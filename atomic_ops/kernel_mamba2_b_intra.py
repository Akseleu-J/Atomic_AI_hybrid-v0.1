"""
Milestone M3 -- Pallas kernel for Mamba2 SSD intra-chunk output (Y_diag) and
chunk-end state write (state_end), consuming `cumdecay` from M2
(kernel_mamba2_a_decay.py) directly -- the pairwise (i,j,d) decay weight is
built ONLY inside (BC,BC,headdim) sub-blocks (BC=128, looped via static
Python range like kernel_a_scores.py's N_SUB split), never as a full
(BT,BT,headdim) tensor -- same "only ever materialize (BC,BC[,D]) at the
sub-block level" discipline GDN-2's Kernel A (_weighted_pair_sum) already
uses; the (BT,BT,D)-at-full-BT-avoidance rule refers to the FULL chunk size,
not the BC=128 sub-block, which GDN-2 already does hold in VMEM transiently.

Per (batch, head, chunk):
  dBx[j,d]        = dt_c[j,d] * x_c[j,d]
  BC_inner[i,j]   = C_c[i,:] . B_c[j,:]                         (d_state contraction)
  decay_diff[i,j,d] = cumdecay_c[i,d] - cumdecay_c[j,d]          (i>=j causal)
  weight[i,j,d]   = exp(clip(decay_diff,-20,20)) * BC_inner[i,j] * causal[i,j]
  Y_diag[i,d]     = sum_j weight[i,j,d] * dBx[j,d]
  decay_to_end[j,d] = exp(clip(cumdecay_c[-1,d] - cumdecay_c[j,d], -20, 20))
  write[j,d]      = dBx[j,d] * decay_to_end[j,d]
  state_end[d,s]  = sum_j B_c[j,s] * write[j,d]

NOTE: this is ONLY the intra-chunk piece -- y_off (reading state_prev) and
state_new (state_prev*decay + state_end) are M4's job (inter-chunk
lax.scan, mirroring kernel_d_pipeline.py's gdn2_inter_chunk_combine). With
state_prev=0 (single chunk), Y_diag==full y and state_end==full state_final
from mamba2_ssd_reference -- exactly what the unit test below cross-checks
against, reusing the already-validated M1 reference instead of duplicating
a second independent reference implementation.

Sanitization: same convention as kernel_a_scores.py/kernel_bwd_b4_intra.py
-- clip decay_diff exponent to +-20 before exp (bounded quantity feeding
exp), nan_to_num+clip(+-1e4) on every accumulation write (READ-MODIFY-WRITE
across sub-block iterations, same "an inf from one iteration can meet an
opposite-sign inf from another and produce an unrecoverable NaN" risk
documented in kernel_bwd_b4_intra.py's own docstring).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HIGHEST = jax.lax.Precision.HIGHEST
_STEP_CLIP = 20.0
_ACC_CLIP = 1e4

# FIX: BC was a fixed module-level constant (128), matching GDN-2's
# kernel_a_scores.py -- but GDN-2's BT is ALSO fixed at 256 project-wide, so
# BC=128 always divided it evenly. Mamba2's chunk_size is a free ModelConfig
# knob (deltanet_chunk_size) and M1's own test suite explicitly validates
# chunk_size=64 (test_matches_token_serial_multi_chunk) as well as
# chunk_size-invariance across 64/128/256 -- a hardcoded BC=128 kernel
# constant silently assumed chunk_size was always >=128 and a multiple of
# it, which broke the moment M4's test reused M1's own chunk_size=64 case.
# BC is now derived from chunk_size itself: use the smaller of (128,
# chunk_size), so chunk_size>=128 keeps the original multi-sub-block
# behavior unchanged (already validated by test_larger_chunk_multi_subblock),
# and chunk_size<128 degrades to a single sub-block (n_sub=1) covering the
# whole chunk in one causal-masked pass.
def _resolve_bc(chunk_size):
    return min(128, chunk_size)


def _sanitize(x, clip=_ACC_CLIP):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


def _kernel_body_b(dt_ref, x_ref, b_ref, c_ref, cumdecay_ref,
                    ydiag_ref, stateend_ref, *, chunk_size, headdim, d_state):
    bc = _resolve_bc(chunk_size)
    assert chunk_size % bc == 0, f"chunk_size={chunk_size} must be divisible by BC={bc}."
    n_sub = chunk_size // bc

    dt_full = dt_ref[0, 0, 0].astype(jnp.float32)
    x_full = x_ref[0, 0, 0].astype(jnp.float32)
    B_full = b_ref[0, 0, 0].astype(jnp.float32)
    C_full = c_ref[0, 0, 0].astype(jnp.float32)
    cumdecay_full = cumdecay_ref[0, 0, 0].astype(jnp.float32)

    dBx_full = _sanitize(dt_full * x_full)

    BC_inner = jnp.dot(C_full, B_full.T, precision=_HIGHEST)
    BC_inner = _sanitize(BC_inner)

    ydiag_ref[0, 0, 0] = jnp.zeros((chunk_size, headdim), dtype=jnp.float32)

    for si in range(n_sub):
        for sj in range(si + 1):
            i0, i1 = si * bc, (si + 1) * bc
            j0, j1 = sj * bc, (sj + 1) * bc

            cum_i = cumdecay_full[i0:i1]
            cum_j = cumdecay_full[j0:j1]
            decay_diff = cum_i[:, None, :] - cum_j[None, :, :]
            edecay = jnp.exp(jnp.clip(decay_diff, -_STEP_CLIP, _STEP_CLIP))

            bc_blk = BC_inner[i0:i1, j0:j1]
            weight = edecay * bc_blk[:, :, None]

            if si == sj:
                idx = jnp.arange(bc)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                weight = weight * causal[:, :, None]

            dBx_j = dBx_full[j0:j1]
            y_blk = jnp.einsum("ijd,jd->id", weight, dBx_j, precision=_HIGHEST)

            ydiag_ref[0, 0, 0, i0:i1] = _sanitize(ydiag_ref[0, 0, 0, i0:i1] + y_blk)

    cum_last = cumdecay_full[chunk_size - 1]
    decay_to_end = jnp.exp(jnp.clip(cum_last[None, :] - cumdecay_full, -_STEP_CLIP, _STEP_CLIP))
    write = _sanitize(dBx_full * decay_to_end)

    state_end = jnp.dot(write.T, B_full, precision=_HIGHEST)
    stateend_ref[0, 0, 0] = _sanitize(state_end)


def intra_chunk_ssd_pallas(dt, x, B, C, cumdecay, chunk_size, interpret=False):
    """dt, x: (bsz, L, n_heads_ssm, headdim). B, C: (bsz, L, d_state).
    cumdecay: (bsz, n_heads_ssm, n_chunks, chunk_size, headdim) -- output of
    build_chunk_cumdecay_pallas (M2), same layout convention.

    Returns Y_diag: (bsz, L, n_heads_ssm, headdim),
            state_end: (bsz, n_heads_ssm, n_chunks, headdim, d_state).
    """
    bsz, L, n_heads_ssm, headdim = dt.shape
    d_state = B.shape[-1]
    assert L % chunk_size == 0
    n_chunks = L // chunk_size

    def reshape_dx(t):
        t = t.reshape(bsz, n_chunks, chunk_size, n_heads_ssm, headdim)
        return jnp.moveaxis(t, (1, 3), (2, 1))   # -> (bsz, n_heads_ssm, n_chunks, BT, D)

    def reshape_bc(t):
        # B/C are shared across heads -- broadcast into the head axis so
        # every (batch,head,chunk) grid cell gets its own contiguous block
        # (simplest correct approach; B/C are small relative to dt/x/state).
        t = t.reshape(bsz, n_chunks, chunk_size, d_state)
        t = jnp.broadcast_to(t[:, None], (bsz, n_heads_ssm, n_chunks, chunk_size, d_state))
        return t

    dt_r = reshape_dx(dt)
    x_r = reshape_dx(x)
    B_r = reshape_bc(B)
    C_r = reshape_bc(C)

    grid = (bsz, n_heads_ssm, n_chunks)
    dx_spec = pl.BlockSpec((1, 1, 1, chunk_size, headdim), lambda i, h, c: (i, h, c, 0, 0))
    bc_spec = pl.BlockSpec((1, 1, 1, chunk_size, d_state), lambda i, h, c: (i, h, c, 0, 0))
    state_spec = pl.BlockSpec((1, 1, 1, headdim, d_state), lambda i, h, c: (i, h, c, 0, 0))

    y_diag, state_end = pl.pallas_call(
        lambda *refs: _kernel_body_b(*refs, chunk_size=chunk_size, headdim=headdim, d_state=d_state),
        grid=grid,
        in_specs=[dx_spec, dx_spec, bc_spec, bc_spec, dx_spec],
        out_specs=[dx_spec, state_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, n_heads_ssm, n_chunks, chunk_size, headdim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, n_heads_ssm, n_chunks, headdim, d_state), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
        interpret=interpret,
    )(dt_r, x_r, B_r, C_r, cumdecay)

    y_diag_out = jnp.moveaxis(y_diag, (1, 2), (3, 1)).reshape(bsz, L, n_heads_ssm, headdim)
    return y_diag_out, state_end
