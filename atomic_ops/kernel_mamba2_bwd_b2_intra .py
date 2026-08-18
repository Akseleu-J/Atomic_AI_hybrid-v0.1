"""
Milestone MB2 -- Pallas backward kernel for M3's intra-chunk stage
(Y_diag/state_end), i.e. the "intra-chunk" half of _chunk_ssd_bwd -- see
mamba2_bwd_intra_reference.py's module docstring for the full ownership
accounting and why this half can be tested in isolation from MB1
(state-recurrence).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HIGHEST = jax.lax.Precision.HIGHEST
_STEP_CLIP = 20.0
_ACC_CLIP = 1e4


def _resolve_bc(chunk_size):
    return min(128, chunk_size)


def _sanitize(x, clip=_ACC_CLIP):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


def _clip_mask(x, lo, hi):
    return ((x >= lo) & (x <= hi)).astype(jnp.float32)


def _kernel_body_b2(dt_ref, x_ref, b_ref, c_ref, cumdecay_ref, dydiag_ref, dstateend_ref,
                     ddt_ref, dx_ref, db_ref, dc_ref, dcum_ref,
                     *, chunk_size, headdim, d_state):
    bc = _resolve_bc(chunk_size)
    assert chunk_size % bc == 0, f"chunk_size={chunk_size} must be divisible by BC={bc}."
    n_sub = chunk_size // bc

    dt_full = dt_ref[0, 0, 0].astype(jnp.float32)
    x_full = x_ref[0, 0, 0].astype(jnp.float32)
    B_full = b_ref[0, 0, 0].astype(jnp.float32)
    C_full = c_ref[0, 0, 0].astype(jnp.float32)
    cumdecay_full = cumdecay_ref[0, 0, 0].astype(jnp.float32)
    dy_diag_full = dydiag_ref[0, 0, 0].astype(jnp.float32)
    dstate_end = dstateend_ref[0, 0, 0].astype(jnp.float32)   # (D, S)

    dBx_full = _sanitize(dt_full * x_full)
    BC_inner = _sanitize(jnp.dot(C_full, B_full.T, precision=_HIGHEST))

    cum_last = cumdecay_full[chunk_size - 1]
    decay_to_end_diff_raw = cum_last[None, :] - cumdecay_full
    decay_to_end_full = jnp.exp(jnp.clip(decay_to_end_diff_raw, -_STEP_CLIP, _STEP_CLIP))
    write_full = _sanitize(dBx_full * decay_to_end_full)

    # ---- state_end backward: full-chunk matmuls ----
    dwrite = _sanitize(jnp.dot(B_full, dstate_end.T, precision=_HIGHEST))     # (C, D)
    dB_c1 = _sanitize(jnp.dot(write_full, dstate_end, precision=_HIGHEST))    # (C, S)

    d_dBx_1 = _sanitize(dwrite * decay_to_end_full)
    d_decay_to_end = _sanitize(dwrite * dBx_full)
    d_decay_to_end_diff = _sanitize(
        d_decay_to_end * decay_to_end_full * _clip_mask(decay_to_end_diff_raw, -_STEP_CLIP, _STEP_CLIP)
    )
    d_cumdecay_from_dte = -d_decay_to_end_diff                       # (C, D)
    d_cumdecay_last_b = jnp.sum(d_decay_to_end_diff, axis=0)         # (D,)

    # ---- init accumulators (dx_ref = scratch for d_dBx_2 during the loop) ----
    dx_ref[0, 0, 0] = jnp.zeros((chunk_size, headdim), dtype=jnp.float32)
    db_ref[0, 0, 0] = dB_c1
    dc_ref[0, 0, 0] = jnp.zeros((chunk_size, d_state), dtype=jnp.float32)
    dcum_ref[0, 0, 0] = jnp.zeros((chunk_size, headdim), dtype=jnp.float32)

    for si in range(n_sub):
        for sj in range(si + 1):
            i0, i1 = si * bc, (si + 1) * bc
            j0, j1 = sj * bc, (sj + 1) * bc

            cum_i = cumdecay_full[i0:i1]
            cum_j = cumdecay_full[j0:j1]
            decay_diff = cum_i[:, None, :] - cum_j[None, :, :]
            edecay = jnp.exp(jnp.clip(decay_diff, -_STEP_CLIP, _STEP_CLIP))

            bc_blk = BC_inner[i0:i1, j0:j1]
            if si == sj:
                idx = jnp.arange(bc)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                bc_blk = bc_blk * causal

            weight = edecay * bc_blk[:, :, None]

            dy_i = dy_diag_full[i0:i1]
            dBx_j = dBx_full[j0:j1]

            dweight = dy_i[:, None, :] * dBx_j[None, :, :]                     # (bc,bc,d)
            d_dBx_2_blk = jnp.einsum("ijd,id->jd", weight, dy_i, precision=_HIGHEST)

            dedecay = dweight * bc_blk[:, :, None]
            dBC_inner_blk = jnp.sum(dweight * edecay, axis=-1)
            if si == sj:
                dBC_inner_blk = dBC_inner_blk * causal

            dC_blk = jnp.dot(dBC_inner_blk, B_full[j0:j1], precision=_HIGHEST)      # (bc, S)
            dB_blk = jnp.dot(dBC_inner_blk.T, C_full[i0:i1], precision=_HIGHEST)    # (bc, S)

            d_decay_diff = dedecay * edecay * _clip_mask(decay_diff, -_STEP_CLIP, _STEP_CLIP)
            dcum_i_blk = jnp.sum(d_decay_diff, axis=1)
            dcum_j_blk = -jnp.sum(d_decay_diff, axis=0)

            dx_ref[0, 0, 0, j0:j1] = _sanitize(dx_ref[0, 0, 0, j0:j1] + d_dBx_2_blk)
            db_ref[0, 0, 0, j0:j1] = _sanitize(db_ref[0, 0, 0, j0:j1] + dB_blk)
            dc_ref[0, 0, 0, i0:i1] = _sanitize(dc_ref[0, 0, 0, i0:i1] + dC_blk)
            dcum_ref[0, 0, 0, i0:i1] = _sanitize(dcum_ref[0, 0, 0, i0:i1] + dcum_i_blk)
            dcum_ref[0, 0, 0, j0:j1] = _sanitize(dcum_ref[0, 0, 0, j0:j1] + dcum_j_blk)

    d_dBx_2_final = dx_ref[0, 0, 0]
    d_dBx_total = _sanitize(d_dBx_1 + d_dBx_2_final)

    ddt_final = d_dBx_total * x_full
    dx_final = d_dBx_total * dt_full

    dcum_final = dcum_ref[0, 0, 0] + d_cumdecay_from_dte
    row_mask = (jnp.arange(chunk_size) == (chunk_size - 1)).astype(jnp.float32)[:, None]
    dcum_final = dcum_final + row_mask * d_cumdecay_last_b[None, :]

    ddt_ref[0, 0, 0] = _sanitize(ddt_final)
    dx_ref[0, 0, 0] = _sanitize(dx_final)
    dcum_ref[0, 0, 0] = _sanitize(dcum_final)


def intra_chunk_ssd_bwd_pallas(dt, x, B, C, cumdecay, dy, dstate_end_grad, chunk_size, interpret=False):
    bsz, L, n_heads_ssm, headdim = dt.shape
    d_state = B.shape[-1]
    assert L % chunk_size == 0
    n_chunks = L // chunk_size

    def reshape_dx(t):
        t = t.reshape(bsz, n_chunks, chunk_size, n_heads_ssm, headdim)
        return jnp.moveaxis(t, (1, 3), (2, 1))

    def reshape_bc(t):
        t = t.reshape(bsz, n_chunks, chunk_size, d_state)
        t = jnp.broadcast_to(t[:, None], (bsz, n_heads_ssm, n_chunks, chunk_size, d_state))
        return t

    dt_r, x_r, dy_r = reshape_dx(dt), reshape_dx(x), reshape_dx(dy)
    B_r, C_r = reshape_bc(B), reshape_bc(C)

    grid = (bsz, n_heads_ssm, n_chunks)
    dx_spec = pl.BlockSpec((1, 1, 1, chunk_size, headdim), lambda i, h, c: (i, h, c, 0, 0))
    bc_spec = pl.BlockSpec((1, 1, 1, chunk_size, d_state), lambda i, h, c: (i, h, c, 0, 0))
    state_spec = pl.BlockSpec((1, 1, 1, headdim, d_state), lambda i, h, c: (i, h, c, 0, 0))

    ddt, dx_raw, dB_raw, dC_raw, dcum = pl.pallas_call(
        lambda *refs: _kernel_body_b2(*refs, chunk_size=chunk_size, headdim=headdim, d_state=d_state),
        grid=grid,
        in_specs=[dx_spec, dx_spec, bc_spec, bc_spec, dx_spec, dx_spec, state_spec],
        out_specs=[dx_spec, dx_spec, bc_spec, bc_spec, dx_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, n_heads_ssm, n_chunks, chunk_size, headdim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, n_heads_ssm, n_chunks, chunk_size, headdim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, n_heads_ssm, n_chunks, chunk_size, d_state), jnp.float32),
            jax.ShapeDtypeStruct((bsz, n_heads_ssm, n_chunks, chunk_size, d_state), jnp.float32),
            jax.ShapeDtypeStruct((bsz, n_heads_ssm, n_chunks, chunk_size, headdim), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
        interpret=interpret,
    )(dt_r, x_r, B_r, C_r, cumdecay, dy_r, dstate_end_grad)

    def unreshape_dx(t):
        return jnp.moveaxis(t, 1, 3).reshape(bsz, L, n_heads_ssm, headdim)

    def unreshape_bc(t):
        t = jnp.sum(t, axis=1)
        return t.reshape(bsz, L, d_state)

    ddt_out = unreshape_dx(ddt)
    dx_out = unreshape_dx(dx_raw)
    dB_out = unreshape_bc(dB_raw)
    dC_out = unreshape_bc(dC_raw)
    return ddt_out, dx_out, dB_out, dC_out, dcum
