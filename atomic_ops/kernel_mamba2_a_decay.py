"""
Milestone M2 -- Pallas Kernel A-equivalent for Mamba2 SSD: builds per-chunk
cumulative log-decay (`cumdecay`) from dt and per-head A, matching the exact
tril-matmul-cumsum trick and clip discipline used in
`atomic_ops/mamba2_ssd_reference.py`'s `_chunk_ssd` (and, before that,
`kernel_a_scores.py`'s `gc` computation for GDN-2).

SCOPE NOTE (deviation from the plan doc's literal "L[h,i,j]" pseudocode):
Mamba2's decay is per-HEAD (`A: (n_heads_ssm,)`) but `dt` -- and therefore
the actual cumulative decay used in the pairwise `L` matrix -- varies per
CHANNEL too (see mamba2_ssd_reference.py's own module docstring: "dt still
varies per-channel, so decay CAN still differ per channel through dt").
`cumdecay` is therefore `(b, i, h, d)`-shaped, not `(b, h, i, j)` -- pairwise
`L[h,i,j,d] = exp(cumdecay[i,h,d] - cumdecay[j,h,d])` is NEVER materialized
here (that would be a `(BT,BT,headdim)` tensor per head per chunk -- e.g.
256*256*192 ~= 12.6M floats just for ONE head/chunk/batch element at typical
project sizes). `cumdecay` itself is the same size as `dt`/`x` (`(BT,D)` per
head/chunk), safe to materialize and hand to M3, which will build the
pairwise `(i,j)` weighting fused inside its Y_diag/state einsums -- same
"never materialize the O(BT^2 * D) intermediate" discipline GDN-2's Kernel
A/B/C/D pipeline already follows (`gc` is `(BT,D)`, `Aqk`/`Akk` are the only
`(BT,BT)` tensors that ever hit HBM, and those have NO extra `D` axis
because GDN-2's decay doesn't vary per-channel-within-D the way Mamba2's
does).

Sanitization: same two-line convention as kernel_a_scores.py --
  (1) per-step dA_exponent clipped to +-20 (bounded quantity feeding exp
      indirectly via the cumsum -- matches mamba2_ssd_reference.py's own
      FIX comment on why the CUMULATIVE sum must NOT be clipped this
      tightly),
  (2) the cumulative `cumdecay` clipped only to the generic overflow guard
      (+-1e4), never to +-20 -- clipping the running sum to the per-step
      bound was exactly the M1 bug documented in
      mamba2_ssd_reference.py's own FIX comment; repeating it here would
      reintroduce the same ~2x rel_err corruption at the kernel level.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HIGHEST = jax.lax.Precision.HIGHEST
_STEP_CLIP = 20.0
_ACC_CLIP = 1e4


def _sanitize(x, clip):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


def _kernel_body_a(dt_ref, a_ref, cumdecay_ref, *, chunk_size):
    dt_full = dt_ref[0, 0, 0].astype(jnp.float32)   # (BT, D)

    h_idx = pl.program_id(1)
    a_val = a_ref[h_idx].astype(jnp.float32)   # scalar read from SMEM -- no alignment constraint here

    dA_exponent = dt_full * a_val
    dA_exponent = _sanitize(dA_exponent, _STEP_CLIP)

    idx = jnp.arange(chunk_size)
    tril_ones = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    cumdecay = jnp.dot(tril_ones, dA_exponent, precision=_HIGHEST)

    cumdecay = _sanitize(cumdecay, _ACC_CLIP)

    cumdecay_ref[0, 0, 0] = cumdecay


def build_chunk_cumdecay_pallas(dt, A, chunk_size, interpret=False):
    bsz, L, n_heads_ssm, headdim = dt.shape
    assert L % chunk_size == 0, f"seq_len={L} must be divisible by chunk_size={chunk_size}."
    n_chunks = L // chunk_size
    assert A.shape == (n_heads_ssm,), f"A must be (n_heads_ssm,)={n_heads_ssm}, got {A.shape}."

    def reshape_in(t):
        t = t.reshape(bsz, n_chunks, chunk_size, n_heads_ssm, headdim)
        return jnp.moveaxis(t, (1, 3), (2, 1))

    dt_r = reshape_in(dt)

    grid = (bsz, n_heads_ssm, n_chunks)
    io_spec = pl.BlockSpec((1, 1, 1, chunk_size, headdim), lambda i, h, c: (i, h, c, 0, 0))
    # FIX: A must live in SMEM, not VMEM. Mosaic requires dynamic VMEM
    # vector indices to be provably a multiple of 128 (lane-tiling
    # constraint) -- a tiny (n_heads_ssm,) array indexed by an arbitrary
    # h_idx can never satisfy that, regardless of whether the BlockSpec
    # hands in a full or partial block (both previous attempts hit this
    # from different angles). SMEM has no such alignment requirement --
    # it's exactly the scalar/lookup-table memory space Pallas TPU expects
    # for this pattern.
    a_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    cumdecay = pl.pallas_call(
        lambda *refs: _kernel_body_a(*refs, chunk_size=chunk_size),
        grid=grid,
        in_specs=[io_spec, a_spec],
        out_specs=io_spec,
        out_shape=jax.ShapeDtypeStruct((bsz, n_heads_ssm, n_chunks, chunk_size, headdim), jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
        interpret=interpret,
    )(dt_r, A.astype(jnp.float32))

    return cumdecay
