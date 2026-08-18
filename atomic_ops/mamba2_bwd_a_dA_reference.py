"""
Milestone MB4 -- hand-derived backward for M2 (kernel_mamba2_a_decay.py):
  dA_exponent_raw = einsum('bchd,h->bchd', dt, A)
  dA_exponent     = clip(dA_exponent_raw, -20, 20)
  cumdecay_raw    = tril_cumsum(dA_exponent)          (matmul-cumsum trick)
  cumdecay        = clip(cumdecay_raw, -1e4, 1e4)

This is the ONE piece of _chunk_ssd_bwd (MB0, mamba2_bwd_reference.py) not
yet split out on its own -- MB1 (state-recurrence) and MB2 (intra-chunk,
Pallas) both consume `cumdecay` as an already-given forward value and never
touch `dt`/`A` through this path; MB0's own `ddt_c_2`/`dA` block is lifted
here VERBATIM (same "hand derivation already exists inside MB0, MB4 just
extracts it" relationship MB1/MB2 have to their own MB0 sections).

Input contract: `dcumdecay` here is the COMBINED per-chunk cotangent for
`cumdecay` that MB3's orchestrator (kernel_mamba2_bwd_b1_state.py) already
produces -- i.e. `dcumdecay_state + dcumdecay_intra`, in the SAME
(bsz, n_heads_ssm, n_chunks, chunk_size, headdim) Pallas layout M2's own
`build_chunk_cumdecay_pallas` outputs. MB4 does NOT recompute or duplicate
that sum -- it is purely the last leg of the chain: combined dcumdecay ->
(mask by cumdecay's own clip) -> reverse-cumsum -> (mask by dA_exponent's
own clip) -> ddt contribution #2 (added to MB2's ddt_intra to finally
complete `ddt`) + dA (the only NON-per-chunk, NON-per-token gradient in
this whole backward chain -- a single (n_heads_ssm,) reduction over every
batch/chunk/token/channel).

Same clip-mask convention as mamba2_bwd_reference.py's own MB0 and
mamba2_bwd_state_reference.py/mamba2_bwd_intra_reference.py's MB1/MB2
splits: every jnp.clip in the forward gets a straight-through gradient
mask (1.0 inside the window, 0.0 outside) -- this is bit-for-bit what
JAX's own autodiff produces for jnp.clip, and is the reason
`_chunk_ssd_bwd_with_partials` in the MB1+MB3 integration test could be
cross-checked exactly against jax.vjp.

Sanitization: final `nan_to_num+clip(+-1e4)` on both returned gradients,
same "custom_vjp return boundary is the last chance to catch a non-finite"
discipline as every other backward wrapper in this project (B5's
reverse_cumsum_bwd, kernel_trainable_B6.py's `_final_sanitize`, etc).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST
_STEP_CLIP = 20.0
_ACC_CLIP = 1e4


def _sanitize(x, clip=_ACC_CLIP):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


def _clip_mask(x, lo, hi):
    """Straight-through gradient mask for jnp.clip -- same convention as
    mamba2_bwd_reference.py's own _clip_mask."""
    return ((x >= lo) & (x <= hi)).astype(jnp.float32)


def mamba2_dA_backward(dt_pallas, A, dcumdecay_combined, chunk_size):
    """dt_pallas: (bsz, n_heads_ssm, n_chunks, chunk_size, headdim) -- SAME
    layout M2's build_chunk_cumdecay_pallas uses internally for dt (i.e.
    the caller must reshape raw (bsz,L,n_heads_ssm,headdim) dt the same way
    kernel_mamba2_a_decay.py's own `reshape_in` does -- see
    _reshape_dt_to_pallas below for a ready-made helper).
    A: (n_heads_ssm,), float32.
    dcumdecay_combined: (bsz, n_heads_ssm, n_chunks, chunk_size, headdim) --
    MB3's output (dcumdecay_state + dcumdecay_intra).

    Returns:
      ddt_contrib2: same layout as dt_pallas -- ADD to MB2's ddt_intra to
        get the final, complete ddt (see kernel_mamba2_bwd_b1_state.py's
        own docstring: "ddt -- PARTIAL... The second contribution... is
        MB4's job").
      dA: (n_heads_ssm,) -- the FULL, final gradient for A (nothing else
        in the whole backward chain touches A).
    """
    f32 = jnp.float32
    dt_f = dt_pallas.astype(f32)
    A_f = A.astype(f32)
    dcum = dcumdecay_combined.astype(f32)

    bsz, n_heads_ssm, n_chunks, C, headdim = dt_f.shape
    assert C == chunk_size

    # ---- re-run the forward (cheap; same "recompute, don't stash"
    # tradeoff used throughout atomic_ops/) ----
    # dA_exponent_raw[b,h,c,i,d] = dt[b,h,c,i,d] * A[h]
    dA_exponent_raw = dt_f * A_f[None, :, None, None, None]
    dA_exponent_clipped = jnp.clip(dA_exponent_raw, -_STEP_CLIP, _STEP_CLIP)

    idx = jnp.arange(C)
    tril_ones = (idx[:, None] >= idx[None, :]).astype(f32)
    cumdecay_raw = jnp.einsum("ij,bhcjd->bhcid", tril_ones, dA_exponent_clipped, precision=_HIGHEST)

    # ================= REVERSE PASS =================
    # cumdecay = clip(cumdecay_raw, -1e4, 1e4)
    d_cumdecay_raw = dcum * _clip_mask(cumdecay_raw, -_ACC_CLIP, _ACC_CLIP)

    # cumdecay_raw = tril_cumsum(dA_exponent_clipped)  -- reverse cumsum,
    # same trick as kernel_bwd_b5_reverse_cumsum.py.
    triu_ones = (idx[:, None] <= idx[None, :]).astype(f32)
    d_dA_exponent_clipped = jnp.einsum("ij,bhcjd->bhcid", triu_ones, d_cumdecay_raw, precision=_HIGHEST)

    # dA_exponent_clipped = clip(dA_exponent_raw, -20, 20)
    d_dA_exponent_raw = d_dA_exponent_clipped * _clip_mask(dA_exponent_raw, -_STEP_CLIP, _STEP_CLIP)

    # dA_exponent_raw = dt * A[h]  (broadcast over b,c,i,d)
    ddt_contrib2 = d_dA_exponent_raw * A_f[None, :, None, None, None]
    dA = jnp.sum(d_dA_exponent_raw * dt_f, axis=(0, 2, 3, 4))   # reduce everything except h

    ddt_contrib2 = _sanitize(ddt_contrib2)
    dA = _sanitize(dA)

    return ddt_contrib2, dA


def reshape_dt_to_pallas(dt, chunk_size):
    """dt: (bsz, L, n_heads_ssm, headdim) -> (bsz, n_heads_ssm, n_chunks,
    chunk_size, headdim) -- byte-for-byte the same reshape/moveaxis M2's
    own `build_chunk_cumdecay_pallas` applies internally, exposed here so
    MB5's final orchestrator can call `mamba2_dA_backward` with a plain
    (b,l,h,d) `dt` without duplicating this reshape logic at every call
    site."""
    bsz, L, n_heads_ssm, headdim = dt.shape
    assert L % chunk_size == 0
    n_chunks = L // chunk_size
    t = dt.reshape(bsz, n_chunks, chunk_size, n_heads_ssm, headdim)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def unreshape_dt_from_pallas(t):
    """Inverse of reshape_dt_to_pallas: (bsz, n_heads_ssm, n_chunks,
    chunk_size, headdim) -> (bsz, L, n_heads_ssm, headdim)."""
    bsz, n_heads_ssm, n_chunks, chunk_size, headdim = t.shape
    out = jnp.moveaxis(t, (1, 2), (3, 1))
    return out.reshape(bsz, n_chunks * chunk_size, n_heads_ssm, headdim)
