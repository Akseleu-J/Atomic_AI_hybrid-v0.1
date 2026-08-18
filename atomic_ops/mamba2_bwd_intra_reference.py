"""
atomic_ops/mamba2_bwd_intra_reference.py -- correctness target for MB2.

Isolates the "intra-chunk" half of _chunk_ssd_bwd (mamba2_bwd_reference.py,
MB0) -- backward of ONLY (y_diag, state_end), i.e. exactly what
kernel_mamba2_bwd_b2_intra.py's Pallas kernel must reproduce -- from the
"state-recurrence" half (y_off/state_prev/decay_h/decay_chunk_end, MB1's
job, already extracted in kernel_mamba2_bwd_b1_state.py).

Why this split is safe to test standalone: `_chunk_ssd` with state_prev=0
gives y_c==y_diag and state_new==state_end IDENTICALLY (not just
"usually") -- every MB1-owned term (dC_c_1, d_cumdecay_from_yoff,
d_cumdecay_last_a, dstate_prev) is a linear function of state_prev and
therefore zero when state_prev=0. `_intra_only_fwd` below is literally
`_chunk_ssd` with the y_off/state_new lines deleted -- same numerics, not
an approximation.

`_intra_only_bwd` is the hand derivation (same "clip = straight-through
mask" adjoint convention as mamba2_bwd_reference.py's MB0), cross-checked
against jax.vjp(_intra_only_fwd) in test_kernel_mamba2_bwd_b2_intra.py --
same "hand derivation -> jax.vjp cross-check -> Pallas port -> TPU test"
discipline as the rest of atomic_ops/.
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


def _intra_only_fwd(dt_c, B_c, C_c, x_c, cumdecay_c):
    """dt_c,x_c,cumdecay_c: (b,C,h,d). B_c,C_c: (b,C,s) -- shared across
    heads (no h axis), same convention as mamba2_ssd_reference._chunk_ssd.
    Returns y_diag: (b,C,h,d), state_end: (b,h,d,s)."""
    f32 = jnp.float32
    dt_c, B_c, C_c, x_c, cumdecay_c = (t.astype(f32) for t in (dt_c, B_c, C_c, x_c, cumdecay_c))
    Cs = dt_c.shape[1]
    idx = jnp.arange(Cs)
    causal = (idx[:, None] >= idx[None, :]).astype(f32)[None, :, :, None, None]

    decay_diff = cumdecay_c[:, :, None, :, :] - cumdecay_c[:, None, :, :, :]
    L = jnp.exp(jnp.clip(decay_diff, -_STEP_CLIP, _STEP_CLIP)) * causal
    L = _sanitize(L, clip=1e6)

    BC_inner = _sanitize(jnp.einsum("bis,bjs->bij", C_c, B_c, precision=_HIGHEST))
    dBx = _sanitize(dt_c * x_c)

    weight = L * BC_inner[:, :, :, None, None]
    y_diag = _sanitize(jnp.einsum("bijhd,bjhd->bihd", weight, dBx, precision=_HIGHEST))

    decay_to_end = jnp.exp(jnp.clip(cumdecay_c[:, -1:, :, :] - cumdecay_c, -_STEP_CLIP, _STEP_CLIP))
    decay_to_end = _sanitize(decay_to_end, clip=1e6)
    write = _sanitize(dBx * decay_to_end)
    state_end = _sanitize(jnp.einsum("bcs,bchd->bhds", B_c, write, precision=_HIGHEST))

    return y_diag, state_end


def _intra_only_bwd(dt_c, B_c, C_c, x_c, cumdecay_c, dy_diag_c, dstate_end_c):
    """Hand-derived adjoint of _intra_only_fwd.
    Returns: ddt_intra (PARTIAL -- MB4 adds the dcumdecay->dA_exponent
    contribution), dB_intra (FULL -- B never touches state_prev/y_off),
    dC_intra (PARTIAL -- MB1 adds its own y_off contribution), dx_c (FULL),
    dcumdecay_intra (PARTIAL -- MB1 adds its own contribution)."""
    f32 = jnp.float32
    dt_c, B_c, C_c, x_c, cumdecay_c, dy_diag_c, dstate_end_c = (
        t.astype(f32) for t in (dt_c, B_c, C_c, x_c, cumdecay_c, dy_diag_c, dstate_end_c)
    )
    Cs = dt_c.shape[1]
    idx = jnp.arange(Cs)
    causal = (idx[:, None] >= idx[None, :]).astype(f32)[None, :, :, None, None]

    decay_diff_raw = cumdecay_c[:, :, None, :, :] - cumdecay_c[:, None, :, :, :]
    L = jnp.exp(jnp.clip(decay_diff_raw, -_STEP_CLIP, _STEP_CLIP)) * causal

    BC_inner_raw = jnp.einsum("bis,bjs->bij", C_c, B_c, precision=_HIGHEST)
    BC_inner = _sanitize(BC_inner_raw)

    dBx = _sanitize(dt_c * x_c)
    weight = L * BC_inner[:, :, :, None, None]

    decay_to_end_diff_raw = cumdecay_c[:, -1:, :, :] - cumdecay_c
    decay_to_end = _sanitize(jnp.exp(jnp.clip(decay_to_end_diff_raw, -_STEP_CLIP, _STEP_CLIP)), clip=1e6)
    write = _sanitize(dBx * decay_to_end)

    # ---- state_end = sum_c B_c[c] (x) write[c] ----
    dwrite = jnp.einsum("bhds,bcs->bchd", dstate_end_c, B_c, precision=_HIGHEST)
    dB_1 = jnp.einsum("bhds,bchd->bcs", dstate_end_c, write, precision=_HIGHEST)

    d_dBx_1 = dwrite * decay_to_end
    d_decay_to_end = dwrite * dBx
    d_decay_to_end_diff = d_decay_to_end * decay_to_end * _clip_mask(decay_to_end_diff_raw, -_STEP_CLIP, _STEP_CLIP)
    d_cumdecay_from_dte = -d_decay_to_end_diff
    d_cumdecay_last_b = jnp.sum(d_decay_to_end_diff, axis=1)   # (b,h,d) -- broadcast row -> sum over C

    # ---- y_diag = sum_j weight[i,j]*dBx[j] ----
    dweight = jnp.einsum("bihd,bjhd->bijhd", dy_diag_c, dBx, precision=_HIGHEST)
    d_dBx_2 = jnp.einsum("bijhd,bihd->bjhd", weight, dy_diag_c, precision=_HIGHEST)

    dL = dweight * BC_inner[:, :, :, None, None]
    dBC_inner = jnp.sum(dweight * L, axis=(-2, -1))   # sum over (h,d)

    dC_2 = jnp.einsum("bij,bjs->bis", dBC_inner, B_c, precision=_HIGHEST)
    dB_2 = jnp.einsum("bij,bis->bjs", dBC_inner, C_c, precision=_HIGHEST)

    d_decay_diff = dL * L * _clip_mask(decay_diff_raw, -_STEP_CLIP, _STEP_CLIP)
    d_cumdecay_i = jnp.sum(d_decay_diff, axis=2)
    d_cumdecay_j = -jnp.sum(d_decay_diff, axis=1)

    d_dBx_total = _sanitize(d_dBx_1 + d_dBx_2)
    ddt_intra = d_dBx_total * x_c
    dx_c = d_dBx_total * dt_c

    dcumdecay_intra = _sanitize(d_cumdecay_i + d_cumdecay_j + d_cumdecay_from_dte)
    row_mask = (idx == (Cs - 1)).astype(f32)[None, :, None, None]
    dcumdecay_intra = dcumdecay_intra + row_mask * d_cumdecay_last_b[:, None, :, :]

    dB_intra = _sanitize(dB_1 + dB_2)
    dC_intra = _sanitize(dC_2)

    return _sanitize(ddt_intra), dB_intra, dC_intra, _sanitize(dx_c), dcumdecay_intra
