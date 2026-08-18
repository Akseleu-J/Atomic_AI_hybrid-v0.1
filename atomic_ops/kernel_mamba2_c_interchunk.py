"""
Milestone M4 -- inter-chunk lax.scan for Mamba2 SSD, combining M2's
cumdecay and M3's per-chunk (Y_diag, state_end) with a carried state_prev
across chunks. Plain JAX (not Pallas) -- mirrors kernel_d_pipeline.py's
gdn2_inter_chunk_combine: Pallas handles the O(BT^2) intra-chunk work (M2 +
M3), a plain lax.scan handles the O(n_chunks) sequential carry, same
division of labor as GDN-2's Kernel A/B/C (Pallas) -> Kernel D (plain scan).

Per chunk (mirrors mamba2_ssd_reference.py's `_chunk_ssd` y_off/state_new
formulas exactly -- this is the SAME math, just computed here from M2/M3's
kernel outputs instead of inline):

  decay_from_start[i,d] = exp(clip(cumdecay_c[i,d], -20, 0))
  y_off[i,d]  = sum_s C_c[i,s] * state_prev[d,s] * decay_from_start[i,d]
  decay_chunk_end[d]    = exp(clip(cumdecay_c[-1,d], -20, 0))
  state_new[d,s] = state_prev[d,s] * decay_chunk_end[d] + state_end_c[d,s]
  Y[i,d] = Y_diag[i,d] + y_off[i,d]

Sanitization: same clip(+-1e4)+nan_to_num convention as
kernel_d_pipeline.py's own scan step (h_new/o_c sanitized every step).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .kernel_mamba2_a_decay import build_chunk_cumdecay_pallas
from .kernel_mamba2_b_intra import intra_chunk_ssd_pallas

_HIGHEST = jax.lax.Precision.HIGHEST
_ACC_CLIP = 1e4


def _sanitize(x, clip=_ACC_CLIP):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


def mamba2_inter_chunk_combine(y_diag, state_end, cumdecay, C, chunk_size, state0=None):
    """y_diag: (bsz, L, n_heads_ssm, headdim) -- M3 output, already flattened.
    state_end: (bsz, n_heads_ssm, n_chunks, headdim, d_state) -- M3 output.
    cumdecay: (bsz, n_heads_ssm, n_chunks, chunk_size, headdim) -- M2 output.
    C: (bsz, L, d_state).

    Returns y: (bsz, L, n_heads_ssm, headdim), state_final: (bsz, n_heads_ssm, headdim, d_state).
    """
    bsz, L, n_heads_ssm, headdim = y_diag.shape
    d_state = C.shape[-1]
    assert L % chunk_size == 0
    n_chunks = L // chunk_size

    y_diag_ch = y_diag.reshape(bsz, n_chunks, chunk_size, n_heads_ssm, headdim)
    y_diag_ch = jnp.moveaxis(y_diag_ch, (1, 3), (2, 1))  # -> (bsz, n_heads_ssm, n_chunks, BT, D)

    C_ch = C.reshape(bsz, n_chunks, chunk_size, d_state)

    if state0 is None:
        state0 = jnp.zeros((bsz, n_heads_ssm, headdim, d_state), dtype=jnp.float32)
    state0 = _sanitize(state0)

    # move n_chunks to the leading (scan) axis
    to_scan = (
        jnp.moveaxis(y_diag_ch, 2, 0),      # (n_chunks, bsz, n_heads_ssm, BT, D)
        jnp.moveaxis(state_end, 2, 0),       # (n_chunks, bsz, n_heads_ssm, D, S)
        jnp.moveaxis(cumdecay, 2, 0),        # (n_chunks, bsz, n_heads_ssm, BT, D)
        jnp.moveaxis(C_ch, 1, 0),            # (n_chunks, bsz, BT, S)
    )

    def step(state_prev, inputs):
        y_diag_c, state_end_c, cumdecay_c, C_c = inputs

        decay_from_start = jnp.exp(jnp.clip(cumdecay_c, -20.0, 0.0))  # (bsz,h,BT,D)
        y_off_raw = jnp.einsum("bis,bhds->bhid", C_c, state_prev, precision=_HIGHEST)
        y_off = _sanitize(y_off_raw * decay_from_start)

        decay_chunk_end = jnp.exp(jnp.clip(cumdecay_c[:, :, -1, :], -20.0, 0.0))  # (bsz,h,D)
        state_new = _sanitize(state_prev * decay_chunk_end[..., None] + state_end_c)

        y_c = _sanitize(y_diag_c + y_off)

        return state_new, y_c

    step = jax.checkpoint(step)
    state_final, y_scanned = jax.lax.scan(step, state0, to_scan)

    y_scanned = jnp.moveaxis(y_scanned, 0, 2)   # -> (bsz, n_heads_ssm, n_chunks, BT, D)
    y = jnp.moveaxis(y_scanned, (1, 2), (3, 1)).reshape(bsz, L, n_heads_ssm, headdim)
    return y, state_final


def mamba2_pallas_forward(dt, x, B, C, A, chunk_size, state0=None, interpret=False):
    """Full staged pipeline: M2 (cumdecay) -> M3 (Y_diag, state_end) -> M4
    (inter-chunk combine). dt, x: (bsz, L, n_heads_ssm, headdim).
    B, C: (bsz, L, d_state). A: (n_heads_ssm,).

    Returns y: (bsz, L, n_heads_ssm, headdim), state_final: (bsz, n_heads_ssm, headdim, d_state).
    """
    cumdecay = build_chunk_cumdecay_pallas(dt, A, chunk_size, interpret=interpret)
    y_diag, state_end = intra_chunk_ssd_pallas(dt, x, B, C, cumdecay, chunk_size, interpret=interpret)
    y, state_final = mamba2_inter_chunk_combine(y_diag, state_end, cumdecay, C, chunk_size, state0=state0)
    return y, state_final
