"""
Milestone MB0 -- hand-derived backward for one Mamba2 SSD chunk
(`_chunk_ssd` in mamba2_ssd_reference.py), written by hand in plain JAX --
NO jax.vjp, NO Pallas. This is the cross-check target for MB1-MB4 (the
eventual fused Pallas backward): every future Pallas kernel's output must
match THIS function (which itself is validated against jax.vjp on
mamba2_ssd_reference.py in test_mamba2_bwd_reference.py) before being
trusted, same "hand derivation -> numpy/jax.vjp cross-check -> Pallas port
-> TPU test" discipline as the rest of atomic_ops/.

Forward being differentiated (see mamba2_ssd_reference.py's _chunk_ssd for
the authoritative version -- this docstring's formulas mirror it exactly,
including its clip/sanitize placement):

  dA_exponent   = clip(dt_c * A[h], -20, 20)
  cumdecay      = clip(tril_cumsum(dA_exponent), -1e4, 1e4)
  decay_diff    = cumdecay_i - cumdecay_j
  L             = exp(clip(decay_diff, -20, 20)) * causal
  BC_inner      = C_c . B_c^T                      (d_state contraction)
  dBx           = dt_c * x_c
  y_diag[i]     = sum_j L[i,j]*BC_inner[i,j]*dBx[j]
  decay_to_end  = exp(clip(cumdecay[-1] - cumdecay, -20, 20))
  write         = dBx * decay_to_end
  state_end     = sum_j B_c[j] (x) write[j]
  decay_h       = exp(clip(cumdecay, -20, 0))
  y_off[i]      = (C_c[i] . state_prev) * decay_h[i]
  decay_chunk_end = exp(clip(cumdecay[-1], -20, 0))
  state_new     = state_prev*decay_chunk_end + state_end
  y_c           = y_diag + y_off

Adjoint convention: every jnp.clip in the forward is treated with a
straight-through gradient (pass-through inside the clip window, zero
outside) -- this is EXACTLY what JAX's own autodiff produces for
jnp.clip, so a correct hand derivation must match jax.vjp bit-for-bit up
to floating point noise. This is the actual cross-check performed in
test_mamba2_bwd_reference.py.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .mamba2_ssd_reference import _chunk_ssd, _sanitize
 
_HIGHEST = jax.lax.Precision.HIGHEST
 
 
def _clip_mask(x, lo, hi):
    """Straight-through gradient mask for jnp.clip(x, lo, hi) -- 1.0 inside
    the window (gradient passes through unchanged), 0.0 outside (gradient
    is killed, matching what clip's own JVP/VJP rule does)."""
    return ((x >= lo) & (x <= hi)).astype(jnp.float32)
 
 
def _chunk_ssd_bwd(dt_c, A, B_c, C_c, x_c, state_prev, dy_c, dstate_new):
    """Hand-derived backward for one chunk. Shapes match _chunk_ssd's
    forward exactly (see mamba2_ssd_reference.py):
      dt_c: (b,C,h,d)  A: (h,)  B_c,C_c: (b,C,s)  x_c: (b,C,h,d)
      state_prev: (b,h,d,s)
      dy_c: (b,C,h,d)  dstate_new: (b,h,d,s)  (incoming cotangents)
    Returns: ddt_c, dB_c, dC_c, dx_c, dA, dstate_prev -- same shapes as the
    corresponding forward inputs.
    """
    f32 = jnp.float32
    dt_c = dt_c.astype(f32)
    B_c = B_c.astype(f32)
    C_c = C_c.astype(f32)
    x_c = x_c.astype(f32)
    state_prev = state_prev.astype(f32)
    Cs = dt_c.shape[1]
 
    # ---- re-run the forward (cheap relative to the matmuls; keeps this
    # function self-contained and avoids a fragile hand-plumbed residual
    # tuple, same "recompute, don't stash everything" tradeoff kernel_c
    # already makes for w_pseudo/u in GDN-2) ----
    dA_exponent_raw = jnp.einsum("bchd,h->bchd", dt_c, A.astype(f32))
    dA_exponent = _sanitize(dA_exponent_raw, clip=20.0)
 
    idx = jnp.arange(Cs)
    tril_ones = (idx[:, None] >= idx[None, :]).astype(f32)
    cumdecay_raw = jnp.einsum("ij,bjhd->bihd", tril_ones, dA_exponent, precision=_HIGHEST)
    cumdecay = _sanitize(cumdecay_raw, clip=1e4)
 
    decay_diff_raw = cumdecay[:, :, None, :, :] - cumdecay[:, None, :, :, :]
    causal = (idx[:, None] >= idx[None, :]).astype(f32)[None, :, :, None, None]
    decay_diff_clipped = jnp.clip(decay_diff_raw, -20.0, 20.0)
    L_raw = jnp.exp(decay_diff_clipped) * causal
    L = _sanitize(L_raw, clip=1e6)
 
    BC_inner_raw = jnp.einsum("bis,bjs->bij", C_c, B_c, precision=_HIGHEST)
    BC_inner = _sanitize(BC_inner_raw)
 
    dBx = _sanitize(dt_c * x_c)
 
    weight = L * BC_inner[:, :, :, None, None]
 
    decay_to_end_raw = cumdecay[:, -1:, :, :] - cumdecay
    decay_to_end_clipped = jnp.clip(decay_to_end_raw, -20.0, 20.0)
    decay_to_end = _sanitize(jnp.exp(decay_to_end_clipped), clip=1e6)
    write = _sanitize(dBx * decay_to_end)
 
    decay_h_raw = jnp.clip(cumdecay, -20.0, 0.0)
    decay_h = jnp.exp(decay_h_raw)
 
    y_off_raw = jnp.einsum("bis,bhds->bihd", C_c, state_prev, precision=_HIGHEST)
 
    decay_chunk_end_raw = jnp.clip(cumdecay[:, -1], -20.0, 0.0)
    decay_chunk_end = jnp.exp(decay_chunk_end_raw)
 
    # ================= REVERSE PASS =================
    dy_diag = dy_c
    dy_off = dy_c
 
    # state_new = state_prev*decay_chunk_end[...,None] + state_end
    dstate_prev = dstate_new * decay_chunk_end[..., None]
    d_decay_chunk_end = jnp.sum(dstate_new * state_prev, axis=-1)          # (b,h,d)
    dstate_end = dstate_new
 
    # decay_chunk_end = exp(clip(cumdecay[-1], -20, 0))
    d_cumdecay_last_a = d_decay_chunk_end * decay_chunk_end * _clip_mask(cumdecay[:, -1], -20.0, 0.0)
 
    # y_off = (C_c . state_prev) * decay_h
    dy_off_raw = dy_off * decay_h
    d_decay_h = jnp.sum(dy_off * y_off_raw, axis=... ) if False else jnp.sum(dy_off * y_off_raw, axis=-1, keepdims=False) * 0  # placeholder, corrected below
    # NOTE: decay_h is (b,C,h,d) NOT reduced over d -- elementwise product with y_off_raw (b,C,h,d)
    d_decay_h = dy_off * y_off_raw
    dC_c_1 = jnp.einsum("bihd,bhds->bis", dy_off_raw, state_prev, precision=_HIGHEST)
    dstate_prev = dstate_prev + jnp.einsum("bihd,bis->bhds", dy_off_raw, C_c, precision=_HIGHEST)
 
    # decay_h = exp(clip(cumdecay, -20, 0))
    d_cumdecay_from_yoff = d_decay_h * decay_h * _clip_mask(cumdecay, -20.0, 0.0)
 
    # state_end = sum_j B_c[j] (x) write[j]
    dwrite = jnp.einsum("bhds,bcs->bchd", dstate_end, B_c, precision=_HIGHEST)
    dB_c_1 = jnp.einsum("bhds,bchd->bcs", dstate_end, write, precision=_HIGHEST)
 
    # write = dBx * decay_to_end
    d_dBx_1 = dwrite * decay_to_end
    d_decay_to_end = dwrite * dBx
 
    # decay_to_end = exp(clip(cumdecay[-1:] - cumdecay, -20, 20))
    d_decay_to_end_diff = d_decay_to_end * decay_to_end * _clip_mask(decay_to_end_raw, -20.0, 20.0)
    d_cumdecay_last_b = jnp.sum(d_decay_to_end_diff, axis=1)                # broadcast over C -> sum
    d_cumdecay_from_decay_to_end = -d_decay_to_end_diff
 
    # y_diag = sum_j weight[i,j] * dBx[j]
    dweight = jnp.einsum("bihd,bjhd->bijhd", dy_diag, dBx, precision=_HIGHEST)
    d_dBx_2 = jnp.einsum("bijhd,bihd->bjhd", weight, dy_diag, precision=_HIGHEST)
 
    # weight = L * BC_inner[...,None,None]
    dL = dweight * BC_inner[:, :, :, None, None]
    dBC_inner = jnp.sum(dweight * L, axis=(-2, -1))
 
    # BC_inner = C_c . B_c^T
    dC_c_2 = jnp.einsum("bij,bjs->bis", dBC_inner, B_c, precision=_HIGHEST)
    dB_c_2 = jnp.einsum("bij,bis->bjs", dBC_inner, C_c, precision=_HIGHEST)
 
    # L = exp(clip(decay_diff, -20, 20)) * causal
    d_decay_diff = dL * L * _clip_mask(decay_diff_raw, -20.0, 20.0)
 
    # decay_diff[i,j] = cumdecay_i - cumdecay_j
    d_cumdecay_i = jnp.sum(d_decay_diff, axis=2)
    d_cumdecay_j = -jnp.sum(d_decay_diff, axis=1)
 
    # dBx = dt_c * x_c
    d_dBx_total = _sanitize(d_dBx_1 + d_dBx_2)
    ddt_c_1 = d_dBx_total * x_c
    dx_c = d_dBx_total * dt_c
 
    # ---- combine every contribution to cumdecay (b,C,h,d) ----
    d_cumdecay_total = _sanitize(
        d_cumdecay_i + d_cumdecay_j + d_cumdecay_from_yoff + d_cumdecay_from_decay_to_end
    )
    # the "last row" (index C-1) also receives two extra scalar-per-(b,h,d)
    # contributions from decay_chunk_end and decay_to_end's broadcast term.
    row_mask = (idx == (Cs - 1)).astype(jnp.float32)[None, :, None, None]
    d_cumdecay_total = d_cumdecay_total + row_mask * (d_cumdecay_last_a + d_cumdecay_last_b)[:, None, :, :]
 
    # cumdecay = clip(tril_cumsum(dA_exponent), -1e4, 1e4)  -- reverse cumsum,
    # same trick as GDN-2's kernel_bwd_b5_reverse_cumsum.py.
    d_cumdecay_total = d_cumdecay_total * _clip_mask(cumdecay_raw, -1e4, 1e4)
    triu_ones = (idx[:, None] <= idx[None, :]).astype(jnp.float32)
    d_dA_exponent = jnp.einsum("ij,bjhd->bihd", triu_ones, d_cumdecay_total, precision=_HIGHEST)
 
    # dA_exponent = clip(dt_c * A[h], -20, 20)
    d_dA_exponent = d_dA_exponent * _clip_mask(dA_exponent_raw, -20.0, 20.0)
    ddt_c_2 = jnp.einsum("bchd,h->bchd", d_dA_exponent, A.astype(f32))
    dA = jnp.sum(d_dA_exponent * dt_c, axis=(0, 1, 3))
 
    ddt_c = _sanitize(ddt_c_1 + ddt_c_2)
    dB_c = _sanitize(dB_c_1 + dB_c_2)
    dC_c = _sanitize(dC_c_1 + dC_c_2)
    dx_c = _sanitize(dx_c)
    dA = _sanitize(dA)
    dstate_prev = _sanitize(dstate_prev)
 
    return ddt_c, dB_c, dC_c, dx_c, dA, dstate_prev
 
 
def chunk_ssd_bwd_scan(dt, A, B, C, x, chunk_size, do, dstate_final, state0=None):
    """Full-sequence hand-derived backward: reverse lax.scan over chunks,
    calling _chunk_ssd_bwd per chunk -- mirrors mamba2_ssd_reference's own
    forward scan structure, and (once MB1-MB4 exist) is the reference this
    project's fused Pallas backward chain must match, exactly the role
    gdn2_wy_reference.gdn2_chunked_wy_reference plays for GDN-2.
 
    dt,x: (b,l,h,d). A: (h,). B,C: (b,l,s). do: (b,l,h,d) cotangent for y.
    dstate_final: (b,h,d,s) cotangent for the final carried state.
    Returns ddt, dB, dC, dx, dA, dstate0 -- same shapes as forward inputs.
    """
    b, l, h, d = dt.shape
    s = B.shape[-1]
    assert l % chunk_size == 0
    n_chunks = l // chunk_size
 
    if state0 is None:
        state0 = jnp.zeros((b, h, d, s), dtype=jnp.float32)
 
    def to_chunks(t):
        shp = t.shape
        t = t.reshape(b, n_chunks, chunk_size, *shp[2:])
        return jnp.moveaxis(t, 1, 0)
 
    dt_ch, B_ch, C_ch, x_ch, do_ch = map(to_chunks, (dt, B, C, x, do))
 
    # ---- forward re-run to recover state_prev per chunk (needed as a
    # residual for the backward -- same "recompute forward, don't stash
    # everything" tradeoff already used throughout atomic_ops/) ----
    def fwd_step(state_prev, inputs):
        dt_c, B_c, C_c, x_c = inputs
        y_c, state_new = _chunk_ssd(dt_c, A, B_c, C_c, x_c, state_prev)
        return state_new, state_prev  # stash state_prev (pre-update) for bwd
 
    _, state_prev_all = jax.lax.scan(fwd_step, state0, (dt_ch, B_ch, C_ch, x_ch))
 
    def bwd_step(dstate_carry, inputs):
        dt_c, B_c, C_c, x_c, do_c, state_prev_c = inputs
        ddt_c, dB_c, dC_c, dx_c, dA_c, dstate_prev_c = _chunk_ssd_bwd(
            dt_c, A, B_c, C_c, x_c, state_prev_c, do_c, dstate_carry
        )
        return dstate_prev_c, (ddt_c, dB_c, dC_c, dx_c, dA_c)
 
    dstate0, (ddt_rev, dB_rev, dC_rev, dx_rev, dA_rev) = jax.lax.scan(
        bwd_step, dstate_final,
        (dt_ch, B_ch, C_ch, x_ch, do_ch, state_prev_all),
        reverse=True,
    )
 
    def from_chunks(t):
        # t: (n_chunks, b, C, ...) -- move n_chunks next to b, then merge
        # (n_chunks, C) -> l. Previous version reshaped to
        # (b, l, *t.shape[2:]) right after moveaxis, but at that point
        # t.shape is (b, n_chunks, C, ...) -- t.shape[2:] still includes
        # C, leaving a spurious extra axis instead of merging n_chunks*C
        # into l. Use t.shape[3:] (the axes AFTER C) instead.
        t = jnp.moveaxis(t, 0, 1)
        return t.reshape(b, l, *t.shape[3:])
 
    ddt = from_chunks(ddt_rev)
    dB = from_chunks(dB_rev)
    dC = from_chunks(dC_rev)
    dx = from_chunks(dx_rev)
    dA = jnp.sum(dA_rev, axis=0)
    dstate0 = _sanitize(dstate0)
 
    return ddt, dB, dC, dx, dA, dstate0
