"""
Fallback trainable wrapper -- Pallas forward + backward via jax.vjp on the
pure-JAX WY reference (gdn2_wy_reference.gdn2_chunked_wy_reference). Keep
this around as a known-good fallback / cross-check target: if the fused
Pallas backward (kernel_trainable_B6.gdn2_pallas_forward_trainable) is ever
suspected of a regression, swap this in and compare gradients.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .kernel_a_scores import BT
from .kernel_d_pipeline import gdn2_pallas_forward
from .gdn2_wy_reference import gdn2_chunked_wy_reference

_FINAL_CLIP = 1e4


def _final_sanitize(x):
    return jnp.nan_to_num(jnp.clip(x, -_FINAL_CLIP, _FINAL_CLIP), nan=0.0,
                           posinf=_FINAL_CLIP, neginf=-_FINAL_CLIP)


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _gdn2_core(q, k, v, w, b, g, scale, h0):
    return gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=h0)


def _gdn2_core_fwd(q, k, v, w, b, g, scale, h0):
    out = gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=h0)
    residuals = (q, k, v, w, b, g, h0)
    return out, residuals


def _gdn2_core_bwd(scale, residuals, cotangents):
    q, k, v, w, b, g, h0 = residuals
    do, dh_final = cotangents

    def ref_forward(q_, k_, v_, w_, b_, g_, h0_):
        return gdn2_chunked_wy_reference(q_, k_, v_, g_, b_, w_, scale, chunk_size=BT, h0=h0_)

    _, vjp_fn = jax.vjp(ref_forward, q, k, v, w, b, g, h0)
    dq, dk, dv, dw, db, dg, dh0 = vjp_fn((do, dh_final))

    # FIX (this packaging pass): same final sanitization pass added to the
    # B6 fused-Pallas backward -- applied here too so both backward
    # implementations give equally-safe gradients, and gradient comparisons
    # between them aren't skewed by one path clipping and the other not.
    dq, dk, dv, dw, db, dg, dh0 = map(
        _final_sanitize, (dq, dk, dv, dw, db, dg, dh0)
    )
    return dq, dk, dv, dw, db, dg, dh0


_gdn2_core.defvjp(_gdn2_core_fwd, _gdn2_core_bwd)


def gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale, h0=None):
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    return _gdn2_core(q, k, v, w, b, g, scale, h0)
