"""
Milestone B1 -- backward through the inter-chunk state recurrence.
Unchanged from validated project version (already has clip/nan_to_num on
dv_new_c and dh_pre_c each reverse-scan step).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


def gdn2_dhu_backward(do, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht=None):
    bsz, H, n_chunks, BT, D = qg.shape
    if dht is None:
        dht = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    dht = jnp.nan_to_num(jnp.clip(dht, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (do, dv_partial, w_pseudo, qg, kg, gc_last))

    def step(dh_carry, inputs):
        do_c, dvp_c, wp_c, qg_c, kg_c, gclast_c = inputs
        decay_c = jnp.exp(gclast_c)[..., None]

        dqh = scale * do_c
        contrib_from_output = jnp.einsum("bhid,bhiv->bhdv", qg_c, dqh, precision=_HIGHEST)
        contrib_from_state = dh_carry * decay_c

        dv_write = jnp.einsum("bhid,bhdv->bhiv", kg_c, dh_carry, precision=_HIGHEST)
        dv_new_c = dvp_c + dv_write
        dv_new_c = jnp.nan_to_num(jnp.clip(dv_new_c, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)

        contrib_from_vnew = -jnp.einsum("bhjd,bhjv->bhdv", wp_c, dv_new_c, precision=_HIGHEST)

        dh_pre_c = contrib_from_output + contrib_from_state + contrib_from_vnew
        dh_pre_c = jnp.nan_to_num(jnp.clip(dh_pre_c, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)

        return dh_pre_c, (dh_pre_c, dv_new_c)

    dh0, (dh_all_rev, dv_all_rev) = jax.lax.scan(step, dht, to_scan, reverse=True)
    dh_all = jnp.moveaxis(dh_all_rev, 0, 2)
    dv_all = jnp.moveaxis(dv_all_rev, 0, 2)
    return dh_all, dh0, dv_all
