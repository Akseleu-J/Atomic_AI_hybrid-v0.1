"""
Milestone B2 -- backward for intra: dAqk and local dv_new contribution.
Unchanged from validated project version (already nan_to_num'd on output).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernel_a_scores import BT

_HIGHEST = jax.lax.Precision.HIGHEST


def _kernel_b2_body(aqk_ref, vnew_ref, do_ref, daqk_ref, dvnew_ref):
    Aqk = aqk_ref[0, 0, 0].astype(jnp.float32)
    v_new = vnew_ref[0, 0, 0].astype(jnp.float32)
    do = do_ref[0, 0, 0].astype(jnp.float32)

    idx = jnp.arange(BT)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)

    dAqk = jnp.dot(do, v_new.T, precision=_HIGHEST) * causal
    dv_new = jnp.dot(Aqk.T, do, precision=_HIGHEST)

    daqk_ref[0, 0, 0] = jnp.nan_to_num(dAqk, nan=0.0, posinf=1e4, neginf=-1e4)
    dvnew_ref[0, 0, 0] = jnp.nan_to_num(dv_new, nan=0.0, posinf=1e4, neginf=-1e4)


def dav_backward_pallas(Aqk, v_new, do):
    bsz, H, n_chunks, _BT, D = v_new.shape
    grid = (bsz, H, n_chunks)

    aqk_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))
    io_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))

    dAqk, dv_new = pl.pallas_call(
        _kernel_b2_body,
        grid=grid,
        in_specs=[aqk_spec, io_spec, io_spec],
        out_specs=[aqk_spec, io_spec],
        out_shape=[
            jax.ShapeDtypeStruct(Aqk.shape, jnp.float32),
            jax.ShapeDtypeStruct(v_new.shape, jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
    )(Aqk, v_new, do)

    return dAqk, dv_new
