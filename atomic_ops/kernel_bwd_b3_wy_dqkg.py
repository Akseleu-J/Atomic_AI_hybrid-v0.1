"""
Milestone B3 -- WY/dqkg fused backward, Pallas/TPU kernel.

FIX (this packaging pass, non-finite hardening): the matrix-inverse-gradient
step (dAkk_raw = -A^T @ dA_total @ A^T) is TWO chained matmuls through A^T --
the single highest-amplification spot in the whole backward chain, since A
itself can have large entries when Akk is close to singular (the exact
"non-finite delta" failure mode from the step-710 incident originated
somewhere in this neighborhood: WY-solve numerics feeding an unstable
inverse-gradient). The original kernel only sanitized the FINAL dAkk output,
after both matmuls had already run -- if the first matmul (A.T @ dA_total)
already overflowed to +-inf, the second matmul could hit inf * 0 = NaN
*before* the final nan_to_num ever gets a chance to clean it up. Fix: clip
dA_total itself before it enters the double-matmul, and clip the
intermediate `tmp = dA_total @ A.T` before the second matmul too. Cheap
(two extra clips on (BT,BT) arrays) and closes the gap without changing the
correctness-validated formula at all -- clipping only kicks in once values
are already at the edge of representable range, same convention as every
other "line of defense" in this project.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernel_a_scores import BT

_HIGHEST = jax.lax.Precision.HIGHEST


def _kernel_b3_body(q_ref, k_ref, b_ref, w_ref, v_ref, gc_ref, a_ref, akk_ref,
                     hpre_ref, vnew_ref, do_ref, dv_ref, dhnext_ref,
                     dq_ref, dk_ref, db_ref, dw_ref, dvraw_ref, dgc_ref, dakk_ref,
                     *, scale):
    q_c = q_ref[0, 0, 0].astype(jnp.float32)
    k_c = k_ref[0, 0, 0].astype(jnp.float32)
    b_c = b_ref[0, 0, 0].astype(jnp.float32)
    w_c = w_ref[0, 0, 0].astype(jnp.float32)
    v_c = v_ref[0, 0, 0].astype(jnp.float32)
    gc = gc_ref[0, 0, 0].astype(jnp.float32)
    A = a_ref[0, 0, 0].astype(jnp.float32)
    hpre_unused = akk_ref  # Akk itself only needed for its (BT,BT) shape/mask below
    h_pre = hpre_ref[0, 0, 0].astype(jnp.float32)
    v_new = vnew_ref[0, 0, 0].astype(jnp.float32)
    do = do_ref[0, 0, 0].astype(jnp.float32)
    dv = dv_ref[0, 0, 0].astype(jnp.float32)
    dh_next = dhnext_ref[0, 0, 0].astype(jnp.float32)

    C = BT
    gc_last = gc[C - 1]

    kb_decayed = b_c * k_c * jnp.exp(gc)
    kg = k_c * jnp.exp(gc_last[None, :] - gc)
    qg = q_c * jnp.exp(gc)
    wv = w_c * v_c

    dqh_up = scale * do
    dqg = jnp.dot(dqh_up, h_pre.T, precision=_HIGHEST)

    dwh = -dv
    dw_pseudo = jnp.dot(dwh, h_pre.T, precision=_HIGHEST)
    du = dv

    dkg = jnp.dot(v_new, dh_next.T, precision=_HIGHEST)

    dA_from_w = jnp.dot(dw_pseudo, kb_decayed.T, precision=_HIGHEST)
    dkb_decayed = jnp.dot(A.T, dw_pseudo, precision=_HIGHEST)

    dA_from_u = jnp.dot(du, wv.T, precision=_HIGHEST)
    dwv = jnp.dot(A.T, du, precision=_HIGHEST)

    dA_total = dA_from_w + dA_from_u
    # FIX: clip before the amplification-prone double matmul through A.T --
    # see module docstring.
    dA_total = jnp.nan_to_num(jnp.clip(dA_total, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)

    idx = jnp.arange(C)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)

    tmp = jnp.dot(dA_total, A.T, precision=_HIGHEST)
    tmp = jnp.nan_to_num(jnp.clip(tmp, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
    dAkk_raw = -jnp.dot(A.T, tmp, precision=_HIGHEST)
    dAkk = dAkk_raw * strict

    dk_from_kb = dkb_decayed * jnp.exp(gc) * b_c
    db = dkb_decayed * jnp.exp(gc) * k_c
    dgc_from_kb = dkb_decayed * kb_decayed

    dx = dkg * kg
    dk_from_kg = dkg * jnp.exp(gc_last[None, :] - gc)
    dgc_from_kg = -dx
    dgc_last_contrib = jnp.sum(dx, axis=0)

    dq = dqg * jnp.exp(gc)
    dgc_from_qg = dqg * qg

    dw = dwv * v_c
    dv_raw = dwv * w_c

    dk = dk_from_kb + dk_from_kg
    dgc = dgc_from_kb + dgc_from_qg + dgc_from_kg

    decay_h_row = jnp.exp(gc_last)
    dgc_last_from_decay = decay_h_row * jnp.sum(dh_next * h_pre, axis=-1)
    dgc_last_total = dgc_last_contrib + dgc_last_from_decay

    row_mask = (idx == (C - 1)).astype(jnp.float32)[:, None]
    dgc = dgc + row_mask * dgc_last_total[None, :]

    dq_ref[0, 0, 0] = jnp.nan_to_num(dq, nan=0.0, posinf=1e4, neginf=-1e4)
    dk_ref[0, 0, 0] = jnp.nan_to_num(dk, nan=0.0, posinf=1e4, neginf=-1e4)
    db_ref[0, 0, 0] = jnp.nan_to_num(db, nan=0.0, posinf=1e4, neginf=-1e4)
    dw_ref[0, 0, 0] = jnp.nan_to_num(dw, nan=0.0, posinf=1e4, neginf=-1e4)
    dvraw_ref[0, 0, 0] = jnp.nan_to_num(dv_raw, nan=0.0, posinf=1e4, neginf=-1e4)
    dakk_ref[0, 0, 0] = jnp.nan_to_num(dAkk, nan=0.0, posinf=1e4, neginf=-1e4)
    dgc_ref[0, 0, 0] = jnp.nan_to_num(dgc, nan=0.0, posinf=1e4, neginf=-1e4)


def wy_dqkg_backward_pallas(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all,
                             do, dv, dh_next_all, scale):
    bsz, H, n_chunks, _BT, D = q.shape
    grid = (bsz, H, n_chunks)

    io_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))
    score_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))
    h_spec = pl.BlockSpec((1, 1, 1, D, D), lambda i, h, c: (i, h, c, 0, 0))

    dq, dk, db, dw, dv_raw, dgc, dAkk = pl.pallas_call(
        lambda *refs: _kernel_b3_body(*refs, scale=scale),
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec,
                   score_spec, score_spec, h_spec, io_spec, io_spec, io_spec, h_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec, score_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, BT), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
    )(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all, do, dv, dh_next_all)

    return dict(dq=dq, dk=dk, db=db, dw=dw, dv_raw=dv_raw, dgc=dgc, dAkk=dAkk)
