"""
Milestone B4 -- intra-score-build backward (Kernel A's backward).

FIX (this packaging pass, non-finite hardening): dq/dk/db/dgc are each
written to via READ-MODIFY-WRITE accumulation across multiple (si,sj)
sub-block iterations (up to 3 additive contributions per region for
N_SUB=2). The original kernel only sanitized the FINAL accumulated value
after the whole loop finished. If any single contribution overflowed to
+-inf mid-loop, a LATER contribution of the opposite sign landing in the
same region produces inf + (-inf) = NaN, which no downstream nan_to_num can
recover (NaN is a fixed point of clip/nan_to_num only at the very last
write -- once it's NaN, it stays NaN through every subsequent read-add-write
in the loop). Fix: clip after every accumulation write, not just at the
end -- keeps each partial sum inside the representable range so a later
opposite-sign contribution cannot produce inf-inf. Same reasoning as the B3
fix in kernel_bwd_b3_wy_dqkg.py.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .kernel_a_scores import BT, BC, N_SUB

_HIGHEST = jax.lax.Precision.HIGHEST
_CLIP = 20.0
_ACC_CLIP = 1e4

assert N_SUB == 2, "Kernel B4 currently implements only the 2-subblock (BT=2*BC) case, matching Kernel A/B."


def _clip_acc(x):
    return jnp.nan_to_num(jnp.clip(x, -_ACC_CLIP, _ACC_CLIP), nan=0.0, posinf=_ACC_CLIP, neginf=-_ACC_CLIP)


def _dL_pair_sum(dM, edecay, R):
    tmp = dM[:, :, None] * edecay
    tmp = tmp * R[None, :, :]
    return jnp.sum(tmp, axis=1)


def _dR_pair_sum(dM, edecay, L):
    tmp = dM[:, :, None] * edecay
    tmp = tmp * L[:, None, :]
    return jnp.sum(tmp, axis=0)


def _dgc_pair_sum(dM, edecay, L, R, clipmask):
    weight = dM[:, :, None] * L[:, None, :] * R[None, :, :] * edecay * clipmask
    dgc_i = jnp.sum(weight, axis=1)
    dgc_j = -jnp.sum(weight, axis=0)
    return dgc_i, dgc_j


def _kernel_b4_body(q_ref, k_ref, b_ref, g_ref, daqk_ref, dakk_ref,
                     dq_ref, dk_ref, db_ref, dgc_ref, *, scale):
    q_full = q_ref[0, 0, 0].astype(jnp.float32)
    k_full = k_ref[0, 0, 0].astype(jnp.float32)
    b_full = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)
    dAqk = daqk_ref[0, 0, 0].astype(jnp.float32)
    dAkk = dakk_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(BT)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

    bk_full = b_full * k_full

    dq_ref[0, 0, 0] = jnp.zeros_like(q_full)
    dk_ref[0, 0, 0] = jnp.zeros_like(k_full)
    db_ref[0, 0, 0] = jnp.zeros_like(k_full)   # scratch: accumulates dbk during the loop
    dgc_ref[0, 0, 0] = jnp.zeros_like(g_raw)

    for si in range(N_SUB):
        for sj in range(si + 1):
            i0, i1 = si * BC, (si + 1) * BC
            j0, j1 = sj * BC, (sj + 1) * BC

            q_i = q_full[i0:i1]
            k_j = k_full[j0:j1]
            bk_i = bk_full[i0:i1]
            gc_i = gc[i0:i1]
            gc_j = gc[j0:j1]

            decay_diff = gc_i[:, None, :] - gc_j[None, :, :]
            clipmask = ((decay_diff >= -_CLIP) & (decay_diff <= _CLIP)).astype(jnp.float32)
            edecay = jnp.exp(jnp.clip(decay_diff, -_CLIP, _CLIP))

            dM_qk = dAqk[i0:i1, j0:j1]
            dM_kk = dAkk[i0:i1, j0:j1]
            if si == sj:
                idx = jnp.arange(BC)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
                dM_qk = dM_qk * causal
                dM_kk = dM_kk * strict

            L_qk = scale * q_i
            R_qk = k_j
            dL_qk = _dL_pair_sum(dM_qk, edecay, R_qk)
            dR_qk = _dR_pair_sum(dM_qk, edecay, L_qk)
            dgc_i_qk, dgc_j_qk = _dgc_pair_sum(dM_qk, edecay, L_qk, R_qk, clipmask)

            L_kk = bk_i
            R_kk = k_j
            dL_kk = _dL_pair_sum(dM_kk, edecay, R_kk)
            dR_kk = _dR_pair_sum(dM_kk, edecay, L_kk)
            dgc_i_kk, dgc_j_kk = _dgc_pair_sum(dM_kk, edecay, L_kk, R_kk, clipmask)

            # FIX: clip every accumulation write, not just the final one --
            # see module docstring. Without this, an inf from one (si,sj)
            # iteration can meet an opposite-sign inf from another iteration
            # writing the same region and produce an unrecoverable NaN.
            dq_ref[0, 0, 0, i0:i1] = _clip_acc(dq_ref[0, 0, 0, i0:i1] + dL_qk * scale)
            db_ref[0, 0, 0, i0:i1] = _clip_acc(db_ref[0, 0, 0, i0:i1] + dL_kk)
            dk_ref[0, 0, 0, j0:j1] = _clip_acc(dk_ref[0, 0, 0, j0:j1] + dR_qk + dR_kk)
            dgc_ref[0, 0, 0, i0:i1] = _clip_acc(dgc_ref[0, 0, 0, i0:i1] + dgc_i_qk + dgc_i_kk)
            dgc_ref[0, 0, 0, j0:j1] = _clip_acc(dgc_ref[0, 0, 0, j0:j1] + dgc_j_qk + dgc_j_kk)

    dbk_final = db_ref[0, 0, 0]
    dk_final = dk_ref[0, 0, 0] + dbk_final * b_full
    db_final = dbk_final * k_full
    dq_final = dq_ref[0, 0, 0]
    dgc_final = dgc_ref[0, 0, 0]

    dq_ref[0, 0, 0] = jnp.nan_to_num(jnp.clip(dq_final, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
    dk_ref[0, 0, 0] = jnp.nan_to_num(jnp.clip(dk_final, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
    db_ref[0, 0, 0] = jnp.nan_to_num(jnp.clip(db_final, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
    dgc_ref[0, 0, 0] = jnp.nan_to_num(jnp.clip(dgc_final, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def intra_backward_pallas(dAqk, dAkk, q, k, b, g, scale):
    bsz, L, H, D = q.shape
    assert D == 128, f"Kernel B4 assumes d_head=128 (MXU tile); got D={D}."
    assert L % BT == 0, f"seq_len={L} must be divisible by BT={BT}."
    n_chunks = L // BT

    def reshape_in(t):
        t = t.reshape(bsz, n_chunks, BT, H, D)
        return jnp.moveaxis(t, (1, 3), (2, 1))

    q_r, k_r, b_r, g_r = map(reshape_in, (q, k, b, g))

    grid = (bsz, H, n_chunks)
    io_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))
    score_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))

    dq, dk, db, dgc = pl.pallas_call(
        lambda *refs: _kernel_b4_body(*refs, scale=scale),
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, score_spec, score_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=150 * 1024 * 1024),
    )(q_r, k_r, b_r, g_r, dAqk, dAkk)

    return dq, dk, db, dgc
