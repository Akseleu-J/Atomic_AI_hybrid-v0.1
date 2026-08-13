"""
Milestone 3 -- Kernel D (plain JAX scan) + full forward pipeline glue
(A -> B -> C -> D). Includes the "with_state" variant (originally shipped
as a separate PATCH file) merged in directly here, since backward needs it.

FIX (this packaging pass, non-finite hardening): h0 is now sanitized on
entry to gdn2_inter_chunk_combine / _with_state. Previously only h_new was
sanitized each step; if a caller ever passed a non-finite h0 across a
segment boundary (e.g. carried state from a previous training step that
went non-finite before the auto-stop machinery caught it), that NaN would
silently poison the very first chunk. Sanitizing h0 costs nothing and closes
that gap.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from kernel_a_scores import build_chunk_scores_pallas, BT
from kernel_b_solve import wy_solve_pallas
from kernel_c_recompute import recompute_wy_pallas

_HIGHEST = jax.lax.Precision.HIGHEST


def _sanitize_h0(h0):
    return jnp.nan_to_num(jnp.clip(h0, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None):
    """Aqk: (B,H,n_chunks,BT,BT). w_pseudo,u,kg,qg: (B,H,n_chunks,BT,D).
    gc_last: (B,H,n_chunks,D). Returns o: (B,H,n_chunks,BT,D), h_final: (B,H,D,D).
    """
    bsz, H, n_chunks, _BT, D = w_pseudo.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    h0 = _sanitize_h0(h0)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, w_pseudo, u, kg, qg, gc_last))

    def step(h_pre, inputs):
        Aqk_c, w_pseudo_c, u_c, kg_c, qg_c, gclast_c = inputs
        wh = jnp.einsum("bhid,bhdv->bhiv", w_pseudo_c, h_pre, precision=_HIGHEST)
        v_new = u_c - wh
        qh = jnp.einsum("bhid,bhdv->bhiv", qg_c, h_pre, precision=_HIGHEST)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_c, v_new, precision=_HIGHEST)
        o_c = scale * qh + intra

        decay_h = jnp.exp(gclast_c)[..., None]
        write = jnp.einsum("bhid,bhiv->bhdv", kg_c, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write
        h_new = jnp.nan_to_num(jnp.clip(h_new, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
        o_c = jnp.nan_to_num(o_c, nan=0.0, posinf=1e4, neginf=-1e4)
        return h_new, o_c

    h_final, o_scanned = jax.lax.scan(step, h0, to_scan)
    o = jnp.moveaxis(o_scanned, 0, 2)
    return o, h_final


def gdn2_inter_chunk_combine_with_state(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None):
    """Same math as gdn2_inter_chunk_combine, plus per-chunk h_pre/v_new
    outputs needed by the backward chain (Milestone B1 reverse-scan, B3).
    (Originally shipped separately as kernel_d_pipeline_PATCH.py -- merged
    in here so forward and its backward-support variant live together.)
    """
    bsz, H, n_chunks, _BT, D = w_pseudo.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    h0 = _sanitize_h0(h0)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, w_pseudo, u, kg, qg, gc_last))

    def step(h_pre, inputs):
        Aqk_c, w_pseudo_c, u_c, kg_c, qg_c, gclast_c = inputs
        wh = jnp.einsum("bhid,bhdv->bhiv", w_pseudo_c, h_pre, precision=_HIGHEST)
        v_new = u_c - wh
        qh = jnp.einsum("bhid,bhdv->bhiv", qg_c, h_pre, precision=_HIGHEST)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_c, v_new, precision=_HIGHEST)
        o_c = scale * qh + intra

        decay_h = jnp.exp(gclast_c)[..., None]
        write = jnp.einsum("bhid,bhiv->bhdv", kg_c, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write

        h_new = jnp.nan_to_num(jnp.clip(h_new, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
        o_c = jnp.nan_to_num(o_c, nan=0.0, posinf=1e4, neginf=-1e4)

        return h_new, (o_c, h_pre, v_new)

    h_final, (o_scanned, h_pre_all, v_new_all) = jax.lax.scan(step, h0, to_scan)
    o = jnp.moveaxis(o_scanned, 0, 2)
    return o, h_final, h_pre_all, v_new_all


def gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=None):
    """Full staged pipeline: Kernel A -> B -> C -> D. q,k,v,w,b,g: (B,L,H,D).
    Returns o: (B,L,H,D), h_final: (B,H,D,D).

    NOTE: no shard_map here -- sharding is the CALLER's responsibility
    (GatedDeltaNet2J in model.py), see project handoff sec 5.6.
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    A = wy_solve_pallas(Akk)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    o_chunks, h_final = gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0)

    n_chunks = L // BT
    o = jnp.moveaxis(o_chunks, 1, 3)
    o = o.reshape(bsz, n_chunks * BT, H, D)
    return o, h_final


def gdn2_pallas_forward_with_residuals(q, k, v, w, b, g, scale, h0=None):
    """Same as gdn2_pallas_forward, but also returns per-chunk residuals
    (Aqk, h_pre_all, v_new_all, ...) needed by the backward Pallas kernels.
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    A = wy_solve_pallas(Akk)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    o_chunks, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0
    )
    n_chunks = L // BT
    o = jnp.moveaxis(o_chunks, 1, 3).reshape(bsz, n_chunks * BT, H, D)
    residuals = dict(Aqk=Aqk, Akk=Akk, A=A, h_pre_all=h_pre_all, v_new_all=v_new_all,
                      w_pseudo=w_pseudo, u=u, kg=kg, qg=qg, gc_last=gc_last)
    return o, h_final, residuals
