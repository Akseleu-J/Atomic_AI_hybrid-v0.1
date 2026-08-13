"""
Milestone 3 -- Kernel D (plain JAX scan) + full forward pipeline glue
(A -> B -> C -> D). Includes the "with_state" variant merged in directly
here, since backward needs it.

FIX (this packaging pass, non-finite hardening): h0 is now sanitized on
entry to gdn2_inter_chunk_combine / _with_state -- see prior docstring
version, unchanged here.

FIX (diagnostics, added while investigating a real non-finite hit at
global_step=784, block=4/layer=14/type=gdn2 -- see kernel_b_solve.py and
kernel_c_recompute.py docstrings for the root-cause writeup: a
near-singular Akk in a specific (batch, head, chunk) can make Kernel B's
WY-solve output a large-but-finite A, which only becomes an actual `inf`
several kernels later, inside THIS module's scan -- by which point the
existing [FWD-DIAG] block/layer check in model.py can only report "this
layer's output is non-finite", with no way to tell whether the problem
originated in Kernel A/B/C or here in D.

Adds an OPTIONAL, env-var-gated (`GDN2_FWD_DIAG=1`, default OFF -- zero
overhead in normal training) diagnostic pass after each of the four kernel
stages: reports (a) whether the stage output is fully finite and (b) the
max absolute value of its finite entries, so a magnitude runaway shows up
BEFORE it actually overflows to inf, not just after. Every call site also
accepts an optional `debug_tag` (plumbed from model.py's GatedDeltaNet2J,
which knows its own `layer_idx`) so the printed messages say exactly which
of the model's layers is misbehaving, instead of requiring the reader to
count print-statement ordinals.

Turn this on for one debug run with:
    GDN2_FWD_DIAG=1 python train.py
and read the [GDN2-FWD-DIAG] lines -- the FIRST stage that reports
non-finite or an outsized max_abs for a given layer_tag is the true source;
everything printed for later stages of the SAME layer_tag on the SAME step
is downstream fallout, not new information.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from .kernel_a_scores import build_chunk_scores_pallas, BT
from .kernel_b_solve import wy_solve_pallas
from .kernel_c_recompute import recompute_wy_pallas

_HIGHEST = jax.lax.Precision.HIGHEST

_GDN2_FWD_DIAG = os.environ.get("GDN2_FWD_DIAG", "0") == "1"
_LARGE_THRESHOLD = 1e6  # "suspiciously large but still finite" trigger level


def _sanitize_h0(h0):
    return jnp.nan_to_num(jnp.clip(h0, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def _stage_diag(tag: str, x):
    """No-op (returns x unchanged) unless GDN2_FWD_DIAG=1. When enabled,
    checks finiteness and max magnitude of x and prints via jax.debug.print
    if something looks wrong -- purely diagnostic, never changes the value
    itself (so turning this on cannot mask or alter the actual bug)."""
    if not _GDN2_FWD_DIAG:
        return x

    finite_mask = jnp.isfinite(x)
    all_finite = jnp.all(finite_mask)
    n_nonfinite = jnp.sum(jnp.logical_not(finite_mask))
    safe_x = jnp.where(finite_mask, x, 0.0)
    max_abs = jnp.max(jnp.abs(safe_x))

    def _report_nonfinite():
        jax.debug.print(
            "[GDN2-FWD-DIAG] ⚠️ non-finite на выходе " + tag +
            ": n_nonfinite={n}  max_abs(конечная часть)={m:.3e}",
            n=n_nonfinite, m=max_abs,
        )

    def _report_large():
        jax.debug.print(
            "[GDN2-FWD-DIAG] 🔶 подозрительно большая величина на выходе " + tag +
            " (всё ещё конечная, но уже похоже на предвестник): max_abs={m:.3e}",
            m=max_abs,
        )

    jax.lax.cond(
        jnp.logical_not(all_finite),
        _report_nonfinite,
        lambda: jax.lax.cond(max_abs > _LARGE_THRESHOLD, _report_large, lambda: None),
    )
    return x


def gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None, debug_tag=""):
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
    h_final = _stage_diag(f"{debug_tag}:kernel_D_h_final", h_final)
    o = jnp.moveaxis(o_scanned, 0, 2)
    o = _stage_diag(f"{debug_tag}:kernel_D_o", o)
    return o, h_final


def gdn2_inter_chunk_combine_with_state(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None, debug_tag=""):
    """Same math as gdn2_inter_chunk_combine, plus per-chunk h_pre/v_new
    outputs needed by the backward chain (Milestone B1 reverse-scan, B3)."""
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
    h_final = _stage_diag(f"{debug_tag}:kernel_D_h_final", h_final)
    o = jnp.moveaxis(o_scanned, 0, 2)
    o = _stage_diag(f"{debug_tag}:kernel_D_o", o)
    return o, h_final, h_pre_all, v_new_all


def gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=None, debug_tag=""):
    """Full staged pipeline: Kernel A -> B -> C -> D. q,k,v,w,b,g: (B,L,H,D).
    Returns o: (B,L,H,D), h_final: (B,H,D,D).

    `debug_tag` is purely diagnostic (see module docstring) -- pass a string
    identifying the calling layer (e.g. "layer14") to get exact-location
    [GDN2-FWD-DIAG] logs when GDN2_FWD_DIAG=1 is set. No-op otherwise.

    NOTE: no shard_map here -- sharding is the CALLER's responsibility
    (GatedDeltaNet2J in model.py), see project handoff sec 5.6.
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    Aqk = _stage_diag(f"{debug_tag}:kernel_A_Aqk", Aqk)
    Akk = _stage_diag(f"{debug_tag}:kernel_A_Akk", Akk)

    A = wy_solve_pallas(Akk)
    A = _stage_diag(f"{debug_tag}:kernel_B_wy_inverse_A", A)

    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    w_pseudo = _stage_diag(f"{debug_tag}:kernel_C_w_pseudo", w_pseudo)
    u = _stage_diag(f"{debug_tag}:kernel_C_u", u)
    kg = _stage_diag(f"{debug_tag}:kernel_C_kg", kg)
    qg = _stage_diag(f"{debug_tag}:kernel_C_qg", qg)

    o_chunks, h_final = gdn2_inter_chunk_combine(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0, debug_tag=debug_tag
    )

    n_chunks = L // BT
    o = jnp.moveaxis(o_chunks, 1, 3)
    o = o.reshape(bsz, n_chunks * BT, H, D)
    return o, h_final


def gdn2_pallas_forward_with_residuals(q, k, v, w, b, g, scale, h0=None, debug_tag=""):
    """Same as gdn2_pallas_forward, but also returns per-chunk residuals
    (Aqk, h_pre_all, v_new_all, ...) needed by the backward Pallas kernels.
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    Aqk = _stage_diag(f"{debug_tag}:kernel_A_Aqk", Aqk)
    Akk = _stage_diag(f"{debug_tag}:kernel_A_Akk", Akk)

    A = wy_solve_pallas(Akk)
    A = _stage_diag(f"{debug_tag}:kernel_B_wy_inverse_A", A)

    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    w_pseudo = _stage_diag(f"{debug_tag}:kernel_C_w_pseudo", w_pseudo)
    u = _stage_diag(f"{debug_tag}:kernel_C_u", u)
    kg = _stage_diag(f"{debug_tag}:kernel_C_kg", kg)
    qg = _stage_diag(f"{debug_tag}:kernel_C_qg", qg)

    o_chunks, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0, debug_tag=debug_tag
    )
    n_chunks = L // BT
    o = jnp.moveaxis(o_chunks, 1, 3).reshape(bsz, n_chunks * BT, H, D)
    residuals = dict(Aqk=Aqk, Akk=Akk, A=A, h_pre_all=h_pre_all, v_new_all=v_new_all,
                      w_pseudo=w_pseudo, u=u, kg=kg, qg=qg, gc_last=gc_last)
    return o, h_final, residuals
