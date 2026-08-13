"""
Milestone B6 -- orchestration: replaces the "cheat" backward (jax.vjp on the
pure-JAX reference) with the honest fused-Pallas chain B2 -> B1 -> B3 -> B4
-> B5.

FIX (this packaging pass, non-finite hardening): every individual kernel in
the chain (B1-B5) already sanitizes ITS OWN outputs. What was missing is a
sanitization pass on the COMBINED gradients this orchestrator produces --
`dq = b3_out["dq"] + dq4` etc. Each summand is independently clipped to
+-1e4, but nothing stopped the SUM of two independently-clipped-but-
opposite-signed-and-still-large values, or a chain of such sums across
db/dw/dg, from drifting the optimizer step to a large-but-finite outlier
that (per the step-710 incident notes) can itself be the seed of a future
non-finite cascade several steps later. Adding one final clip/nan_to_num
pass right before the values leave this function is the same "belt and
braces" pattern used everywhere else in this project, applied at the one
place it was still missing: the actual custom_vjp return boundary.

FIX #2 (dtype mismatch, found when this backward was actually wired into
real training in model.py): custom_vjp requires the bwd function to return
cotangents with the SAME dtype as the corresponding forward input. In real
usage (GatedDeltaNet2J in model.py), q/k/v/w_gate/b_gate arrive as
bfloat16, while g (log-decay) arrives as float32 (computed explicitly in
fp32 for precision -- see model.py's GatedDeltaNet2J.__call__). Every
intermediate value in this backward chain (B1-B5) is float32, and the
final-sanitize pass did NOT change that -- so dq/dk/dv/dw/db were returned
as float32 for bfloat16 primal inputs. JAX accepts that silently at the
custom_vjp boundary itself, but training then crashes one level up the
graph the moment that float32 cotangent has to combine with a
bfloat16-typed gradient elsewhere (e.g. through a bf16 nn.Dense/conv
weight upstream) -- `TypeError: lax.mul requires arguments to have the
same dtypes, got float32, bfloat16`.

The jax.vjp-on-reference path in kernel_trainable.py never hit this because
gdn2_chunked_wy_reference does an explicit `.astype(f32)` as the very first
thing it does to EACH input inside `_build_chunk_wy` -- autodiff then casts
the cotangent back down to that input's original dtype automatically, for
free, on the way out of that `.astype` call. Here, since the backward is
computed by hand (not by differentiating through an `.astype`), the cast
back to the original dtype has to be explicit -- added below.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .kernel_a_scores import BT, build_chunk_scores_pallas
from .kernel_b_solve import wy_solve_pallas
from .kernel_c_recompute import recompute_wy_pallas
from .kernel_d_pipeline import gdn2_pallas_forward, gdn2_inter_chunk_combine_with_state

from .kernel_bwd_b1_dhu import gdn2_dhu_backward
from .kernel_bwd_b2_dav import dav_backward_pallas
from .kernel_bwd_b3_wy_dqkg import wy_dqkg_backward_pallas
from .kernel_bwd_b4_intra import intra_backward_pallas
from .kernel_bwd_b5_reverse_cumsum import reverse_cumsum_bwd

_HIGHEST = jax.lax.Precision.HIGHEST
_FINAL_CLIP = 1e4


def _final_sanitize(x):
    return jnp.nan_to_num(jnp.clip(x, -_FINAL_CLIP, _FINAL_CLIP), nan=0.0,
                           posinf=_FINAL_CLIP, neginf=-_FINAL_CLIP)


def _reshape_in(t, bsz, n_chunks, H, D):
    t = t.reshape(bsz, n_chunks, BT, H, D)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def _reshape_out(t):
    bsz, H, n_chunks, _BT, D = t.shape
    t2 = jnp.moveaxis(t, (1, 2, 3), (3, 1, 2))
    return t2.reshape(bsz, n_chunks * BT, H, D)


def _build_dh_next_all(dh_all, dht):
    shifted = dh_all[:, :, 1:]
    dht_expanded = dht[:, :, None]
    return jnp.concatenate([shifted, dht_expanded], axis=2)


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

    bsz, L, H, D = q.shape
    n_chunks = L // BT

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    A = wy_solve_pallas(Akk)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    o_all, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0
    )
    h_pre_all = jnp.moveaxis(h_pre_all, 0, 2)
    v_new_all = jnp.moveaxis(v_new_all, 0, 2)

    g_r = _reshape_in(g, bsz, n_chunks, H, D)
    idx = jnp.arange(BT)
    tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST)

    q_r = _reshape_in(q, bsz, n_chunks, H, D)
    k_r = _reshape_in(k, bsz, n_chunks, H, D)
    b_r = _reshape_in(b, bsz, n_chunks, H, D)
    w_r = _reshape_in(w, bsz, n_chunks, H, D)
    v_r = _reshape_in(v, bsz, n_chunks, H, D)
    do_r = _reshape_in(do, bsz, n_chunks, H, D)

    dAqk, dv_partial = dav_backward_pallas(Aqk, v_new_all, do_r)

    dh_all, dh0, dv_all = gdn2_dhu_backward(
        do_r, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht=dh_final
    )
    dh_next_all = _build_dh_next_all(dh_all, dh_final)

    b3_out = wy_dqkg_backward_pallas(
        q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
        do_r, dv_all, dh_next_all, scale,
    )

    dq4, dk4, db4, dgc4 = intra_backward_pallas(dAqk, b3_out["dAkk"], q, k, b, g, scale)

    dgc_total = b3_out["dgc"] + dgc4
    dg_raw = reverse_cumsum_bwd(dgc_total, chunk_size=BT)

    dq = _reshape_out(b3_out["dq"] + dq4)
    dk = _reshape_out(b3_out["dk"] + dk4)
    db = _reshape_out(b3_out["db"] + db4)
    dw = _reshape_out(b3_out["dw"])
    dv = _reshape_out(b3_out["dv_raw"])
    dg = _reshape_out(dg_raw)

    # FIX: final sanitization pass on every gradient this function returns
    # -- see module docstring. This is the custom_vjp return boundary; an
    # optimizer step consumes exactly these values next, so this is the
    # last place a non-finite or absurdly-large gradient can be caught
    # before it reaches the model's weights.
    #
    # FIX #2 (dtype): custom_vjp requires the bwd function to return
    # cotangents with the SAME dtype as the corresponding forward input.
    # q/k/v/w/b arrive from model.py as bfloat16, while g (log-decay)
    # arrives as float32 (computed explicitly in fp32 for precision --
    # see GatedDeltaNet2J in model.py). Every intermediate value in this
    # backward chain (B1-B5) is float32, and the sanitize step above does
    # NOT change that -- so without this cast, dq/dk/dv/dw/db would be
    # returned as float32 for bfloat16 primal inputs. JAX accepts that
    # silently at the custom_vjp boundary itself, but it then fails one
    # level up the graph with a dtype-mismatch error (float32 vs bfloat16)
    # the moment that cotangent has to combine with a bfloat16-typed
    # gradient elsewhere (e.g. through a bf16 nn.Dense/conv weight) --
    # exactly the "lax.mul requires ... float32, bfloat16" failure. The
    # jax.vjp-on-reference path in kernel_trainable.py never hit this
    # because gdn2_chunked_wy_reference does an explicit `.astype(f32)` as
    # the very first thing it does to EACH input -- autodiff then casts the
    # cotangent back down to that input's original dtype automatically for
    # free, on the way out of `.astype`. Here the cast has to be explicit.
    dq = _final_sanitize(dq).astype(q.dtype)
    dk = _final_sanitize(dk).astype(k.dtype)
    db = _final_sanitize(db).astype(b.dtype)
    dw = _final_sanitize(dw).astype(w.dtype)
    dv = _final_sanitize(dv).astype(v.dtype)
    dg = _final_sanitize(dg).astype(g.dtype)
    dh0 = _final_sanitize(dh0).astype(h0.dtype)

    return dq, dk, dv, dw, db, dg, dh0


_gdn2_core.defvjp(_gdn2_core_fwd, _gdn2_core_bwd)


def gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale, h0=None):
    """Drop-in trainable version of gdn2_pallas_forward -- fused-Pallas
    backward (B1-B5), not the jax.vjp-on-reference "cheat" backward.
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    return _gdn2_core(q, k, v, w, b, g, scale, h0)
