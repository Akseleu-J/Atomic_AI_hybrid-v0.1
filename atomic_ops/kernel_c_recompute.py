"""
Milestone 3 -- Kernel C (Pallas/TPU): w_pseudo, u, kg, qg, gc_last.

FIX (non-finite hardening, found by Акселеу while auditing this file against
the same failure mode already documented in kernel_b_solve.py's docstring):
this kernel had TWO separate gaps, both allowing a large-but-finite `A`
(produced by Kernel B when Akk is near-singular -- see kernel_b_solve.py's
own docstring for the full story) to flow downstream unbounded, only to
overflow to an actual `inf` later inside Kernel D's inter-chunk scan (where
the existing clip/nan_to_num finally catches it -- too late to localize the
real source):

  1. `w_pseudo`/`u` were only `nan_to_num`'d, never `clip`'d. Exactly the
     gap kernel_b_solve.py's docstring warns about: `nan_to_num` alone only
     replaces values that are ALREADY nan/inf -- it does nothing to a value
     that is merely huge but still finite (e.g. ~1e20, which a near-singular
     A can easily produce through `A @ kb_decayed` / `A @ (w*v)`).

  2. `kg`, `qg`, and `gc_last_row` had NO sanitization at all -- not even
     `nan_to_num`. `kg = k * exp(gc_last - gc)` and `qg = q * exp(gc)` both
     exponentiate a cumulative decay term; if that ever leaks slightly
     positive (same `decay_diff` numerical-leak risk documented in
     gdn2_wy_reference.py and kernel_a_scores.py) or `A` upstream is huge,
     these can overflow directly here, unclipped, straight into Kernel D's
     `wh = einsum(w_pseudo, h_pre)` / `write = einsum(kg, v_new)` matmuls.

Fix: same `clip(+-1e4) + nan_to_num` combination used everywhere else in
this project (kernel_b_solve.py, kernel_d_pipeline.py's scan step, B3, B4,
B5) applied to all five outputs of this kernel, not just two of them.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .kernel_a_scores import BT

_HIGHEST = jax.lax.Precision.HIGHEST
_CLIP = 1e4


def _sanitize(x):
    return jnp.nan_to_num(jnp.clip(x, -_CLIP, _CLIP), nan=0.0, posinf=_CLIP, neginf=-_CLIP)


def _kernel_c_body(q_ref, k_ref, v_ref, w_ref, b_ref, g_ref, a_ref,
                    w_pseudo_ref, u_ref, kg_ref, qg_ref, gc_last_ref):
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    v = v_ref[0, 0, 0].astype(jnp.float32)
    w = w_ref[0, 0, 0].astype(jnp.float32)
    b = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)
    A = a_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(BT)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

    kb_decayed = b * k * jnp.exp(gc)
    w_pseudo = jnp.dot(A, kb_decayed, precision=_HIGHEST)
    u = jnp.dot(A, w * v, precision=_HIGHEST)
    # FIX: clip, not just nan_to_num -- see module docstring. A near-singular
    # Akk upstream (Kernel B) can make A large-but-finite, which propagates
    # here through these matmuls; nan_to_num alone lets that straight through.
    w_pseudo = _sanitize(w_pseudo)
    u = _sanitize(u)

    gc_last_row = gc[BT - 1]
    kg = k * jnp.exp(gc_last_row[None, :] - gc)
    qg = q * jnp.exp(gc)

    # FIX: kg/qg/gc_last_row previously had NO sanitization at all. These
    # feed directly into Kernel D's inter-chunk einsums (wh, write) --
    # unclipped, an overflow here only surfaces as an `inf` several kernels
    # downstream, exactly the diagnosability gap kernel_b_solve.py's
    # docstring describes for A itself.
    kg = _sanitize(kg)
    qg = _sanitize(qg)
    gc_last_row = _sanitize(gc_last_row)

    w_pseudo_ref[0, 0, 0] = w_pseudo
    u_ref[0, 0, 0] = u
    kg_ref[0, 0, 0] = kg
    qg_ref[0, 0, 0] = qg
    gc_last_ref[0, 0, 0, 0] = gc_last_row


def recompute_wy_pallas(q, k, v, w, b, g, A):
    bsz, L, H, D = q.shape
    assert L % BT == 0
    n_chunks = L // BT

    def reshape_in(t):
        t = t.reshape(bsz, n_chunks, BT, H, D)
        return jnp.moveaxis(t, (1, 3), (2, 1))

    q_r, k_r, v_r, w_r, b_r, g_r = map(reshape_in, (q, k, v, w, b, g))

    grid = (bsz, H, n_chunks)
    io_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))
    a_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))
    gclast_spec = pl.BlockSpec((1, 1, 1, 1, D), lambda i, h, c: (i, h, c, 0, 0))

    w_pseudo, u, kg, qg, gc_last = pl.pallas_call(
        _kernel_c_body,
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec, a_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec, gclast_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, 1, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
    )(q_r, k_r, v_r, w_r, b_r, g_r, A)

    gc_last = gc_last.reshape(bsz, H, n_chunks, D)
    return w_pseudo, u, kg, qg, gc_last
