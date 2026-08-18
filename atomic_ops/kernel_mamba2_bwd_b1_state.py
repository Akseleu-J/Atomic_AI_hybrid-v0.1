"""
Milestone MB1 -- inter-chunk state-recurrence backward, extracted as a
standalone module from mamba2_bwd_reference.py's chunk_ssd_bwd_scan (MB0).

Rationale (mirrors kernel_d_pipeline.py's forward split of Kernel D --
plain-JAX O(n_chunks) scan -- from Kernels A/B/C -- Pallas O(BT^2) intra-
chunk work): MB0's _chunk_ssd_bwd bundled TWO structurally independent
pieces of the backward into one function:

  (1) the STATE-RECURRENCE adjoint -- backward through
      `state_new = state_prev*decay_chunk_end + state_end` and the
      `y_off = (C_c . state_prev) * decay_h` cross-term. Both only need
      quantities that are already known from the FORWARD pass (state_prev,
      C_c, decay_h, decay_chunk_end) plus this chunk's incoming cotangents
      (dy_c, dstate_new-from-next-chunk). No dependency on anything MB2-4
      will compute.

  (2) the INTRA-CHUNK adjoint -- backward through the (BT,BT)-shaped
      L/BC_inner/weight machinery (y_diag, state_end themselves, and the
      cumdecay contributions that flow through them). This is the O(BT^2)
      part that MB2 (Pallas port of the y_diag/state_end adjoint) and MB4
      (Pallas port of the cumdecay/dA adjoint) will replace piece by piece.

This module implements ONLY (1), reverse-scanned across chunks -- exactly
the role `gdn2_dhu_backward` (kernel_bwd_b1_dhu.py) plays for GDN-2: a
pure-JAX carry recurrence, sanitized every step, that the intra-chunk
Pallas kernels feed into / read from, but never themselves compute.

Validation: mamba2_bwd_scan (this module) is REQUIRED to reproduce
mamba2_bwd_reference.chunk_ssd_bwd_scan's output bit-for-bit (same math,
same _chunk_ssd_bwd call inside the loop -- this is a pure refactor, not a
new derivation) -- see test_kernel_mamba2_bwd_b1_state.py. Once MB2-4
exist, THIS file's `_chunk_bwd_fn` seam (currently defaulted to
mamba2_bwd_reference._chunk_ssd_bwd) is what gets swapped for the
Pallas-backed version, and mamba2_bwd_reference.py remains the permanent
cross-check target for whatever replaces it -- same "reference stays as
fallback/cross-check" convention as kernel_trainable.py's own docstring.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .mamba2_ssd_reference import _chunk_ssd, _sanitize
from .mamba2_bwd_reference import _chunk_ssd_bwd

_HIGHEST = jax.lax.Precision.HIGHEST
_ACC_CLIP = 1e4


def _sanitize_state(x):
    return jnp.nan_to_num(jnp.clip(x, -_ACC_CLIP, _ACC_CLIP), nan=0.0, posinf=_ACC_CLIP, neginf=-_ACC_CLIP)


def mamba2_bwd_scan(dt, A, B, C, x, chunk_size, do, dstate_final, state0=None,
                     chunk_bwd_fn: Callable = _chunk_ssd_bwd):
    """Reverse lax.scan across chunks -- the ONLY place this project's
    Mamba2 backward is allowed to carry state sequentially. Everything
    inside `chunk_bwd_fn` (default: the MB0 hand-derived reference) is
    per-chunk, embarrassingly parallel work -- exactly the kind MB2-4 will
    port to Pallas, plugged in here via the `chunk_bwd_fn` parameter
    without touching this scan.

    Same call signature and return values as
    mamba2_bwd_reference.chunk_ssd_bwd_scan -- this function IS that
    function, refactored to (a) live in its own module (MB1's actual
    deliverable) and (b) expose the per-chunk backward as a swappable
    argument instead of a hardcoded call, so MB2-4 don't need to touch
    this file again once they exist.

    dt,x: (b,l,h,d). A: (h,). B,C: (b,l,s). do: (b,l,h,d) cotangent for y.
    dstate_final: (b,h,d,s) cotangent for the final carried state.
    Returns ddt, dB, dC, dx, dA, dstate0 -- same shapes as forward inputs.
    """
    b, l, h, d = dt.shape
    s = B.shape[-1]
    assert l % chunk_size == 0, f"seq_len={l} must be divisible by chunk_size={chunk_size}."
    n_chunks = l // chunk_size

    if state0 is None:
        state0 = jnp.zeros((b, h, d, s), dtype=jnp.float32)
    state0 = _sanitize_state(state0)

    def to_chunks(t):
        shp = t.shape
        t = t.reshape(b, n_chunks, chunk_size, *shp[2:])
        return jnp.moveaxis(t, 1, 0)

    dt_ch, B_ch, C_ch, x_ch, do_ch = map(to_chunks, (dt, B, C, x, do))

    # ---- forward re-run to recover per-chunk state_prev residuals -- same
    # "recompute forward, don't stash everything" tradeoff kernel_c makes
    # for w_pseudo/u in GDN-2 (cheaper than threading a huge residual
    # pytree through the training loop). ----
    def fwd_step(state_prev, inputs):
        dt_c, B_c, C_c, x_c = inputs
        _, state_new = _chunk_ssd(dt_c, A, B_c, C_c, x_c, state_prev)
        return state_new, state_prev

    _, state_prev_all = jax.lax.scan(fwd_step, state0, (dt_ch, B_ch, C_ch, x_ch))

    def bwd_step(dstate_carry, inputs):
        dt_c, B_c, C_c, x_c, do_c, state_prev_c = inputs
        # FIX (this milestone, belt-and-braces): sanitize the incoming
        # carry every step, same convention as gdn2_dhu_backward's own
        # dht/dh_carry sanitization -- protects against a large-but-finite
        # dstate drifting across many chunks before any Pallas kernel gets
        # a chance to clip its own output.
        dstate_carry = _sanitize_state(dstate_carry)
        ddt_c, dB_c, dC_c, dx_c, dA_c, dstate_prev_c = chunk_bwd_fn(
            dt_c, A, B_c, C_c, x_c, state_prev_c, do_c, dstate_carry
        )
        dstate_prev_c = _sanitize_state(dstate_prev_c)
        return dstate_prev_c, (ddt_c, dB_c, dC_c, dx_c, dA_c)

    dstate0, (ddt_rev, dB_rev, dC_rev, dx_rev, dA_rev) = jax.lax.scan(
        bwd_step, dstate_final,
        (dt_ch, B_ch, C_ch, x_ch, do_ch, state_prev_all),
        reverse=True,
    )

    def from_chunks(t):
        # t: (n_chunks, b, C, ...) -- merge (n_chunks, C) -> l. t.shape
        # right after moveaxis is (b, n_chunks, C, ...); the axes AFTER C
        # are t.shape[3:] (same fix as MB0's chunk_ssd_bwd_scan).
        t = jnp.moveaxis(t, 0, 1)
        return t.reshape(b, l, *t.shape[3:])

    ddt = from_chunks(ddt_rev)
    dB = from_chunks(dB_rev)
    dC = from_chunks(dC_rev)
    dx = from_chunks(dx_rev)
    dA = jnp.sum(dA_rev, axis=0)
    dstate0 = _sanitize_state(dstate0)

    return ddt, dB, dC, dx, dA, dstate0
