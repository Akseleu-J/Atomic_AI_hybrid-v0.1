"""
atomic_ops/mamba2_bwd_state_reference.py -- correctness target for MB1's
own Pallas-eligible half (state-recurrence), complementary to
mamba2_bwd_intra_reference.py's already-existing MB2 split.

mamba2_bwd_reference.py's `_chunk_ssd_bwd` (MB0) bundles TWO structurally
independent pieces (see kernel_mamba2_bwd_b1_state.py's own module
docstring for the same accounting, stated there but never actually acted
on beyond the reverse-scan skeleton):

  (1) STATE-RECURRENCE adjoint (THIS file) -- backward through
        state_carry[d,s] = state_prev[d,s] * decay_chunk_end[d]
        y_off[i,d]        = (C_c[i,:] . state_prev[d,:]) * decay_h[i,d]
      Both only need state_prev, C_c, cumdecay (all already known from the
      forward) plus this chunk's incoming cotangents (dy_off, dstate_carry).
      No dependency on anything MB2 owns.

  (2) INTRA-CHUNK adjoint (mamba2_bwd_intra_reference.py / MB2, already
      done, already Pallas-ported and TPU-validated) -- backward through
      y_diag/state_end (the (BT,BT)-shaped L/BC_inner/weight machinery).

Why the split is safe to test standalone, same argument as MB2's own
docstring but mirrored: `_chunk_ssd` with state_prev=0 gives
`state_carry==0` and `y_off==0` IDENTICALLY (both are literally
`state_prev * (...)` or `(... . state_prev) * (...)` -- linear in
state_prev, zero when it's zero). `_state_only_fwd` below is `_chunk_ssd`
with the y_diag/state_end lines deleted and `state_new` replaced by just
its state_prev-carry term (`state_end` is MB2's own output, added back in
by whichever orchestrator wires MB1+MB2 together -- NOT reproduced here).

`_state_only_bwd` is lifted verbatim (not re-derived) from the
already-validated `_chunk_ssd_bwd` in mamba2_bwd_reference.py -- every line
below corresponds 1:1 to a line already present there under the
"y_off"/"state_new"/"decay_h"/"decay_chunk_end" headings. This file just
gives that subset its own forward to be cross-checked against
(`test_kernel_mamba2_bwd_b1_state.py` cross-checks it via jax.vjp on
`_state_only_fwd`, same "hand derivation -> jax.vjp cross-check -> (Pallas
port, if ever needed) -> TPU test" discipline as MB2), same convention
`_intra_only_bwd` already uses in the sibling file.

NOTE: unlike MB2 (which got a dedicated Pallas kernel because its cost is
O(BT^2)), this state-only piece is O(BT) per chunk (no (BT,BT) intermediate
anywhere) -- kernel_mamba2_bwd_b1_state.py's own docstring already commits
to keeping the whole state-recurrence half as plain JAX inside the
reverse-scan carry, matching gdn2_dhu_backward's role for GDN-2. This file
therefore is NOT expected to grow a Pallas port; it exists so
kernel_mamba2_bwd_b1_state.py's `chunk_bwd_fn` seam can be pointed at
JUST this half (instead of the full MB0 `_chunk_ssd_bwd`) once MB3's
orchestrator combines it with MB2's Pallas kernel for the intra half.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .mamba2_ssd_reference import _sanitize

_HIGHEST = jax.lax.Precision.HIGHEST
_STEP_CLIP = 20.0


def _clip_mask(x, lo, hi):
    """Straight-through gradient mask for jnp.clip -- same convention as
    mamba2_bwd_reference.py's own _clip_mask / mamba2_bwd_intra_reference's
    own copy of it."""
    return ((x >= lo) & (x <= hi)).astype(jnp.float32)


def _state_only_fwd(state_prev, C_c, cumdecay_c):
    """state_prev: (b,h,d,s). C_c: (b,C,s). cumdecay_c: (b,C,h,d).
    Returns y_off: (b,C,h,d), state_carry: (b,h,d,s) -- ONLY the
    state_prev-decay term of state_new (state_end, MB2's own output, is
    NOT added here -- see module docstring)."""
    f32 = jnp.float32
    state_prev, C_c, cumdecay_c = (t.astype(f32) for t in (state_prev, C_c, cumdecay_c))

    decay_h = jnp.exp(jnp.clip(cumdecay_c, -_STEP_CLIP, 0.0))          # (b,C,h,d)
    y_off_raw = jnp.einsum("bis,bhds->bihd", C_c, state_prev, precision=_HIGHEST)
    y_off = _sanitize(y_off_raw * decay_h)

    decay_chunk_end = jnp.exp(jnp.clip(cumdecay_c[:, -1], -_STEP_CLIP, 0.0))   # (b,h,d)
    state_carry = _sanitize(state_prev * decay_chunk_end[..., None])

    return y_off, state_carry


def _state_only_bwd(state_prev, C_c, cumdecay_c, dy_off, dstate_carry):
    """Hand-derived adjoint of _state_only_fwd -- lines lifted verbatim
    from mamba2_bwd_reference._chunk_ssd_bwd (see module docstring).
    Returns dC_intra_partial (renamed dC_c here -- FULL contribution from
    this half; MB2 owns the other half via BC_inner), dstate_prev (FULL --
    both y_off and state_carry are linear in state_prev, both contribute),
    dcumdecay_state (PARTIAL -- MB2 adds its own contribution from the
    y_diag/state_end side)."""
    f32 = jnp.float32
    state_prev, C_c, cumdecay_c, dy_off, dstate_carry = (
        t.astype(f32) for t in (state_prev, C_c, cumdecay_c, dy_off, dstate_carry)
    )
    Cs = cumdecay_c.shape[1]
    idx = jnp.arange(Cs)

    # ---- re-run the forward (cheap, self-contained residuals -- same
    # "recompute, don't stash" tradeoff used throughout atomic_ops/) ----
    decay_h_raw = jnp.clip(cumdecay_c, -_STEP_CLIP, 0.0)
    decay_h = jnp.exp(decay_h_raw)
    y_off_raw = jnp.einsum("bis,bhds->bihd", C_c, state_prev, precision=_HIGHEST)

    decay_chunk_end_raw = jnp.clip(cumdecay_c[:, -1], -_STEP_CLIP, 0.0)
    decay_chunk_end = jnp.exp(decay_chunk_end_raw)

    # ================= REVERSE PASS (verbatim from MB0) =================
    # state_carry = state_prev * decay_chunk_end[...,None]
    dstate_prev = dstate_carry * decay_chunk_end[..., None]
    d_decay_chunk_end = jnp.sum(dstate_carry * state_prev, axis=-1)          # (b,h,d)
    d_cumdecay_last_a = d_decay_chunk_end * decay_chunk_end * _clip_mask(cumdecay_c[:, -1], -_STEP_CLIP, 0.0)

    # y_off = (C_c . state_prev) * decay_h
    dy_off_raw = dy_off * decay_h
    d_decay_h = dy_off * y_off_raw
    dC_c = jnp.einsum("bihd,bhds->bis", dy_off_raw, state_prev, precision=_HIGHEST)
    dstate_prev = dstate_prev + jnp.einsum("bihd,bis->bhds", dy_off_raw, C_c, precision=_HIGHEST)

    # decay_h = exp(clip(cumdecay, -20, 0))
    d_cumdecay_from_yoff = d_decay_h * decay_h * _clip_mask(decay_h_raw, -_STEP_CLIP, 0.0)

    # ---- combine every contribution to cumdecay this half owns ----
    row_mask = (idx == (Cs - 1)).astype(jnp.float32)[None, :, None, None]
    dcumdecay_state = d_cumdecay_from_yoff + row_mask * d_cumdecay_last_a[:, None, :, :]

    dC_c = _sanitize(dC_c)
    dstate_prev = _sanitize(dstate_prev)
    dcumdecay_state = _sanitize(dcumdecay_state)

    return dC_c, dstate_prev, dcumdecay_state
