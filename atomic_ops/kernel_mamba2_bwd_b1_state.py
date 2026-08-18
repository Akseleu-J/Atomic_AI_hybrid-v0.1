"""
Milestone MB1+MB3 -- orchestrator combining the state-recurrence backward
(mamba2_bwd_state_reference.py, pure JAX, reverse lax.scan) with the
intra-chunk Pallas backward (kernel_mamba2_bwd_b2_intra.py, MB2, already
TPU-validated) into a single per-chunk backward pass over the whole
sequence.

This REPLACES the previous version of this file, which pointed its
`chunk_bwd_fn` seam at the full MB0 `_chunk_ssd_bwd` (mamba2_bwd_reference)
by default -- see that version's own docstring, which already announced
this as the intended next step ("Once MB2-4 exist, THIS file's
`_chunk_bwd_fn` seam ... is what gets swapped for the Pallas-backed
version").

WHY THIS IS ONE FILE, NOT TWO SEPARATE MB1/MB3 FILES
-----------------------------------------------------------------------
mamba2_bwd_state_reference.py's own docstring commits the state-recurrence
half to staying plain JAX inside the reverse-scan carry (it is O(BT) per
chunk, not O(BT^2) -- no Pallas kernel was ever planned for it, unlike
MB2). That means MB1's "reverse-scan skeleton" and MB3's "combine MB1+MB2
outputs" are the SAME function in practice: the scan produces the
state-only partial results AND the exact per-chunk `dstate_end_grad` array
MB2 needs (see below), and the combination happens the moment both are in
hand -- there is no intermediate artifact worth a separate module.

TWO-PASS STRUCTURE
-----------------------------------------------------------------------
Unlike MB0's `chunk_ssd_bwd_scan` (which calls the ENTIRE per-chunk
backward, both halves, inside one `jax.lax.scan` step), this orchestrator
cannot do that: MB2 is a single Pallas kernel call spanning ALL chunks at
once (its own `grid=(bsz, n_heads_ssm, n_chunks)` -- see
kernel_mamba2_bwd_b2_intra.py), not something callable once per scan step.
So the two halves run in two passes, exactly mirroring how GDN-2 already
splits this same problem (see kernel_bwd_b1_dhu.py's `gdn2_dhu_backward`
producing `dh_all` for every chunk via one reverse-scan, which
kernel_bwd_b3_wy_dqkg.py's Pallas kernel then consumes ACROSS all chunks
in one call, via `_build_dh_next_all`'s shift):

  PASS 1 (this file's reverse scan, pure JAX, state-only half):
    For every chunk (reverse order), consume the incoming state cotangent
    `dstate_carry` (the recurrence's own carry) to compute:
      - dC_state_c, dstate_prev_c, dcumdecay_state_c  (via
        mamba2_bwd_state_reference._state_only_bwd)
    AND record `dstate_carry` itself (the value used as INPUT this step --
    exactly the "dstate_end_grad" MB2 needs for this same chunk) as a
    SCANNED OUTPUT, not just an internal carry -- same "carry doubles as a
    recorded scan output" trick `gdn2_dhu_backward` already uses for
    `dh_pre_c`.

  PASS 2 (one Pallas call spanning every chunk, MB2):
    `intra_chunk_ssd_bwd_pallas(dt, x, B, C, cumdecay, dy=do,
    dstate_end_grad=<PASS 1's recorded array>, chunk_size)` -> ddt_intra,
    dx (full), dB_intra (full), dC_intra (partial), dcumdecay_intra
    (partial).

  COMBINE (this is MB3 proper): dC and dcumdecay each have contributions
  from BOTH passes (state-only via C_c/cumdecay_c's role in y_off, intra
  via C_c/cumdecay_c's role in BC_inner/L) -- summed with the SAME
  clip-after-every-accumulation-write discipline
  kernel_bwd_b4_intra.py's own docstring documents ("an inf from one
  contribution can meet an opposite-sign inf from another and produce an
  unrecoverable NaN" -- here the two contributions come from two different
  backward passes instead of two loop iterations, same risk).

OWNERSHIP OF EACH RETURNED GRADIENT AT THIS MILESTONE (MB1+MB2+MB3 done,
MB4 NOT done yet)
-----------------------------------------------------------------------
  ddt        -- PARTIAL. Only the dBx=dt*x contribution (MB2-owned). The
                second contribution (through dA_exponent=dt*A, via the
                reverse tril-cumsum of dcumdecay) is MB4's job -- NOT
                computed here. Do not treat this ddt as final.
  dx         -- FULL. x only ever appears inside dBx=dt*x (MB2-owned);
                state-only half never touches x.
  dB         -- FULL. B only ever appears inside BC_inner (MB2-owned);
                state-only half never touches B (see
                mamba2_bwd_state_reference.py's own docstring / the
                `test_zero_state_prev_gives_zero_grads` cross-check that
                pins this down).
  dC         -- FULL for this milestone's scope, i.e. sum of BOTH halves'
                contributions (state-only's dC_c_1-equivalent + MB2's
                dC_c_2-equivalent) -- C has no further MB4-owned
                contribution in MB0's accounting (see
                mamba2_bwd_reference._chunk_ssd_bwd: dC_c = dC_c_1 + dC_c_2,
                nothing else touches dC downstream of that).
  dA         -- NOT computed here. MB4-only (depends on the full ddt
                dA_exponent chain).
  dstate0    -- FULL. Purely state-only-owned (state_prev never appears in
                MB2's forward at all).
  dcumdecay  -- Returned (NOT part of MB0's public return signature, but
                needed downstream by MB4) as the SUM of both halves'
                per-chunk contributions, in the same per-chunk
                `(bsz, n_heads_ssm, n_chunks, chunk_size, headdim)` Pallas
                layout `build_chunk_cumdecay_pallas` (M2) itself produces
                -- MB4 consumes it directly in this layout to run the
                reverse-cumsum/dA_exponent/dA chain and to complete `ddt`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .kernel_mamba2_a_decay import build_chunk_cumdecay_pallas
from .kernel_mamba2_bwd_b2_intra import intra_chunk_ssd_bwd_pallas
from .mamba2_bwd_state_reference import _state_only_bwd
from .mamba2_ssd_reference import _chunk_ssd, _sanitize

_HIGHEST = jax.lax.Precision.HIGHEST
_ACC_CLIP = 1e4


def _sanitize_state(x):
    return jnp.nan_to_num(jnp.clip(x, -_ACC_CLIP, _ACC_CLIP), nan=0.0, posinf=_ACC_CLIP, neginf=-_ACC_CLIP)


def _cumdecay_pallas_to_natural(cumdecay):
    """(bsz, n_heads_ssm, n_chunks, C, d) -> (n_chunks, bsz, C, n_heads_ssm, d)
    -- puts n_chunks first (scan axis, matching to_chunks' own convention
    for dt/B/C/x elsewhere in this project) and restores the (b,C,h,d)
    layout mamba2_bwd_state_reference._state_only_bwd expects (same
    convention mamba2_ssd_reference._chunk_ssd's own cumdecay uses)."""
    return jnp.moveaxis(cumdecay, (1, 2), (3, 0))


def _cumdecay_natural_to_pallas(cumdecay_natural_stacked):
    """Inverse of _cumdecay_pallas_to_natural: (n_chunks, bsz, C, h, d) ->
    (bsz, h, n_chunks, C, d) -- the layout kernel_mamba2_bwd_b2_intra.py's
    `dcum` output already uses, so the two contributions can be summed
    directly without a further reshape."""
    return jnp.moveaxis(cumdecay_natural_stacked, (0, 3), (2, 1))


def mamba2_bwd_scan(dt, A, B, C, x, chunk_size, do, dstate_final, state0=None,
                     interpret=False):
    """Drop-in-shaped (but NOT identical -- see module docstring's
    "ownership" section) replacement for the previous version's
    `chunk_bwd_fn`-parametrized reverse scan. Returns
    (ddt_partial, dB, dC, dx, dstate0, dcumdecay_combined) -- NOTE the
    changed return signature vs the old file (no `dA`, extra
    `dcumdecay_combined`): dA cannot be produced until MB4 exists, and
    dcumdecay_combined is a NEW artifact MB4 needs that MB0's public
    signature never exposed (MB0 keeps cumdecay/dA_exponent entirely
    internal to `_chunk_ssd_bwd`). Callers wanting MB0's exact old
    signature should keep using `mamba2_bwd_reference.chunk_ssd_bwd_scan`
    directly (still the cross-check target -- see the module docstring's
    validation discipline note) until MB4 lands and a final orchestrator
    can restore that exact signature.

    dt,x: (b,l,h,d). A: (h,). B,C: (b,l,s). do: (b,l,h,d) cotangent for y.
    dstate_final: (b,h,d,s) cotangent for the final carried state.
    `interpret`: forwarded to MB2's Pallas call (True for CPU/dev-time
    cross-checking, same convention as every other Pallas kernel in this
    project -- False, the default, for real TPU training).
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

    # ---- M2: cumdecay, computed ONCE, reused by both PASS 1 (state-only
    # scan, needs the natural (b,C,h,d)-per-chunk layout) and PASS 2 (MB2
    # Pallas call, needs the (b,h,n_chunks,C,d) layout it was already
    # produced in). ----
    cumdecay_pallas_layout = build_chunk_cumdecay_pallas(dt, A, chunk_size, interpret=interpret)
    cumdecay_natural_ch = _cumdecay_pallas_to_natural(cumdecay_pallas_layout)

    # ---- forward re-run for per-chunk state_prev residuals -- same
    # "recompute, don't stash" tradeoff every other backward in this
    # project makes (kernel_c's w_pseudo/u, MB0's own state_prev_all). ----
    def fwd_step(state_prev, inputs):
        dt_c, B_c, C_c, x_c = inputs
        _, state_new = _chunk_ssd(dt_c, A, B_c, C_c, x_c, state_prev)
        return state_new, state_prev

    _, state_prev_all = jax.lax.scan(fwd_step, state0, (dt_ch, B_ch, C_ch, x_ch))

    # ======================================================================
    # PASS 1 -- state-only reverse scan. Produces, per chunk:
    #   dC_state_c, dcumdecay_state_c  (to be summed with MB2's own halves)
    #   dstate_carry_used_c            (= the array MB2 needs as
    #                                     dstate_end_grad -- see module
    #                                     docstring's "carry doubles as a
    #                                     recorded output" note)
    # and, as the final carry: dstate0.
    # ======================================================================
    def bwd_step_state_only(dstate_carry, inputs):
        C_c, cumdecay_c, do_c, state_prev_c = inputs
        dstate_carry = _sanitize_state(dstate_carry)
        dC_c, dstate_prev_c, dcumdecay_c = _state_only_bwd(
            state_prev_c, C_c, cumdecay_c, dy_off=do_c, dstate_carry=dstate_carry
        )
        dstate_prev_c = _sanitize_state(dstate_prev_c)
        # record the carry VALUE USED THIS STEP (not the updated one) --
        # this is exactly dstate_end_grad for MB2, for this same chunk.
        return dstate_prev_c, (dC_c, dcumdecay_c, dstate_carry)

    dstate0, (dC_state_rev, dcumdecay_state_rev, dstate_carry_all_rev) = jax.lax.scan(
        bwd_step_state_only, dstate_final,
        (C_ch, cumdecay_natural_ch, do_ch, state_prev_all),
        reverse=True,
    )
    dstate0 = _sanitize_state(dstate0)

    def from_chunks(t):
        t = jnp.moveaxis(t, 0, 1)
        return t.reshape(b, l, *t.shape[3:])

    dC_state = from_chunks(dC_state_rev)                       # (b, l, s)

    # dstate_carry_all_rev: (n_chunks, b, h, d, s) -- move n_chunks to the
    # position MB2 expects it (axis 2, matching cumdecay's own
    # (b,h,n_chunks,...) convention).
    dstate_end_grad = jnp.moveaxis(dstate_carry_all_rev, 0, 2)  # (b, h, n_chunks, d, s)
    dstate_end_grad = _sanitize_state(dstate_end_grad)

    # dcumdecay_state_rev: (n_chunks, b, C, h, d) -- back to Pallas layout
    # to combine with MB2's own dcumdecay output.
    dcumdecay_state = _cumdecay_natural_to_pallas(dcumdecay_state_rev)

    # ======================================================================
    # PASS 2 -- MB2, one Pallas call spanning every chunk at once.
    # ======================================================================
    ddt_intra, dx_full, dB_full, dC_intra, dcumdecay_intra = intra_chunk_ssd_bwd_pallas(
        dt, x, B, C, cumdecay_pallas_layout, dy=do, dstate_end_grad=dstate_end_grad,
        chunk_size=chunk_size, interpret=interpret,
    )

    # ======================================================================
    # COMBINE (MB3) -- clipped sum, same discipline as
    # kernel_bwd_b4_intra.py's per-accumulation-write clip.
    # ======================================================================
    dC = _sanitize(dC_state + dC_intra)
    dcumdecay = _sanitize(dcumdecay_state + dcumdecay_intra)

    ddt = _sanitize(ddt_intra)   # PARTIAL -- see module docstring
    dx = _sanitize(dx_full)
    dB = _sanitize(dB_full)

    return ddt, dB, dC, dx, dstate0, dcumdecay
