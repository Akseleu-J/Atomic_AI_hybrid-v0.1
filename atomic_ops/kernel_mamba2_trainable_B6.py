"""
Milestone MB5 -- honest fused-Pallas trainable wrapper for Mamba2 SSD.

Forward: unchanged from kernel_mamba2_trainable.py -- Pallas M2 (cumdecay)
-> M3 (intra-chunk, Pallas) -> M4 (inter-chunk, plain lax.scan), via
kernel_mamba2_c_interchunk.mamba2_pallas_forward.

Backward: REPLACES kernel_mamba2_trainable.py's jax.vjp-on-
mamba2_ssd_reference "cheat" backward with the actual fused chain built up
over MB1-MB4:
  MB1 (state-recurrence, plain JAX reverse-scan) +
  MB2 (intra-chunk, Pallas kernel, TPU-validated) +
  MB3 (combine, same file as MB1 -- kernel_mamba2_bwd_b1_state.mamba2_bwd_scan)
  -> ddt_partial, dB, dC, dx, dstate0, dcumdecay_combined
  MB4 (mamba2_bwd_a_dA_reference.mamba2_dA_backward) consumes
  dcumdecay_combined -> ddt_contrib2, dA
  ddt_final = ddt_partial + ddt_contrib2

Every one of these six returned gradients (ddt, dx, dB, dC, dA, dstate0)
was independently cross-checked against MB0 (mamba2_bwd_reference.py, the
hand-derived single-pass reference) with rel_err ~1e-7-1e-8 on real TPU
(see test_mamba2_bwd_a_dA.py / test_kernel_mamba2_bwd_b1_state.py) -- same
validation bar kernel_trainable_B6.py's own GDN-2 fused backward was held
to before it replaced kernel_trainable.py in model.py.

Same "cheat backward stays around as fallback / cross-check target"
convention as GDN-2: kernel_mamba2_trainable.py (jax.vjp-on-reference) is
NOT deleted or modified -- keep it as the comparison target for MB6 (see
that milestone's own test file), same role kernel_trainable.py still plays
for GDN-2 per its own docstring.

Same dtype-cast-at-boundary discipline as kernel_trainable_B6.py's FIX #2
and kernel_mamba2_trainable.py's own FIX: dt/x/B/C arrive as bfloat16 from
Mamba2J, A/state0 as float32. Every intermediate in the MB1-MB4 backward
chain is float32 (mamba2_bwd_a_dA_reference / mamba2_bwd_state_reference /
kernel_mamba2_bwd_b2_intra all upcast internally), so an explicit
`.astype(orig_dtype)` cast is required on every returned gradient at the
custom_vjp boundary -- there is no `.astype(f32)`-on-the-input-itself for
autodiff to piggyback a free downcast off of here (this backward is
hand-assembled, not differentiated through the forward).

Sanitization: one more clip(+-1e4)+nan_to_num pass on every gradient right
before it leaves this function -- same "last chance before the optimizer"
reasoning as kernel_trainable_B6.py's own `_final_sanitize`. Each
individual MB1-MB4 piece already sanitizes its own output, but (same
argument as kernel_trainable_B6.py's own docstring) nothing upstream
guards against the SUM `ddt_partial + ddt_contrib2` drifting to a
large-but-finite outlier even when both summands are individually
in-range.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .kernel_mamba2_c_interchunk import mamba2_pallas_forward
from .kernel_mamba2_bwd_b1_state import mamba2_bwd_scan
from .mamba2_bwd_a_dA_reference import (
    mamba2_dA_backward, reshape_dt_to_pallas, unreshape_dt_from_pallas,
)

_FINAL_CLIP = 1e4


def _final_sanitize(x):
    return jnp.nan_to_num(jnp.clip(x, -_FINAL_CLIP, _FINAL_CLIP), nan=0.0,
                           posinf=_FINAL_CLIP, neginf=-_FINAL_CLIP)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _mamba2_core(dt, x, B, C, A, chunk_size, state0):
    return mamba2_pallas_forward(dt, x, B, C, A, chunk_size, state0=state0)


def _mamba2_core_fwd(dt, x, B, C, A, chunk_size, state0):
    out = mamba2_pallas_forward(dt, x, B, C, A, chunk_size, state0=state0)
    residuals = (dt, x, B, C, A, state0)
    return out, residuals


def _mamba2_core_bwd(chunk_size, residuals, cotangents):
    dt, x, B, C, A, state0 = residuals
    do, dstate_final = cotangents

    # MB1+MB2+MB3 -- state-recurrence (plain JAX reverse-scan) + intra-chunk
    # (Pallas, MB2) + combine. Returns ddt PARTIAL (only the dBx=dt*x
    # contribution) and dcumdecay COMBINED (both halves summed) -- see
    # kernel_mamba2_bwd_b1_state.py's own docstring "ownership" section.
    ddt_partial, dB, dC, dx, dstate0, dcumdecay_combined = mamba2_bwd_scan(
        dt, A, B, C, x, chunk_size, do, dstate_final, state0=state0,
    )

    # MB4 -- completes ddt (second contribution, through dA_exponent=dt*A)
    # and produces dA (the only non-per-chunk gradient in this chain).
    dt_pallas = reshape_dt_to_pallas(dt, chunk_size)
    ddt_contrib2_pallas, dA = mamba2_dA_backward(dt_pallas, A, dcumdecay_combined, chunk_size)
    ddt_contrib2 = unreshape_dt_from_pallas(ddt_contrib2_pallas)

    ddt = ddt_partial + ddt_contrib2

    # FIX: final sanitization pass on the COMBINED gradients this
    # orchestrator produces -- same reasoning as kernel_trainable_B6.py's
    # own "FIX" docstring: each individual MB1-MB4 piece is already
    # clipped, but the SUM (ddt_partial + ddt_contrib2 here) is a new
    # accumulation boundary nothing upstream guards.
    ddt = _final_sanitize(ddt)
    dx = _final_sanitize(dx)
    dB = _final_sanitize(dB)
    dC = _final_sanitize(dC)
    dA = _final_sanitize(dA)
    dstate0 = _final_sanitize(dstate0)

    # FIX (dtype): custom_vjp requires cotangents to match the
    # corresponding forward input's dtype at the boundary -- dt/x/B/C are
    # bfloat16 in real usage (Mamba2J), A/state0 are float32. Every
    # intermediate in the MB1-MB4 chain is float32 throughout (no
    # `.astype(f32)`-on-input for autodiff to piggyback a free downcast
    # off of, since this backward is hand-assembled, not obtained by
    # differentiating through the forward) -- same failure mode
    # kernel_trainable_B6.py's own FIX #2 and kernel_mamba2_trainable.py's
    # own FIX document at length for GDN-2/Mamba2 respectively.
    ddt = ddt.astype(dt.dtype)
    dx = dx.astype(x.dtype)
    dB = dB.astype(B.dtype)
    dC = dC.astype(C.dtype)
    dA = dA.astype(A.dtype)
    dstate0 = dstate0.astype(state0.dtype)

    return ddt, dx, dB, dC, dA, dstate0


_mamba2_core.defvjp(_mamba2_core_fwd, _mamba2_core_bwd)


def mamba2_pallas_forward_trainable_B6(dt, x, B, C, A, chunk_size, state0=None):
    """Drop-in replacement for kernel_mamba2_trainable.mamba2_pallas_forward_trainable
    -- SAME call signature, but backward is the honest fused MB1-MB4 chain
    instead of jax.vjp on mamba2_ssd_reference. Do NOT wire this into
    model.py until MB6 (gradient comparison against
    kernel_mamba2_trainable.py, several seeds/shapes, on real TPU
    interpret=False) has been run and confirmed finite/rel_diff-small --
    same discipline kernel_trainable_B6.py itself was held to before
    replacing kernel_trainable.py in model.py's GDN-2 import.

    dt, x: (bsz, L, n_heads_ssm, headdim). B, C: (bsz, L, d_state).
    A: (n_heads_ssm,), float32. state0: optional (bsz, n_heads_ssm,
    headdim, d_state), float32 -- zeros if omitted.
    """
    bsz, L, n_heads_ssm, headdim = dt.shape
    d_state = B.shape[-1]
    if state0 is None:
        state0 = jnp.zeros((bsz, n_heads_ssm, headdim, d_state), dtype=jnp.float32)
    return _mamba2_core(dt, x, B, C, A, chunk_size, state0)
