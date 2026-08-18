"""
Milestone M5 -- trainable wrapper for the Mamba2 SSD Pallas pipeline
(M2 cumdecay -> M3 intra-chunk -> M4 inter-chunk, all wired together via
kernel_mamba2_c_interchunk.mamba2_pallas_forward).

Same ordering discipline this project already used for GDN-2
(atomic_ops/kernel_trainable.py before kernel_trainable_B6.py): "cheat"
backward first -- Pallas forward, but backward computed via jax.vjp on the
already-validated pure-JAX reference (mamba2_ssd_reference.py), NOT a
hand-fused Pallas backward. This is cheap to get right and lets training
start / gradients get validated before investing in a fully fused backward
(a later milestone, if the reference-vjp backward ever turns out to be a
real bottleneck -- same reasoning kernel_trainable.py's own docstring gives
for keeping the "cheat" path around as a permanent cross-check target, not
just a stepping stone).

FIX (dtype, anticipating the same failure mode kernel_trainable_B6.py's own
docstring documents at length for GDN-2): dt/x/B/C arrive from Mamba2J as
bfloat16 (all four go through nn.Dense(..., dtype=jnp.bfloat16) or the bf16
depthwise conv), while every internal computation in mamba2_ssd_reference.py
(and therefore its jax.vjp cotangents) is float32. Because this wrapper's
`_fwd`/`_bwd` are hand-written (not obtained by literally differentiating
through an `.astype(f32)` the way gdn2_chunked_wy_reference's OWN inputs
are cast, which lets JAX auto-downcast the cotangent for free) an explicit
`.astype(orig_dtype)` cast is required on every returned gradient, exactly
like kernel_trainable_B6.py's FIX #2. Unlike kernel_trainable_B6.py, this
wrapper's `_bwd` calls jax.vjp on `mamba2_ssd_reference` directly -- and
mamba2_ssd_reference.py's `_chunk_ssd` DOES do `.astype(f32)` on its own
inputs internally (see its own module docstring: "dt_c.astype(f32)" etc.)
-- so in principle autodiff through that cast alone might already return
correctly-downcast cotangents. The explicit final `.astype()` below is kept
anyway as a belt-and-braces measure (cheap, and removes any doubt) -- same
"don't rely on an implicit cast surviving custom_vjp's boundary" caution
kernel_trainable_B6.py's docstring itself recommends.

A: passed through as float32 always (matches mamba2_ssd_reference.py's own
convention -- A is `-jnp.exp(A_log_safe)`, already float32 in Mamba2J,
never bf16), so no cast needed for its gradient.

Sanitization: same final clip(+-1e4)+nan_to_num pass on every returned
gradient as kernel_trainable.py/kernel_trainable_B6.py -- the custom_vjp
return boundary is the last place a non-finite or absurdly-large gradient
can be caught before an optimizer step consumes it.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .kernel_mamba2_c_interchunk import mamba2_pallas_forward
from .mamba2_ssd_reference import mamba2_ssd_reference

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

    def ref_forward(dt_, x_, B_, C_, A_, state0_):
        return mamba2_ssd_reference(dt_, A_, B_, C_, x_, chunk_size=chunk_size, state0=state0_)

    _, vjp_fn = jax.vjp(ref_forward, dt, x, B, C, A, state0)
    ddt, dx, dB, dC, dA, dstate0 = vjp_fn((do, dstate_final))

    ddt, dx, dB, dC, dA, dstate0 = map(
        _final_sanitize, (ddt, dx, dB, dC, dA, dstate0)
    )

    # FIX: cast every gradient back to its corresponding forward input's
    # dtype -- see module docstring. dt/x/B/C are bfloat16 in real usage
    # (Mamba2J), A/state0 are float32. custom_vjp requires cotangents to
    # match forward-input dtype at the boundary or training crashes one
    # level up the graph the moment this cotangent has to combine with a
    # bfloat16-typed gradient elsewhere.
    ddt = ddt.astype(dt.dtype)
    dx = dx.astype(x.dtype)
    dB = dB.astype(B.dtype)
    dC = dC.astype(C.dtype)
    dA = dA.astype(A.dtype)
    dstate0 = dstate0.astype(state0.dtype)

    return ddt, dx, dB, dC, dA, dstate0


_mamba2_core.defvjp(_mamba2_core_fwd, _mamba2_core_bwd)


def mamba2_pallas_forward_trainable(dt, x, B, C, A, chunk_size, state0=None):
    """Drop-in trainable version of mamba2_pallas_forward -- Pallas forward
    (M2->M3->M4), backward via jax.vjp on the pure-JAX SSD reference
    (mamba2_ssd_reference.py). Same call signature as mamba2_pallas_forward
    minus `interpret` (always compiled -- this is the path meant for real
    training, not kernel-development debugging).

    dt, x: (bsz, L, n_heads_ssm, headdim). B, C: (bsz, L, d_state).
    A: (n_heads_ssm,), float32. state0: optional (bsz, n_heads_ssm, headdim,
    d_state), float32 -- zeros if omitted.
    """
    bsz, L, n_heads_ssm, headdim = dt.shape
    d_state = B.shape[-1]
    if state0 is None:
        state0 = jnp.zeros((bsz, n_heads_ssm, headdim, d_state), dtype=jnp.float32)
    return _mamba2_core(dt, x, B, C, A, chunk_size, state0)
