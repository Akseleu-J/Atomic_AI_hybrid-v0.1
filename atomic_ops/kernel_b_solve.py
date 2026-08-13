"""
Milestone 3 -- Kernel B (Pallas/TPU): WY solve A = (I + Akk)^{-1}.

FIX (non-finite hardening, found while diagnosing a forward non-finite that
appeared at step ~780 of a real training run, block=4/layer=14/type=gdn2):
this was the ONE place in the whole project using only `nan_to_num` on its
output, without a `clip` first -- every other kernel (B3, B4, B5,
kernel_d_pipeline's scan step) uses the `clip(...) + nan_to_num(...)`
combination specifically because `nan_to_num` alone only replaces VALUES
THAT ARE ALREADY nan/inf; it does nothing to a value that is merely huge but
still finite (e.g. ~1e25). Forward substitution against a near-singular
`Akk` (which can happen for a specific (batch, head, chunk) once decay/erase
gate parameters have drifted far enough into training) can produce exactly
that: a large-but-finite `A`. That unbounded magnitude then flows on,
unclipped, into Kernel C's `w_pseudo`/`u`/`kg`/`qg` and finally overflows to
an actual `inf` only once it hits a later matmul in Kernel D's inter-chunk
scan -- by which point the ORIGINAL cause (this near-singular WY-solve) is
several kernels removed from where the `inf` finally appears, which is
exactly why the existing [FWD-DIAG] block/layer-level check couldn't
pinpoint it. Clipping A here, at the actual source, closes the gap instead
of only reacting to its downstream symptom.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .kernel_a_scores import BT, BC, N_SUB

_HIGHEST = jax.lax.Precision.HIGHEST
_CLIP = 1e4

assert N_SUB == 2, "Kernel B currently implements only the 2-subblock (BT=2*BC) case."


def _sanitize(x):
    return jnp.nan_to_num(jnp.clip(x, -_CLIP, _CLIP), nan=0.0, posinf=_CLIP, neginf=-_CLIP)


def _bc_forward_substitution(T):
    bc = T.shape[-1]
    idx = jnp.arange(bc)

    def body(i, A):
        onehot_i = (idx == i).astype(jnp.float32)
        t_row = jnp.sum(T * onehot_i[:, None], axis=0)
        contrib = jnp.sum(t_row[:, None] * A, axis=0)
        new_row = onehot_i - contrib
        mask_col = onehot_i[:, None]
        A = A * (1.0 - mask_col) + mask_col * new_row[None, :]
        return A

    A0 = jnp.zeros((bc, bc), dtype=jnp.float32)
    return jax.lax.fori_loop(0, bc, body, A0)


def _kernel_b_body(akk_ref, a_ref):
    Akk = akk_ref[0, 0, 0].astype(jnp.float32)

    T00 = Akk[0:BC, 0:BC]
    T11 = Akk[BC:2 * BC, BC:2 * BC]
    T10 = Akk[BC:2 * BC, 0:BC]

    A00 = _bc_forward_substitution(T00)
    A11 = _bc_forward_substitution(T11)

    tmp = jnp.dot(T10, A00, precision=_HIGHEST)
    A10 = -jnp.dot(A11, tmp, precision=_HIGHEST)

    # FIX: clip, not just nan_to_num -- see module docstring. A near-singular
    # Akk can make forward substitution produce a large-but-finite A that
    # nan_to_num alone would let straight through.
    A00 = _sanitize(A00)
    A10 = _sanitize(A10)
    A11 = _sanitize(A11)

    a_ref[0, 0, 0] = jnp.zeros((BT, BT), dtype=jnp.float32)
    a_ref[0, 0, 0, 0:BC, 0:BC] = A00
    a_ref[0, 0, 0, BC:2 * BC, 0:BC] = A10
    a_ref[0, 0, 0, BC:2 * BC, BC:2 * BC] = A11


def wy_solve_pallas(Akk):
    bsz, H, n_chunks = Akk.shape[:3]
    assert Akk.shape[-2:] == (BT, BT)
    grid = (bsz, H, n_chunks)

    spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))

    A = pl.pallas_call(
        _kernel_b_body,
        grid=grid,
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(Akk.shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
    )(Akk)

    return A
