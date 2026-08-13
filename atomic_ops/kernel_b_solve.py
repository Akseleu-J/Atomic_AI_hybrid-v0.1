"""
Milestone 3 -- Kernel B (Pallas/TPU): WY solve A = (I + Akk)^{-1}.
Unchanged from validated project version.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .kernel_a_scores import BT, BC, N_SUB

_HIGHEST = jax.lax.Precision.HIGHEST

assert N_SUB == 2, "Kernel B currently implements only the 2-subblock (BT=2*BC) case."


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

    A00 = jnp.nan_to_num(A00, nan=0.0, posinf=1e4, neginf=-1e4)
    A10 = jnp.nan_to_num(A10, nan=0.0, posinf=1e4, neginf=-1e4)
    A11 = jnp.nan_to_num(A11, nan=0.0, posinf=1e4, neginf=-1e4)

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
