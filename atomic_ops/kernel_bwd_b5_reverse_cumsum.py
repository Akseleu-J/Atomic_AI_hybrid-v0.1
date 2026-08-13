"""
Milestone B5 -- reverse cumsum for dg. Unchanged from validated project
version. Added a final nan_to_num as a cheap belt-and-braces (the original
had none at all since it's "just" an adjoint matmul -- but reverse-cumsum
SUMS up to BT=256 upstream dgc rows into one dg_raw row, so a single
large-but-finite dgc entry can blow this up to a magnitude worth clamping
before it hits the optimizer).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


def reverse_cumsum_bwd(dgc, chunk_size):
    C = chunk_size
    idx = jnp.arange(C)
    triu_ones = (idx[:, None] <= idx[None, :]).astype(jnp.float32)

    dg_raw = jnp.einsum("ij,...jd->...id", triu_ones, dgc.astype(jnp.float32), precision=_HIGHEST)
    dg_raw = jnp.nan_to_num(jnp.clip(dg_raw, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
    return dg_raw
