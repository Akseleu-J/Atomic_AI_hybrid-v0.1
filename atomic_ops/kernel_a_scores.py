"""
Milestone 3 -- Kernel A (Pallas/TPU): builds the intra-chunk Aqk/Akk score
matrices for one (batch, head, chunk). Unchanged from the validated project
version (already carries the decay_diff clip + nan_to_num defenses).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HIGHEST = jax.lax.Precision.HIGHEST

BT = 256
BC = 128
N_SUB = BT // BC


def _weighted_pair_sum(a_i, edecay, b_j):
    tmp = a_i[:, None, :] * edecay
    tmp = tmp * b_j[None, :, :]
    return jnp.sum(tmp, axis=-1)


def _kernel_a_body(q_ref, k_ref, b_ref, g_ref, aqk_ref, akk_ref, *, scale):
    q_full = q_ref[0, 0, 0].astype(jnp.float32)
    k_full = k_ref[0, 0, 0].astype(jnp.float32)
    b_full = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(BT)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

    aqk_ref[0, 0, 0] = jnp.zeros((BT, BT), dtype=jnp.float32)
    akk_ref[0, 0, 0] = jnp.zeros((BT, BT), dtype=jnp.float32)

    for si in range(N_SUB):
        for sj in range(si + 1):
            i0, i1 = si * BC, (si + 1) * BC
            j0, j1 = sj * BC, (sj + 1) * BC

            q_i = q_full[i0:i1]
            k_i = k_full[i0:i1]
            k_j = k_full[j0:j1]
            b_i = b_full[i0:i1]
            gc_i = gc[i0:i1]
            gc_j = gc[j0:j1]

            decay_diff = gc_i[:, None, :] - gc_j[None, :, :]
            edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))

            aqk_blk = scale * _weighted_pair_sum(q_i, edecay, k_j)
            bk_i = b_i * k_i
            akk_blk = _weighted_pair_sum(bk_i, edecay, k_j)

            if si == sj:
                idx = jnp.arange(BC)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
                aqk_blk = aqk_blk * causal
                akk_blk = akk_blk * strict

            aqk_ref[0, 0, 0, i0:i1, j0:j1] = jnp.nan_to_num(aqk_blk, nan=0.0, posinf=1e4, neginf=-1e4)
            akk_ref[0, 0, 0, i0:i1, j0:j1] = jnp.nan_to_num(akk_blk, nan=0.0, posinf=1e4, neginf=-1e4)


def build_chunk_scores_pallas(q, k, b, g, scale):
    bsz, L, H, D = q.shape
    assert D == 128, f"Kernel A assumes d_head=128 (MXU tile); got D={D}."
    assert L % BT == 0, f"seq_len={L} must be divisible by BT={BT}."
    n_chunks = L // BT

    def reshape_in(t):
        t = t.reshape(bsz, n_chunks, BT, H, D)
        return jnp.moveaxis(t, (1, 3), (2, 1))

    q_r, k_r, b_r, g_r = map(reshape_in, (q, k, b, g))

    grid = (bsz, H, n_chunks)

    in_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))
    out_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))

    aqk, akk = pl.pallas_call(
        lambda *refs: _kernel_a_body(*refs, scale=scale),
        grid=grid,
        in_specs=[in_spec, in_spec, in_spec, in_spec],
        out_specs=[out_spec, out_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, BT), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, BT), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
    )(q_r, k_r, b_r, g_r)

    return aqk, akk
