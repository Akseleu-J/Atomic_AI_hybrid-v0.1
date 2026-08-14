"""
Milestone 3 -- Kernel B (Pallas/TPU): WY solve A = (I + Akk)^{-1}.

REWRITE (this pass, BC=128 -> MB=16 block-recursive stability + perf fix):
numerical stress-testing (see project notes / test_bc_solve_stability*.py)
compared the original flat 128-step forward substitution against a
block-recursive version with MB=16 micro-blocks (matching the original
Triton authors' sub-block scale, vs our MXU-driven BC=128):

  - Accuracy: on structured near-singular Akk (the class most relevant to
    the step-710/784 instability incidents -- decay collapsed to similar
    values across a chunk), the blocked version gave consistently lower
    relative error vs an fp64 reference across the whole tested magnitude
    range (roughly 1-3x lower error), though NOT the dramatic "cliff" a
    single lucky/unlucky synthetic sample first suggested -- that was
    sampling noise, not a real threshold effect. Treat this as "somewhat
    more numerically robust", not "eliminates the failure mode".
  - Performance: on the real production grid shape (bsz=8, H=6, n_chunks=16,
    768 total (BC,BC) solves), the blocked version was ~3.5x FASTER than
    the flat version (4.24ms vs 14.75ms steady-state) -- the flat
    algorithm's 128 strictly-sequential fori_loop steps do not parallelize
    well on TPU, while the blocked version's statically-unrolled 8-block
    structure gives XLA much more to fuse/schedule on the MXU.

Net: this rewrite is adopted primarily because it is a clear performance
win with no observed accuracy downside, AND a modest accuracy upside on
the adversarial case it was designed to test. It is NOT presented as a
proven fix for the step-710 incident -- see kernel_trainable_B6.py's Akk
dump hook for how to validate that directly against real training data.

Same clip+nan_to_num convention as before (kernel_b_solve.py's original
docstring's reasoning about near-singular Akk producing large-but-finite
A still applies -- see that reasoning preserved below), just applied at
every block-matmul boundary instead of only at the two 2x2-block boundaries
the old N_SUB=2 top-level split had.

Original top-level 2x2 split (T00/T11/T10 -> A00/A10/A11, driven by
N_SUB=BT//BC=2) is now itself expressed via this same block-recursive solve
with N_MICRO=8, MB=16 blocks inside each BC=128 sub-chunk -- i.e. the old
special-cased 2-block code path is now just a special case of the general
block solve below and has been removed in favor of it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .kernel_a_scores import BT, BC, N_SUB

_HIGHEST = jax.lax.Precision.HIGHEST
_CLIP = 1e4

MB = 16
assert BC % MB == 0, f"BC={BC} must be divisible by MB={MB}."
N_MICRO = BC // MB

assert N_SUB == 2, "Kernel B currently implements only the 2-subblock (BT=2*BC) top-level case."


def _sanitize(x):
    return jnp.nan_to_num(jnp.clip(x, -_CLIP, _CLIP), nan=0.0, posinf=_CLIP, neginf=-_CLIP)


def _micro_forward_substitution(T_mb):
    """Forward substitution for one MB x MB strictly-lower-triangular block.
    Same one-hot-masked fori_loop pattern as the original _bc_forward_substitution
    (Mosaic doesn't lower dynamic_slice/dynamic_update_slice inside fori_loop --
    see project handoff sec 5.3), just at depth MB=16 instead of BC=128. Also
    sanitizes each row write (not just the final block) -- this recurrence is
    exactly the deep-dependency-chain risk the rewrite targets, so it gets the
    same per-step defense B4's accumulation loop uses."""
    idx = jnp.arange(MB)

    def body(i, A):
        onehot_i = (idx == i).astype(jnp.float32)
        t_row = jnp.sum(T_mb * onehot_i[:, None], axis=0)
        contrib = jnp.sum(t_row[:, None] * A, axis=0)
        new_row = onehot_i - contrib
        new_row = _sanitize(new_row)
        mask_col = onehot_i[:, None]
        A = A * (1.0 - mask_col) + mask_col * new_row[None, :]
        return A

    A0 = jnp.zeros((MB, MB), dtype=jnp.float32)
    return jax.lax.fori_loop(0, MB, body, A0)


def _block_solve(T_full):
    """T_full: (BC, BC) strictly lower triangular in VMEM (already sliced out
    of the (BT,BT) Akk block by the caller). Returns A = (I + T_full)^{-1}
    via block forward substitution over N_MICRO=8 blocks of size MB=16.

    All (m, n, k) loop bounds are static Python ints (N_MICRO=8 is a
    compile-time constant) -- this unrolls into a static sequence of
    (BC,BC)-resident slice/matmul ops, no dynamic indexing, matching the
    project's Mosaic constraints (handoff sec 5.3/5.5).
    """
    blocks = [[None] * N_MICRO for _ in range(N_MICRO)]

    for m in range(N_MICRO):
        T_mm = T_full[m * MB:(m + 1) * MB, m * MB:(m + 1) * MB]
        A_mm = _sanitize(_micro_forward_substitution(T_mm))
        blocks[m][m] = A_mm

        for n in range(m - 1, -1, -1):
            acc = jnp.zeros((MB, MB), dtype=jnp.float32)
            for k in range(n, m):
                T_mk = T_full[m * MB:(m + 1) * MB, k * MB:(k + 1) * MB]
                A_kn = blocks[k][n]
                contrib = jnp.dot(T_mk, A_kn, precision=_HIGHEST)
                acc = _sanitize(acc + contrib)
            A_mn = -jnp.dot(A_mm, acc, precision=_HIGHEST)
            A_mn = _sanitize(A_mn)
            blocks[m][n] = A_mn

    rows = []
    for m in range(N_MICRO):
        row_blocks = []
        for n in range(N_MICRO):
            if n > m:
                row_blocks.append(jnp.zeros((MB, MB), dtype=jnp.float32))
            else:
                row_blocks.append(blocks[m][n])
        rows.append(jnp.concatenate(row_blocks, axis=1))
    return jnp.concatenate(rows, axis=0)


def _kernel_b_body(akk_ref, a_ref):
    Akk = akk_ref[0, 0, 0].astype(jnp.float32)

    T00 = Akk[0:BC, 0:BC]
    T11 = Akk[BC:2 * BC, BC:2 * BC]
    T10 = Akk[BC:2 * BC, 0:BC]

    A00 = _block_solve(T00)
    A11 = _block_solve(T11)

    tmp = jnp.dot(T10, A00, precision=_HIGHEST)
    tmp = _sanitize(tmp)
    A10 = -jnp.dot(A11, tmp, precision=_HIGHEST)
    A10 = _sanitize(A10)

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
        # NOTE: block-recursive solve holds more short-lived (MB,MB)
        # intermediates than the old 2-block version -- bumped vmem
        # headroom accordingly. Reduce back toward 64MB if profiling
        # shows this is unnecessarily high.
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=96 * 1024 * 1024),
    )(Akk)

    return A
