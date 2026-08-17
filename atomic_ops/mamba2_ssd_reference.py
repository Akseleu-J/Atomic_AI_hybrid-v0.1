"""
Milestone M1 -- pure-JAX chunked SSD reference for Mamba2, to replace the
current associative_scan-based _chunk_step in model.py's Mamba2J.

ASSUMES M0 DECISION (a): decay is grouped PER HEAD, not per channel.
    A_log: (n_heads_ssm,)   -- was (d_inner,) in the current associative_scan code
    d_inner = n_heads_ssm * headdim
This matches upstream Mamba2 / HF modeling_mamba2.py and is what makes the
intra-chunk decay matrix L a manageable (n_heads_ssm, BT, BT) instead of
(d_inner, BT, BT). If you decide to keep true per-channel decay instead,
this file (and the Pallas ports M2/M3 built on it) need a different design
-- see atomic_ops/MAMBA2_PALLAS_PLAN.md M0 note.

Algorithm (standard Mamba2 SSD chunk decomposition, same "quadratic form
inside chunk + carried state across chunks" shape as GDN-2's chunk-parallel
approach, but WITHOUT the WY/delta-rule correction -- Mamba2 has no erase
term, so there's no matrix inverse here, just:

  decay_diff[h,i,j]  = cumdecay[h,i] - cumdecay[h,j]      (i>=j causal)
  L[h,i,j]           = exp(clip(decay_diff, -20, 20)) * causal_mask
  Y_diag[b,i,h,p]    = sum_j L[h,i,j] * (C[b,i,:] . B[b,j,:]) * dBx[b,j,h,p]
                        where dBx = dt * x  (the "input" being written, per
                        head/channel), same role as GDN-2's `bk` write term
                        but WITHOUT the b_gate erase -- Mamba2 always fully
                        writes, only decay ever erases.
  state_end[b,h,p,s] = sum_j B[b,j,s] * dBx[b,j,h,p] * exp(cumdecay[h,-1]-cumdecay[h,j])
  Y_off[b,i,h,p]     = sum_s C[b,i,s] * state_prev[b,h,p,s] * exp(cumdecay[h,i])
  Y[b,i,h,p]         = Y_diag + Y_off
  state_new          = state_prev * exp(cumdecay[h,-1]) + state_end

Same sanitization convention as the rest of atomic_ops/: clip+nan_to_num at
every accumulation boundary, precision=HIGHEST on all matmuls.

FIX (this pass, correctness bug found via M1 validation against
token_serial_ground_truth in test_mamba2_ssd_reference.py --
rel_err(y) ~= 2.1, way above the 1e-4 bar): `cumdecay` -- the RUNNING SUM
of `dA_exponent` over up to `chunk_size` steps (64-256 in the tests) -- was
being clipped to the same +-20 bound used for the per-step `dA_exponent`
itself. Each individual per-step exponent is legitimately bounded to
+-20 (matches the clip convention used everywhere else in this project),
but the CUMULATIVE sum over many steps can legitimately reach far beyond
that (e.g. -2 * 64 = -128 over just a 64-step chunk with a fast-decaying
head). Clamping the cumulative value itself to +-20 silently truncated
real decay history -- and since `decay_diff = cumdecay_i - cumdecay_j` is
built by subtracting two INDEPENDENTLY clipped cumdecay values, the
resulting pairwise decay used to build the whole `L` matrix came out wrong
for essentially every (i,j) pair whose true cumulative decay exceeded the
clip window. The token-serial ground truth never clips at all, so this was
purely an artifact of an over-tight sanitize call, not a real numerical
overflow risk (module-level `_CLIP=1e4` is the right bound for that
purpose -- see `_sanitize`'s own default). The per-pair DIFFERENCE
(`decay_diff`, right before its own `exp()`) is still clipped to +-20,
which is the numerically-sensitive/correct place for that bound, since at
that point it's a bounded quantity feeding an `exp`, not an unbounded
running sum.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST
_CLIP = 1e4


def _sanitize(x, clip=_CLIP):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


def _chunk_ssd(dt_c, A, B_c, C_c, x_c, state_prev):
    """One chunk. Shapes (chunk_size=C):
    dt_c:  (b, C, n_heads, headdim)   -- per-(head,channel) timestep, dt already clipped upstream
    A:     (n_heads,)                 -- log-decay-rate-per-unit-dt is A itself (already = -exp(A_log))
    B_c:   (b, C, d_state)
    C_c:   (b, C, d_state)
    x_c:   (b, C, n_heads, headdim)
    state_prev: (b, n_heads, headdim, d_state)   fp32 carry
    Returns: y_c (b, C, n_heads, headdim), state_new (same shape as state_prev)
    """
    f32 = jnp.float32
    Cs = dt_c.shape[1]

    # dA per (head, channel, time) = dt * A[head] -- same clip discipline as
    # the current associative_scan code's dA_exponent/dA clip.
    dA_exponent = jnp.einsum("bchd,h->bchd", dt_c.astype(f32), A.astype(f32))
    dA_exponent = _sanitize(dA_exponent, clip=20.0)
    # decay is per (head, channel) now, not shared across channel within head
    # -- NOTE: if M0=(a) (decay shared per head), A is (n_heads,), so
    # dA_exponent is already constant across the 'headdim' axis in effect
    # (same A broadcast), but dt still varies per-channel, so decay CAN
    # still differ per channel through dt. This matches upstream Mamba2
    # (dt varies per channel even though A doesn't).

    # cumulative log-decay within chunk, per (b,c,h,d) -- tril-matmul trick
    # (jnp.cumsum doesn't lower in Pallas; keeping the SAME trick here even
    # in the pure-JAX reference so M1 and the eventual M2/M3 Pallas port
    # share identical numerics, not just "close enough").
    idx = jnp.arange(Cs)
    tril_ones = (idx[:, None] >= idx[None, :]).astype(f32)
    cumdecay = jnp.einsum("ij,bjhd->bihd", tril_ones, dA_exponent, precision=_HIGHEST)
    # FIX: cumdecay is a RUNNING SUM of up to chunk_size per-step
    # dA_exponent values, each already legitimately clipped to +-20. The
    # cumulative sum itself can legitimately reach magnitudes far beyond
    # +-20 (e.g. -2 * 64 = -128 over a 64-step chunk) -- clipping the
    # CUMULATIVE value itself to the same +-20 bound as the per-step clip
    # silently truncates real decay history. Since decay_diff below is
    # built by SUBTRACTING two independently-clipped cumdecay values
    # (cumdecay_i - cumdecay_j), that truncation corrupted the pairwise
    # decay used to build L for every (i,j) pair in the chunk -- this was
    # the actual source of the ~2x rel_err against the token-serial ground
    # truth (which never clips at all). Only guard against genuine overflow
    # here (module-level _CLIP=1e4); the numerically-sensitive clip belongs
    # on the per-pair DIFFERENCE (decay_diff, still clipped to +-20 right
    # before its own exp() below), not on the raw cumulative sum.
    cumdecay = _sanitize(cumdecay, clip=_CLIP)

    decay_diff = cumdecay[:, :, None, :, :] - cumdecay[:, None, :, :, :]  # (b,i,j,h,d)
    causal = (idx[:, None] >= idx[None, :]).astype(f32)[None, :, :, None, None]
    L = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0)) * causal
    L = _sanitize(L, clip=1e6)

    BC_inner = jnp.einsum("bis,bjs->bij", C_c.astype(f32), B_c.astype(f32), precision=_HIGHEST)
    BC_inner = _sanitize(BC_inner)

    dBx = dt_c.astype(f32) * x_c.astype(f32)
    dBx = _sanitize(dBx)

    # Y_diag[b,i,h,d] = sum_j L[b,i,j,h,d] * BC_inner[b,i,j] * dBx[b,j,h,d]
    weight = L * BC_inner[:, :, :, None, None]
    y_diag = jnp.einsum("bijhd,bjhd->bihd", weight, dBx, precision=_HIGHEST)
    y_diag = _sanitize(y_diag)

    decay_to_end = jnp.exp(jnp.clip(cumdecay[:, -1:, :, :] - cumdecay, -20.0, 20.0))
    decay_to_end = _sanitize(decay_to_end, clip=1e6)
    write = dBx * decay_to_end  # (b,C,h,d)
    state_end = jnp.einsum("bcs,bchd->bhds", B_c.astype(f32), write, precision=_HIGHEST)
    state_end = _sanitize(state_end)

    decay_h = jnp.exp(jnp.clip(cumdecay, -20.0, 0.0))  # (b,C,h,d), for reading old state
    y_off = jnp.einsum("bis,bhds->bihd", C_c.astype(f32), state_prev, precision=_HIGHEST)
    y_off = y_off * decay_h
    y_off = _sanitize(y_off)

    decay_chunk_end = jnp.exp(jnp.clip(cumdecay[:, -1], -20.0, 0.0))  # (b,h,d)
    state_new = state_prev * decay_chunk_end[..., None] + state_end
    state_new = _sanitize(state_new)

    y_c = _sanitize(y_diag + y_off)
    return y_c, state_new


def mamba2_ssd_reference(dt, A, B, C, x, chunk_size, state0=None):
    """dt,x: (b,l,n_heads,headdim). A: (n_heads,). B,C: (b,l,d_state).
    Returns y: (b,l,n_heads,headdim), state_final: (b,n_heads,headdim,d_state).
    """
    b, l, n_heads, headdim = dt.shape
    d_state = B.shape[-1]
    assert l % chunk_size == 0
    n_chunks = l // chunk_size

    def to_chunks(t):
        shp = t.shape
        t = t.reshape(b, n_chunks, chunk_size, *shp[2:])
        return jnp.moveaxis(t, 1, 0)

    dt_ch, B_ch, C_ch, x_ch = map(to_chunks, (dt, B, C, x))

    if state0 is None:
        state0 = jnp.zeros((b, n_heads, headdim, d_state), dtype=jnp.float32)

    def step(state_prev, inputs):
        dt_c, B_c, C_c, x_c = inputs
        y_c, state_new = _chunk_ssd(dt_c, A, B_c, C_c, x_c, state_prev)
        return state_new, y_c

    step = jax.checkpoint(step)
    state_final, y_scanned = jax.lax.scan(step, state0, (dt_ch, B_ch, C_ch, x_ch))
    y = jnp.moveaxis(y_scanned, 0, 1).reshape(b, l, n_heads, headdim)
    return y, state_final
