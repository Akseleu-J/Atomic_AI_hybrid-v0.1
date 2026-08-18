"""MB1+MB3 orchestrator integration test.

Cross-checks kernel_mamba2_bwd_b1_state.mamba2_bwd_scan (two-pass:
state-only reverse scan + MB2 Pallas call, combined) against
mamba2_bwd_reference.chunk_ssd_bwd_scan (MB0, full single-pass reference)
-- same "hand derivation -> jax.vjp cross-check -> Pallas port -> TPU
test" discipline, at the INTEGRATION level this time (individual pieces
already passed their own cross-checks in test_mamba2_bwd_state_reference.py
and test_kernel_mamba2_bwd_b2_intra.py).

dx, dB, dC, dstate0 are expected to match MB0 EXACTLY (same math, only
reorganized into two passes -- no MB4-dependent partiality for these four).

ddt and dcumdecay are only PARTIAL/internal at this milestone (see
kernel_mamba2_bwd_b1_state.py's own docstring's "ownership" section) --
MB0's public chunk_ssd_bwd_scan never exposes dcumdecay at all, and its
ddt is the FULL ddt (ddt_c_1 + ddt_c_2, the latter requiring MB4). To
still get a real cross-check on these two (not just "trust the pieces"),
`_chunk_ssd_bwd_with_partials` below is a byte-for-byte copy of MB0's own
`_chunk_ssd_bwd` with two extra return values spliced in (ddt_c_1 alone,
and d_cumdecay_total before it's consumed by the reverse-cumsum/dA chain)
-- test-only, not a new derivation, just exposing internals MB0's public
function intentionally keeps private.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from atomic_ops.mamba2_ssd_reference import _chunk_ssd, _sanitize
from atomic_ops.mamba2_bwd_reference import chunk_ssd_bwd_scan, _clip_mask
from atomic_ops.kernel_mamba2_bwd_b1_state import mamba2_bwd_scan

_HIGHEST = jax.lax.Precision.HIGHEST


def _rel_err(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    return float(jnp.linalg.norm((a - b).ravel()) / (jnp.linalg.norm(b.ravel()) + 1e-8))


def _chunk_ssd_bwd_with_partials(dt_c, A, B_c, C_c, x_c, state_prev, dy_c, dstate_new):
    """Test-only: identical to mamba2_bwd_reference._chunk_ssd_bwd, plus
    returns ddt_c_1 (dBx-only contribution -- what the orchestrator's
    partial ddt should match) and d_cumdecay_total (pre-reverse-cumsum --
    what the orchestrator's combined dcumdecay should match)."""
    f32 = jnp.float32
    dt_c = dt_c.astype(f32)
    B_c = B_c.astype(f32)
    C_c = C_c.astype(f32)
    x_c = x_c.astype(f32)
    state_prev = state_prev.astype(f32)
    Cs = dt_c.shape[1]

    dA_exponent_raw = jnp.einsum("bchd,h->bchd", dt_c, A.astype(f32))
    dA_exponent = _sanitize(dA_exponent_raw, clip=20.0)

    idx = jnp.arange(Cs)
    tril_ones = (idx[:, None] >= idx[None, :]).astype(f32)
    cumdecay_raw = jnp.einsum("ij,bjhd->bihd", tril_ones, dA_exponent, precision=_HIGHEST)
    cumdecay = _sanitize(cumdecay_raw, clip=1e4)

    decay_diff_raw = cumdecay[:, :, None, :, :] - cumdecay[:, None, :, :, :]
    causal = (idx[:, None] >= idx[None, :]).astype(f32)[None, :, :, None, None]
    L = _sanitize(jnp.exp(jnp.clip(decay_diff_raw, -20.0, 20.0)) * causal, clip=1e6)

    BC_inner = _sanitize(jnp.einsum("bis,bjs->bij", C_c, B_c, precision=_HIGHEST))
    dBx = _sanitize(dt_c * x_c)
    weight = L * BC_inner[:, :, :, None, None]

    decay_to_end_raw = cumdecay[:, -1:, :, :] - cumdecay
    decay_to_end = _sanitize(jnp.exp(jnp.clip(decay_to_end_raw, -20.0, 20.0)), clip=1e6)
    write = _sanitize(dBx * decay_to_end)

    decay_h_raw = jnp.clip(cumdecay, -20.0, 0.0)
    decay_h = jnp.exp(decay_h_raw)
    y_off_raw = jnp.einsum("bis,bhds->bihd", C_c, state_prev, precision=_HIGHEST)
    decay_chunk_end_raw = jnp.clip(cumdecay[:, -1], -20.0, 0.0)
    decay_chunk_end = jnp.exp(decay_chunk_end_raw)

    dy_diag = dy_c
    dy_off = dy_c

    dstate_prev = dstate_new * decay_chunk_end[..., None]
    d_decay_chunk_end = jnp.sum(dstate_new * state_prev, axis=-1)
    dstate_end = dstate_new
    d_cumdecay_last_a = d_decay_chunk_end * decay_chunk_end * _clip_mask(cumdecay[:, -1], -20.0, 0.0)

    dy_off_raw = dy_off * decay_h
    d_decay_h = dy_off * y_off_raw
    dC_c_1 = jnp.einsum("bihd,bhds->bis", dy_off_raw, state_prev, precision=_HIGHEST)
    dstate_prev = dstate_prev + jnp.einsum("bihd,bis->bhds", dy_off_raw, C_c, precision=_HIGHEST)
    d_cumdecay_from_yoff = d_decay_h * decay_h * _clip_mask(cumdecay, -20.0, 0.0)

    dwrite = jnp.einsum("bhds,bcs->bchd", dstate_end, B_c, precision=_HIGHEST)
    dB_c_1 = jnp.einsum("bhds,bchd->bcs", dstate_end, write, precision=_HIGHEST)
    d_dBx_1 = dwrite * decay_to_end
    d_decay_to_end = dwrite * dBx
    d_decay_to_end_diff = d_decay_to_end * decay_to_end * _clip_mask(decay_to_end_raw, -20.0, 20.0)
    d_cumdecay_last_b = jnp.sum(d_decay_to_end_diff, axis=1)
    d_cumdecay_from_decay_to_end = -d_decay_to_end_diff

    dweight = jnp.einsum("bihd,bjhd->bijhd", dy_diag, dBx, precision=_HIGHEST)
    d_dBx_2 = jnp.einsum("bijhd,bihd->bjhd", weight, dy_diag, precision=_HIGHEST)
    dL = dweight * BC_inner[:, :, :, None, None]
    dBC_inner = jnp.sum(dweight * L, axis=(-2, -1))
    dC_c_2 = jnp.einsum("bij,bjs->bis", dBC_inner, B_c, precision=_HIGHEST)
    dB_c_2 = jnp.einsum("bij,bis->bjs", dBC_inner, C_c, precision=_HIGHEST)

    d_decay_diff = dL * L * _clip_mask(decay_diff_raw, -20.0, 20.0)
    d_cumdecay_i = jnp.sum(d_decay_diff, axis=2)
    d_cumdecay_j = -jnp.sum(d_decay_diff, axis=1)

    d_dBx_total = _sanitize(d_dBx_1 + d_dBx_2)
    ddt_c_1 = d_dBx_total * x_c   # <-- TEST-ONLY EXTRA RETURN #1
    dx_c = d_dBx_total * dt_c

    d_cumdecay_total = _sanitize(
        d_cumdecay_i + d_cumdecay_j + d_cumdecay_from_yoff + d_cumdecay_from_decay_to_end
    )
    row_mask = (idx == (Cs - 1)).astype(jnp.float32)[None, :, None, None]
    d_cumdecay_total = d_cumdecay_total + row_mask * (d_cumdecay_last_a + d_cumdecay_last_b)[:, None, :, :]
    # <-- TEST-ONLY EXTRA RETURN #2 (d_cumdecay_total, pre-clip-mask/pre-reverse-cumsum)

    dB_c = _sanitize(dB_c_1 + dB_c_2)
    dC_c = _sanitize(dC_c_1 + dC_c_2)

    return ddt_c_1, dB_c, dC_c, dx_c, d_cumdecay_total, dstate_prev


def _reference_partial_ddt_and_dcumdecay(dt, A, B, C, x, chunk_size, do, dstate_final, state0):
    """Assembles the full-sequence ddt_c_1 / d_cumdecay_total reference by
    running the same reverse scan structure MB0 uses, but with
    _chunk_ssd_bwd_with_partials instead of the public _chunk_ssd_bwd."""
    b, l, h, d = dt.shape
    n_chunks = l // chunk_size

    def to_chunks(t):
        shp = t.shape
        t = t.reshape(b, n_chunks, chunk_size, *shp[2:])
        return jnp.moveaxis(t, 1, 0)

    dt_ch, B_ch, C_ch, x_ch, do_ch = map(to_chunks, (dt, B, C, x, do))

    def fwd_step(state_prev, inputs):
        dt_c, B_c, C_c, x_c = inputs
        _, state_new = _chunk_ssd(dt_c, A, B_c, C_c, x_c, state_prev)
        return state_new, state_prev

    _, state_prev_all = jax.lax.scan(fwd_step, state0, (dt_ch, B_ch, C_ch, x_ch))

    def bwd_step(dstate_carry, inputs):
        dt_c, B_c, C_c, x_c, do_c, state_prev_c = inputs
        ddt_c1, dB_c, dC_c, dx_c, dcum_c, dstate_prev_c = _chunk_ssd_bwd_with_partials(
            dt_c, A, B_c, C_c, x_c, state_prev_c, do_c, dstate_carry
        )
        return dstate_prev_c, (ddt_c1, dcum_c)

    _, (ddt1_rev, dcum_rev) = jax.lax.scan(
        bwd_step, dstate_final, (dt_ch, B_ch, C_ch, x_ch, do_ch, state_prev_all), reverse=True
    )

    def from_chunks(t):
        t = jnp.moveaxis(t, 0, 1)
        return t.reshape(b, l, *t.shape[3:])

    ddt1_full = from_chunks(ddt1_rev)          # (b, l, h, d)
    # dcum_rev: (n_chunks, b, C, h, d) -- move to Pallas layout (b,h,n_chunks,C,d)
    # for direct comparison with the orchestrator's own dcumdecay output.
    dcum_pallas_layout = jnp.moveaxis(dcum_rev, (0, 3), (2, 1))
    return ddt1_full, dcum_pallas_layout


def _make_problem(seed=0, b=2, l=128, h=4, d=32, s=16, chunk_size=64):
    keys = jax.random.split(jax.random.PRNGKey(seed), 8)
    dt = jax.nn.softplus(jax.random.normal(keys[0], (b, l, h, d))) * 0.1 + 1e-2
    A = -jnp.exp(jax.random.uniform(keys[1], (h,), minval=-1.0, maxval=1.0))
    B = jax.random.normal(keys[2], (b, l, s)) * 0.1
    C = jax.random.normal(keys[3], (b, l, s)) * 0.1
    x = jax.random.normal(keys[4], (b, l, h, d)) * 0.1
    do = jax.random.normal(keys[5], (b, l, h, d)) * 0.1
    dstate_final = jax.random.normal(keys[6], (b, h, d, s)) * 0.1
    state0 = jax.random.normal(keys[7], (b, h, d, s)) * 0.1
    return dt, A, B, C, x, chunk_size, do, dstate_final, state0


def test_orchestrator_matches_mb0_full_outputs():
    dt, A, B, C, x, chunk_size, do, dstate_final, state0 = _make_problem()

    ddt_ref, dB_ref, dC_ref, dx_ref, dA_ref, dstate0_ref = chunk_ssd_bwd_scan(
        dt, A, B, C, x, chunk_size, do, dstate_final, state0=state0
    )

    ddt_orch, dB_orch, dC_orch, dx_orch, dstate0_orch, dcum_orch = mamba2_bwd_scan(
        dt, A, B, C, x, chunk_size, do, dstate_final, state0=state0
    )

    # these four are expected to match MB0 EXACTLY -- see module docstring.
    for name, orch, ref in [
        ("dx", dx_orch, dx_ref),
        ("dB", dB_orch, dB_ref),
        ("dC", dC_orch, dC_ref),
        ("dstate0", dstate0_orch, dstate0_ref),
    ]:
        err = _rel_err(orch, ref)
        print(f"[MB1+MB3][orch vs MB0-full] {name:10s} rel_err={err:.3e}  {'OK' if err < 1e-4 else 'FAIL'}")
        assert err < 1e-4 and bool(jnp.all(jnp.isfinite(orch)))

    # ddt is only PARTIAL here (no MB4 yet) -- cross-check against the
    # test-only reference's ddt_c_1, not MB0's full ddt.
    ddt1_ref, dcum_ref = _reference_partial_ddt_and_dcumdecay(
        dt, A, B, C, x, chunk_size, do, dstate_final, state0
    )
    for name, orch, ref in [
        ("ddt_partial", ddt_orch, ddt1_ref),
        ("dcumdecay", dcum_orch, dcum_ref),
    ]:
        err = _rel_err(orch, ref)
        print(f"[MB1+MB3][orch vs MB0-internals] {name:12s} rel_err={err:.3e}  {'OK' if err < 1e-4 else 'FAIL'}")
        assert err < 1e-4 and bool(jnp.all(jnp.isfinite(orch)))


def test_multi_chunk_and_batch_shapes():
    """Same check on a different (b, l, h, d, s, chunk_size) combination --
    guards against shape/layout bugs that a single fixed size could hide
    (e.g. an accidental axis swap that happens to be a no-op when two
    dimensions are equal)."""
    dt, A, B, C, x, chunk_size, do, dstate_final, state0 = _make_problem(
        seed=7, b=3, l=192, h=5, d=24, s=12, chunk_size=64
    )
    ddt_ref, dB_ref, dC_ref, dx_ref, dA_ref, dstate0_ref = chunk_ssd_bwd_scan(
        dt, A, B, C, x, chunk_size, do, dstate_final, state0=state0
    )
    ddt_orch, dB_orch, dC_orch, dx_orch, dstate0_orch, dcum_orch = mamba2_bwd_scan(
        dt, A, B, C, x, chunk_size, do, dstate_final, state0=state0
    )
    for name, orch, ref in [
        ("dx", dx_orch, dx_ref),
        ("dB", dB_orch, dB_ref),
        ("dC", dC_orch, dC_ref),
        ("dstate0", dstate0_orch, dstate0_ref),
    ]:
        err = _rel_err(orch, ref)
        print(f"[MB1+MB3][multi-shape] {name:10s} rel_err={err:.3e}  {'OK' if err < 1e-4 else 'FAIL'}")
        assert err < 1e-4 and bool(jnp.all(jnp.isfinite(orch)))


if __name__ == "__main__":
    print(f"[MB1+MB3] backend: {jax.default_backend()}")
    test_orchestrator_matches_mb0_full_outputs()
    test_multi_chunk_and_batch_shapes()
    print("[MB1+MB3] ✅ passed (interpret=True)")
