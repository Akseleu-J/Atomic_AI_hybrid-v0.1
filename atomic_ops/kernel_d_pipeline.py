"""
Milestone 3 -- "Kernel D" (deliberately plain JAX, not Pallas -- see chat for
rationale) + full pipeline glue chaining Kernels A -> B -> C -> D.

Math (chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_kda.py L977-1263,
plus the derived output-combine formula from Milestone 1, both already
validated against the token-serial ground truth):
    v_new   = u - w_pseudo @ h_pre
    o       = scale * (qg @ h_pre) + Aqk @ v_new
    h_new   = h_pre * exp(gc_last) + kg^T @ v_new
sequential over chunks, h_pre carried via jax.lax.scan.

NOT YET TESTED ON REAL TPU (though this part reuses ordinary XLA ops with no
Mosaic/Pallas involvement, so it should be much lower-risk than Kernels A-C).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from kernel_a_scores import build_chunk_scores_pallas, BT
from kernel_b_solve import wy_solve_pallas
from kernel_c_recompute import recompute_wy_pallas

_HIGHEST = jax.lax.Precision.HIGHEST


def gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None):
    """Aqk: (B,H,n_chunks,BT,BT). w_pseudo,u,kg,qg: (B,H,n_chunks,BT,D).
    gc_last: (B,H,n_chunks,D). Returns o: (B,H,n_chunks,BT,D), h_final: (B,H,D,D).
    """
    bsz, H, n_chunks, _BT, D = w_pseudo.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, w_pseudo, u, kg, qg, gc_last))

    def step(h_pre, inputs):
        Aqk_c, w_pseudo_c, u_c, kg_c, qg_c, gclast_c = inputs
        wh = jnp.einsum("bhid,bhdv->bhiv", w_pseudo_c, h_pre, precision=_HIGHEST)
        v_new = u_c - wh                                                              # (B,H,BT,D)
        qh = jnp.einsum("bhid,bhdv->bhiv", qg_c, h_pre, precision=_HIGHEST)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_c, v_new, precision=_HIGHEST)
        o_c = scale * qh + intra                                                      # (B,H,BT,D)

        decay_h = jnp.exp(gclast_c)[..., None]                                          # (B,H,D,1)
        write = jnp.einsum("bhid,bhiv->bhdv", kg_c, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write
        # Defensive clip/sanitize, matching the safeguards already present in
        # model.py's associative_scan gdn2_recurrence_safe (Frobenius-norm
        # clip in _combine + nan_to_num on global_S) -- this codebase has a
        # documented history of GDN-2 state blowing up to non-finite over
        # long training runs (see train.py's NONFINITE_* auto-stop machinery
        # and the "train_loss -> ln(vocab_size)" incident noted there). The
        # WY-chunked state update has no equivalent per-step contractive
        # mechanism, so this last-line-of-defense is added explicitly.
        h_new = jnp.nan_to_num(jnp.clip(h_new, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
        o_c = jnp.nan_to_num(o_c, nan=0.0, posinf=1e4, neginf=-1e4)
        return h_new, o_c

    h_final, o_scanned = jax.lax.scan(step, h0, to_scan)
    o = jnp.moveaxis(o_scanned, 0, 2)  # (B,H,n_chunks,BT,D)
    return o, h_final


def gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=None):
    """Full staged pipeline: Kernel A (Pallas) -> B (Pallas) -> C (Pallas)
    -> D (plain JAX scan). q,k,v,w,b,g: (B,L,H,D). Returns o: (B,L,H,D),
    h_final: (B,H,D,D).

    ФИКС (shard_map): pallas_call внутри Kernels A/B/C cannot be
    automatically partitioned by GSPMD ("Mosaic kernels cannot be
    automatically partitioned. Please wrap the call in a shard_map.", same
    root cause and same fix as MLAJ's pallas_flash_attention in model.py).
    Deferred import of get_model_mesh/get_batch_axis from model.py avoids a
    circular import (model.py imports this module) -- it only resolves at
    call time, by which point model.py is fully loaded. Kernel D (plain JAX
    scan) doesn't strictly need shard_map -- ordinary JAX ops partition fine
    under GSPMD -- but wrapping the whole thing together is simpler and
    matches MLAJ's granularity (one shard_map per sub-layer).
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    def _impl(q, k, v, w, b, g, h0):
        Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
        A = wy_solve_pallas(Akk)
        w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
        o_chunks, h_final = gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0)
        n_chunks = L // BT
        o = jnp.moveaxis(o_chunks, 1, 3)          # (B,n_chunks,BT,H,D)
        o = o.reshape(bsz, n_chunks * BT, H, D)     # (B,L,H,D)
        return o, h_final

    try:
        from model import get_model_mesh, get_batch_axis
        mesh = get_model_mesh()
        batch_axis = get_batch_axis()
    except ImportError:
        mesh, batch_axis = None, None

    if mesh is not None:
        spec = P(batch_axis, None, None, None)
        sharded_impl = jax.shard_map(
            _impl, mesh=mesh,
            in_specs=(spec, spec, spec, spec, spec, spec, spec),
            out_specs=(spec, spec),
            check_vma=False,
        )
        return sharded_impl(q, k, v, w, b, g, h0)
    else:
        return _impl(q, k, v, w, b, g, h0)
