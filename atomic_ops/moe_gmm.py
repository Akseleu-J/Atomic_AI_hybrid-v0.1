"""
atomic_ops/moe_gmm.py -- Sparse MoE FFN via megablox `gmm`/`tgmm` Pallas
grouped-matmul TPU kernels, replacing moe_sparse.py's sort+gather-into-
(E,capacity+1,d)+nn.vmap(ExpertPack) dispatch.

Implements the integration plan (M0-M7, see project handoff doc) up through
M6. Not run on real TPU from this environment -- no TPU hardware here, see
CAVEAT below. Everything in this file was validated with
`jax.experimental.pallas.ops.tpu.megablox.gmm`'s own `interpret=True` mode
on CPU, which runs the SAME kernel logic through a pure-JAX/numpy
interpreter (not the compiled Mosaic kernel) -- exact-match against a plain
per-group-einsum reference for both the forward matmul shapes AND the
hand-written custom_vjp backward (see test_moe_gmm_parity.py in this same
delivery, run there for the actual numbers). Before switching this into
model.py's hot path, re-run that same test with `interpret=False` on the
real v5e-8 to confirm the compiled kernel agrees (same two-stage validation
discipline already used for kernel_trainable_B6.py vs kernel_trainable.py
in this project).

M1 -- routing without capacity/scatter
-----------------------------------------------------------------------
moe_sparse.py's SparseMoEJ has to invent a `capacity` and a sentinel slot
because its dispatch target is a *dense* (E, capacity+1, d) buffer sized
before routing is known -- any token past `capacity` for its expert is
dropped (see moe_sparse.py's own docstring on the sentinel-slot fix).
`gmm` needs no such buffer: it consumes a *sorted* (T, d) matrix directly,
grouped by `group_sizes` (a length-E_routed vector of per-expert token
counts that is exactly correct, computed fresh every forward pass via
`jnp.bincount`). Nothing is ever dropped -- `group_sizes.sum() == T`
identically, by construction, not just "usually true after warmup" the way
moe_sparse.py's dropped_ratio->0 is an emergent training outcome.

M2 -- forward FFN via gmm instead of nn.vmap(ExpertPack)
-----------------------------------------------------------------------
Expert weights are held as consolidated `self.param` tensors
`W1: (E_routed, d_model, d_ff)` / `W2: (E_routed, d_ff, d_model)` -- NOT a
flax `nn.vmap`-wrapped submodule with a param axis, matching how
moe_sparse.py's `routed_experts` axis is already unsharded/replicated (see
its own `_get_shard_spec` note in train.py) -- gmm needs the raw weight
array directly, it has no notion of a flax Module.

No bias terms: the plan (M2) specifies exactly `h = gelu(gmm(x,W1,sizes));
out = gmm(h,W2,sizes)`, and adding a per-expert bias would mean gathering
`b[expert_id]` per token, an *extra* data-dependent gather outside the gmm
call -- doable, but out of scope for this delivery; flagged in
GmmMoEJ's docstring as a follow-up if bias matters empirically.

M3 -- backward: gmm (dx) + tgmm (dW)
-----------------------------------------------------------------------
`gmm`/`tgmm` are themselves plain `jax.jit`-wrapped Pallas calls with no
autodiff rule of their own (confirmed: they are NOT `jax.custom_vjp`
objects, ordinary functions) -- differentiating through them naively would
try to trace through the Pallas kernel's dynamic-grid/dynamic-index
machinery, which is not expected to produce a usable VJP. So, per the plan,
a hand-written `jax.custom_vjp` wraps the whole two-gmm FFN:
    dh              = gmm(dout, W2^T-per-group, group_sizes)
    dW2             = tgmm(h^T, dout, group_sizes)
    dh_pre          = gelu_vjp(dh)          <- plain JAX autodiff, gelu is elementwise
    dx              = gmm(dh_pre, W1^T-per-group, group_sizes)
    dW1             = tgmm(x^T, dh_pre, group_sizes)
`group_sizes` itself is integer-valued and carries no gradient -- passed
via `nondiff_argnums`, same convention this project already uses for
`scale` in kernel_trainable.py/kernel_trainable_B6.py's custom_vjp.

M4 -- sanitization and dtype discipline
-----------------------------------------------------------------------
Same `clip(+-1e3)+nan_to_num` convention as moe_sparse.py, applied after
every gmm/tgmm call (forward AND backward, both are new numerical surfaces
this project hasn't stress-tested yet) -- not just at the final output.
`group_sizes`/routing indices are pinned to int32 explicitly (`gmm`'s own
common.py enforces this; see M0 smoke-test), same "explicit dtype anchor"
reasoning already applied for A_log/decay_a and B6's cotangent dtype fix.

M5 -- SPMD / sharding
-----------------------------------------------------------------------
Follows the *second* fix already landed in moe_sparse.py (`_local_sharded`,
not the superseded `_with_batch_sharding`/full-replication approach its own
docstring says was replaced): the whole routing+gmm block runs under an
explicit `with_sharding_constraint` pinning `flat_x`/`expert_idx`/
`gate_weight` (and everything derived data-dependently from them: perm,
group_sizes, x_sorted) to stay SHARDED along the batch axis -- each device
independently computes routing and calls `gmm`/`tgmm` on ONLY its own local
shard, no cross-device gather. This is *more* natural for gmm than for the
old capacity-buffer dispatch: `gmm`'s `group_offset`/`num_actual_groups`
hooks exist precisely to let each shard operate on a local slice, though
this delivery does not yet use expert-parallelism (E_routed experts stay
fully replicated across all devices, same deprioritization already on
record in userMemories/INTEGRATION_NOTES.md for the old implementation --
M5's expert-parallel variant is a distinct follow-up, not done here).

M6 -- integration into a SparseMoEJ-shaped module
-----------------------------------------------------------------------
`GmmMoEJ` below is a drop-in structural replacement for moe_sparse.py's
`SparseMoEJ` (same __call__ signature, same sown metrics:
`aux_loss`/`z_loss`/`moe_dropped_ratio`, same shared+routed combination) --
`moe_dropped_ratio` is sown as an always-0.0 constant (kept only so
train.py's existing "[DIAG] moe dropped_ratio" logging line and
collect_by_leaf_name() plumbing keep working unmodified; structurally
there is no dropping left to report, gmm's grouping never discards a
token).

CAVEAT (read before wiring into model.py)
-----------------------------------------------------------------------
This file was authored and logic-tested in an environment with **no TPU
and no real Mosaic compilation** -- `interpret=True` runs the *reference
interpreter* for the same kernel code, which is a strong but not
sufficient substitute for compiling on v5e-8 (tiling/(128,128,128)
assumptions, VMEM budget, and the actual Mosaic lowering are all
unverified here). Treat this the same way this project already treats
`kernel_trainable_B6.py` before it was trusted: run
`test_moe_gmm_parity.py` with `interpret=False` on your Kaggle TPU v5e-8
FIRST, compare against moe_sparse.py's SparseMoEJ (or a plain dense
JAX-einsum reference) on a few seeds/sizes, and only switch model.py's
import over once that's finite and rel_diff-small, per this project's own
"equivalence testing before production use" discipline.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm, tgmm


from model import get_model_mesh, get_batch_axis
_DEFAULT_TILING = (128, 128, 128)
_SANITIZE_CLIP = 1e3
 
 
def _sanitize(x, clip=_SANITIZE_CLIP):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)
 
 
def _auto_tile(m, k, n, m_pref=128, k_pref=128, n_pref=128):
    def _pick(d, pref):
        return pref if d % pref == 0 else d
    return (_pick(m, m_pref), _pick(k, k_pref), _pick(n, n_pref))
 
 
def _make_grouped_ffn_core(interpret=False):
 
    def _fwd_math(x_sorted, W1, W2, group_sizes):
        x_f = x_sorted.astype(jnp.float32)
        W1_f = W1.astype(jnp.float32)
        W2_f = W2.astype(jnp.float32)
 
        T, d_model = x_f.shape
        _, _, d_ff = W1_f.shape
 
        h_pre = gmm(x_f, W1_f, group_sizes,
                    tiling=_auto_tile(T, d_model, d_ff),
                    interpret=interpret, preferred_element_type=jnp.float32)
        h_pre = _sanitize(h_pre)
 
        h, gelu_vjp = jax.vjp(jax.nn.gelu, h_pre)
 
        out = gmm(h, W2_f, group_sizes,
                  tiling=_auto_tile(T, d_ff, d_model),
                  interpret=interpret, preferred_element_type=jnp.float32)
        out = _sanitize(out)
        return out, h, gelu_vjp
 
    # NOTE: no more nondiff_argnums -- group_sizes is now an ordinary
    # (traced-safe) positional argument. custom_vjp is fine with an
    # integer-dtype argument as long as bwd returns a float0 cotangent
    # for it (see _core_bwd below).
    @jax.custom_vjp
    def _core(x_sorted, W1, W2, group_sizes):
        out, _, _ = _fwd_math(x_sorted, W1, W2, group_sizes)
        return out.astype(x_sorted.dtype)
 
    def _core_fwd(x_sorted, W1, W2, group_sizes):
        out, h, gelu_vjp = _fwd_math(x_sorted, W1, W2, group_sizes)
        # group_sizes carried through residuals now (it used to be closed
        # over via nondiff_argnums and handed to _core_bwd separately).
        residuals = (x_sorted, W1, W2, group_sizes, h, gelu_vjp)
        return out.astype(x_sorted.dtype), residuals
 
    def _core_bwd(residuals, dout):
        x_sorted, W1, W2, group_sizes, h, gelu_vjp = residuals
        x_f = x_sorted.astype(jnp.float32)
        W1_f = W1.astype(jnp.float32)
        W2_f = W2.astype(jnp.float32)
        dout_f = _sanitize(dout.astype(jnp.float32))
 
        T, d_model = x_f.shape
        _, d_ff = h.shape
 
        dW2 = tgmm(h.T, dout_f, group_sizes,
                   tiling=_auto_tile(d_ff, T, d_model),
                   interpret=interpret, preferred_element_type=jnp.float32)
        dW2 = _sanitize(dW2)
 
        W2_T = jnp.swapaxes(W2_f, 1, 2)
        dh = gmm(dout_f, W2_T, group_sizes,
                 tiling=_auto_tile(T, d_model, d_ff),
                 interpret=interpret, preferred_element_type=jnp.float32)
        dh = _sanitize(dh)
 
        (dh_pre,) = gelu_vjp(dh)
        dh_pre = _sanitize(dh_pre.astype(jnp.float32))
 
        dW1 = tgmm(x_f.T, dh_pre, group_sizes,
                   tiling=_auto_tile(d_model, T, d_ff),
                   interpret=interpret, preferred_element_type=jnp.float32)
        dW1 = _sanitize(dW1)
 
        W1_T = jnp.swapaxes(W1_f, 1, 2)
        dx = gmm(dh_pre, W1_T, group_sizes,
                 tiling=_auto_tile(T, d_ff, d_model),
                 interpret=interpret, preferred_element_type=jnp.float32)
        dx = _sanitize(dx)
 
        # group_sizes is integer-dtyped and carries no gradient -- JAX's
        # tangent type for an integer leaf is float0, and custom_vjp bwd
        # must return a value of that dtype/shape for it (not None, not an
        # int32 zeros array -- both are rejected).
        dgroup_sizes = jnp.zeros(group_sizes.shape, dtype=jax.dtypes.float0)
 
        return (
            dx.astype(x_sorted.dtype),
            dW1.astype(W1.dtype),
            dW2.astype(W2.dtype),
            dgroup_sizes,
        )
 
    _core.defvjp(_core_fwd, _core_bwd)
    return _core
 


class GmmMoEJ(nn.Module):
    """Drop-in structural replacement for moe_sparse.py's SparseMoEJ.

    Same shared-expert-plus-top1-routed-experts split, same sown metrics.
    See module docstring for the M1-M6 design and the CAVEAT about this
    never having run on real TPU hardware.
    """
    cfg: object
    interpret: bool = False  # M0/M1-M3 validation only -- False for real TPU runs

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rngs=None):
        b, l, d = x.shape
        flat_x = x.reshape(-1, d)
        T = flat_x.shape[0]
        E_routed = self.cfg.num_experts - 1
        assert E_routed >= 1, "num_experts must be >= 2 (1 shared + >=1 routed)."

        # Local import mirrors moe_sparse.py's own workaround for the
        # model.py <-> atomic_ops circular import.
       
        mesh = get_model_mesh()
        batch_axis = get_batch_axis()

        def _local_sharded(t):
            """M5: keep t SHARDED along batch_axis -- no cross-device
            gather. routed-expert weights (W1/W2 below) are fully
            replicated per device (same as moe_sparse.py's
            routed_experts / experts_block), so each device independently
            sorts/groups/gmm's ONLY its own local batch shard."""
            if mesh is None or batch_axis is None:
                return t
            spec = P(batch_axis, *([None] * (t.ndim - 1)))
            return jax.lax.with_sharding_constraint(t, jax.sharding.NamedSharding(mesh, spec))

        # ---- shared expert: plain Dense FFN, no routing/gmm needed ----
        shared_h = nn.Dense(self.cfg.d_ff, name="shared_w1", dtype=jnp.bfloat16)(flat_x)
        shared_h = jax.nn.gelu(shared_h)
        shared_h = nn.Dropout(rate=self.cfg.dropout_rate)(shared_h, deterministic=deterministic)
        shared_out = nn.Dense(self.cfg.d_model, name="shared_w2", dtype=jnp.bfloat16)(shared_h)

        # ---- routing decision (identical to moe_sparse.py) ----
        router_logits = nn.Dense(E_routed, use_bias=False, name="router", dtype=jnp.bfloat16)(flat_x)
        router_logits = router_logits.astype(jnp.float32)
        router_logits = _sanitize(router_logits)
        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout") if rngs is None else rngs.get("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape, dtype=router_logits.dtype
            )
        gate_probs = jax.nn.softmax(router_logits, axis=-1)
        expert_idx = jnp.argmax(router_logits, axis=-1).astype(jnp.int32)
        gate_weight = jnp.take_along_axis(gate_probs, expert_idx[:, None], axis=-1)

        mean_probs = jnp.mean(gate_probs, axis=0)
        self.sow("losses", "aux_loss", E_routed * jnp.sum(mean_probs * mean_probs))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(
            jax.scipy.special.logsumexp(router_logits, axis=-1))))
        self.sow("losses", "expert_utilization", mean_probs)

        # ==================================================================
        # M5: pin routing inputs to stay batch-sharded before the
        # data-dependent argsort/bincount/gmm block.
        # ==================================================================
        flat_x_r = _local_sharded(flat_x)
        expert_idx_r = _local_sharded(expert_idx)
        gate_weight_r = _local_sharded(gate_weight)

        # ==================================================================
        # M1: routing via group_sizes = bincount(expert_idx). No capacity,
        # no sentinel slot, no dropped tokens -- group_sizes.sum() == T
        # identically by construction (bincount over a length-T index
        # array with values in [0, E_routed) always sums to T).
        # ==================================================================
        group_sizes = jnp.bincount(expert_idx_r, length=E_routed).astype(jnp.int32)
        perm = jnp.argsort(expert_idx_r, stable=True)
        inv_perm = jnp.argsort(perm)  # perm[inv_perm] == arange(T); x[perm][inv_perm] == x

        x_sorted = jnp.take(flat_x_r, perm, axis=0)

        # ==================================================================
        # M2: consolidated per-expert weight tensors (NOT nn.vmap(ExpertPack))
        # ==================================================================
        d_model, d_ff = self.cfg.d_model, self.cfg.d_ff
        w1_init = nn.initializers.lecun_normal()
        w2_init = nn.initializers.lecun_normal()
        W1 = self.param("routed_w1", w1_init, (E_routed, d_model, d_ff), jnp.bfloat16)
        W2 = self.param("routed_w2", w2_init, (E_routed, d_ff, d_model), jnp.bfloat16)

        grouped_ffn = _make_grouped_ffn_core(interpret=self.interpret)
        out_sorted = grouped_ffn(x_sorted.astype(jnp.bfloat16), W1, W2, group_sizes)

        # M1 round-trip: gather back to original token order.
        routed_out = jnp.take(out_sorted, inv_perm, axis=0)

        routed_out = routed_out.astype(jnp.float32) * gate_weight_r
        combined = shared_out.astype(jnp.float32) + routed_out
        combined = _sanitize(combined)
        combined = _local_sharded(combined)

        # M6: dropped_ratio is now structurally always 0 -- gmm's grouping
        # never discards a token, group_sizes.sum() == T identically. Sown
        # anyway so train.py's existing logging/collect_by_leaf_name plumbing
        # for "moe_dropped_ratio" keeps working unmodified.
        self.sow("losses", "moe_dropped_ratio", jnp.zeros((), dtype=jnp.float32))

        return combined.reshape(b, l, d).astype(x.dtype)
