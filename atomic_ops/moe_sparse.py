"""
Sparse MoE -- 1 shared (always-on) expert + top-1 routing among E_routed
experts, dispatch via sort+gather (NOT one-hot-matmul).

See original docstring (unchanged rationale) for why sort+gather instead of
one-hot-matmul dispatch, and why this needs no Pallas kernel (plain
JAX/XLA: argsort, take, searchsorted, .at[].set() on STATIC shapes).

FIX (this pass, PRODUCTION correctness -- found while reviewing for
production-readiness, NOT caught by moe_quality_check_tpu.py's toy task):
capacity-overflow tokens used to be routed via
`slot_sorted = jnp.clip(pos_in_expert_sorted, 0, capacity - 1)` -- this
CLIPS every overflowing position down into the last valid slot
(capacity - 1), colliding with the actually-valid token that legitimately
occupies that slot. `.at[].set()` with duplicate indices does not guarantee
"last real value wins" semantics in XLA -- the legitimate token's data can
be silently overwritten by a zeroed-out dropped-token write landing on the
same (expert, slot) index. This is NOT hypothetical: the project's own
quality-check run hit dropped_ratio=0.32 at step 0 (router not yet
balanced), which is exactly the regime where this collision fires --
and it fires hardest exactly when it's hardest to notice (early steps,
loss dominated by other noise, no per-token diagnostic).

Fix: give dropped tokens a dedicated SENTINEL slot (index `capacity`,
buffer sized `capacity + 1`) instead of clipping them into the valid
range. Valid positions (0..capacity-1) are then guaranteed unique per
expert (pos_in_expert_sorted is strictly increasing within an expert's
sorted block, so no two valid tokens ever share a slot) -- only dropped
tokens share the sentinel slot with each other, and that slot's output is
never read into `combined` (overflow_sorted is False for all tokens that
land there, so the existing `jnp.where(overflow_sorted[:, None], ..., 0.0)`
in the combine step already zeroes them out correctly; the sentinel row's
compute is just discarded, not incorrect).

FIX #2 (input sanitization, matches project convention): router_logits was
the only major pre-activation in this project's forward path with no
clip/nan_to_num before it feeds a softmax/argmax decision. Every other
router-adjacent or decay-adjacent computation in the project (kernel_c
kg/qg, model.py's g/alpha, decay_a/a_log) is defended this way; a
large-but-finite router_logits value (e.g. from an under-trained or
drifted `router` Dense early in training) could otherwise make argmax
routing behave unpredictably before softmax saturates it, or make z_loss
(logsumexp of router_logits) blow up. Cheap clip added, same ±1e3
convention as model.py's other pre-activation sanitization.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class MoEDiagnostics(NamedTuple):
    dropped_ratio: jnp.ndarray          # scalar, fraction of routed tokens dropped by capacity
    expert_utilization: jnp.ndarray     # (E_routed,) fraction of routed tokens per expert


class ExpertPack(nn.Module):
    """Same shape/signature as model.py's ExpertPack -- kept identical so
    this drops in without touching the rest of the model."""
    d_model: int
    d_ff: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        h = nn.Dense(self.d_ff, name="w1", dtype=jnp.bfloat16)(x)
        h = jax.nn.gelu(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        return nn.Dense(self.d_model, name="w2", dtype=jnp.bfloat16)(h)


def _sanitize(x, clip=1e3):
    # Same convention as the rest of this project (model.py/kernel_*.py):
    # nan_to_num alone doesn't catch large-but-finite blowups, clip first.
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


class SparseMoEJ(nn.Module):
    """1 shared (always-on) expert + top-1 routing among `num_experts - 1`
    routed experts. Drop-in alternative to model.py's dense MoEJ.

    cfg fields used: d_model, d_ff, num_experts, dropout_rate,
    router_noise_std, moe_capacity_factor.
    `num_experts` includes the shared one, so routed count = num_experts - 1
    (e.g. num_experts=8 -> 1 shared + 7 routed, matching current config).
    """
    cfg: object  # ModelConfig -- typed loosely to avoid a hard model.py import here

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rngs=None):
        b, l, d = x.shape
        flat_x = x.reshape(-1, d)
        T = flat_x.shape[0]
        E_routed = self.cfg.num_experts - 1
        assert E_routed >= 1, "num_experts must be >= 2 (1 shared + >=1 routed)."

        # ---- shared expert: always on, no routing/dispatch needed ----
        shared_out = ExpertPack(
            d_model=self.cfg.d_model, d_ff=self.cfg.d_ff,
            dropout_rate=self.cfg.dropout_rate, name="shared_expert",
        )(flat_x, deterministic)

        # ---- routing decision (top-1 among E_routed) ----
        router_logits = nn.Dense(E_routed, use_bias=False, name="router", dtype=jnp.bfloat16)(flat_x)
        router_logits = router_logits.astype(jnp.float32)
        # FIX #2: sanitize before it drives argmax/softmax -- see module docstring.
        router_logits = _sanitize(router_logits)
        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout") if rngs is None else rngs.get("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape, dtype=router_logits.dtype
            )
        gate_probs = jax.nn.softmax(router_logits, axis=-1)
        expert_idx = jnp.argmax(router_logits, axis=-1)                      # (T,)
        gate_weight = jnp.take_along_axis(gate_probs, expert_idx[:, None], axis=-1)  # (T,1)

        # load-balancing / z-loss aux, same formulas as the dense MoEJ so
        # optimizer.py's collect_by_leaf_name("aux_loss"/"z_loss") keeps working
        mean_probs = jnp.mean(gate_probs, axis=0)
        self.sow("losses", "aux_loss", E_routed * jnp.sum(mean_probs * mean_probs))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(
            jax.scipy.special.logsumexp(router_logits, axis=-1))))
        self.sow("losses", "expert_utilization", mean_probs)

        # ---- dispatch: sort + gather (linear), NOT one-hot-matmul ----
        capacity = max(1, int(self.cfg.moe_capacity_factor * T / E_routed))

        perm = jnp.argsort(expert_idx, stable=True)               # (T,)
        sorted_idx = expert_idx[perm]
        boundaries = jnp.searchsorted(sorted_idx, jnp.arange(E_routed))
        pos_in_expert_sorted = jnp.arange(T) - boundaries[sorted_idx]
        overflow_sorted = pos_in_expert_sorted < capacity

        # FIX: dropped tokens get a dedicated SENTINEL slot (index=capacity)
        # instead of being clipped into the valid range -- see module
        # docstring. Buffer is sized capacity+1; the sentinel row's compute
        # is discarded, never read into `combined`.
        slot_sorted = jnp.where(overflow_sorted, pos_in_expert_sorted, capacity)

        x_padded = jnp.concatenate([flat_x, jnp.zeros((1, d), flat_x.dtype)], axis=0)
        gather_idx = jnp.where(overflow_sorted, perm, T)          # T -> zero-pad row
        gathered_x = jnp.take(x_padded, gather_idx, axis=0)       # (T, d), expert-grouped order

        # scatter into STATIC (E_routed, capacity + 1, d) blocks -- static
        # shapes throughout, `.at[].set()` on a statically-shaped output,
        # not the `.at[].add()` scatter pattern flagged as non-lowering in
        # this project's Pallas kernels (this runs under plain XLA, and is
        # a `set` not an `add`, so no accumulation-race concern; and with
        # the sentinel-slot fix above, no VALID token ever shares an index
        # with another token, so ordering of duplicate writes -- which XLA
        # does not otherwise guarantee -- can no longer corrupt real data).
        expert_in = jnp.zeros((E_routed, capacity + 1, d), flat_x.dtype)
        expert_in = expert_in.at[sorted_idx, slot_sorted].set(
            jnp.where(overflow_sorted[:, None], gathered_x, 0.0)
        )

        # ---- run each routed expert once over its (capacity+1, d) block
        # (the extra sentinel row is wasted compute -- 1/(capacity+1) of
        # this expert's work -- negligible at realistic capacity, and
        # cheaper than the alternative of a dynamic-shape slice) ----
        run_experts = nn.vmap(
            ExpertPack,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, None),
            out_axes=0,
            axis_size=E_routed,
        )(d_model=self.cfg.d_model, d_ff=self.cfg.d_ff,
          dropout_rate=self.cfg.dropout_rate, name="routed_experts")
        expert_out = run_experts(expert_in, deterministic)        # (E_routed, capacity+1, d)

        # ---- combine: gather each token's own expert's output back ----
        out_sorted = expert_out[sorted_idx, slot_sorted]           # (T, d), still in sorted order
        out_sorted = jnp.where(overflow_sorted[:, None], out_sorted, 0.0)

        routed_out = jnp.zeros((T, d), flat_x.dtype)
        routed_out = routed_out.at[perm].set(out_sorted)           # back to original token order

        routed_out = routed_out.astype(jnp.float32) * gate_weight
        combined = shared_out.astype(jnp.float32) + routed_out
        combined = _sanitize(combined)

        dropped_ratio = 1.0 - jnp.mean(overflow_sorted.astype(jnp.float32))
        self.sow("losses", "moe_dropped_ratio", dropped_ratio)

        return combined.reshape(b, l, d).astype(x.dtype)
