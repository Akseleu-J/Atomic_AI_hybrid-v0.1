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

FIX (this pass, SPMD dispatch correctness): argsort/searchsorted/.at[].set()
dispatch (see above) is a data-dependent-index operation -- a class XLA's
GSPMD auto-partitioner handles far less reliably than a plain batched einsum
(the old dense MoEJ's only op, which trivially shards along the batch axis
with zero collectives). Without an explicit sharding annotation here, the
partitioner may infer an incorrect layout around argsort/scatter -- e.g.
computing `capacity`/`boundaries` against a PARTIAL per-device view while
the code's own Python-level `capacity = int(...T...)` assumes T is the
full logical (replicated) batch size seen during tracing. This mismatch
was traced (see project handoff) to non-finite gradients appearing on the
very FIRST training micro-step after switching from dense MoEJ to this
sparse implementation -- immediate, not the "accumulated instability after
N steps" pattern seen elsewhere in this project, which points to a
structural SPMD issue rather than a numerical one.

Fix: explicit jax.lax.with_sharding_constraint around the dispatch/combine
block -- forces flat_x (and the indices computed from it) to be REPLICATED
across the mesh's batch axis right before argsort/scatter (so every device
computes the SAME capacity/boundaries/slot assignment from the SAME full
view), then re-applies the batch-sharded constraint on the final `combined`
output before returning, so nothing downstream loses its expected FSDP
batch sharding. Same "make the SPMD boundary explicit" pattern already
used for GDN-2 (kernel_trainable_B6 wrapped in jax.shard_map) and MLA
(flash attention wrapped in jax.shard_map) elsewhere in this project --
applied here via with_sharding_constraint instead of shard_map because
this block calls flax submodules (nn.Dense, nn.vmap(ExpertPack)) with
their own parameter state, which jax.shard_map cannot wrap directly.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P


class MoEDiagnostics(NamedTuple):
    dropped_ratio: jnp.ndarray
    expert_utilization: jnp.ndarray


class ExpertPack(nn.Module):
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
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


class SparseMoEJ(nn.Module):
    cfg: object

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rngs=None):
        b, l, d = x.shape
        flat_x = x.reshape(-1, d)
        T = flat_x.shape[0]
        E_routed = self.cfg.num_experts - 1
        assert E_routed >= 1, "num_experts must be >= 2 (1 shared + >=1 routed)."

        # ФИКС: локальный import, чтобы не создавать циклический импорт
        # model.py <-> atomic_ops.moe_sparse (model.py импортирует
        # SparseMoEJ из этого файла на верхнем уровне; здесь -- обратный
        # import, только внутри функции, тем же способом, что utils.py
        # локально импортирует jax).
        from model import get_model_mesh, get_batch_axis

        mesh = get_model_mesh()
        batch_axis = get_batch_axis()

        def _with_batch_sharding(t, extra_dims=0):
            """t: (T, ...) -- применяет constraint с шардированием по
            batch_axis на первую ось, None на все остальные. No-op вне
            mesh-контекста (например, при model.init() без установленного
            mesh)."""
            if mesh is None or batch_axis is None:
                return t
            spec = P(batch_axis, *([None] * (t.ndim - 1)))
            return jax.lax.with_sharding_constraint(t, jax.sharding.NamedSharding(mesh, spec))

        def _local_sharded(t):
        """Explicit constraint: t stays SHARDED along batch_axis, never
        gathered. Correct semantics for this dispatch: routed_experts params
        are already fully replicated per device (see train.py's
        _get_shard_spec), so NO cross-device communication is needed for
        top-1 routing -- each device independently sorts/scatters/gathers
        ONLY its own local batch shard. This is the fix for BOTH problems:
        (a) the crash without any constraint (GSPMD couldn't infer a valid
        partitioning for argsort/scatter on an implicitly-sharded input on
        its own), and (b) the previous _replicated() fix's hidden cost (an
        unnecessary full-batch all-gather onto every device before dispatch,
        duplicating compute 8x and defeating much of sparse MoE's point)."""
            if mesh is None or batch_axis is None:
                return t
            spec = P(batch_axis, *([None] * (t.ndim - 1)))
            return jax.lax.with_sharding_constraint(t, jax.sharding.NamedSharding(mesh, spec))
        # ---- shared expert: без диспатча, обычный einsum-путь, шардится
        # автоматически как раньше -- constraint здесь не нужен ----
        shared_out = ExpertPack(
            d_model=self.cfg.d_model, d_ff=self.cfg.d_ff,
            dropout_rate=self.cfg.dropout_rate, name="shared_expert",
        )(flat_x, deterministic)

        # ---- routing decision ----
        router_logits = nn.Dense(E_routed, use_bias=False, name="router", dtype=jnp.bfloat16)(flat_x)
        router_logits = router_logits.astype(jnp.float32)
        router_logits = _sanitize(router_logits)
        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout") if rngs is None else rngs.get("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape, dtype=router_logits.dtype
            )
        gate_probs = jax.nn.softmax(router_logits, axis=-1)
        expert_idx = jnp.argmax(router_logits, axis=-1)
        gate_weight = jnp.take_along_axis(gate_probs, expert_idx[:, None], axis=-1)

        mean_probs = jnp.mean(gate_probs, axis=0)
        self.sow("losses", "aux_loss", E_routed * jnp.sum(mean_probs * mean_probs))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(
            jax.scipy.special.logsumexp(router_logits, axis=-1))))
        self.sow("losses", "expert_utilization", mean_probs)

        # ==================================================================
        # ФИКС: явная репликация ПЕРЕД data-dependent диспатчем -- см.
        # докстринг модуля. flat_x/expert_idx/gate_weight ниже форсируются
        # в полностью реплицированный вид, чтобы argsort/searchsorted/
        # capacity считались одинаково на КАЖДОМ устройстве, независимо от
        # того, как auto-partitioner решил бы шардировать их по умолчанию.
        # ==================================================================
        flat_x_r = _local_sharded(flat_x)
        expert_idx_r = _local_sharded(expert_idx)
        gate_weight_r = _local_sharded(gate_weight)

        capacity = max(1, int(self.cfg.moe_capacity_factor * T / E_routed))

        perm = jnp.argsort(expert_idx_r, stable=True)
        sorted_idx = expert_idx_r[perm]
        boundaries = jnp.searchsorted(sorted_idx, jnp.arange(E_routed))
        pos_in_expert_sorted = jnp.arange(T) - boundaries[sorted_idx]
        overflow_sorted = pos_in_expert_sorted < capacity

        slot_sorted = jnp.where(overflow_sorted, pos_in_expert_sorted, capacity)

        x_padded = jnp.concatenate([flat_x_r, jnp.zeros((1, d), flat_x_r.dtype)], axis=0)
        gather_idx = jnp.where(overflow_sorted, perm, T)
        gathered_x = jnp.take(x_padded, gather_idx, axis=0)

        expert_in = jnp.zeros((E_routed, capacity + 1, d), flat_x_r.dtype)
        expert_in = expert_in.at[sorted_idx, slot_sorted].set(
            jnp.where(overflow_sorted[:, None], gathered_x, 0.0)
        )
        # expert_in реплицирован -- это ОК и ожидаемо: каждое устройство
        # прогоняет ОДИНАКОВЫЙ expert_in через СВОИ (реплицированные по
        # параметрам, variable_axes={"params":0} -- каждый девайс держит
        # ВСЕ параметры routed_experts) веса и получит одинаковый результат.
        # Дублирование вычислений -- цена корректности; не шардируем эту
        # ось намеренно (см. _get_shard_spec в train.py: "experts_block"/
        # "routed_experts" уже был не шардирован и до этого фикса).

        run_experts = nn.vmap(
            ExpertPack,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, None),
            out_axes=0,
            axis_size=E_routed,
        )(d_model=self.cfg.d_model, d_ff=self.cfg.d_ff,
          dropout_rate=self.cfg.dropout_rate, name="routed_experts")
        expert_out = run_experts(expert_in, deterministic)

        out_sorted = expert_out[sorted_idx, slot_sorted]
        out_sorted = jnp.where(overflow_sorted[:, None], out_sorted, 0.0)

        routed_out = jnp.zeros((T, d), flat_x_r.dtype)
        routed_out = routed_out.at[perm].set(out_sorted)

        routed_out = routed_out.astype(jnp.float32) * gate_weight_r
        combined = shared_out.astype(jnp.float32) + routed_out
        combined = _sanitize(combined)

        # ФИКС: возвращаем нормальный batch-шардинг на выходе -- ниже по
        # графу (residual stream в BlockDAR) продолжает ожидать FSDP-шардинг
        # по batch_axis, как было ДО этого MoE-блока.
        combined = _with_batch_sharding(combined)

        dropped_ratio = 1.0 - jnp.mean(overflow_sorted.astype(jnp.float32))
        self.sow("losses", "moe_dropped_ratio", dropped_ratio)

        return combined.reshape(b, l, d).astype(x.dtype)
