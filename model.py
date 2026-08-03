import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from typing import List
from jax.sharding import PartitionSpec as P

# Lazy/fault-tolerant: this import chain (jax.experimental.pallas.ops...) has been
# breaking on some Kaggle TPU environments with
#   AttributeError: module 'jax' has no attribute '_src'
# raised from INSIDE jax's own jax/experimental/pallas/ops/__init__.py -- a
# jax/jaxlib version-skew issue, not something in this file. pallas_flash_attention
# is only ever called behind `if self.cfg.use_flash_attention:` further down, so a
# broken/unavailable Pallas import should not prevent the whole module (and every
# other config with use_flash_attention=False) from loading. If use_flash_attention
# is turned on later and this import genuinely failed, MLAJ raises a clear error at
# that point instead of at module-import time.
try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as pallas_flash_attention,
        BlockSizes as FlashBlockSizes,
    )
    _PALLAS_FLASH_ATTENTION_IMPORT_ERROR = None
except Exception as _e:  # noqa: BLE001 -- deliberately broad: import-time failures
    pallas_flash_attention = None
    FlashBlockSizes = None
    _PALLAS_FLASH_ATTENTION_IMPORT_ERROR = _e

_model_mesh = None
_batch_axis = None  # None -> batch axis is fully replicated; "tpu_nodes" -> sharded across the mesh

def set_model_mesh(mesh, batch_axis=None):
    """Registers the mesh (and how the batch axis is actually sharded) so MLAJ's
    shard_map spec can never silently diverge from train.py's data_sharding again --
    that divergence is exactly what caused:
        ValueError: shard_map applied to the function ... axis sizes that are not
        evenly divisible by the corresponding mesh axis sizes ... float32[2,...]
    when data_sharding used P(None, None) (batch fully replicated, to allow
    batch_size < n_devices) while MLAJ's shard_map still hardcoded
    in_specs=P("tpu_nodes", ...) as if the batch axis were actually split 8 ways.
    Pass batch_axis=None when data_sharding replicates the batch (any batch_size);
    pass batch_axis="tpu_nodes" when data_sharding actually shards the batch axis on
    that mesh axis (requires batch_size % n_devices == 0)."""
    global _model_mesh, _batch_axis
    _model_mesh = mesh
    _batch_axis = batch_axis

def get_model_mesh():
    return _model_mesh

def get_batch_axis():
    return _batch_axis


@struct.dataclass
class ModelConfig:
    d_model: int = 384
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 6
    d_latent: int = 128
    d_ff: int = 1536
    num_experts: int = 4
    top_k: int = 2
    num_layers: int = 6
    vocab_size: int = 151936
    dropout_rate: float = 0.1
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.0001
    moe_capacity_factor: float = 1.25
    tie_embeddings: bool = True
    label_smoothing: float = 0.05
    router_noise_std: float = 0.3
    use_flash_attention: bool = False
    deltanet_chunk_size: int = 1024


# ==========================================
# RoPE
# ==========================================
class RoPEEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, seq_len):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2)[: (self.dim // 2)] / self.dim))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated_x = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated_x * sin


# ==========================================
# Multi-head Latent Attention
# ==========================================
class MLAJ(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, causal_mask, cos, sin, deterministic: bool = True, rngs=None):
        mesh = get_model_mesh()
        batch_axis = get_batch_axis()
        b, l, _ = x.shape
        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_model // n_heads

        Q = nn.Dense(self.cfg.d_model, use_bias=False, name="W_q", dtype=jnp.bfloat16)(x)
        Q = Q.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)
        Q_rope = apply_rope(Q, cos[None, None, :, :d_head], sin[None, None, :, :d_head])

        kv_latent = nn.Dense(self.cfg.d_latent, use_bias=False, name="W_kv_down", dtype=jnp.bfloat16)(x)
        K = nn.Dense(self.cfg.d_model, use_bias=False, name="W_k_up", dtype=jnp.bfloat16)(kv_latent)
        V = nn.Dense(self.cfg.d_model, use_bias=False, name="W_v_up", dtype=jnp.bfloat16)(kv_latent)

        K = K.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)
        K_rope = apply_rope(K, cos[None, None, :, :d_head], sin[None, None, :, :d_head])
        V = V.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)

        sm_scale = 1.0 / math.sqrt(d_head)

        if self.cfg.use_flash_attention:
            if pallas_flash_attention is None:
                raise ImportError(
                    "cfg.use_flash_attention=True but jax.experimental.pallas.ops.tpu."
                    "flash_attention failed to import (see model.py's module-level "
                    "try/except) -- original error: "
                    f"{_PALLAS_FLASH_ATTENTION_IMPORT_ERROR!r}. Either fix the jax/"
                    "jaxlib environment, or set use_flash_attention=False."
                )

            def _flash_call(q_local, k_local, v_local):
                local_b = q_local.shape[0]
                block_sizes = FlashBlockSizes.get_default(local_b, n_heads, l, l, d_head)
                return pallas_flash_attention(
                    q_local, k_local, v_local,
                    causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
                )

            if mesh is not None:
                # in_specs/out_specs use batch_axis from get_batch_axis(), NOT a
                # hardcoded "tpu_nodes" -- this MUST match how data_sharding actually
                # shards the batch axis in train.py's make_shard_and_compile, or you
                # get exactly the crash this comment used to describe:
                #   ValueError: ... axis sizes that are not evenly divisible by the
                #   corresponding mesh axis sizes ... float32[2,...]
                # (batch_size=2 replicated across an 8-device mesh, while this spec
                # claimed the batch axis was split 8 ways). set_model_mesh() is the
                # single source of truth for this now -- change it there, not here.
                spec = P(batch_axis, None, None, None)
                sharded_flash = jax.shard_map(
                    _flash_call,
                    mesh=mesh,
                    in_specs=spec,
                    out_specs=spec,
                    check_vma=False,
                )
                out = sharded_flash(
                    Q_rope.astype(jnp.bfloat16),
                    K_rope.astype(jnp.bfloat16),
                    V.astype(jnp.bfloat16)
                ).astype(x.dtype)
            else:
                # CPU / single-device debug path -- no mesh registered at all.
                out = _flash_call(
                    Q_rope.astype(jnp.bfloat16),
                    K_rope.astype(jnp.bfloat16),
                    V.astype(jnp.bfloat16)
                ).astype(x.dtype)
        else:
            # Naive fallback -- only for CPU debugging / small seq_len smoke tests.
            scores = jnp.einsum("bhqd,bhkd->bhqk", Q_rope, K_rope) * sm_scale
            scores = jnp.where(causal_mask == 0, -1e9, scores)
            attn = jax.nn.softmax(scores, axis=-1)
            if not deterministic:
                dropout_rng = rngs['dropout'] if rngs is not None and 'dropout' in rngs else self.make_rng('dropout')
                keep_prob = 1.0 - self.cfg.dropout_rate
                mask_drop = jax.random.bernoulli(dropout_rng, keep_prob, attn.shape)
                attn = attn * mask_drop / keep_prob
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, V)

        out = out.transpose(0, 2, 1, 3).reshape(b, l, self.cfg.d_model)
        return nn.Dense(self.cfg.d_model, use_bias=False, name="W_o", dtype=jnp.bfloat16)(out)


# ==========================================
# Mamba-2 (SSM)
# ==========================================
class Mamba2J(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        b, l, d = x.shape
        d_inner = d * self.cfg.expand
        d_state = self.cfg.d_state

        in_proj = nn.Dense(d_inner * 2, use_bias=False, name="in_proj", dtype=jnp.bfloat16)(x)
        x_bc, res = jnp.split(in_proj, 2, axis=-1)

        conv_w = self.param("conv_w", nn.initializers.normal(stddev=0.02), (d_inner, self.cfg.d_conv))
        conv_b = self.param("conv_b", nn.initializers.zeros, (d_inner,))

        rhs = conv_w.T[:, None, :]
        res_conv = jax.lax.conv_general_dilated(
            lhs=x_bc,
            rhs=rhs,
            window_strides=(1,),
            padding=[(self.cfg.d_conv - 1, 0)],
            feature_group_count=d_inner,
            dimension_numbers=('NHC', 'HIO', 'NHC')
        )
        x_conv = jax.nn.silu(res_conv + conv_b[None, None, :])

        A = -jnp.exp(self.param("A_log", nn.initializers.uniform(scale=1.0), (d_inner,)))
        B = nn.Dense(d_state, use_bias=False, name="B_proj", dtype=jnp.bfloat16)(x_bc)
        C = nn.Dense(d_state, use_bias=False, name="C_proj", dtype=jnp.bfloat16)(x_bc)
        dt = jax.nn.softplus(nn.Dense(d_inner, use_bias=True, name="dt_proj", dtype=jnp.bfloat16)(x_bc))

        dA = jnp.exp(jnp.einsum("bld,d->bld", dt, A))

        chunk_size = min(self.cfg.deltanet_chunk_size, l)
        if l % chunk_size != 0:
            raise ValueError(
                f"seq_len={l} must be divisible by deltanet_chunk_size={chunk_size} "
                "(chunked Mamba2 scan requires equal-sized chunks)."
            )
        num_chunks = l // chunk_size

        def _combine(state1, state2):
            da1, c1 = state1
            da2, c2 = state2
            da2e = da2[..., None]
            return da2 * da1, da2e * c1 + c2

        def _to_chunks(t):
            trailing = t.shape[2:]
            t = t.reshape(b, num_chunks, chunk_size, *trailing)
            return jnp.moveaxis(t, 1, 0)

        dA_ch = _to_chunks(dA)
        dt_ch = _to_chunks(dt)
        B_ch = _to_chunks(B)
        C_ch = _to_chunks(C)
        x_conv_ch = _to_chunks(x_conv)

        carry_da_init = jnp.ones((b, d_inner), dtype=x.dtype)
        carry_h_init = jnp.zeros((b, d_inner, d_state), dtype=x.dtype)

        def _chunk_step(carry, chunk_inputs):
            carry_da, carry_h = carry
            da_c, dt_c, B_c, C_c, xconv_c = chunk_inputs

            dB_c = jnp.einsum("bcd,bcs->bcds", dt_c, B_c)
            C_input_c = dB_c * xconv_c[..., None]

            P_local, S_local = jax.lax.associative_scan(_combine, (da_c, C_input_c), axis=1)

            global_da = P_local * carry_da[:, None, :]
            global_h = P_local[..., None] * carry_h[:, None, :, :] + S_local

            y_c = jnp.einsum("bcds,bcs->bcd", global_h, C_c)
            new_carry = (global_da[:, -1], global_h[:, -1])
            return new_carry, y_c

        _, y_chunks = jax.lax.scan(
            _chunk_step, (carry_da_init, carry_h_init), (dA_ch, dt_ch, B_ch, C_ch, x_conv_ch)
        )
        y = jnp.moveaxis(y_chunks, 0, 1).reshape(b, l, d_inner)

        out = y * jax.nn.silu(res)
        return nn.Dense(d, use_bias=False, name="out_proj", dtype=jnp.bfloat16)(out)


# ==========================================
# Gated DeltaNet-2
# ==========================================
class GatedDeltaNet2J(nn.Module):
    """Gated Delta Rule-2 (Hatamizadeh, Choi, Kautz -- NVIDIA, arXiv:2605.22791, May 2026)."""

    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        b, l, d = x.shape
        n_heads = self.cfg.n_heads
        d_head = d // n_heads
        eps = 1e-6

        def short_causal_conv(name, u):
            conv_w = self.param(f"{name}_conv_w", nn.initializers.normal(stddev=0.02), (d, self.cfg.d_conv))
            conv_b = self.param(f"{name}_conv_b", nn.initializers.zeros, (d,))
            rhs = conv_w.T[:, None, :]
            out = jax.lax.conv_general_dilated(
                lhs=u,
                rhs=rhs,
                window_strides=(1,),
                padding=[(self.cfg.d_conv - 1, 0)],
                feature_group_count=d,
                dimension_numbers=('NHC', 'HIO', 'NHC')
            )
            return out + conv_b[None, None, :]

        q_lin = nn.Dense(d, use_bias=False, name="q_proj", dtype=jnp.bfloat16)(x)
        k_lin = nn.Dense(d, use_bias=False, name="k_proj", dtype=jnp.bfloat16)(x)
        v_lin = nn.Dense(d, use_bias=False, name="v_proj", dtype=jnp.bfloat16)(x)

        q = jax.nn.silu(short_causal_conv("q", q_lin)).reshape(b, l, n_heads, d_head)
        k = jax.nn.silu(short_causal_conv("k", k_lin)).reshape(b, l, n_heads, d_head)
        v = jax.nn.silu(short_causal_conv("v", v_lin)).reshape(b, l, n_heads, d_head)

        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + eps)

        b_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="erase_gate", dtype=jnp.bfloat16)(x)).reshape(b, l, n_heads, d_head)
        w_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="write_gate", dtype=jnp.bfloat16)(x)).reshape(b, l, n_heads, d_head)

        a_param = self.param("decay_a", nn.initializers.zeros, (n_heads,))
        f_proj = nn.Dense(d, use_bias=True, name="decay_proj", dtype=jnp.bfloat16)(x).reshape(b, l, n_heads, d_head)
        g = -jnp.exp(a_param)[None, None, :, None] * jax.nn.softplus(f_proj)
        alpha = jnp.exp(g)

        out_gate = nn.Dense(d, use_bias=False, name="out_gate", dtype=jnp.bfloat16)(x)

        e = b_gate * k
        z = w_gate * v
        ea = e * alpha

        chunk_size = min(self.cfg.deltanet_chunk_size, l)
        if l % chunk_size != 0:
            raise ValueError(
                f"seq_len={l} must be divisible by deltanet_chunk_size={chunk_size} "
                "(chunked GatedDeltaNet2 scan requires equal-sized chunks)."
            )
        num_chunks = l // chunk_size

        def _combine(state1, state2):
            m1, c1 = state1
            m2, c2 = state2
            return m2 @ m1, m2 @ c1 + c2

        def _to_chunks(t):
            t = t.reshape(b, num_chunks, chunk_size, n_heads, d_head)
            return jnp.moveaxis(t, 1, 0)

        k_ch, ea_ch, z_ch, alpha_ch, q_ch = map(_to_chunks, (k, ea, z, alpha, q))

        eye_bh = jnp.broadcast_to(jnp.eye(d_head, dtype=x.dtype), (b, n_heads, d_head, d_head))
        zero_bh = jnp.zeros((b, n_heads, d_head, d_head), dtype=x.dtype)

        def _chunk_step(carry, chunk_inputs):
            carry_M, carry_S = carry
            k_c, ea_c, z_c, alpha_c, q_c = chunk_inputs

            eye = jnp.eye(d_head, dtype=x.dtype)[None, None, None, :, :]
            M_c = eye * alpha_c[:, :, :, None, :] - k_c[:, :, :, :, None] @ ea_c[:, :, :, None, :]
            C_c = k_c[:, :, :, :, None] @ z_c[:, :, :, None, :]

            P_local, S_local = jax.lax.associative_scan(_combine, (M_c, C_c), axis=1)

            global_M = jnp.einsum("bchmn,bhnp->bchmp", P_local, carry_M)
            global_S = jnp.einsum("bchmn,bhnp->bchmp", P_local, carry_S) + S_local

            out_c = jnp.einsum("bchij,bchi->bchj", global_S, q_c)
            new_carry = (global_M[:, -1], global_S[:, -1])
            return new_carry, out_c

        _, out_chunks = jax.lax.scan(
            _chunk_step, (eye_bh, zero_bh), (k_ch, ea_ch, z_ch, alpha_ch, q_ch)
        )
        out = jnp.moveaxis(out_chunks, 0, 1).reshape(b, l, d)

        out = nn.RMSNorm(epsilon=1e-6, name="out_norm", dtype=jnp.float32)(out)
        return nn.Dense(d, use_bias=False, name="out_proj", dtype=jnp.bfloat16)(out * jax.nn.silu(out_gate))


# ==========================================
# MoE -- gather/scatter dispatch, O(N) instead of O(N * capacity)
# ==========================================
class ExpertPack(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        h = nn.Dense(self.cfg.d_ff, name="w1", dtype=jnp.bfloat16)(x)
        h = jax.nn.gelu(h)
        h = nn.Dropout(rate=self.cfg.dropout_rate)(h, deterministic=deterministic)
        return nn.Dense(self.cfg.d_model, name="w2", dtype=jnp.bfloat16)(h)


class MoEJ(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rngs=None):
        b, l, d = x.shape
        flat_x = x.reshape(-1, d)
        num_tokens = flat_x.shape[0]
        E, K = self.cfg.num_experts, self.cfg.top_k
        n_assign = num_tokens * K

        router_logits = nn.Dense(E, use_bias=False, name="router", dtype=jnp.bfloat16)(flat_x)
        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape
            )

        router_probs = jax.nn.softmax(router_logits, axis=-1)
        top_k_vals, top_k_idx = jax.lax.top_k(router_probs, K)

        gate = top_k_vals / (jnp.sum(top_k_vals, axis=-1, keepdims=True) + 1e-9)

        flat_expert_idx = top_k_idx.reshape(-1)
        flat_gate = gate.reshape(-1)
        flat_token_idx = jnp.repeat(jnp.arange(num_tokens), K)

        mean_probs = jnp.mean(router_probs, axis=0)
        expert_gate_frac = jnp.zeros(E, dtype=flat_gate.dtype).at[flat_expert_idx].add(flat_gate) / num_tokens
        self.sow("losses", "aux_loss", E * jnp.sum(mean_probs * expert_gate_frac))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(jax.scipy.special.logsumexp(router_logits, axis=-1))))
        expert_assign_frac = jnp.zeros(E, dtype=flat_gate.dtype).at[flat_expert_idx].add(1.0) / n_assign
        self.sow("losses", "expert_utilization", expert_assign_frac)

        capacity = max(1, int(self.cfg.moe_capacity_factor * num_tokens * K / E))

        sort_order = jnp.argsort(flat_expert_idx)
        sorted_expert = flat_expert_idx[sort_order]

        expert_counts = jnp.zeros(E, dtype=jnp.int32).at[flat_expert_idx].add(1)
        expert_start = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(expert_counts)[:-1]])

        pos_in_bucket_sorted = jnp.arange(n_assign) - expert_start[sorted_expert]
        valid_sorted = pos_in_bucket_sorted < capacity

        dest_row = sorted_expert * capacity + jnp.minimum(pos_in_bucket_sorted, capacity - 1)

        gathered_x = flat_x[flat_token_idx][sort_order]
        buffer = jnp.zeros((E * capacity, d), dtype=flat_x.dtype)
        buffer = buffer.at[dest_row].add(jnp.where(valid_sorted[:, None], gathered_x, 0.0))
        expert_inputs = buffer.reshape(E, capacity, d)

        run_experts = nn.vmap(
            ExpertPack,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, None),
            out_axes=0,
        )(cfg=self.cfg, name="experts_block")
        expert_outputs = run_experts(expert_inputs, deterministic)

        flat_expert_outputs = expert_outputs.reshape(E * capacity, d)
        gathered_out_sorted = flat_expert_outputs[dest_row]
        gathered_out_sorted = jnp.where(valid_sorted[:, None], gathered_out_sorted, 0.0)

        unsort_order = jnp.argsort(sort_order)
        gathered_out = gathered_out_sorted[unsort_order]
        weighted_out = gathered_out * flat_gate[:, None]

        flat_outputs = jnp.zeros_like(flat_x).at[flat_token_idx].add(weighted_out)
        return flat_outputs.reshape(b, l, d)


# ==========================================
# Delta-Attention Residual Block
# ==========================================
class DeltaAttentionResidualBlockJ(nn.Module):
    cfg: ModelConfig
    layer_idx: int

    @nn.compact
    def __call__(self, current_x, history_deltas, causal_mask, cos, sin, deterministic: bool = True, rngs=None):
        norm_1 = nn.RMSNorm(epsilon=1e-6, name="norm_1", dtype=jnp.float32)(current_x)
        gdn_out = GatedDeltaNet2J(cfg=self.cfg, name="gdn")(norm_1)
        mamba_out = Mamba2J(cfg=self.cfg, name="mamba")(norm_1)
        mla_out = MLAJ(cfg=self.cfg, name="mla")(
            norm_1, causal_mask, cos, sin, deterministic=deterministic, rngs=rngs
        )
        alpha = jax.nn.softmax(self.param("alpha", nn.initializers.zeros, (3,)))
        current_delta = jnp.einsum("i,ibld->bld", alpha, jnp.stack([gdn_out, mamba_out, mla_out], axis=0))

        updated_history = history_deltas.at[self.layer_idx].set(current_delta)

        q_route = nn.Dense(self.cfg.d_latent, use_bias=False, name="q_route", dtype=jnp.bfloat16)(current_x)
        k_route = nn.Dense(self.cfg.d_latent, use_bias=False, name="k_route", dtype=jnp.bfloat16)(updated_history)
        routing_scores = jnp.einsum("bld,vbld->blv", q_route, k_route) / jnp.sqrt(self.cfg.d_latent)

        depth_mask = jnp.arange(self.cfg.num_layers) <= self.layer_idx
        routing_scores = jnp.where(depth_mask[None, None, :], routing_scores, -1e9)
        routing_weights = jax.nn.softmax(routing_scores, axis=-1)

        moe_in = current_x + jnp.einsum("blv,vbld->bld", routing_weights, updated_history)
        norm_2 = nn.RMSNorm(epsilon=1e-6, name="norm_2", dtype=jnp.float32)(moe_in)
        moe_out = MoEJ(cfg=self.cfg, name="moe")(norm_2, deterministic=deterministic, rngs=rngs)
        return moe_in + moe_out, updated_history


# ==========================================
# Block of 2 consecutive Delta-Attention Residual layers, sharing one remat scope
# ==========================================
class DeltaResidualBlockPairJ(nn.Module):
    cfg: ModelConfig
    layer_idx_0: int

    @nn.compact
    def __call__(self, current_x, history_deltas, causal_mask, cos, sin, deterministic: bool = True, rngs=None):
        current_x, history_deltas = DeltaAttentionResidualBlockJ(
            cfg=self.cfg, layer_idx=self.layer_idx_0, name=f"layer_{self.layer_idx_0}"
        )(current_x, history_deltas, causal_mask, cos, sin, deterministic, rngs)
        current_x, history_deltas = DeltaAttentionResidualBlockJ(
            cfg=self.cfg, layer_idx=self.layer_idx_0 + 1, name=f"layer_{self.layer_idx_0 + 1}"
        )(current_x, history_deltas, causal_mask, cos, sin, deterministic, rngs)
        return current_x, history_deltas


# ==========================================
# Full model
# ==========================================
class FullHybridMoEModel(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True, rngs=None, return_hidden: bool = False):
        b, l = input_ids.shape
        embed_layer = nn.Embed(num_embeddings=self.cfg.vocab_size, features=self.cfg.d_model, name="embed", dtype=jnp.bfloat16)
        x = embed_layer(input_ids)
        causal_mask = jnp.tril(jnp.ones((l, l))).astype(jnp.bool_)[None, None, :, :]

        d_head = self.cfg.d_model // self.cfg.n_heads
        cos, sin = RoPEEmbedding(dim=d_head)(l)

        history_deltas = jnp.zeros((self.cfg.num_layers, b, l, self.cfg.d_model), dtype=x.dtype)

        RematPair = nn.remat(DeltaResidualBlockPairJ, static_argnums=(6,))
        RematSingle = nn.remat(DeltaAttentionResidualBlockJ, static_argnums=(6,))

        num_full_pairs = self.cfg.num_layers // 2
        for p in range(num_full_pairs):
            i = p * 2
            x, history_deltas = RematPair(
                cfg=self.cfg, layer_idx_0=i, name=f"layer_pair_{i}"
            )(x, history_deltas, causal_mask, cos, sin, deterministic, rngs)

        if self.cfg.num_layers % 2 == 1:
            i = num_full_pairs * 2
            x, history_deltas = RematSingle(
                cfg=self.cfg, layer_idx=i, name=f"layer_{i}"
            )(x, history_deltas, causal_mask, cos, sin, deterministic, rngs)

        final = nn.RMSNorm(epsilon=1e-6, name="final_norm", dtype=jnp.float32)(x)

        # return_hidden=True skips the vocab projection entirely. (batch, seq,
        # vocab) logits + log_probs together dominate memory at vocab_size=151936
        # (~2.5GB EACH at batch=2, seq=2048, fp32) and this projection sits outside
        # the nn.remat scopes above (those only cover the transformer block pairs),
        # so nothing here gets recomputed instead of stored during backward. The
        # only way to avoid materializing the full tensor is to never build it in
        # the first place -- compute_loss's chunked_cross_entropy does the
        # projection itself, chunk by chunk, straight from `final`. Default is
        # False so any other caller (generation, eval scripts, etc.) keeps getting
        # full logits unchanged.
        if return_hidden:
            return final

        if self.cfg.tie_embeddings:
            logits = embed_layer.attend(final)
        else:
            logits = nn.Dense(self.cfg.vocab_size, use_bias=False, name="lm_head", dtype=jnp.bfloat16)(final)
        return logits
