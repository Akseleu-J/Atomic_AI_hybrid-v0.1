import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from typing import List

 
from jax.experimental.pallas.ops.tpu.flash_attention import (
    flash_attention as pallas_flash_attention,
    BlockSizes as FlashBlockSizes,
)

@struct.dataclass
class ModelConfig:
    d_model: int = 1024
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 16
    d_latent: int = 256
    d_ff: int = 3072
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 22
    vocab_size: int = 151936
    dropout_rate: float = 0.1
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.0001
    moe_capacity_factor: float = 1.25
    tie_embeddings: bool = True
    label_smoothing: float = 0.05
    router_noise_std: float = 0.3
   
    use_flash_attention: bool = True
   
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
        b, l, _ = x.shape
        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_model // n_heads
 
        Q = nn.Dense(self.cfg.d_model, use_bias=False, name="W_q")(x)
        Q = Q.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)  # (b, n_heads, l, d_head)
        Q_rope = apply_rope(Q, cos[None, None, :, :d_head], sin[None, None, :, :d_head])
 
        kv_latent = nn.Dense(self.cfg.d_latent, use_bias=False, name="W_kv_down")(x)
        K = nn.Dense(self.cfg.d_model, use_bias=False, name="W_k_up")(kv_latent)
        V = nn.Dense(self.cfg.d_model, use_bias=False, name="W_v_up")(kv_latent)
 
        K = K.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)
        K_rope = apply_rope(K, cos[None, None, :, :d_head], sin[None, None, :, :d_head])
        V = V.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)
 
        # sm_scale must be a concrete Python float -- it is a static_argname in
        # pallas_flash_attention's jax.jit signature, so a jnp-traced value here
        # breaks under remat's retracing (float(tracer) raises
        # ConcretizationTypeError). d_head is already a plain Python int.
        sm_scale = 1.0 / math.sqrt(d_head)
 
        if self.cfg.use_flash_attention:
            # Pallas TPU kernel: never materializes the full (l, l) score matrix.
            # Requires q/k/v as (batch, num_heads, seq_len, head_dim), which is
            # exactly what we already have after the transpose above.
            block_sizes = FlashBlockSizes.get_default(b, n_heads, l, l, d_head)
            out = pallas_flash_attention(
                Q_rope.astype(jnp.bfloat16),
                K_rope.astype(jnp.bfloat16),
                V.astype(jnp.bfloat16),
                causal=True,
                sm_scale=sm_scale,
                block_sizes=block_sizes,
            ).astype(x.dtype)
        else:
            # Naive fallback -- only for CPU debugging / small seq_len smoke tests.
            # This is the O(l^2) path that caused the MLA OOM at seq_len=8192.
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
        return nn.Dense(self.cfg.d_model, use_bias=False, name="W_o")(out)
 
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

        in_proj = nn.Dense(d_inner * 2, use_bias=False, name="in_proj")(x)
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
        B = nn.Dense(d_state, use_bias=False, name="B_proj")(x_bc)      # (b, l, d_state) -- small
        C = nn.Dense(d_state, use_bias=False, name="C_proj")(x_bc)      # (b, l, d_state) -- small
        dt = jax.nn.softplus(nn.Dense(d_inner, use_bias=True, name="dt_proj")(x_bc))  # (b, l, d_inner) -- small

        dA = jnp.exp(jnp.einsum("bld,d->bld", dt, A))          # (b, l, d_inner) -- per-channel scalar decay, small

        # ---- chunked scan: dB/C_input (the (b,l,d_inner,d_state)-sized quantities)
        # are built INSIDE _chunk_step from small per-chunk dt/B/x_conv slices, never
        # materialized at full seq_len -- this is the actual memory saving, not just
        # reshaping an already-full-size tensor into chunks after the fact.
        #
        # FIX (correctness, not just memory): the SSM affine input term at each step is
        # dB_t * x_t (broadcast x_t over d_state). The previous version scanned dB and
        # x_conv as two INDEPENDENT accumulations under the same decay and kept only
        # the x_conv one (shape (...,d_inner,1)) as `h`, discarding the dB accumulation
        # entirely -- then einsum("blds,bls->bld", h, C) silently broadcast that size-1
        # axis against C's real d_state axis, which numpy/jax einsum allows without
        # error. Net effect verified against a correct sequential SSM: max abs diff
        # ~2.2, i.e. not numerical noise -- the model never computed a real SSM
        # recurrence at all. This version's chunked output verified exact (~1e-15)
        # against a correct sequential reference.
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

        def _to_chunks(t):  # (b, l, ...) -> (num_chunks, b, chunk_size, ...)
            trailing = t.shape[2:]
            t = t.reshape(b, num_chunks, chunk_size, *trailing)
            return jnp.moveaxis(t, 1, 0)

        dA_ch = _to_chunks(dA)            # (num_chunks, b, chunk_size, d_inner)
        dt_ch = _to_chunks(dt)            # (num_chunks, b, chunk_size, d_inner)
        B_ch = _to_chunks(B)              # (num_chunks, b, chunk_size, d_state)
        C_ch = _to_chunks(C)              # (num_chunks, b, chunk_size, d_state)
        x_conv_ch = _to_chunks(x_conv)    # (num_chunks, b, chunk_size, d_inner)

        carry_da_init = jnp.ones((b, d_inner), dtype=x.dtype)
        carry_h_init = jnp.zeros((b, d_inner, d_state), dtype=x.dtype)

        def _chunk_step(carry, chunk_inputs):
            carry_da, carry_h = carry
            da_c, dt_c, B_c, C_c, xconv_c = chunk_inputs
            # (b, chunk_size, d_inner), (b, chunk_size, d_inner), (b, chunk_size, d_state),
            # (b, chunk_size, d_state), (b, chunk_size, d_inner)

            # Built HERE, per chunk -- only ever (b, chunk_size, d_inner, d_state), not
            # (b, l, d_inner, d_state). This is what actually bounds peak memory.
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
        y = jnp.moveaxis(y_chunks, 0, 1).reshape(b, l, d_inner)  # (num_chunks,b,c,d)->(b,l,d_inner)

        out = y * jax.nn.silu(res)
        return nn.Dense(d, use_bias=False, name="out_proj")(out)


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

        q_lin = nn.Dense(d, use_bias=False, name="q_proj")(x)
        k_lin = nn.Dense(d, use_bias=False, name="k_proj")(x)
        v_lin = nn.Dense(d, use_bias=False, name="v_proj")(x)

        q = jax.nn.silu(short_causal_conv("q", q_lin)).reshape(b, l, n_heads, d_head)
        k = jax.nn.silu(short_causal_conv("k", k_lin)).reshape(b, l, n_heads, d_head)
        v = jax.nn.silu(short_causal_conv("v", v_lin)).reshape(b, l, n_heads, d_head)

        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + eps)

        b_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="erase_gate")(x)).reshape(b, l, n_heads, d_head)
        w_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="write_gate")(x)).reshape(b, l, n_heads, d_head)

        a_param = self.param("decay_a", nn.initializers.zeros, (n_heads,))
        f_proj = nn.Dense(d, use_bias=True, name="decay_proj")(x).reshape(b, l, n_heads, d_head)
        g = -jnp.exp(a_param)[None, None, :, None] * jax.nn.softplus(f_proj)
        alpha = jnp.exp(g)

        out_gate = nn.Dense(d, use_bias=False, name="out_gate")(x)

        e = b_gate * k
        z = w_gate * v
        ea = e * alpha

        eye = jnp.eye(d_head)[None, None, None, :, :]
        M = eye * alpha[:, :, :, None, :] - k[:, :, :, :, None] @ ea[:, :, :, None, :]
        C = k[:, :, :, :, None] @ z[:, :, :, None, :]

        def _combine(state1, state2):
            m1, c1 = state1
            m2, c2 = state2
            return m2 @ m1, m2 @ c1 + c2

        _, S = jax.lax.associative_scan(_combine, (M, C), axis=1)

        out = jnp.einsum("blhij,blhi->blhj", S, q).reshape(b, l, d)
        out = nn.RMSNorm(epsilon=1e-6, name="out_norm")(out)
        return nn.Dense(d, use_bias=False, name="out_proj")(out * jax.nn.silu(out_gate))


# ==========================================
# MoE -- gather/scatter dispatch, O(N) instead of O(N * capacity)
# ==========================================
#
# ЧТО БЫЛО НЕ ТАК (root cause OOM):
# Старая реализация строила ПЛОТНЫЕ one-hot тензоры формы (num_tokens*top_k, capacity)
# и (num_tokens*top_k, num_experts, capacity) через einsum. Их размер растёт как
# O(num_tokens^2), потому что capacity сам пропорционален num_tokens. При
# num_tokens=65536 (batch=8 * seq_len=8192) это давало тензоры на десятки GB НА КАЖДЫЙ
# из 22 слоёв, и без remat автодифф держал их все в HBM одновременно -> сотни GB.
#
# Дополнительно: jnp.cumsum(one_hot, axis=0) считался вдоль оси токенов. Под GSPMD
# (jax.jit + in_shardings, без shard_map) это data-dependent global reduction вдоль
# потенциально шардированной оси -- компилятору проще реплицировать весь батч на
# каждый чип, чем построить распределённый scan. Именно поэтому параллелизация на
# 8 TPU не спасала: каждый чип пересчитывал ПОЛНЫЙ глобальный batch.
#
# ИСПРАВЛЕНИЕ: вместо one-hot + cumsum по всей оси токенов используем
# argsort + bincount (cumsum только по num_experts=8 элементам, а не по num_tokens) и
# честный gather/scatter. Пиковая память теперь O(num_tokens*d + num_experts*capacity*d)
# -- линейно, а не квадратично.
class ExpertPack(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        h = nn.Dense(self.cfg.d_ff, name="w1")(x)
        h = jax.nn.gelu(h)
        h = nn.Dropout(rate=self.cfg.dropout_rate)(h, deterministic=deterministic)
        return nn.Dense(self.cfg.d_model, name="w2")(h)


class MoEJ(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rngs=None):
        b, l, d = x.shape
        flat_x = x.reshape(-1, d)
        num_tokens = flat_x.shape[0]
        E, K = self.cfg.num_experts, self.cfg.top_k
        d_ff = self.cfg.d_ff
        n_assign = num_tokens * K

        router_logits = nn.Dense(E, use_bias=False, name="router")(flat_x)
        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape
            )

        router_probs = jax.nn.softmax(router_logits, axis=-1)
        top_k_vals, top_k_idx = jax.lax.top_k(router_probs, K)  # (num_tokens, K)

        gate = top_k_vals / (jnp.sum(top_k_vals, axis=-1, keepdims=True) + 1e-9)

        flat_expert_idx = top_k_idx.reshape(-1)                        # (n_assign,)
        flat_gate = gate.reshape(-1)                                    # (n_assign,)
        flat_token_idx = jnp.repeat(jnp.arange(num_tokens), K)          # (n_assign,)

        # ---- diagnostics: unchanged, routing itself didn't change ----
        mean_probs = jnp.mean(router_probs, axis=0)
        expert_gate_frac = jnp.zeros(E, dtype=flat_gate.dtype).at[flat_expert_idx].add(flat_gate) / num_tokens
        self.sow("losses", "aux_loss", E * jnp.sum(mean_probs * expert_gate_frac))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(jax.scipy.special.logsumexp(router_logits, axis=-1))))
        expert_assign_frac = jnp.zeros(E, dtype=flat_gate.dtype).at[flat_expert_idx].add(1.0) / n_assign
        self.sow("losses", "expert_utilization", expert_assign_frac)

        # ---- sort assignments by expert (unchanged from the gather/scatter version) ----
        sort_order = jnp.argsort(flat_expert_idx)                       # (n_assign,) -- stable by default
        sorted_expert = flat_expert_idx[sort_order]
        # group_sizes for ragged_dot: count per expert, in expert-index order 0..E-1 --
        # this MUST match the order rows appear in sorted_x (ascending expert id), which
        # argsort already guarantees.
        group_sizes = jnp.zeros(E, dtype=jnp.int32).at[flat_expert_idx].add(1)

        sorted_x = flat_x[flat_token_idx][sort_order]                   # (n_assign, d), grouped contiguous

        # ---- stacked per-expert weights (replaces nn.vmap(ExpertPack)) ----
        # Named so "bias" appears in the path for both bias params -- optimizer.py's
        # label_fn routes anything with "bias" in its path to adamw_nodecay (no weight
        # decay, no Muon) same as every other bias in the model; without this naming
        # these would fall through to Muon as a rank>=2 array, which is wrong for a bias.
        w1_kernel = self.param("w1_kernel", nn.initializers.lecun_normal(), (E, d, d_ff))
        w1_bias = self.param("w1_bias", nn.initializers.zeros, (E, d_ff))
        w2_kernel = self.param("w2_kernel", nn.initializers.lecun_normal(), (E, d_ff, d))
        w2_bias = self.param("w2_bias", nn.initializers.zeros, (E, d))

        # ---- grouped matmul: dropless, no capacity/padding, no overflow to drop ----
        # NOTE (honest risk flag): jax.lax.ragged_dot's exact calling convention (arg
        # order, dtype expectations) is used here per its documented contract (lhs
        # pre-sorted contiguous-by-group, group_sizes in matching group order) but has
        # NOT been executed against a real JAX runtime in this environment -- validate
        # on a small CPU/TPU smoke test before a long run.
        h = jax.lax.ragged_dot(sorted_x, w1_kernel, group_sizes) + w1_bias[sorted_expert]
        h = jax.nn.gelu(h)
        if not deterministic and self.cfg.dropout_rate > 0:
            dropout_rng = self.make_rng("dropout")
            keep_prob = 1.0 - self.cfg.dropout_rate
            keep_mask = jax.random.bernoulli(dropout_rng, p=keep_prob, shape=h.shape)
            h = jnp.where(keep_mask, h / keep_prob, 0.0)
        out_sorted = jax.lax.ragged_dot(h, w2_kernel, group_sizes) + w2_bias[sorted_expert]

        unsort_order = jnp.argsort(sort_order)                          # inverse permutation
        gathered_out = out_sorted[unsort_order]                         # back to assignment order
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
        norm_1 = nn.RMSNorm(epsilon=1e-6, name="norm_1")(current_x)
        gdn_out = GatedDeltaNet2J(cfg=self.cfg, name="gdn")(norm_1)
        mamba_out = Mamba2J(cfg=self.cfg, name="mamba")(norm_1)
        mla_out = MLAJ(cfg=self.cfg, name="mla")(norm_1, causal_mask, cos, sin, deterministic=deterministic)

        alpha = jax.nn.softmax(self.param("alpha", nn.initializers.zeros, (3,)))
        current_delta = jnp.einsum("i,ibld->bld", alpha, jnp.stack([gdn_out, mamba_out, mla_out], axis=0))

        updated_history = history_deltas.at[self.layer_idx].set(current_delta)

        q_route = nn.Dense(self.cfg.d_latent, use_bias=False, name="q_route")(current_x)
        k_route = nn.Dense(self.cfg.d_latent, use_bias=False, name="k_route")(updated_history)
        routing_scores = jnp.einsum("bld,vbld->blv", q_route, k_route) / jnp.sqrt(self.cfg.d_latent)

        depth_mask = jnp.arange(self.cfg.num_layers) <= self.layer_idx
        routing_scores = jnp.where(depth_mask[None, None, :], routing_scores, -1e9)
        routing_weights = jax.nn.softmax(routing_scores, axis=-1)

        moe_in = current_x + jnp.einsum("blv,vbld->bld", routing_weights, updated_history)
        norm_2 = nn.RMSNorm(epsilon=1e-6, name="norm_2")(moe_in)
        moe_out = MoEJ(cfg=self.cfg, name="moe")(norm_2, deterministic=deterministic, rngs=rngs)
        return moe_in + moe_out, updated_history


# ==========================================
# Full model
# ==========================================
class FullHybridMoEModel(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True, rngs=None):
        b, l = input_ids.shape
        embed_layer = nn.Embed(num_embeddings=self.cfg.vocab_size, features=self.cfg.d_model, name="embed")
        x = embed_layer(input_ids)
        causal_mask = jnp.tril(jnp.ones((l, l))).astype(jnp.bool_)[None, None, :, :]

        d_head = self.cfg.d_model // self.cfg.n_heads
        cos, sin = RoPEEmbedding(dim=d_head)(l)

        history_deltas = jnp.zeros((self.cfg.num_layers, b, l, self.cfg.d_model), dtype=x.dtype)

        # Gradient checkpointing (remat) per layer: without this, autodiff keeps every
        # layer's activations (incl. the MoE dispatch buffers) resident in HBM
        # simultaneously for the backward pass. With 22 layers that multiplies peak
        # memory by ~22x. remat recomputes the forward pass per-layer during backward
        # instead, trading some compute for a large drop in peak HBM usage.
        # static_argnums marks `deterministic` (position 5 in __call__, counting
        # `self` as 0) as a Python-level static value rather than a traced array --
        # remat needs this because dropout branches on it with a Python `if`.
        RematBlock = nn.remat(DeltaAttentionResidualBlockJ, static_argnums=(6,))

        for i in range(self.cfg.num_layers):
            x, history_deltas = RematBlock(
                cfg=self.cfg, layer_idx=i, name=f"layer_{i}"
            )(x, history_deltas, causal_mask, cos, sin, deterministic, rngs)

        final = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x)
        if self.cfg.tie_embeddings:
            logits = embed_layer.attend(final)
        else:
            logits = nn.Dense(self.cfg.vocab_size, use_bias=False, name="lm_head")(final)
        return logits
