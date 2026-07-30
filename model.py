import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from typing import List


@struct.dataclass
class ModelConfig:
    # Scaled up from the original 768/18-layer config (~1.03B params, embed+head ~23% of
    # total) to ~1.76B params with embed+head down to ~9% of total -- verified with a
    # standalone parameter counter before touching this file. d_head = 1024/16 = 64,
    # matching d_state for a clean SSM/attention head-dim split.
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
    # Anti-overfitting / anti-routing-collapse additions:
    tie_embeddings: bool = True     # share embed<->lm_head weights: saves ~155M params
                                     # (vocab_size*d_model) AND regularizes the huge vocab
                                     # table, since it must now serve both roles at once.
    label_smoothing: float = 0.05   # softens the CE target distribution; set to 0.0 to disable.
    router_noise_std: float = 0.3   # Gaussian noise added to router logits during training
                                     # ONLY (Switch Transformer / ST-MoE-style jitter) --
                                     # discourages the router from collapsing onto a fixed
                                     # small subset of experts early in training. Annealed
                                     # to 0 is common in the literature; kept constant here
                                     # for simplicity -- lower it if routing looks TOO noisy
                                     # in the expert-utilization logs (see MoEJ).


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
    def __call__(self, x, causal_mask, cos, sin, deterministic: bool = True):
        b, l, _ = x.shape
        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_model // n_heads

        Q = nn.Dense(self.cfg.d_model, use_bias=False, name="W_q")(x)
        Q = Q.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)
        Q_rope = apply_rope(Q, cos[None, None, :, :d_head], sin[None, None, :, :d_head])

        kv_latent = nn.Dense(self.cfg.d_latent, use_bias=False, name="W_kv_down")(x)
        K = nn.Dense(self.cfg.d_model, use_bias=False, name="W_k_up")(kv_latent)
        V = nn.Dense(self.cfg.d_model, use_bias=False, name="W_v_up")(kv_latent)

        K = K.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)
        K_rope = apply_rope(K, cos[None, None, :, :d_head], sin[None, None, :, :d_head])
        V = V.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhqd,bhkd->bhqk", Q_rope, K_rope) / jnp.sqrt(d_head)
        scores = jnp.where(causal_mask == 0, -1e9, scores)
        attn = jax.nn.softmax(scores, axis=-1)
        attn = nn.Dropout(rate=self.cfg.dropout_rate)(attn, deterministic=deterministic)

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

        in_proj = nn.Dense(d_inner * 2, use_bias=False, name="in_proj")(x)
        x_bc, res = jnp.split(in_proj, 2, axis=-1)

        conv_w = self.param("conv_w", nn.initializers.normal(stddev=0.02), (d_inner, self.cfg.d_conv))
        conv_b = self.param("conv_b", nn.initializers.zeros, (d_inner,))

        rhs = conv_w.T[:, None, :]  # (d_conv, 1, d_inner)
        res_conv = jax.lax.conv_general_dilated(
            lhs=x_bc,
            rhs=rhs,
            window_strides=(1,),
            padding=[(self.cfg.d_conv - 1, 0)],
            feature_group_count=d_inner,
        )
        x_conv = jax.nn.silu(res_conv + conv_b[None, None, :])

        A = -jnp.exp(self.param("A_log", nn.initializers.uniform(scale=1.0), (d_inner,)))
        B = nn.Dense(self.cfg.d_state, use_bias=False, name="B_proj")(x_bc)
        C = nn.Dense(self.cfg.d_state, use_bias=False, name="C_proj")(x_bc)
        dt = jax.nn.softplus(nn.Dense(d_inner, use_bias=True, name="dt_proj")(x_bc))

        dA = jnp.exp(jnp.einsum("bld,d->bld", dt, A))
        dB = jnp.einsum("bld,bls->blds", dt, B)

        def _associative_scan_mamba(a, b_val):
            da1, db1, xc1 = a
            da2, db2, xc2 = b_val
            da2e = da2[..., None]
            return da2 * da1, da2e * db1 + db2, da2e * xc1 + xc2

        _, _, h = jax.lax.associative_scan(
            _associative_scan_mamba, (dA, dB, x_conv[..., None]), axis=1
        )
        y = jnp.einsum("blds,bls->bld", h, C)
        out = y * jax.nn.silu(res)
        return nn.Dense(d, use_bias=False, name="out_proj")(out)


# ==========================================
# Gated DeltaNet-2
# ==========================================
class GatedDeltaNet2J(nn.Module):
    """Gated Delta Rule-2 (Hatamizadeh, Choi, Kautz -- NVIDIA, arXiv:2605.22791, May 2026).

    Implements the paper's recurrence exactly (Eq. 8-12, Eq. 29):
        e_t = b_t (.) k_t,           z_t = w_t (.) v_t                              (Eq. 8)
        S_t = (I - k_t e_t^T) Diag(alpha_t) S_{t-1} + k_t z_t^T,  o_t = S_t^T q_t    (Eq. 10/29)
        alpha_t = exp(g_t),  g_t = -exp(a) (.) softplus(W_f x_t + delta)            (Eq. 12)
    with a channel-wise erase gate b_t in [0,1]^{d_k} (key axis) and a channel-wise write
    gate w_t in [0,1]^{d_v} (value axis) -- the paper's core change vs. plain Gated DeltaNet /
    KDA, which tie erase and write to one scalar beta_t.

    Honest scope note: this uses jax.lax.associative_scan over the sequence axis, which is
    mathematically equivalent to Eq. 29 but is NOT the paper's chunkwise WY / Triton kernel
    (Sec. 3.3, Appendix A-C) -- that's a separate hardware-level optimization of the same
    recurrence, needed for real throughput at scale but out of scope for a from-scratch JAX
    reimplementation here. Also simplified vs. the paper: the decay bias `delta` (Eq. 12) is
    folded into the Dense layer's own bias rather than kept as a separate parameter, and
    grouped-value-head repetition and the negative-eigenvalue erase-range extension
    (Appendix, b_t in [0,2]^{d_k}) are omitted.
    """

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
            rhs = conv_w.T[:, None, :]  # (d_conv, 1, d)
            out = jax.lax.conv_general_dilated(
                lhs=u, rhs=rhs, window_strides=(1,),
                padding=[(self.cfg.d_conv - 1, 0)], feature_group_count=d,
            )
            return out + conv_b[None, None, :]

        q_lin = nn.Dense(d, use_bias=False, name="q_proj")(x)
        k_lin = nn.Dense(d, use_bias=False, name="k_proj")(x)
        v_lin = nn.Dense(d, use_bias=False, name="v_proj")(x)

        q = jax.nn.silu(short_causal_conv("q", q_lin)).reshape(b, l, n_heads, d_head)
        k = jax.nn.silu(short_causal_conv("k", k_lin)).reshape(b, l, n_heads, d_head)
        v = jax.nn.silu(short_causal_conv("v", v_lin)).reshape(b, l, n_heads, d_head)

        # L2 normalization of q, k per head (Appendix D.2) -- stabilizes the recurrent scores
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + eps)

        # channel-wise erase gate b_t and write gate w_t (Eq. 11) -- the paper's core novelty:
        # DECOUPLED from each other, unlike plain Gated DeltaNet/KDA's single scalar beta_t
        b_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="erase_gate")(x)).reshape(b, l, n_heads, d_head)
        w_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="write_gate")(x)).reshape(b, l, n_heads, d_head)

        # channel-wise log-decay (Eq. 12): g_t = -exp(a) * softplus(W_f x_t + delta)
        # `a` is one scalar per head, broadcast across that head's d_head channels, per the paper
        a_param = self.param("decay_a", nn.initializers.zeros, (n_heads,))
        f_proj = nn.Dense(d, use_bias=True, name="decay_proj")(x).reshape(b, l, n_heads, d_head)
        g = -jnp.exp(a_param)[None, None, :, None] * jax.nn.softplus(f_proj)
        alpha = jnp.exp(g)  # (b, l, n_heads, d_head)

        out_gate = nn.Dense(d, use_bias=False, name="out_gate")(x)

        e = b_gate * k  # erase direction, key-side   (Eq. 8)
        z = w_gate * v  # write target, value-side     (Eq. 8)
        ea = e * alpha  # decay folded into the erase factor, per Eq. 10's (k_t e_t^T) Diag(alpha_t)

        # Per-step affine transition on the state S in R^{d_head x d_head} (key-dim x value-dim):
        #   M_t = Diag(alpha_t) - k_t (e_t * alpha_t)^T   ==  (I - k_t e_t^T) Diag(alpha_t)
        #   C_t = k_t z_t^T
        eye = jnp.eye(d_head)[None, None, None, :, :]
        M = eye * alpha[:, :, :, None, :] - k[:, :, :, :, None] @ ea[:, :, :, None, :]
        C = k[:, :, :, :, None] @ z[:, :, :, None, :]

        def _combine(state1, state2):
            m1, c1 = state1
            m2, c2 = state2
            return m2 @ m1, m2 @ c1 + c2

        # scan along the SEQUENCE axis (axis=1) -- (b, l, ...) has batch on axis 0
        _, S = jax.lax.associative_scan(_combine, (M, C), axis=1)  # S: (b, l, h, d_head, d_head)

        out = jnp.einsum("blhij,blhi->blhj", S, q).reshape(b, l, d)  # o_t = S_t^T q_t
        out = nn.RMSNorm(epsilon=1e-6, name="out_norm")(out)
        return nn.Dense(d, use_bias=False, name="out_proj")(out * jax.nn.silu(out_gate))


# ==========================================
# MoE (capacity-based sparse dispatch, JIT-safe: no dynamic-shape boolean indexing)
# ==========================================
# ==========================================
# MoE (capacity-based sparse dispatch, JIT-safe: no dynamic-shape boolean indexing)
# ==========================================
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

        router_logits = nn.Dense(E, use_bias=False, name="router")(flat_x)

        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape
            )

        router_probs = jax.nn.softmax(router_logits, axis=-1)
        top_k_vals, top_k_idx = jax.lax.top_k(router_probs, K)

        mask = jnp.zeros_like(router_probs)
        t_idx = jnp.arange(num_tokens)[:, None]
        for i in range(K):
            mask = mask.at[t_idx, top_k_idx[:, [i]]].set(top_k_vals[:, [i]])
        mask = mask / (mask.sum(axis=-1, keepdims=True) + 1e-9)

        mean_probs = jnp.mean(router_probs, axis=0)
        expert_counts = jnp.mean(mask, axis=0)
        self.sow("losses", "aux_loss", E * jnp.sum(mean_probs * expert_counts))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(jax.scipy.special.logsumexp(router_logits, axis=-1))))
        self.sow("losses", "expert_utilization", expert_counts)

        capacity = max(1, int(self.cfg.moe_capacity_factor * num_tokens * K / E))

        flat_expert_idx = top_k_idx.reshape(-1)
        flat_token_idx = jnp.repeat(jnp.arange(num_tokens), K)
        flat_gate = mask[flat_token_idx, flat_expert_idx]

        one_hot_expert = jax.nn.one_hot(flat_expert_idx, E)
        position_in_expert = jnp.sum(
            (jnp.cumsum(one_hot_expert, axis=0) - 1) * one_hot_expert, axis=-1
        ).astype(jnp.int32)

        dispatch_pos = jax.nn.one_hot(position_in_expert, capacity)
        dispatch_tensor = one_hot_expert[:, :, None] * dispatch_pos[:, None, :]

        assignment_x = flat_x[flat_token_idx]
        expert_inputs = jnp.einsum("nec,nd->ecd", dispatch_tensor, assignment_x)

        run_experts = nn.vmap(
            ExpertPack,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, None),
            out_axes=0,
        )(cfg=self.cfg, name="experts_block")
        expert_outputs = run_experts(expert_inputs, deterministic)

        combine_tensor = dispatch_tensor * flat_gate[:, None, None]
        per_assignment_out = jnp.einsum("nec,ecd->nd", combine_tensor, expert_outputs)

        flat_outputs = jnp.zeros_like(flat_x).at[flat_token_idx].add(per_assignment_out)
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

        for i in range(self.cfg.num_layers):
            x, history_deltas = DeltaAttentionResidualBlockJ(
                cfg=self.cfg, layer_idx=i, name=f"layer_{i}"
            )(x, history_deltas, causal_mask, cos, sin, deterministic=deterministic, rngs=rngs)

        final = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x)
        if self.cfg.tie_embeddings:
            logits = embed_layer.attend(final)
        else:
            logits = nn.Dense(self.cfg.vocab_size, use_bias=False, name="lm_head")(final)
        return logits
