import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from typing import List, Tuple
from jax.sharding import PartitionSpec as P

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as pallas_flash_attention,
        BlockSizes as FlashBlockSizes,
    )
    _PALLAS_FLASH_ATTENTION_IMPORT_ERROR = None
except Exception as _e:
    pallas_flash_attention = None
    FlashBlockSizes = None
    _PALLAS_FLASH_ATTENTION_IMPORT_ERROR = _e

# jax.checkpoint_policies нет в этой версии JAX — используем строковые policy в nn.remat
_model_mesh = None
_batch_axis = None

def set_model_mesh(mesh, batch_axis=None):
    global _model_mesh, _batch_axis
    _model_mesh = mesh
    _batch_axis = batch_axis

def get_model_mesh():
    return _model_mesh

def get_batch_axis():
    return _batch_axis


# ==========================================================================
# ДИАГНОСТИКА (2-й уровень): forward-активации уже проверены (FWD-DIAG) и
# оказались finite, а градиент всё равно non-finite -- значит проблема
# именно в backward конкретного узла (например, custom VJP flash-attention
# ядра, или матмул с бOльшим диапазоном значений, чем видно по forward
# значению после clip/nan_to_num). identity-функция с custom_vjp пропускает
# forward без изменений, а в backward проверяет входящий котангент.
# ==========================================================================
def make_grad_probe(tag: str):
    @jax.custom_vjp
    def _probe(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        finite = jnp.all(jnp.isfinite(g))
        jax.lax.cond(
            jnp.logical_not(finite),
            lambda: jax.debug.print("[BWD-DIAG] ⚠️ non-finite ВХОДЯЩИЙ градиент в узле: " + tag),
            lambda: None,
        )
        return (g,)

    _probe.defvjp(_fwd, _bwd)
    return _probe


@struct.dataclass
class ModelConfig:
    d_model: int = 512
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 8
    d_latent: int = 384
    d_ff: int = 3072
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 21          # 7 blocks × 3 layers
    layers_per_block: int = 3
    vocab_size: int = 151936
    dropout_rate: float = 0.1
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.0001
    moe_capacity_factor: float = 1.25
    tie_embeddings: bool = True
    label_smoothing: float = 0.05
    router_noise_std: float = 0.3
    use_flash_attention: bool = True
    deltanet_chunk_size: int = 256

    # Layer type schedule: "gdn2", "mamba2", "mla"
    # 21 layers: 16 gdn2, 2 mamba2, 3 mla
    layer_types: Tuple[str, ...] = (
        "gdn2", "gdn2", "mla",      # block 0
        "gdn2", "mamba2", "gdn2",   # block 1
        "gdn2", "gdn2", "gdn2",     # block 2
        "gdn2", "gdn2", "mla",      # block 3
        "gdn2", "mamba2", "gdn2",   # block 4
        "gdn2", "gdn2", "gdn2",     # block 5
        "gdn2", "gdn2", "mla",      # block 6
    )


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
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated_x = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated_x * sin


# ==========================================
# Multi-head Latent Attention (MLA)
# ==========================================
class MLAJ(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, causal_mask, cos, sin, deterministic: bool = True, rngs=None):
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
                    "flash_attention failed to import -- original error: "
                    f"{_PALLAS_FLASH_ATTENTION_IMPORT_ERROR!r}."
                )

            def _flash_call(q_local, k_local, v_local):
                local_b = q_local.shape[0]
                block_sizes = FlashBlockSizes(
                    block_q=1024,
                    block_k_major=1024,
                    block_k=1024,
                    block_b=local_b,
                    block_q_major_dkv=1024,
                    block_k_major_dkv=1024,
                    block_k_dkv=256,
                    block_q_dkv=1024,
                    block_k_major_dq=512,
                    block_k_dq=256,
                    block_q_dq=1024,
                )
                return pallas_flash_attention(
                    q_local, k_local, v_local,
                    causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
                )
            mesh = get_model_mesh()
            batch_axis = get_batch_axis()

            if mesh is not None:
                spec = P(batch_axis, None, None, None)
                sharded_flash = jax.shard_map(
                    _flash_call,
                    mesh=mesh,
                    in_specs=spec,
                    out_specs=spec,
                    check_vma=False,
                )
                out = sharded_flash(Q_rope, K_rope, V).astype(x.dtype)
            else:
                out = _flash_call(Q_rope, K_rope, V).astype(x.dtype)
        else:
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
        out = make_grad_probe(f"mla_flash_attn_out")(out)
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

        rhs = conv_w.T[:, None, :].astype(x_bc.dtype)
        res_conv = jax.lax.conv_general_dilated(
            lhs=x_bc,
            rhs=rhs,
            window_strides=(1,),
            padding=[(self.cfg.d_conv - 1, 0)],
            feature_group_count=d_inner,
            dimension_numbers=('NHC', 'HIO', 'NHC')
        )
        x_conv = jax.nn.silu(res_conv + conv_b[None, None, :].astype(x_bc.dtype))

        A = -jnp.exp(self.param("A_log", nn.initializers.uniform(scale=1.0), (d_inner,))).astype(x.dtype)
        B = nn.Dense(d_state, use_bias=False, name="B_proj", dtype=jnp.bfloat16)(x_bc)
        C = nn.Dense(d_state, use_bias=False, name="C_proj", dtype=jnp.bfloat16)(x_bc)
        dt = jax.nn.softplus(nn.Dense(d_inner, use_bias=True, name="dt_proj", dtype=jnp.bfloat16)(x_bc))

        dA = jnp.exp(jnp.einsum("bld,d->bld", dt, A))

        chunk_size = min(self.cfg.deltanet_chunk_size, l)
        if l % chunk_size != 0:
            raise ValueError(f"seq_len={l} must be divisible by deltanet_chunk_size={chunk_size}.")
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

        _chunk_step = jax.checkpoint(_chunk_step)

        _, y_chunks = jax.lax.scan(
            _chunk_step, (carry_da_init, carry_h_init), (dA_ch, dt_ch, B_ch, C_ch, x_conv_ch)
        )
        y = jnp.moveaxis(y_chunks, 0, 1).reshape(b, l, d_inner)

        out = y * jax.nn.silu(res)
        return nn.Dense(d, use_bias=False, name="out_proj", dtype=jnp.bfloat16)(out)


# ==========================================
# Gated DeltaNet-2 (GDN-2)
# ==========================================
class GatedDeltaNet2J(nn.Module):
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
            rhs = conv_w.T[:, None, :].astype(u.dtype)
            out = jax.lax.conv_general_dilated(
                lhs=u,
                rhs=rhs,
                window_strides=(1,),
                padding=[(self.cfg.d_conv - 1, 0)],
                feature_group_count=d,
                dimension_numbers=('NHC', 'HIO', 'NHC')
            )
            return out + conv_b[None, None, :].astype(u.dtype)

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

        a_param = self.param("decay_a", nn.initializers.zeros, (n_heads,)).astype(x.dtype)
        f_proj = nn.Dense(d, use_bias=True, name="decay_proj", dtype=jnp.bfloat16)(x).reshape(b, l, n_heads, d_head)
        g = -jnp.exp(a_param)[None, None, :, None] * jax.nn.softplus(f_proj)
        alpha = jnp.exp(g)

        out_gate = nn.Dense(d, use_bias=False, name="out_gate", dtype=jnp.bfloat16)(x)

        e = b_gate * k
        z = w_gate * v
        ea = e * alpha

        chunk_size = min(self.cfg.deltanet_chunk_size, l)
        if l % chunk_size != 0:
            raise ValueError(f"seq_len={l} must be divisible by deltanet_chunk_size={chunk_size}.")
        num_chunks = l // chunk_size

        # ФИКС: M_c = eye*alpha - k⊗ea в общем случае НЕ гарантированно имеет
        # спектральную норму <=1 (b_gate/w_gate -- независимые сигмоиды, а не
        # согласованный erase/write gate классического delta rule), поэтому
        # даже небольшое превышение normы 1 на одном токене после
        # associative_scan-произведения ПОДРЯД chunk_size=256 таких матриц
        # даёт экспоненциальный рост (1.05^256 ~ 1.6e5) вплоть до inf --
        # именно это наблюдалось эмпирически (block=5, layer=16, gdn2, после
        # 190 стабильных шагов, когда веса гейтов сместились в опасную зону).
        # bf16 и fp32 имеют ОДИНАКОВЫЙ диапазон экспоненты (8 бит), так что
        # апкаст сам по себе точку overflow не сдвигает -- нужно ограничивать
        # саму норму матрицы на каждом шаге combine, а не только точность.
        def _combine(state1, state2):
            m1, c1 = state1
            m2, c2 = state2
            m_new = m2 @ m1
            # Ограничиваем Frobenius-норму произведения сверху -- дешёвая
            # аппроксимация spectral clipping, которая не даёт норме расти
            # неограниченно внутри scan, но не трогает "здоровые" M (норма
            # которых и так <=1 почти везде).
            fro_norm = jnp.sqrt(jnp.sum(jnp.square(m_new), axis=(-2, -1), keepdims=True))
            scale = jnp.minimum(1.0, 1.0 / (fro_norm + 1e-6))
            m_new = m_new * scale
            c_new = m2 @ c1 + c2
            c_new = jnp.nan_to_num(c_new, nan=0.0, posinf=1e4, neginf=-1e4)
            return m_new, c_new

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
            # ФИКС: последняя линия обороны -- если несмотря на clipping в
            # _combine что-то всё же дало inf/nan (например, carry_M/carry_S,
            # пришедшие с предыдущего чанка), не даём этому уйти дальше по
            # scan через новый carry.
            global_S = jnp.nan_to_num(global_S, nan=0.0, posinf=1e4, neginf=-1e4)

            out_c = jnp.einsum("bchij,bchi->bchj", global_S, q_c)
            new_carry = (global_M[:, -1], global_S[:, -1])
            return new_carry, out_c

        _chunk_step = jax.checkpoint(_chunk_step)

        _, out_chunks = jax.lax.scan(
            _chunk_step, (eye_bh, zero_bh), (k_ch, ea_ch, z_ch, alpha_ch, q_ch)
        )
        out = jnp.moveaxis(out_chunks, 0, 1).reshape(b, l, d)

        out = nn.RMSNorm(epsilon=1e-6, name="out_norm")(out).astype(x.dtype)
        return nn.Dense(d, use_bias=False, name="out_proj", dtype=jnp.bfloat16)(out * jax.nn.silu(out_gate))


# ==========================================
# MoE
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
    """DENSE MoE (тестовый режим): все E экспертов считают все токены,
    без routing/capacity/sort/gather-scatter. Используется для диагностики
    того, был ли TPU compute bottleneck именно в routing-механике старого
    top-k + capacity-buffer варианта (см. историю: 3.3с/микрошаг при MFU ~2%)."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rngs=None):
        b, l, d = x.shape
        flat_x = x.reshape(-1, d)
        E = self.cfg.num_experts

        router_logits = nn.Dense(E, use_bias=False, name="router", dtype=jnp.bfloat16)(flat_x)
        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape, dtype=router_logits.dtype
            )
        gate = jax.nn.softmax(router_logits, axis=-1)  # (tokens, E)

        mean_probs = jnp.mean(gate, axis=0)
        self.sow("losses", "aux_loss", E * jnp.sum(mean_probs * mean_probs))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(
            jax.scipy.special.logsumexp(router_logits, axis=-1))))
        self.sow("losses", "expert_utilization", mean_probs)

        # ФИКС: in_axes=(None, None) значит НИ ОДИН аргумент не несёт ось
        # экспертов -- в отличие от старого in_axes=(0, None), где
        # expert_inputs уже имел ось E_routed и axis_size выводился сам.
        # Здесь axis_size нужно указать явно, иначе падает при компиляции.
        run_experts = nn.vmap(
            ExpertPack,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(None, None),
            out_axes=0,
            axis_size=E,
        )(cfg=self.cfg, name="experts_block")
        all_outputs = run_experts(flat_x, deterministic)  # (E, tokens, d)

        weighted = jnp.einsum("te,etd->td", gate, all_outputs)
        return weighted.reshape(b, l, d)
# ==========================================
# Intra-block attention (lightweight mixing)
# ==========================================
class IntraBlockAttention(nn.Module):
    """Mix sources via lightweight attention for quality.
    sources: list of (B, L, D) tensors
    block_input: (B, L, D) — the original block input, used as query
    Returns: mixed (B, L, D)
    """
    cfg: ModelConfig

    @nn.compact
    def __call__(self, sources, block_input):
        # sources: list of (B, L, D)
        n_sources = len(sources)
        if n_sources == 1:
            return sources[0]

        b, l, d = block_input.shape

        # Query from block_input
        q = nn.Dense(self.cfg.d_latent, use_bias=False, name="q_proj", dtype=jnp.bfloat16)(block_input)

        # Key for each source (separate projection per source for quality)
        k_list = []
        for i, src in enumerate(sources):
            k_i = nn.Dense(self.cfg.d_latent, use_bias=False, name=f"k_proj_{i}", dtype=jnp.bfloat16)(src)
            k_list.append(k_i)

        # Stack keys: (N, B, L, d_latent)
        k_stack = jnp.stack(k_list, axis=0)

        # Scores: (N, B, L)
        scores = jnp.einsum("bld,nbld->nbl", q, k_stack) / jnp.sqrt(jnp.array(self.cfg.d_latent, dtype=q.dtype))

        # Softmax over sources
        weights = jax.nn.softmax(scores, axis=0)  # (N, B, L)

        # Stack sources: (N, B, L, D)
        src_stack = jnp.stack(sources, axis=0)

        # Weighted mix
        mixed = jnp.einsum("nbl,nbld->bld", weights, src_stack)
        return mixed.astype(block_input.dtype)

class HybridDARAttention(nn.Module):
    """DAR over variable source count. Shared k-projection via flattening
    so Flax params don't depend on n_sources."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, current_x, all_sources):
        n = len(all_sources)
        if n == 0:
            return jnp.zeros_like(current_x)
        
        b, l, d = current_x.shape
        stack = jnp.stack(all_sources, axis=0)  # (n, B, L, D)
        
        # Shared projection: params (D, d_latet), shape-agnostic to n
        flat = stack.reshape(n * b * l, d)
        k_flat = nn.Dense(
            self.cfg.d_latent, use_bias=False, name="k_proj", dtype=jnp.bfloat16
        )(flat)
        k = k_flat.reshape(n, b, l, self.cfg.d_latent)
        
        q = nn.Dense(
            self.cfg.d_latent, use_bias=False, name="q_proj", dtype=jnp.bfloat16
        )(current_x)
        
        scores = jnp.einsum(
            "bld,nbld->nbl", q, k
        ) / jnp.sqrt(jnp.array(self.cfg.d_latent, dtype=q.dtype))
        
        weights = jax.nn.softmax(scores, axis=0)
        retrieved = jnp.einsum("nbl,nbld->bld", weights, stack)
        return retrieved.astype(current_x.dtype)
# ==========================================
# Specialized Sublayer (dispatches to GDN-2 / Mamba2 / MLA)
# ==========================================
class SpecializedSublayer(nn.Module):
    cfg: ModelConfig
    layer_type: str  # "gdn2", "mamba2", "mla"

    @nn.compact
    def __call__(self, x, causal_mask=None, cos=None, sin=None, deterministic=True, rngs=None):
        if self.layer_type == "gdn2":
            return GatedDeltaNet2J(cfg=self.cfg, name="gdn2")(x)
        elif self.layer_type == "mamba2":
            return Mamba2J(cfg=self.cfg, name="mamba2")(x)
        elif self.layer_type == "mla":
            return MLAJ(cfg=self.cfg, name="mla")(
                x, causal_mask, cos, sin, deterministic=deterministic, rngs=rngs
            )
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")


# ==========================================
# Block-level DAR Layer
# ==========================================
class BlockDARLayer(nn.Module):
    cfg: ModelConfig
    layer_type: str
    layer_idx: int

    @nn.compact
    def __call__(self, current_x, x_input, block_input, local_deltas, history_blocks,
                 causal_mask, cos, sin, deterministic=True, rngs=None):
        b, l, d = current_x.shape
        
        # --- Hybrid DAR: retrieve from [x_input, Δ..., δ...] ---
        dar_sources = []
        if history_blocks.shape[0] > 0:
            dar_sources.extend([history_blocks[j] for j in range(history_blocks.shape[0])])
        dar_sources.extend(local_deltas)
        
        retrieved = HybridDARAttention(cfg=self.cfg, name="dar")(
            current_x, dar_sources
        )
        retrieved = make_grad_probe(f"block{self.layer_idx}_dar_out")(retrieved)
        current_x = current_x + retrieved
        
        # --- Intra-block mixing (ваш оригинальный механизм) ---
        intra_sources = [block_input] + list(local_deltas)
        mixed = IntraBlockAttention(cfg=self.cfg, name="intra")(
            intra_sources, block_input
        )
        mixed = make_grad_probe(f"block{self.layer_idx}_intra_out")(mixed)
        
        # --- Sublayer ---
        delta = SpecializedSublayer(
            cfg=self.cfg, layer_type=self.layer_type, name="sublayer"
        )(mixed, causal_mask=causal_mask, cos=cos, sin=sin,
          deterministic=deterministic, rngs=rngs)
        
        return delta

# ==========================================
# Block-level DAR Block (3 consecutive layers)
# ==========================================
class BlockDAR(nn.Module):
    cfg: ModelConfig
    block_idx: int
    layer_idx_start: int

    @nn.compact
    def __call__(self, current_x, x_input, history_blocks, causal_mask, cos, sin,
                 deterministic=True, rngs=None):
        b, l, d = current_x.shape
        block_input = current_x  # snapshot входа блока
        
        local_deltas = []
        
        for i in range(self.cfg.layers_per_block):
            layer_idx = self.layer_idx_start + i
            layer_type = self.cfg.layer_types[layer_idx]
            
            delta = BlockDARLayer(
                cfg=self.cfg,
                layer_type=layer_type,
                layer_idx=layer_idx,
                name=f"layer_{layer_idx}"
            )(current_x, x_input, block_input, local_deltas, history_blocks,
              causal_mask, cos, sin, deterministic, rngs)

            # ДИАГНОСТИКА: ловим ПЕРВЫЙ non-finite delta по forward-активациям,
            # до того как residual/DAR размажет его по всему графу (что и
            # даёт симметричную картину "все группы параметров non-finite"
            # в backward-диагностике). layer_type печатается статически
            # (python f-string), non-finite флаг -- динамически.
            delta_finite = jnp.all(jnp.isfinite(delta))
            jax.lax.cond(
                jnp.logical_not(delta_finite),
                lambda: jax.debug.print(
                    "[FWD-DIAG] ⚠️ non-finite delta: block={b} layer={l} type=" + layer_type,
                    b=self.block_idx, l=layer_idx,
                ),
                lambda: None,
            )

            local_deltas.append(delta)
            current_x = current_x + delta  # residual
        
        # Агрегируем и сбрасываем локальные δ
        block_delta = sum(local_deltas)
        new_history = jnp.concatenate(
            [history_blocks, block_delta[None, ...]], axis=0
        )
        
        # MoE после блока (DAR уже был на последнем слое)
        norm_2 = nn.RMSNorm(epsilon=1e-6, name="norm_2")(current_x).astype(current_x.dtype)
        moe_out = MoEJ(cfg=self.cfg, name="moe")(norm_2, deterministic=deterministic, rngs=rngs)

        # ДИАГНОСТИКА: то же самое для MoE-выхода блока.
        moe_finite = jnp.all(jnp.isfinite(moe_out))
        jax.lax.cond(
            jnp.logical_not(moe_finite),
            lambda: jax.debug.print("[FWD-DIAG] ⚠️ non-finite moe_out: block={b}", b=self.block_idx),
            lambda: None,
        )

        output = current_x + moe_out
        
        return output, new_history


# ==========================================
# Full Model
# ==========================================
class FullHybridMoEModel(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True, rngs=None, return_hidden: bool = False):
        b, l = input_ids.shape
        embed_layer = nn.Embed(
            num_embeddings=self.cfg.vocab_size,
            features=self.cfg.d_model,
            name="embed",
            dtype=jnp.bfloat16,
        )
        x_input = embed_layer(input_ids)
        x = x_input
        causal_mask = jnp.tril(jnp.ones((l, l))).astype(jnp.bool_)[None, None, :, :]

        d_head = self.cfg.d_model // self.cfg.n_heads
        cos, sin = RoPEEmbedding(dim=d_head)(l)

        # History starts empty: (0, B, L, D)
        history_blocks = jnp.zeros((0, b, l, self.cfg.d_model), dtype=x.dtype)

        num_blocks = self.cfg.num_layers // self.cfg.layers_per_block


        RematBlock = BlockDAR
        
        for block_idx in range(num_blocks):
            layer_idx_start = block_idx * self.cfg.layers_per_block
            x, history_blocks = RematBlock(
                cfg=self.cfg, block_idx=block_idx, layer_idx_start=layer_idx_start,
                name=f"block_{block_idx}"
            )(x, x_input, history_blocks, causal_mask, cos, sin, deterministic, rngs)

        final = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x).astype(x.dtype)

        if return_hidden:
            return final

        if self.cfg.tie_embeddings:
            logits = embed_layer.attend(final)
        else:
            logits = nn.Dense(
                self.cfg.vocab_size,
                use_bias=False,
                name="lm_head",
                dtype=jnp.bfloat16,
            )(final)
        return logits
