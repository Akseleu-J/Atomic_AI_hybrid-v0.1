import math
import os
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from typing import List, Tuple
from jax.sharding import PartitionSpec as P

# ==========================================================================
# ФИКС (интеграция fused-Pallas backward B1-B6, atomic_ops/INTEGRATION_NOTES.md
# п. "Что стоит сделать" #3): раньше model.py импортировал ТОЛЬКО
# kernel_trainable.gdn2_pallas_forward_trainable ("читерский" backward --
# jax.vjp на чистом JAX-референсе). kernel_trainable_B6.py (честный
# fused-Pallas forward+backward B1->B2->B3->B4->B5, с финальной
# санитизацией градиентов на границе custom_vjp) провалидирован сравнением
# градиентов против kernel_trainable.py (atomic_ops/gdn2_backward_compare.py,
# compare_suite() по нескольким seed/размерам -- все finite, rel_diff < 5%)
# и подтверждён на реальном TPU. Переключаем основной импорт на него --
# это теперь единственный путь, которым обучается GDN-2 (и forward, и
# backward идут через кернелы Pallas A/B/C/D + B1-B5, никакого jax.vjp на
# JAX-референсе в горячем пути обучения больше нет).
#
# kernel_trainable.py (jax.vjp-на-референсе) остаётся в репозитории как
# cross-check / fallback -- см. его собственный docstring -- на случай, если
# понадобится снова сравнить градиенты при подозрении на регрессию.
# ==========================================================================
from atomic_ops.kernel_trainable_B6 import gdn2_pallas_forward_trainable
from atomic_ops.kernel_a_scores import BT as GDN2_PALLAS_BT
from atomic_ops.moe_gmm import GmmMoEJ
print("[MODEL] ⚙️ GDN-2: используется честный fused-Pallas forward+backward "
      "(kernel_d_pipeline.py + kernel_bwd_b1..b5, склеены в "
      "kernel_trainable_B6.py). jax.vjp-на-референсе backward "
      "(kernel_trainable.py) больше не используется в горячем пути "
      "обучения.")

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
# ДИАГНОСТИКА (сквозная, переиспользует тот же флаг, что kernel_d_pipeline.py
# использует для GDN-2 -- GDN2_FWD_DIAG=1, по умолчанию OFF, нулевой оверхед
# в обычном обучении). Добавлена сюда (module-level, model.py) для
# использования внутри Mamba2J -- см. её докстринг ниже: диагностика нужна,
# чтобы увидеть, реально ли новый RMSNorm перед out_proj сбивает амплитуду
# SSM-выхода, а не просто гадать по [RESID-DIAG] на несколько слоёв позже.
# ==========================================================================
_GDN2_FWD_DIAG = os.environ.get("GDN2_FWD_DIAG", "0") == "1"


def _diag_maxabs(tag: str, x):
    """No-op (возвращает x без изменений) если GDN2_FWD_DIAG=0. Иначе печатает
    finite/non-finite статус и max|abs| конечной части -- чисто диагностика,
    значение не меняет ни в каком режиме."""
    if not _GDN2_FWD_DIAG:
        return x

    finite_mask = jnp.isfinite(x)
    all_finite = jnp.all(finite_mask)
    safe_x = jnp.where(finite_mask, x, 0.0)
    max_abs = jnp.max(jnp.abs(safe_x))

    jax.debug.print(
        "[MAMBA2-FWD-DIAG] " + tag + ": all_finite={f}  max_abs={m:.3e}",
        f=all_finite, m=max_abs,
    )
    return x


# ==========================================================================
# ДИАГНОСТИКА #2 (этот пасс -- точечная локализация RESID-DIAG на
# layer=22/block=7/mamba2 при round_robin dataloader): [RESID-DIAG] в
# BlockDAR печатает max|abs| ПОСЛЕ current_x+delta, но не говорит, откуда
# пришла величина -- current_x УЖЕ был близко к порогу ДО этого слоя
# (значит проблема в накоплении по предыдущим слоям/блокам), или delta
# именно ЭТОГО sublayer'а (mamba2 на layer=22) добавила скачок сама по
# себе (значит проблема локальна для этого слоя). Печатает ОБА числа
# отдельно, только когда суммарный результат уже приближается к клипу --
# порог 5e2 выбран НИЖЕ порога самого RESID-DIAG (1e3), чтобы эта
# диагностика срабатывала РАНЬШЕ и давала предупреждение, а не только
# постфактум объяснение уже случившегося события. Управляется тем же
# GDN2_FWD_DIAG=1 флагом -- по умолчанию OFF, нулевой оверхед.
# ==========================================================================
def _diag_resid_breakdown(tag: str, current_x_before, delta):
    if not _GDN2_FWD_DIAG:
        return

    cx_before_abs = jnp.max(jnp.abs(jnp.nan_to_num(current_x_before, nan=0.0, posinf=0.0, neginf=0.0)))
    delta_abs = jnp.max(jnp.abs(jnp.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)))

    def _report():
        jax.debug.print(
            "[RESID-BREAKDOWN-DIAG] " + tag +
            ": current_x_before={cx:.3e}  delta={d:.3e}  "
            "(если current_x_before уже большой -- накопление ИЗ ПРЕДЫДУЩИХ слоёв; "
            "если delta большая, а current_x_before маленький -- источник ЭТОТ слой)",
            cx=cx_before_abs, d=delta_abs,
        )

    jax.lax.cond(
        jnp.maximum(cx_before_abs, delta_abs) > 5e2,
        _report,
        lambda: None,
    )


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


def make_grad_sanitizer(tag: str, clip_val: float = 1e3):
    """Как make_grad_probe, но не только печатает, а АКТИВНО чинит non-finite
    градиент (nan_to_num + клип) прежде чем отдать его дальше по backward
    графу. Используется в узлах со своим сложным custom VJP (pallas
    flash-attention) или длинной scan-рекуррентностью (Mamba2), где
    санитизация только forward-входов (как для GDN-2/Mamba2 сделано выше)
    не гарантирует конечность именно ГРАДИЕНТА, вычисляемого внутри."""
    @jax.custom_vjp
    def _sanitizer(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        finite = jnp.all(jnp.isfinite(g))
        jax.lax.cond(
            jnp.logical_not(finite),
            lambda: jax.debug.print("[BWD-FIX] 🩹 non-finite градиент в узле {t} -- санитизирован", t=tag),
            lambda: None,
        )
        g_safe = jnp.nan_to_num(jnp.clip(g, -clip_val, clip_val), nan=0.0, posinf=clip_val, neginf=-clip_val)
        return (g_safe,)

    _sanitizer.defvjp(_fwd, _bwd)
    return _sanitizer


@struct.dataclass
class ModelConfig:
    d_model: int = 768
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

        # ФИКС: единственный модуль без входной санитизации перед
        # flash-attention (GDN-2 нормирует q/k, Mamba2 клипает B/C/dt) --
        # подтверждено логом, где mla_flash_attn_out регулярно требовал
        # спасения градиента. Клип входа тем же способом (±1e3), что и везде.
        def _sanitize(t):
            return jnp.nan_to_num(jnp.clip(t, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

        Q_rope, K_rope, V = map(_sanitize, (Q_rope, K_rope, V))

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

        # ФИКС (по аналогии с out_norm в GatedDeltaNet2J / ssm_out_norm в
        # Mamba2J -- см. их докстринги): MLA была единственным из трёх типов
        # саблеера БЕЗ нормировки выхода перед финальной проекцией. До этого
        # момента make_grad_sanitizer чинил только градиент на backward, но
        # ничего не ограничивал в forward -- амплитуда выхода attention могла
        # свободно плавать перед W_o и уходить в history_blocks/local_deltas
        # ненормированной, ровно как это было прослежено для Mamba2 (RESID-DIAG,
        # рост current_x после mla-слоёв в block0/3/6). RMSNorm здесь -- тот же
        # приём, применённый к последнему из трёх недостающих мест.
        out = nn.RMSNorm(epsilon=1e-6, name="attn_out_norm")(out).astype(x.dtype)

        out = make_grad_sanitizer("mla_flash_attn_out")(out)
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
 
        # ФИКС: тот же inf*0=nan риск, что и в GDN-2 decay -- если A_log
        # уйдёт в большое положительное значение, exp(A_log) переполняется в
        # inf, A=-inf; если dt в какой-то позиции округлится до 0 в bf16,
        # dt*A = 0*(-inf) = nan. Клипаем A_log перед exp и санитизируем
        # итоговый показатель степени перед exp.
        A_log_safe = jnp.clip(self.param("A_log", nn.initializers.uniform(scale=1.0), (d_inner,)), -20.0, 20.0)
        # ФИКС (root cause найденной ошибки lax.concatenate bf16 vs f32):
        # раньше было `.astype(x.dtype)` -- но x (аргумент __call__, после
        # nn.RMSNorm без dtype=) на практике может оказаться float32
        # (RMSNorm.scale по умолчанию float32, bf16*float32 -> float32 по
        # правилам promotion), тогда как dt/B/C/x_conv ВСЕ идут через
        # nn.Dense(..., dtype=bfloat16) и гарантированно bf16. Из-за этого A
        # (и все производные dA/da_c) становились float32, а C_input_c
        # оставался bf16 -- смешение dtype внутри одной пары элементов
        # jax.lax.associative_scan, что и роняло lax.concatenate внутри
        # scan'а. Берём dtype у x_bc (выход in_proj, гарантированно bf16),
        # а не у сырого x.
        A = -jnp.exp(A_log_safe).astype(x_bc.dtype)
        B = nn.Dense(d_state, use_bias=False, name="B_proj", dtype=jnp.bfloat16)(x_bc)
        C = nn.Dense(d_state, use_bias=False, name="C_proj", dtype=jnp.bfloat16)(x_bc)
        dt = jax.nn.softplus(nn.Dense(d_inner, use_bias=True, name="dt_proj", dtype=jnp.bfloat16)(x_bc))
 
        # ФИКС #1: dt-коридор. Если dt микроскопический (округляется к 0 в
        # bf16), dA = exp(dt*A) -> 1, decay фактически отсутствует, и state
        # копит бесконечность через associative_scan. Если dt гигантский --
        # SSM делает неадекватно большой шаг за один токен. Ограничиваем
        # разумным коридором [1e-2, 1.0] до того, как dt участвует в
        # dA_exponent.
        dt = jnp.clip(dt, jnp.array(1e-2, dtype=dt.dtype), jnp.array(1.0, dtype=dt.dtype))
 
        dA_exponent = jnp.nan_to_num(jnp.einsum("bld,d->bld", dt, A), nan=0.0, posinf=0.0, neginf=-20.0)
        dA = jnp.exp(dA_exponent)
        # ФИКС #2 (главный тормоз): даже при "плохих" весах decay не может
        # приближаться к 1 -- иначе на chunk_size=256 произведение decay по
        # чанку почти не затухает и state растёт неограниченно.
        # 0.99**256 ~= 0.08 -- гарантирует, что state забывает старое внутри
        # одного чанка независимо от того, что выучили A_log/dt_proj.
        dA = jnp.clip(dA, jnp.array(0.0, dtype=dA.dtype), jnp.array(0.99, dtype=dA.dtype))
 
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
 
        # ФИКС: та же санитизация-рубеж, что и в GDN-2 -- независимо от
        # источника nan/inf выше по графу (A_log/exp, dt/softplus, B/C
        # проекции), гарантируем, что в рекуррентный scan всегда приходят
        # конечные значения.
        def _sanitize(t):
            # ФИКС #4: явный bound той же разрядности, что и t. Голый
            # Python-scalar в jnp.clip обычно ок, но внутри associative_scan
            # смешение carry/init разной разрядности (bf16 vs fp32-scalar
            # default) может дать TypeError на некоторых backend'ах --
            # закрываем это явным bound с astype под dtype самого t.
            bound = jnp.array(1e3, dtype=t.dtype)
            return jnp.nan_to_num(jnp.clip(t, -bound, bound), nan=0.0, posinf=1e3, neginf=-1e3)
 
        dA, dt, B, C, x_conv = map(_sanitize, (dA, dt, B, C, x_conv))
 
        dA_ch = _to_chunks(dA)
        dt_ch = _to_chunks(dt)
        B_ch = _to_chunks(B)
        C_ch = _to_chunks(C)
        x_conv_ch = _to_chunks(x_conv)
 
        carry_da_init = jnp.ones((b, d_inner), dtype=x_bc.dtype)
        carry_h_init = jnp.zeros((b, d_inner, d_state), dtype=x_bc.dtype)
 
        def _chunk_step(carry, chunk_inputs):
            carry_da, carry_h = carry
            da_c, dt_c, B_c, C_c, xconv_c = chunk_inputs
 
            dB_c = jnp.einsum("bcd,bcs->bcds", dt_c, B_c)
            # ФИКС #3: dB_c -- это input gate, записывающий сигнал в state за
            # один шаг. Диагностика на реальном прогоне ловила dB_c до ~2384
            # -- клип здесь бьёт именно по источнику того выброса, до того
            # как он попадёт в C_input_c/associative_scan.
            dB_c = jnp.clip(dB_c, jnp.array(-1e3, dtype=dB_c.dtype), jnp.array(1e3, dtype=dB_c.dtype))
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

        # ДИАГНОСТИКА (GDN2_FWD_DIAG=1): величина SSM-выхода ДО нормировки --
        # чтобы видеть, насколько велика амплитуда, которую ниже гасит
        # RMSNorm, вместо того чтобы узнавать о ней только по [RESID-DIAG]
        # в BlockDAR несколько слоёв спустя.
        y = _diag_maxabs("mamba2_ssm_out_pre_norm", y)

        # ФИКС (закрывает дыру, симметричную out_norm в GatedDeltaNet2J):
        # раньше выход SSM-скана шёл прямо в out_proj через
        # y * silu(res) -> clip(±1e4) -> out_proj, БЕЗ какой-либо нормировки --
        # единственный из трёх типов саблеера без неё на момент инцидентов
        # RESID-DIAG на layer=13/14 (block4, mamba2). Клип ограничивает
        # диапазон, но не масштаб/распределение -- значение могло свободно
        # плавать от ~1 до 1e4 и оттуда накапливаться в local_deltas/
        # history_blocks, откуда его читают все последующие блоки через
        # HybridDARAttention/IntraBlockAttention. RMSNorm здесь -- ровно тот
        # же приём, что уже стоит на выходе GDN-2 (out_norm), применённый
        # ДО гейта silu(res), а не после -- чтобы сам гейт не мог снова
        # растащить масштаб уже отнормированного значения.
        y = nn.RMSNorm(epsilon=1e-6, name="ssm_out_norm")(y).astype(x_bc.dtype)

        y = _diag_maxabs("mamba2_ssm_out_post_norm", y)

        out = y * jax.nn.silu(res)
        # ФИКС #5: последний рубеж перед residual stream -- даже если весь
        # scan остался конечным, финальное умножение y*silu(res) может дать
        # усиление (оба множителя уже прошли клип по отдельности, но их
        # произведение -- нет). Не пускаем большие значения в out_proj/residual.
        out = jnp.clip(out, jnp.array(-3e2, dtype=out.dtype), jnp.array(3e2, dtype=out.dtype))
        return nn.Dense(d, use_bias=False, name="out_proj", dtype=jnp.bfloat16)(out)
 
 


# ==========================================
# Gated DeltaNet-2 (GDN-2)
# ==========================================
def _gdn2_recurrence_impl(k, e, z, alpha, q, dtype):
    """Reference-only fallback, not part of the active forward path -- see
    original file for full explanation. Left unchanged."""
    b, l, n_heads, d_head = k.shape

    def _to_time_major(t):
        return jnp.moveaxis(t, 1, 0)

    k_t, e_t, z_t, alpha_t, q_t = map(_to_time_major, (k, e, z, alpha, q))

    S0 = jnp.zeros((b, n_heads, d_head, d_head), dtype=jnp.float32)

    def _step(S, inputs):
        k_i, e_i, z_i, alpha_i, q_i = inputs
        k_f = k_i.astype(jnp.float32)
        e_f = e_i.astype(jnp.float32)
        z_f = z_i.astype(jnp.float32)
        alpha_f = alpha_i.astype(jnp.float32)
        q_f = q_i.astype(jnp.float32)

        S_bar = alpha_f[..., :, None] * S
        r = jnp.einsum('bhkv,bhk->bhv', S_bar, e_f)
        S_new = S_bar + jnp.einsum('bhk,bhv->bhkv', k_f, z_f - r)
        o = jnp.einsum('bhkv,bhk->bhv', S_new, q_f)
        return S_new, o.astype(dtype)

    _step = jax.checkpoint(_step)
    _, out_t = jax.lax.scan(_step, S0, (k_t, e_t, z_t, alpha_t, q_t))

    return jnp.moveaxis(out_t, 0, 1)


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
        v = jnp.clip(v, -50.0, 50.0)

        # ФИКС (найдено через probe sublayer_out_layer{N}/intra_out: forward
        # ПОЛНОСТЬЮ конечен, но backward регулярно даёт NaN между этими
        # двумя точками -- т.е. внутри GatedDeltaNet2J, между выходом B1-B6
        # chain (уже санитизирован на своей границе в kernel_trainable_B6.py)
        # и входом в conv/q_proj/k_proj). Единственный кандидат на этом
        # участке -- нормализация q/k. jnp.linalg.norm(x) САМ ПО СЕБЕ имеет
        # NaN-градиент РОВНО в x=0 (d‖x‖/dx = x/‖x‖, неопределённость 0/0) --
        # прибавление +eps в форвард-делении (q/(‖q‖+eps)) защищает ТОЛЬКО
        # forward-деление, backward самого jnp.linalg.norm() эту особенность
        # не лечит вообще. Если silu(conv(...)) даёт точный или почти точный
        # нулевой вектор для каких-то токенов/каналов -- невидимо в forward
        # (там всё гладко благодаря +eps), но backward взрывается в NaN.
        # Прямая проверка (jax.vjp на x=jnp.zeros(4)): текущая формула даёт
        # grad=[nan,nan,nan,nan]; rsqrt(sum(x^2)+eps^2) даёт конечный (хоть
        # и большой, ~1e6 у сингулярности) градиент везде, включая x=0.
        # Оборачиваем дополнительно в grad_sanitizer (тот же приём, что для
        # mla_flash_attn_out) -- большой-но-конечный градиент у сингулярности
        # всё ещё стоит подрезать, чтобы не разгонять его дальше по conv/Dense.
        def _safe_normalize(t):
            return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps ** 2)

        q = make_grad_sanitizer("gdn2_q_normalize")(_safe_normalize(q))
        k = make_grad_sanitizer("gdn2_k_normalize")(_safe_normalize(k))

        b_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="erase_gate", dtype=jnp.bfloat16)(x)).reshape(b, l, n_heads, d_head)
        w_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="write_gate", dtype=jnp.bfloat16)(x)).reshape(b, l, n_heads, d_head)

        a_param = self.param("decay_a", nn.initializers.zeros, (n_heads,)).astype(jnp.float32)
        f_proj = nn.Dense(d, use_bias=True, name="decay_proj", dtype=jnp.bfloat16)(x).reshape(b, l, n_heads, d_head)
        a_param_safe = jnp.clip(a_param, -20.0, 20.0)
        g = -jnp.exp(a_param_safe)[None, None, :, None] * jax.nn.softplus(f_proj.astype(jnp.float32))
        g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-20.0)
        alpha = jnp.exp(g)

        out_gate = nn.Dense(d, use_bias=False, name="out_gate", dtype=jnp.bfloat16)(x)
        # ФИКС (закрывает дыру, названную прямо в докстринге BlockDARLayer
        # ниже по файлу -- "GatedDeltaNet2J: out_gate ничем не клипируется до
        # silu(out_gate)*out -> out_proj"): out (после out_norm) уже
        # нормирован, но out_gate -- сырой выход Dense, ничем не ограниченный.
        # jax.nn.silu растёт линейно на больших положительных x (не насыщается,
        # в отличие от sigmoid), так что при разросшихся весах out_gate
        # (Muon-таргет без weight decay в среднем растёт быстрее AdamW/Lion --
        # см. optimizer.py) итоговый множитель silu(out_gate) может быть
        # произвольно большим и растащить уже отнормированный out обратно в
        # residual stream -- ровно тот путь усиления, который приводил к
        # RESID-DIAG выбросам после gdn2-слоёв. Клип на уровне входа в out_gate
        # тем же способом (±1e2), что и клип параметров в train.py после
        # apply_updates -- дёшево, не меняет остальную математику.
        out_gate = jnp.clip(out_gate, -1e2, 1e2)

        def _sanitize(t):
            return jnp.nan_to_num(jnp.clip(t, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

        q, k, v, w_gate, b_gate, g = map(_sanitize, (q, k, v, w_gate, b_gate, g))

        if l % GDN2_PALLAS_BT != 0:
            raise ValueError(
                f"seq_len={l} must be divisible by the Pallas GDN-2 chunk size "
                f"({GDN2_PALLAS_BT}); this is currently a separate constant from "
                f"cfg.deltanet_chunk_size, see kernel_a_scores.py."
            )

        mesh = get_model_mesh()
        batch_axis = get_batch_axis()

        _gdn2_fixed = partial(gdn2_pallas_forward_trainable, scale=1.0, h0=None)

        if mesh is not None:
            in_spec = P(batch_axis, None, None, None)
            out_spec = (
                P(batch_axis, None, None, None),
                P(batch_axis, None, None, None),
            )
            sharded_gdn2 = jax.shard_map(
                _gdn2_fixed,
                mesh=mesh,
                in_specs=(in_spec, in_spec, in_spec, in_spec, in_spec, in_spec),
                out_specs=out_spec,
                check_vma=False,
            )
            out, _h_final = sharded_gdn2(q, k, v, w_gate, b_gate, g)
        else:
            out, _h_final = _gdn2_fixed(q, k, v, w_gate, b_gate, g)

        out = out.reshape(b, l, d)

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
        gate = jax.nn.softmax(router_logits, axis=-1)

        mean_probs = jnp.mean(gate, axis=0)
        self.sow("losses", "aux_loss", E * jnp.sum(mean_probs * mean_probs))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(
            jax.scipy.special.logsumexp(router_logits, axis=-1))))
        self.sow("losses", "expert_utilization", mean_probs)

        run_experts = nn.vmap(
            ExpertPack,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(None, None),
            out_axes=0,
            axis_size=E,
        )(cfg=self.cfg, name="experts_block")
        all_outputs = run_experts(flat_x, deterministic)

        weighted = jnp.einsum("te,etd->td", gate, all_outputs)
        return weighted.reshape(b, l, d)


class IntraBlockAttention(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, sources, block_input):
        n_sources = len(sources)
        if n_sources == 1:
            return sources[0]

        b, l, d = block_input.shape

        q = nn.Dense(self.cfg.d_latent, use_bias=False, name="q_proj", dtype=jnp.bfloat16)(block_input)

        k_list = []
        for i, src in enumerate(sources):
            k_i = nn.Dense(self.cfg.d_latent, use_bias=False, name=f"k_proj_{i}", dtype=jnp.bfloat16)(src)
            k_list.append(k_i)

        k_stack = jnp.stack(k_list, axis=0)

        scores = jnp.einsum("bld,nbld->nbl", q, k_stack) / jnp.sqrt(jnp.array(self.cfg.d_latent, dtype=q.dtype))

        weights = jax.nn.softmax(scores, axis=0)

        src_stack = jnp.stack(sources, axis=0)

        mixed = jnp.einsum("nbl,nbld->bld", weights, src_stack)
        return mixed.astype(block_input.dtype)


class HybridDARAttention(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, current_x, all_sources):
        n = len(all_sources)
        if n == 0:
            return jnp.zeros_like(current_x)

        b, l, d = current_x.shape
        stack = jnp.stack(all_sources, axis=0)

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


class SpecializedSublayer(nn.Module):
    cfg: ModelConfig
    layer_type: str

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


class BlockDARLayer(nn.Module):
    cfg: ModelConfig
    layer_type: str
    layer_idx: int

    @nn.compact
    def __call__(self, current_x, x_input, block_input, local_deltas, history_blocks,
                 causal_mask, cos, sin, deterministic=True, rngs=None):
        b, l, d = current_x.shape

        dar_sources = []
        if history_blocks.shape[0] > 0:
            dar_sources.extend([history_blocks[j] for j in range(history_blocks.shape[0])])
        dar_sources.extend(local_deltas)

        retrieved = HybridDARAttention(cfg=self.cfg, name="dar")(
            current_x, dar_sources
        )
        retrieved = make_grad_probe(f"block{self.layer_idx}_dar_out")(retrieved)
        current_x = current_x + retrieved

        intra_sources = [block_input] + list(local_deltas)
        mixed = IntraBlockAttention(cfg=self.cfg, name="intra")(
            intra_sources, block_input
        )
        mixed = make_grad_probe(f"block{self.layer_idx}_intra_out")(mixed)

        # ФИКС: нормализация перед рекуррентным/внимательным саблеером
        # Убирает разогретый residual (1032+) на вход в Mamba2/GDN2/MLA
        mixed = nn.RMSNorm(epsilon=1e-6, name="pre_sublayer_norm")(mixed)

        delta = SpecializedSublayer(
            cfg=self.cfg, layer_type=self.layer_type, name="sublayer"
        )(mixed, causal_mask=causal_mask, cos=cos, sin=sin,
          deterministic=deterministic, rngs=rngs)

        # ФИКС (найдено по инциденту, где sublayer_out_layer16_gdn2 САМ был
        # non-finite -- т.е. входящий градиент уже испорчен ДО backward
        # самого GDN-2/Mamba2/MLA, в отличие от предыдущего инцидента, где
        # ломалась именно нормализация q/k ВНУТРИ sublayer'а). delta имеет
        # широкий fan-out: current_x (этот блок), local_deltas ->
        # IntraBlockAttention для СЛЕДУЮЩИХ слоёв ЭТОГО блока, и, самое
        # далекобойное, history_blocks -> HybridDARAttention ВСЕХ будущих
        # блоков. Градиент от всех этих потребителей суммируется в одну
        # точку (dL/ddelta) -- если хотя бы один из многих путей даёт NaN,
        # сумма NaN. make_grad_sanitizer здесь чистит именно эту
        # аккумулированную точку ДО того, как она уходит в backward самого
        # sublayer'а (аналитический B1-B6 chain для gdn2 -- наиболее хрупкий
        # участок, т.к. это ручной VJP, не автодифф). Ставится МЕЖДУ
        # SpecializedSublayer и диагностическим probe ниже -- probe
        # по-прежнему видит сырую (незачищенную) картину для диагностики,
        # сама защита работает на уровень глубже.
        delta = make_grad_sanitizer(f"delta_fanin_layer{self.layer_idx}_{self.layer_type}")(delta)

        # ДИАГНОСТИКА (найдено по инцидентам 1067/1142: forward теперь
        # ПОЛНОСТЬЮ конечен -- ни одного [FWD-DIAG] -- а backward всё равно
        # даёт non-finite. Существующий пробник intra_out висит на ВХОДЕ в
        # sublayer и загорается каскадом по всем блокам через
        # HybridDARAttention -- не локализует. Этот пробник -- на ВЫХОДЕ
        # sublayer'а (до клипа delta), с тегом layer_type+layer_idx -- если
        # входящий градиент здесь уже non-finite, проблема НИЖЕ по потоку
        # (IntraBlockAttention/HybridDARAttention/следующие слои); если
        # здесь finite, а intra_out дальше внутри ЭТОГО ЖЕ layer/backward
        # (см. отдельный tag) -- non-finite, значит проблема ВНУТРИ
        # backward самого sublayer'а (B1-B6 аналитический chain для gdn2,
        # associative_scan для mamba2, или custom VJP flash-attention).
        delta = make_grad_probe(f"sublayer_out_layer{self.layer_idx}_{self.layer_type}")(delta)

        # ФИКС (подтверждено логом инцидента на шаге 943 -- без этого клипа
        # delta от layer13/mamba2 достигала ~6.25e8, а от layer14/gdn2 --
        # ~5.3e6, ОБЕ поверх current_x, который уже был ограничен фиксом #1
        # (клип после current_x+delta). Фикс #1 сам по себе НЕ защищает,
        # потому что клипается СУММА current_x+delta уже ПОСЛЕ того, как
        # delta успевает разойтись в local_deltas/block_delta/history_blocks
        # -- т.е. клип current_x спасает следующий шаг накопления residual
        # stream, но не спасает IntraBlockAttention/HybridDARAttention
        # следующих слоёв и history_blocks будущих блоков, которые читают
        # local_deltas/history_blocks НАПРЯМУЮ, в обход current_x. Вероятный
        # источник самой амплитуды -- GatedDeltaNet2J: out_gate ничем не
        # клипируется до silu(out_gate)*out -> out_proj, то есть даже
        # ограниченный вход (mixed) может дать неограниченный delta на
        # выходе. Клип здесь, в источнике, закрывает это независимо от
        # внутреннего механизма амплификации.
        #
        # ОБНОВЛЕНИЕ: сам источник (out_gate без клипа в GatedDeltaNet2J,
        # отсутствие RMSNorm на выходе Mamba2 и MLA) закрыт выше по стеку
        # (см. фиксы в GatedDeltaNet2J/Mamba2J/MLAJ) -- этот клип остаётся
        # как второй, независимый рубеж обороны, а не единственная защита.
        delta = jnp.nan_to_num(jnp.clip(delta, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

        return delta

class BlockDAR(nn.Module):
    cfg: ModelConfig
    block_idx: int
    layer_idx_start: int

    @nn.compact
    def __call__(self, current_x, x_input, history_blocks, causal_mask, cos, sin,
                 deterministic=True, rngs=None):
        b, l, d = current_x.shape
        block_input = current_x

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

            delta_finite = jnp.all(jnp.isfinite(delta))
            jax.lax.cond(
                jnp.logical_not(delta_finite),
                lambda: jax.debug.print(
                    "[FWD-DIAG] ⚠️ non-finite delta: block={b} layer={l} type=" + layer_type,
                    b=self.block_idx, l=layer_idx,
                ),
                lambda: None,
            )

            # ДИАГНОСТИКА (этот пасс): показывает, приходит ли амплитуда,
            # которую ниже поймает [RESID-DIAG], от УЖЕ большого current_x
            # (накопление из предыдущих слоёв/блоков) или от delta ИМЕННО
            # этого слоя (локальный источник) -- см. _diag_resid_breakdown
            # выше. Управляется GDN2_FWD_DIAG=1, нулевой оверхед по
            # умолчанию. Вызывается ДО клипа, чтобы видеть сырые величины.
            _diag_resid_breakdown(f"block={self.block_idx}_layer={layer_idx}_{layer_type}",
                                   current_x, delta)

            # ФИКС (найдено по логу инцидента на шаге 780, см. переписку /
            # HANDOFF: единственная non-finite точка -- block=4 layer=14
            # gdn2 -- каскадом ломает ВСЁ, что идёт после неё, вплоть до
            # backward-градиентов на layer=0). Причина: residual stream
            # current_x был ЕДИНСТВЕННЫМ местом во всём проекте без
            # clip/nan_to_num -- только finite-диагностика, ничего активно
            # не чинящая. В отличие от него параметры (train.py, клип ±1e2
            # после apply_updates), входы GDN-2/Mamba2 (клип ±1e3 внутри
            # каждого модуля), градиенты на границе custom_vjp (клип ±1e4)
            # -- все защищены. current_x накапливается через 18 сложений
            # (6 блоков x 3 слоя) без единого ограничения -- один
            # большой-но-конечный выброс (например, из Mamba2 после дрейфа
            # весов на 780 шагах) уходит непосредственно на вход следующего
            # слоя (bf16 nn.Dense в q_proj/k_proj/v_proj) ДО того, как
            # собственная внутренняя санитизация этого слоя (клип ±1e3
            # ПОСЛЕ conv+silu) вообще успевает сработать -- overflow
            # происходит в первых же матмулах.
            #
            # Печатаем max|abs| ДО санитизации (не только finite/non-finite)
            # -- чтобы на следующем прогоне видеть сам факт и величину
            # дрейфа задолго до фактического overflow, а не только момент
            # уже случившейся катастрофы.
            _cx_next = current_x + delta
            cx_abs_max = jnp.max(jnp.abs(jnp.nan_to_num(_cx_next, nan=0.0, posinf=0.0, neginf=0.0)))
            jax.lax.cond(
                cx_abs_max > 1e3,
                lambda: jax.debug.print(
                    "[RESID-DIAG] ⚠️ current_x после layer={l} (block={b}) max|abs|={m} "
                    "ДО санитизации -- дрейф residual stream", b=self.block_idx, l=layer_idx, m=cx_abs_max,
                ),
                lambda: None,
            )

            local_deltas.append(delta)

            # ФИКС (точечный, layer=22/block=7/mamba2 -- воспроизводимо даёт
            # RESID-DIAG чуть выше порога 1e3 при round_robin dataloader,
            # хотя остальные mamba2-слои на меньшей глубине (layer=4,
            # layer=13) этого не делают): общий клип ±1e3 был одинаков для
            # ВСЕХ типов слоёв. mamba2 -- документированно наименее
            # контрактный тип (associative_scan) на максимальной
            # накопленной глубине (block=7 -- последний блок) -- сужаем
            # порог именно для него, не трогая gdn2/mla. Это НЕ устраняет
            # причину дрейфа (структурная хрупкость layer=22 на
            # максимальной глубине, см. HANDOFF раздел про
            # depth x instability interaction) -- только отодвигает порог
            # раньше, чтобы санитизация срабатывала с большим запасом
            # прочности, вместо того чтобы регулярно подходить вплотную
            # к 1e3.
            _resid_clip = 5e2 if layer_type == "mamba2" else 1e3
            current_x = jnp.nan_to_num(
                jnp.clip(_cx_next, -_resid_clip, _resid_clip),
                nan=0.0, posinf=_resid_clip, neginf=-_resid_clip,
            )

        block_delta = sum(local_deltas)
        new_history = jnp.concatenate(
            [history_blocks, block_delta[None, ...]], axis=0
        )

        norm_2 = nn.RMSNorm(epsilon=1e-6, name="norm_2")(current_x).astype(current_x.dtype)
        # ФИКС (M6): top_k GmmMoEJ раньше брался ТОЛЬКО из class-level
        # default (GmmMoEJ.top_k=2), никак не связанного с cfg.top_k --
        # т.е. реальный роутинг был top-2 "случайно", а cfg.top_k оставался
        # мёртвым полем с устаревшим комментарием про SparseMoEJ (top-1).
        # Явно прокидываем cfg.top_k -- теперь это единственная точка
        # конфигурации top-k, видимая в train.py рядом с остальными
        # MoE-гиперпараметрами.
        moe_out = GmmMoEJ(cfg=self.cfg, top_k=self.cfg.top_k, name="moe")(
            norm_2, deterministic=deterministic, rngs=rngs
        )                       
        moe_finite = jnp.all(jnp.isfinite(moe_out))
        jax.lax.cond(
            jnp.logical_not(moe_finite),
            lambda: jax.debug.print("[FWD-DIAG] ⚠️ non-finite moe_out: block={b}", b=self.block_idx),
            lambda: None,
        )

        # ФИКС: тот же пробел на выходе блока -- output = current_x + moe_out
        # был последним необработанным сложением residual stream перед тем,
        # как результат уходит в history_blocks / следующий BlockDAR.
        output = jnp.nan_to_num(jnp.clip(current_x + moe_out, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

        return output, new_history


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
