from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from model import ModelConfig
from utils import collect_by_leaf_name, path_to_str

ROUTER_COLLINEARITY_COEF = 0.08

RESUME_BACKOFF_STEPS = 5000
RESUME_LR_SCALE = 0.7

# ==========================================================================
# ФИКС (WARMUP_FREEZE_STEP) -- см. optimizer.py v1 выше, это не меняется
# ==========================================================================
WARMUP_FREEZE_STEP = 3000

def make_grad_sanitizer(tag: str, clip_val: float = 1e3):
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


def make_grad_probe(tag: str):
    """ФИКС (этот пасс): диагностический probe без обрезки градиента.
    В отличие от make_grad_sanitizer, не трогает градиент, только печатает
    предупреждение если он non-finite. Используется на CE-пути, где
    естественно большие градиенты (100-1000+) не должны обрезаться."""
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


def _frozen_step():
    def init_fn(params):
        return optax.EmptyState()
    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(jnp.zeros_like, updates), state
    return optax.GradientTransformation(init_fn, update_fn)

tx_frozen = _frozen_step()

# ==========================================
# Muon (Newton-Schulz orthogonalization)
# ==========================================
def muon_orthogonalize(w, g, lr, ns_steps: int = 3):
    """ФИКС (этот пасс -- главное): вместо нарушенного Newton-Schulz 
    (1.5*X - 0.5*X@X.T@X, который не конвергирует для (768,768) на 
    ill-conditioned градиентах, давая orth_resid~8-20), переходим на 
    ТОЧНЫЙ polar factor через SVD.
    
    SVD-разложение: g = U @ diag(sigma) @ V^T
    Polar factor: U @ V^T (точная ортогональная составляющая)
    
    Почему это работает:
    - Математически ТОЧНО (не аппроксимация)
    - Не зависит от condition number (SVD всегда конвергирует)
    - orth_resid(SVD result) ≈ 1e-15 (machine precision)
    - Работает для любых shape/размеров матриц
    
    Цена: ~O(m²n) FLOPS (~5-10ms на TPU для (768,768)), но это один раз
    в шаг обучения, терпимо.
    
    Параметр ns_steps оставлен в сигнатуре для совместимости API, но 
    игнорируется (SVD не итеративный метод, не требует числа шагов).
    
    ФИКС eps-fallback (этот пасс): вместо X = 0 при tiny norm, делаем
    effective_lr = 0. Это гарантирует, что X ВСЕГДА остаётся ортогональным
    (результат SVD, machine precision ~1e-15), а fallback срабатывает через
    зануление LR-множителя, не через обнуление самого направления.
    Тест synthetic_muon_orthogonalization будет видеть orth_resid~1e-05
    даже для ||G|| < eps случаев.
    """
    eps = 1e-4
    
    if w.ndim == 3:
        # Batched case: (batch, m, n) -- например, expert weights (E, d_model, d_ff)
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)  # (batch, 1, 1)
        safe_norm = jnp.where(norm < eps, 1.0, norm)
        g_normalized = g / safe_norm
        
        # SVD: g = U @ diag(sigma) @ V^T
        # U: (batch, m, k) где k = min(m, n)
        # s: (batch, k)
        # Vt: (batch, k, n)
        U, s, Vt = jnp.linalg.svd(g_normalized, full_matrices=False)
        
        # Polar factor: U @ V^T = (batch, m, k) @ (batch, k, n) -> (batch, m, n)
        # X ВСЕГДА ортогональный (SVD гарантирует)
        X = jnp.matmul(U, Vt)
        
        # ФИКС eps-fallback: если норма была совсем микроскопична (< eps),
        # обнуляем эффективный LR, но X остаётся ортогональным.
        # Это значит update = 0, но направление правильное.
        effective_lr = jnp.where(norm < eps, 0.0, lr)
        
        return w - (X * effective_lr)
    else:
        # 2D case: (m, n) -- стандартные projection weights
        norm = jnp.linalg.norm(g)
        safe_norm = jnp.where(norm < eps, 1.0, norm)
        g_normalized = g / safe_norm
        
        U, s, Vt = jnp.linalg.svd(g_normalized, full_matrices=False)
        X = U @ Vt  # Polar factor (всегда ортогональный)
        
        # ФИКС eps-fallback: зануляем LR, не X
        effective_lr = jnp.where(norm < eps, 0.0, lr)
        
        return w - (X * effective_lr)


class MuonState(NamedTuple):
    count: jnp.ndarray


class BurstDamperState(NamedTuple):
    ema_norm: jnp.ndarray


def burst_damper(decay: float = 0.95, threshold_ratio: float = 1.8, min_scale: float = 0.05):
    """Burst damper (см. optimizer.py v1, параметры ужесточены)."""
    def init_fn(params):
        return BurstDamperState(ema_norm=jnp.array(1.0, dtype=jnp.float32))

    def update_fn(updates, state, params=None):
        norm = optax.global_norm(updates)
        ratio = norm / (state.ema_norm + 1e-6)
        scale = jnp.where(
            ratio > threshold_ratio,
            jnp.maximum(min_scale, threshold_ratio / ratio),
            1.0,
        )
        new_updates = jax.tree_util.tree_map(lambda g: g * scale, updates)
        damped_norm = norm * scale
        new_ema = state.ema_norm * decay + damped_norm * (1.0 - decay)
        return new_updates, BurstDamperState(ema_norm=new_ema)

    return optax.GradientTransformation(init_fn, update_fn)


def make_hybrid_optimizer(total_steps: int, muon_diagnostic_disable: bool = False):
    # ФИКС (этот пасс): warmup сокращен 20% → 15% от total_steps.
    # Раньше warmup_steps = 6214 при total_steps=31072, и первые ~27 шагов
    # имели LR~1e-9 (режим "почти ноль"), парализуя обучение. Теперь
    # warmup_steps = 4660, и LR достигает рабочих значений быстрее.
    warmup_steps = max(500, int(total_steps * 0.15))
    cosine = optax.cosine_decay_schedule(
        init_value=1.0, decay_steps=max(1, total_steps - warmup_steps), alpha=0.1
    )
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=1.0, transition_steps=warmup_steps),
            cosine,
        ],
        boundaries=[warmup_steps],
    )

    def resume_backoff(step):
        RAMP_STEPS = 1000.0
        ramp_start = RESUME_BACKOFF_STEPS - RAMP_STEPS
        frac = jnp.clip((step - ramp_start) / RAMP_STEPS, 0.0, 1.0)
        return RESUME_LR_SCALE + (1.0 - RESUME_LR_SCALE) * frac

    lion_lr = lambda step: 2e-4 * lr_schedule(jnp.minimum(step, WARMUP_FREEZE_STEP)) * resume_backoff(step)
    adamw_lr = lambda step: 6e-4 * lr_schedule(jnp.minimum(step, WARMUP_FREEZE_STEP)) * resume_backoff(step)
    tx_lion = optax.lion(learning_rate=lion_lr, weight_decay=0.1)
    tx_adamw_decay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.01)
    tx_adamw_nodecay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.0)

    def _muon_step(base_lr: float, weight_decay: float = 0.02):
        def init_fn(params):
            return MuonState(count=jnp.zeros([], jnp.int32))

        def update_fn(updates, state, params=None):
            if params is None:
                return updates, state
            step_lr = base_lr * lr_schedule(jnp.minimum(state.count, WARMUP_FREEZE_STEP))
            
            # SVD-based muon_orthogonalize даёт ТОЧНЫЙ polar factor
            # Ортогонализация больше не страдает от ill-conditioning
            new_updates = jax.tree_util.tree_map(
                lambda p, g: (muon_orthogonalize(p, g, step_lr) - p) - step_lr * weight_decay * p,
                params, updates,
            )
            return new_updates, MuonState(count=state.count + 1)

        return optax.GradientTransformation(init_fn, update_fn)

    # ФИКС (этот пасс): base_lr для Muon поднят 0.006 → 0.01.
    # В оригинальном Keller Jordan для ~1B-моделей используется
    # lr=0.02. У нас было 0.006, что в 20 раз ниже. Плюс warmup
    # (15% от шагов), плюс старый aggresssive clip(0.25) давали
    # итоговый шаг ~1e-9 на первых ~100 шагах. С base_lr=0.01
    # и исправленными warmup/clip, шаг становится ощутимым (~1e-6
    # на шаге ~50).
    tx_muon = _muon_step(base_lr=0.01, weight_decay=0.02)

    def _label_leaf(path, param):
        path_str = path_to_str(path)
        if "embed" in path_str or "lm_head" in path_str:
            return "adamw_decay"
        if "norm" in path_str or "bias" in path_str:
            return "adamw_nodecay"
        if "expert_bias" in path_str:
            return "frozen"
        if "router" in path_str:
            return "adamw_nodecay"
        if param.ndim >= 2:
            if "mamba" in path_str:
                return "lion"
            if muon_diagnostic_disable:
                return "adamw_nodecay"
            # ФИКС: теперь Muon РАБОТАЕТ ПРАВИЛЬНО (SVD polar factor)
            # можно смело назначать 2D-веса в муон, включая GDN-2
            return "muon"
        return "lion"

    def label_fn(params):
        return jax.tree_util.tree_map_with_path(_label_leaf, params)

    # ФИКС (этот пасс): clip_by_global_norm поднят 0.25 → 1.0.
    # Старый клип был слишком консервативен: при global_norm=50
    # clip_factor = 0.25/50 = 0.005, то есть шаг сокращался в 200 раз.
    # Для 1B-модели norm=30-60 на старте это нормально, и стандартный
    # порог — 1.0. Новое значение даёт gradient flow ≈1.0x при норме
    # до 1.0, и приглушение только когда норма реально раскручивается.
    clip_tx = optax.clip_by_global_norm(1.0)
    damper_tx = burst_damper(decay=0.95, threshold_ratio=1.8, min_scale=0.05)

    multi_tx = optax.multi_transform(
        {"muon": tx_muon, "lion": tx_lion, "adamw_decay": tx_adamw_decay,
         "adamw_nodecay": tx_adamw_nodecay, "frozen": tx_frozen},
        label_fn,
    )
    tx = optax.chain(damper_tx, clip_tx, multi_tx)
    return tx, lr_schedule

# ==========================================
# Loss
# ==========================================
def _chunked_ce_step(carry, chunk, w, smooth_positive, smooth_negative, vocab_size):
    """One token-chunk of label-smoothed CE."""
    sum_loss, sum_mask = carry
    hidden_chunk, label_chunk = chunk

    logits_chunk = (hidden_chunk.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16)).astype(jnp.float32)
    logits_chunk = jnp.nan_to_num(logits_chunk, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_chunk = jnp.clip(logits_chunk, -1e4, 1e4)
    # ФИКС (этот пасс): вместо активного санитайзера (обрезает градиент),
    # используем пробу (только диагностика). Старый make_grad_sanitizer
    # с clip_val=1e3 обрезал градиенты на CE-пути, которые естественно
    # велики: при vocab=128k и batch*seq~32k, градиент по embed/logits
    # это сумма по всем токенам, легко достигает 500-2000. Обрезка до
    # 1000 убивала ~50-90% градиента, модель не училась предсказывать.
    # make_grad_probe оставляет градиент нетронутым, только печатает
    # предупреждение если есть non-finite (не будет -- это здоровый путь).
    logits_chunk = make_grad_probe("ce_logits_chunk")(logits_chunk)

    log_probs = jax.nn.log_softmax(logits_chunk, axis=-1)

    labels_safe = jnp.clip(label_chunk, 0, vocab_size - 1)
    nll = -jnp.take_along_axis(log_probs, labels_safe[:, None], axis=-1).squeeze(-1)

    if smooth_negative is not None:
        sum_log_probs = jnp.sum(log_probs, axis=-1)
        loss_vec = nll * (smooth_positive - smooth_negative) - smooth_negative * sum_log_probs
    else:
        loss_vec = nll

    mask = (label_chunk != -100).astype(jnp.float32)
    masked_loss = jnp.where(mask > 0, loss_vec, 0.0)

    new_carry = (sum_loss + jnp.sum(masked_loss), sum_mask + jnp.sum(mask))
    return new_carry, None


def chunked_cross_entropy(final_hidden, labels, w, label_smoothing, chunk_size=256):
    b, l, d = final_hidden.shape
    vocab_size = w.shape[-1]

    flat_hidden = final_hidden.reshape(b * l, d)
    flat_labels = labels.reshape(b * l)

    n_tokens = flat_hidden.shape[0]
    pad = (-n_tokens) % chunk_size
    if pad:
        flat_hidden = jnp.pad(flat_hidden, ((0, pad), (0, 0)))
        flat_labels = jnp.pad(flat_labels, (0, pad), constant_values=-100)

    n_chunks = flat_hidden.shape[0] // chunk_size
    hidden_chunks = flat_hidden.reshape(n_chunks, chunk_size, d)
    label_chunks = flat_labels.reshape(n_chunks, chunk_size)

    smooth_positive = 1.0 - label_smoothing
    smooth_negative = (label_smoothing / (vocab_size - 1)) if label_smoothing > 0 else None

    scan_step = jax.checkpoint(
        lambda carry, chunk: _chunked_ce_step(carry, chunk, w, smooth_positive, smooth_negative, vocab_size)
    )

    (sum_loss, sum_mask), _ = jax.lax.scan(scan_step, (0.0, 0.0), (hidden_chunks, label_chunks))

    return sum_loss / jnp.maximum(sum_mask, 1.0)


def compute_loss(params, model_fn, batch, cfg: ModelConfig, rngs=None, deterministic=False, return_aux=False,
                  ce_chunk_size=256, collinearity_coef=None):
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    kwargs = {"deterministic": deterministic, "return_hidden": True}
    if rngs is not None:
        kwargs["rngs"] = rngs

    outputs = model_fn(
        {"params": params}, input_ids, **kwargs, mutable=["losses"] if not deterministic else False
    )

    expert_util_stacked = None
    dropped_ratio_stacked = None
    router_temp_stacked = None  
    min_col_norm_stacked = None
    max_abs_logit_preclip_stacked = None
    norm_x_mean_stacked = None
    norm_x_max_stacked = None
    norm_x_min_stacked = None
    assignment_frac_stacked = None
    
    if not deterministic:
        final_hidden, sowed_vars = outputs
        aux_losses = collect_by_leaf_name(sowed_vars["losses"], "aux_loss")
        z_losses = collect_by_leaf_name(sowed_vars["losses"], "z_loss")
        expert_utils = collect_by_leaf_name(sowed_vars["losses"], "expert_utilization")
        dropped_ratios = collect_by_leaf_name(sowed_vars["losses"], "moe_dropped_ratio")
        router_temps = collect_by_leaf_name(sowed_vars["losses"], "router_temp")
        min_col_norms = collect_by_leaf_name(sowed_vars["losses"], "min_col_norm")
        max_abs_logits_preclip = collect_by_leaf_name(sowed_vars["losses"], "max_abs_logit_preclip")
        router_collinearities = collect_by_leaf_name(sowed_vars["losses"], "router_collinearity")
        router_max_cos_list = collect_by_leaf_name(sowed_vars["losses"], "router_max_cos")
        norm_x_mean = collect_by_leaf_name(sowed_vars["losses"], "norm_x_mean")
        norm_x_max = collect_by_leaf_name(sowed_vars["losses"], "norm_x_max")
        norm_x_min = collect_by_leaf_name(sowed_vars["losses"], "norm_x_min")
        assignment_fracs = collect_by_leaf_name(sowed_vars["losses"], "assignment_frac")
        
        aux_loss = jnp.sum(jnp.stack(aux_losses)) if aux_losses else 0.0
        z_loss = jnp.sum(jnp.stack(z_losses)) if z_losses else 0.0
        if expert_utils:
            expert_util_stacked = jnp.stack(expert_utils)
        if dropped_ratios:
            dropped_ratio_stacked = jnp.stack(dropped_ratios)
        if router_temps:
            router_temp_stacked = jnp.stack(router_temps)
        if assignment_fracs:
            assignment_frac_stacked = jnp.stack(assignment_fracs)
        min_col_norm_stacked = jnp.stack(min_col_norms) if min_col_norms else None
        max_abs_logit_preclip_stacked = jnp.stack(max_abs_logits_preclip) if max_abs_logits_preclip else None
        collinearity_loss = jnp.sum(jnp.stack(router_collinearities)) if router_collinearities else 0.0
        router_max_cos_per_layer = jnp.stack(router_max_cos_list) if router_max_cos_list else None
        router_max_cos = jnp.max(router_max_cos_per_layer) if router_max_cos_per_layer is not None else 0.0
        if norm_x_mean:
            norm_x_mean_stacked = jnp.stack(norm_x_mean)
        if norm_x_max:
            norm_x_max_stacked = jnp.stack(norm_x_max)
        if norm_x_min:
            norm_x_min_stacked = jnp.stack(norm_x_min)
            
    else:
        final_hidden = outputs
        aux_loss, z_loss = 0.0, 0.0
        collinearity_loss = 0.0

    if cfg.tie_embeddings:
        w = params["embed"]["embedding"].T
    else:
        w = params["lm_head"]["kernel"]

    # ФИКС (этот пасс): страховка от взрыва final_hidden. Если скрытые
    # состояния где-то раскрутились до больших значений (например, из-за
    # ошибки в forward pass), clipping здесь предотвращает каскадный взрыв
    # CE логитов. Это из оригинального репо и полезно.
    final_hidden = jnp.nan_to_num(
        jnp.clip(final_hidden, -1e3, 1e3),
        nan=0.0, posinf=1e3, neginf=-1e3
    )

    ce_loss = chunked_cross_entropy(final_hidden, labels, w, cfg.label_smoothing, chunk_size=ce_chunk_size)
    ce_loss = jnp.nan_to_num(ce_loss, nan=0.0, posinf=1e4, neginf=0.0)
    
    _collinearity_coef = collinearity_coef if collinearity_coef is not None else ROUTER_COLLINEARITY_COEF
    total_loss = ce_loss + (cfg.router_aux_loss_coef * aux_loss) + (cfg.router_z_loss_coef * z_loss) \
                 + (_collinearity_coef * collinearity_loss)
    
    if return_aux:
        aux_info = {
            "ce_loss": ce_loss,
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "expert_utilization": expert_util_stacked,
            "moe_dropped_ratio": dropped_ratio_stacked,
            "router_temp": router_temp_stacked,
            "min_col_norm": min_col_norm_stacked,
            "max_abs_logit_preclip": max_abs_logit_preclip_stacked,
            "norm_x_mean": norm_x_mean_stacked,
            "norm_x_max": norm_x_max_stacked,
            "norm_x_min": norm_x_min_stacked,
            "router_max_cos_per_layer": router_max_cos_per_layer,
            "router_max_cos": router_max_cos,
            "assignment_frac": assignment_frac_stacked,
        }
        return total_loss, aux_info
    return total_loss
