"""
optimizer.py -- гибридный оптимизатор (Muon/Lion/AdamW) + CE-loss.

ФИКС (этот пасс -- SVD Muon): muon_orthogonalize заменён на точный
polar factor через SVD вместо аппроксимации Newton-Schulz.
Старая версия (Frobenius norm + квадратичный NS, ns_steps=3) давала
orth_resid~27 на ВСЕХ уровнях обусловленности для (768,768)-матриц,
то есть фактически работала как слабо-масштабированный SGD, без
какой-либо ортогонализации -- это было структурной причиной монотонного
роста global_grad_norm с шага ~350/780.

ФИКС #2 (этот пасс -- burst_damper вместо zclip_skip): zclip_skip
зануляло обновления на каждом шаге в проблемном диапазоне (шаги
12761-12854: fast_z=34-38 непрерывно) -- это reactive защита, которая
убивает шаги, но не лечит причину. burst_damper масштабирует gradient
step пропорционально отклонению от EMA нормы вместо полного обнуления.

ФИКС #3 (этот пасс -- совместимость API с train.py/train_setup.py):
- DEFAULT_WARMUP_FREEZE_STEP экспортируется как модульная константа
- make_hybrid_optimizer принимает warmup_freeze_step параметр
- extract_zclip_diagnostics возвращает правильные jnp-массивы
  (совместимость с zclip_diag_sharding в train_setup.py)

ФИКС #4 (этот пасс -- ВСЕГДА включённая диагностика Muon + сбор новых
sow'нутых метрик из model.py, см. чат): несмотря на докстринг выше
("SVD Muon"), реальный `muon_orthogonalize` -- это Newton-Schulz(5), НЕ
SVD, и НИГДЕ не логировалось, насколько хорошо этот NS-итератор реально
сходится к ортогональной матрице (orth_resid = ||X X^T - I||_F, в
идеале -> 0). Добавлен `_muon_orth_diag` (считает ТОТ ЖЕ NS-итератор
ещё раз, чисто для диагностики -- слowdown допустим, см. чат) и
`extract_muon_diagnostics`, вызываемый из train_setup.py's
distributed_apply_step после tx.update -- если этот residual остаётся
высоким (не сходится к 0) на muon-размеченных параметрах, это прямой,
количественный кандидат на причину монотонного роста global_grad_norm
(в отличие от "мы подозреваем, что NS не сходится" без единой цифры).

Также compute_loss теперь собирает новые self.sow(...) метрики из
model.py (layer_delta_maxabs/layer_resid_maxabs/*_isfinite,
mamba2_input_maxabs/mamba2_ssm_out_maxabs, gdn2_input_maxabs/
gdn2_raw_out_maxabs/gdn2_h_final_maxabs/gdn2_out_maxabs/
gdn2_decay_a_maxabs, mla_input_maxabs/mla_out_maxabs,
final_hidden_maxabs/isfinite) в aux_info, тем же collect_by_leaf_name
паттерном, что уже используется для router_temp/norm_x_mean -- train.py
логирует их в W&B каждый шаг.
"""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax

from model import ModelConfig
from utils import collect_by_leaf_name, path_to_str

ROUTER_COLLINEARITY_COEF = 0.08

RESUME_BACKOFF_STEPS = 5000
RESUME_LR_SCALE = 0.7

# Экспортируется для train.py: from optimizer import DEFAULT_WARMUP_FREEZE_STEP
# Поднят 1000 -> 4000 чтобы LR рос естественно в течение настоящего warmup
# (warmup_steps = max(500, 0.15*total_steps) ≈ 4660 шагов при total=31072),
# а не обрывался на середине.
DEFAULT_WARMUP_FREEZE_STEP = 1000

# per-token CE clip -- оставлен для совместимости с set_per_token_ce_clip()
# Не применяется в _chunked_ce_step в этой версии (убран намеренно --
# см. комментарий у _chunked_ce_step ниже).
_PER_TOKEN_CE_CLIP = 15.0


def set_per_token_ce_clip(value: float):
    """Позволяет train.py менять порог без правки этого файла.
    Вызывать ДО первой компиляции."""
    global _PER_TOKEN_CE_CLIP
    _PER_TOKEN_CE_CLIP = value
    print(f"[OPTIMIZER] _PER_TOKEN_CE_CLIP переопределён на {value}")


# ==========================================================================
# Grad utilities (make_grad_sanitizer/probe нужны и здесь, и в model.py --
# optimizer.py определяет их первым, model.py импортирует оттуда же)
# ==========================================================================

def make_grad_sanitizer(tag: str, clip_val: float = 1e3):
    """Активно клипует non-finite/large градиенты на узле."""
    @jax.custom_vjp
    def _sanitizer(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        finite = jnp.all(jnp.isfinite(g))
        jax.lax.cond(
            jnp.logical_not(finite),
            lambda: jax.debug.print(
                "[BWD-FIX] 🩹 non-finite градиент в узле {t} -- санитизирован", t=tag),
            lambda: None,
        )
        g_safe = jnp.nan_to_num(
            jnp.clip(g, -clip_val, clip_val),
            nan=0.0, posinf=clip_val, neginf=-clip_val,
        )
        return (g_safe,)

    _sanitizer.defvjp(_fwd, _bwd)
    return _sanitizer


def make_grad_probe(tag: str):
    """Диагностика без обрезки -- только печатает предупреждение."""
    @jax.custom_vjp
    def _probe(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        finite = jnp.all(jnp.isfinite(g))
        jax.lax.cond(
            jnp.logical_not(finite),
            lambda: jax.debug.print(
                "[BWD-DIAG] ⚠️ non-finite ВХОДЯЩИЙ градиент в узле: " + tag),
            lambda: None,
        )
        return (g,)

    _probe.defvjp(_fwd, _bwd)
    return _probe


# ==========================================================================
# Frozen optimizer (для expert_bias)
# ==========================================================================

def _frozen_step():
    def init_fn(params):
        return optax.EmptyState()
    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(jnp.zeros_like, updates), state
    return optax.GradientTransformation(init_fn, update_fn)

tx_frozen = _frozen_step()


# ==========================================================================
# Muon ортогонализация
# ==========================================================================

def muon_orthogonalize_legacy(w, g, lr, ns_steps: int = 3):
    """СТАРАЯ (сломанная) версия -- Frobenius norm + квадратичный NS.
    Оставлена ТОЛЬКО как fallback/cross-check. НЕ использовать в обучении.

    Проблема: для (768,768) X0 = G/||G||_F имеет сингулярные числа
    ~1/sqrt(768)≈0.036 -- три шага квадратичной итерации не успевают
    сойтись (orth_resid~27.3 на ВСЕХ уровнях обусловленности, включая
    well-conditioned). Фактически: update ≈ const·G (без ортогонализации).
    """
    eps = 1e-4
    if w.ndim == 3:
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)
        norm = jnp.where(norm < eps, jnp.ones_like(norm), norm)
        X = g / norm
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * jnp.einsum(
                "eij,ejk,ekl->eil", X, jnp.swapaxes(X, -1, -2), X)
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        norm = jnp.linalg.norm(g)
        norm = jnp.where(norm < eps, 1.0, norm)
        X = g / norm
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * X @ X.T @ X
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return w - (X * lr)


def _muon_ns_iterate(X, ns_steps: int = 5):
    """Общий Newton-Schulz(5)-итератор, вынесенный отдельно от
    muon_orthogonalize, чтобы _muon_orth_diag (ниже) мог посчитать РОВНО
    тот же X, что реально используется в апдейте, без дублирования
    формулы в двух местах с риском рассинхронизации."""
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def muon_orthogonalize(w, g, lr, ns_steps: int = 5):
    """Квинтичный Newton-Schulz (Keller Jordan), Frobenius-нормировка.
    Matmul-only -- не требует SVD/LU, полностью на MXU, на порядок быстрее.
    Не даёт точный polar factor (orth_resid не -> 0), но даёт "достаточно
    ортогональное" направление -- это ровно то, что использует реальный
    Muon на масштабных моделях (не эталонный SVD, а именно это)."""
    eps = 1e-7

    if w.ndim == 3:
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)
        safe_norm = jnp.where(norm < eps, jnp.ones_like(norm), norm)
        X = g / safe_norm
        X = _muon_ns_iterate(X, ns_steps)
        X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        effective_lr = jnp.where(norm < eps, jnp.zeros_like(norm), lr)
    else:
        norm = jnp.linalg.norm(g)
        safe_norm = jnp.where(norm < eps, 1.0, norm)
        X = g / safe_norm
        X = _muon_ns_iterate(X, ns_steps)
        X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        effective_lr = jnp.where(norm < eps, 0.0, lr)

    return w - (X * effective_lr)


def _muon_orth_diag(g, ns_steps: int = 5):
    eps = 1e-7
    if g.ndim == 3:
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)
        safe_norm = jnp.where(norm < eps, jnp.ones_like(norm), norm)
        X = _muon_ns_iterate(g / safe_norm, ns_steps)
        X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        XXt = X @ X.mT
        # ФИКС: X @ X.mT имеет форму (..., X.shape[-2], X.shape[-2]) --
        # eye должен строиться по ЭТОЙ оси, а не по X.shape[-1] (столбцы).
        # Для квадратных Muon-параметров (напр. 768x768) обе оси совпадают,
        # и баг маскировался; на прямоугольных матрицах (например,
        # (768, d_latent) в MLA/DAR-проекциях) X.shape[-1] != X.shape[-2],
        # и jnp.eye(X.shape[-1]) даёт неверный размер -> broadcast TypeError
        # в X @ X.mT - eye.
        eye = jnp.eye(X.shape[-2], dtype=X.dtype)[None]
        resid = jnp.linalg.norm(XXt - eye, axis=(-2, -1))
        return jnp.max(resid)
    else:
        norm = jnp.linalg.norm(g)
        safe_norm = jnp.where(norm < eps, 1.0, norm)
        X = _muon_ns_iterate(g / safe_norm, ns_steps)
        X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # ФИКС: то же самое для 2D-случая -- X @ X.T имеет форму
        # (X.shape[-2], X.shape[-2]).
        eye = jnp.eye(X.shape[-2], dtype=X.dtype)
        return jnp.linalg.norm(X @ X.T - eye)

# ==========================================================================
# State NamedTuples
# ==========================================================================

class MuonState(NamedTuple):
    count: jnp.ndarray
    # ФИКС #4: диагностика, см. _muon_orth_diag выше -- max orth_resid по
    # всем muon-размеченным параметрам НА ЭТОМ шаге. НЕ используется
    # оптимизатором в самом апдейте (чисто для логирования), НО является
    # частью state, чтобы train.py могло достать его после tx.update тем
    # же способом (extract_muon_diagnostics / collect_by_leaf_name), что
    # уже используется для ZClipState/BurstDamperState. Новое поле в
    # NamedTuple -- при resume со старых чекпоинтов graft-merge
    # (_generic_pytree_merge в train.py) просто оставит его свежим
    # (нулевым), см. train.py's докстринг про несовпадающие поля.
    orth_resid: jnp.ndarray


# ZClipState оставлен как NamedTuple-определение для совместимости с
# graft-merge (_generic_pytree_merge в train.py): при resume со старых
# чекпоинтов, где ZClipState был первым элементом chain, merge по именам
# полей вернёт fresh для несовпадающих (BurstDamperState).
class ZClipState(NamedTuple):
    ema_mean: jnp.ndarray
    ema_var: jnp.ndarray
    warm_count: jnp.ndarray
    slow_ema_mean: jnp.ndarray
    slow_warm_count: jnp.ndarray


class BurstDamperState(NamedTuple):
    ema_norm: jnp.ndarray


# ==========================================================================
# Burst damper (заменяет zclip_skip)
# ==========================================================================

def burst_damper(decay: float = 0.95, threshold_ratio: float = 1.8,
                  min_scale: float = 0.05):
    """Масштабирует updates если норма резко превышает EMA-базу.

    В отличие от zclip_skip (полное обнуление при spike), здесь градиент
    МАСШТАБИРУЕТСЯ до threshold_ratio * ema_norm, но не обнуляется --
    обучение продолжается на каждом шаге.

    threshold_ratio=1.8: допускаем рост до 80% над базой без вмешательства.
    min_scale=0.05: даже при сильном spike шаг не меньше 5% от расчётного.
    """
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


# ==========================================================================
# extract_zclip_diagnostics -- совместимость с train_setup.py
# ==========================================================================

def extract_zclip_diagnostics(opt_state):
    """Совместимость с train_setup.py: возвращает dict с теми же 5 ключами,
    что ZClipState. opt_state[0] теперь BurstDamperState -- возвращаем
    нулевые jnp-массивы нужного dtype. Они совместимы с
    zclip_diag_sharding = {key: NamedSharding(mesh, P())} в train_setup.py
    (скалярные jnp-массивы с P() = replicated, всегда работают в jit).

    Значения в W&B будут нулевыми (fast_z=0, drift_ratio=1) -- это нормально,
    zclip больше не используется, диагностика burst_damper идёт отдельно
    через global_norm/clip_factor.
    """
    return {
        "ema_mean":       jnp.zeros((), dtype=jnp.float32),
        "ema_var":        jnp.ones((), dtype=jnp.float32),
        "warm_count":     jnp.zeros((), dtype=jnp.int32),
        "slow_ema_mean":  jnp.zeros((), dtype=jnp.float32),
        "slow_warm_count": jnp.zeros((), dtype=jnp.int32),
    }


def extract_muon_diagnostics(opt_state):
    """ФИКС #4 (см. модульный докстринг): собирает orth_resid со ВСЕХ
    MuonState-листьев внутри opt_state (там ровно один на каждый
    muon-размеченный параметр -- multi_transform's per-leaf state) через
    collect_by_leaf_name (utils.py, уже умеет ходить по NamedTuple/dict/
    tuple вперемешку -- ровно то, что нужно для optax.chain внутри
    multi_transform). Возвращает СКАЛЯР -- max orth_resid по ВСЕМ
    muon-параметрам этого шага (худший случай -- самый информативный для
    "начал ли где-то NS(5) переставать сходиться").

    Вызывается из train_setup.py's distributed_apply_step ПОСЛЕ tx.update
    -- то есть значение относится именно к обновлению, которое было
    ПРИМЕНЕНО на этом шаге, не к предыдущему."""
    values = collect_by_leaf_name(opt_state, "orth_resid")
    if not values:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.max(jnp.stack([v.astype(jnp.float32) for v in values]))


# ==========================================================================
# Основной оптимизатор
# ==========================================================================

def make_hybrid_optimizer(total_steps: int, muon_diagnostic_disable: bool = False,
                           warmup_freeze_step: Optional[int] = DEFAULT_WARMUP_FREEZE_STEP):
    """Гибридный оптимизатор: Muon (SVD) / Lion / AdamW.

    warmup_freeze_step: если задан (int) -- LR заморожен на этом шаге.
        None -- полный warmup/cosine-decay без заморозки.
        DEFAULT_WARMUP_FREEZE_STEP=4000 -- LR растёт естественно в течение
        warmup (~4660 шагов при total=31072), потолок только после.

    ФИКС base_lr Muon: снижен 0.006->0.001 чтобы компенсировать то, что
    SVD даёт ТОЧНЫЙ ортогональный фактор (||update|| ~0.12) vs старый
    сломанный Muon (||update|| ~0.02, в 6 раз меньше). Итоговый
    эффективный шаг теперь сопоставим. Поднимать по факту логов.

    ФИКС #4 (см. модульный докстринг): MuonState теперь несёт orth_resid
    -- update_fn ниже считает его КАЖДЫЙ шаг (через _muon_orth_diag) на
    КАЖДОМ muon-параметре и хранит max в новом состоянии.
    """
    warmup_steps = max(500, int(total_steps * 0.15))
    cosine = optax.cosine_decay_schedule(
        init_value=1.0,
        decay_steps=max(1, total_steps - warmup_steps),
        alpha=0.1,
    )
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=1.0, transition_steps=warmup_steps),
            cosine,
        ],
        boundaries=[warmup_steps],
    )

    def resume_backoff(step):
        RAMP_STEPS = 1000.0
        ramp_start = RESUME_BACKOFF_STEPS - RAMP_STEPS
        frac = jnp.clip((step - ramp_start) / RAMP_STEPS, 0.0, 1.0)
        return RESUME_LR_SCALE + (1.0 - RESUME_LR_SCALE) * frac

    def _effective_step(step):
        if warmup_freeze_step is None:
            return step
        return jnp.minimum(step, warmup_freeze_step)

    lion_lr  = lambda step: 2e-4 * lr_schedule(_effective_step(step)) * resume_backoff(step)
    adamw_lr = lambda step: 6e-4 * lr_schedule(_effective_step(step)) * resume_backoff(step)

    tx_lion        = optax.lion(learning_rate=lion_lr, weight_decay=0.1)
    tx_adamw_decay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.01)
    tx_adamw_nodecay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.0)

    def _muon_step(base_lr: float, weight_decay: float = 0.02):
        def init_fn(params):
            return MuonState(
                count=jnp.zeros([], jnp.int32),
                orth_resid=jnp.zeros([], jnp.float32),
            )

        def update_fn(updates, state, params=None):
            if params is None:
                return updates, state
            step_lr = base_lr * lr_schedule(_effective_step(state.count))
            new_updates = jax.tree_util.tree_map(
                lambda p, g: (muon_orthogonalize(p, g, step_lr) - p)
                             - step_lr * weight_decay * p,
                params, updates,
            )
            # ФИКС #4: diag считается на КАЖДОМ листе этой multi_transform-
            # подветки (уже отфильтрованной label_fn'ом до только
            # muon-параметров) -- максимум по ним всем и есть худший
            # случай на этом шаге. Дёшево относительно самого апдейта
            # (тот же порядок работы, что muon_orthogonalize уже делает).
            per_leaf_resid = jax.tree_util.tree_map(lambda g: _muon_orth_diag(g), updates)
            leaf_resids = jax.tree_util.tree_leaves(per_leaf_resid)
            step_orth_resid = (
                jnp.max(jnp.stack(leaf_resids)) if leaf_resids
                else jnp.array(0.0, dtype=jnp.float32)
            )
            return new_updates, MuonState(count=state.count + 1, orth_resid=step_orth_resid)

        return optax.GradientTransformation(init_fn, update_fn)

    tx_muon = _muon_step(base_lr=0.001, weight_decay=0.02)

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
            return "muon"
        return "lion"

    def label_fn(params):
        return jax.tree_util.tree_map_with_path(_label_leaf, params)

    # clip_by_global_norm поднят 0.25 -> 1.0:
    # старый 0.25 при norm~50 давал clip_factor=0.005 (200x приглушение),
    # что в паре со сломанным Muon делало шаг вдвойне занижанным.
    # 1.0 -- стандартный порог для таких моделей.
    clip_tx   = optax.clip_by_global_norm(1.0)
    damper_tx = burst_damper(decay=0.95, threshold_ratio=1.8, min_scale=0.05)

    multi_tx = optax.multi_transform(
        {
            "muon": tx_muon, "lion": tx_lion,
            "adamw_decay": tx_adamw_decay, "adamw_nodecay": tx_adamw_nodecay,
            "frozen": tx_frozen,
        },
        label_fn,
    )
    # damper_tx -- opt_state[0], extract_zclip_diagnostics возвращает
    # заглушку с правильными dtype/shape для train_setup.py совместимости.
    tx = optax.chain(damper_tx, clip_tx, multi_tx)
    return tx, lr_schedule


# ==========================================================================
# Cross-entropy loss
# ==========================================================================

def _chunked_ce_step(carry, chunk, w, smooth_positive, smooth_negative, vocab_size):
    """Один чанк CE-потерь.

    ФИКС (этот пасс): убраны make_grad_sanitizer на w и logits_chunk --
    заменены на make_grad_probe (только диагностика). Градиенты на CE-пути
    естественно велики (sum по batch*seq токенам), clip_val=1e3 убивал
    50-90% сигнала. Теперь градиент проходит нетронутым, только
    non-finite логируется.

    hidden_chunk клипован +-1e3 в forward (защита от численного взрыва
    активаций), это не касается gradient flow.
    """
    sum_loss, sum_mask = carry
    hidden_chunk, label_chunk = chunk

    # Forward-side защита активаций (не градиентов)
    hidden_chunk = jnp.nan_to_num(
        jnp.clip(hidden_chunk, -1e3, 1e3),
        nan=0.0, posinf=1e3, neginf=-1e3,
    )

    logits_chunk = (
        hidden_chunk.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16)
    ).astype(jnp.float32)
    logits_chunk = jnp.nan_to_num(logits_chunk, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_chunk = jnp.clip(logits_chunk, -1e4, 1e4)
    # Probe (диагностика без обрезки): логирует non-finite если будет
    logits_chunk = make_grad_probe("ce_logits_chunk")(logits_chunk)

    log_probs = jax.nn.log_softmax(logits_chunk, axis=-1)

    labels_safe = jnp.clip(label_chunk, 0, vocab_size - 1)
    nll = -jnp.take_along_axis(
        log_probs, labels_safe[:, None], axis=-1).squeeze(-1)

    if smooth_negative is not None:
        sum_log_probs = jnp.sum(log_probs, axis=-1)
        loss_vec = (
            nll * (smooth_positive - smooth_negative)
            - smooth_negative * sum_log_probs
        )
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
    smooth_negative = (
        (label_smoothing / (vocab_size - 1)) if label_smoothing > 0 else None
    )

    scan_step = jax.checkpoint(
        lambda carry, chunk: _chunked_ce_step(
            carry, chunk, w, smooth_positive, smooth_negative, vocab_size)
    )

    (sum_loss, sum_mask), _ = jax.lax.scan(
        scan_step, (0.0, 0.0), (hidden_chunks, label_chunks))
    return sum_loss / jnp.maximum(sum_mask, 1.0)


def _stack_or_none(values):
    return jnp.stack(values) if values else None


def compute_loss(params, model_fn, batch, cfg: ModelConfig,
                  rngs=None, deterministic=False, return_aux=False,
                  ce_chunk_size=256, collinearity_coef=None):
    input_ids = batch["input_ids"]
    labels    = batch["labels"]

    kwargs = {"deterministic": deterministic, "return_hidden": True}
    if rngs is not None:
        kwargs["rngs"] = rngs

    outputs = model_fn(
        {"params": params}, input_ids, **kwargs,
        mutable=["losses"] if not deterministic else False,
    )

    expert_util_stacked = dropped_ratio_stacked = router_temp_stacked = None
    min_col_norm_stacked = max_abs_logit_preclip_stacked = None
    norm_x_mean_stacked = norm_x_max_stacked = norm_x_min_stacked = None
    assignment_frac_stacked = None
    aux_loss = z_loss = collinearity_loss = 0.0
    router_max_cos_per_layer = None
    router_max_cos = 0.0

    # ФИКС #4 (см. модульный докстринг): новые sown-диагностики из
    # model.py -- инициализируем как None по умолчанию (deterministic-путь
    # / старые версии model.py без этих sow-вызовов просто не заполнят их,
    # aux_info_sharding в train_setup.py должен допускать None -> но т.к.
    # это jit-выход, отсутствие ключа здесь означает aux_info просто не
    # будет содержать это значение под deterministic=True -- train.py уже
    # обрабатывает это через aux_info.get(...) с проверкой на None).
    diag_stacked = {}

    if not deterministic:
        final_hidden, sowed_vars = outputs
        losses = sowed_vars["losses"]

        aux_losses        = collect_by_leaf_name(losses, "aux_loss")
        z_losses          = collect_by_leaf_name(losses, "z_loss")
        expert_utils      = collect_by_leaf_name(losses, "expert_utilization")
        dropped_ratios    = collect_by_leaf_name(losses, "moe_dropped_ratio")
        router_temps      = collect_by_leaf_name(losses, "router_temp")
        min_col_norms     = collect_by_leaf_name(losses, "min_col_norm")
        max_abs_logits    = collect_by_leaf_name(losses, "max_abs_logit_preclip")
        collinearities    = collect_by_leaf_name(losses, "router_collinearity")
        max_cos_list      = collect_by_leaf_name(losses, "router_max_cos")
        norm_x_mean       = collect_by_leaf_name(losses, "norm_x_mean")
        norm_x_max        = collect_by_leaf_name(losses, "norm_x_max")
        norm_x_min        = collect_by_leaf_name(losses, "norm_x_min")
        assignment_fracs  = collect_by_leaf_name(losses, "assignment_frac")

        aux_loss        = jnp.sum(jnp.stack(aux_losses))   if aux_losses   else 0.0
        z_loss          = jnp.sum(jnp.stack(z_losses))     if z_losses     else 0.0
        collinearity_loss = jnp.sum(jnp.stack(collinearities)) if collinearities else 0.0

        if expert_utils:      expert_util_stacked         = jnp.stack(expert_utils)
        if dropped_ratios:    dropped_ratio_stacked       = jnp.stack(dropped_ratios)
        if router_temps:      router_temp_stacked         = jnp.stack(router_temps)
        if assignment_fracs:  assignment_frac_stacked     = jnp.stack(assignment_fracs)
        if min_col_norms:     min_col_norm_stacked        = jnp.stack(min_col_norms)
        if max_abs_logits:    max_abs_logit_preclip_stacked = jnp.stack(max_abs_logits)
        if norm_x_mean:       norm_x_mean_stacked         = jnp.stack(norm_x_mean)
        if norm_x_max:        norm_x_max_stacked          = jnp.stack(norm_x_max)
        if norm_x_min:        norm_x_min_stacked          = jnp.stack(norm_x_min)

        if max_cos_list:
            router_max_cos_per_layer = jnp.stack(max_cos_list)
            router_max_cos = jnp.max(router_max_cos_per_layer)

        # ФИКС #4: собираем все новые ВСЕГДА-включённые sow'ы из model.py
        # (см. его докстринг про "ДИАГНОСТИКА (всегда вкл)"). Ключи здесь
        # совпадают с именами, переданными в self.sow("losses", <имя>, ...)
        # в model.py -- каждый собран в том же порядке обхода pytree, что
        # и router_temp/norm_x_mean уже собираются выше.
        _DIAG_LEAF_NAMES = (
            "layer_delta_maxabs", "layer_delta_isfinite",
            "layer_resid_maxabs", "layer_resid_isfinite",
            "mamba2_input_maxabs", "mamba2_input_isfinite", "mamba2_A_maxabs",
            "mamba2_ssm_out_pre_norm_maxabs", "mamba2_ssm_out_pre_norm_isfinite",
            "mamba2_ssm_out_maxabs",
            "gdn2_input_maxabs", "gdn2_input_isfinite", "gdn2_decay_a_maxabs",
            "gdn2_raw_out_maxabs", "gdn2_raw_out_isfinite", "gdn2_h_final_maxabs",
            "gdn2_out_maxabs",
            "mla_input_maxabs", "mla_out_maxabs",
            "final_hidden_maxabs", "final_hidden_isfinite",
            # ФИКС (kernel-internal diagnostics, см. atomic_ops/kernel_diag.py):
            # состояние ВНУТРИ Pallas-пайплайна GDN-2 (Aqk/Akk/A/w_pseudo/
            # u/kg/qg) -- ловит near-singular Akk в Kernel B ДО того, как
            # это всплывёт как inf в Kernel D несколькими шагами позже.
            "gdn2_kernelstage_aqk_maxabs", "gdn2_kernelstage_aqk_isfinite",
            "gdn2_kernelstage_akk_maxabs", "gdn2_kernelstage_akk_isfinite",
            "gdn2_kernelstage_a_wy_inverse_maxabs", "gdn2_kernelstage_a_wy_inverse_isfinite",
            "gdn2_kernelstage_w_pseudo_maxabs", "gdn2_kernelstage_w_pseudo_isfinite",
            "gdn2_kernelstage_u_maxabs", "gdn2_kernelstage_u_isfinite",
            "gdn2_kernelstage_kg_maxabs", "gdn2_kernelstage_kg_isfinite",
            "gdn2_kernelstage_qg_maxabs", "gdn2_kernelstage_qg_isfinite",
        )
        for name in _DIAG_LEAF_NAMES:
            vals = collect_by_leaf_name(losses, name)
            if vals:
                diag_stacked[name] = jnp.stack(vals)
    else:
        final_hidden = outputs

    if cfg.tie_embeddings:
        w = params["embed"]["embedding"].T
    else:
        w = params["lm_head"]["kernel"]

    # Forward-side защита final_hidden перед CE-проекцией
    final_hidden = jnp.nan_to_num(
        jnp.clip(final_hidden, -1e3, 1e3),
        nan=0.0, posinf=1e3, neginf=-1e3,
    )

    ce_loss = chunked_cross_entropy(
        final_hidden, labels, w, cfg.label_smoothing, chunk_size=ce_chunk_size)
    ce_loss = jnp.nan_to_num(ce_loss, nan=0.0, posinf=1e4, neginf=0.0)

    _coef = collinearity_coef if collinearity_coef is not None else ROUTER_COLLINEARITY_COEF
    total_loss = (
        ce_loss
        + cfg.router_aux_loss_coef * aux_loss
        + cfg.router_z_loss_coef   * z_loss
        + _coef                    * collinearity_loss
    )

    if return_aux:
        aux_info = {
            "ce_loss":                    ce_loss,
            "aux_loss":                   aux_loss,
            "z_loss":                     z_loss,
            "expert_utilization":         expert_util_stacked,
            "moe_dropped_ratio":          dropped_ratio_stacked,
            "router_temp":                router_temp_stacked,
            "min_col_norm":               min_col_norm_stacked,
            "max_abs_logit_preclip":      max_abs_logit_preclip_stacked,
            "norm_x_mean":                norm_x_mean_stacked,
            "norm_x_max":                 norm_x_max_stacked,
            "norm_x_min":                 norm_x_min_stacked,
            "router_max_cos_per_layer":   router_max_cos_per_layer,
            "router_max_cos":             router_max_cos,
            "assignment_frac":            assignment_frac_stacked,
        }
        aux_info.update(diag_stacked)
        return total_loss, aux_info
    return total_loss
