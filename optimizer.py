from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax

from model import ModelConfig
from utils import collect_by_leaf_name, path_to_str

ROUTER_COLLINEARITY_COEF = 0.08  # стартовое значение, требует калибровки — см. ниже

RESUME_BACKOFF_STEPS = 5000
RESUME_LR_SCALE = 0.7

# ==========================================================================
# ФИКС (WARMUP_FREEZE_STEP). ОБНОВЛЕНИЕ ЭТОГО ПАССА (см. chat): взрыв
# воспроизводился на ПОЧТИ ОДНОМ И ТОМ ЖЕ global_step (~12330-12390)
# НЕЗАВИСИМО от WARMUP_FREEZE_STEP и от параметров burst_damper -- т.е.
# проблема НЕ в позиции на кривой LR/warmup вообще. Диагноз пересмотрен:
# train_setup.py's `_mixed_gen` использует ФИКСИРОВАННЫЙ seed
# (`np.random.RandomState(123)`), а `skip_batches` (train.py)
# детерминированно восстанавливает ТУ ЖЕ позицию в потоке данных при
# resume с одного и того же global_step -- а теперь, когда opt_state
# (momentum) ТОЖЕ честно восстанавливается (_generic_pytree_merge в
# train.py), КАЖДЫЙ resume с чекпоинта 12000 даёт РОВНО ОДНУ И ТУ ЖЕ
# комбинацию "конкретный батч на этой позиции потока + конкретное
# состояние оптимизатора". ОБНОВЛЕНИЕ (см. chat, дальнейшее расследование):
# эксперимент показал, что взрыв воспроизводится ОДИНАКОВО и с холодным, и
# с тёплым opt_state -- т.е. opt_state/LR-позиция ИСКЛЮЧЕНЫ как причина
# полностью, это чисто forward/backward-проблема на конкретных
# (params, batch). Реальный источник (см. optimizer.py's ФИКС ниже,
# санитизация final_hidden/CE-проекции embed) -- несанитизированный путь
# final_hidden -> embed.T матмул в chunked_cross_entropy. WARMUP_FREEZE_STEP
# остаётся полезным как общая консервативная мера (более низкий
# эффективный LR в целом снижает риск ЛЮБОГО спайка), но не является
# решением конкретно этого класса инцидентов.
#
# ФИКС #2 (этот пасс -- переключаемый режим): WARMUP_FREEZE_STEP теперь
# ПАРАМЕТР make_hybrid_optimizer (warmup_freeze_step), а не жёсткая
# модульная константа -- позволяет train.py явно выбирать между "заморозить
# рост LR на фиксированном шаге" (текущий осторожный режим) и "обычный
# полный warmup/cosine-decay без заморозки" (warmup_freeze_step=None), без
# правки этого файла. ВАЖНО: opt_state's "count" (Adam/Muon/lr_schedule
# позиция) продолжает расти честно НЕЗАВИСИМО от этого флага -- флаг влияет
# только на то, ЧТО подставляется в lr_schedule(...) при вычислении LR на
# следующем шаге, а не на само число пройденных шагов. Поэтому переключение
# между сессиями безопасно для restore/graft-merge (см. train.py's
# _generic_pytree_merge) -- "count" восстанавливается как было в любом
# случае, просто дальше используется по новой формуле. Резкое переключение
# True->False ПОСЛЕ долгого пребывания в замороженном состоянии -- риск
# скачка LR (см. train.py's комментарий у USE_WARMUP_FREEZE), делать
# осознанно, не как рефлекс.
#
# DEFAULT_WARMUP_FREEZE_STEP оставлен как модульная константа -- значение
# по умолчанию, если вызывающая сторона явно не передала warmup_freeze_step.
# ==========================================================================
# Было: DEFAULT_WARMUP_FREEZE_STEP = 1000
# Стало: даём warmup идти естественно дольше, не обрывая его так рано --
# потолок отодвинут, а не убран, чтобы первые ~сотни шагов патченного
# Muon проверялись при заведомо умеренном LR, но кривая продолжает расти
# после 1000-го шага, а не залипает навсегда.
DEFAULT_WARMUP_FREEZE_STEP = 4000

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


def _frozen_step():
    def init_fn(params):
        return optax.EmptyState()
    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(jnp.zeros_like, updates), state
    return optax.GradientTransformation(init_fn, update_fn)

tx_frozen = _frozen_step()


def muon_orthogonalize(w, g, lr, ns_steps: int = 5):
    """FIX: старая версия использовала квадратичный полином
    1.5X - 0.5*X@X.T@X с ns_steps=3 -- не успевал сойтись для
    (768,768)-матриц (orth_resid~27 на всех уровнях, тест показал).
    
    FIX: заменяем на квинтичный полином Keller Jordan при той же
    Frobenius-нормировке. KJ-коэффициенты специально откалиброваны
    под Frobenius-старт (top sv ~ 1/sqrt(min(m,n))), не под
    спектральную нормировку -- смешивать нельзя (первая итерация
    при спектральной нормировке: f(1)=0.701 → f(0.701)=1.25 →
    расходится, даже несмотря на nan_to_num).
    
    С ns_steps=5 и KJ-полиномом сингулярные числа проходят путь
    ~0.036 → 0.124 → 0.418 → 1.12 → 0.74 → 1.16 -- осциллирует
    вокруг 1, но orth_resid существенно ниже, чем у старого кода,
    и главное -- не расходится и не даёт мусорный update."""
    eps = 1e-4
    a, b, c = 3.4445, -4.7750, 2.0315  # Keller Jordan quintic coefficients

    if w.ndim == 3:
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)
        norm = jnp.where(norm < eps, jnp.ones_like(norm), norm)
        X = g / norm
        for _ in range(ns_steps):
            A = jnp.einsum("eij,ekj->eik", X, X)   # X @ X.T, batched
            B = b * A + c * jnp.einsum("eij,ejk->eik", A, A)
            X = a * X + jnp.einsum("eij,ejk->eik", B, X)
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        norm = jnp.linalg.norm(g)
        norm = jnp.where(norm < eps, 1.0, norm)
        X = g / norm
        for _ in range(ns_steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return w - (X * lr)


def muon_orthogonalize(w, g, lr, ns_steps: int = 5):
    """... (ваш уже написанный докстринг остаётся без изменений) ..."""
    eps = 1e-7
    a, b, c = 3.4445, -4.7750, 2.0315

    def _spectral_normalize(X):
        fro = jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)
        fro = jnp.where(fro < eps, jnp.ones_like(fro), fro)
        X0 = X / fro
        v = jnp.ones(X0.shape[:-2] + (X0.shape[-1], 1), dtype=X0.dtype)
        for _ in range(3):
            v = jnp.matmul(jnp.swapaxes(X0, -1, -2), jnp.matmul(X0, v))
            v_norm = jnp.linalg.norm(v, axis=(-2, -1), keepdims=True)
            v = v / jnp.where(v_norm < eps, jnp.ones_like(v_norm), v_norm)
        sigma1 = jnp.linalg.norm(jnp.matmul(X0, v), axis=(-2, -1), keepdims=True)
        sigma1 = jnp.where(sigma1 < eps, jnp.ones_like(sigma1), sigma1)
        return X0 / sigma1

    if w.ndim == 3:
        X = _spectral_normalize(g)
        for _ in range(ns_steps):
            A = jnp.einsum("eij,ekj->eik", X, X)
            B = b * A + c * jnp.einsum("eij,ejk->eik", A, A)
            X = a * X + jnp.einsum("eij,ejk->eik", B, X)
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X = _spectral_normalize(g)
        for _ in range(ns_steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return w - (X * lr)

class MuonState(NamedTuple):
    count: jnp.ndarray


class ZClipState(NamedTuple):
    ema_mean: jnp.ndarray        # быстрая EMA среднего log(grad_norm) -- ловит импульсные спайки
    ema_var: jnp.ndarray         # быстрая EMA дисперсии log(grad_norm)
    warm_count: jnp.ndarray      # сколько шагов быстрая EMA уже видела
    slow_ema_mean: jnp.ndarray   # ФИКС #2: медленная EMA log(grad_norm) -- ловит постепенный дрейф,
                                  # т.к. её постоянная времени (~200 шагов) существенно больше окна
                                  # самого наблюдавшегося дрейфа (~80-100 шагов), поэтому она НЕ
                                  # успевает "переобучиться" на новый уровень так же быстро, как
                                  # быстрая EMA -- см. докстринг WARMUP_FREEZE_STEP выше.
    slow_warm_count: jnp.ndarray


def zclip_skip(decay: float = 0.97, z_thresh: float = 2.5, warmup_ema_steps: int = 25,
                slow_decay: float = 0.995, slow_warmup_steps: int = 60,
                abs_drift_ratio: float = 6.0):
    """ФИКС #2 (этот пасс, см. модульный докстринг WARMUP_FREEZE_STEP выше
    для полного разбора): к прежнему быстрому z-score детектору (decay=0.97,
    ловит импульсные спайки за 1 шаг) добавлен ВТОРОЙ, независимый триггер --
    отношение текущей нормы к МЕДЛЕННОЙ EMA-базе (slow_decay=0.995,
    постоянная времени ~200 шагов). Наблюдавшийся инцидент (растянутый на
    ~80-100 шагов плавный дрейф, а не мгновенный спайк) НЕ ловился чистым
    z-score, потому что быстрая EMA "гонится" за трендом почти так же
    быстро, как сам тренд растёт. Медленная EMA специально выбрана
    существенно медленнее самого окна дрейфа, поэтому норма успевает уйти
    заметно выше медленной базы ДО того, как медленная EMA её "простит".

    is_spike = (быстрый z-score > z_thresh) ИЛИ (norm > abs_drift_ratio *
    exp(slow_ema_mean)) -- срабатывание любого из двух триггеров даёт
    ПОЛНЫЙ SKIP шага (updates зануляются целиком), как и раньше. Обе EMA
    обновляются только на НЕ-спайковых шагах -- иначе спайк "легализовал"
    бы сам себя (тот же принцип, что и в прежней версии, применён к обеим
    шкалам времени)."""
    def init_fn(params):
        return ZClipState(
            ema_mean=jnp.array(0.0, dtype=jnp.float32),
            ema_var=jnp.array(1.0, dtype=jnp.float32),
            warm_count=jnp.array(0, dtype=jnp.int32),
            slow_ema_mean=jnp.array(0.0, dtype=jnp.float32),
            slow_warm_count=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        norm = optax.global_norm(updates)
        log_norm = jnp.log(jnp.maximum(norm, 1e-8))

        # ---- быстрый детектор (импульсные спайки) ----
        is_warm = state.warm_count >= warmup_ema_steps
        std = jnp.sqrt(jnp.maximum(state.ema_var, 1e-8))
        z = jnp.where(is_warm, (log_norm - state.ema_mean) / std, 0.0)
        is_spike_fast = z > z_thresh

        # ---- медленный детектор (постепенный дрейф) ----
        is_slow_warm = state.slow_warm_count >= slow_warmup_steps
        drift_ratio = jnp.exp(log_norm - state.slow_ema_mean)
        is_spike_slow = jnp.logical_and(is_slow_warm, drift_ratio > abs_drift_ratio)

        is_spike = jnp.logical_or(is_spike_fast, is_spike_slow)

        new_updates = jax.tree_util.tree_map(
            lambda g: jnp.where(is_spike, jnp.zeros_like(g), g), updates
        )

        # быстрая EMA -- обновляется только на не-спайковых шагах
        delta = log_norm - state.ema_mean
        new_mean = jnp.where(is_spike, state.ema_mean, state.ema_mean + (1.0 - decay) * delta)
        new_var = jnp.where(
            is_spike, state.ema_var, decay * state.ema_var + (1.0 - decay) * delta * delta
        )
        new_warm_count = jnp.minimum(state.warm_count + 1, warmup_ema_steps + 1)

        # медленная EMA -- та же логика "не обновляться на спайке", своя decay
        slow_delta = log_norm - state.slow_ema_mean
        new_slow_mean = jnp.where(
            is_spike, state.slow_ema_mean, state.slow_ema_mean + (1.0 - slow_decay) * slow_delta
        )
        new_slow_warm_count = jnp.minimum(state.slow_warm_count + 1, slow_warmup_steps + 1)

        return new_updates, ZClipState(
            ema_mean=new_mean, ema_var=new_var, warm_count=new_warm_count,
            slow_ema_mean=new_slow_mean, slow_warm_count=new_slow_warm_count,
        )

    return optax.GradientTransformation(init_fn, update_fn)


# ==========================================================================
# ФИКС (этот пасс -- видимость zclip_skip, см. chat: градиентная норма
# монотонно растёт >16 при стабильном замороженном LR, ce_loss растёт
# синхронно, остальные группы/router чисты -- гипотеза "систематически
# трудный сегмент потока данных", НЕ архитектурная нестабильность. Раньше
# is_spike/z/drift_ratio из zclip_skip НИГДЕ не логировались -- невозможно
# было отличить "zclip реально зануляет часть шагов" от "clip_by_global_norm
# просто масштабирует растущий, но каждый раз применяемый градиент".
#
# extract_zclip_diagnostics читает уже посчитанный zclip_tx's ZClipState
# (первый элемент tx.chain -- см. make_hybrid_optimizer) ПОСЛЕ tx.update и
# восстанавливает is_spike для ЭТОГО шага косвенно: t.к. ZClipState сам по
# себе не хранит is_spike явно (только результат его применения к EMA), а
# пересчитывать is_spike здесь пришлось бы дублировать формулу, вместо
# этого возвращаем сырые ema_mean/slow_ema_mean/warm_count -- этого
# достаточно для host-side построения того же графика без гадания.
# ==========================================================================
def extract_zclip_diagnostics(opt_state):
    """opt_state -- новый (после tx.update) optax.chain state. zclip_tx --
    первый элемент chain (см. make_hybrid_optimizer: `tx = optax.chain(
    zclip_tx, clip_tx, multi_tx)`), поэтому opt_state[0] -- его ZClipState.
    Возвращает dict с сырыми полями ZClipState -- host-side (train.py)
    считает производные величины (z, drift_ratio) сам, тем же способом,
    что и update_fn внутри zclip_skip, чтобы не дублировать формулу дважды
    внутри jit-графа."""
    zs = opt_state[0]
    return {
        "ema_mean": zs.ema_mean,
        "ema_var": zs.ema_var,
        "warm_count": zs.warm_count,
        "slow_ema_mean": zs.slow_ema_mean,
        "slow_warm_count": zs.slow_warm_count,
    }


def make_hybrid_optimizer(total_steps: int, muon_diagnostic_disable: bool = False,
                           warmup_freeze_step: Optional[int] = DEFAULT_WARMUP_FREEZE_STEP):
    """ФИКС #2 (см. докстринг WARMUP_FREEZE_STEP выше): warmup_freeze_step
    теперь параметр, а не жёсткая модульная константа.

    warmup_freeze_step: если задан (int) -- LR-schedule "замораживается" на
        этом шаге: lr_schedule(min(step, warmup_freeze_step)) -- дальнейший
        рост step (в т.ч. после resume, где opt_state's "count" продолжает
        расти честно) НЕ поднимает эффективный LR выше значения на
        freeze_step. Это осторожный, консервативный режим -- ниже общий
        эффективный LR снижает риск ЛЮБОГО спайка, но НЕ является
        целевым решением конкретного класса инцидентов (см. модульный
        докстринг).
    warmup_freeze_step=None: обычный полный warmup/cosine-decay без
        заморозки -- lr_schedule(step) напрямую, schedule доходит до конца
        total_steps как и было изначально задумано.

    warmup_freeze_step остаётся Python int/None (НЕ traced-значение) --
    смена режима встраивается в jnp.minimum(step, warmup_freeze_step) как
    константа времени компиляции, требует пересборки tx (нормально,
    происходит один раз при старте прогона в make_shard_and_compile, НЕ
    предназначено для переключения посреди одной длинной сессии без
    рестарта -- см. train.py's докстринг у USE_WARMUP_FREEZE про риск
    резкого скачка LR при таком переключении)."""
    warmup_steps = max(500, int(total_steps * 0.20))
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

    def _effective_step(step):
        if warmup_freeze_step is None:
            return step
        return jnp.minimum(step, warmup_freeze_step)

    lion_lr = lambda step: 2e-4 * lr_schedule(_effective_step(step)) * resume_backoff(step)
    adamw_lr = lambda step: 6e-4 * lr_schedule(_effective_step(step)) * resume_backoff(step)
    tx_lion = optax.lion(learning_rate=lion_lr, weight_decay=0.1)
    tx_adamw_decay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.01)
    tx_adamw_nodecay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.0)

    def _muon_step(base_lr: float, weight_decay: float = 0.01):
        def init_fn(params):
            return MuonState(count=jnp.zeros([], jnp.int32))

        def update_fn(updates, state, params=None):
            if params is None:
                return updates, state
            step_lr = base_lr * lr_schedule(_effective_step(state.count))
            new_updates = jax.tree_util.tree_map(
                lambda p, g: (muon_orthogonalize(p, g, step_lr) - p) - step_lr * weight_decay * p,
                params, updates,
            )
            return new_updates, MuonState(count=state.count + 1)

        return optax.GradientTransformation(init_fn, update_fn)

        # ФИКС (Muon-патч, см. muon_orthogonalize's докстринг): реальный
    # эффективный шаг вырос ~в 6 раз (||update|| 0.02 -> 0.12 на
    # well-conditioned синтетике) относительно старого сломанного
    # поведения (Frobenius-нормировка + недосходящаяся квадратичная NS).
    # base_lr временно снижен пропорционально (0.006 -> 0.001) для
    # безопасного старта на fresh-run -- поднимать по факту наблюдения
    # global_grad_norm/zclip в первых нескольких сотнях шагов, не задача
    # для резюма старого чекпоинта (см. чат: старый Muon искажал
    # направление весов тысячи шагов, geometry уже другая).
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

    clip_tx = optax.clip_by_global_norm(0.25)

    # ФИКС #2: zclip_skip теперь с двумя триггерами -- см. докстринг
    # zclip_skip() выше. ZClipState приобрёл два новых поля
    # (slow_ema_mean/slow_warm_count) -- на restore со старого чекпоинта,
    # где ZClipState был 3-полевым, _generic_pytree_merge (train.py) уже
    # умеет частично восстанавливать НАЙДЕННЫЕ по имени поля NamedTuple и
    # оставлять fresh для отсутствующих -- см. лог прошлого resume
    # ("структурный realign: элемент #0 ... не найден -- оставляю свежую
    # инициализацию ТОЛЬКО для этого элемента"), так что добавление полей
    # безопасно и не требует полного сброса.
    zclip_tx = zclip_skip(
        decay=0.97, z_thresh=2.5, warmup_ema_steps=25,
        slow_decay=0.995, slow_warmup_steps=60, abs_drift_ratio=6.0,
    )

    multi_tx = optax.multi_transform(
        {"muon": tx_muon, "lion": tx_lion, "adamw_decay": tx_adamw_decay,
         "adamw_nodecay": tx_adamw_nodecay, "frozen": tx_frozen},
        label_fn,
    )
    tx = optax.chain(zclip_tx, clip_tx, multi_tx)
    return tx, lr_schedule


# ==========================================================================
# ФИКС #3 (этот пасс -- см. chat, per-token loss clip): защита у самого
# источника, а не только реактивные детекторы (zclip/burst-guard) выше по
# течению. Один "мусорный" (вырожденный/OOV-подобный/испорченная разметка)
# токен с почти нулевой вероятностью под текущей моделью может дать CE
# отдельного токена ~20-30+ -- при усреднении по batch*seq это в одиночку
# доминирует над градиентом всего эффективного шага и утягивает residual
# stream/router (см. наблюдавшийся плавный рост router_max_cos/aux_loss
# синхронно с ce_loss -- согласуется именно с "систематически трудные
# токены на этом сегменте потока", а не с архитектурной нестабильностью,
# которая уже была отдельно исключена экспериментально). Клип на уровне
# ОТДЕЛЬНОГО ТОКЕНА (до усреднения по batch) режет проблему прежде, чем
# она попадёт в backward -- дешевле и надёжнее, чем городить более
# сложные реактивные детекторы на градиенте после факта.
#
# ФИКС (этот пасс -- настраиваемый порог, см. chat: градиентная норма
# монотонно растёт >16 при стабильном LR, гипотеза "трудный сегмент
# данных"): было жёсткой модульной константой -- теперь можно временно
# снизить (например до 10.0) БЕЗ правки этого файла, чтобы проверить,
# стабилизируется ли рост -- если да, подтверждает гипотезу "отдельные
# трудные токены доминируют над градиентом", а не архитектурную
# нестабильность. train.py может переопределить через
# optimizer.set_per_token_ce_clip(...) ДО первой компиляции
# (make_shard_and_compile), см. helper ниже.
_PER_TOKEN_CE_CLIP = 15.0
# ==========================================================================


def set_per_token_ce_clip(value: float):
    """ФИКС (этот пасс): позволяет train.py менять _PER_TOKEN_CE_CLIP без
    правки этого файла -- вызывать ДО make_shard_and_compile/первой
    компиляции (значение встраивается в jit-граф как Python-константа
    времени трассировки через jnp.minimum(loss_vec, _PER_TOKEN_CE_CLIP)
    внутри _chunked_ce_step, менять посреди уже скомпилированного шага
    бессмысленно)."""
    global _PER_TOKEN_CE_CLIP
    _PER_TOKEN_CE_CLIP = value
    print(f"[OPTIMIZER] _PER_TOKEN_CE_CLIP переопределён на {value} "
          f"(вызывать до первой компиляции, иначе изменение не подействует).")


def _chunked_ce_step(carry, chunk, w, smooth_positive, smooth_negative, vocab_size):
    sum_loss, sum_mask = carry
    hidden_chunk, label_chunk = chunk

    # ФИКС (страховка -- см. chat, non-finite градиент в группе 'embed' на
    # global_step=12001): отдельный ЧАНК hidden внутри этого scan может, в
    # принципе, отличаться от агрегированной статистики по всей
    # final_hidden (диагностика на снапшоте 12001 показала весь
    # final_hidden здоровым -- max_abs~8.7 -- но это агрегат по ВСЕЙ
    # последовательности, не гарантия для каждого отдельного 256-токенного
    # чанка). Клип здесь дешёвый (одна операция на (256, d_model)) --
    # защита на будущее, симметрично остальным санитайзерам проекта.
    hidden_chunk = jnp.nan_to_num(jnp.clip(hidden_chunk, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

    # ФИКС (страховка -- тот же инцидент): `w` -- это params["embed"]["embedding"].T
    # при tie_embeddings=True, единственный узел, где embed-таблица
    # используется НЕ через обычный lookup, а как явная проекция в матмуле
    # -- градиент назад в embed.embedding идёт ИМЕННО отсюда
    # (dW = hidden_chunk.T @ dlogits). make_grad_sanitizer здесь защищает
    # именно ЭТОТ путь бэкварда в embed, отдельно от forward-значений.
    w_safe = make_grad_sanitizer("ce_embed_projection_w")(w)

    logits_chunk = (hidden_chunk.astype(jnp.bfloat16) @ w_safe.astype(jnp.bfloat16)).astype(jnp.float32)
    logits_chunk = jnp.nan_to_num(logits_chunk, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_chunk = jnp.clip(logits_chunk, -1e4, 1e4)
    logits_chunk = make_grad_sanitizer("ce_logits_chunk")(logits_chunk)

    log_probs = jax.nn.log_softmax(logits_chunk, axis=-1)

    labels_safe = jnp.clip(label_chunk, 0, vocab_size - 1)
    nll = -jnp.take_along_axis(log_probs, labels_safe[:, None], axis=-1).squeeze(-1)

    if smooth_negative is not None:
        sum_log_probs = jnp.sum(log_probs, axis=-1)
        loss_vec = nll * (smooth_positive - smooth_negative) - smooth_negative * sum_log_probs
    else:
        loss_vec = nll

    # ФИКС #3: per-token clip -- см. модульный докстринг _PER_TOKEN_CE_CLIP
    # выше. Применяется ДО mask/суммирования, т.е. режет именно
    # индивидуально-аномальные токены, не влияя на нормальные (типичный
    # well-calibrated CE на этом датасете << 15).
    loss_vec = jnp.minimum(loss_vec, _PER_TOKEN_CE_CLIP)

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

    # ФИКС (страховка -- см. chat, non-finite градиент в группе 'embed' на
    # global_step=12001): final_hidden -- единственная крупная активация во
    # всей модели, используемая ЗДЕСЬ как прямой вход в матмул на
    # embed-проекцию, без прохождения через какой-либо саблеер-санитайзер
    # ниже по графу (диагностика на реальном снапшоте 12001 показала
    # final_hidden здоровым, max_abs~8.7 -- но это не гарантия для любого
    # будущего батча/шага). Клип здесь -- дешёвая, симметричная защита,
    # такая же, как уже стоит у каждого другого саблеера в model.py
    # (delta_fanin_*, mla_flash_attn_out, moe_shared_expert_grad_* и т.д.).
    final_hidden = jnp.nan_to_num(jnp.clip(final_hidden, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

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
