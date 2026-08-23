from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from model import ModelConfig
from utils import collect_by_leaf_name, path_to_str

ROUTER_COLLINEARITY_COEF = 0.08  # стартовое значение, требует калибровки — см. ниже

# ФИКС (resume LR jump): RESUME_BACKOFF_STEPS/RESUME_LR_SCALE -- явные
# модульные константы. resume_backoff сглажена линейной рампой
# (RAMP_STEPS=1000) в теле функции ниже -- скачка LR на step=RESUME_BACKOFF_STEPS
# больше нет.
RESUME_BACKOFF_STEPS = 5000
RESUME_LR_SCALE = 0.7

# ==========================================================================
# ФИКС (этот пасс -- ГЛАВНЫЙ, см. chat: взрыв градиентов стабильно и
# ВОСПРОИЗВОДИМО случается на ~50-80% пройденного warmup, дважды подряд на
# ИДЕНТИЧНОМ шаге ~3300/6034 warmup_steps, НЕЗАВИСИМО от того, восстановлен
# ли честный opt_state с прошлой сессии (count реально продолжается с 3000,
# не с 0) -- т.е. проблема НЕ в холодном старте оптимизатора (это уже
# закрыто _generic_pytree_merge в train.py), а в АБСОЛЮТНОЙ ВЕЛИЧИНЕ LR,
# которую расписание достигает именно в этой точке рампы. BlockDAR
# (накопительный residual через HybridDARAttention + history_blocks по 7
# блокам) органически не выдерживает LR в этом диапазоне при УЖЕ прогретом
# (не нулевом) моментуме Adam/Lion/Muon -- совместное движение всех
# GDN-2/Mamba2/MoE весов на этом уровне LR входит в резонанс.
#
# WARMUP_FREEZE_STEP -- эффективная позиция на кривой lr_schedule ЗАМОРОЖЕНА
# на безопасном значении: lr_schedule(min(step, WARMUP_FREEZE_STEP)) вместо
# lr_schedule(step). Это НЕ останавливает сам счётчик state.count -- он
# продолжает расти как обычно (Adam/Lion/Muon momentum и любая другая
# внутренняя бухгалтерия оптимизатора копится нормально, opt_state
# структурно не меняется, чекпоинты остаются полностью совместимыми). 
# Замораживается ТОЛЬКО то, какое значение LR-кривой (0..1 множитель)
# используется в качестве множителя -- т.е. эффективный LR перестаёт расти
# дальше точки WARMUP_FREEZE_STEP и держится на этом безопасном плато сколь
# угодно долго, вместо того чтобы продолжать рампу к пику и упираться в
# нестабильную зону.
#
# Значение 3000 выбрано по вашим собственным наблюдениям: это последний
# ПОДТВЕРЖДЁННО безопасный шаг (сохранённый чекпоинт, дважды воспроизведённый
# взрыв на следующем ~3300). При желании поднять эту границу позже (когда
# захотите попробовать более высокий LR) -- меняется ОДНА константа, без
# пересборки оптимизатора/чекпоинтов.
#
# Побочный эффект: обучение больше НЕ доходит до пика LR/полного cosine
# decay в рамках текущей конфигурации -- это осознанный компромисс между
# стабильностью и скоростью сходимости (см. обсуждение в чате про то, что
# доучивание с меньшим, но стабильным LR дешевле, чем повторяющиеся сбросы
# opt_state на каждом взрыве). Поднимать WARMUP_FREEZE_STEP стоит только
# ПОСЛЕ того, как текущий уровень отработает стабильно продолжительное
# время (например, весь оставшийся бюджет до воркшопа) -- как отдельный,
# намеренный эксперимент, не как автоматическую рампу.
# ==========================================================================
WARMUP_FREEZE_STEP = 2600

# ДИАГНОСТИКА (2-й уровень, backward-only): см. аналогичную в model.py.
# Здесь отдельная копия, чтобы не тянуть зависимость optimizer.py -> model.py
# для одной internal-функции.
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


# ==========================================================================
# ФИКС (этот пасс -- см. чат: [DIAG] стабильно светил non-finite ИМЕННО в
# группах moe/embed/other, при том что router_max_cos/min_col_norm
# полностью здоровы (cos 0.02-0.08, min_col_norm 0.916-0.918) -- MoE-
# роутинг исключён как причина). Один из двух конкретных пробелов: узел
# "ce_logits_chunk" был единственным probe'ом в проекте, который ТОЛЬКО
# печатал non-finite градиент (make_grad_probe), но НЕ чинил его -- в
# отличие от model.py's gdn2_q_normalize/gdn2_k_normalize/
# mla_flash_attn_out/delta_fanin_* и moe_gmm.py's moe_router_input_grad,
# которые все используют активный sanitizer (клип + nan_to_num). Через
# этот узел градиент из label-smoothing (sum_log_probs по всему vocab=
# 128256) идёт НАПРЯМУЮ в embed/lm_head без какой-либо защиты -- ровно та
# группа, что светится в [DIAG]. Локальная копия model.py's
# make_grad_sanitizer (тот же паттерн, что moe_gmm.py уже применяет для
# СВОЕЙ internal-копии -- избегаем optimizer.py -> model.py импорта ради
# одной функции).
# ==========================================================================
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

# ==========================================
# Muon (Newton-Schulz orthogonalization)
# ==========================================
def muon_orthogonalize(w, g, lr, ns_steps: int = 3):
    """Orthogonalize the gradient via Newton-Schulz iteration, then take a step."""
    # ФИКС: eps увеличен — bfloat16 не держит 1e-7, норма обнуляется, 
    # деление на ~0 дает inf, Newton-Schulz взрывается.
    eps = 1e-4
    
    if w.ndim == 3:
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)
        # Если норма слишком мала — считаем градиент нулевым,
        # иначе деление на ~0 дает inf и заражает все параметры nan.
        norm = jnp.where(norm < eps, jnp.ones_like(norm), norm)
        X = g / norm
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * jnp.einsum("eij,ejk,ekl->eil", X, jnp.swapaxes(X, -1, -2), X)
            # Если итерация разошлась — обнуляем, не даем nan расползтись
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        norm = jnp.linalg.norm(g)
        norm = jnp.where(norm < eps, 1.0, norm)
        X = g / norm
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * X @ X.T @ X
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return w - (X * lr)


class MuonState(NamedTuple):
    count: jnp.ndarray


class BurstDamperState(NamedTuple):
    ema_norm: jnp.ndarray


def burst_damper(decay: float = 0.95, threshold_ratio: float = 1.8, min_scale: float = 0.05):
    """ФИКС (этот пасс -- см. chat: затяжные периоды global_grad_norm>20,
    держащиеся сотни эффективных шагов ПОДРЯД перед тем, как дойти до
    настоящего non-finite -- наблюдалось ДВАЖДЫ на РАЗНЫХ позициях
    warmup-рампы (~66-80% в первый раз, ~50% во второй), т.е. проблема НЕ
    завязана на конкретную точку кривой LR/на то, восстановлен ли
    честный momentum или нет -- она завязана на САМ ДИАПАЗОН LR/momentum,
    в котором архитектура (BlockDAR -- накопительный residual через
    history_blocks + GDN-2/Mamba2/MoE) периодически заходит в нестабильный
    режим. BURST-GUARD в train.py уже ВИДИТ этот режим заранее (несколько
    эффективных шагов подряд с global_norm>20), но раньше только логировал
    предупреждение -- ничего не предпринимал, чтобы реально затормозить
    расходящийся шаг.

    ФИКС (этот пасс -- ужесточение параметров): наблюдение показало, что
    прежние значения (decay=0.98, threshold_ratio=3.0, min_scale=0.1) НЕ
    успевают погасить именно ЗАТЯЖНУЮ СЕРИЮ всплесков (BURST-GUARD
    срабатывал раз в 2-3 шага несколько раз подряд непосредственно перед
    non-finite, см. лог global_step=12316,12319,12322,12325,12328,12331) --
    EMA с decay=0.98 слишком инертна и "не поспевает" за реальным уровнем,
    а threshold_ratio=3.0/min_scale=0.1 недостаточно агрессивны, чтобы
    реально сбить нарастающий тренд, а не просто дать одноразовый укол.
    Новые значения: decay=0.95 (EMA быстрее реагирует на текущий уровень),
    threshold_ratio=1.8 (приглушение включается раньше, не ждёт 3-кратного
    всплеска), min_scale=0.05 (сильнее давит при подтверждённом всплеске).
    Совместно с WARMUP_FREEZE_STEP (см. выше) -- это ВТОРАЯ, дополняющая
    линия обороны: freeze не даёт LR-кривой физически дойти до опасной
    зоны, burst_damper продолжает подавлять контекстные всплески даже на
    безопасном плато (архитектурная нестабильность в принципе может
    случиться и там, просто с меньшей вероятностью).

    Это -- ОТДЕЛЬНАЯ, ДОПОЛНИТЕЛЬНАЯ (не заменяющая) линия обороны против
    optax.clip_by_global_norm (фиксированный порог 0.25 -- см. ниже):
    clip_by_global_norm обрезает КАЖДЫЙ шаг до одной и той же абсолютной
    нормы, независимо от контекста -- он не различает "стабильно большой
    grad_norm, потому что модель обучается на сложных данных" и "grad_norm
    внезапно вырос в 5x относительно того, что было последние сотни шагов,
    похоже на начало runaway". burst_damper -- ВТОРАЯ, more контекстно-
    зависимая линия: держит экспоненциальное скользящее среднее (EMA)
    нормы градиента (`ema_norm`, персистентный скаляр в opt_state -- т.е.
    переживает resume ровно так же, как momentum AdamW/Lion/Muon, начиная
    с фикса _generic_pytree_merge в train.py) и, если СЫРАЯ (до clip)
    норма ЭТОГО шага более чем в `threshold_ratio` раз превышает EMA,
    приглушает обновление коэффициентом `max(min_scale, threshold_ratio /
    ratio)` -- то есть чем сильнее всплеск относительно недавней "нормы",
    тем сильнее приглушение (но не более чем до min_scale, чтобы не
    занулять шаг полностью -- полное занижение уже делает существующий
    is_finite skip-step в train_setup.py на настоящих non-finite шагах,
    это разные механизмы).

    ВАЖНО: EMA обновляется УЖЕ ПРИГЛУШЁННОЙ нормой этого шага (`damped_norm
    = norm * scale`), а НЕ сырой -- иначе сам burst "утягивал" бы EMA
    вверх и через несколько шагов "легализовывал" бы уже случившийся
    всплеск как новую норму, вместо того чтобы продолжать сопротивляться
    ему, пока он реально не затихнет.

    Вставляется ПЕРВЫМ элементом chain (до clip_by_global_norm) -- работает
    на СЫРОМ (после nan_to_num/scale в distributed_apply_step, но до
    clip_by_global_norm) градиенте, той же формы, что и остальной chain
    (весь pytree разом, до multi_transform-разметки по группам -- та же
    точка, где сейчас уже стоит clip_by_global_norm)."""
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
    # ФИКС (этот пасс -- см. chat, router collapse эпизод + non-finite взрыв
    # сразу после resume): warmup-доля увеличена 10% -> 20%. Раньше короткий
    # warmup (~10%, при total_steps=30172 это warmup_steps=3017) в сочетании
    # с тем, что opt_state (momentum AdamW/Lion/Muon) ПОЛНОСТЬЮ обнулялся на
    # каждом resume (см. train.py's докстринг про restore), давал двойной
    # удар: холодный оптимизатор + быстро растущий LR -- крах стабильно
    # ловился на ~66-80% локального прогресса по warmup (~2300-2400 при
    # warmup_steps=3017). Теперь opt_state (train.py's
    # _compatible_restore_params_and_opt_state) реально ВОССТАНАВЛИВАЕТСЯ из
    # HF-чекпоинта -- momentum и внутренние step-счётчики ("count") больше
    # НЕ обнуляются на resume для параметров, чья структура не изменилась
    # (т.е. почти всё, кроме router/expert_bias, которые и так новые/
    # несовместимые листья). Поскольку momentum теперь настоящий (не
    # холодный), оставшийся риск -- уже не "холодный старт", а "слишком
    # быстрый рост LR относительно того, как архитектура (BlockDAR,
    # накопительный residual stream) успевает адаптироваться" -- удлинённый
    # warmup даёт более плавную рампу для этого случая. Комбинируется со
    # сниженным пиком LR ниже (adamw/lion) -- обе меры решают РАЗНЫЕ грани
    # одной проблемы (высота пика + скорость его достижения), применяются
    # вместе.
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
        # ФИКС: было jnp.where(step < RESUME_BACKOFF_STEPS, RESUME_LR_SCALE, 1.0) --
        # мгновенный скачок LR в 1/0.7≈1.43x РОВНО на step=RESUME_BACKOFF_STEPS.
        # Линейная рампа на последних 1000 шагов убирает разрыв, сохраняя тот
        # же итоговый диапазон [0.7, 1.0]. Теперь, когда opt_state's count
        # ЧЕСТНО восстанавливается из чекпоинта (train.py), эта функция
        # видит РЕАЛЬНУЮ позицию обучения, а не 0 на каждой сессии -- т.е.
        # этот переход тоже происходит РОВНО ОДИН раз за всё обучение
        # (на глобальном шаге ~5000), как и было задумано изначально, а не
        # на каждом resume.
        #
        # ПРИМЕЧАНИЕ (WARMUP_FREEZE_STEP, см. константу выше): resume_backoff
        # НЕ заморожена -- она использует СЫРОЙ (не капнутый) step. Раз
        # WARMUP_FREEZE_STEP=3000 < RESUME_BACKOFF_STEPS=5000, при текущей
        # конфигурации resume_backoff всё ещё будет находиться в фазе
        # рампы 0.7->1.0 к моменту, когда lr_schedule уже заморожен -- то
        # есть итоговый LR первое время после step=3000 будет ПРОДОЛЖАТЬ
        # медленно расти (за счёт resume_backoff), даже когда lr_schedule
        # сам по себе уже не растёт. Это осознанно оставлено как есть --
        # эффект небольшой (0.7->1.0, плавно за 1000 шагов) и НЕ похож по
        # характеру на резкий рост warmup-кривой, который и был источником
        # проблемы. Если после внедрения freeze всё равно будут наблюдаться
        # всплески в момент окончания resume_backoff-рампы (~step 4000-5000)
        # -- следующий кандидат на заморозку именно эта функция.
        RAMP_STEPS = 1000.0
        ramp_start = RESUME_BACKOFF_STEPS - RAMP_STEPS
        frac = jnp.clip((step - ramp_start) / RAMP_STEPS, 0.0, 1.0)
        return RESUME_LR_SCALE + (1.0 - RESUME_LR_SCALE) * frac

    # ==========================================================================
    # ФИКС (пик LR снижен, см. chat -- router collapse эпизод 9300-9568 и
    # non-finite взрыв на ~2300-2400 локальных шагах warmup): BlockDAR --
    # накопительный, некaнонический residual stream (HybridDARAttention +
    # history_blocks через 7 блоков) в сочетании с насыщенным/
    # структурированным датасетом (код/математика/агентные траектории,
    # более коррелированный градиент) органически не терпит такой же пик
    # LR, как более "обычные" архитектуры. Пик снижен ~0.6-0.67x по обеим
    # группам (adamw: 1e-3 -> 6e-4, lion: 3e-4 -> 2e-4). Применяется ВМЕСТЕ
    # с удлинённым warmup (выше) и восстановлением реального opt_state
    # (train.py) -- три независимые, дополняющие друг друга меры против
    # одной и той же категории non-finite/router-collapse инцидентов.
    #
    # ФИКС (этот пасс -- ГЛАВНЫЙ, WARMUP_FREEZE_STEP): step, который видит
    # lr_schedule, теперь капается сверху константой WARMUP_FREEZE_STEP
    # (см. её докстринг выше) -- jnp.minimum(step, WARMUP_FREEZE_STEP).
    # Сам step (внутренний счётчик каждой optax-схемы -- ScaleByScheduleState
    # для adamw/lion, MuonState.count для muon) продолжает расти как обычно
    # и восстанавливается из чекпоинта штатно через _generic_pytree_merge --
    # НИКАКИХ изменений в структуре opt_state, полная совместимость с уже
    # сохранёнными чекпоинтами (в т.ч. с текущим на шаге ~3300). Меняется
    # ТОЛЬКО то, какую точку lr_schedule(...) эти LR-лямбды читают -- она
    # больше не продвигается дальше WARMUP_FREEZE_STEP, LR держится на
    # безопасном плато вместо продолжения рампы к пику.
    # ==========================================================================
    lion_lr = lambda step: 2e-4 * lr_schedule(jnp.minimum(step, WARMUP_FREEZE_STEP)) * resume_backoff(step)
    adamw_lr = lambda step: 6e-4 * lr_schedule(jnp.minimum(step, WARMUP_FREEZE_STEP)) * resume_backoff(step)
    tx_lion = optax.lion(learning_rate=lion_lr, weight_decay=0.1)
    tx_adamw_decay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.01)
    tx_adamw_nodecay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.0)

    def _muon_step(base_lr: float, weight_decay: float = 0.01):
        def init_fn(params):
            return MuonState(count=jnp.zeros([], jnp.int32))

        def update_fn(updates, state, params=None):
            if params is None:
                return updates, state
            # ФИКС (WARMUP_FREEZE_STEP): та же заморозка эффективной точки
            # расписания, что и для lion_lr/adamw_lr выше -- state.count
            # продолжает расти нормально (нужен и дальше растущим, т.к.
            # используется только ЗДЕСЬ, внутри lr_schedule и нигде больше),
            # но передаваемое в lr_schedule значение капается сверху.
            step_lr = base_lr * lr_schedule(jnp.minimum(state.count, WARMUP_FREEZE_STEP))
            # ФИКС: у Muon-ветки (в отличие от AdamW/Lion) не было НИКАКОГО
            # weight decay -- ортогонализованное обновление ничего не тянет
            # к нулю, поэтому норма параметров могла годами (в масштабе шагов
            # обучения) медленно дрейфовать вверх без противовеса. Это
            # правдоподобный вклад в наблюдавшийся "взрыв параметров"
            # (см. диагностику [PARAM-DIAG] в train.py). Добавляем простой
            # decoupled weight decay: w <- w - step_lr*weight_decay*w,
            # применяется ПОСЛЕ основного muon-шага, тем же способом, что
            # AdamW/Lion делают decoupled decay.
            new_updates = jax.tree_util.tree_map(
                lambda p, g: (muon_orthogonalize(p, g, step_lr) - p) - step_lr * weight_decay * p,
                params, updates,
            )
            return new_updates, MuonState(count=state.count + 1)

        return optax.GradientTransformation(init_fn, update_fn)

    # ФИКС (этот пасс -- см. chat: затяжной период global_grad_norm>20 на
    # протяжении ~800 эффективных шагов ПОСЛЕ того, как честный
    # opt_state-merge наконец заработал, т.е. Muon больше не получает
    # случайный "сброс момента" на каждом resume, который РАНЬШЕ,
    # похоже, непреднамеренно периодически чинил медленный дрейф нормы
    # параметров вверх): base_lr снижен ещё раз (0.008 -> 0.006),
    # weight_decay увеличен (0.01 -> 0.02) -- сильнее противовес
    # ортогонализованным обновлениям, которые сами по себе ничего не
    # тянут к нулю. Раньше этот дрейф периодически (и случайно) обнулялся
    # холодным restart'ом оптимизатора при каждом resume -- теперь, когда
    # momentum реально переживает resume (см. _generic_pytree_merge's
    # критический фикс в train.py), эта защита должна идти НЕ от
    # случайных сбросов, а от самого механизма (сильнее decay, ниже LR).
    tx_muon = _muon_step(base_lr=0.006, weight_decay=0.02)

    # ФИКС: НЕ добавляем отдельную группу multi_transform для decay_a/A_log --
    # это меняет СТРУКТУРУ opt_state (новый ключ в multi_transform), что
    # ломает restore со старых чекпоинтов (несовпадение pytree). Тот же эффект
    # "замедленного LR для decay-параметров" реализован в train.py на уровне
    # МАСШТАБИРОВАНИЯ ГРАДИЕНТА (avg_grads *= 0.2 для этих leaf) ДО входа в
    # tx.update() -- функционально эквивалентно, но не трогает состояние
    # оптимизатора, поэтому совместимо с уже сохранёнными чекпоинтами.
    def _label_leaf(path, param):
        path_str = path_to_str(path)
        if "embed" in path_str or "lm_head" in path_str:
            return "adamw_decay"
        if "norm" in path_str or "bias" in path_str:
            return "adamw_nodecay"
        # ФИКС (интеграция SparseMoEJ, atomic_ops/moe_sparse.py): router --
        # маленький, чувствительный к начальной балансировке Dense(d_model,
        # E_routed). Muon-ортогонализация (агрессивное обновление
        # направления, без weight decay до фикса выше) на этом конкретном
        # слое рискует резко раскачать routing-решения до того, как
        # утилизация экспертов успеет устаканиться -- именно тот режим
        # (высокий dropped_ratio на первых шагах, пока роутер не
        # сбалансирован), где ошибка маршрутизации дороже всего. AdamW без
        # decay -- мягче и предсказуемее для этого конкретного слоя, тот же
        # выбор, что уже сделан для norm/bias.
        #
        # ФИКС (bias-балансировка, DeepSeek-V3 style): expert_bias остаётся
        # "frozen" для ГРАДИЕНТНОГО пути -- обновляется decoupled-путём в
        # train_setup.py's apply_expert_bias_update, на основе ФАКТИЧЕСКОЙ
        # частоты top-k-назначений (assignment_frac), не градиента loss.
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

    # ФИКС (этот пасс -- см. chat: затяжной ~800-шаговый период
    # global_grad_norm>20 перед non-finite в moe/embed/other, PARAM-DIAG
    # ни разу не сработал -- т.е. параметры сами по себе не разрослись, но
    # градиент долго оставался нездоровым): clip ужесточён ещё раз
    # (0.35 -> 0.25). clip_by_global_norm не имеет состояния (EmptyState),
    # поэтому это изменение НЕ влияет на совместимость чекпоинтов.
    clip_tx = optax.clip_by_global_norm(0.25)

    # ФИКС (этот пасс -- ужесточённые параметры burst_damper, см. её
    # собственный докстринг выше про decay=0.95/threshold_ratio=1.8/
    # min_scale=0.05 вместо прежних 0.98/3.0/0.1): вставляется ПЕРВЫМ
    # элементом chain -- работает на СЫРОМ градиенте, до clip_by_global_norm
    # и до multi_transform-разметки по группам. BurstDamperState.ema_norm --
    # персистентный скаляр в opt_state, уже присутствовал в chain до этого
    # пасса (см. docstring про structural realign в train.py) -- изменение
    # только гиперпараметров функции НЕ меняет структуру/форму состояния,
    # т.е. полностью совместимо с уже сохранёнными чекпоинтами (значение
    # ema_norm восстановится как было, продолжит обновляться уже под новыми,
    # более строгими порогами).
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
    hidden_chunk, label_chunk = chunk  # (chunk_size, d_model), (chunk_size,)

    # Матмул в bf16 для памяти (chunk_size x d_model x vocab — самый большой
    # single matmul в модели). Bfloat16 дает ~2x меньше памяти и полную
    # throughput TPU MXU. НО: при больших значениях hidden/w bf16 overflow'ится
    # в inf. Решение: upcast в fp32 -> nan_to_num (inf->clip) -> clip -> softmax.
    logits_chunk = (hidden_chunk.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16)).astype(jnp.float32)

    # ФИКС: sanitize bfloat16 overflow. Если значение inf или nan —
    # заменяем на крайние допустимые, чтобы log_softmax не дал nan.
    logits_chunk = jnp.nan_to_num(logits_chunk, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_chunk = jnp.clip(logits_chunk, -1e4, 1e4)

    # ФИКС (этот пасс -- см. chat: [DIAG] стабильно светил non-finite
    # именно в группах moe/embed/other; MoE-роутинг исключён как причина
    # свежими метриками (router_max_cos/min_col_norm здоровы) -- этот узел
    # был единственным probe'ом в проекте, который ТОЛЬКО печатал, но не
    # чинил non-finite градиент. label smoothing тянет через
    # sum_log_probs по всему vocab_size -- градиент здесь реально может
    # улетать при насыщенных логитах и идёт НАПРЯМУЮ в embed/lm_head без
    # защиты. make_grad_probe -> make_grad_sanitizer, тот же паттерн, что
    # уже применяется во всех остальных узлах проекта (model.py's
    # gdn2_q_normalize/mla_flash_attn_out/delta_fanin_*, moe_gmm.py's
    # moe_router_input_grad).
    logits_chunk = make_grad_sanitizer("ce_logits_chunk")(logits_chunk)

    log_probs = jax.nn.log_softmax(logits_chunk, axis=-1)

    labels_safe = jnp.clip(label_chunk, 0, vocab_size - 1)
    nll = -jnp.take_along_axis(log_probs, labels_safe[:, None], axis=-1).squeeze(-1)

    if smooth_negative is not None:
        sum_log_probs = jnp.sum(log_probs, axis=-1)
        loss_vec = nll * (smooth_positive - smooth_negative) - smooth_negative * sum_log_probs
    else:
        loss_vec = nll

    mask = (label_chunk != -100).astype(jnp.float32)

    # ФИКС: jnp.where вместо умножения. Если nll содержит nan (например, от
    # inf logits до sanitize), то nan * 0.0 = nan, и jnp.sum все равно даст nan.
    # jnp.where(mask>0, loss_vec, 0.0) берет 0.0 из false-branch и игнорирует
    # nan в true-branch — pad-токены дают ровно 0 вклада.
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

    # ФИКС: защита от пустого батча (все токены -100). jnp.maximum с 1.0
    # вместо +1e-9 — если sum_mask=0, возвращаем 0.0 (не огромное число).
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
    min_col_norm_stacked = None               # NEW
    max_abs_logit_preclip_stacked = None      # NEW
    norm_x_mean_stacked = None    # NEW
    norm_x_max_stacked = None     # NEW
    norm_x_min_stacked = None     # NEW
    # ФИКС (bias-балансировка, DeepSeek-V3 style): assignment_frac -- см.
    # docstring в moe_gmm.py's патче ("assignment_frac", sown сразу после
    # top_gate). Собирается тем же collect_by_leaf_name-путём, что и
    # router_temp -- порядок листьев ДОЛЖЕН совпадать с порядком
    # expert_bias-параметров в params (оба идут по блокам model.py's
    # BlockDAR в порядке block_0, block_1, ... -- см. предупреждение в
    # train_setup.py's _build_expert_bias_index_map).
    assignment_frac_stacked = None  # NEW
    if not deterministic:
        final_hidden, sowed_vars = outputs
        aux_losses = collect_by_leaf_name(sowed_vars["losses"], "aux_loss")
        z_losses = collect_by_leaf_name(sowed_vars["losses"], "z_loss")
        expert_utils = collect_by_leaf_name(sowed_vars["losses"], "expert_utilization")
        # ФИКС (интеграция SparseMoEJ): moe_dropped_ratio sown per-layer by
        # SparseMoEJ (atomic_ops/moe_sparse.py) -- same collection pattern
        # as expert_utilization/aux_loss/z_loss above. Absent for the dense
        # MoEJ path, so this stays None (and downstream consumers must
        # handle that, same as expert_utilization already does) if the
        # model is ever switched back to the dense MoE for cross-checking.
        dropped_ratios = collect_by_leaf_name(sowed_vars["losses"], "moe_dropped_ratio")
        router_temps = collect_by_leaf_name(sowed_vars["losses"], "router_temp")
        min_col_norms = collect_by_leaf_name(sowed_vars["losses"], "min_col_norm")
        max_abs_logits_preclip = collect_by_leaf_name(sowed_vars["losses"], "max_abs_logit_preclip")
        router_collinearities = collect_by_leaf_name(sowed_vars["losses"], "router_collinearity")
        router_max_cos_list = collect_by_leaf_name(sowed_vars["losses"], "router_max_cos")
        norm_x_mean = collect_by_leaf_name(sowed_vars["losses"], "norm_x_mean")
        norm_x_max = collect_by_leaf_name(sowed_vars["losses"], "norm_x_max")
        norm_x_min = collect_by_leaf_name(sowed_vars["losses"], "norm_x_min")
        # NEW: assignment_frac -- список из E_routed-векторов, по одному на
        # MoE-блок, в порядке обхода дерева (тот же порядок, что router_temp).
        assignment_fracs = collect_by_leaf_name(sowed_vars["losses"], "assignment_frac")
        aux_loss = jnp.sum(jnp.stack(aux_losses)) if aux_losses else 0.0
        z_loss = jnp.sum(jnp.stack(z_losses)) if z_losses else 0.0
        if expert_utils:
            expert_util_stacked = jnp.stack(expert_utils)
        if dropped_ratios:
            dropped_ratio_stacked = jnp.stack(dropped_ratios)
        if router_temps:                # добавлено
            router_temp_stacked = jnp.stack(router_temps)
        if assignment_fracs:            # NEW
            assignment_frac_stacked = jnp.stack(assignment_fracs)   # (n_moe_layers, E_routed)
        min_col_norm_stacked = jnp.stack(min_col_norms) if min_col_norms else None
        max_abs_logit_preclip_stacked = jnp.stack(max_abs_logits_preclip) if max_abs_logits_preclip else None
        # Анти-коллинеарный штраф
        collinearity_loss = jnp.sum(jnp.stack(router_collinearities)) if router_collinearities else 0.0
        # Для логирования в W&B: возьмём максимум по слоям, чтобы видеть наихудший случай
        router_max_cos_per_layer = jnp.stack(router_max_cos_list) if router_max_cos_list else None
        # Вычисляем максимум по слоям для мониторинга в проде
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

    ce_loss = chunked_cross_entropy(final_hidden, labels, w, cfg.label_smoothing, chunk_size=ce_chunk_size)

    # ФИКС: последняя линия обороны. Если в params уже есть nan (например,
    # от предыдущего взорвавшегося шага), обнуляем ce_loss чтобы не заразить
    # opt_state. Обучение продолжится с плохим loss — это сигнал смотреть
    # предыдущие шаги, но не убивает процесс.
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
            "min_col_norm": min_col_norm_stacked,                     # NEW
            "max_abs_logit_preclip": max_abs_logit_preclip_stacked,   # NEW
            "norm_x_mean": norm_x_mean_stacked,   # NEW
            "norm_x_max": norm_x_max_stacked,     # NEW
            "norm_x_min": norm_x_min_stacked,     # NEW
            "router_max_cos_per_layer": router_max_cos_per_layer,
            "router_max_cos": router_max_cos,
            "assignment_frac": assignment_frac_stacked,   # NEW (bias-балансировка)
        }
        return total_loss, aux_info
    return total_loss
