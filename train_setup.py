"""
train_setup.py -- диагностика non-finite по группам параметров, TPU mesh /
шардинг / компиляция train-step'ов, multi-source dataloader.

Вынесено из train.py. Логика не менялась -- перенос как есть.

ФИКС (dataloader, по гипотезе "смешение источников в одном батче триггерит
RESID-DIAG на layer=22/mamba2" -- см. чат): dataloader_multi_source теперь
поддерживает три режима подачи данных (mode="mixed"/"sequential"/
"round_robin") и per-source fraction (третий элемент в file_pairs). Все
остальные функции в этом файле -- БЕЗ ИЗМЕНЕНИЙ.

ФИКС #2 (этот пасс): _round_robin_gen был переписан на "быстрый пропуск без
чтения с диска" (принимает batch_idx через next(src_gens[s]), затем сам
вызывает _gather_batch) -- но функция-генератор индексов, которую он
вызывал (_infinite_source_indices), нигде не была определена, только
_infinite_source_batches (которая сразу возвращает ГОТОВЫЕ данные, а не
индексы) -- это NameError при первом же вызове mode="round_robin".
Добавлена _infinite_source_indices -- тот же паттерн, что и
_infinite_source_batches, но yield'ит idx_local[...] вместо
_gather_batch(idx_local[...]), чтобы _round_robin_gen мог пропускать уже
пройденные микрошаги (skip_first) БЕЗ чтения с диска на каждый из них --
именно так, как и было задумано в его собственном докстринге/принте
"Быстрый пропуск N микрошагов (без чтения с диска)".

ФИКС #3 (этот пасс -- W&B диагностика без jax.debug.print/callback):
раньше _group_nonfinite_check и distributed_apply_step (для was_clipped)
печатали диагностику через jax.debug.print ПРЯМО ВНУТРИ jit-графа. Две
проблемы с этим:
  (a) jax.debug.print/jax.debug.callback открывают host-callback канал на
      КАЖДЫЙ фактический вызов -- на длинных сессиях (десятки тысяч шагов)
      это упирается в лимит открытых файлов рантайма (OSError: Too many
      open files), реально наблюдавшийся баг в этом проекте;
  (b) даже если бы это не падало, вывод шёл ТОЛЬКО в консоль Kaggle,
      которая не сохраняется между сессиями, и НИКОГДА не попадал в W&B --
      то есть самая ценная диагностика (какая именно группа параметров
      даёт non-finite, насколько близко global_norm подошёл к клипу)
      была невидима постфактум.
Обе диагностики теперь ЧИСТЫЕ функции: возвращают jnp-массивы/скаляры как
ОБЫЧНЫЕ ВЫХОДЫ jitted distributed_apply_step, без единого debug.print/
debug.callback внутри графа. Печать и W&B-логирование делаются на
host-стороне в train.py, в том же месте, где уже логируются train_loss/
aux_info -- ноль дополнительных host-callback каналов, полное покрытие
W&B.

ФИКС #4 (этот пасс): make_hybrid_optimizer теперь дополнительно
возвращает lr_schedule (0..1 множитель, до умножения на конкретный
base_lr каждой группы) -- нужно, чтобы train.py мог логировать
train/lr_scale в W&B без необходимости пересчитывать warmup/cosine-decay
формулу отдельно.

ФИКС #5 (этот пасс -- AttributeError: 'function' object has no attribute
'init'): optax.GradientTransformation сам является NamedTuple из двух
полей (init, update) -- то есть тоже обычный 2-tuple. Пока optimizer.py
не обновлён на фактический возврат (tx, lr_schedule) (ФИКС #4 выше
описывает НАМЕРЕНИЕ, но реализация в optimizer.py может отставать),
безусловная распаковка `tx, lr_schedule = make_hybrid_optimizer(...)` НЕ
падает с ошибкой сразу -- она молча распаковывает сам
GradientTransformation: tx становится функцией tx.init, а lr_schedule --
функцией tx.update. Дальше `tx.init(...)` закономерно падает с
"'function' object has no attribute 'init'", и трейсбек указывает на
СЛЕДУЮЩУЮ строку (jax.eval_shape(lambda: tx.init(...))), что маскирует
реальную причину. make_shard_and_compile теперь явно проверяет тип
результата make_hybrid_optimizer вместо слепой распаковки -- см.
make_shard_and_compile ниже.

ФИКС #6 (этот пасс -- router_temp runaway, см. test_synthetic_router_temp*.py
в чате): GmmMoEJ's router_temp (atomic_ops/moe_gmm.py, _ROUTER_TEMP_INIT=
10.0, hard-clipped to [1,15] только НА ЧТЕНИЕ через router_temp_clipped)
не имеет структурного сопротивления росту САМОГО параметра к потолку
клипа. Подтверждено на полностью изолированном синтетическом тесте (один
слой GmmMoEJ, случайный шум на входе, loss=mean(out**2), без реальных
данных, без GDN-2/Mamba2): router_temp детерминированно упирается в 15.0
за ~9000 шагов чисто из-за task-loss давления, независимо от датасета.

L2-штраф-в-loss (router_temp - init)**2 был опробован ПЕРВЫМ и отклонён:
под Adam адаптивная пер-параметрная нормализация шага (sqrt(v)/m) делает
ЭФФЕКТИВНЫЙ размер шага почти независимым от величины L2-коэффициента --
100-кратное изменение coef (1e-4 -> 1e-2) дало почти ИДЕНТИЧНЫЕ траектории
отката в синтетическом resume-shock тесте (старт=15.0, все три coef сошлись
к ~13.6 за 1500 шагов). Тот же класс проблемы, что уже встречался в этом
проекте с Muon weight decay (см. optimizer.py's _muon_step докстринг) --
там решено применением decay ВНЕ градиентного пути (после ортогонализации),
а не через loss.

Фикс здесь следует тому же паттерну "decoupled decay, не через градиент
loss": применяется прямо в пост-обработке параметров после
optax.apply_updates (distributed_apply_step, сразу после существующего
клипа параметров ±1e2), не через compute_loss/optimizer.py's tx.update().
Синтетический перебор (test_synthetic_router_temp_decoupled.py) подтвердил
decay_rate=0.02 как хороший баланс:
  - Сценарий A (рост от init под task-loss давлением, 3000 шагов):
    final=10.03, max=10.03 -- рост к потолку полностью подавлен.
  - Сценарий B (откат от уже упёршегося 15.0, симулирует чекпоинт ДО
    фикса, 1500 шагов): стабилизируется в [9,11] к шагу 81, БЕЗ overshoot
    ниже init (min=10.01) -- в отличие от decay_rate=0.001/0.005, которые
    либо не стабилизируются за 1500 шагов вовсе, либо стабилизируются
    слишком медленно (шаг 339), чтобы быть практически полезными сразу
    после реального resume.
decay_rate=0.05 был чуть быстрее (шаг 31) без видимого overshoot, но 0.02
оставляет больше запаса против более высокой дисперсии градиентов
реальных данных по сравнению с безшумным синтетическим тестом --
пересмотреть только если 0.02 окажется слишком медленным для удержания
роста на реальных TPU-прогонах.
"""
from __future__ import annotations

import glob
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from model import FullHybridMoEModel, ModelConfig, set_model_mesh
from optimizer import compute_loss, make_hybrid_optimizer
from utils import path_to_str

DATASET_FRACTION = 1
DATASET_FRACTION_SEED = 777

# ФИКС: автостоп при частых non-finite градиентах -- см. train.py, где эти
# константы реально используются в цикле обучения. Оставлены здесь же,
# рядом с диагностикой групп, которую они венчают логически, но train.py
# импортирует их напрямую отсюда, чтобы не дублировать значения в двух
# местах.
NONFINITE_CONSECUTIVE_LIMIT = 4
NONFINITE_WINDOW_SIZE = 15
NONFINITE_WINDOW_RATIO = 0.25

# ==========================================================================
# ФИКС #6 (router_temp runaway) -- см. докстринг модуля выше для полного
# обоснования (isolated synthetic test + отклонённый L2-в-loss вариант +
# принятый decoupled-decay вариант, decay_rate=0.02).
# ==========================================================================
ROUTER_TEMP_DECAY_RATE = 0.02
ROUTER_TEMP_INIT = 10.0  # должно совпадать с atomic_ops/moe_gmm.py's _ROUTER_TEMP_INIT


def _router_temp_decay_leaf(path, param):
    """tree_map_with_path leaf fn -- тот же паттерн, что _decay_scale_leaf
    ниже: определяет router_temp-листья по подстроке в пути, возвращает
    СТАВКУ decay для этого листа (0.0 для всех остальных -- т.е. no-op)."""
    path_str = path_to_str(path)
    return ROUTER_TEMP_DECAY_RATE if "router_temp" in path_str else 0.0


def apply_router_temp_decay(new_params, decay_map):
    """Decoupled decay-to-init ТОЛЬКО для router_temp-листьев, применяется
    ПОСЛЕ optax.apply_updates (т.е. после собственного шага Adam) -- см.
    докстринг модуля выше про то, почему это НЕ должно идти через
    градиент/loss. decay_map -- pytree той же формы, что new_params, со
    ROUTER_TEMP_DECAY_RATE на router_temp-листьях и 0.0 везде остальном
    (см. _router_temp_decay_leaf) -- строится один раз в
    make_shard_and_compile, тот же паттерн, что _decay_grad_scale, так что
    здесь остаётся чистый jax.tree_map внутри jitted distributed_apply_step
    (никакого Python-side string matching внутри jit-трейса)."""
    return jax.tree_util.tree_map(
        lambda p, rate: p - rate * (p - ROUTER_TEMP_INIT),
        new_params, decay_map,
    )


SESSION_TIME_BUDGET_SECONDS = 9 * 3600 - 5 * 60  # 9 часов минус запас на graceful stop


# ==========================================================================
# ДИАГНОСТИКА non-finite градиентов: относим каждый лист параметров к одной
# из "подозреваемых" групп (GDN-2, Mamba2, MLA, MoE, embed, остальное),
# затем на каждом шаге, где итоговый global_norm не конечен, ВОЗВРАЩАЕМ
# (не печатаем -- см. ФИКС #3 выше) булев вектор -- у КАКИХ ИМЕННО групп
# есть non-finite градиент.
# ==========================================================================
_DIAG_GROUPS = ("gdn2", "mamba2", "mla", "moe", "muon_decay", "embed", "other")


def _classify_leaf_group(path_str: str) -> str:
    if "gdn2" in path_str:
        return "gdn2"
    if "mamba2" in path_str:
        return "mamba2"
    if "mla" in path_str:
        return "mla"
    if "experts_block" in path_str or "moe" in path_str or "router" in path_str:
        return "moe"
    if "embed" in path_str or "lm_head" in path_str:
        return "embed"
    return "other"


def make_grad_group_map(params):
    """Строит pytree той же формы, что params/grads, где каждый лист -- это
    ИМЯ группы (питоновская строка, статична, не трейсится)."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _: _classify_leaf_group(path_to_str(path)), params
    )


def build_group_nonfinite_flags(grad_group_map):
    """ФИКС (заменяет старую build_group_nonfinite_check): возвращает
    ЧИСТУЮ функцию (avg_grads) -> jnp.ndarray формы (len(_DIAG_GROUPS),)
    bool -- по каждой группе: есть ли в ней хоть один non-finite лист.
    НИКАКОГО jax.debug.print/debug.callback внутри -- просто обычный
    JAX-массив, который distributed_apply_step возвращает как один из
    своих выходов (см. ФИКС #3 в докстринге модуля). Порядок элементов
    массива соответствует порядку _DIAG_GROUPS -- host-сторона (train.py)
    делает zip(_DIAG_GROUPS, flags) для печати/W&B."""
    leaves_g, _ = jax.tree_util.tree_flatten(grad_group_map)

    def _flags(avg_grads):
        leaves_grad = jax.tree_util.tree_leaves(avg_grads)
        flags = []
        for group in _DIAG_GROUPS:
            idxs = [i for i, g in enumerate(leaves_g) if g == group]
            if not idxs:
                flags.append(jnp.array(False))
                continue
            group_flags = jnp.stack([
                jnp.logical_not(jnp.all(jnp.isfinite(leaves_grad[i]))) for i in idxs
            ])
            flags.append(jnp.any(group_flags))
        return jnp.stack(flags)

    return _flags


def make_tpu_mesh():
    devices = jax.devices()
    n = len(devices)
    mesh_devices = mesh_utils.create_device_mesh((n,), devices)
    return Mesh(mesh_devices, axis_names=("tpu_nodes",))


def make_shard_and_compile(config: ModelConfig, total_steps: int, batch_size: int,
                           seq_len: int = 8192, accum_steps: int = 1):
    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]

    if batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by n_devices={n_devices}."
        )

    batch_axis = "tpu_nodes"
    set_model_mesh(mesh, batch_axis=batch_axis)

    # ФИКС #4: make_hybrid_optimizer теперь возвращает (tx, lr_schedule) --
    # lr_schedule прокидывается наружу для W&B-логирования в train.py
    # (train/lr_scale), не только для внутреннего использования optax'ом.
    #
    # ФИКС #5 (этот пасс, AttributeError: 'function' object has no attribute
    # 'init'): optax.GradientTransformation САМ является NamedTuple из двух
    # полей (init, update) -- то есть тоже обычный 2-tuple. Если
    # optimizer.py ещё не обновлён и make_hybrid_optimizer по-прежнему
    # возвращает ГОЛЫЙ tx (а не (tx, lr_schedule)), то безусловная
    # распаковка `tx, lr_schedule = make_hybrid_optimizer(...)` НЕ падает
    # с ошибкой -- она молча распаковывает сам GradientTransformation:
    # tx становится функцией tx.init, а lr_schedule -- функцией tx.update.
    # Дальше `tx.init(...)` закономерно падает с
    # "'function' object has no attribute 'init'" (именно эта ошибка и
    # наблюдалась -- трейсбек указывал на СЛЕДУЮЩУЮ строку,
    # jax.eval_shape(lambda: tx.init(...)), что маскировало реальную
    # причину). Явно проверяем тип результата вместо слепой распаковки.
    _opt_result = make_hybrid_optimizer(total_steps=total_steps)
    if isinstance(_opt_result, (optax.GradientTransformation, optax.GradientTransformationExtraArgs)):
        # optimizer.py ещё не обновлён -- вернул голый tx, lr_schedule нет.
        tx = _opt_result
        lr_schedule = None
        print("[OPTIMIZER] ⚠️ make_hybrid_optimizer вернул голый tx (без lr_schedule) -- "
              "train/lr_scale логироваться в W&B не будет, пока optimizer.py не обновлён "
              "на возврат (tx, lr_schedule).")
    else:
        tx, lr_schedule = _opt_result
    model = FullHybridMoEModel(cfg=config)

    init_rng = jax.random.PRNGKey(0)
    abstract_params = jax.eval_shape(
        lambda: model.init(init_rng, jnp.zeros((batch_size, seq_len), dtype=jnp.int32))
    )["params"]

    data_sharding = NamedSharding(mesh, P("tpu_nodes", None))

    MIN_SHARD_SIZE = 128

    def _get_shard_spec(path, param):
        if not hasattr(param, "shape") or param.ndim == 0:
            return NamedSharding(mesh, P())
        path_str = path_to_str(path)
        if ("experts_block" in path_str or "routed_experts" in path_str
                or "routed_w1" in path_str or "routed_w2" in path_str):
            return NamedSharding(mesh, P(*([None] * param.ndim)))
        best_axis, best_size = None, -1
        for i, size in enumerate(param.shape):
            if size % n_devices == 0 and (size // n_devices) >= MIN_SHARD_SIZE and size > best_size:
                best_axis, best_size = i, size
        if best_axis is None:
            return NamedSharding(mesh, P(*([None] * param.ndim)))
        spec = [None] * param.ndim
        spec[best_axis] = "tpu_nodes"
        return NamedSharding(mesh, P(*spec))

    param_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, abstract_params)

    grad_group_map = make_grad_group_map(abstract_params)
    _group_nonfinite_flags = build_group_nonfinite_flags(grad_group_map)

    def _decay_scale_leaf(path, param):
        path_str = path_to_str(path)
        return 0.2 if ("decay_a" in path_str or "a_log" in path_str) else 1.0

    _decay_grad_scale = jax.tree_util.tree_map_with_path(_decay_scale_leaf, abstract_params)

    # ФИКС #6 (router_temp runaway): decay-карта строится один раз здесь,
    # тем же паттерном, что _decay_grad_scale выше -- чистая pytree-структура
    # (0.0/ROUTER_TEMP_DECAY_RATE на листьях), без строковых операций
    # внутри jitted distributed_apply_step.
    _router_temp_decay_map = jax.tree_util.tree_map_with_path(
        _router_temp_decay_leaf, abstract_params
    )

    opt_state_abstract = jax.eval_shape(lambda: tx.init(abstract_params))
    opt_state_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, opt_state_abstract)

    def model_apply_wrapped(variables, input_ids, rngs=None, deterministic=True, **kwargs):
        return model.apply(
            variables, input_ids,
            rngs=rngs, deterministic=deterministic,
            **kwargs
        )

    def distributed_train_step_micro(p, s, b, r, accum_grads):
        def loss_fn(param):
            return compute_loss(
                param, model_apply_wrapped, b, config,
                rngs={"dropout": r},
                deterministic=False, return_aux=True,
                ce_chunk_size=2048,
            )
        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
        return p, s, new_accum, loss, aux_info

    def distributed_apply_step(p, s, accum_grads, n_accum):
        avg_grads = jax.tree_util.tree_map(lambda g: g / n_accum, accum_grads)

        # ФИКС #3: чистая функция, возвращает jnp-массив -- НЕ печатает
        # ничего внутри jit. Разбор/печать/W&B -- на host-стороне в
        # train.py, после device_get.
        group_nonfinite_flags = _group_nonfinite_flags(avg_grads)

        global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(avg_grads)))

        is_finite = jnp.isfinite(global_norm)
        safe_norm = jnp.where(is_finite, global_norm, 1.0)
        clip_factor = jnp.where(is_finite, jnp.minimum(1.0, 1.0 / (safe_norm + 1e-6)), 0.0)

        avg_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) * clip_factor,
            avg_grads,
        )

        avg_grads = jax.tree_util.tree_map(lambda g, s: g * s, avg_grads, _decay_grad_scale)

        updates, new_s = tx.update(avg_grads, s, p)
        new_p = optax.apply_updates(p, updates)

        new_p = jax.tree_util.tree_map(
            lambda pp: jnp.nan_to_num(jnp.clip(pp, -1e2, 1e2), nan=0.0, posinf=1e2, neginf=-1e2),
            new_p,
        )

        # ФИКС #6 (router_temp runaway): decoupled decay-to-init для
        # router_temp-листьев, вне градиентного пути -- см. докстринг
        # модуля / ROUTER_TEMP_DECAY_RATE выше. Применяется ПОСЛЕ клипа
        # параметров: клип держит |w|<=1e2 общим правилом на случай взрыва,
        # decay здесь решает узкую задачу -- удержать router_temp у
        # разумной рабочей точки (~init=10.0), а не только в пределах
        # overflow-safe диапазона. Никакого конфликта между ними: клип
        # применяется первым и почти никогда не заденет router_temp (его
        # диапазон [1,15] << 1e2), decay применяется вторым и трогает
        # ТОЛЬКО router_temp-листья (везде остальном decay_map=0.0, т.е.
        # no-op).
        new_p = apply_router_temp_decay(new_p, _router_temp_decay_map)

        # ФИКС #3: was_clipped тоже больше не печатается через
        # jax.lax.cond(...jax.debug.print...) -- просто bool-скаляр,
        # возвращаемый как обычный выход, ровно как is_finite уже был.
        was_clipped = jnp.any(jnp.stack([
            jnp.any(jnp.abs(leaf) >= 1e2) for leaf in jax.tree_util.tree_leaves(p)
        ]))

        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        return (new_p, new_s, zero_accum, is_finite,
                global_norm, clip_factor, group_nonfinite_flags, was_clipped)

    def distributed_val_step(p, b):
        return compute_loss(
            p, model_apply_wrapped, b, config,
            rngs=None,
            deterministic=True,
        )

    aux_info_sharding = {
        "ce_loss": NamedSharding(mesh, P()),
        "aux_loss": NamedSharding(mesh, P()),
        "z_loss": NamedSharding(mesh, P()),
        "expert_utilization": NamedSharding(mesh, P(None)),
        "moe_dropped_ratio": NamedSharding(mesh, P(None)),
        "router_temp": NamedSharding(mesh, P(None)),   # добавлено
    }

    compiled_train_micro = jax.jit(
        distributed_train_step_micro,
        donate_argnums=(0, 1, 4),
        in_shardings=(
            param_sharding,
            opt_state_sharding,
            {"input_ids": data_sharding, "labels": data_sharding},
            NamedSharding(mesh, P(None)),
            param_sharding,
        ),
        out_shardings=(
            param_sharding,
            opt_state_sharding,
            param_sharding,
            NamedSharding(mesh, P()),
            aux_info_sharding,
        ),
    )

    compiled_apply = jax.jit(
        distributed_apply_step,
        donate_argnums=(0, 1, 2),
        in_shardings=(
            param_sharding,
            opt_state_sharding,
            param_sharding,
            NamedSharding(mesh, P()),
        ),
        out_shardings=(
            param_sharding,
            opt_state_sharding,
            param_sharding,
            NamedSharding(mesh, P()),        # is_finite
            NamedSharding(mesh, P()),        # global_norm
            NamedSharding(mesh, P()),        # clip_factor
            NamedSharding(mesh, P(None)),    # group_nonfinite_flags, shape (len(_DIAG_GROUPS),)
            NamedSharding(mesh, P()),        # was_clipped
        ),
    )

    compiled_val = jax.jit(
        distributed_val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=NamedSharding(mesh, P()),
    )

    return (compiled_train_micro, compiled_apply, compiled_val, mesh, tx, model,
            param_sharding, opt_state_sharding, data_sharding, lr_schedule)


def resolve_source_files(output_dir, prefix):
    merged_ids = os.path.join(output_dir, f"{prefix}_input_ids.npy")
    merged_lbls = os.path.join(output_dir, f"{prefix}_labels.npy")
    if os.path.exists(merged_ids) and os.path.exists(merged_lbls):
        return [(merged_ids, merged_lbls)]

    shard_ids_paths = sorted(
        glob.glob(os.path.join(output_dir, f"{prefix}_shard_ids_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    pairs = []
    for ids_path in shard_ids_paths:
        lbls_path = ids_path.replace("_shard_ids_", "_shard_lbls_")
        if os.path.exists(lbls_path):
            pairs.append((ids_path, lbls_path))
    if not pairs:
        raise FileNotFoundError(
            f"Не найдены файлы для prefix={prefix!r} в {output_dir} -- ни объединённого "
            f"{prefix}_input_ids.npy, ни шардов {prefix}_shard_ids_*.npy. Проверьте путь."
        )
    return pairs


def build_manifest(file_pairs):
    """ФИКС: теперь принимает и 2-tuple (ids_path, lbls_path), и 3-tuple
    (ids_path, lbls_path, fraction) -- третий элемент здесь просто
    игнорируется (используется выше по стеку, в dataloader_multi_source, до
    вызова build_manifest), чтобы старые вызовы с 2-tuple продолжали
    работать без изменений."""
    manifest = []
    total = 0
    for entry in file_pairs:
        ids_path, lbls_path = entry[0], entry[1]
        n_rows = np.load(ids_path, mmap_mode="r").shape[0]
        manifest.append((ids_path, lbls_path, n_rows))
        total += n_rows
        print(f"[DATA] {os.path.basename(ids_path)}: {n_rows:,} блоков")
    print(f"[DATA] Комбинированный пул: {total:,} блоков из {len(manifest)} файл(ов)")
    return manifest


def dataloader_multi_source(file_pairs, batch_size, data_sharding, seq_len, val_split=0.05,
                             dataset_fraction=1.0, fraction_seed=777, skip_batches=0,
                             mode="mixed"):
    """
    file_pairs: список (ids_path, lbls_path) ИЛИ (ids_path, lbls_path, fraction)
    -- fraction (0.0-1.0) для ИМЕННО ЭТОГО источника, по умолчанию 1.0, если
    не указан (полная обратная совместимость со старым 2-tuple форматом).

    ФИКС (гипотеза: смешанные батчи из разнородных источников -- один из
    возможных факторов частого RESID-DIAG на layer=22/mamba2 при полном
    6-датасетном пуле против стабильных 4000-6000 шагов на 2 источниках --
    см. чат): три режима подачи данных, mode=

      "mixed" (default, СТАРОЕ поведение, byte-for-byte то же самое, если
        вызвать без явного mode=) -- все источники в одном глобально
        перемешанном пуле, один батч может содержать строки из НЕСКОЛЬКИХ
        источников сразу.

      "sequential" -- источники проходятся ПО ОЧЕРЕДИ целиком, в порядке
        file_pairs: сначала ВСЕ шаги первого источника (с локальным shuffle
        внутри источника на каждый повторный проход), потом полностью
        второй, и т.д. Ни один батч не смешивает разные источники, но
        модель подолгу (сотни-тысячи шагов) видит только один источник
        подряд -- ближе к curriculum learning, чем к обычному перемешанному
        обучению; риск временного смещения градиентного сигнала в сторону
        "текущего" источника, если LR всё ещё заметен (cosine decay ещё не
        близко к alpha).

      "round_robin" -- на каждом МИКРО-шаге ровно один источник, источники
        чередуются по кругу в порядке file_pairs (0,1,...,S-1,0,1,...).
        Внутри батча источники не смешиваются, но и не залипают надолго --
        за accum_steps подряд идущих микрошага эффективный шаг (после
        суммирования градиентов) обычно видит несколько РАЗНЫХ источников.
        ВАЖНО: посещает каждый источник с РАВНОЙ частотой (1/n_sources за
        цикл) НЕЗАВИСИМО от его размера -- в отличие от "mixed", где
        вероятность строки из источника ~ пропорциональна его размеру.
        Для маленьких источников (agentpack/rstar/syntheticcode) это
        эффективно ПЕРЕВЕШИВАЕТ их относительно природной доли -- нормально
        для короткого диагностического прогона, но не для финального
        полного обучения без явного контроля через per-source fraction.

    Каждый батч в mode="round_robin" несёт дополнительное поле
    "_source_idx" (индекс источника в file_pairs, для диагностики -- если
    сработает RESID-DIAG/non-finite, можно сохранить это поле рядом со
    снапшотом и сразу узнать источник-виновник).
    """
    normalized_pairs = []
    for entry in file_pairs:
        if len(entry) == 3:
            ids_path, lbls_path, frac = entry
        else:
            ids_path, lbls_path = entry
            frac = 1.0
        normalized_pairs.append((ids_path, lbls_path, float(frac)))

    manifest = build_manifest([(p[0], p[1]) for p in normalized_pairs])
    sizes = np.array([n for _, _, n in manifest])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    context_length = np.load(manifest[0][0], mmap_mode="r").shape[1]
    if context_length > seq_len:
        context_length = seq_len

    mmap_cache = {}

    def _get_mmap(path):
        arr = mmap_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            mmap_cache[path] = arr
        return arr

    def _gather_batch(global_indices):
        shard_of = np.searchsorted(offsets, global_indices, side="right") - 1
        ids_out = np.empty((len(global_indices), context_length), dtype=np.int32)
        lbls_out = np.empty((len(global_indices), context_length), dtype=np.int32)
        for s in np.unique(shard_of):
            m = shard_of == s
            local_idx = global_indices[m] - offsets[s]
            ids_path, lbls_path, _ = manifest[s]
            ids_full = _get_mmap(ids_path)[local_idx]
            lbls_full = _get_mmap(lbls_path)[local_idx]
            ids_out[m] = ids_full[:, :seq_len]
            lbls_out[m] = lbls_full[:, :seq_len]
        return ids_out, lbls_out

    # ---- per-source fraction: применяем К КАЖДОМУ источнику отдельно,
    # ДО train/val split и ДО глобального dataset_fraction ----
    frac_rng_per_source = np.random.RandomState(fraction_seed)
    per_source_idx = []
    for s, (ids_path, _, n) in enumerate(manifest):
        local_idx = np.arange(offsets[s], offsets[s + 1])
        frac = normalized_pairs[s][2]
        if frac < 1.0:
            n_keep = max(1, int(len(local_idx) * frac))
            local_idx = frac_rng_per_source.choice(local_idx, size=n_keep, replace=False)
            local_idx.sort()
        if frac != 1.0:
            print(f"[DATA] Источник {os.path.basename(ids_path)}: "
                  f"{frac*100:.0f}% -> {len(local_idx):,} из {int(n):,} блоков")
        per_source_idx.append(local_idx)

    all_idx = np.concatenate(per_source_idx)

    # ---- старый ГЛОБАЛЬНЫЙ dataset_fraction -- оставлен для обратной
    # совместимости, применяется ПОВЕРХ уже отфильтрованного по источникам ----
    if dataset_fraction < 1.0:
        frac_rng = np.random.RandomState(fraction_seed)
        n_keep = int(len(all_idx) * dataset_fraction)
        all_idx = frac_rng.choice(all_idx, size=n_keep, replace=False)
        all_idx.sort()
        print(f"[DATA] Общая подвыборка {dataset_fraction*100:.0f}%: {n_keep:,} блоков "
              f"(после per-source фильтра, seed={fraction_seed})")

    pool_size = len(all_idx)
    val_size = int(pool_size * val_split)
    train_size = pool_size - val_size

    split_rng = np.random.RandomState(42)
    shuffled = np.copy(all_idx)
    split_rng.shuffle(shuffled)
    train_idx_pool = shuffled[:train_size]
    val_idx_pool = shuffled[train_size:]

    train_pool_by_source = None
    if mode in ("sequential", "round_robin"):
        val_set = set(val_idx_pool.tolist())
        train_pool_by_source = []
        for s in range(len(manifest)):
            src_train_idx = np.array(
                [i for i in per_source_idx[s] if i not in val_set], dtype=np.int64
            )
            train_pool_by_source.append(src_train_idx)
            print(f"[DATA] (mode={mode}) источник #{s} ({os.path.basename(manifest[s][0])}): "
                  f"{len(src_train_idx):,} train-блоков")

    def _infinite_source_batches(src_idx, seed):
        """Бесконечный генератор ГОТОВЫХ (ids_np, lbls_np) батчей одного
        источника -- используется там, где пропуск уже пройденных
        микрошагов не нужен (или где чтение с диска на каждый шаг
        приемлемо). Оставлена как есть -- см. _infinite_source_indices
        ниже для варианта, который умеет пропускать без чтения с диска."""
        local_rng = np.random.RandomState(seed)
        idx_local = np.copy(src_idx)
        while True:
            local_rng.shuffle(idx_local)
            n_steps = len(idx_local) // batch_size
            for step in range(n_steps):
                batch_idx = idx_local[step * batch_size:(step + 1) * batch_size]
                yield _gather_batch(batch_idx)

    def _infinite_source_indices(src_idx, seed):
        """ФИКС: недостающая функция -- _round_robin_gen ниже вызывал
        _infinite_source_indices(...), которая нигде не была определена
        (NameError при первом же mode="round_robin"). Тот же паттерн, что
        _infinite_source_batches выше, но yield'ит СЫРЫЕ ИНДЕКСЫ
        (idx_local[...]), а не результат _gather_batch(...) -- это и
        позволяет _round_robin_gen пропускать уже пройденные микрошаги при
        resume БЕЗ обращения к диску на каждый из них: во время пропуска
        просто прокручивается RNG/цикл индексов, а _gather_batch (реальное
        mmap-чтение) вызывается только начиная с первого НЕ пропускаемого
        шага -- см. _round_robin_gen."""
        local_rng = np.random.RandomState(seed)
        idx_local = np.copy(src_idx)
        while True:
            local_rng.shuffle(idx_local)
            n_steps = len(idx_local) // batch_size
            for step in range(n_steps):
                yield idx_local[step * batch_size:(step + 1) * batch_size]

    def _round_robin_gen(pool_by_source, skip_first=0):
        src_gens = [_infinite_source_indices(pool_by_source[s], seed=1000 + s)
                    for s in range(len(pool_by_source))]
        n_sources = len(src_gens)
        step_i = 0
        if skip_first > 0:
            print(f"[DATA] (round_robin) Быстрый пропуск {skip_first} микрошагов "
                  f"(без чтения с диска)...")
        while True:
            s = step_i % n_sources
            batch_idx = next(src_gens[s])
            step_i += 1
            if step_i <= skip_first:
                continue
            ids_np, lbls_np = _gather_batch(batch_idx)
            yield {
                "input_ids": jax.device_put(jnp.array(ids_np), data_sharding),
                "labels": jax.device_put(jnp.array(lbls_np), data_sharding),
                "_source_idx": s,
            }

    def _sequential_gen(pool_by_source, skip_first=0):
        local_rng = np.random.RandomState(123)
        skip_remaining = skip_first
        first_pass = True
        while True:
            for s, src_idx in enumerate(pool_by_source):
                idx_local = np.copy(src_idx)
                local_rng.shuffle(idx_local)
                n_steps = len(idx_local) // batch_size
                start_step = 0
                if first_pass and skip_remaining > 0:
                    start_step = min(skip_remaining, n_steps)
                    skip_remaining -= start_step
                    if start_step > 0:
                        print(f"[DATA] Resume (sequential): пропускаем {start_step} "
                              f"микрошагов источника #{s}.")
                for step in range(start_step, n_steps):
                    batch_idx = idx_local[step * batch_size:(step + 1) * batch_size]
                    ids_np, lbls_np = _gather_batch(batch_idx)
                    yield {
                        "input_ids": jax.device_put(jnp.array(ids_np), data_sharding),
                        "labels": jax.device_put(jnp.array(lbls_np), data_sharding),
                        "_source_idx": s,
                    }
            first_pass = False

    def _mixed_gen(pool, is_train=True, skip_first=0):
        idx_local = np.copy(pool)
        local_rng = np.random.RandomState(123)
        first_pass = True
        while True:
            if is_train:
                local_rng.shuffle(idx_local)
            n_steps = len(idx_local) // batch_size
            start_step = 0
            if first_pass and is_train:
                start_step = skip_first % max(n_steps, 1)
                if start_step > 0:
                    print(f"[DATA] Resume: пропускаем первые {start_step} микрошагов "
                          f"текущего прохода датасета (уже были пройдены раньше).")
            first_pass = False
            for step in range(start_step, n_steps):
                batch_idx = idx_local[step * batch_size: (step + 1) * batch_size]
                ids_np, lbls_np = _gather_batch(batch_idx)
                yield {
                    "input_ids": jax.device_put(jnp.array(ids_np), data_sharding),
                    "labels": jax.device_put(jnp.array(lbls_np), data_sharding),
                }
            if not is_train:
                break

    if mode == "round_robin":
        train_gen = _round_robin_gen(train_pool_by_source, skip_first=skip_batches)
    elif mode == "sequential":
        train_gen = _sequential_gen(train_pool_by_source, skip_first=skip_batches)
    elif mode == "mixed":
        train_gen = _mixed_gen(train_idx_pool, True, skip_first=skip_batches)
    else:
        raise ValueError(f"Неизвестный mode={mode!r}, ожидается 'mixed'/'sequential'/'round_robin'.")

    return (
        train_gen,
        lambda: _mixed_gen(val_idx_pool, False),
        train_size // batch_size,
        val_size // batch_size,
    )
