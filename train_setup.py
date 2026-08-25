"""
train_setup.py -- диагностика non-finite по группам параметров, TPU mesh /
шардинг / компиляция train-step'ов, multi-source dataloader.

Вынесено из train.py.

ФИКС (dataloader, по гипотезе "смешение источников в одном батче триггерит
RESID-DIAG на layer=22/mamba2" -- см. чат): dataloader_multi_source теперь
поддерживает три режима подачи данных (mode="mixed"/"sequential"/
"round_robin") и per-source fraction (третий элемент в file_pairs).

ФИКС #2 (этот пасс): _round_robin_gen переписан на "быстрый пропуск без
чтения с диска".

ФИКС #3 (этот пасс -- W&B диагностика без jax.debug.print/callback):
group_nonfinite_flags и was_clipped -- чистые функции, возвращают
jnp-массивы/скаляры как ОБЫЧНЫЕ ВЫХОДЫ jitted distributed_apply_step.

ФИКС #4 (этот пасс): make_hybrid_optimizer теперь дополнительно
возвращает lr_schedule.

ФИКС #5 (этот пасс -- AttributeError: 'function' object has no attribute
'init'): make_shard_and_compile явно проверяет тип результата
make_hybrid_optimizer вместо слепой распаковки.

ФИКС #6 (этот пасс -- router_temp runaway): GmmMoEJ's router_temp
decoupled decay-to-init в apply_router_temp_decay.

ФИКС #7 (этот пасс -- bias-балансировка экспертов, DeepSeek-V3 style):
GmmMoEJ's expert_bias, decoupled update в apply_expert_bias_update.

ФИКС #8 (этот пасс -- переключаемый WARMUP_FREEZE_STEP, см. chat):
make_shard_and_compile теперь принимает warmup_freeze_step и прокидывает
его в make_hybrid_optimizer(warmup_freeze_step=...) -- раньше это была
жёсткая модульная константа WARMUP_FREEZE_STEP внутри optimizer.py,
теперь train.py явно решает (USE_WARMUP_FREEZE/WARMUP_FREEZE_STEP_VALUE),
None означает обычный полный warmup/cosine-decay без заморозки.

ФИКС #9 (этот пасс -- видимость zclip_skip, см. chat: градиентная норма
монотонно растёт при стабильном замороженном LR, ce_loss растёт синхронно,
остальные группы чисты -- гипотеза "систематически трудный сегмент
данных"): distributed_apply_step теперь ДОПОЛНИТЕЛЬНО возвращает сырые
поля ZClipState (через optimizer.extract_zclip_diagnostics) ПОСЛЕ
tx.update -- раньше было невозможно отличить "zclip реально зануляет
часть шагов" от "clip_by_global_norm просто масштабирует растущий, но
каждый раз применяемый градиент", потому что is_spike/z/drift_ratio
нигде не логировались. host-side (train.py) пересчитывает z-score и
drift_ratio из этих сырых полей ТЕМ ЖЕ способом, что update_fn внутри
zclip_skip -- не дублирует логику принятия решения (is_spike), только
воспроизводит формулу для логирования.

ФИКС #10 (этот пасс -- ВСЕГДА включённая по-слойная диагностика, см.
чат): добавлен diagnostics.py -- гранулярность "один тег на физический
(block_idx, layer_idx) слой" вместо "gdn2"/"mamba2"/"mla" одной кучей.
distributed_apply_step теперь дополнительно возвращает:
  - layer_grad_norms/layer_grad_maxabs/layer_grad_nonfinite -- по
    avg_grads, тегированным через diagnostics.make_leaf_layer_map
  - layer_w_norms/layer_w_maxabs/layer_w_nonfinite -- то же самое, но по
    ВЕСАМ (new_p) после апдейта -- отдельный сигнал "разбухает ли слой
    сам по себе", независимый от того, что градиент показывает В МОМЕНТ
  - muon_orth_resid -- см. optimizer.py's ФИКС #4 (extract_muon_diagnostics)
_PARAM_LAYER_TAGS (regex по путям params, "b{N}_l{M}") и _SOW_LAYER_TAGS
(из diagnostics.layer_tags_in_sow_order, совпадает с порядком, в котором
model.py реально sow'ит layer_delta_maxabs/layer_resid_maxabs) оба
экспортируются -- train.py использует их как подписи столбцов в W&B.
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
from optimizer import compute_loss, make_hybrid_optimizer, extract_zclip_diagnostics, extract_muon_diagnostics
from utils import path_to_str
from diagnostics import (
    make_leaf_layer_map, param_layer_tags, layer_tags_in_sow_order, build_leaf_stats_fn,
)

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
# ФИКС #6 (router_temp runaway) -- см. модульный докстринг выше.
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
    ПОСЛЕ optax.apply_updates."""
    return jax.tree_util.tree_map(
        lambda p, rate: p - rate * (p - ROUTER_TEMP_INIT),
        new_params, decay_map,
    )


# ==========================================================================
# ФИКС #7 (bias-балансировка экспертов, DeepSeek-V3 style) -- см. модульный
# докстринг выше.
# ==========================================================================
EXPERT_BIAS_GAMMA = 0.02


def _build_expert_bias_index_map(abstract_params):
    """Присваивает каждому expert_bias-листу СТАТИЧЕСКИЙ (Python int, не
    jnp-массив) индекс, в порядке обхода params-pytree через
    tree_map_with_path."""
    counter = {"i": 0}

    def _mark(path, leaf):
        path_str = path_to_str(path)
        if "expert_bias" in path_str and hasattr(leaf, "shape"):
            idx = counter["i"]
            counter["i"] += 1
            return idx
        return -1

    return jax.tree_util.tree_map_with_path(_mark, abstract_params)


def apply_expert_bias_update(new_params, bias_index_map, assignment_frac_stacked, gamma=EXPERT_BIAS_GAMMA):
    """Decoupled, вне градиента, обновление expert_bias-листьев по правилу
    DeepSeek-V3."""
    if assignment_frac_stacked is None:
        return new_params

    def _update(p, idx):
        if idx < 0:
            return p
        frac = assignment_frac_stacked[idx]
        e_routed = p.shape[-1]
        target = 1.0 / e_routed
        overloaded = frac > target
        return p + gamma * jnp.where(overloaded, -1.0, 1.0)

    return jax.tree_util.tree_map(_update, new_params, bias_index_map)


SESSION_TIME_BUDGET_SECONDS = 9 * 3600 - 5 * 60  # 9 часов минус запас на graceful stop


# ==========================================================================
# ДИАГНОСТИКА non-finite градиентов: относим каждый лист параметров к одной
# из "подозреваемых" групп (GDN-2, Mamba2, MLA, MoE, embed, остальное),
# затем на каждом шаге, где итоговый global_norm не конечен, ВОЗВРАЩАЕМ
# (не печатаем -- см. ФИКС #3 выше) булев вектор -- у КАКИХ ИМЕННО групп
# есть non-finite градиент.
#
# ФИКС #10 (см. модульный докстринг): эта группировка ("gdn2" одной кучей
# на 16+ слоёв) остаётся КАК ЕСТЬ (дёшево, полезно как быстрый top-level
# фильтр), но ДОПОЛНЕНА diagnostics.py's более гранулярной по-физическому-
# слою группировкой ниже -- используйте _DIAG_GROUPS для "какой ТИП слоя
# затронут", и _PARAM_LAYER_TAGS/layer_grad_* для "какой ИМЕННО слой".
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
    """Возвращает ЧИСТУЮ функцию (avg_grads) -> jnp.ndarray формы
    (len(_DIAG_GROUPS),) bool -- по каждой группе: есть ли в ней хоть
    один non-finite лист."""
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


# ФИКС #10: заполняется внутри make_shard_and_compile (нужен abstract_params,
# которые строятся там же) и экспортируется как модульный атрибут -- train.py
# импортирует их напрямую (`from train_setup import _PARAM_LAYER_TAGS,
# _SOW_LAYER_TAGS`) уже ПОСЛЕ первого вызова make_shard_and_compile.
_PARAM_LAYER_TAGS = None
_SOW_LAYER_TAGS = None


def make_shard_and_compile(config: ModelConfig, total_steps: int, batch_size: int,
                           seq_len: int = 8192, accum_steps: int = 1,
                           warmup_freeze_step=None):
    """ФИКС #8 (переключаемый WARMUP_FREEZE_STEP, см. модульный докстринг
    выше): warmup_freeze_step прокидывается напрямую в
    make_hybrid_optimizer(warmup_freeze_step=...). None -- обычный полный
    warmup/cosine-decay без заморозки; int -- LR-schedule заморожена на
    этом шаге (см. optimizer.py's make_hybrid_optimizer докстринг)."""
    global _PARAM_LAYER_TAGS, _SOW_LAYER_TAGS

    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]

    if batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by n_devices={n_devices}."
        )

    batch_axis = "tpu_nodes"
    set_model_mesh(mesh, batch_axis=batch_axis)

    _opt_result = make_hybrid_optimizer(total_steps=total_steps, warmup_freeze_step=warmup_freeze_step)
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

    # ФИКС #10: по-физическому-слою тегирование + stats-функции (см.
    # diagnostics.py и модульный докстринг выше). Работает и для grads
    # (avg_grads), и для params (new_p) -- та же листовая структура.
    grad_layer_map = make_leaf_layer_map(abstract_params)
    _PARAM_LAYER_TAGS = param_layer_tags(grad_layer_map)
    _SOW_LAYER_TAGS = layer_tags_in_sow_order(config)
    _layer_grad_stats_fn = build_leaf_stats_fn(grad_layer_map, _PARAM_LAYER_TAGS)
    _layer_weight_stats_fn = build_leaf_stats_fn(grad_layer_map, _PARAM_LAYER_TAGS)
    print(f"[DIAG] По-слойная диагностика (ФИКС #10): {len(_PARAM_LAYER_TAGS)} физических "
          f"тегов параметров ({_PARAM_LAYER_TAGS}), {len(_SOW_LAYER_TAGS)} sown-тегов активаций.")

    def _decay_scale_leaf(path, param):
        path_str = path_to_str(path)
        return 0.2 if ("decay_a" in path_str or "a_log" in path_str) else 1.0
    def _router_scale_leaf(path, param):
        path_str = path_to_str(path)
        return 0.3 if "router" in path_str else 1.0

    _router_grad_scale = jax.tree_util.tree_map_with_path(_router_scale_leaf, abstract_params)

    _decay_grad_scale = jax.tree_util.tree_map_with_path(_decay_scale_leaf, abstract_params)

    _router_temp_decay_map = jax.tree_util.tree_map_with_path(
        _router_temp_decay_leaf, abstract_params
    )

    _expert_bias_index_map = _build_expert_bias_index_map(abstract_params)

    opt_state_abstract = jax.eval_shape(lambda: tx.init(abstract_params))
    opt_state_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, opt_state_abstract)

    def model_apply_wrapped(variables, input_ids, rngs=None, deterministic=True, **kwargs):
        return model.apply(
            variables, input_ids,
            rngs=rngs, deterministic=deterministic,
            **kwargs
        )

    def distributed_train_step_micro(p, s, b, r, accum_grads, collinearity_coef=None):
        def loss_fn(param):
            return compute_loss(
                param, model_apply_wrapped, b, config,
                rngs={"dropout": r},
                deterministic=False, return_aux=True,
                ce_chunk_size=2048,
                collinearity_coef=collinearity_coef,
            )
        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        micro_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
        )
        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
        return p, s, new_accum, loss, aux_info, micro_grad_norm

    def distributed_apply_step(p, s, accum_grads, n_accum, assignment_frac_stacked=None):
        avg_grads = jax.tree_util.tree_map(lambda g: g / n_accum, accum_grads)

        # ФИКС #3: чистая функция, возвращает jnp-массив -- НЕ печатает
        # ничего внутри jit. Разбор/печать/W&B -- на host-стороне в
        # train.py, после device_get.
        group_nonfinite_flags = _group_nonfinite_flags(avg_grads)

        # ФИКС #10: по-физическому-слою норма/max|abs|/nonfinite ГРАДИЕНТА
        # -- на СЫРОМ avg_grads (до nan_to_num-санитизации ниже), чтобы
        # nonfinite-флаг был содержательным (после nan_to_num всё уже
        # искусственно конечно).
        layer_grad_norms, layer_grad_maxabs, layer_grad_nonfinite = _layer_grad_stats_fn(avg_grads)

        global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(avg_grads)))
        is_finite = jnp.isfinite(global_norm)
        # clip_factor теперь только для логов/диагностики
        safe_norm = jnp.where(is_finite, global_norm, 1.0)
        clip_factor = jnp.where(is_finite, jnp.minimum(1.0, 1.0 / (safe_norm + 1e-6)), 0.0)

        # only sanitize non-finite, без дополнительного ручного clip (optax clip уже есть в tx)
        avg_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
            avg_grads,
        )
        avg_grads = jax.tree_util.tree_map(lambda g, s: g * s, avg_grads, _decay_grad_scale)
        avg_grads = jax.tree_util.tree_map(lambda g, s: g * s, avg_grads, _router_grad_scale)

        updates, new_s = tx.update(avg_grads, s, p)
        new_p = optax.apply_updates(p, updates)

        new_p = jax.tree_util.tree_map(
            lambda pp: jnp.nan_to_num(jnp.clip(pp, -1e2, 1e2), nan=0.0, posinf=1e2, neginf=-1e2),
            new_p,
        )

        new_p = apply_router_temp_decay(new_p, _router_temp_decay_map)
        new_p = apply_expert_bias_update(new_p, _expert_bias_index_map, assignment_frac_stacked)

        # ФИКС #10: по-физическому-слою норма/max|abs|/nonfinite ВЕСОВ
        # (new_p) ПОСЛЕ апдейта -- независимый сигнал "разбухает ли слой
        # сам по себе", в дополнение к градиенту В МОМЕНТ.
        layer_w_norms, layer_w_maxabs, layer_w_nonfinite = _layer_weight_stats_fn(new_p)

        # ФИКС #10 (см. optimizer.py's ФИКС #4): Muon orth_resid ПОСЛЕ
        # tx.update -- относится к апдейту, реально применённому на этом
        # шаге.
        muon_orth_resid = extract_muon_diagnostics(new_s)

        # ФИКС #3: was_clipped тоже больше не печатается через
        # jax.lax.cond(...jax.debug.print...) -- просто bool-скаляр,
        # возвращаемый как обычный выход, ровно как is_finite уже был.
        was_clipped = jnp.any(jnp.stack([
            jnp.any(jnp.abs(leaf) >= 1e2) for leaf in jax.tree_util.tree_leaves(p)
        ]))

        # ФИКС #9 (видимость zclip_skip, см. модульный докстринг выше):
        # сырые поля ZClipState ПОСЛЕ tx.update -- host-side (train.py)
        # пересчитывает z-score/drift_ratio из них тем же способом, что
        # zclip_skip's update_fn, для логирования (НЕ для повторного
        # принятия решения -- решение уже принято внутри tx.update).
        zclip_diag = extract_zclip_diagnostics(new_s)

        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        return (new_p, new_s, zero_accum, is_finite,
                global_norm, clip_factor, group_nonfinite_flags, was_clipped,
                zclip_diag,
                layer_grad_norms, layer_grad_maxabs, layer_grad_nonfinite,
                layer_w_norms, layer_w_maxabs, layer_w_nonfinite,
                muon_orth_resid)

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
        "router_temp": NamedSharding(mesh, P(None)),
        "min_col_norm": NamedSharding(mesh, P(None)),
        "max_abs_logit_preclip": NamedSharding(mesh, P(None)),
        "norm_x_mean": NamedSharding(mesh, P(None)),
        "norm_x_max": NamedSharding(mesh, P(None)),
        "norm_x_min": NamedSharding(mesh, P(None)),
        "router_max_cos_per_layer": NamedSharding(mesh, P()),
        "router_max_cos": NamedSharding(mesh, P()),
        "assignment_frac": NamedSharding(mesh, P(None, None)),
        # ФИКС #10: sharding-записи для ВСЕХ новых ВСЕГДА-включённых
        # sow-диагностик из model.py -- optimizer.py's compute_loss
        # собирает их под ЭТИМИ ЖЕ именами в aux_info, jit's out_shardings
        # требует точного совпадения структуры pytree.
        "layer_delta_maxabs": NamedSharding(mesh, P(None)),
        "layer_delta_isfinite": NamedSharding(mesh, P(None)),
        "layer_resid_maxabs": NamedSharding(mesh, P(None)),
        "layer_resid_isfinite": NamedSharding(mesh, P(None)),
        "mamba2_input_maxabs": NamedSharding(mesh, P(None)),
        "mamba2_input_isfinite": NamedSharding(mesh, P(None)),
        "mamba2_A_maxabs": NamedSharding(mesh, P(None)),
        "mamba2_ssm_out_pre_norm_maxabs": NamedSharding(mesh, P(None)),
        "mamba2_ssm_out_pre_norm_isfinite": NamedSharding(mesh, P(None)),
        "mamba2_ssm_out_maxabs": NamedSharding(mesh, P(None)),
        "gdn2_input_maxabs": NamedSharding(mesh, P(None)),
        "gdn2_input_isfinite": NamedSharding(mesh, P(None)),
        "gdn2_decay_a_maxabs": NamedSharding(mesh, P(None)),
        "gdn2_raw_out_maxabs": NamedSharding(mesh, P(None)),
        "gdn2_raw_out_isfinite": NamedSharding(mesh, P(None)),
        "gdn2_h_final_maxabs": NamedSharding(mesh, P(None)),
        "gdn2_out_maxabs": NamedSharding(mesh, P(None)),
        "mla_input_maxabs": NamedSharding(mesh, P(None)),
        "mla_out_maxabs": NamedSharding(mesh, P(None)),
        "final_hidden_maxabs": NamedSharding(mesh, P(None)),
        "final_hidden_isfinite": NamedSharding(mesh, P(None)),
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
            NamedSharding(mesh, P()),   # <-- для collinearity_coef (скаляр)

        ),
        out_shardings=(
            param_sharding,
            opt_state_sharding,
            param_sharding,
            NamedSharding(mesh, P()),
            aux_info_sharding,
            NamedSharding(mesh, P()),        # micro_grad_norm (скаляр)
        ),
    )

    # ФИКС #9: zclip_diag -- девятый выход distributed_apply_step, dict
    # скаляров той же формы, что ZClipState.
    zclip_diag_sharding = {
        "ema_mean": NamedSharding(mesh, P()),
        "ema_var": NamedSharding(mesh, P()),
        "warm_count": NamedSharding(mesh, P()),
        "slow_ema_mean": NamedSharding(mesh, P()),
        "slow_warm_count": NamedSharding(mesh, P()),
    }

    compiled_apply = jax.jit(
        distributed_apply_step,
        donate_argnums=(0, 1, 2),
        in_shardings=(
            param_sharding,
            opt_state_sharding,
            param_sharding,
            NamedSharding(mesh, P()),
            NamedSharding(mesh, P(None, None)),   # <-- assignment_frac_stacked
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
            zclip_diag_sharding,              # ФИКС #9: zclip diagnostics
            # ФИКС #10: по-физическому-слою диагностика (grads, weights) +
            # Muon orth_resid.
            NamedSharding(mesh, P(None)),    # layer_grad_norms
            NamedSharding(mesh, P(None)),    # layer_grad_maxabs
            NamedSharding(mesh, P(None)),    # layer_grad_nonfinite
            NamedSharding(mesh, P(None)),    # layer_w_norms
            NamedSharding(mesh, P(None)),    # layer_w_maxabs
            NamedSharding(mesh, P(None)),    # layer_w_nonfinite
            NamedSharding(mesh, P()),        # muon_orth_resid
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
    """Принимает и 2-tuple (ids_path, lbls_path), и 3-tuple
    (ids_path, lbls_path, fraction)."""
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
    file_pairs: список (ids_path, lbls_path) ИЛИ (ids_path, lbls_path, fraction).

    mode="mixed" (default) -- все источники в одном глобально перемешанном
      пуле, один батч может содержать строки из НЕСКОЛЬКИХ источников.
    mode="sequential" -- источники проходятся ПО ОЧЕРЕДИ целиком.
    mode="round_robin" -- на каждом МИКРО-шаге ровно один источник,
      источники чередуются по кругу, посещает каждый источник с РАВНОЙ
      частотой независимо от размера, батч несёт "_source_idx".
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
        local_rng = np.random.RandomState(seed)
        idx_local = np.copy(src_idx)
        while True:
            local_rng.shuffle(idx_local)
            n_steps = len(idx_local) // batch_size
            for step in range(n_steps):
                batch_idx = idx_local[step * batch_size:(step + 1) * batch_size]
                yield _gather_batch(batch_idx)

    def _infinite_source_indices(src_idx, seed):
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
