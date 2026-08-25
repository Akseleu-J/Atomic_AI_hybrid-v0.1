"""
train.py -- главный orchestration-цикл обучения.

ФИКС (разбивка файла): раньше все ~1400 строк -- HF-релей, orbax
чекпоинтинг, диагностика non-finite по группам, mesh/шардинг/компиляция,
dataloader и сам цикл обучения -- жили в одном файле, что усложняло навигацию.
Разнесено на:
  - checkpointing.py  -- HF Hub relay + orbax save/restore/async
  - train_setup.py    -- диагностика групп, mesh/shard/compile, dataloader
  - wandb_logging.py  -- W&B логирование (новое)
  - diagnostics.py    -- ФИКС #10 (этот пасс): по-физическому-слою
    статистика + host-side EMA burst-трекер (новое)
  - train.py (этот файл) -- только main_execution(), сам цикл обучения

ФИКС (этот пасс -- САМЫЙ ВАЖНЫЙ, см. chat): opt_state (momentum AdamW/
Lion/Muon + внутренние step-счётчики "count", по которым optimizer.py's
lr_schedule/resume_backoff вычисляют позицию на кривой warmup/cosine)
РЕАЛЬНО восстанавливается из HF-чекпоинта при resume, а не пересоздаётся
с нуля целиком через tx.init(params).

Раньше было ДВА независимых источника нестабильности сразу после каждого
resume, которые складывались: (1) opt_state пересоздавался с нуля --
холодный momentum на всех группах параметров; (2) lr_schedule's warmup
считался от opt_state's внутреннего "count" (не от global_step), который
после пересоздания тоже стоял на 0 -- warmup ПЕРЕЗАПУСКАЛСЯ заново на
каждой Kaggle-сессии. Комбинация этих двух факторов (быстро растущий LR +
абсолютно холодный оптимизатор) и была вероятной причиной router collapse/
non-finite взрывов, стабильно ловившихся на одном и том же ЛОКАЛЬНОМ (не
глобальном) диапазоне шагов после resume, независимо от того, на каком
глобальном шаге сессия стартовала.

Была опробована промежуточная версия фикса (_resume_opt_state_count --
принудительно выставлять "count" в global_step СРАЗУ ПОСЛЕ полного
tx.init()), но она оказалась хуже исходного поведения -- откачена.

Правильное решение -- восстанавливать НАСТОЯЩИЙ opt_state тем же принципом
"graft-merge", что уже применялся для params (_generic_pytree_merge).

ОБНОВЛЕНИЕ (этот пасс -- см. chat, дальнейшее расследование инцидента
"non-finite embed на global_step=12001"): эксперимент показал, что взрыв
воспроизводится ОДИНАКОВО и с холодным, и с тёплым opt_state -- то есть
opt_state/LR-позиция ИСКЛЮЧЕНЫ как причина ЭТОГО конкретного инцидента.
Реальный источник -- несанитизированный путь final_hidden -> embed.T
матмул в optimizer.py's chunked_cross_entropy (см. optimizer.py's ФИКС
"страховка"). Тем не менее WARMUP_FREEZE_STEP остаётся полезной
консервативной мерой сам по себе (ниже общий эффективный LR снижает риск
ЛЮБОГО другого спайка) -- см. USE_WARMUP_FREEZE ниже, теперь явный
переключаемый флаг вместо жёсткой константы в optimizer.py.

ФИКС (этот пасс -- переключаемый WARMUP_FREEZE_STEP): USE_WARMUP_FREEZE /
WARMUP_FREEZE_STEP_VALUE -- теперь явные флаги здесь, в train.py, а не
жёстко зашитая константа в optimizer.py. opt_state's "count" продолжает
расти честно НЕЗАВИСИМО от этого флага (graft-merge restore не завязан на
него вообще) -- флаг влияет только на формулу lr_schedule(...) на
следующем шаге. ВАЖНО: переключение USE_WARMUP_FREEZE=True -> False ПОСЛЕ
долгого пребывания в замороженном состоянии (count уже далеко за
freeze_step) -- это одномоментный СКАЧОК LR вверх (lr_schedule(count)
внезапно начинает использовать реальный, уже далеко ушедший count вместо
потолка), а не плавный переход -- переключать осознанно, не как рефлекс,
и предпочтительно на fresh start / при явном намерении изменить LR-режим,
а не посреди воспроизводимого инцидента.

ФИКС (этот пасс, has_nan -> isfinite): предыдущая проверка восстановленных
params использовала jnp.isnan, которая пропускает +-inf (нашли при
расследовании non-finite embed -- params печатались как "здоровые" даже
если бы там был inf, а не nan). Заменено на jnp.isfinite.

ФИКС (этот пасс, dataloader, per-source fraction как гиперпараметры):
SOURCE_FRACTIONS рядом с file_pairs.

ФИКС (этот пасс, host-side group-diagnostics + W&B): per-group non-finite
флаги, global grad norm, clip factor, флаг клипа параметров -- обычные
outputs compiled_apply, разбираются и логируются в W&B на host-стороне.

ФИКС (этот пасс -- router collapse на чекпоинте шага 2000): RESET_ROUTER_ON_RESUME
-- после появления graft-merge (и для params, и теперь для opt_state)
остаётся no-op предупреждением.

ФИКС (этот пасс -- structural realign при несовпадении длины chain-tuple):
_generic_pytree_merge's list/tuple ветка при несовпадении ДЛИНЫ tuple
пытается сопоставить элементы ПО СТРУКТУРНОЙ СИГНАТУРЕ (_structural_tuple_realign),
а не безусловно отбрасывать весь tuple на fresh.

ФИКС (этот пасс -- ВСЕГДА включённая по-слойная диагностика "на лету",
100% покрытие, см. chat): compiled_apply теперь дополнительно возвращает
layer_grad_norms/layer_grad_maxabs/layer_grad_nonfinite,
layer_w_norms/layer_w_maxabs/layer_w_nonfinite (см. train_setup.py's
ФИКС #10, diagnostics.py) и muon_orth_resid (см. optimizer.py's ФИКС #4)
-- все логируются в W&B КАЖДЫЙ эффективный шаг, по КАЖДОМУ физическому
(block_idx, layer_idx) слою отдельно, плюс host-side EMA burst-tracker
(HostLayerBurstTracker) ловит момент, когда ОДИН конкретный слой начинает
отклоняться от своей же базовой линии -- раньше, чем это дойдёт до
агрегатного burst_streak/global_norm. aux_info's новые sown-активационные
диагностики (layer_delta_maxabs/layer_resid_maxabs и т.д. из model.py)
логируются рядом, с теми же именами столбцов, что и физические grad/weight
теги -- см. diagnostics.layer_tags_in_sow_order.
"""
import os
import time
import json
import signal
import shutil
import sys

from collections import deque
from dataclasses import asdict

import wandb

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from checkpointing import (
    make_manager, save_slot, save_slot_async, _finalize_pending_save,
    upload_slot, download_slot, _HAS_HF,
    CHECKPOINT_EVERY_SECONDS, HF_LATEST_KEEP_N,
)
import train_setup
from train_setup import (
    make_shard_and_compile, dataloader_multi_source, build_manifest,
    DATASET_FRACTION, DATASET_FRACTION_SEED,
    NONFINITE_CONSECUTIVE_LIMIT, NONFINITE_WINDOW_SIZE, NONFINITE_WINDOW_RATIO,
    SESSION_TIME_BUDGET_SECONDS,
    _DIAG_GROUPS,   # ФИКС: нужно для host-side разбора group_nonfinite_flags
)
import wandb_logging
from diagnostics import HostLayerBurstTracker

from model import ModelConfig, get_model_mesh
from utils import path_to_str  # ФИКС (router-reset): нужно для поиска router-листьев в params/opt_state
# ФИКС (pjit in_shardings length mismatch): нужен тот же коэффициент,
# что optimizer.py's compute_loss уже использует как дефолт для
# collinearity_coef -- см. докстринг модуля выше.
from optimizer import ROUTER_COLLINEARITY_COEF, DEFAULT_WARMUP_FREEZE_STEP

# ==========================================================================
# Источник RESUME (откуда восстанавливаться) отделён от места SAVE
# (куда сохранять) — это две независимые вещи, а не один флаг:
#   - resume: последний ЗДОРОВЫЙ чекпоинт лежит в HF Bucket (сохранён туда
#     вручную/аварийно), путь указывается явно.
#   - save: основная модель должна продолжать копиться в HF Models
#     (repo_id из Kaggle secret HF_REPO_ID) — upload_slot() это уже делает,
#     трогать не нужно.
# ==========================================================================
RESUME_SOURCE = "hf_model"     # "bucket" | "hf_model" | "local_only"
RESUME_BUCKET_ID = "atomic-ai-labs/atomic-light-v0.5-bucket"   # <-- ваш реальный bucket id
RESUME_BUCKET_SUBDIR = "best_train"# <-- какой слот внутри бакета

# ==========================================================================
# ФИКС (этот пасс -- переключаемый WARMUP_FREEZE_STEP, см. модульный
# докстринг выше и optimizer.py's ФИКС #2 у make_hybrid_optimizer):
#   USE_WARMUP_FREEZE=True  -> LR-schedule заморожена на
#       WARMUP_FREEZE_STEP_VALUE (текущий осторожный режим, было жёстко
#       зашито как WARMUP_FREEZE_STEP=1000 в optimizer.py).
#   USE_WARMUP_FREEZE=False -> обычный полный warmup/cosine-decay без
#       заморозки, LR-schedule использует честный opt_state's "count"
#       напрямую.
# opt_state's "count" ВСЕГДА растёт честно независимо от этого флага --
# graft-merge restore (_generic_pytree_merge ниже) не завязан на него.
# Флаг только меняет, ЧТО подставляется в lr_schedule(...) на следующем
# шаге -- безопасно менять между сессиями, НЕБЕЗОПАСНО дёргать посреди
# воспроизводимого инцидента как попытку "магически" его погасить (см.
# докстринг выше про риск скачка LR при True->False после долгой
# заморозки).
# ==========================================================================
USE_WARMUP_FREEZE = True
WARMUP_FREEZE_STEP_VALUE = DEFAULT_WARMUP_FREEZE_STEP  # используется только если USE_WARMUP_FREEZE=True

# ==========================================================================
# ФИКС (этот пасс -- по-слойный burst-tracker, см. модульный докстринг
# выше): порог/decay для HostLayerBurstTracker. threshold_ratio=4.0 --
# конкретный физический слой должен вырасти относительно своей же EMA
# больше чем в 4 раза, ДВА шага подряд, прежде чем это считается триггером
# (фильтр разового шумового выброса) -- отдельно от train_setup.py's
# _BURST_GUARD_THRESHOLD=8.0 (тот -- абсолютный порог на АГРЕГАТНЫЙ
# global_norm, этот -- относительный, per-layer).
# ==========================================================================
LAYER_BURST_THRESHOLD_RATIO = 4.0
LAYER_BURST_EMA_DECAY = 0.98


# ==========================================================================
# ФИКС (router collapse, см. докстринг модуля выше): переинициализация
# router-листьев (kernel 2D + router_temp скаляр) + обнуление их Adam
# mu/nu внутри opt_state сразу после restore. Обе функции -- ЧИСТЫЕ
# (params/opt_state -> новый params/opt_state), без побочных эффектов.
#
# ПРИМЕЧАНИЕ: после появления _generic_pytree_merge (graft-merge на
# restore для params И opt_state, см. ниже) router/router_temp/expert_bias
# УЖЕ приходят со свежей инициализацией автоматически (несовпадающие по
# структуре листья). Эти функции оставлены в коде на случай отката на
# строгий StandardRestore или для ручного форс-сброса router независимо
# от графа причин.
# ==========================================================================
def _reset_router_params(params, seed=1234, stddev=0.02):
    """Переинициализирует ТОЛЬКО router-листья:
       - 2D (kernel) -- случайным нормальным с stddev
       - 0D с "temp" в имени (router_temp) -- сбрасывает на _ROUTER_TEMP_INIT (10.0)
    """
    rng = jax.random.PRNGKey(seed)
    _ROUTER_TEMP_INIT = 10.0  # должно совпадать с константой в moe_gmm.py

    def _reset_leaf(path, leaf):
        nonlocal rng
        path_str = path_to_str(path)
        if "router" in path_str and hasattr(leaf, "shape"):
            if leaf.ndim == 2:
                rng, sub = jax.random.split(rng)
                new_leaf = jax.random.normal(sub, leaf.shape, dtype=leaf.dtype) * stddev
                print(f"[ROUTER-RESET] Переинициализирован params: {path_str}, shape={leaf.shape}")
                return new_leaf
            if leaf.ndim == 0 and "temp" in path_str:
                print(f"[ROUTER-RESET] Сброшен router_temp: {path_str} -> {_ROUTER_TEMP_INIT}")
                return jnp.array(_ROUTER_TEMP_INIT, dtype=leaf.dtype)
        return leaf

    return jax.tree_util.tree_map_with_path(_reset_leaf, params)


def _reset_router_opt_state(opt_state):
    """Обнуляет Adam-моменты (mu/nu, и любые другие числовые буферы той
    же формы, что параметры) ТОЛЬКО для router-параметров внутри
    opt_state -- лечит momentum, накопленный ДО чекпоинта. НЕ трогает
    остальные группы (muon/lion/adamw для GDN-2/Mamba2/MLA/embed/experts)."""
    def _reset_leaf(path, leaf):
        path_str = path_to_str(path)
        if "router" in path_str and hasattr(leaf, "shape") and leaf.ndim >= 1:
            print(f"[ROUTER-RESET] Обнулён opt_state момент: {path_str}, shape={leaf.shape}")
            return jnp.zeros_like(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(_reset_leaf, opt_state)


def _download_resume_checkpoint(local_dir, repo_subdir):
    """Единственное место, которое решает, ОТКУДА тянуть чекпоинт при
    resume. save (upload_slot) эту функцию не трогает и не использует —
    он всегда пишет в HF_REPO_ID (models), независимо от RESUME_SOURCE."""
    if RESUME_SOURCE == "bucket":
        print(f"[RESUME] 🪣 Источник: HF Bucket {RESUME_BUCKET_ID}/{repo_subdir}")
        return download_slot(local_dir, repo_subdir, bucket_id=RESUME_BUCKET_ID)
    elif RESUME_SOURCE == "hf_model":
        print(f"[RESUME] 🤗 Источник: HF Model repo ({os.environ.get('HF_REPO_ID', '?')})/{repo_subdir}")
        return download_slot(local_dir, repo_subdir)  # repo_id=None -> HF_REPO_ID внутри checkpointing.py
    elif RESUME_SOURCE == "local_only":
        return None
    else:
        raise ValueError(f"Неизвестный RESUME_SOURCE={RESUME_SOURCE!r}")


# ==========================================================================
# ФИКС (см. докстринг модуля, раздел "structural realign"): helper для
# _generic_pytree_merge's list/tuple ветки. Вычисляет "сигнатуру" элемента
# tuple/list -- используется, когда длина raw и fresh НЕ совпадает
# (например, optax.chain получил новый элемент), чтобы сопоставлять
# элементы по СТРУКТУРЕ, а не по позиции.
#
# ВАЖНО: raw приходит из orbax БЕЗ типовой информации (NamedTuple всегда
# десериализуется как plain dict -- см. _generic_pytree_merge's докстринг
# про NamedTuple-ветку выше), поэтому сигнатура для dict и для
# NamedTuple-с-теми-же-именами-полей ДОЛЖНА совпадать -- обе сводятся к
# ("fields", frozenset(...)).
# ==========================================================================
def _tuple_elem_signature(x):
    if isinstance(x, dict):
        return ("fields", frozenset(x.keys()))
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return ("fields", frozenset(x._fields))
    if isinstance(x, (list, tuple)):
        return ("seq", len(x))
    if hasattr(x, "shape"):
        return ("leaf", tuple(x.shape))
    return ("other", type(x).__name__)


def _structural_tuple_realign(fresh_seq, raw_seq, path):
    """fresh_seq/raw_seq -- list/tuple разной длины. Пытается сопоставить
    каждый элемент fresh_seq С ПЕРВЫМ ещё не использованным элементом
    raw_seq той же структурной сигнатуры (_tuple_elem_signature) --
    порядок внутри каждой сигнатурной группы сохраняется (если сигнатур
    несколько с одинаковым ключом, они сопоставляются по порядку
    появления, не идеально надёжно в вырожденных случаях, но
    ЗНАЧИТЕЛЬНО лучше, чем безусловный fresh для ВСЕГО tuple).
    Элементы fresh_seq, для которых не нашлось соответствия в raw_seq
    (например, только что добавленный optax-трансформ), остаются fresh
    -- ровно так же, как несовпадающие/новые dict-ключи в основной
    _generic_pytree_merge ветке.
    Возвращает list той же длины, что fresh_seq.
    """
    used = [False] * len(raw_seq)
    raw_sigs = [_tuple_elem_signature(r) for r in raw_seq]
    out = []
    unmatched_fresh_idx = []
    for i, f in enumerate(fresh_seq):
        f_sig = _tuple_elem_signature(f)
        match_idx = None
        for j, (r_sig, is_used) in enumerate(zip(raw_sigs, used)):
            if not is_used and r_sig == f_sig:
                match_idx = j
                break
        if match_idx is not None:
            used[match_idx] = True
            out.append(_generic_pytree_merge(f, raw_seq[match_idx], path + (i,)))
        else:
            print(f"[MERGE] структурный realign: элемент #{i} на "
                  f"{'/'.join(map(str, path))} (сигнатура {f_sig}) не найден среди "
                  f"оставшихся элементов чекпоинта -- оставляю свежую инициализацию "
                  f"ТОЛЬКО для этого элемента (остальные элементы tuple восстановлены "
                  f"как обычно).")
            out.append(f)
            unmatched_fresh_idx.append(i)
    return out


def _generic_pytree_merge(fresh, raw, path=()):
    """ФИКС (см. докстринг модуля): обобщённая версия старой
    _compatible_restore_params's внутренней merge(), которая умела
    рекурсировать ТОЛЬКО по dict-узлам -- этого хватало для params
    (чисто dict-based pytree flax), но НЕ хватает для opt_state, чьи узлы
    вдобавок включают NamedTuple (optax.ScaleByAdamState(count, mu, nu),
    наш собственный MuonState(count, orth_resid), optax.EmptyState() для
    frozen/clip) и обычные tuple/list (например, tx.chain внутри optax
    строит tuple состояний по числу трансформаций).

    Контракт: "fresh" -- только что созданная (tx.init(params) /
    model.init(...)) структура правильной формы, "raw" -- сырое, БЕЗ
    навязанной структуры содержимое чекпоинта
    (mngr.restore(step, args=ocp.args.StandardRestore()), без item=).
    Рекурсивно идёт по ОБОИМ деревьям параллельно:

      - dict            -> рекурсия по ключам fresh; ключ, которого нет в
                            raw (новый/переименованный лист, например
                            router.kernel/router_temp/expert_bias) --
                            остаётся fresh (свежая инициализация), с
                            печатью [MERGE] предупреждения.
      - NamedTuple       -> рекурсия по полям (_fields), ТОЛЬКО если raw --
                            тоже NamedTuple с ТЕМИ ЖЕ полями (иначе тип
                            optax-состояния сам изменился между версиями
                            optax/кода -- fresh целиком, без попытки
                            частичного мёржа полей вслепую). ФИКС #4
                            (optimizer.py) добавил MuonState.orth_resid --
                            старые чекпоинты без этого поля просто
                            получат fresh (нулевой) orth_resid, count
                            восстановится штатно (совпадающее поле).
      - list/tuple       -> рекурсия поэлементно при равной длине; ПРИ
                            НЕСОВПАДЕНИИ ДЛИНЫ -- структурный realign через
                            _structural_tuple_realign вместо безусловного
                            fresh для ВСЕГО tuple -- см. докстринг модуля.
      - leaf со .shape   -> восстанавливается из raw, если формы СОВПАДАЮТ
                            (jnp.asarray(raw, dtype=fresh.dtype)); при
                            несовпадении формы -- fresh, с [MERGE]
                            предупреждением.
      - прочее (EmptyState(), optax.MaskedNode, None, Python int/float
        без .shape) -- возвращается fresh КАК ЕСТЬ, без попытки сравнения.

    ЧИСТАЯ функция (fresh, raw) -> merged, без побочных эффектов кроме
    print()-диагностики."""
    # --- dict ---
    if isinstance(fresh, dict):
        out = {}
        for k, v in fresh.items():
            if isinstance(raw, dict) and k in raw:
                out[k] = _generic_pytree_merge(v, raw[k], path + (k,))
            else:
                print(f"[MERGE] новый/отсутствующий в чекпоинте лист "
                      f"{'/'.join(map(str, path + (k,)))} -- оставляю свежую инициализацию")
                out[k] = v
        return out

    # --- NamedTuple (optax states: ScaleByAdamState, MuonState, EmptyState,
    # optax.MultiTransformState, ...) ---
    if isinstance(fresh, tuple) and hasattr(fresh, "_fields"):
        if not fresh._fields:
            # Пустой NamedTuple (например, optax.EmptyState()) -- нечего
            # мёржить в принципе, просто вернуть fresh как есть.
            return fresh

        def _get_field_source(field, idx):
            if isinstance(raw, tuple) and hasattr(raw, "_fields") and field in raw._fields:
                return True, getattr(raw, field)
            if isinstance(raw, dict) and field in raw:
                return True, raw[field]
            if isinstance(raw, (list, tuple)) and len(raw) == len(fresh._fields):
                return True, raw[idx]
            return False, None

        merged_fields = {}
        all_found = True
        for idx, field in enumerate(fresh._fields):
            found, raw_field_val = _get_field_source(field, idx)
            if not found:
                all_found = False
                break
            merged_fields[field] = _generic_pytree_merge(
                getattr(fresh, field), raw_field_val, path + (field,)
            )

        if all_found:
            return type(fresh)(**merged_fields)

        print(f"[MERGE] несовпадение структуры NamedTuple на "
              f"{'/'.join(map(str, path))}: fresh_fields={fresh._fields}, "
              f"raw_type={type(raw).__name__} -- оставляю свежую")
        return fresh

    # --- plain list/tuple (например, tx.chain внутри optax) ---
    if isinstance(fresh, (list, tuple)):
        if isinstance(raw, (list, tuple)) and len(raw) == len(fresh):
            merged = [
                _generic_pytree_merge(f, r, path + (i,)) for i, (f, r) in enumerate(zip(fresh, raw))
            ]
            return type(fresh)(merged) if not isinstance(fresh, tuple) else tuple(merged)
        if isinstance(raw, (list, tuple)) and len(raw) != len(fresh):
            print(f"[MERGE] несовпадение длины list/tuple на {'/'.join(map(str, path))} "
                  f"(fresh={len(fresh)}, raw={len(raw)}) -- пробую структурный realign "
                  f"вместо полного сброса на свежую инициализацию...")
            merged = _structural_tuple_realign(list(fresh), list(raw), path)
            return type(fresh)(merged) if not isinstance(fresh, tuple) else tuple(merged)
        print(f"[MERGE] несовпадение длины list/tuple на {'/'.join(map(str, path))} -- оставляю свежую")
        return fresh

    # --- leaf с .shape (jnp/np массивы, включая 0-мерные скаляры count) ---
    if hasattr(fresh, "shape"):
        if hasattr(raw, "shape") and tuple(fresh.shape) == tuple(raw.shape):
            return jnp.asarray(raw, dtype=fresh.dtype)
        print(f"[MERGE] несовпадение формы на {'/'.join(map(str, path))}: "
              f"fresh={getattr(fresh, 'shape', None)} raw={getattr(raw, 'shape', None)} -- оставляю свежую")
        return fresh

    # --- всё остальное (optax.MaskedNode, None, голые Python-числа и т.п.) ---
    return fresh


def _compatible_restore_params_and_opt_state(mngr, step, fresh_params, fresh_opt_state):
    """ФИКС (см. докстринг модуля): читает ОБА поддерева ("params" и
    "opt_state") чекпоинта ОДНИМ вызовом mngr.restore (без строгого item=)
    и мёрджит КАЖДОЕ из них через _generic_pytree_merge в соответствующую
    свежесозданную структуру.

    Если чекпоинт был сохранён ДО того, как opt_state вообще начал
    сохраняться, opt_state остаётся fresh_opt_state целиком, с
    предупреждением в консоль.

    Возвращает (merged_params, merged_opt_state) -- НЕ шардированные
    (raw host-side структуры) -- шардинг (jax.device_put) остаётся
    ответственностью вызывающего кода."""
    raw = mngr.restore(step, args=ocp.args.StandardRestore())  # без item -> без строгого таргета
    raw_params = raw["params"]
    merged_params = _generic_pytree_merge(fresh_params, raw_params)

    if "opt_state" in raw:
        raw_opt_state = raw["opt_state"]
        merged_opt_state = _generic_pytree_merge(fresh_opt_state, raw_opt_state)
    else:
        print("[MERGE] ⚠️ В чекпоинте нет ключа 'opt_state' -- opt_state остаётся полностью "
              "свежим (холодный momentum). Это ожидаемо только для очень старых чекпоинтов, "
              "сохранённых до того, как save_slot начал писать opt_state вместе с params.")
        merged_opt_state = fresh_opt_state

    return merged_params, merged_opt_state


def main_execution():
    ckpt_root = "/kaggle/working/orbax_checkpoints"
    latest_dir = os.path.join(ckpt_root, "latest")
    best_train_dir = os.path.join(ckpt_root, "best_train")
    best_val_dir = os.path.join(ckpt_root, "best_val")
    for d in (latest_dir, best_train_dir, best_val_dir):
        os.makedirs(d, exist_ok=True)

    mngr_latest = make_manager(latest_dir, max_to_keep=HF_LATEST_KEEP_N)
    mngr_best_train = make_manager(best_train_dir, max_to_keep=1)
    mngr_best_val = make_manager(best_val_dir, max_to_keep=1)

    FORCE_FRESH_START = False  # <-- поставьте False, чтобы вернуть обычный resume
    RESUME_FROM_SLOT = "best_val"  # <-- используется только если FORCE_FRESH_START=False

    # ФИКС (router collapse): см. докстринг модуля выше. После graft-merge
    # (и для params, и для opt_state) router/router_temp/expert_bias уже
    # приходят свежими автоматически -- этот флаг оставлен как
    # предупреждающий no-op.
    RESET_ROUTER_ON_RESUME = False

    if FORCE_FRESH_START:
        resume_step = None
        print("[RESUME] 🆕 FORCE_FRESH_START=True -- пропускаю поиск чекпоинтов, начинаю с нуля.")
    elif RESUME_FROM_SLOT == "latest":
        resume_step = mngr_latest.latest_step()
        if resume_step is not None:
            print(f"[LOCAL] 📦 Found checkpoint (latest): step {resume_step}")
        if resume_step is None and _HAS_HF and RESUME_SOURCE != "local_only":
            resume_step = _download_resume_checkpoint(latest_dir, "latest")
            if resume_step is not None:
                mngr_latest = make_manager(latest_dir, max_to_keep=HF_LATEST_KEEP_N)
    else:
        override_dir = os.path.join(ckpt_root, RESUME_FROM_SLOT)
        os.makedirs(override_dir, exist_ok=True)
        override_mngr = make_manager(override_dir, max_to_keep=1)
        resume_step = override_mngr.latest_step()
        if resume_step is not None:
            print(f"[LOCAL] 📦 Found checkpoint ({RESUME_FROM_SLOT}): step {resume_step}")

        if resume_step is None and _HAS_HF and RESUME_SOURCE != "local_only":
            resume_step = _download_resume_checkpoint(override_dir, RESUME_FROM_SLOT)
            if resume_step is not None:
                override_mngr = make_manager(override_dir, max_to_keep=1)

        if resume_step is not None:
            src = os.path.join(override_dir, str(resume_step))
            dst = os.path.join(latest_dir, str(resume_step))
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
                print(f"[RESUME OVERRIDE] Скопировано {RESUME_FROM_SLOT}/{resume_step} -> latest/{resume_step}")
            mngr_latest = make_manager(latest_dir, max_to_keep=HF_LATEST_KEEP_N)
        else:
            print(f"[RESUME OVERRIDE] ⚠️ Не найден чекпоинт в слоте '{RESUME_FROM_SLOT}' ни локально, ни на HF.")

    resume = (resume_step is not None)
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")

    config = ModelConfig(
        d_model=768,
        d_state=128,
        d_conv=4,
        expand=2,
        n_heads=6,
        d_latent=768,
        d_ff=4096,
        num_experts=8,
        top_k=2,
        moe_capacity_factor=1.25,
        router_aux_loss_coef=0.03,
        router_z_loss_coef=0.0003,
        num_layers=24,
        layers_per_block=3,
        vocab_size=128256,
        tie_embeddings=True,
        label_smoothing=0.0,
        router_noise_std=0.1,
        use_flash_attention=True,
        deltanet_chunk_size=256,
        layer_types=(
            "gdn2", "gdn2", "mla",
            "gdn2", "mamba2", "gdn2",
            "gdn2", "gdn2", "mla",
            "gdn2", "mamba2", "gdn2",
            "gdn2", "gdn2", "mla",
            "gdn2", "mamba2", "gdn2",
            "gdn2", "gdn2", "mla",
            "gdn2", "gdn2", "gdn2",
        ),
    )

    # ФИКС (assignment_frac fallback shape): сколько MoE-блоков реально в
    # архитектуре -- GmmMoEJ висит на каждом BlockDAR (см. model.py), т.е.
    # ровно num_layers // layers_per_block блоков. Нужно для нулевого
    # fallback-массива assignment_frac_arr в основном цикле ниже, на
    # случай (структурная защита, не ожидаемый рабочий путь), если
    # aux_info["assignment_frac"] почему-то не пришёл ни на одном микрошаге
    # эффективного шага.
    n_moe_layers = config.num_layers // config.layers_per_block
    n_experts_routed = config.num_experts - 1

    # ==========================================================================
    # ФИКС: per-source fraction теперь ГИПЕРПАРАМЕТРЫ здесь же, рядом с
    # file_pairs, а не магические числа внутри кортежей ниже. dataloader_multi_source
    # (train_setup.py) уже поддерживает 3-tuple (ids_path, lbls_path, fraction) --
    # этот блок просто делает точку конфигурации явной и удобной для правки без
    # необходимости лезть в сами пути. 1.0 = использовать источник полностью
    # (старое поведение, обратная совместимость), 0.0 < frac < 1.0 = случайная
    # подвыборка ИМЕННО этого источника (см. dataloader_multi_source per-source
    # сэмплинг, применяется ДО train/val split и ДО глобального DATASET_FRACTION).
    #
    # Порядок ключей соответствует порядку источников в file_pairs ниже.
    # ==========================================================================
    SOURCE_FRACTIONS = {
        "kodcode": 1.0,
        "math": 1.0,
        "codex": 1.0,
        "agentpack": 1.0,
        "rstar": 1.0,
        "syntheticcode": 1.0,
    }

    file_pairs = [
        (
            "/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_labels.npy",
            SOURCE_FRACTIONS["kodcode"],
        ),  # kodcode
        (
            "/kaggle/input/datasets/umirbayulgaisha/math-data/math_input_ids.npy",
            "/kaggle/input/datasets/umirbayulgaisha/math-data/math_labels.npy",
            SOURCE_FRACTIONS["math"],
        ),  # math
        (
            "/kaggle/input/datasets/akseleu1j/codex-dataset/codex_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/codex-dataset/codex_labels.npy",
            SOURCE_FRACTIONS["codex"],
        ),  # codex
        (
            "/kaggle/input/datasets/akseleu1j/agentpack/agentpack_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/agentpack/agentpack_labels.npy",
            SOURCE_FRACTIONS["agentpack"],
        ),  # agentpack
        (
            "/kaggle/input/datasets/akseleu1j/rstar-dataset/rstar_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/rstar-dataset/rstar_labels.npy",
            SOURCE_FRACTIONS["rstar"],
        ),  # rstar
        (
            "/kaggle/input/datasets/akseleu1j/sytetic-dataset/syntheticcode_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/sytetic-dataset/syntheticcode_labels.npy",
            SOURCE_FRACTIONS["syntheticcode"],
        ),  # syntheticcode
    ]

    # ФИКС: file_pairs теперь состоит из 3-tuple (ids_path, lbls_path, fraction)
    # -- распаковка ниже обновлена под 3 элемента (_frac здесь не используется,
    # это чисто проверка существования файлов).
    for ids_path, lbls_path, _frac in file_pairs:
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"Не найден файл: {ids_path}")
        if not os.path.exists(lbls_path):
            raise FileNotFoundError(f"Не найден файл: {lbls_path}")
    print("Все файлы найдены.")

    manifest = build_manifest(file_pairs)
    total_blocks_full = sum(n for _, _, n in manifest)
    total_blocks = int(total_blocks_full * DATASET_FRACTION)
    print(f"Всего блоков (полный пул): {total_blocks_full:,}")
    print(f"Всего блоков (после {DATASET_FRACTION*100:.0f}% подвыборки): {total_blocks:,}")

    micro_batch_size = 8
    accum_steps = 4
    effective_batch_size = micro_batch_size * accum_steps
    seq_len = 4096
    epochs = 1
    early_stop_patience = 2
    eval_every_steps = 1000
    eval_batches = 40
    eval_patience = 4

    val_split = 0.05
    val_size = int(total_blocks * val_split)
    train_size = total_blocks - val_size

    train_steps_per_epoch = train_size // effective_batch_size
    total_train_steps = train_steps_per_epoch * epochs
    micro_steps_per_epoch = train_size // micro_batch_size

    print(f"[TPU] Компиляция XLA графа под {total_train_steps} эффективных шагов "
          f"({epochs} эпох(и) x {train_steps_per_epoch} шагов, accum={accum_steps})...")

    # ФИКС (переключаемый WARMUP_FREEZE_STEP): warmup_freeze_step передаётся
    # явным keyword-аргументом -- см. USE_WARMUP_FREEZE/WARMUP_FREEZE_STEP_VALUE
    # выше. make_shard_and_compile (train_setup.py) должен принимать этот
    # параметр и прокидывать его в make_hybrid_optimizer(warmup_freeze_step=...)
    # -- см. optimizer.py's make_hybrid_optimizer.
    _warmup_freeze_step_effective = WARMUP_FREEZE_STEP_VALUE if USE_WARMUP_FREEZE else None
    print(f"[LR] warmup_freeze_step={_warmup_freeze_step_effective!r} "
          f"(USE_WARMUP_FREEZE={USE_WARMUP_FREEZE})")

    # ФИКС: make_shard_and_compile (train_setup.py) теперь возвращает
    # дополнительно lr_schedule 10-м элементом -- unpacking обновлён под
    # 10 значений, иначе ValueError: too many values to unpack.
    (compiled_train_micro, compiled_apply, compiled_val, mesh, tx, model,
     param_sharding, opt_state_sharding, data_sharding, lr_schedule) = (
        make_shard_and_compile(
            config, total_train_steps, micro_batch_size, seq_len, accum_steps,
            warmup_freeze_step=_warmup_freeze_step_effective,
        )
    )
    print(f"[TPU] Устройств в mesh: {mesh.shape['tpu_nodes']} (FSDP: params, state и батч шардированы).")

    # ФИКС (этот пасс -- по-физическому-слою диагностика, см. модульный
    # докстринг выше): _PARAM_LAYER_TAGS/_SOW_LAYER_TAGS заполняются ВНУТРИ
    # train_setup.make_shard_and_compile (нужны abstract_params, которые
    # строятся там же) -- читаем их отсюда как атрибуты модуля train_setup
    # ПОСЛЕ вызова выше, не через `from train_setup import ...` (на момент
    # импорта модуля они ещё None).
    _PARAM_LAYER_TAGS = train_setup._PARAM_LAYER_TAGS
    _SOW_LAYER_TAGS = train_setup._SOW_LAYER_TAGS
    _layer_grad_burst_tracker = HostLayerBurstTracker(
        _PARAM_LAYER_TAGS, threshold_ratio=LAYER_BURST_THRESHOLD_RATIO, decay=LAYER_BURST_EMA_DECAY,
    )
    print(f"[DIAG] По-слойный burst-tracker: {len(_PARAM_LAYER_TAGS)} физических тегов, "
          f"{len(_SOW_LAYER_TAGS)} sown-тегов активаций. threshold_ratio={LAYER_BURST_THRESHOLD_RATIO}.")

    print(f"[LR-DIAG] warmup_steps={max(500, int(total_train_steps * 0.20))}, "
          f"total_train_steps={total_train_steps} -- opt_state (momentum + count) теперь "
          f"реально восстанавливается из чекпоинта при resume, так что warmup БОЛЬШЕ НЕ "
          f"перезапускается заново на каждой сессии (см. докстринг модуля).")

    # ФИКС (pjit in_shardings length mismatch, см. докстринг модуля выше):
    # collinearity_coef теперь ЯВНЫЙ 6-й позиционный jit-аргумент
    # compiled_train_micro (train_setup.py's in_shardings уже ожидает
    # ровно 6 элементов) -- строим его здесь ОДИН раз как jnp-скаляр той
    # же величины, что optimizer.py's compute_loss уже использовал как
    # дефолт (ROUTER_COLLINEARITY_COEF), и передаём этот же объект во
    # ВСЕ вызовы compiled_train_micro ниже (и под memory-analysis .lower(),
    # и в основном цикле обучения).
    collinearity_coef_arr = jnp.asarray(ROUTER_COLLINEARITY_COEF, dtype=jnp.float32)

    # ФИКС (W&B): инициализация run'а. resume_id читаем из локального файла
    # (не из orbax metadata.json, чтобы не трогать сигнатуры save_slot/
    # save_slot_async в checkpointing.py) -- если он есть, W&B продолжит
    # существующий run вместо создания нового при каждом рестарте
    # Kaggle-сессии, и графики (loss/step) останутся непрерывными.
    wandb_id_path = os.path.join(ckpt_root, "wandb_run_id.txt")
    _resume_wandb_id = None
    if not FORCE_FRESH_START and os.path.exists(wandb_id_path):
        with open(wandb_id_path) as f:
            _resume_wandb_id = f.read().strip() or None

    wandb_run_id = wandb_logging.init_wandb(
        project="atomic-ai",
        run_name=f"gdn2-hybrid-{time.strftime('%Y%m%d-%H%M%S')}",
        config={**asdict(config), "micro_batch_size": micro_batch_size,
                "accum_steps": accum_steps, "seq_len": seq_len,
                "n_devices": mesh.shape["tpu_nodes"],
                "source_fractions": SOURCE_FRACTIONS,
                "router_reset_on_resume": RESET_ROUTER_ON_RESUME,
                "router_collinearity_coef": ROUTER_COLLINEARITY_COEF,
                "use_warmup_freeze": USE_WARMUP_FREEZE,
                "warmup_freeze_step": _warmup_freeze_step_effective,
                "layer_burst_threshold_ratio": LAYER_BURST_THRESHOLD_RATIO,
                "n_param_layer_tags": len(_PARAM_LAYER_TAGS)},
        resume_id=_resume_wandb_id,
    )
    if wandb_run_id is not None:
        os.makedirs(ckpt_root, exist_ok=True)
        with open(wandb_id_path, "w") as f:
            f.write(wandb_run_id)

    _sanity_stream, _, _, _ = dataloader_multi_source(
        file_pairs, micro_batch_size, data_sharding, seq_len=seq_len,
        dataset_fraction=DATASET_FRACTION, fraction_seed=DATASET_FRACTION_SEED,
    )
    print("[SANITY] Проверка первого батча...")
    test_batch = next(_sanity_stream)
    max_label = int(jnp.max(test_batch['labels']))
    min_label = int(jnp.min(test_batch['labels']))
    print(f"[SANITY] Labels range: [{min_label}, {max_label}], vocab_size={config.vocab_size}")
    assert max_label < config.vocab_size, f"max_label={max_label} >= vocab_size!"
    valid_mask = test_batch['labels'] >= 0
    n_valid = int(jnp.sum(valid_mask))
    print(f"[SANITY] Валидных labels в батче: {n_valid}/{valid_mask.size} ({100*n_valid/valid_mask.size:.1f}%)")
    if n_valid == 0:
        raise ValueError("Все labels в первом батче маскированы (pad) — loss будет NaN!")

    ids_np_chk = jax.device_get(test_batch["input_ids"])
    lbls_np_chk = jax.device_get(test_batch["labels"])
    valid_chk = lbls_np_chk[:, :-1] != -100
    shift_match = np.mean(lbls_np_chk[:, :-1][valid_chk] == ids_np_chk[:, 1:][valid_chk]) if valid_chk.any() else float("nan")
    same_pos_match = np.mean(lbls_np_chk == ids_np_chk)
    print(f"[SANITY] labels[i]==ids[i+1] (должно быть высоким): {shift_match:.2%}")
    print(f"[SANITY] labels[i]==ids[i]   (должно быть низким): {same_pos_match:.2%}")
    if same_pos_match > 0.5:
        raise ValueError(
            "labels совпадают с input_ids на тех же позициях в >50% случаев -- "
            "датасет не сдвинут на 1 токен. Останавливаю обучение до фикса данных."
        )
    del _sanity_stream

    global_rng = jax.random.PRNGKey(42)
    init_params_fn = jax.jit(
        lambda rng: model.init(rng, jnp.zeros((micro_batch_size, seq_len), dtype=jnp.int32))["params"],
        out_shardings=param_sharding,
    )
    params = init_params_fn(global_rng)
    print(f"[MEM] Доступно памяти на чипе 0: {jax.local_devices()[0].memory_stats()}")
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Общее количество параметров: {total_params:,} (≈ {total_params / 1e9:.2f} млрд)")

    weights_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params))
    n_devices_display = mesh.shape["tpu_nodes"]
    print(f"Размер весов модели (глобально): {weights_bytes / 1e9:.2f} ГБ "
          f"(с FSDP на чип реально хранится в среднем ~{weights_bytes / 1e9 / n_devices_display:.2f} ГБ)")

    wandb_logging.log_metrics(0, {"model/total_params": total_params, "model/weights_gb": weights_bytes / 1e9})

    # ФИКС (главный патч, см. докстринг модуля): fresh_opt_state строится
    # здесь и ДЕРЖИТСЯ рядом как "структурный шаблон" для
    # _generic_pytree_merge ниже -- переменная `opt_state` может быть
    # переприсвоена восстановленным (merged) значением в resume-блоке, но
    # fresh_opt_state (правильная СТРУКТУРА на fresh params) нужна ИМЕННО
    # для этого merge-вызова, поэтому сохраняем её под отдельным именем
    # ДО входа в resume-блок.
    fresh_opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)
    opt_state = fresh_opt_state

    zero_accum = jax.jit(
        lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),
        out_shardings=param_sharding,
    )(params)
    accum_grads = zero_accum

    if resume and resume_step is not None:
        print(f"[RESUME] ⬆️ Restoring step {resume_step} из 'latest' (совместимый merge, "
              f"params И opt_state)...")
        try:
            # 1-3. ФИКС (главный патч): params И opt_state восстанавливаются
            #    ОДНИМ вызовом _compatible_restore_params_and_opt_state --
            #    несовпадающие/новые листья (router, router_temp,
            #    expert_bias, и их соответствующие moments/count внутри
            #    opt_state) остаются со свежей инициализацией автоматически,
            #    ВСЁ ОСТАЛЬНОЕ (GDN-2/Mamba2/MLA/эксперты/embed -- и их
            #    params, И их AdamW/Lion/Muon momentum, И step-счётчики
            #    "count") восстанавливается как было.
            params_merged, opt_state_merged = _compatible_restore_params_and_opt_state(
                mngr_latest, resume_step, params, fresh_opt_state
            )
            params = jax.device_put(params_merged, param_sharding)
            opt_state = jax.device_put(opt_state_merged, opt_state_sharding)

            # 4. Обнуляем аккумулятор градиентов (accum_grads НЕ является
            #    частью персистентного состояния оптимизатора -- это чисто
            #    внутрисессионный буфер, обнулять его при каждом resume
            #    корректно и раньше, и сейчас).
            accum_grads = jax.jit(
                lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),
                out_shardings=param_sharding
            )(params)

            # 5. Метаданные (если есть) -- читаем из metadata.json
            meta_path = os.path.join(latest_dir, str(resume_step), "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                start_epoch = meta.get("epoch", 0)
                global_step = meta.get("global_step", resume_step)
                best_val_loss = meta.get("best_val_loss", float("inf"))
                best_train_loss = meta.get("best_train_loss", float("inf"))
            else:
                global_step = resume_step
            global_rng = jax.random.PRNGKey(42 + global_step)

            # 5.5. ФИКС (диагностика, см. докстринг модуля): печатаем
            #      восстановленные "count"-значения из opt_state -- чтобы
            #      сразу видеть, что merge реально сработал (count близок
            #      к global_step, а НЕ 0), без необходимости лезть в
            #      device-side структуру вручную.
            def _find_counts(path, leaf):
                path_str = path_to_str(path)
                if "count" in path_str and hasattr(leaf, "shape") and leaf.ndim == 0:
                    print(f"[RESUME DEBUG] opt_state count восстановлен: {path_str} = "
                          f"{int(jax.device_get(leaf))} (global_step={global_step})")
                return leaf
            jax.tree_util.tree_map_with_path(_find_counts, opt_state)

            # 6. Валидация восстановленных параметров.
            #    ФИКС (этот пасс -- isfinite вместо isnan): jnp.isnan
            #    пропускает +-inf -- restore мог бы "пройти" валидацию, даже
            #    если в params затесался inf, а не nan. Заменено на
            #    jnp.isfinite, которая ловит оба случая разом (см. чат,
            #    расследование non-finite embed на global_step=12001).
            param_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(params))))
            has_bad = any(bool(jnp.any(~jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(params))
            print(f"[RESUME DEBUG] param_norm={param_norm:.4f}, has_nonfinite={has_bad}")
            if has_bad:
                raise ValueError("Восстановленные params содержат non-finite (NaN/Inf) -- чекпоинт повреждён.")

            print(f"[RESUME] ✅ Restored: step={global_step}, best_val={best_val_loss:.4f}, best_train={best_train_loss:.4f}")

            # 7. ФИКС (router collapse): после graft-merge (params+opt_state)
            #    router.kernel/router_temp/expert_bias уже приходят свежими
            #    автоматически. Повторный явный сброс здесь избыточен --
            #    оставлен как предупреждение, а не как действие.
            if RESET_ROUTER_ON_RESUME:
                print("[ROUTER-RESET] ⚠️ RESET_ROUTER_ON_RESUME=True, но после graft-merge "
                      "router уже свежий -- пропускаю повторный явный сброс.")
                wandb_logging.log_metrics(global_step, {"router/reset_applied": 0})

            # 8. Если восстанавливали не из "latest" (override) -- перерегистрируем в latest
            if RESUME_FROM_SLOT != "latest":
                print(f"[RESUME OVERRIDE] Перерегистрирую шаг {global_step} в mngr_latest "
                      f"(бухгалтерия CheckpointManager была в обход при копировании)...")
                mngr_latest.save(
                    global_step,
                    args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}),
                )
                mngr_latest.wait_until_finished()
                print(f"[RESUME OVERRIDE] ✅ Шаг {global_step} перерегистрирован штатно.")

        except Exception as e:
            print(f"[RESUME] ❌ Error during compatible restore: {e}. Starting fresh.")
            resume = False
            global_step = 0
    else:
        print("[RESUME] 🆕 Fresh start.")

    skip_micro_steps = global_step * accum_steps
    train_stream, val_factory, _, val_steps = dataloader_multi_source(
        file_pairs, micro_batch_size, data_sharding, seq_len=seq_len,
        dataset_fraction=DATASET_FRACTION, fraction_seed=DATASET_FRACTION_SEED,
        skip_batches=skip_micro_steps,
        mode="mixed",  # ФИКС: было "round_robin" -- переход на пропорциональный
                         # размеру источника сэмплинг, устраняет ~6x переупотребление
                         # kodcode относительно его естественной доли (2.7% пула).
    )

    _dummy_batch = {
        "input_ids": jax.device_put(jnp.zeros((micro_batch_size, seq_len), dtype=jnp.int32), data_sharding),
        "labels": jax.device_put(jnp.zeros((micro_batch_size, seq_len), dtype=jnp.int32), data_sharding),
    }
    # ФИКС (pjit in_shardings length mismatch): compiled_train_micro теперь
    # требует ровно 6 позиционных аргументов (см. докстринг модуля выше) --
    # шестым передаём collinearity_coef_arr, тот же объект, что и в
    # основном цикле ниже.
    _lowered = compiled_train_micro.lower(
        params, opt_state, _dummy_batch, global_rng, accum_grads, collinearity_coef_arr
    )
    _compiled_exec = _lowered.compile()
    _analysis = _compiled_exec.memory_analysis()
    print(f"[MEM ANALYSIS] HBM temp:      {_analysis.temp_size_in_bytes / 1e9:.2f} ГБ")
    print(f"[MEM ANALYSIS] HBM arguments: {_analysis.argument_size_in_bytes / 1e9:.2f} ГБ")
    print(f"[MEM ANALYSIS] HBM output:    {_analysis.output_size_in_bytes / 1e9:.2f} ГБ")
    print("[TPU] Компиляция готова -- переходим к реальному обучению.")

    stopped_early = False
    stopped_by_time_budget = False
    stopped_by_nonfinite_limit = False
    eval_no_improve_count = 0
    epochs_without_improvement = 0
    best_eval_loss = float("inf")
    epoch = start_epoch

    nonfinite_consecutive_count = 0
    nonfinite_window = deque(maxlen=NONFINITE_WINDOW_SIZE)
    _accum_window = deque(maxlen=accum_steps)
    _micro_grad_norms = []   # сбор per‑micro норм для текущего эффективного шага
    # ФИКС (compiled_apply 5-й позиционный аргумент): копит
    # aux_info["assignment_frac"] (форма (n_moe_layers, E_routed)) с
    # каждого микрошага ОДНОГО эффективного шага -- обычный Python-список
    # (не deque с maxlen, т.к. явно сбрасывается сразу после использования
    # на apply-шаге, а не скользящее окно как _accum_window).
    _assignment_frac_window = []
    _burst_dumped_steps = set()
    _layer_burst_dumped_steps = set()

    def _save_all_needed_slots(step, cur_train_loss_val, force_latest=True, tag="", skip_hf_upload=False):
        nonlocal best_train_loss
        finalized = _finalize_pending_save(mngr_latest)
        if finalized is not None and not skip_hf_upload:
            upload_slot(latest_dir, "latest", finalized["step"], "", keep_last_n=HF_LATEST_KEEP_N)

        if force_latest:
            save_slot(mngr_latest, latest_dir, step, params, opt_state, epoch, best_val_loss, best_train_loss, cur_train_loss_val)
            if not skip_hf_upload:
                upload_slot(latest_dir, "latest", step, tag, keep_last_n=HF_LATEST_KEEP_N)
        if cur_train_loss_val is not None:
            tl = float(jax.device_get(cur_train_loss_val))
            if tl < best_train_loss:
                best_train_loss = tl
                save_slot(mngr_best_train, best_train_dir, step, params, opt_state, epoch, best_val_loss, best_train_loss, cur_train_loss_val)
                if not skip_hf_upload:
                    upload_slot(best_train_dir, "best_train", step, f"train_loss={tl:.4f}", keep_last_n=1)
                print(f"[BEST_TRAIN] Новый лучший train_loss: {tl:.4f} на шаге {step}")
                wandb_logging.log_metrics(step, {"checkpoint/best_train_loss": tl})

    def emergency_save(signum=None, frame=None):
        print(f"\n🚨 [EMERGENCY] Saving step {global_step}...")
        try:
            _save_all_needed_slots(global_step, None, force_latest=True, tag="EMERGENCY")
            print(f"🚨 ✅ Emergency save done (local + HF): step {global_step}")
        except Exception as e:
            print(f"🚨 ❌ Emergency save failed: {e}")
        wandb_logging.log_alert("Emergency save", f"Обучение прервано (SIGTERM/SIGINT) на шаге {global_step}.", level="WARN")
        wandb_logging.finish()
        sys.exit(0)

    signal.signal(signal.SIGTERM, emergency_save)
    signal.signal(signal.SIGINT, emergency_save)

    burst_streak = 0
    total_tokens_processed = 0
    epoch_start_time = time.perf_counter()
    last_ckpt_time = time.perf_counter()
    session_start_time = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        for micro_step in range(micro_steps_per_epoch):
            global_rng, step_rng = jax.random.split(global_rng)

            _t0 = time.perf_counter()
            try:
                batch = next(train_stream)
            except StopIteration:
                print("[DATA] Поток данных исчерпан для этой эпохи.")
                break
            _t_data = time.perf_counter() - _t0

            # ФИКС: round_robin/sequential кладут диагностический ключ
            # "_source_idx" в батч -- compiled_train_micro скомпилирован
            # под строгую pjit-сигнатуру {"input_ids","labels"} (см.
            # train_setup.py in_shardings), лишний ключ рушит pytree-match
            # с ValueError "Mismatch details... 3 child keys". Достаём его
            # здесь, ДО того как батч попадёт в compiled_train_micro или
            # _accum_window. .pop(..., None) безопасен и для mixed-режима
            # (там ключа нет -- просто вернёт None).
            source_idx = batch.pop("_source_idx", None)

            _accum_window.append({
                "input_ids": jax.device_get(batch["input_ids"]),
                "labels": jax.device_get(batch["labels"]),
                "step_rng": jax.device_get(step_rng),
                "source_idx": source_idx,
            })

            total_tokens_processed += micro_batch_size * seq_len

            _t1 = time.perf_counter()
            # ФИКС (pjit in_shardings length mismatch, см. докстринг модуля
            # выше): шестой позиционный аргумент collinearity_coef_arr
            # обязателен -- compiled_train_micro скомпилирован с
            # in_shardings длины 6.
            params, opt_state, accum_grads, train_loss, aux_info, micro_grad_norm = compiled_train_micro(
                params, opt_state, batch, step_rng, accum_grads, collinearity_coef_arr
            )
            _micro_grad_norms.append(float(jax.device_get(micro_grad_norm)))
            if micro_step < 30:
                jax.block_until_ready(train_loss)
            _t_compute = time.perf_counter() - _t1

            # ФИКС (compiled_apply 5-й позиционный аргумент): копим
            # assignment_frac с КАЖДОГО микрошага -- aux_info["assignment_frac"]
            # уже возвращается compiled_train_micro (собран в optimizer.py's
            # compute_loss через collect_by_leaf_name), просто раньше нигде
            # не сохранялся между микрошагами одного эффективного шага.
            if aux_info.get("assignment_frac") is not None:
                _assignment_frac_window.append(jax.device_get(aux_info["assignment_frac"]))

            if (micro_step + 1) % accum_steps == 0:
                effective_step = (micro_step + 1) // accum_steps

                _params_pre_apply_host = jax.tree_util.tree_map(jax.device_get, params)

                # ФИКС (compiled_apply 5-й позиционный аргумент): усредняем
                # по всем микрошагам ЭТОГО эффективного шага (тот же смысл,
                # что accum_grads усредняется в distributed_apply_step через
                # n_accum) и сбрасываем окно для следующего эффективного
                # шага сразу после использования. Fallback -- нулевой
                # массив нужной формы (n_moe_layers, n_experts_routed),
                # если по какой-то причине окно пусто (не ожидается на
                # практике, пока в архитектуре есть MoE и deterministic=False).
                if _assignment_frac_window:
                    assignment_frac_arr = jnp.asarray(
                        np.mean(np.stack(_assignment_frac_window), axis=0), dtype=jnp.float32
                    )
                else:
                    print(f"[WARN] assignment_frac_window пуст на global_step={global_step + 1} -- "
                          f"использую нулевой fallback (n_moe_layers={n_moe_layers}, "
                          f"E_routed={n_experts_routed}). Это НЕ ожидаемый рабочий путь -- "
                          f"если видите это часто, проверьте, что MoE реально sow'ит "
                          f"assignment_frac (moe_gmm.py) и deterministic=False в train-forward.")
                    assignment_frac_arr = jnp.zeros((n_moe_layers, n_experts_routed), dtype=jnp.float32)
                _assignment_frac_window = []

                _t_apply = time.perf_counter()
                (params, opt_state, accum_grads, was_finite,
                 global_norm, clip_factor, group_nonfinite_flags, was_clipped,
                 zclip_diag,
                 layer_grad_norms, layer_grad_maxabs, layer_grad_nonfinite,
                 layer_w_norms, layer_w_maxabs, layer_w_nonfinite,
                 muon_orth_resid) = compiled_apply(
                    params, opt_state, accum_grads, accum_steps, assignment_frac_arr
                )
                if micro_step < 30:
                    jax.block_until_ready(params)
                _t_apply_total = time.perf_counter() - _t_apply

                step_was_finite = bool(jax.device_get(was_finite))

                # ФИКС: раньше это печаталось внутри jit через debug.print
                # (и НЕ логировалось в W&B вообще). Теперь -- обычный
                # host-side разбор, ноль host-callback каналов, полное
                # покрытие W&B.
                _global_norm_val = float(jax.device_get(global_norm))

                # ==============================================================
                # ФИКС (этот пасс -- ВСЕГДА включённая по-физическому-слою
                # диагностика "на лету", см. модульный докстринг): логируем
                # КАЖДЫЙ эффективный шаг норму/max|abs|/nonfinite градиента и
                # весов по КАЖДОМУ физическому (block_idx, layer_idx) слою
                # отдельно (не одной кучей "gdn2"), плюс Muon orth_resid.
                # Дёшево (несколько device_get на маленькие (n_tags,)
                # массивы), но именно ЭТО даёт возможность увидеть, в КАКОМ
                # ИМЕННО слое начинается монотонный дрейф, а не только
                # постфактум по агрегату.
                # ==============================================================
                _lgn = jax.device_get(layer_grad_norms)
                _lgm = jax.device_get(layer_grad_maxabs)
                _lgnf = jax.device_get(layer_grad_nonfinite)
                _lwn = jax.device_get(layer_w_norms)
                _lwm = jax.device_get(layer_w_maxabs)
                _lwnf = jax.device_get(layer_w_nonfinite)
                _muon_orth_val = float(jax.device_get(muon_orth_resid))

                _layerdiag_wandb = {"muon/orth_resid_max": _muon_orth_val}
                _nonfinite_layer_tags_this_step = []
                _maxabs_by_tag = {}
                for i, tag in enumerate(_PARAM_LAYER_TAGS):
                    gn, gm, gnf = float(_lgn[i]), float(_lgm[i]), bool(_lgnf[i])
                    wn, wm, wnf = float(_lwn[i]), float(_lwm[i]), bool(_lwnf[i])
                    _layerdiag_wandb[f"layerdiag/{tag}_grad_norm"] = gn
                    _layerdiag_wandb[f"layerdiag/{tag}_grad_maxabs"] = gm
                    _layerdiag_wandb[f"layerdiag/{tag}_weight_norm"] = wn
                    _layerdiag_wandb[f"layerdiag/{tag}_weight_maxabs"] = wm
                    _maxabs_by_tag[tag] = gm
                    if gnf or wnf:
                        _nonfinite_layer_tags_this_step.append((tag, gnf, wnf))
                wandb_logging.log_metrics(global_step + 1, _layerdiag_wandb)

                if _nonfinite_layer_tags_this_step:
                    print(f"[LAYER-DIAG] ⚠️ non-finite на global_step={global_step + 1} в физических "
                          f"слоях: {_nonfinite_layer_tags_this_step} (grad, weight)")

                if _muon_orth_val > 3.0:
                    print(f"[MUON-DIAG] ⚠️ orth_resid={_muon_orth_val:.3f} на global_step={global_step + 1} "
                          f"-- Newton-Schulz(5) для Muon плохо сходится (идеал -> 0, ~27 -- полностью "
                          f"не ортогонализирует, см. optimizer.py's muon_orthogonalize_legacy докстринг). "
                          f"Возможный кандидат на источник роста global_grad_norm.")
                    wandb_logging.log_alert(
                        "Muon orth_resid высокий",
                        f"orth_resid={_muon_orth_val:.3f} на step={global_step + 1} -- NS(5) не сходится.",
                        level="WARN",
                    )

                # ФИКС (этот пасс -- локализованный по-слойный burst-tracker,
                # см. модульный докстринг): триггерится, когда ОДИН
                # конкретный физический слой отклоняется от СВОЕЙ ЖЕ EMA
                # сильнее threshold_ratio ДВА шага подряд -- раньше, чем это
                # дойдёт до общего burst_streak (агрегатный global_norm>8.0
                # три шага подряд). Снимок делается ОДИН РАЗ на (tag, step),
                # той же схемой, что уже используется для AUTO-STOP/
                # nonfinite снапшотов (params ДО apply + вся accum_window).
                _layer_triggered = _layer_grad_burst_tracker.update(_maxabs_by_tag)
                if _layer_triggered:
                    print(f"[LAYER-BURST] ⚠️ на global_step={global_step + 1}: {_layer_triggered} "
                          f"(tag, prev_ema, new_val, streak)")
                    _dump_key = (tuple(t for t, *_ in _layer_triggered), global_step + 1)
                    if _dump_key not in _layer_burst_dumped_steps:
                        _tags_str = "_".join(t for t, *_ in _layer_triggered)
                        snap_dir = os.path.join(ckpt_root, "layer_burst_snapshots",
                                                 f"{global_step + 1}_{_tags_str}")
                        os.makedirs(snap_dir, exist_ok=True)
                        for i, entry in enumerate(_accum_window):
                            np.save(os.path.join(snap_dir, f"micro_{i}_input_ids.npy"), entry["input_ids"])
                            np.save(os.path.join(snap_dir, f"micro_{i}_labels.npy"), entry["labels"])
                        with open(os.path.join(snap_dir, "LAYER_BURST_META.json"), "w") as f:
                            json.dump({
                                "global_step": int(global_step + 1),
                                "triggered": [
                                    {"tag": t, "prev_ema": p, "new_val": v, "streak": s}
                                    for t, p, v, s in _layer_triggered
                                ],
                                "muon_orth_resid": _muon_orth_val,
                                "source_idx_per_micro": [entry["source_idx"] for entry in _accum_window],
                            }, f)
                        print(f"[LAYER-BURST] 📸 Снимок сохранён: {snap_dir}")
                        wandb_logging.log_alert(
                            "Layer-level burst triggered",
                            f"Слои {_tags_str} отклонились от своей EMA >{LAYER_BURST_THRESHOLD_RATIO}x "
                            f"на step={global_step + 1}.", level="WARN",
                        )
                        _layer_burst_dumped_steps.add(_dump_key)

                # ФИКС (видимость zclip_skip, см. train_setup.py's ФИКС #9
                # докстринг): пересчитываем z-score/drift_ratio host-side из
                # СЫРЫХ полей ZClipState (уже ПОСЛЕ tx.update, т.е. это
                # состояние ЭТОГО шага) -- ТОЙ ЖЕ формулой, что update_fn
                # внутри optimizer.py's zclip_skip использует для принятия
                # решения is_spike (но здесь только для логирования, само
                # решение уже принято внутри jit-графа). Позволяет отличить
                # "zclip реально зануляет часть шагов" от "clip_by_global_norm
                # просто масштабирует растущий, но каждый раз применяемый
                # градиент" -- см. chat, инцидент "градиентная норма
                # монотонно растёт >16 при стабильном замороженном LR".
                _zc = jax.device_get(zclip_diag)
                _zc_ema_std = float(np.sqrt(max(float(_zc["ema_var"]), 1e-8)))
                _zc_log_norm = float(np.log(max(_global_norm_val, 1e-8)))
                _zc_is_warm = int(_zc["warm_count"]) >= 25       # warmup_ema_steps default в zclip_skip
                _zc_is_slow_warm = int(_zc["slow_warm_count"]) >= 60  # slow_warmup_steps default
                _zc_fast_z = ((_zc_log_norm - float(_zc["ema_mean"])) / _zc_ema_std) if _zc_is_warm else 0.0
                _zc_drift_ratio = (
                    float(np.exp(_zc_log_norm - float(_zc["slow_ema_mean"]))) if _zc_is_slow_warm else 1.0
                )
                _zc_likely_spike = bool(
                    (_zc_is_warm and _zc_fast_z > 2.5) or
                    (_zc_is_slow_warm and _zc_drift_ratio > 6.0)
                )
                if _zc_likely_spike:
                    print(f"[ZCLIP-DIAG] ⚠️ похоже, zclip_skip ЗАНУЛИЛ обновление на "
                          f"global_step={global_step + 1}: fast_z={_zc_fast_z:.2f} "
                          f"drift_ratio={_zc_drift_ratio:.2f}")
                wandb_logging.log_metrics(global_step + 1, {
                    "zclip/ema_mean": float(_zc["ema_mean"]),
                    "zclip/ema_std": _zc_ema_std,
                    "zclip/slow_ema_mean": float(_zc["slow_ema_mean"]),
                    "zclip/fast_z": _zc_fast_z,
                    "zclip/drift_ratio": _zc_drift_ratio,
                    "zclip/likely_spike_skip": int(_zc_likely_spike),
                })
                # Логирование per‑micro градиентных норм
                if _micro_grad_norms:
                  max_micro = max(_micro_grad_norms)
                  mean_micro = sum(_micro_grad_norms) / len(_micro_grad_norms)
                  wandb_step_metrics = {
                    "train/micro_grad_norm_max": max_micro,
                    "train/micro_grad_norm_mean": mean_micro,
                  }
                  wandb_logging.log_metrics(global_step + 1, wandb_step_metrics)
                  _micro_grad_norms = []   # очистка для следующего эффективного шага
                # ФИКС (этот пасс -- см. chat, порог снижен с 20.0 до 8.0):
                # раньше диагностический снимок данных срабатывал только
                # при global_norm>20 три шага подряд -- при монотонном
                # дрейфе (наблюдался рост от единиц до >16 за сотни шагов)
                # снимок делался СЛИШКОМ ПОЗДНО, чтобы поймать данные,
                # соответствующие НАЧАЛУ дрейфа. Более низкий порог даёт
                # ранний снимок конкретных input_ids -- нужно для проверки
                # гипотезы "систематически трудный сегмент потока данных"
                # (см. _PER_TOKEN_CE_CLIP / mode="mixed" обсуждение).
                _BURST_GUARD_THRESHOLD = 8.0
                if _global_norm_val > _BURST_GUARD_THRESHOLD:
                    burst_streak += 1
                else:
                    burst_streak = 0

                if burst_streak >= 3:
                    print(f"[BURST-GUARD] ⚠️ global_grad_norm>20 три эффективных шага подряд "
                          f"(global_step={global_step + 1}). Вероятен runaway-режим.")
                    if global_step not in _burst_dumped_steps:
                        snap_dir = os.path.join(ckpt_root, "burst_snapshots", str(global_step + 1))
                        os.makedirs(snap_dir, exist_ok=True)
                        for i, entry in enumerate(_accum_window):
                            np.save(os.path.join(snap_dir, f"micro_{i}_input_ids.npy"), entry["input_ids"])
                            # ФИКС (этот пасс): раньше сохранялись только
                            # input_ids -- для проверки гипотезы "трудные
                            # токены доминируют над CE" нужны и labels
                            # (какие именно позиции промаскированы/что
                            # модель должна предсказать), не только сырые
                            # входные ids.
                            np.save(os.path.join(snap_dir, f"micro_{i}_labels.npy"), entry["labels"])
                        with open(os.path.join(snap_dir, "BURST_META.json"), "w") as f:
                            json.dump({
                                "n_micro": len(_accum_window),
                                "global_step": int(global_step + 1),
                                "global_norm_at_dump": _global_norm_val,
                                "ce_loss_at_dump": float(jax.device_get(aux_info["ce_loss"])),
                                "source_idx_per_micro": [entry["source_idx"] for entry in _accum_window],
                                # ФИКС (этот пасс): дублируем по-слойную
                                # диагностику ЭТОГО ЖЕ шага прямо в
                                # BURST_META.json -- при агрегатном
                                # burst_streak срабатывании сразу видно,
                                # какие ИМЕННО слои были хуже всех в этот
                                # момент, без похода в W&B.
                                "layer_grad_maxabs_at_dump": {t: float(_lgm[i]) for i, t in enumerate(_PARAM_LAYER_TAGS)},
                                "muon_orth_resid_at_dump": _muon_orth_val,
                            }, f)
                        print(f"[BURST-GUARD] 📸 Снимок сохранён: {snap_dir} "
                              f"(global_norm={_global_norm_val:.2f}, ce_loss="
                              f"{float(jax.device_get(aux_info['ce_loss'])):.3f})")
                        _burst_dumped_steps.add(global_step)
                    # остальное (alert, сброс burst_streak) – уже есть
                    wandb_logging.log_alert(
                        "Burst guard triggered",
                        f"global_grad_norm > 20 три шага подряд на step={global_step + 1}",
                        level="WARN",
                    )
                    burst_streak = 0
                _clip_factor_val = float(jax.device_get(clip_factor))
                _group_flags_np = jax.device_get(group_nonfinite_flags)
                _was_clipped_val = bool(jax.device_get(was_clipped))

                _nonfinite_groups_this_step = [
                    name for name, flag in zip(_DIAG_GROUPS, _group_flags_np) if bool(flag)
                ]
                if _nonfinite_groups_this_step:
                    print(f"[DIAG] ⚠️ non-finite градиент в группах: {_nonfinite_groups_this_step} "
                          f"на global_step={global_step + 1}")
                if _was_clipped_val:
                    print(f"[PARAM-DIAG] ⚠️ Обнаружен параметр с |w|>=100 ДО клипа -- веса разрослись "
                          f"на global_step={global_step + 1}")

                wandb_step_diag_metrics = {
                    "train/global_grad_norm": _global_norm_val,
                    "train/clip_factor": _clip_factor_val,
                    "train/param_clip_triggered": int(_was_clipped_val),
                }
                for name in _DIAG_GROUPS:
                    wandb_step_diag_metrics[f"nonfinite/group_{name}"] = int(name in _nonfinite_groups_this_step)
                wandb_logging.log_metrics(global_step + 1, wandb_step_diag_metrics)

                if not step_was_finite:
                    print(f"[WARNING] ⚠️ Non-finite градиент на global_step={global_step + 1} -- "
                          f"обновление ПРОПУЩЕНО, веса не изменены. Если это повторяется часто, "
                          f"стоит посмотреть на LR/warmup или численную стабильность GDN-2/Mamba2/Muon.")
                    wandb_logging.log_metrics(global_step + 1, {"train/step_skipped_nonfinite": 1})

                    snap_dir = os.path.join(ckpt_root, "nonfinite_snapshots", str(global_step + 1))
                    os.makedirs(snap_dir, exist_ok=True)
                    snap_mngr = make_manager(snap_dir, max_to_keep=1)
                    snap_mngr.save(global_step + 1, args=ocp.args.StandardSave(
                        {"params": _params_pre_apply_host, "opt_state": opt_state}
                    ))
                    snap_mngr.wait_until_finished()
                    for i, entry in enumerate(_accum_window):
                        np.save(os.path.join(snap_dir, f"micro_{i}_input_ids.npy"), entry["input_ids"])
                        np.save(os.path.join(snap_dir, f"micro_{i}_labels.npy"), entry["labels"])
                        np.save(os.path.join(snap_dir, f"micro_{i}_step_rng.npy"), np.asarray(entry["step_rng"]))
                    # ФИКС: source_idx теперь реально прокидывается в снапшот
                    # (раньше был бы недоступен -- batch["_source_idx"] уже
                    # ронял бы compiled_train_micro задолго до этой точки).
                    with open(os.path.join(snap_dir, "SNAPSHOT_META.json"), "w") as f:
                        json.dump({
                            "n_micro": len(_accum_window),
                            "global_step": int(global_step + 1),
                            "source_idx_per_micro": [entry["source_idx"] for entry in _accum_window],
                            "nonfinite_groups": _nonfinite_groups_this_step,
                            "nonfinite_layer_tags": _nonfinite_layer_tags_this_step,
                            "muon_orth_resid_at_dump": _muon_orth_val,
                        }, f)
                    print(f"[SNAPSHOT] Saved {len(_accum_window)} micro-steps + PRE-APPLY params to {snap_dir}")
                else:
                    wandb_logging.log_metrics(global_step + 1, {"train/step_skipped_nonfinite": 0})

                nonfinite_window.append(0 if step_was_finite else 1)
                if step_was_finite:
                    nonfinite_consecutive_count = 0
                else:
                    nonfinite_consecutive_count += 1

                window_ratio = sum(nonfinite_window) / len(nonfinite_window)
                hit_consecutive_limit = nonfinite_consecutive_count >= NONFINITE_CONSECUTIVE_LIMIT
                hit_window_limit = (
                    len(nonfinite_window) >= NONFINITE_WINDOW_SIZE and window_ratio >= NONFINITE_WINDOW_RATIO
                )
                if hit_consecutive_limit or hit_window_limit:
                    reason = (
                        f"{nonfinite_consecutive_count} non-finite шагов ПОДРЯД"
                        if hit_consecutive_limit else
                        f"{window_ratio*100:.0f}% non-finite за последние {len(nonfinite_window)} эффективных шагов"
                    )
                    print(f"\n🛑 [AUTO-STOP] Похоже на СИСТЕМНУЮ проблему, не разовый выброс: {reason}. "
                          f"Продолжать долгий фоновый запуск бессмысленно -- сохраняюсь ЛОКАЛЬНО (без HF-заливки, "
                          f"чтобы не ждать сеть) на последнем известном ЗДОРОВОМ состоянии (params не менялись "
                          f"с последнего успешного шага) и останавливаюсь, чтобы не потратить впустую всю сессию.")
                    wandb_logging.log_alert(
                        "AUTO-STOP: частые non-finite градиенты",
                        f"Остановлено на шаге {global_step}: {reason}. Чекпоинт сохранён локально в {latest_dir}.",
                        level="ERROR",
                    )
                    try:
                        _save_all_needed_slots(
                            global_step, None, force_latest=True, tag="AUTO_STOP_NONFINITE", skip_hf_upload=True
                        )
                        print(f"🛑 ✅ Сохранено ЛОКАЛЬНО на шаге {global_step} (HF-заливка пропущена ради скорости "
                              f"остановки -- при желании залейте вручную позже, локальный чекпоинт лежит в "
                              f"{latest_dir}/{global_step}). Разберитесь с причиной перед повторным запуском -- "
                              f"см. [WARNING]/[FWD-DIAG]/[BWD-DIAG]/[PARAM-DIAG]/[LAYER-DIAG]/[LAYER-BURST]/"
                              f"[MUON-DIAG] логи выше для локализации источника.")
                    except Exception as e:
                        print(f"🛑 ❌ Save при автостопе не удался: {e}")
                    stopped_by_nonfinite_limit = True
                    stopped_early = True
                    break

                global_step += 1

                now = time.perf_counter()
                if now - last_ckpt_time >= CHECKPOINT_EVERY_SECONDS:
                    save_slot_async(mngr_latest, latest_dir, global_step, params, opt_state,
                                    epoch, best_val_loss, best_train_loss, train_loss)
                    tl = float(jax.device_get(train_loss))
                    if tl < best_train_loss:
                        best_train_loss = tl
                        save_slot(mngr_best_train, best_train_dir, global_step, params, opt_state,
                                  epoch, best_val_loss, best_train_loss, train_loss)
                        upload_slot(best_train_dir, "best_train", global_step, f"train_loss={tl:.4f}", keep_last_n=1)
                        print(f"[BEST_TRAIN] Новый лучший train_loss: {tl:.4f} на шаге {global_step}")
                        wandb_logging.log_metrics(global_step, {"checkpoint/best_train_loss": tl})
                    last_ckpt_time = time.perf_counter()

                elapsed_session = time.perf_counter() - session_start_time
                if elapsed_session >= SESSION_TIME_BUDGET_SECONDS:
                    print(f"[SESSION LIMIT] Достигнут бюджет времени сессии "
                          f"({elapsed_session/3600:.2f} ч) -- сохраняюсь и завершаюсь gracefully...")
                    wandb_logging.log_alert(
                        "Session time budget reached",
                        f"Достигнут бюджет времени сессии на шаге {global_step} -- graceful stop, resume вручную.",
                        level="WARN",
                    )
                    _save_all_needed_slots(global_step, train_loss, force_latest=True, tag="SESSION_LIMIT")
                    stopped_by_time_budget = True
                    stopped_early = True
                    break

                if micro_step < 30:
                    total_step_time = _t_compute + _t_apply_total
                    print(f"[TIMING] effective step {effective_step}: "
                          f"данные={_t_data*1000:.0f}мс  "
                          f"TPU compute={_t_compute*1000:.0f}мс  "
                          f"apply={_t_apply_total*1000:.0f}мс  "
                          f"(доля данных: {_t_data/(total_step_time+_t_data)*100:.0f}%)")

                if effective_step % 10 == 0:
                    print(
                        f"Epoch: {epoch} | Step: {effective_step}/{train_steps_per_epoch} | "
                        f"Global Step: {global_step} | Train Loss: {jax.device_get(train_loss):.4f} "
                        f"(ce={jax.device_get(aux_info['ce_loss']):.4f} "
                        f"aux={jax.device_get(aux_info['aux_loss']):.4f} "
                        f"z={jax.device_get(aux_info['z_loss']):.5f}) | "
                        f"best_train={best_train_loss:.4f} | muon_orth_resid={_muon_orth_val:.3f}"
                    )
                    # ФИКС (W&B): та же информация, что уже печатается в лог,
                    # плюс throughput (токены/сек) для графиков скорости.
                    tok_per_sec = (micro_batch_size * accum_steps * seq_len) / max(
                        (_t_compute + _t_apply_total) * accum_steps, 1e-6
                    )
                    wandb_step_metrics = {
                        "train/loss": float(jax.device_get(train_loss)),
                        "train/ce_loss": float(jax.device_get(aux_info["ce_loss"])),
                        "train/aux_loss": float(jax.device_get(aux_info["aux_loss"])),
                        "train/z_loss": float(jax.device_get(aux_info["z_loss"])),
                        "train/best_train_loss": best_train_loss,
                        "train/tokens_per_sec": tok_per_sec,
                        "train/epoch": epoch,
                    }
                    if aux_info["expert_utilization"] is not None:
                        util = np.atleast_2d(jax.device_get(aux_info["expert_utilization"]))
                        util_std_per_layer = util.std(axis=-1)
                        worst_layer = int(util_std_per_layer.argmax())
                        print(
                            f"           expert utilization std (max over layers, layer {worst_layer}): "
                            f"{util_std_per_layer[worst_layer]:.4f} | ideal ~= 0, uniform = 1/{config.num_experts - 1}"
                        )
                        wandb_step_metrics["moe/expert_util_std_max"] = float(util_std_per_layer[worst_layer])
                        wandb_step_metrics["moe/expert_util_std_worst_layer"] = worst_layer

                    if aux_info.get("router_temp") is not None:
                        rt = np.atleast_1d(jax.device_get(aux_info["router_temp"]))
                        worst_temp_layer = int(rt.argmax())
                        print(
                            f"           router_temp: mean={rt.mean():.4f} "
                            f"min={rt.min():.4f} max={rt.max():.4f} (worst_layer={worst_temp_layer})"
                        )
                        wandb_step_metrics["moe/router_temp_mean"] = float(rt.mean())
                        wandb_step_metrics["moe/router_temp_min"] = float(rt.min())
                        wandb_step_metrics["moe/router_temp_max"] = float(rt.max())
                        wandb_step_metrics["moe/router_temp_worst_layer"] = worst_temp_layer
                        for i, v in enumerate(rt):
                            wandb_step_metrics[f"moe/router_temp_layer{i}"] = float(v)
                    if aux_info.get("min_col_norm") is not None:
                        col_norms = np.atleast_1d(jax.device_get(aux_info["min_col_norm"]))
                        worst_col_layer = int(col_norms.argmin())
                        print(
                            f"           min router column norm (min over layers, layer {worst_col_layer}): "
                            f"{col_norms[worst_col_layer]:.6f}  (watch for drift toward 0)"
                        )
                        wandb_step_metrics["moe/min_col_norm_worst"] = float(col_norms[worst_col_layer])
                        if col_norms[worst_col_layer] < 1e-3:
                            print(f"[MOE-DIAG] ⚠️ router column near-collapse on layer {worst_col_layer} "
                                  f"at global_step={global_step}: min_col_norm={col_norms[worst_col_layer]:.6e}")
                            wandb_logging.log_alert(
                                "Router column near-collapse",
                                f"layer={worst_col_layer} min_col_norm={col_norms[worst_col_layer]:.6e} "
                                f"at global_step={global_step}.", level="WARN",
                            )
                    if aux_info.get("router_max_cos_per_layer") is not None:
                        max_cos = np.atleast_1d(jax.device_get(aux_info["router_max_cos_per_layer"]))
                        worst_cos_layer = int(max_cos.argmax())
                        print(f"           router_max_cos (max over layers, layer {worst_cos_layer}): {max_cos[worst_cos_layer]:.4f}")
                        wandb_step_metrics["moe/router_max_cos_worst"] = float(max_cos[worst_cos_layer])
                        if max_cos[worst_cos_layer] > 0.85:
                            wandb_logging.log_alert(
                                "Router collinearity high",
                                f"layer={worst_cos_layer} max_cos={max_cos[worst_cos_layer]:.4f} at step={global_step}",
                                level="WARN",
                            )

                    if aux_info.get("max_abs_logit_preclip") is not None:
                        max_logits = np.atleast_1d(jax.device_get(aux_info["max_abs_logit_preclip"]))
                        worst_logit_layer = int(max_logits.argmax())
                        print(
                            f"           max|router logit| pre-clip (max over layers, layer {worst_logit_layer}): "
                            f"{max_logits[worst_logit_layer]:.3f}  (clip is at ±8)"
                        )
                        wandb_step_metrics["moe/max_abs_logit_preclip_worst"] = float(max_logits[worst_logit_layer])
                        if max_logits[worst_logit_layer] > 12.0:
                            print(f"[MOE-DIAG] ⚠️ router logits far past clip on layer {worst_logit_layer} "
                                  f"at global_step={global_step}: max|logit|={max_logits[worst_logit_layer]:.3f} "
                                  f"(clip=±8) -- router is being saturated hard, worth investigating even "
                                  f"though the clip itself keeps it numerically safe.")
                    if aux_info.get("norm_x_mean") is not None:
                        norm_means = jax.device_get(aux_info["norm_x_mean"])
                        norm_maxes = jax.device_get(aux_info["norm_x_max"])
                        norm_mins = jax.device_get(aux_info["norm_x_min"])
                        mean_all = float(norm_means.mean())
                        max_all = float(norm_maxes.max())
                        min_all = float(norm_mins.min())
                        print(
                            f"           ||x|| (L2 norm per token): mean={mean_all:.3f}, "
                            f"max={max_all:.3f}, min={min_all:.3f}"
                        )
                        wandb_step_metrics["moe/norm_x_mean"] = mean_all
                        wandb_step_metrics["moe/norm_x_max"] = max_all
                        wandb_step_metrics["moe/norm_x_min"] = min_all

                    if aux_info.get("moe_dropped_ratio") is not None:
                        dropped = jax.device_get(aux_info["moe_dropped_ratio"])
                        worst_drop_layer = int(dropped.argmax())
                        print(
                            f"           moe dropped_ratio (max over layers, layer {worst_drop_layer}): "
                            f"{dropped[worst_drop_layer]:.4f}  (ideal ~= 0 after warmup)"
                        )
                        wandb_step_metrics["moe/dropped_ratio_max"] = float(dropped[worst_drop_layer])

                    # ==============================================================
                    # ФИКС (этот пасс -- ВСЕГДА включённые sow-активационные
                    # диагностики из model.py, см. модульный докстринг):
                    # логируем max|abs|/isfinite для каждого физического
                    # слоя (layer_delta_maxabs/layer_resid_maxabs, порядок
                    # СОВПАДАЕТ с _SOW_LAYER_TAGS -- см.
                    # diagnostics.layer_tags_in_sow_order/model.py's
                    # BlockDAR/BlockDARLayer докстринги), плюс отдельные
                    # per-sublayer-type диагностики (mamba2_*/gdn2_*/mla_*)
                    # и final_hidden_maxabs.
                    # ==============================================================
                    if aux_info.get("layer_delta_maxabs") is not None:
                        _ldm = np.atleast_1d(jax.device_get(aux_info["layer_delta_maxabs"]))
                        _ldf = np.atleast_1d(jax.device_get(aux_info.get("layer_delta_isfinite", np.ones_like(_ldm))))
                        _lrm = np.atleast_1d(jax.device_get(aux_info.get("layer_resid_maxabs", np.zeros_like(_ldm))))
                        n = min(len(_SOW_LAYER_TAGS), len(_ldm))
                        for i in range(n):
                            tag = _SOW_LAYER_TAGS[i]
                            wandb_step_metrics[f"actdiag/{tag}_delta_maxabs"] = float(_ldm[i])
                            wandb_step_metrics[f"actdiag/{tag}_resid_maxabs"] = float(_lrm[i]) if i < len(_lrm) else 0.0
                        _nonfinite_sow_tags = [_SOW_LAYER_TAGS[i] for i in range(n) if not bool(_ldf[i])]
                        if _nonfinite_sow_tags:
                            print(f"[ACT-DIAG] ⚠️ non-finite delta (sown) на global_step={global_step}: "
                                  f"{_nonfinite_sow_tags}")
                    for _diag_key, _wb_key in (
                        ("mamba2_ssm_out_maxabs", "actdiag/mamba2_ssm_out_maxabs_worst"),
                        ("gdn2_out_maxabs", "actdiag/gdn2_out_maxabs_worst"),
                        ("gdn2_h_final_maxabs", "actdiag/gdn2_h_final_maxabs_worst"),
                        ("gdn2_decay_a_maxabs", "actdiag/gdn2_decay_a_maxabs_worst"),
                        ("mla_out_maxabs", "actdiag/mla_out_maxabs_worst"),
                        ("final_hidden_maxabs", "actdiag/final_hidden_maxabs"),
                    ):
                      for _kstage in ("aqk", "akk", "a_wy_inverse", "w_pseudo", "u", "kg", "qg"):
                        p_mk = f"gdn2_kernelstage_{_kstage}_maxabs"
                        _fk = f"gdn2_kernelstage_{_kstage}_isfinite"
                        if aux_info.get(_mk) is not None:
                            _v = np.atleast_1d(jax.device_get(aux_info[_mk]))
                            _f = np.atleast_1d(jax.device_get(aux_info.get(_fk, np.ones_like(_v))))
                            worst_idx = int(_v.argmax())
                            wandb_step_metrics[f"kerneldiag/{_kstage}_maxabs_worst"] = float(_v[worst_idx])
                            _bad = [i for i in range(len(_f)) if not bool(_f[i])]
                            if _bad:
                                print(f"[KERNEL-DIAG] ⚠️ non-finite {_kstage} (внутри Pallas-пайплайна) "
                                      f"на GDN-2 слоях {_bad} на global_step={global_step}")
                                wandb_logging.log_alert(
                                    "GDN-2 kernel-internal non-finite",
                                    f"{_kstage} non-finite на слоях {_bad}, step={global_step}.",
                                    level="WARN",
  )
                        if aux_info.get(_diag_key) is not None:
                            _v = np.atleast_1d(jax.device_get(aux_info[_diag_key]))
                            wandb_step_metrics[_wb_key] = float(np.max(_v))

                    wandb_logging.log_metrics(global_step, wandb_step_metrics)

                if global_step % eval_every_steps == 0:
                    val_stream = val_factory()
                    eval_loss = 0.0
                    n_batches_done = 0
                    for _ in range(eval_batches):
                        try:
                            eval_batch = next(val_stream)
                        except StopIteration:
                            break
                        eval_loss += jax.device_get(compiled_val(params, eval_batch))
                        n_batches_done += 1
                    eval_loss /= max(n_batches_done, 1)
                    print(f"[EVAL] Step {global_step}: val loss (частичный, {n_batches_done} батчей) = {eval_loss:.4f}")
                    wandb_logging.log_metrics(global_step, {"eval/partial_val_loss": eval_loss})

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        eval_no_improve_count = 0
                        if eval_loss < best_val_loss:
                            best_val_loss = eval_loss
                            save_slot(mngr_best_val, best_val_dir, global_step, params, opt_state, epoch, best_val_loss, best_train_loss)
                            upload_slot(best_val_dir, "best_val", global_step, f"val_loss={eval_loss:.4f}", keep_last_n=1)
                            print(f"[BEST_VAL] Новый лучший val_loss: {best_val_loss:.4f} на шаге {global_step}")
                            wandb_logging.log_metrics(global_step, {"checkpoint/best_val_loss": best_val_loss})
                    else:
                        eval_no_improve_count += 1
                        if eval_no_improve_count >= eval_patience:
                            print(
                                f"[EARLY STOP] Частичный val loss не улучшался {eval_patience} "
                                "проверок подряд. Останавливаю обучение немедленно."
                            )
                            wandb_logging.log_alert(
                                "Early stop", f"val loss не улучшался {eval_patience} проверок подряд, шаг {global_step}.",
                                level="WARN",
                            )
                            _save_all_needed_slots(global_step, train_loss, force_latest=True, tag="EARLY_STOP")
                            print(f"[ORBAX] Финальные чекпоинты (шаг {global_step}) сохранены.")
                            stopped_early = True
                            break

            else:
                if micro_step < 30:
                    print(f"[TIMING] micro step {micro_step}: "
                          f"данные={_t_data*1000:.0f}мс  "
                          f"TPU={_t_compute*1000:.0f}мс  (accumulating)")

        if stopped_early:
            break

        print(f"--- Эпоха {epoch} завершена. Запуск распределенной кросс-валидации ---")
        val_stream = val_factory()
        total_val_loss = 0.0
        for _ in range(val_steps):
            total_val_loss += jax.device_get(compiled_val(params, next(val_stream)))

        mean_val_loss = total_val_loss / val_steps
        print(f"===> Эпоха: {epoch} | ИТОГОВЫЙ СРЕДНИЙ VALIDATION LOSS: {mean_val_loss:.4f} <===")
        wandb_logging.log_metrics(global_step, {"eval/epoch_val_loss": mean_val_loss, "eval/epoch": epoch})

        epoch_elapsed = time.perf_counter() - epoch_start_time
        tokens_per_sec = total_tokens_processed / epoch_elapsed
        print(f"Средняя скорость эпохи: {tokens_per_sec / 1e6:.2f} млн токенов/сек")

        total_tokens_processed = 0
        epoch_start_time = time.perf_counter()

        _save_all_needed_slots(global_step, None, force_latest=True, tag="EPOCH_END")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_without_improvement = 0
            save_slot(mngr_best_val, best_val_dir, global_step, params, opt_state, epoch, best_val_loss, best_train_loss)
            upload_slot(best_val_dir, "best_val", global_step, f"val_loss={mean_val_loss:.4f} EPOCH_END", keep_last_n=1)
            print(f"[BEST_VAL] Новый лучший val_loss ({best_val_loss:.4f}) -- сохранён")
            wandb_logging.log_metrics(global_step, {"checkpoint/best_val_loss": best_val_loss})
        else:
            epochs_without_improvement += 1
            print(
                f"[EARLY STOP] val loss не улучшился {epochs_without_improvement} эпох(и) подряд "
                f"(лучший: {best_val_loss:.4f})"
            )
            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"[EARLY STOP] Останавливаю обучение -- val loss не улучшался "
                    f"{early_stop_patience} эпохи подряд."
                )
                break

    if stopped_by_time_budget:
        print(f"[SESSION LIMIT] Обучение остановлено по бюджету времени сессии на шаге {global_step}. "
              f"Запустите скрипт заново для продолжения.")
    if stopped_by_nonfinite_limit:
        print(f"[AUTO-STOP] Обучение остановлено на шаге {global_step} из-за частых non-finite градиентов "
              f"(похоже на системную проблему, не разовый выброс). Чекпоинт сохранён на последнем здоровом "
              f"состоянии. НЕ запускайте повторный resume вслепую -- сначала разберитесь с причиной "
              f"(численная стабильность, LR/warmup), иначе с высокой вероятностью упрётесь в то же самое.")

    finalized = _finalize_pending_save(mngr_latest)
    if finalized is not None:
        upload_slot(latest_dir, "latest", finalized["step"], "FINAL", keep_last_n=HF_LATEST_KEEP_N)

    print("Обучение завершено (для этой сессии).")

    # ФИКС: финальная сводка прогона -- фиксируется как W&B summary (не
    # временной ряд), чтобы в таблице/сравнении ранов сразу были видны
    # итоговые best_train/best_val, на каком шаге всё закончилось и по
    # какой причине (обычное завершение / time budget / non-finite auto-stop).
    wandb_logging.set_summary({
        "final/best_train_loss": best_train_loss,
        "final/best_val_loss": best_val_loss,
        "final/global_step": global_step,
        "final/stopped_by_time_budget": stopped_by_time_budget,
        "final/stopped_by_nonfinite_limit": stopped_by_nonfinite_limit,
    })
    wandb_logging.finish()


if __name__ == "__main__":
    # ФИКС (этот пасс -- убран стрей os.environ-присвоение, которое раньше
    # висело здесь же неиндентированным и было вставлено ПОСЛЕ комментария,
    # но перед вызовом main_execution() -- работало по чистой случайности
    # порядка исполнения, но было легко сломать. GDN2_FWD_DIAG управляется
    # явно снаружи (переменная окружения перед запуском скрипта), либо
    # можно раскомментировать строку ниже для постоянной диагностики --
    # ПОМНИТЕ: это добавляет host-callback накладные расходы на каждый шаг,
    # включайте только для целевой отладки, не для обычных прогонов). ФИКС
    # #10 (по-физическому-слою диагностика, см. модульный докстринг) НЕ
    # зависит от этого флага -- она всегда включена независимо от
    # GDN2_FWD_DIAG, поскольку идёт через self.sow(...)/чистые jnp-выходы,
    # а не debug.print/callback.
    os.environ["GDN2_FWD_DIAG"] = "1"
    main_execution()
