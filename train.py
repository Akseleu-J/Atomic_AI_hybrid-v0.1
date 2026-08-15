import glob
import os
import re
import time
import json
import signal
import shutil
import sys
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# ==================== HF HUB RELAY ====================
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    from huggingface_hub import HfApi, snapshot_download, upload_folder, create_repo, login
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    HF_REPO_ID = user_secrets.get_secret("HF_REPO_ID")
    _HAS_HF = bool(HF_TOKEN)
    login(HF_TOKEN)
    if _HAS_HF:
        print(f"[HF] ✅ Интеграция: {HF_REPO_ID}")
    else:
        raise ImportError("Hugging face worck's uncorrectly")
except ImportError:
    _HAS_HF = False
    print("[WARN] pip install -q huggingface_hub")

# ФИКС: раз в 25 минут. ВАЖНО: с синхронным чекпоинтингом (см. ниже) реальная
# длительность записи будет видна в логах как время выполнения save_all_slots() --
# следите за первыми 2-3 циклами и увеличьте интервал, если запись занимает
# больше половины интервала (иначе TPU будет простаивать в ожидании I/O больше,
# чем считать).
CHECKPOINT_EVERY_SECONDS = 27 * 60

# ФИКС: автостоп при частых non-finite градиентах. Раньше скрипт тихо
# пропускал битые шаги и полз дальше сколько угодно -- на долгом фоновом
# запуске (ночь, 8+ часов) это означает риск потратить всю сессию впустую,
# если проблема системная (а не разовый выброс), а не единичный редкий сбой.
# Два независимых триггера (срабатывает любой):
#   NONFINITE_CONSECUTIVE_LIMIT -- N подряд non-finite эффективных шагов
#     без единого успешного обновления между ними -- явный признак того,
#     что модель "застряла" в нездоровом состоянии и сама не восстановится.
#   NONFINITE_WINDOW_SIZE / NONFINITE_WINDOW_RATIO -- если в скользящем окне
#     последних NONFINITE_WINDOW_SIZE эффективных шагов доля non-finite
#     превышает NONFINITE_WINDOW_RATIO -- тоже сигнал системной проблемы,
#     даже если они не идут строго подряд.
NONFINITE_CONSECUTIVE_LIMIT = 4
NONFINITE_WINDOW_SIZE = 15
NONFINITE_WINDOW_RATIO = 0.25

SESSION_TIME_BUDGET_SECONDS = 9 * 3600 - 15 * 60  # 9 часов минус запас на graceful stop

# ФИКС: 4 именованных слота на HF вместо "последние N" -- защищает от того,
# что один плохой шаг (как на 414-м) перезатирает единственную сохранённую
# копию. Слоты:
#   latest      -- держит 2 последних чекпоинта (N и N-1), для обычного resume
#   best_train  -- лучший train_loss за всё время
#   best_val    -- лучший val_loss (по частичной или полной валидации)
HF_LATEST_KEEP_N = 1  # ФИКС: было 2 -- со слотами best_train/best_val локально одновременно
                       # хранилось до 4 полных копий (params+opt_state), это, похоже, и
                       # переполняло диск /kaggle/working (см. диагностику в save_slot).
                       # N-1 всё равно доступен на HF при необходимости отката.

DATASET_FRACTION = 1
DATASET_FRACTION_SEED = 777

# ПАТЧ: глобальный словарь для отслеживания асинхронных сохранений
_PENDING_SAVES = {}

# ==========================================================================
# ФИКС от гонки на шаге 414: enable_async_checkpointing=False.
# Async-сохранение копирует params/opt_state с device на host в ФОНОВОМ
# потоке и возвращает управление сразу; следующий compiled_apply() при этом
# донирует (donate_argnums) те же самые буферы памяти под перезапись. Если
# фоновый writer не успел дочитать буфер до того, как XLA его переиспользовал
# (см. лог "Waiting for previous save to complete took 256s" -- явный признак
# отставания фонового воркера), live-параметры после этого момента портятся
# необратимо. Синхронный save() блокирует до полного завершения записи, что
# делает эту гонку структурно невозможной ценой простоя TPU во время записи.
# ==========================================================================

# ПАТЧ: make_manager — включить async
def make_manager(local_dir, max_to_keep):
    os.makedirs(local_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=True,
        enable_async_checkpointing=True,   # ПАТЧ: было False
    )
    return ocp.CheckpointManager(local_dir, ocp.StandardCheckpointer(), options)


def save_slot(mngr, local_dir, step, params, opt_state, epoch, best_val_loss, best_train_loss, train_loss=None):
    step_dir = os.path.join(local_dir, str(step))

    # ФИКС: если предыдущая попытка сейва этого же шага упала (например, из-за
    # нехватки места на диске), могла остаться "осиротевшая" пустая директория
    # step_dir. orbax mngr.save() может спутаться с уже существующим (пустым)
    # путём и молча ничего не записать -- именно это похоже произошло на шаге
    # 379 дважды подряд. Чистим перед повторной попыткой, чтобы orbax писал
    # в гарантированно чистое место.
    if os.path.exists(step_dir) and not os.listdir(step_dir):
        print(f"[CKPT] ⚠️ Найдена пустая осиротевшая директория {step_dir}, удаляю перед сейвом...")
        os.rmdir(step_dir)

    # ФИКС: диагностика места на диске прямо в логе -- если сейв упадёт снова,
    # сразу будет видно, было ли место или проблема в чём-то другом.
    try:
        du = shutil.disk_usage(local_dir)
        print(f"[CKPT] Диск перед сейвом: свободно {du.free / 1e9:.2f} ГБ из {du.total / 1e9:.2f} ГБ "
              f"({100 * du.free / du.total:.1f}% свободно)")
    except Exception as e_du:
        print(f"[CKPT] ⚠️ Не удалось проверить место на диске: {e_du}")

    t0 = time.perf_counter()
    mngr.save(step, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}))
    mngr.wait_until_finished()
    elapsed = time.perf_counter() - t0

    # ФИКС: раньше metadata.json писался без проверки/создания директории --
    # обычно orbax успевал создать step_dir к этому моменту, но не гарантированно
    # (например, если предыдущий сейв был прерван и внутренняя бухгалтерия
    # CheckpointManager разошлась с реальным диском). Явно создаём директорию
    # и проверяем, что orbax реально записал чекпоинт, вместо того чтобы
    # падать с неинформативным FileNotFoundError на open().
    os.makedirs(step_dir, exist_ok=True)
    if not os.listdir(step_dir):
        du = shutil.disk_usage(local_dir)
        raise RuntimeError(
            f"orbax mngr.save() для шага {step} в {local_dir} не создал ожидаемых файлов "
            f"({step_dir} пуст после wait_until_finished()). Свободно на диске: "
            f"{du.free / 1e9:.2f} ГБ из {du.total / 1e9:.2f} ГБ -- если свободного места мало, "
            f"это, скорее всего, и есть причина (см. HF_LATEST_KEEP_N и очистку старых локальных "
            f"чекпоинтов). Если места достаточно -- возможна рассинхронизация внутренней "
            f"бухгалтерии CheckpointManager, проверьте состояние {local_dir}."
        )

    meta = {
        "global_step": int(step),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "best_train_loss": float(best_train_loss),
        "timestamp": time.time(),
    }
    if train_loss is not None:
        meta["train_loss"] = float(jax.device_get(train_loss))
    meta_path = os.path.join(step_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"[CKPT] Сохранён локально: {local_dir}/{step} (заняло {elapsed:.1f}с)")
    return elapsed

# ========================== ПАТЧ: новые асинхронные функции ==========================
# ФИКС (переход на async для периодического 'latest'-сейва): гонка
# donate_argnums, из-за которой раньше стоял enable_async_checkpointing=False,
# устраняется тем, что фоновый writer работает НЕ с live-буферами params/
# opt_state (которые донируются следующему compiled_apply), а с их
# независимой device-side копией (jnp.array(..., copy=True)). Копирование
# само по себе быстрое (device-to-device, не диск) и делается синхронно
# ЗДЕСЬ -- медленная часть (запись на /kaggle/working, наблюдавшиеся ~300с)
# уходит в фон, не блокируя TPU.
#
# Используется ТОЛЬКО для периодического 'latest'-сейва во время обучения
# (единственное место, где скорость реально важна -- срабатывает часто).
# best_train/best_val/emergency/auto-stop/session-limit/epoch-end остаются
# на синхронном save_slot() -- эти пути редкие, и там важнее гарантия "уже
# на диске", чем несколько секунд экономии.

def _finalize_pending_save(mngr):
    """Дожидается фоновой записи (если есть), проверяет что файлы реально
    появились на диске, пишет metadata.json. No-op (возвращает None), если
    для этого mngr ничего не в процессе. ОБЯЗАТЕЛЬНО вызывать перед любым
    следующим save() на том же mngr (orbax не документирует поведение при
    перекрывающихся async-сохранениях на одном CheckpointManager) и перед
    выходом из процесса (иначе фоновый writer может быть убит на середине
    записи -- ровно та порча, от которой изначально стоял sync-режим)."""
    key = id(mngr)
    pending = _PENDING_SAVES.pop(key, None)
    if pending is None:
        return None

    mngr.wait_until_finished()
    elapsed = time.perf_counter() - pending["t0"]

    step_dir = os.path.join(pending["local_dir"], str(pending["step"]))
    os.makedirs(step_dir, exist_ok=True)
    if not os.listdir(step_dir):
        du = shutil.disk_usage(pending["local_dir"])
        raise RuntimeError(
            f"async mngr.save() для шага {pending['step']} в {pending['local_dir']} не создал "
            f"ожидаемых файлов после wait_until_finished(). Свободно на диске: "
            f"{du.free / 1e9:.2f} ГБ из {du.total / 1e9:.2f} ГБ."
        )

    meta = {
        "global_step": int(pending["step"]),
        "epoch": int(pending["epoch"]),
        "best_val_loss": float(pending["best_val_loss"]),
        "best_train_loss": float(pending["best_train_loss"]),
        "timestamp": time.time(),
    }
    if pending["train_loss"] is not None:
        meta["train_loss"] = float(jax.device_get(pending["train_loss"]))
    with open(os.path.join(step_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)

    # ФИКС: elapsed здесь -- время от ЗАПУСКА до момента, когда мы решили
    # проверить (может включать время, прошедшее в фоне за несколько
    # последующих шагов обучения, если finalize вызван не сразу) -- не
    # путать со старой метрикой "чистое время блокирующей записи".
    print(f"[CKPT] ✅ Async-сейв подтверждён: {pending['local_dir']}/{pending['step']} "
          f"(от запуска до подтверждения прошло {elapsed:.1f}с, включая параллельно шедшее обучение)")
    return pending

def save_slot_async(mngr, local_dir, step, params, opt_state, epoch, best_val_loss, best_train_loss, train_loss=None):
    """Запускает async-сейв на НЕЗАВИСИМОЙ копии params/opt_state и сразу
    возвращает управление -- TPU продолжает следующие шаги, пока диск пишется
    в фоне. См. модульный комментарий выше про устранение гонки с
    donate_argnums."""
    _finalize_pending_save(mngr)  # завершить предыдущий async-сейв на этом mngr, если есть

    params_snapshot = jax.block_until_ready(
        jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)
    )
    opt_state_snapshot = jax.block_until_ready(
        jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), opt_state)
    )

    try:
        du = shutil.disk_usage(local_dir)
        print(f"[CKPT] Диск перед async-сейвом: свободно {du.free / 1e9:.2f} ГБ из {du.total / 1e9:.2f} ГБ "
              f"({100 * du.free / du.total:.1f}% свободно)")
    except Exception as e_du:
        print(f"[CKPT] ⚠️ Не удалось проверить место на диске: {e_du}")

    t0 = time.perf_counter()
    mngr.save(step, args=ocp.args.StandardSave({"params": params_snapshot, "opt_state": opt_state_snapshot}))
    _PENDING_SAVES[id(mngr)] = dict(
        step=step, local_dir=local_dir, epoch=epoch,
        best_val_loss=best_val_loss, best_train_loss=best_train_loss,
        train_loss=train_loss, t0=t0,
    )
    print(f"[CKPT] 🚀 Async-сейв запущен для шага {step} -- TPU продолжает без ожидания.")
# ========================== КОНЕЦ ПАТЧА ==========================

def upload_slot(local_dir, repo_subdir, step, msg="", keep_last_n=1):
    """Заливает {local_dir}/{step} -> HF под path_in_repo={repo_subdir}/{step},
    затем чистит старые шаги ИМЕННО внутри этого repo_subdir (не трогая
    другие слоты)."""
    if not _HAS_HF:
        return
    step_dir = os.path.join(local_dir, str(step))
    if not os.path.exists(step_dir):
        print(f"[HF] ⚠️ {step_dir} не найден, пропускаю upload")
        return
    try:
        api = HfApi(token=HF_TOKEN)
        create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
        st_path = os.path.join(step_dir, "STATUS.txt")
        with open(st_path, "w") as f:
            f.write(f"IDLE: slot={repo_subdir} last_step={step} | t={time.time()}\n")
        upload_folder(
            folder_path=step_dir,
            repo_id=HF_REPO_ID,
            repo_type="model",
            path_in_repo=f"{repo_subdir}/{step}",
            commit_message=f"[{repo_subdir}] step {step} {msg}",
        )
        print(f"[HF] ✅ Uploaded: {repo_subdir}/{step}")

        try:
            all_files = api.list_repo_files(HF_REPO_ID, repo_type="model")
            prefix = f"{repo_subdir}/"
            found_steps = set()
            for f_path in all_files:
                if f_path.startswith(prefix):
                    rest = f_path[len(prefix):]
                    m = re.match(r"^(\d+)/", rest)
                    if m:
                        found_steps.add(int(m.group(1)))
            steps_sorted = sorted(found_steps, reverse=True)
            for old_step in steps_sorted[keep_last_n:]:
                try:
                    api.delete_folder(
                        path_in_repo=f"{repo_subdir}/{old_step}",
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                    )
                    print(f"[HF] 🗑️ [{repo_subdir}] удалён старый шаг: {old_step}")
                except Exception as e_del:
                    print(f"[HF] ⚠️ Не удалось удалить {repo_subdir}/{old_step}: {e_del}")
        except Exception as e_list:
            print(f"[HF] ⚠️ Не удалось получить список файлов для чистки {repo_subdir}: {e_list}")
    except Exception as e:
        print(f"[HF] ❌ Upload error ({repo_subdir}): {e}")


def download_slot(local_dir, repo_subdir):
    """Скачивает все шаги указанного слота с HF в local_dir. Возвращает
    максимальный найденный локально номер шага (или None)."""
    if not _HAS_HF:
        return None
    try:
        print(f"[HF] Downloading slot '{repo_subdir}' from {HF_REPO_ID}...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=local_dir,
            repo_type="model",
            allow_patterns=[f"{repo_subdir}/**"],
        )
        # snapshot_download кладёт файлы как local_dir/{repo_subdir}/{step}/... --
        # переносим на плоскую структуру local_dir/{step}/..., которую ожидает
        # CheckpointManager.
        src_root = os.path.join(local_dir, repo_subdir)
        if not os.path.isdir(src_root):
            return None
        for step_name in os.listdir(src_root):
            src = os.path.join(src_root, step_name)
            dst = os.path.join(local_dir, step_name)
            if os.path.isdir(src) and not os.path.exists(dst):
                os.rename(src, dst)
        items = [d for d in os.listdir(local_dir) if d.isdigit()]
        if not items:
            return None
        latest = max(int(d) for d in items)
        print(f"[HF] Slot '{repo_subdir}': найден шаг {latest}")
        return latest
    except Exception as e:
        print(f"[HF] Download failed для слота '{repo_subdir}': {e}")
        return None


from model import FullHybridMoEModel, ModelConfig, set_model_mesh, get_model_mesh
from optimizer import compute_loss, make_hybrid_optimizer
from utils import path_to_str


# ==========================================================================
# ДИАГНОСТИКА non-finite градиентов: относим каждый лист параметров к одной
# из "подозреваемых" групп (те же кандидаты, что обсуждали: GDN-2, Mamba2,
# MLA, MoE, Muon-таргеты типа >=2D веса, остальное), затем на каждом шаге,
# где итоговый global_norm не конечен, печатаем через jax.debug.print,
# у КАКИХ ИМЕННО групп есть non-finite градиент. Работает под jax.jit.
# Временная мера -- после локализации источника этот блок можно убрать.
# ==========================================================================
_DIAG_GROUPS = ("gdn2", "mamba2", "mla", "moe", "muon_decay", "embed", "other")


def _classify_leaf_group(path_str: str) -> str:
    if "gdn2" in path_str:
        return "gdn2"
    if "mamba2" in path_str:
        return "mamba2"
    if "mla" in path_str:
        return "mla"
    # ФИКС (интеграция SparseMoEJ, atomic_ops/moe_sparse.py): SparseMoEJ's
    # submodules are named "shared_expert"/"routed_experts"/"router" (not
    # the dense MoEJ's "experts_block") -- all still live under the same
    # top-level "moe" module name in BlockDARLayer, so the existing "moe"
    # substring match already covers them; "router" kept explicit too so
    # this still works if router logic is ever pulled out to its own name.
    if "experts_block" in path_str or "moe" in path_str or "router" in path_str:
        return "moe"
    if "embed" in path_str or "lm_head" in path_str:
        return "embed"
    return "other"


def make_grad_group_map(params):
    """Строит pytree той же формы, что params/grads, где каждый лист -- это
    ИМЯ группы (питоновская строка, статична, не трейсится). Вычисляется
    один раз вне jit по abstract params."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _: _classify_leaf_group(path_to_str(path)), params
    )


def build_group_nonfinite_check(grad_group_map):
    """Возвращает функцию (avg_grads) -> None, которая ВНУТРИ jit печатает
    через jax.debug.print, в каких группах есть non-finite градиент.
    grad_group_map должен быть уже посчитан (статические python-строки),
    его листья используются только для группировки на python-уровне --
    сам traversal и суммирование masks делается в jax."""
    leaves_g, treedef = jax.tree_util.tree_flatten(grad_group_map)

    def _check(avg_grads):
        leaves_grad = jax.tree_util.tree_leaves(avg_grads)
        for group in _DIAG_GROUPS:
            idxs = [i for i, g in enumerate(leaves_g) if g == group]
            if not idxs:
                continue
            any_nonfinite = False
            flags = [jnp.logical_not(jnp.all(jnp.isfinite(leaves_grad[i]))) for i in idxs]
            group_flag = jnp.any(jnp.stack(flags)) if len(flags) > 1 else flags[0]
            jax.lax.cond(
                group_flag,
                lambda g=group: jax.debug.print(
                    "[DIAG] ⚠️ non-finite градиент обнаружен в группе: {g}", g=g
                ),
                lambda: None,
            )
    return _check


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

    tx = make_hybrid_optimizer(total_steps=total_steps)
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
        # ФИКС (интеграция SparseMoEJ, atomic_ops/moe_sparse.py):
        # "routed_experts" params carry an nn.vmap expert axis (axis 0,
        # size E_routed) the same way the old dense "experts_block" did --
        # standard FSDP axis-picking below doesn't know about that vmap
        # axis and could pick it (or another axis) to shard across
        # devices in a way that breaks the vmap structure. Kept
        # unsharded, same treatment as "experts_block" always had.
        # "shared_expert"/"router" are plain Dense layers (no vmap axis),
        # so they fall through to the normal FSDP logic below unchanged.
        if "experts_block" in path_str or "routed_experts" in path_str:
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

    # ДИАГНОСТИКА: строим карту групп один раз (вне jit, на python-уровне) и
    # компилируем функцию-проверку non-finite по группам для использования
    # внутри distributed_apply_step.
    grad_group_map = make_grad_group_map(abstract_params)
    _group_nonfinite_check = build_group_nonfinite_check(grad_group_map)

    # ФИКС: масштаб градиента для decay_a (GDN-2) / A_log (Mamba2) -- те же
    # экспоненцируемые decay-параметры, что многократно оказывались
    # источником non-finite. Реализовано на уровне градиента (а НЕ отдельной
    # группой в multi_transform в optimizer.py), чтобы НЕ менять структуру
    # opt_state и остаться совместимыми с уже сохранёнными чекпоинтами.
    # Множитель 0.2 функционально эквивалентен LR в 5 раз меньше для этих
    # конкретных параметров.
    def _decay_scale_leaf(path, param):
        path_str = path_to_str(path)
        return 0.2 if ("decay_a" in path_str or "a_log" in path_str) else 1.0

    _decay_grad_scale = jax.tree_util.tree_map_with_path(_decay_scale_leaf, abstract_params)

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

        # ДИАГНОСТИКА: печатаем, в каких группах параметров градиент
        # non-finite -- до клиппинга/nan_to_num, чтобы видеть "сырой" источник.
        _group_nonfinite_check(avg_grads)

        global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(avg_grads)))

        # ФИКС: если в градиентах просочился NaN/Inf (bf16-overflow в chunked CE,
        # associative_scan в GDN-2/Mamba2, Muon-ортогонализация и т.п.), старый
        # код давал global_norm=NaN -> clip_factor=NaN -> НОВЫЕ params = NaN*p
        # для КАЖДОГО параметра за один шаг, необратимо и мгновенно (это и
        # произошло на шаге ~300: train_loss 4.49 -> следующий шаг 11.7618 =
        # ln(vocab_size), т.е. модель откатилась к состоянию "как при случайной
        # инициализации"). Теперь: если global_norm не конечен, обновление
        # ПОЛНОСТЬЮ пропускается (clip_factor=0 + nan_to_num на всякий случай,
        # т.к. 0*NaN=NaN), веса остаются как были, а вызывающий код узнаёт об
        # этом через is_finite и логирует предупреждение вместо тихой порчи.
        is_finite = jnp.isfinite(global_norm)
        safe_norm = jnp.where(is_finite, global_norm, 1.0)
        clip_factor = jnp.where(is_finite, jnp.minimum(1.0, 1.0 / (safe_norm + 1e-6)), 0.0)

        avg_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) * clip_factor,
            avg_grads,
        )

        # ФИКС: масштабируем градиент для decay_a/A_log (см. _decay_grad_scale
        # выше) -- эффективно снижает LR для этой узкой, повторяющейся
        # "горячей точки" non-finite, не трогая структуру opt_state.
        avg_grads = jax.tree_util.tree_map(lambda g, s: g * s, avg_grads, _decay_grad_scale)

        updates, new_s = tx.update(avg_grads, s, p)
        new_p = optax.apply_updates(p, updates)

        # ФИКС: последний общий рубеж -- клип самих ПАРАМЕТРОВ (не
        # активаций/градиентов, это уже сделано выше и в model.py). Диагностика
        # показала, что даже после санитизации градиента в gdn2_recurrence_safe
        # и mla_flash_attn_out non-finite градиент продолжает появляться на
        # тех же слоях -- значит проблема сместилась в backward через сами
        # nn.Dense (q_proj/k_proj/decay_proj и т.п.): если веса W уже успели
        # разрастись за 640+ шагов (особенно у Muon-параметров, где НЕТ
        # weight decay, в отличие от AdamW/Lion), то grad * W может дать inf
        # даже при конечном grad. Ограничиваем абсолютную величину параметров
        # после каждого apply-шага -- дёшево и не меняет оптимизатор.
        new_p = jax.tree_util.tree_map(
            lambda pp: jnp.nan_to_num(jnp.clip(pp, -1e2, 1e2), nan=0.0, posinf=1e2, neginf=-1e2),
            new_p,
        )
        # ДИАГНОСТИКА: подтверждаем/опровергаем гипотезу "веса разрослись" --
        # печатаем, если клип реально что-то обрезал на этом шаге.
        was_clipped = jnp.any(jnp.stack([
            jnp.any(jnp.abs(leaf) >= 1e2) for leaf in jax.tree_util.tree_leaves(p)
        ]))
        jax.lax.cond(
            was_clipped,
            lambda: jax.debug.print("[PARAM-DIAG] ⚠️ Обнаружен параметр с |w|>=100 ДО клипа -- веса разрослись."),
            lambda: None,
        )

        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        return new_p, new_s, zero_accum, is_finite

    def distributed_val_step(p, b):
        return compute_loss(
            p, model_apply_wrapped, b, config,
            rngs=None,
            deterministic=True,
        )

    # ФИКС (интеграция SparseMoEJ): moe_dropped_ratio -- новый sown value,
    # same (n_moe_layers,)-vector shape as expert_utilization, needs its
    # own out_sharding entry or jax.jit's donate/out_shardings machinery
    # will not know how to place it.
    aux_info_sharding = {
        "ce_loss": NamedSharding(mesh, P()),
        "aux_loss": NamedSharding(mesh, P()),
        "z_loss": NamedSharding(mesh, P()),
        "expert_utilization": NamedSharding(mesh, P(None)),
        "moe_dropped_ratio": NamedSharding(mesh, P(None)),
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
            NamedSharding(mesh, P()),  # is_finite
        ),
    )

    compiled_val = jax.jit(
        distributed_val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=NamedSharding(mesh, P()),
    )

    return (compiled_train_micro, compiled_apply, compiled_val, mesh, tx, model,
            param_sharding, opt_state_sharding, data_sharding)


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
    manifest = []
    total = 0
    for ids_path, lbls_path in file_pairs:
        n_rows = np.load(ids_path, mmap_mode="r").shape[0]
        manifest.append((ids_path, lbls_path, n_rows))
        total += n_rows
        print(f"[DATA] {os.path.basename(ids_path)}: {n_rows:,} блоков")
    print(f"[DATA] Комбинированный пул: {total:,} блоков из {len(manifest)} файл(ов)")
    return manifest


def dataloader_multi_source(file_pairs, batch_size, data_sharding, seq_len, val_split=0.05,
                             dataset_fraction=1.0, fraction_seed=777, skip_batches=0):
    manifest = build_manifest(file_pairs)
    sizes = np.array([n for _, _, n in manifest])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    total_blocks = int(offsets[-1])
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

    all_idx = np.arange(total_blocks)
    if dataset_fraction < 1.0:
        frac_rng = np.random.RandomState(fraction_seed)
        n_keep = int(total_blocks * dataset_fraction)
        all_idx = frac_rng.choice(all_idx, size=n_keep, replace=False)
        all_idx.sort()
        print(f"[DATA] Подвыборка {dataset_fraction*100:.0f}%: {n_keep:,} из {total_blocks:,} блоков "
              f"(seed={fraction_seed}, детерминированно между рестартами)")

    pool_size = len(all_idx)
    val_size = int(pool_size * val_split)
    train_size = pool_size - val_size

    split_rng = np.random.RandomState(42)
    shuffled = np.copy(all_idx)
    split_rng.shuffle(shuffled)
    train_idx_pool = shuffled[:train_size]
    val_idx_pool = shuffled[train_size:]

    def _generator(pool, is_train=True, skip_first=0):
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

    return (
        _generator(train_idx_pool, True, skip_first=skip_batches),
        lambda: _generator(val_idx_pool, False),
        train_size // batch_size,
        val_size // batch_size,
    )


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

    # ФИКС: явный флаг чистого старта -- гарантирует свежий запуск НЕЗАВИСИМО
    # от того, остался ли где-то (локально или на HF) недочищенный чекпоинт.
    # Раньше "чистый старт" зависел от того, что resume_step случайно
    # окажется None -- если забыть удалить один из слотов на HF, скрипт
    # молча резюмировался бы со старого (потенциально "нездорового") чекпоинта.
    # FORCE_FRESH_START=True полностью пропускает поиск/скачивание чекпоинтов.
    #
    # ВАЖНО для этого прогона: переход MoEJ -> SparseMoEJ меняет структуру
    # params pytree (experts_block -> shared_expert/routed_experts, router
    # shape меняется с num_experts на num_experts-1 выходов) -- resume со
    # старого dense-чекпоинта СТРУКТУРНО несовместим и упадёт на
    # restore(). FORCE_FRESH_START=True здесь обязателен для первого
    # sparse-прогона, не только "по умолчанию безопасно".
    FORCE_FRESH_START = True  # <-- поставьте False, чтобы вернуть обычный resume

    # ФИКС: ручной override источника восстановления -- на случай, если
    # 'latest' испорчен (NaN-эпизод и т.п.), а известный здоровый чекпоинт
    # лежит в другом слоте. "latest" -- обычное поведение. "best_train" 
    # "best_val" -- форсированное восстановление именно из этого слота (один
    # раз, для отката после инцидента), ПОСЛЕ чего он копируется в 'latest' и
    # обучение дальше продолжается как обычно.
    RESUME_FROM_SLOT = "best_train"  # <-- используется только если FORCE_FRESH_START=False

    if FORCE_FRESH_START:
        resume_step = None
        print("[RESUME] 🆕 FORCE_FRESH_START=True -- пропускаю поиск чекпоинтов, начинаю с нуля.")
    # --- Resume: сначала локально ищем нужный слот, потом HF ---
    elif RESUME_FROM_SLOT == "latest":
        resume_step = mngr_latest.latest_step()
        if resume_step is not None:
            print(f"[LOCAL] 📦 Found checkpoint (latest): step {resume_step}")
        if resume_step is None and _HAS_HF:
            resume_step = download_slot(latest_dir, "latest")
            if resume_step is not None:
                mngr_latest = make_manager(latest_dir, max_to_keep=HF_LATEST_KEEP_N)
    else:
        # Форсированный откат: тянем слот RESUME_FROM_SLOT (например best_train)
        # и локально, и с HF, затем зеркалим его в latest_dir, чтобы
        # mngr_latest мог его прочитать и дальнейшие save() продолжали работать
        # штатно через обычный 'latest'-путь.
        override_dir = os.path.join(ckpt_root, RESUME_FROM_SLOT)
        os.makedirs(override_dir, exist_ok=True)
        override_mngr = make_manager(override_dir, max_to_keep=1)
        resume_step = override_mngr.latest_step()
        if resume_step is not None:
            print(f"[LOCAL] 📦 Found checkpoint ({RESUME_FROM_SLOT}): step {resume_step}")

        if resume_step is None and _HAS_HF:
            resume_step = download_slot(override_dir, RESUME_FROM_SLOT)
            if resume_step is not None:
                override_mngr = make_manager(override_dir, max_to_keep=1)

        if resume_step is not None:
            src = os.path.join(override_dir, str(resume_step))
            dst = os.path.join(latest_dir, str(resume_step))
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
                print(f"[RESUME OVERRIDE] Скопировано {RESUME_FROM_SLOT}/{resume_step} -> latest/{resume_step}")
            # Пересоздаём mngr_latest, чтобы он увидел скопированный шаг
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
        n_heads=6,            # 768/6 = 128 = MXU tile, required by kernel_a_scores.py assert
        d_latent=768,
        d_ff=4096,
        num_experts=8,        # 1 shared + 7 routed (atomic_ops/moe_sparse.py's SparseMoEJ)
        top_k=1,              # ФИКС: SparseMoEJ routes top-1 among the 7 routed experts (argmax
                               # over router_logits) -- was 2 (unused, dense MoEJ ignored this
                               # field entirely). Now documents the ACTUAL routing behavior;
                               # SparseMoEJ still doesn't read this field directly (top-1 is
                               # hardcoded via jnp.argmax), this is documentation, not wiring.
        moe_capacity_factor=1.25,  # confirmed via atomic_ops_moe_bench_tpu.py: dropped_ratio->0
                                     # by the end of the quality-check run at this factor.
        # ФИКС: поднят с 0.01 -- quality-check (moe_quality_check_tpu.py)
        # валидировал балансировку роутера именно при coef=0.1, где
        # dropped_ratio уверенно сходится к 0 за 400 шагов. 0.01 на порядок
        # ниже и НЕ был проверен на предмет того, достаточно ли давления на
        # балансировку при таком уровне -- 0.03 выбран как промежуточное,
        # более консервативное значение для первого реального прогона;
        # следить за dropped_ratio/expert_util std в логе и поднять до 0.1,
        # если балансировка на реальных данных окажется хуже, чем в тесте.
        router_aux_loss_coef=0.03,
        router_z_loss_coef=0.0001,
        num_layers=24,         # 8 blocks x layers_per_block=3
        layers_per_block=3,
        vocab_size=128256,
        tie_embeddings=True,
        label_smoothing=0.0,
        router_noise_std=0.1,
        use_flash_attention=True,
        deltanet_chunk_size=256,   # must equal kernel_a_scores.BT
        layer_types=(
            "gdn2", "gdn2", "mla",
            "gdn2", "mamba2", "gdn2",
            "gdn2", "gdn2", "gdn2",
            "gdn2", "gdn2", "mla",
            "gdn2", "mamba2", "gdn2",
            "gdn2", "gdn2", "mla",
            "gdn2", "gdn2", "mla",       # new block 6 — mirrors block 0
            "gdn2", "mamba2", "gdn2",    # new block 7 — mirrors block 1
        ),
    )
    file_pairs = [
        (
            "/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_labels.npy",
        ),  # kodcode
        (
            "/kaggle/input/datasets/umirbayulgaisha/math-data/math_input_ids.npy",
            "/kaggle/input/datasets/umirbayulgaisha/math-data/math_labels.npy",
        ), #math
    ]
    """        
        (
        "/kaggle/input/datasets/akseleu1j/codex-dataset/codex_input_ids.npy",
        "/kaggle/input/datasets/akseleu1j/codex-dataset/codex_labels.npy",
        ),  # codex
         (
        "/kaggle/input/datasets/akseleu1j/agentpack/agentpack_input_ids.npy",
        "/kaggle/input/datasets/akseleu1j/agentpack/agentpack_labels.npy",
        ),  #agentpack
        
        
        (
            "/kaggle/input/datasets/akseleu1j/rstar-dataset/rstar_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/rstar-dataset/rstar_labels.npy",
        ),  # rstar
        (
             "/kaggle/input/datasets/akseleu1j/sytetic-dataset/syntheticcode_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/sytetic-dataset/syntheticcode_labels.npy",
        ),  # syntheticcode
    """

    for ids_path, lbls_path in file_pairs:
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

    (compiled_train_micro, compiled_apply, compiled_val, mesh, tx, model,
     param_sharding, opt_state_sharding, data_sharding) = (
        make_shard_and_compile(config, total_train_steps, micro_batch_size, seq_len, accum_steps)
    )
    print(f"[TPU] Устройств в mesh: {mesh.shape['tpu_nodes']} (FSDP: params, state и батч шардированы).")

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

    opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)

    zero_accum = jax.jit(
        lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),
        out_shardings=param_sharding,
    )(params)
    accum_grads = zero_accum

    if resume and resume_step is not None:
        print(f"[RESUME] ⬆️ Restoring step {resume_step} из 'latest'...")
        try:
            restored = mngr_latest.restore(
                resume_step,
                args=ocp.args.StandardRestore({"params": params, "opt_state": opt_state}),
            )
            params = restored["params"]
            opt_state = restored["opt_state"]
            accum_grads = jax.jit(
                lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),
                out_shardings=param_sharding,
            )(params)

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

            # ФИКС: sanity-проверка после restore -- ловим "застрял на ln(vocab)"
            # сразу, не через сотни шагов. Не идеальная защита (веса могут быть
            # тихо повреждены не до NaN/init-уровня), но отсекает самый частый случай.
            param_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(params))))
            has_nan = any(bool(jnp.any(jnp.isnan(x))) for x in jax.tree_util.tree_leaves(params))
            print(f"[RESUME DEBUG] param_norm={param_norm:.4f}, has_nan={has_nan}")
            if has_nan:
                raise ValueError("Восстановленные params содержат NaN -- чекпоинт повреждён.")

            print(f"[RESUME] ✅ Restored: step={global_step}, best_val={best_val_loss:.4f}, best_train={best_train_loss:.4f}")

            # ФИКС: если resume_step попал в latest_dir через ручной
            # shutil.copytree (override-путь RESUME_FROM_SLOT != "latest"),
            # CheckpointManager НЕ знает об этом шаге через свою внутреннюю
            # бухгалтерию (она обновляется только при вызовах mngr.save()).
            # Расхождение между "что реально на диске" и "что менеджер считает
            # валидным" может ломать последующие mngr_latest.save() на новых
            # шагах (похоже, именно это вызвало FileNotFoundError на шаге 379).
            # Чиним явным пересохранением через сам mngr_latest -- теперь
            # бухгалтерия корректна, а параметры те же самые, что были
            # восстановлены (никакого реального переобучения/потери прогресса).
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
            print(f"[RESUME] ❌ Error: {e}. Starting fresh.")
            resume = False
            global_step = 0
    else:
        print("[RESUME] 🆕 Fresh start.")

    skip_micro_steps = global_step * accum_steps
    train_stream, val_factory, _, val_steps = dataloader_multi_source(
        file_pairs, micro_batch_size, data_sharding, seq_len=seq_len,
        dataset_fraction=DATASET_FRACTION, fraction_seed=DATASET_FRACTION_SEED,
        skip_batches=skip_micro_steps,
    )

    _dummy_batch = {
        "input_ids": jax.device_put(jnp.zeros((micro_batch_size, seq_len), dtype=jnp.int32), data_sharding),
        "labels": jax.device_put(jnp.zeros((micro_batch_size, seq_len), dtype=jnp.int32), data_sharding),
    }
    _lowered = compiled_train_micro.lower(params, opt_state, _dummy_batch, global_rng, accum_grads)
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

    # ФИКС: состояние для автостопа при частых non-finite (см. константы
    # NONFINITE_* вверху файла).
    nonfinite_consecutive_count = 0
    nonfinite_window = deque(maxlen=NONFINITE_WINDOW_SIZE)
    _accum_window = deque(maxlen=accum_steps)

    # ПАТЧ: изменённая _save_all_needed_slots с финализацией
    def _save_all_needed_slots(step, cur_train_loss_val, force_latest=True, tag="", skip_hf_upload=False):
        """Сохраняет 'latest' всегда; 'best_train' -- если побит рекорд train_loss.
        skip_hf_upload=True -- только локально, без сетевой заливки на HF (для
        случаев, когда важна скорость завершения, а не немедленная доступность
        чекпоинта на HF -- см. auto-stop по non-finite ниже)."""
        nonlocal best_train_loss
        # ПАТЧ: финализируем любой незавершённый async-сейв 'latest'
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

    def emergency_save(signum=None, frame=None):
        print(f"\n🚨 [EMERGENCY] Saving step {global_step}...")
        try:
            _save_all_needed_slots(global_step, None, force_latest=True, tag="EMERGENCY")
            print(f"🚨 ✅ Emergency save done (local + HF): step {global_step}")
        except Exception as e:
            print(f"🚨 ❌ Emergency save failed: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, emergency_save)
    signal.signal(signal.SIGINT, emergency_save)

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

            _accum_window.append({
                "input_ids": jax.device_get(batch["input_ids"]),
                "labels": jax.device_get(batch["labels"]),
                "step_rng": jax.device_get(step_rng),
            })

            total_tokens_processed += micro_batch_size * seq_len

            _t1 = time.perf_counter()
            params, opt_state, accum_grads, train_loss, aux_info = compiled_train_micro(
                params, opt_state, batch, step_rng, accum_grads
            )
            if micro_step < 30:
                jax.block_until_ready(train_loss)
            _t_compute = time.perf_counter() - _t1

            if (micro_step + 1) % accum_steps == 0:
                effective_step = (micro_step + 1) // accum_steps

                _params_pre_apply_host = jax.tree_util.tree_map(jax.device_get, params)

                _t_apply = time.perf_counter()
                params, opt_state, accum_grads, was_finite = compiled_apply(
                    params, opt_state, accum_grads, accum_steps
                )
                if micro_step < 30:
                    jax.block_until_ready(params)
                _t_apply_total = time.perf_counter() - _t_apply

                # ФИКС: узнаём, был ли шаг пропущен из-за NaN/Inf в градиенте
                # (веса НЕ обновились -- см. distributed_apply_step). Раньше
                # это происходило тихо и необратимо портило все параметры.
                step_was_finite = bool(jax.device_get(was_finite))
                if not step_was_finite:
                    print(f"[WARNING] ⚠️ Non-finite градиент на global_step={global_step + 1} -- "
                          f"обновление ПРОПУЩЕНО, веса не изменены. Если это повторяется часто, "
                          f"стоит посмотреть на LR/warmup или численную стабильность GDN-2/Mamba2/Muon.")

                    # ВСТАВКА: снэпшот с pre-apply params + весь accum-window
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
                    with open(os.path.join(snap_dir, "SNAPSHOT_META.json"), "w") as f:
                        json.dump({"n_micro": len(_accum_window), "global_step": int(global_step + 1)}, f)
                    print(f"[SNAPSHOT] Saved {len(_accum_window)} micro-steps + PRE-APPLY params to {snap_dir}")

                # ФИКС: автостоп при частых non-finite -- см. константы
                # NONFINITE_* и комментарий там же. Считаем ДО global_step+=1,
                # чтобы номер шага в логе соответствовал именно проблемному шагу.
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
                    try:
                        _save_all_needed_slots(
                            global_step, None, force_latest=True, tag="AUTO_STOP_NONFINITE", skip_hf_upload=True
                        )
                        print(f"🛑 ✅ Сохранено ЛОКАЛЬНО на шаге {global_step} (HF-заливка пропущена ради скорости "
                              f"остановки -- при желании залейте вручную позже, локальный чекпоинт лежит в "
                              f"{latest_dir}/{global_step}). Разберитесь с причиной перед повторным запуском -- "
                              f"см. [WARNING]/[FWD-DIAG]/[BWD-DIAG]/[PARAM-DIAG] логи выше для локализации источника.")
                    except Exception as e:
                        print(f"🛑 ❌ Save при автостопе не удался: {e}")
                    stopped_by_nonfinite_limit = True
                    stopped_early = True
                    break

                global_step += 1

                now = time.perf_counter()
                # ПАТЧ: заменён блок периодического сохранения на асинхронный
                if now - last_ckpt_time >= CHECKPOINT_EVERY_SECONDS:
                    # ФИКС: 'latest' теперь запускается асинхронно (см. save_slot_async) --
                    # TPU не простаивает на время записи (~300с наблюдалось синхронно).
                    # best_train проверяем отдельно, синхронно как раньше -- срабатывает
                    # только на новый рекорд, не является узким местом.
                    save_slot_async(mngr_latest, latest_dir, global_step, params, opt_state,
                                    epoch, best_val_loss, best_train_loss, train_loss)
                    tl = float(jax.device_get(train_loss))
                    if tl < best_train_loss:
                        best_train_loss = tl
                        save_slot(mngr_best_train, best_train_dir, global_step, params, opt_state,
                                  epoch, best_val_loss, best_train_loss, train_loss)
                        upload_slot(best_train_dir, "best_train", global_step, f"train_loss={tl:.4f}", keep_last_n=1)
                        print(f"[BEST_TRAIN] Новый лучший train_loss: {tl:.4f} на шаге {global_step}")
                    last_ckpt_time = time.perf_counter()

                elapsed_session = time.perf_counter() - session_start_time
                if elapsed_session >= SESSION_TIME_BUDGET_SECONDS:
                    print(f"[SESSION LIMIT] Достигнут бюджет времени сессии "
                          f"({elapsed_session/3600:.2f} ч) -- сохраняюсь и завершаюсь gracefully...")
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
                        f"best_train={best_train_loss:.4f}"
                    )
                    if aux_info["expert_utilization"] is not None:
                        util = jax.device_get(aux_info["expert_utilization"])
                        util_std_per_layer = util.std(axis=-1)
                        worst_layer = int(util_std_per_layer.argmax())
                        print(
                            f"           expert utilization std (max over layers, layer {worst_layer}): "
                            f"{util_std_per_layer[worst_layer]:.4f} | ideal ~= 0, uniform = 1/{config.num_experts - 1}"
                        )
                    if aux_info.get("moe_dropped_ratio") is not None:
                        dropped = jax.device_get(aux_info["moe_dropped_ratio"])
                        worst_drop_layer = int(dropped.argmax())
                        print(
                            f"           moe dropped_ratio (max over layers, layer {worst_drop_layer}): "
                            f"{dropped[worst_drop_layer]:.4f}  (ideal ~= 0 after warmup)"
                        )

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
 
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        eval_no_improve_count = 0
                        if eval_loss < best_val_loss:
                            best_val_loss = eval_loss
                            save_slot(mngr_best_val, best_val_dir, global_step, params, opt_state, epoch, best_val_loss, best_train_loss)
                            upload_slot(best_val_dir, "best_val", global_step, f"val_loss={eval_loss:.4f}", keep_last_n=1)
                            print(f"[BEST_VAL] Новый лучший val_loss: {best_val_loss:.4f} на шаге {global_step}")
                    else:
                        eval_no_improve_count += 1
                        if eval_no_improve_count >= eval_patience:
                            print(
                                f"[EARLY STOP] Частичный val loss не улучшался {eval_patience} "
                                "проверок подряд. Останавливаю обучение немедленно."
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

    # ПАТЧ: финализировать любой незавершённый async-сейв перед выходом
    finalized = _finalize_pending_save(mngr_latest)
    if finalized is not None:
        upload_slot(latest_dir, "latest", finalized["step"], "FINAL", keep_last_n=HF_LATEST_KEEP_N)

    print("Обучение завершено (для этой сессии).")
 
 
if __name__ == "__main__":
    main_execution()
