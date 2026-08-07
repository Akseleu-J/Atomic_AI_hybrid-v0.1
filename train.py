import glob
import os
import re
import time
import json
import signal
import sys

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
    from huggingface_hub import HfApi, snapshot_download, upload_folder, create_repo
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_REPO_ID = os.environ.get("HF_REPO_ID", "atomic-ai-labs/atomic-light-v0.1")
    _HAS_HF = bool(HF_TOKEN)
    if _HAS_HF:
        print(f"[HF] ✅ Интеграция: {HF_REPO_ID}")
except ImportError:
    _HAS_HF = False
    print("[WARN] pip install -q huggingface_hub")

# ФИКС: чекпоинт по ВРЕМЕНИ, а не по шагам. При переменной скорости (например,
# после смены архитектуры MoE) фиксированный CHECKPOINT_EVERY=1000 шагов может
# означать от 1 до 4+ часов между сейвами -- на Kaggle с обрывами по таймауту
# это слишком дорого при промахе. Раз в 15-20 минут -- разумный компромисс
# между накладными расходами на сейв (I/O + upload) и риском потери прогресса.
CHECKPOINT_EVERY_SECONDS = 15 * 60

# Держим на HF Hub не больше这 N последних чекпоинтов -- иначе репозиторий растёт
# бесконечно и КАЖДЫЙ холодный старт скачивает всё больше мусора.
HF_KEEP_LAST_N = 2

# ФИКС: детерминированная доля датасета для тестового прогона. Задаётся ДО
# train/val split, чтобы при каждом рестарте (новый процесс python) выборка
# была той же самой -- иначе resume будет "плавать" по разным подмножествам
# данных между сессиями.
DATASET_FRACTION = 0.30
DATASET_FRACTION_SEED = 777


def upload_ckpt(ckpt_dir, step, msg=""):
    """Upload ONLY the specific step_N folder to HF Hub, then prune old ones."""
    if not _HAS_HF:
        return
    step_dir = os.path.join(ckpt_dir, f"step_{step}")
    if not os.path.exists(step_dir):
        print(f"[HF] ⚠️ step_{step} not found, skipping upload")
        return
    try:
        api = HfApi(token=HF_TOKEN)
        create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
        st_path = os.path.join(step_dir, "STATUS.txt")
        with open(st_path, "w") as f:
            f.write(f"IDLE: last_step={step} | t={time.time()}\n")
        upload_folder(
            folder_path=step_dir,
            repo_id=HF_REPO_ID,
            repo_type="model",
            path_in_repo=f"step_{step}",
            commit_message=f"Step {step} {msg}",
        )
        print(f"[HF] ✅ Uploaded: step {step}")

        # ФИКС: чистим старые чекпоинты на хабе, оставляя HF_KEEP_LAST_N последних.
        try:
            all_files = api.list_repo_files(HF_REPO_ID, repo_type="model")
            found_steps = set()
            for f_path in all_files:
                m = re.match(r"^step_(\d+)/", f_path)
                if m:
                    found_steps.add(int(m.group(1)))
            steps_sorted = sorted(found_steps, reverse=True)
            steps_to_delete = steps_sorted[HF_KEEP_LAST_N:]
            for old_step in steps_to_delete:
                try:
                    api.delete_folder(
                        path_in_repo=f"step_{old_step}",
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                    )
                    print(f"[HF] 🗑️ Удалён старый чекпоинт: step {old_step}")
                except Exception as e_del:
                    print(f"[HF] ⚠️ Не удалось удалить step {old_step}: {e_del}")
        except Exception as e_list:
            print(f"[HF] ⚠️ Не удалось получить список файлов для чистки: {e_list}")
    except Exception as e:
        print(f"[HF] ❌ Upload error: {e}")


def download_latest(ckpt_dir):
    """Download latest checkpoint from HF Hub"""
    if not _HAS_HF:
        return None
    try:
        print(f"[HF] Downloading from {HF_REPO_ID}...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=ckpt_dir,
            repo_type="model",
            allow_patterns=["step_*/**", "STATUS.txt", "metadata.json"],
        )
        items = [d for d in os.listdir(ckpt_dir) if d.startswith("step_")]
        if not items:
            return None
        latest = max(int(d.split("_")[1]) for d in items)
        print(f"[HF] Found checkpoint: step {latest}")
        return latest
    except Exception as e:
        print(f"[HF] Download failed: {e}")
        return None


from model import FullHybridMoEModel, ModelConfig, set_model_mesh, get_model_mesh
from optimizer import compute_loss, make_hybrid_optimizer
from utils import path_to_str


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
        if "experts_block" in path_to_str(path):
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

    opt_state_abstract = jax.eval_shape(lambda: tx.init(abstract_params))
    opt_state_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, opt_state_abstract)

    def model_apply_wrapped(variables, input_ids, rngs=None, deterministic=True, **kwargs):
        return model.apply(
            variables, input_ids,
            rngs=rngs, deterministic=deterministic,
            **kwargs
        )

    # --- Micro-step: compute grads, accumulate ---
    def distributed_train_step_micro(p, s, b, r, accum_grads):
        loss_fn = lambda param: compute_loss(
            param, model_apply_wrapped, b, config,
            rngs={"dropout": r},
            deterministic=False, return_aux=True,
            ce_chunk_size=2048
        )
        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
        return p, s, new_accum, loss, aux_info

    # --- Apply-step: average grads, update params ---
    def distributed_apply_step(p, s, accum_grads, n_accum):
        avg_grads = jax.tree_util.tree_map(lambda g: g / n_accum, accum_grads)

        # Gradient clipping — только если NaN останется после фикса маскировки
        global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(avg_grads)))
        clip_factor = jnp.minimum(1.0, 1.0 / (global_norm + 1e-6))
        avg_grads = jax.tree_util.tree_map(lambda g: g * clip_factor, avg_grads)

        updates, new_s = tx.update(avg_grads, s, p)
        new_p = optax.apply_updates(p, updates)
        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        return new_p, new_s, zero_accum

    def distributed_val_step(p, b):
        return compute_loss(
            p, model_apply_wrapped, b, config,
            rngs=None,
            deterministic=True
        )

    aux_info_sharding = {
        "ce_loss": NamedSharding(mesh, P()),
        "aux_loss": NamedSharding(mesh, P()),
        "z_loss": NamedSharding(mesh, P()),
        "expert_utilization": NamedSharding(mesh, P(None)),
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

    # ФИКС: детерминированная подвыборка ДО train/val split. Тот же seed при
    # каждом рестарте процесса -> тот же набор блоков используется всегда,
    # так что resume (skip_batches) не "уезжает" на другое подмножество данных.
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

    # Отдельный shuffle-seed на разбиение train/val -- тоже детерминированный.
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
                # ФИКС: при resume пропускаем уже пройденные микрошаги текущей
                # эпохи БЕЗ чтения данных (дёшево -- это просто арифметика
                # индексов, реальный I/O начинается только с start_step).
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
    checkpoint_dir = "/kaggle/working/orbax_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Resume logic: local first, then HF Hub ---
    resume_step = None
    local_items = [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
    if local_items:
        resume_step = max(int(d.split("_")[1]) for d in local_items)
        print(f"[LOCAL] 📦 Found checkpoint: step {resume_step}")

    if resume_step is None and _HAS_HF:
        resume_step = download_latest(checkpoint_dir)

    resume = (resume_step is not None)
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    config = ModelConfig(
        d_model=768,
        d_state=128,
        d_conv=4,
        expand=2,
        n_heads=8,
        d_latent=512,
        d_ff=4096,
        num_experts=8,          # DenseMoE тестовый режим (см. обсуждение)
        top_k=2,                # не используется DenseMoE, оставлено для совместимости конфига
        num_layers=21,
        layers_per_block=3,
        vocab_size=151936,
        dropout_rate=0.1,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.0001,
        moe_capacity_factor=1.0,  # не используется DenseMoE
        tie_embeddings=True,
        label_smoothing=0.0,
        router_noise_std=0.1,
        use_flash_attention=True,
        deltanet_chunk_size=256,
    )
    file_pairs = [
        (
            "/kaggle/input/datasets/akseleu1j/atentic-data/agentic_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/atentic-data/agentic_labels.npy",
        ),
        (
            "/kaggle/input/datasets/akseleu1j/coding-labels/coding_A_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/coding-labels/coding_A_labels.npy",
        ),
        (
            "/kaggle/input/datasets/akseleu1j/coding-ids/coding_B_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/coding-ids/coding_B_labels.npy",
        ),
        (
            "/kaggle/input/datasets/akseleu1j/simple-data/common_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/simple-data/common_labels.npy",
        ),
        (
            "/kaggle/input/datasets/akseleu1j/math-ids/math_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/math-ids/math_labels.npy",
        ),
        (
            "/kaggle/input/datasets/akseleu1j/reasoning-ids/reasoning_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/reasoning-ids/reasoning_labels.npy",
        ),
    ]

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

    # --- Gradient Accumulation config ---
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

    # --- Sanity check: временный генератор без skip, только для проверки формата ---
    _sanity_stream, _, _, _ = dataloader_multi_source(
        file_pairs, micro_batch_size, data_sharding, seq_len=seq_len,
        dataset_fraction=DATASET_FRACTION, fraction_seed=DATASET_FRACTION_SEED,
    )
    print("[SANITY] Проверка первого батча...")
    test_batch = next(_sanity_stream)
    max_label = int(jnp.max(test_batch['labels']))
    min_label = int(jnp.min(test_batch['labels']))
    print(f"[SANITY] Labels range: [{min_label}, {max_label}], vocab_size={config.vocab_size}")

    # -100 — стандартный pad token, это нормально
    assert max_label < config.vocab_size, f"max_label={max_label} >= vocab_size!"

    valid_mask = test_batch['labels'] >= 0
    n_valid = int(jnp.sum(valid_mask))
    print(f"[SANITY] Валидных labels в батче: {n_valid}/{valid_mask.size} ({100*n_valid/valid_mask.size:.1f}%)")
    if n_valid == 0:
        raise ValueError("Все labels в первом батче маскированы (pad) — loss будет NaN!")
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
          f"(с FSDP на чип реально хранится в среднем ~{weights_bytes / 1e9 / n_devices_display:.2f} ГБ -- "
          "точная цифра зависит от того, какие оси делимы на n_devices, см. _get_shard_spec)")

    opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)

    zero_accum = jax.jit(
        lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),
        out_shardings=param_sharding,
    )(params)
    accum_grads = zero_accum

    # --- Resume: restore BOTH params + opt_state ---
    if resume and resume_step is not None:
        print(f"[RESUME] ⬆️ Restoring step {resume_step}...")
        ckpt_path = os.path.join(checkpoint_dir, f"step_{resume_step}")
        restorer = ocp.StandardCheckpointer()
        try:
            ckpt = restorer.restore(
                ckpt_path, item={"params": params, "opt_state": opt_state}
            )
            params = ckpt["params"]
            opt_state = ckpt["opt_state"]
            accum_grads = jax.jit(
                lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),
                out_shardings=param_sharding,
            )(params)

            meta_path = os.path.join(ckpt_path, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                start_epoch = meta.get("epoch", 0)
                global_step = meta.get("global_step", resume_step)
                best_val_loss = meta.get("best_val_loss", float("inf"))
            else:
                global_step = resume_step
            global_rng = jax.random.PRNGKey(42 + global_step)
            print(f"[RESUME] ✅ Restored: step={global_step}, best_val={best_val_loss:.4f}")
        except Exception as e:
            print(f"[RESUME] ❌ Error: {e}. Starting fresh.")
            resume = False
            global_step = 0
    else:
        print("[RESUME] 🆕 Fresh start.")

    # ФИКС: создаём боевой train_stream ПОСЛЕ восстановления global_step,
    # передавая skip_batches -- иначе каждый рестарт начинает читать данные
    # с начала эпохи заново, и модель никогда не доходит до хвоста датасета.
    skip_micro_steps = global_step * accum_steps
    train_stream, val_factory, _, val_steps = dataloader_multi_source(
        file_pairs, micro_batch_size, data_sharding, seq_len=seq_len,
        dataset_fraction=DATASET_FRACTION, fraction_seed=DATASET_FRACTION_SEED,
        skip_batches=skip_micro_steps,
    )

    # --- Pre-compile ---
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

    # --- Checkpoint managers ---
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer(), options)
    best_checkpoint_dir = "/kaggle/working/orbax_checkpoints_best"
    best_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    best_mngr = ocp.CheckpointManager(best_checkpoint_dir, ocp.StandardCheckpointer(), best_options)

    # --- Init loop variables ---
    stopped_early = False
    eval_no_improve_count = 0
    epochs_without_improvement = 0
    best_eval_loss = float("inf")
    epoch = start_epoch  # ФИКС: определена до цикла, чтобы emergency_save видела актуальное значение даже до первой итерации for

    def _save_checkpoint_and_meta(step, tag=""):
        mngr.save(
            step,
            args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}),
        )
        mngr.wait_until_finished()
        meta = {
            "global_step": int(step),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "timestamp": time.time(),
        }
        meta_path = os.path.join(checkpoint_dir, f"step_{step}", "metadata.json")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        upload_ckpt(checkpoint_dir, step, tag)

    # --- Emergency save handler ---
    def emergency_save(signum=None, frame=None):
        print(f"\n🚨 [EMERGENCY] Saving step {global_step}...")
        try:
            _save_checkpoint_and_meta(global_step, "EMERGENCY")
            print(f"🚨 ✅ Emergency save done: step {global_step}")
        except Exception as e:
            print(f"🚨 ❌ Emergency save failed: {e}")
        # Kaggle обычно убивает процесс жёстко вскоре после SIGTERM -- явно
        # завершаем сами, чтобы не оставлять процесс в неопределённом состоянии.
        sys.exit(0)

    signal.signal(signal.SIGTERM, emergency_save)
    signal.signal(signal.SIGINT, emergency_save)

    total_tokens_processed = 0
    epoch_start_time = time.perf_counter()
    last_ckpt_time = time.perf_counter()

    # ==================== TRAINING LOOP ====================
    for epoch in range(start_epoch, epochs):
        for micro_step in range(micro_steps_per_epoch):
            global_rng, step_rng = jax.random.split(global_rng)

            _t0 = time.perf_counter()
            try:
                batch = next(train_stream)
            except StopIteration:
                # Достигли конца прохода по данным (может случиться, если
                # skip_batches почти догнал длину эпохи) -- просто выходим,
                # эпоха фактически завершена.
                print("[DATA] Поток данных исчерпан для этой эпохи.")
                break
            _t_data = time.perf_counter() - _t0

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

                _t_apply = time.perf_counter()
                params, opt_state, accum_grads = compiled_apply(
                    params, opt_state, accum_grads, accum_steps
                )
                if micro_step < 30:
                    jax.block_until_ready(params)
                _t_apply_total = time.perf_counter() - _t_apply

                global_step += 1

                # ФИКС: чекпоинт по прошедшему времени, а не по количеству шагов.
                now = time.perf_counter()
                if now - last_ckpt_time >= CHECKPOINT_EVERY_SECONDS:
                    print(f"[CKPT] 💾 Saving step {global_step} (прошло {(now - last_ckpt_time)/60:.1f} мин)...")
                    _save_checkpoint_and_meta(global_step)
                    last_ckpt_time = now

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
                        f"z={jax.device_get(aux_info['z_loss']):.5f})"
                    )
                    if aux_info["expert_utilization"] is not None:
                        util = jax.device_get(aux_info["expert_utilization"])
                        # DenseMoE: expert_utilization теперь (num_blocks, E) --
                        # среднее по всем токенам на блок, std по экспертам
                        # внутри каждого блока по-прежнему валиден.
                        util_std_per_layer = util.std(axis=-1)
                        worst_layer = int(util_std_per_layer.argmax())
                        print(
                            f"           expert utilization std (max over layers, layer {worst_layer}): "
                            f"{util_std_per_layer[worst_layer]:.4f} | ideal ~= 0, uniform = 1/{config.num_experts}"
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
                    else:
                        eval_no_improve_count += 1
                        if eval_no_improve_count >= eval_patience:
                            print(
                                f"[EARLY STOP] Частичный val loss не улучшался {eval_patience} "
                                "проверок подряд. Останавливаю обучение немедленно."
                            )
                            _save_checkpoint_and_meta(global_step, "EARLY_STOP")
                            best_mngr.save(
                                global_step,
                                args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}),
                            )
                            print(f"[ORBAX] Финальный чекпоинт (шаг {global_step}) сохранён в оба каталога.")
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

        best_val_loss_before = best_val_loss
        _save_checkpoint_and_meta(global_step, "EPOCH_END")
        print(f"[ORBAX] Чекпоинт для шага {global_step} успешно зафиксирован.")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_without_improvement = 0
            best_mngr.save(
                global_step,
                args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}),
            )
            print(f"[ORBAX] Новый лучший val loss ({best_val_loss:.4f}) -- сохранён в {best_checkpoint_dir}")
        else:
            epochs_without_improvement += 1
            print(
                f"[EARLY STOP] val loss не улучшился {epochs_without_improvement} эпох(и) подряд "
                f"(лучший: {best_val_loss:.4f})"
            )
            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"[EARLY STOP] Останавливаю обучение -- val loss не улучшался "
                    f"{early_stop_patience} эпохи подряд. Лучшие веса лежат в {best_checkpoint_dir}."
                )
                break

    print("Обучение завершено.")


if __name__ == "__main__":
    main_execution()
