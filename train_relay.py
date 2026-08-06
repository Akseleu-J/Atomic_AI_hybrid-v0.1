"""
train_relay.py — обучение с авто-resume и HF Hub relay.
Вставьте этот файл в репо Atomic_AI_hybrid-v0.1.
В ноутбуке: from train_relay import main_execution; main_execution()
"""
import glob
import os
import re
import signal
import sys
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from huggingface_hub import HfApi, snapshot_download, upload_folder, hf_hub_download, create_repo
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_REPO_ID = os.environ.get("HF_REPO_ID", "your-team/atomic-ai")
    _HAS_HF = bool(HF_TOKEN)
except ImportError:
    _HAS_HF = False
    print("[WARN] pip install -q huggingface_hub")

from model import FullHybridMoEModel, ModelConfig, set_model_mesh
from optimizer import compute_loss, make_hybrid_optimizer
from utils import path_to_str

# ────────────────────────────────
# HF Hub helpers
# ────────────────────────────────
CHECKPOINT_EVERY = 500
MAX_POLL_MINUTES = 30


def _write_status(ckpt_dir, text):
    p = os.path.join(ckpt_dir, "STATUS.txt")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(p, "w") as f:
        f.write(text + "\n")
    if _HAS_HF:
        try:
            api = HfApi(token=HF_TOKEN)
            create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=p, path_in_repo="STATUS.txt",
                            repo_id=HF_REPO_ID, repo_type="model")
        except Exception:
            pass


def upload_ckpt(ckpt_dir, step, msg=""):
    if not _HAS_HF:
        return
    try:
        api = HfApi(token=HF_TOKEN)
        create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
        _write_status(ckpt_dir, f"IDLE: last_step={step} | user={os.environ.get('KAGGLE_USER','?')} | t={time.time()}")
        upload_folder(folder_path=ckpt_dir, repo_id=HF_REPO_ID, repo_type="model",
                      commit_message=f"Step {step} {msg}")
        print(f"[HF] ✅ Загружено: step {step}")
    except Exception as e:
        print(f"[HF] ❌ Ошибка upload: {e}")


def download_latest(ckpt_dir):
    if not _HAS_HF:
        return None
    try:
        print(f"[HF] ⬇️ Скачиваю {HF_REPO_ID}...")
        snapshot_download(repo_id=HF_REPO_ID, local_dir=ckpt_dir, repo_type="model",
                          allow_patterns=["checkpoints/**", "STATUS.txt", "metadata.json"])
        cp = os.path.join(ckpt_dir, "checkpoints")
        if not os.path.exists(cp):
            return None
        items = [d for d in os.listdir(cp) if d.startswith("step_")]
        if not items:
            return None
        latest = max(int(d.split("_")[1]) for d in items)
        print(f"[HF] 📦 Найден чекпоинт: step {latest}")
        return latest
    except Exception as e:
        print(f"[HF] Скачивание не удалось: {e}")
        return None


def poll_idle(max_min=MAX_POLL_MINUTES):
    if not _HAS_HF:
        return False
    print(f"[RELAY] ⏳ Жду освобождения (макс {max_min} мин)...")
    for m in range(max_min):
        try:
            hf_hub_download(repo_id=HF_REPO_ID, filename="STATUS.txt",
                            local_dir="/tmp/hf_poll", repo_type="model")
            with open("/tmp/hf_poll/STATUS.txt") as f:
                st = f.read().strip()
            if "IDLE" in st:
                print(f"[RELAY] ✅ Предыдущий закончил: {st[:80]}")
                return True
            if m % 5 == 0:
                print(f"[RELAY] ⏳ Жду... ({m} мин) {st[:60]}")
        except Exception:
            if m % 10 == 0:
                print(f"[RELAY] ⏳ HF пустой ({m} мин)...")
        time.sleep(60)
    print("[RELAY] ⚠️ Таймаут. Запускаюсь с нуля.")
    return False


# ────────────────────────────────
# TPU mesh & shard helpers
# ────────────────────────────────
def make_tpu_mesh():
    devices = jax.devices()
    n = len(devices)
    mesh_devices = mesh_utils.create_device_mesh((n,), devices)
    return Mesh(mesh_devices, axis_names=("tpu_nodes",))


def make_shard_and_compile(config, total_steps, batch_size, seq_len=8192):
    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]
    if batch_size % n_devices != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by n_devices={n_devices}")

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
        return model.apply(variables, input_ids, rngs=rngs, deterministic=deterministic, **kwargs)

    def distributed_train_step(p, s, b, r):
        loss_fn = lambda param: compute_loss(
            param, model_apply_wrapped, b, config,
            rngs={"dropout": r}, deterministic=False, return_aux=True, ce_chunk_size=2048
        )
        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, new_s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), new_s, loss, aux_info

    def distributed_val_step(p, b):
        return compute_loss(p, model_apply_wrapped, b, config, rngs=None, deterministic=True)

    aux_info_sharding = {
        "ce_loss": NamedSharding(mesh, P()),
        "aux_loss": NamedSharding(mesh, P()),
        "z_loss": NamedSharding(mesh, P()),
        "expert_utilization": NamedSharding(mesh, P(None, None)),
    }

    compiled_train = jax.jit(
        distributed_train_step,
        donate_argnums=(0, 1),
        in_shardings=(
            param_sharding, opt_state_sharding,
            {"input_ids": data_sharding, "labels": data_sharding},
            NamedSharding(mesh, P(None)),
        ),
        out_shardings=(
            param_sharding, opt_state_sharding,
            NamedSharding(mesh, P()), aux_info_sharding,
        ),
    )
    compiled_val = jax.jit(
        distributed_val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=NamedSharding(mesh, P()),
    )
    return compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding


# ────────────────────────────────
# Data loading
# ────────────────────────────────
def build_manifest(file_pairs):
    manifest = []
    total = 0
    for ids_path, lbls_path in file_pairs:
        n_rows = np.load(ids_path, mmap_mode="r").shape[0]
        manifest.append((ids_path, lbls_path, n_rows))
        total += n_rows
        print(f"[DATA] {os.path.basename(ids_path)}: {n_rows:,} блоков")
    print(f"[DATA] Всего: {total:,} блоков")
    return manifest


def dataloader_multi_source(file_pairs, batch_size, data_sharding, seq_len, val_split=0.05):
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

    val_size = int(total_blocks * val_split)
    train_size = total_blocks - val_size
    all_idx = np.arange(total_blocks)
    np.random.RandomState(42).shuffle(all_idx)
    train_idx_pool = all_idx[:train_size]
    val_idx_pool = all_idx[train_size:]

    def _generator(pool, is_train=True):
        idx_local = np.copy(pool)
        local_rng = np.random.RandomState(123)
        while True:
            if is_train:
                local_rng.shuffle(idx_local)
            for step in range(len(idx_local) // batch_size):
                batch_idx = idx_local[step * batch_size: (step + 1) * batch_size]
                ids_np, lbls_np = _gather_batch(batch_idx)
                yield {
                    "input_ids": jax.device_put(jnp.array(ids_np), data_sharding),
                    "labels": jax.device_put(jnp.array(lbls_np), data_sharding),
                }
            if not is_train:
                break

    return (
        _generator(train_idx_pool, True),
        lambda: _generator(val_idx_pool, False),
        train_size // batch_size,
        val_size // batch_size,
    )


# ────────────────────────────────
# Main
# ────────────────────────────────
def main_execution():
    config = ModelConfig(
        d_model=768, d_state=128, d_conv=4, expand=2, n_heads=8,
        d_latent=512, d_ff=6144, num_experts=8, top_k=2,
        num_layers=21, layers_per_block=3, vocab_size=131072,
        dropout_rate=0.1, router_aux_loss_coef=0.01,
        router_z_loss_coef=0.0001, moe_capacity_factor=1.0,
        tie_embeddings=True, label_smoothing=0.0,
        router_noise_std=0.3, use_flash_attention=True,
        deltanet_chunk_size=256,
    )

    file_pairs = [
        ("/kaggle/input/datasets/akseleu1j/atentic-data/agentic_input_ids.npy",
         "/kaggle/input/datasets/akseleu1j/atentic-data/agentic_labels.npy"),
        ("/kaggle/input/datasets/akseleu1j/coding-labels/coding_A_input_ids.npy",
         "/kaggle/input/datasets/akseleu1j/coding-labels/coding_A_labels.npy"),
        ("/kaggle/input/datasets/akseleu1j/coding-ids/coding_B_input_ids.npy",
         "/kaggle/input/datasets/akseleu1j/coding-ids/coding_B_labels.npy"),
        ("/kaggle/input/datasets/akseleu1j/simple-data/common_input_ids.npy",
         "/kaggle/input/datasets/akseleu1j/simple-data/common_labels.npy"),
        ("/kaggle/input/datasets/akseleu1j/math-ids/math_input_ids.npy",
         "/kaggle/input/datasets/akseleu1j/math-ids/math_labels.npy"),
        ("/kaggle/input/datasets/akseleu1j/reasoning-ids/reasoning_input_ids.npy",
         "/kaggle/input/datasets/akseleu1j/reasoning-ids/reasoning_labels.npy"),
    ]

    for ids_path, lbls_path in file_pairs:
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"Не найден: {ids_path}")
        if not os.path.exists(lbls_path):
            raise FileNotFoundError(f"Не найден: {lbls_path}")
    print("✅ Все файлы найдены.")

    manifest = build_manifest(file_pairs)
    total_blocks = sum(n for _, _, n in manifest)
    print(f"Всего блоков: {total_blocks:,}")

    batch_size = 8
    seq_len = 4096
    epochs = 1
    early_stop_patience = 2
    eval_every_steps = 1000
    eval_batches = 40
    eval_patience = 4
    checkpoint_every = CHECKPOINT_EVERY

    val_split = 0.05
    val_size = int(total_blocks * val_split)
    train_size = total_blocks - val_size
    train_steps_per_epoch = train_size // batch_size
    total_train_steps = train_steps_per_epoch * epochs

    print(f"[TPU] Компиляция под {total_train_steps} шагов...")
    compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding = (
        make_shard_and_compile(config, total_train_steps, batch_size, seq_len)
    )
    print(f"[TPU] Устройств: {mesh.shape['tpu_nodes']}")

    train_stream, val_factory, _, val_steps = dataloader_multi_source(
        file_pairs, batch_size, data_sharding, seq_len=seq_len
    )

    # ── HF resume logic ──
    checkpoint_dir = "/kaggle/working/orbax_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_step = None
    local_items = [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
    if local_items:
        resume_step = max(int(d.split("_")[1]) for d in local_items)
        print(f"[LOCAL] 📦 Чекпоинт: step {resume_step}")

    if resume_step is None and _HAS_HF:
        poll_idle(max_min=MAX_POLL_MINUTES)
        resume_step = download_latest(checkpoint_dir)

    resume = (resume_step is not None)
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    global_rng = jax.random.PRNGKey(42)
    init_params_fn = jax.jit(
        lambda rng: model.init(rng, jnp.zeros((batch_size, seq_len), dtype=jnp.int32))["params"],
        out_shardings=param_sharding,
    )
    params = init_params_fn(global_rng)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Параметров: {total_params:,} (≈ {total_params/1e9:.2f} млрд)")

    opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)

    if resume and resume_step is not None:
        print(f"[RESUME] ⬆️ Восстанавливаю шаг {resume_step}...")
        try:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{resume_step}")
            restorer = ocp.StandardCheckpointer()
            params = restorer.restore(ckpt_path, item=params)
            opt_state = restorer.restore(ckpt_path, item=opt_state)
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
            print(f"[RESUME] ✅ step={global_step}, best_val={best_val_loss:.4f}")
        except Exception as e:
            print(f"[RESUME] ❌ {e}. С нуля.")
            resume = False
            global_step = 0
    else:
        print("[RESUME] 🆕 Новое обучение.")

    _write_status(checkpoint_dir, f"RUNNING: step={global_step} | user={os.environ.get('KAGGLE_USER','?')} | t={time.time()}")

    # ── Emergency save ──
    def emergency_save(signum=None, frame=None):
        print(f"\n🚨 [EMERGENCY] Сохраняю {global_step}...")
        try:
            mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer(),
                                         options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
            mngr.save(global_step, args=ocp.args.StandardSave(params))
            meta = {"global_step": int(global_step), "epoch": int(start_epoch),
                    "best_val_loss": float(best_val_loss), "timestamp": time.time()}
            with open(os.path.join(checkpoint_dir, f"step_{global_step}", "metadata.json"), "w") as f:
                json.dump(meta, f)
            upload_ckpt(checkpoint_dir, global_step, "EMERGENCY")
            print(f"🚨 ✅ Сохранено: {global_step}")
        except Exception as e:
            print(f"🚨 ❌ {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, emergency_save)
    signal.signal(signal.SIGINT, emergency_save)

    mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer(),
                                 options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
    best_mngr = ocp.CheckpointManager(
        "/kaggle/working/orbax_checkpoints_best", ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    )

    epochs_without_improvement = 0
    eval_no_improve_count = 0
    stopped_early = False
    total_tokens_processed = 0
    epoch_start_time = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        for step in range(global_step, train_steps_per_epoch):
            global_rng, step_rng = jax.random.split(global_rng)

            _t0 = time.perf_counter()
            batch = next(train_stream)
            _t_data = time.perf_counter() - _t0
            total_tokens_processed += batch_size * seq_len

            _t1 = time.perf_counter()
            params, opt_state, train_loss, aux_info = compiled_train(params, opt_state, batch, step_rng)
            if step < 30:
                jax.block_until_ready(train_loss)
            _t_compute = time.perf_counter() - _t1

            global_step += 1

            if step < 30:
                print(f"[TIMING] step {step}: данные={_t_data*1000:.0f}мс TPU={_t_compute*1000:.0f}мс")

            if global_step % 10 == 0:
                print(f"Epoch:{epoch} Step:{global_step}/{train_steps_per_epoch} | "
                      f"Loss:{jax.device_get(train_loss):.4f} "
                      f"(ce={jax.device_get(aux_info['ce_loss']):.4f} "
                      f"aux={jax.device_get(aux_info['aux_loss']):.4f} "
                      f"z={jax.device_get(aux_info['z_loss']):.5f})")

            # ── Checkpoint every N ──
            if global_step % checkpoint_every == 0:
                print(f"[CKPT] 💾 Сохраняю {global_step}...")
                mngr.save(global_step, args=ocp.args.StandardSave(params))
                meta = {"global_step": int(global_step), "epoch": int(epoch),
                        "best_val_loss": float(best_val_loss),
                        "train_loss": float(jax.device_get(train_loss)),
                        "timestamp": time.time()}
                with open(os.path.join(checkpoint_dir, f"step_{global_step}", "metadata.json"), "w") as f:
                    json.dump(meta, f)
                upload_ckpt(checkpoint_dir, global_step)
                _write_status(checkpoint_dir, f"RUNNING: step={global_step} | user={os.environ.get('KAGGLE_USER','?')}")

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
                print(f"[EVAL] Step {global_step}: val loss = {eval_loss:.4f}")
                if eval_loss < best_val_loss:
                    best_val_loss = eval_loss
                    eval_no_improve_count = 0
                    best_mngr.save(global_step, args=ocp.args.StandardSave(params))
                else:
                    eval_no_improve_count += 1
                    if eval_no_improve_count >= eval_patience:
                        print("[EARLY STOP] Останавливаю.")
                        upload_ckpt(checkpoint_dir, global_step, "EARLY_STOP")
                        stopped_early = True
                        break

        if stopped_early:
            break

        print(f"--- Эпоха {epoch} завершена ---")
        val_stream = val_factory()
        total_val_loss = sum(jax.device_get(compiled_val(params, next(val_stream))) for _ in range(val_steps))
        mean_val_loss = total_val_loss / val_steps
        print(f"===> VALIDATION LOSS: {mean_val_loss:.4f} <===")

        epoch_elapsed = time.perf_counter() - epoch_start_time
        print(f"Скорость: {total_tokens_processed/epoch_elapsed/1e6:.2f} млн токенов/сек")
        total_tokens_processed = 0
        epoch_start_time = time.perf_counter()

        mngr.save(global_step, args=ocp.args.StandardSave(params))
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_without_improvement = 0
            best_mngr.save(global_step, args=ocp.args.StandardSave(params))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print("[EARLY STOP] Останавливаю.")
                break

    upload_ckpt(checkpoint_dir, global_step, "FINAL")
    print("Обучение завершено.")


if __name__ == "__main__":
    main_execution()
