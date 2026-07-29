import glob
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from model import FullHybridMoEModel, ModelConfig
from optimizer import compute_loss, make_hybrid_optimizer
from utils import path_to_str


def make_tpu_mesh():
    devices = jax.devices()
    n = len(devices)
    # FIX: was hard-coded to (8,), assuming a TPU v5e-8 pod. That crashes with
    # "Number of devices N must equal the product of mesh_shape (8,)" on ANY other
    # device count -- including a 1-device CPU smoke test, which is exactly what
    # produced this error. Always build the mesh to match whatever's actually available:
    # 1 device for a CPU sanity check, 8 for the real TPU v5e-8 run, etc.
    mesh_devices = mesh_utils.create_device_mesh((n,), devices)
    return Mesh(mesh_devices, axis_names=("tpu_nodes",))


def make_shard_and_compile(config: ModelConfig, total_steps: int, batch_size: int):
    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]

    # Fail fast with a clear message instead of a deep XLA ValueError three frames down.
    if batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by the number of devices "
            f"({n_devices}); data is sharded along the batch axis across devices."
        )
    experts_divide_evenly = (config.num_experts % n_devices == 0)
    if not experts_divide_evenly:
        print(
            f"[WARN] num_experts={config.num_experts} is not divisible by n_devices={n_devices}; "
            "expert weights will be REPLICATED instead of sharded across devices for this run "
            "(fine for a CPU/small-scale smoke test, but check pod topology before a real TPU run)."
        )

    tx = make_hybrid_optimizer(total_steps=total_steps)
    model = FullHybridMoEModel(cfg=config)

    init_rng = jax.random.PRNGKey(0)
    abstract_params = jax.eval_shape(
        lambda: model.init(init_rng, jnp.zeros((batch_size, 8192), dtype=jnp.int32))
    )["params"]

    data_sharding = NamedSharding(mesh, P("tpu_nodes", None))

    def _get_param_sharding(path, param):
        path_str = path_to_str(path)
        if "experts_block" in path_str and experts_divide_evenly:
            return NamedSharding(mesh, P("tpu_nodes", None, None))
        return NamedSharding(mesh, P(None))

    param_sharding = jax.tree_util.tree_map_with_path(_get_param_sharding, abstract_params)

    # Optimizer state for expert weights is kept on the same chips as the expert weights
    # themselves, to avoid unnecessary resharding/communication every step.
    opt_state_abstract = jax.eval_shape(lambda: tx.init(abstract_params))
    opt_state_sharding = jax.tree_util.tree_map_with_path(_get_param_sharding, opt_state_abstract)

    @jax.jit
    def distributed_train_step(p, s, b, r):
        loss_fn = lambda param: compute_loss(
            param, model.apply, b, config, rngs={"dropout": r}, deterministic=False, return_aux=True
        )
        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, new_s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), new_s, loss, aux_info

    @jax.jit
    def distributed_val_step(p, b):
        return compute_loss(p, model.apply, b, config, rngs=None, deterministic=True)

    compiled_train = jax.jit(
        distributed_train_step,
        in_shardings=(
            param_sharding,
            opt_state_sharding,
            {"input_ids": data_sharding, "labels": data_sharding},
            NamedSharding(mesh, P(None)),
        ),
        out_shardings=(
            param_sharding,
            opt_state_sharding,
            NamedSharding(mesh, P(None)),
            NamedSharding(mesh, P(None)),  # aux_info (ce_loss/aux_loss/z_loss/expert_utilization)
        ),
    )
    compiled_val = jax.jit(
        distributed_val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=NamedSharding(mesh, P(None)),
    )
    return compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding


def resolve_source_files(output_dir, prefix):
    """Finds one dataset-prep run's output files regardless of whether it finished as
    one merged file (prefix_input_ids.npy / prefix_labels.npy) or was left as
    individual numbered shards (prefix_shard_ids_N.npy / prefix_shard_lbls_N.npy) --
    e.g. because the final merge ran out of disk on a very large run, the way the
    reasoning set did before it got combined by hand. Works with either state."""
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
    """file_pairs: list of (ids_path, labels_path), from one or several dataset-prep
    runs. Returns [(ids_path, labels_path, n_rows), ...] -- just reads each .npy
    header via mmap (cheap), never loads the actual token data into RAM."""
    manifest = []
    total = 0
    for ids_path, lbls_path in file_pairs:
        n_rows = np.load(ids_path, mmap_mode="r").shape[0]
        manifest.append((ids_path, lbls_path, n_rows))
        total += n_rows
        print(f"[DATA] {os.path.basename(ids_path)}: {n_rows:,} блоков")
    print(f"[DATA] Комбинированный пул: {total:,} блоков из {len(manifest)} файл(ов)")
    return manifest


def dataloader_multi_source(file_pairs, batch_size, data_sharding, val_split=0.05):
    """Combines multiple (ids_path, labels_path) sources -- e.g. the separate agentic /
    coding / reasoning dataset-prep outputs -- into ONE training pool, WITHOUT
    physically concatenating them into a single file. That concatenation is exactly
    what ran the reasoning-set prep out of disk before (2.62B tokens -> ~21GB for one
    contiguous array). Instead this builds a global block index -> (source file, local
    row) lookup and reads each batch's rows directly from the relevant memmapped .npy
    files, so combining three (or any number of) large sources costs no extra disk or
    RAM beyond what's already on disk."""
    manifest = build_manifest(file_pairs)
    sizes = np.array([n for _, _, n in manifest])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    total_blocks = int(offsets[-1])
    context_length = np.load(manifest[0][0], mmap_mode="r").shape[1]

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
            ids_out[m] = _get_mmap(ids_path)[local_idx]
            lbls_out[m] = _get_mmap(lbls_path)[local_idx]
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


def main_execution():
    config = ModelConfig()

    # Точки монтирования трёх отдельно подготовленных датасетов -- ПОПРАВЬТЕ пути под
    # реальные имена ваших Kaggle Datasets (Add Data -> выбираете каждый из трёх выводов
    # notebook'ов подготовки, Kaggle монтирует их под /kaggle/input/<slug>/...).
    # resolve_source_files сам разберётся, попал ли каждый датасет в один объединённый
    # файл или остался шардами -- ничего вручную склеивать не нужно.
    DATASET_SOURCES = [
        ("/kaggle/input/agentic-dataset/processed_jax_data", "agentic"),
        ("/kaggle/input/coding-dataset/processed_jax_data", "coding"),
        ("/kaggle/input/reasoning-dataset/processed_jax_data", "reasoning"),
    ]
    file_pairs = []
    for output_dir, prefix in DATASET_SOURCES:
        file_pairs.extend(resolve_source_files(output_dir, prefix))

    manifest_preview = build_manifest(file_pairs)
    total_blocks = sum(n for _, _, n in manifest_preview)

    checkpoint_dir = "/kaggle/working/orbax_checkpoints"
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer(), options)
    best_checkpoint_dir = "/kaggle/working/orbax_checkpoints_best"
    best_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    best_mngr = ocp.CheckpointManager(best_checkpoint_dir, ocp.StandardCheckpointer(), best_options)

    batch_size = 32
    epochs = 6
    # Early stopping: if val loss doesn't improve for this many epochs in a row, stop
    # early instead of continuing to fit noise once the model has stopped generalizing.
    early_stop_patience = 2

    train_steps_per_epoch = (int(total_blocks * 0.95)) // batch_size
    total_train_steps = train_steps_per_epoch * epochs

    print(f"[TPU] Компиляция XLA графа под {total_train_steps} общих шагов обучения...")
    compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding = (
        make_shard_and_compile(config, total_train_steps, batch_size)
    )

    global_rng = jax.random.PRNGKey(42)
    with mesh:
        init_params_fn = jax.jit(
            lambda rng: model.init(rng, jnp.zeros((batch_size, 8192), dtype=jnp.int32))["params"],
            out_shardings=param_sharding,
        )
        params = init_params_fn(global_rng)
        opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)

    train_stream, val_factory, train_steps, val_steps = dataloader_multi_source(
        file_pairs, batch_size, data_sharding
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(epochs):
        with mesh:
            for step in range(train_steps):
                global_rng, step_rng = jax.random.split(global_rng)
                batch = next(train_stream)

                params, opt_state, train_loss, aux_info = compiled_train(params, opt_state, batch, step_rng)
                global_step += 1

                if step % 10 == 0:
                    print(
                        f"Epoch: {epoch} | Step: {step}/{train_steps} | "
                        f"Global Step: {global_step} | Train Loss: {jax.device_get(train_loss):.4f} "
                        f"(ce={jax.device_get(aux_info['ce_loss']):.4f} "
                        f"aux={jax.device_get(aux_info['aux_loss']):.4f} "
                        f"z={jax.device_get(aux_info['z_loss']):.5f})"
                    )
                    # Anti-routing-collapse monitoring: if any expert's utilization share
                    # spikes far above 1/num_experts (or drops near 0) across layers, the
                    # router is collapsing -- this shows up here well before val loss
                    # visibly suffers. A perfectly balanced router would print ~1/num_experts
                    # for every entry.
                    if aux_info["expert_utilization"] is not None:
                        util = jax.device_get(aux_info["expert_utilization"])  # (num_layers, num_experts)
                        util_std_per_layer = util.std(axis=-1)
                        worst_layer = int(util_std_per_layer.argmax())
                        print(
                            f"           expert utilization std (max over layers, layer {worst_layer}): "
                            f"{util_std_per_layer[worst_layer]:.4f} | ideal ~= 0, uniform = 1/{config.num_experts}"
                        )

            print(f"--- Эпоха {epoch} завершена. Запуск распределенной кросс-валидации ---")
            val_stream = val_factory()
            total_val_loss = 0.0
            for _ in range(val_steps):
                total_val_loss += jax.device_get(compiled_val(params, next(val_stream)))

            mean_val_loss = total_val_loss / val_steps
            print(f"===> Эпоха: {epoch} | ИТОГОВЫЙ СРЕДНИЙ VALIDATION LOSS: {mean_val_loss:.4f} <===")

            mngr.save(global_step, args=ocp.args.StandardSave(params))
            print(f"[ORBAX] Чекпоинт для шага {global_step} успешно зафиксирован.")

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_without_improvement = 0
                best_mngr.save(global_step, args=ocp.args.StandardSave(params))
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


if __name__ == "__main__":
    main_execution()
