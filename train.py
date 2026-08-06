import glob
import os
import re
import time
import signal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from model import FullHybridMoEModel, ModelConfig, set_model_mesh, get_model_mesh
from optimizer import compute_loss, make_hybrid_optimizer
from utils import path_to_str


def make_tpu_mesh():
    devices = jax.devices()
    n = len(devices)
    mesh_devices = mesh_utils.create_device_mesh((n,), devices)
    return Mesh(mesh_devices, axis_names=("tpu_nodes",))


def make_shard_and_compile(config: ModelConfig, total_steps: int, batch_size: int, seq_len: int = 8192):
    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]

    if batch_size % n_devices != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by n_devices={n_devices}.")

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

    # --- Single step: forward + backward + update ---
    def distributed_train_step(p, s, b, r):
        loss_fn = lambda param: compute_loss(
            param, model_apply_wrapped, b, config,
            rngs={"dropout": r},
            deterministic=False, return_aux=True,
            ce_chunk_size=2048
        )
        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, new_s = tx.update(grads, s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_s, loss, aux_info

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
        "expert_utilization": NamedSharding(mesh, P(None, None)),
    }

    compiled_train = jax.jit(
        distributed_train_step,
        donate_argnums=(0, 1),
        in_shardings=(
            param_sharding,
            opt_state_sharding,
            {"input_ids": data_sharding, "labels": data_sharding},
            NamedSharding(mesh, P(None)),
        ),
        out_shardings=(
            param_sharding,
            opt_state_sharding,
            NamedSharding(mesh, P()),
            aux_info_sharding,
        ),
    )

    compiled_val = jax.jit(
        distributed_val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=NamedSharding(mesh, P()),
    )

    return compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding


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


def main_execution():
    config = ModelConfig(
        d_model=768,
        d_state=128,
        d_conv=4,
        expand=2,
        n_heads=8,
        d_latent=512,
        d_ff=6144,
        num_experts=8,
        top_k=2,
        num_layers=21,
        layers_per_block=3,
        vocab_size=151936,
        dropout_rate=0.1,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.0001,
        moe_capacity_factor=1.0,
        tie_embeddings=True,
        label_smoothing=0.0,
        router_noise_std=0.3,
        use_flash_attention=True,
        deltanet_chunk_size=512,
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
    total_blocks = sum(n for _, _, n in manifest)
    print(f"Всего блоков: {total_blocks:,}")

    batch_size = 8
    seq_len = 4096
    epochs = 1
    early_stop_patience = 2
    eval_every_steps = 1000
    eval_batches = 40
    eval_patience = 4

    val_split = 0.05
    val_size = int(total_blocks * val_split)
    train_size = total_blocks - val_size

    train_steps_per_epoch = train_size // batch_size
    total_train_steps = train_steps_per_epoch * epochs

    print(f"[TPU] Компиляция XLA графа под {total_train_steps} шагов "
          f"({epochs} эпох(и) x {train_steps_per_epoch} шагов)...")

    compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding = (
        make_shard_and_compile(config, total_train_steps, batch_size, seq_len)
    )
    print(f"[TPU] Устройств в mesh: {mesh.shape['tpu_nodes']} (FSDP: params, state и батч шардированы).")

    train_stream, val_factory, _, val_steps = dataloader_multi_source(
        file_pairs, batch_size, data_sharding, seq_len=seq_len
    )

    global_rng = jax.random.PRNGKey(42)
    init_params_fn = jax.jit(
        lambda rng: model.init(rng, jnp.zeros((batch_size, seq_len), dtype=jnp.int32))["params"],
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

    _dummy_batch = {
        "input_ids": jax.device_put(jnp.zeros((batch_size, seq_len), dtype=jnp.int32), data_sharding),
        "labels": jax.device_put(jnp.zeros((batch_size, seq_len), dtype=jnp.int32), data_sharding),
    }
    _lowered = compiled_train.lower(params, opt_state, _dummy_batch, global_rng)
    _compiled_exec = _lowered.compile()
    _analysis = _compiled_exec.memory_analysis()
    print(f"[MEM ANALYSIS] HBM temp:      {_analysis.temp_size_in_bytes / 1e9:.2f} ГБ")
    print(f"[MEM ANALYSIS] HBM arguments: {_analysis.argument_size_in_bytes / 1e9:.2f} ГБ")
    print(f"[MEM ANALYSIS] HBM output:    {_analysis.output_size_in_bytes / 1e9:.2f} ГБ")
    print("[TPU] Компиляция готова -- переходим к реальному обучению.")

    checkpoint_dir = "/kaggle/working/orbax_checkpoints"
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer(), options)
    best_checkpoint_dir = "/kaggle/working/orbax_checkpoints_best"
    best_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    best_mngr = ocp.CheckpointManager(best_checkpoint_dir, ocp.StandardCheckpointer(), best_options)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    best_eval_loss = float("inf")
    eval_no_improve_count = 0
    stopped_early = False

    total_tokens_processed = 0
    epoch_start_time = time.perf_counter()

    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            global_rng, step_rng = jax.random.split(global_rng)

            _t0 = time.perf_counter()
            batch = next(train_stream)
            _t_data = time.perf_counter() - _t0

            total_tokens_processed += batch_size * seq_len

            _t1 = time.perf_counter()
            params, opt_state, train_loss, aux_info = compiled_train(
                params, opt_state, batch, step_rng
            )
            if step < 30:
                jax.block_until_ready(train_loss)
            _t_compute = time.perf_counter() - _t1

            if step < 30:
                print(f"[TIMING] step {step}: данные={_t_data*1000:.0f}мс  "
                      f"TPU={_t_compute*1000:.0f}мс  "
                      f"(доля данных: {_t_data/(_t_data+_t_compute)*100:.0f}%)")

            global_step += 1

            if step % 10 == 0:
                print(
                    f"Epoch: {epoch} | Step: {step}/{train_steps_per_epoch} | "
                    f"Global Step: {global_step} | Train Loss: {jax.device_get(train_loss):.4f} "
                    f"(ce={jax.device_get(aux_info['ce_loss']):.4f} "
                    f"aux={jax.device_get(aux_info['aux_loss']):.4f} "
                    f"z={jax.device_get(aux_info['z_loss']):.5f})"
                )
                if aux_info["expert_utilization"] is not None:
                    util = jax.device_get(aux_info["expert_utilization"])
                    util_std_per_layer = util.std(axis=-1)
                    worst_layer = int(util_std_per_layer.argmax())
                    print(
                        f"           expert utilization std (max over layers, layer {worst_layer}): "
                        f"{util_std_per_layer[worst_layer]:.4f} | ideal ~= 0, uniform = 1/{config.num_experts}"
                    )

            if (step + 1) % 10 == 0:
                print(f"[Успех] Тестовой запуск успешно проверен!")
                os.kill(os.getpid(), signal.SIGKILL)

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
                        mngr.save(global_step, args=ocp.args.StandardSave(params))
                        best_mngr.save(global_step, args=ocp.args.StandardSave(params))
                        print(f"[ORBAX] Финальный чекпоинт (шаг {global_step}) сохранён в оба каталога.")
                        stopped_early = True
                        break

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

    print("Обучение завершено.")


if __name__ == "__main__":
    main_execution()
