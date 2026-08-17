"""
train.py -- главный orchestration-цикл обучения.
...
"""
import os
import time
import json
import signal
import shutil
import sys
from collections import deque
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from checkpointing import (
    make_manager, save_slot, save_slot_async, _finalize_pending_save,
    upload_slot, download_slot, _HAS_HF,
    CHECKPOINT_EVERY_SECONDS, HF_LATEST_KEEP_N,
)
from train_setup import (
    make_shard_and_compile, dataloader_multi_source, build_manifest,
    DATASET_FRACTION, DATASET_FRACTION_SEED,
    NONFINITE_CONSECUTIVE_LIMIT, NONFINITE_WINDOW_SIZE, NONFINITE_WINDOW_RATIO,
    SESSION_TIME_BUDGET_SECONDS,
)
import wandb_logging

from model import ModelConfig, get_model_mesh


# ==========================================================================
# ДИАГНОСТИКА (не изменяет params/opt_state): сравнивает статистику
# A_log/dt_proj.bias между всеми mamba2-слоями после resume -- вместо
# слепой "хирургии" переинициализации, которая была бы преждевременна:
# все mamba2-слои получили ОДИНАКОВОЕ число градиентных шагов в этом
# прогоне (num_layers=24 задан с самого начала, не расширялся посреди
# обучения), поэтому гипотеза "layer=22 моложе остальных" отпадает.
# Реальные кандидаты -- структурная позиция (близость к выходу, глубина
# history_blocks для block_7), а не недостаток шагов -- и их стоит сначала
# УВИДЕТЬ в цифрах, а не лечить вслепую.
# ==========================================================================
def diagnose_mamba2_decay_params(params, mamba2_layer_indices, layers_per_block):
    """Печатает mean/std/min/max для A_log и dt_proj.bias КАЖДОГО указанного
    mamba2-слоя, плюс производные величины (dt на первом шаге при нулевом
    входе dt_proj: softplus(bias), и итоговый decay exp(-softplus(bias)*A)
    в характерной точке A=-exp(A_log).mean()) -- чтобы увидеть, действительно
    ли layer=22 численно отличается от layer=4/layer=13, а не гадать."""
    print("\n" + "=" * 70)
    print("[MAMBA2-DIAG] Сравнение decay-параметров по mamba2-слоям")
    print("=" * 70)

    found = {}

    def _collect(path, leaf):
        if len(path) < 6:
            return
        keys = [str(getattr(p, "key", p)) for p in path]
        for layer_idx in mamba2_layer_indices:
            block_idx = layer_idx // layers_per_block
            prefix_ok = (
                keys[0] == f"block_{block_idx}" and keys[1] == f"layer_{layer_idx}"
                and keys[2] == "sublayer" and keys[3] == "mamba2"
            )
            if not prefix_ok:
                continue
            if keys[4] == "A_log":
                found.setdefault(layer_idx, {})["A_log"] = leaf
            elif len(keys) >= 6 and keys[4] == "dt_proj" and keys[5] == "bias":
                found.setdefault(layer_idx, {})["dt_bias"] = leaf

    jax.tree_util.tree_map_with_path(lambda p, l: (_collect(p, l), l)[1], params)

    for layer_idx in mamba2_layer_indices:
        entry = found.get(layer_idx)
        if entry is None or "A_log" not in entry or "dt_bias" not in entry:
            print(f"[MAMBA2-DIAG] ⚠️ layer_{layer_idx}: не удалось найти A_log/dt_proj.bias "
                  f"по ожидаемому пути -- пропускаю (проверьте пути вручную).")
            continue

        A_log = jax.device_get(entry["A_log"]).astype("float32")
        dt_bias = jax.device_get(entry["dt_bias"]).astype("float32")

        A_log_clipped = jnp.clip(A_log, -20.0, 20.0)
        A = -jnp.exp(A_log_clipped)
        dt_at_zero_input = jax.nn.softplus(dt_bias)  # dt если dt_proj-выход самой сети ~0
        dt_clipped = jnp.clip(dt_at_zero_input, 1e-2, 1.0)  # тот же forward-клип, что в Mamba2J
        # decay за один шаг в характерной точке -- exp(dt*A), усреднённый по каналам
        decay_per_step = jnp.exp(jnp.clip(dt_clipped * A, -20.0, 0.0))

        print(f"\n[MAMBA2-DIAG] layer_{layer_idx} (block_{layer_idx // layers_per_block}):")
        print(f"    A_log:        mean={float(jnp.mean(A_log)):+.4f}  std={float(jnp.std(A_log)):.4f}  "
              f"min={float(jnp.min(A_log)):+.4f}  max={float(jnp.max(A_log)):+.4f}")
        print(f"    dt_proj.bias: mean={float(jnp.mean(dt_bias)):+.4f}  std={float(jnp.std(dt_bias)):.4f}  "
              f"min={float(jnp.min(dt_bias)):+.4f}  max={float(jnp.max(dt_bias)):+.4f}")
        print(f"    dt(at zero input, post-clip): mean={float(jnp.mean(dt_clipped)):.5f}  "
              f"std={float(jnp.std(dt_clipped)):.5f}")
        print(f"    decay_per_step (exp(dt*A)):   mean={float(jnp.mean(decay_per_step)):.5f}  "
              f"std={float(jnp.std(decay_per_step)):.5f}  "
              f"(1.0=не забывает вообще, 0.0=забывает мгновенно)")

    print("\n[MAMBA2-DIAG] Как читать: если у layer_22 decay_per_step систематически "
          "ближе к 0 или к 1 относительно layer_4/layer_13 (не просто другой std, а "
          "смещённое mean), это говорит о специфичном для этого слоя режиме -- тогда "
          "стоит смотреть на A_log/dt_bias std как на признак недостаточной "
          "межканальной дифференциации decay именно здесь. Если все три слоя похожи -- "
          "проблема НЕ в decay-параметрах, и стоит смотреть выше по стеку (DAR "
          "history_blocks на block_7, близость к выходу).")
    print("=" * 70 + "\n")


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

    FORCE_FRESH_START = False  # ФИКС: продолжаем с чекпоинта шага 4000, не с нуля
    RESUME_FROM_SLOT = "best_val"

    if FORCE_FRESH_START:
        resume_step = None
        print("[RESUME] 🆕 FORCE_FRESH_START=True -- пропускаю поиск чекпоинтов, начинаю с нуля.")
    elif RESUME_FROM_SLOT == "latest":
        resume_step = mngr_latest.latest_step()
        if resume_step is not None:
            print(f"[LOCAL] 📦 Found checkpoint (latest): step {resume_step}")
        if resume_step is None and _HAS_HF:
            resume_step = download_slot(latest_dir, "latest")
            if resume_step is not None:
                mngr_latest = make_manager(latest_dir, max_to_keep=HF_LATEST_KEEP_N)
    else:
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
        router_z_loss_coef=0.0001,
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
            "gdn2", "gdn2", "gdn2",
            "gdn2", "gdn2", "mla",
            "gdn2", "mamba2", "gdn2",
            "gdn2", "gdn2", "mla",
            "gdn2", "gdn2", "mla",
            "gdn2", "mamba2", "gdn2",
        ),
    )
    file_pairs = [
        (
            "/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_labels.npy",
        ),
        (
            "/kaggle/input/datasets/umirbayulgaisha/math-data/math_input_ids.npy",
            "/kaggle/input/datasets/umirbayulgaisha/math-data/math_labels.npy",
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
                "n_devices": mesh.shape["tpu_nodes"]},
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

            param_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(params))))
            has_nan = any(bool(jnp.any(jnp.isnan(x))) for x in jax.tree_util.tree_leaves(params))
            print(f"[RESUME DEBUG] param_norm={param_norm:.4f}, has_nan={has_nan}")
            if has_nan:
                raise ValueError("Восстановленные params содержат NaN -- чекпоинт повреждён.")

            print(f"[RESUME] ✅ Restored: step={global_step}, best_val={best_val_loss:.4f}, best_train={best_train_loss:.4f}")

            # ==================================================================
            # ФИКС: диагностика (read-only) decay-параметров ВСЕХ трёх mamba2-
            # слоёв сразу после restore -- см. handoff про RESID-DIAG на
            # layer=22 и опровержение гипотезы "layer=22 моложе" (все три
            # получили одинаковое число шагов в этом прогоне). НЕ изменяет
            # params -- только печатает сравнение, чтобы решить, действительно
            # ли численно layer=22 отличается, прежде чем что-либо трогать.
            # ==================================================================
            mamba2_layer_indices = [
                idx for idx, t in enumerate(config.layer_types) if t == "mamba2"
            ]
            diagnose_mamba2_decay_params(params, mamba2_layer_indices, config.layers_per_block)

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

    nonfinite_consecutive_count = 0
    nonfinite_window = deque(maxlen=NONFINITE_WINDOW_SIZE)
    _accum_window = deque(maxlen=accum_steps)

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

                step_was_finite = bool(jax.device_get(was_finite))
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
                    with open(os.path.join(snap_dir, "SNAPSHOT_META.json"), "w") as f:
                        json.dump({"n_micro": len(_accum_window), "global_step": int(global_step + 1)}, f)
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
                              f"см. [WARNING]/[FWD-DIAG]/[BWD-DIAG]/[PARAM-DIAG] логи выше для локализации источника.")
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
                        f"best_train={best_train_loss:.4f}"
                    )
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
                        util = jax.device_get(aux_info["expert_utilization"])
                        util_std_per_layer = util.std(axis=-1)
                        worst_layer = int(util_std_per_layer.argmax())
                        print(
                            f"           expert utilization std (max over layers, layer {worst_layer}): "
                            f"{util_std_per_layer[worst_layer]:.4f} | ideal ~= 0, uniform = 1/{config.num_experts - 1}"
                        )
                        wandb_step_metrics["moe/expert_util_std_max"] = float(util_std_per_layer[worst_layer])
                        wandb_step_metrics["moe/expert_util_std_worst_layer"] = worst_layer
                    if aux_info.get("moe_dropped_ratio") is not None:
                        dropped = jax.device_get(aux_info["moe_dropped_ratio"])
                        worst_drop_layer = int(dropped.argmax())
                        print(
                            f"           moe dropped_ratio (max over layers, layer {worst_drop_layer}): "
                            f"{dropped[worst_drop_layer]:.4f}  (ideal ~= 0 after warmup)"
                        )
                        wandb_step_metrics["moe/dropped_ratio_max"] = float(dropped[worst_drop_layer])
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
    wandb_logging.finish()


if __name__ == "__main__":
    main_execution()
