from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from model import ModelConfig
from utils import collect_by_leaf_name, path_to_str


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


def make_hybrid_optimizer(total_steps: int, muon_diagnostic_disable: bool = False):
    # ФИКС: минимальный пол на длину warmup. Если total_steps окажется
    # небольшим (например, при короткой сессии/маленьком датасете), 10%
    # может дать слишком короткий warmup для стабилизации только что
    # инициализированных GDN-2/Mamba2 блоков.
    warmup_steps = max(500, int(total_steps * 0.10))
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

    lion_lr = lambda step: 3e-4 * lr_schedule(step)
    adamw_lr = lambda step: 1e-3 * lr_schedule(step)
    tx_lion = optax.lion(learning_rate=lion_lr, weight_decay=0.1)
    tx_adamw_decay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.01)
    tx_adamw_nodecay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.0)

    def _muon_step(base_lr: float, weight_decay: float = 0.01):
        def init_fn(params):
            return MuonState(count=jnp.zeros([], jnp.int32))

        def update_fn(updates, state, params=None):
            if params is None:
                return updates, state
            step_lr = base_lr * lr_schedule(state.count)
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

    # ФИКС: base_lr слегка снижен (0.01 -> 0.008) как дополнительная
    # предосторожность параллельно с добавленным weight decay -- снижает
    # скорость, с которой ортогонализованные обновления могут толкать норму
    # параметров вверх на старте обучения, пока decay ещё не успел накопить
    # эффект (decay пропорционален w, на малых w в начале почти не действует).
    tx_muon = _muon_step(base_lr=0.008, weight_decay=0.01)

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

    # ФИКС: общий global-norm clip слегка ужесточён (0.5 -> 0.35) как
    # дополнительный запас прочности -- дешёвая мера, не требующая
    # архитектурных изменений, снижает амплитуду отдельных "плохих" шагов
    # по всем группам параметров одновременно. clip_by_global_norm не имеет
    # состояния (EmptyState), поэтому это изменение НЕ влияет на совместимость
    # чекпоинтов.
    clip_tx = optax.clip_by_global_norm(0.35)
    multi_tx = optax.multi_transform(
        {"muon": tx_muon, "lion": tx_lion, "adamw_decay": tx_adamw_decay, "adamw_nodecay": tx_adamw_nodecay},
        label_fn,
    )
    tx = optax.chain(clip_tx, multi_tx)
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
    logits_chunk = make_grad_probe("ce_logits_chunk")(logits_chunk)

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
                  ce_chunk_size=256):
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
        aux_loss = jnp.sum(jnp.stack(aux_losses)) if aux_losses else 0.0
        z_loss = jnp.sum(jnp.stack(z_losses)) if z_losses else 0.0
        if expert_utils:
            expert_util_stacked = jnp.stack(expert_utils)
        if dropped_ratios:
            dropped_ratio_stacked = jnp.stack(dropped_ratios)
    else:
        final_hidden = outputs
        aux_loss, z_loss = 0.0, 0.0

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

    total_loss = ce_loss + (cfg.router_aux_loss_coef * aux_loss) + (cfg.router_z_loss_coef * z_loss)
    if return_aux:
        aux_info = {
            "ce_loss": ce_loss,
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "expert_utilization": expert_util_stacked,
            "moe_dropped_ratio": dropped_ratio_stacked,
            "router_temp": jnp.stack(router_temps) if router_temps else None,
        }
        return total_loss, aux_info
