from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from model import ModelConfig
from utils import collect_by_leaf_name, path_to_str


# ==========================================
# Muon (Newton-Schulz orthogonalization)
# ==========================================
def muon_orthogonalize(w, g, lr, ns_steps: int = 3):
    """Orthogonalize the gradient via Newton-Schulz iteration, then take a step."""
    eps = 1e-7
    if w.ndim == 3:
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)
        X = g / (norm + eps)
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * jnp.einsum("eij,ejk,ekl->eil", X, jnp.swapaxes(X, -1, -2), X)
    else:
        norm = jnp.linalg.norm(g)
        X = g / (norm + eps)
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * X @ X.T @ X

    return w - (X * lr)


class MuonState(NamedTuple):
    count: jnp.ndarray


def make_hybrid_optimizer(total_steps: int, muon_diagnostic_disable: bool = False):
    """muon_diagnostic_disable=True routes every normally-muon leaf to
    adamw_nodecay instead. This is a ONE-CALL diagnostic, not a real fix: it
    exists to answer a single question via memory_analysis() (compile-only, no
    execution needed) -- is the 17.57 GB `temp` figure caused by Newton-Schulz
    needing the FULL (all-gathered) matrix under FSDP for ~250+ muon-labeled
    leaves at once, or is it something else? If HBM temp collapses with this
    flag on, the hypothesis is confirmed and we go fix Muon's update_fn
    (sequence the all-gathers instead of leaving them to jax.tree_map's
    unconstrained scheduling) rather than guessing further blind. If it does
    NOT collapse, look elsewhere instead of chasing Muon."""
    warmup_steps = max(1, int(total_steps * 0.05))
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

    def _muon_step(base_lr: float):
        def init_fn(params):
            return MuonState(count=jnp.zeros([], jnp.int32))

        def update_fn(updates, state, params=None):
            if params is None:
                return updates, state
            step_lr = base_lr * lr_schedule(state.count)
            new_updates = jax.tree_util.tree_map(
                lambda p, g: (muon_orthogonalize(p, g, step_lr) - p), params, updates
            )
            return new_updates, MuonState(count=state.count + 1)

        return optax.GradientTransformation(init_fn, update_fn)

    tx_muon = _muon_step(base_lr=0.02)

    def _label_leaf(path, param):
        path_str = path_to_str(path)
        if "embed" in path_str or "lm_head" in path_str:
            return "adamw_decay"
        # ФИКС: слои нормализации в model.py называются norm_1/norm_2/out_norm/
        # final_norm -- подстрока "rmsnorm" никогда не встречается в пути, поэтому
        # все scale-векторы RMSNorm (1D) проскальзывали мимо этой ветки и падали в
        # `else: return "lion"` ниже -- обучались Lion'ом с weight decay вместо
        # AdamW без decay, как задумывалось. Проверено на коллизии: "norm" не
        # цепляет router/q_route/k_route/decay_proj/out_proj и т.д.
        if "norm" in path_str or "bias" in path_str:
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

    clip_tx = optax.clip_by_global_norm(1.0)
    multi_tx = optax.multi_transform(
        {"muon": tx_muon, "lion": tx_lion, "adamw_decay": tx_adamw_decay, "adamw_nodecay": tx_adamw_nodecay},
        label_fn,
    )
    return optax.chain(clip_tx, multi_tx)


# ==========================================
# Loss
# ==========================================
def _chunked_ce_step(carry, chunk, w, smooth_positive, smooth_negative, vocab_size):
    """One token-chunk of label-smoothed CE. Wrapped in jax.checkpoint by the
    caller so its forward tensors (chunk_size, vocab) are recomputed during
    backward instead of being kept alive for the whole scan -- see
    chunked_cross_entropy below for why this exists."""
    sum_loss, sum_mask = carry
    hidden_chunk, label_chunk = chunk  # (chunk_size, d_model), (chunk_size,)

    logits_chunk = hidden_chunk @ w  # (chunk_size, vocab) -- the only large tensor,
                                       # and only for ONE chunk at a time
    log_probs = jax.nn.log_softmax(logits_chunk, axis=-1)

    labels_safe = jnp.clip(label_chunk, 0, vocab_size - 1)  # -100 (ignore_index) -> valid
                                                               # gather index; zeroed by mask below
    nll = -jnp.take_along_axis(log_probs, labels_safe[:, None], axis=-1).squeeze(-1)

    if smooth_negative is not None:
        sum_log_probs = jnp.sum(log_probs, axis=-1)
        loss_vec = nll * (smooth_positive - smooth_negative) - smooth_negative * sum_log_probs
    else:
        loss_vec = nll

    mask = (label_chunk != -100).astype(jnp.float32)
    new_carry = (sum_loss + jnp.sum(loss_vec * mask), sum_mask + jnp.sum(mask))
    return new_carry, None  # (carry, y) -- the shape jax.lax.scan's step fn must return


def chunked_cross_entropy(final_hidden, labels, w, label_smoothing, chunk_size=256):
    """Same label-smoothed cross-entropy as before, but the (batch, seq, vocab)
    logits/log_probs tensor is NEVER materialized in full.

    ЧТО БЫЛО (две итерации назад): jax.nn.one_hot(labels, vocab_size) строил
    плотный таргет (batch, seq, vocab) -- ~2.5 ГБ на fp32 при batch=2, seq=2048,
    vocab=151936, и таких тензоров строилось 3-4 разом.

    ЧТО СТАЛО ПОТОМ (прошлая правка): убрали one-hot через take_along_axis --
    но это не сдвинуло OOM (16.76 -> 16.77 ГБ), потому что сам `logits` и
    `log_probs` -- тензоры ТОЙ ЖЕ формы (batch, seq, vocab) -- строились самой
    моделью (`embed_layer.attend(final)` в model.py) ДО того, как loss вообще
    начинал считаться, и это происходит вне nn.remat-скоупов модели (remat там
    оборачивает только transformer-блоки, не финальную проекцию). На стыке
    forward/backward нужно ОДНОВРЕМЕННО держать forward-тензор и backward-тензор
    того же размера -- отсюда ~8-10 ГБ только на этот узел, и remat здесь не
    помогает (нечего "растягивать" -- промежуток между вычислением и
    использованием и так короткий).

    ЧТО СТАЛО ТЕПЕРЬ: model.py возвращает `final` (b, l, d_model) вместо
    `logits` (return_hidden=True) -- маленький тензор, не зависящий от vocab_size.
    Проекция в vocab-пространство и вся loss-арифметика делается здесь, чанками
    по `chunk_size` токенов через jax.lax.scan, и тело каждого чанка обёрнуто в
    jax.checkpoint: во время backward JAX пересчитывает logits/log_probs для
    ОДНОГО чанка за раз и сразу их выбрасывает, вместо того чтобы хранить все
    чанки разом. Пиковая память по этому узлу падает с O(batch*seq*vocab) до
    O(chunk_size*vocab) -- при chunk_size=256 это ~148 МБ вместо ~2.5 ГБ.
    Градиент по `w` (embedding-таблица/lm_head) накапливается через обычный
    autodiff по scan (JAX суммирует вклад каждой итерации в grad замкнутого
    аргумента корректно, это стандартное поведение, не что-то специальное).
    Численно эквивалентно прежней формуле -- то же разложение, тот же
    take_along_axis + sum(log_probs), просто по кускам вместо разом.
    """
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

    # jax.checkpoint on the per-chunk step itself: during backward, jax.lax.scan
    # recomputes logits_chunk/log_probs for one chunk at a time from the (small,
    # cheap-to-keep) hidden_chunk, uses them, then discards them -- instead of
    # every chunk's (chunk_size, vocab) tensor being kept alive simultaneously
    # for the whole scan.
    scan_step = jax.checkpoint(
        lambda carry, chunk: _chunked_ce_step(carry, chunk, w, smooth_positive, smooth_negative, vocab_size)
    )

    (sum_loss, sum_mask), _ = jax.lax.scan(scan_step, (0.0, 0.0), (hidden_chunks, label_chunks))
    return sum_loss / (sum_mask + 1e-9)


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
    if not deterministic:
        final_hidden, sowed_vars = outputs
        aux_losses = collect_by_leaf_name(sowed_vars["losses"], "aux_loss")
        z_losses = collect_by_leaf_name(sowed_vars["losses"], "z_loss")
        expert_utils = collect_by_leaf_name(sowed_vars["losses"], "expert_utilization")
        aux_loss = jnp.sum(jnp.stack(aux_losses)) if aux_losses else 0.0
        z_loss = jnp.sum(jnp.stack(z_losses)) if z_losses else 0.0
        if expert_utils:
            expert_util_stacked = jnp.stack(expert_utils)
    else:
        final_hidden = outputs
        aux_loss, z_loss = 0.0, 0.0

    # w must be (d_model, vocab) for `hidden_chunk @ w` in chunked_cross_entropy.
    # Tied embeddings: nn.Embed's kernel is stored (vocab, d_model) -- transpose.
    # Untied: nn.Dense("lm_head") kernel is already (d_model, vocab).
    if cfg.tie_embeddings:
        w = params["embed"]["embedding"].T
    else:
        w = params["lm_head"]["kernel"]

    ce_loss = chunked_cross_entropy(final_hidden, labels, w, cfg.label_smoothing, chunk_size=ce_chunk_size)

    total_loss = ce_loss + (cfg.router_aux_loss_coef * aux_loss) + (cfg.router_z_loss_coef * z_loss)
    if return_aux:
        aux_info = {
            "ce_loss": ce_loss,
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "expert_utilization": expert_util_stacked,
        }
        return total_loss, aux_info
    return total_loss
