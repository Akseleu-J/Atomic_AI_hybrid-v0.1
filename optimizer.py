from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from model import ModelConfig
from utils import collect_by_leaf_name, path_to_str


# ==========================================
# Muon (Newton-Schulz orthogonalization)
# ==========================================
def muon_orthogonalize(w, g, lr, ns_steps: int = 5):
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


def make_hybrid_optimizer(total_steps: int):
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
def compute_loss(params, model_fn, batch, cfg: ModelConfig, rngs=None, deterministic=False, return_aux=False):
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    kwargs = {"deterministic": deterministic}
    if rngs is not None:
        kwargs["rngs"] = rngs

    outputs = model_fn(
        {"params": params}, input_ids, **kwargs, mutable=["losses"] if not deterministic else False
    )

    expert_util_stacked = None
    if not deterministic:
        logits, sowed_vars = outputs
        aux_losses = collect_by_leaf_name(sowed_vars["losses"], "aux_loss")
        z_losses = collect_by_leaf_name(sowed_vars["losses"], "z_loss")
        expert_utils = collect_by_leaf_name(sowed_vars["losses"], "expert_utilization")
        aux_loss = jnp.sum(jnp.stack(aux_losses)) if aux_losses else 0.0
        z_loss = jnp.sum(jnp.stack(z_losses)) if z_losses else 0.0
        if expert_utils:
            expert_util_stacked = jnp.stack(expert_utils)
    else:
        logits = outputs
        aux_loss, z_loss = 0.0, 0.0

    vocab_size = logits.shape[-1]
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # ЧТО БЫЛО: jax.nn.one_hot(labels, vocab_size) строил ПЛОТНЫЙ таргет формы
    # (batch, seq, vocab_size). При vocab_size=151936 (сознательно не уменьшенном
    # для теста) один такой тензор -- уже ~2.5 ГБ на fp32 при batch=2, seq=2048.
    # А их строилось МИНИМУМ 3-4 разом: one_hot_labels, сглаженный one_hot_labels,
    # и log_probs*one_hot_labels перед суммированием -- 10+ ГБ только на loss, пока
    # сама модель (d_model=384, num_layers=6) занимает памяти на порядок меньше.
    # Это и объясняет OOM "в притык" (16.76 ГБ нужно / 15.75 ГБ доступно) при
    # уже уменьшенном конфиге -- урезали всё, кроме vocab_size, а именно он тут
    # доминировал.
    #
    # ЧТО СТАЛО: та же самая label-smoothed cross-entropy, но без one-hot --
    # gather нужного log_prob через take_along_axis (O(batch*seq), не
    # O(batch*seq*vocab)) плюс sum(log_probs) для члена сглаживания (это
    # редукция уже вычисленного log_probs, а не новый большой массив). Разложение:
    #   loss = -sum_v target[v] * log_probs[v]
    #        = -log_probs[label]*(smooth_pos - smooth_neg) - smooth_neg*sum_v(log_probs[v])
    # Проверено численно против старой one-hot формулы (разные label_smoothing,
    # включая ignore_index=-100): совпадает с точностью float32 (~5e-7), градиенты
    # -- та же точность.
    labels_safe = jnp.clip(labels, 0, vocab_size - 1)  # -100 (ignore_index) -> валидный индекс для gather;
                                                          # реальное значение всё равно обнулится через `mask` ниже
    nll = -jnp.take_along_axis(log_probs, labels_safe[..., None], axis=-1).squeeze(-1)

    if cfg.label_smoothing > 0:
        smooth_positive = 1.0 - cfg.label_smoothing
        smooth_negative = cfg.label_smoothing / (vocab_size - 1)
        sum_log_probs = jnp.sum(log_probs, axis=-1)
        loss_matrix = nll * (smooth_positive - smooth_negative) - smooth_negative * sum_log_probs
    else:
        loss_matrix = nll

    mask = (labels != -100).astype(jnp.float32)
    ce_loss = jnp.sum(loss_matrix * mask) / (jnp.sum(mask) + 1e-9)

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
