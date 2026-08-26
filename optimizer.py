"""
optimizer.py -- гибридный оптимизатор (Muon/Lion/AdamW) + CE-loss.

ФИКС (этот пасс -- диагностика сведена ТОЛЬКО к оптимизатору): compute_loss
больше НЕ собирает kernel/activation-level sow'нутые метрики
(layer_delta_*, layer_resid_*, mamba2_input_*, mamba2_ssm_out_*, gdn2_*,
gdn2_kernelstage_*, mla_*, final_hidden_*) -- эти self.sow(...) вызовы
удалены из model.py (см. его докстринг), поэтому collect_by_leaf_name для
них теперь всегда вернёт пустой список. aux_info больше не тащит
diag_stacked. Оптимизаторская диагностика (per-group nonfinite flags,
per-layer grad/weight norm/maxabs/nonfinite, muon orth_resid+worst_leaf_idx,
per-group grad norms) остаётся ПОЛНОСТЬЮ -- она живёт в train_setup.py и
здесь не трогалась.

ФИКС (локализация orth_resid, предыдущий пасс, без изменений в этом
пассе): MuonState несёт worst_leaf_idx -- индекс худшего по orth_resid
листа ВНУТРИ muon-подветки params, на КАЖДОМ шаге. extract_muon_diagnostics
возвращает (max_resid, worst_leaf_idx). build_muon_leaf_paths -- чистая
Python-функция (вызывается один раз в train_setup.py), строит список путей
в ТОМ ЖЕ порядке flatten(), в котором multi_transform обходит muon-подветку.

ФИКС (структурные исключения из Muon, без изменений в этом пассе):
- conv_w -- rank<=4, структурно вырождена для спектральной ортогонализации
  -- переведена на Lion.
- routed/shared MoE-экспертов w1/w2 -- экстремально прямоугольные,
  давали orth_resid=58-59 стабильно -- переведены на Lion.
"""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax

from model import ModelConfig
from utils import collect_by_leaf_name, path_to_str

ROUTER_COLLINEARITY_COEF = 0.08

RESUME_BACKOFF_STEPS = 5000
RESUME_LR_SCALE = 0.7

DEFAULT_WARMUP_FREEZE_STEP = 5000

_PER_TOKEN_CE_CLIP = 15.0


def set_per_token_ce_clip(value: float):
    global _PER_TOKEN_CE_CLIP
    _PER_TOKEN_CE_CLIP = value
    print(f"[OPTIMIZER] _PER_TOKEN_CE_CLIP переопределён на {value}")


# ==========================================================================
# Grad utilities
# ==========================================================================

def make_grad_sanitizer(tag: str, clip_val: float = 1e3):
    @jax.custom_vjp
    def _sanitizer(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        finite = jnp.all(jnp.isfinite(g))
        jax.lax.cond(
            jnp.logical_not(finite),
            lambda: jax.debug.print(
                "[BWD-FIX] 🩹 non-finite градиент в узле {t} -- санитизирован", t=tag),
            lambda: None,
        )
        g_safe = jnp.nan_to_num(
            jnp.clip(g, -clip_val, clip_val),
            nan=0.0, posinf=clip_val, neginf=-clip_val,
        )
        return (g_safe,)

    _sanitizer.defvjp(_fwd, _bwd)
    return _sanitizer


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
            lambda: jax.debug.print(
                "[BWD-DIAG] ⚠️ non-finite ВХОДЯЩИЙ градиент в узле: " + tag),
            lambda: None,
        )
        return (g,)

    _probe.defvjp(_fwd, _bwd)
    return _probe


# ==========================================================================
# Frozen optimizer (для expert_bias)
# ==========================================================================

def _frozen_step():
    def init_fn(params):
        return optax.EmptyState()
    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(jnp.zeros_like, updates), state
    return optax.GradientTransformation(init_fn, update_fn)

tx_frozen = _frozen_step()


# ==========================================================================
# Muon ортогонализация
# ==========================================================================

def muon_orthogonalize_legacy(w, g, lr, ns_steps: int = 3):
    """СТАРАЯ (сломанная) версия -- fallback/cross-check only."""
    eps = 1e-4
    if w.ndim == 3:
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)
        norm = jnp.where(norm < eps, jnp.ones_like(norm), norm)
        X = g / norm
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * jnp.einsum(
                "eij,ejk,ekl->eil", X, jnp.swapaxes(X, -1, -2), X)
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        norm = jnp.linalg.norm(g)
        norm = jnp.where(norm < eps, 1.0, norm)
        X = g / norm
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * X @ X.T @ X
            X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return w - (X * lr)


def _muon_ns_iterate(X, ns_steps: int = 7):
    a, b, c = 3.4445, -4.7750, 2.0315

    was_tall = X.shape[-2] > X.shape[-1]
    if was_tall:
        X = jnp.swapaxes(X, -2, -1)

    norm = jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)
    X = X / (norm * 1.01 + 1e-7)

    for _ in range(ns_steps):
        A = X @ jnp.swapaxes(X, -2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if was_tall:
        X = jnp.swapaxes(X, -2, -1)

    return X


def muon_orthogonalize(w, g, lr, ns_steps: int = 7):
    eps = 1e-7

    if w.ndim == 3:
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)
        safe_norm = jnp.where(norm < eps, jnp.ones_like(norm), norm)
        X = g / safe_norm
        X = _muon_ns_iterate(X, ns_steps)
        X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        effective_lr = jnp.where(norm < eps, jnp.zeros_like(norm), lr)
    else:
        norm = jnp.linalg.norm(g)
        safe_norm = jnp.where(norm < eps, 1.0, norm)
        X = g / safe_norm
        X = _muon_ns_iterate(X, ns_steps)
        X = jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        effective_lr = jnp.where(norm < eps, 0.0, lr)

    return w - (X * effective_lr)


def _muon_orth_diag(g, ns_steps: int = 7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = jnp.asarray(g)

    was_tall = X.shape[-2] > X.shape[-1]
    if was_tall:
        X = jnp.swapaxes(X, -2, -1)

    norm = jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)
    eps = 1e-7
    X = X / (norm * 1.01 + eps)
    for _ in range(ns_steps):
        A = X @ jnp.swapaxes(X, -2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if was_tall:
        X = jnp.swapaxes(X, -2, -1)

    if X.shape[-2] >= X.shape[-1]:
        prod = jnp.swapaxes(X, -2, -1) @ X
        n = X.shape[-1]
    else:
        prod = X @ jnp.swapaxes(X, -2, -1)
        n = X.shape[-2]
    I = jnp.eye(n, dtype=prod.dtype)
    return jnp.linalg.norm(prod - I)


# ==========================================================================
# State NamedTuples
# ==========================================================================

class MuonState(NamedTuple):
    count: jnp.ndarray
    orth_resid: jnp.ndarray
    worst_leaf_idx: jnp.ndarray
    worst_leaf_grad_norm: jnp.ndarray      # НОВОЕ
    worst_leaf_grad_maxabs: jnp.ndarray    # НОВОЕ
    mean_orth_resid: jnp.ndarray    

class ZClipState(NamedTuple):
    ema_mean: jnp.ndarray
    ema_var: jnp.ndarray
    warm_count: jnp.ndarray
    slow_ema_mean: jnp.ndarray
    slow_warm_count: jnp.ndarray


class BurstDamperState(NamedTuple):
    ema_norm: jnp.ndarray


# ==========================================================================
# Burst damper (заменяет zclip_skip)
# ==========================================================================

def burst_damper(decay: float = 0.95, threshold_ratio: float = 1.8,
                  min_scale: float = 0.05):
    def init_fn(params):
        return BurstDamperState(ema_norm=jnp.array(1.0, dtype=jnp.float32))

    def update_fn(updates, state, params=None):
        norm = optax.global_norm(updates)
        ratio = norm / (state.ema_norm + 1e-6)
        scale = jnp.where(
            ratio > threshold_ratio,
            jnp.maximum(min_scale, threshold_ratio / ratio),
            1.0,
        )
        new_updates = jax.tree_util.tree_map(lambda g: g * scale, updates)
        damped_norm = norm * scale
        new_ema = state.ema_norm * decay + damped_norm * (1.0 - decay)
        return new_updates, BurstDamperState(ema_norm=new_ema)

    return optax.GradientTransformation(init_fn, update_fn)


# ==========================================================================
# extract_zclip_diagnostics -- совместимость с train_setup.py
# ==========================================================================

def extract_zclip_diagnostics(opt_state):
    return {
        "ema_mean":       jnp.zeros((), dtype=jnp.float32),
        "ema_var":        jnp.ones((), dtype=jnp.float32),
        "warm_count":     jnp.zeros((), dtype=jnp.int32),
        "slow_ema_mean":  jnp.zeros((), dtype=jnp.float32),
        "slow_warm_count": jnp.zeros((), dtype=jnp.int32),
    }


def extract_muon_diagnostics(opt_state):
    resid_values = collect_by_leaf_name(opt_state, "orth_resid")
    idx_values = collect_by_leaf_name(opt_state, "worst_leaf_idx")
    norm_values = collect_by_leaf_name(opt_state, "worst_leaf_grad_norm")
    maxabs_values = collect_by_leaf_name(opt_state, "worst_leaf_grad_maxabs")
    mean_values = collect_by_leaf_name(opt_state, "mean_orth_resid")   # НОВОЕ
    if not resid_values:
        z = jnp.array(0.0, dtype=jnp.float32)
        return z, jnp.array(-1, dtype=jnp.int32), z, z
    max_resid = jnp.max(jnp.stack([v.astype(jnp.float32) for v in resid_values]))
    worst_idx = idx_values[0].astype(jnp.int32) if idx_values else jnp.array(-1, dtype=jnp.int32)
    worst_norm = norm_values[0].astype(jnp.float32) if norm_values else jnp.array(0.0, dtype=jnp.float32)
    worst_maxabs = maxabs_values[0].astype(jnp.float32) if maxabs_values else jnp.array(0.0, dtype=jnp.float32)
    mean_resid = mean_values[0].astype(jnp.float32) if mean_values else jnp.array(0.0, dtype=jnp.float32)   # НОВОЕ
    return max_resid, worst_idx, worst_norm, worst_maxabs, mean_resid


def build_muon_leaf_paths(params, label_fn):
    """ЧИСТАЯ Python-функция (НЕ jit, вызывается ОДИН РАЗ при старте
    обучения, в train_setup.py's make_shard_and_compile) -- строит список
    путей параметров, помеченных label_fn как "muon", В ТОМ ЖЕ ПОРЯДКЕ, в
    котором jax.tree_util.tree_flatten обходит ИСХОДНОЕ дерево params."""
    labels = label_fn(params)
    flat_params_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
    flat_labels, _ = jax.tree_util.tree_flatten(labels)

    muon_paths = [
        path_to_str(path) for (path, _), lbl in zip(flat_params_with_path, flat_labels)
        if lbl == "muon"
    ]
    return muon_paths


def _label_leaf(path, param):
    path_str = path_to_str(path)
    if "embed" in path_str or "lm_head" in path_str:
        return "adamw_decay"
    if "norm" in path_str or "bias" in path_str:
        return "adamw_nodecay"
    if "expert_bias" in path_str:
        return "frozen"
    if "router" in path_str:
        return "adamw_nodecay"
    if "conv_w" in path_str:
        return "lion"
    if param.ndim >= 2:
        if "mamba" in path_str:
            return "lion"
        if ("w1" in path_str or "w2" in path_str) and (
            "expert" in path_str or "moe" in path_str or "routed" in path_str or "shared" in path_str
        ):
            return "lion"
        return "muon"
    return "lion"


def _make_label_fn(muon_diagnostic_disable: bool):
    def label_fn(params):
        def _leaf(path, param):
            lbl = _label_leaf(path, param)
            if lbl == "muon" and muon_diagnostic_disable:
                return "adamw_nodecay"
            return lbl
        return jax.tree_util.tree_map_with_path(_leaf, params)
    return label_fn


# ==========================================================================
# Основной оптимизатор
# ==========================================================================

def make_hybrid_optimizer(total_steps: int, muon_diagnostic_disable: bool = False,
                           warmup_freeze_step: Optional[int] = DEFAULT_WARMUP_FREEZE_STEP,
                           muon_ns_steps: int = 7):
    warmup_steps = max(500, int(total_steps * 0.15))
    cosine = optax.cosine_decay_schedule(
        init_value=1.0,
        decay_steps=max(1, total_steps - warmup_steps),
        alpha=0.1,
    )
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=1.0, transition_steps=warmup_steps),
            cosine,
        ],
        boundaries=[warmup_steps],
    )

    def resume_backoff(step):
        RAMP_STEPS = 1000.0
        ramp_start = RESUME_BACKOFF_STEPS - RAMP_STEPS
        frac = jnp.clip((step - ramp_start) / RAMP_STEPS, 0.0, 1.0)
        return RESUME_LR_SCALE + (1.0 - RESUME_LR_SCALE) * frac

    def _effective_step(step):
        if warmup_freeze_step is None:
            return step
        return jnp.minimum(step, warmup_freeze_step)

    lion_lr  = lambda step: 2e-4 * lr_schedule(_effective_step(step)) * resume_backoff(step)
    adamw_lr = lambda step: 6e-4 * lr_schedule(_effective_step(step)) * resume_backoff(step)

    tx_lion        = optax.lion(learning_rate=lion_lr, weight_decay=0.1)
    tx_adamw_decay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.01)
    tx_adamw_nodecay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.0)

    def _muon_step(base_lr: float, weight_decay: float = 0.02, ns_steps: int = muon_ns_steps):
        def init_fn(params):
            return MuonState(
                count=jnp.zeros([], jnp.int32),
                orth_resid=jnp.zeros([], jnp.float32),
                worst_leaf_idx=jnp.zeros([], jnp.int32),
                worst_leaf_grad_norm=jnp.zeros([], jnp.float32),      # НОВОЕ
                worst_leaf_grad_maxabs=jnp.zeros([], jnp.float32),    # НОВОЕ
                mean_orth_resid=jnp.zeros([], jnp.float32),   # НОВОЕ
            )

        def update_fn(updates, state, params=None):
          if params is None:
              return updates, state
          step_lr = base_lr * lr_schedule(_effective_step(state.count))
          new_updates = jax.tree_util.tree_map(
              lambda p, g: (muon_orthogonalize(p, g, step_lr, ns_steps=ns_steps) - p)
                           - step_lr * weight_decay * p,
              params, updates,
          )

          leaves_g, _ = jax.tree_util.tree_flatten(updates)
          per_leaf_resid = jnp.stack([
              _muon_orth_diag(g, ns_steps=ns_steps) for g in leaves_g
          ])

          leaf_norms_pre = jnp.stack([jnp.linalg.norm(g.astype(jnp.float32)) for g in leaves_g])
          _EPS_GRAD = 1e-8
          per_leaf_resid_masked = jnp.where(leaf_norms_pre < _EPS_GRAD, -jnp.inf, per_leaf_resid)

          worst_idx = jnp.argmax(per_leaf_resid_masked)
          step_orth_resid = per_leaf_resid[worst_idx]

          _valid_mask = leaf_norms_pre >= _EPS_GRAD
          step_mean_orth_resid = jnp.sum(jnp.where(_valid_mask, per_leaf_resid, 0.0)) / jnp.maximum(jnp.sum(_valid_mask), 1)

          leaf_norms = leaf_norms_pre
          leaf_maxabs = jnp.stack([jnp.max(jnp.abs(g.astype(jnp.float32))) for g in leaves_g])
          worst_leaf_grad_norm = leaf_norms[worst_idx]
          worst_leaf_grad_maxabs = leaf_maxabs[worst_idx]
          _valid_mask = leaf_norms_pre >= _EPS_GRAD
          step_mean_orth_resid = jnp.sum(jnp.where(_valid_mask, per_leaf_resid, 0.0)) / jnp.maximum(jnp.sum(_valid_mask), 1)
          # НОВОЕ: норма и maxabs градиента ИМЕННО худшего листа -- дёшево
          # (один-два reduce поверх уже посчитанного stacked tensor нельзя,
          # т.к. листья разной формы, но динамический индекс через lax.switch
          # избыточен -- проще посчитать по ВСЕМ листьям через tree_map и
          # выбрать через jnp.take на flat-массиве скаляров).
          leaf_norms = jnp.stack([jnp.linalg.norm(g.astype(jnp.float32)) for g in leaves_g])
          leaf_maxabs = jnp.stack([jnp.max(jnp.abs(g.astype(jnp.float32))) for g in leaves_g])
          worst_leaf_grad_norm = leaf_norms[worst_idx]
          worst_leaf_grad_maxabs = leaf_maxabs[worst_idx]

          return new_updates, MuonState(
              count=state.count + 1,
              orth_resid=step_orth_resid,
              worst_leaf_idx=worst_idx.astype(jnp.int32),
              worst_leaf_grad_norm=worst_leaf_grad_norm,
              worst_leaf_grad_maxabs=worst_leaf_grad_maxabs,
              mean_orth_resid=step_mean_orth_resid,
        )

        return optax.GradientTransformation(init_fn, update_fn)

    tx_muon = _muon_step(base_lr=0.001, weight_decay=0.02, ns_steps=muon_ns_steps)

    label_fn = _make_label_fn(muon_diagnostic_disable)

    clip_tx   = optax.clip_by_global_norm(1.0)
    damper_tx = burst_damper(decay=0.95, threshold_ratio=1.8, min_scale=0.05)

    multi_tx = optax.multi_transform(
        {
            "muon": tx_muon, "lion": tx_lion,
            "adamw_decay": tx_adamw_decay, "adamw_nodecay": tx_adamw_nodecay,
            "frozen": tx_frozen,
        },
        label_fn,
    )
    tx = optax.chain(damper_tx, clip_tx, multi_tx)
    return tx, lr_schedule


# ==========================================================================
# Cross-entropy loss
# ==========================================================================

def _chunked_ce_step(carry, chunk, w, smooth_positive, smooth_negative, vocab_size):
    sum_loss, sum_mask = carry
    hidden_chunk, label_chunk = chunk

    hidden_chunk = jnp.nan_to_num(
        jnp.clip(hidden_chunk, -1e3, 1e3),
        nan=0.0, posinf=1e3, neginf=-1e3,
    )

    logits_chunk = (
        hidden_chunk.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16)
    ).astype(jnp.float32)
    logits_chunk = jnp.nan_to_num(logits_chunk, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_chunk = jnp.clip(logits_chunk, -1e4, 1e4)
    logits_chunk = make_grad_probe("ce_logits_chunk")(logits_chunk)

    log_probs = jax.nn.log_softmax(logits_chunk, axis=-1)

    labels_safe = jnp.clip(label_chunk, 0, vocab_size - 1)
    nll = -jnp.take_along_axis(
        log_probs, labels_safe[:, None], axis=-1).squeeze(-1)

    if smooth_negative is not None:
        sum_log_probs = jnp.sum(log_probs, axis=-1)
        loss_vec = (
            nll * (smooth_positive - smooth_negative)
            - smooth_negative * sum_log_probs
        )
    else:
        loss_vec = nll

    mask = (label_chunk != -100).astype(jnp.float32)
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
    smooth_negative = (
        (label_smoothing / (vocab_size - 1)) if label_smoothing > 0 else None
    )

    scan_step = jax.checkpoint(
        lambda carry, chunk: _chunked_ce_step(
            carry, chunk, w, smooth_positive, smooth_negative, vocab_size)
    )

    (sum_loss, sum_mask), _ = jax.lax.scan(
        scan_step, (0.0, 0.0), (hidden_chunks, label_chunks))
    return sum_loss / jnp.maximum(sum_mask, 1.0)


def _stack_or_none(values):
    return jnp.stack(values) if values else None


def compute_loss(params, model_fn, batch, cfg: ModelConfig,
                  rngs=None, deterministic=False, return_aux=False,
                  ce_chunk_size=256, collinearity_coef=None):
    input_ids = batch["input_ids"]
    labels    = batch["labels"]

    kwargs = {"deterministic": deterministic, "return_hidden": True}
    if rngs is not None:
        kwargs["rngs"] = rngs

    outputs = model_fn(
        {"params": params}, input_ids, **kwargs,
        mutable=["losses"] if not deterministic else False,
    )

    expert_util_stacked = dropped_ratio_stacked = router_temp_stacked = None
    min_col_norm_stacked = max_abs_logit_preclip_stacked = None
    norm_x_mean_stacked = norm_x_max_stacked = norm_x_min_stacked = None
    assignment_frac_stacked = None
    aux_loss = z_loss = collinearity_loss = 0.0
    router_max_cos_per_layer = None
    router_max_cos = 0.0

    if not deterministic:
        final_hidden, sowed_vars = outputs
        losses = sowed_vars["losses"]

        aux_losses        = collect_by_leaf_name(losses, "aux_loss")
        z_losses          = collect_by_leaf_name(losses, "z_loss")
        expert_utils      = collect_by_leaf_name(losses, "expert_utilization")
        dropped_ratios    = collect_by_leaf_name(losses, "moe_dropped_ratio")
        router_temps      = collect_by_leaf_name(losses, "router_temp")
        min_col_norms     = collect_by_leaf_name(losses, "min_col_norm")
        max_abs_logits    = collect_by_leaf_name(losses, "max_abs_logit_preclip")
        collinearities    = collect_by_leaf_name(losses, "router_collinearity")
        max_cos_list      = collect_by_leaf_name(losses, "router_max_cos")
        norm_x_mean       = collect_by_leaf_name(losses, "norm_x_mean")
        norm_x_max        = collect_by_leaf_name(losses, "norm_x_max")
        norm_x_min        = collect_by_leaf_name(losses, "norm_x_min")
        assignment_fracs  = collect_by_leaf_name(losses, "assignment_frac")

        aux_loss        = jnp.sum(jnp.stack(aux_losses))   if aux_losses   else 0.0
        z_loss          = jnp.sum(jnp.stack(z_losses))     if z_losses     else 0.0
        collinearity_loss = jnp.sum(jnp.stack(collinearities)) if collinearities else 0.0

        if expert_utils:      expert_util_stacked         = jnp.stack(expert_utils)
        if dropped_ratios:    dropped_ratio_stacked       = jnp.stack(dropped_ratios)
        if router_temps:      router_temp_stacked         = jnp.stack(router_temps)
        if assignment_fracs:  assignment_frac_stacked     = jnp.stack(assignment_fracs)
        if min_col_norms:     min_col_norm_stacked        = jnp.stack(min_col_norms)
        if max_abs_logits:    max_abs_logit_preclip_stacked = jnp.stack(max_abs_logits)
        if norm_x_mean:       norm_x_mean_stacked         = jnp.stack(norm_x_mean)
        if norm_x_max:        norm_x_max_stacked          = jnp.stack(norm_x_max)
        if norm_x_min:        norm_x_min_stacked          = jnp.stack(norm_x_min)

        if max_cos_list:
            router_max_cos_per_layer = jnp.stack(max_cos_list)
            router_max_cos = jnp.max(router_max_cos_per_layer)

        # ФИКС (этот пасс): kernel/activation-level diag collection
        # (layer_delta_*, layer_resid_*, mamba2_*, gdn2_*, gdn2_kernelstage_*,
        # mla_*, final_hidden_*) удалён -- эти self.sow(...) больше не
        # существуют в model.py, оптимизаторская диагностика (per-group/
        # per-layer grad/weight stats, muon) полностью живёт в
        # train_setup.py и не зависит от этого блока.
    else:
        final_hidden = outputs

    if cfg.tie_embeddings:
        w = params["embed"]["embedding"].T
    else:
        w = params["lm_head"]["kernel"]

    final_hidden = jnp.nan_to_num(
        jnp.clip(final_hidden, -1e3, 1e3),
        nan=0.0, posinf=1e3, neginf=-1e3,
    )

    ce_loss = chunked_cross_entropy(
        final_hidden, labels, w, cfg.label_smoothing, chunk_size=ce_chunk_size)
    ce_loss = jnp.nan_to_num(ce_loss, nan=0.0, posinf=1e4, neginf=0.0)

    _coef = collinearity_coef if collinearity_coef is not None else ROUTER_COLLINEARITY_COEF
    total_loss = (
        ce_loss
        + cfg.router_aux_loss_coef * aux_loss
        + cfg.router_z_loss_coef   * z_loss
        + _coef                    * collinearity_loss
    )

    if return_aux:
        aux_info = {
            "ce_loss":                    ce_loss,
            "aux_loss":                   aux_loss,
            "z_loss":                     z_loss,
            "expert_utilization":         expert_util_stacked,
            "moe_dropped_ratio":          dropped_ratio_stacked,
            "router_temp":                router_temp_stacked,
            "min_col_norm":               min_col_norm_stacked,
            "max_abs_logit_preclip":      max_abs_logit_preclip_stacked,
            "norm_x_mean":                norm_x_mean_stacked,
            "norm_x_max":                 norm_x_max_stacked,
            "norm_x_min":                 norm_x_min_stacked,
            "router_max_cos_per_layer":   router_max_cos_per_layer,
            "router_max_cos":             router_max_cos,
            "assignment_frac":            assignment_frac_stacked,
        }
        return total_loss, aux_info
    return total_loss
