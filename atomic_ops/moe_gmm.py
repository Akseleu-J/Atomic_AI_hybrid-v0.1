"""
atomic_ops/moe_gmm.py -- Sparse MoE FFN via megablox `gmm`/`tgmm` Pallas
grouped-matmul TPU kernels, replacing moe_sparse.py's sort+gather-into-
(E,capacity+1,d)+nn.vmap(ExpertPack) dispatch.

Implements the integration plan (M0-M7, see project handoff doc) up through
M6. Not run on real TPU from this environment -- no TPU hardware here, see
CAVEAT below. Everything in this file was validated with
`jax.experimental.pallas.ops.tpu.megablox.gmm`'s own `interpret=True` mode
on CPU, which runs the SAME kernel logic through a pure-JAX/numpy
interpreter (not the compiled Mosaic kernel) -- exact-match against a plain
per-group-einsum reference for both the forward matmul shapes AND the
hand-written custom_vjp backward (see test_moe_gmm_parity.py in this same
delivery, run there for the actual numbers). Before switching this into
model.py's hot path, re-run that same test with `interpret=False` on the
real v5e-8 to confirm the compiled kernel agrees (same two-stage validation
discipline already used for kernel_trainable_B6.py vs kernel_trainable.py
in this project).

M1 -- routing without capacity/scatter
-----------------------------------------------------------------------
moe_sparse.py's SparseMoEJ has to invent a `capacity` and a sentinel slot
because its dispatch target is a *dense* (E, capacity+1, d) buffer sized
before routing is known -- any token past `capacity` for its expert is
dropped (see moe_sparse.py's own docstring on the sentinel-slot fix).
`gmm` needs no such buffer: it consumes a *sorted* (T, d) matrix directly,
grouped by `group_sizes` (a length-E_routed vector of per-expert token
counts that is exactly correct, computed fresh every forward pass via
`jnp.bincount`). Nothing is ever dropped -- `group_sizes.sum() == T`
identically, by construction, not just "usually true after warmup" the way
moe_sparse.py's dropped_ratio->0 is an emergent training outcome.

M2 -- forward FFN via gmm instead of nn.vmap(ExpertPack)
-----------------------------------------------------------------------
Expert weights are held as consolidated `self.param` tensors
`W1: (E_routed, d_model, d_ff)` / `W2: (E_routed, d_ff, d_model)` -- NOT a
flax `nn.vmap`-wrapped submodule with a param axis, matching how
moe_sparse.py's `routed_experts` axis is already unsharded/replicated (see
its own `_get_shard_spec` note in train.py) -- gmm needs the raw weight
array directly, it has no notion of a flax Module.

No bias terms: the plan (M2) specifies exactly `h = gelu(gmm(x,W1,sizes));
out = gmm(h,W2,sizes)`, and adding a per-expert bias would mean gathering
`b[expert_id]` per token, an *extra* data-dependent gather outside the gmm
call -- doable, but out of scope for this delivery; flagged in
GmmMoEJ's docstring as a follow-up if bias matters empirically.

M3 -- backward: gmm (dx) + tgmm (dW)
-----------------------------------------------------------------------
`gmm`/`tgmm` are themselves plain `jax.jit`-wrapped Pallas calls with no
autodiff rule of their own (confirmed: they are NOT `jax.custom_vjp`
objects, ordinary functions) -- differentiating through them naively would
try to trace through the Pallas kernel's dynamic-grid/dynamic-index
machinery, which is not expected to produce a usable VJP. So, per the plan,
a hand-written `jax.custom_vjp` wraps the whole two-gmm FFN:
    dh              = gmm(dout, W2^T-per-group, group_sizes)
    dW2             = tgmm(h^T, dout, group_sizes)
    dh_pre          = gelu_vjp(dh)          <- plain JAX autodiff, gelu is elementwise
    dx              = gmm(dh_pre, W1^T-per-group, group_sizes)
    dW1             = tgmm(x^T, dh_pre, group_sizes)
`group_sizes` itself is integer-valued and carries no gradient -- passed
via `nondiff_argnums`, same convention this project already uses for
`scale` in kernel_trainable.py/kernel_trainable_B6.py's custom_vjp.

M4 -- sanitization and dtype discipline
-----------------------------------------------------------------------
Same `clip(+-1e3)+nan_to_num` convention as moe_sparse.py, applied after
every gmm/tgmm call (forward AND backward, both are new numerical surfaces
this project hasn't stress-tested yet) -- not just at the final output.
`group_sizes`/routing indices are pinned to int32 explicitly (`gmm`'s own
common.py enforces this; see M0 smoke-test), same "explicit dtype anchor"
reasoning already applied for A_log/decay_a and B6's cotangent dtype fix.

M5 -- SPMD / sharding
-----------------------------------------------------------------------
Follows the *second* fix already landed in moe_sparse.py (`_local_sharded`,
not the superseded `_with_batch_sharding`/full-replication approach its own
docstring says was replaced): the whole routing+gmm block runs under an
explicit `with_sharding_constraint` pinning `flat_x`/`expert_idx`/
`gate_weight` (and everything derived data-dependently from them: perm,
group_sizes, x_sorted) to stay SHARDED along the batch axis -- each device
independently computes routing and calls `gmm`/`tgmm` on ONLY its own local
shard, no cross-device gather. This is *more* natural for gmm than for the
old capacity-buffer dispatch: `gmm`'s `group_offset`/`num_actual_groups`
hooks exist precisely to let each shard operate on a local slice, though
this delivery does not yet use expert-parallelism (E_routed experts stay
fully replicated across all devices, same deprioritization already on
record in userMemories/INTEGRATION_NOTES.md for the old implementation --
M5's expert-parallel variant is a distinct follow-up, not done here).

M6 -- integration into a SparseMoEJ-shaped module
-----------------------------------------------------------------------
`GmmMoEJ` below is a drop-in structural replacement for moe_sparse.py's
`SparseMoEJ` (same __call__ signature, same sown metrics:
`aux_loss`/`z_loss`/`moe_dropped_ratio`, same shared+routed combination) --
`moe_dropped_ratio` is sown as an always-0.0 constant (kept only so
train.py's existing "[DIAG] moe dropped_ratio" logging line and
collect_by_leaf_name() plumbing keep working unmodified; structurally
there is no dropping left to report, gmm's grouping never discards a
token).

CAVEAT (read before wiring into model.py)
-----------------------------------------------------------------------
This file was authored and logic-tested in an environment with **no TPU
and no real Mosaic compilation** -- `interpret=True` runs the *reference
interpreter* for the same kernel code, which is a strong but not
sufficient substitute for compiling on v5e-8 (tiling/(128,128,128)
assumptions, VMEM budget, and the actual Mosaic lowering are all
unverified here). Treat this the same way this project already treats
`kernel_trainable_B6.py` before it was trusted: run
`test_moe_gmm_parity.py` with `interpret=False` on your Kaggle TPU v5e-8
FIRST, compare against moe_sparse.py's SparseMoEJ (or a plain dense
JAX-einsum reference) on a few seeds/sizes, and only switch model.py's
import over once that's finite and rel_diff-small, per this project's own
"equivalence testing before production use" discipline.

ФИКС (этот пасс -- router collapse, см. чат: expert_utilization_std рос
монотонно 0.005 -> 0.30+ на протяжении нескольких сотен шагов сразу после
resume, независимо от router_z_loss_coef/router_noise_std -- z_loss
штрафует величину логитов через градиент, но ничего структурно не
ОГРАНИЧИВАЕТ саму величину; под Adam норма router.kernel может свободно
расти, и с ней растёт разброс логитов, даже если штраф формально
уменьшает его СКОРОСТЬ роста):

  1. router.kernel теперь используется через L2-НОРМАЛИЗОВАННЫЕ СТОЛБЦЫ
     (по одному направлению на эксперта) -- величина логита теперь зависит
     ТОЛЬКО от угла между входом и направлением эксперта (умноженного на
     ||flat_x||), а не от того, насколько разрослась норма самого
     router.kernel под оптимизатором. Это СТРУКТУРНЫЙ потолок, а не
     штраф -- в отличие от z_loss, коллапс логитов через рост весов
     физически невозможен независимо от того, что накопил Adam-momentum
     ДО этого шага.
  2. Добавлен обучаемый router_temp (скаляр, init=10.0) -- после
     нормализации величина "сырого" логита ~ ||x||*cos(angle), что
     заметно меньше произвольного диапазона до фикса; без температуры
     softmax/top_k стал бы слишком плоским (все эксперты почти
     равновероятны) и МЕДЛЕННЕЕ обучался бы отличать токены. Temperature
     обучается тем же градиентным путём, что и остальная модель -- НЕ
     добавляет новую multi_transform-группу в optimizer.py (попадает в
     "other"/default-группу _label_leaf, тот же LR, что у большинства
     параметров).
  3. Узкий clip(-8, 8) на router_logits ДО softmax/noise -- exp(8)/exp(-8)
     ~ 3000x разницы между самым и наименее вероятным экспертом при
     top_gate softmax -- этого более чем достаточно для уверенной
     маршрутизации, но структурно не даёт уйти в экстремальный
     почти-one-hot коллапс, даже если temperature сама по себе окажется
     плохо откалиброванной на первых шагах. Старый широкий _sanitize
     (clip ±1e3) сохранён КАК ЕСТЬ для остальных величин в файле (h_pre,
     out, dW1/dW2/dx/dh) -- он защищает от overflow, не от router
     collapse, и трогать его не нужно.

Обратная совместимость с чекпоинтами: router.kernel остаётся тем же
self.param с той же формой (d_model, E_routed) -- ТОЛЬКО способ его
использования внутри forward меняется (нормализация -- чистая функция
существующего параметра, не новый параметр). router_temp -- НОВЫЙ
параметр, которого не было в старых чекпоинтах; restore существующего
чекпоинта БЕЗ router_temp потребует либо FORCE_FRESH_START, либо (что
дешевле и правильнее здесь) переинициализации ТОЛЬКО router-группы после
restore -- см. train.py's router-reset patch, который в любом случае
рекомендован отдельно для лечения уже накопленного router collapse на
чекпоинте до этого фикса.

ФИКС (этот пасс -- shared expert без градиентной защиты, см. чат: [DIAG]
стабильно светил non-finite именно в группах moe/embed/other, при этом
router_max_cos/min_col_norm полностью здоровы (0.02-0.08 / 0.916-0.918) --
MoE-роутинг исключён как источник. Разобрано по узлам: routed-путь уже
защищён -- КАЖДЫЙ gmm/tgmm вызов в _core_bwd проходит через _sanitize.
shared_w1/shared_w2 -- голый nn.Dense БЕЗ какой-либо защиты градиента,
единственный узел в GmmMoEJ без sanitizer'а. Добавлен _moe_grad_sanitizer
на shared_out, тот же паттерн/тег-конвенция, что уже применяется к
flat_x_for_router (moe_router_input_grad_*) -- активно клипует несущийся
назад градиент (не только детектирует, как make_grad_probe), прежде чем
он попадёт в residual stream через combined/current_x.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm, tgmm


_DEFAULT_TILING = (128, 128, 128)
_SANITIZE_CLIP = 1e3

# ФИКС: узкий clip специально для router_logits -- отдельная константа от
# _SANITIZE_CLIP (которая остаётся широкой ±1e3 overflow-защитой для
# остальных величин в файле). exp(8) ~ 2981, exp(-8) ~ 0.000335 -- диапазон
# ~9e6 между самым уверенным и самым неуверенным логитом даже ДО softmax,
# более чем достаточно для полной уверенности маршрутизации без риска
# экстремального one-hot коллапса, устойчивого к любому текущему
# router_z_loss_coef/router_noise_std.
_ROUTER_LOGIT_CLIP = 8.0
_ROUTER_TEMP_INIT = 10.0
import os

_MOE_FWD_DIAG = os.environ.get("GDN2_FWD_DIAG", "0") == "1"
def _safe_normalize(t, eps=1e-6):
    """L2-нормализация по последней оси (по строкам)."""
    return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps)

def _moe_grad_sanitizer(tag: str, clip_val: float = 1e3):
    """Local copy of model.py's make_grad_sanitizer (same reasoning
    optimizer.py already gives for its own local copy: avoid a
    moe_gmm.py -> model.py import for one internal helper).

    Placed on the flat_x copy that feeds router_logits (see call site
    in GmmMoEJ.__call__ below) -- isolation there means the cotangent this
    node sees on the real backward pass is EXACTLY the router's own
    contribution to dflat_x. This replaces the synthetic dL/dflat_x check
    from the offline diagnostic script -- here it's the ACTUAL gradient
    from the ACTUAL training step, not a proxy, and it costs nothing extra
    since it's already being computed by autodiff.

    ФИКС (этот пасс): ТАКЖЕ placed on shared_out (see call site below) --
    previously the shared expert (shared_w1/shared_w2, plain nn.Dense) was
    the ONLY node in this whole file with no gradient protection at all,
    while the routed gmm/tgmm path already sanitizes every backward output
    inside _core_bwd. Same helper, same reasoning, different call site --
    see module docstring's "ФИКС (этот пасс -- shared expert...)" section.

    Also ACTIVELY clips (not just reports) -- same "diagnose AND fix"
    pattern already used for mla_flash_attn_out/gdn2_q_normalize/
    gdn2_k_normalize/delta_fanin_* in model.py."""
    @jax.custom_vjp
    def _sanitizer(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        finite = jnp.all(jnp.isfinite(g))
        max_abs = jnp.max(jnp.abs(jnp.where(jnp.isfinite(g), g, 0.0)))
        if _MOE_FWD_DIAG:
            jax.lax.cond(
                jnp.logical_or(jnp.logical_not(finite), max_abs > clip_val),
                lambda: jax.debug.print(
                    "[MOE-BWD-DIAG] ⚠️ " + tag + ": incoming grad finite={f} "
                    "max_abs={m:.3e} -- sanitized to +-{c}",
                    f=finite, m=max_abs, c=clip_val,
                ),
                lambda: None,
            )
        g_safe = jnp.nan_to_num(jnp.clip(g, -clip_val, clip_val),
                                 nan=0.0, posinf=clip_val, neginf=-clip_val)
        return (g_safe,)

    _sanitizer.defvjp(_fwd, _bwd)
    return _sanitizer
def _moe_grad_probe(tag: str):
    """Как model.py's make_grad_probe -- ТОЛЬКО детектирует non-finite во
    входящем backward-градиенте, не чинит. Нужен, чтобы локализовать
    ТОЧНУЮ операцию внутри gmm/tgmm forward+backward цепочки, где
    non-finite впервые появляется -- в отличие от _sanitize (который уже
    есть на каждом шаге), probe ничего не маскирует и печатает ПЕРВОЕ
    место появления."""
    @jax.custom_vjp
    def _probe(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        finite = jnp.all(jnp.isfinite(g))
        max_abs = jnp.max(jnp.abs(jnp.where(jnp.isfinite(g), g, 0.0)))
        jax.lax.cond(
            jnp.logical_not(finite),
            lambda: jax.debug.print(
                "[MOE-GMM-DIAG] 🔴 non-finite ВХОДЯЩИЙ градиент в узле: " + tag +
                "  (max_abs конечной части={m:.3e})", m=max_abs,
            ),
            lambda: None,
        )
        return (g,)

    _probe.defvjp(_fwd, _bwd)
    return _probe
def _sanitize(x, clip=_SANITIZE_CLIP):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


def _auto_tile(m, k, n, m_pref=128, k_pref=128, n_pref=128):
    def _pick(d, pref):
        return pref if d % pref == 0 else d
    return (_pick(m, m_pref), _pick(k, k_pref), _pick(n, n_pref))


def _make_grouped_ffn_core(interpret=False, diag_tag="moe"):

    def _fwd_math(x_sorted, W1, W2, group_sizes):
        x_f = x_sorted.astype(jnp.float32)
        W1_f = W1.astype(jnp.float32)
        W2_f = W2.astype(jnp.float32)

        T, d_model = x_f.shape
        _, _, d_ff = W1_f.shape

        # ФИКС (диагностика, см. чат): group_sizes min/max -- прямая
        # проверка гипотезы "вырожденная (0- или почти-0-токенная) группа
        # ломает gmm/tgmm". Sown КАЖДЫЙ forward-вызов, недорого (пара
        # reduce поверх уже вычисленного bincount).
        _min_group = jnp.min(group_sizes).astype(jnp.float32)
        _max_group = jnp.max(group_sizes).astype(jnp.float32)

        h_pre = gmm(x_f, W1_f, group_sizes,
                    tiling=_auto_tile(T, d_model, d_ff),
                    interpret=interpret, preferred_element_type=jnp.float32)
        # ФИКС: probe ДО _sanitize -- ловит non-finite ИЗ САМОГО gmm-вызова
        # (h_pre), прежде чем _sanitize его замаскирует.
        h_pre_finite = jnp.all(jnp.isfinite(h_pre))
        h_pre_maxabs = jnp.max(jnp.abs(jnp.where(jnp.isfinite(h_pre), h_pre, 0.0)))
        jax.lax.cond(
            jnp.logical_or(jnp.logical_not(h_pre_finite), h_pre_maxabs > 1e2),
            lambda: jax.debug.print(
                "[MOE-GMM-FWD-DIAG] " + diag_tag + ": gmm(x,W1) h_pre "
                "finite={f} maxabs={m:.3e} min_group={mn:.0f} max_group={mx:.0f}",
                f=h_pre_finite, m=h_pre_maxabs, mn=_min_group, mx=_max_group,
            ),
            lambda: None,
        )
        h_pre = _sanitize(h_pre)

        h, gelu_vjp = jax.vjp(jax.nn.gelu, h_pre)

        out = gmm(h, W2_f, group_sizes,
                  tiling=_auto_tile(T, d_ff, d_model),
                  interpret=interpret, preferred_element_type=jnp.float32)
        out_finite = jnp.all(jnp.isfinite(out))
        out_maxabs = jnp.max(jnp.abs(jnp.where(jnp.isfinite(out), out, 0.0)))
        jax.lax.cond(
            jnp.logical_or(jnp.logical_not(out_finite), out_maxabs > 1e2),
            lambda: jax.debug.print(
                "[MOE-GMM-FWD-DIAG] " + diag_tag + ": gmm(h,W2) out "
                "finite={f} maxabs={m:.3e}", f=out_finite, m=out_maxabs,
            ),
            lambda: None,
        )
        out = _sanitize(out)
        return out, h, gelu_vjp, group_sizes

    @jax.custom_vjp
    def _core(x_sorted, W1, W2, group_sizes):
        out, _, _, _ = _fwd_math(x_sorted, W1, W2, group_sizes)
        return out.astype(x_sorted.dtype)

    def _core_fwd(x_sorted, W1, W2, group_sizes):
        out, h, gelu_vjp, group_sizes = _fwd_math(x_sorted, W1, W2, group_sizes)
        residuals = (x_sorted, W1, W2, group_sizes, h, gelu_vjp)
        return out.astype(x_sorted.dtype), residuals

    def _core_bwd(residuals, dout):
        x_sorted, W1, W2, group_sizes, h, gelu_vjp = residuals
        x_f = x_sorted.astype(jnp.float32)
        W1_f = W1.astype(jnp.float32)
        W2_f = W2.astype(jnp.float32)

        # ФИКС: probe на ВХОДЯЩИЙ cotangent dout -- если он уже non-finite
        # ЗДЕСЬ, источник находится ВЫШЕ по графу (не внутри самого MoE
        # блока), и всё нижеследующее -- уже вторичный эффект.
        dout_finite = jnp.all(jnp.isfinite(dout))
        jax.lax.cond(
            jnp.logical_not(dout_finite),
            lambda: jax.debug.print(
                "[MOE-GMM-BWD-DIAG] 🔴 " + diag_tag + ": ВХОДЯЩИЙ dout "
                "уже non-finite ДО входа в MoE backward -- источник ВЫШЕ по графу"),
            lambda: None,
        )
        dout_f = _sanitize(dout.astype(jnp.float32))

        T, d_model = x_f.shape
        _, d_ff = h.shape

        dW2 = tgmm(h.T, dout_f, group_sizes,
                   tiling=_auto_tile(d_ff, T, d_model),
                   interpret=interpret, preferred_element_type=jnp.float32)
        dW2_finite = jnp.all(jnp.isfinite(dW2))
        jax.lax.cond(
            jnp.logical_not(dW2_finite),
            lambda: jax.debug.print("[MOE-GMM-BWD-DIAG] 🔴 " + diag_tag + ": tgmm dW2 non-finite"),
            lambda: None,
        )
        dW2 = _sanitize(dW2)

        W2_T = jnp.swapaxes(W2_f, 1, 2)
        dh = gmm(dout_f, W2_T, group_sizes,
                 tiling=_auto_tile(T, d_model, d_ff),
                 interpret=interpret, preferred_element_type=jnp.float32)
        dh_finite = jnp.all(jnp.isfinite(dh))
        jax.lax.cond(
            jnp.logical_not(dh_finite),
            lambda: jax.debug.print("[MOE-GMM-BWD-DIAG] 🔴 " + diag_tag + ": gmm dh non-finite"),
            lambda: None,
        )
        dh = _sanitize(dh)

        (dh_pre,) = gelu_vjp(dh)
        dh_pre = _sanitize(dh_pre.astype(jnp.float32))

        dW1 = tgmm(x_f.T, dh_pre, group_sizes,
                   tiling=_auto_tile(d_model, T, d_ff),
                   interpret=interpret, preferred_element_type=jnp.float32)
        dW1_finite = jnp.all(jnp.isfinite(dW1))
        jax.lax.cond(
            jnp.logical_not(dW1_finite),
            lambda: jax.debug.print("[MOE-GMM-BWD-DIAG] 🔴 " + diag_tag + ": tgmm dW1 non-finite"),
            lambda: None,
        )
        dW1 = _sanitize(dW1)

        W1_T = jnp.swapaxes(W1_f, 1, 2)
        dx = gmm(dh_pre, W1_T, group_sizes,
                 tiling=_auto_tile(T, d_ff, d_model),
                 interpret=interpret, preferred_element_type=jnp.float32)
        dx_finite = jnp.all(jnp.isfinite(dx))
        jax.lax.cond(
            jnp.logical_not(dx_finite),
            lambda: jax.debug.print("[MOE-GMM-BWD-DIAG] 🔴 " + diag_tag + ": gmm dx non-finite"),
            lambda: None,
        )
        dx = _sanitize(dx)

        dgroup_sizes = jnp.zeros(group_sizes.shape, dtype=jax.dtypes.float0)

        return (
            dx.astype(x_sorted.dtype),
            dW1.astype(W1.dtype),
            dW2.astype(W2.dtype),
            dgroup_sizes,
        )

    _core.defvjp(_core_fwd, _core_bwd)
    return _core

class GmmMoEJ(nn.Module):
    """Top-k=2 версия. Каждый токен маршрутизируется к ДВУМ из E_routed
    экспертов (top_k=2 через jax.lax.top_k), веса гейта перенормируются
    внутри выбранной пары (softmax только по 2 значениям, не по всем
    E_routed) -- тот же приём, что Switch/GShard-стиль top-k MoE.

    Диспетчинг реализован как ОДИН gmm/tgmm-проход над 2T строками:
    каждый токен дублируется дважды (по одной копии на каждый из двух
    выбранных экспертов), затем ОБЫЧНАЯ top-1-механика (argsort по
    expert_idx / bincount / gmm) применяется к этому 2T-массиву без
    изменений в самом _make_grouped_ffn_core -- gmm/tgmm агностичны к
    происхождению group_sizes, им всё равно T это или 2T строк.

    combine: для каждого исходного токена суммируются его ДВЕ выходные
    строки (первая и вторая копия), взвешенные перенормированными
    top-2 гейтами.

    ФИКС (router collapse): router теперь использует L2-нормализованные
    столбцы + обучаемую температуру -- см. module docstring "ФИКС (этот
    пасс...)" выше для полного обоснования.

    ФИКС (shared expert grad protection): см. module docstring's
    "ФИКС (этот пасс -- shared expert...)" -- shared_out теперь тоже
    проходит через _moe_grad_sanitizer.
    """
    cfg: object
    interpret: bool = False
    top_k: int = 2

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rngs=None):
        b, l, d = x.shape
        flat_x = x.reshape(-1, d)
        # ФИКС (live router diagnostics -- replaces offline synthetic
        # replay, which twice proved the router's forward math is
        # structurally bounded and can't be the source of a runaway on its
        # own; see project notes on test_synthetic_router_collapse*.py).
        # tag is unique per block via the flax scope path (e.g.
        # "block_3/moe"), evaluated at trace time -- one distinct probe
        # per real MoE block, not a single shared one.
        _moe_tag = "/".join(str(p) for p in self.scope.path) if self.scope is not None else "moe"
        flat_x_for_router = _moe_grad_sanitizer(f"moe_router_input_grad_{_moe_tag}")(flat_x)
        
        # ---- новая диагностика: ||flat_x|| (по строкам) ----
        _norm_per_token = jnp.linalg.norm(flat_x_for_router, axis=1, keepdims=False)
        self.sow("losses", "norm_x_mean", jnp.mean(_norm_per_token))
        self.sow("losses", "norm_x_max", jnp.max(_norm_per_token))
        self.sow("losses", "norm_x_min", jnp.min(_norm_per_token))
        # ----------------------------------------------------
        T = flat_x.shape[0]
        E_routed = self.cfg.num_experts - 1
        k = self.top_k
        assert E_routed >= k, f"num_experts-1={E_routed} must be >= top_k={k}."
        from model import get_model_mesh, get_batch_axis
        mesh = get_model_mesh()
        batch_axis = get_batch_axis()

        # ---- shared expert ----
        shared_h = nn.Dense(self.cfg.d_ff, name="shared_w1", dtype=jnp.bfloat16)(flat_x)
        shared_h = jax.nn.gelu(shared_h)
        shared_h = nn.Dropout(rate=self.cfg.dropout_rate)(shared_h, deterministic=deterministic)
        shared_out = nn.Dense(self.cfg.d_model, name="shared_w2", dtype=jnp.bfloat16)(shared_h)
        # ФИКС (этот пасс -- см. module docstring "ФИКС (этот пасс --
        # shared expert...)"): shared_w1/shared_w2 были ЕДИНСТВЕННЫМ узлом
        # в GmmMoEJ без градиентной защиты -- routed-путь уже санитизирован
        # на каждом шаге backward внутри _core_bwd. Активный клип (не
        # только детект), тот же паттерн, что уже используется для
        # flat_x_for_router выше.
        shared_out = _moe_grad_sanitizer(f"moe_shared_expert_grad_{_moe_tag}")(shared_out)

        # ---- routing: top-k среди E_routed ----
        # ФИКС (router collapse): router.kernel -- явный self.param (не
        # nn.Dense), т.к. нужен прямой доступ к сырой матрице ДЛЯ
        # нормализации столбцов ПЕРЕД матмулом. Форма (d_model, E_routed)
        # -- та же, что раньше выдавал nn.Dense(E_routed, use_bias=False),
        # так что чекпоинт-совместимость по ЭТОМУ параметру сохранена
        # (веса можно даже restore'ить напрямую, если имя/форма совпадают
        # -- restore обычно матчит по имени пути, "router"/"kernel").
        router_kernel = self.param(
            "router", nn.initializers.lecun_normal(), (d, E_routed), jnp.float32
        )
        # L2-нормализация СТОЛБЦОВ (одно направление на эксперта) --
        # структурный потолок на величину логита, не зависящий от того,
        # насколько разрослась норма router_kernel под оптимизатором.
        router_kernel_normed = router_kernel * jax.lax.rsqrt(
            jnp.sum(router_kernel ** 2, axis=0, keepdims=True) + 1e-6
        )

        # ДОБАВИТЬ: штраф за схожесть направлений экспертов (router_max_cos=0.96 —
        # это и есть измеренное этим членом). Считаем только off-diagonal Грам-матрицу.
        gram = jnp.dot(router_kernel_normed.T, router_kernel_normed,
                        precision=jax.lax.Precision.HIGHEST)  # (E_routed, E_routed), диагональ ~1.0
        eye = jnp.eye(E_routed, dtype=gram.dtype)
        off_diag_sq = jnp.square(gram - eye) * (1.0 - eye)   # обнулить диагональ явно
        router_collinearity = jnp.sum(off_diag_sq) / (E_routed * (E_routed - 1))
        self.sow("losses", "router_collinearity", router_collinearity)

        _max_cos_offdiag = jnp.max(jnp.abs(gram - eye))       # тот же max_cos, что вы мерили офлайн
        self.sow("losses", "router_max_cos", _max_cos_offdiag)
        router_temp = self.param(
            "router_temp", nn.initializers.constant(_ROUTER_TEMP_INIT), ()
        )
        router_temp_clipped = jnp.clip(router_temp, 1.0, 15.0)   # структурный потолок
        # ФИКС (router saturation): нормируем вход по L2, чтобы logit ∈ [-temp, temp]
        flat_x_normed_for_router = _safe_normalize(flat_x_for_router.astype(jnp.float32))
        router_logits = jnp.dot(
        flat_x_normed_for_router, router_kernel_normed, precision=jax.lax.Precision.HIGHEST
        ) * router_temp_clipped
        # ФИКС (live diagnostics, forward side): capture pre-clip logit
        # magnitude and pre-normalization column norm BEFORE the existing
        # narrow clip/sanitize -- these are the two forward-side
        # early-warning signals the offline snapshot script computed
        # after-the-fact; sowing them here makes them visible every real
        # step, not just at the one step a non-finite snapshot happened to
        # catch.
        _max_abs_logit_preclip = jnp.max(jnp.abs(router_logits))
        _min_col_norm = jnp.min(jnp.sqrt(jnp.sum(router_kernel ** 2, axis=0)))
        self.sow("losses", "max_abs_logit_preclip", _max_abs_logit_preclip)
        self.sow("losses", "min_col_norm", _min_col_norm)

        if _MOE_FWD_DIAG:
            jax.lax.cond(
                jnp.logical_or(_max_abs_logit_preclip > _ROUTER_LOGIT_CLIP * 1.5, _min_col_norm < 1e-3),
                lambda: jax.debug.print(
                    "[MOE-FWD-DIAG] ⚠️ " + _moe_tag + ": max|logit|(pre-clip)={m:.3f}  "
                    "min_col_norm={c:.6f}", m=_max_abs_logit_preclip, c=_min_col_norm,
                ),
                lambda: None,
            )

        router_logits = jnp.clip(router_logits, -_ROUTER_LOGIT_CLIP, _ROUTER_LOGIT_CLIP)
        router_logits = _sanitize(router_logits)
        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout") if rngs is None else rngs.get("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape, dtype=router_logits.dtype
            )

        # M1': top-k выбор + перенормировка гейтов ВНУТРИ выбранной пары
        # (не softmax по всем E_routed -- иначе вес каждого выбранного
        # эксперта занижен относительно "честного" top-k распределения).
        expert_bias = self.param("expert_bias", nn.initializers.zeros, (E_routed,), jnp.float32)
        self.sow("losses", "expert_bias", expert_bias)  # для W&B, тот же паттерн что router_temp

        router_logits_biased = router_logits + jax.lax.stop_gradient(expert_bias)[None, :]
        top_vals, top_idx = jax.lax.top_k(router_logits_biased, k=k)
        top_idx = top_idx.astype(jnp.int32)
        # ВАЖНО: вес гейта считается по НЕбиасированным логитам выбранных экспертов --
        # bias влияет только на ВЫБОР top-k, не на итоговый вес (как в DeepSeek-V3).
        top_vals_unbiased = jnp.take_along_axis(router_logits, top_idx, axis=-1)
        top_gate = jax.nn.softmax(top_vals_unbiased, axis=-1)
        _assign_counts = jnp.zeros((E_routed,), dtype=jnp.float32)
        for _j in range(k):
            _assign_counts = _assign_counts + jnp.sum(
                jax.nn.one_hot(top_idx[:, _j], E_routed), axis=0
            )
        assignment_frac = _assign_counts / (T * k)
        self.sow("losses", "assignment_frac", assignment_frac)
 
        # aux_loss/z_loss считаются по ПОЛНОМУ softmax(router_logits) --
        # это диагностика балансировки роутера как такового (Switch-style
        # load-balancing loss смотрит на распределение по ВСЕМ экспертам,
        # не только выбранным), а не по факту 2T-дисптача.
        full_probs = jax.nn.softmax(router_logits, axis=-1)
        full_probs = jnp.nan_to_num(full_probs, nan=0.0, posinf=0.0, neginf=0.0)
        mean_probs = jnp.mean(full_probs, axis=0)
        self.sow("losses", "aux_loss", E_routed * jnp.sum(mean_probs * mean_probs))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(
            jax.scipy.special.logsumexp(router_logits, axis=-1))))
        self.sow("losses", "expert_utilization", mean_probs)
        # ФИКС (диагностика): router_temp сам по себе -- полезный
        # индикатор ("растёт ли температура сама по себе, компенсируя
        # узкий clip"). Сожено сюда же, тем же collect_by_leaf_name-путём,
        # что и остальные MoE-метрики -- optimizer.py уже собирает
        # aux_loss/z_loss/expert_utilization по имени листа, router_temp
        # добавлен по аналогии для W&B-видимости.
        self.sow("losses", "router_temp", router_temp)

        d_model, d_ff = self.cfg.d_model, self.cfg.d_ff
        W1 = self.param("routed_w1", nn.initializers.lecun_normal(),
                         (E_routed, d_model, d_ff), jnp.bfloat16)
        W2 = self.param("routed_w2", nn.initializers.lecun_normal(),
                         (E_routed, d_ff, d_model), jnp.bfloat16)

        # ==================================================================
        # M2': дублирование токенов под 2T-диспетчинг (k копий на токен,
        # каждая помечена одним из k выбранных экспертов, взвешена своим
        # перенормированным гейтом). Порядок по оси-k сохраняется через
        # concatenate -- используется для обратного split на combine.
        # ==================================================================
        
        def _dispatch_and_ffn(flat_x_local, expert_idx_local, W1_local, W2_local):
            T_rep = flat_x_local.shape[0]  # = k * T_local_device
            group_sizes = jnp.bincount(expert_idx_local, length=E_routed).astype(jnp.int32)
            perm = jnp.argsort(expert_idx_local, stable=True)
            inv_perm = jnp.argsort(perm)

            x_sorted = jnp.take(flat_x_local, perm, axis=0)
            grouped_ffn = _make_grouped_ffn_core(interpret=self.interpret, diag_tag=_moe_tag)
            out_sorted = grouped_ffn(x_sorted.astype(jnp.bfloat16), W1_local, W2_local, group_sizes)

            out_unsorted = jnp.take(out_sorted, inv_perm, axis=0)  # (T_rep, d), тот же порядок, что flat_x_local
            # ФИКС: min/max group_sizes для диагностики -- возвращаются явно, а не
            # sow'ятся здесь напрямую, т.к. эта функция вызывается ВНУТРИ
            # jax.shard_map (см. _dispatch_local_topk ниже) -- self.sow там
            # недоступен/небезопасен, поэтому значения прокидываются наружу через
            # return и sow'ятся уже СНАРУЖИ shard_map, в основном теле __call__.
            _min_gs = jnp.min(group_sizes).astype(jnp.float32)
            _max_gs = jnp.max(group_sizes).astype(jnp.float32)
            return out_unsorted, _min_gs, _max_gs
        # flat_x_rep: (k*T, d) -- k конкатенированных копий flat_x, в
        # порядке [копия под top-1-выбор для каждого токена][копия под
        # top-2-выбор для каждого токена]...
        flat_x_rep = jnp.concatenate([flat_x] * k, axis=0)
        expert_idx_rep = jnp.concatenate(
            [top_idx[:, j] for j in range(k)], axis=0
        )  # (k*T,)

        if mesh is not None and batch_axis is not None:
            def _dispatch_local_topk(flat_x_local, top_idx_local, top_gate_local, W1_local, W2_local):
                flat_x_rep_local = jnp.concatenate([flat_x_local] * k, axis=0)
                expert_idx_rep_local = jnp.concatenate(
                    [top_idx_local[:, j] for j in range(k)], axis=0
                )
                out_rep_local, _min_gs, _max_gs = _dispatch_and_ffn(
                    flat_x_rep_local, expert_idx_rep_local, W1_local, W2_local
                )
                out_chunks_local = jnp.split(out_rep_local, k, axis=0)
                combined_local = jnp.zeros_like(flat_x_local, dtype=jnp.float32)
                for j in range(k):
                    combined_local = combined_local + out_chunks_local[j].astype(jnp.float32) * top_gate_local[:, j:j+1]
                # ФИКС: min/max group_sizes прокидываются наружу третьим/четвёртым
                # выходом shard_map -- нужно добавить соответствующий out_spec ниже.
                return combined_local, _min_gs, _max_gs
            in_specs = (
                P(batch_axis, None),   # flat_x
                P(batch_axis, None),   # top_idx
                P(batch_axis, None),   # top_gate
                P(None, None, None),   # W1
                P(None, None, None),   # W2
            )
            out_specs = (P(batch_axis, None), P(), P())   # ФИКС: +2 скалярных выхода
            sharded_dispatch = jax.shard_map(
                _dispatch_local_topk, mesh=mesh,
                in_specs=in_specs, out_specs=out_specs,
                check_vma=False,
            )
            routed_out, _moe_min_group, _moe_max_group = sharded_dispatch(flat_x, top_idx, top_gate, W1, W2)

            if mesh is None:
                raise RuntimeError(
                    "GmmMoEJ: get_model_mesh() вернул None во время трассировки — "
                    "set_model_mesh() должен быть вызван до первого model.init()/jit."
               )
            self.sow("losses", "moe_min_group_size", jnp.min(_min_gs_all))
            self.sow("losses", "moe_max_group_size", jnp.max(_max_gs_all))
        else:
            routed_out_rep, _moe_min_group, _moe_max_group = _dispatch_and_ffn(flat_x_rep, expert_idx_rep, W1, W2)
            out_chunks = jnp.split(routed_out_rep, k, axis=0)
            routed_out = jnp.zeros_like(flat_x, dtype=jnp.float32)
            for j in range(k):
                routed_out = routed_out + out_chunks[j].astype(jnp.float32) * top_gate[:, j:j+1]
            combined = shared_out.astype(jnp.float32) + routed_out
            combined = _sanitize(combined)

            # ФИКС: sow group_sizes min/max -- единое место после обеих веток
            # (mesh/non-mesh), чтобы не дублировать sow-вызов.
            self.sow("losses", "moe_min_group_size", _moe_min_group)
            self.sow("losses", "moe_max_group_size", _moe_max_group)

            self.sow("losses", "moe_dropped_ratio", jnp.zeros((), dtype=jnp.float32))
            return combined.reshape(b, l, d).astype(x.dtype)
        # ==================================================================
        # M3': combine -- разбить (k*T, d) обратно на k кусков по T строк
        # каждый (тот же порядок конкатенации, что при dispatch), взвесить
        # top_gate[:, j] и просуммировать.
        # ==================================================================

        combined = shared_out.astype(jnp.float32) + routed_out
        combined = _sanitize(combined)

        self.sow("losses", "moe_dropped_ratio", jnp.zeros((), dtype=jnp.float32))
        return combined.reshape(b, l, d).astype(x.dtype)
