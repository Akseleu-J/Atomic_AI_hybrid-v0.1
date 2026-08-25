"""
atomic_ops/kernel_diag.py -- ВСЕГДА включённая, дешёвая диагностика ВНУТРИ
GDN-2 Pallas-пайплайна (Aqk/Akk/A/w_pseudo/u/kg/qg), в дополнение к
kernel_d_pipeline.py's _stage_diag (который требует GDN2_FWD_DIAG=1 и идёт
через jax.debug.print/host-callback -- дорого и не логируется в W&B).

Зачем это отдельный файл, а не правка kernel_d_pipeline.py: диагностика
здесь работает ПОД jax.lax.stop_gradient и вызывается ДОПОЛНИТЕЛЬНО,
рядом с основным (дифференцируемым) вызовом gdn2_pallas_forward_trainable
в model.py -- она НЕ участвует в custom_vjp, не меняет форму
forward/backward, не требует трогать shard_map's in_specs/out_specs в
model.py (которые сейчас жёстко ожидают ровно (o, h_final)).

Стоимость: Kernel A + Kernel B (build_chunk_scores_pallas + wy_solve_pallas)
пересчитываются ЕЩЁ РАЗ, под stop_gradient -- то же вычисление, что уже
делает _gdn2_core_bwd внутри kernel_trainable_B6.py на backward-проходе
(residuals там всё равно пересчитываются с нуля, "recompute, don't stash"
конвенция этого проекта), так что относительная добавка стоимости на
ПОЛНЫЙ шаг (forward+backward) умеренная -- один лишний forward A+B+C на
каждый физический GDN-2 слой, каждый эффективный шаг.

Все значения -- ЧИСТЫЕ jnp-скаляры (max|abs| конечной части + isfinite),
без jax.debug.print/callback -- ровно тот же "self.sow(...) + чистый jnp"
паттерн, что diagnostics.py/model.py уже используют повсеместно.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .kernel_a_scores import build_chunk_scores_pallas
from .kernel_b_solve import wy_solve_pallas
from .kernel_c_recompute import recompute_wy_pallas


def _stat(x):
    """(max|abs| конечной части, isfinite-флаг как float32 0/1) для
    произвольного jnp-массива -- дешёвые reduce, никакого device->host
    внутри jit."""
    finite_mask = jnp.isfinite(x)
    safe_x = jnp.where(finite_mask, x, 0.0)
    maxabs = jnp.max(jnp.abs(safe_x))
    all_finite = jnp.all(finite_mask).astype(jnp.float32)
    return maxabs, all_finite


def gdn2_kernel_stage_diagnostics(q, k, v, w, b, g, scale, axis_name = None):
    """Пересчитывает Kernel A -> B -> C ПОД stop_gradient (эта функция
    НИКОГДА не участвует в backward -- чисто диагностический побочный
    прогон) и возвращает плоский dict СКАЛЯРОВ:

        {stage}_maxabs, {stage}_isfinite

    для stage in {aqk, akk, a_wy_inverse, w_pseudo, u, kg, qg}.

    Вызывать из GatedDeltaNet2J.__call__ СРАЗУ ПОСЛЕ основного
    (дифференцируемого) вызова gdn2_pallas_forward_trainable, на тех же
    (уже санитизированных) q/k/v/w/b/g -- см. model.py.
    axis_name: если задан (например, "tpu_nodes" -- batch_axis mesh'а) --
    функция ДОЛЖНА вызываться внутри jax.shard_map с этим axis_name,
    иначе pmax/pmin ниже упадут с ошибкой "unbound axis name". Нужен,
    чтобы agregировать maxabs/isfinite ГЛОБАЛЬНО по всем шардам батча,
    а не только по локальному шарду -- без этого out_specs=P() (replicated)
    либо падает, либо тихо даёт значение произвольного шарда.
    """
    q_sg, k_sg, v_sg, w_sg, b_sg, g_sg = jax.lax.stop_gradient((q, k, v, w, b, g))

    Aqk, Akk = build_chunk_scores_pallas(q_sg, k_sg, b_sg, g_sg, scale)
    A = wy_solve_pallas(Akk)
    w_pseudo, u, kg, qg, _gc_last = recompute_wy_pallas(q_sg, k_sg, v_sg, w_sg, b_sg, g_sg, A)

    out = {}
    for name, val in (
        ("aqk", Aqk), ("akk", Akk), ("a_wy_inverse", A),
        ("w_pseudo", w_pseudo), ("u", u), ("kg", kg), ("qg", qg),
    ):
        maxabs, isfinite = _stat(val)
        if axis_name is not None:
            maxabs = jax.lax.pmax(maxabs, axis_name=axis_name)
            isfinite = jax.lax.pmin(isfinite, axis_name=axis_name)
        out[f"{name}_maxabs"] = maxabs
        out[f"{name}_isfinite"] = isfinite
    return out
