"""
wandb_logging.py -- тонкая обёртка над Weights & Biases для train.py.

Следует тому же паттерну, что HF-интеграция в checkpointing.py: пытаемся
достать ключ из Kaggle secrets, если не получилось -- логирование молча
отключается (_HAS_WANDB=False), обучение продолжает работать как раньше,
просто без W&B. Ни один вызов log_* не должен уметь уронить обучение --
все обёрнуты в try/except, т.к. сетевой сбой логирования не повод терять
TPU-сессию.

ФИКС (этот пасс): entity раньше НЕ передавался из train.py (всегда None
через дефолтный параметр функции) -- если resume_id задан и resume="must",
а run изначально создавался под ДРУГИМ entity (например, team-аккаунт
вместо личного, или наоборот), wandb.init() тихо падает с CommError,
ловится в except ниже, и обучение продолжается БЕЗ W&B -- ровно в момент
resume, когда непрерывность графиков нужнее всего, и без явного сигнала
пользователю, что произошло (только один print в консоли Kaggle, которая
не сохраняется). WANDB_ENTITY -- явная константа модуля, единственное
место, которое нужно поменять, если используется team/organization entity
вместо личного аккаунта, залогиненного через WANDB_API_KEY.

ФИКС #2 (этот пасс): добавлен set_summary() -- финальные метрики
(best_train_loss, best_val_loss, итоговый global_step и т.п.) раньше нигде
не фиксировались как wandb.summary, только на графике по step -- неудобно
сравнивать несколько runs в таблице W&B без summary-колонок. Вызывается
один раз в конце main_execution(), перед finish().
"""
from __future__ import annotations

import time

try:
    from kaggle_secrets import UserSecretsClient
    _user_secrets = UserSecretsClient()
    import wandb
    WANDB_API_KEY = _user_secrets.get_secret("WANDB_API_KEY")
    _HAS_WANDB = bool(WANDB_API_KEY)
    if _HAS_WANDB:
        wandb.login(key=WANDB_API_KEY)
        print("[WANDB] ✅ Ключ найден, логирование включено.")
    else:
        raise ImportError("W&B API key пуст")
except ImportError:
    _HAS_WANDB = False
    print("[WARN] W&B недоступен (не установлен пакет).")
except Exception as e:
    _HAS_WANDB = False 
    print(f"[WARN] W&B недоступен, в Kaggle secters добавь API")

# ФИКС: явная константа entity -- поставьте сюда ваш team/organization
# entity, если используете его в W&B. None (по умолчанию) означает личный
# аккаунт, под которым выполнен wandb.login(key=WANDB_API_KEY) выше.
# ВАЖНО: если когда-либо резюмируете run (resume_id задан), entity ЗДЕСЬ
# должен совпадать с entity, под которым run был изначально создан --
# иначе wandb.init(..., resume="must") упадёт с CommError (см. докстринг
# модуля выше).
WANDB_ENTITY = None

_run = None


def init_wandb(project: str, run_name: str, config: dict, resume_id: str | None = None,
                entity: str | None = None):
    """entity: если None (по умолчанию, и именно так вызывается из
    train.py), берётся модульная константа WANDB_ENTITY -- так что для
    большинства случаев менять нужно ТОЛЬКО WANDB_ENTITY выше, а не каждый
    call site."""
    global _run
    if not _HAS_WANDB:
        return None
    entity = entity if entity is not None else WANDB_ENTITY
    try:
        _run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            id=resume_id,
            resume="must" if resume_id else None,   # fail loudly if the run isn't found, not silently create a new one
        )
        print(f"[WANDB] 🚀 Run начат: {_run.name} (id={_run.id}, entity={entity})"
              + (f", resumed from {resume_id}" if resume_id else ""))
        return _run.id
    except Exception as e:
        print(f"[WANDB] ❌ Не удалось инициализировать run: {e}. Продолжаю без W&B.")
        _run = None
        return None


def log_metrics(step: int, metrics: dict):
    """Логирует произвольный словарь метрик на данном шаге. No-op, если
    W&B недоступен или init_wandb не вызывался/упал. Никогда не бросает
    исключение наружу -- сетевой сбой логирования не должен останавливать
    обучение."""
    if _run is None:
        return
    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        print(f"[WANDB] ⚠️ Не удалось залогировать метрики на шаге {step}: {e}")


def set_summary(values: dict):
    """ФИКС (новое): записывает значения в wandb.summary -- видно как
    отдельные, сортируемые/фильтруемые колонки в таблице всех runs
    проекта, а не только точки на графике по step. Вызывать один раз в
    конце обучения (или в любой момент, когда нужно зафиксировать текущий
    "лучший" результат) -- перезаписывает существующие ключи. No-op без
    W&B, никогда не бросает исключение наружу."""
    if _run is None:
        return
    try:
        for k, v in values.items():
            wandb.run.summary[k] = v
    except Exception as e:
        print(f"[WANDB] ⚠️ Не удалось записать summary: {e}")


def log_alert(title: str, text: str, level: str = "WARN"):
    """Отправляет W&B Alert (email/slack, если настроено в аккаунте) --
    используется для событий, которые стоит заметить не листая логи Kaggle:
    auto-stop по non-finite, session-limit graceful stop и т.п."""
    if _run is None:
        return
    try:
        wandb.alert(title=title, text=text,
                    level=wandb.AlertLevel.WARN if level == "WARN" else wandb.AlertLevel.ERROR)
    except Exception as e:
        print(f"[WANDB] ⚠️ Не удалось отправить alert '{title}': {e}")


def finish():
    global _run
    if _run is None:
        return
    try:
        wandb.finish()
        print("[WANDB] Run завершён.")
    except Exception as e:
        print(f"[WANDB] ⚠️ Ошибка при finish(): {e}")
    finally:
        _run = None
