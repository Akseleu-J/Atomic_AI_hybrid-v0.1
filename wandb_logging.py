"""
wandb_logging.py -- тонкая обёртка над Weights & Biases для train.py.

Следует тому же паттерну, что HF-интеграция в checkpointing.py: пытаемся
достать ключ из Kaggle secrets, если не получилось -- логирование молча
отключается (_HAS_WANDB=False), обучение продолжает работать как раньше,
просто без W&B. Ни один вызов log_* не должен уметь уронить обучение --
все обёрнуты в try/except, т.к. сетевой сбой логирования не повод терять
TPU-сессию.
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
    print("[WARN] W&B недоступен (нет kaggle secret WANDB_API_KEY или пакета wandb) "
          "-- обучение продолжится без логирования в W&B.")

_run = None


def init_wandb(project: str, run_name: str, config: dict, resume_id: str | None = None):
    """Инициализирует W&B run. resume_id -- если передан, пытается
    продолжить существующий run (полезно после resume из чекпоинта, чтобы
    графики в W&B не начинались заново с шага 0 при каждом рестарте
    Kaggle-сессии). Если resume_id=None или resume не удался -- создаёт
    новый run. Возвращает run.id (строку) для сохранения в metadata.json
    следующего чекпоинта, или None если W&B недоступен."""
    global _run
    if not _HAS_WANDB:
        return None
    try:
        _run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            id=resume_id,
            resume="allow" if resume_id else None,
        )
        print(f"[WANDB] 🚀 Run начат: {_run.name} (id={_run.id})"
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
