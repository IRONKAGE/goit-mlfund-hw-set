# ==============================================================================
# 🧠 CORE PACKAGE INITIALIZATION (MLOps Infrastructure)
# ==============================================================================
"""
Ядро інфраструктури проекту
Цей пакет об'єднує ETL-пайплайни (Data Engineering) та ML-утиліти (Hardware & Seeds)
Імпортування з цього модуля гарантує єдину точку доступу до всіх спільних інструментів
"""

import os
import logging
import warnings
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 0. МЕТАДАНІ ПАКЕТУ
# ------------------------------------------------------------------------------
__version__ = "1.0.0"
__author__ = "IRONKAGE (Oleh Hatsenko)"

# ------------------------------------------------------------------------------
# 1. ЗАВАНТАЖЕННЯ СЕКРЕТІВ (.env)
# ------------------------------------------------------------------------------
# Автоматично підтягуємо змінні середовища при першому ж імпорті ядра
# override=True гарантує, що локальний .env має пріоритет
load_dotenv(override=True)

# =======================================================
# 2. НАЛАШТУВАННЯ MLFLOW ТРЕКІНГУ (Global Infrastructure)
# =======================================================
try:
    import mlflow

    # База даних спільна для всіх завдань (один файл mlruns.db на весь проект)
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlruns.db")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # ⚠️ УВАГА: mlflow.set_experiment() -  кожен Marimo-ноутбук має самостійно задавати назву свого експерименту
    # у комірці ініціалізації, щоб розділяти метрики різних домашніх завдань

except ImportError:
    # Якщо пакет використовується на Production сервері без MLflow — просто ігноруємо
    pass

# ------------------------------------------------------------------------------
# 3. СИСТЕМА ЛОГУВАННЯ ТА ПОПЕРЕДЖЕНЬ (Global Logging & Warnings)
# ------------------------------------------------------------------------------
# Глушимо набридливі UserWarning та FutureWarning від Pandas/Sklearn у ноутбуках,
# щоб зберегти чистоту виводу в Marimo
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Налаштовуємо єдиний стандарт логування для всього проекту
# Він виводитиме час, рівень важливості та назву модуля, що дуже допоможе при дебаггінгу
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Створюємо екземпляр логера, який можна буде імпортувати
logger = logging.getLogger("core")

# Вимикаємо зайвий спам від сторонніх C++ бібліотек під час рендерингу
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)


# ------------------------------------------------------------------------------
# 4. ФАСАД API (Package API Export)
# ------------------------------------------------------------------------------
# Імпортуємо ключові функції з внутрішніх файлів, щоб вони були доступні
# безпосередньо з пакету `core`. Використовуємо безпечний імпорт (try-except)

try:
    from .ml_utils import (
        set_global_seed,
        get_hardware_config,
        clear_vram,
        log_system_info,
        get_boosting_kwargs
    )
    from .etl_core import SecureDownloader, smart_read_csv

    # Спеціальна змінна __all__ жорстко контролює,
    # що саме буде експортовано при виклику `from core import *`
    __all__ = [
        "set_global_seed",
        "get_hardware_config",
        "clear_vram",
        "log_system_info",
        "get_boosting_kwargs",
        "SecureDownloader",
        "smart_read_csv",
        "logger"
    ]

except ImportError as e:
    logger.warning(f"Модулі ядра ще знаходяться в стадії розробки. Відсутній імпорт: {e}")
    # Експортуємо хоча б логер, щоб не ламати скрипти, які на нього чекають
    __all__ = ["logger"]
