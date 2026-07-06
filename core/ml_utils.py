# ==============================================================================
# 🛠️ ML UTILITIES (Hardware Routing, Memory Management & Reproducibility)
# ==============================================================================
"""
Утиліти машинного навчання.
Відповідають за інтелектуальну маршрутизацію обчислень, очищення відеопам'яті (VRAM)
та строгу фіксацію випадковості для відтворюваності експериментів.
"""

import os
import gc
import random
import platform
import logging
import numpy as np

# Отримуємо логер, який ми попередньо налаштували у __init__.py
logger = logging.getLogger("core.ml_utils")


# ------------------------------------------------------------------------------
# 1. ЗАБЕЗПЕЧЕННЯ БАЗОВОЇ ВІДТВОРЮВАНОСТІ (Base Reproducibility)
# ------------------------------------------------------------------------------
def set_global_seed(seed: int = None) -> None:
    """
    Жорстко фіксує базовий seed для Python та NumPy.
    (PyTorch фіксується окремо в апаратному маршрутизаторі)
    """
    if seed is None:
        seed = int(os.getenv("GLOBAL_SEED", 42))

    # 1. Фіксація стандартного модуля Python
    random.seed(seed)

    # 2. Фіксація хешування вбудованих структур (dict, set)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 3. Фіксація NumPy (використовується в Pandas та Scikit-Learn)
    np.random.seed(seed)
    logger.info(f"🌱 Базовий seed зафіксовано: {seed} (Python, NumPy)")


# ------------------------------------------------------------------------------
# 2. УЛЬТИМАТИВНИЙ МАРШРУТИЗАТОР ОБЧИСЛЮВАЛЬНИХ ПОТУЖНОСТЕЙ
# ------------------------------------------------------------------------------
def get_hardware_config(global_seed: int = None):
    """
    Ультимативний апаратний маршрутизатор PyTorch
    Автоматично шукає GPU. Якщо драйвери відсутні або "биті" — провалюється на CPU
    Підтримує примусовий Override через `.env` (ML_BACKEND=cpu) для Docker-контейнерів
    """
    if global_seed is None:
        global_seed = int(os.getenv("GLOBAL_SEED", 42))

    try:
        import torch
    except ImportError:
        logger.warning("⚠️ PyTorch не встановлено. Повертаю базовий 'cpu'")
        return "cpu", "CPU (No PyTorch)"

    # 🎛️ 0. ПРИМУСОВИЙ OVERRIDE (Для Docker та CI/CD)
    env_backend = os.getenv("ML_BACKEND", "").lower().strip()
    if env_backend == "cpu":
        device = torch.device("cpu")
        device_ui_name = "CPU (Примусово через .env)"
        torch.manual_seed(global_seed)
        logger.info(f"⚙️ Ініціалізовано: {device_ui_name}")
        return device, device_ui_name

    # 🥇 1. NVIDIA (CUDA) або AMD (ROCm)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Архітектурна перевірка: чи це зібрано під AMD HIP/ROCm?
        if getattr(torch.version, "hip", None):
            device_ui_name = "ROCm (AMD Instinct / Radeon)"
        else:
            device_ui_name = f"CUDA ({torch.cuda.get_device_name(0)})"

        torch.cuda.manual_seed_all(global_seed)
        # Вимикаємо евристичну оптимізацію CuDNN для досягнення 100% детермінованості
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 🥈 2. Apple Metal API (MPS)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_ui_name = "MPS (Apple Metal API)"
        torch.mps.manual_seed(global_seed)

    # 🥉 3. Intel (Data Center Max Series / Arc)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        device_ui_name = "XPU (Intel GPU / AI Accelerators)"
        torch.xpu.manual_seed_all(global_seed)

    # 🌐 4. Huawei Ascend (Enterprise Cloud NPU)
    elif hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
        device = torch.device("npu")
        device_ui_name = "NPU (Huawei Ascend Accelerator)"
        torch.npu.manual_seed_all(global_seed)

    # 🧠 5. Meta MTIA (Специфічні акселератори Llama-інфраструктури)
    elif hasattr(torch, "mtia") and getattr(torch.mtia, "is_available", lambda: False)():
        device = torch.device("mtia")
        device_ui_name = "MTIA (Meta Training & Inference Accelerator)"
        torch.mtia.manual_seed_all(global_seed)

    # 🐢 6. АВТОМАТИЧНИЙ ФОЛБЕК НА CPU (Якщо GPU немає або драйвери биті)
    else:
        device = torch.device("cpu")
        arch = platform.machine()
        device_ui_name = f"CPU ({arch})"
        torch.manual_seed(global_seed)

    logger.info(f"🚀 Апаратне прискорення: {device_ui_name}")
    return device, device_ui_name


# ------------------------------------------------------------------------------
# 3. МЕНЕДЖЕР ПАМ'ЯТІ (VRAM / RAM Garbage Collector)
# ------------------------------------------------------------------------------
def clear_vram(device) -> None:
    """Архітектурне звільнення відеопам'яті (VRAM) та оперативної пам'яті (RAM)"""
    gc.collect()

    if device is None or type(device) == str:
        return

    try:
        import torch
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

        logger.debug(f"🧹 VRAM ({device.type.upper()}) та оперативну пам'ять успішно очищено")
    except Exception as e:
        logger.debug(f"🧹 Очищення VRAM пропущено: {e}")


# ------------------------------------------------------------------------------
# 4. ІНСПЕКТОР СЕРЕДОВИЩА (Environment Profiler)
# ------------------------------------------------------------------------------
def log_system_info() -> None:
    """
    Виводить короткий аудит системи під час старту пайплайну
    Допомагає зрозуміти, на якому саме "залізі" запустився контейнер або скрипт
    """
    os_name = platform.system()
    os_release = platform.release()
    py_version = platform.python_version()

    logger.info(f"🖥️  Система: {os_name} {os_release} | Python: {py_version}")

# ------------------------------------------------------------------------------
# 5. МАРШРУТИЗАТОР ДЛЯ ДЕРЕВ РІШЕНЬ (XGBoost / LightGBM)
# ------------------------------------------------------------------------------
def get_boosting_kwargs(device):
    """
    Генерує оптимальні параметри апаратного прискорення для XGBoost та LightGBM
    на основі виявленого PyTorch-пристрою. Відповідає за безпечний fallback
    """
    device_type = getattr(device, "type", "cpu")

    # Базові налаштування (завжди навантажуємо всі ядра процесора через n_jobs=-1)
    xgb_kwargs = {"tree_method": "hist", "n_jobs": -1}
    lgbm_kwargs = {"n_jobs": -1, "verbose": -1}

    # 1. NVIDIA (CUDA) або AMD (ROCm)
    # Згідно з доками, XGBoost працює з ROCm через alias 'cuda' або 'gpu'
    if device_type == "cuda":
        xgb_kwargs["device"] = "cuda"
        lgbm_kwargs["device_type"] = "gpu"

    # 2. Intel GPU (SYCL / XPU)
    # Згідно з XGBoost 2.0+, додано нативну підтримку SYCL
    elif device_type == "xpu":
        xgb_kwargs["device"] = "sycl"
        lgbm_kwargs["device_type"] = "gpu"

    # 3. Apple Metal (MPS) або інші
    # XGBoost/LGBM з коробки вимагають специфічної OpenCL компіляції під Mac
    # Найстабільніший і найбезпечніший варіант — CPU з максимальною багатопоточністю
    else:
        xgb_kwargs["device"] = "cpu"
        lgbm_kwargs["device_type"] = "cpu"

    logger.info(f"🌲 Бустинги (XGB/LGBM) змаршрутизовано на: {xgb_kwargs['device'].upper()} (n_jobs={xgb_kwargs['n_jobs']})")

    return xgb_kwargs, lgbm_kwargs
