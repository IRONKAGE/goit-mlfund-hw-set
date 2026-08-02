import pandas as pd
import numpy as np
import os

# Ініціалізація глобального seed з урахуванням змінних середовища (.env)
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", 42))

def get_concrete_mock(save_path: str):
    """
    Fallback-адаптер: Створює синтетичний набір даних міцності бетону (Concrete Strength)
    Використовується для гарантії безперебійної роботи пайплайну (Zero-Trust Data Source)
    """
    np.random.seed(GLOBAL_SEED)
    n = 1030  # Розмір оригінального набору даних UCI

    data = {
        "Cement": np.random.uniform(102.0, 540.0, n),
        "BlastFurnaceSlag": np.random.choice([0.0, np.random.uniform(20, 359)], n), # Часто дорівнює 0
        "FlyAsh": np.random.choice([0.0, np.random.uniform(20, 200)], n),           # Часто дорівнює 0
        "Water": np.random.uniform(121.0, 247.0, n),
        "Superplasticizer": np.random.choice([0.0, np.random.uniform(2, 32)], n),
        "CoarseAggregate": np.random.uniform(801.0, 1145.0, n),
        "FineAggregate": np.random.uniform(594.0, 992.0, n),
        "Age": np.random.choice([1, 3, 7, 14, 28, 56, 90, 180, 365], n) # Стандартні дні випробувань
    }

    df = pd.DataFrame(data)

    # 🧠 Фізико-хімічна нелінійна функція (щоб XGBoost та SHAP мали що вивчати)
    # Міцність базується на Водоцементному співвідношенні (W/C ratio) та логарифмі віку
    water_cement_ratio = df["Cement"] / df["Water"]

    df["csMPa"] = (
        water_cement_ratio * 18.0 +               # Цемент - головний драйвер
        df["Superplasticizer"] * 0.6 +            # Пластифікатор трохи допомагає
        np.log1p(df["Age"]) * 5.0 +               # Логарифмічний набір міцності з часом
        (df["BlastFurnaceSlag"] > 0) * 3.0 -      # Бонус за шлак
        np.random.normal(0, 4.0, n)               # Природний шум/похибка лабораторії
    )

    # Захист: міцність не може бути від'ємною або космічно великою
    df["csMPa"] = np.clip(df["csMPa"], a_min=2.0, a_max=85.0).round(2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df
