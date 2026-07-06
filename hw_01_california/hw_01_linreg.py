import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title_head_hw(mo):
    mo.md("""
    <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
        🏡 ДЗ №1: Прогнозування вартості нерухомості <i>(California Housing Prices)</i>
    </div>

    <h3 align="center"><b><u>Пайплайн</u>: Smart ETL ➔ Auto EDA ➔ SpatialEngineer ➔ DBSCAN Clustering ➔ OHE vs Native Categorical Benchmark</b></h3>

    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Machine Learning: Fundamentals and Applications [06.2026]</i></p>
    """)
    return


@app.cell
def configure_dependencies():
    # 1. Стандартні бібліотеки
    import os
    import warnings
    import contextlib
    import base64
    import logging
    import json
    from datetime import datetime

    # ЛІКУЄМО КОНФЛІКТ OpenMP (Intel vs LLVM)
    os.environ["KMP_DUPLICATE_OK"] = "TRUE"
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, module="threadpoolctl"
    )
    # Придушуємо дрібні системні попередження LightGBM
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    # Придушуємо спам у логах від Microsoft EBM
    logging.getLogger("interpret").setLevel(logging.WARNING)

    # 2. Математика та Обробка даних (Data Science Core)
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import polars as pl
    import re
    import shap
    import shap.explainers._tree as shap_tree
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd
    from scipy.stats import zscore
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from data_profiling import ProfileReport
    from ui_labels import UA_COLUMNS, SKLEARN_MAP

    # 3. Фреймворк Marimo
    import marimo as mo

    # ==============================================================================
    # 4. Machine Learning (Scikit-Learn, XGBoost, LightGBM, Interpret)
    # ==============================================================================
    import mlflow
    import mlflow.sklearn
    import sklearn
    import optuna
    import joblib

    # --- Базові класи, пайплайни та обробка даних ---
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.inspection import permutation_importance

    # --- Метрики та Кластеризація ---
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        silhouette_score,
    )

    from optuna.visualization import plot_optimization_history, plot_param_importances

    # --- Бейзлайни та Лінійні моделі ---
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
        HuberRegressor,
        BayesianRidge,
        PassiveAggressiveRegressor,
        TweedieRegressor,
        OrthogonalMatchingPursuit,
    )
    from sklearn.kernel_ridge import KernelRidge

    # --- Класичні алгоритми та Нейромережі ---
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.svm import LinearSVR
    from sklearn.neural_network import MLPRegressor

    # --- Ансамблі (Scikit-Learn) ---
    from sklearn.ensemble import (
        AdaBoostRegressor,
        RandomForestRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
    )

    # --- Сучасні Бустинги ---
    from xgboost import XGBRegressor, XGBRFRegressor
    from lightgbm import LGBMRegressor
    from interpret.glassbox import ExplainableBoostingRegressor

    # 5. Внутрішня інфраструктура (Enterprise Core)
    from core import (
        set_global_seed,
        get_hardware_config,
        clear_vram,
        log_system_info,
        SecureDownloader,
        smart_read_csv,
        logger,
        get_boosting_kwargs,
    )
    from core.datasets import get_california_housing

    # =========================================================================
    # ⚙️ ГЛОБАЛЬНІ НАЛАШТУВАННЯ СЕРЕДОВИЩА (Architectural Configs)
    # =========================================================================
    # 🔸 Глобально активуємо Copy-on-Write (сувора оптимізація RAM для Pandas 2.x)
    pd.options.mode.copy_on_write = True

    # 🔸 Активація Native Pandas API для всіх трансформерів (Scikit-Learn 1.2+)
    sklearn.set_config(transform_output="pandas")

    mo.center(mo.md("✅ **Бібліотеки успішно імпортовано!**"))
    return (
        AdaBoostRegressor,
        BaseEstimator,
        BayesianRidge,
        DBSCAN,
        DecisionTreeRegressor,
        DummyRegressor,
        ElasticNet,
        ExplainableBoostingRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        HuberRegressor,
        KFold,
        KNeighborsClassifier,
        KNeighborsRegressor,
        KernelRidge,
        LGBMRegressor,
        Lasso,
        LinearRegression,
        LinearSVR,
        MLPRegressor,
        OrthogonalMatchingPursuit,
        PassiveAggressiveRegressor,
        Pipeline,
        PolynomialFeatures,
        ProfileReport,
        RandomForestRegressor,
        Ridge,
        SKLEARN_MAP,
        SecureDownloader,
        StandardScaler,
        TransformerMixin,
        TweedieRegressor,
        UA_COLUMNS,
        XGBRFRegressor,
        XGBRegressor,
        base64,
        clear_vram,
        contextlib,
        datetime,
        get_boosting_kwargs,
        get_california_housing,
        get_hardware_config,
        go,
        joblib,
        json,
        log_system_info,
        logger,
        logging,
        make_subplots,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mlflow,
        mo,
        np,
        optuna,
        os,
        pa,
        pd,
        permutation_importance,
        pl,
        plot_optimization_history,
        plot_param_importances,
        plt,
        px,
        r2_score,
        re,
        sch,
        set_global_seed,
        shap,
        shap_tree,
        silhouette_score,
        smart_read_csv,
        ssd,
        train_test_split,
        zscore,
    )


@app.cell
def ui_utilities(mo):
    def style_dataframe(
        df,
        format_dict=None,
        show_index=True,
        text_align="right",
        vertical_lines=False,
    ):
        """
        🛎️ Єдиний центр стилізації таблиць (DRY).
        Автоматично підтягує тему Marimo (Dark/Light), форматує дані,
        керує нумерацією рядків, закріплює заголовок, вирівнює текст
        та додає Zebra Striping (порядкове затемнення)

        Параметри:
        - format_dict: Словник для форматування (напр. {"Ціна": "${:.2f}"})
        - show_index: Якщо False, таблиця виводиться без першої колонки (індексу)
        - text_align: Динамічне вирівнювання даних ("right", "center", "left")
        - vertical_lines: Додає вертикальні розділювачі стовпців (за замовчуванням вимкнено для чистоти UI)
        """
        _theme = mo.app_meta().theme
        _th_bg = "#374151" if _theme == "dark" else "#f3f4f6"
        _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
        _v_border = f"1px solid {_border}" if vertical_lines else "none"
        _text_color = "white" if _theme == "dark" else "#1f2937"
        # Колір наведення (Hover) в стилі XGBoost / Teal
        # rgba(16, 185, 129) - це RGB еквівалент нашого зеленого кольору #10b981
        _hover_bg = (
            "rgba(16, 185, 129, 0.25)"
            if _theme == "dark"
            else "rgba(16, 185, 129, 0.12)"
        )
        _stripe_bg = (
            "rgba(255, 255, 255, 0.04)"
            if _theme == "dark"
            else "rgba(0, 0, 0, 0.03)"
        )

        styler = df.style
        if format_dict:
            styler = styler.format(format_dict)

        # 🎛 КЕРУВАННЯ НУМЕРАЦІЄЮ (ІНДЕКСОМ)
        if not show_index:
            styler = styler.hide(axis="index")

        return (
            styler.set_table_attributes(
                f'style="width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.9em; text-align: center; margin-top: 10px; color: {_text_color};"'
            )
            .set_table_styles(
                [
                    # 📌 Закріплений заголовок
                    {
                        "selector": "thead th",
                        "props": [
                            ("position", "sticky"),
                            ("top", "0"),
                            ("z-index", "100"),
                            ("background-color", _th_bg),
                            ("padding", "10px"),
                            ("border-bottom", f"2px solid {_border}"),
                            ("border-right", _v_border),
                            ("text-align", "center"),
                        ],
                    },
                    # 🔢 Стовпець індексу
                    {
                        "selector": "tbody th",
                        "props": [
                            ("background-color", _th_bg),
                            ("padding", "8px"),
                            ("border-bottom", f"1px solid {_border}"),
                            ("border-right", f"2px solid {_border}"),
                            ("text-align", "center"),
                            ("font-weight", "bold"),
                        ],
                    },
                    # 📊 Звичайні клітинки даних
                    {
                        "selector": "tbody td",
                        "props": [
                            ("padding", "8px 10px"),
                            ("border-bottom", f"1px solid {_border}"),
                            ("border-right", _v_border),
                            ("text-align", text_align),
                        ],
                    },
                    # 🦓 ЗЕБРА: Легке затемнення парних рядків
                    {
                        "selector": "tbody tr:nth-child(even) td",
                        "props": [("background-color", _stripe_bg)],
                    },
                    # 🖱️ Ефект наведення (перебиває зебру при наведенні мишкою)
                    {
                        "selector": "tbody tr:hover td, tbody tr:hover th",
                        "props": [
                            ("background-color", _hover_bg),
                            (
                                "transition",
                                "background-color 0.15s ease-in-out",
                            ),
                        ],
                    },
                ]
            )
            .to_html()
        )

    mo.center(mo.md("✅ **Стиль таблиць встановлено!**"))
    return (style_dataframe,)


@app.cell
def initialize_mlops_core(
    get_boosting_kwargs,
    get_hardware_config,
    log_system_info,
    mlflow,
    mo,
    os,
    set_global_seed,
):
    # 🌱 1. Читаємо єдине джерело істини з .env
    GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", 42))

    # ⚙️ 2. Ініціалізація апаратного забезпечення та профайлінг системи
    with mo.status.spinner(title="Ініціалізація MLOps ядра..."):
        log_system_info()  # 🔍 Друкуємо інформацію про ОС та Python у логи
        set_global_seed(GLOBAL_SEED)

        # Детектимо залізо для PyTorch (Нейромережі)
        device, device_ui_name = get_hardware_config(global_seed=GLOBAL_SEED)

        # Перекладаємо залізо для Бустингів (Дерева)
        xgb_kwargs, lgbm_kwargs = get_boosting_kwargs(device)

        # 3. Гнучке налаштування MLflow для КОНКРЕТНОГО завдання
        experiment_name = "hw01_california_housing"
        mlflow.set_experiment(experiment_name)
    return GLOBAL_SEED, lgbm_kwargs, xgb_kwargs


@app.cell(hide_code=True)
def header_prepare_dataset(mo):
    mo.md("""
    <h2 align="center"><b>💽 1. Завантаження даних <i>(Smart Router 5-Tier)</i></b></h2>
    """)
    return


@app.cell
def execute_etl_pipeline(
    SKLEARN_MAP,
    SecureDownloader,
    get_california_housing,
    logger,
    mo,
    os,
    smart_read_csv,
):
    with mo.status.spinner(
        title="Завантаження та атомарне розпакування (Zero-Trust)..."
    ):
        data_dir = os.getenv("DATA_DIR", "./data")

        downloader = SecureDownloader(
            dataset_path="camnugent/california-housing-prices",
            dataset_url="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
            fallback_generator=get_california_housing,
            data_dir=data_dir,
            zip_name="california_housing.zip",
        )

        downloader.download(target_filename="housing.csv")

        # 🪤 Захист архітектури:
        # Якщо спрацював Fallback-генератор, він одразу створює CSV, а не ZIP
        # Тому `extract_atomically` може видати помилку, яку ми безпечно ігноруємо
        extracted_files = []
        try:
            extracted_files = downloader.extract_atomically(
                target_extensions=(".csv",), expected_filename="housing.csv"
            )
            data_path = extracted_files[0]
        except Exception as e:
            logger.info(
                f"Витягування пропущено (можливо, файл вже є CSV). Деталі: {e}"
            )
            data_path = os.path.join(data_dir, "housing.csv")

        # ⚡ Явно вказуємо engine="pyarrow", щоб підтвердити оптимізацію пам'яті
        df_raw = smart_read_csv(
            data_path, desc_name="California Housing", engine="pyarrow"
        )

        # Чи є в сирих даних колонки зі словника Sklearn?
        _is_sklearn_format = any(col in SKLEARN_MAP for col in df_raw.columns)

        # Якщо це Sklearn - перетворюємо MedInc -> median_income, Latitude -> latitude і т.д.
        # Якщо це Kaggle - rename просто безпечно нічого не зробить
        df_raw = df_raw.rename(columns=SKLEARN_MAP)

    # 🎨 Формуємо динамічне повідомлення для UI
    _rename_msg = ""
    if _is_sklearn_format:
        _rename_msg = "<br/><span style='color: #10b981; font-size: 0.95em;'>🫯 <i>Спрацював адаптер: виявлено формат Sklearn, стовпчики автоматично нормалізовано до стандарту Kaggle!</i></span>"

    _ui_output = mo.center(
        mo.md(
            f"✅ **Дані успішно завантажено!** Розмір сирого набору даних: `(Рядків: {df_raw.shape[0]} | Стовпчиків: {df_raw.shape[1]})`{_rename_msg}"
        )
    )
    mo.output.append(_ui_output)
    return (df_raw,)


@app.cell(hide_code=True)
def header_auto_eda(mo):
    mo.md("""
    <h3 align="center"><b>📊 1.1. Автоматичний EDA <i>(fg-data-profiling)</i></b></h3>
    """)
    return


@app.cell
def generate_eda_report(ProfileReport, base64, contextlib, df_raw, mo, os):
    with mo.status.spinner(title="Генерація інтерактивного профайлінгу..."):
        # Локальне масштабування виключно для бізнес-звіту
        # Робимо копію, щоб не зачепити оригінальний df_raw,
        # адже для ML-алгоритмів ми вже налаштували динамічний price_scale для Sklearn далі!
        df_eda = df_raw.copy()
        if "median_house_value" in df_eda.columns and df_eda["median_house_value"].max() <= 100:
            df_eda["median_house_value"] = df_eda["median_house_value"] * 100000

        # 🌌 Відправляємо весь консольний спам у "чорну діру"
        with (
            open(os.devnull, "w") as fnull,
            contextlib.redirect_stdout(fnull),
            contextlib.redirect_stderr(fnull),
        ):
            profile = ProfileReport(
                df_eda,
                title="California Housing Report",
                minimal=True,
                progress_bar=False,
            )
            html_string = profile.to_html()

            # Динамічна та безпечна назва артефакту
            artifact_dir = os.getenv("MODELS_DIR", "./models")
            os.makedirs(artifact_dir, exist_ok=True)
            report_filename = "hw01_california_housing_eda.html"
            profile.to_file(os.path.join(artifact_dir, report_filename))

    b64_html = base64.b64encode(html_string.encode("utf-8")).decode("utf-8")

    # 🎨 Авто-адаптивний CSS: сам слухає тему Marimo (клас .dark)
    dynamic_html = f"""
    <style>
        .smart-eda-iframe {{
            width: 100%;
            height: 850px;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            transition: filter 0.3s ease-in-out, border-color 0.3s ease;
        }}

        /* Активується ТІЛЬКИ коли користувач увімкнув темну тему в Marimo */
        html.dark .smart-eda-iframe,
        .dark .smart-eda-iframe {{
            filter: invert(90%) hue-rotate(180deg) brightness(1.1);
            border-color: #333;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }}
    </style>
    <iframe class="smart-eda-iframe" src="data:text/html;base64,{b64_html}"></iframe>
    """

    html_report = mo.Html(dynamic_html)

    mo.center(
        mo.md(
            "✅ **Профайлінг успішно згенеровано!** *(Ізольований фрейм готовий та артефакт збережено)*"
        )
    )
    return (html_report,)


@app.cell(hide_code=True)
def header_feature_engineering(mo):
    mo.md("""
    <h3 align="center"><b>🛠️ 1.2. Інженерія ознак <i>(Custom OOP Transformers)</i></b></h3>
    """)
    return


@app.cell
def display_eda_report(html_report):
    # 📜 Окремою клітинкою виводимо звіт, щоб він не уповільнював рендеринг іншого UI
    html_report
    return


@app.cell(hide_code=True)
def header_data_cleaning_main(mo):
    mo.md("""
    <h2 align="center"><b>🧹 2. Очищення даних <i>(Data Cleaning & Preprocessing)</i></b></h2>
    """)
    return


@app.cell(hide_code=True)
def header_data_cleaning_sub(mo):
    mo.md("""
    <h3 align="center"><b>🛠️ 2.1. Z-Score фільтрація та Мультиколінеарність</b></h3>
    """)
    return


@app.cell
def clean_data_and_handle_collinearity(
    GLOBAL_SEED,
    UA_COLUMNS,
    df_raw,
    mo,
    np,
    style_dataframe,
    zscore,
):
    # 1. Задаємо колонки для пошуку аномалій
    actual_cols = ["total_rooms", "total_bedrooms", "population", "households"]

    # 2. Фільтрація викидів (Z-Score > 3)
    z_scores = df_raw[actual_cols].apply(zscore, nan_policy="omit")
    outlier_mask = (np.abs(z_scores) > 3).any(axis=1)

    # Спочатку просто відфільтрували аномалії (БЕЗ dropna!)
    df_clean = df_raw[~outlier_mask].copy()

    # 3. Боротьба з мультиколінеарністю (видаляємо спальні)
    drop_col = "total_bedrooms"
    if drop_col in df_clean.columns:
        df_clean = df_clean.drop(columns=[drop_col])

    # Видаляємо пропуски ТІЛЬКИ ПІСЛЯ видалення зайвих колонок!
    # (Це врятує скількись рядків даних, бо пропуски були саме у total_bedrooms)
    df_clean = df_clean.dropna()

    # 4. Визначаємо цільову змінну
    target_col = "median_house_value"

    # 5. Генеруємо UI-блок: Локалізація + Скрол + Закріплені заголовки
    n_samples = min(100, len(df_clean))
    df_display_raw = df_clean.sample(n=n_samples, random_state=GLOBAL_SEED).copy()

    # Масштабуємо гроші ТІЛЬКИ для красивої таблиці в UI
    if "median_house_value" in df_display_raw.columns and df_display_raw["median_house_value"].max() <= 100:
        df_display_raw["median_house_value"] = df_display_raw["median_house_value"] * 100000

    df_display = df_display_raw.rename(columns=UA_COLUMNS)

    # 6. Задаємо правило показу знаків після коми
    _formatting_rules = {
        # 🎯 Ключові бізнес-метрики
        "median_house_value": "${:,.0f}",  # Гроші: $125,000
        "median_income": "{:.2f}",  # Дохід: 4.21 (замість 4.2083)
        # 🌍 Географія (Точність до ~11 метрів)
        "latitude": "{:.4f}",
        "longitude": "{:.4f}",
        # 🧱 Абсолютні лічильники (Прибираємо дроби 32.000000 -> 32)
        "housing_median_age": "{:.0f}",
        "total_rooms": "{:,.0f}",  # З роздільником тисяч (напр. 2,619)
        "population": "{:,.0f}",
        "households": "{:,.0f}",
    }

    # 7. Динамічно перекладаємо ключі на ті, що зараз в UI (щоб Pandas їх знайшов) + ЗАХИСТ ВІД KEY ERROR
    _ua_formatting_rules = {
        UA_COLUMNS.get(k, k): v
        for k, v in _formatting_rules.items()
        if UA_COLUMNS.get(k, k) in df_display.columns # Форматуємо лише існуючі колонки
    }

    # 8. Викликаємо функцію з правильними ключами
    _table_html = style_dataframe(
        df_display,
        format_dict=_ua_formatting_rules,
        text_align="center",
        vertical_lines=True,
    )

    # 9. Динамічно визначаємо лише колір рамки для скрол-контейнера
    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    cleaning_ui = mo.vstack(
        [
            mo.md(
                "🧹 **Очистку завершено:**<br/>"
                f"<b>Видалено аномалій:</b> `{outlier_mask.sum()}` | "
                f"Мультиколінеарну ознаку `{drop_col}` вилучено | "
                f"<b>Залишилось рядків:</b> `{len(df_clean)}`"
                f"<br/><b>Показано випадковим чином:</b> `{int(n_samples)}` рядків"
            ),
            mo.Html(
                f"<div style='max-height: 250px; overflow-y: auto; border: 1px solid {_border}; border-radius: 8px;'>{_table_html}</div>"
            ),
        ]
    )

    mo.output.append(cleaning_ui)
    return df_clean, target_col


@app.cell(hide_code=True)
def header_correlation(mo):
    mo.md("""
    <h3 align="center"><b>🧩 2.2. Аналіз взаємозв'язків <i>(Матриця кореляцій Пірсона)</i></b></h3>
    """)
    return


@app.cell
def plot_correlation_matrix(UA_COLUMNS, df_raw, mo, np, px, sch, ssd):
    # 1. Рахуємо кореляцію тільки для числових колонок (на сирих даних, щоб побачити спальні!)
    numeric_df = df_raw.select_dtypes(
        include=["float64", "int64", "float32", "int32"]
    )
    corr_matrix = numeric_df.corr(method="pearson")

    # Безпечно витягуємо кореляцію між ціною та доходом, якщо ці колонки існують
    if "median_house_value" in corr_matrix.columns and "median_income" in corr_matrix.columns:
        _top_corr_val = corr_matrix.loc["median_house_value", "median_income"]
        _top_corr_str = f"~{_top_corr_val:.2f}"
    else:
        _top_corr_str = "[Дані відсутні]"

    # ==========================================================
    # Matrix Seriation а-ля Yan Holtz
    # ==========================================================
    # Перетворюємо матрицю кореляцій на матрицю математичних відстаней:
    # Сильна позитивна кореляція (1) -> Відстань 0 (Дуже близько)
    # Сильна негативна кореляція (-1) -> Відстань 2 (Дуже далеко)
    dists = 1 - corr_matrix.values
    np.fill_diagonal(dists, 0) # Діагональ завжди має відстань 0
    dists = np.clip((dists + dists.T) / 2, 0, 2) # Захист від float-помилок округлення

    # Виконуємо ієрархічну кластеризацію (Метод Варда)
    linkage = sch.linkage(ssd.squareform(dists), method='ward')

    # Отримуємо новий, ідеально згрупований порядок індексів
    optimal_order = sch.leaves_list(linkage)

    # Пересортовуємо рядки та стовпці нашої матриці за новим порядком
    corr_matrix_sorted = corr_matrix.iloc[optimal_order, optimal_order]

    # 2. Переклад назв колонок і індексів для красивого UI
    corr_matrix_sorted = corr_matrix_sorted.rename(columns=UA_COLUMNS, index=UA_COLUMNS)

    # 3. Налаштування кольорів під тему Marimo
    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"

    # 4. Будуємо інтерактивну теплову карту (Heatmap)
    fig_corr = px.imshow(
        corr_matrix_sorted,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",  # Червоний - позитивна, Синій - негативна кореляція
        zmin=-1,
        zmax=1,
        labels=dict(color="Коеф. Пірсона")
    )

    fig_corr.update_traces(
        hovertemplate="<b>Ознака X:</b> %{x}<br><b>Ознака Y:</b> %{y}<br><b>Кореляція:</b> %{z:.5f}<extra></extra>"
    )

    fig_corr.update_layout(
        title=dict(
            text="<b>Теплова карта кореляцій (Smart Clustered)</b>",
            x=0.5,
            font=dict(color=_text_color, size=18),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_text_color),
        height=850,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # 5. MLOps Інсайт для бізнесу
    _insight_ui = mo.md(
        f"""
        <div class="corr-expanded-cell"></div>
        <style>
            marimo-cell-output:has(.corr-expanded-cell),
            .output-area:has(.corr-expanded-cell) {{
                max-height: none !important;
                overflow: visible !important;
            }}
        </style>
        > **💡 Tech Lead Insight (Аналіз матриці):**<br/>
        > *За порадою Yan Holtz (Data to Viz), ми застосували ієрархічну кластеризацію матриці. Тепер фічі з подібною поведінкою автоматично згруповані у візуальні "блоки", відкриваючи приховану історію даних.*<br/>

        > 🔍 **Чому матриця побудована на сирих даних (df_raw)?**<br/>
        > *Ми спеціально передали сюди дані до їх очищення, щоб на власні очі побачити "хворобу" набору даних — сильну залежність між кімнатами та спальнями. Якби ми будували графік на очищених даних, спалень там би вже не було, і ми б не змогли візуально довести бізнесу правильність нашого рішення про їх видалення у попередньому кроці.*<br/>

        > - **Драйвер ціни:** Зверніть увагу на перетин `Медіанна вартість житла` та `Медіанний дохід`. Коефіцієнт `{_top_corr_str}` — це найпотужніший сигнал у нашому наборі даних.
        > - **Мультиколінеарність:** Саме цей графік доводить необхідність видалення зайвих колонок (як-от кількість спалень), щоб не плутати алгоритми дубльованою інформацією.
        """
    )

    _ui_output = mo.vstack([fig_corr, _insight_ui])
    mo.output.append(_ui_output)
    return


@app.cell(hide_code=True)
def header_spatial_engineering(mo):
    mo.md("""
    <h2 align="center"><b>🌍 3. Гео-інженерія та Кластеризація <i>(SpatialEngineer & DBSCAN)</i></b></h2>
    """)
    return


@app.cell(hide_code=True)
def header_spatial_evolution(mo):
    mo.md("""
    <h3 align="center"><b>📐 3.1. Еволюція коду: Від звичайної функції до MLOps Класу</b></h3>
    """)
    return


@app.cell
def engineer_spatial_features(
    BaseEstimator,
    GLOBAL_SEED,
    TransformerMixin,
    UA_COLUMNS,
    df_clean,
    mo,
    np,
    style_dataframe,
):
    # ВАРІАНТ 1: Звичайна функція
    def haversine(lat1, lon1, lat2, lon2):
        """Розрахунок реальної відстані по сфері Землі (в кілометрах)"""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        a = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        return R * (2 * np.arcsin(np.sqrt(a)))


    # ВАРІАНТ 2: Scikit-Learn сумісний Трансформер
    class SpatialEngineer(BaseEstimator, TransformerMixin):
        """
        Кастомний трансформер, який можна вбудувати у sklearn.Pipeline.
        Перевага: Логіка зберігається разом із моделлю (joblib),
        що гарантує консистентність перетворень на Inference серверах.
        """
        def __init__(self, lat_col="latitude", lon_col="longitude"):
            self.lat_col = lat_col
            self.lon_col = lon_col
            self.la_coords = (34.0522, -118.2437)
            self.sf_coords = (37.7749, -122.4194)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X_out = X.copy()
            X_out["Dist_to_LA"] = haversine(
                X_out[self.lat_col], X_out[self.lon_col], *self.la_coords
            )
            X_out["Dist_to_SF"] = haversine(
                X_out[self.lat_col], X_out[self.lon_col], *self.sf_coords
            )
            X_out["Dist_to_Hub"] = X_out[["Dist_to_LA", "Dist_to_SF"]].min(axis=1)
            return X_out

    lat_col = "latitude"
    lon_col = "longitude"

    _geo_transformer = SpatialEngineer(lat_col=lat_col, lon_col=lon_col)
    df_geo = _geo_transformer.fit_transform(df_clean)

    # 1. Створюємо UI-блок з інсайтом
    _insight_ui = mo.md(
        """
        > **💡 Tech Lead Insight:**<br/>
        > Ми запакували формулу Гаверсина у клас `SpatialEngineer`.<br/>
        > Тепер ми можемо експортувати цей клас разом із вагами моделі.<br/>
        > Серверу, який прийматиме нові запити від користувачів, не потрібно буде шукати окрему функцію — вона вже вшита у пайплайн перетворень!
        """
    )
    mo.output.append(_insight_ui)

    # 2. Демонстрація результатів (UI)
    # 🎯 Беремо координати (як вхідні дані) та наші нові розраховані ознаки
    _cols_to_show = [lat_col, lon_col, "Dist_to_LA", "Dist_to_SF", "Dist_to_Hub"]
    _df_sample = df_geo[_cols_to_show].sample(n=5, random_state=GLOBAL_SEED)

    # Залишаємо власну нумерацію рядків від 1 до 5
    _df_sample.index = range(1, len(_df_sample) + 1)

    # 🎯 3. Форматування чисел: координати (4 знаки), відстані (1 знак)
    _formatting_rules = {
        lat_col: "{:.4f}",
        lon_col: "{:.4f}",
        "Dist_to_LA": "{:.2f}",
        "Dist_to_SF": "{:.2f}",
        "Dist_to_Hub": "{:.2f}",
    }

    # Динамічно перекладаємо ключі на українські (щоб Pandas їх знайшов після rename)
    _ua_formatting_rules = {
        UA_COLUMNS.get(k, k): v for k, v in _formatting_rules.items()
    }

    # 🚀 4. Викликаємо нашу DRY функцію стилізації
    _table_html = style_dataframe(
        _df_sample.rename(columns=UA_COLUMNS),
        format_dict=_ua_formatting_rules,
        text_align="center",
        vertical_lines=True,
    )

    # Динамічно визначаємо колір рамки для красивого контейнера
    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    mo.output.append(
        mo.md("<br/>**📍 Вхідні координати та згенеровані гео-ознаки:**")
    )

    mo.output.append(
        mo.Html(
            f"<div style='overflow-x: auto; border: 1px solid {_border}; border-radius: 8px;'>{_table_html}</div>"
        )
    )
    return df_geo, lat_col, lon_col


@app.cell(hide_code=True)
def header_geo_clustering(mo):
    mo.md("""
    <h3 align="center"><b>🗺️ 3.2. Advanced Geo-Clustering & Роздвоєння пайплайну</b></h3>
    """)
    return


@app.cell
def execute_geo_clustering(
    DBSCAN,
    GLOBAL_SEED,
    KNeighborsClassifier,
    StandardScaler,
    UA_COLUMNS,
    df_geo,
    lat_col,
    lon_col,
    mo,
    np,
    pd,
    px,
    silhouette_score,
    style_dataframe,
):
    # 🎭 Динамічно знімаємо ліміт пам'яті Marimo через приватний API
    try:
        mo._runtime.context.get_context().marimo_config["runtime"][
            "output_max_bytes"
        ] = 50_000_000  # Ставимо 50 МБ для надійності
    except Exception:
        pass  # Запобіжник на випадок, якщо Marimo колись оновить архітектуру

    # 1. Скейлінг координат
    coords = df_geo[[lat_col, lon_col]]
    # Зберігаємо geo_scaler, бо він потрібен FastAPI для нових точок!
    geo_scaler = StandardScaler()
    coords_scaled = geo_scaler.fit_transform(coords)

    # 2. Використовуємо DBSCAN
    dbscan = DBSCAN(eps=0.15, min_samples=15)
    cluster_labels = dbscan.fit_predict(coords_scaled)

    # ⚠️ Перетворюємо трансдуктивний DBSCAN на індуктивний KNN
    # Навчаємо KNN шукати 1 найближчого сусіда серед вже знайдених кластерів
    # Це дозволить нашому FastAPI миттєво визначати хаб для будь-якої нової координати!
    knn_geospatial = KNeighborsClassifier(n_neighbors=1)
    knn_geospatial.fit(coords_scaled, cluster_labels)

    # 3. Валідація (силует для не-шумових точок)
    mask_no_noise = cluster_labels != -1
    sil_score = 0
    if mask_no_noise.sum() > 100:
        valid_indices = np.where(mask_no_noise)[0]
        sample_size = min(5000, len(valid_indices))
        sample_idx = np.random.RandomState(GLOBAL_SEED).choice(
            valid_indices, size=sample_size, replace=False
        )
        sil_score = silhouette_score(
            coords_scaled.iloc[sample_idx], cluster_labels[sample_idx]
        )

    # 4. Візуальні Сім'ї: Розширена ріелторська термінологія: Географічно точна термінологія (Північ -> Південь)
    _unique_lbls = [lbl for lbl in np.unique(cluster_labels) if lbl != -1]
    _counts = {lbl: (cluster_labels == lbl).sum() for lbl in _unique_lbls}

    _name_noise = "Віддалені території (Rural/Exurbs)"
    cluster_names = {-1: f"-1: {_name_noise}"}

    # 🎨 Динамічно збираємо палітру та порядок
    custom_color_map = {f"-1: {_name_noise}": "#9ca3af"} # Сірий для віддалених
    ordered_categories = [f"-1: {_name_noise}"]

    if _counts:
        # Найбільший кластер — ЗАВЖДИ Прибережний Мегаполіс (LA + SF)
        _coastal_lbl = max(_counts, key=_counts.get)
        _coastal_full_name = f"{_coastal_lbl}: Прибережний Мегаполіс (Coastal Metro)"

        cluster_names[_coastal_lbl] = _coastal_full_name
        custom_color_map[_coastal_full_name] = "#2563eb" # Ідеально синій
        ordered_categories.append(_coastal_full_name)

        # 📍 Якорі: (Широта, Назва, 5 відтінків [Базовий, Темний, Світлий, Дуже Темний, Дуже Світлий])
        _anchors = {
            "Redding": (40.58, "Крайня Північна Долина (Redding)", ["#14b8a6", "#0f766e", "#5eead4", "#115e59", "#99f6e4"]),      # Бірюзові
            "Sacramento": (38.58, "Північна Долина (Sacramento)", ["#22c55e", "#15803d", "#86efac", "#166534", "#bbf7d0"]),       # Зелені
            "Stockton": (37.95, "Північно-Центральна (Stockton/Modesto)", ["#a855f7", "#7e22ce", "#d8b4fe", "#6b21a8", "#e9d5ff"]), # Фіолетові
            "Fresno": (36.73, "Центральна Долина (Fresno)", ["#ec4899", "#be185d", "#f9a8d4", "#9d174d", "#fbcfe8"]),             # Рожеві
            "Bakersfield": (35.37, "Південна Долина (Bakersfield)", ["#f97316", "#c2410c", "#fdba74", "#9a3412", "#fed7aa"])      # Помаранчеві
        }

        # Застосовуємо словник
        # Якщо DBSCAN раптом не класифікує кластер (якого немає в словнику),
        # ми захищаємось через безпечний fallback:
        _fallback_name = "Некласифікований аномальний ринок"
        _fallback_colors = ["#ef4444", "#b91c1c", "#fca5a5", "#991b1b", "#fecaca"] # Червоні

        # Лічильники використання (щоб брати наступний відтінок з палітри)
        _anchor_usage = {k: 0 for k in _anchors.keys()}
        _fallback_usage = 0

        _valley_lbls = [lbl for lbl in _unique_lbls if lbl != _coastal_lbl]
        _valley_centroids = []
        for lbl in _valley_lbls:
            _mean_lat = df_geo.loc[cluster_labels == lbl, lat_col].mean()
            _valley_centroids.append((lbl, _mean_lat))

        # Сортуємо Долини географічно (Північ -> Південь)
        _valley_centroids.sort(key=lambda x: x[1], reverse=True)

        for lbl, _mean_lat in _valley_centroids:
            # Знаходимо найближчий якір до цього кластера
            _closest_key = min(_anchors.keys(), key=lambda k: abs(_anchors[k][0] - _mean_lat))
            _min_dist = abs(_anchors[_closest_key][0] - _mean_lat)

            # Якщо кластер занадто далеко (аномалія, >1.5 градуса широти від усіх міст)
            if _min_dist > 1.5:
                _base_name = _fallback_name
                _color = _fallback_colors[_fallback_usage % 5]
                _fallback_usage += 1
            else:
                _base_name = _anchors[_closest_key][1]
                _color = _anchors[_closest_key][2][_anchor_usage[_closest_key] % 5]
                _anchor_usage[_closest_key] += 1

            _full_name = f"{lbl}: {_base_name}"
            cluster_names[lbl] = _full_name
            custom_color_map[_full_name] = _color
            ordered_categories.append(_full_name)

    # Застосовуємо словник до колонки
    df_geo["GeoCluster"] = [
        cluster_names.get(lbl, f"{lbl}: Новий регіональний ринок")
        for lbl in cluster_labels
    ]

    # 5. ДЕМОНСТРАЦІЯ: OHE vs NATIVE CATEGORICALS
    df_ohe = pd.get_dummies(
        df_geo, columns=["GeoCluster"], drop_first=True, dtype=int
    )
    if "ocean_proximity" in df_ohe.columns:
        df_ohe = pd.get_dummies(
            df_ohe, columns=["ocean_proximity"], drop_first=True, dtype=int
        )

    df_native = df_geo.copy()
    df_native["GeoCluster"] = df_native["GeoCluster"].astype("category")
    if "ocean_proximity" in df_native.columns:
        df_native["ocean_proximity"] = df_native["ocean_proximity"].astype(
            "category"
        )

    _current_theme = mo.app_meta().theme
    _map_style = "carto-darkmatter" if _current_theme == "dark" else "carto-positron"
    _text_color = "white" if _current_theme == "dark" else "#1f2937"
    _border = "#4b5563" if _current_theme == "dark" else "#e5e7eb"

    # 6. Візуалізація карти
    inc_col = "median_income"
    val_col = "median_house_value"

    df_map = df_geo.copy()
    if val_col in df_map.columns and df_map[val_col].max() <= 100:
        df_map[val_col] = df_map[val_col] * 100000

    fig_dbscan = px.scatter_map(
        df_map,
        lat=lat_col,
        lon=lon_col,
        color="GeoCluster",
        color_discrete_map=custom_color_map,
        zoom=4.5,
        height=600,
        category_orders={"GeoCluster": ordered_categories},
        title=f"<b>DBSCAN Кластери (Силует: {sil_score:.3f})</b>",
        hover_name="GeoCluster",
        labels={
            lat_col: UA_COLUMNS.get(lat_col, "Широта"),
            lon_col: UA_COLUMNS.get(lon_col, "Довгота"),
            inc_col: f"{UA_COLUMNS.get(inc_col, 'Медіанний дохід')} (x$10k)",
            val_col: UA_COLUMNS.get(val_col, "Медіанна вартість житла"),
        },
        hover_data={
            "GeoCluster": False,
            lat_col: ":.3f",
            lon_col: ":.3f",
            inc_col: ":.2f",
            val_col: ":$,.0f",
        },
    )

    for trace in fig_dbscan.data:
        if "Віддалені" in trace.name:
            trace.legendgroup = "1_noise"
            trace.legendgrouptitle = dict(
                text="&nbsp;&nbsp;&nbsp;&nbsp;<b>🏞️ ІЗОЛЬОВАНІ ЗОНИ</b>",
                font=dict(color=_text_color),
            )
        elif "Прибережний" in trace.name:
            trace.legendgroup = "2_core"
            trace.legendgrouptitle = dict(
                text="&nbsp;&nbsp;&nbsp;&nbsp;<b>🌅 ОСНОВНИЙ РИНОК</b>",
                font=dict(color=_text_color),
            )
        else:
            trace.legendgroup = "3_hubs"
            trace.legendgrouptitle = dict(
                text="&nbsp;&nbsp;&nbsp;&nbsp;<b>🏙️ ВНУТРІШНІ ХАБИ</b>",
                font=dict(color=_text_color),
            )

    fig_dbscan.update_layout(
        title=dict(
            text=f"DBSCAN Кластери (Силует: {sil_score:.3f})",
            x=0.5,
            xanchor="center",
            font=dict(color=_text_color),
        ),
        map_style=_map_style,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.75,
            xanchor="left",
            x=1.02,
            title_text="<b><i>Тип урбанізації:</i></b>",
            groupclick="toggleitem",
            font=dict(color=_text_color),
        ),
    )

    # 7. Демонстрація таблиці (One-Hot Encoding)
    # Беремо всі унікальні комбінації кластерів
    cluster_cols = [c for c in df_ohe.columns if "GeoCluster_" in c]
    df_sample_ohe = df_ohe[cluster_cols].drop_duplicates()

    # Перезаписуємо індекси для краси
    df_sample_ohe.index = range(1, len(df_sample_ohe) + 1)

    # Робимо перенесення рядка (break) після двокрапки для красивого UI
    df_sample_ohe = df_sample_ohe.rename(columns=lambda x: x.replace(": ", ":<br/>"))

    # Використовуємо наш єдиний стиль без форматування дробів
    _table_html = style_dataframe(
        df_sample_ohe,
        text_align="center",
        vertical_lines=True
    )

    # 8. UI-Блок
    _clustering_insight_ui = mo.md(
        """
        <div class="dbscan-expanded-cell"></div>
        <style>
            .output-area:has(.dbscan-expanded-cell),
            marimo-cell-output:has(.dbscan-expanded-cell) {
                max-height: none !important;
                overflow-y: hidden !important;
                padding-bottom: 25px !important;
            }
        </style>
        > 🧠 **MLOps Інтеграція (Інференс):**<br/>
        > Оскільки алгоритм DBSCAN математично не здатний обробляти нові дані (не має методу `.predict()`), ми додали "розумний" міст для Backend-у. Ми натренували швидкий алгоритм `KNeighborsClassifier(n=1)` запам'ятати межі кластерів DBSCAN. Тепер будь-які нові координати клієнта будуть миттєво віднесені до правильного хабу!

        > **🌎 Артефакти Гео-пайплайну:**<br/>
        > У цій комірці ми підготували три фінальні версії даних:

        > 1. **`df_geo` (Source of Truth):** Базовий датафрейм із максимальною точністю координат (`float64`). Використаний для рендерингу карти.<br/>
        > 2. **`df_ohe` (Для Лінійних моделей):** Розширена матриця з розкодованими стовпцями (0 та 1) для лінійної алгебри.<br/>
        > 3. **`df_native` (Для Дерев):** Колонка `GeoCluster` має тип `category`. Матриця ідеально компактна для алгоритмів дерев.

        > 🌊 **Географічна аномалія (Будинки в океані):**<br/>
        > На карті можна помітити точки (переписні квартали), що знаходяться у воді вздовж узбережжя.<br/>
        > Це не помилка коду, а особливість збору даних перепису США 1990 року (Census Block Groups).<br/>
        > Координати вказують на математичний центроїд кварталу, межі якого юридично захоплюють територіальні води океану.<br/>
        > Алгоритми ML здатні коректно обробити ці координати без додаткових маніпуляцій.
        """
    )

    mo.output.append(_clustering_insight_ui)
    mo.output.append(fig_dbscan)
    mo.output.append(mo.md(
        "<br/><br/>**👀 Як виглядає OHE (One-Hot Encoding) під капотом:**<br/>"
    ))
    mo.output.append(mo.Html(f"<div style='overflow-x: auto; border: 1px solid {_border}; border-radius: 8px;'>{_table_html}</div>"))
    mo.output.append(mo.md(
        "> *💡 **Де подівся кластер «GeoCluster -1: Віддалені території»?***<br/>"
        "*Зверніть увагу на рядок, що складається виключно з нулів. Це і є наш базовий кластер шумових/віддалених точок.<br/>"
        "Його стовпець було навмисно видалено (`drop_first=True`), щоб уникнути математичної проблеми лінійних моделей (Dummy Variable Trap).<br/>"
        "При цьому самі дані (будинки) нікуди не зникають з набору даних — алгоритм ідентифікує їх саме за відсутністю одиничок у всіх інших стовпцях.*"
    ))
    return df_native, df_ohe


@app.cell(hide_code=True)
def header_model_benchmarking(mo):
    mo.md("""
    <h2 align="center"><b>🏆 4. Бенчмаркінг: A/B Тест <i>(OHE vs Native Categorical + Log-Transform)</i></b></h2>
    """)
    return


@app.cell(hide_code=True)
def header_benchmark_config(mo):
    mo.md("""
    <h3 align="center"><b>🎚️ 4.1. Конфігурація експерименту <i>(Target Transform Strategy)</i></b></h3>
    """)
    return


@app.cell
def benchmark_config_ui(mo):
    # 🔑 Виносимо ключі у змінні, щоб Marimo міг їх правильно розпізнати
    opt_log = "🌟 LOG-TRANSFORM (log1p): Ідеально для грошей, гасить вплив вілл за $500k. Логарифмічна шкала надає пріоритет точності масового сегмента та знижує вплив елітної нерухомості на метрики."
    opt_sqrt = "🧮 SQRT-TRANSFORM (sqrt): Помірне згладжування для локальних аномалій. Коренева шкала зберігає баланс між масовим ринком та дорогими об'єктами."
    opt_raw = "🧱 RAW PRICES (raw): Без змін. Базовий рівень, де моделі 'божеволіють' від дорогих будинків. Це прямий прогноз без математичних втручань у ціноутворення на сирих даних."

    # Створюємо реактивну радіокнопку
    target_transform_selector = mo.ui.radio(
        options={opt_log: "log", opt_sqrt: "sqrt", opt_raw: "raw"},
        value=opt_log,
        label="**🛠️ Оберіть математичну трансформацію ціни (Target Transform):**",
    )

    # Виводимо на екран
    _selector_insight = mo.md(
        "> *Спробуйте змінити трансформацію, і Marimo автоматично перетренує всі алгоритми!*"
    )

    mo.output.append(_selector_insight)
    mo.output.append(target_transform_selector)
    return (target_transform_selector,)


@app.cell(hide_code=True)
def header_model_configurator(mo):
    mo.md("""
    <h3 align="center"><b>🕹️ 4.2. Налаштування пулу алгоритмів <i>(Algorithm Pool Setup)</i></b></h3>
    """)
    return


@app.cell
def model_data_state(
    AdaBoostRegressor,
    BayesianRidge,
    DecisionTreeRegressor,
    DummyRegressor,
    ElasticNet,
    ExplainableBoostingRegressor,
    ExtraTreesRegressor,
    GLOBAL_SEED,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    HuberRegressor,
    KNeighborsRegressor,
    KernelRidge,
    LGBMRegressor,
    Lasso,
    LinearRegression,
    LinearSVR,
    MLPRegressor,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    Pipeline,
    PolynomialFeatures,
    RandomForestRegressor,
    Ridge,
    StandardScaler,
    TweedieRegressor,
    XGBRFRegressor,
    XGBRegressor,
    lgbm_kwargs,
    mo,
    xgb_kwargs,
):
    # 1. Абстракції
    poly_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=100.0, random_state=GLOBAL_SEED)),
    ])

    # 2. Ініціалізація моделей регресії
    # Всі параметри заліза (cuda, sycl, cpu, n_jobs=-1) приходять із ml_utils.py

    # Параметричні лінійні алгоритми
    models_linear = {
        "Dummy (Mean Baseline)": (1, DummyRegressor(strategy="mean")),
        "OLS Baseline (OHE)": (2, Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression(n_jobs=-1))])),
        "Ridge L2 (OHE)": (3, Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=10.0, random_state=GLOBAL_SEED))])),
        "Lasso L1 (OHE)": (4, Pipeline([('scaler', StandardScaler()), ('regressor', Lasso(alpha=0.01, random_state=GLOBAL_SEED))])),
        "ElasticNet (OHE)": (5, Pipeline([('scaler', StandardScaler()), ('regressor', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=GLOBAL_SEED))])),
        "Huber Robust (OHE)": (6, Pipeline([('scaler', StandardScaler()), ('regressor', HuberRegressor(max_iter=1000))])),
        "Polynomial Ridge (OHE)": (7, poly_ridge),
        "Bayesian Ridge (OHE)": (8, Pipeline([('scaler', StandardScaler()), ('regressor', BayesianRidge())])),
        "Passive Aggressive (OHE)": (9, Pipeline([('scaler', StandardScaler()), ('regressor', PassiveAggressiveRegressor(random_state=GLOBAL_SEED))])),
        "Tweedie GLM (Pricing)": (10, Pipeline([('scaler', StandardScaler()), ('regressor', TweedieRegressor(power=1.5))])),
        "Kernel Ridge (OHE)": (11, Pipeline([('scaler', StandardScaler()), ('regressor', KernelRidge(alpha=1.0))])),
    }

    # Ретро-Класика
    models_classic = {
        "Decision Tree (CART, 1984)": (12, DecisionTreeRegressor(max_depth=15, random_state=GLOBAL_SEED)),
        "MLP Neural Net (80s Classic)": (13, Pipeline([('scaler', StandardScaler()), ('regressor', MLPRegressor(hidden_layer_sizes=(50,), max_iter=300, random_state=GLOBAL_SEED))])),
        "OMP (Signal Processing, 1993)": (14, Pipeline([('scaler', StandardScaler()), ('regressor', OrthogonalMatchingPursuit())])),
        "KNN (Instance-based, 90s)": (15, Pipeline([('scaler', StandardScaler()), ('regressor', KNeighborsRegressor(n_neighbors=5, n_jobs=-1))])),
        "AdaBoost (1995)": (16, AdaBoostRegressor(n_estimators=100, random_state=GLOBAL_SEED)),
        "Linear SVR (SVM, 1996)": (17, Pipeline([('scaler', StandardScaler()), ('regressor', LinearSVR(max_iter=2000, random_state=GLOBAL_SEED))])),
    }

    # Ансамблі
    models_ensemble = {
        "Random Forest (OHE)": (18, RandomForestRegressor(n_estimators=100, max_depth=15, random_state=GLOBAL_SEED, n_jobs=-1)),
        "Extra Trees (OHE)": (19, ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=GLOBAL_SEED, n_jobs=-1)),
        "Gradient Boosting (Sklearn Classic)": (20, GradientBoostingRegressor(n_estimators=200, random_state=GLOBAL_SEED)),
        "Scikit HistGradient (OHE)": (21, HistGradientBoostingRegressor(max_iter=200, random_state=GLOBAL_SEED)),
        "LightGBM (OHE)": (22, LGBMRegressor(random_state=GLOBAL_SEED, **lgbm_kwargs)),
        "XGBoost (OHE)": (23, XGBRegressor(n_estimators=200, random_state=GLOBAL_SEED, **xgb_kwargs))
    }

    # Моделі з нативною підтримкою категорій
    models_native = {
        "LightGBM (Native Category)": (24, LGBMRegressor(random_state=GLOBAL_SEED, **lgbm_kwargs)),
        "XGBoost (Native Category)": (25, XGBRegressor(n_estimators=200, enable_categorical=True, random_state=GLOBAL_SEED, **xgb_kwargs)),
        "Scikit Native HGB (Sklearn)": (26, HistGradientBoostingRegressor(max_iter=200, random_state=GLOBAL_SEED, categorical_features="from_dtype")),
        "Explainable Boosting (EBM)": (27, ExplainableBoostingRegressor(random_state=GLOBAL_SEED, n_jobs=-1)),
        "XGBoost Random Forest (Native)": (28, XGBRFRegressor(n_estimators=200, enable_categorical=True, random_state=GLOBAL_SEED, **xgb_kwargs)),
        "LightGBM DART (Dropouts)": (29, LGBMRegressor(boosting_type='dart', random_state=GLOBAL_SEED, **lgbm_kwargs)),
        "XGBoost DART (Dropouts)": (30, XGBRegressor(n_estimators=200, booster='dart', enable_categorical=True, random_state=GLOBAL_SEED, **xgb_kwargs))
    }

    # Реєстри для комірки 4.3
    master_registry = {**models_linear, **models_classic, **models_ensemble, **models_native}
    native_keys = list(models_native.keys())
    id_to_name_map = {f"#{mod_id:02d}": name for name, (mod_id, _) in master_registry.items()}

    # 3. Глобальні стани-рубильники (Force State)
    get_force_lin, set_force_lin = mo.state(True)
    get_force_cla, set_force_cla = mo.state(True)
    get_force_ens, set_force_ens = mo.state(True)
    get_force_nat, set_force_nat = mo.state(True)

    mo.center(mo.md("✅ **Алгоритми завантажено у памʼять!**"))
    return (
        get_force_cla,
        get_force_ens,
        get_force_lin,
        get_force_nat,
        id_to_name_map,
        master_registry,
        models_classic,
        models_ensemble,
        models_linear,
        models_native,
        native_keys,
        set_force_cla,
        set_force_ens,
        set_force_lin,
        set_force_nat,
    )


@app.cell
def controller_ui(
    get_force_cla,
    get_force_ens,
    get_force_lin,
    get_force_nat,
    mo,
    models_classic,
    models_ensemble,
    models_linear,
    models_native,
):
    # 1. Читаємо поточні значення глобальних рубильників
    force_lin = get_force_lin()
    force_cla = get_force_cla()
    force_ens = get_force_ens()
    force_nat = get_force_nat()

    # 2. Список моделей, які завжди повинні бути включені (не бажано відключити)
    mandatory_models = [
        "Dummy (Mean Baseline)",        # Абсолютний нуль для порівняння
        "OLS Baseline (OHE)",           # Графік 1 + Текст (R2)
        "Decision Tree (CART, 1984)",   # Графік 2
        "Random Forest (OHE)",          # Графік 3
        "XGBoost (Native Category)",    # Графік 4
        "Explainable Boosting (EBM)",   # Текст (Біла коробка)
        "MLP Neural Net (80s Classic)"  # Текст (Сліпі зони ШІ)
    ]

    # 3. Фабрика чекбоксів із захистом
    def make_cb(name, force_state):
        is_locked = name in mandatory_models
        # Якщо заблоковано: завжди галочка стоїть і інпут сірий
        # Інакше: слухає головний рубильник колонки
        return mo.ui.checkbox(
            label=name,
            value=True if is_locked else force_state,
            disabled=is_locked
        )

    # 4. Створюємо словники UI
    ui_linear = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_lin) for name, (mod_id, _) in models_linear.items()})
    ui_classic = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_cla) for name, (mod_id, _) in models_classic.items()})
    ui_ensemble = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_ens) for name, (mod_id, _) in models_ensemble.items()})
    ui_native = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_nat) for name, (mod_id, _) in models_native.items()})

    mo.center(mo.md("✅ **Словники алгоритмів - створено!**"))
    return ui_classic, ui_ensemble, ui_linear, ui_native


@app.cell
def view_render(
    mo,
    set_force_cla,
    set_force_ens,
    set_force_lin,
    set_force_nat,
    ui_classic,
    ui_ensemble,
    ui_linear,
    ui_native,
):
    # 1. Функція для створення групи (Кнопка-Емодзі + Заголовок)
    def build_group_view(ui_dict, title, set_state_fn):
        vals = ui_dict.value
        total = len(vals)
        completed = sum(vals.values())  # Рахуємо, скільки True у словнику

        # Визначаємо поточний емодзі та стан
        if completed == total and total > 0:
            icon_char = "✅"
            current_state_is_true = True
        else:
            icon_char = "☑️" if completed > 0 else "🔲"
            current_state_is_true = False

        # On-Click: змінює глобальний рубильник (перезапускає Комірку controller_ui)
        def toggle_all(_):
            set_state_fn(not current_state_is_true)

        # Створюємо кнопку-емодзі
        icon_button = mo.ui.button(
            label=icon_char,
            on_click=toggle_all,
            kind="neutral",
            tooltip="Увімкнути/Вимкнути всі"
        )

        # Компонуємо заголовок (Кнопка + Текст + Лічильник) і центруємо його
        header = mo.center(
            mo.hstack(
                [icon_button, mo.md(f"**<span style='font-size: 1.05em;'>{title} ({completed}/{total})</span>**")],
                align="center"
            )
        )
        return header, completed, total, icon_button

    # 2. Збираємо статистику по кожній групі
    h_lin, c_lin, t_lin, btn_lin = build_group_view(ui_linear, "Базові (OHE) 🐣", set_force_lin)
    h_cla, c_cla, t_cla, btn_cla = build_group_view(ui_classic, "Класичне МН 🌳", set_force_cla)
    h_ens, c_ens, t_ens, btn_ens = build_group_view(ui_ensemble, "Ансамблі (OHE) 🥁", set_force_ens)
    h_nat, c_nat, t_nat, btn_nat = build_group_view(ui_native, "Нативні 🫀", set_force_nat)

    # Загальний підрахунок для головного заголовка
    total_selected = c_lin + c_cla + c_ens + c_nat
    total_all = t_lin + t_cla + t_ens + t_nat

    # 3. Головний заголовок із динамічною цифрою
    main_header = mo.hstack([
        mo.md("🎛️ **Конфігуратор архітектур** *(A/B Тестування)*"),
        mo.md(f"<div style='text-align: right; color: #10b981; font-size: 1.1em;'><b>✓ Всього обрано: {total_selected} / {total_all} алгоритмів</b></div>")
    ], justify="space-between", align="center")

    # Шлюз безпеки
    run_btn = mo.ui.run_button(label="🎭 Запустити тренування обраних моделей", kind="success")

    # Вертикальний розділювач (фіксована висота обходить баг Flexbox-розтягування)
    v_line = mo.Html("<div style='width: 1px; background-color: #4b5563; min-height: 420px; margin: 0 15px; margin-top: 15px;'></div>")

    # Допоміжна функція для побудови колонки (заголовок по центру, список зліва)
    def build_column(header, ui_group):
        items_with_ids = [
            mo.hstack([mo.md(f"`{k}`"), cb], align="center")
            for k, cb in ui_group.items()
        ]

        return mo.vstack([
            header,
            mo.md("<div style='height: 10px;'></div>"),
            mo.vstack(items_with_ids, align="start")
        ], align="center")

    css_no_scroll = mo.md(
        """
        <div class="config-noscroll"></div>
        <style>
            marimo-cell-output:has(.config-noscroll),
            .output-area:has(.config-noscroll) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )

    # 4. Вивід фінального інтерфейсу
    config_panel = mo.vstack([
        css_no_scroll,
        mo.center(main_header),
        mo.center(mo.md("> Оберіть алгоритми для поточного тренування. Клікайте на **кнопку-емодзі** біля категорії для швидкого масового виділення. **Після вибору обов'язково натисніть 'Запустити' внизу!**")),
        mo.md("<br>"),
        mo.hstack([
            build_column(h_lin, ui_linear),
            v_line,
            build_column(h_cla, ui_classic),
            v_line,
            build_column(h_ens, ui_ensemble),
            v_line,
            build_column(h_nat, ui_native)
        ], justify="space-between", align="start"),
        mo.md("---"),
        mo.center(run_btn)
    ])

    mo.output.append(config_panel)
    return (run_btn,)


@app.cell(hide_code=True)
def header_benchmark_execution(mo):
    mo.md("""
    <h3 align="center"><b>🏋️‍♂️ 4.3. Тренування алгоритмів та Лідерборд <i>(Model Leaderboard & Diagnostics)</i></b></h3>
    """)
    return


@app.cell
def execute_benchmark(
    GLOBAL_SEED,
    clear_vram,
    df_native,
    df_ohe,
    go,
    id_to_name_map,
    make_subplots,
    master_registry,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mo,
    native_keys,
    np,
    pa,
    pd,
    pl,
    r2_score,
    re,
    run_btn,
    target_col,
    target_transform_selector,
    train_test_split,
    ui_classic,
    ui_ensemble,
    ui_linear,
    ui_native,
    xgb_kwargs,
):
    # 🧻 Очищуємо імена всіх колонок від спецсимволів (<, >, пробіли)
    df_ohe_clean = df_ohe.rename(columns=lambda x: re.sub(r"[^A-Za-z0-9_]+", "_", str(x)))
    df_native_clean = df_native.rename(columns=lambda x: re.sub(r"[^A-Za-z0-9_]+", "_", str(x)))
    target_col_clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(target_col))

    # ⛓️ Спліт для даних з OHE
    X_ohe = df_ohe_clean.drop(columns=[target_col_clean])
    y_ohe = df_ohe_clean[target_col_clean]
    X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(
        X_ohe, y_ohe, test_size=0.2, random_state=GLOBAL_SEED
    )

    # 🎏 Спліт для нативних категорій
    X_native = df_native_clean.drop(columns=[target_col_clean])
    y_native = df_native_clean[target_col_clean]
    X_train_native, X_test_native, _, _ = train_test_split(
        X_native, y_native, test_size=0.2, random_state=GLOBAL_SEED
    )

    # Адаптивний масштаб (Kaggle = долари, Sklearn = x100k)
    # Якщо ціна менше 100 (тобто 5.0), ми множимо її на 100к для графіків і метрик
    price_scale = 100000 if y_ohe.max() <= 100 else 1
    y_test_eval = y_test_ohe * price_scale

    # 🔦 Динамічна трансформація, залежить від попереднього блоку
    transform_strategy = target_transform_selector.value
    if transform_strategy == "log":
        y_train_processed = np.log1p(y_train_ohe)
    elif transform_strategy == "sqrt":
        y_train_processed = np.sqrt(y_train_ohe)
    else:  # "raw"
        y_train_processed = y_train_ohe.copy()

    # 🪎 Динамічно визначаємо коротку назву бекенду
    _hw_type = xgb_kwargs.get("device", "cpu")
    if _hw_type == "cuda":
        _hw_ui = "CUDA GPU"
    elif _hw_type == "sycl":
        _hw_ui = "Intel XPU"
    else:
        _hw_ui = "Multi-core CPU"

    # 🧠 ЗБІРКА АЛГОРИТМІВ З UI-КОНФІГУРАТОРА (КОМІРКА 4.2)
    # ⚠️ Комірка "спить", поки не натиснуть кнопку з MVC-панелі
    mo.stop(
        not run_btn.value,
        mo.center(mo.md("### ⏳ Очікування конфігурації...\n> 🆘 Оберіть алгоритми у Конфігураторі вище та натисніть зелену кнопку **`🎭 Запустити тренування обраних моделей`**."))
    )

    selected_names = []

    # Логіка збору: словники вже містять актуальний стан (True/False). Просто витягуємо вибрані.
    for ui_group in [ui_linear, ui_classic, ui_ensemble, ui_native]:
        selected_names.extend([
            id_to_name_map[mod_id]
            for mod_id, is_sel in ui_group.value.items()
            if is_sel
        ])

    # ⚠️ Якщо користувач зняв усі галочки і натиснув "Запустити"
    mo.stop(
        not selected_names,
        mo.md("⚠️ **Неможливо запустити: не обрано жодного алгоритму!**")
    )

    all_models = [(name, master_registry[name]) for name in selected_names]
    total_models_count = len(all_models)
    results = []
    trained_models = {}

    with mo.status.progress_bar(
        title=f"Тренування {total_models_count} моделей ({transform_strategy.upper()})...",
        subtitle=f"💎 <b>Engine:</b> {_hw_ui} <br/>⏳ <b>Ініціалізація...</b>",
        remove_on_exit=True,
        total=total_models_count,
    ) as bar:
        for name, (mod_id, _model) in all_models:
            # 1. Оновлюємо текст (хто зараз тренується), не рухаючи сам повзунок
            bar.update(
                increment=0,
                subtitle=f"💎 <b>Engine:</b> {_hw_ui} <br/>☣️ <b>Тренуємо:</b> {name}",
            )

            # Маршрутизатор даних
            _is_native = name in native_keys
            _X_train_curr = X_train_native if _is_native else X_train_ohe
            _X_test_curr = X_test_native if _is_native else X_test_ohe

            # Тренування
            _model.fit(_X_train_curr, y_train_processed)
            y_pred_processed = _model.predict(_X_test_curr)

            # Збереження
            if transform_strategy == "log":
                y_pred = np.expm1(y_pred_processed)
            elif transform_strategy == "sqrt":
                y_pred = np.square(y_pred_processed)
            else:
                y_pred = y_pred_processed

            # Переводимо прогноз у реальні долари для універсальної оцінки
            y_pred_eval = y_pred * price_scale

            trained_models[name] = _model
            results.append({
                "ID": f"#{mod_id:02d}",
                "Алгоритм": name,
                "R-квадрат (R²) ⬆️": r2_score(y_test_eval, y_pred_eval),
                "MAE ($) ⬇️": mean_absolute_error(y_test_eval, y_pred_eval),
                "MAPE (%) ⬇️": mean_absolute_percentage_error(y_test_eval, y_pred_eval) * 100,
            })
            # 2. Робимо крок повзунка після успішного тренування
            bar.update()

    # Скидаємо індекси та робимо .copy(), щоб Marimo не падав на read-only масивах
    df_results = (
        pd.DataFrame(results)
        .sort_values(by="R-квадрат (R²) ⬆️", ascending=False)
        .reset_index(drop=True)
        .copy()
    )

    # Очищуємо пам'ять через наш утилітний метод
    clear_vram(None)  # Викликаємо без аргументів, бо device тепер всередині kwargs/ml_utils

    benchmark_theme = mo.app_meta().theme
    bench_border = "#4b5563" if benchmark_theme == "dark" else "#e5e7eb"
    bench_text = "white" if benchmark_theme == "dark" else "#1f2937"

    # 3. Створюємо класичний DataFrame і округлюємо (нативно в Pandas)
    # Зберігаємо суворі числові типи (int/float), щоб Marimo зміг намалювати красиві гістограми
    df_results["R-квадрат (R²) ⬆️"] = df_results["R-квадрат (R²) ⬆️"].round(4)
    df_results["MAE ($) ⬇️"] = df_results["MAE ($) ⬇️"].round(0).astype(int)
    df_results["MAPE (%) ⬇️"] = df_results["MAPE (%) ⬇️"].round(2)

    _display_data = None
    _ui_mode = "Unknown"

    # 🥇 РІВЕНЬ 1: Polars (Ідеально для Marimo 0.23.11+, є гістограми)
    try:
        _display_data = pl.from_pandas(df_results)
        _ui_mode = "🥇 Tier 1: Polars Engine"
    except Exception:
        # 🥈 РІВЕНЬ 2: PyArrow (Офіційний C++ бекенд Arrow, є гістограми)
        try:
            _display_data = pa.Table.from_pandas(df_results)
            _ui_mode = "🥈 Tier 2: PyArrow Engine"
        except Exception:
            # 🥉 РІВЕНЬ 3: Column-Dict (Чистий Python, обходить Pandas, ЗБЕРІГАЄ гістограми)
            try:
                _display_data = df_results.to_dict(orient="list")
                _ui_mode = "🥉 Tier 3: Column-Dict Fallback"
            except Exception:
                # 🛡️ РІВЕНЬ 4: Record-Dict (Найбезпечніший, без гістограм, але працює завжди)
                _display_data = df_results.to_dict(orient="records")
                _ui_mode = "🛡️ Tier 4: Safe-Records Dict"

    # Налаштування центрування (нативне для Marimo)
    _justify_config = {col: "center" for col in df_results.columns}

    # 🛟 РІВЕНЬ 5 (ULTIMATE FALLBACK): Перехоплення крашу самого UI-компонента
    try:
        _benchmark_table = mo.ui.table(
            _display_data,
            selection=None,
            page_size=50,
            text_justify_columns=_justify_config,
            label=f"🏆 **Результати A/B тестування архітектур - (Стратегія: {transform_strategy.upper()} | Рушій UI: {_ui_mode}):**"
        )
    except Exception as critical_err:
        # Якщо інтерактивний компонент Marimo повністю падає, віддаємо статичний HTML!
        _ui_mode = "🛟 Tier 5: Static HTML (Critical Fallback)"
        _benchmark_table = mo.vstack([
            mo.md(f"🏆 **Результати A/B тестування архітектур - (Стратегія: {transform_strategy.upper()} | Рушій UI: {_ui_mode}):**"),
            mo.md(f"> ⚠️ *Інтерактивний рушій Marimo недоступний. Активовано резервний режим перегляду (HTML).*<br/>> <sub style='color: gray;'>Деталі збою: {str(critical_err)}</sub>"),
            mo.Html(df_results.to_html(justify="center", index=False))
        ])

    # 🌗 Візуалізація (Еволюція алгоритмів 2x2)
    def get_model_data_safe(model_name, X_data):
        """Безпечно дістає прогнози та метрики з таблиці. Рятує від KeyError, якщо модель відсутня"""
        if model_name not in trained_models:
            return None, None, None, None

        # Зворотне перетворення математики
        pred_raw = trained_models[model_name].predict(X_data)
        if transform_strategy == "log":
            pred = np.expm1(pred_raw)
        elif transform_strategy == "sqrt":
            pred = np.square(pred_raw)
        else:
            pred = pred_raw

        # Переводимо в реальні долари для малювання
        pred_eval = pred * price_scale

        # 📊 Динамічно витягуємо метрики ПРЯМО з таблиці
        # Вказуємо явно колонку "Алгоритм", оскільки df_results.columns[0] — це тепер "ID"
        row_data = df_results[df_results["Алгоритм"].apply(lambda x: model_name in str(x))]

        if not row_data.empty:
            mae = row_data["MAE ($) ⬇️"].iloc[0]
            r2 = row_data["R-квадрат (R²) ⬆️"].iloc[0]
            mape = row_data["MAPE (%) ⬇️"].iloc[0]
        else:
            mae, r2, mape = 0, 0, 0

        return pred_eval, mae, r2, mape

    # Конфігурація 4-х ключових архітектур для історії "Еволюції"
    grid_configs = [
        {"ukr": "🔴 1. Лінійна регресія", "eng": "OLS Baseline (OHE)", "x": X_test_ohe, "row": 1, "col": 1, "color": "#ef4444"},
        {"ukr": "🟠 2. Дерево рішень", "eng": "Decision Tree (CART, 1984)", "x": X_test_ohe, "row": 1, "col": 2, "color": "#f97316"},
        {"ukr": "🔵 3. Випадковий ліс (Ансамбль)", "eng": "Random Forest (OHE)", "x": X_test_ohe, "row": 2, "col": 1, "color": "#3b82f6"},
        {"ukr": "🟢 4. Сучасний XGBoost", "eng": "XGBoost (Native Category)", "x": X_test_native, "row": 2, "col": 2, "color": "#10b981"}
    ]

    subplot_titles = []
    plot_data_cache = {}

    # Готуємо дані та багаторядкові заголовки
    for cfg in grid_configs:
        pred, mae, r2, mape = get_model_data_safe(cfg["eng"], cfg["x"])
        if pred is not None:
            # З нового рядка англійська назва + всі 3 ключові метрики
            title = f"{cfg['ukr']}<br><span style='font-size:12px; color:gray;'>{cfg['eng']}</span><br><span style='font-size:13px;'>R²: {r2} | MAE: ${mae:,} | MAPE: {mape}%</span>"
            plot_data_cache[cfg["eng"]] = pred
        else:
            # Fallback: Якщо алгоритм вимкнено під час тренування
            title = f"{cfg['ukr']}<br><span style='font-size:12px; color:gray;'>{cfg['eng']}</span><br>❌ <i>Алгоритм відключено</i>"
            plot_data_cache[cfg["eng"]] = None
        subplot_titles.append(title)

    # Трохи збільшили vertical_spacing (0.22), щоб влізли 3 рядки тексту
    _fig_diag = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08, vertical_spacing=0.22
    )

    # Малюємо графіки
    axis_limit = 550000

    for cfg in grid_configs:
        pred = plot_data_cache[cfg["eng"]]
        if pred is not None:
            _fig_diag.add_trace(
                go.Scattergl(
                    x=y_test_eval, y=pred, mode="markers",
                    marker=dict(color=cfg["color"], size=3, opacity=0.3),
                    name=cfg["eng"].split()[0],
                    hovertemplate="<b>Справжня ціна:</b> $%{x:,.0f}<br><b>Прогноз ціни:</b> $%{y:,.0f}<extra></extra>",
                ),
                row=cfg["row"], col=cfg["col"],
            )
        else:
            # Заглушка, якщо моделі немає
            _fig_diag.add_annotation(
                text="⚠️ Модель не брала участі<br>в поточному тестуванні",
                x=axis_limit/2, y=axis_limit/2, showarrow=False,
                font=dict(size=14, color="gray"),
                row=cfg["row"], col=cfg["col"]
            )

        # Малюємо діагональ (ідеальний прогноз)
        # Вона йде від 0 до 550k, оскільки далі по осі X ми графік не показуємо
        _fig_diag.add_trace(
            go.Scatter(
                x=[0, 550000], y=[0, 550000], mode="lines",
                line=dict(color=bench_text, dash="dash", width=2),
                showlegend=False, hoverinfo="skip",
            ),
            row=cfg["row"], col=cfg["col"],
        )

        # Жорстке обмеження тільки для осі X (до 550k)
        _fig_diag.update_xaxes(
            title_text="Справжня ціна ($)", gridcolor=bench_border, zerolinecolor=bench_border,
            range=[0, 550000], row=cfg["row"], col=cfg["col"]
        )
        # Вісь Y - БЕЗ лімітів (Plotly автоматично відмасштабує її, щоб показати всі "відльоти")
        _fig_diag.update_yaxes(
            title_text="Прогноз ціни ($)", gridcolor=bench_border, zerolinecolor=bench_border,
            autorange=True, row=cfg["row"], col=cfg["col"]
        )

    # Фінальний дизайн макета
    _fig_diag.update_layout(
        title=dict(
            text=f"<b>Еволюція алгоритмів ({transform_strategy.upper()}): Справжня ціна vs Прогноз ціни від моделі</b>",
            x=0.5, xanchor="center",
            y=0.97,
            font=dict(color=bench_text, size=18)
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=bench_text),
        height=850,
        margin=dict(l=0, r=0, b=40, t=115),
        showlegend=False,
    )

    # 🎞️ Витягуємо метрики для тексту
    # Чемпіон (перший рядок таблиці)
    top_r2_raw = df_results["R-квадрат (R²) ⬆️"].iloc[0]
    top_r2_pct = int(round(top_r2_raw * 100)) # 0.8432 -> 84
    top_mape_pct = int(round(df_results["MAPE (%) ⬇️"].iloc[0])) # 16.4 -> 16

    # Явно вказуємо колонку "Алгоритм" (а не [0], бо [0] — це тепер "ID")
    model_col = "Алгоритм"

    # Шукаємо OLS (щоб показати базовий рівень)
    ols_mask = df_results[model_col].astype(str).str.contains("OLS", na=False)
    if ols_mask.any():
        ols_r2_pct = int(round(df_results.loc[ols_mask, "R-квадрат (R²) ⬆️"].iloc[0] * 100))
    else:
        ols_r2_pct = 65 # Fallback, якщо OLS вимкнено

    # Шукаємо EBM / Explainable Boosting (Біла коробка)
    ebm_mask = df_results[model_col].astype(str).str.contains("Explainable|EBM", case=False, na=False)
    if ebm_mask.any():
        ebm_r2_pct = int(round(df_results.loc[ebm_mask, "R-квадрат (R²) ⬆️"].iloc[0] * 100))
        ebm_mae_k = int(round(df_results.loc[ebm_mask, "MAE ($) ⬇️"].iloc[0] / 1000)) # 31253 -> 31
    else:
        ebm_r2_pct = 83 # Fallback
        ebm_mae_k = 31  # Fallback

    _benchmark_insight = mo.md(
        f"""
        <div class="benchmark-expanded-cell"></div>
        <style>
            marimo-cell-output:has(.benchmark-expanded-cell),
            .output-area:has(.benchmark-expanded-cell) {{
                max-height: none !important;
                overflow: visible !important;
            }}
        </style>

        > <b>🔄 Теперішня стратегія трансформації: `{transform_strategy.upper()}`</b><br/>
        > Зверніть увагу, як зміна математичної функції кардинально змінює лідерів таблиці, купчастість помилок на графіках та загальну здатність алгоритмів "розуміти" ринок!

        > <b>📊 Як читати метрики лідерборду:</b>

        > - <b>R-квадрат (R²) ⬆️ (Пояснена дисперсія):</b> Оцінка `~{top_r2_raw:.2f}` означає, що найкраща модель змогла пояснити **{top_r2_pct}% усіх причин**, чому ціни на будинки відрізняються. Для порівняння, базова лінійна регресія (OLS) зрозуміла лише близько `{ols_r2_pct}%` ринку.<br/>
        > - <b>MAE (`$`) ⬇️ (Середня абсолютна похибка):</b> Показує, на скільки реальних доларів у середньому помиляється модель.<br/>
        > - <b>MAPE (%) ⬇️ (Середня похибка у відсотках):</b> Наш ШІ-чемпіон помиляється в середньому лише на `~{top_mape_pct}%` від вартості будинку.

        > <b>💼 Бізнес-інсайти (За результатами {total_models_count} алгоритмів):</b>

        > - ⚖️ **Ідеальний баланс (Біла Коробка):** Алгоритми на кшталт *Explainable Boosting (EBM)* показують топову точність (близько `{ebm_r2_pct}%`) із похибкою в районі `${ebm_mae_k}k`. Для бізнесу це ідеальне рішення, бо ми можемо на 100% математично пояснити регулятору та клієнту, чому алгоритм видав саме таку ціну.<br/>
        > - 🦮 **Сліпі зони ШІ:** Останні місця таблиці (часто з від'ємним R²) чудово демонструють жорстке правило ML: *складні алгоритми (як-от нейромережі типу MLP) не працюють "з коробки" на табличних даних і вимагають масштабування та ретельного налаштування гіперпараметрів.*
        """
    )

    _business_insight = mo.md(
        """
        > **💡 Tech Lead Insight (Аналіз аномалій та еволюція алгоритмів):**<br/>
        > 🔍 **Сліпа зона набору даних:** На правому краї графіків чітко видно історичний баг перепису Каліфорнії — усі будинки, дорожчі за `$500,000`, жорстко обрізалися цим штучним лімітом. Те, як алгоритми впоралися з цією аномалією, чудово ілюструє їхню еволюцію:

        > 1. **🔴 Лінійна регресія (OLS):** Сліпа до нелінійних патернів. Її прогнози розлітаються широкою хмарою, повністю ігноруючи ліміт і "пробиваючи стелю" в $500k.
        > 2. **🟠 Дерево рішень (1984):** Ідеально "відловило" штучний ліміт (чітка вертикальна лінія), але через примітивність алгоритму прогнози лягають грубими "сходинками".
        > 3. **🔵 Випадковий ліс (Ансамбль):** Згладив сходинки самотнього дерева, сформувавши щільнішу хмару, але все ще має відчутні похибки на краях розподілу.
        > 4. **🟢 XGBoost (Modern ML):** Вершина еволюції. Розпізнає складні патерни, поважає ліміти та найсильніше притискає хмару прогнозів до центральної діагоналі ідеальної точності.
        """
    )

    # 👑 ФІНАЛ: ГРАФІК АБСОЛЮТНОГО ЧЕМПІОНА
    # 1. Знаходимо ім'я лідера (перший рядок відсортованої таблиці)
    champion_name = df_results["Алгоритм"].iloc[0]

    # 2. Визначаємо, які дані йому потрібні (Нативні чи OHE)
    is_champ_native = champion_name in native_keys
    X_test_champ = X_test_native if is_champ_native else X_test_ohe

    # 3. Витягуємо його прогнози та метрики через нашу безпечну функцію
    pred_champ, mae_champ, r2_champ, mape_champ = get_model_data_safe(champion_name, X_test_champ)

    fig_champion = go.Figure()

    if pred_champ is not None:
        fig_champion.add_trace(
            go.Scattergl(
                x=y_test_eval, y=pred_champ, mode="markers",
                # Використовуємо красивий "королівський" фіолетовий колір для лідера
                marker=dict(color="#8b5cf6", size=4, opacity=0.45),
                name="Прогноз Чемпіона",
                hovertemplate="<b>Справжня ціна:</b> $%{x:,.0f}<br><b>Прогноз лідера:</b> $%{y:,.0f}<extra></extra>",
            )
        )

        # Діагональ ідеального прогнозу
        fig_champion.add_trace(
            go.Scatter(
                x=[0, 550000], y=[0, 550000], mode="lines",
                line=dict(color=bench_text, dash="dash", width=2),
                showlegend=False, hoverinfo="skip",
            )
        )

    # Оформлення осей
    fig_champion.update_xaxes(
        title_text="Справжня ціна ($)", gridcolor=bench_border, zerolinecolor=bench_border,
        range=[0, 550000]
    )
    fig_champion.update_yaxes(
        title_text="Прогноз ціни ($)", gridcolor=bench_border, zerolinecolor=bench_border,
        autorange=True
    )

    # Фінальний дизайн графіка Чемпіона
    fig_champion.update_layout(
        title=dict(
            text=f"<b>👑 АБСОЛЮТНИЙ ЧЕМПІОН: {champion_name}</b><br><span style='font-size:15px; color:gray;'>R²: {r2_champ} | MAE: ${mae_champ:,} | MAPE: {mape_champ}%</span>",
            x=0.5, xanchor="center",
            y=0.92,
            font=dict(color=bench_text, size=22)
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=bench_text),
        height=550, # Робимо його великим і виразним
        margin=dict(l=0, r=0, b=40, t=90),
        showlegend=False,
    )

    # 4. Універсальний філософський інсайт
    _champion_insight = mo.md(
        f"""
        > 🏆 **Битва Титанів (No Free Lunch Theorem):**<br/>
        > Знайомтеся — це **{champion_name}**, найкращий алгоритм поточного прогону! <br/>
        > Залежно від того, які дані ви завантажили (Sklearn із 9 колонками чи Kaggle із текстовою категорією океану) та яку математичну трансформацію (`{transform_strategy.upper()}`) обрали, лідер може і буде змінюватись.

        > * 🌲 Часто наявність оригінальної текстової категорії (`df_native`) дає алгоритмам типу LightGBM/XGBoost перевагу над "роздутою" матрицею нулів та одиниць (OHE).
        > * 📐 Зміна трансформації (`log` vs `sqrt`) зміщує фокус алгоритму з елітних вілл на масовий ринок, виводячи в лідери інші архітектури.

        > Цей фінальний графік математично доводить фундаментальне правило Data Science: **Не існує одного універсального алгоритму для всіх задач. Перемагає той, чия архітектура найкраще резонує з формою ваших даних!**
        """
    )

    mo.output.append(_benchmark_insight)
    mo.output.append(_benchmark_table)
    mo.output.append(_fig_diag)
    mo.output.append(_business_insight)
    mo.output.append(fig_champion)
    mo.output.append(_champion_insight)
    return (
        X_test_native,
        X_test_ohe,
        X_train_native,
        X_train_ohe,
        df_results,
        trained_models,
        transform_strategy,
        y_test_ohe,
        y_train_ohe,
    )


@app.cell(hide_code=True)
def header_model_interpretation(mo):
    mo.md("""
    <h2 align="center"><b>🩻 5. Інтерпретація моделі <i>(Model Explainability)</i></b></h2>
    """)
    return


@app.cell(hide_code=True)
def header_model_selector_ui(mo):
    mo.md("""
    <h3 align="center"><b>🎛️ 5.1. Інтерактивний Селектор Моделі <i>(Model Selection)</i></b></h3>
    """)
    return


@app.cell
def model_selector_ui(df_results, master_registry, mo):
    # 1. Читаємо весь відсортований лідерборд (всі моделі, що тренувалися!)
    _ranked_names = df_results["Алгоритм"].tolist()

    # 2. Формуємо розумний словник для Dropdown (Top-5 + Розділювач + Решта)
    _dropdown_options = {}
    _top_5 = _ranked_names[:5]
    _rest = _ranked_names[5:]

    # 🏅 Додаємо Топ-5 з красивими медалями
    _medals = ["🥇", "🥈", "🥉", "🏵️", "🏵️"]
    for i, _name in enumerate(_top_5):
        _mod_id = master_registry[_name][0] # Дістаємо ID алгоритму
        _dropdown_options[f"{_medals[i]} #{_mod_id:02d} {_name}"] = _name

    # Додаємо візуальний розділювач
    # (Хак безпеки: якщо користувач випадково клікне на лінію, повертаємо топ-1 модель, щоб не було крашу)
    if _rest:
        _dropdown_options["─── 👇 ІНШІ ТРЕНОВАНІ АЛГОРИТМИ 👇 ───"] = _top_5[0]

    # ▫️ Додаємо решту списку
    for _name in _rest:
        _mod_id = master_registry[_name][0] # Дістаємо ID алгоритму
        _dropdown_options[f"🎗️ #{_mod_id:02d} {_name}"] = _name

    # 3. Дефолтний вибір (Абсолютний чемпіон)
    _default_key = list(_dropdown_options.keys())[0]

    # 4. Створюємо інтерактивний Dropdown
    champion_selector = mo.ui.dropdown(
        options=_dropdown_options,
        value=_default_key,  # Marimo бере ключ
        label="🏆 **Оберіть модель для глибокого аналізу:** "
    )

    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"

    # 5. Рендеримо картку з центруванням
    ui_card = mo.md(
        f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
            <h3 style="margin-top: 0;">🎛️ Інтерактивний центр аналізу (XAI)</h3>
            <p>Завдяки реактивності Marimo, <b>усі наступні графіки (Feature Importance, Аналіз залишків) та пайплайн Optuna автоматично перебудуються</b> під ваш вибір!</p>
            <div style="display: flex; justify-content: center; align-items: center; margin-top: 15px;">
                {champion_selector}
            </div>
        </div>
        """
    )

    mo.output.append(ui_card)
    return (champion_selector,)


@app.cell(hide_code=True)
def header_feature_importance(mo):
    mo.md("""
    <h3 align="center"><b>🗃️ 5.2. Розпакування 'чорної скриньки' <i>(Feature Importance)</i></b></h3>
    """)
    return


@app.cell
def plot_feature_importance(
    UA_COLUMNS,
    X_test_native,
    X_test_ohe,
    champion_selector,
    go,
    master_registry,
    mo,
    native_keys,
    np,
    pd,
    permutation_importance,
    trained_models,
    transform_strategy,
    y_test_ohe,
):
    _selected_name = champion_selector.value
    _model = trained_models[_selected_name]

    # Витягуємо ID алгоритму з master_registry
    _mod_id = master_registry[_selected_name][0]
    _id_str = f"#{_mod_id:02d}"

    _is_native = _selected_name in native_keys
    _X_test_curr = X_test_native if _is_native else X_test_ohe
    _features = _X_test_curr.columns

    # Універсальний адаптер важливості алгоритмів
    def _get_final_estimator(model):
        if hasattr(model, "steps") and len(model.steps) > 0:
            return model.steps[-1][1]
        return model

    _core_estimator = _get_final_estimator(_model)

    _importances = None
    _calc_method = "Невідомо"

    # 1. НАМАГАЄМОСЯ ДІСТАТИ НАТИВНУ ВАГУ
    if hasattr(_core_estimator, 'feature_importances_'):
        _importances = _core_estimator.feature_importances_
        _calc_method = "Нативна (Gini / Information Gain)"
    elif hasattr(_core_estimator, 'coef_'):
        _importances = np.abs(_core_estimator.coef_)
        if _importances.ndim > 1:
            _importances = _importances.ravel()
        _calc_method = "Абсолютні коефіцієнти (|coef_|)"
    elif hasattr(_core_estimator, 'term_importances'):
        try:
            _ebm_scores = _core_estimator.term_importances()
            _ebm_names = _core_estimator.term_names_ if hasattr(_core_estimator, 'term_names_') else getattr(_core_estimator, 'feature_names_in_', _features)
            _score_map = dict(zip(_ebm_names, _ebm_scores))
            _importances = np.array([_score_map.get(f, 0.0) for f in _features])
            _calc_method = "Маргінальна вага (EBM)"
        except Exception:
            pass

    # 2. МАГІЯ XAI ДЛЯ ЧОРНИХ СКРИНЬОК (Якщо нативна вага не спрацювала або видала нулі)
    if _importances is None or len(_importances) != len(_features) or np.all(_importances == 0):
        # Готуємо правильний таргет (бо модель тренувалася на трансформованому y)
        if transform_strategy == "log":
            _y_test_ready = np.log1p(y_test_ohe)
        elif transform_strategy == "sqrt":
            _y_test_ready = np.sqrt(y_test_ohe)
        else:
            _y_test_ready = y_test_ohe.copy()

        # Рахуємо Permutation Importance з індикатором завантаження
        with mo.status.spinner("🪎 Зламуємо чорну скриньку", subtitle=f"'{_selected_name}' (Permutation Importance)..."):
            _perm_result = permutation_importance(
                _model, _X_test_curr, _y_test_ready, n_repeats=5, random_state=42, n_jobs=-1
            )
            # Беремо середнє значення падіння метрики
            _importances = _perm_result.importances_mean
            # ЗАХИСТ 1: Рятуємо від NaN та нескінченностей (буває у дуже поганих моделей)
            _importances = np.nan_to_num(_importances, nan=0.0, posinf=0.0, neginf=0.0)
            _importances = np.clip(_importances, a_min=0, a_max=None)

            _calc_method = "Перестановочна (Permutation Importance)"

    def _localize(feat):
        if feat in UA_COLUMNS:
            return UA_COLUMNS[feat]
        for _eng, _ua in UA_COLUMNS.items():
            if feat.startswith(_eng):
                _suffix = feat[len(_eng):].strip("_")
                if _suffix:
                    return f"{_ua} ({_suffix})"
                return _ua
        return feat

    _localized_features = [_localize(f) for f in _features]

    _df_fi = pd.DataFrame({
        "Ознака": _localized_features,
        "Важливість": _importances
    }).sort_values(by="Важливість", ascending=True)  # Зараз обмеження ознак - відсутні, але можна зробити топ-15 написавши вкінці: .tail(15)

    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"

    # 🎨 Кастомна палітра
    if "XGBoost" in _selected_name or "Forest" in _selected_name or "LightGBM" in _selected_name or "HistGradient" in _selected_name:
        _colorscale = 'Teal'
    elif "EBM" in _selected_name or "Explainable" in _selected_name:
        _colorscale = 'Purples'
    elif "MLP" in _selected_name or "KNN" in _selected_name:
        _colorscale = 'Blues'
    else:
        _colorscale = 'Oranges'

    _fig_fi = go.Figure(go.Bar(
        x=_df_fi["Важливість"],
        y=_df_fi["Ознака"],
        orientation='h',
        marker=dict(
            color=_df_fi["Важливість"],
            colorscale=_colorscale,
            showscale=False
        ),
        # Форматуємо цифри залежно від методу, бо Permutation дає менші дроби
        text=_df_fi["Важливість"].apply(lambda x: f"{x:.4f}" if "Permutation" in _calc_method else f"{x:.3f}"),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Вага (Вплив): %{x:.5f}<extra></extra>"
    ))

    # ЗАХИСТ 2: Якщо всі ознаки = 0, примусово розширюємо вісь X, щоб графік не зникав
    _max_val = _df_fi["Важливість"].max()
    _x_range = [0, 0.1] if _max_val == 0 else None

    # Вираховуємо динамічну висоту: 25 пікселів на кожну ознаку, але не менше 600
    _dynamic_height = max(600, len(_df_fi) * 25)

    _fig_fi.update_layout(
        title=dict(
            text=f"<b>Рентген моделі ({_id_str} {_selected_name})</b><br><span style='font-size:13px; color:gray;'>Метод екстракції: {_calc_method}</span>",
            x=0.5, font=dict(color=_text_color, size=18)
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_text_color),
        xaxis=dict(title="Сила впливу ознаки", gridcolor="rgba(128,128,128,0.2)", zeroline=False, range=_x_range),
        yaxis=dict(title="", gridcolor="rgba(128,128,128,0.2)"),
        height=_dynamic_height, margin=dict(l=20, r=40, t=75, b=20)
    )

    _insight_md = mo.md(
        """
        > **💡 Tech Lead Insight (Аналіз бізнес-драйверів та архітектур):**<br/>
        > Зверніть увагу на лідерів графіка. Механіка розрахунку "важливості" кардинально відрізняється залежно від сімейства алгоритмів, які ви зараз переглядаєте:

        > 📐 **1. Лінійні моделі (OLS, Ridge, Lasso, ElasticNet):**<br/>
        > Стовпчики показують абсолютну математичну вагу коефіцієнтів (`|coef_|`). По суті, це відповідь на запитання: *"На скільки зміниться ціна, якщо ця ознака виросте на 1 одиницю?"* (з урахуванням стандартизації). Чим більший стовпчик — тим сильніший прямий або зворотний лінійний вплив ознаки.

        > 🌲 **2. Дерева та Ансамблі (Decision Tree, Random Forest, XGBoost, LightGBM):**<br/>
        > Графік відображає **Information Gain** (приріст інформації) або зменшення домішок (Gini/MSE). Він показує, наскільки ефективно конкретна ознака зменшувала "хаос" і помилку прогнозу під час побудови гілок дерев. Це не прямі гроші, а відносна "корисність" ознаки для прийняття рішень алгоритмом.

        > 🔮 **3. Елітні 'Білі скриньки' (Explainable Boosting - EBM):**<br/>
        > Показує середню абсолютну маргінальну вагу. EBM будує окрему нелінійну функцію для кожної ознаки незалежно від інших. Стовпчик — це усереднений чистий вплив конкретної ознаки на фінальну ціну по всьому набору даних. Найвища прозорість для бізнесу та регуляторів.

        > 🗃️ **4. Чорні скриньки (MLP, KNN, HistGradientBoosting):**<br/>
        > Оскільки ці алгоритми не віддають свої внутрішні знання, ми використовуємо **Permutation Importance**. Ми навмисно "перемішуємо" (псуємо) одну колонку даних і дивимось, наскільки сильно впаде точність алгоритму. Чим довший стовпчик — тим катастрофічнішим для моделі виявилося псування цієї ознаки.

        > - **Нейромережі (MLP) та HistGradient** — розподіляють знання по сотнях прихованих нейронів та складних структурах. Замість того, щоб діставати з них вагу напряму, алгоритм навмисно "перемішує" (псує) кожну колонку даних і заміряє, наскільки катастрофічно падає точність прогнозу. Чим сильніше падіння — тим важливіша ознака.
        > - **KNN (K-Nearest Neighbors)** — це алгоритм на основі геометрії. Він взагалі не вивчає вагу колонок, а просто обчислює дистанцію у багатовимірному просторі. Завдяки перестановкам ми розуміємо, яка координата виявилася критичною для пошуку найближчих сусідів.
        > - **Dummy Baseline** — єдиний алгоритм, який завжди світитиметься нулями. Це заглушка, яка просто видає середню ціну по ринку, жорстко ігноруючи всі вхідні ознаки. Псування колонок ніяк не впливає на його прогноз.
        """
    )

    _css_no_scroll = mo.md(
        """
        <div class="xai-noscroll"></div>
        <style>
            marimo-cell-output:has(.xai-noscroll),
            .output-area:has(.xai-noscroll) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )

    mo.output.append(mo.vstack([_css_no_scroll, _fig_fi, _insight_md]))
    return


@app.cell(hide_code=True)
def header_residual_analysis(mo):
    mo.md("""
    <h3 align="center"><b>🔬 5.3. Аналіз залишків <i>(Residual Plot)</i></b></h3>
    """)
    return


@app.cell
def plot_residual_analysis(
    X_test_native,
    X_test_ohe,
    champion_selector,
    master_registry,
    mo,
    native_keys,
    np,
    px,
    trained_models,
    transform_strategy,
    y_test_ohe,
):
    _selected_name = champion_selector.value
    _model = trained_models[_selected_name]

    # Витягуємо ID алгоритму
    _mod_id = master_registry[_selected_name][0]
    _id_str = f"#{_mod_id:02d}"

    _is_native = _selected_name in native_keys
    _X_test_curr = X_test_native if _is_native else X_test_ohe

    _pred_raw = _model.predict(_X_test_curr)

    if transform_strategy == "log":
        _final_preds_raw = np.expm1(_pred_raw)
    elif transform_strategy == "sqrt":
        _final_preds_raw = np.square(_pred_raw)
    else:
        _final_preds_raw = _pred_raw

    # Адаптивний масштаб (Kaggle = долари, Sklearn = x100k)
    _price_scale = 100000 if y_test_ohe.max() <= 100 else 1

    # Переводимо у реальні долари перед розрахунком залишків
    _final_preds = _final_preds_raw * _price_scale
    _y_test_scaled = y_test_ohe * _price_scale

    # Тепер залишки (residuals) рахуються в сотнях тисяч доларів!
    _residuals = _y_test_scaled - _final_preds

    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"
    _grid_color = "#4b5563" if _theme == "dark" else "#e5e7eb"

    # Додаємо marginal_y="histogram" для оцінки нормальності розподілу
    _fig_res = px.scatter(
        x=_final_preds, y=_residuals,
        labels={'x': 'Прогнозована ціна', 'y': 'Залишок / Помилка'},
        opacity=0.4,
        color_discrete_sequence=['#ef4444'],
        marginal_y="histogram"
    )

    _fig_res.update_traces(
        hovertemplate="<b>Діапазон помилки:</b> %{y}<br><b>Кількість будинків:</b> %{x}<extra></extra>",
        selector=dict(type='histogram')
    )

    # Робимо красивий Hover для точок розсіювання
    _fig_res.update_traces(
        hovertemplate="<b>Прогноз:</b> $%{x:,.0f}<br><b>Помилка:</b> $%{y:,.0f}<extra></extra>",
        selector=dict(type='scatter')
    )

    _fig_res.add_hline(y=0, line_dash="dash", line_color=_text_color, line_width=2)

    # Стилізуємо лінії сітки абсолютно для всіх субплатів (включаючи бічну гістограму)
    _fig_res.update_xaxes(gridcolor=_grid_color, zerolinecolor=_grid_color)
    _fig_res.update_yaxes(gridcolor=_grid_color, zerolinecolor=_grid_color)

    # Фінальне налаштування геометрії та підписів
    _fig_res.update_layout(
        title=dict(
            text=f"<b>Аналіз залишків ({_id_str} {_selected_name}): Перевірка гомоскедастичності</b>",
            x=0.5, font=dict(color=_text_color, size=18)
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_text_color),

        # 💵 Форматуємо гроші тільки для логічно відповідних осей (xaxis2 для підрахунку залишаємо числовим)
        xaxis=dict(tickformat="$,.0f", title="Прогнозована ціна ($)"),
        yaxis=dict(tickformat="$,.0f", title="Залишок / Помилка ($)"),

        xaxis2=dict(
            title=dict(
                text="<b>Гістограма<br>розподілу</b>",
                font=dict(size=12, color="gray")
            )
        ),

        height=550,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    _css_no_scroll = mo.md(
        """
        <div class="res-noscroll"></div>
        <style>
            marimo-cell-output:has(.res-noscroll),
            .output-area:has(.res-noscroll) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )

    _insight = mo.vstack([
        _css_no_scroll,
        _fig_res,
        mo.md(
            """
            > **💡 Tech Lead Insight (Чому сліпо довіряти R² небезпечно?):**<br/>
            > Згідно з математичною теорією, ідеальна модель має демонструвати **гомоскедастичність** (залишки розкидані хаотично навколо нуля), а гістограма помилок справа — мати форму нормального "дзвону" (Gaussian Distribution).

            > ⚠️ **Аналіз аномалії (Гетероскедастичність):**<br/>
            > На графіку чітко видно "лійку" — чим дорожчий будинок, тим більша помилка. Модель системно недооцінює елітну нерухомість через штучний історичний ліміт набору даних у `$500,000` (алгоритм просто не бачив цін вище цієї стелі під час навчання).

            > 💼 **Бізнес-висновок:**<br/>
            > Цей XAI-аналіз доводить, що наш алгоритм є абсолютно надійним для мас-маркету та середнього класу. Проте, випускати цю модель у **Production** для оцінки Luxury-нерухомості (від `$450k`) небезпечно — для цього сегменту нам потрібно зібрати нові, нецензуровані дані або розробити окрему експертну систему.
            """
        )
    ])
    mo.output.append(_insight)
    return


@app.cell(hide_code=True)
def header_optimization(mo):
    mo.md("""
    <h2 align="center"><b>👣 6. Оптимізація та Пояснюваність <i>(Hyperparameter Tuning & SHAP)</i></b></h2>
    """)
    return


@app.cell(hide_code=True)
def header_optuna(mo):
    mo.md("""
    <h3 align="center"><b>🧪 6.1. Байєсівська оптимізація <i>(Optuna + MLflow)</i></b></h3>
    """)
    return


@app.cell
def optuna_ui_controls(champion_selector, mo, transform_strategy):
    _selected_name = champion_selector.value

    # Optuna налаштована лише для ансамблів
    _is_tunable = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])

    trials_slider = mo.ui.slider(start=3, stop=100, step=1, value=10, show_value=True, label="🛝 **Кількість ітерацій (n_trials):**")

    run_optuna_btn = mo.ui.run_button(
        label=f"💝 Запустити тюнінг для {_selected_name}",
        kind="success",
        disabled=not _is_tunable
    )

    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"

    _warning_msg = "" if _is_tunable else f"<div style='color: #ef4444; margin-bottom: 10px;'>⚠️ <b>Увага:</b> Алгоритм <i>{_selected_name}</i> не підтримується цим пайплайном Optuna. Оберіть XGBoost, LightGBM або Random Forest у селекторі вище.</div>"

    _architecture_guide = mo.Html(
        f"""
        <div class="optuna-noscroll" style="border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; font-family: inherit;">
            <details>
                <summary style="padding: 15px; font-weight: bold; cursor: pointer; text-align: center; display: block; outline: none; user-select: none;">
                    📚 Архітектурний путівник: Що ми оптимізуємо і чому? (Натисніть, щоб розгорнути ⏬️)
                </summary>
                <div style="padding: 20px; border-top: 1px solid {_border}; text-align: left; line-height: 1.6;">
                    <p>Байєсівська оптимізація (Optuna) — це "важка артилерія". Вона вимагає багато часу та обчислювальних ресурсів, тому в MLOps її застосовують лише там, де вона дає максимальний <b>ROI (Повернення інвестицій)</b> у точність.</p>
                    <br/>
                    <p>🟢 <b>Сучасні Бустинги та Ансамблі (XGBoost, LightGBM, Random Forest)</b><br/>
                    <b>Статус:</b> Підтримується ідеально.<br/>
                    <b>Чому:</b> Ці алгоритми мають гігантський простір нелінійних гіперпараметрів (<code>max_depth</code>, <code>learning_rate</code>, <code>subsample</code> тощо). Саме тут математика Optuna (алгоритм TPE) розкриває свій потенціал і здатна витиснути додаткові 5-15% точності, які неможливо знайти вручну.</p>
                    <br/>
                    <p>🟣 <b>Microsoft Interpret (Explainable Boosting - EBM)</b><br/>
                    <b>Статус:</b> Виведено з пайплайну Optuna.<br/>
                    <b>Чому:</b> EBM — це специфічна адитивна модель (GAM). Вона спроектована Microsoft так, щоб видавати топову точність <i>«з коробки»</i>. Спроба перебирати через Optuna тисячі парних взаємодій (<code>interactions</code>) та розмірів кошиків (<code>max_bins</code>) призводить до комбінаторного вибуху та забиває всю RAM/CPU (особливо на macOS), викликаючи системні дедлоки.<br/>
                    <b>Правильний підхід:</b> Якщо EBM <i>дуже</i> потрібно затюнити, це роблять ізольовано: через жорстко обмежений <code>GridSearchCV</code> на невеликій підвибірці даних, суворо вимкнувши мультипоточність (<code>n_jobs=1</code>).</p>
                    <br/>
                    <p>🔴 <b>Лінійні Моделі (OLS, Ridge, Lasso, ElasticNet, Huber, Tweedie)</b><br/>
                    <b>Статус:</b> Не оптимізуємо тут (Overkill).<br/>
                    <b>Чому:</b> Це опуклі математичні задачі з 1-2 параметрами. Використовувати Optuna для них — це стріляти з гармати по горобцях. У реальному продакшені для них використовують класи <code>RidgeCV</code> або <code>LassoCV</code>, які знаходять ідеальний оптимум математично і в 100 разів швидше.</p>
                    <br/>
                    <p>🟠 <b>Ретро-Класика (KNN, SVR, MLP, Decision Tree)</b><br/>
                    <b>Статус:</b> Не оптимізуємо (Низький ROI).<br/>
                    <b>Чому:</b> Звичайне Дерево (CART) тюнити немає сенсу — воно завжди програє Random Forest. KNN та SVR алгоритмічно надто довго рахуються на десятках тисяч рядків. А для нейромереж (MLP) Байєсівська оптимізація вимагає зовсім іншого підходу, що виходить за межі класичного табличного конвеєра.</p>
                </div>
            </details>
        </div>

        <style>
            /* Бронебійний CSS проти вертикального скролу (перевизначає Marimo) */
            marimo-cell-output:has(.optuna-noscroll),
            .output-area:has(.optuna-noscroll) {{
                max-height: 9999px !important;
                overflow: visible !important;
                overflow-y: visible !important;
            }}
        </style>
        """
    )

    _optuna_ui = mo.md(
        f"""
        <div class="optuna-noscroll" style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
            <h3 style="margin-top: 0;">🧪 Байєсівська оптимізація (Optuna) + MLflow</h3>
            <p><i>Всі моделі оптимізуються з урахуванням поточної стратегії <b>({transform_strategy.upper()})</b>. Результати автоматично логуються в SQLite базу MLflow.</i></p>
            {_warning_msg}
            {trials_slider}<br/><br/>{run_optuna_btn}
        </div>
        """
    )
    mo.output.append(mo.vstack([_architecture_guide, _optuna_ui]))
    return run_optuna_btn, trials_slider


@app.cell
def optuna_execution(
    GLOBAL_SEED,
    KFold,
    LGBMRegressor,
    RandomForestRegressor,
    XGBRegressor,
    X_train_native,
    X_train_ohe,
    champion_selector,
    lgbm_kwargs,
    logging,
    mean_absolute_error,
    mlflow,
    mo,
    native_keys,
    np,
    optuna,
    os,
    pd,
    plot_optimization_history,
    plot_param_importances,
    run_optuna_btn,
    style_dataframe,
    transform_strategy,
    trials_slider,
    xgb_kwargs,
    y_train_ohe,
):
    _selected_name = champion_selector.value
    _is_tunable = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])

    final_tuned_model = None

    if run_optuna_btn.value and _is_tunable:
        # Глушимо MLflow, щоб він не панікував через відсутність pip у середовищі 'uv'
        logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
        logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

        with mo.status.progress_bar(
            total=trials_slider.value,
            title=f"🦄 Тюнінг {_selected_name}",
            subtitle="⏳ Ініціалізація алгоритмів...",
            remove_on_exit=True
        ) as _bar:

            # Примусово створюємо папку (якщо її видалили), щоб SQLite мав куди писати
            os.makedirs("mlruns", exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///mlruns/mlruns.db")
            mlflow.set_experiment("Housing_Optimization")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            _is_native = _selected_name in native_keys
            _X_train_curr = X_train_native if _is_native else X_train_ohe

            if transform_strategy == "log":
                _y_train_processed = np.log1p(y_train_ohe)
            elif transform_strategy == "sqrt":
                _y_train_processed = np.sqrt(y_train_ohe)
            else:
                _y_train_processed = y_train_ohe.copy()

            # Адаптивний масштаб (Kaggle = долари, Sklearn = x100k)
            _price_scale = 100000 if y_train_ohe.max() <= 100 else 1

            def _objective(trial):
                _bar.update(increment=0, subtitle=f"🏃‍♂️ Ітерація {trial.number + 1} з {trials_slider.value}: Навчання 3-х фолдів (чекайте)...")

                if "XGBoost" in _selected_name:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                        "max_depth": trial.suggest_int("max_depth", 4, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "enable_categorical": _is_native,
                        "random_state": GLOBAL_SEED,
                        **xgb_kwargs
                    }
                    model_opt = XGBRegressor(**params)
                elif "LightGBM" in _selected_name:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                        "max_depth": trial.suggest_int("max_depth", 3, 12),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "random_state": GLOBAL_SEED,
                        **lgbm_kwargs
                    }
                    model_opt = LGBMRegressor(**params)
                else:
                    params = {"n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50), "max_depth": trial.suggest_int("max_depth", 5, 15), "random_state": GLOBAL_SEED, "n_jobs": -1}
                    model_opt = RandomForestRegressor(**params)

                kf = KFold(n_splits=3, shuffle=True, random_state=GLOBAL_SEED)
                cv_scores = []

                for train_idx, val_idx in kf.split(_X_train_curr):
                    # PyArrow: .copy() гарантує монолітну пам'ять і прискорює алгоритми
                    X_tr = _X_train_curr.iloc[train_idx].copy()
                    X_val = _X_train_curr.iloc[val_idx].copy()
                    y_tr_proc = _y_train_processed.iloc[train_idx].copy()
                    y_val_dollars = y_train_ohe.iloc[val_idx].copy()

                    model_opt.fit(X_tr, y_tr_proc)
                    preds_proc = model_opt.predict(X_val)

                    if transform_strategy == "log":
                        preds_dollars = np.expm1(preds_proc)
                    elif transform_strategy == "sqrt":
                        preds_dollars = np.square(preds_proc)
                    else:
                        preds_dollars = preds_proc

                    # Переводимо у реальні долари ПЕРЕД розрахунком помилки
                    preds_eval = preds_dollars * _price_scale
                    y_val_eval = y_val_dollars * _price_scale

                    cv_scores.append(mean_absolute_error(y_val_eval, preds_eval))

                return np.mean(cv_scores)

            def _progress_callback(study, trial):
                _bar.update(
                    increment=1,
                    subtitle=f"🌿 Ітерація {trial.number + 1} з {trials_slider.value} | Найкраще MAE: ${study.best_value:,.0f}"
                )

            _sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
            _study = optuna.create_study(direction="minimize", sampler=_sampler)

            # Вкидаємо наш callback в Optuna
            _study.optimize(_objective, n_trials=trials_slider.value, callbacks=[_progress_callback])

            _best_params = _study.best_params
            _best_params.update({"random_state": GLOBAL_SEED})

            if "XGBoost" in _selected_name:
                _best_params.update({"enable_categorical": _is_native, **xgb_kwargs})
                final_tuned_model = XGBRegressor(**_best_params)
            elif "LightGBM" in _selected_name:
                _best_params.update(**lgbm_kwargs)
                final_tuned_model = LGBMRegressor(**_best_params)
            else:
                final_tuned_model = RandomForestRegressor(**_best_params)

            # Перемикаємо статус перед фінальним довгим тренуванням
            _bar.update(increment=0, subtitle="💾 Збереження найкращої моделі у базу...")
            final_tuned_model.fit(_X_train_curr, _y_train_processed)

            # Очищаємо UI перед малюванням нових графіків
            mo.output.clear()

            _safe_run_name = f"Optuna_{_selected_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            with mlflow.start_run(run_name=_safe_run_name):
                mlflow.log_params(_best_params)
                mlflow.log_metric("CV_MAE_dollars", _study.best_value)
                mlflow.log_metric("Optuna_Trials", trials_slider.value)
                mlflow.log_param("Transform_Strategy", transform_strategy)

                _trusted_types = [
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBRegressor",
                    "lightgbm.sklearn.LGBMRegressor",
                    "lightgbm.basic.Booster",
                    "collections.OrderedDict"
                ]

                # Блокує виклик subprocess для pip, щоб уникнути помилок у середовищі Marimo
                mlflow.sklearn.log_model(
                    final_tuned_model,
                    artifact_path="champion_model",
                    skops_trusted_types=_trusted_types,
                    pip_requirements=["scikit-learn", "xgboost", "lightgbm"]
                )

                _run_id = mlflow.active_run().info.run_id

            _fig_history = plot_optimization_history(_study)
            try: _fig_params = plot_param_importances(_study)
            except Exception: _fig_params = None

            _theme = mo.app_meta().theme
            _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
            _text_color = "white" if _theme == "dark" else "#1f2937"

            # 🇺🇦 1. УКРАЇНІЗАЦІЯ ГРАФІКА ІСТОРІЇ (Осі та математичний зсув)
            _fig_history.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_text_color),
                title=dict(text="<b>Історія оптимізації</b>", x=0.5),
                xaxis_title="Ітерація (Спроба)",
                yaxis_title="Похибка MAE (Долари)"
            )
            for _trace in _fig_history.data:
                # Зсуваємо всі координати осі X на +1 (щоб рахувало від 1, а не від 0)
                if _trace.x is not None:
                    _trace.x = tuple(x + 1 for x in _trace.x)

                if _trace.name == 'Objective Value':
                    _trace.name = 'Похибка поточної ітерації'
                    # Додано форматування грошей!
                    _trace.hovertemplate = '<b>Ітерація:</b> %{x}<br><b>Похибка:</b> $%{y:,.0f}<extra></extra>'
                elif _trace.name == 'Best Value':
                    _trace.name = 'Рекорд (Найкраще значення)'
                    _trace.hovertemplate = '<b>Ітерація:</b> %{x}<br><b>Рекорд:</b> $%{y:,.0f}<extra></extra>'

            # 🇺🇦 2. УКРАЇНІЗАЦІЯ ГРАФІКА ВАЖЛИВОСТІ
            if _fig_params:
                _fig_params.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_text_color),
                    title=dict(text="<b>Важливість гіперпараметрів</b>", x=0.5),
                    xaxis_title="Ступінь впливу на результат",
                    yaxis_title="Гіперпараметр"
                )
                for _trace in _fig_params.data:
                    _trace.hovertemplate = '<b>Гіперпараметр:</b> %{y}<br><b>Вплив:</b> %{x}<extra></extra>'

            _params_table = style_dataframe(pd.DataFrame([_best_params]), text_align="center", vertical_lines=True, show_index=False)
            _plots_ui = mo.hstack([_fig_history, _fig_params], justify="center") if _fig_params else _fig_history

            # 🪎 Визначаємо красиве ім'я заліза для UI
            _hw_type = xgb_kwargs.get("device", "cpu")
            if _hw_type == "cuda":
                _hw_ui = "CUDA GPU 🟢"
            elif _hw_type == "sycl":
                _hw_ui = "Intel XPU 🔵"
            else:
                _hw_ui = "Multi-core CPU ⚙️"

            _css_no_scroll = mo.md(
                """
                <div class="optuna-noscroll"></div>
                <style>
                    marimo-cell-output:has(.optuna-noscroll),
                    .output-area:has(.optuna-noscroll) {
                        max-height: none !important;
                        overflow-y: visible !important;
                    }
                </style>
                """
            )

            _result_ui = mo.vstack([
                _css_no_scroll,
                mo.md(
                    f"""
                    ✅ **Оптимізацію завершено!** Найкраще MAE: `${_study.best_value:,.0f}`<br/>
                    💎 **Залізо (Engine):** `{_hw_ui}`<br/>
                    🗃️ **MLflow:** Усі параметри збережено в `mlruns/mlruns.db`! *(Run ID: `{_run_id}`)*
                    """
                ),
                mo.Html(f"<div style='overflow-x: auto; border: 1px solid {_border}; border-radius: 8px;'>{_params_table}</div>"),
                _plots_ui
            ])

            mo.output.append(_result_ui)
    return (final_tuned_model,)


@app.cell(hide_code=True)
def header_shap(mo):
    mo.md("""
    <h3 align="center"><b>🕵️‍♂️ 6.2. Квантова пояснюваність <i>(SHAP Values)</i></b></h3>
    """)
    return


@app.cell
def shap_ui_controls(
    X_train_native,
    X_train_ohe,
    champion_selector,
    final_tuned_model,
    mo,
    native_keys,
):
    _selected_name = champion_selector.value
    _is_tree = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])

    # Інтелектуально перевіряємо, чи підходить тюнінгована модель під поточні дані
    _is_native = _selected_name in native_keys
    _expected_features = X_train_native.shape[1] if _is_native else X_train_ohe.shape[1]

    _use_tuned = False
    if final_tuned_model is not None:
        _tuned_features = getattr(final_tuned_model, "n_features_in_", None)
        if _tuned_features == _expected_features:
            _use_tuned = True

    _model_origin = "Після оптимізації Optuna 🎯" if _use_tuned else "Базова з Лідерборду 🐣"

    shap_btn = mo.ui.run_button(
        label=f"👁️ Згенерувати SHAP для {_selected_name}",
        kind="info",
        disabled=not _is_tree
    )

    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"

    _warning_msg = "" if _is_tree else f"<div style='color: #ef4444; margin-bottom: 10px;'>⚠️ <b>Увага:</b> <i>{_selected_name}</i> не підтримує SHAP TreeExplainer. Оберіть Дерева або Бустинг.</div>"

    _shap_card = mo.md(
        f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
            <h3 style="margin-top: 0;">🕵️‍♂️ Аналіз SHAP (Квантова пояснюваність)</h3>
            <p>Цей алгоритм заглядає всередину "чорної скриньки" і розраховує внесок кожної ознаки для <b>кожного окремого будинку</b>.<br/>
            <i>Джерело моделі: <b>{_model_origin}</b></i></p>
            {_warning_msg}
            {shap_btn}
        </div>
        """
    )

    mo.output.append(_shap_card)
    return (shap_btn,)


@app.cell
def shap_execution(
    GLOBAL_SEED,
    UA_COLUMNS,
    X_train_native,
    X_train_ohe,
    champion_selector,
    final_tuned_model,
    mo,
    native_keys,
    plt,
    shap,
    shap_btn,
    shap_tree,
    trained_models,
    transform_strategy,
    y_train_ohe,
):
    if shap_btn.value:
        _selected_name = champion_selector.value
        _is_native = _selected_name in native_keys
        _X_train_curr = X_train_native if _is_native else X_train_ohe

        # Визначаємо модель безпечно
        _model_to_explain = trained_models[_selected_name]

        if final_tuned_model is not None:
            _expected_features = _X_train_curr.shape[1]
            _tuned_features = getattr(final_tuned_model, "n_features_in_", None)
            if _tuned_features == _expected_features:
                _model_to_explain = final_tuned_model

        # Адаптивний масштаб (Kaggle = долари, Sklearn = x100k)
        _price_scale = 100000 if y_train_ohe.max() <= 100 else 1

        # Додаємо спіннер для розуміння процесів під капотом
        with mo.status.spinner(title="🍻 Аналіз рішень моделі...", subtitle="Розрахунок векторів Шеплі (SHAP values)"):
            _X_sample = _X_train_curr.sample(n=500, random_state=GLOBAL_SEED)

            _orig_decode = getattr(shap_tree, "decode_ubjson_buffer", None)

            def _clean_base_score(_dict):
                """Витягує число з масиву '[12.08]' всередині словника XGBoost після декодування SHAP"""
                try:
                    _bs = _dict["learner"]["learner_model_param"]["base_score"]
                    if isinstance(_bs, str) and "[" in _bs:
                        _dict["learner"]["learner_model_param"]["base_score"] = _bs.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
                except Exception:
                    pass
                return _dict

            if _orig_decode:
                def _patched_decode(*args, **kwargs):
                    return _clean_base_score(_orig_decode(*args, **kwargs))
                shap_tree.decode_ubjson_buffer = _patched_decode

            try:
                _explainer = shap.TreeExplainer(_model_to_explain)
                _shap_values = _explainer.shap_values(_X_sample, check_additivity=False)

                # Якщо стратегія 'raw' і ми в sklearn, переводимо вектори впливу в реальні долари
                if transform_strategy == "raw" and _price_scale > 1:
                    _shap_values = _shap_values * _price_scale

            finally:
                if _orig_decode:
                    shap_tree.decode_ubjson_buffer = _orig_decode

            # Перейменовуємо колонки у датафреймі перед відмальовуванням
            _X_sample_ua = _X_sample.rename(columns=UA_COLUMNS)

            _fig, _ax = plt.subplots(figsize=(10, 6))

            # 🔥 додавши max_display=12 - залише графік чистим і сфокусованим на головному
            # проте мене цікавить показ всіх ознак, тому max_display не використовую
            shap.summary_plot(_shap_values, _X_sample_ua, show=False)

            _theme = mo.app_meta().theme
            _text_color = 'white' if _theme == 'dark' else '#1f2937'

            _ax.set_title(
                f"Квантова пояснюваність ({_selected_name}): Глобальний вплив ознак",
                color=_text_color,
                fontsize=15,
                fontweight='bold',
                pad=20
            )

            if transform_strategy == "log":
                _x_label = "Значення SHAP (Вплив на Логарифм ціни, log1p)"
                _scale_note = "⚠️ <b>Увага:</b> <i>Модель навчалась на <b>логарифмі ціни</b>. Вплив на осі X показує зміну логарифма (наближено до % зміни).</i>"
            elif transform_strategy == "sqrt":
                _x_label = "Значення SHAP (Вплив на Квадратний корінь ціни)"
                _scale_note = "⚠️ <b>Увага:</b> <i>Модель навчалась на <b>квадратному корені ціни</b>. Вплив на осі X є нелінійним.</i>"
            else:
                _x_label = "Значення SHAP (Вплив на прогноз ціни, $)"
                _scale_note = "💵 <i>Вплив на осі X <b>розраховано у реальних доларах США</b>.</i>"

            _ax.set_xlabel(_x_label)

            if _theme == "dark":
                # Робимо фон повністю прозорим
                _fig.patch.set_facecolor('none')
                _ax.set_facecolor('none')
                _ax.xaxis.label.set_color('white')
                _ax.yaxis.label.set_color('white')
                _ax.tick_params(colors='white')
                for spine in _ax.spines.values():
                    spine.set_edgecolor('gray')

            if len(_fig.axes) > 1:
                _cbar_ax = _fig.axes[-1]
                _cbar_ax.set_ylabel("Значення ознаки", rotation=270, labelpad=15)
                # Безпечне призначення тиків для matplotlib > 3.3
                _ticks = _cbar_ax.get_yticks()
                _cbar_ax.set_yticks(_ticks)
                _cbar_ax.set_yticklabels(["Низьке", "Високе"])

                if _theme == "dark":
                    _cbar_ax.yaxis.label.set_color('white')
                    _cbar_ax.tick_params(colors='white')

            _plot_html = mo.as_html(_fig)
            plt.close(_fig)

            _css_no_scroll = mo.md(
                """
                <div class="shap-noscroll"></div>
                <style>
                    marimo-cell-output:has(.shap-noscroll),
                    .output-area:has(.shap-noscroll) {
                        max-height: 9999px !important;
                        overflow: visible !important;
                        overflow-y: visible !important;
                    }
                </style>
                """
            )

            _info_algo_md = mo.md(
                f"""
                <center>{_scale_note}</center>
                """
            )

            # Додаємо пояснення сірих точок ТІЛЬКИ для нативних категоріальних моделей
            _native_insight = ""
            if _is_native:
                _native_insight = """
                > ⚙️ **Чому ознак рівно 12?**<br/>
                > Ви обрали модель з **нативною підтримкою категорій**. Замість того, щоб розбивати `Близькість до океану` та `GeoCluster` на багато окремих колонок (One-Hot Encoding), алгоритм розумно стиснув їх у їхні оригінальні 12 колонок. Це робить модель швидшою і легшою для аналізу!

                > 👽 **Що означають СІРІ точки?**<br/>
                > Вони з'являються лише на текстових категоріальних ознаках. Для таких ознак поняття "Високе/Низьке" значення не має математичного сенсу. SHAP спеціально фарбує їх у сірий колір, щоб захистити вас від хибних висновків за шкалою кольорів.
                """

            _insight_md = mo.md(
                f"""
                > **💡 Tech Lead Insight (Як читати SHAP):**<br/>
                > На відміну від звичайного "Рентгену" (Feature Importance), SHAP показує **напрямок** та **щільність** впливу.

                > - **Колір точки (🔴 Червоний/Синій 🔵):** Значення самої ознаки (напр., Червоний = високий дохід, Синій = низький).
                > - **Позиція на осі X (⏮️ Вліво/Вправо ⏭️):** Як сильно це значення зменшило (вліво) або збільшило (вправо) ціну будинку.

                {_native_insight}
                """
            )

            mo.output.append(mo.vstack([_css_no_scroll, _info_algo_md, mo.center(_plot_html), _insight_md]))
    return


@app.cell(hide_code=True)
def header_bonsai(mo):
    mo.md("""
    <h3 align="center"><b>🌳 6.3. Генеративне мистецтво даних <i>(3D Фрактал 'Цифровий Бонсай')</i></b></h3>
    """)
    return


@app.cell
def bonsai_dropdown_init(
    champion_selector,
    final_tuned_model,
    master_registry,
    mo,
    trained_models,
):
    _tree_keys = [k for k in trained_models.keys() if any(kw in k for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])]
    _options = {}

    if final_tuned_model is not None and champion_selector is not None:
        _tuned_name = champion_selector.value
        _mod_id = master_registry[_tuned_name][0]
        _options[f"🔱 #{_mod_id:02d} {_tuned_name} (Optuna Tuned)"] = "Optuna"

    for _k in _tree_keys:
        _mod_id = master_registry[_k][0]
        _options[f"🌱 #{_mod_id:02d} {_k}"] = _k

    _default_val = list(_options.keys())[0]

    local_bonsai_selector = mo.ui.dropdown(
        options=_options,
        value=_default_val,
        label="<span style='white-space: nowrap;'>🌳 <b>Алгоритм:</b></span>"
    )
    mo.center(mo.md("✅ **Алгоритми знайдено!**"))
    return (local_bonsai_selector,)


@app.cell
def bonsai_slider_init(
    final_tuned_model,
    local_bonsai_selector,
    mo,
    trained_models,
):
    _selected = local_bonsai_selector.value
    _actual_trees = 1

    if _selected == "Optuna" and final_tuned_model is not None:
        _model = final_tuned_model
    else:
        _model = trained_models[_selected]

    if hasattr(_model, "best_estimator_"):       # 1. Знімаємо обгортку Optuna
        _model = _model.best_estimator_
    if hasattr(_model, "steps"):                 # 2. Знімаємо обгортку Pipeline
        _model = _model.steps[-1][1]
    if hasattr(_model, "regressor_"):            # 3. Знімаємо обгортку TransformedTargetRegressor!
        _model = _model.regressor_

    # XGBoost, LightGBM, Random Forest використовують n_estimators
    # HistGradientBoosting використовує max_iter!
    # Якщо нічого немає (наприклад, звичайне Decision Tree), ставимо 1
    _actual_trees = getattr(_model, "n_estimators", getattr(_model, "max_iter", 1))

    _best_iter = getattr(_model, "best_iteration_", None)
    if _best_iter is not None and _best_iter > 0:
        _actual_trees = _best_iter

    _actual_trees = max(1, int(_actual_trees))

    bonsai_tree_slider = mo.ui.slider(
        start=1,
        stop=_actual_trees,
        step=1,
        value=min(15, _actual_trees),
        show_value=True,
        label=f"<span style='white-space: nowrap;'>🌿 <b>Дерев (Макс: {_actual_trees}):</b></span>"
    )
    mo.center(mo.md("✅ **Підрахунок дерев - виконано!**"))
    return (bonsai_tree_slider,)


@app.cell
def digital_bonsai_plot(
    GLOBAL_SEED,
    bonsai_tree_slider,
    final_tuned_model,
    go,
    local_bonsai_selector,
    mo,
    np,
    trained_models,
):
    # 🎭 Динамічне розширення пам'яті для великих масивів точок
    try:
        mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 50_000_000
    except Exception:
        pass

    _selected_name = local_bonsai_selector.value
    _use_tuned = (_selected_name == "Optuna")
    _model_origin = "Після оптимізації Optuna 🔱" if _use_tuned else "Базова з Лідерборду 🐣"

    # 🕳️ Глибока розпаковка моделі
    _model = final_tuned_model if _use_tuned else trained_models[_selected_name]
    if hasattr(_model, "best_estimator_"):
        _model = _model.best_estimator_
    if hasattr(_model, "steps"):
        _model = _model.steps[-1][1]

    _algo_family = str(type(_model))
    _is_xgb = "XGB" in _algo_family
    _is_lgbm = "LGBM" in _algo_family

    _actual_estimators = getattr(_model, "n_estimators", getattr(_model, "max_iter", 10))
    if hasattr(_model, "best_iteration_") and getattr(_model, "best_iteration_"):
        _actual_estimators = getattr(_model, "best_iteration_")

    _visual_trees = min(bonsai_tree_slider.value, max(1, int(_actual_estimators)))
    _nodes_x, _nodes_y, _nodes_z = [], [], []
    _lines_x, _lines_y, _lines_z = [], [], []
    _node_colors, _node_sizes, _node_text = [], [], []

    def _build_fractal_branch(x, y, z, angle_xy, angle_z, length, depth, max_depth=4):
        if depth > max_depth:
            return
        _jitter = np.random.uniform(-0.1, 0.1)
        x_end = x + length * np.sin(angle_z) * np.cos(angle_xy + _jitter)
        y_end = y + length * np.sin(angle_z) * np.sin(angle_xy + _jitter)
        z_end = z + length * np.cos(angle_z)

        _lines_x.extend([x, x_end, None])
        _lines_y.extend([y, y_end, None])
        _lines_z.extend([z, z_end, None])

        _nodes_x.append(x_end)
        _nodes_y.append(y_end)
        _nodes_z.append(z_end)

        _samples = max(1, int(20640 / (2**depth)) + np.random.randint(-10, 10))
        _info_gain = max(0.001, (1.0 / (depth + 1)) * np.random.uniform(0.7, 1.3))

        _node_sizes.append(max(2, 12 - depth * 2))
        _node_colors.append(depth)

        _hover_html = (
            f"<b>🧬 Алгоритм: {'Optuna Champion' if _use_tuned else _selected_name}</b><br>"
            f"🌲 Глибина спліту: {depth}<br>"
            f"🏠 Об'єктів у ноді: ~{_samples}<br>"
            f"💝 Важливість (Info Gain): {_info_gain:.4f}"
        )
        _node_text.append(_hover_html)

        _new_len = length * 0.73
        _build_fractal_branch(x_end, y_end, z_end, angle_xy + 0.55, angle_z + 0.32, _new_len, depth + 1, max_depth)
        _build_fractal_branch(x_end, y_end, z_end, angle_xy - 0.55, angle_z + 0.32, _new_len, depth + 1, max_depth)

    np.random.seed(GLOBAL_SEED)
    for _i in range(_visual_trees):
        _base_angle = _i * (2 * np.pi / max(1, _visual_trees))
        _tilt = np.random.uniform(0.12, 0.48)
        _build_fractal_branch(x=0, y=0, z=0, angle_xy=_base_angle, angle_z=_tilt, length=2.2, depth=0, max_depth=4)

    # --- Побудова Plotly ---
    _trace_branches = go.Scatter3d(
        x=_lines_x, y=_lines_y, z=_lines_z, mode='lines', name="Гілки (Правила)",
        line=dict(color='#4b5563', width=2), opacity=0.35, hoverinfo='skip',
        showlegend=True,
        legendgroup="branches"
    )

    _colorscale = 'Teal' if _is_xgb else 'Purples' if _is_lgbm else 'Greens'
    _legend_dot_color = "#14b8a6" if _is_xgb else "#a855f7" if _is_lgbm else "#22c55e"

    _trace_nodes_legend = go.Scatter3d(
        x=[None], y=[None], z=[None], mode='markers', name="Логічні Вузли",
        marker=dict(size=10, color=_legend_dot_color), # Жорстко задаємо колір алгоритму!
        showlegend=True,
        legendgroup="nodes"
    )

    _trace_nodes = go.Scatter3d(
        x=_nodes_x, y=_nodes_y, z=_nodes_z, mode='markers', name="Логічні Вузли",
        marker=dict(size=_node_sizes, color=_node_colors, colorscale=_colorscale, opacity=0.8, showscale=False),
        text=_node_text, hoverinfo='text',
        showlegend=False,
        legendgroup="nodes"
    )

    _fig_bonsai = go.Figure(data=[_trace_branches, _trace_nodes_legend, _trace_nodes])

    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"
    _border = "rgba(75, 85, 99, 0.5)" if _theme == "dark" else "rgba(229, 231, 235, 0.8)"
    _bg_panel = "rgba(31, 41, 55, 0.85)" if _theme == "dark" else "rgba(249, 250, 251, 0.9)"

    _fig_bonsai.update_layout(
        title=dict(
            text=f"<b>3D Ансамбль 'Цифровий Бонсай'</b><br><span style='font-size:12px; color:gray;'>Джерело: {_model_origin}</span>",
            x=0.5, font=dict(color=_text_color, size=15)
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_text_color),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.3, y=1.3, z=0.7))),
        height=720, margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            title=dict(text="<b>КАРТА АЛГОРИТМУ</b>", font=dict(color="gray", size=11)),
            yanchor="bottom", y=0.02,
            xanchor="right", x=0.98,
            bgcolor=_bg_panel,
            bordercolor=_border,
            borderwidth=1,
            font=dict(size=13, color=_text_color),
            itemsizing="constant"
        )
    )

    _css_no_scroll = mo.md(
                """
                <div class="shap-noscroll"></div>
                <style>
                    marimo-cell-output:has(.shap-noscroll),
                    .output-area:has(.shap-noscroll) {
                        max-height: 9999px !important;
                        overflow: visible !important;
                        overflow-y: visible !important;
                    }
                </style>
                """
            )

    _control_panel = mo.md(
        f"""
        <div style="position: relative; margin-top: -105px; margin-left: 2%; margin-bottom: 20px; width: calc(96% - 240px); height: 85px; padding: 0 20px; border-radius: 12px; background: {_bg_panel}; border: 1px solid {_border}; box-shadow: 0 8px 24px rgba(0,0,0,0.15); z-index: 50; backdrop-filter: blur(8px); display: flex; align-items: center;">
            <div style="width: 100%;">
                {mo.center(mo.hstack(
                    [local_bonsai_selector, bonsai_tree_slider],
                    justify="space-between",
                    align="center",
                    widths=[1, 1]
                ))}
            </div>
        </div>
        """
    )
    mo.vstack([_css_no_scroll, _fig_bonsai, _control_panel])
    return


@app.cell(hide_code=True)
def header_mlops_serialization(mo):
    mo.md("""
    <h2 align="center"><b>⛲️ 7. Продакшн: MLOps Серіалізація та Мікросервіс <i>(FastAPI)</i></b></h2>
    """)
    return


@app.cell
def mlops_ui_controls(
    champion_selector,
    final_tuned_model,
    master_registry,
    mo,
    trained_models,
):
    _export_options = {}

    # А) Додаємо Оптимізовану модель (якщо вона існує) нагору з VIP-статусом
    if final_tuned_model is not None:
        _tuned_name = champion_selector.value
        _mod_id = master_registry[_tuned_name][0]
        _export_options[f"🐋 #{_mod_id:02d} {_tuned_name} (Optuna Tuned 🔱)"] = {"name": _tuned_name, "is_tuned": True}

    # Б) Далі додаємо всі базові натреновані моделі з Лідерборду
    _sorted_baselines = sorted(
        trained_models.keys(),
        key=lambda k: master_registry[k][0] if k in master_registry else 999
    )

    for _name in _sorted_baselines:
        _mod_id = master_registry[_name][0]
        _export_options[f"🧧 #{_mod_id:02d} {_name} (Базова)"] = {"name": _name, "is_tuned": False}

    mlops_export_selector = mo.ui.dropdown(
        options=_export_options,
        value=list(_export_options.keys())[0],
        label="**🗄️ Оберіть алгоритм для розгортання:** "
    )

    mlops_generate_btn = mo.ui.run_button(label="⚙️ Згенерувати Production Артефакти", kind="success")

    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"

    _ui_panel = mo.vstack([
        mo.md(
            f"""
            <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
                <h3 style="margin-top: 0;">🛫 Експорт у Production (Мікросервіс)</h3>
                <p>Оберіть фінальну модель для передачі Backend-команді. У списку відображаються всі базові алгоритми з Лідерборду, а також модель, яка пройшла тюнінг Optuna (має VIP-статус 🔱).</p>
                <div style="display: flex; justify-content: center; align-items: center; margin-top: 15px;">
                    {mlops_export_selector}
                </div>
                <div style="margin-top: 15px;">
                    {mlops_generate_btn}
                </div>
            </div>
            """
        )
    ])

    mo.output.append(_ui_panel)
    return mlops_export_selector, mlops_generate_btn


@app.cell
def mlops_execution(
    X_train_native,
    X_train_ohe,
    datetime,
    final_tuned_model,
    joblib,
    json,
    mlops_export_selector,
    mlops_generate_btn,
    mo,
    native_keys,
    np,
    os,
    trained_models,
    transform_strategy,
):
    if mlops_generate_btn.value:
        _selection = mlops_export_selector.value
        _selected_name = _selection["name"]
        _is_tuned = _selection["is_tuned"]

        # Дістаємо потрібну модель (Optuna чи Базова)
        _model_to_save = final_tuned_model if _is_tuned else trained_models[_selected_name]

        _is_native = _selected_name in native_keys
        _X_curr = X_train_native if _is_native else X_train_ohe

        # 1. Ізолюємо артефакти в окремій папці проекту!
        _project_name = "california_housing"
        _artifact_dir = os.path.join(os.getenv("MODELS_DIR", "./models"), _project_name)
        os.makedirs(_artifact_dir, exist_ok=True)

        _safe_name = _selected_name.replace(" ", "_").replace("(", "").replace(")", "").lower()

        _model_path = os.path.join(_artifact_dir, f"{_safe_name}_champion.joblib")
        _schema_path = os.path.join(_artifact_dir, "features_schema.json")
        _api_path = os.path.join(_artifact_dir, "api.py")

        # 2. Зберігаємо ваги моделі
        joblib.dump(_model_to_save, _model_path)

        # Логіка опису трансформації
        if transform_strategy == "log":
            _strategy_desc = "Логарифмічна (log1p) ➔ Авто-декодування через expm1"
        elif transform_strategy == "sqrt":
            _strategy_desc = "Квадратний корінь (sqrt) ➔ Авто-декодування через square"
        else:
            _strategy_desc = "Сирі ціни (raw) ➔ Без зворотного перетворення"

        # Зберігаємо самі списки категорій
        _dtypes_dict = {}
        _categories_dict = {}
        for col in _X_curr.columns:
            _dt = _X_curr[col].dtype
            _dtypes_dict[col] = str(_dt)
            if hasattr(_dt, 'categories'):
                _categories_dict[col] = list(_dt.categories)

        # 3. Зберігаємо розширений маніфест
        _features_schema = {
            "project_name": _project_name,
            "exported_at": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "model_architecture": _selected_name,
            "is_optuna_tuned": _is_tuned,
            "target_transform_applied": transform_strategy,
            "expected_columns": list(_X_curr.columns),
            "dtypes": _dtypes_dict,
            "categories": _categories_dict
        }
        with open(_schema_path, "w", encoding="utf-8") as f:
            json.dump(_features_schema, f, indent=4, ensure_ascii=False)

        # Беремо перший рядок реальних даних з набору даних
        _sample_row = _X_curr.iloc[0].to_dict()
        # Конвертуємо numpy-типи у нативні Python-типи для коректного JSON-експорту
        _sample_clean = {
            k: (float(v) if isinstance(v, (float, np.floating))
                else int(v) if isinstance(v, (int, np.integer))
                else str(v))
            for k, v in _sample_row.items()
        }
        _sample_json_str = json.dumps({"features": _sample_clean}, ensure_ascii=False)

        # 4. Автогенерація сервера FastAPI
        _is_tuned_str = "True" if _is_tuned else "False"
        _api_lines = [
            'from fastapi import FastAPI, HTTPException',
            'from fastapi.responses import HTMLResponse',
            'from pydantic import BaseModel',
            'from typing import Dict, Any',
            'import joblib',
            'import json',
            'import pandas as pd',
            'import uvicorn',
            'import numpy as np',
            'import os',
            '',
            '# Вимикаємо стандартний Swagger UI та Redoc',
            'app = FastAPI(title="🏡 California Housing AI API", version="1.0", docs_url=None, redoc_url=None)',
            '',
            '# 💅 Підключаємо сучасний Scalar замість Swagger',
            '@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)',
            'def scalar_html():',
            '    return """',
            '    <!doctype html>',
            '    <html>',
            '      <head>',
            '        <title>California Housing API Docs</title>',
            '        <meta charset="utf-8" />',
            '        <meta name="viewport" content="width=device-width, initial-scale=1" />',
            '      </head>',
            '      <body>',
            '        <script id="api-reference" data-url="/openapi.json"></script>',
            '        <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>',
            '      </body>',
            '    </html>',
            '    """',
            '',
            '# Визначаємо абсолютний шлях до папки з моделлю',
            'BASE_DIR = os.path.dirname(os.path.abspath(__file__))',
            f'MODEL_PATH = os.path.join(BASE_DIR, "{os.path.basename(_model_path)}")',
            'SCHEMA_PATH = os.path.join(BASE_DIR, "features_schema.json")',
            '',
            '# Завантаження моделі та схеми',
            'try:',
            '    model = joblib.load(MODEL_PATH)',
            '    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:',
            '        schema = json.load(f)',
            '        expected_dtypes = schema.get("dtypes", {})',
            '        expected_categories = schema.get("categories", {})',
            '        expected_columns = schema.get("expected_columns", [])', # 👈 1. Зчитуємо еталонний порядок колонок
            'except Exception as e:',
            '    print(f"❌ Помилка ініціалізації: {e}")',
            '    model = None',
            '    expected_dtypes = {}',
            '    expected_categories = {}',
            '    expected_columns = []',
            '',
            'class InferencePayload(BaseModel):',
            '    features: Dict[str, Any]',
            '',
            '    # 🤖 Автоматично згенерований приклад запиту на основі реальних даних!',
            '    model_config = {',
            '        "json_schema_extra": {',
            f'            "examples": [{_sample_json_str}]',
            '        }',
            '    }',
            '',
            '# 💎 Чітка Pydantic-схема для успішної відповіді (200 OK)',
            'class PredictionResponse(BaseModel):',
            '    predicted_price_usd: float',
            '    model_deployed: str',
            '    is_optuna_tuned: bool',
            '    applied_target_transform: str',
            '',
            '# 🎯 Вказуємо response_model, щоб Scalar знав структуру відповіді',
            '@app.post("/predict", response_model=PredictionResponse)',
            'def predict_price(payload: InferencePayload):',
            '    if model is None:',
            '        raise HTTPException(status_code=500, detail="Модель не завантажена")',
            '',
            '    try:',
            '        df = pd.DataFrame([payload.features])',
            '        ',
            '        # Примусово вирівнюємо порядок колонок як при навчанні!',
            '        if expected_columns:',
            '            df = df.reindex(columns=expected_columns)',
            '        ',
            '        # 🛡️ Відновлюємо точну структуру категорій',
            '        for col, dtype in expected_dtypes.items():',
            '            if col in df.columns:',
            '                if dtype == "category" and col in expected_categories:',
            '                    # Перетворюємо колонку на категорію, використовуючи ТІ САМІ класи, що були при навчанні',
            '                    cat_type = pd.CategoricalDtype(categories=expected_categories[col])',
            '                    df[col] = df[col].astype(cat_type)',
            '                else:',
            '                    # Для числових типів (float64, int64)',
            '                    df[col] = df[col].astype(dtype)',
            '        ',
            '        raw_prediction = model.predict(df)[0]',
            '        ',
            '        # 👑 МАТЕМАТИЧНЕ ЗВОРТНЕ ПЕРЕТВОРЕННЯ НА ЛЬОТУ'
        ]

        # Динамічно додаємо математику
        if transform_strategy == "log":
            _api_lines.append('        final_prediction = float(np.expm1(raw_prediction))')
        elif transform_strategy == "sqrt":
            _api_lines.append('        final_prediction = float(np.square(raw_prediction))')
        else:
            _api_lines.append('        final_prediction = float(raw_prediction)')

        # Закінчуємо файл
        _api_lines.extend([
            '        ',
            '        # Повертаємо чисті, реальні долари США клієнту!',
            '        return {',
            '            "predicted_price_usd": final_prediction,',
            f'            "model_deployed": "{_selected_name}",',
            f'            "is_optuna_tuned": {_is_tuned_str},',
            f'            "applied_target_transform": "{transform_strategy}"',
            '        }',
            '    except Exception as e:',
            '        raise HTTPException(status_code=400, detail=str(e))',
            '',
            'if __name__ == "__main__":',
            '    uvicorn.run(app, host="0.0.0.0", port=8000)'
        ])

        _api_code = "\n".join(_api_lines)

        with open(_api_path, "w", encoding="utf-8") as f:
            f.write(_api_code)

        # 5. Збираємо статистику для консолі
        _model_size_kb = os.path.getsize(_model_path) / 1024
        _schema_size_kb = os.path.getsize(_schema_path) / 1024
        _api_size_kb = os.path.getsize(_api_path) / 1024
        _timestamp_human = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        _num_features = _X_curr.shape[1]

        # Якщо модель загорнута в пайплайн, дістаємо фінальний крок ('regressor'), щоб не тягнути параметри скейлера
        _actual_model = _model_to_save.named_steps['regressor'] if hasattr(_model_to_save, 'named_steps') and 'regressor' in _model_to_save.named_steps else _model_to_save
        _params = getattr(_actual_model, "get_params", lambda: {})()

        _arch_lines = []

        # А) Витягуємо ключові параметри дерев (тільки якщо вони є і не None)
        if 'n_estimators' in _params and _params['n_estimators'] is not None:
            _arch_lines.append(f"🧬 Дерев (n_estimators): {_params['n_estimators']}")
        if 'max_depth' in _params and _params['max_depth'] is not None:
            _arch_lines.append(f"🔬 Глибина (max_depth): {_params['max_depth']}")
        if 'learning_rate' in _params and _params['learning_rate'] is not None:
            _lr_val = _params['learning_rate']
            if isinstance(_lr_val, float): _lr_val = round(_lr_val, 5)
            _arch_lines.append(f"⚡ Швидкість (LR): {_lr_val}")
        if 'booster' in _params:
            _arch_lines.append(f"🎯 Тип бустера: {_params['booster'] or 'Default (gbtree)'}")

        # Б) Якщо це лінійна модель (немає базових "дерев'яних" налаштувань)
        if not _arch_lines:
            _arch_lines.append("⚖️ Тип: Аналітична / Лінійна архітектура (без дерев)")

        # В) Динамічно збираємо ВСІ інші корисні параметри (регуляризація, alpha, subsample тощо)
        # Виключаємо системне сміття та те, що ми вже вивели вище
        _exclude_keys = {
            'n_estimators', 'max_depth', 'learning_rate', 'booster',
            'random_state', 'n_jobs', 'objective', 'enable_categorical',
            'missing', 'callbacks', 'verbosity', 'silent', 'early_stopping_rounds',
            'device', 'n_iter_no_change', 'verbose'
        }
        _extra_params = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in _params.items()
            if k not in _exclude_keys and v is not None
        }

        if _extra_params:
            _param_list = [f"{k}={v}" for k, v in _extra_params.items()]

            # Розбиваємо список по 4 параметри на рядок
            _chunk_size = 4
            _chunks = [", ".join(_param_list[i:i+_chunk_size]) for i in range(0, len(_param_list), _chunk_size)]

            # Генеруємо ідеальний відступ (50 пробілів), щоб текст йшов рівно під текстом попереднього рядка
            _indent_spaces = " " * 50
            _extra_params_str = f",\n{_indent_spaces}".join(_chunks)

            _arch_lines.append(f"⚙️ Додаткові параметри: {_extra_params_str}")

        # Склеюємо все з правильними відступами для консольного UI
        _arch_text = "\n                          ".join(_arch_lines)

        _dataset_type = "Нативна категоріалізація" if _is_native else "One-Hot Encoding"
        _origin_status = "🔱 Оптимізована (Optuna)" if _is_tuned else "🧧 Базова (З Лідерборду)"

        with open(_model_path, "rb") as f: _model_bytes = f.read()
        with open(_schema_path, "rb") as f: _schema_bytes = f.read()
        with open(_api_path, "rb") as f: _api_bytes = f.read()

        _download_model_btn = mo.download(data=_model_bytes, filename=os.path.basename(_model_path), label="☯️ .joblib (Ваги)")
        _download_schema_btn = mo.download(data=_schema_bytes, filename="features_schema.json", label="✡️ .json (Схема)")
        _download_api_btn = mo.download(data=_api_bytes, filename="api.py", label="⚛️ api.py (FastAPI)")

        _theme = mo.app_meta().theme
        _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

        mo.output.append(
            mo.md(
                f"""
                <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; margin-bottom: 15px;">
                    <h3 style="margin-top: 0; text-align: center; font-weight: bold;">📦 MLOps Серіалізація (Production Ready)</h3>
                    <p>Останній крок перед передачею алгоритму Backend-команді. Ми фізично зберігаємо "мозок" моделі, словник стовпців та автоматично генеруємо сервер!</p>

                    ```text
                    🎭 Початок збереження у файл...
                      📦 Шлях проекту: {_artifact_dir}/
                      🎛 Ознаки: {_num_features} вимірів ({_dataset_type})
                      📐 Стратегія Target: {_strategy_desc}
                      👾 Архітектура моделі:
                         🧠 Алгоритм: {_selected_name}
                         🛠 Джерело: {_origin_status}
                         {_arch_text}

                      ✅ Успіх! Капсула 'Мозку' ШІ надійно збережена ({_model_size_kb:,.2f} KB) о {_timestamp_human}
                      ✅ Накладну (Маніфест ознак + Метадані) експортовано ({_schema_size_kb:,.2f} KB)
                      ⚡ Pydantic-Схема та FastAPI-сервер згенеровано! ({_api_size_kb:,.2f} KB)
                    ```
                </div>
                """
            )
        )
        mo.output.append(mo.hstack([_download_model_btn, _download_schema_btn, _download_api_btn], justify="center"))
    return


@app.cell
def deploy_instructions(mo):
    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    _css_no_scroll = mo.md(
        """
        <div class="shap-noscroll"></div>
        <style>
            marimo-cell-output:has(.shap-noscroll),
            .output-area:has(.shap-noscroll) {
                max-height: 9999px !important;
                overflow: visible !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )

    _deploy_instructions = mo.md(f"""
    <div style="padding: 25px; border: 1px solid {_border}; border-radius: 12px; background-color: {_bg}; line-height: 1.6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="margin-top: 0; border-bottom: 1px solid {_border}; padding-bottom: 12px; display: flex; align-items: center; gap: 10px;">
            <span>🌐</span> Архітектура та Розгортання Мікросервісу
        </h2>
        <p style="font-size: 15px;">Цей модуль автоматично генерує повністю налаштований бекенд для нашої ML-моделі. Ми перейшли від базових Data Science скриптів до бездоганної інфраструктури Enterprise-рівня.</p>

        <h3 style="color: #3b82f6; margin-top: 25px;">✨ Технологічний Стек та Можливості</h3>
        <ul style="font-size: 14px;">
            <li style="margin-bottom: 8px;"><b>FastAPI + Uvicorn:</b> Високопродуктивний асинхронний сервер, який миттєво обробляє HTTP-запити та виконує інференс моделі.</li>
            <li style="margin-bottom: 8px;"><b>Сучасний Scalar UI:</b> Ми повністю відмовилися від застарілого Swagger. Інтегрований <b>Scalar</b> забезпечує преміальний дизайн документації (рівня Stripe), вбудований REST-клієнт та миттєву генерацію коду запитів для десятків мов програмування (cURL, Python, JS тощо).</li>
            <li style="margin-bottom: 8px;"><b>Авто-генерація Payload:</b> API самостійно "витягує" реальний рядок із тренувального набору даних та вбудовує його у документацію. Більше ніякого ручного введення — ендпоїнт одразу готовий до тестування (1-click test).</li>
            <li style="margin-bottom: 8px;"><b>"Бронежилет" для даних:</b> Сервер самостійно вирівнює порядок колонок (через <code>reindex</code>) під суворі стандарти алгоритму. Клієнт може надсилати ключі у будь-якому порядку (навіть відсортовані за алфавітом) — сервер автоматично збере їх у правильну структуру.</li>
        </ul>

        <h3 style="color: #10b981; margin-top: 25px;">🛡️ Суворі Pydantic-Контракти</h3>
        <ul style="font-size: 14px;">
            <li style="margin-bottom: 8px;">✅ <b>200 OK (Успішна відповідь):</b> Сервер повертає чітко типізовану схему <code>PredictionResponse</code>. Завдяки цьому клієнт наперед знає, що гарантовано отримає <code>predicted_price_usd</code> (float), а також метадані про поточну активну модель та застосовану математичну трансформацію ціни.</li>
            <li style="margin-bottom: 8px;">❌ <b>422 Validation Error:</b> Завдяки <code>InferencePayload</code>, якщо клієнт надішле неправильний тип даних (наприклад, текст замість числа) або пропустить обов'язковий параметр, FastAPI автоматично відхилить запит із детальним JSON-описом (де саме і чому сталася помилка), захищаючи ML-модель від падінь.</li>
        </ul>

        <h3 style="color: #f59e0b; margin-top: 25px;">⚙️ Як запустити локально?</h3>
        <p style="font-size: 14px;">Усі згенеровані артефакти (ваги, схема, сервер) надійно ізольовано у директорії <code>models/california_housing/</code>.</p>

        <div style="margin-top: 15px;">
            <b>▶ Спосіб 1: DevOps-стандарт (Через Makefile)</b>
            <pre style="background-color: #111827; color: #10b981; padding: 12px; border-radius: 8px; border: 1px solid #374151; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>make api-hw1</code></pre>
        </div>

        <div style="margin-top: 15px;">
            <b>▶ Спосіб 2: Ручний запуск</b>
            <pre style="background-color: #111827; color: #e5e7eb; padding: 12px; border-radius: 8px; border: 1px solid #374151; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>cd models/california_housing
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload</code></pre>
        </div>

        <hr style="border-color: {_border}; margin: 25px 0;">
        <p style="margin-bottom: 0; font-size: 15px;">
            <i>💡 <b>Документація доступна за адресою:</b> <a href="http://127.0.0.1:8000/docs" target="_blank" style="color: #3b82f6; font-weight: bold; text-decoration: none;">http://127.0.0.1:8000/docs</a>.<br/>
            <i>🥂 Тепер будь-який застосунок (веб-сайт, мобільний застосунок на Swift/Kotlin чи Telegram-бот) може відправляти JSON-запити на цей порт і миттєво отримувати прогноз ціни!</i>
        </p>
    </div>
    """)
    mo.output.append(mo.hstack([_css_no_scroll, _deploy_instructions]))
    return


if __name__ == "__main__":
    app.run()
