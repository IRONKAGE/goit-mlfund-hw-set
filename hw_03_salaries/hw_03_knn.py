import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title_head_hw(mo):
    mo.md("""
    <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
        💼 ДЗ №3: Прогнозування зарплат <i>(Salaries Estimation)</i>
    </div>

    <h3 align="center"><b><u>Пайплайн</u>: PowerTransformer ➔ TargetEncoder ➔ 3D Corporate Twin Radar ➔ Optuna ➔ XAI ➔ FastAPI ➔ TimesFM</b></h3>

    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Machine Learning: Fundamentals and Applications [07.2026]</i></p>
    """)
    return


@app.cell
def configure_dependencies():
    import os
    import warnings
    import sys
    import contextlib
    import base64
    import json
    import html
    import textwrap
    import urllib.request
    from datetime import datetime

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("interpret").setLevel(logging.ERROR)
    from sklearn.exceptions import ConvergenceWarning

    # 🛡️ ПІДКЛЮЧЕННЯ АРХІТЕКТУРНОГО ЯДРА
    _core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
    if _core_path not in sys.path:
        sys.path.append(_core_path)

    from core import (
        SecureDownloader, smart_read_csv, get_hardware_config, clear_vram,
        set_global_seed, log_system_info, get_boosting_kwargs, logger
    )

    # 📍 ЛОКАЛЬНІ ІМПОРТИ
    from data_adapters import get_salary_mock
    from ui_labels import UA_COLUMNS

    # Data Science
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
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from data_profiling import ProfileReport

    import marimo as mo

    # MLOps & Machine Learning
    import mlflow
    import mlflow.sklearn
    import sklearn
    import optuna
    import joblib

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, PowerTransformer, TargetEncoder, OneHotEncoder
    from sklearn.model_selection import KFold
    from sklearn.inspection import permutation_importance
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn import set_config
    from sklearn.utils import estimator_html_repr

    # Метрики
    from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

    # Моделі
    from sklearn.dummy import DummyRegressor
    from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from interpret.glassbox import ExplainableBoostingRegressor

    from optuna.visualization import plot_optimization_history, plot_param_importances

    pd.options.mode.copy_on_write = True
    sklearn.set_config(transform_output="pandas")

    mo.center(mo.md("✅ **Бібліотеки, Локальні Адаптери та Ядро MLOps успішно імпортовано!**"))
    return (
        ColumnTransformer,
        ConvergenceWarning,
        DummyRegressor,
        ElasticNet,
        ExplainableBoostingRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        KFold,
        KNeighborsRegressor,
        LGBMRegressor,
        Lasso,
        NearestNeighbors,
        PCA,
        Pipeline,
        PowerTransformer,
        ProfileReport,
        RadiusNeighborsRegressor,
        RandomForestRegressor,
        Ridge,
        SimpleImputer,
        TargetEncoder,
        UA_COLUMNS,
        XGBRegressor,
        base64,
        clear_vram,
        contextlib,
        datetime,
        estimator_html_repr,
        get_boosting_kwargs,
        get_hardware_config,
        get_salary_mock,
        go,
        html,
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
        set_config,
        set_global_seed,
        shap,
        shap_tree,
        smart_read_csv,
        ssd,
        textwrap,
        urllib,
        warnings,
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
    logger,
    mlflow,
    mo,
    os,
    set_global_seed,
):
    # 🌱 1. Читаємо єдине джерело істини з .env
    GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", 42))

    # ⚙️ 2. Ініціалізація апаратного забезпечення та профайлінг системи
    with mo.status.spinner(title="Ініціалізація MLOps ядра..."):
        log_system_info()
        set_global_seed(GLOBAL_SEED)

        # Детектимо залізо для PyTorch (Нейромережі)
        device, device_ui_name = get_hardware_config(global_seed=GLOBAL_SEED)

        # Перекладаємо конфіги заліза для дерев (XGBoost/LightGBM)
        xgb_kwargs, lgbm_kwargs = get_boosting_kwargs(device)

        # 3. Налаштування MLflow для поточного завдання
        experiment_name = "hw03_salary_knn"
        mlflow.set_experiment(experiment_name)

        # 📝 Фіксуємо подію в глобальний аудит-лог
        logger.info(f"✅ Налаштовано експеримент MLflow: {experiment_name}")
    return GLOBAL_SEED, device, device_ui_name, lgbm_kwargs, xgb_kwargs


@app.cell(hide_code=True)
def header_prepare_dataset(mo):
    mo.md("""
    <h2 align='center'><b>💽 1. Завантаження даних та Smart EDA</b></h2>
    """)
    return


@app.cell
def execute_etl_pipeline(
    get_salary_mock,
    go,
    logger,
    make_subplots,
    mo,
    np,
    os,
    smart_read_csv,
    urllib,
):
    # 🎭 Динамічно знімаємо ліміт пам'яті Marimo через приватний API
    try:
        mo._runtime.context.get_context().marimo_config["runtime"][
            "output_max_bytes"
        ] = 50_000_000  # Ставимо 50 МБ для надійності
    except Exception:
        pass  # Запобіжник на випадок, якщо Marimo колись оновить архітектуру

    logger.info("Починаємо ETL-процес для HR Salary Dataset...")

    _train_url = "https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/refs/heads/main/datasets/mod_04_hw_train_data.csv"
    _valid_url = "https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/refs/heads/main/datasets/mod_04_hw_valid_data.csv"

    _data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(_data_path, exist_ok=True)

    _train_file = os.path.join(_data_path, "mod_04_hw_train_data.csv")
    _valid_file = os.path.join(_data_path, "mod_04_hw_valid_data.csv")

    with mo.status.spinner(title="Отримання даних, фільтрація та візуалізація..."):
        logger.info("🔍 Перевірка локальних файлів...")

        # 🌐 Елегантний роутинг даних (Завантаження або Генерація) з детальним логуванням
        for url, path, is_val, desc in [
            (_train_url, _train_file, False, "mod_04_hw_train_data.csv"),
            (_valid_url, _valid_file, True, "mod_04_hw_valid_data.csv")
        ]:
            if os.path.exists(path):
                logger.info(f"🔋 Знайдено готовий файл: {desc}. Пропускаємо мережевий запит...")
            else:
                try:
                    logger.info(f"🌐 Ініціалізація завантаження {desc} з GitHub...")
                    urllib.request.urlretrieve(url, path)
                    logger.info(f"✅ Файл {desc} успішно завантажено!")
                except Exception as e:
                    logger.warning(f"⚠️ Мережевий збій при зверненні до GitHub: {e}")
                    logger.info("🧪 Активація користувацького генератора даних (Fallback)...")
                    get_salary_mock(save_path=path, is_valid=is_val)
                    logger.info(f"✅ Синтетичний набір даних {desc} згенеровано успішно!")

        # ⚡ Читання сирих даних через Smart Reader
        df_train_raw = smart_read_csv(_train_file, "Train Data", engine="pyarrow")
        df_valid_raw = smart_read_csv(_valid_file, "Valid Data", engine="pyarrow")

        salary_raw = df_train_raw['Salary'].copy()

        # 🛡️ ПРАВИЛЬНИЙ Z-SCORE (Стійкий до пропусків NaN у реальних даних)
        _initial_len = len(df_train_raw)
        _num_cols = df_train_raw.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
        _df_num = df_train_raw[_num_cols]

        # Обчислюємо Z-оцінку через pandas (він безпечно ігнорує NaN)
        _z_scores = np.abs((_df_num - _df_num.mean()) / _df_num.std(ddof=0))

        # Залишаємо рядки, де ЖОДНА колонка не перевищує 3 сигми
        _mask = ~(_z_scores >= 3).any(axis=1)

        df_train_cleaned = df_train_raw[_mask].copy()
        _outliers_removed = _initial_len - len(df_train_cleaned)

        logger.info(f"🛡️ Z-Score сканування завершено. Видалено аномальних записів (викидів): {_outliers_removed}")

        salary_cleaned = df_train_cleaned['Salary'].copy()

        target_col = 'Salary'
        X_train = df_train_cleaned.drop(columns=[target_col, 'Hire_Date'], errors='ignore')
        y_train = df_train_cleaned[target_col]
        X_valid = df_valid_raw.drop(columns=[target_col, 'Hire_Date'], errors='ignore')
        y_valid = df_valid_raw[target_col]

        logger.info("🚀 Пайплайн ETL-процесу успішно завершено.")

        _theme = mo.app_meta().theme
        _template = "plotly_dark" if _theme == "dark" else "plotly_white"
        _text_color = "white" if _theme == "dark" else "#1f2937"
        _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
        _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

        _fig_eda = make_subplots(
            rows=1, cols=2,
            subplot_titles=["До фільтрації (Сирі дані)", "Після 3-Sigma Rule (Очищені)"],
        )

        _fig_eda.add_trace(
            go.Histogram(
                x=salary_raw,
                name="Сирі дані",
                marker_color='#ef4444',
                nbinsx=50,
                hovertemplate="<b>Діапазон ЗП:</b> %{x}$<br><b>Співробітників:</b> %{y}<extra></extra>"
            ),
            row=1, col=1
        )

        _fig_eda.add_trace(
            go.Histogram(
                x=salary_cleaned,
                name="Очищені",
                marker_color='#10b981',
                nbinsx=50,
                hovertemplate="<b>Діапазон ЗП:</b> %{x}$<br><b>Співробітників:</b> %{y}<extra></extra>"
            ),
            row=1, col=2
        )

        # Додавання підписів осей
        _fig_eda.update_xaxes(title_text="Заробітна плата ($)", row=1, col=1)
        _fig_eda.update_yaxes(title_text="Кількість співробітників", row=1, col=1)
        _fig_eda.update_xaxes(title_text="Заробітна плата ($)", row=1, col=2)
        _fig_eda.update_yaxes(title_text="Кількість співробітників", row=1, col=2)

        _fig_eda.update_layout(
            showlegend=False,
            template=_template,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                text="<b>Вплив Z-Score на розподіл заробітної плати</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=20, color=_text_color)
            ),
            height=450,
            margin=dict(t=70, b=40, l=40, r=20)
        )

    _css_no_scroll = mo.md(
        """
        <div class="marimo-noscroll-override"></div>
        <style>
            marimo-cell-output:has(.marimo-noscroll-override),
            .output-area:has(.marimo-noscroll-override) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )

    mo.output.append(mo.vstack([
        _css_no_scroll,
        mo.center(
            mo.md(f"✅ **Дані успішно завантажено та очищено!**<br>Тренувальний набір: `{X_train.shape}` рядків<br>Валідаційний набір: `{X_valid.shape}` рядків")
        ),
        mo.md(f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-top: 15px; margin-bottom: 15px;">
            <h3 style="margin-top: 0; color: #10b981;">🛡️ Математична фільтрація аномалій (3-Sigma Rule)</h3>
            <p>Багатовимірний простір проскановано. Успішно виявлено та видалено <b>{_outliers_removed}</b> аномальних записів (викидів), які відхилялися від середнього більше ніж на 3 стандартні відхилення (Z ≥ 3). Це критично важливо для захисту розрахунків Евклідової відстані в алгоритмі KNN від спотворення.</p>
            <h3 style="margin-top: 15px; color: #3b82f6;">🚀 Готовність до трансформування</h3>
            <p>Наступним кроком ми застосуємо <b>PowerTransformer</b> для вирівнювання розподілу цільової змінної та числових ознак, а також <b>TargetEncoder</b> для кодування категорій.</p>
        </div>
        """),
        mo.ui.plotly(_fig_eda)
    ]))
    return X_train, X_valid, df_train_raw, y_train, y_valid


@app.cell(hide_code=True)
def header_auto_eda(mo):
    mo.md("""
    <h3 align="center"><b>📊 1.1. Автоматичний EDA <i>(fg-data-profiling)</i></b></h3>
    """)
    return


@app.cell
def generate_eda_report(
    ProfileReport,
    UA_COLUMNS,
    contextlib,
    df_train_raw,
    html,
    mo,
    os,
    re,
):
    with mo.status.spinner(title="Генерація інтерактивного профайлінгу..."):
        df_eda = df_train_raw.copy()

        # 🌌 Відправляємо весь консольний спам у "чорну діру"
        with (
            open(os.devnull, "w") as fnull,
            contextlib.redirect_stdout(fnull),
            contextlib.redirect_stderr(fnull),
        ):
            profile = ProfileReport(
                df_eda,
                title="Salary Profiling Report",
                minimal=True,
                progress_bar=False,
            )
            html_string = profile.to_html()

            # Динамічна та безпечна назва артефакту (зберігаємо звіт у файл)
            artifact_dir = os.getenv("MODELS_DIR", "./models")
            os.makedirs(artifact_dir, exist_ok=True)
            report_filename = "hw03_salary_eda.html"
            profile.to_file(os.path.join(artifact_dir, report_filename))

    # Перейменовуємо колонки лише для візуалізації, зберігаючи оригінал недоторканим
    df_display = df_train_raw.rename(columns=UA_COLUMNS)

    # ⚡ Нативна таблиця з українськими заголовками
    df_explorer = mo.ui.table(df_display, selection=None, pagination=True)

    html_string = re.sub(
        r'<a\s+([^>]*)href=["\']#([^"\']+)["\']([^>]*)>',
        r'<a \1 href="javascript:void(0);" data-target="#\2" data-bs-target="#\2" onclick="var el=document.getElementById(\'\2\'); if(el) el.scrollIntoView({behavior: \'smooth\'});" \3>',
        html_string
    )

    # Замість Base64 використовуємо DOM-атрибут srcdoc
    safe_html = html.escape(html_string)

    dynamic_html = f"""
    <div class="marimo-noscroll-override"></div>
    <style>
        marimo-cell-output:has(.marimo-noscroll-override),
        .output-area:has(.marimo-noscroll-override) {{
            max-height: none !important;
            overflow: visible !important;
            overflow-y: visible !important;
        }}

        .smart-eda-iframe {{
            width: 100%;
            height: 850px;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            transition: filter 0.3s ease-in-out, border-color 0.3s ease;
        }}

        html.dark .smart-eda-iframe,
        .dark .smart-eda-iframe {{
            filter: invert(90%) hue-rotate(180deg) brightness(1.1);
            border-color: #333;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }}
    </style>

    <!-- Вставка звіту з дотриманням безпеки -->
    <iframe class="smart-eda-iframe" srcdoc="{safe_html}" sandbox="allow-scripts allow-same-origin"></iframe>
    """

    html_report = mo.Html(dynamic_html)

    mo.output.append(
        mo.vstack([
            mo.md("### 📊 Інтерактивний огляд даних"),
            df_explorer,
            mo.md("<div style='height: 15px;'></div>"),
            mo.center(mo.md("✅ **Профайлінг успішно згенеровано!** *(Ізольований фрейм готовий до виводу у наступній клітинці)*"))
        ])
    )
    return (html_report,)


@app.cell
def display_eda_report(html_report, mo):
    # 📜 Окремою клітинкою гарантовано виводимо звіт
    mo.output.append(html_report)
    return


@app.cell(hide_code=True)
def header_correlation(mo):
    mo.md("""
    <h3 align="center"><b>🧩 1.2. Аналіз мультиколінеарності <i>(Smart Correlation Matrix)</i></b></h3>
    """)
    return


@app.cell
def plot_correlation_matrix(
    UA_COLUMNS,
    X_train,
    mo,
    np,
    px,
    sch,
    ssd,
    y_train,
):
    # 1. Підготовка даних
    _df_corr = X_train.copy()
    _df_corr['Salary'] = y_train
    _num_df = _df_corr.select_dtypes(include=["float32", "float64", "int32", "int64"])
    _corr_matrix = _num_df.corr(method="pearson")

    # 2. Smart Clustering (Розумне групування кореляцій)
    _dists = 1 - np.abs(_corr_matrix.values)
    np.fill_diagonal(_dists, 0)
    _linkage = sch.linkage(ssd.squareform(_dists), method='ward')
    _opt_order = sch.leaves_list(_linkage)
    _corr_sorted = _corr_matrix.iloc[_opt_order, _opt_order]

    # 3. Переклад осей X та Y за допомогою нашого словника
    _corr_sorted = _corr_sorted.rename(columns=UA_COLUMNS, index=UA_COLUMNS)

    # 4. Налаштування UI теми
    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"

    # 5. Побудова інтерактивного графіка Plotly
    fig_corr = px.imshow(
        _corr_sorted,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
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
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    _insight_ui = mo.md(
        """
        <div class="corr-expanded-cell"></div>
        <style>
            marimo-cell-output:has(.corr-expanded-cell),
            .output-area:has(.corr-expanded-cell) {
                max-height: none !important;
                overflow: visible !important;
                overflow-y: visible !important;
            }
        </style>
        > **💡 Tech Lead Insight (Вплив простору на алгоритм KNN):**<br/>
        > Алгоритм **K-Nearest Neighbors (KNN)** приймає рішення на основі вимірювання Евклідової відстані у багатовимірному просторі. Тому на етапі трансформації даних критично важливо зробити дві речі:

        > 1. **Використати скалери** (наприклад, `StandardScaler`), щоб збалансувати вплив ознак із різними масштабами (одиниці років vs тисячі доларів)
        > 2. **Усунути мультиколінеарність**. Якщо незалежні змінні сильно корелюють між собою (наприклад, якби ми додали `Вік` та `Досвід роботи`), вони діятимуть як "подвійна вага", штучно розтягуючи простір уздовж своєї осі. Такі дублікати слід видаляти або застосовувати PCA
        """
    )

    mo.output.append(mo.vstack([fig_corr, _insight_ui]))
    return


@app.cell(hide_code=True)
def header_pipeline(mo):
    mo.md("""
    <h2 align='center'><b>🛠️ 2. Трансформаційний пайплайн <i>(PowerTransformer + TargetEncoder)</i></b></h2>
    """)
    return


@app.cell
def build_transformation_pipeline(
    ColumnTransformer,
    Pipeline,
    PowerTransformer,
    SimpleImputer,
    TargetEncoder,
    X_train,
    base64,
    estimator_html_repr,
    mo,
    set_config,
):
    set_config(display="diagram")
    _num_cols = X_train.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist()
    _cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    with mo.status.spinner("Побудова трансформаційного графа..."):
        # 1. Pipeline для числових
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('power_scaler', PowerTransformer(method='yeo-johnson'))
        ])

        # 2. Pipeline для категоріальних
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target_enc', TargetEncoder(target_type='continuous', smooth="auto"))
        ])

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, _num_cols),
            ('cat', cat_pipeline, _cat_cols)
        ])

    _insight_ui = mo.md(
        """
        > **💡 Tech Lead Insight (Трансформація для алгоритму KNN):**<br/>
        > Алгоритм K-Nearest Neighbors надзвичайно чутливий до масштабу ознак, оскільки він математично рахує геометричну відстань між точками.
        >
        > ⚡ Ми застосували **PowerTransformer (Yeo-Johnson)** замість звичайного StandardScaler. Він намагається зробити розподіл кожної числової ознаки Гаусовим (нормальним), що є ідеальною умовою для розрахунку Евклідової відстані<br/>
        > ⚡ Для категоріальних ознак ми застосували **TargetEncoder**. Він замінює, наприклад, відділ 'IT' на середню зарплату в цьому відділі, даючи алгоритму KNN ідеальну числову шкалу для обчислення відстаней між посадами
        """
    )

    _pipeline_vis = mo.vstack([
        mo.center(mo.md("### ⚙️ Архітектура Pipeline (Інтерактивний граф)")),
        mo.md("*Натисніть на блоки графа нижче, щоб розгорнути їх та переглянути налаштування параметрів.*"),
    ])

    _css_no_scroll = mo.md(
        """
        <div class="marimo-noscroll-override"></div>
        <style>
            marimo-cell-output:has(.marimo-noscroll-override),
            .output-area:has(.marimo-noscroll-override) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )

    # 1. Отримуємо сирий, 100% повний HTML безпосередньо від scikit-learn
    raw_html = estimator_html_repr(preprocessor)

    # 2. Обгортаємо у повноцінний HTML-документ (інакше iframe не відрендерить стилі)
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 20px;
                background-color: #ffffff; /* Жорстко задаємо білий фон */
                color: #000000;
                font-family: system-ui, -apple-system, sans-serif;
            }}
        </style>
    </head>
    <body>
        <div style="max-width: 800px; margin: 0 auto;">
            {raw_html}
        </div>
    </body>
    </html>
    """

    # 3. Кодуємо у Base64
    b64_encoded = base64.b64encode(full_html.encode('utf-8')).decode('utf-8')

    # 4. Виводимо через ізольований iframe з фоном
    _html_arch = mo.Html(
        f'<iframe src="data:text/html;base64,{b64_encoded}" '
        f'width="640px" height="450px" '
        f'style="border: 1px solid #4b5563; border-radius: 8px; background-color: #ffffff;">'
        f'</iframe>'
    )

    mo.output.append(
        mo.vstack([
            _css_no_scroll,
            _insight_ui,
            mo.center(_pipeline_vis),
            mo.center(
                _html_arch
            )
        ])
    )
    return (preprocessor,)


@app.cell(hide_code=True)
def header_radar(mo):
    mo.md("""
    <h2 align='center'><b>🌌 3. Corporate Twin Radar <i>(3D Локатор Сусідів)</i></b></h2>
    """)
    return


@app.cell
def wow_factor_twin_radar(
    NearestNeighbors,
    PCA,
    X_train,
    X_valid,
    go,
    mo,
    np,
    preprocessor,
    y_train,
    y_valid,
):
    with mo.status.spinner("Проекція багатовимірного простору у 3D та пошук сусідів..."):
        # 1. Трансформуємо дані для математики (Тренувальні та Валідаційні)
        _X_train_proc = preprocessor.fit_transform(X_train, y_train)
        _X_valid_proc = preprocessor.transform(X_valid)

        # 2. Зменшуємо розмірність до 3D для візуалізації
        pca_3d = PCA(n_components=3, random_state=42)
        X_train_3d = pca_3d.fit_transform(_X_train_proc)
        X_valid_3d = pca_3d.transform(_X_valid_proc)

        # 3. Навчаємо KNN для пошуку сусідів
        k_neighbors = 5
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        nn.fit(X_train_3d)

        # 4. Беремо випадкового співробітника-жертву з ВАЛІДАЦІЙНОГО набору
        target_idx = np.random.randint(0, len(X_valid_3d))
        target_point = X_valid_3d[target_idx] if isinstance(X_valid_3d, np.ndarray) else X_valid_3d.iloc[target_idx]
        actual_salary = y_valid.iloc[target_idx]

        # Шукаємо його "Близнюків" у тренувальному наборі
        distances, indices = nn.kneighbors([target_point])
        neighbor_indices = indices[0]

    # 🎨 Тематичне оформлення
    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"
    _grid_color = "rgba(255, 255, 255, 0.1)" if _theme == "dark" else "rgba(0, 0, 0, 0.1)"

    # Створюємо ефект "напівпрозорого скла" для стінок 3D-простору
    _pane_bg = "rgba(255, 255, 255, 0.04)" if _theme == "dark" else "rgba(0, 0, 0, 0.04)"

    _fig = go.Figure()

    # Всі тренувальні співробітники (фон)
    _y_train_vals = y_train.values if hasattr(y_train, 'values') else y_train

    _fig.add_trace(go.Scatter3d(
        x=X_train_3d[:, 0] if isinstance(X_train_3d, np.ndarray) else X_train_3d.iloc[:, 0],
        y=X_train_3d[:, 1] if isinstance(X_train_3d, np.ndarray) else X_train_3d.iloc[:, 1],
        z=X_train_3d[:, 2] if isinstance(X_train_3d, np.ndarray) else X_train_3d.iloc[:, 2],
        mode='markers', marker=dict(size=3, color='gray', opacity=0.1),
        name='Корпоративний простір (Train)',
        # Видаляємо hoverinfo='skip' і додаємо дані для підказки:
        text=[f"&#36;{val:,.0f}" for val in _y_train_vals],
        hovertemplate="<b>Базовий Працівник</b><br>Зарплата: %{text}<extra></extra>"
    ))

    # Цільовий співробітник (Валідація)
    _fig.add_trace(go.Scatter3d(
        x=[target_point[0]], y=[target_point[1]], z=[target_point[2]],
        mode='markers', marker=dict(size=12, color='#f59e0b', symbol='diamond', line=dict(color='white', width=2)),
        name='Цільовий Працівник (Valid)',
        hovertemplate="<b>Цільовий Працівник</b><br>Фактична ЗП: &#36;%s<extra></extra>" % f"{actual_salary:,.0f}"
    ))

    # К-найближчих сусідів
    _X_neighbors = X_train_3d[neighbor_indices] if isinstance(X_train_3d, np.ndarray) else X_train_3d.iloc[neighbor_indices]
    _y_neighbors = y_train.iloc[neighbor_indices].values
    _predicted_salary = np.mean(_y_neighbors)

    _fig.add_trace(go.Scatter3d(
        x=_X_neighbors[:, 0] if isinstance(_X_neighbors, np.ndarray) else _X_neighbors.iloc[:, 0],
        y=_X_neighbors[:, 1] if isinstance(_X_neighbors, np.ndarray) else _X_neighbors.iloc[:, 1],
        z=_X_neighbors[:, 2] if isinstance(_X_neighbors, np.ndarray) else _X_neighbors.iloc[:, 2],
        mode='markers', marker=dict(size=8, color='#3b82f6'),
        name=f'Топ-{k_neighbors} Близнюків (Train)',
        hovertemplate="<b>Знайдений Сусід</b><br>Зарплата: %{text}<extra></extra>",
        text=[f"&#36;{val:,.0f}" for val in _y_neighbors]
    ))

    # Лазерні промені
    _lines_x, _lines_y, _lines_z = [], [], []
    for _n_pt in (_X_neighbors if isinstance(_X_neighbors, np.ndarray) else _X_neighbors.values):
        _lines_x.extend([target_point[0], _n_pt[0], None])
        _lines_y.extend([target_point[1], _n_pt[1], None])
        _lines_z.extend([target_point[2], _n_pt[2], None])

    _fig.add_trace(go.Scatter3d(
        x=_lines_x, y=_lines_y, z=_lines_z, mode='lines',
        line=dict(color='#10b981', width=3, dash='dash'),
        name='Вектори математичної відстані', hoverinfo='skip'
    ))

    _fig.update_layout(
        # 1. Глобальна тема Plotly (автоматично підлаштовує багато дрібниць)
        template="plotly_dark" if _theme == "dark" else "plotly_white",

        # 2. Фікс MathJax: Використовуємо HTML-код долара (&#36;) замість символу $
        title=dict(
            text=f"<b>Corporate Twin Radar<br>Прогноз ЗП: &#36;{_predicted_salary:,.0f} vs Факт: &#36;{actual_salary:,.0f}</b>",
            x=0.5
        ),
        font=dict(color=_text_color),

        # 3. Адаптивна сітка координат з ВИМКНЕНИМ фоном стінок (showbackground=False)
        scene=dict(
            xaxis=dict(title="PCA 1", gridcolor=_grid_color, showbackground=False, zerolinecolor=_grid_color),
            yaxis=dict(title="PCA 2", gridcolor=_grid_color, showbackground=False, zerolinecolor=_grid_color),
            zaxis=dict(title="PCA 3", gridcolor=_grid_color, showbackground=False, zerolinecolor=_grid_color),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
    )

    _css_no_scroll = mo.md(
        """
        <div class="marimo-noscroll-override"></div>
        <style>
            marimo-cell-output:has(.marimo-noscroll-override),
            .output-area:has(.marimo-noscroll-override) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )
    _insight = mo.md(f"""
    > 🧠 **Як мислить алгоритм KNN:**<br/>
    > **Алгоритм KNeighborsRegressor ідеально прозорий:** він проектує нового співробітника у багатовимірний простір і шукає $K$ ({k_neighbors}) історичних записів, чий корпоративний "відбиток" максимально схожий на нього<br/>
    > **Прогнозована зарплата** — це середнє арифметичне зарплат цих знайдених "близнюків". На радарі жовтий ромб — це реальний співробітник з валідаційної вибірки, а зелені промені вказують на його сусідів з тренувального набору даних
    """)

    mo.output.append(mo.vstack([_css_no_scroll, mo.ui.plotly(_fig), _insight]))
    return


@app.cell(hide_code=True)
def header_model_configurator(mo):
    mo.md("""
    <h2 align='center'><b>🕹️ 4. Конфігурація пулу алгоритмів <i>(Algorithm Pool)</i></b></h2>
    """)
    return


@app.cell
def model_data_state(
    DummyRegressor,
    ElasticNet,
    ExplainableBoostingRegressor,
    ExtraTreesRegressor,
    GLOBAL_SEED,
    GradientBoostingRegressor,
    KNeighborsRegressor,
    LGBMRegressor,
    Lasso,
    RadiusNeighborsRegressor,
    RandomForestRegressor,
    Ridge,
    XGBRegressor,
    lgbm_kwargs,
    mo,
    xgb_kwargs,
):
    # 1. Архітектури сусідів
    models_knn = {
        "KNN (K=3, Euclidean)": (1, KNeighborsRegressor(n_neighbors=3, metric='euclidean', n_jobs=-1)),
        "KNN (K=5, Euclidean)": (2, KNeighborsRegressor(n_neighbors=5, metric='euclidean', n_jobs=-1)),
        "KNN (K=10, Manhattan)": (3, KNeighborsRegressor(n_neighbors=10, metric='manhattan', weights='distance', n_jobs=-1)),
        "KNN (K=15, Minkowski p=3)": (4, KNeighborsRegressor(n_neighbors=15, metric='minkowski', p=3, n_jobs=-1)),
        "Radius Neighbors (R=2.0)": (5, RadiusNeighborsRegressor(radius=2.0, n_jobs=-1))
    }

    # 2. Лінійні моделі
    models_linear = {
        "Dummy (Mean Baseline)": (6, DummyRegressor(strategy="mean")),
        "Ridge Regressor (L2)": (7, Ridge(alpha=1.0, random_state=GLOBAL_SEED)),
        "Lasso Regressor (L1)": (8, Lasso(alpha=0.01, random_state=GLOBAL_SEED)),
        "ElasticNet Regressor": (9, ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=GLOBAL_SEED))
    }

    # 3. Ансамблі та Бустинг
    models_ensembles = {
        "Random Forest": (10, RandomForestRegressor(n_estimators=100, random_state=GLOBAL_SEED, n_jobs=-1)),
        "Extra Trees": (11, ExtraTreesRegressor(n_estimators=100, random_state=GLOBAL_SEED, n_jobs=-1)),
        "Gradient Boosting": (12, GradientBoostingRegressor(n_estimators=100, random_state=GLOBAL_SEED)),
        "XGBoost": (13, XGBRegressor(n_estimators=100, random_state=GLOBAL_SEED, **xgb_kwargs)),
        "LightGBM": (14, LGBMRegressor(random_state=GLOBAL_SEED, **lgbm_kwargs)),
        "Explainable Boosting (EBM)": (15, ExplainableBoostingRegressor(random_state=GLOBAL_SEED, n_jobs=-1))
    }

    master_registry = {**models_knn, **models_linear, **models_ensembles}
    id_to_name_map = {f"#{mod_id:02d}": name for name, (mod_id, _) in master_registry.items()}

    get_force_knn, set_force_knn = mo.state(True)
    get_force_lin, set_force_lin = mo.state(True)
    get_force_ens, set_force_ens = mo.state(True)

    mo.center(mo.md(f"✅ **Всі {len(master_registry)} алгоритмів (з акцентом на архітектури сусідів) завантажено у памʼять!**"))
    return (
        get_force_ens,
        get_force_knn,
        get_force_lin,
        id_to_name_map,
        master_registry,
        models_ensembles,
        models_knn,
        models_linear,
        set_force_ens,
        set_force_knn,
        set_force_lin,
    )


@app.cell
def controller_ui(
    get_force_ens,
    get_force_knn,
    get_force_lin,
    mo,
    models_ensembles,
    models_knn,
    models_linear,
):
    force_knn = get_force_knn()
    force_lin = get_force_lin()
    force_ens = get_force_ens()

    mandatory_models = [
        "Dummy (Mean Baseline)",
        "KNN (K=5, Euclidean)",
        "XGBoost",
        "Explainable Boosting (EBM)"
    ]

    def make_cb(name, force_state):
        is_locked = name in mandatory_models
        return mo.ui.checkbox(label=name, value=True if is_locked else force_state, disabled=is_locked)

    ui_knn = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_knn) for name, (mod_id, _) in models_knn.items()})
    ui_lin = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_lin) for name, (mod_id, _) in models_linear.items()})
    ui_ens = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_ens) for name, (mod_id, _) in models_ensembles.items()})

    mo.center(mo.md("✅ **Словники інтерфейсу створено!**"))
    return ui_ens, ui_knn, ui_lin


@app.cell
def view_render(
    mo,
    set_force_ens,
    set_force_knn,
    set_force_lin,
    ui_ens,
    ui_knn,
    ui_lin,
):
    def build_group_view(ui_dict, title, set_state_fn):
        vals = ui_dict.value
        total = len(vals)
        completed = sum(vals.values())

        if completed == total and total > 0:
            icon_char = "✅"
            current_state_is_true = True
        else:
            icon_char = "☑️" if completed > 0 else "🔲"
            current_state_is_true = False

        def toggle_all(_): set_state_fn(not current_state_is_true)

        icon_button = mo.ui.button(label=icon_char, on_click=toggle_all, kind="neutral")
        header = mo.center(mo.hstack([icon_button, mo.md(f"**<span style='font-size: 1.05em;'>{title} ({completed}/{total})</span>**")], align="center"))
        return header, completed, total, icon_button

    h_knn, c_knn, t_knn, _ = build_group_view(ui_knn, "KNN Варіації 👯", set_force_knn)
    h_lin, c_lin, t_lin, _ = build_group_view(ui_lin, "Лінійні моделі 📉", set_force_lin)
    h_ens, c_ens, t_ens, _ = build_group_view(ui_ens, "Ансамблі/Бустинг 🌳", set_force_ens)

    total_selected = c_knn + c_lin + c_ens
    total_all = t_knn + t_lin + t_ens

    main_header = mo.hstack([
        mo.md("🎛️ **Конфігуратор архітектур (A/B Тестування)**"),
        mo.md(f"<div style='text-align: right; color: #10b981; font-size: 1.1em;'><b>✓ Всього обрано: {total_selected} / {total_all}</b></div>")
    ], justify="space-between", align="center")

    run_btn = mo.ui.run_button(label="🎭 Запустити тренування", kind="success")
    v_line = mo.Html("<div style='width: 1px; background-color: #4b5563; min-height: 240px; margin: 0 15px; margin-top: 15px;'></div>")

    def build_column(header, ui_group):
        items_with_ids = [mo.hstack([mo.md(f"`{k}`"), cb], align="center") for k, cb in ui_group.items()]
        return mo.vstack([header, mo.md("<div style='height: 10px;'></div>"), mo.vstack(items_with_ids, align="start")], align="center")

    _css_no_scroll = mo.md('<div class="config-noscroll"></div><style>marimo-cell-output:has(.config-noscroll),.output-area:has(.config-noscroll){max-height: none !important; overflow-y: visible !important; overflow-x: visible !important;}</style>')

    config_panel = mo.vstack([
        _css_no_scroll,
        mo.center(main_header),
        mo.hstack([
            build_column(h_knn, ui_knn), v_line,
            build_column(h_lin, ui_lin), v_line,
            build_column(h_ens, ui_ens)
        ], justify="space-between", align="start"),
        mo.center(run_btn)
    ])

    mo.output.append(config_panel)
    return (run_btn,)


@app.cell(hide_code=True)
def header_benchmark(mo):
    mo.md("""
    <h2 align='center'><b>🏋️‍♂️ 5. Тренування та Лідерборд <i>(MAPE Focus)</i></b></h2>
    """)
    return


@app.cell
def execute_benchmark(
    ConvergenceWarning,
    X_train,
    X_valid,
    clear_vram,
    go,
    id_to_name_map,
    logger,
    logging,
    make_subplots,
    master_registry,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mo,
    pa,
    pd,
    pl,
    preprocessor,
    r2_score,
    run_btn,
    ui_ens,
    ui_knn,
    ui_lin,
    warnings,
    xgb_kwargs,
    y_train,
    y_valid,
):
    mo.stop(not run_btn.value, mo.center(mo.md("### ⏳ Очікування конфігурації...\n> 🆘 Оберіть алгоритми у Конфігураторі вище та натисніть зелену кнопку.")))

    # 🧯 Придушення зайвих попереджень та налаштування логера
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", module="interpret.*")
    logging.getLogger("interpret").setLevel(logging.ERROR)

    logger.info("Початок бенчмаркінгу обраних моделей...")

    # 🪎 Динамічне визначення апаратного забезпечення (Hardware)
    _hw_type = xgb_kwargs.get("device", "cpu") if xgb_kwargs else "cpu"
    if _hw_type == "cuda":
        _hw_ui = "CUDA GPU"
    elif _hw_type == "sycl":
        _hw_ui = "Intel XPU"
    else:
        _hw_ui = "Multi-core CPU"

    selected_names = []
    for ui_group in [ui_knn, ui_lin, ui_ens]:
        selected_names.extend([id_to_name_map[mod_id] for mod_id, is_sel in ui_group.value.items() if is_sel])

    mo.stop(not selected_names, mo.md("⚠️ Неможливо запустити: не обрано жодного алгоритму!"))

    # 1. Трансформація даних через єдиний граф
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_valid_proc = preprocessor.transform(X_valid)

    results = []
    trained_models = {}

    total_models = len(selected_names)

    # 📊 Прогрес-бар з інформацією про залізо
    with mo.status.progress_bar(total=total_models, title=f"Тренування {total_models} моделей...", subtitle=f"💎 <b>Engine:</b> {_hw_ui} <br/>⏳ Ініціалізація...", remove_on_exit=True) as bar:
        for name in selected_names:
            bar.update(increment=0, subtitle=f"💎 <b>Engine:</b> {_hw_ui} <br/>☣️ <b>Тренуємо:</b> {name}")
            mod_id, model = master_registry[name]
            try:
                model.fit(X_train_proc, y_train)
                y_pred = model.predict(X_valid_proc)

                trained_models[name] = model
                results.append({
                    "ID": f"#{mod_id:02d}",
                    "Алгоритм": name,
                    "R-квадрат (R²) ⬆️": r2_score(y_valid, y_pred),
                    "MAE ($) ⬇️": mean_absolute_error(y_valid, y_pred),
                    "MAPE (%) ⬇️": mean_absolute_percentage_error(y_valid, y_pred) * 100
                })
            except Exception as e:
                logger.error(f"Помилка при тренуванні {name}: {e}")
            bar.update()

    logger.info(f"Бенчмаркінг {total_models} моделей завершено.")

    df_results = pd.DataFrame(results).sort_values(by="MAPE (%) ⬇️", ascending=True).reset_index(drop=True).copy()

    # Зберігаємо строгі типи для коректного малювання гістограм у таблиці
    df_results["R-квадрат (R²) ⬆️"] = df_results["R-квадрат (R²) ⬆️"].round(4)
    df_results["MAE ($) ⬇️"] = df_results["MAE ($) ⬇️"].round(0).astype(int)
    df_results["MAPE (%) ⬇️"] = df_results["MAPE (%) ⬇️"].round(2)

    clear_vram(None)

    _theme = mo.app_meta().theme
    _text = "white" if _theme == "dark" else "#1f2937"
    _bg = "rgba(0,0,0,0)"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    # =========================================================================
    # 🗄️ 5-РІВНЕВА СИСТЕМА ВІДОБРАЖЕННЯ ТАБЛИЦІ (Tier 1-5 Fallback)
    # =========================================================================
    _display_data = None
    _ui_mode = "Unknown"

    try:
        _display_data = pl.from_pandas(df_results)
        _ui_mode = "🥇 Tier 1: Polars Engine"
    except Exception:
        try:
            _display_data = pa.Table.from_pandas(df_results)
            _ui_mode = "🥈 Tier 2: PyArrow Engine"
        except Exception:
            try:
                _display_data = df_results.to_dict(orient="list")
                _ui_mode = "🥉 Tier 3: Column-Dict Fallback"
            except Exception:
                _display_data = df_results.to_dict(orient="records")
                _ui_mode = "🛡️ Tier 4: Safe-Records Dict"

    _justify_config = {col: "center" for col in df_results.columns}

    try:
        _benchmark_table = mo.ui.table(
            _display_data,
            selection=None,
            page_size=50,
            text_justify_columns=_justify_config,
            label=f"🏆 **Лідерборд результатів (Рушій UI: {_ui_mode} | Обчислення: {_hw_ui}):**"
        )
    except Exception as critical_err:
        _ui_mode = "🛟 Tier 5: Static HTML (Critical Fallback)"
        _benchmark_table = mo.vstack([
            mo.md(f"🏆 **Лідерборд результатів (Рушій UI: {_ui_mode} | Обчислення: {_hw_ui}):**"),
            mo.md(f"> ⚠️ *Інтерактивний рушій недоступний. Активовано резервний режим (HTML).*<br/>> <sub style='color: gray;'>Деталі збою: {str(critical_err)}</sub>"),
            mo.Html(df_results.to_html(justify="center", index=False))
        ])

    # =========================================================================
    # 📝 ДИНАМІЧНІ ВИСНОВКИ (INSIGHTS)
    # =========================================================================
    _dynamic_bullets = []
    if any(ui_lin.value.values()):
        _dynamic_bullets.append("> 1. **Лінійні моделі:** Зазвичай показують найгірший результат на складних нелінійних даних, оскільки намагаються провести пряму лінію через складну \"хмару\" залежностей.")
    if any(ui_knn.value.values()):
        _dynamic_bullets.append("> 2. **Архітектури сусідів (KNN):** Добре знаходять локальні патерни (схожих співробітників), але можуть сильно помилятися на викидах (аномально високих або низьких зарплатах).")
    if any(ui_ens.value.values()):
        _dynamic_bullets.append("> 3. **Ансамблі та Бустинг:** Найпотужніші алгоритми, здатні вивчити багаторівневі правила. Саме вони зазвичай \"збирають\" точки найближче до ідеальної діагоналі.")

    _insights_text = "\n".join(_dynamic_bullets)

    _benchmark_insight = mo.md(
        "> **📊 Як читати метрики лідерборду (Регресія):**\n"
        "\n"
        "> - **MAPE (%) ⬇️:** Головна бізнес-метрика. Показує, на скільки відсотків у середньому модель помиляється в прогнозі зарплати.\n"
        "> - **MAE ($) ⬇️:** Скільки доларів у середньому ми недоплачуємо або переплачуємо при прогнозі.\n"
        "> - **R-квадрат (R²) ⬆️:** Чим ближче до 1.0, тим краще модель розуміє логіку формування зарплат (0.0 означає сліпе вгадування середнього значення).\n"
        "\n"
        "> **💡 Tech Lead Insight (Аналіз розкиду):**\n"
        "\n"
        "> Подивіться на графіки нижче. Ідеальна модель повинна вибудувати всі крапки чітко вздовж пунктирної діагоналі. Чим сильніше крапки \"розлітаються\" в сторони — тим більша похибка.\n"
        "\n"
        f"{_insights_text}"
    )

    # =========================================================================
    # 📊 ВІЗУАЛІЗАЦІЯ БАЗОВИХ МОДЕЛЕЙ (Обов'язкові до порівняння)
    # =========================================================================
    mandatory_to_plot = [
        {"name": "KNN (K=5, Euclidean)", "title": "👯 Класичний Підхід (Сусіди)"},
        {"name": "XGBoost", "title": "🫀 Сучасний Підхід (Бустинг)"}
    ]

    subplot_titles = []
    plot_data_cache = {}

    for conf in mandatory_to_plot:
        model_name = conf["name"]
        if model_name in trained_models:
            row = df_results[df_results["Алгоритм"] == model_name]
            r2 = row["R-квадрат (R²) ⬆️"].iloc[0]
            mae = row["MAE ($) ⬇️"].iloc[0]
            mape = row["MAPE (%) ⬇️"].iloc[0]
            title = f"{conf['title']}<br><span style='font-size:12px; color:gray;'>{model_name}</span><br><span style='font-size:13px;'>R²: {r2:.4f} | MAE: &#36;{mae:,.0f} | MAPE: {mape:.2f}%</span>"
            plot_data_cache[model_name] = trained_models[model_name].predict(X_valid_proc)
        else:
            title = f"{conf['title']}<br><span style='font-size:12px; color:gray;'>{model_name}</span><br>❌ <i>Вимкнено</i>"
            plot_data_cache[model_name] = None
        subplot_titles.append(title)

    _fig_diag = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles, horizontal_spacing=0.1)

    max_val_global = y_valid.max()
    for idx, conf in enumerate(mandatory_to_plot):
        col_idx = idx + 1
        model_name = conf["name"]
        preds = plot_data_cache[model_name]

        if preds is not None:
            curr_max = max(y_valid.max(), preds.max())
            if curr_max > max_val_global: max_val_global = curr_max

            _fig_diag.add_trace(go.Scattergl(
                x=y_valid, y=preds, mode="markers",
                marker=dict(color="#3b82f6" if col_idx == 1 else "#10b981", size=6, opacity=0.6),
                name=model_name, showlegend=False,
                hovertemplate="Факт: &#36;%{x:,.0f}<br>Прогноз: &#36;%{y:,.0f}<extra></extra>"
            ), row=1, col=col_idx)

            _fig_diag.add_trace(go.Scatter(
                x=[0, curr_max], y=[0, curr_max], mode="lines",
                line=dict(color=_text, dash="dash", width=1.5),
                showlegend=False, hoverinfo="skip"
            ), row=1, col=col_idx)
        else:
             _fig_diag.add_annotation(text="⚠️ Модель відсутня", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"), row=1, col=col_idx)

        _fig_diag.update_xaxes(title_text="Фактична ЗП ($)", gridcolor=_border, row=1, col=col_idx)
        _fig_diag.update_yaxes(title_text="Прогнозована ЗП ($)", gridcolor=_border, row=1, col=col_idx)

    _fig_diag.update_layout(
        title=dict(text="<b>Еволюція Алгоритмів: Здатність моделі зрозуміти дані</b>", x=0.5, xanchor="center", y=0.98, font=dict(color=_text, size=18)),
        paper_bgcolor=_bg, plot_bgcolor=_bg, font=dict(color=_text), height=500, margin=dict(t=100, b=40, l=40, r=40)
    )

    # =========================================================================
    # 👑 ГРАФІК АБСОЛЮТНОГО ЧЕМПІОНА
    # =========================================================================
    champion_name = df_results["Алгоритм"].iloc[0]
    champ_pred = trained_models[champion_name].predict(X_valid_proc)

    row_champ = df_results[df_results["Алгоритм"] == champion_name]
    r2_champ = row_champ["R-квадрат (R²) ⬆️"].iloc[0]
    mae_champ = row_champ["MAE ($) ⬇️"].iloc[0]
    mape_champ = row_champ["MAPE (%) ⬇️"].iloc[0]

    fig_champ = go.Figure()
    fig_champ.add_trace(go.Scattergl(
        x=y_valid, y=champ_pred, mode="markers",
        marker=dict(color="#8b5cf6", size=10, opacity=0.7, line=dict(color="white", width=1)),
        name="Прогноз моделі",
        hovertemplate="Фактична ЗП: &#36;%{x:,.0f}<br>Прогноз моделі: &#36;%{y:,.0f}<extra></extra>"
    ))

    max_val = max(y_valid.max(), champ_pred.max())
    fig_champ.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines",
        line=dict(color=_text, dash="dash", width=2),
        name="Ідеальний прогноз", hoverinfo="skip"
    ))

    fig_champ.update_layout(
        title=dict(text=f"<b>👑 АБСОЛЮТНИЙ ЧЕМПІОН: {champion_name}</b><br><span style='font-size:15px; color:gray;'>R-квадрат: {r2_champ:.4f} | MAE: &#36;{mae_champ:,.0f} | MAPE: {mape_champ:.2f}%</span>", x=0.5, xanchor="center", y=0.92, font=dict(color=_text, size=22)),
        xaxis_title="Справжня Зарплата ($)", yaxis_title="Прогнозована Зарплата ($)",
        paper_bgcolor=_bg, plot_bgcolor=_bg, font=dict(color=_text), height=550, margin=dict(t=90, b=40, l=40, r=40)
    )

    fig_champ.update_xaxes(gridcolor=_border)
    fig_champ.update_yaxes(gridcolor=_border)

    _css_no_scroll = mo.md(
        """
        <div class="marimo-noscroll-override"></div>
        <style>
            marimo-cell-output:has(.marimo-noscroll-override),
            .output-area:has(.marimo-noscroll-override) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )

    mo.output.append(_css_no_scroll)
    mo.output.append(_benchmark_insight)
    mo.output.append(_benchmark_table)
    mo.output.append(_fig_diag)
    mo.output.append(fig_champ)
    return X_train_proc, df_results, trained_models


@app.cell(hide_code=True)
def header_model_selector_ui(mo):
    mo.md("""
    <h3 align='center'><b>🎛️ 5.1. Інтерактивний Селектор Моделі <i>(Model Selection)</i></b></h3>
    """)
    return


@app.cell
def model_selector_ui(df_results, master_registry, mo):
    _ranked_names = df_results["Алгоритм"].tolist()
    _dropdown_options = {}
    _split_idx = min(5, len(_ranked_names))
    _top_5 = _ranked_names[:_split_idx]
    _rest = _ranked_names[_split_idx:]

    _medals = ["🥇", "🥈", "🥉", "🏵️", "🏵️"]
    for _i, _name in enumerate(_top_5):
        _mod_id = master_registry[_name][0]
        _dropdown_options[f"{_medals[_i]} #{_mod_id:02d} {_name}"] = _name

    if _rest:
        _dropdown_options["─── 👇 ІНШІ ТРЕНОВАНІ АЛГОРИТМИ 👇 ───"] = _top_5[0]

    for _name in _rest:
        _mod_id = master_registry[_name][0]
        _dropdown_options[f"🎗️ #{_mod_id:02d} {_name}"] = _name

    _default_key = list(_dropdown_options.keys())[0]

    champion_selector = mo.ui.dropdown(
        options=_dropdown_options,
        value=_default_key,
        label="🏆 **Оберіть алгоритм для Оптимізації та XAI:** "
    )

    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"

    ui_card = mo.md(
        f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
            <h3 style="margin-top: 0;">🎛️ Інтерактивний центр аналізу (XAI)</h3>
            <p>Завдяки реактивності Marimo, <b>усі наступні графіки та пайплайн Optuna автоматично перебудуються</b> під ваш вибір!</p>
            <div style="display: flex; justify-content: center; align-items: center; margin-top: 15px;">
                {champion_selector}
            </div>
        </div>
        """
    )
    mo.output.append(ui_card)
    return (champion_selector,)


@app.cell(hide_code=True)
def header_optuna(mo):
    mo.md("""
    <h3 align='center'><b>🧪 5.2. Байєсівська оптимізація <i>(Optuna + MLflow)</i></b></h3>
    """)
    return


@app.cell
def optuna_ui_controls(champion_selector, mo):
    _selected_name = champion_selector.value
    _is_tunable = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree", "KNN"])

    trials_slider = mo.ui.slider(start=3, stop=50, step=1, value=10, show_value=True, label="🛝 **Кількість ітерацій (n_trials):**")
    run_optuna_btn = mo.ui.run_button(label=f"💝 Запустити тюнінг для {_selected_name}", kind="success", disabled=not _is_tunable)

    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    mo.output.append(mo.md(f"""
    <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; text-align: center;">
        <p><i>Оптимізація гіперпараметрів TPE для мінімізації метрики MAPE.</i></p>
        {trials_slider}<br/><br/>{run_optuna_btn}
    </div>
    """))
    return run_optuna_btn, trials_slider


@app.cell
def optuna_execution(
    GLOBAL_SEED,
    KFold,
    KNeighborsRegressor,
    RandomForestRegressor,
    XGBRegressor,
    X_train_proc,
    champion_selector,
    logger,
    logging,
    mean_absolute_percentage_error,
    mlflow,
    mo,
    np,
    optuna,
    os,
    pd,
    plot_optimization_history,
    plot_param_importances,
    run_optuna_btn,
    style_dataframe,
    trials_slider,
    xgb_kwargs,
    y_train,
):
    _selected_name = champion_selector.value
    _is_tunable = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree", "KNN"])
    final_tuned_model = None

    if run_optuna_btn.value and _is_tunable:
        # Глушимо MLflow, щоб він не панікував через відсутність pip у деяких середовищах
        logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
        logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

        # 🪎 Визначаємо красиве ім'я заліза для UI
        _hw_type = xgb_kwargs.get("device", "cpu") if xgb_kwargs else "cpu"
        if _hw_type == "cuda":
            _hw_ui = "CUDA GPU 🟢"
        elif _hw_type == "sycl":
            _hw_ui = "Intel XPU 🔵"
        else:
            _hw_ui = "Multi-core CPU ⚙️"

        with mo.status.progress_bar(
            total=trials_slider.value,
            title=f"🦄 Тюнінг {_selected_name}",
            subtitle="⏳ Ініціалізація алгоритмів...",
            remove_on_exit=True
        ) as _bar:

            os.makedirs("mlruns", exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///mlruns/mlruns.db")
            mlflow.set_experiment("Salary_Optimization")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def _objective(trial):
                _bar.update(increment=0, subtitle=f"🏃‍♂️ Ітерація {trial.number + 1} з {trials_slider.value}: Навчання 3-х фолдів...")

                if "KNN" in _selected_name:
                    params = {
                        "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
                        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                        "p": trial.suggest_int("p", 1, 3)
                    }
                    model_opt = KNeighborsRegressor(**params)
                elif "XGBoost" in _selected_name:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                        "max_depth": trial.suggest_int("max_depth", 3, 9),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "random_state": GLOBAL_SEED,
                        **xgb_kwargs
                    }
                    model_opt = XGBRegressor(**params)
                else:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
                        "max_depth": trial.suggest_int("max_depth", 5, 15),
                        "random_state": GLOBAL_SEED,
                        "n_jobs": -1
                    }
                    model_opt = RandomForestRegressor(**params)

                kf = KFold(n_splits=3, shuffle=True, random_state=GLOBAL_SEED)
                cv_scores = []

                for train_idx, val_idx in kf.split(X_train_proc):
                    X_tr = X_train_proc.iloc[train_idx].copy() if isinstance(X_train_proc, pd.DataFrame) else X_train_proc[train_idx].copy()
                    X_val = X_train_proc.iloc[val_idx].copy() if isinstance(X_train_proc, pd.DataFrame) else X_train_proc[val_idx].copy()
                    y_tr = y_train.iloc[train_idx].copy()
                    y_val = y_train.iloc[val_idx].copy()

                    model_opt.fit(X_tr, y_tr)
                    y_pred = model_opt.predict(X_val)
                    cv_scores.append(mean_absolute_percentage_error(y_val, y_pred))

                return np.mean(cv_scores)

            def _progress_callback(study, trial):
                _bar.update(
                    increment=1,
                    subtitle=f"🌿 Ітерація {trial.number + 1} з {trials_slider.value} | Найкращий MAPE: {study.best_value * 100:.2f}%"
                )

            _sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
            _study = optuna.create_study(direction="minimize", sampler=_sampler)
            _study.optimize(_objective, n_trials=trials_slider.value, callbacks=[_progress_callback])

            _best_params = _study.best_params
            _best_params.update({"random_state": GLOBAL_SEED})
            logger.info(f"Optuna знайшла найкращі параметри: {_best_params}")

            if "KNN" in _selected_name:
                final_tuned_model = KNeighborsRegressor(**_best_params)
            elif "XGBoost" in _selected_name:
                _best_params.update({"random_state": GLOBAL_SEED, **xgb_kwargs})
                final_tuned_model = XGBRegressor(**_best_params)
            else:
                _best_params.update({"random_state": GLOBAL_SEED, "n_jobs": -1})
                final_tuned_model = RandomForestRegressor(**_best_params)

            _bar.update(increment=0, subtitle="💾 Збереження найкращої моделі у базу...")
            final_tuned_model.fit(X_train_proc, y_train)
            mo.output.clear()

            _safe_run_name = f"Optuna_{_selected_name.replace(' ', '_').replace('(', '').replace(')', '')}"

            with mlflow.start_run(run_name=_safe_run_name):
                mlflow.log_params(_best_params)
                mlflow.log_metric("CV_MAPE", _study.best_value)
                mlflow.log_metric("Optuna_Trials", trials_slider.value)

                _trusted_types = [
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBRegressor",
                    "sklearn.neighbors._regression.KNeighborsRegressor",
                    "sklearn.ensemble._forest.RandomForestRegressor",
                    "collections.OrderedDict"
                ]

                _pip_reqs = ["scikit-learn"]
                if "XGBoost" in _selected_name: _pip_reqs.append("xgboost")

                mlflow.sklearn.log_model(
                    final_tuned_model,
                    artifact_path="champion_model",
                    skops_trusted_types=_trusted_types,
                    pip_requirements=_pip_reqs
                )

                _run_id = mlflow.active_run().info.run_id

            # ==========================================
            # 🎨 БЛОК ВІЗУАЛІЗАЦІЇ ТА UX
            # ==========================================
            _fig_history = plot_optimization_history(_study)
            try:
                _fig_params = plot_param_importances(_study)
            except Exception:
                _fig_params = None

            _theme = mo.app_meta().theme
            _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
            _text_color = "white" if _theme == "dark" else "#1f2937"

            # 🇺🇦 1. УКРАЇНІЗАЦІЯ ГРАФІКА ІСТОРІЇ
            _fig_history.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_text_color),
                title=dict(text="<b>Історія оптимізації</b>", x=0.5),
                xaxis_title="Ітерація (Спроба)",
                yaxis_title="Похибка MAPE (Частка)"
            )
            for _trace in _fig_history.data:
                # Зсуваємо координати осі X на +1
                if _trace.x is not None:
                    _trace.x = tuple(x + 1 for x in _trace.x)

                if _trace.name == 'Objective Value':
                    _trace.name = 'Похибка поточної ітерації'
                    _trace.hovertemplate = '<b>Ітерація:</b> %{x}<br><b>MAPE:</b> %{y:.4f}<extra></extra>'
                    _trace.marker.color = '#60a5fa'
                elif _trace.name == 'Best Value':
                    _trace.name = 'Рекорд (Найкраще значення)'
                    _trace.hovertemplate = '<b>Ітерація:</b> %{x}<br><b>Рекорд:</b> %{y:.4f}<extra></extra>'
                    _trace.line.color = '#ef4444'

            # 🇺🇦 2. УКРАЇНІЗАЦІЯ ГРАФІКА ВАЖЛИВОСТІ
            if _fig_params:
                _fig_params.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_text_color),
                    title=dict(text="<b>Важливість гіперпараметрів</b>", x=0.5),
                    xaxis_title="Ступінь впливу на результат",
                    yaxis_title="Гіперпараметр"
                )
                for _trace in _fig_params.data:
                    _trace.hovertemplate = '<b>Гіперпараметр:</b> %{y}<br><b>Вплив:</b> %{x:.3f}<extra></extra>'
                    _trace.marker.color = '#8b5cf6'

            # 3. ТАБЛИЦЯ ТА КОМПОНУВАННЯ
            _params_table = style_dataframe(pd.DataFrame([_best_params]), text_align="center", vertical_lines=True, show_index=False)
            _plots_ui = mo.hstack([_fig_history, _fig_params], justify="center") if _fig_params else _fig_history

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
                    ✅ **Оптимізацію завершено!** Найкращий MAPE: `{_study.best_value * 100:.2f}%`<br/>
                    💎 **Залізо (Engine):** `{_hw_ui}`<br/>
                    🗃️ **MLflow:** Усі параметри збережено в `mlruns/mlruns.db`! *(Run ID: `{_run_id}`)*
                    """
                ),
                mo.Html(f"<div style='overflow-x: auto; border: 1px solid {_border}; border-radius: 8px;'>{_params_table}</div>"),
                _plots_ui
            ])

            mo.output.append(_result_ui)
    return (final_tuned_model,)


@app.cell
def feature_importance_analysis(
    UA_COLUMNS,
    X_train_proc,
    champion_selector,
    go,
    master_registry,
    mo,
    np,
    pd,
    permutation_importance,
    preprocessor,
    trained_models,
    y_train,
):
    _selected_name = champion_selector.value

    # 🛡️ ЗАХИСТ: зупиняємо комірку, якщо алгоритм ще не натреновано
    mo.stop(
        _selected_name not in trained_models,
        mo.md(f"⚠️ **Модель `{_selected_name}` ще не натренована!** Будь ласка, запустіть тренування у блоці вище.")
    )

    _model = trained_models[_selected_name]
    _mod_id = master_registry[_selected_name][0]
    _id_str = f"#{_mod_id:02d}"

    # Використовуємо твій єдиний масив даних
    _X_data = X_train_proc

    # Безпечно витягуємо імена колонок, навіть якщо X_train_proc це Numpy-масив
    if hasattr(_X_data, "columns"):
        _features = _X_data.columns.tolist()
    elif hasattr(preprocessor, "get_feature_names_out"):
        _features = preprocessor.get_feature_names_out()
    else:
        _features = [f"Ознака_{i}" for i in range(_X_data.shape[1])]

    # Універсальний адаптер для пайплайнів
    def _get_final_estimator(model):
        if hasattr(model, "steps") and len(model.steps) > 0:
            return model.steps[-1][1]
        return model

    _core_estimator = _get_final_estimator(_model)
    _importances = None
    _calc_method = "Невідомо"

    # =========================================================================
    # 🔍 1. НАМАГАЄМОСЯ ДІСТАТИ НАТИВНУ ВАГУ
    # =========================================================================
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

    # =========================================================================
    # 🪎 2. МАГІЯ XAI ДЛЯ ЧОРНИХ СКРИНЬОК (Permutation Importance)
    # =========================================================================
    if _importances is None or len(_importances) != len(_features) or np.all(_importances == 0):
        with mo.status.spinner("🪎 Зламуємо чорну скриньку", subtitle=f"'{_selected_name}' (Permutation Importance)..."):
            try:
                _sample_size = min(2000, _X_data.shape[0])

                # БЕЗПЕЧНИЙ семплінг (працює і для DataFrame, і для Numpy)
                if isinstance(_X_data, (pd.DataFrame, pd.Series)):
                    _X_sample = _X_data.sample(n=_sample_size, random_state=42)
                    _y_sample = y_train.loc[_X_sample.index]
                else:
                    np.random.seed(42)
                    _indices = np.random.choice(_X_data.shape[0], _sample_size, replace=False)
                    _X_sample = _X_data[_indices]
                    _y_sample = y_train.iloc[_indices] if hasattr(y_train, 'iloc') else y_train[_indices]

                _perm_result = permutation_importance(
                    _model, _X_sample, _y_sample, n_repeats=5, random_state=42, n_jobs=-1
                )

                _importances = _perm_result.importances_mean

                # ЗАХИСТ: Рятуємо від NaN та нескінченностей
                _importances = np.nan_to_num(_importances, nan=0.0, posinf=0.0, neginf=0.0)
                _importances = np.clip(_importances, a_min=0, a_max=None)
                _calc_method = "Перестановочна (Permutation Importance)"
            except Exception as e:
                mo.stop(True, mo.md(f"❌ **Помилка обчислення Permutation Importance:** {str(e)}"))

    # =========================================================================
    # 📝 3. ФОРМУВАННЯ ДАНИХ ТА РОЗУМНИЙ ПЕРЕКЛАД OHE
    # =========================================================================
    def _localize(feat):
        if feat in UA_COLUMNS:
            return UA_COLUMNS[feat]
        # Переклад складених ознак типу OHE
        for _eng, _ua in UA_COLUMNS.items():
            if feat.startswith(_eng):
                _suffix = feat[len(_eng):].strip("_")
                if _suffix:
                    return f"{_ua} ({_suffix})"
                return _ua
        return feat

    _localized_features = [_localize(f) for f in _features]
    _min_len = min(len(_localized_features), len(_importances))

    _df_fi = pd.DataFrame({
        "Ознака": _localized_features[:_min_len],
        "Важливість": _importances[:_min_len]
    }).sort_values(by="Важливість", ascending=True)

    # =========================================================================
    # 📊 4. ВІЗУАЛІЗАЦІЯ PLOTLY
    # =========================================================================
    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"
    _border_color = "rgba(128,128,128,0.2)"

    # 🎨 Кастомна палітра (ОНОВЛЕНО)
    if any(x in _selected_name for x in ["XGBoost", "Forest", "LightGBM", "Trees", "Gradient"]):
        _colorscale = 'Teal'
    elif "EBM" in _selected_name or "Explainable" in _selected_name:
        _colorscale = 'Purples'
    elif "KNN" in _selected_name or "Radius" in _selected_name:
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
        text=_df_fi["Важливість"].apply(lambda x: f"{x:.4f}" if "Permutation" in _calc_method else f"{x:.3f}"),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Вага (Вплив): %{x:.5f}<extra></extra>"
    ))

    _max_val = _df_fi["Важливість"].max()
    _x_range = [0, 0.1] if _max_val == 0 else None

    # Динамічна висота графіка
    _dynamic_height = max(600, len(_df_fi) * 25)

    _fig_fi.update_layout(
        title=dict(
            text=f"<b>Рентген алгоритму ({_id_str} {_selected_name})</b><br><span style='font-size:13px; color:gray;'>Метод екстракції: {_calc_method}</span>",
            x=0.5, font=dict(color=_text_color, size=18)
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_text_color),
        xaxis=dict(title="Сила впливу ознаки", gridcolor=_border_color, zerolinecolor=_border_color, range=_x_range),
        yaxis=dict(title="", gridcolor=_border_color),
        height=_dynamic_height, margin=dict(l=20, r=40, t=75, b=20)
    )

    # =========================================================================
    # 🧠 5. ДИНАМІЧНІ БІЗНЕС-ІНСАЙТИ
    # =========================================================================
    # Безпечне витягування імен лідерів (якщо ознак менше ніж 2)
    _top_1_feat = _df_fi.iloc[-1]["Ознака"] if len(_df_fi) > 0 else "Невідомо"
    _top_2_feat = _df_fi.iloc[-2]["Ознака"] if len(_df_fi) > 1 else "Невідомо"

    _insight_md = mo.md(
        f"""
        > **💡 Tech Lead Insight (Інтерпретація Моделі):**<br/>
        > Цей графік ілюструє глобальну стратегію прийняття рішень моделлю — на які саме метрики вона звертає найбільшу увагу перед тим, як видати прогноз.
        >
        > - 🥇 **Головний предиктор:** Ознака **«{_top_1_feat}»** має найвищу питому вагу. Математично алгоритм вважає її найбільш критичною.
        > - 🥈 **Другорядний фактор:** Ознака **«{_top_2_feat}»** також відіграє вагому роль, корегуючи або підтверджуючи логіку першої.
        > - 🗑️ **Вектор оптимізації:** Ознаки, які знаходяться внизу списку, додають мінімальну інформаційну цінність.
        >
        > **Механіка розрахунку залежить від сімейства алгоритмів:**
        >
        > 📐 **1. Лінійні моделі (Ridge, Lasso, ElasticNet):**<br/>
        > Показують абсолютну вагу коефіцієнтів (`|coef_|`). На скільки зміниться цільовий показник при зміні ознаки на 1 одиницю. *(Dummy Baseline — заглушка, її вага завжди нульова)*.
        >
        > 🌲 **2. Дерева та Ансамблі (Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM):**<br/>
        > Показують **Information Gain** (приріст інформації). Наскільки ефективно ознака зменшувала "хаос" під час побудови гілок дерев. Це відносна "корисність" для прийняття рішень.
        >
        > 🔮 **3. Елітні 'Білі скриньки' (Explainable Boosting - EBM):**<br/>
        > Показують середню абсолютну маргінальну вагу. Стовпчик — це усереднений чистий вплив ознаки на фінальний результат. Найвища прозорість для бізнесу.
        >
        > 🗃️ **4. Чорні скриньки (KNN та Radius Neighbors):**<br/>
        > Використовують **Permutation Importance**. Ми навмисно "перемішуємо" колонку і дивимось, наскільки катастрофічно падає точність алгоритму. Для "сусідів" це показує, яка координата виявилася найважливішою для пошуку відстані.
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
def header_xai(mo):
    mo.md("""
    <h2 align='center'><b>🕵️‍♂️ 6. Квантова пояснюваність <i>(XAI & SHAP)</i></b></h2>
    """)
    return


@app.cell
def shap_ui_controls(champion_selector, final_tuned_model, mo):
    _selected_name = champion_selector.value
    # SHAP TreeExplainer працює ТІЛЬКИ з деревами/бустингом
    _is_tree = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])

    # Перевіряємо, чи є в пам'яті свіжа оптимізована модель
    _use_tuned = final_tuned_model is not None
    _model_origin = "Після оптимізації Optuna 🎯" if _use_tuned else "Базова з Лідерборду 🐣"

    shap_btn = mo.ui.run_button(
        label=f"👁️ Згенерувати SHAP для {_selected_name}",
        kind="info",
        disabled=not _is_tree
    )

    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"

    _warning_msg = "" if _is_tree else f"<div style='color: #ef4444; margin-bottom: 10px;'>⚠️ <b>Увага:</b> <i>{_selected_name}</i> не підтримує швидкий SHAP TreeExplainer. Оберіть Ансамблі або Бустинг у конфігураторі.</div>"

    _shap_card = mo.md(
        f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
            <h3 style="margin-top: 0;">🕵️‍♂️ Аналіз SHAP (Квантова пояснюваність)</h3>
            <p>Цей алгоритм заглядає всередину "чорної скриньки" і розраховує математичний внесок кожної ознаки для <b>кожного окремого співробітника</b>.<br/>
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
    X_train_proc,
    champion_selector,
    final_tuned_model,
    mo,
    plt,
    shap,
    shap_btn,
    shap_tree,
    trained_models,
):
    if shap_btn.value:
        _selected_name = champion_selector.value

        # Визначаємо, чи належить модель до сімейства нативних бустингів
        _native_names = ["LightGBM", "XGBoost", "Explainable Boosting"]
        _is_native = any(n in _selected_name for n in _native_names)

        # Беремо модель: або оптимізовану, або базову
        _model_to_explain = final_tuned_model if final_tuned_model is not None else trained_models[_selected_name]

        with mo.status.spinner(title="🍻 Аналіз рішень моделі...", subtitle="Розрахунок векторів Шеплі (SHAP values)"):
            # Використовуємо процесований набір даних (як навчалася модель)
            _X_sample = X_train_proc.sample(n=min(500, len(X_train_proc)), random_state=GLOBAL_SEED)

            # 🛠️ Патч для сумісності нових версій XGBoost та SHAP
            _orig_decode = getattr(shap_tree, "decode_ubjson_buffer", None)

            def _clean_base_score(_dict):
                try:
                    _bs = _dict.get("learner", {}).get("learner_model_param", {}).get("base_score")
                    if isinstance(_bs, str) and "[" in _bs:
                        _dict["learner"]["learner_model_param"]["base_score"] = _bs.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
                except Exception:
                    pass
                return _dict

            if _orig_decode:
                def _patched_decode(*args, **kwargs):
                    return _clean_base_score(_orig_decode(*args, **kwargs))
                shap_tree.decode_ubjson_buffer = _patched_decode

            # Розрахунок SHAP
            try:
                _explainer = shap.TreeExplainer(_model_to_explain)
                _shap_values = _explainer.shap_values(_X_sample, check_additivity=False)
            finally:
                if _orig_decode:
                    shap_tree.decode_ubjson_buffer = _orig_decode

            # Локалізація колонок
            def _localize(feat):
                if feat in UA_COLUMNS: return UA_COLUMNS[feat]
                for _eng, _ua in UA_COLUMNS.items():
                    if str(feat).startswith(_eng):
                        _suffix = str(feat)[len(_eng):].strip("_")
                        return f"{_ua} ({_suffix})" if _suffix else _ua
                return feat

            _X_sample_ua = _X_sample.rename(columns={col: _localize(col) for col in _X_sample.columns})

            _theme = mo.app_meta().theme
            _style = 'dark_background' if _theme == 'dark' else 'default'

            with plt.style.context(_style):
                plt.rcParams['savefig.transparent'] = True
                _fig, _ax = plt.subplots(figsize=(10, 6))

                # Відмальовка бджолиного рою
                shap.summary_plot(_shap_values, _X_sample_ua, show=False)

                _text_color = 'white' if _theme == 'dark' else '#1f2937'

                _ax.set_title(
                    f"Квантова пояснюваність ({_selected_name}): Глобальний вплив ознак",
                    color=_text_color, fontsize=15, fontweight='bold', pad=20
                )

                _ax.set_xlabel("Значення SHAP (Вплив на прогноз зарплати, $)", color=_text_color)

                _fig.patch.set_facecolor('none')
                _ax.set_facecolor('none')
                _ax.tick_params(colors=_text_color)
                _ax.yaxis.label.set_color(_text_color)
                _ax.xaxis.label.set_color(_text_color)

                if _theme == "dark":
                    for spine in _ax.spines.values():
                        spine.set_edgecolor('gray')

                if len(_fig.axes) > 1:
                    _cbar_ax = _fig.axes[-1]
                    _cbar_ax.set_ylabel("Трансформоване значення ознаки", rotation=270, labelpad=15)
                    _ticks = _cbar_ax.get_yticks()
                    _cbar_ax.set_yticks(_ticks)
                    _cbar_ax.set_yticklabels(["Низьке", "Високе"])
                    _cbar_ax.yaxis.label.set_color(_text_color)
                    _cbar_ax.tick_params(colors=_text_color)
                    if _theme == "dark":
                        for spine in _cbar_ax.spines.values():
                            spine.set_edgecolor('gray')

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

            _native_insight = ""
            if _is_native:
                _native_insight = """
                > 👽 **Що означають СІРІ точки?**<br/>
                > Якщо ви вимкнете TargetEncoder і передасте в модель сирі текстові дані, SHAP зафарбує їх у сірий колір. Для тексту (напр. "Менеджер") поняття "Високе/Низьке" значення не має математичного сенсу, тому SHAP захищає вас від хибних висновків за шкалою кольорів.
                """

            _insight_md = mo.md(
                f"""
                <center>💵 <i>Вплив на осі X <b>розраховано у реальних доларах США</b>.</i></center>

                > **💡 Tech Lead Insight (Як читати SHAP після TargetEncoder):**<br/>
                > Оскільки перед потраплянням у модель дані пройшли через Pipeline трансформації (PowerTransformer та TargetEncoder), колір крапки означає математично нормалізоване значення:
                >
                > - **Колір точки (🔵 Синій/Червоний 🔴):** Високе чи низьке значення ознаки. Для категорій типу "Посада" або "Відділ", червоний колір означатиме ті категорії, які TargetEncoder закодував вищими числами (вищою історичною зарплатою)
                > - **Позиція на осі X (⏮️ Вліво/Вправо ⏭️):** Як сильно характеристика конкретного співробітника "переконала" алгоритм підвищити або зменшити фінальний прогноз зарплати

                {_native_insight}
                """
            )

            mo.output.append(mo.vstack([_css_no_scroll, mo.center(_plot_html), _insight_md]))
    return


@app.cell(hide_code=True)
def header_mlops(mo):
    mo.md("""
    <h2 align='center'><b>⛲️ 7. Продакшн: MLOps Серіалізація та Мікросервіс <i>(FastAPI)</i></b></h2>
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
                <p>Оберіть фінальну модель для передачі Backend-команді. У списку відображаються всі базові алгоритми з Лідерборду, а також регресор, який пройшов тюнінг Optuna (має VIP-статус 🔱).</p>
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
def mlops_serialization(
    Pipeline,
    X_train,
    datetime,
    final_tuned_model,
    joblib,
    json,
    logger,
    mlops_export_selector,
    mlops_generate_btn,
    mo,
    np,
    os,
    preprocessor,
    textwrap,
    trained_models,
):
    if mlops_generate_btn.value:
        _selection = mlops_export_selector.value
        _selected_name = _selection["name"]
        _is_tuned = _selection["is_tuned"]

        _model_to_save = final_tuned_model if _is_tuned else trained_models[_selected_name]

        # 1. Об'єднуємо Preprocessor + Model у єдиний Pipeline для деплою!
        _full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', _model_to_save)
        ])

        _project_name = "salary_prediction"
        _artifact_dir = os.path.join(os.getenv("MODELS_DIR", "./models"), _project_name)
        os.makedirs(_artifact_dir, exist_ok=True)

        _safe_name = _selected_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        if _is_tuned:
            _safe_name += "_tuned"

        _model_path = os.path.join(_artifact_dir, f"{_safe_name}_champion.joblib")
        _schema_path = os.path.join(_artifact_dir, "features_schema.json")
        _api_path = os.path.join(_artifact_dir, "api.py")
        _docker_path = os.path.join(_artifact_dir, "Dockerfile")

        joblib.dump(_full_pipeline, _model_path)

        _dtypes_dict = {}
        for _col in X_train.columns:
            _dt = X_train[_col].dtype
            _dtypes_dict[_col] = str(_dt)

        _features_schema = {
            "project_name": _project_name,
            "exported_at": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "model_architecture": _selected_name,
            "is_optuna_tuned": _is_tuned,
            "expected_columns": list(X_train.columns),
            "dtypes": _dtypes_dict
        }
        with open(_schema_path, "w", encoding="utf-8") as f:
            json.dump(_features_schema, f, indent=4, ensure_ascii=False)

        _sample_row = X_train.iloc[0].to_dict()
        _sample_clean = {
            k: (float(v) if isinstance(v, (float, np.floating))
                else int(v) if isinstance(v, (int, np.integer))
                else str(v))
            for k, v in _sample_row.items()
        }
        _sample_json_str = json.dumps({"features": _sample_clean}, ensure_ascii=False)

        # =====================================================================
        # 🏗️ АРХІТЕКТУРНИЙ ШЕДЕВР: Генерація коду FastAPI
        # =====================================================================
        _api_content = textwrap.dedent(f"""
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import HTMLResponse
            from pydantic import BaseModel
            from typing import Dict, Any
            import joblib
            import json
            import pandas as pd
            import uvicorn
            import os

            app = FastAPI(
                title="💼 Salary Prediction API",
                version="1.0",
                docs_url=None,
                redoc_url=None
            )

            @app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
            def scalar_html():
                return '''
                <!doctype html>
                <html>
                <head>
                    <title>Salary Prediction API Docs</title>
                    <meta charset="utf-8" />
                </head>
                <body>
                    <script id="api-reference" data-url="/openapi.json"></script>
                    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
                </body>
                </html>
                '''

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            MODEL_PATH = os.path.join(BASE_DIR, "{os.path.basename(_model_path)}")
            SCHEMA_PATH = os.path.join(BASE_DIR, "features_schema.json")

            try:
                pipeline = joblib.load(MODEL_PATH)
                with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                    expected_dtypes = schema.get("dtypes", {{}})
                    expected_columns = schema.get("expected_columns", [])
                    is_tuned = schema.get("is_optuna_tuned", False)
            except Exception as e:
                pipeline, expected_columns, expected_dtypes, is_tuned = None, [], {{}}, False

            class InferencePayload(BaseModel):
                features: Dict[str, Any]
                model_config = {{"json_schema_extra": {{"examples": [{_sample_json_str}]}}}}

            class PredictionResponse(BaseModel):
                predicted_salary_usd: float
                model_deployed: str
                is_optuna_tuned: bool
                features_processed_count: int

            @app.post("/predict", response_model=PredictionResponse)
            def predict_salary(payload: InferencePayload):
                if pipeline is None:
                    raise HTTPException(status_code=500, detail="Пайплайн не знайдено")

                try:
                    df = pd.DataFrame([payload.features])

                    if expected_columns:
                        df = df.reindex(columns=expected_columns)

                    for col, dtype in expected_dtypes.items():
                        if col in df.columns:
                            df[col] = df[col].astype(dtype)

                    # Трансформаційний пайплайн сам застосує TargetEncoder та PowerTransformer!
                    pred = pipeline.predict(df)[0]

                    return {{
                        "predicted_salary_usd": round(float(pred), 2),
                        "model_deployed": "{_selected_name}",
                        "is_optuna_tuned": is_tuned,
                        "features_processed_count": len(expected_columns)
                    }}

                except Exception as e:
                    raise HTTPException(status_code=400, detail=str(e))

            if __name__ == "__main__":
                api_host = os.getenv("API_HOST", "0.0.0.0")
                api_port = int(os.getenv("API_PORT", 8000))
                uvicorn.run(app, host=api_host, port=api_port)
        """).strip()

        with open(_api_path, "w", encoding="utf-8") as f:
            f.write(_api_content)

        # =====================================================================
        # 🛠️ ГЕНЕРАЦІЯ DOCKERFILE
        # =====================================================================
        _docker_libs = "fastapi uvicorn pydantic joblib pandas scikit-learn numpy"
        if "XGBoost" in _selected_name: _docker_libs += " xgboost"
        elif "LightGBM" in _selected_name: _docker_libs += " lightgbm"
        elif "EBM" in _selected_name or "Explainable" in _selected_name: _docker_libs += " interpret"

        _docker_content = textwrap.dedent(f"""
            FROM python:3.12-slim
            WORKDIR /app
            RUN pip install --no-cache-dir {_docker_libs}
            COPY . /app
            EXPOSE 8000
            CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
        """).strip()

        with open(_docker_path, "w", encoding="utf-8") as f:
            f.write(_docker_content)

        # =====================================================================
        # 📊 ЗБІР СТАТИСТИКИ ТА ПАРАМЕТРІВ
        # =====================================================================
        _model_size_kb = os.path.getsize(_model_path) / 1024
        _schema_size_kb = os.path.getsize(_schema_path) / 1024
        _api_size_kb = os.path.getsize(_api_path) / 1024
        _timestamp_human = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        _num_features = X_train.shape[1]

        _params = getattr(_model_to_save, "get_params", lambda: {})()
        _arch_lines = []

        if 'n_estimators' in _params and _params['n_estimators'] is not None:
            _arch_lines.append(f"🧬 Дерев (n_estimators): {_params['n_estimators']}")
        if 'n_neighbors' in _params and _params['n_neighbors'] is not None:
            _arch_lines.append(f"👯 Сусідів (n_neighbors): {_params['n_neighbors']}")
            _arch_lines.append(f"📏 Метрика (metric): {_params.get('metric', 'minkowski')}")
        if 'learning_rate' in _params and _params['learning_rate'] is not None:
            _lr_val = _params['learning_rate']
            if isinstance(_lr_val, float): _lr_val = round(_lr_val, 5)
            _arch_lines.append(f"⚡ Швидкість (LR): {_lr_val}")

        if not _arch_lines:
            _arch_lines.append("⚖️ Тип: Аналітична / Лінійна архітектура (без дерев або сусідів)")

        _exclude_keys = {
            'n_estimators', 'max_depth', 'learning_rate', 'booster',
            'random_state', 'n_jobs', 'objective', 'enable_categorical',
            'n_neighbors', 'metric', 'weights', 'p', 'algorithm'
        }
        _extra_params = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in _params.items()
            if k not in _exclude_keys and v is not None
        }

        if _extra_params:
            _param_list = [f"{k}={v}" for k, v in _extra_params.items()]
            _chunk_size = 4
            _chunks = [", ".join(_param_list[i:i+_chunk_size]) for i in range(0, len(_param_list), _chunk_size)]
            _indent_spaces = " " * 41
            _extra_params_str = f",\n{_indent_spaces}".join(_chunks)
            _arch_lines.append(f"⚙️ Додаткові параметри: {_extra_params_str}")

        _arch_text = "\n                 ".join(_arch_lines)
        _origin_status = "🔱 Оптимізована (Optuna)" if _is_tuned else "🧧 Базова (З Лідерборду)"

        # Зчитуємо файли для кнопок завантаження
        with open(_model_path, "rb") as f: _model_bytes = f.read()
        with open(_schema_path, "rb") as f: _schema_bytes = f.read()
        with open(_api_path, "rb") as f: _api_bytes = f.read()
        with open(_docker_path, "rb") as f: _docker_bytes = f.read()

        _download_model_btn = mo.download(data=_model_bytes, filename=os.path.basename(_model_path), label="☯️ .joblib (Ваги Pipeline)")
        _download_schema_btn = mo.download(data=_schema_bytes, filename="features_schema.json", label="✡️ .json (Схема)")
        _download_api_btn = mo.download(data=_api_bytes, filename="api.py", label="⚛️ api.py (FastAPI)")
        _download_docker_btn = mo.download(data=_docker_bytes, filename="Dockerfile", label="🐳 Dockerfile")

        _theme = mo.app_meta().theme
        _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
        _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

        _version_tag = " (Optuna Tuned 🔱)" if _is_tuned else " (Базова версія)"
        logger.info(f"Експорт MLOps артефактів завершено для {_selected_name}{_version_tag}")

        mo.output.append(mo.md(f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-top: 20px;">
            <h3 style="margin-top: 0; color: #3b82f6; text-align: center;">📦 MLOps Серіалізація (Production Ready)</h3>
            <p style="text-align: center;">Ми запакували <code>PowerTransformer</code>, <code>TargetEncoder</code> та <code>{_selected_name}</code> у єдиний нерозривний <code>.joblib</code> файл. Серверу <code>FastAPI</code> не потрібно знати про трансформації — він просто передає сирі дані у пайплайн!</p>

            ```text
            🎭 Початок збереження у файл...
              📦 Шлях проекту: {_artifact_dir}/
              🎛 Ознаки: {_num_features} вимірів
              👾 Архітектура моделі:
                 🧠 Алгоритм: {_selected_name}
                 🛠 Джерело: {_origin_status}
                 {_arch_text}

              ✅ Успіх! Капсула 'Пайплайну' надійно збережена ({_model_size_kb:,.2f} KB) о {_timestamp_human}
              ✅ Накладну (Маніфест ознак + Метадані) експортовано ({_schema_size_kb:,.2f} KB)
              ⚡ Pydantic-Схема та FastAPI-сервер згенеровано! ({_api_size_kb:,.2f} KB)
            ```
        </div>
        """))
        mo.output.append(mo.hstack([_download_model_btn, _download_schema_btn, _download_api_btn, _download_docker_btn], justify="center"))
    return


@app.cell
def deploy_instructions(mo):
    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _pre_bg = "#111827" if _theme == "dark" else "#f3f4f6"
    _pre_border = "#374151" if _theme == "dark" else "#d1d5db"
    _pre_text_cmd = "#10b981" if _theme == "dark" else "#059669"
    _pre_text_code = "#e5e7eb" if _theme == "dark" else "#374151"

    _css_no_scroll = mo.md(
        """
        <div class="marimo-noscroll-override"></div>
        <style>
            marimo-cell-output:has(.marimo-noscroll-override),
            .output-area:has(.marimo-noscroll-override) {
                max-height: none !important;
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
            <p style="font-size: 15px;">Цей модуль автоматично генерує повністю налаштований бекенд для нашої ML-моделі прогнозування зарплат. Ми перейшли від базових Data Science скриптів до бездоганної інфраструктури Enterprise-рівня.</p>

            <h3 style="color: #3b82f6; margin-top: 25px;">✨ Технологічний Стек та Можливості</h3>
            <ul style="font-size: 14px;">
                <li style="margin-bottom: 8px;"><b>FastAPI + Uvicorn:</b> Високопродуктивний асинхронний сервер, який миттєво обробляє HTTP-запити та виконує інференс (регресію).</li>
                <li style="margin-bottom: 8px;"><b>Scikit-Learn Pipeline 🔗:</b> Уся логіка (<code>PowerTransformer</code> та <code>TargetEncoder</code>) запечена всередину моделі. Бекенд не займається обробкою даних — він просто передає сирий JSON у пайплайн.</li>
                <li style="margin-bottom: 8px;"><b>Docker Контейнеризація 🐳:</b> Автоматична генерація <code>Dockerfile</code> з точними залежностями алгоритму. Це забезпечує 100% ізоляцію середовища — модель гарантовано працюватиме однаково на вашому ноутбуці або сервері AWS.</li>
                <li style="margin-bottom: 8px;"><b>Сучасний Scalar UI:</b> Ми повністю відмовилися від застарілого Swagger. Інтегрований <b>Scalar</b> забезпечує преміальний дизайн документації, вбудований REST-клієнт та миттєву генерацію коду запитів для десятків мов програмування.</li>
            </ul>

            <h3 style="color: #10b981; margin-top: 25px;">🛡️ Суворі Pydantic-Контракти</h3>
            <ul style="font-size: 14px;">
                <li style="margin-bottom: 8px;">✅ <b>200 OK (Успішна відповідь):</b> Сервер повертає чітко типізовану схему <code>PredictionResponse</code>. Завдяки цьому клієнт наперед знає, що гарантовано отримає <code>predicted_salary_usd</code> (float) та метадані про поточну активну модель <code>model_deployed</code>.</li>
                <li style="margin-bottom: 8px;">❌ <b>422 Validation Error:</b> Завдяки <code>InferencePayload</code>, якщо клієнт надішле неправильний тип даних або пропустить параметр, FastAPI автоматично відхилить запит із детальним JSON-описом, захищаючи ML-модель від падінь.</li>
            </ul>

            <h3 style="color: #f59e0b; margin-top: 25px;">⚙️ Як запустити сервер?</h3>
            <p style="font-size: 14px;">Усі згенеровані артефакти надійно ізольовано у директорії <code>models/salary_prediction/</code>.</p>

            <div style="margin-top: 15px;">
                <b>▶ Спосіб 1: DevOps-стандарт (Через Makefile)</b>
                <pre style="background-color: {_pre_bg}; color: {_pre_text_cmd}; padding: 12px; border-radius: 8px; border: 1px solid {_pre_border}; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>make api-hw3</code></pre>
            </div>

            <div style="margin-top: 15px;">
                <b>▶ Спосіб 2: Запуск у Docker (Cloud Ready ☁️)</b>
                <pre style="background-color: {_pre_bg}; color: {_pre_text_code}; padding: 12px; border-radius: 8px; border: 1px solid {_pre_border}; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>cd models/salary_prediction
        docker build -t salary-api .
        docker run -p 8000:8000 salary-api</code></pre>
            </div>

            <div style="margin-top: 15px;">
                <b>▶ Спосіб 3: Ручний запуск (Без Docker)</b>
                <pre style="background-color: {_pre_bg}; color: {_pre_text_code}; padding: 12px; border-radius: 8px; border: 1px solid {_pre_border}; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>cd models/salary_prediction
        uvicorn api:app --host 0.0.0.0 --port 8000 --reload</code></pre>
            </div>

            <hr style="border-color: {_border}; margin: 25px 0;">
            <p style="margin-bottom: 0; font-size: 15px;">
                <i>💡 <b>Документація доступна за адресою:</b> <a href="http://127.0.0.1:8000/docs" target="_blank" style="color: #3b82f6; font-weight: bold; text-decoration: none;">http://127.0.0.1:8000/docs</a>.</i><br/>
                <i>🥂 Тепер будь-який веб-сайт, Telegram-бот чи мобільний застосунок на Swift/Kotlin може відправляти JSON-запити на цей порт і миттєво отримувати прогноз зарплати!</i>
            </p>
        </div>
        """)

    mo.output.append(mo.hstack([_css_no_scroll, _deploy_instructions]))
    return


@app.cell(hide_code=True)
def header_timesfm(mo):
    mo.md("""
    <h2 align='center'><b>🔮 8. TimesFM <i>(Zero-Shot Прогноз інфляції зарплат)</i></b></h2>
    """)
    return


@app.cell
def timesfm_ui_controls(mo):
    import timesfm
    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    timesfm_btn = mo.ui.run_button(
        label="🚀 Згенерувати 2-річний прогноз росту зарплат",
        kind="success"
    )

    _ui_panel = mo.vstack([
        mo.md(
            f"""
            <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
                <p>Фундаментальна модель <b>Google TimesFM</b> (<i>Рушій v2.0, Ваги: 1.0-200m</i>) здатна робити високоточні прогнози часових рядів <b>без жодного донавчання</b>!<br/>Натисніть кнопку нижче для генерації Zero-Shot прогнозу середньої базової ставки на <b>наступні 24 місяці</b>.</p>
                <div style="margin-top: 20px;">
                    {timesfm_btn}
                </div>
            </div>
            """
        )
    ])

    mo.output.append(_ui_panel)
    return timesfm, timesfm_btn


@app.cell
def execute_timesfm(
    UA_COLUMNS,
    clear_vram,
    device,
    device_ui_name,
    df_train_raw,
    go,
    logger,
    mo,
    np,
    pd,
    timesfm,
    timesfm_btn,
):
    mo.stop(
        not timesfm_btn.value,
        mo.center(mo.md("⏳ **Очікування:** Натисніть кнопку 'Згенерувати 2-річний прогноз росту зарплат' 🚀"))
    )

    _df_time = df_train_raw.copy()

    # 1. Знаходимо колонку із зарплатою
    _salary_col_name = UA_COLUMNS.get('Salary', 'Фактична Заробітна плата ($)')
    _salary_col = _salary_col_name if _salary_col_name in _df_time.columns else 'Salary'

    mo.stop(
        _salary_col not in _df_time.columns,
        mo.md(f"❌ **Помилка пайплайну:** Не знайдено колонку із зарплатою (`{_salary_col}`). Доступні: `{list(_df_time.columns)}`")
    )

    # 2. Динамічний пошук або реконструювання дати найму
    _date_col_name = UA_COLUMNS.get('Hire_Date', 'Дата найму')
    _date_col = _date_col_name if _date_col_name in _df_time.columns else ('Hire_Date' if 'Hire_Date' in _df_time.columns else None)

    _insight_extra = ""

    if _date_col is None:
        _exp_col_name = UA_COLUMNS.get('Experience', 'Досвід роботи')
        _exp_col = _exp_col_name if _exp_col_name in _df_time.columns else ('Experience' if 'Experience' in _df_time.columns else None)

        mo.stop(
            _exp_col is None,
            mo.md(f"❌ **Помилка пайплайну:** У наборі даних немає ні дати найму, ні досвіду роботи. Доступні: `{list(_df_time.columns)}`")
        )

        _df_time['Hire_Date'] = pd.Timestamp.now() - pd.to_timedelta(_df_time[_exp_col] * 365.25, unit='D')
        _date_col = 'Hire_Date'
        _insight_extra = f"<br/><i>🔧 <b>Data Engineering:</b> Оскільки в поточному наборі даних відсутня дата найму, алгоритм реконструював її на основі колонки `{_exp_col}`.</i>"

    # Підготовка часового ряду
    _df_time[_date_col] = pd.to_datetime(_df_time[_date_col])
    _df_time['Month'] = _df_time[_date_col].dt.to_period('M').dt.to_timestamp()

    _ts = _df_time.groupby('Month')[_salary_col].mean().reset_index().sort_values('Month')

    if _ts[_salary_col].sum() == 0:
        np.random.seed(42)
        _ts[_salary_col] = np.random.uniform(1500, 5000, len(_ts))
        _insight_extra += "<br/><i>⚠️ <b>Data Warning:</b> Значення зарплат дорівнювали нулю. Застосовано синтетичні базові ставки.</i>"

    _context_dates = _ts['Month'].values
    _context_values = _ts[_salary_col].values

    _is_mock = False
    _error_msg = ""
    _pred_values = []

    # 4. Інференс або Soft-Fallback
    if len(_context_values) < 128:
        _is_mock = True
        _error_msg = f"Недостатньо історичної глибини (потрібно 128 місяців, маємо {len(_context_values)})"

        np.random.seed(42)
        # Зменшуємо агресивність тренду (* 0.2), щоб лінія не падала в нуль так швидко
        _trend = ((_context_values[-1] - _context_values[0]) / len(_context_values)) * 0.2 if len(_context_values) > 1 else 50
        _noise = np.random.normal(0, np.std(_context_values) * 0.1 if len(_context_values) > 1 and np.std(_context_values) > 0 else 500, 24)
        _pred_values = _context_values[-1] + np.arange(1, 25) * _trend + np.cumsum(_noise)
        _pred_values = np.clip(_pred_values, a_min=0, a_max=None)
    else:
        try:
            with mo.status.spinner(f"Агрегація історичних даних та інференс TimesFM (Бекенд: `{device_ui_name}`)..."):
                _TfmClass = getattr(timesfm, 'TimesFm', getattr(timesfm, 'TimesFM', None))
                if _TfmClass is None: raise AttributeError("Клас TimesFm відсутній")

                _tfm = _TfmClass(context_len=128, horizon_len=24, input_patch_len=32, output_patch_len=128, num_layers=20, model_dims=1280, backend=device.type)
                _tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

                _forecast_values, _ = _tfm.forecast([_context_values[-128:]])
                _pred_values = _forecast_values[0][:, 1]

                clear_vram(device)
                logger.info("Інференс TimesFM успішно завершено.")

        except Exception as e:
            _is_mock = True
            _error_msg = str(e)
            logger.warning(f"⚠️ Fallback TimesFM: {_error_msg}")

            np.random.seed(42)
            _trend = ((_context_values[-1] - _context_values[0]) / len(_context_values)) * 0.2
            _noise = np.random.normal(0, np.std(_context_values) * 0.1 if np.std(_context_values) > 0 else 500, 24)
            _pred_values = _context_values[-1] + np.arange(1, 25) * _trend + np.cumsum(_noise)
            _pred_values = np.clip(_pred_values, a_min=0, a_max=None)

    # 5. Візуалізація
    _last_date = pd.to_datetime(_context_dates[-1])
    _future_dates = pd.date_range(start=_last_date + pd.DateOffset(months=1), periods=24, freq='MS')

    _theme = mo.app_meta().theme
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"

    _fig_ts = go.Figure()

    _fig_ts.add_trace(go.Scatter(
        x=_context_dates, y=_context_values,
        mode='lines', name='Історична Середня ЗП',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='<b>%{data.name}</b>: %{y:,.0f} $<extra></extra>'
    ))

    _fig_ts.add_trace(go.Scatter(
        x=_future_dates, y=_pred_values,
        mode='lines', name='Прогноз на 2 роки',
        line=dict(color='#ef4444', width=2.5, dash='dash'),
        hovertemplate='<b>%{data.name}</b>: %{y:,.0f} $<extra></extra>'
    ))

    _fig_ts.add_trace(go.Scatter(
        x=[_context_dates[-1], _future_dates[0]], y=[_context_values[-1], _pred_values[0]],
        mode='lines', showlegend=False, hoverinfo='skip',
        line=dict(color='#ef4444', width=2.5, dash='dash')
    ))

    _fig_ts.update_layout(
        title=dict(text=f"<b>Zero-Shot Foundation Model (TimesFM) | Backend: {device_ui_name}</b><br><sup>Прогноз інфляції базової ставки (На 24 місяці)</sup>", x=0.5),
        xaxis_title="", yaxis_title=_salary_col_name,
        template=_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified", margin=dict(t=80, b=80, l=40, r=40),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        hoverlabel=dict(
            namelength=-1,
            bgcolor="#1f2937" if _theme == "dark" else "#ffffff",
            font_color="#ffffff" if _theme == "dark" else "#000000",
            bordercolor="#374151" if _theme == "dark" else "#e5e7eb"
        )
    )

    _fig_ts.update_xaxes(
        hoverformat="%m.%Y"
    )

    _css_no_scroll = mo.md(
        """
        <div class="timesfm-noscroll"></div>
        <style>
            marimo-cell-output:has(.timesfm-noscroll),
            .output-area:has(.timesfm-noscroll) {
                max-height: none !important;
                overflow: visible !important;
                overflow-y: visible !important;
                overflow-x: visible !important;
            }
        </style>
        """
    )

    if _is_mock:
        _insight_text = (
            f"> ⚠️ **Симуляція (Fallback Mode):**<br/>"
            f"> Через апаратні обмеження (Intel Mac) або зміну API рушія, офіційна бібліотека `timesfm` не змогла ініціалізуватися (`{_error_msg}`).<br/>"
            f"> Активовано алгоритм симуляції (Mock), щоб продемонструвати структуру виводу. На сумісному сервері тут буде реальний AI-прогноз."
            f"{_insight_extra}"
        )
        _insight = mo.md(_insight_text)
    else:
        _insight_text = (
            f"> 💡 **Tech Lead Insight (Глобальний корпоративний тренд):**<br/>"
            f"> Ми агрегували дані з набору даних і передали їх фундаментальній моделі Google TimesFM. Вона не використовувала жодної з наших ознак (досвід, освіта), а лише досліджувала історичний часовий ряд середньої ставки найму.<br/>"
            f"> Завдяки 200 мільйонам параметрів вона виконала **Zero-Shot** інференс, передбачивши очікувану динаміку компенсацій для нових співробітників на наступні 2 роки."
            f"{_insight_extra}"
        )
        _insight = mo.md(_insight_text)

    mo.output.append(mo.vstack([_css_no_scroll, mo.ui.plotly(_fig_ts, config={'responsive': True}), _insight], align="stretch"))
    return


if __name__ == "__main__":
    app.run()
