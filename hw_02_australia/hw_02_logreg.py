import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title_head_hw(mo):
    mo.md("""
    <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
        🌦️ ДЗ №2: Прогнозування погоди <i>(Rain in Australia)</i>
    </div>

    <h3 align="center"><b><u>Пайплайн</u>: OOT Split ➔ SMOTE ➔ Benchmark (30 Models) ➔ Threshold Opt ➔ Optuna ➔ XAI ➔ TimesFM</b></h3>

    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Machine Learning: Fundamentals and Applications [07.2026]</i></p>
    """)
    return


@app.cell
def configure_dependencies():
    import os
    import warnings

    # Блокуємо конфлікт C-бібліотек OpenMP на рівні ОС
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Глушимо візуальний спам про LLVM/Intel OpenMP
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

    import sys
    import contextlib
    import base64
    import json
    import html
    import textwrap
    from datetime import datetime

    # 🛡️ ПІДКЛЮЧЕННЯ АРХІТЕКТУРНОГО ЯДРА (Фасад)
    # Згідно з інфраструктурними стандартами, ядро автоматично підтягує .env,
    # налаштовує MLflow Tracking URI та придушує зайві Warnings
    _core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
    if _core_path not in sys.path:
        sys.path.append(_core_path)

    # Єдина точка доступу до всіх спільних інструментів
    from core import (
        SecureDownloader, smart_read_csv, get_hardware_config, clear_vram,
        set_global_seed, log_system_info, get_boosting_kwargs, logger
    )

    # 📍 ЛОКАЛЬНІ ІМПОРТИ ДОМЕНУ (Domain Isolation)
    from ui_labels import UA_COLUMNS
    from data_adapters import get_rain_australia_mock

    # Core Data Science
    import logging
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
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import roc_curve

    import marimo as mo

    # Machine Learning Core
    import mlflow
    import mlflow.sklearn
    import sklearn
    import optuna
    import joblib

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import KFold, TimeSeriesSplit
    from sklearn.inspection import permutation_importance

    # 🧬 SMOTE (Боротьба з дисбалансом класів)
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    # Метрики Класифікації
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score, log_loss,
        precision_recall_curve, confusion_matrix, classification_report
    )
    from optuna.visualization import plot_optimization_history, plot_param_importances

    # --- КЛАСИФІКАТОРИ ---
    from sklearn.dummy import DummyClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
    from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
    from sklearn.svm import LinearSVC, SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.ensemble import (
        AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier
    )
    from xgboost import XGBClassifier, XGBRFClassifier
    from lightgbm import LGBMClassifier
    from interpret.glassbox import ExplainableBoostingClassifier

    pd.options.mode.copy_on_write = True
    sklearn.set_config(transform_output="pandas")

    mo.center(mo.md("✅ **Бібліотеки, Локальні Адаптери та Ядро MLOps успішно імпортовано!**"))
    return (
        AdaBoostClassifier,
        BaggingClassifier,
        BernoulliNB,
        CalibratedClassifierCV,
        ColumnTransformer,
        ConvergenceWarning,
        DecisionTreeClassifier,
        DummyClassifier,
        ExplainableBoostingClassifier,
        ExtraTreeClassifier,
        ExtraTreesClassifier,
        GaussianNB,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        ImbPipeline,
        KFold,
        KNeighborsClassifier,
        LGBMClassifier,
        LinearDiscriminantAnalysis,
        LinearSVC,
        LogisticRegression,
        MLPClassifier,
        NearestCentroid,
        OneHotEncoder,
        ProfileReport,
        QuadraticDiscriminantAnalysis,
        RandomForestClassifier,
        RidgeClassifier,
        SGDClassifier,
        SMOTE,
        SVC,
        SecureDownloader,
        SimpleImputer,
        StandardScaler,
        UA_COLUMNS,
        XGBClassifier,
        XGBRFClassifier,
        accuracy_score,
        classification_report,
        clear_vram,
        confusion_matrix,
        contextlib,
        datetime,
        f1_score,
        get_boosting_kwargs,
        get_hardware_config,
        get_rain_australia_mock,
        go,
        html,
        joblib,
        json,
        log_system_info,
        logger,
        logging,
        make_subplots,
        mlflow,
        mo,
        np,
        optuna,
        os,
        pd,
        permutation_importance,
        pl,
        plot_optimization_history,
        plt,
        precision_recall_curve,
        px,
        re,
        roc_auc_score,
        roc_curve,
        sch,
        set_global_seed,
        shap,
        smart_read_csv,
        ssd,
        textwrap,
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
        log_system_info()  # 🔍 Друкуємо інформацію про ОС та Python у логи
        set_global_seed(GLOBAL_SEED)

        # Детектимо залізо для PyTorch (Нейромережі)
        device, device_ui_name = get_hardware_config(global_seed=GLOBAL_SEED)

        # Перекладаємо залізо для Бустингів (Дерева)
        xgb_kwargs, lgbm_kwargs = get_boosting_kwargs(device)

        # 3. Гнучке налаштування MLflow для КОНКРЕТНОГО завдання
        experiment_name = "hw02_rain_classification"
        mlflow.set_experiment(experiment_name)

        # 📝 Фіксуємо подію в глобальний аудит-лог
        logger.info(f"✅ Налаштовано експеримент MLflow: {experiment_name}")
    return GLOBAL_SEED, device, device_ui_name, lgbm_kwargs, xgb_kwargs


@app.cell(hide_code=True)
def header_prepare_dataset(mo):
    mo.md("""
    <h2 align="center"><b>💽 1. Завантаження даних, OOT Split та Smart EDA</b></h2>
    """)
    return


@app.cell
def execute_etl_pipeline(
    SecureDownloader,
    UA_COLUMNS,
    get_rain_australia_mock,
    go,
    logger,
    make_subplots,
    mo,
    np,
    os,
    pd,
    smart_read_csv,
):
    # 🎭 Динамічно знімаємо ліміт пам'яті Marimo через приватний API
    try:
        mo._runtime.context.get_context().marimo_config["runtime"][
            "output_max_bytes"
        ] = 50_000_000  # Ставимо 50 МБ для надійності
    except Exception:
        pass  # Запобіжник на випадок, якщо Marimo колись оновить архітектуру

    _data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    logger.info("Починаємо ETL-процес для Weather AUS...")

    with mo.status.spinner(title="Завантаження, атомарне очищення та Downcasting даних..."):
        _downloader = SecureDownloader(
            dataset_path="jsphyg/weather-dataset-rattle-package",
            dataset_url="https://github.com/zayaanmoez/eda-weather-aus/archive/refs/heads/master.zip",
            fallback_generator=get_rain_australia_mock,
            data_dir=_data_path,
            zip_name="weather_aus.zip"
        )
        _downloader.download(target_filename="weatherAUS.csv")
        _extracted = _downloader.extract_atomically(target_extensions=('.csv',), expected_filename="weatherAUS.csv")

        # ⚡ Читаємо сирі дані
        df_raw = smart_read_csv(_extracted[0], "Weather AUS", engine="pyarrow")
        _df = df_raw.copy()

        # Початок очищення
        _df = _df.dropna(subset=['RainTomorrow']).copy()

        # Видалення дублікатів
        _duplicates_count = _df.duplicated().sum()
        _df.drop_duplicates(inplace=True)
        logger.info(f"Видалено повних дублікатів: {_duplicates_count}")

        # Тотальний Downcasting (Тепер і Float, і Int стискаються)
        for _col in _df.select_dtypes(include=[np.float64]).columns:
            _df[_col] = _df[_col].astype(np.float32)
        for _col in _df.select_dtypes(include=[np.int64]).columns:
            _df[_col] = _df[_col].astype(np.int32)

        # Очищення сильно розріджених ознак
        _missing_matrix = _df.isnull().mean()
        _cols_to_drop = _missing_matrix[_missing_matrix > 0.3].index.tolist()
        _df.drop(columns=_cols_to_drop, inplace=True)

        # Кругове тригонометричне кодування вітру
        _wind_compass = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        for _w_col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
            if _w_col in _df.columns:
                # Прибрано fillna(0), щоб уникнути штучного "Північного вітру"
                _df[_w_col + '_Rad'] = _df[_w_col].map(_wind_compass).apply(np.radians)
                _df[_w_col + '_Sin'] = np.sin(_df[_w_col + '_Rad']).astype(np.float32)
                _df[_w_col + '_Cos'] = np.cos(_df[_w_col + '_Rad']).astype(np.float32)
                _df.drop(columns=[_w_col, _w_col + '_Rad'], inplace=True)

        # Out-of-Time (OOT) Split
        _df['Date'] = pd.to_datetime(_df['Date'])
        _df['Year'] = _df['Date'].dt.year.astype(np.int32)

        _month_num = _df['Date'].dt.month
        _df['Month_Sin'] = np.sin(2 * np.pi * _month_num / 12).astype(np.float32)
        _df['Month_Cos'] = np.cos(2 * np.pi * _month_num / 12).astype(np.float32)

        max_year = _df['Year'].max()
        _train_mask = _df['Year'] < max_year
        _test_mask = _df['Year'] == max_year

        _X = _df.drop(columns=['RainTomorrow', 'Date'])
        _y = _df['RainTomorrow'].map({'Yes': 1, 'No': 0}).astype(np.int8)

        X_train = _X[_train_mask].copy()
        X_test = _X[_test_mask].copy()
        y_train = _y[_train_mask].copy()
        y_test = _y[_test_mask].copy()

        target_col = 'RainTomorrow'
        all_features = [col for col in _df.columns if col != target_col and col != 'Date']
        cont_features = [col for col in all_features if _df[col].nunique() >= 25]

        _theme = mo.app_meta().theme
        _template = "plotly_dark" if _theme == "dark" else "plotly_white"
        _text_color = "white" if _theme == "dark" else "#1f2937"

        _plot_feats = [f for f in cont_features if pd.api.types.is_numeric_dtype(_df[f])][:6]

        _fig_eda = make_subplots(
            rows=2, cols=3,
            subplot_titles=[UA_COLUMNS.get(f, f) for f in _plot_feats],
            vertical_spacing=0.15
        )

        for i, feat in enumerate(_plot_feats):
            _row = i // 3 + 1
            _col = i % 3 + 1
            _ua_name = UA_COLUMNS.get(feat, feat)

            _fig_eda.add_trace(
                go.Histogram(
                    x=_df[feat],
                    name=_ua_name,
                    marker_color='#3b82f6',
                    nbinsx=40,
                    hovertemplate="<b>Діапазон:</b> %{x}<br><b>Кількість:</b> %{y}<extra></extra>"
                ),
                row=_row,
                col=_col
            )

            _fig_eda.update_xaxes(title_text="Значення", title_font=dict(size=11, color='gray'), row=_row, col=_col)

            if feat == "Rainfall":
                _fig_eda.update_yaxes(title_text="Кількість (Log)", type="log", title_font=dict(size=11, color='gray'), row=_row, col=_col)
            else:
                _fig_eda.update_yaxes(title_text="Кількість", title_font=dict(size=11, color='gray'), row=_row, col=_col)

        _fig_eda.update_layout(
            showlegend=False,
            template=_template,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                text="<b>Розподіл неперервних ознак</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=20, color=_text_color)
            ),
            height=600,
            margin=dict(t=70, b=40, l=40, r=20)
        )

    logger.info("ETL-процес успішно завершено.")
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

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

    mo.output.append(
        mo.center(
            mo.md(f"✅ **Дані успішно завантажено!** Розмір сирого набору даних: `(Рядків: {df_raw.shape[0]} | Стовпчиків: {df_raw.shape[1]})`")
        )
    )

    mo.output.append(mo.vstack([
        _css_no_scroll,
        mo.md(f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-top: 15px;">
            <h3 style="margin-top: 0; color: #10b981;">🚀 Пайплайн виконано: Дані захищено від витоку (Zero Data Leakage)</h3>
            <ul>
                <li><b>Стиснення RAM:</b> Ознаки конвертовано у високопродуктивний тип <code>float32 / int32</code>. Видалено <b>{_duplicates_count}</b> дублікатів.</li>
                <li><b>Круговий час та простір:</b> Напрямки вітру та Місяці року переведені у безперервні хвилі Sin/Cos (Без штучних заповнень).</li>
                <li><b>Out-of-Time Спліт:</b> Тестова вибірка ізольована суворо за останнім роком спостережень (<b>{max_year} рік</b>).</li>
            </ul>
        </div>
        """),
        mo.ui.plotly(_fig_eda)
    ]))
    return X_test, X_train, df_raw, y_test, y_train


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
    df_raw,
    html,
    mo,
    os,
    re,
):
    with mo.status.spinner(title="Генерація інтерактивного профайлінгу..."):
        df_eda = df_raw.copy()

        # 🌌 Відправляємо весь консольний спам у "чорну діру"
        with (
            open(os.devnull, "w") as fnull,
            contextlib.redirect_stdout(fnull),
            contextlib.redirect_stderr(fnull),
        ):
            profile = ProfileReport(
                df_eda,
                title="Weather AUS Profiling Report",
                minimal=True,
                progress_bar=False,
            )
            html_string = profile.to_html()

            # Динамічна та безпечна назва артефакту
            artifact_dir = os.getenv("MODELS_DIR", "./models")
            os.makedirs(artifact_dir, exist_ok=True)
            report_filename = "hw02_australia_weather_eda.html"
            profile.to_file(os.path.join(artifact_dir, report_filename))

    # Перейменовуємо колонки лише для візуалізації, зберігаючи df_raw недоторканим для ML-ядра
    df_display = df_raw.rename(columns=UA_COLUMNS)

    # ⚡ Нативна таблиця з українськими заголовками
    df_explorer = mo.ui.table(df_display, selection=None, pagination=True)

    html_string = re.sub(
        r'<a\s+([^>]*)href=["\']#([^"\']+)["\']([^>]*)>',
        r'<a \1 href="javascript:void(0);" data-target="#\2" data-bs-target="#\2" onclick="var el=document.getElementById(\'\2\'); if(el) el.scrollIntoView({behavior: \'smooth\'});" \3>',
        html_string
    )

    # Замість Base64 (який падає через ліміти URL), використовуємо DOM-атрибут srcdoc
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
def plot_correlation_matrix(UA_COLUMNS, df_raw, mo, np, px, sch, ssd):
    numeric_df = df_raw.select_dtypes(include=["float32", "float64", "int32", "int64"])
    corr_matrix = numeric_df.corr(method="pearson")

    dists = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(dists, 0)
    linkage = sch.linkage(ssd.squareform(dists), method='ward')
    optimal_order = sch.leaves_list(linkage)
    corr_matrix_sorted = corr_matrix.iloc[optimal_order, optimal_order]

    corr_matrix_sorted = corr_matrix_sorted.rename(columns=UA_COLUMNS, index=UA_COLUMNS)

    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"

    fig_corr = px.imshow(
        corr_matrix_sorted,
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
            text="<b>Теплова карта кореляцій метеоданих (Smart Clustered)</b>",
            x=0.5,
            font=dict(color=_text_color, size=18),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_text_color),
        height=850,
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
        > **💡 Tech Lead Insight (Мультиколінеарність):**<br/>
        > Подивіться на темно-червоні квадрати (наприклад, між `Атм. тиск (9 ранку)` та `Атм. тиск (3 дня)`). Це яскраво виражена мультиколінеарність. Звичайна логістична регресія "ламається" на таких ознаках, розмиваючи ваги між ними. Саме тому ми застосували **ElasticNet-регуляризацію** — вона обходить цей недолік, автоматично ізолюючи дублюючі фактори.
        """
    )

    mo.output.append(mo.vstack([fig_corr, _insight_ui]))
    return


@app.cell(hide_code=True)
def header_model_configurator(mo):
    mo.md("""
    <h2 align="center"><b>🕹️ 2. Конфігурація пулу алгоритмів <i>(Classification Pool)</i></b></h2>
    """)
    return


@app.cell
def model_data_state(
    AdaBoostClassifier,
    BaggingClassifier,
    BernoulliNB,
    CalibratedClassifierCV,
    DecisionTreeClassifier,
    DummyClassifier,
    ExplainableBoostingClassifier,
    ExtraTreeClassifier,
    ExtraTreesClassifier,
    GLOBAL_SEED,
    GaussianNB,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ImbPipeline,
    KNeighborsClassifier,
    LGBMClassifier,
    LinearDiscriminantAnalysis,
    LinearSVC,
    LogisticRegression,
    MLPClassifier,
    NearestCentroid,
    QuadraticDiscriminantAnalysis,
    RandomForestClassifier,
    RidgeClassifier,
    SGDClassifier,
    SMOTE,
    SVC,
    StandardScaler,
    XGBClassifier,
    XGBRFClassifier,
    lgbm_kwargs,
    mo,
    xgb_kwargs,
):
    # 1. Лінійні та Базові моделі (Linear & Baseline)
    models_linear = {
        "Dummy (Mode Baseline)": (1, DummyClassifier(strategy="prior")),
        "Logistic Regression (OHE+SMOTE)": (2, ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=GLOBAL_SEED)), ('regressor', LogisticRegression(l1_ratio=0.5, solver='saga', max_iter=1000, random_state=GLOBAL_SEED))])),
        # Обгортаємо не-імовірнісні лінійні моделі у CalibratedClassifierCV
        "Ridge Classifier (OHE+SMOTE)": (3, ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=GLOBAL_SEED)), ('regressor', CalibratedClassifierCV(RidgeClassifier(random_state=GLOBAL_SEED), cv=3))])),
        "SGD Classifier (Log Loss)": (4, SGDClassifier(class_weight='balanced', loss='log_loss', penalty='elasticnet', l1_ratio=0.5, random_state=GLOBAL_SEED)),
        "SGD Classifier (Hinge/PA)": (5, CalibratedClassifierCV(SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, class_weight='balanced', random_state=GLOBAL_SEED), cv=3)),
        "Linear SVC (SVM)": (6, CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=2000, random_state=GLOBAL_SEED), cv=3))
    }

    # 2. Імовірнісні та Дистанційні (Probabilistic & Instance)
    models_prob_inst = {
        "Gaussian Naive Bayes": (7, GaussianNB()),
        "Bernoulli Naive Bayes": (8, BernoulliNB()),
        "Linear Discriminant (LDA)": (9, LinearDiscriminantAnalysis()),
        "Quadratic Discriminant (QDA)": (10, QuadraticDiscriminantAnalysis(reg_param=0.1)),
        "KNN (Instance-based)": (11, ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=GLOBAL_SEED)), ('regressor', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])),
        "Nearest Centroid": (12, ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=GLOBAL_SEED)), ('regressor', NearestCentroid())])) # Nearest Centroid не підтримує predict_proba, але для нього це ок
    }

    # 3. Дерева рішень та Нейромережі (Trees, NN & Non-linear SVM)
    models_trees_nn = {
        "Decision Tree (CART)": (13, DecisionTreeClassifier(class_weight='balanced', max_depth=15, random_state=GLOBAL_SEED)),
        "Extra Tree (Single CART)": (14, ExtraTreeClassifier(class_weight='balanced', max_depth=15, random_state=GLOBAL_SEED)),
        # Вмикаємо probability=True для класичного SVC (Platt Scaling під капотом)
        "SVC (RBF Kernel)": (15, ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=GLOBAL_SEED)), ('regressor', SVC(kernel='rbf', probability=True, class_weight='balanced', max_iter=2000, random_state=GLOBAL_SEED))])),
        "SVC (Poly Kernel)": (16, ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=GLOBAL_SEED)), ('regressor', SVC(kernel='poly', degree=3, probability=True, class_weight='balanced', max_iter=2000, random_state=GLOBAL_SEED))])),
        "MLP Neural Net (Shallow)": (17, ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=GLOBAL_SEED)), ('regressor', MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=GLOBAL_SEED))])),
        "MLP Neural Net (Deep)": (18, ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=GLOBAL_SEED)), ('regressor', MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=300, random_state=GLOBAL_SEED))]))
    }

    # 4. Класичні Ансамблі (Classical Ensembles)
    models_ensembles = {
        "Random Forest (OHE)": (19, RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=15, random_state=GLOBAL_SEED, n_jobs=-1)),
        "Extra Trees Ensemble": (20, ExtraTreesClassifier(class_weight='balanced', n_estimators=100, max_depth=15, random_state=GLOBAL_SEED, n_jobs=-1)),
        "Bagging Classifier": (21, BaggingClassifier(n_estimators=50, random_state=GLOBAL_SEED, n_jobs=-1)),
        "AdaBoost (1995)": (22, AdaBoostClassifier(n_estimators=100, random_state=GLOBAL_SEED)),
        "Gradient Boosting (Classic)": (23, GradientBoostingClassifier(n_estimators=200, random_state=GLOBAL_SEED)),
        "XGBoost Random Forest": (24, XGBRFClassifier(scale_pos_weight=4, n_estimators=200, random_state=GLOBAL_SEED, eval_metric='logloss', **xgb_kwargs))
    }

    # 5. Передовий Бустинг та Нативні (Advanced Boosting)
    models_advanced = {
        "Scikit HistGradient (OHE)": (25, HistGradientBoostingClassifier(max_iter=200, random_state=GLOBAL_SEED)),
        "LightGBM (OHE)": (26, LGBMClassifier(class_weight='balanced', random_state=GLOBAL_SEED, **lgbm_kwargs)),
        "XGBoost (OHE)": (27, XGBClassifier(scale_pos_weight=4, n_estimators=200, random_state=GLOBAL_SEED, eval_metric='logloss', **xgb_kwargs)),
        "LightGBM (Native)": (28, LGBMClassifier(class_weight='balanced', random_state=GLOBAL_SEED, **lgbm_kwargs)),
        "XGBoost (Native)": (29, XGBClassifier(scale_pos_weight=4, n_estimators=200, enable_categorical=True, random_state=GLOBAL_SEED, eval_metric='logloss', **xgb_kwargs)),
        "Explainable Boosting (EBM)": (30, ExplainableBoostingClassifier(random_state=GLOBAL_SEED, n_jobs=-1))
    }

    master_registry = {**models_linear, **models_prob_inst, **models_trees_nn, **models_ensembles, **models_advanced}
    id_to_name_map = {f"#{mod_id:02d}": name for name, (mod_id, _) in master_registry.items()}

    get_force_lin, set_force_lin = mo.state(True)
    get_force_pro, set_force_pro = mo.state(True)
    get_force_tre, set_force_tre = mo.state(True)
    get_force_ens, set_force_ens = mo.state(True)
    get_force_adv, set_force_adv = mo.state(True)

    mo.center(mo.md(f"✅ **Всі {len(master_registry)} класифікаторів (з підтримкою калібрування) завантажено у памʼять!**"))
    return (
        get_force_adv,
        get_force_ens,
        get_force_lin,
        get_force_pro,
        get_force_tre,
        id_to_name_map,
        master_registry,
        models_advanced,
        models_ensembles,
        models_linear,
        models_prob_inst,
        models_trees_nn,
        set_force_adv,
        set_force_ens,
        set_force_lin,
        set_force_pro,
        set_force_tre,
    )


@app.cell
def controller_ui(
    get_force_adv,
    get_force_ens,
    get_force_lin,
    get_force_pro,
    get_force_tre,
    mo,
    models_advanced,
    models_ensembles,
    models_linear,
    models_prob_inst,
    models_trees_nn,
):
    force_lin = get_force_lin()
    force_pro = get_force_pro()
    force_tre = get_force_tre()
    force_ens = get_force_ens()
    force_adv = get_force_adv()

    mandatory_models = [
        "Dummy (Mode Baseline)",
        "Logistic Regression (OHE+SMOTE)",
        "Random Forest (OHE)",
        "XGBoost (Native)",
        "Explainable Boosting (EBM)"
    ]

    def make_cb(name, force_state):
        is_locked = name in mandatory_models
        return mo.ui.checkbox(label=name, value=True if is_locked else force_state, disabled=is_locked)

    ui_lin = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_lin) for name, (mod_id, _) in models_linear.items()})
    ui_pro = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_pro) for name, (mod_id, _) in models_prob_inst.items()})
    ui_tre = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_tre) for name, (mod_id, _) in models_trees_nn.items()})
    ui_ens = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_ens) for name, (mod_id, _) in models_ensembles.items()})
    ui_adv = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_adv) for name, (mod_id, _) in models_advanced.items()})

    mo.center(mo.md("✅ **Словники інтерфейсу створено!**"))
    return ui_adv, ui_ens, ui_lin, ui_pro, ui_tre


@app.cell
def view_render(
    mo,
    set_force_adv,
    set_force_ens,
    set_force_lin,
    set_force_pro,
    set_force_tre,
    ui_adv,
    ui_ens,
    ui_lin,
    ui_pro,
    ui_tre,
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

    h_lin, c_lin, t_lin, _ = build_group_view(ui_lin, "Лінійні 🐣", set_force_lin)
    h_pro, c_pro, t_pro, _ = build_group_view(ui_pro, "Імовірність 🎲", set_force_pro)
    h_tre, c_tre, t_tre, _ = build_group_view(ui_tre, "Дерева/НМ 🧠", set_force_tre)
    h_ens, c_ens, t_ens, _ = build_group_view(ui_ens, "Ансамблі 🥁", set_force_ens)
    h_adv, c_adv, t_adv, _ = build_group_view(ui_adv, "Сучасний Бустинг 🫀", set_force_adv)

    total_selected = c_lin + c_pro + c_tre + c_ens + c_adv
    total_all = t_lin + t_pro + t_tre + t_ens + t_adv

    main_header = mo.hstack([
        mo.md("🎛️ **Ультимативний Конфігуратор (God Mode)**"),
        mo.md(f"<div style='text-align: right; color: #10b981; font-size: 1.1em;'><b>✓ Всього обрано: {total_selected} / {total_all}</b></div>")
    ], justify="space-between", align="center")

    run_btn = mo.ui.run_button(label="🎭 Запустити тренування пулу моделей", kind="success")
    v_line = mo.Html("<div style='width: 1px; background-color: #4b5563; min-height: 240px; margin: 0 15px; margin-top: 15px;'></div>")

    def build_column(header, ui_group):
        items_with_ids = [mo.hstack([mo.md(f"`{k}`"), cb], align="center") for k, cb in ui_group.items()]
        return mo.vstack([header, mo.md("<div style='height: 10px;'></div>"), mo.vstack(items_with_ids, align="start")], align="center")

    _css_no_scroll = mo.md('<div class="config-noscroll"></div><style>marimo-cell-output:has(.config-noscroll),.output-area:has(.config-noscroll){max-height: none !important; overflow-y: visible !important; overflow-x: visible !important;}</style>')

    config_panel = mo.vstack([
        _css_no_scroll,
        mo.center(main_header),
        mo.hstack([
            build_column(h_lin, ui_lin), v_line,
            build_column(h_pro, ui_pro), v_line,
            build_column(h_tre, ui_tre), v_line,
            build_column(h_ens, ui_ens), v_line,
            build_column(h_adv, ui_adv)
        ], justify="space-between", align="start"),
        mo.center(run_btn)
    ])
    mo.output.append(config_panel)
    return (run_btn,)


@app.cell(hide_code=True)
def header_benchmark_execution(mo):
    mo.md("""
    <h3 align="center"><b>🏋️‍♂️ 3. Тренування алгоритмів та Лідерборд <i>(Model Leaderboard & Diagnostics)</i></b></h3>
    """)
    return


@app.cell
def execute_benchmark(
    ColumnTransformer,
    ConvergenceWarning,
    ImbPipeline,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
    X_test,
    X_train,
    accuracy_score,
    clear_vram,
    confusion_matrix,
    f1_score,
    go,
    id_to_name_map,
    logger,
    logging,
    make_subplots,
    master_registry,
    mo,
    np,
    pd,
    pl,
    roc_auc_score,
    roc_curve,
    run_btn,
    ui_adv,
    ui_ens,
    ui_lin,
    ui_pro,
    ui_tre,
    warnings,
    xgb_kwargs,
    y_test,
    y_train,
):
    mo.stop(not run_btn.value, mo.center(mo.md("### ⏳ Очікування конфігурації...\n> 🆘 Оберіть алгоритми у Конфігураторі вище та натисніть зелену кнопку.")))

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", module="interpret.*")
    logging.getLogger("interpret").setLevel(logging.ERROR)

    logger.info("Початок бенчмаркінгу обраних моделей...")

    _hw_type = xgb_kwargs.get("device", "cpu")
    _hw_ui = "CUDA GPU" if _hw_type == "cuda" else "Intel XPU" if _hw_type == "sycl" else "Multi-core CPU"

    selected_names = []
    for ui_group in [ui_lin, ui_pro, ui_tre, ui_ens, ui_adv]:
        selected_names.extend([id_to_name_map[mod_id] for mod_id, is_sel in ui_group.value.items() if is_sel])

    mo.stop(not selected_names, mo.md("⚠️ **Неможливо запустити: не обрано жодного алгоритму!**"))

    all_models = [(name, master_registry[name]) for name in selected_names]
    results = []
    trained_models = {}

    # Замість довіри зовнішнім спискам, самі скануємо типи даних у X_train
    _num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    _cat_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()

    # Використовуємо ImbPipeline для сумісності зі SMOTE
    preprocessor_ohe = ColumnTransformer([
        ('num', ImbPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), _num_cols),
        ('cat', ImbPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), _cat_cols)
    ])

    preprocessor_native = ColumnTransformer([
        ('num', ImbPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), _num_cols),
        ('cat', ImbPipeline([('imputer', SimpleImputer(strategy='most_frequent'))]), _cat_cols)
    ], remainder='passthrough')

    X_train_ohe = preprocessor_ohe.fit_transform(X_train)
    X_test_ohe = preprocessor_ohe.transform(X_test)
    X_train_native = preprocessor_native.fit_transform(X_train)
    X_test_native = preprocessor_native.transform(X_test)

    # Залізобетонна конвертація у 'category' ПІСЛЯ імп'ютації (Для Native моделей)
    for c in X_train_native.columns:
        if X_train_native[c].dtype in ['object', 'string']:
            X_train_native[c] = X_train_native[c].astype('category')
            X_test_native[c] = X_test_native[c].astype('category')

    _native_names = {"LightGBM (Native)", "XGBoost (Native)", "Explainable Boosting (EBM)"}

    with mo.status.progress_bar(total=len(all_models), title=f"Тренування {len(all_models)} моделей...", subtitle=f"💎 Engine: {_hw_ui}", remove_on_exit=True) as bar:
        for name, (mod_id, _model) in all_models:
            bar.update(increment=0, subtitle=f"💎 <b>Engine:</b> {_hw_ui} <br/>☣️ <b>Тренуємо:</b> {name}")

            _is_native = name in _native_names
            _X_train_curr = X_train_native if _is_native else X_train_ohe
            _X_test_curr = X_test_native if _is_native else X_test_ohe

            _model.fit(_X_train_curr, y_train)
            y_pred = _model.predict(_X_test_curr)

            if hasattr(_model, "predict_proba"):
                y_proba = _model.predict_proba(_X_test_curr)[:, 1]
            elif hasattr(_model, "decision_function"):
                y_proba = 1 / (1 + np.exp(-_model.decision_function(_X_test_curr)))
            else:
                y_proba = y_pred

            _roc = roc_auc_score(y_test, y_proba)
            _tn, _fp, _fn, _tp = confusion_matrix(y_test, y_pred).ravel()
            _specificity = _tn / (_tn + _fp) if (_tn + _fp) > 0 else 0

            trained_models[name] = _model
            results.append({
                "ID": f"#{mod_id:02d}",
                "Алгоритм": name,
                "ROC-AUC ⬆️": _roc,
                "F1-Score ⬆️": f1_score(y_test, y_pred, zero_division=0), # Захист від ділення на нуль
                "Specificity (TN Rate) ⬆️": _specificity,
                "Accuracy ⬆️": accuracy_score(y_test, y_pred)
            })
            bar.update()

    logger.info(f"Бенчмаркінг {len(all_models)} моделей завершено.")
    df_results = pd.DataFrame(results).sort_values(by="ROC-AUC ⬆️", ascending=False).reset_index(drop=True).copy()

    df_results["ROC-AUC ⬆️"] = df_results["ROC-AUC ⬆️"].round(4)
    df_results["F1-Score ⬆️"] = df_results["F1-Score ⬆️"].round(4)
    df_results["Specificity (TN Rate) ⬆️"] = df_results["Specificity (TN Rate) ⬆️"].round(4)
    df_results["Accuracy ⬆️"] = df_results["Accuracy ⬆️"].round(4)

    clear_vram(None)

    _display_data = None
    try: _display_data = pl.from_pandas(df_results)
    except Exception: _display_data = df_results.to_dict(orient="records")

    _justify_config = {col: "center" for col in df_results.columns}
    _benchmark_table = mo.ui.table(_display_data, selection=None, page_size=50, text_justify_columns=_justify_config, label="🏆 **Результати A/B тестування класифікаторів:**")

    # =========================================================================
    # 📊 ВІЗУАЛІЗАЦІЯ ЕВОЛЮЦІЇ АЛГОРИТМІВ (2 колонки x 3 ряди)
    # =========================================================================
    benchmark_theme = mo.app_meta().theme
    bench_border = "#4b5563" if benchmark_theme == "dark" else "#e5e7eb"
    bench_text = "white" if benchmark_theme == "dark" else "#1f2937"

    def get_model_prob_safe(model_name, X_data):
        if model_name not in trained_models: return None, 0, 0, 0
        _m = trained_models[model_name]
        if hasattr(_m, "predict_proba"): proba = _m.predict_proba(X_data)[:, 1]
        elif hasattr(_m, "decision_function"): proba = 1 / (1 + np.exp(-_m.decision_function(X_data)))
        else: proba = _m.predict(X_data)

        row = df_results[df_results["Алгоритм"] == model_name]
        roc, f1, spec = (row["ROC-AUC ⬆️"].iloc[0], row["F1-Score ⬆️"].iloc[0], row["Specificity (TN Rate) ⬆️"].iloc[0]) if not row.empty else (0, 0, 0)
        return proba, roc, f1, spec

    grid_configs = [
        {"ukr": "🐣 1. Лінійний Baseline", "eng": "Logistic Regression (OHE+SMOTE)", "x": X_test_ohe, "row": 1, "col": 1},
        {"ukr": "🎲 2. Імовірнісний підхід", "eng": "Gaussian Naive Bayes", "x": X_test_ohe, "row": 1, "col": 2},
        {"ukr": "🧠 3. Дерева Рішень", "eng": "Decision Tree (CART)", "x": X_test_ohe, "row": 2, "col": 1},
        {"ukr": "🥁 4. Класичні Ансамблі", "eng": "Random Forest (OHE)", "x": X_test_ohe, "row": 2, "col": 2},
        {"ukr": "🫀 5. Сучасний Бустинг", "eng": "XGBoost (Native)", "x": X_test_native, "row": 3, "col": 1},
        {"ukr": "🔬 6. Explainable AI", "eng": "Explainable Boosting (EBM)", "x": X_test_native, "row": 3, "col": 2}
    ]

    subplot_titles, plot_data_cache = [], {}
    for cfg in grid_configs:
        proba, roc, f1, spec = get_model_prob_safe(cfg["eng"], cfg["x"])
        if proba is not None:
            title = f"{cfg['ukr']}<br><span style='font-size:12px; color:gray;'>{cfg['eng']}</span><br><span style='font-size:13px;'>ROC: {roc} | F1: {f1} | Spec: {spec}</span>"
            plot_data_cache[cfg["eng"]] = proba
        else:
            title = f"{cfg['ukr']}<br><span style='font-size:12px; color:gray;'>{cfg['eng']}</span><br>❌ <i>Вимкнено</i>"
            plot_data_cache[cfg["eng"]] = None
        subplot_titles.append(title)

    _fig_diag = make_subplots(rows=3, cols=2, subplot_titles=subplot_titles, horizontal_spacing=0.08, vertical_spacing=0.15)

    y_test_arr = np.array(y_test)
    for cfg in grid_configs:
        proba = plot_data_cache[cfg["eng"]]
        if proba is not None:
            # Малюємо гістограми розподілу ймовірностей (Розділення класів)
            _fig_diag.add_trace(go.Histogram(
                x=proba[y_test_arr==0], name="Без опадів", marker_color='#3b82f6', opacity=0.6, nbinsx=30,
                showlegend=(cfg["row"]==1 and cfg["col"]==1),
                hovertemplate="<b>Клас:</b> Без опадів<br><b>Ймовірність:</b> %{x}<br><b>Кількість днів:</b> %{y}<extra></extra>"
            ), row=cfg["row"], col=cfg["col"])

            _fig_diag.add_trace(go.Histogram(
                x=proba[y_test_arr==1], name="Дощ", marker_color='#ef4444', opacity=0.6, nbinsx=30,
                showlegend=(cfg["row"]==1 and cfg["col"]==1),
                hovertemplate="<b>Клас:</b> Дощ<br><b>Ймовірність:</b> %{x}<br><b>Кількість днів:</b> %{y}<extra></extra>"
            ), row=cfg["row"], col=cfg["col"])
        else:
            _fig_diag.add_annotation(text="⚠️ Модель не обрана", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"), row=cfg["row"], col=cfg["col"])

        _fig_diag.update_xaxes(title_text="Ймовірність Дощу (0 до 1)", range=[0, 1], gridcolor=bench_border, row=cfg["row"], col=cfg["col"])
        _fig_diag.update_yaxes(title_text="Кількість", autorange=True, gridcolor=bench_border, row=cfg["row"], col=cfg["col"])

    _fig_diag.update_layout(
        title=dict(text="<b>Еволюція Класифікаторів: Здатність розділяти класи (Дощ проти Без опадів)</b>", x=0.5, xanchor="center", y=0.98, font=dict(color=bench_text, size=18)),
        barmode='overlay', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=bench_text), height=1100, margin=dict(l=0, r=0, b=40, t=100)
    )

    # =========================================================================
    # 📝 ДИНАМІЧНІ ВИСНОВКИ (INSIGHTS)
    # =========================================================================
    top_roc = df_results["ROC-AUC ⬆️"].iloc[0]
    top_f1 = df_results["F1-Score ⬆️"].iloc[0]
    _dynamic_bullets = []

    # Перевіряємо, чи обрана хоча б одна модель з групи Лінійних (ui_lin) або Імовірнісних (ui_pro)
    if any(ui_lin.value.values()) or any(ui_pro.value.values()):
        _dynamic_bullets.append("> 1. **Базові моделі (Лінійні/Наївний Баєс):** Їхні прогнози часто \"розмазані\" по центру. Вони невпевнені у своїх рішеннях.")

    # Перевіряємо групу Дерев та Нейромереж (ui_tre)
    if any(ui_tre.value.values()):
        _dynamic_bullets.append("> 2. **Дерева рішень (та базові НМ):** Працюють грубо. Замість плавного розподілу, дерева збирають ймовірності у кілька жорстких \"стовпів\".")

    # Перевіряємо групи Класичних Ансамблів (ui_ens) або Сучасного Бустингу (ui_adv)
    if any(ui_ens.value.values()) or any(ui_adv.value.values()):
        _dynamic_bullets.append("> 3. **Сучасні Ансамблі та Бустинг:** Вершина еволюції. Вони максимально \"розштовхують\" класи по кутках, мінімізуючи зону перетину в центрі.")

    if not _dynamic_bullets:
        _dynamic_bullets.append("> *Для поточного набору алгоритмів специфічні візуальні інсайти не згенеровані. Спробуйте додати базові моделі та бустинги для порівняння.*")

    _insights_text = "\n".join(_dynamic_bullets)

    _benchmark_insight = mo.md(
        "> **📊 Як читати метрики лідерборду:**\n"
        "\n"
        f"> - **ROC-AUC ⬆️:** Оцінка `~{top_roc:.3f}` означає, що якби ми взяли один випадковий дощовий день і один сонячний, наша найкраща модель у `{int(top_roc*100)}%` випадків правильно дала б дощовому дню вищу ймовірність!\n"
        "> - **F1-Score ⬆️:** Баланс між Точністю (Precision) та Повнотою (Recall). Рятує нас від ілюзії високої Accuracy на незбалансованих даних (адже сонячних днів в Австралії набагато більше).\n"
        "> - **Specificity (TN Rate) ⬆️:** Відсоток правильно вгаданих сонячних днів. Допомагає не змушувати користувача брати парасольку дарма (False Positives).\n"
        "\n"
        "> **💡 Tech Lead Insight (Аналіз гістограм):**\n"
        "\n"
        "> Подивіться на графіки вище. Ідеальна модель повинна зсунути всі сині стовпчики вліво (до 0.0), а всі червоні — вправо (до 1.0).\n"
        "\n"
        f"{_insights_text}"
    )

    # =========================================================================
    # 👑 ГРАФІК АБСОЛЮТНОГО ЧЕМПІОНА (ROC Curve)
    # =========================================================================
    champion_name = df_results["Алгоритм"].iloc[0]
    is_champ_native = champion_name in _native_names
    X_test_champ = X_test_native if is_champ_native else X_test_ohe
    proba_champ, roc_champ, f1_champ, spec_champ = get_model_prob_safe(champion_name, X_test_champ)

    fig_champion = go.Figure()
    if proba_champ is not None:
        fpr, tpr, _ = roc_curve(y_test, proba_champ)
        fig_champion.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', line=dict(color='#8b5cf6', width=4),
            name=f'{champion_name} (AUC = {roc_champ:.3f})', fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)',
            hovertemplate="<b>Частка хибних (FPR):</b> %{x:.4f}<br><b>Частка істинних (TPR):</b> %{y:.4f}<extra></extra>"
        ))
        fig_champion.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color=bench_text, dash='dash'), showlegend=False, hoverinfo='skip'))

    fig_champion.update_xaxes(title_text="Частка хибнопозитивних (FPR)", gridcolor=bench_border, range=[0, 1])
    fig_champion.update_yaxes(title_text="Частка істиннопозитивних (TPR)", gridcolor=bench_border, range=[0, 1])
    fig_champion.update_layout(
        title=dict(text=f"<b>👑 АБСОЛЮТНИЙ ЧЕМПІОН: {champion_name}</b><br><span style='font-size:15px; color:gray;'>ROC-AUC: {roc_champ} | F1-Score: {f1_champ} | Specificity: {spec_champ}</span>", x=0.5, xanchor="center", y=0.92, font=dict(color=bench_text, size=22)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=bench_text), height=550, margin=dict(l=0, r=0, b=40, t=90), showlegend=True, legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
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

    mo.output.append(_css_no_scroll)
    mo.output.append(_benchmark_table)
    mo.output.append(_benchmark_insight)
    mo.output.append(_fig_diag)
    mo.output.append(fig_champion)
    return (
        X_test_native,
        X_test_ohe,
        X_train_native,
        X_train_ohe,
        df_results,
        preprocessor_native,
        preprocessor_ohe,
        trained_models,
    )


@app.cell(hide_code=True)
def header_model_selector_ui(mo):
    mo.md("""
    <h3 align="center"><b>🎛️ 3.1. Інтерактивний Селектор Моделі <i>(Model Selection)</i></b></h3>
    """)
    return


@app.cell
def model_selector_ui(df_results, master_registry, mo):
    _ranked_names = df_results["Алгоритм"].tolist()
    _dropdown_options = {}
    _top_5 = _ranked_names[:5]
    _rest = _ranked_names[5:]

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
        label="🏆 **Оберіть модель для глибокого аналізу:** "
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
def header_thresholds(mo):
    mo.md("""
    <h3 align="center"><b>🎯 3.2. Оптимізація порогу та Матриця Помилок <i>(Cost-Sensitive Thresholding)</i></b></h3>
    """)
    return


@app.cell(hide_code=True)
def threshold_optimization(
    X_test_native,
    X_test_ohe,
    champion_selector,
    classification_report,
    confusion_matrix,
    go,
    mo,
    np,
    pd,
    precision_recall_curve,
    px,
    style_dataframe,
    trained_models,
    y_test,
):
    _selected_name = champion_selector.value
    _model = trained_models[_selected_name]

    # Локальний реєстр нативних моделей замість фантомного native_keys
    _native_names = {"LightGBM (Native)", "XGBoost (Native)", "Explainable Boosting (EBM)"}
    _is_native = _selected_name in _native_names
    _X_test_curr = X_test_native if _is_native else X_test_ohe

    # Відмовостійкість для моделей без predict_proba (SVM, Ridge, Hinge PA)
    if hasattr(_model, "predict_proba"):
        _y_pred_proba = _model.predict_proba(_X_test_curr)[:, 1]
    elif hasattr(_model, "decision_function"):
        # Використовуємо сигмоїду для перетворення відстані у псевдоймовірність [0, 1]
        _dfunc = _model.decision_function(_X_test_curr)
        _y_pred_proba = 1 / (1 + np.exp(-_dfunc))
    else:
        # Fallback для найпримітивніших класифікаторів
        _y_pred_proba = _model.predict(_X_test_curr)

    _precisions, _recalls, _thresholds = precision_recall_curve(y_test, _y_pred_proba)
    _f1_scores = 2 * (_precisions * _recalls) / (_precisions + _recalls + 1e-10)
    _opt_idx = np.argmax(_f1_scores)
    optimal_threshold = _thresholds[_opt_idx] if _opt_idx < len(_thresholds) else 0.5
    _y_pred_opt = (_y_pred_proba >= optimal_threshold).astype(int)

    _theme = mo.app_meta().theme
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"
    _bg = "#111827" if _theme == "dark" else "#f3f4f6"
    _text = "#e5e7eb" if _theme == "dark" else "#1f2937"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    # =========================================================================
    # 📈 ГРАФІК 1: ОПТИМІЗАЦІЯ ПОРОГУ (F1-SCORE)
    # =========================================================================
    _fig_pr = go.Figure()
    _fig_pr.add_trace(go.Scatter(
        x=_thresholds,
        y=_f1_scores[:-1],
        mode='lines',
        name='F1-Міра',
        line=dict(color='#10b981', width=3),
        hovertemplate="<b>Поріг:</b> %{x:.3f}<br><b>F1-Міра:</b> %{y:.3f}<extra></extra>"
    ))

    _fig_pr.add_vline(
        x=optimal_threshold,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f" Опт. Поріг: {optimal_threshold:.2f} ",
        annotation_position="top",
        annotation_bgcolor="#ef4444",
        annotation_font_color="white",
        annotation_borderpad=4
    )

    _fig_pr.update_layout(
        title=dict(text="<b>Оптимізація порогу прийняття рішення</b>", x=0.5, xanchor="center"),
        xaxis_title="Поріг класифікації (Threshold)",
        yaxis_title="Значення F1-Міри",
        template=_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=40),
        height=350
    )

    # =========================================================================
    # 🧩 ГРАФІК 2: МАТРИЦЯ ПОМИЛОК
    # =========================================================================
    _cm = confusion_matrix(y_test, _y_pred_opt)
    _colorscale = 'Blues' if _theme == 'light' else 'ice'

    _fig_cm = px.imshow(
        _cm,
        text_auto=True,
        color_continuous_scale=_colorscale,
        labels=dict(x="Прогноз моделі", y="Фактичний клас", color="Кількість"),
        x=['Без опадів', 'Дощ'],
        y=['Без опадів', 'Дощ'],
        template=_template
    )
    _fig_cm.update_layout(
        title=dict(text="<b>Матриця помилок (Оптимізована)</b>", x=0.5, xanchor="center"),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False, margin=dict(l=20, r=20, t=60, b=40), height=350
    )

    # =========================================================================
    # 📝 УКРАЇНІЗАЦІЯ ЗВІТУ КЛАСИФІКАЦІЇ (HTML Бронебійна таблиця)
    # =========================================================================
    _report_dict = classification_report(
        y_test,
        _y_pred_opt,
        target_names=['Без опадів', 'Дощ'],
        output_dict=True,
        zero_division=0
    )
    _df_report = pd.DataFrame(_report_dict).T
    _df_report = _df_report.rename(columns={
        'precision': 'Точність (Precision)', 'recall': 'Повнота (Recall)',
        'f1-score': 'F1-Міра', 'support': "Об'єм (Support)"
    })
    _df_report.index = _df_report.index.map(lambda x: {'accuracy': 'Точність (Accuracy)', 'macro avg': 'Середнє макро', 'weighted avg': 'Середнє зважене'}.get(x, x))

    _formatted_df = _df_report.copy()
    if 'Точність (Accuracy)' in _formatted_df.index:
        _formatted_df.loc['Точність (Accuracy)', ['Точність (Precision)', 'Повнота (Recall)']] = np.nan

    for col in _formatted_df.columns:
        if col == "Об'єм (Support)":
            _formatted_df[col] = _formatted_df[col].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "")
        else:
            _formatted_df[col] = _formatted_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    _html_table = style_dataframe(
        _formatted_df,
        show_index=True,
        text_align="center"
    )

    _report_ui = mo.Html(f"""
    <div style='background: {_bg}; border: 1px solid {_border}; padding: 15px; border-radius: 8px; margin-top: 15px;'>
        <b style='color: {_text};'>📋 МЕТРИКИ КЛАСИФІКАЦІЇ:</b><br><br>
        {_html_table}
    </div>
    """)

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

    left_column = mo.vstack([
        mo.ui.plotly(_fig_pr),
        mo.md(f"<div style='text-align: center; margin-top: -10px; font-size: 1.05em;'>🎯 Ідеальний поріг рішення для максимізації F1-Score: <b>{optimal_threshold:.3f}</b></div>"),
        mo.center(_report_ui)
    ], align="stretch")

    right_column = mo.vstack([
        mo.ui.plotly(_fig_cm),
        mo.md("<div style='text-align: center; margin-top: -10px; font-size: 1.05em;'>🧩 Візуалізація розподілу помилок<br/>першого та другого роду</div>")
    ], align="stretch")

    side_by_side_layout = mo.hstack([left_column, right_column], widths=[3, 1], justify="space-between", align="start")

    mo.output.append(mo.vstack([
        _css_no_scroll,
        mo.center(mo.md(f"### 👑 Аналіз Моделі: {_selected_name}")),
        mo.md("<div style='height: 10px;'></div>"),
        side_by_side_layout
    ]))
    return (optimal_threshold,)


@app.cell(hide_code=True)
def header_what_if(mo):
    mo.md("""
    <h3 align="center"><b>🧪 4. What-If Simulator <i>(Інтерактивний Інференс)</i></b></h3>
    """)
    return


@app.cell
def simulator_ui(UA_COLUMNS, X_train, mo):
    _num_cols = X_train.select_dtypes(include=['float32', 'int32', 'float64', 'int64']).columns.tolist()
    _key_features = ['Humidity3pm', 'Pressure3pm', 'WindGustSpeed', 'MaxTemp']
    _sim_features = [f for f in _key_features if f in _num_cols]
    if len(_sim_features) < 4: _sim_features = _num_cols[:4]

    _sliders_raw = {}
    for _feat in _sim_features:
        _min, _max = float(X_train[_feat].min()), float(X_train[_feat].max())
        _mean = float(X_train[_feat].mean())
        _range = _max - _min
        _step = _range / 100.0 if _range > 0 else 0.1

        if str(X_train[_feat].dtype).startswith('int'):
            _step = max(1, int(round(_step)))
            _mean = int(round(_mean))
        else:
            _step = max(0.01, round(_step, 2))
            _mean = round(_mean, 2)

        _sliders_raw[_feat] = mo.ui.slider(
            start=_min, stop=_max, value=_mean, step=_step, show_value=True, full_width=True
        )

    sim_panel = mo.ui.dictionary(_sliders_raw)

    _rows = []
    for _k in _sim_features:
        _ua_label = UA_COLUMNS.get(_k, _k)
        _rows.append(
            mo.hstack(
                [
                    mo.md(f"<div style='width: 250px; text-align: right; padding-right: 20px; font-size: 1.05em;'>🎛️ <b>{_ua_label}</b></div>"),
                    mo.md(f"<div style='width: 600px;'>{sim_panel[_k]}</div>")
                ],
                align="center", justify="center"
            )
        )

    _custom_layout = mo.vstack(_rows, gap="15px")

    _info_md = mo.md(
        """
        Покрутіть погодні показники, щоб побачити, як модель миттєво змінює ймовірність дощу!
        > 💡 **Математика Логіту:**

        > - Логістична регресія оцінює співвідношення шансів ($OR = p / (1 - p)$)
        > - Цільова мітка алгоритму визначається як $\ln(p / (1 - p)) = wx + b$
        > - Змінюючи повзунки, ви безпосередньо впливаєте на вектор $x$, що експоненційно змінює ймовірність $p$ на спідометрі!
        """
    )

    _bg = "#1f2937" if mo.app_meta().theme == "dark" else "#f9fafb"
    _border = "#4b5563" if mo.app_meta().theme == "dark" else "#e5e7eb"
    _ui_card = mo.md(
        f"""
        <div style="padding: 25px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-top: 15px;">
            {_custom_layout}
        </div>
        """
    )

    mo.output.append(mo.vstack([
        mo.md("### 🧪 Лабораторія інтерактивного інференсу (What-If Simulator)"),
        _info_md,
        _ui_card
    ]))
    return (sim_panel,)


@app.cell
def simulator_engine(
    X_train,
    champion_selector,
    go,
    mo,
    np,
    optimal_threshold,
    pd,
    preprocessor_native,
    preprocessor_ohe,
    sim_panel,
    trained_models,
):
    _selected_name = champion_selector.value
    _native_names = {"LightGBM (Native)", "XGBoost (Native)", "Explainable Boosting (EBM)"}
    _is_native = _selected_name in _native_names

    _model = trained_models[_selected_name]

    # 1. Формуємо базовий рядок із СИРИХ даних (X_train)
    _base_row = X_train.median(numeric_only=True).to_dict()
    for _cat in X_train.select_dtypes(include=['object', 'category']).columns:
        _base_row[_cat] = X_train[_cat].mode()[0]

    # 2. Накладаємо зверху те, що "накрутив" користувач у UI
    for _feat, _val in sim_panel.value.items():
        _base_row[_feat] = _val

    # Перетворюємо в DataFrame з ідеальним порядком колонок
    _df_user_raw = pd.DataFrame([_base_row])[X_train.columns]

    # Елегантна трансформація через справжній пайплайн!
    if _is_native:
        _df_sim = preprocessor_native.transform(_df_user_raw)
        # Для LightGBM/XGBoost нативно відновлюємо категоріальний тип
        for _c in _df_sim.columns:
            if _df_sim[_c].dtype in ['object', 'string']:
                _df_sim[_c] = _df_sim[_c].astype('category')
    else:
        _df_sim = preprocessor_ohe.transform(_df_user_raw)

    # 3. Чесно забираємо ймовірність
    if hasattr(_model, "predict_proba"):
        _prob_rain = float(_model.predict_proba(_df_sim)[0][1])
    elif hasattr(_model, "decision_function"):
        _dfunc = _model.decision_function(_df_sim)[0]
        _prob_rain = float(1 / (1 + np.exp(-_dfunc))) # Застосовуємо сигмоїду
    else:
        _prob_rain = float(_model.predict(_df_sim)[0])

    _theme = mo.app_meta().theme

    _color_sun_bg = 'rgba(245, 158, 11, 0.15)'
    _color_rain_bg = 'rgba(59, 130, 246, 0.15)'
    _color_bar = '#2563eb'
    _color_threshold = "white" if _theme == 'dark' else "#1f2937"

    _fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta", value = _prob_rain * 100,
        title = {'text': "Ймовірність дощу (%)", 'font': {'size': 24}},
        # Дельта логічна: росте = більше дощу (синій), падає = більше сонця (жовтий)
        delta = {'reference': optimal_threshold * 100, 'increasing': {'color': "#3b82f6"}, 'decreasing': {'color': "#f59e0b"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': _color_bar},
            'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': [
                {'range': [0, optimal_threshold * 100], 'color': _color_sun_bg},
                {'range': [optimal_threshold * 100, 100], 'color': _color_rain_bg}
            ],
            'threshold': {'line': {'color': _color_threshold, 'width': 3}, 'thickness': 0.75, 'value': optimal_threshold * 100}
        }))

    _fig_gauge.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white' if _theme == 'dark' else 'black'})

    _is_rain = _prob_rain >= optimal_threshold
    if _is_rain:
        _verdict_html = f"""
        <div style="text-align: center; margin-top: -20px; margin-bottom: 25px;">
            <h2 style="color: #3b82f6; margin: 0; font-size: 2em;">🌧️ Прогноз: ДОЩ</h2>
            <p style="color: gray; margin-top: 5px; font-size: 1.1em;"><i>Ймовірність перевищує поріг чутливості ({optimal_threshold * 100:.1f}%). Рекомендовано взяти парасольку!</i></p>
        </div>
        """
    else:
        _verdict_html = f"""
        <div style="text-align: center; margin-top: -20px; margin-bottom: 25px;">
            <h2 style="color: #f59e0b; margin: 0; font-size: 2em;">☀️ Прогноз: БЕЗ ОПАДІВ</h2>
            <p style="color: gray; margin-top: 5px; font-size: 1.1em;"><i>Ймовірність нижча за поріг чутливості ({optimal_threshold * 100:.1f}%). Чудова погода!</i></p>
        </div>
        """

    _insight_md = mo.md(f"""
    > **💡 Tech Lead Insight (Анатомія Симулятора):**<br/>
    > Ми використовуємо справжній **ML Pipeline** для трансляції ваших значень у формат, зрозумілий моделі. Жодних "костилів" — симулятор генерує один рядок сирих даних і пропускає його через ті самі `StandardScaler` та `OneHotEncoder`, на яких тренувалася нейромережа.

    > ⚖️ **1. Чому різні алгоритми показують різні відсотки?**<br/>
    > Логістична регресія має плавну S-подібну криву і часто видає 60-70%. Ансамблі схильні до категоричних рішень і можуть одразу стрибати на 15% або 95%. Саме тому ми використовуємо динамічний `optimal_threshold`, який калібрує поріг чутливості під кожен алгоритм!

    > 🪨 **2. Чому повзунки іноді не впливають на спідометр?**<br/>
    > Це залежить від глобальної "ваги" ознаки (Feature Importance). Якщо ви потягнете повзунок малозначимої ознаки, алгоритм може її просто проігнорувати.
    """)

    _debug_panel = mo.accordion({
        "🛠️ Режим Архітектора (Дебаг)": mo.md(f"**Сирі значення з UI:** `{sim_panel.value}`<br>**Трансформований вектор (у Модель):** `{_df_sim.iloc[0].to_dict()}`<br>**Фінальна ймовірність:** `{_prob_rain:.6f}`")
    })

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
        mo.ui.plotly(_fig_gauge),
        mo.Html(_verdict_html),
        _insight_md,
        _debug_panel
    ]))
    return


@app.cell(hide_code=True)
def header_optuna(mo):
    mo.md("""
    <h3 align="center"><b>🧪 5. Байєсівська оптимізація <i>(Optuna + MLflow)</i></b></h3>
    """)
    return


@app.cell
def optuna_ui_controls(champion_selector, mo):
    _selected_name = champion_selector.value
    _is_tunable = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])
    trials_slider = mo.ui.slider(start=3, stop=50, step=1, value=5, show_value=True, label="🛝 **Кількість ітерацій (n_trials):**")
    run_optuna_btn = mo.ui.run_button(label=f"💝 Запустити тюнінг для {_selected_name}", kind="success", disabled=not _is_tunable)

    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    mo.output.append(mo.md(f"""
    <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; text-align: center;">
        <h3 style="margin-top: 0;">🧪 Optuna (Оптимізація F1-Score)</h3>
        <p><i>На відміну від сліпого випадкового пошуку <code>RandomizedSearchCV</code>, Optuna використовує алгоритм TPE (Tree-structured Parzen Estimator), який враховує історію попередніх кроків для швидшого знаходження ідеальних гіперпараметрів.</i></p>
        {trials_slider}<br/><br/>{run_optuna_btn}
    </div>
    """))
    return run_optuna_btn, trials_slider


@app.cell
def optuna_execution(
    GLOBAL_SEED,
    KFold,
    LGBMClassifier,
    RandomForestClassifier,
    XGBClassifier,
    X_train_native,
    X_train_ohe,
    champion_selector,
    f1_score,
    lgbm_kwargs,
    logger,
    logging,
    mlflow,
    mo,
    np,
    optuna,
    os,
    pd,
    plot_optimization_history,
    run_optuna_btn,
    style_dataframe,
    trials_slider,
    xgb_kwargs,
    y_train,
):
    _selected_name = champion_selector.value
    _is_tunable = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])
    final_tuned_model = None

    if run_optuna_btn.value and _is_tunable:
        # Глушимо MLflow, щоб він не панікував через відсутність pip у середовищі
        logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
        logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

        with mo.status.progress_bar(
            total=trials_slider.value,
            title=f"🦄 Тюнінг {_selected_name}",
            subtitle="⏳ Ініціалізація алгоритмів...",
            remove_on_exit=True
        ) as _bar:

            os.makedirs("mlruns", exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///mlruns/mlruns.db")
            mlflow.set_experiment("Rain_Optimization")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            _native_names = {"LightGBM (Native)", "XGBoost (Native)", "Explainable Boosting (EBM)"}
            _is_native = _selected_name in _native_names
            _X_train_curr = X_train_native if _is_native else X_train_ohe

            def _objective(trial):
                _bar.update(increment=0, subtitle=f"🏃‍♂️ Ітерація {trial.number + 1} з {trials_slider.value}: Навчання 3-х фолдів...")

                if "XGBoost" in _selected_name:
                    params = {"n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50), "max_depth": trial.suggest_int("max_depth", 3, 9), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), "enable_categorical": _is_native, "scale_pos_weight": 4, "eval_metric": 'logloss', "random_state": GLOBAL_SEED, **xgb_kwargs}
                    model_opt = XGBClassifier(**params)
                elif "LightGBM" in _selected_name:
                    params = {"n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50), "max_depth": trial.suggest_int("max_depth", 3, 10), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), "class_weight": 'balanced', "random_state": GLOBAL_SEED, **lgbm_kwargs}
                    model_opt = LGBMClassifier(**params)
                else:
                    params = {"n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50), "max_depth": trial.suggest_int("max_depth", 5, 15), "class_weight": 'balanced', "random_state": GLOBAL_SEED, "n_jobs": -1}
                    model_opt = RandomForestClassifier(**params)

                kf = KFold(n_splits=3, shuffle=True, random_state=GLOBAL_SEED)
                cv_scores = []

                for train_idx, val_idx in kf.split(_X_train_curr):
                    # PyArrow: .copy() гарантує монолітну пам'ять і прискорює алгоритми
                    X_tr = _X_train_curr.iloc[train_idx].copy()
                    X_val = _X_train_curr.iloc[val_idx].copy()
                    y_tr = y_train.iloc[train_idx].copy()
                    y_val = y_train.iloc[val_idx].copy()

                    model_opt.fit(X_tr, y_tr)
                    y_pred = model_opt.predict(X_val)
                    cv_scores.append(f1_score(y_val, y_pred))

                return np.mean(cv_scores)

            def _progress_callback(study, trial):
                _bar.update(
                    increment=1,
                    subtitle=f"🌿 Ітерація {trial.number + 1} з {trials_slider.value} | Найкращий F1: {study.best_value:.4f}"
                )

            _sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
            _study = optuna.create_study(direction="maximize", sampler=_sampler)

            # Вкидаємо наш callback в Optuna
            _study.optimize(_objective, n_trials=trials_slider.value, callbacks=[_progress_callback])

            _best_params = _study.best_params
            _best_params.update({"random_state": GLOBAL_SEED})
            logger.info(f"Optuna знайшла найкращі параметри: {_best_params}")

            if "XGBoost" in _selected_name:
                _best_params.update({"enable_categorical": _is_native, "scale_pos_weight": 4, "eval_metric": 'logloss', **xgb_kwargs})
                final_tuned_model = XGBClassifier(**_best_params)
            elif "LightGBM" in _selected_name:
                _best_params.update({"class_weight": 'balanced', **lgbm_kwargs})
                final_tuned_model = LGBMClassifier(**_best_params)
            else:
                _best_params.update({"class_weight": 'balanced'})
                final_tuned_model = RandomForestClassifier(**_best_params)

            # Перемикаємо статус перед фінальним довгим тренуванням
            _bar.update(increment=0, subtitle="💾 Збереження найкращої моделі у базу...")
            final_tuned_model.fit(_X_train_curr, y_train)
            mo.output.clear()

            _safe_run_name = f"Optuna_{_selected_name.replace(' ', '_').replace('(', '').replace(')', '')}"

            with mlflow.start_run(run_name=_safe_run_name):
                mlflow.log_params(_best_params)
                mlflow.log_metric("CV_F1_Score", _study.best_value)
                mlflow.log_metric("Optuna_Trials", trials_slider.value)

                _trusted_types = [
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                    "lightgbm.sklearn.LGBMClassifier",
                    "lightgbm.basic.Booster",
                    "collections.OrderedDict"
                ]

                _pip_reqs = ["scikit-learn", "imbalanced-learn"]
                if "XGBoost" in _selected_name: _pip_reqs.append("xgboost")
                if "LightGBM" in _selected_name: _pip_reqs.append("lightgbm")
                if "EBM" in _selected_name or "Explainable" in _selected_name: _pip_reqs.append("interpret")

                mlflow.sklearn.log_model(
                    final_tuned_model,
                    artifact_path="champion_model",
                    skops_trusted_types=_trusted_types,
                    pip_requirements=_pip_reqs
                )

                _run_id = mlflow.active_run().info.run_id

            # ==========================================
            # 🎨 БЛОК ВІЗУАЛІЗАЦІЇ ТА UX (МАГІЯ ТУТ)
            # ==========================================
            _fig_history = plot_optimization_history(_study)
            try:
                _fig_params = optuna.visualization.plot_param_importances(_study)
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
                yaxis_title="Значення F1-Міри"
            )
            for _trace in _fig_history.data:
                # Зсуваємо координати осі X на +1
                if _trace.x is not None:
                    _trace.x = tuple(x + 1 for x in _trace.x)

                if _trace.name == 'Objective Value':
                    _trace.name = 'Результат поточної ітерації'
                    _trace.hovertemplate = '<b>Ітерація:</b> %{x}<br><b>F1-Score:</b> %{y:.4f}<extra></extra>'
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
                    ✅ **Оптимізацію завершено!** Найкращий F1-Score: `{_study.best_value:.4f}`<br/>
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
    X_train_native,
    X_train_ohe,
    champion_selector,
    go,
    mo,
    np,
    pd,
    permutation_importance,
    trained_models,
    y_train,
):
    _selected_name = champion_selector.value
    _model = trained_models.get(_selected_name)

    # Локальний реєстр нативних моделей
    _native_names = {"LightGBM (Native)", "XGBoost (Native)", "Explainable Boosting (EBM)"}
    _is_native = _selected_name in _native_names
    _X_data = X_train_native if _is_native else X_train_ohe

    _feature_names = _X_data.columns.tolist()
    _importances = None
    _imp_type = "Невідомо"

    # =========================================================================
    # 🔍 1. РОЗПІЗНАВАННЯ АРХІТЕКТУРИ ТА ВИТЯГУВАННЯ ВАГ
    # =========================================================================
    if hasattr(_model, "feature_importances_"):
        _importances = _model.feature_importances_
        _imp_type = "Gini / Entropy Importance (Дерева/Бустинг)"
    elif hasattr(_model, "coef_"):
        # Для лінійних моделей беремо абсолютні значення коефіцієнтів
        _importances = np.abs(_model.coef_[0])
        _imp_type = "Absolute Weights (Лінійні моделі)"
    elif "Explainable Boosting" in _selected_name and hasattr(_model, "term_importances"):
        _importances = _model.term_importances()
        # EBM може мати додаткові терміни взаємодії, тому беремо їхні імена
        if hasattr(_model, "term_names_"):
            _feature_names = _model.term_names_
        _imp_type = "Global Term Importances (EBM)"

    # Якщо модель (HistGradient, KNN тощо) "лінива" і не має вбудованих ваг
    if _importances is None:
        with mo.status.spinner(f"Обчислення Permutation Importance для {_selected_name}..."):
            # Беремо невелику випадкову вибірку, щоб графік побудувався миттєво
            _sample_size = min(2000, len(_X_data))
            _X_sample = _X_data.sample(n=_sample_size, random_state=42)
            _y_sample = y_train.loc[_X_sample.index]

            # Рахуємо агностичну важливість на льоту!
            _perm_result = permutation_importance(_model, _X_sample, _y_sample, n_repeats=5, random_state=42, n_jobs=-1)
            _importances = _perm_result.importances_mean
            _imp_type = "Permutation Importance (Модельно-агностичний метод)"

    # =========================================================================
    # 📝 2. ФОРМУВАННЯ ДАНИХ ТА РОЗУМНИЙ ПЕРЕКЛАД OHE
    # =========================================================================
    # Відкидаємо можливі розбіжності у довжині масивів (захист для специфічних ансамблів)
    _min_len = min(len(_feature_names), len(_importances))
    _df_imp = pd.DataFrame({"Ознака": _feature_names[:_min_len], "Важливість": _importances[:_min_len]})

    def translate_feature(feat):
        # 1. Прямий збіг (нативні колонки)
        if feat in UA_COLUMNS:
            return UA_COLUMNS[feat]

        # 2. Переклад складених OHE ознак (наприклад, Location_Sydney)
        parts = str(feat).split('_', 1)
        if len(parts) == 2 and parts[0] in UA_COLUMNS:
            return f"{UA_COLUMNS[parts[0]]} ({parts[1]})"

        return feat

    _df_imp["Укр_Ознака"] = _df_imp["Ознака"].apply(translate_feature)

    # Беремо Топ-15 і сортуємо за ЗРОСТАННЯМ (найважливіша фіча буде нагорі у горизонтальному графіку)
    _df_imp = _df_imp.sort_values(by="Важливість", ascending=True).tail(15)

    # =========================================================================
    # 📊 3. ВІЗУАЛІЗАЦІЯ PLOTLY
    # =========================================================================
    _theme = mo.app_meta().theme
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"
    _text_color = "white" if _theme == "dark" else "#1f2937"

    _fig_imp = go.Figure(go.Bar(
        x=_df_imp["Важливість"],
        y=_df_imp["Укр_Ознака"],
        orientation='h',
        marker=dict(
            color=_df_imp["Важливість"],
            colorscale='Tealgrn' if _theme == 'light' else 'Viridis',
            showscale=False
        ),
        hovertemplate="<b>Ознака:</b> %{y}<br><b>Відносна важливість:</b> %{x:.5f}<extra></extra>"
    ))

    _fig_imp.update_layout(
        title=dict(text=f"<b>Топ-15 найважливіших ознак ({_imp_type})</b>", x=0.5, xanchor="center"),
        xaxis_title="Вплив на прогноз моделі (Absolute Impact)",
        yaxis_title=None,
        template=_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=40),
        height=550,
        font=dict(color=_text_color)
    )

    # =========================================================================
    # 🧠 4. ДИНАМІЧНІ БІЗНЕС-ІНСАЙТИ
    # =========================================================================
    # Безпечне витягування імен лідерів (якщо ознак менше ніж 2)
    _top_1_feat = _df_imp.iloc[-1]["Укр_Ознака"] if len(_df_imp) > 0 else "Невідомо"
    _top_2_feat = _df_imp.iloc[-2]["Укр_Ознака"] if len(_df_imp) > 1 else "Невідомо"

    _insight_md = mo.md(f"""
    > **💡 Tech Lead Insight (Інтерпретація Моделі):**<br/>
    > Ми щойно зазирнули всередину "чорної скриньки" алгоритму `{_selected_name}`. Цей графік ілюструє глобальну стратегію прийняття рішень моделлю — на які саме датчики вона звертає найбільшу увагу перед тим, як видати прогноз.
    >
    > - 🥇 **Головний предиктор:** Ознака **«{_top_1_feat}»** має найвищу питому вагу. Математично модель вважає її найбільш критичною метрикою.
    > - 🥈 **Другорядний фактор:** Ознака **«{_top_2_feat}»** також відіграє вагому роль, корегуючи або підтверджуючи логіку першої.
    > - 🗑️ **Вектор оптимізації (Feature Selection):** Ознаки, які знаходяться внизу списку (або взагалі не потрапили у цей Топ-15), додають мінімальну інформаційну цінність. У майбутніх ітераціях ми можемо сміливо видалити їх з пайплайну для прискорення тренування без втрати фінальної якості (F1-Score).
    """)

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
        mo.center(mo.md(f"### 🧬 Глобальна пояснюваність (Feature Importance): {_selected_name}")),
        mo.md("<div style='height: 10px;'></div>"),
        mo.ui.plotly(_fig_imp),
        _insight_md
    ]))
    return


@app.cell(hide_code=True)
def header_shap(mo):
    mo.md("""
    <h3 align="center"><b>🕵️‍♂️ 6. Квантова пояснюваність <i>(SHAP Values)</i></b></h3>
    """)
    return


@app.cell
def shap_ui(
    X_train_native,
    X_train_ohe,
    champion_selector,
    final_tuned_model,
    mo,
):
    _selected_name = champion_selector.value
    _is_tree = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])

    # Створюємо локальний реєстр (без залежності від глобальних змінних)
    _native_names = {"LightGBM (Native)", "XGBoost (Native)", "Explainable Boosting (EBM)"}
    _is_native = _selected_name in _native_names

    # Інтелектуально перевіряємо, чи підходить тюнінгована модель під поточні дані
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
            <p>Цей алгоритм заглядає всередину "чорної скриньки" і розраховує математичний внесок кожної ознаки для <b>кожного окремого дня</b>.<br/>
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
    UA_COLUMNS,
    X_train_native,
    X_train_ohe,
    champion_selector,
    final_tuned_model,
    mo,
    plt,
    shap,
    shap_btn,
    trained_models,
    warnings,
):
    if shap_btn.value:
        _selected_name = champion_selector.value

        _native_names = {"LightGBM (Native)", "XGBoost (Native)", "Explainable Boosting (EBM)"}
        _is_native = _selected_name in _native_names
        _X_train_curr = X_train_native if _is_native else X_train_ohe

        # Визначаємо модель безпечно
        _model_to_explain = trained_models[_selected_name]
        if final_tuned_model is not None:
            _expected_features = _X_train_curr.shape[1]
            _tuned_features = getattr(final_tuned_model, "n_features_in_", None)
            if _tuned_features == _expected_features:
                _model_to_explain = final_tuned_model

        with mo.status.spinner(title="🍻 Аналіз рішень моделі...", subtitle="Розрахунок векторів Шеплі (SHAP values)"):
            _X_sample = _X_train_curr.sample(n=500, random_state=42)
            _explainer = shap.Explainer(_model_to_explain)
            _shap_exp = _explainer(_X_sample)

            # Українізація "на льоту" всередині об'єкта Explanation
            _shap_exp.feature_names = [UA_COLUMNS.get(c, c) for c in _X_sample.columns]

            # Броня для 3D-масивів (мультиклас/ансамблі)
            if len(_shap_exp.shape) == 3:
                _shap_exp = _shap_exp[:, :, 1]

            _theme = mo.app_meta().theme
            _style = 'dark_background' if _theme == 'dark' else 'default'

            with plt.style.context(_style):
                plt.rcParams['savefig.transparent'] = True
                plt.close('all')

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    shap.plots.beeswarm(_shap_exp, show=False)

                _fig = plt.gcf()
                _ax = plt.gca()

                _fig.set_size_inches(10, 6)
                _text_color = 'white' if _theme == 'dark' else '#1f2937'

                _ax.set_title(
                    f"Квантова пояснюваність ({_selected_name}): Глобальний вплив ознак",
                    color=_text_color, fontsize=15, fontweight='bold', pad=20
                )

                _ax.set_xlabel("Значення SHAP (Вплив на Логарифм Шансів Дощу / Log-Odds)", color=_text_color)

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
                    _cbar_ax.set_ylabel("Фактичне значення датчика", rotation=270, labelpad=15)
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

            _native_insight = ""
            if _is_native:
                _native_insight = """
                > 👽 **Що означають СІРІ точки?**<br/>
                > Ви обрали модель з нативною підтримкою категорій. Сірі точки з'являються на текстових ознаках (напр., напрямок вітру "NNW"). Для таких ознак поняття "Високе/Низьке" не має математичного сенсу (Північ не більша за Південь), тому SHAP фарбує їх у сірий колір.
                """

            _insight_md = mo.md(
                f"""
                <center>⚠️ <b>Увага:</b> <i>Дерева рішень у класифікації передбачають не відсотки (0-100%), а математичний <b>Логарифм Шансів (Log-Odds)</b>. Тому вплив на осі X є нелінійним.</i></center>

                > **💡 Tech Lead Insight (Як читати класифікаційний SHAP):**<br/>
                > На відміну від звичайного "Рентгену" (Feature Importance), SHAP розраховує маргінальний внесок кожної фічі на основі векторів Шеплі з теорії кооперативних ігор. Графік показує **напрямок** та **щільність** впливу.
                >
                > - **Колір точки (🔴 Червоний/Синій 🔵):** Фізичне значення погодного датчика (Червоний = високий тиск, Синій = низький).
                > - **Позиція на осі X (⏮️ Вліво/Вправо ⏭️):** Зсув вправо від нуля тягне прогноз до "Дощ" (клас 1), зсув вліво тягне прогноз до "Без опадів" (клас 0).

                {_native_insight}
                """
            )

            mo.output.append(mo.vstack([_css_no_scroll, mo.center(_plot_html), _insight_md]))
    return


@app.cell(hide_code=True)
def header_geomap(mo):
    mo.md("""
    <h3 align="center"><b>🗺️ 7. Геопросторовий аналіз <i>(Кліматична Мапа Австралії)</i></b></h3>
    """)
    return


@app.cell(hide_code=True)
def geo_map_execution(df_raw, go, mo, pd):
    # Координати основних метеостанцій Австралії
    _geo_dict = {
        'Sydney': (-33.8688, 151.2093), 'Melbourne': (-37.8136, 144.9631),
        'Brisbane': (-27.4698, 153.0251), 'Perth': (-31.9505, 115.8605),
        'Adelaide': (-34.9285, 138.6007), 'Hobart': (-42.8821, 147.3272),
        'Darwin': (-12.4634, 130.8456), 'AliceSprings': (-23.6980, 133.8807),
        'Cairns': (-16.9186, 145.7781), 'Townsville': (-19.2590, 146.8169)
    }

    _map_data = []
    # Розраховуємо % дощових днів для кожного міста
    if 'Location' in df_raw.columns and 'RainTomorrow' in df_raw.columns:
        _grouped = df_raw.groupby('Location')['RainTomorrow'].apply(lambda x: (x == 'Yes').mean() * 100).to_dict()

        for loc, coords in _geo_dict.items():
            _rain_prob = _grouped.get(loc, 0)
            if _rain_prob > 0:
                _map_data.append({"Location": loc, "Lat": coords[0], "Lon": coords[1], "RainRisk": _rain_prob})

    _df_map = pd.DataFrame(_map_data)

    _df_map['HoverText'] = (
        '<b>🏠 ' + _df_map['Location'] + '</b><br>' +
        '🌏 Координати: ' + _df_map['Lat'].astype(str) + '°, ' + _df_map['Lon'].astype(str) + '°<br>' +
        '🌧️ Історична частота дощів: <b>' + _df_map['RainRisk'].round(1).astype(str) + '%</b>'
    )

    _theme = mo.app_meta().theme
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"
    _text_color = "white" if _theme == "dark" else "#1f2937"

    # Запобіжник для математики маркерів
    _max_risk = _df_map['RainRisk'].max() if not _df_map.empty else 100

    _fig_geo = go.Figure(data=go.Scattergeo(
        lon=_df_map['Lon'], lat=_df_map['Lat'],
        text=_df_map['HoverText'],
        mode='markers',
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            size=_df_map['RainRisk'],
            sizemode='area',
            sizeref=2. * _max_risk / (45.**2),
            sizemin=8,
            color=_df_map['RainRisk'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Частота дощів (%)", thickness=15, len=0.7),
            line_color='white',
            line_width=1.5
        )
    ))

    _fig_geo.update_layout(
        title=dict(text="<b>Кліматична Мапа Австралії: Історична частота опадів по містах</b>", x=0.5, font=dict(size=18)),
        geo=dict(
            scope='world',
            projection_type='mercator',
            lataxis_range=[-50, -10],
            lonaxis_range=[100, 165],
            showland=True,
            landcolor="#334155" if _theme == "dark" else "#e2e8f0",
            showocean=True,
            oceancolor="rgba(0,0,0,0)",
            bgcolor="rgba(0,0,0,0)",
            framecolor="#4b5563" if _theme == "dark" else "#e5e7eb"
        ),
        template=_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=20, l=0, r=0),
        autosize=True,
        font=dict(color=_text_color)
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

    _insight = mo.md(
        """
        > 💡 **Tech Lead Insight:**<br/>
        > Географія вирішує все. Тропічний Дарвін (Північ) та прибережні регіони (Сідней, Мельбурн) мають найвищі базові ризики опадів, тоді як центр материка (Alice Springs) залишається екстремально посушливим. Наш пайплайн автоматично вивчає ці географічні патерни, надаючи кожному місту унікальну вагу при класифікації.
        """
    )

    _map_ui = mo.hstack([_fig_geo], justify="center")

    mo.output.append(mo.vstack([_css_no_scroll, _map_ui, _insight]))
    return


@app.cell(hide_code=True)
def header_mlops_serialization(mo):
    mo.md("""
    <h2 align="center"><b>⛲️ 8. Продакшн: MLOps Серіалізація та Мікросервіс <i>(FastAPI)</i></b></h2>
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
def mlopps_serialization(
    X_train_native,
    X_train_ohe,
    datetime,
    final_tuned_model,
    joblib,
    json,
    logger,
    mlops_export_selector,
    mlops_generate_btn,
    mo,
    np,
    optimal_threshold,
    os,
    textwrap,
    trained_models,
):
    if mlops_generate_btn.value:
        _selection = mlops_export_selector.value
        _selected_name = _selection["name"]
        _is_tuned = _selection["is_tuned"]

        _native_names = {"LightGBM (Native)", "XGBoost (Native)", "Explainable Boosting (EBM)"}
        _is_native = _selected_name in _native_names
        _X_curr = X_train_native if _is_native else X_train_ohe

        _model_to_save = final_tuned_model if _is_tuned else trained_models[_selected_name]

        _project_name = "rain_in_australia"
        _artifact_dir = os.path.join(os.getenv("MODELS_DIR", "./models"), _project_name)
        os.makedirs(_artifact_dir, exist_ok=True)

        _safe_name = _selected_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        if _is_tuned:
            _safe_name += "_tuned"

        _model_path = os.path.join(_artifact_dir, f"{_safe_name}_champion.joblib")
        _schema_path = os.path.join(_artifact_dir, "features_schema.json")
        _api_path = os.path.join(_artifact_dir, "api.py")
        _docker_path = os.path.join(_artifact_dir, "Dockerfile")

        joblib.dump(_model_to_save, _model_path)

        _dtypes_dict = {}
        _categories_dict = {}
        for _col in _X_curr.columns:
            _dt = _X_curr[_col].dtype
            _dtypes_dict[_col] = str(_dt)
            if hasattr(_dt, 'categories'):
                _categories_dict[_col] = list(_dt.categories)

        _features_schema = {
            "project_name": _project_name,
            "exported_at": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "model_architecture": _selected_name,
            "is_optuna_tuned": _is_tuned,
            "expected_columns": list(_X_curr.columns),
            "dtypes": _dtypes_dict,
            "categories": _categories_dict,
            "optimal_threshold": float(optimal_threshold)
        }
        with open(_schema_path, "w", encoding="utf-8") as f:
            json.dump(_features_schema, f, indent=4, ensure_ascii=False)

        _sample_row = _X_curr.iloc[0].to_dict()
        _sample_clean = {
            k: (float(v) if isinstance(v, (float, np.floating))
                else int(v) if isinstance(v, (int, np.integer))
                else str(v))
            for k, v in _sample_row.items()
        }
        _sample_json_str = json.dumps({"features": _sample_clean}, ensure_ascii=False)

        # =====================================================================
        # 🏗️ АРХІТЕКТУРНИЙ ШЕДЕВР: Генерація коду через textwrap.dedent
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
                title="🌦️ Rain in Australia AI API",
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
                    <title>Rain Classification API</title>
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
                model = joblib.load(MODEL_PATH)
                with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                    expected_dtypes = schema.get("dtypes", {{}})
                    expected_categories = schema.get("categories", {{}})
                    expected_columns = schema.get("expected_columns", [])
                    optimal_threshold = schema.get("optimal_threshold", 0.5)
            except Exception as e:
                model, expected_columns, expected_dtypes, expected_categories, optimal_threshold = None, [], {{}}, {{}}, 0.5

            class InferencePayload(BaseModel):
                features: Dict[str, Any]
                model_config = {{"json_schema_extra": {{"examples": [{_sample_json_str}]}}}}

            class RainPredictionResponse(BaseModel):
                will_it_rain_tomorrow: bool
                rain_probability_percent: float
                model_deployed: str
                optimal_decision_threshold: float

            @app.post("/predict", response_model=RainPredictionResponse)
            def predict_rain(payload: InferencePayload):
                if model is None:
                    raise HTTPException(status_code=500, detail="Модель не завантажена")

                try:
                    df = pd.DataFrame([payload.features])

                    if expected_columns:
                        df = df.reindex(columns=expected_columns)

                    for col, dtype in expected_dtypes.items():
                        if col in df.columns:
                            if dtype == "category" and col in expected_categories:
                                cat_type = pd.CategoricalDtype(categories=expected_categories[col])
                                df[col] = df[col].astype(cat_type)
                            else:
                                df[col] = df[col].astype(dtype)

                    probabilities = model.predict_proba(df)[0]
                    prob_rain = float(probabilities[1])

                    return {{
                        "will_it_rain_tomorrow": bool(prob_rain >= optimal_threshold),
                        "rain_probability_percent": round(prob_rain * 100, 2),
                        "model_deployed": "{_selected_name}",
                        "optimal_decision_threshold": optimal_threshold
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
        _docker_libs = "fastapi uvicorn pydantic joblib pandas scikit-learn imbalanced-learn"
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
        _num_features = _X_curr.shape[1]

        # Безпечно витягуємо фінальний класифікатор (якщо це пайплайн)
        _actual_model = getattr(_model_to_save, "named_steps", {}).get(
            "classifier", getattr(_model_to_save, "named_steps", {}).get("model", _model_to_save)
        )
        _params = getattr(_actual_model, "get_params", lambda: {})()

        _arch_lines = []

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

        if not _arch_lines:
            _arch_lines.append("⚖️ Тип: Аналітична / Лінійна архітектура (без дерев)")

        _exclude_keys = {
            'n_estimators', 'max_depth', 'learning_rate', 'booster',
            'random_state', 'n_jobs', 'objective', 'enable_categorical',
            'missing', 'callbacks', 'verbosity', 'silent', 'early_stopping_rounds',
            'device', 'n_iter_no_change', 'verbose', 'class_weight'
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

        _arch_text = "\n                  ".join(_arch_lines)

        _dataset_type = "Нативна категоріалізація" if _is_native else "One-Hot Encoding"
        _origin_status = "🔱 Оптимізована (Optuna)" if _is_tuned else "🧧 Базова (З Лідерборду)"

        # Зчитуємо файли для кнопок завантаження
        with open(_model_path, "rb") as f: _model_bytes = f.read()
        with open(_schema_path, "rb") as f: _schema_bytes = f.read()
        with open(_api_path, "rb") as f: _api_bytes = f.read()
        with open(_docker_path, "rb") as f: _docker_bytes = f.read()

        _download_model_btn = mo.download(data=_model_bytes, filename=os.path.basename(_model_path), label="☯️ .joblib (Ваги)")
        _download_schema_btn = mo.download(data=_schema_bytes, filename="features_schema.json", label="✡️ .json (Схема)")
        _download_api_btn = mo.download(data=_api_bytes, filename="api.py", label="⚛️ api.py (FastAPI)")
        _download_docker_btn = mo.download(data=_docker_bytes, filename="Dockerfile", label="🐳 Dockerfile")

        _theme = mo.app_meta().theme
        _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
        _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

        _version_tag = " (Optuna Tuned 🔱)" if _is_tuned else " (Базова версія)"
        logger.info(f"Експорт MLOps артефактів завершено для {_selected_name}{_version_tag}")

        # Рендеринг красивого UI з логами
        mo.output.append(mo.md(f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-top: 20px;">
            <h3 style="margin-top: 0; color: #3b82f6; text-align: center;">📦 MLOps Серіалізація (Production Ready)</h3>
            <p style="text-align: center;">Останній крок перед передачею алгоритму Backend-команді. Ми фізично зберігаємо "мозок" моделі, маніфест ознак та автоматично генеруємо Docker-капсулу!</p>

            ```text
            🎭 Початок збереження у файл...
              📦 Шлях проекту: {_artifact_dir}/
              🎛 Ознаки: {_num_features} вимірів ({_dataset_type})
              🎯 Decision Threshold (Оптимальний поріг): {optimal_threshold}
              👾 Архітектура моделі:
                 🧠 Алгоритм: {_selected_name}
                 🛠 Джерело: {_origin_status}
                 {_arch_text}

              ✅ Успіх! Капсула 'Мозку' ШІ надійно збережена ({_model_size_kb:,.2f} KB) о {_timestamp_human}
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
        <p style="font-size: 15px;">Цей модуль автоматично генерує повністю налаштований бекенд для нашої ML-моделі класифікації погоди. Ми перейшли від базових Data Science скриптів до бездоганної інфраструктури Enterprise-рівня.</p>

        <h3 style="color: #3b82f6; margin-top: 25px;">✨ Технологічний Стек та Можливості</h3>
        <ul style="font-size: 14px;">
            <li style="margin-bottom: 8px;"><b>FastAPI + Uvicorn:</b> Високопродуктивний асинхронний сервер, який миттєво обробляє HTTP-запити та виконує інференс (передбачення ймовірності).</li>
            <li style="margin-bottom: 8px;"><b>Docker Контейнеризація 🐳:</b> Автоматична генерація <code>Dockerfile</code> з точними залежностями алгоритму (включно з <i>XGBoost/LightGBM</i>). Це забезпечує 100% ізоляцію середовища — модель гарантовано працюватиме однаково на вашому ноутбуці, сервері AWS чи Google Cloud. Ніяких проблем <i>"а на моєму комп'ютері працювало"</i>!</li>
            <li style="margin-bottom: 8px;"><b>Сучасний Scalar UI:</b> Ми повністю відмовилися від застарілого Swagger. Інтегрований <b>Scalar</b> забезпечує преміальний дизайн документації (рівня Stripe), вбудований REST-клієнт та миттєву генерацію коду запитів для десятків мов програмування.</li>
            <li style="margin-bottom: 8px;"><b>"Бронежилет" для даних:</b> Сервер самостійно вирівнює порядок колонок під суворі стандарти алгоритму та конвертує типи даних, захищаючи процес від людського фактору.</li>
        </ul>

        <h3 style="color: #10b981; margin-top: 25px;">🛡️ Суворі Pydantic-Контракти</h3>
        <ul style="font-size: 14px;">
            <li style="margin-bottom: 8px;">✅ <b>200 OK (Успішна відповідь):</b> Сервер повертає чітко типізовану схему <code>RainPredictionResponse</code>. Завдяки цьому клієнт наперед знає, що гарантовано отримає <code>will_it_rain_tomorrow</code> (bool) та <code>rain_probability_percent</code> (float), а також метадані про поточну активну модель та застосований <code>optimal_decision_threshold</code>.</li>
            <li style="margin-bottom: 8px;">❌ <b>422 Validation Error:</b> Завдяки <code>InferencePayload</code>, якщо клієнт надішле неправильний тип даних або пропустить параметр, FastAPI автоматично відхилить запит із детальним JSON-описом (де саме сталася помилка), захищаючи ML-модель від падінь.</li>
        </ul>

        <h3 style="color: #f59e0b; margin-top: 25px;">⚙️ Як запустити сервер?</h3>
        <p style="font-size: 14px;">Усі згенеровані артефакти надійно ізольовано у директорії <code>models/rain_in_australia/</code>.</p>

        <div style="margin-top: 15px;">
            <b>▶ Спосіб 1: DevOps-стандарт (Через Makefile)</b>
            <pre style="background-color: {_pre_bg}; color: {_pre_text_cmd}; padding: 12px; border-radius: 8px; border: 1px solid {_pre_border}; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>make api-hw2</code></pre>
        </div>

        <div style="margin-top: 15px;">
            <b>▶ Спосіб 2: Запуск у Docker (Cloud Ready ☁️)</b>
            <pre style="background-color: {_pre_bg}; color: {_pre_text_code}; padding: 12px; border-radius: 8px; border: 1px solid {_pre_border}; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>cd models/rain_in_australia
    docker build -t rain-api .
    docker run -p 8000:8000 rain-api</code></pre>
        </div>

        <div style="margin-top: 15px;">
            <b>▶ Спосіб 3: Ручний запуск (Без Docker)</b>
            <pre style="background-color: {_pre_bg}; color: {_pre_text_code}; padding: 12px; border-radius: 8px; border: 1px solid {_pre_border}; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>cd models/rain_in_australia
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload</code></pre>
        </div>

        <hr style="border-color: {_border}; margin: 25px 0;">
        <p style="margin-bottom: 0; font-size: 15px;">
            <i>💡 <b>Документація доступна за адресою:</b> <a href="http://127.0.0.1:8000/docs" target="_blank" style="color: #3b82f6; font-weight: bold; text-decoration: none;">http://127.0.0.1:8000/docs</a>.<br/>
            <i>🥂 Тепер будь-який веб-сайт, Telegram-бот чи мобільний застосунок на Swift/Kotlin може відправляти JSON-запити на цей порт і миттєво отримувати прогноз імовірності дощу!</i>
        </p>
    </div>
    """)

    mo.output.append(mo.hstack([_css_no_scroll, _deploy_instructions]))
    return


@app.cell(hide_code=True)
def header_timesfm(mo):
    mo.md("""
    <h3 align="center"><b>🔮 9. Zero-Shot прогнозування за допомогою Google TimesFM</b></h3>
    """)
    return


@app.cell
def timesfm_ui_controls(UA_COLUMNS, df_raw, mo):
    import timesfm
    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    # Динамічно збираємо всі доступні міста
    _locations = sorted(df_raw['Location'].dropna().unique().tolist())

    # Системні назви метрик
    _features = ['MaxTemp', 'MinTemp', 'Humidity3pm', 'Pressure3pm', 'WindGustSpeed']
    _feat_options = {UA_COLUMNS.get(f, f): f for f in _features}

    timesfm_loc_selector = mo.ui.dropdown(
        options=_locations,
        value='Sydney',
        label="**🏙️ Місто:** "
    )

    _default_feat_label = UA_COLUMNS.get('MaxTemp', 'MaxTemp')

    timesfm_feat_selector = mo.ui.dropdown(
        options=_feat_options,
        value=_default_feat_label,
        label="**📈 Метрика:** "
    )

    timesfm_btn = mo.ui.run_button(
        label="🚀 Згенерувати Zero-Shot Прогноз",
        kind="success"
    )

    # Пряма інтерполяція змінних без .batch()
    _ui_panel = mo.vstack([
        mo.md(
            f"""
            <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
                <p>Фундаментальна модель <b>Google TimesFM</b> (<i>Рушій v2.0, Ваги: 1.0-200m</i>) здатна робити високоточні прогнози часових рядів <b>без жодного донавчання</b>! Оберіть місто та метрику для створення прогнозу на наступні 30 днів.</p>
                <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-top: 15px;">
                    {timesfm_loc_selector} {timesfm_feat_selector}
                </div>
                <div style="margin-top: 20px;">
                    {timesfm_btn}
                </div>
            </div>
            """
        )
    ])

    mo.output.append(_ui_panel)
    return timesfm, timesfm_btn, timesfm_feat_selector, timesfm_loc_selector


@app.cell
def execute_timesfm(
    UA_COLUMNS,
    clear_vram,
    device,
    device_ui_name,
    df_raw,
    go,
    logger,
    mo,
    np,
    pd,
    timesfm,
    timesfm_btn,
    timesfm_feat_selector,
    timesfm_loc_selector,
):
    mo.stop(
        not timesfm_btn.value,
        mo.center(mo.md("⏳ **Очікування:** Оберіть місто та метрику, а потім натисніть кнопку 'Згенерувати Zero-Shot Прогноз' 🚀"))
    )

    _loc = timesfm_loc_selector.value
    _feat = timesfm_feat_selector.value
    _ua_feat = UA_COLUMNS.get(_feat, _feat)

    _theme = mo.app_meta().theme
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"

    # Підготовка часового ряду під обране місто та метрику
    _df_city = df_raw[(df_raw['Location'] == _loc) & df_raw[_feat].notna()].copy()
    if 'Date' in _df_city.columns:
        _df_city['Date'] = pd.to_datetime(_df_city['Date'])
        _df_city = _df_city.sort_values('Date')

    _valid_data = _df_city[['Date', _feat]].dropna()
    _context_dates = _valid_data['Date'].values
    _context_values = _valid_data[_feat].values

    # Захист від нестачі даних
    mo.stop(
        len(_context_values) < 256,
        mo.md(f"❌ **Помилка:** Недостатньо даних для міста {_loc} по метриці <b>{_ua_feat}</b>. TimesFM потребує щонайменше 256 днів історичного контексту.")
    )

    _pred_values = []
    _is_mock = False
    _error_msg = ""

    # Інференс TimesFM (Бронебійний блок)
    try:
        with mo.status.spinner(f"Завантаження ваг TimesFM 1.0 та інференс (Бекенд: `{device_ui_name}`)..."):

            _TfmClass = getattr(timesfm, 'TimesFm', getattr(timesfm, 'TimesFM', None))

            if _TfmClass is None:
                raise AttributeError("Клас TimesFm відсутній у модулі (можливий конфлікт бекенду на Intel Mac)")

            _tfm = _TfmClass(
                context_len=256, horizon_len=30, input_patch_len=32, output_patch_len=128, num_layers=20, model_dims=1280, backend=device.type
            )
            _tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

            _forecast_values, _ = _tfm.forecast([_context_values[-256:]])
            _pred_values = _forecast_values[0][:, 1]

            clear_vram(device)
            logger.info(f"Інференс TimesFM ({_feat} для {_loc}) успішно завершено.")

    except Exception as e:
        _is_mock = True
        _error_msg = str(e)
        logger.warning(f"⚠️ Перехід на Fallback (Симуляцію) через помилку рушія TimesFM: {_error_msg}")

        np.random.seed(42)
        _trend = (_context_values[-1] - _context_values[-30]) / 30
        _noise = np.random.normal(0, np.std(_context_values[-30:]) * 0.4, 30)
        _pred_values = _context_values[-1] + np.arange(1, 31) * _trend + np.cumsum(_noise)

    # Підготовка дат для графіка
    _last_date = pd.to_datetime(_context_dates[-1])
    _future_dates = pd.date_range(start=_last_date + pd.Timedelta(days=1), periods=30)

    # Побудова графіка
    _fig_ts = go.Figure()

    _fig_ts.add_trace(go.Scatter(
        x=_context_dates[-100:], y=_context_values[-100:],
        mode='lines', name=f'Історія ({_loc})',
        line=dict(color='#3b82f6', width=2)
    ))

    _fig_ts.add_trace(go.Scatter(
        x=_future_dates, y=_pred_values,
        mode='lines', name='Zero-Shot Прогноз TimesFM',
        line=dict(color='#ef4444', width=2.5, dash='dash')
    ))

    _fig_ts.add_trace(go.Scatter(
        x=[_context_dates[-1], _future_dates[0]], y=[_context_values[-1], _pred_values[0]],
        mode='lines', showlegend=False,
        line=dict(color='#ef4444', width=2.5, dash='dash')
    ))

    _fig_ts.update_layout(
        title=dict(text=f"<b>TimesFM Foundation Model | Hardware Backend: {device_ui_name}</b><br><sup>Прогноз метрики: {_ua_feat}</sup>", x=0.5),
        xaxis_title="",
        yaxis_title=_ua_feat,
        template=_template,
        margin=dict(t=70, b=80, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )

    _fig_ts.update_xaxes(
        tickformat="%d.%m.%Y",
        hoverformat="%d.%m.%Y"
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
        _insight_text = f"> ⚠️ **Симуляція (Fallback Mode):**<br/>Через апаратні обмеження (Intel Mac) або зміну API рушія, офіційна бібліотека `timesfm` не змогла ініціалізуватися (`{_error_msg}`).<br/>Активовано алгоритм симуляції (Mock), щоб продемонструвати структуру виводу. На сумісному сервері тут буде реальний AI-прогноз."
    else:
        _insight_text = f"> 💡 **Tech Lead Insight:**<br/>Google TimesFM демонструє магію **Zero-Shot** інференсу. Модель не тренувалася на даних міста {_loc}, але завдяки вагам, отриманим з мільярдів інших часових рядів, вона здатна блискавично вловлювати сезонність та тренди нашої погодної метрики (<b>{_ua_feat}</b>). Це економить тижні на розробку власних ARIMA чи LSTM моделей!"

    _insight = mo.md(_insight_text)
    mo.output.append(mo.vstack([_css_no_scroll, mo.ui.plotly(_fig_ts, config={'responsive': True, 'locale': 'uk'}), _insight], align="stretch"))
    return


if __name__ == "__main__":
    app.run()
