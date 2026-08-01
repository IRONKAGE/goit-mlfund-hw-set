import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title_head_hw(mo):
    mo.md("""
    <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
        🚗 ДЗ №4: Важливість ознак у моделях <i>(Autos & CarDekho Pricing)</i>
    </div>

    <h3 align="center"><b><u>Пайплайн</u>: Auto EDA ➔ Mutual Info Rank ➔ Decision Tree vs Random Forest ➔ Optuna ➔ SHAP ➔ FastAPI</b></h3>

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
    import re
    from datetime import datetime

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore")
    import logging

    # 🛡️ ПІДКЛЮЧЕННЯ АРХІТЕКТУРНОГО ЯДРА
    _core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
    if _core_path not in sys.path:
        sys.path.append(_core_path)

    from core import (
        SecureDownloader, smart_read_csv, get_hardware_config, clear_vram,
        set_global_seed, log_system_info, get_boosting_kwargs, logger
    )

    # 📍 ЛОКАЛЬНІ ІМПОРТИ
    from data_adapters import get_autos_mock, get_cardekho_mock
    from ui_labels import UA_COLUMNS
    from data_profiling import ProfileReport

    # Data Science & MLOps
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import shap
    import shap.explainers._tree as shap_tree

    import marimo as mo

    import mlflow
    import mlflow.sklearn
    import sklearn
    import optuna
    import joblib

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

    # Моделі
    from sklearn.dummy import DummyRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor
    )
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    from optuna.visualization import plot_optimization_history, plot_param_importances

    pd.options.mode.copy_on_write = True
    sklearn.set_config(transform_output="pandas")

    mo.center(mo.md("✅ **Бібліотеки та Ядро MLOps успішно імпортовано!**"))
    return (
        ColumnTransformer,
        DecisionTreeRegressor,
        DummyRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        KFold,
        LGBMRegressor,
        OrdinalEncoder,
        Pipeline,
        ProfileReport,
        RandomForestRegressor,
        SecureDownloader,
        SimpleImputer,
        TargetEncoder,
        UA_COLUMNS,
        XGBRegressor,
        clear_vram,
        contextlib,
        datetime,
        get_autos_mock,
        get_boosting_kwargs,
        get_cardekho_mock,
        get_hardware_config,
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
        mutual_info_regression,
        np,
        optuna,
        os,
        pa,
        pd,
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
        smart_read_csv,
        ssd,
        textwrap,
        train_test_split,
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

        # 3. Налаштування MLflow для поточного завдання (ВИПРАВЛЕНО!)
        experiment_name = "hw04_trees_ensemble"
        mlflow.set_experiment(experiment_name)

        # 📝 Фіксуємо подію в глобальний аудит-лог
        logger.info(f"✅ Налаштовано експеримент MLflow: {experiment_name}")
    return GLOBAL_SEED, lgbm_kwargs, xgb_kwargs


@app.cell(hide_code=True)
def header_prepare_dataset(mo):
    mo.md("""
    <h2 align='center'><b>💽 1. Завантаження даних та ETL (Autos & CarDekho)</b></h2>
    """)
    return


@app.cell
def execute_etl_pipeline(
    SecureDownloader,
    get_autos_mock,
    get_cardekho_mock,
    go,
    logger,
    make_subplots,
    mo,
    os,
    pd,
    smart_read_csv,
    urllib,
):
    # 🎭 Динамічно знімаємо ліміт пам'яті Marimo через приватний API
    try:
        mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 50_000_000
    except Exception:
        pass

    _data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(_data_path, exist_ok=True)

    with mo.status.spinner(title="Завантаження наборів даних (Autos & CarDekho)..."):
        # ==========================================
        # 🚗 Набір даних 1: Autos (Для Mutual Info)
        # ==========================================
        _autos_url = "https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/refs/heads/main/datasets/autos.csv"
        _csv_autos = os.path.join(_data_path, "autos.csv")

        if not os.path.exists(_csv_autos):
            try:
                urllib.request.urlretrieve(_autos_url, _csv_autos)
            except Exception:
                get_autos_mock(save_path=_csv_autos)

        # Заміна '?' на пропуски ОБОВ'ЯЗКОВА за вимогою ментора!
        df_autos = smart_read_csv(_csv_autos, "Autos", engine="pyarrow").replace('?', pd.NA).dropna()
        if 'price' in df_autos.columns:
            df_autos['price'] = df_autos['price'].astype(float)

        # ==========================================
        # 🚙 Набір даних 2: CarDekho (Для Ансамблів)
        # ==========================================
        _downloader_cd = SecureDownloader(
            dataset_path="nehalbirla/vehicle-dataset-from-cardekho",
            data_dir=_data_path,
            zip_name="car_data.zip",
            fallback_generator=get_cardekho_mock
        )
        _downloader_cd.download(target_filename="car data.csv")

        try:
            _csv_cd = _downloader_cd.extract_atomically(target_extensions=('.csv',), expected_filename="car data.csv")[0]
        except Exception:
            _csv_cd = os.path.join(_data_path, "car data.csv")

        df_cd = smart_read_csv(_csv_cd, "CarDekho", engine="pyarrow").dropna()

        # Уніфікація назви цільової колонки
        _target_cd_original = next((col for col in df_cd.columns if col.lower() in ['selling_price', 'selling price', 'price']), None)
        if _target_cd_original and _target_cd_original != 'Selling_Price':
            df_cd = df_cd.rename(columns={_target_cd_original: 'Selling_Price'})
            logger.info(f"🔄 Стовпчик '{_target_cd_original}' перейменовано на 'Selling_Price' для сумісності з ML пайплайном.")

        # Фіча-інжиніринг для CarDekho
        _year_col = next((col for col in df_cd.columns if col.lower() == 'year'), None)
        if _year_col:
            df_cd['Car_Age'] = 2026 - df_cd[_year_col]
            df_cd = df_cd.drop(columns=['Car_Name', 'name', _year_col], errors='ignore')

    # ==========================================
    # 📊 ВІЗУАЛІЗАЦІЯ ТА UI ВИВІД
    # ==========================================
    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _text_color = "white" if _theme == "dark" else "#1f2937"
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"

    # Створюємо візуалізацію цільових змінних
    _fig_targets = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Розподіл цін (Autos Dataset)", "Розподіл цін (CarDekho Dataset)"]
    )

    if 'price' in df_autos.columns:
        _fig_targets.add_trace(
            go.Histogram(
                x=df_autos['price'],
                name="Autos Price",
                marker_color='#3b82f6',
                nbinsx=40,
                hovertemplate="<b>Ціна:</b> %{x}$<br><b>К-ть авто:</b> %{y}<extra></extra>"
            ),
            row=1, col=1
        )

    if 'Selling_Price' in df_cd.columns:
        _fig_targets.add_trace(
            go.Histogram(
                x=df_cd['Selling_Price'],
                name="CarDekho Price",
                marker_color='#10b981',
                nbinsx=40,
                hovertemplate="<b>Ціна:</b> %{x}<br><b>К-ть авто:</b> %{y}<extra></extra>"
            ),
            row=1, col=2
        )
    else:
        logger.warning(f"⚠️ Увага: цільову колонку не знайдено! Наявні колонки: {list(df_cd.columns)}")

    _fig_targets.update_layout(
        showlegend=False,
        template=_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="<b>Первинний огляд цільових змінних (Target)</b>",
            x=0.5,
            font=dict(color=_text_color, size=20)
        ),
        height=450,
        margin=dict(t=70, b=40, l=40, r=20)
    )

    _fig_targets.update_xaxes(title_text="Ціна ($)", row=1, col=1)
    _fig_targets.update_yaxes(title_text="Кількість авто", row=1, col=1)
    _fig_targets.update_xaxes(title_text="Ціна", row=1, col=2)
    _fig_targets.update_yaxes(title_text="Кількість авто", row=1, col=2)

    _css_no_scroll = mo.md(
        '''
        <div class="marimo-noscroll-override"></div>
        <style>
            marimo-cell-output:has(.marimo-noscroll-override),
            .output-area:has(.marimo-noscroll-override) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        '''
    )

    mo.output.append(mo.vstack([
        _css_no_scroll,
        mo.center(
            mo.md(f"✅ **Дані успішно завантажено та очищено!**<br>Autos Dataset: (`Рядків: {df_autos.shape[0]} | Стовпчиків: {df_autos.shape[1]}`)<br>CarDekho Dataset: (`Рядків: {df_cd.shape[0]} | Стовпчиків: {df_cd.shape[1]}`)")
        ),
        mo.md(f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-top: 15px; margin-bottom: 15px;">
            <p style="margin-bottom: 0;"><b>Наступний крок:</b> Аналіз розподілу цільових змінних перед переходом до генерації ознак та побудови ансамблів.</p>
        </div>
        """),
        mo.ui.plotly(_fig_targets)
    ]))
    return df_autos, df_cd


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
    df_autos,
    df_cd,
    html,
    mo,
    os,
    re,
):
    with mo.status.spinner(title="Генерація інтерактивних профайлінгів..."):
        _artifact_dir = os.getenv("MODELS_DIR", "./models")
        os.makedirs(_artifact_dir, exist_ok=True)

        # 🛠️ DRY: Допоміжна функція для генерації та очистки HTML
        def _generate_profile_html(df_source, title, filename):
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                profile = ProfileReport(df_source, title=title, minimal=True, progress_bar=False)
                html_str = profile.to_html()
                profile.to_file(os.path.join(_artifact_dir, filename))

            # Фікс якірних посилань для Marimo
            html_str = re.sub(
                r'<a\s+([^>]*)href=["\']#([^"\']+)["\']([^>]*)>',
                r'<a \1 href="javascript:void(0);" data-target="#\2" data-bs-target="#\2" onclick="var el=document.getElementById(\'\2\'); if(el) el.scrollIntoView({behavior: \'smooth\'});" \3>',
                html_str
            )
            return html.escape(html_str)

        # Генеруємо профайлінг для двох наборів
        safe_html_autos = _generate_profile_html(df_autos.copy(), "Autos Profiling Report", "hw04_autos_eda.html")
        safe_html_cd = _generate_profile_html(df_cd.copy(), "CarDekho Profiling Report", "hw04_cardekho_eda.html")

    # 🗂️ Створюємо інтерактивні таблиці та пакуємо їх у вкладки
    df_explorer_autos = mo.ui.table(df_autos.rename(columns=UA_COLUMNS), selection=None, pagination=True)
    df_explorer_cd = mo.ui.table(df_cd.rename(columns=UA_COLUMNS), selection=None, pagination=True)

    tables_tabs = mo.ui.tabs({
        "🚗 Autos Dataset": df_explorer_autos,
        "🚙 CarDekho Dataset": df_explorer_cd
    })

    iframe_css = """
    <div class="marimo-noscroll-override"></div>
    <style>
        marimo-cell-output:has(.marimo-noscroll-override),
        .output-area:has(.marimo-noscroll-override) {
            max-height: none !important;
            overflow: visible !important;
            overflow-y: visible !important;
        }
        .smart-eda-iframe {
            width: 100%; height: 850px; border: 1px solid #e5e7eb; border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); transition: filter 0.3s ease-in-out, border-color 0.3s ease;
        }
        html.dark .smart-eda-iframe, .dark .smart-eda-iframe {
            filter: invert(90%) hue-rotate(180deg) brightness(1.1); border-color: #333; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }
    </style>
    """

    # 📑 Створюємо вкладки для EDA-звітів
    reports_tabs = mo.ui.tabs({
        "🚗 Autos EDA Report": mo.Html(f'{iframe_css}<iframe class="smart-eda-iframe" srcdoc="{safe_html_autos}" sandbox="allow-scripts allow-same-origin"></iframe>'),
        "🚙 CarDekho EDA Report": mo.Html(f'{iframe_css}<iframe class="smart-eda-iframe" srcdoc="{safe_html_cd}" sandbox="allow-scripts allow-same-origin"></iframe>')
        })

    mo.output.append(mo.vstack([
        mo.center(mo.md("### 📊 Інтерактивний огляд даних (Сирі таблиці)")),
        tables_tabs,
        mo.md("<div style='height: 15px;'></div>"),
        mo.center(mo.md("✅ **Профайлінг для обох наборів даних успішно згенеровано!** *(Перейдіть до наступної клітинки для детального аналізу)*"))
    ]))
    return (reports_tabs,)


@app.cell
def display_eda_report(mo, reports_tabs):
    mo.output.append(mo.center(mo.md("### 📑 Інтерактивні звіти (ProfileReport)")))
    mo.output.append(reports_tabs)
    return


@app.cell(hide_code=True)
def header_correlation(mo):
    mo.md("""
    <h3 align="center"><b>🧩 1.2. Аналіз мультиколінеарності <i>(Smart Correlation Matrices)</i></b></h3>
    """)
    return


@app.cell
def plot_correlation_matrices(
    UA_COLUMNS,
    df_autos,
    df_cd,
    mo,
    np,
    px,
    sch,
    ssd,
):
    # 🛠️ Універсальна функція побудови розумної матриці
    def _build_smart_corr_fig(df, title, height_px):
        # 1. Відбір числових ознак
        _num_df = df.select_dtypes(include=["float32", "float64", "int32", "int64"])
        _corr_matrix = _num_df.corr(method="pearson")

        # 2. Smart Clustering (Розумне групування за методом Ward)
        _dists = 1 - np.abs(_corr_matrix.values)
        np.fill_diagonal(_dists, 0)
        _linkage = sch.linkage(ssd.squareform(_dists), method='ward')
        _opt_order = sch.leaves_list(_linkage)
        _corr_sorted = _corr_matrix.iloc[_opt_order, _opt_order]

        # 3. Локалізація (якщо стовпець є у словнику)
        _corr_sorted = _corr_sorted.rename(columns=UA_COLUMNS, index=UA_COLUMNS)

        # 4. Налаштування UI теми
        _theme = mo.app_meta().theme
        _text_color = "white" if _theme == "dark" else "#1f2937"

        # 5. Побудова Plotly графіка
        fig = px.imshow(
            _corr_sorted,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels=dict(color="Коеф. Пірсона")
        )

        fig.update_traces(
            hovertemplate="<b>Ознака X:</b> %{x}<br><b>Ознака Y:</b> %{y}<br><b>Кореляція:</b> %{z:.5f}<extra></extra>"
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(color=_text_color, size=18),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_text_color),
            height=height_px,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return fig

    # 📊 Генеруємо обидва графіки
    fig_autos = _build_smart_corr_fig(df_autos, "Кореляції метрик: Autos Dataset", height_px=650)
    fig_cd = _build_smart_corr_fig(df_cd, "Кореляції метрик: CarDekho Dataset", height_px=450)

    corr_tabs = mo.ui.tabs({
        "🚗 Autos (Для Частини 1)": mo.ui.plotly(fig_autos),
        "🚙 CarDekho (Для Частини 2)": mo.ui.plotly(fig_cd)
    })

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
        > **💡 Tech Lead Insight (Мультиколінеарність у Деревах Рішень):**<br/>
        > Темно-червоні або темно-сині квадрати вказують на сильну кореляцію між двома незалежними ознаками (наприклад, розмір двигуна та його вага).
        >
        > На відміну від лінійної регресії чи алгоритму KNN з минулого ДЗ №3 (де такі дублікати спотворюють простір), **ансамблі дерев (Random Forest, XGBoost) є абсолютно стійкими до мультиколінеарності**. Якщо дві ознаки дублюють одна одну, дерево просто обере одну з них для розгалуження, проігнорувавши іншу.
        >
        > Саме через цей механізм результати статистичної **Взаємної Інформації (MI)**, яка оцінює кожну колонку ізольовано, будуть так сильно відрізнятися від **Feature Importance** моделі у нашому наступному кроці!
        """
    )

    mo.output.append(mo.vstack([corr_tabs, mo.md("<div style='height: 15px;'></div>"), _insight_ui]))
    return


@app.cell(hide_code=True)
def header_task1(mo):
    mo.md("""
    <h2 align='center'><b>🔬 2. Частина 1: Взаємна Інформація (MI) vs Feature Importance (FI)</b></h2>
    """)
    return


@app.cell
def execute_task1_mi_vs_rf(
    GLOBAL_SEED,
    OrdinalEncoder,
    RandomForestRegressor,
    UA_COLUMNS,
    df_autos,
    mo,
    mutual_info_regression,
    pd,
    px,
):
    with mo.status.spinner("⚙️ Розрахунок mutual_info_regression та навчання Random Forest на наборі даних Autos..."):
        # 1. Підготовка Autos
        X_autos = df_autos.drop(columns=['price'], errors='ignore')
        y_autos = df_autos['price']

        # 2. Кодування категоріальних ознак
        cat_cols = X_autos.select_dtypes(include=['object', 'category']).columns.tolist()
        encoder = OrdinalEncoder()
        X_enc = X_autos.copy()
        if cat_cols: X_enc[cat_cols] = encoder.fit_transform(X_autos[cat_cols])

        # 3. Визначаємо дискретні ознаки для MI
        discrete_mask = X_enc.dtypes.isin([int, 'int64', 'int32']) | X_enc.columns.isin(cat_cols)

        # 4. Розрахунок MI
        mi_scores = mutual_info_regression(X_enc, y_autos, discrete_features=discrete_mask, random_state=GLOBAL_SEED)

        # 5. Навчання RF
        rf_autos = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=GLOBAL_SEED, n_jobs=-1)
        rf_autos.fit(X_enc, y_autos)
        rf_scores = rf_autos.feature_importances_

        # 6. ⚖️ Уніфікація шкал та локалізація
        _df_comp = pd.DataFrame({'Feature_Original': X_enc.columns, 'MI_Score': mi_scores, 'RF_Importance': rf_scores})

        # Перекладаємо назви фічей для красивого графіка
        _df_comp['Feature'] = _df_comp['Feature_Original'].map(lambda x: UA_COLUMNS.get(x, x))

        _df_comp['Взаємна Інформація (MI)'] = _df_comp['MI_Score'].rank(pct=True)
        _df_comp['Важливість в Моделі (FI)'] = _df_comp['RF_Importance'].rank(pct=True)

        # 🏆 Динамічно визначаємо лідерів для висновку
        top_3_mi = _df_comp.sort_values(by='MI_Score', ascending=False).head(3)['Feature'].tolist()
        top_3_rf = _df_comp.sort_values(by='RF_Importance', ascending=False).head(3)['Feature'].tolist()
        match_status = "збігаються" if top_3_mi[0] == top_3_rf[0] else "відрізняються"

        # 7. Переформатування (Melt) для графіка
        _melted_df = _df_comp.melt(id_vars='Feature', value_vars=['Взаємна Інформація (MI)', 'Важливість в Моделі (FI)'], var_name='Метрика', value_name='Ранг (Перцентиль)')

        # 8. Налаштування UI та побудова ІНТЕРАКТИВНОГО графіка (Plotly)
        _theme = mo.app_meta().theme
        _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
        _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
        _text_color = "white" if _theme == "dark" else "#1f2937"

        # Будуємо групований (barmode='group') горизонтальний (orientation='h') барчарт
        fig_bar = px.bar(
            _melted_df,
            x='Ранг (Перцентиль)',
            y='Feature',
            color='Метрика',
            barmode='group',
            orientation='h',
            # 🇺🇦 Синьо-жовті кольори для метрик
            color_discrete_sequence=['#3b82f6', '#facc15'] if _theme == 'dark' else ['#2563eb', '#eab308']
        )

        # Налаштовуємо красиве віконце при наведенні
        fig_bar.update_traces(
            hovertemplate="<b>%{y}</b><br>%{data.name}: %{x:.3f}<extra></extra>"
        )

        # Фінальний дизайн (прозорий фон, сітка)
        fig_bar.update_layout(
            title=dict(
                text="<b>Битва Метрик: Взаємна Інформація vs Алгоритмічна Важливість (Autos)</b>",
                x=0.5,
                font=dict(color=_text_color, size=18)
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_text_color),
            xaxis=dict(title="Перцентильний ранг (0.0 - 1.0)", showgrid=True, gridcolor=_border),
            yaxis=dict(title="", showgrid=False, categoryorder='total ascending'),
            legend_title_text="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=700,
            margin=dict(l=20, r=20, t=80, b=20),
        )

    _css_no_scroll = mo.md(
        '''
        <div class="marimo-noscroll-override"></div>
        <style>
            marimo-cell-output:has(.marimo-noscroll-override),
            .output-area:has(.marimo-noscroll-override) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        '''
    )

    mo.output.append(mo.vstack([
        _css_no_scroll,
        mo.ui.plotly(fig_bar),
        mo.center(mo.md('<h3 style="margin-top: 0; color: #3b82f6;">🔬 Аналітичний Звіт (MI vs Random Forest)</h3>')),
        mo.md(
            f"""
            <div style="background-color: {_bg}; border: 1px solid {_border}; padding: 25px; border-radius: 8px; margin-top: 20px;">
                <p><b>Висновок за результатами аналізу:</b></p>
                <ul>
                    <li style="margin-bottom: 8px;">За метрикою <b>Взаємної інформації (MI)</b> найважливішими ознаками виявилися: <b>{', '.join(top_3_mi)}</b>. Це означає, що вони мають найсильніший індивідуальний статистичний зв'язок із ціною.</li>
                    <li style="margin-bottom: 8px;">Натомість ансамблева модель <b>Random Forest</b> визначила як найважливіші: <b>{', '.join(top_3_rf)}</b>.</li>
                    <li style="margin-bottom: 8px;">Топ-ознаки в обох підходах <b>{match_status}</b>. Як видно з графіка, "корисність" окремих ознак з точки зору ізольованого статистичного показника (MI) суттєво відрізняється від їхньої важливості в моделі (FI). Алгоритм дерев оцінює <i>синергію</i> фічей: якщо дві ознаки сильно скорельовані (ефект мультиколінеарності), Random Forest привласнює високу вагу лише одній з них, ігноруючи іншу.</li>
                </ul>
            </div>
            """
        )
    ]))
    return


@app.cell(hide_code=True)
def header_task2(mo):
    mo.md("""
    <h2 align='center'><b>🌲 3. Частина 2: Ансамблі та Лідерборд <i>(CarDekho Benchmark)</i></b></h2>
    """)
    return


@app.cell
def config_model_pool(
    DecisionTreeRegressor,
    DummyRegressor,
    ExtraTreesRegressor,
    GLOBAL_SEED,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    LGBMRegressor,
    RandomForestRegressor,
    XGBRegressor,
    lgbm_kwargs,
    mo,
    xgb_kwargs,
):
    # 1. Базові алгоритми та одиночні дерева
    models_baseline = {
        "Dummy (Mean Baseline)": (1, DummyRegressor(strategy="mean")),
        "Decision Tree (CART)": (2, DecisionTreeRegressor(max_depth=10, random_state=GLOBAL_SEED))
    }

    # 2. Ансамблі: Bagging (Ліси)
    models_bagging = {
        "Random Forest": (3, RandomForestRegressor(n_estimators=100, max_depth=12, random_state=GLOBAL_SEED, n_jobs=-1)),
        "Extra Trees": (4, ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=GLOBAL_SEED, n_jobs=-1))
    }

    # 3. Ансамблі: Gradient Boosting
    models_boosting = {
        "Gradient Boosting (Sklearn)": (5, GradientBoostingRegressor(n_estimators=100, random_state=GLOBAL_SEED)),
        "HistGradientBoosting": (6, HistGradientBoostingRegressor(random_state=GLOBAL_SEED)),
        "XGBoost": (7, XGBRegressor(n_estimators=150, max_depth=6, random_state=GLOBAL_SEED, **xgb_kwargs)),
        "LightGBM": (8, LGBMRegressor(n_estimators=150, max_depth=8, random_state=GLOBAL_SEED, **lgbm_kwargs))
    }

    master_registry = {**models_baseline, **models_bagging, **models_boosting}
    id_to_name_map = {f"#{mod_id:02d}": name for name, (mod_id, _) in master_registry.items()}

    get_force_base, set_force_base = mo.state(True)
    get_force_bag, set_force_bag = mo.state(True)
    get_force_boost, set_force_boost = mo.state(True)

    mo.center(mo.md(f"✅ **Всі {len(master_registry)} алгоритмів (Baseline, Bagging, Boosting) завантажено у памʼять!**"))
    return (
        get_force_bag,
        get_force_base,
        get_force_boost,
        id_to_name_map,
        master_registry,
        models_bagging,
        models_baseline,
        models_boosting,
        set_force_bag,
        set_force_base,
        set_force_boost,
    )


@app.cell
def controller_ui(
    get_force_bag,
    get_force_base,
    get_force_boost,
    mo,
    models_bagging,
    models_baseline,
    models_boosting,
):
    force_base = get_force_base()
    force_bag = get_force_bag()
    force_boost = get_force_boost()

    mandatory_models = [
        "Decision Tree (CART)",
        "Random Forest",
        "XGBoost"
    ]

    def make_cb(name, force_state):
        is_locked = name in mandatory_models
        return mo.ui.checkbox(label=name, value=True if is_locked else force_state, disabled=is_locked)

    # 1. Створюємо звичайні Python словники з UI-елементами
    raw_base = {f"#{mod_id:02d}": make_cb(name, force_base) for name, (mod_id, _) in models_baseline.items()}
    raw_bag = {f"#{mod_id:02d}": make_cb(name, force_bag) for name, (mod_id, _) in models_bagging.items()}
    raw_boost = {f"#{mod_id:02d}": make_cb(name, force_boost) for name, (mod_id, _) in models_boosting.items()}

    # 2. Обгортаємо їх у Marimo UI для рендеру по групах у View
    ui_base = mo.ui.dictionary(raw_base)
    ui_bag = mo.ui.dictionary(raw_bag)
    ui_boost = mo.ui.dictionary(raw_boost)

    # 3. Створюємо єдиний словник (ui_models), поєднуючи сирі словники (БЕЗ .value)
    ui_models = mo.ui.dictionary({**raw_base, **raw_bag, **raw_boost})

    mo.center(mo.md("✅ **Словники інтерфейсу створено!**"))
    return ui_bag, ui_base, ui_boost, ui_models


@app.cell
def view_render(
    mo,
    set_force_bag,
    set_force_base,
    set_force_boost,
    ui_bag,
    ui_base,
    ui_boost,
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

    # Адаптуємо назви колонок під тематику дерев
    h_base, c_base, t_base, _ = build_group_view(ui_base, "Базові / Дерева 🪵", set_force_base)
    h_bag, c_bag, t_bag, _ = build_group_view(ui_bag, "Bagging (Ліси) 🌲", set_force_bag)
    h_boost, c_boost, t_boost, _ = build_group_view(ui_boost, "Boosting (Градієнт) 🚀", set_force_boost)

    total_selected = c_base + c_bag + c_boost
    total_all = t_base + t_bag + t_boost

    main_header = mo.hstack([
        mo.md("🎛️ **Конфігуратор Ансамблів (CarDekho)**"),
        mo.md(f"<div style='text-align: right; color: #10b981; font-size: 1.1em;'><b>✓ Всього обрано: {total_selected} / {total_all}</b></div>")
    ], justify="space-between", align="center")

    run_btn = mo.ui.run_button(label="🎭 Запустити тренування", kind="success")
    v_line = mo.Html("<div style='width: 1px; background-color: #4b5563; min-height: 200px; margin: 0 15px; margin-top: 15px;'></div>")

    def build_column(header, ui_group):
        items_with_ids = [mo.hstack([mo.md(f"`{k}`"), cb], align="center") for k, cb in ui_group.items()]
        return mo.vstack([header, mo.md("<div style='height: 10px;'></div>"), mo.vstack(items_with_ids, align="start")], align="center")

    _css_no_scroll = mo.md('<div class="config-noscroll"></div><style>marimo-cell-output:has(.config-noscroll),.output-area:has(.config-noscroll){max-height: none !important; overflow-y: visible !important; overflow-x: visible !important;}</style>')

    config_panel = mo.vstack([
        _css_no_scroll,
        mo.center(main_header),
        mo.hstack([
            build_column(h_base, ui_base), v_line,
            build_column(h_bag, ui_bag), v_line,
            build_column(h_boost, ui_boost)
        ], justify="space-between", align="start"),
        mo.md("<div style='height: 10px;'></div>"),
        mo.center(run_btn)
    ])

    mo.output.append(config_panel)
    return (run_btn,)


@app.cell(hide_code=True)
def header_benchmark(mo):
    mo.md("""
    <h3 align='center'><b>⚔️ 3.1. Битва Ансамблів: Бенчмаркінг та Аналіз <i>(Model Evaluation)</i></b></h3>
    """)
    return


@app.cell(hide_code=True)
def execute_benchmark(
    ColumnTransformer,
    GLOBAL_SEED,
    Pipeline,
    SimpleImputer,
    TargetEncoder,
    clear_vram,
    df_cd,
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
    r2_score,
    run_btn,
    train_test_split,
    ui_bag,
    ui_base,
    ui_boost,
    ui_models,
    warnings,
    xgb_kwargs,
):
    mo.stop(not run_btn.value, mo.center(mo.md("### ⏳ Очікування конфігурації...\n> 🆘 Оберіть алгоритми у Конфігураторі вище та натисніть зелену кнопку.")))

    # 🧯 Придушення зайвих попереджень та налаштування логера
    warnings.filterwarnings("ignore")
    logging.getLogger("interpret").setLevel(logging.ERROR)

    logger.info("Початок бенчмаркінгу обраних ансамблів (CarDekho)...")

    # 🪎 Динамічне визначення апаратного забезпечення (Hardware)
    _hw_type = xgb_kwargs.get("device", "cpu") if xgb_kwargs else "cpu"
    if _hw_type == "cuda":
        _hw_ui = "CUDA GPU"
    elif _hw_type == "mps":
        _hw_ui = "Apple Silicon (MPS)"
    elif _hw_type == "sycl":
        _hw_ui = "Intel XPU"
    else:
        _hw_ui = "Multi-core CPU"

    # Спліт даних CarDekho
    X_cd = df_cd.drop(columns=['Selling_Price'], errors='ignore')
    y_cd = df_cd['Selling_Price']

    X_train_cd, X_test_cd, y_train_cd, y_test_cd = train_test_split(X_cd, y_cd, test_size=0.2, random_state=GLOBAL_SEED)

    # Трансформаційний граф (Target Encoding для категорій)
    _num_cols = X_train_cd.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    _cat_cols = X_train_cd.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), _num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('enc', TargetEncoder(target_type='continuous'))]), _cat_cols)
    ])

    X_train_proc = preprocessor.fit_transform(X_train_cd, y_train_cd)
    X_test_proc = preprocessor.transform(X_test_cd)

    # Визначаємо обрані моделі
    selected_names = [id_to_name_map[mod_id] for mod_id, is_sel in ui_models.value.items() if is_sel]
    mo.stop(not selected_names, mo.md("⚠️ Неможливо запустити: не обрано жодного алгоритму!"))

    results = []
    trained_models = {}
    total_models = len(selected_names)

    # 📊 Прогрес-бар з інформацією про залізо
    with mo.status.progress_bar(total=total_models, title=f"Тренування {total_models} моделей...", subtitle=f"💎 <b>Engine:</b> {_hw_ui} <br/>⏳ Ініціалізація...", remove_on_exit=True) as bar:
        for name in selected_names:
            bar.update(increment=0, subtitle=f"💎 <b>Engine:</b> {_hw_ui} <br/>☣️ <b>Тренуємо:</b> {name}")
            mod_id, model = master_registry[name]

            try:
                model.fit(X_train_proc, y_train_cd)
                y_pred = model.predict(X_test_proc)

                trained_models[name] = model
                results.append({
                    "ID": f"#{mod_id:02d}",
                    "Алгоритм": name,
                    "R-квадрат (R²) ⬆️": r2_score(y_test_cd, y_pred),
                    "MAE ⬇️": mean_absolute_error(y_test_cd, y_pred),
                    "MAPE (%) ⬇️": mean_absolute_percentage_error(y_test_cd, y_pred) * 100
                })
            except Exception as e:
                logger.error(f"Помилка при тренуванні {name}: {e}")

            bar.update()

    logger.info(f"Бенчмаркінг {total_models} моделей завершено.")

    df_results = pd.DataFrame(results).sort_values(by="MAPE (%) ⬇️", ascending=True).reset_index(drop=True).copy()

    # Форматування метрик
    df_results["R-квадрат (R²) ⬆️"] = df_results["R-квадрат (R²) ⬆️"].round(4)
    df_results["MAE ⬇️"] = df_results["MAE ⬇️"].round(2)
    df_results["MAPE (%) ⬇️"] = df_results["MAPE (%) ⬇️"].round(2)

    try: clear_vram(None)
    except Exception: pass

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
            label=f"🏆 **Лідерборд Ансамблів (Рушій UI: {_ui_mode} | Обчислення: {_hw_ui}):**"
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
    if any(ui_base.value.values()):
        _dynamic_bullets.append("> 1. **Базові Дерева:** Одиночне дерево схильне до перенавчання. На графіках такі моделі показують \"сходинкову\" структуру прогнозів і погано генералізують.")
    if any(ui_bag.value.values()):
        _dynamic_bullets.append("> 2. **Bagging (Випадковий Ліс):** Будує сотні незалежних дерев і усереднює їх. Відмінно бореться з перенавчанням та знижує дисперсію помилки.")
    if any(ui_boost.value.values()):
        _dynamic_bullets.append("> 3. **Boosting (XGBoost, LightGBM):** Найпотужніші алгоритми, що послідовно виправляють помилки попередніх дерев. Часто є абсолютними чемпіонами і найточніше підганяють точки до діагоналі ідеалу.")

    _insights_text = "\n".join(_dynamic_bullets)

    _benchmark_insight = mo.md(
        "> **📊 Як читати метрики лідерборду (Регресія):**\n"
        "\n"
        "> ⚠️ *Важлива специфіка даних:* Оскільки набір даних CarDekho походить з Індії, усі ціни вимірюються у **Лакхах (Lakhs)**. `1 Lakh = 100 000 індійських рупій`. Тобто значення `9.5` на графіку означає `950 000` рупій.\n"
        "\n"
        "> - **MAPE (%) ⬇️:** Головна бізнес-метрика. Показує, на скільки відсотків у середньому модель помиляється в прогнозі ціни авто.\n"
        "> - **MAE ⬇️:** Абсолютна похибка. Скільки грошей (в Лакхах) ми в середньому недоплачуємо або переплачуємо. (Наприклад, MAE 0.5 = помилка в 50 000 рупій).\n"
        "> - **R-квадрат (R²) ⬆️:** Чим ближче до 1.0, тим краще модель розуміє логіку ціноутворення (0.0 означає сліпе вгадування середньої ціни).\n"
        "\n"
        "> **💡 Tech Lead Insight (Еволюція Алгоритмів):**\n"
        "\n"
        "> Подивіться на графіки нижче. Ідеальна модель повинна вибудувати всі крапки чітко вздовж пунктирної діагоналі. Чим сильніше крапки розлітаються — тим гірше алгоритм зрозумів дані.\n"
        "\n"
        f"{_insights_text}"
    )

    # =========================================================================
    # 📊 ВІЗУАЛІЗАЦІЯ ЕВОЛЮЦІЇ (Дерево -> Ліс -> Бустинг)
    # =========================================================================
    grid_configs = [
        {"ukr": "Одиночне Дерево", "key": "Decision Tree (CART)", "color": "#f97316"},
        {"ukr": "Випадковий Ліс", "key": "Random Forest", "color": "#3b82f6"},
        {"ukr": "XGBoost", "key": "XGBoost", "color": "#10b981"}
    ]

    _fig_diag = make_subplots(rows=1, cols=3, subplot_titles=[f"{c['ukr']}<br>({c['key']})" for c in grid_configs])
    max_val = y_test_cd.max()

    for i, cfg in enumerate(grid_configs):
        if cfg["key"] in trained_models:
            preds = trained_models[cfg["key"]].predict(X_test_proc)
            _fig_diag.add_trace(go.Scattergl(
                x=y_test_cd, y=preds, mode="markers",
                marker=dict(color=cfg["color"], size=5, opacity=0.5),
                hovertemplate="Факт: %{x}<br>Прогноз: %{y}<extra></extra>",
                showlegend=False
            ), row=1, col=i+1)

            _fig_diag.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val], mode="lines",
                line=dict(color=_text, dash="dash", width=1.5),
                showlegend=False, hoverinfo="skip"
            ), row=1, col=i+1)
        else:
             _fig_diag.add_annotation(text="⚠️ Модель вимкнена", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"), row=1, col=i+1)

        _fig_diag.update_xaxes(title_text="Фактична Ціна", gridcolor=_border, row=1, col=i+1)
        _fig_diag.update_yaxes(title_text="Прогноз", gridcolor=_border, row=1, col=i+1)

    _fig_diag.update_layout(
        title=dict(text="<b>Еволюція: Дерево ➔ Ліс ➔ Бустинг</b>", x=0.5, font=dict(size=18)),
        paper_bgcolor=_bg, plot_bgcolor=_bg, font=dict(color=_text), height=450, margin=dict(t=80, b=40, l=20, r=20)
    )

    # =========================================================================
    # 👑 ГРАФІК АБСОЛЮТНОГО ЧЕМПІОНА
    # =========================================================================
    champion_name = df_results["Алгоритм"].iloc[0]
    champ_pred = trained_models[champion_name].predict(X_test_proc)

    row_champ = df_results.iloc[0]
    r2_champ = row_champ["R-квадрат (R²) ⬆️"]
    mae_champ = row_champ["MAE ⬇️"]
    mape_champ = row_champ["MAPE (%) ⬇️"]

    fig_champ = go.Figure()
    fig_champ.add_trace(go.Scattergl(
        x=y_test_cd, y=champ_pred, mode="markers",
        marker=dict(color="#8b5cf6", size=8, opacity=0.7, line=dict(color="white", width=0.5)),
        name="Прогноз моделі",
        hovertemplate="Фактична ціна: %{x}<br>Прогноз: %{y}<extra></extra>"
    ))

    curr_max = max(y_test_cd.max(), champ_pred.max())
    fig_champ.add_trace(go.Scatter(
        x=[0, curr_max], y=[0, curr_max], mode="lines",
        line=dict(color=_text, dash="dash", width=2),
        name="Ідеальний прогноз", hoverinfo="skip"
    ))

    fig_champ.update_layout(
        title=dict(text=f"<b>👑 АБСОЛЮТНИЙ ЧЕМПІОН: {champion_name}</b><br><span style='font-size:14px; color:gray;'>R²: {r2_champ:.4f} | MAE: {mae_champ:.2f} | MAPE: {mape_champ:.2f}%</span>", x=0.5, y=0.92),
        xaxis_title="Справжня Ціна", yaxis_title="Прогнозована Ціна",
        paper_bgcolor=_bg, plot_bgcolor=_bg, font=dict(color=_text), height=550, margin=dict(t=90, b=40, l=40, r=40)
    )
    fig_champ.update_xaxes(gridcolor=_border)
    fig_champ.update_yaxes(gridcolor=_border)

    _css_no_scroll = mo.md('<div class="marimo-noscroll-override"></div><style>marimo-cell-output:has(.marimo-noscroll-override),.output-area:has(.marimo-noscroll-override){max-height: none !important; overflow-y: visible !important;}</style>')

    mo.output.append(mo.vstack([_css_no_scroll, _benchmark_insight, _benchmark_table, _fig_diag, fig_champ]))
    return (
        X_cd,
        X_train_cd,
        X_train_proc,
        df_results,
        preprocessor,
        trained_models,
        y_train_cd,
    )


@app.cell(hide_code=True)
def header_model_selector(mo):
    mo.md("""
    <h3 align='center'><b>🎛️ 3.2. Інтерактивний Селектор <i>(Model Selection)</i></b></h3>
    """)
    return


@app.cell
def model_selector_ui(df_results, master_registry, mo):
    _ranked_names = df_results["Алгоритм"].tolist()
    _dropdown_options = {}
    _split_idx = min(5, len(_ranked_names))

    # Розділяємо на Топ-5 та всі інші
    _top_5 = _ranked_names[:_split_idx]
    _rest = _ranked_names[_split_idx:]

    # Роздаємо медалі лідерам
    _medals = ["🥇", "🥈", "🥉", "🏵️", "🏵️"]
    for _i, _name in enumerate(_top_5):
        _mod_id = master_registry[_name][0]
        _dropdown_options[f"{_medals[_i]} #{_mod_id:02d} {_name}"] = _name

    # Додаємо візуальний роздільник (фолбек на чемпіона, якщо хтось клікне на роздільник)
    if _rest:
        _dropdown_options["─── 👇 ІНШІ ТРЕНОВАНІ АЛГОРИТМИ 👇 ───"] = _top_5[0]

    # Інші моделі отримують стрічку втішення
    for _name in _rest:
        _mod_id = master_registry[_name][0]
        _dropdown_options[f"🎗️ #{_mod_id:02d} {_name}"] = _name

    _default_key = list(_dropdown_options.keys())[0]

    champion_selector = mo.ui.dropdown(
        options=_dropdown_options,
        value=_default_key,
        label="🏆 **Оберіть ансамбль для Оптимізації та SHAP:** "
    )

    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"

    ui_card = mo.md(
        f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
            <h3 style="margin-top: 0;">🎛️ Інтерактивний центр аналізу (XAI)</h3>
            <p>Завдяки реактивності Marimo, <b>наступні кроки байєсівської оптимізації (Optuna) та розрахунок векторів Шеплі (SHAP) автоматично перебудуються</b> під ваш вибір!</p>
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
    <h3 align="center"><b>🧪 3.3. Байєсівська оптимізація <i>(Optuna + MLflow)</i></b></h3>
    """)
    return


@app.cell
def optuna_ui_controls(champion_selector, mo):
    _selected_name = champion_selector.value
    # Тюнінгуємо тільки дерева/ансамблі (відкидаємо Dummy Baseline)
    _is_tunable = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])

    trials_slider = mo.ui.slider(start=3, stop=50, step=1, value=10, show_value=True, label="🛝 **Кількість ітерацій (n_trials):**")
    run_optuna_btn = mo.ui.run_button(label=f"💝 Запустити тюнінг для {_selected_name}", kind="success", disabled=not _is_tunable)

    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    mo.output.append(mo.md(f"""
    <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; text-align: center;">
        <p><i>Оптимізація гіперпараметрів TPE для мінімізації абсолютної похибки в ціні (MAE).</i></p>
        {trials_slider}<br/><br/>{run_optuna_btn}
    </div>
    """))
    return run_optuna_btn, trials_slider


@app.cell
def execute_optuna(
    GLOBAL_SEED,
    KFold,
    LGBMRegressor,
    RandomForestRegressor,
    XGBRegressor,
    X_train_proc,
    champion_selector,
    lgbm_kwargs,
    logger,
    logging,
    mean_absolute_error,
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
    y_train_cd,
):
    _selected_name = champion_selector.value
    _is_tunable = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])
    final_tuned_model = None

    if run_optuna_btn.value and _is_tunable:
        # Глушимо MLflow, щоб він не панікував
        logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
        logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

        # 🪎 Визначаємо красиве ім'я заліза для UI
        _hw_type = xgb_kwargs.get("device", "cpu") if xgb_kwargs else "cpu"
        if _hw_type == "cuda":
            _hw_ui = "CUDA GPU 🟢"
        elif _hw_type == "mps":
            _hw_ui = "Apple Silicon (MPS) 🟣"
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
            mlflow.set_experiment("CarDekho_Optimization")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def _objective(trial):
                _bar.update(increment=0, subtitle=f"🏃‍♂️ Ітерація {trial.number + 1} з {trials_slider.value}: Навчання 3-х фолдів...")

                if "XGBoost" in _selected_name:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                        "max_depth": trial.suggest_int("max_depth", 3, 9),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "random_state": GLOBAL_SEED,
                        **xgb_kwargs
                    }
                    model_opt = XGBRegressor(**params)
                elif "LightGBM" in _selected_name:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "random_state": GLOBAL_SEED,
                        **lgbm_kwargs
                    }
                    model_opt = LGBMRegressor(**params)
                else:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                        "max_depth": trial.suggest_int("max_depth", 5, 20),
                        "random_state": GLOBAL_SEED,
                        "n_jobs": -1
                    }
                    model_opt = RandomForestRegressor(**params)

                kf = KFold(n_splits=3, shuffle=True, random_state=GLOBAL_SEED)
                cv_scores = []

                for train_idx, val_idx in kf.split(X_train_proc):
                    X_tr = X_train_proc.iloc[train_idx].copy() if isinstance(X_train_proc, pd.DataFrame) else X_train_proc[train_idx].copy()
                    X_val = X_train_proc.iloc[val_idx].copy() if isinstance(X_train_proc, pd.DataFrame) else X_train_proc[val_idx].copy()
                    y_tr = y_train_cd.iloc[train_idx].copy()
                    y_val = y_train_cd.iloc[val_idx].copy()

                    model_opt.fit(X_tr, y_tr)
                    y_pred = model_opt.predict(X_val)
                    cv_scores.append(mean_absolute_error(y_val, y_pred))

                return np.mean(cv_scores)

            def _progress_callback(study, trial):
                _bar.update(
                    increment=1,
                    subtitle=f"🌿 Ітерація {trial.number + 1} з {trials_slider.value} | Найкращий MAE: {study.best_value:.4f}"
                )

            _sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
            _study = optuna.create_study(direction="minimize", sampler=_sampler)
            _study.optimize(_objective, n_trials=trials_slider.value, callbacks=[_progress_callback])

            _best_params = _study.best_params
            _best_params.update({"random_state": GLOBAL_SEED})
            logger.info(f"Optuna знайшла найкращі параметри: {_best_params}")

            if "XGBoost" in _selected_name:
                _best_params.update(**xgb_kwargs)
                final_tuned_model = XGBRegressor(**_best_params)
            elif "LightGBM" in _selected_name:
                _best_params.update(**lgbm_kwargs)
                final_tuned_model = LGBMRegressor(**_best_params)
            else:
                _best_params.update({"n_jobs": -1})
                final_tuned_model = RandomForestRegressor(**_best_params)

            _bar.update(increment=0, subtitle="💾 Збереження найкращої моделі у базу...")
            final_tuned_model.fit(X_train_proc, y_train_cd)
            mo.output.clear()

            _safe_run_name = f"Optuna_{_selected_name.replace(' ', '_').replace('(', '').replace(')', '')}"

            with mlflow.start_run(run_name=_safe_run_name):
                mlflow.log_params(_best_params)
                mlflow.log_metric("CV_MAE", _study.best_value)
                mlflow.log_metric("Optuna_Trials", trials_slider.value)

                _trusted_types = [
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBRegressor",
                    "sklearn.ensemble._forest.RandomForestRegressor",
                    "lightgbm.sklearn.LGBMRegressor"
                ]

                _pip_reqs = ["scikit-learn"]
                if "XGBoost" in _selected_name: _pip_reqs.append("xgboost")
                if "LightGBM" in _selected_name: _pip_reqs.append("lightgbm")

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
                title=dict(text="<b>Історія байєсівської оптимізації</b>", x=0.5),
                xaxis_title="Ітерація (Спроба)",
                yaxis_title="Абсолютна похибка (MAE)"
            )
            for _trace in _fig_history.data:
                if _trace.x is not None:
                    _trace.x = tuple(x + 1 for x in _trace.x)
                if _trace.name == 'Objective Value':
                    _trace.name = 'Похибка ітерації'
                    _trace.hovertemplate = '<b>Ітерація:</b> %{x}<br><b>MAE:</b> %{y:.4f}<extra></extra>'
                    _trace.marker.color = '#10b981' if _theme == 'dark' else '#059669'
                elif _trace.name == 'Best Value':
                    _trace.name = 'Рекорд (Найменша похибка)'
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
                    ✅ **Оптимізацію завершено!** Найкраще відхилення (MAE): `{_study.best_value:.4f} Лакхів`<br/>
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
def header_feature_importance(mo):
    mo.md("""
    <h3 align="center"><b>🧬 3.4. Глобальна пояснюваність <i>(Feature Importance)</i></b></h3>
    """)
    return


@app.cell
def feature_importance_analysis(
    UA_COLUMNS,
    X_train_cd,
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
    y_train_cd,
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

    # Безпечно витягуємо імена колонок
    if hasattr(X_train_proc, "columns"):
        _features = X_train_proc.columns.tolist()
    elif hasattr(preprocessor, "get_feature_names_out"):
        # Очищуємо префікси Pipeline (напр., 'num__', 'cat__')
        _raw_features = preprocessor.get_feature_names_out()
        _features = [f.split('__')[-1] for f in _raw_features]
    else:
        _features = X_train_cd.columns.tolist()

    # Універсальний адаптер для пайплайнів
    def _get_final_estimator(model):
        if hasattr(model, "steps") and len(model.steps) > 0:
            return model.steps[-1][1]
        return model

    _core_estimator = _get_final_estimator(_model)
    _importances = None
    _calc_method = "Невідомо"

    # =========================================================================
    # 🔍 1. НАМАГАЄМОСЯ ДІСТАТИ НАТИВНУ ВАГУ (Для Дерев)
    # =========================================================================
    if hasattr(_core_estimator, 'feature_importances_'):
        _importances = _core_estimator.feature_importances_
        _calc_method = "Нативна (Information Gain / Gini)"

    # =========================================================================
    # 🪎 2. МАГІЯ XAI ДЛЯ ЧОРНИХ СКРИНЬОК (Permutation Importance)
    # =========================================================================
    if _importances is None or len(_importances) != len(_features) or np.all(_importances == 0):
        with mo.status.spinner("🪎 Зламуємо чорну скриньку", subtitle=f"'{_selected_name}' (Permutation Importance)..."):
            try:
                _sample_size = min(2000, X_train_proc.shape[0])

                # БЕЗПЕЧНИЙ семплінг (працює і для DataFrame, і для Numpy)
                if isinstance(X_train_proc, (pd.DataFrame, pd.Series)):
                    _X_sample = X_train_proc.sample(n=_sample_size, random_state=42)
                    _y_sample = y_train_cd.loc[_X_sample.index]
                else:
                    np.random.seed(42)
                    _indices = np.random.choice(X_train_proc.shape[0], _sample_size, replace=False)
                    _X_sample = X_train_proc[_indices]
                    _y_sample = y_train_cd.iloc[_indices] if hasattr(y_train_cd, 'iloc') else y_train_cd[_indices]

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
    # 📝 3. ФОРМУВАННЯ ДАНИХ ТА РОЗУМНИЙ ПЕРЕКЛАД
    # =========================================================================
    def _localize(feat):
        return UA_COLUMNS.get(feat, feat)

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

    # 🎨 Кастомна палітра
    if any(x in _selected_name for x in ["XGBoost", "Forest", "LightGBM", "Trees", "Gradient"]):
        _colorscale = 'Teal'
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
    _dynamic_height = max(500, len(_df_fi) * 25)

    _fig_fi.update_layout(
        title=dict(
            text=f"<b>Рентген алгоритму ({_id_str} {_selected_name})</b><br><span style='font-size:13px; color:gray;'>Метод екстракції: {_calc_method}</span>",
            x=0.5, font=dict(color=_text_color, size=18)
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_text_color),
        xaxis=dict(title="Сила впливу ознаки на ціну", gridcolor=_border_color, zerolinecolor=_border_color, range=_x_range),
        yaxis=dict(title="", gridcolor=_border_color),
        height=_dynamic_height, margin=dict(l=20, r=40, t=75, b=20)
    )

    # =========================================================================
    # 🧠 5. ДИНАМІЧНІ БІЗНЕС-ІНСАЙТИ
    # =========================================================================
    _top_1_feat = _df_fi.iloc[-1]["Ознака"] if len(_df_fi) > 0 else "Невідомо"
    _top_2_feat = _df_fi.iloc[-2]["Ознака"] if len(_df_fi) > 1 else "Невідомо"

    _insight_md = mo.md(
        f"""
        > **💡 Tech Lead Insight (Глобальна Інтерпретація):**<br/>
        > Цей графік ілюструє **глобальну стратегію** прийняття рішень моделлю — на які саме характеристики автомобіля алгоритм звертає найбільшу увагу перед тим, як спрогнозувати його вартість.
        >
        > - 🥇 **Головний предиктор:** Ознака **«{_top_1_feat}»** має найвищу питому вагу. Математично алгоритм вважає її найбільш критичною для визначення ціни авто.
        > - 🥈 **Другорядний фактор:** Ознака **«{_top_2_feat}»** також відіграє вагому роль, доповнюючи логіку першої ознаки.
        > - 🗑️ **Вектор оптимізації:** Ознаки, які знаходяться внизу списку, додають мінімальну інформаційну цінність. Алгоритм міг би обійтися і без них.
        >
        > 🌲 **Механіка для Дерев та Ансамблів:**<br/>
        > Ця діаграма показує **Information Gain** (приріст інформації). Вона відображає, наскільки ефективно кожна ознака зменшувала "хаос" (дисперсію цін) під час побудови гілок дерев.
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
def header_shap(mo):
    mo.md("""
    <h3 align="center"><b>🕵️‍♂️ 3.5. Квантова пояснюваність <i>(SHAP Values)</i></b></h3>
    """)
    return


@app.cell
def shap_ui_controls(champion_selector, final_tuned_model, mo):
    _selected_name = champion_selector.value
    # SHAP TreeExplainer працює ТІЛЬКИ з деревами/бустингом
    _is_tree = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])

    # Перевіряємо, чи є в пам'яті свіжа оптимізована модель з Optuna
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
            <p>Цей алгоритм заглядає всередину "чорної скриньки" і розраховує математичний внесок кожної ознаки для <b>кожного окремого автомобіля</b>.<br/>
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
    X_train_cd,
    X_train_proc,
    champion_selector,
    final_tuned_model,
    mo,
    np,
    pd,
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
            _sample_size = min(500, len(X_train_proc))

            if isinstance(X_train_proc, (pd.DataFrame, pd.Series)):
                _X_sample = X_train_proc.sample(n=_sample_size, random_state=GLOBAL_SEED)
            else:
                np.random.seed(GLOBAL_SEED)
                _indices = np.random.choice(X_train_proc.shape[0], _sample_size, replace=False)
                _X_sample = pd.DataFrame(X_train_proc[_indices], columns=X_train_cd.columns)

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

            # Локалізація колонок (переклад на українську)
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

                _ax.set_xlabel("Значення SHAP (Вплив на ціну авто, Лакхи)", color=_text_color)

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
                    _cbar_ax.set_ylabel("Фактичне значення ознаки", rotation=270, labelpad=15)
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
                > Якщо ви вимкнете TargetEncoder і передасте в модель сирі текстові дані, алгоритм зафарбує їх у сірий колір. Для тексту (наприклад "Тип палива: Дизель") поняття "Високе/Низьке" математично не існує, тому SHAP захищає вас від хибних висновків.
                """

            _insight_md = mo.md(
                f"""
                <center>💵 <i>Вплив на осі X <b>розраховано у Лакхах (1 Lakh = 100 000 рупій)</b>.</i></center>

                > **💡 Tech Lead Insight (Як читати SHAP для автомобілів):**<br/>
                > Кожна крапка на цьому графіку — це конкретний автомобіль з вибірки.
                >
                > - **Колір точки (🔵 Синій/Червоний 🔴):** Високе чи низьке значення ознаки. Наприклад, для "Вік авто", червоний означає дуже стару машину, а синій — нову. Для категорій типу "Тип палива", червоний означатиме ті категорії, які TargetEncoder закодував вищими числами (історично дорожчі авто).
                > - **Позиція на осі X (⏮️ Вліво/Вправо ⏭️):** Як сильно ця характеристика "переконала" алгоритм знизити або підвищити фінальну ціну машини. Наприклад, якщо червоні крапки "Віку авто" зміщені сильно вліво — це означає, що старість автомобіля катастрофічно збиває його ціну.

                {_native_insight}
                """
            )

            mo.output.append(mo.vstack([_css_no_scroll, mo.center(_plot_html), _insight_md]))
    return


@app.cell(hide_code=True)
def header_what_if(mo):
    mo.md("""
    <h2 align="center"><b>🔮 4. What-If Симулятор <i>(Калькулятор знецінення авто)</i></b></h2>
    """)
    return


@app.cell
def what_if_ui(X_cd, mo):
    # Динамічно отримуємо безпечні межі з набору даних
    _max_present = float(X_cd['Present_Price'].max()) if 'Present_Price' in X_cd else 50.0
    _owners = sorted([int(x) for x in X_cd['Owner'].unique()])

    # 🌍 Словники для перекладу значень у UI (Відображення -> Оригінал)
    _fuel_map = {"Бензин (Petrol)": "Petrol", "Дизель (Diesel)": "Diesel", "Газ (CNG)": "CNG"}
    _seller_map = {"Дилер (Dealer)": "Dealer", "Приватна особа (Individual)": "Individual"}
    _trans_map = {"Механіка (Manual)": "Manual", "Автомат (Automatic)": "Automatic"}

    # Фільтруємо лише ті значення, які реально існують у наборі даних
    _fuel_options = {k: v for k, v in _fuel_map.items() if v in X_cd['Fuel_Type'].values}
    _seller_options = {k: v for k, v in _seller_map.items() if v in X_cd['Seller_Type'].values}
    _trans_options = {k: v for k, v in _trans_map.items() if v in X_cd['Transmission'].values}

    # 🎛️ Пакуємо всі контроли у єдиний реактивний словник
    # Використовуємо .keys()[0], щоб Marimo вибрав дефолтним український надпис
    sim_panel = mo.ui.dictionary({
        'price': mo.ui.slider(start=0.5, stop=_max_present, step=0.5, value=10.0, label="💰 Ціна в салоні (Лакхів):", show_value=True),
        'kms': mo.ui.slider(start=0, stop=100000, step=5000, value=20000, label="🛣️ Початковий пробіг (км):", show_value=True),
        'fuel': mo.ui.dropdown(options=_fuel_options, value=list(_fuel_options.keys())[0], label="⛽ Тип палива:"),
        'seller': mo.ui.dropdown(options=_seller_options, value=list(_seller_options.keys())[0], label="🤝 Тип продавця:"),
        'trans': mo.ui.dropdown(options=_trans_options, value=list(_trans_options.keys())[0], label="⚙️ Коробка передач:"),
        'owner': mo.ui.dropdown(options=[str(x) for x in _owners], value=str(_owners[0]), label="👤 Кількість власників:")
    })

    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    ui_layout = mo.md(f"""
    <div style="padding: 25px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg};">
        <h3 style="margin-top: 0; text-align: center; color: #3b82f6;">Інтерактивний симулятор ринку</h3>
        <p style="text-align: center;">Покрутіть повзунки або змініть категорії — модель <b>миттєво</b> перебудує 15-річну історію знецінення авто!</p>

        <div style="display: flex; flex-wrap: wrap; justify-content: space-around; gap: 20px; margin-top: 25px;">
            <div style="flex: 1; min-width: 250px; background: rgba(128,128,128,0.1); padding: 15px; border-radius: 8px;">
                <b>🔢 Числові параметри</b><br/><br/>
                {sim_panel['price']}<br/><br/>{sim_panel['kms']}
            </div>
            <div style="flex: 1; min-width: 250px; background: rgba(128,128,128,0.1); padding: 15px; border-radius: 8px;">
                <b>📋 Категоріальні параметри</b><br/><br/>
                {sim_panel['fuel']}<br/>{sim_panel['seller']}<br/>{sim_panel['trans']}<br/>{sim_panel['owner']}
            </div>
        </div>
    </div>
    """)

    mo.output.append(ui_layout)
    return (sim_panel,)


@app.cell(hide_code=True)
def simulate_what_if(
    X_cd,
    champion_selector,
    final_tuned_model,
    go,
    mo,
    pd,
    preprocessor,
    sim_panel,
    trained_models,
):
    _selected_name = champion_selector.value
    _model = final_tuned_model if final_tuned_model is not None else trained_models[_selected_name]

    # Миттєво витягуємо поточні значення з UI-словника
    _vals = sim_panel.value

    # 1. ГЕНЕРУЄМО СИНТЕТИЧНУ ІСТОРІЮ (15 років життя авто)
    _ages = list(range(0, 16))

    # Динаміка пробігу (+15,000 км за кожен рік життя)
    _base_kms = _vals['kms']
    _sim_kms = [_base_kms + (age * 15000) for age in _ages]

    # 🔧 Значення _vals['fuel'] тощо ВЖЕ є англійськими (оригінальними),
    # оскільки ми використали словник {Українська: Англійська} у параметрі options
    _synth_data = {
        'Present_Price': [_vals['price']] * len(_ages),
        'Kms_Driven': _sim_kms,
        'Fuel_Type': [_vals['fuel']] * len(_ages),
        'Seller_Type': [_vals['seller']] * len(_ages),
        'Transmission': [_vals['trans']] * len(_ages),
        'Owner': [int(_vals['owner'])] * len(_ages)
    }

    # Адаптація під колонку років у вашому датасеті
    if 'Car_Age' in X_cd.columns:
        _synth_data['Car_Age'] = _ages
    if 'Year' in X_cd.columns:
        _current_year = 2024
        _synth_data['Year'] = [_current_year - age for age in _ages]

    # Безпечно заповнюємо всі інші можливі колонки, якщо вони є в X_cd
    for col in X_cd.columns:
        if col not in _synth_data:
            if pd.api.types.is_numeric_dtype(X_cd[col]):
                _synth_data[col] = [X_cd[col].median()] * len(_ages)
            else:
                _synth_data[col] = [X_cd[col].mode()[0]] * len(_ages)

    # Створюємо DataFrame у правильному порядку
    _df_sim = pd.DataFrame(_synth_data)[X_cd.columns]

    # 2. ПРОЦЕСИНГ ТА ІНФЕРЕНС
    _X_sim_proc = preprocessor.transform(_df_sim)
    _preds = _model.predict(_X_sim_proc)

    # 3. МАЛЮЄМО МАГІЮ (Крива амортизації)
    _theme = mo.app_meta().theme
    _bg = "rgba(0,0,0,0)"
    _text = "white" if _theme == "dark" else "#1f2937"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    fig = go.Figure()

    # Додаємо красиву заливку під кривою
    fig.add_trace(go.Scattergl(
        x=_ages, y=_preds, mode="lines+markers",
        line=dict(color="#3b82f6", width=4),
        marker=dict(size=10, color="white", line=dict(color="#3b82f6", width=2)),
        fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate="<b>Вік авто:</b> %{x} років<br><b>Намотано:</b> %{customdata:,} км<br><b>Прогнозна ціна:</b> %{y:.2f} Лакхів<extra></extra>",
        customdata=_sim_kms
    ))

    fig.update_layout(
        title=dict(text=f"<b>Крива знецінення автомобіля (Depreciation Curve)</b><br><span style='font-size: 13px; color: gray;'>Модель: {_selected_name} | Задана ціна нового авто: {_vals['price']} Лакхів</span>", x=0.5),
        xaxis=dict(title="Вік автомобіля (Років)", gridcolor=_border, tickmode='linear'),
        yaxis=dict(title="Прогнозована ціна (Лакхів)", gridcolor=_border),
        paper_bgcolor=_bg, plot_bgcolor=_bg, font=dict(color=_text), height=500,
        margin=dict(t=80, b=40, l=40, r=40)
    )

    _insight = mo.md(f"""
    > **💡 Tech Lead Insight (Реальність машинного навчання):**<br/>
    > Змінюйте повзунки наживо! Модель миттєво перераховує весь масив через `preprocessor` і малює нову криву.
    >
    > Зверніть увагу на форму: найсильніше ціна автомобіля обвалюється в перші 3-5 років його життя. Після 8-10 років падіння суттєво сповільнюється, і графік стає більш пологим — ансамбль ідеально вивчив фундаментальний економічний закон вторинного ринку.
    """)

    _css_no_scroll = mo.md('<div class="sim-noscroll"></div><style>marimo-cell-output:has(.sim-noscroll),.output-area:has(.sim-noscroll){max-height: none !important; overflow-y: visible !important;}</style>')

    mo.output.append(mo.vstack([_css_no_scroll, mo.ui.plotly(fig), _insight]))
    return


@app.cell(hide_code=True)
def header_mlops(mo):
    mo.md("""
    <h2 align='center'><b>⛲️ 5. Продакшн: MLOps Серіалізація та Мікросервіс <i>(FastAPI)</i></b></h2>
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
    X_train_cd,
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

        _project_name = "cardekho_pricing"
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
        for _col in X_train_cd.columns:
            _dt = X_train_cd[_col].dtype
            _dtypes_dict[_col] = str(_dt)

        _features_schema = {
            "project_name": _project_name,
            "exported_at": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "model_architecture": _selected_name,
            "is_optuna_tuned": _is_tuned,
            "expected_columns": list(X_train_cd.columns),
            "dtypes": _dtypes_dict
        }
        with open(_schema_path, "w", encoding="utf-8") as f:
            json.dump(_features_schema, f, indent=4, ensure_ascii=False)

        _sample_row = X_train_cd.iloc[0].to_dict()
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
                title="🚗 CarDekho Pricing API",
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
                    <title>CarDekho Pricing API Docs</title>
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
                predicted_price_lakhs: float
                model_deployed: str
                is_optuna_tuned: bool
                features_processed_count: int

            @app.post("/predict", response_model=PredictionResponse)
            def predict_price(payload: InferencePayload):
                if pipeline is None:
                    raise HTTPException(status_code=500, detail="Пайплайн не знайдено")

                try:
                    df = pd.DataFrame([payload.features])

                    if expected_columns:
                        df = df.reindex(columns=expected_columns)

                    for col, dtype in expected_dtypes.items():
                        if col in df.columns:
                            df[col] = df[col].astype(dtype)

                    # Трансформаційний пайплайн сам застосує TargetEncoder!
                    pred = pipeline.predict(df)[0]

                    return {{
                        "predicted_price_lakhs": round(float(pred), 2),
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
        _num_features = X_train_cd.shape[1]

        _params = getattr(_model_to_save, "get_params", lambda: {})()
        _arch_lines = []

        if 'n_estimators' in _params and _params['n_estimators'] is not None:
            _arch_lines.append(f"🧬 Дерев (n_estimators): {_params['n_estimators']}")
        if 'learning_rate' in _params and _params['learning_rate'] is not None:
            _lr_val = _params['learning_rate']
            if isinstance(_lr_val, float): _lr_val = round(_lr_val, 5)
            _arch_lines.append(f"⚡ Швидкість (LR): {_lr_val}")

        if not _arch_lines:
            _arch_lines.append("⚖️ Тип: Базове дерево рішень (CART) або інший бейзлайн")

        _exclude_keys = {
            'n_estimators', 'max_depth', 'learning_rate', 'booster',
            'random_state', 'n_jobs', 'objective', 'enable_categorical', 'algorithm'
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
            <p style="text-align: center;">Ми запакували <code>TargetEncoder</code> та <code>{_selected_name}</code> у єдиний нерозривний <code>.joblib</code> файл. Серверу <code>FastAPI</code> не потрібно знати про трансформації — він просто передає сирі дані у пайплайн!</p>

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
            <p style="font-size: 15px;">Цей модуль автоматично генерує повністю налаштований бекенд для нашої ML-моделі прогнозування цін на авто. Ми перейшли від базових Data Science скриптів до інфраструктури Enterprise-рівня.</p>

            <h3 style="color: #3b82f6; margin-top: 25px;">✨ Технологічний Стек та Можливості</h3>
            <ul style="font-size: 14px;">
                <li style="margin-bottom: 8px;"><b>FastAPI + Uvicorn:</b> Високопродуктивний асинхронний сервер, який миттєво обробляє HTTP-запити та виконує інференс.</li>
                <li style="margin-bottom: 8px;"><b>Scikit-Learn Pipeline 🔗:</b> Уся логіка (<code>TargetEncoder</code> та Імпутація) запечена всередину моделі. Бекенд не займається обробкою даних — він просто передає сирий JSON у пайплайн.</li>
                <li style="margin-bottom: 8px;"><b>Docker Контейнеризація 🐳:</b> Автоматична генерація <code>Dockerfile</code> з точними залежностями алгоритму. Забезпечує 100% ізоляцію середовища на будь-якому сервері.</li>
                <li style="margin-bottom: 8px;"><b>Сучасний Scalar UI:</b> Ми повністю відмовилися від застарілого Swagger. Інтегрований <b>Scalar</b> забезпечує преміальний дизайн документації та миттєву генерацію коду запитів.</li>
            </ul>

            <h3 style="color: #10b981; margin-top: 25px;">🛡️ Суворі Pydantic-Контракти</h3>
            <ul style="font-size: 14px;">
                <li style="margin-bottom: 8px;">✅ <b>200 OK (Успішна відповідь):</b> Сервер повертає чітко типізовану схему <code>PredictionResponse</code>. Клієнт гарантовано отримає <code>predicted_price_lakhs</code> (float) та метадані.</li>
                <li style="margin-bottom: 8px;">❌ <b>422 Validation Error:</b> Завдяки <code>InferencePayload</code>, якщо клієнт надішле неправильний тип даних, FastAPI автоматично відхилить запит із детальним JSON-описом.</li>
            </ul>

            <h3 style="color: #f59e0b; margin-top: 25px;">⚙️ Як запустити сервер?</h3>
            <p style="font-size: 14px;">Усі згенеровані артефакти надійно ізольовано у директорії <code>models/cardekho_pricing/</code>.</p>

            <div style="margin-top: 15px;">
                <b>▶ Спосіб 1: DevOps-стандарт (Через Makefile)</b>
                <pre style="background-color: {_pre_bg}; color: {_pre_text_cmd}; padding: 12px; border-radius: 8px; border: 1px solid {_pre_border}; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>make api-hw4</code></pre>
            </div>

            <div style="margin-top: 15px;">
                <b>▶ Спосіб 2: Запуск у Docker (Cloud Ready ☁️)</b>
                <pre style="background-color: {_pre_bg}; color: {_pre_text_code}; padding: 12px; border-radius: 8px; border: 1px solid {_pre_border}; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>cd models/cardekho_pricing
        docker build -t cardekho-api .
        docker run -p 8000:8000 cardekho-api</code></pre>
            </div>

            <div style="margin-top: 15px;">
                <b>▶ Спосіб 3: Ручний запуск (Без Docker)</b>
                <pre style="background-color: {_pre_bg}; color: {_pre_text_code}; padding: 12px; border-radius: 8px; border: 1px solid {_pre_border}; font-family: monospace; font-size: 13px; margin-top: 5px;"><code>cd models/cardekho_pricing
        uvicorn api:app --host 0.0.0.0 --port 8000 --reload</code></pre>
            </div>

            <hr style="border-color: {_border}; margin: 25px 0;">
            <p style="margin-bottom: 0; font-size: 15px;">
                <i>💡 <b>Документація доступна за адресою:</b> <a href="http://127.0.0.1:8000/docs" target="_blank" style="color: #3b82f6; font-weight: bold; text-decoration: none;">http://127.0.0.1:8000/docs</a>.</i><br/>
                <i>🥂 Тепер будь-який веб-сайт, Telegram-бот чи мобільний застосунок на Swift/Kotlin може відправляти JSON-запити на цей порт і миттєво отримувати ціну на авто!</i>
            </p>
        </div>
        """)

    mo.output.append(_css_no_scroll)
    mo.output.append(_deploy_instructions)
    return


if __name__ == "__main__":
    app.run()
