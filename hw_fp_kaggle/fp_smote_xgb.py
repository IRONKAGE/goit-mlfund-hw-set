import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title_head_fp(mo):
    mo.md("""
    <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
        🏆 Фінальний проект (Kaggle): Прогнозування відтоку клієнтів <i>(Customer Churn)</i>
    </div>

    <h3 align="center"><b><u>Пайплайн</u>: Auto EDA ➔ Model Leaderboard ➔ Optuna Marathon ➔ HITL <i>(Режим Архітектора)</i> ➔ Threshold Tuning ➔ SMOTE+LightGBM ➔ XAI</b></h3>

    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Machine Learning: Fundamentals and Applications [08.2026]</i></p>
    """)
    return


@app.cell(hide_code=True)
def configure_dependencies():
    # 📦 1. Стандартні бібліотеки
    import os
    import sys
    import warnings
    import json
    import base64
    import hashlib
    import contextlib
    import textwrap
    import re
    import html
    import urllib.request
    from datetime import datetime

    # 🤫 Глушимо системні попередження для чистоти UI
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
    logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

    # 🛡️ 2. ПІДКЛЮЧЕННЯ АРХІТЕКТУРНОГО ЯДРА
    _core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
    if _core_path not in sys.path:
        sys.path.append(_core_path)

    from core import (
        SecureDownloader, smart_read_csv, get_hardware_config, clear_vram,
        set_global_seed, log_system_info, get_boosting_kwargs, logger
    )

    from data_profiling import ProfileReport

    # 📊 3. Data Science & Візуалізація
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import matplotlib.pyplot as plt
    import marimo as mo

    # 🤖 4. Machine Learning & MLOps
    import mlflow
    import mlflow.sklearn
    import sklearn
    import optuna

    # Квантова пояснюваність (XAI)
    import shap
    import shap.explainers._tree as shap_tree

    # ⚖️ Балансування класів (Imbalanced-Learn)
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    # 🧠 Scikit-Learn інфраструктура
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
    from sklearn.feature_selection import SelectFromModel

    # 🎯 Метрики оцінки
    from sklearn.metrics import (
        balanced_accuracy_score, roc_auc_score, f1_score,
        confusion_matrix, accuracy_score, precision_recall_curve, classification_report
    )

    # 🌲 Алгоритми класифікації (Розширений пул "God Mode")
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, HistGradientBoostingClassifier,
        ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
    )
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from interpret.glassbox import ExplainableBoostingClassifier

    # 📈 Візуалізація Optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances

    # ⚡ Оптимізація пам'яті Pandas (Copy-on-Write)
    pd.options.mode.copy_on_write = True
    sklearn.set_config(transform_output="pandas")

    mo.output.append(mo.center(mo.md("✅ **Усі Бібліотеки, Ядро MLOps та Розширені Алгоритми успішно імпортовано!**")))
    return (
        AdaBoostClassifier,
        ColumnTransformer,
        DummyClassifier,
        ExplainableBoostingClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        ImbPipeline,
        LGBMClassifier,
        LogisticRegression,
        OneHotEncoder,
        ProfileReport,
        RandomForestClassifier,
        SMOTE,
        SelectFromModel,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        XGBClassifier,
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        clear_vram,
        clone,
        confusion_matrix,
        contextlib,
        f1_score,
        get_boosting_kwargs,
        get_hardware_config,
        go,
        hashlib,
        html,
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
        plot_optimization_history,
        plot_param_importances,
        plt,
        px,
        re,
        roc_auc_score,
        set_global_seed,
        shap,
        shap_tree,
        smart_read_csv,
        train_test_split,
        urllib,
    )


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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

        # Детектимо оптимальне залізо для алгоритмів
        device, device_ui_name = get_hardware_config(global_seed=GLOBAL_SEED)

        # Перекладаємо конфіги заліза для бустингів (XGBoost / LightGBM)
        xgb_kwargs, lgbm_kwargs = get_boosting_kwargs(device)

        # 3. Налаштування MLflow для фінального змагання
        experiment_name = "fp_kaggle_customer_churn"
        mlflow.set_experiment(experiment_name)

        logger.info(f"✅ Налаштовано експеримент MLflow: {experiment_name}")
    return GLOBAL_SEED, device, lgbm_kwargs, xgb_kwargs


@app.cell(hide_code=True)
def header_data(mo):
    mo.md("""
    <h2 align='center'><b>💽 1. Завантаження даних та Smart EDA</b></h2>
    """)
    return


@app.cell(hide_code=True)
def execute_etl(go, make_subplots, mo, os, smart_read_csv, urllib):
    # 🎭 Динамічно знімаємо ліміт пам'яті Marimo через приватний API
    try:
        mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 50_000_000
    except Exception:
        pass

    _data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(_data_path, exist_ok=True)

    _train_path = os.path.join(_data_path, 'final_proj_data.csv')
    _test_path = os.path.join(_data_path, 'final_proj_test.csv')

    # Прямі посилання на файли у репозиторії GitHub
    _github_base = "https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/main/competition/"

    with mo.status.spinner(title="Завантаження та обробка набору даних Kaggle..."):
        # 🌐 Автоматичне завантаження (Idempotency)
        for _filename, _filepath in [("final_proj_data.csv", _train_path), ("final_proj_test.csv", _test_path)]:
            if not os.path.exists(_filepath):
                urllib.request.urlretrieve(_github_base + _filename, _filepath)

        # Використовуємо оптимізований pyarrow рушій
        df_train = smart_read_csv(_train_path, "Kaggle Train Data", engine="pyarrow")
        df_test = smart_read_csv(_test_path, "Kaggle Test Data", engine="pyarrow")

        X_train_full = df_train.drop(columns=['y'])
        y_train_full = df_train['y']

        # Підрахунок типів ознак та дисбалансу
        _num_cols = X_train_full.select_dtypes(include=['int64', 'float64']).shape[1]
        _cat_cols = X_train_full.shape[1] - _num_cols
        _target_counts = y_train_full.value_counts().sort_index()

    # ==========================================
    # 📊 ВІЗУАЛІЗАЦІЯ ТА UI ВИВІД
    # ==========================================
    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _text_color = "white" if _theme == "dark" else "#1f2937"
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"

    _fig_eda = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        subplot_titles=["Дисбаланс класів (Target 'y')", "Структура ознак (Features)"]
    )

    _fig_eda.add_trace(
        go.Bar(
            x=['Клас 0 (Залишились)', 'Клас 1 (Відтік)'],
            y=[_target_counts.get(0, 0), _target_counts.get(1, 0)],
            marker_color=['#3b82f6', '#ef4444'],
            text=[_target_counts.get(0, 0), _target_counts.get(1, 0)],
            textposition='auto',
            name="Target",
            hovertemplate="<b>%{x}</b><br>Кількість клієнтів: %{y}<extra></extra>"
        ),
        row=1, col=1
    )

    _fig_eda.add_trace(
        go.Pie(
            labels=['Числові (Numeric)', 'Категоріальні (Categorical)'],
            values=[_num_cols, _cat_cols],
            marker=dict(colors=['#10b981', '#f59e0b']),
            hole=0.4,
            name="Features",
            hovertemplate="<b>%{label}</b><br>Кількість ознак: %{value}<extra></extra>"
        ),
        row=1, col=2
    )

    _fig_eda.update_layout(
        showlegend=False,
        template=_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="<b>Аналіз Kaggle Набору Даних: Відтік Клієнтів</b>",
            x=0.5,
            font=dict(color=_text_color, size=20)
        ),
        height=450,
        margin=dict(t=70, b=40, l=40, r=20)
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
        mo.center(
            mo.md(f"✅ **Дані успішно завантажено та проаналізовано!**<br>Тренувальний набір: (`Рядків: {X_train_full.shape[0]} | Ознак: {X_train_full.shape[1]}`)")
        ),
        mo.md(f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-top: 15px; margin-bottom: 15px;">
            <p style="margin-bottom: 8px;"><b>1. Завантаження:</b> Файли автоматично стягнуто з офіційного репозиторію через <code>urllib</code>.</p>
            <p style="margin-bottom: 8px;"><b>2. Розподіл ознак:</b> Ідентифіковано {_num_cols} числових та {_cat_cols} категоріальних ознак для <code>ColumnTransformer</code>.</p>
            <p style="margin-bottom: 0;"><b>3. Балансування:</b> Виявлено дисбаланс класів, що підтверджує необхідність використання алгоритму <code>SMOTE</code> у пайплайні.</p>
        </div>
        """),
        mo.ui.plotly(_fig_eda)
    ]))
    return X_train_full, df_test, y_train_full


@app.cell(hide_code=True)
def header_auto_eda(mo):
    mo.md("""
    <h3 align="center"><b>📊 1.1. Автоматичний EDA <i>(fg-data-profiling)</i></b></h3>
    """)
    return


@app.cell
def generate_eda_report(
    ProfileReport,
    X_train_full,
    contextlib,
    hashlib,
    html,
    mo,
    os,
    re,
    y_train_full,
):
    _artifact_dir = os.getenv("MODELS_DIR", "./models")
    os.makedirs(_artifact_dir, exist_ok=True)

    # 🧠 Розумне кешування (Smart Caching)
    # Створюємо унікальний "зліпок" (сигнатуру) поточного стану набору даних
    # Якщо ви видалите колонку або відфільтруєте рядки, сигнатура зміниться!
    _data_signature = f"{X_train_full.shape}_{list(X_train_full.columns)}_{float(y_train_full.sum())}"
    _hash = hashlib.md5(_data_signature.encode()).hexdigest()[:8]

    _artifact_path = os.path.join(_artifact_dir, f"fp_customer_churn_eda_{_hash}.html")

    with mo.status.spinner(title="Генерація інтерактивного профайлінгу..."):
        # Тепер кеш жорстко прив'язаний до КОНКРЕТНОЇ версії даних
        if os.path.exists(_artifact_path):
            with open(_artifact_path, "r", encoding="utf-8") as _f:
                html_str = _f.read()
        else:
            # Збираємо тренувальний набір даних разом з таргетом для повноцінного аналізу
            df_eda = X_train_full.copy()
            df_eda['y'] = y_train_full

            # 🌌 Відправляємо весь консольний спам у "чорну діру"
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                profile = ProfileReport(
                    df_eda,
                    title="Customer Churn Profiling Report",
                    minimal=True,
                    progress_bar=False,
                )
                html_str = profile.to_html()

                # 🧹 Очищаємо від старих неактуальних звітів
                for _f in os.listdir(_artifact_dir):
                    if _f.startswith("fp_customer_churn_eda_") and _f.endswith(".html"):
                        try: os.remove(os.path.join(_artifact_dir, _f))
                        except Exception: pass

                # Зберігаємо новий артефакт
                profile.to_file(_artifact_path)

        # 🛠️ Фікс якірних посилань для Marimo (щоб навігація у звіті працювала плавно)
        html_str = re.sub(
            r'<a\s+([^>]*)href=["\']#([^"\']+)["\']([^>]*)>',
            r'<a \1 href="javascript:void(0);" data-target="#\2" data-bs-target="#\2" onclick="var el=document.getElementById(\'\2\'); if(el) el.scrollIntoView({behavior: \'smooth\'});" \3>',
            html_str
        )
        # Захист HTML від "зламу" структури Marimo
        safe_html = html.escape(html_str)

    # 🎨 Авто-адаптивний CSS: слухає тему Marimo (Dark/Light)
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

    # Пакуємо все в Iframe
    html_report = mo.Html(f'{iframe_css}<iframe class="smart-eda-iframe" srcdoc="{safe_html}" sandbox="allow-scripts allow-same-origin"></iframe>')

    mo.output.append(mo.vstack([
        mo.center(mo.md("### 📊 Інтерактивний огляд даних (EDA)")),
        mo.md("<div style='height: 15px;'></div>"),
        mo.center(mo.md("✅ **Профайлінг успішно згенеровано!** *(Ізольований фрейм готовий до виводу у наступній клітинці)*"))
    ]))
    return (html_report,)


@app.cell
def display_eda_report(html_report, mo):
    # 📜 Окремою клітинкою гарантовано виводимо звіт, щоб він не блокував рендеринг UI
    mo.output.append(html_report)
    return


@app.cell(hide_code=True)
def header_feature_types(mo):
    mo.md("""
    <h3 align='center'><b>🔍 1.2. Розподіл типів ознак</b></h3>
    """)
    return


@app.cell
def extract_feature_types(X_train_full, mo, pd):
    # 🔍 Підготовка списків ознак для ColumnTransformer
    # 🛠️ ВИПРАВЛЕНО (регресія): select_dtypes(['int64','float64','object','category','bool'])
    # розпізнає лише "класичні" numpy-дтайпи. smart_read_csv читає через engine="pyarrow"
    # і часто повертає pandas/pyarrow-розширені типи (Int64, string[pyarrow] тощо) —
    # такі колонки мовчки губилися через remainder='drop' у ColumnTransformer
    # (симптом: ValueError "Shape of passed values (…, 212), indices imply (…, 230)").
    #
    # is_numeric_dtype / is_bool_dtype коректно розпізнають і numpy, і pandas nullable,
    # і pyarrow-backed типи — тож жодна колонка більше не губиться.
    numeric_features = [
        col for col in X_train_full.columns
        if pd.api.types.is_numeric_dtype(X_train_full[col]) and not pd.api.types.is_bool_dtype(X_train_full[col])
    ]
    categorical_features = [col for col in X_train_full.columns if col not in numeric_features]

    # Евристика: якщо категоріальних не знайшлось, шукаємо приховані категорії
    # серед числових (менше 15 унікальних значень)
    if not categorical_features:
        categorical_features = [col for col in X_train_full.columns if X_train_full[col].nunique() < 15]
        numeric_features = [col for col in X_train_full.columns if col not in categorical_features]

    # 🛡️ Захисна перевірка: жодна колонка не повинна загубитись мовчки
    assert len(numeric_features) + len(categorical_features) == X_train_full.shape[1], (
        f"Класифікація ознак не покриває всі колонки: "
        f"{len(numeric_features) + len(categorical_features)} з {X_train_full.shape[1]}"
    )

    mo.output.append(
        mo.center(
            mo.md(f"✅ **Ознаки успішно класифіковано!**<br>Готово до обробки: (`Числових: {len(numeric_features)} | Категоріальних: {len(categorical_features)}`)")
        )
    )
    return categorical_features, numeric_features


@app.cell(hide_code=True)
def header_leaderboard(mo):
    mo.md("""
    <h2 align='center'><b>🕹️ 2. Швидкий Лідерборд алгоритмів <i>(Model Screening)</i></b></h2>
    """)
    return


@app.cell
def holdout_split(
    GLOBAL_SEED,
    X_train_full,
    mo,
    train_test_split,
    y_train_full,
):
    # 🔪 Holdout-спліт лише для швидкого відбору алгоритмів.
    # Фінальна модель (у Optuna-марафоні) все одно валідується через CV
    # і перенавчається на всіх даних перед сабмітом.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=GLOBAL_SEED,
    )

    mo.output.append(
        mo.center(
            mo.md(
                f"✅ **Holdout-спліт для Лідерборду створено!** "
                f"(`Train: {X_train.shape[0]} | Valid: {X_valid.shape[0]} "
                f"| Частка відтоку у Valid: {y_valid.mean():.2%}`)"
            )
        )
    )
    return X_train, X_valid, y_train, y_valid


@app.cell
def model_data_state(
    AdaBoostClassifier,
    DummyClassifier,
    ExplainableBoostingClassifier,
    ExtraTreesClassifier,
    GLOBAL_SEED,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ImbPipeline,
    LGBMClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SMOTE,
    StandardScaler,
    XGBClassifier,
    lgbm_kwargs,
    mo,
    xgb_kwargs,
    y_train,
):
    # ⚖️ Рахуємо РЕАЛЬНИЙ дисбаланс класів для ваг
    _neg, _pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = _neg / max(_pos, 1)

    # 1. Базові моделі
    models_baseline = {
        "Dummy (Mode Baseline)": (1, DummyClassifier(strategy="prior")),
        "Logistic Regression (SMOTE)": (2, ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=GLOBAL_SEED)),
            ('clf', LogisticRegression(max_iter=1000, random_state=GLOBAL_SEED)),
        ])),
    }

    # 2. Класичні Ансамблі
    models_ensembles = {
        "Random Forest": (3, RandomForestClassifier(class_weight='balanced', n_estimators=300, max_depth=15, random_state=GLOBAL_SEED, n_jobs=-1)),
        "Extra Trees": (4, ExtraTreesClassifier(class_weight='balanced', n_estimators=300, max_depth=15, random_state=GLOBAL_SEED, n_jobs=-1)),
        "AdaBoost (1995)": (5, AdaBoostClassifier(n_estimators=100, random_state=GLOBAL_SEED)),
        "Gradient Boosting (Classic)": (6, GradientBoostingClassifier(n_estimators=200, random_state=GLOBAL_SEED)),
    }

    # 3. Бустинг (OHE сумісний)
    models_boosting = {
        "HistGradientBoosting (OHE)": (7, HistGradientBoostingClassifier(max_iter=300, random_state=GLOBAL_SEED)),
        "LightGBM (OHE)": (8, LGBMClassifier(class_weight='balanced', random_state=GLOBAL_SEED, verbosity=-1, **lgbm_kwargs)),
        "XGBoost (OHE)": (9, XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=300, random_state=GLOBAL_SEED, eval_metric='logloss', **xgb_kwargs)),
    }

    # 4. Нативні алгоритми та XAI
    models_native = {
        "LightGBM (Native)": (10, LGBMClassifier(class_weight='balanced', random_state=GLOBAL_SEED, verbosity=-1, **lgbm_kwargs)),
        "XGBoost (Native)": (11, XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=300, enable_categorical=True, random_state=GLOBAL_SEED, eval_metric='logloss', **xgb_kwargs)),
        "Explainable Boosting (EBM)": (12, ExplainableBoostingClassifier(random_state=GLOBAL_SEED, n_jobs=-1))
    }

    master_registry = {**models_baseline, **models_ensembles, **models_boosting, **models_native}
    id_to_name_map = {f"#{mod_id:02d}": name for name, (mod_id, _) in master_registry.items()}

    get_force_base, set_force_base = mo.state(True)
    get_force_ens, set_force_ens = mo.state(True)
    get_force_boost, set_force_boost = mo.state(True)
    get_force_nat, set_force_nat = mo.state(True)

    mo.output.append(
        mo.center(mo.md(
            f"✅ **{len(master_registry)} релевантних алгоритмів завантажено!** "
            f"(`scale_pos_weight = {scale_pos_weight:.2f}` — прораховано з реальних даних)"
        ))
    )
    return (
        get_force_base,
        get_force_boost,
        get_force_ens,
        get_force_nat,
        id_to_name_map,
        master_registry,
        models_baseline,
        models_boosting,
        models_ensembles,
        models_native,
        set_force_base,
        set_force_boost,
        set_force_ens,
        set_force_nat,
    )


@app.cell
def controller_ui(
    get_force_base,
    get_force_boost,
    get_force_ens,
    get_force_nat,
    mo,
    models_baseline,
    models_boosting,
    models_ensembles,
    models_native,
):
    force_base = get_force_base()
    force_ens = get_force_ens()
    force_boost = get_force_boost()
    force_nat = get_force_nat()

    # 🔒 Ці 3 моделі завжди увімкнені — вони є опорними точками порівняння
    # 🛠️ ЗМІНЕНО: прибрав "Explainable Boosting (EBM)" зі списку обов'язкових —
    # це найповільніша модель у пулі (адитивний циклічний бустинг на 200+ ознаках),
    # тепер її можна вимкнути для пришвидшення повторних прогонів
    mandatory_models = [
        "Dummy (Mode Baseline)",
        "Logistic Regression (SMOTE)",
        "XGBoost (Native)",
    ]

    # 🐢 ВИПРАВЛЕНО (швидкість): AdaBoost та класичний Gradient Boosting не мають
    # n_jobs (однопоточні) і на табличних даних такого розміру майже завжди
    # програють HistGradientBoosting/XGBoost/LightGBM. Раніше були увімкнені
    # за замовчуванням і помітно уповільнювали кожен прогін лідерборду.
    # Залишаємо доступними, але вимикаємо за замовчуванням.
    slow_optional_models = ["AdaBoost (1995)", "Gradient Boosting (Classic)"]

    def make_cb(name, force_state):
        is_locked = name in mandatory_models
        if is_locked:
            default_value = True
        elif name in slow_optional_models:
            default_value = False
        else:
            default_value = force_state
        return mo.ui.checkbox(label=name, value=default_value, disabled=is_locked)

    ui_base = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_base) for name, (mod_id, _) in models_baseline.items()})
    ui_ens = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_ens) for name, (mod_id, _) in models_ensembles.items()})
    ui_boost = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_boost) for name, (mod_id, _) in models_boosting.items()})
    ui_nat = mo.ui.dictionary({f"#{mod_id:02d}": make_cb(name, force_nat) for name, (mod_id, _) in models_native.items()})

    mo.center(mo.md("✅ **Словники інтерфейсу створено!**"))
    return ui_base, ui_boost, ui_ens, ui_nat


@app.cell
def view_render(
    mo,
    set_force_base,
    set_force_boost,
    set_force_ens,
    set_force_nat,
    ui_base,
    ui_boost,
    ui_ens,
    ui_nat,
):
    def build_group_view(ui_dict, title, set_state_fn):
        vals = ui_dict.value
        total = len(vals)
        completed = sum(vals.values())
        current_state_is_true = completed == total and total > 0
        icon_char = "✅" if current_state_is_true else ("☑️" if completed > 0 else "🔲")

        def toggle_all(_): set_state_fn(not current_state_is_true)
        icon_button = mo.ui.button(label=icon_char, on_click=toggle_all, kind="neutral")
        header = mo.center(mo.hstack([icon_button, mo.md(f"**<span style='font-size: 1.05em;'>{title} ({completed}/{total})</span>**")], align="center"))
        return header, completed, total

    h_base, c_base, t_base = build_group_view(ui_base, "Базові 🐣", set_force_base)
    h_ens, c_ens, t_ens = build_group_view(ui_ens, "Класичні Ансамблі 🥁", set_force_ens)
    h_boost, c_boost, t_boost = build_group_view(ui_boost, "OHE Бустинг 🚀", set_force_boost)
    h_nat, c_nat, t_nat = build_group_view(ui_nat, "Нативні & EBM 🫀", set_force_nat)

    total_selected = c_base + c_ens + c_boost + c_nat
    total_all = t_base + t_ens + t_boost + t_nat

    main_header = mo.hstack([
        mo.md("🎛️ **Конфігуратор алгоритмів (Churn Leaderboard)**"),
        mo.md(f"<div style='text-align: right; color: #10b981; font-size: 1.1em;'><b>✓ Всього обрано: {total_selected} / {total_all}</b></div>"),
    ], justify="space-between", align="center")

    run_btn = mo.ui.run_button(label="🎭 Запустити тренування пулу моделей", kind="success")
    v_line = mo.Html("<div style='width: 1px; background-color: #4b5563; min-height: 180px; margin: 0 15px; margin-top: 15px;'></div>")

    def build_column(header, ui_group):
        items_with_ids = [mo.hstack([mo.md(f"`{k}`"), cb], align="center") for k, cb in ui_group.items()]
        return mo.vstack([header, mo.md("<div style='height: 10px;'></div>"), mo.vstack(items_with_ids, align="start")], align="center")

    _css_no_scroll = mo.md(
        '<div class="config-noscroll"></div><style>'
        'marimo-cell-output:has(.config-noscroll),.output-area:has(.config-noscroll)'
        '{max-height: none !important; overflow-y: visible !important; overflow-x: visible !important;}'
        '</style>'
    )

    config_panel = mo.vstack([
        _css_no_scroll,
        mo.center(main_header),
        mo.hstack([
            build_column(h_base, ui_base), v_line,
            build_column(h_ens, ui_ens), v_line,
            build_column(h_boost, ui_boost), v_line,
            build_column(h_nat, ui_nat)
        ], justify="space-between", align="start"),
        mo.md("<br>"),
        mo.center(run_btn),
    ])
    mo.output.append(config_panel)
    return (run_btn,)


@app.cell(hide_code=True)
def execute_benchmark(
    ColumnTransformer,
    ImbPipeline,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
    X_train,
    X_valid,
    accuracy_score,
    balanced_accuracy_score,
    categorical_features,
    clear_vram,
    confusion_matrix,
    device,
    f1_score,
    go,
    id_to_name_map,
    logger,
    master_registry,
    mo,
    np,
    numeric_features,
    pd,
    roc_auc_score,
    run_btn,
    ui_base,
    ui_boost,
    ui_ens,
    ui_nat,
    xgb_kwargs,
    y_train,
    y_valid,
):
    # 🛑 Чекаємо натискання кнопки
    mo.stop(
        not run_btn.value,
        mo.center(mo.md("### ⏳ Очікування конфігурації...\n> 🆘 Оберіть алгоритми у Конфігураторі вище та натисніть зелену кнопку.")),
    )

    _hw_ui = "CUDA GPU" if xgb_kwargs.get("device") == "cuda" else "Apple Silicon (MPS)" if xgb_kwargs.get("device") == "mps" else "Multi-core CPU"

    logger.info("Початок бенчмаркінгу пулу моделей для Customer Churn...")

    # 1. Збираємо всі обрані моделі з 4-х колонок
    selected_names = []
    for ui_group in [ui_base, ui_ens, ui_boost, ui_nat]:
        selected_names.extend([id_to_name_map[mod_id] for mod_id, is_sel in ui_group.value.items() if is_sel])

    mo.stop(not selected_names, mo.md("⚠️ **Неможливо запустити: не обрано жодного алгоритму!**"))

    all_models = [(name, master_registry[name]) for name in selected_names]
    results = []
    trained_models = {}

    # 🧱 2. Два препроцесори: OHE (для лінійних/дерев) та Native (category dtype для LightGBM/XGBoost/EBM)
    preprocessor_ohe = ColumnTransformer([
        ('num', ImbPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', ImbPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features),
    ])

    preprocessor_native = ColumnTransformer([
        ('num', ImbPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', ImbPipeline([('imputer', SimpleImputer(strategy='most_frequent'))]), categorical_features),
    ], remainder='drop')

    X_train_ohe = preprocessor_ohe.fit_transform(X_train)
    X_valid_ohe = preprocessor_ohe.transform(X_valid)

    X_train_native = pd.DataFrame(
        preprocessor_native.fit_transform(X_train),
        columns=numeric_features + categorical_features,
    )
    X_valid_native = pd.DataFrame(
        preprocessor_native.transform(X_valid),
        columns=numeric_features + categorical_features,
    )

    # Спочатку кастуємо в str, щоб уникнути помилки XGBoost з float-категоріями
    for c in categorical_features:
        X_train_native[c] = X_train_native[c].astype(str).astype('category')
        X_valid_native[c] = X_valid_native[c].astype(str).astype(
            pd.CategoricalDtype(categories=X_train_native[c].cat.categories)
        )

    # Список моделей, які потребують Native набору даних
    _native_names = {"LightGBM (Native)", "XGBoost (Native)", "Explainable Boosting (EBM)"}

    # 🚀 3. Запуск тренування з прогрес-баром
    with mo.status.progress_bar(
        total=len(all_models),
        title=f"Тренування {len(selected_names)} моделей...",
        subtitle=f"💎 <b>Engine:</b> {_hw_ui} <br/>⏳ Ініціалізація...",
        remove_on_exit=True,
    ) as bar:
        for name, (mod_id, _model) in all_models:
            bar.update(increment=0, subtitle=f"💎 <b>Engine:</b> {_hw_ui} <br/>☣️ <b>Тренуємо:</b> {name}")

            # Розумний маршрутизатор наборів даних
            _is_native = name in _native_names
            _X_train_curr = X_train_native if _is_native else X_train_ohe
            _X_valid_curr = X_valid_native if _is_native else X_valid_ohe

            # Тренування та прогноз
            _model.fit(_X_train_curr, y_train)
            y_pred = _model.predict(_X_valid_curr)

            # Безпечне витягування ймовірностей (для ROC-AUC)
            if hasattr(_model, "predict_proba"):
                y_proba = _model.predict_proba(_X_valid_curr)[:, 1]
            elif hasattr(_model, "decision_function"):
                y_proba = 1 / (1 + np.exp(-_model.decision_function(_X_valid_curr)))
            else:
                y_proba = y_pred

            # Розрахунок специфічності (Скільки клієнтів, що ЗАЛИШИЛИСЬ, ми вгадали правильно)
            _tn, _fp, _fn, _tp = confusion_matrix(y_valid, y_pred).ravel()
            _specificity = _tn / (_tn + _fp) if (_tn + _fp) > 0 else 0

            trained_models[name] = _model
            results.append({
                "ID": f"#{mod_id:02d}",
                "Алгоритм": name,
                "Balanced Acc ⬆️": balanced_accuracy_score(y_valid, y_pred),
                "ROC-AUC ⬆️": roc_auc_score(y_valid, y_proba),
                "F1-Score ⬆️": f1_score(y_valid, y_pred, zero_division=0),
                "Specificity ⬆️": _specificity,
                "Accuracy ⬆️": accuracy_score(y_valid, y_pred),
            })
            bar.update()

    logger.info(f"Бенчмаркінг {len(all_models)} моделей завершено.")
    df_results = pd.DataFrame(results).sort_values(by="Balanced Acc ⬆️", ascending=False).reset_index(drop=True)

    for _col in ["Balanced Acc ⬆️", "ROC-AUC ⬆️", "F1-Score ⬆️", "Specificity ⬆️", "Accuracy ⬆️"]:
        df_results[_col] = df_results[_col].round(4)

    clear_vram(device)

    # 📊 4. Візуалізація результатів
    _justify_config = {col: "center" for col in df_results.columns}
    _benchmark_table = mo.ui.table(
        df_results,
        selection=None,
        page_size=50,
        text_justify_columns=_justify_config,
        label="🏆 **Лідерборд алгоритмів (сортовано за Balanced Accuracy — метрика Kaggle):**",
    )

    champion_name = df_results["Алгоритм"].iloc[0]
    champion_score = df_results["Balanced Acc ⬆️"].iloc[0]

    _is_champion_boosting = any(x in champion_name for x in ["LightGBM", "XGBoost", "Gradient", "EBM"])

    _champion_note = (
        "Це очікувано — сучасний бустинг та ансамблі зазвичай виграють на табличних даних."
        if _is_champion_boosting else
        "⚠️ Несподівано: базові моделі обійшли бустинг. Можливо, дані занадто зашумлені або потрібно перевірити препроцесинг."
    )

    _champion_card = mo.md(f"""
    > 👑 **Лідер бенчмарку:** `{champion_name}` — Balanced Accuracy на holdout: `{champion_score}`
    >
    > {_champion_note}
    >
    > 💡 **Наступний крок:** У наступному блоці ми передамо найкращу архітектуру на глибоку оптимізацію гіперпараметрів за допомогою Optuna.
    """)

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

    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"

    _fig_leaderboard = go.Figure(go.Bar(
        x=df_results["Balanced Acc ⬆️"],
        y=df_results["Алгоритм"],
        orientation='h',
        marker_color=['#10b981' if i == 0 else '#3b82f6' for i in range(len(df_results))],
        text=df_results["Balanced Acc ⬆️"],
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>Balanced Acc: %{x}<extra></extra>",
    ))
    _fig_leaderboard.update_layout(
        template=_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text="<b>🏆 Лідерборд: Balanced Accuracy по моделях</b>", x=0.5, font=dict(color=_text_color, size=18)),
        yaxis=dict(autorange="reversed"),
        height=550,
        margin=dict(t=60, b=40, l=20, r=20),
    )

    mo.output.append(_css_no_scroll)
    mo.output.append(_benchmark_table)
    mo.output.append(_champion_card)
    mo.output.append(_fig_leaderboard)
    return (df_results,)


@app.cell(hide_code=True)
def header_optuna_section(mo):
    mo.md("""
    <h2 align='center'><b>🏆 3. Вибір Чемпіона та Optuna Marathon</b></h2>
    """)
    return


@app.cell
def model_selector_ui(df_results, master_registry, mo):
    _ranked_names = df_results["Алгоритм"].tolist()
    _dropdown_options = {}

    # Беремо Топ-3 для швидкого доступу
    _split_idx = min(3, len(_ranked_names))
    _top_3 = _ranked_names[:_split_idx]
    _rest = _ranked_names[_split_idx:]

    _medals = ["🥇", "🥈", "🥉"]
    for _i, _name in enumerate(_top_3):
        _mod_id = master_registry[_name][0]
        _dropdown_options[f"{_medals[_i]} #{_mod_id:02d} {_name}"] = _name

    if _rest:
        _dropdown_options["─── 👇 ІНШІ ТРЕНОВАНІ АЛГОРИТМИ 👇 ───"] = _top_3[0]

    for _name in _rest:
        _mod_id = master_registry[_name][0]
        _dropdown_options[f"🎗️ #{_mod_id:02d} {_name}"] = _name

    _default_key = list(_dropdown_options.keys())[0]

    champion_selector = mo.ui.dropdown(
        options=_dropdown_options,
        value=_default_key,
        label="🏆 **Оберіть алгоритм для Оптимізації (Optuna):** "
    )

    _theme = mo.app_meta().theme
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"

    ui_card = mo.md(
        f"""
        <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-bottom: 15px; text-align: center;">
            <h3 style="margin-top: 0;">🎛️ Центр Байєсівської Оптимізації</h3>
            <p>Завдяки реактивності Marimo, <b>усі наступні графіки та розрахунки автоматично перебудуються</b> під ваш вибір!</p>
            <div style="display: flex; justify-content: center; align-items: center; margin-top: 15px;">
                {champion_selector}
            </div>
        </div>
        """
    )
    mo.output.append(ui_card)
    return (champion_selector,)


@app.cell
def optuna_ui_controls(champion_selector, mo):
    _selected_name = champion_selector.value

    # 🔓 Дозволяємо тюнінгувати ВСЕ, окрім базового Dummy
    _is_tunable = "Dummy" not in _selected_name

    trials_slider = mo.ui.slider(start=5, stop=100, step=1, value=10, show_value=True, label="🛝 **Кількість ітерацій (n_trials):**")
    run_optuna_btn = mo.ui.run_button(label=f"💝 Запустити тюнінг для {_selected_name}", kind="success", disabled=not _is_tunable)

    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    mo.output.append(mo.md(f"""
    <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; text-align: center;">
        <p><i>Оптимізація гіперпараметрів для максимізації метрики змагання (Balanced Accuracy).</i></p>
        {trials_slider}<br/><br/>{run_optuna_btn}
    </div>
    """))
    return run_optuna_btn, trials_slider


@app.cell
def execute_optuna(
    AdaBoostClassifier,
    ColumnTransformer,
    ExplainableBoostingClassifier,
    ExtraTreesClassifier,
    GLOBAL_SEED,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ImbPipeline,
    LGBMClassifier,
    LogisticRegression,
    OneHotEncoder,
    RandomForestClassifier,
    SMOTE,
    SelectFromModel,
    SimpleImputer,
    StandardScaler,
    StratifiedKFold,
    XGBClassifier,
    X_train,
    balanced_accuracy_score,
    categorical_features,
    champion_selector,
    device,
    lgbm_kwargs,
    logging,
    mlflow,
    mo,
    np,
    numeric_features,
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
    _is_tunable = "Dummy" not in _selected_name

    final_tuned_pipeline = None
    best_params = {}
    # 🛠️ ВИПРАВЛЕНО: раніше study жив як `_study` — приватна для цієї клітинки
    # змінна, яку marimo ніколи не публікує назовні. Розділ 7 (XAI) намагався
    # взяти глобальну `study` і падав з NameError. Ініціалізуємо тут, щоб вона
    # завжди була визначена (навіть до першого запуску Optuna), і повертаємо нижче.
    study = None

    if run_optuna_btn.value and _is_tunable:
        logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
        logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

        _hw_ui = "CUDA GPU 🟢" if getattr(device, "type", "") == "cuda" else (
            "Apple Silicon (MPS) 🟣" if getattr(device, "type", "") == "mps" else "Multi-core CPU ⚙️"
        )

        with mo.status.progress_bar(
            total=trials_slider.value,
            title=f"🦄 Тюнінг {_selected_name}",
            subtitle="⏳ Підготовка даних (Preprocessing & Feature Selection)...",
            remove_on_exit=True
        ) as _bar:

            os.makedirs("mlruns", exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///mlruns/mlruns.db")
            mlflow.set_experiment("Customer_Churn_Optimization")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # =================================================================
            # 🚀 1. МЕГА-ОПТИМІЗАЦІЯ ШВИДКОСТІ (Виносимо важке за межі циклу)
            # =================================================================
            _num_tf = ImbPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            _cat_tf = ImbPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            _prep = ColumnTransformer(transformers=[('num', _num_tf, numeric_features), ('cat', _cat_tf, categorical_features)], remainder='drop')

            _X_train_prep = _prep.fit_transform(X_train)
            _feat_sel = SelectFromModel(XGBClassifier(n_estimators=30, random_state=GLOBAL_SEED, **xgb_kwargs), threshold='median')
            _X_train_sel = _feat_sel.fit_transform(_X_train_prep, y_train)

            _X_train_opt = pd.DataFrame(_X_train_sel)
            _y_train_opt = y_train.reset_index(drop=True)

            def _objective(trial):
                _bar.update(increment=0, subtitle=f"🏃‍♂️ Ітерація {trial.number + 1} з {trials_slider.value}: Навчання 3-х фолдів...")

                # 🧠 2. ОПТИМІЗОВАНИЙ ПРОСТІР ПОШУКУ
                if "XGBoost" in _selected_name:
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                        'max_depth': trial.suggest_int('max_depth', 3, 8),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                        'min_child_weight': trial.suggest_int('min_child_weight', 2, 12),
                        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                        'eval_metric': 'logloss',
                        'random_state': GLOBAL_SEED,
                        'n_jobs': -1,
                        **xgb_kwargs
                    }
                    model_cls = XGBClassifier(**param)
                elif "LightGBM" in _selected_name:
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                        'random_state': GLOBAL_SEED,
                        'n_jobs': -1,
                        **lgbm_kwargs
                    }
                    model_cls = LGBMClassifier(**param)
                elif "Random Forest" in _selected_name:
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                        'max_depth': trial.suggest_int('max_depth', 5, 20),
                        'random_state': GLOBAL_SEED,
                        'n_jobs': -1
                    }
                    model_cls = RandomForestClassifier(**param)
                elif "Extra Trees" in _selected_name:
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                        'max_depth': trial.suggest_int('max_depth', 5, 25),
                        'random_state': GLOBAL_SEED,
                        'n_jobs': -1
                    }
                    model_cls = ExtraTreesClassifier(**param)
                elif "Logistic Regression" in _selected_name:
                    param = {
                        'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
                        'max_iter': 1000,
                        'solver': 'saga',
                        'random_state': GLOBAL_SEED,
                        'n_jobs': -1
                    }
                    model_cls = LogisticRegression(**param)
                elif "AdaBoost" in _selected_name:
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                        'random_state': GLOBAL_SEED
                    }
                    model_cls = AdaBoostClassifier(**param)
                elif "HistGradient" in _selected_name:
                    param = {
                        'max_iter': trial.suggest_int('max_iter', 100, 300, step=50),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                        'random_state': GLOBAL_SEED
                    }
                    model_cls = HistGradientBoostingClassifier(**param)
                elif "Gradient Boosting (Classic)" in _selected_name:
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                        'random_state': GLOBAL_SEED
                    }
                    model_cls = GradientBoostingClassifier(**param)
                elif "Explainable Boosting" in _selected_name:
                    param = {
                        'max_bins': trial.suggest_int('max_bins', 128, 512, step=128),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                        'random_state': GLOBAL_SEED,
                        'n_jobs': -1
                    }
                    model_cls = ExplainableBoostingClassifier(**param)
                else:
                    raise ValueError(f"Алгоритм {_selected_name} не підтримується для оптимізації.")

                # =================================================================
                # 🚀 3. ЕТАЛОННИЙ МАНУАЛЬНИЙ K-FOLD
                # =================================================================
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=GLOBAL_SEED)
                scores = []

                for train_idx, val_idx in cv.split(_X_train_opt, _y_train_opt):
                    X_tr = _X_train_opt.iloc[train_idx].copy()
                    X_val = _X_train_opt.iloc[val_idx].copy()
                    y_tr = _y_train_opt.iloc[train_idx].copy()
                    y_val = _y_train_opt.iloc[val_idx].copy()

                    X_tr_smote, y_tr_smote = SMOTE(random_state=GLOBAL_SEED).fit_resample(X_tr, y_tr)

                    model_cls.fit(X_tr_smote, y_tr_smote)
                    y_pred = model_cls.predict(X_val)
                    scores.append(balanced_accuracy_score(y_val, y_pred))

                return np.mean(scores)

            def _progress_callback(study, trial):
                _bar.update(
                    increment=1,
                    subtitle=f"🌿 Ітерація {trial.number + 1} з {trials_slider.value} | Найкращий Bal. Acc: {study.best_value:.5f}"
                )

            _sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
            study = optuna.create_study(direction="maximize")
            study.optimize(_objective, n_trials=trials_slider.value, callbacks=[_progress_callback])

            best_params = study.best_params
            best_params['random_state'] = GLOBAL_SEED

            # =================================================================
            # 💾 4. ВІДНОВЛЕННЯ ФІНАЛЬНОЇ МОДЕЛІ
            # =================================================================
            _bar.update(increment=0, subtitle="💾 Збірка ультимативного пайплайну...")

            if "XGBoost" in _selected_name:
                best_params.update({'eval_metric': 'logloss', 'n_jobs': -1, **xgb_kwargs})
                final_model = XGBClassifier(**best_params)
            elif "LightGBM" in _selected_name:
                best_params.update({'n_jobs': -1, **lgbm_kwargs})
                final_model = LGBMClassifier(**best_params)
            elif "Random Forest" in _selected_name:
                best_params.update({'n_jobs': -1})
                final_model = RandomForestClassifier(**best_params)
            elif "Extra Trees" in _selected_name:
                best_params.update({'n_jobs': -1})
                final_model = ExtraTreesClassifier(**best_params)
            elif "Logistic Regression" in _selected_name:
                best_params.update({'max_iter': 1000, 'solver': 'saga', 'n_jobs': -1})
                final_model = LogisticRegression(**best_params)
            elif "AdaBoost" in _selected_name:
                final_model = AdaBoostClassifier(**best_params)
            elif "HistGradient" in _selected_name:
                final_model = HistGradientBoostingClassifier(**best_params)
            elif "Gradient Boosting (Classic)" in _selected_name:
                final_model = GradientBoostingClassifier(**best_params)
            elif "Explainable Boosting" in _selected_name:
                best_params.update({'n_jobs': -1})
                final_model = ExplainableBoostingClassifier(**best_params)

            final_tuned_pipeline = ImbPipeline([
                ('preprocessor', _prep),
                ('feature_selection', _feat_sel),
                ('smote', SMOTE(random_state=GLOBAL_SEED)),
                ('classifier', final_model)
            ])

            final_tuned_pipeline.fit(X_train, y_train)
            mo.output.clear()

            _safe_run_name = f"Optuna_{_selected_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            with mlflow.start_run(run_name=_safe_run_name):
                mlflow.log_params(best_params)
                mlflow.log_metric("CV_Balanced_Accuracy", study.best_value)

                # 🛡️ ДОДАНО ВІДСУТНІ ТИПИ: lightgbm.basic.Booster та collections.OrderedDict
                _trusted_types = [
                    "imblearn.over_sampling._smote.base.SMOTE",
                    "imblearn.pipeline.Pipeline",
                    "numpy.dtype",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                    "lightgbm.sklearn.LGBMClassifier",
                    "lightgbm.basic.Booster",
                    "collections.OrderedDict",
                    "sklearn.ensemble._forest.RandomForestClassifier",
                    "sklearn.ensemble._forest.ExtraTreesClassifier",
                    "sklearn.linear_model._logistic.LogisticRegression",
                    "sklearn.ensemble._weight_boosting.AdaBoostClassifier",
                    "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier",
                    "sklearn.ensemble._gb.GradientBoostingClassifier",
                    "interpret.glassbox._ebm._ebm.ExplainableBoostingClassifier"
                ]

                mlflow.sklearn.log_model(
                    final_tuned_pipeline,
                    artifact_path="champion_pipeline",
                    skops_trusted_types=_trusted_types,
                    pip_requirements=["scikit-learn", "xgboost", "lightgbm", "imbalanced-learn", "interpret"]
                )

            # 4. Візуалізація
            _fig_history = plot_optimization_history(study)
            try: _fig_params = plot_param_importances(study)
            except Exception: _fig_params = None

            _theme = mo.app_meta().theme
            _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
            _text_color = "white" if _theme == "dark" else "#1f2937"

            _fig_history.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_text_color),
                title=dict(text="<b>Історія байєсівської оптимізації</b>", x=0.5),
                xaxis_title="Ітерація (Спроба)", yaxis_title="Balanced Accuracy"
            )
            if _fig_params:
                _fig_params.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_text_color),
                    title=dict(text="<b>Важливість гіперпараметрів</b>", x=0.5),
                    xaxis_title="Ступінь впливу", yaxis_title="Гіперпараметр"
                )

            _params_table = style_dataframe(pd.DataFrame([best_params]), text_align="center", vertical_lines=True, show_index=False)
            _plots_ui = mo.hstack([_fig_history, _fig_params], justify="center") if _fig_params else _fig_history
            _css_no_scroll = mo.md('<div class="optuna-noscroll"></div><style>marimo-cell-output:has(.optuna-noscroll),.output-area:has(.optuna-noscroll){max-height: none !important; overflow-y: visible !important;}</style>')

            _result_ui = mo.vstack([
                _css_no_scroll,
                mo.md(f"✅ **Оптимізацію завершено!** Найкраща точність: `{study.best_value:.5f}`<br/>💎 **Залізо:** `{_hw_ui}`"),
                mo.Html(f"<div style='overflow-x: auto; border: 1px solid {_border}; border-radius: 8px;'>{_params_table}</div>"),
                _plots_ui
            ])
            mo.output.append(_result_ui)
    return best_params, final_tuned_pipeline, study


@app.cell(hide_code=True)
def header_architect(mo):
    mo.md("""
    <h2 align="center"><b>🧑‍💻 4. Режим Архітектора <i>(Human-in-the-Loop / Manual Override)</i></b></h2>
    """)
    return


@app.cell
def manual_override_ui(best_params, mo):
    # Зупиняємо виконання, якщо Optuna ще не відпрацювала
    mo.stop(not best_params, mo.center(mo.md("⏳ **Очікування:** Спочатку запустіть Optuna Marathon у попередньому блоці!")))

    override_switch = mo.ui.switch(label="**Увімкнути ручне коригування параметрів (Manual Override)**")

    # 🎛️ Динамічно будуємо слайдери залежно від того, що є в best_params
    _sliders = {}

    # Слайдери для Дерев / Бустингів
    if 'n_estimators' in best_params:
        _sliders['n_estimators'] = mo.ui.slider(start=50, stop=1000, step=10, value=best_params['n_estimators'], label="🌲 Кількість дерев (n_estimators)", show_value=True)
    if 'max_depth' in best_params:
        _sliders['max_depth'] = mo.ui.slider(start=2, stop=30, step=1, value=best_params['max_depth'], label="📏 Макс. глибина (max_depth) - контроль перенавчання", show_value=True)
    if 'learning_rate' in best_params:
        _sliders['learning_rate'] = mo.ui.slider(start=0.001, stop=0.5, step=0.001, value=best_params['learning_rate'], label="⚡ Швидкість навчання (learning_rate)", show_value=True)
    if 'subsample' in best_params:
        _sliders['subsample'] = mo.ui.slider(start=0.3, stop=1.0, step=0.05, value=best_params['subsample'], label="🎲 Частка рядків (subsample)", show_value=True)
    if 'colsample_bytree' in best_params:
        _sliders['colsample_bytree'] = mo.ui.slider(start=0.3, stop=1.0, step=0.05, value=best_params['colsample_bytree'], label="🎲 Частка колонок (colsample_bytree)", show_value=True)
    if 'min_child_weight' in best_params:
        _sliders['min_child_weight'] = mo.ui.slider(start=1, stop=20, step=1, value=best_params['min_child_weight'], label="👶 Мін. вага нащадка (min_child_weight)", show_value=True)
    if 'gamma' in best_params:
        _sliders['gamma'] = mo.ui.slider(start=0.0, stop=5.0, step=0.1, value=best_params['gamma'], label="✂️ Гамма (gamma - згладжування)", show_value=True)

    # Слайдери для Лінійних моделей та EBM
    if 'C' in best_params:
        _sliders['C'] = mo.ui.slider(start=0.0001, stop=10.0, step=0.1, value=best_params['C'], label="⚖️ Регуляризація (C)", show_value=True)
    if 'max_iter' in best_params:
        _sliders['max_iter'] = mo.ui.slider(start=100, stop=2000, step=50, value=best_params['max_iter'], label="🔄 Макс. ітерацій (max_iter)", show_value=True)
    if 'max_bins' in best_params:
        _sliders['max_bins'] = mo.ui.slider(start=64, stop=1024, step=64, value=best_params['max_bins'], label="🗑️ Макс. кошиків (max_bins)", show_value=True)

    manual_sliders = mo.ui.dictionary(_sliders)
    test_manual_btn = mo.ui.run_button(label="🧪 Оцінити ручні параметри (3-Fold CV)", kind="warn")

    mo.output.append(mo.center(mo.md(f"✅ **Елементи Режиму Архітектора успішно згенеровано!** (Доступно {len(_sliders)} повзунків)")))
    return manual_sliders, override_switch, test_manual_btn


@app.cell
def execute_manual_override(
    AdaBoostClassifier,
    ColumnTransformer,
    ExplainableBoostingClassifier,
    ExtraTreesClassifier,
    GLOBAL_SEED,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ImbPipeline,
    LGBMClassifier,
    LogisticRegression,
    OneHotEncoder,
    RandomForestClassifier,
    SMOTE,
    SelectFromModel,
    SimpleImputer,
    StandardScaler,
    StratifiedKFold,
    XGBClassifier,
    X_train,
    balanced_accuracy_score,
    best_params,
    categorical_features,
    champion_selector,
    final_tuned_pipeline,
    lgbm_kwargs,
    manual_sliders,
    mo,
    np,
    numeric_features,
    override_switch,
    pd,
    test_manual_btn,
    xgb_kwargs,
    y_train,
):
    mo.stop(not final_tuned_pipeline, mo.md("⏳"))

    _selected_name = champion_selector.value

    # За замовчуванням (Auto) використовуємо результат Optuna
    ultimate_params = best_params.copy()
    ultimate_pipeline = final_tuned_pipeline

    if not override_switch.value:
        _ui_content = mo.md(f"<br>🤖 **Режим Optuna (Auto):** Використовуються найкращі параметри, знайдені машиною.")
    else:
        _manual_result_ui = mo.md("")

        if test_manual_btn.value:
            with mo.status.spinner("Оцінюємо ручні параметри (Stratified 3-Fold CV)..."):

                # 1. Застосовуємо ручні зміни до параметрів
                manual_params = best_params.copy()
                for k, v in manual_sliders.value.items():
                    manual_params[k] = v

                manual_params['random_state'] = GLOBAL_SEED

                # 2. Безпечні конфігурації для оцінки (n_jobs=1 для внутрішніх моделей)
                _eval_xgb_kwargs = xgb_kwargs.copy()
                _eval_xgb_kwargs['n_jobs'] = 1
                _eval_lgbm_kwargs = lgbm_kwargs.copy()
                _eval_lgbm_kwargs['n_jobs'] = 1

                # 3. Універсальна збірка моделі для ОЦІНКИ
                if "XGBoost" in _selected_name:
                    manual_params.update({'eval_metric': 'logloss', **_eval_xgb_kwargs})
                    eval_model = XGBClassifier(**manual_params)
                elif "LightGBM" in _selected_name:
                    manual_params.update({**_eval_lgbm_kwargs})
                    eval_model = LGBMClassifier(**manual_params)
                elif "Random Forest" in _selected_name:
                    manual_params.update({'n_jobs': 1})
                    eval_model = RandomForestClassifier(**manual_params)
                elif "Extra Trees" in _selected_name:
                    manual_params.update({'n_jobs': 1})
                    eval_model = ExtraTreesClassifier(**manual_params)
                elif "Logistic Regression" in _selected_name:
                    manual_params.update({'solver': 'saga', 'n_jobs': 1})
                    eval_model = LogisticRegression(**manual_params)
                elif "AdaBoost" in _selected_name:
                    eval_model = AdaBoostClassifier(**manual_params)
                elif "HistGradient" in _selected_name:
                    eval_model = HistGradientBoostingClassifier(**manual_params)
                elif "Gradient Boosting (Classic)" in _selected_name:
                    eval_model = GradientBoostingClassifier(**manual_params)
                elif "Explainable Boosting" in _selected_name:
                    manual_params.update({'n_jobs': 1})
                    eval_model = ExplainableBoostingClassifier(**manual_params)

                # 4. Швидкий препроцесинг
                _num_tf = ImbPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
                _cat_tf = ImbPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
                _prep = ColumnTransformer(transformers=[('num', _num_tf, numeric_features), ('cat', _cat_tf, categorical_features)], remainder='drop')

                _X_train_prep = _prep.fit_transform(X_train)
                _feat_sel = SelectFromModel(XGBClassifier(n_estimators=30, random_state=GLOBAL_SEED, **_eval_xgb_kwargs), threshold='median')
                _X_train_sel = _feat_sel.fit_transform(_X_train_prep, y_train)

                _X_train_opt = pd.DataFrame(_X_train_sel)
                _y_train_opt = y_train.reset_index(drop=True)

                # 5. Ручна швидка крос-валідація
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=GLOBAL_SEED)
                scores = []

                for train_idx, val_idx in cv.split(_X_train_opt, _y_train_opt):
                    X_tr = _X_train_opt.iloc[train_idx].copy()
                    X_val = _X_train_opt.iloc[val_idx].copy()
                    y_tr = _y_train_opt.iloc[train_idx].copy()
                    y_val = _y_train_opt.iloc[val_idx].copy()

                    X_tr_smote, y_tr_smote = SMOTE(random_state=GLOBAL_SEED).fit_resample(X_tr, y_tr)
                    eval_model.fit(X_tr_smote, y_tr_smote)
                    _y_pred = eval_model.predict(X_val)
                    scores.append(balanced_accuracy_score(y_val, _y_pred))

                manual_score = np.mean(scores)

                # 6. Фінальна збірка моделі для ПРОДАКШЕНУ (на всю міць n_jobs=-1)
                _prod_params = manual_params.copy()
                if "XGBoost" in _selected_name:
                    _prod_params.update({'eval_metric': 'logloss', 'n_jobs': -1, **xgb_kwargs})
                    prod_model = XGBClassifier(**_prod_params)
                elif "LightGBM" in _selected_name:
                    _prod_params.update({'n_jobs': -1, **lgbm_kwargs})
                    prod_model = LGBMClassifier(**_prod_params)
                elif "Random Forest" in _selected_name or "Extra Trees" in _selected_name or "Explainable Boosting" in _selected_name or "Logistic Regression" in _selected_name:
                    _prod_params.update({'n_jobs': -1})
                    prod_model = eval(f"{_selected_name.split(' ')[0]}Classifier")(**_prod_params) if "Logistic" not in _selected_name else LogisticRegression(**_prod_params)
                else:
                    prod_model = eval_model # Для алгоритмів без підтримки n_jobs

                manual_pipeline = ImbPipeline([
                    ('preprocessor', _prep),
                    ('feature_selection', _feat_sel),
                    ('smote', SMOTE(random_state=GLOBAL_SEED)),
                    ('classifier', prod_model)
                ])
                manual_pipeline.fit(X_train, y_train)

                # 🚀 ПЕРЕХОПЛЕННЯ УПРАВЛІННЯ!
                ultimate_params = _prod_params
                ultimate_pipeline = manual_pipeline

                _res_text = f"⚖️ **Результат ручного налаштування:** `Balanced Acc = {manual_score:.5f}`. Ультимативний пайплайн оновлено!"
                _manual_result_ui = mo.md(f"<div style='padding: 10px; border-left: 4px solid #f59e0b; background: rgba(128,128,128,0.1); margin-top: 15px;'>{_res_text}</div>")

        # Розпаковуємо елементи словника, щоб вони вишикувалися вертикально
        # і не відображалися як JSON-дерево
        _sliders_ui = mo.vstack(list(manual_sliders.values()), align="stretch")

        _ui_content = mo.vstack([
            mo.md("🛠️ **Налаштування архітектури:**"),
            _sliders_ui,
            mo.md("<br>"),
            test_manual_btn,
            _manual_result_ui
        ])

    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    _card = mo.md(f"""
    <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-top: 15px;">
        <h3 style="margin-top: 0; color: #f59e0b; display: flex; align-items: center; gap: 10px;">
            🧑‍💻 Режим Архітектора (Human-in-the-Loop) {override_switch}
        </h3>
        <p>Машина сліпо женеться за тисячними долями метрики і може перенавчити модель. Увімкніть перемикач, щоб примусово зрізати глибину дерев або зменшити швидкість навчання. <i>Слайдери ініціалізовані найкращими параметрами Optuna.</i></p>
        {_ui_content}
    </div>
    """)

    mo.output.append(_card)
    return (ultimate_pipeline,)


@app.cell(hide_code=True)
def header_thresholds(mo):
    mo.md("""
    <h2 align="center"><b>🎯 5. Калібрування порогів <i>(Cost-Sensitive Thresholding)</i></b></h2>
    """)
    return


@app.cell
def threshold_optimization(
    X_valid,
    balanced_accuracy_score,
    champion_selector,
    classification_report,
    confusion_matrix,
    f1_score,
    go,
    mo,
    np,
    pd,
    px,
    style_dataframe,
    ultimate_pipeline,
    y_valid,
):
    mo.stop(not ultimate_pipeline, mo.md("⏳ **Очікування:** Спочатку завершіть оптимізацію та налаштування архітектури у Блоках 3 та 4."))

    _selected_name = champion_selector.value

    with mo.status.spinner("Аналіз розподілу ймовірностей на валідаційній вибірці..."):
        # 1. Витягуємо ймовірності з нашого Ультимативного Пайплайну
        if hasattr(ultimate_pipeline, "predict_proba"):
            _y_pred_proba = ultimate_pipeline.predict_proba(X_valid)[:, 1]
        elif hasattr(ultimate_pipeline, "decision_function"):
            _dfunc = ultimate_pipeline.decision_function(X_valid)
            _y_pred_proba = 1 / (1 + np.exp(-_dfunc))
        else:
            _y_pred_proba = ultimate_pipeline.predict(X_valid)

        # 2. Перебираємо 100 порогів для пошуку ідеального балансу
        _thresholds = np.linspace(0.01, 0.99, 100)
        _bal_acc_scores = [balanced_accuracy_score(y_valid, (_y_pred_proba >= t).astype(int)) for t in _thresholds]
        _f1_scores = [f1_score(y_valid, (_y_pred_proba >= t).astype(int), zero_division=0) for t in _thresholds]

        # 3. Максимізуємо Balanced Accuracy (Головна метрика змагання)
        _opt_idx = np.argmax(_bal_acc_scores)
        optimal_threshold = _thresholds[_opt_idx]
        _y_pred_opt = (_y_pred_proba >= optimal_threshold).astype(int)

    _theme = mo.app_meta().theme
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _text = "white" if _theme == "dark" else "#1f2937"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    # =========================================================================
    # 📈 ГРАФІК 1: ОПТИМІЗАЦІЯ ПОРОГУ
    # =========================================================================
    _fig_pr = go.Figure()
    _fig_pr.add_trace(go.Scatter(
        x=_thresholds, y=_bal_acc_scores, mode='lines', name='Balanced Accuracy',
        line=dict(color='#10b981', width=3),
        hovertemplate="<b>Поріг:</b> %{x:.3f}<br><b>Bal. Acc:</b> %{y:.3f}<extra></extra>"
    ))
    _fig_pr.add_trace(go.Scatter(
        x=_thresholds, y=_f1_scores, mode='lines', name='F1-Міра',
        line=dict(color='#3b82f6', width=2, dash='dash'),
        hovertemplate="<b>Поріг:</b> %{x:.3f}<br><b>F1-Міра:</b> %{y:.3f}<extra></extra>"
    ))

    _fig_pr.add_vline(
        x=optimal_threshold, line_dash="dash", line_color="#ef4444",
        annotation_text=f" Опт. Поріг: {optimal_threshold:.2f} ",
        annotation_position="top", annotation_bgcolor="#ef4444", annotation_font_color="white",
        annotation_borderpad=4
    )

    _fig_pr.update_layout(
        title=dict(text="<b>Оптимізація порогу прийняття рішення</b>", x=0.5, xanchor="center"),
        xaxis_title="Поріг класифікації (Threshold)", yaxis_title="Значення метрики",
        template=_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=40), height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # =========================================================================
    # 🧩 ГРАФІК 2: МАТРИЦЯ ПОМИЛОК
    # =========================================================================
    _cm = confusion_matrix(y_valid, _y_pred_opt)
    _colorscale = 'Blues' if _theme == 'light' else 'ice'

    _fig_cm = px.imshow(
        _cm, text_auto=True, color_continuous_scale=_colorscale,
        labels=dict(x="Прогноз моделі", y="Фактичний клас", color="Кількість"),
        x=['Залишився (0)', 'Відтік (1)'], y=['Залишився (0)', 'Відтік (1)'],
        template=_template
    )
    _fig_cm.update_layout(
        title=dict(text="<b>Матриця помилок (Оптимізована)</b>", x=0.5, xanchor="center"),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False, margin=dict(l=20, r=20, t=60, b=40), height=350
    )

    # =========================================================================
    # 📝 ЗВІТ КЛАСИФІКАЦІЇ (Бронебійна таблиця)
    # =========================================================================
    _report_dict = classification_report(
        y_valid, _y_pred_opt, target_names=['Залишився (0)', 'Відтік (1)'], output_dict=True, zero_division=0
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

    _html_table = style_dataframe(_formatted_df, show_index=True, text_align="center")

    _report_ui = mo.Html(f"""
    <div style='background: rgba(128,128,128,0.05); border: 1px solid {_border}; padding: 15px; border-radius: 8px; margin-top: 15px;'>
        <b style='color: {_text};'>📋 ФІНАЛЬНІ МЕТРИКИ КЛАСИФІКАЦІЇ:</b><br><br>
        {_html_table}
    </div>
    """)

    left_column = mo.vstack([
        mo.ui.plotly(_fig_pr),
        mo.md(f"<div style='text-align: center; margin-top: -10px; font-size: 1.05em;'>🎯 Ідеальний поріг для Balanced Accuracy: <b>{optimal_threshold:.3f}</b></div>"),
        mo.center(_report_ui)
    ], align="stretch")

    right_column = mo.vstack([
        mo.ui.plotly(_fig_cm),
        mo.md(f"""
        <div style='padding: 15px; border: 1px solid {_border}; border-radius: 8px; background: rgba(128,128,128,0.05); margin-top: 15px; font-size: 0.95em;'>
        <b>💡 Tech Lead Insight (Помилки 1 та 2 роду):</b><br><br>
        • <b>Хибнопозитивні (FP):</b> Клієнт би залишився, але ми дали йому бонус на утримання (втрата маржі).<br>
        • <b>Хибнонегативні (FN):</b> Клієнт пішов, бо ми не помітили його ризику (втрата довічного прибутку LTV).<br><br>
        Знижуючи поріг рішення, ми "виловлюємо" більше відтоку (зменшуємо FN), але ризикуємо роздати зайві бонуси лояльним клієнтам (збільшуємо FP). Знайдений поріг <code>{optimal_threshold:.3f}</code> є математично оптимальним!
        </div>
        """)
    ], align="stretch")

    side_by_side_layout = mo.hstack([left_column, right_column], widths=[3, 2], justify="space-between", align="start")

    _css_no_scroll = mo.md('<div class="thresh-noscroll"></div><style>marimo-cell-output:has(.thresh-noscroll),.output-area:has(.thresh-noscroll){max-height: none !important; overflow-y: visible !important;}</style>')

    mo.output.append(mo.vstack([
        _css_no_scroll,
        mo.md(f"<h3 style='text-align: center; margin-top: 10px;'>⚖️ Калібрування порогів: {_selected_name}</h3>"),
        side_by_side_layout
    ]))
    return (optimal_threshold,)


@app.cell(hide_code=True)
def header_xai(mo):
    mo.md("""
    <h2 align='center'><b>🕵️‍♂️ 6. Розпакування 'чорної скриньки' <i>(SHAP Values)</i></b></h2>
    """)
    return


@app.cell
def shap_execution(
    GLOBAL_SEED,
    X_train,
    champion_selector,
    mo,
    pd,
    plt,
    shap,
    shap_tree,
    ultimate_pipeline,
):
    mo.stop(not ultimate_pipeline, mo.md("⏳ Очікування ультимативного пайплайну..."))
    _selected_name = champion_selector.value

    # Витягуємо фінальну модель з пайплайну
    _model = ultimate_pipeline.named_steps['classifier']
    _is_tree = any(kw in _selected_name for kw in ["Forest", "XGBoost", "LightGBM", "Gradient", "Tree"])

    if not _is_tree:
        mo.output.append(mo.md(f"⚠️ Модель `{_selected_name}` не підтримує швидкий SHAP TreeExplainer. Оберіть алгоритм на основі дерев."))
    else:
        with mo.status.spinner("Квантова пояснюваність:<br/>Обчислюємо вектори Шеплі..."):
            # Проганяємо сирі дані через препроцесинг та відбір ознак
            _X_transformed = ultimate_pipeline.named_steps['preprocessor'].transform(X_train)
            _X_selected = ultimate_pipeline.named_steps['feature_selection'].transform(_X_transformed)

            # Витягуємо РЕАЛЬНІ назви ознак, що вижили після SelectFromModel
            try:
                _all_feats = ultimate_pipeline.named_steps['preprocessor'].get_feature_names_out()
                _mask = ultimate_pipeline.named_steps['feature_selection'].get_support()
                _feat_names = [f.replace('num__', '').replace('cat__', '') for f in _all_feats[_mask]]
            except Exception:
                _feat_names = [f"Feature {i}" for i in range(_X_selected.shape[1])]

            # Безпечне перейменування без генерації NaN
            if isinstance(_X_selected, pd.DataFrame):
                _X_sample = _X_selected.sample(n=min(500, _X_selected.shape[0]), random_state=GLOBAL_SEED)
                _X_sample.columns = _feat_names
            else:
                _X_sample = pd.DataFrame(_X_selected, columns=_feat_names).sample(n=min(500, _X_selected.shape[0]), random_state=GLOBAL_SEED)

            # Примусово конвертуємо в float, щоб SHAP міг застосувати кольорову шкалу (High/Low)
            _X_sample = _X_sample.apply(pd.to_numeric, errors='coerce').astype(float)

            # 🛠️ Патч для сумісності нових версій XGBoost та SHAP
            _orig_decode = getattr(shap_tree, "decode_ubjson_buffer", None)
            def _clean_base_score(_dict):
                try:
                    _bs = _dict.get("learner", {}).get("learner_model_param", {}).get("base_score")
                    if isinstance(_bs, str) and "[" in _bs:
                        _dict["learner"]["learner_model_param"]["base_score"] = _bs.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
                except Exception: pass
                return _dict

            if _orig_decode:
                shap_tree.decode_ubjson_buffer = lambda *args, **kwargs: _clean_base_score(_orig_decode(*args, **kwargs))

            try:
                _explainer = shap.TreeExplainer(_model)
                _shap_values = _explainer.shap_values(_X_sample, check_additivity=False)

                # (СПИСКИ LightGBM/RF):
                # LightGBM часто повертає список впливів [на Клас 0, на Клас 1].
                # Нас цікавить лише вплив на відтік (Клас 1).
                if isinstance(_shap_values, list):
                    _shap_values = _shap_values[1]

            finally:
                if _orig_decode: shap_tree.decode_ubjson_buffer = _orig_decode

            _theme = mo.app_meta().theme
            _style = 'dark_background' if _theme == 'dark' else 'default'
            _text_color = "white" if _theme == "dark" else "#1f2937"

            with plt.style.context(_style):
                plt.rcParams['savefig.transparent'] = True
                _fig, _ax = plt.subplots(figsize=(10, 6))

                shap.summary_plot(_shap_values, _X_sample, show=False)

                _ax.set_title(f"Вплив ознак на рішення моделі ({_selected_name})", color=_text_color, fontsize=14, fontweight='bold', pad=20)
                _ax.set_xlabel("Значення SHAP (Вплив на прогноз відтоку)", color=_text_color)
                _fig.patch.set_facecolor('none')
                _ax.set_facecolor('none')
                _ax.tick_params(colors=_text_color)
                _ax.xaxis.label.set_color(_text_color)

                if len(_fig.axes) > 1:
                    _cbar_ax = _fig.axes[-1]
                    _cbar_ax.set_ylabel("Фактичне значення ознаки", rotation=270, labelpad=15)
                    _cbar_ax.yaxis.label.set_color(_text_color)
                    _cbar_ax.tick_params(colors=_text_color)

                _plot_html = mo.as_html(_fig)
                plt.close(_fig)

        _css_no_scroll = mo.md('<div class="xai-noscroll"></div><style>marimo-cell-output:has(.xai-noscroll),.output-area:has(.xai-noscroll) {max-height: 9999px !important; overflow: visible !important; overflow-y: visible !important;}</style>')
        mo.output.append(mo.vstack([_css_no_scroll, mo.center(_plot_html), mo.md("> **💡 Tech Lead Insight:** Чим сильніше червоні крапки зміщені вправо (додатний SHAP), тим сильніше це значення збільшує ймовірність відтоку клієнта.")]))
    return


@app.cell(hide_code=True)
def header_submission(mo):
    mo.md("""
    <h2 align='center'><b>🎫 7. Фінальний Сабміт (Production Score)</b></h2>
    """)
    return


@app.cell(hide_code=True)
def create_submission(
    X_train_full,
    clear_vram,
    clone,
    device,
    df_test,
    mo,
    optimal_threshold,
    pd,
    ultimate_pipeline,
    y_train_full,
):
    # 🛠️ ВИПРАВЛЕНО (критичний баг): раніше ця клітинка ІГНОРУВАЛА весь пайплайн
    # (Лідерборд → вибір чемпіона → Optuna → HITL-корекція) і наглухо будувала
    # НОВИЙ, захардкоджений XGBClassifier(**best_params). Якщо чемпіоном ставав
    # не XGBoost (LightGBM/RandomForest/EBM/...), це впало б з TypeError через
    # несумісні параметри. Тепер клонуємо АРХІТЕКТУРУ саме того пайплайна,
    # який реально обрав користувач (ultimate_pipeline), і довчаємо її на 100% даних.
    with mo.status.spinner(title="🔥 Перенавчання Ультимативного Пайплайну на ВСІХ даних..."):
        final_pipeline = clone(ultimate_pipeline)
        final_pipeline.fit(X_train_full, y_train_full)
        clear_vram(device)

        if 'index' in df_test.columns:
            X_test_final = df_test.drop(columns=['index'])
            test_index = df_test['index']
        else:
            X_test_final = df_test
            test_index = df_test.index

        # 🛠️ ВИПРАВЛЕНО: раніше .predict() використовував дефолтний поріг 0.5,
        # ігноруючи весь розділ 5 "Калібрування порогів". Тепер застосовуємо
        # знайдений optimal_threshold — саме заради цього він і рахувався.
        if hasattr(final_pipeline, "predict_proba"):
            _test_proba = final_pipeline.predict_proba(X_test_final)[:, 1]
            test_preds = (_test_proba >= optimal_threshold).astype(int)
        else:
            test_preds = final_pipeline.predict(X_test_final)

        submission = pd.DataFrame({
            'index': test_index,
            'y': test_preds
        })

        submission_path = 'submission_ultimate.csv'
        submission.to_csv(submission_path, index=False)

    # ==========================================
    # 🎨 ВІЗУАЛІЗАЦІЯ ТА UI ВИВІД (IRONKAGE Pattern)
    # ==========================================
    _theme = mo.app_meta().theme
    _bg = "#1f2937" if _theme == "dark" else "#f9fafb"
    _border = "#4b5563" if _theme == "dark" else "#e5e7eb"

    # Читаємо згенерований файл для кнопки завантаження
    with open(submission_path, "rb") as f:
        _csv_data = f.read()

    # Створюємо нативну кнопку Marimo для скачування файлу
    _download_btn = mo.download(
        data=_csv_data,
        filename=submission_path,
        label="📥 Завантажити submission_ultimate.csv",
        mimetype="text/csv"
    )

    # Формуємо красиву картку з результатами
    _ui_card = mo.md(f"""
    <div style="padding: 20px; border: 1px solid {_border}; border-radius: 8px; background-color: {_bg}; margin-top: 15px; margin-bottom: 15px;">
        <h3 style="margin-top: 0;">✅ Фінальний етап успішно завершено!</h3>
        <p style="margin-bottom: 8px;"><b>1. Модель:</b> Пайплайн <code>{type(final_pipeline.named_steps['classifier']).__name__}</code> (обраний вами у розділах 2–4) перенавчено на всіх <b>{X_train_full.shape[0]}</b> рядках тренувального набору.</p>
        <p style="margin-bottom: 8px;"><b>2. Поріг:</b> Застосовано калібрований поріг <code>{optimal_threshold:.3f}</code> з розділу 5, а не дефолтний 0.5.</p>
        <p style="margin-bottom: 15px;"><b>3. Прогноз:</b> Згенеровано прогнози для <b>{X_test_final.shape[0]}</b> клієнтів із валідаційного набору (як вимагає формат змагання).</p>
        {_download_btn}
    </div>
    """)

    mo.output.append(_ui_card)
    return


@app.cell(hide_code=True)
def header_data(mo):
    mo.md("""
    <h2 align='center'><b> 📊 8. Інтерактивна візуалізація та аналітика</b></h2>
    """)
    return


@app.cell
def final_diagnostics(
    mo,
    plot_optimization_history,
    plot_param_importances,
    study,
):
    mo.stop(study is None, mo.center(mo.md("⏳ **Очікування:** Спочатку запустіть Optuna Marathon у попередньому розділі.")))

    _theme = mo.app_meta().theme
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"
    _text_color = "white" if _theme == "dark" else "#1f2937"

    # Графік історії
    fig_history = plot_optimization_history(study)
    fig_history.update_layout(
        title=dict(text="<b>📈 Історія оптимізації (Balanced Acc)</b>", x=0.5, font=dict(color=_text_color)),
        template=_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Графік гіперпараметрів
    try:
        fig_params = plot_param_importances(study)
        fig_params.update_layout(
            title=dict(text="<b>🎛️ Вплив гіперпараметрів</b>", x=0.5, font=dict(color=_text_color)),
            template=_template,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    except Exception:
        fig_params = mo.md("")

    _css_no_scroll = mo.md('<div class="diag-noscroll"></div><style>marimo-cell-output:has(.diag-noscroll),.output-area:has(.diag-noscroll){max-height: none !important; overflow-y: visible !important;}</style>')

    mo.output.append(mo.vstack([
        _css_no_scroll,
        mo.center(mo.md("### 📊 Діагностика Optuna-марафону")),
        mo.ui.plotly(fig_history),
        mo.ui.plotly(fig_params) if hasattr(fig_params, 'update_layout') else fig_params
    ]))
    return


if __name__ == "__main__":
    app.run()
