import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title_head_fp(mo):
    mo.md("""
    <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
        🏆 Фінальний проєкт (Kaggle): Прогнозування відтоку клієнтів <i>(Customer Churn)</i>
    </div>

    <h3 align="center"><b><u>Пайплайн</u>: ColumnTransformer ➔ Feature Selection ➔ SMOTE ➔ Optuna Marathon ➔ XGBoost ➔ Plotly Viz</b></h3>

    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Machine Learning: Fundamentals and Applications [08.2026]</i></p>
    """)
    return


@app.cell(hide_code=True)
def configure_dependencies():
    import os
    import sys
    import warnings

    # 🤫 Глушимо системні попередження для чистоти UI
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

    # 📊 Data Science & UI
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import marimo as mo

    # 🤖 Machine Learning & MLOps
    import mlflow
    import optuna

    # ⚖️ Балансування класів (Imbalanced-Learn)
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    # 🧠 Scikit-Learn інфраструктура
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import balanced_accuracy_score

    # 🌲 Бустинги
    from xgboost import XGBClassifier

    # 📈 Візуалізація Optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances

    # ⚡ Оптимізація пам'яті Pandas (Copy-on-Write)
    pd.options.mode.copy_on_write = True

    mo.output.append(mo.center(mo.md("✅ **Бібліотеки, Ядро MLOps та UI-компоненти успішно імпортовано!**")))

    # Експортуємо всі модулі для інших клітинок
    return (
        ColumnTransformer,
        ImbPipeline,
        OrdinalEncoder,
        SMOTE,
        SelectFromModel,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        XGBClassifier,
        clear_vram,
        cross_val_score,
        get_boosting_kwargs,
        get_hardware_config,
        go,
        log_system_info,
        logger,
        logging,
        make_subplots,
        mlflow,
        mo,
        optuna,
        os,
        pd,
        plot_optimization_history,
        plot_param_importances,
        px,
        set_global_seed,
        smart_read_csv,
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
    return


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

        # Перекладаємо конфіги заліза для бустингів (XGBoost)
        xgb_kwargs, lgbm_kwargs = get_boosting_kwargs(device)

        # 3. Налаштування MLflow для фінального змагання
        experiment_name = "fp_kaggle_customer_churn"
        mlflow.set_experiment(experiment_name)

        # 📝 Фіксуємо подію в глобальний аудит-лог
        logger.info(f"✅ Налаштовано експеримент MLflow: {experiment_name}")
    return (device,)


@app.cell(hide_code=True)
def header_data(mo):
    mo.md("""
    <h2 align='center'><b>💽 1. Завантаження даних та Smart EDA</b></h2>
    """)
    return


@app.cell(hide_code=True)
def execute_etl(go, make_subplots, mo, os, smart_read_csv):
    import urllib.request

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

    with mo.status.spinner(title="Завантаження та обробка датасетів Kaggle..."):
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
            text="<b>Аналіз Kaggle Датасету: Відтік Клієнтів</b>",
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
def header_data(mo):
    mo.md("""
    <h2 align='center'><b>🔍 2. Розподіл типів ознак</b></h2>
    """)
    return


@app.cell(hide_code=True)
def _(X_train_full, mo):
    # 🔍 2. Підготовка списків ознак для ColumnTransformer
    numeric_features = X_train_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train_full.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Евристика: якщо всі колонки зчитались як числа, шукаємо приховані категорії (менше 15 унікальних значень)
    if not categorical_features:
        categorical_features = [col for col in X_train_full.columns if X_train_full[col].nunique() < 15]
        numeric_features = [col for col in X_train_full.columns if col not in categorical_features]

    mo.output.append(
        mo.center(
            mo.md(f"✅ **Ознаки успішно класифіковано!**<br>Готово до обробки: (`Числових: {len(numeric_features)} | Категоріальних: {len(categorical_features)}`)")
        )
    )
    return categorical_features, numeric_features


@app.cell(hide_code=True)
def header_data(mo):
    mo.md("""
    <h2 align='center'><b>🧠 3. Глибока оптимізація <i>(Optuna Marathon)</i></b></h2>
    """)
    return


@app.cell(hide_code=True)
def run_optuna_marathon(
    ColumnTransformer,
    ImbPipeline,
    OrdinalEncoder,
    SMOTE,
    SelectFromModel,
    SimpleImputer,
    StandardScaler,
    StratifiedKFold,
    XGBClassifier,
    X_train_full,
    categorical_features,
    cross_val_score,
    device,
    get_boosting_kwargs,
    get_hardware_config,
    logger,
    logging,
    mlflow,
    mo,
    numeric_features,
    optuna,
    pd,
    plot_optimization_history,
    plot_param_importances,
    y_train_full,
):
    # Глушимо зайві логи MLflow
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
    logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

    # 🛠️ ВИПРАВЛЕНО: Прибрали `_` перед device, щоб передати його далі
    _, device_name = get_hardware_config()
    xgb_hw_kwargs, _ = get_boosting_kwargs(device)

    # Гарна назва для UI
    _hw_ui = "CUDA GPU 🟢" if getattr(device, "type", "") == "cuda" else (
        "Apple Silicon (MPS) 🟣" if getattr(device, "type", "") == "mps" else f"{device_name} ⚙️"
    )

    _n_trials = 10 # Кількість ітерацій

    # 🎬 ВІЗУАЛЬНА АНІМАЦІЯ: Ініціалізуємо прогрес-бар
    with mo.status.progress_bar(
        total=_n_trials,
        title="🏆 Марафон Optuna: XGBoost + SMOTE",
        subtitle="⏳ Ініціалізація пайплайну...",
        remove_on_exit=True
    ) as _bar:

        def objective(trial):
            _bar.update(increment=0, subtitle=f"🏃‍♂️ Ітерація {trial.number + 1} з {_n_trials}: Навчання 5-ти фолдів...")

            # 1. Простір пошуку (Регуляризація)
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 600),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 2, 12),
                'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
                'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
                'random_state': 42,
                **xgb_hw_kwargs
            }

            # 2. Збірка монолітного конвеєра
            numeric_transformer = ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = ImbPipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )

            feature_selector = SelectFromModel(
                XGBClassifier(n_estimators=100, random_state=42, **xgb_hw_kwargs),
                threshold='median'
            )

            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('feature_selection', feature_selector),
                ('smote', SMOTE(random_state=42)),
                ('classifier', XGBClassifier(**param))
            ])

            # 3. Валідація
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(
                pipeline, X_train_full, y_train_full,
                cv=cv, scoring='balanced_accuracy',
                n_jobs=1
            ).mean()

            return score

        # Callback для оновлення прогрес-бару
        def _progress_callback(study, trial):
            _bar.update(
                increment=1,
                subtitle=f"🌿 Ітерація {trial.number + 1} з {_n_trials} | Рекорд (Bal. Accuracy): {study.best_value:.5f}"
            )

        # ЗАПУСК МАРАФОНУ
        # 🛠️ ВИПРАВЛЕНО: Прибрали `_` перед study, щоб передати його в інші клітинки
        # (раніше `_study` був приватним для цієї клітинки, тому клітинка 5
        # ловила NameError при спробі побудувати графіки на його основі)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=_n_trials, callbacks=[_progress_callback])

        best_params = study.best_params
        logger.info(f"Optuna знайшла найкращі параметри: {best_params}")

        # ==========================================
        # 💾 ВІДНОВЛЕННЯ ТА ЗБЕРЕЖЕННЯ В MLFLOW
        # ==========================================
        _bar.update(increment=0, subtitle="💾 Збереження найкращої моделі у базу MLflow...")

        # Створюємо фінальний пайплайн
        _final_xgb_params = {**best_params, **xgb_hw_kwargs, 'random_state': 42}

        _final_num_tf = ImbPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        _final_cat_tf = ImbPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        _final_prep = ColumnTransformer(transformers=[('num', _final_num_tf, numeric_features), ('cat', _final_cat_tf, categorical_features)], remainder='drop')
        _final_sel = SelectFromModel(XGBClassifier(n_estimators=100, random_state=42, **xgb_hw_kwargs), threshold='median')

        final_tuned_pipeline = ImbPipeline([
            ('preprocessor', _final_prep),
            ('feature_selection', _final_sel),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(**_final_xgb_params))
        ])

        # Навчаємо на всіх даних
        final_tuned_pipeline.fit(X_train_full, y_train_full)

        with mlflow.start_run(run_name="Ultimate_Pipeline_SMOTE_XGB"):
            mlflow.log_params(best_params)
            mlflow.log_metric("CV_Balanced_Accuracy", study.best_value)
            mlflow.log_metric("Optuna_Trials", _n_trials)

            # Додаємо дозволи для нестандартних класів (Imblearn + XGBoost)
            _trusted_types = [
                "imblearn.over_sampling._smote.base.SMOTE",
                "imblearn.pipeline.Pipeline",
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]

            mlflow.sklearn.log_model(
                final_tuned_pipeline,
                artifact_path="champion_pipeline",
                skops_trusted_types=_trusted_types,
                pip_requirements=["scikit-learn", "xgboost", "imbalanced-learn"]
            )
            _run_id = mlflow.active_run().info.run_id

        # ==========================================
        # 🎨 БЛОК ВІЗУАЛІЗАЦІЇ ТА UX
        # ==========================================
        _fig_history = plot_optimization_history(study)
        try:
            _fig_params = plot_param_importances(study)
        except Exception:
            _fig_params = None

        _theme = mo.app_meta().theme
        _border = "#4b5563" if _theme == "dark" else "#e5e7eb"
        _text_color = "white" if _theme == "dark" else "#1f2937"

        # УКРАЇНІЗАЦІЯ ГРАФІКІВ
        _fig_history.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_text_color),
            title=dict(text="<b>Історія байєсівської оптимізації</b>", x=0.5),
            xaxis_title="Ітерація (Спроба)",
            yaxis_title="Збалансована точність (Balanced Acc)"
        )
        for _trace in _fig_history.data:
            if _trace.x is not None:
                _trace.x = tuple(x + 1 for x in _trace.x)
            if _trace.name == 'Objective Value':
                _trace.name = 'Точність ітерації'
                _trace.hovertemplate = '<b>Ітерація:</b> %{x}<br><b>Bal. Acc:</b> %{y:.5f}<extra></extra>'
                _trace.marker.color = '#3b82f6'
            elif _trace.name == 'Best Value':
                _trace.name = 'Рекорд (Максимальна точність)'
                _trace.hovertemplate = '<b>Ітерація:</b> %{x}<br><b>Рекорд:</b> %{y:.5f}<extra></extra>'
                _trace.line.color = '#10b981'

        if _fig_params:
            _fig_params.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_text_color),
                title=dict(text="<b>Вплив гіперпараметрів XGBoost</b>", x=0.5),
                xaxis_title="Ступінь впливу на результат",
                yaxis_title="Гіперпараметр"
            )
            for _trace in _fig_params.data:
                _trace.hovertemplate = '<b>Гіперпараметр:</b> %{y}<br><b>Вплив:</b> %{x:.3f}<extra></extra>'
                _trace.marker.color = '#8b5cf6'

        # ТАБЛИЦЯ ТА КОМПОНУВАННЯ UI
        _params_df = pd.DataFrame([best_params])
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
                ✅ **Оптимізацію завершено!** Найкраща збалансована точність (Balanced Accuracy): `{study.best_value:.5f}`<br/>
                💎 **Залізо (Engine):** `{_hw_ui}`<br/>
                🗃️ **MLflow:** Пайплайн (разом зі SMOTE та ColumnTransformer) успішно збережено! *(Run ID: `{_run_id}`)*
                """
            ),
            mo.ui.table(_params_df, selection=None),
            _plots_ui
        ])

        mo.output.append(_result_ui)
    return best_params, study, xgb_hw_kwargs


@app.cell(hide_code=True)
def header_data(mo):
    mo.md("""
    <h2 align='center'><b> 🎯 4. Бойове навчання та створення <i>submission.csv</i></b></h2>
    """)
    return


@app.cell(hide_code=True)
def create_submission(
    ColumnTransformer,
    ImbPipeline,
    OrdinalEncoder,
    SMOTE,
    SelectFromModel,
    SimpleImputer,
    StandardScaler,
    XGBClassifier,
    X_train_full,
    best_params,
    categorical_features,
    clear_vram,
    device,
    df_test,
    mo,
    numeric_features,
    pd,
    xgb_hw_kwargs,
    y_train_full,
):
    with mo.status.spinner(title="🔥 Бойове навчання пайплайну на всіх даних..."):
        # Використовуємо знайдені параметри + конфіги заліза
        final_params = {**best_params, **xgb_hw_kwargs, 'random_state': 42}

        numeric_transformer = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = ImbPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

        feature_selector_final = SelectFromModel(
            XGBClassifier(n_estimators=100, random_state=42, **xgb_hw_kwargs),
            threshold='median'
        )

        final_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', feature_selector_final),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(**final_params))
        ])

        # Навчаємо на ВСІХ тренувальних даних
        final_pipeline.fit(X_train_full, y_train_full)
        clear_vram(device)

        if 'index' in df_test.columns:
            X_test_final = df_test.drop(columns=['index'])
            test_index = df_test['index']
        else:
            X_test_final = df_test
            test_index = df_test.index

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
        <p style="margin-bottom: 8px;"><b>1. Навчання:</b> Пайплайн (<code>SMOTE</code> + <code>XGBoost</code>) перенавчено на всіх <b>{X_train_full.shape[0]}</b> рядках тренувального набору.</p>
        <p style="margin-bottom: 15px;"><b>2. Прогноз:</b> Згенеровано прогнози для <b>{X_test_final.shape[0]}</b> клієнтів із валідаційного набору (як вимагає формат змагання).</p>
        {_download_btn}
    </div>
    """)

    mo.output.append(_ui_card)
    return (final_pipeline,)


@app.cell(hide_code=True)
def header_data(mo):
    mo.md("""
    <h2 align='center'><b> 📊 5. Інтерактивна візуалізація та аналітика</b></h2>
    """)
    return


@app.cell(hide_code=True)
def _(
    final_pipeline,
    mo,
    pd,
    plot_optimization_history,
    plot_param_importances,
    px,
    study,
):
    _theme = mo.app_meta().theme
    _template = "plotly_dark" if _theme == "dark" else "plotly_white"

    # 1. Графік історії оптимізації Optuna
    fig_history = plot_optimization_history(study)
    fig_history.update_layout(title="📈 Історія Байєсівської оптимізації (Balanced Accuracy)", template="plotly_dark")

    # 2. Графік важливості гіперпараметрів
    fig_params = plot_param_importances(study)
    fig_params.update_layout(title="🎛️ Вплив гіперпараметрів на результат", template="plotly_dark")

    # 3. Важливість ознак (Feature Importances) від XGBoost
    xgb_model = final_pipeline.named_steps['classifier']
    importances = xgb_model.feature_importances_

    # 🏷️ Дістаємо РЕАЛЬНІ назви ознак, що пройшли ColumnTransformer + SelectFromModel,
    # а не порожні індекси "Feature_i". Якщо з якоїсь причини pipeline не підтримує
    # get_feature_names_out (напр. стара версія sklearn), падаємо назад на індекси.
    try:
        _all_names = final_pipeline.named_steps['preprocessor'].get_feature_names_out()
        _selected_mask = final_pipeline.named_steps['feature_selection'].get_support()
        _feature_labels = [
            name.replace('num__', '').replace('cat__', '')
            for name in _all_names[_selected_mask]
        ]
        if len(_feature_labels) != len(importances):
            raise ValueError("Розмірність назв ознак не збігається з importances")
    except Exception:
        _feature_labels = [f"Feature_{i}" for i in range(len(importances))]

    feat_imp_df = pd.DataFrame({
        'Ознака': _feature_labels,
        'Важливість': importances
    }).sort_values(by='Важливість', ascending=False).head(20)  # Беремо Топ-20

    fig_features = px.bar(
        feat_imp_df,
        x='Важливість',
        y='Ознака',
        orientation='h',
        title="🧬 Топ-20 найважливіших ознак (XGBoost Feature Importances)",
        template=_template,
        color='Важливість',
        color_continuous_scale='Viridis'
    )
    fig_features.update_layout(yaxis={'categoryorder': 'total ascending'})

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

    mo.output.append(
        mo.vstack([
            _css_no_scroll,
            mo.center(mo.md("### 📊 Діагностика Optuna-марафону та фінальної моделі")),
            mo.ui.plotly(fig_history),
            mo.ui.plotly(fig_params),
            mo.ui.plotly(fig_features),
        ])
    )
    return


if __name__ == "__main__":
    app.run()
