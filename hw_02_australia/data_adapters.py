# hw_02_classification/data_adapters.py

import pandas as pd
import numpy as np

def get_rain_australia_mock(save_path: str):
    """
    Fallback-адаптер: Створює синтетичний метео-набір даних Австралії,
    якщо відсутній інтернет або Kaggle API недоступне (Zero-Trust Data Source).
    """
    np.random.seed(42)
    n_samples = 5000

    dates = pd.date_range(start='2015-01-01', periods=n_samples, freq='h')
    locations = np.random.choice(['Sydney', 'Melbourne', 'Brisbane', 'Perth'], n_samples)

    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Location': locations,
        'MinTemp': np.random.normal(12, 5, n_samples),
        'MaxTemp': np.random.normal(24, 6, n_samples),
        'Rainfall': np.random.exponential(2, n_samples),
        'WindGustDir': np.random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'], n_samples),
        'WindDir9am': np.random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'], n_samples),
        'WindDir3pm': np.random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'], n_samples),
        'WindGustSpeed': np.random.normal(40, 15, n_samples),
        'WindSpeed9am': np.random.normal(15, 10, n_samples),
        'WindSpeed3pm': np.random.normal(20, 10, n_samples),
        'Humidity9am': np.random.normal(60, 15, n_samples),
        'Humidity3pm': np.random.normal(50, 20, n_samples),
        'Pressure9am': np.random.normal(1015, 5, n_samples),
        'Pressure3pm': np.random.normal(1013, 5, n_samples),
        'Cloud9am': np.random.randint(0, 9, n_samples),
        'Cloud3pm': np.random.randint(0, 9, n_samples),
        'Temp9am': np.random.normal(16, 5, n_samples),
        'Temp3pm': np.random.normal(22, 6, n_samples),
        'RainToday': np.random.choice(['Yes', 'No'], n_samples, p=[0.22, 0.78]),
        'RainTomorrow': np.random.choice(['Yes', 'No'], n_samples, p=[0.22, 0.78])
    })

    df.to_csv(save_path, index=False)
    return save_path
