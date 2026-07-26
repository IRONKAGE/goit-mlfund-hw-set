import pandas as pd
import numpy as np

def get_salary_mock(save_path: str, is_valid: bool = False):
    """
    Fallback-адаптер: Створює синтетичний набір даних співробітників компанії
    для тестування алгоритму KNN (ДЗ №3).
    """
    np.random.seed(42 if not is_valid else 43)
    n = 8000 if not is_valid else 2000

    df = pd.DataFrame({
        'Name': [f"Employee_{i}" for i in range(n)],
        'Phone_Number': [f"+38050{np.random.randint(1000000, 9999999)}" for _ in range(n)],
        'Experience': np.random.uniform(0, 30, n).round(1),
        'Qualification': np.random.choice(['BSc', 'MSc', 'PhD', 'High School'], n),
        'University': np.random.choice(['KPI', 'KNU', 'Lviv Tech', 'Stanford', 'None'], n),
        'Role': np.random.choice(['Developer', 'Manager', 'Analyst', 'Data Scientist', 'HR'], n),
        'Cert': np.random.choice(['AWS', 'GCP', 'Azure', 'AWS,GCP', 'None'], n),
        'Date_Of_Birth': pd.to_datetime(np.random.choice(pd.date_range('1960-01-01', '2004-01-01'), n)),
        'Hire_Date': pd.to_datetime(np.random.choice(pd.date_range('2010-01-01', '2023-12-31'), n)),
        'Salary': 0
    })

    # Складна нелінійна функція ціноутворення (щоб KNN мав що вивчати)
    df['Salary'] = (
        25000 +
        df['Experience'] ** 1.2 * 1500 +
        (df['Role'] == 'Data Scientist') * 30000 +
        (df['Role'] == 'Developer') * 20000 +
        (df['Qualification'] == 'PhD') * 15000 +
        (df['Cert'].str.contains('AWS')) * 5000 +
        np.random.normal(0, 4500, n)
    )

    # Фактор інфляції у часі (свідомо закладений для тестування TimesFM у кінці)
    years_since_2015 = (df['Hire_Date'].dt.year - 2015)
    df['Salary'] = df['Salary'] * (1 + 0.04 * years_since_2015)

    df['Salary'] = df['Salary'].astype(int)
    df.to_csv(save_path, index=False)
    return save_path
