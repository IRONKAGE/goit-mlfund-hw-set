import pandas as pd
import numpy as np
import os

# Ініціалізація глобального seed з урахуванням змінних середовища (.env)
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", 42))

def get_autos_mock(save_path: str):
    """Синтетичний генератор набору даних Autos (Automobile)"""
    np.random.seed(GLOBAL_SEED)
    n = 205

    data = {
        "make": np.random.choice(["alfa-romero", "audi", "bmw", "toyota", "honda", "mazda", "nissan", "peugeot", "volkswagen", "volvo"], n),
        "fuel-type": np.random.choice(["gas", "diesel"], n, p=[0.8, 0.2]),
        "aspiration": np.random.choice(["std", "turbo"], n, p=[0.8, 0.2]),
        "num-of-doors": np.random.choice(["two", "four", "?"], n, p=[0.4, 0.58, 0.02]),
        "body-style": np.random.choice(["hardtop", "wagon", "sedan", "hatchback", "convertible"], n),
        "drive-wheels": np.random.choice(["4wd", "fwd", "rwd"], n),
        "engine-location": np.random.choice(["front", "rear"], n, p=[0.98, 0.02]),
        "wheel-base": np.random.uniform(86.6, 120.9, n),
        "length": np.random.uniform(141.1, 208.1, n),
        "width": np.random.uniform(60.3, 72.3, n),
        "height": np.random.uniform(47.8, 59.8, n),
        "curb-weight": np.random.uniform(1488, 4066, n),
        "engine-type": np.random.choice(["dohc", "ohcv", "ohc", "l", "rotor", "ohcf", "dohcv"], n),
        "num-of-cylinders": np.random.choice(["eight", "five", "four", "six", "three", "twelve", "two"], n),
        "engine-size": np.random.uniform(61, 326, n),
        "fuel-system": np.random.choice(["1bbl", "2bbl", "4bbl", "idi", "mfi", "mpfi", "spdi", "spfi"], n),
        "bore": np.random.uniform(2.54, 3.94, n),
        "stroke": np.random.uniform(2.07, 4.17, n),
        "compression-ratio": np.random.uniform(7.0, 23.0, n),
        "horsepower": np.random.uniform(48, 288, n),
        "peak-rpm": np.random.uniform(4150, 6600, n),
        "city-mpg": np.random.uniform(13, 49, n),
        "highway-mpg": np.random.uniform(16, 54, n),
        "price": np.random.uniform(5118, 45400, n)
    }

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def get_cardekho_mock(save_path: str):
    """Синтетичний генератор набору даних CarDekho"""
    np.random.seed(GLOBAL_SEED)
    n = 301

    data = {
        "Car_Name": np.random.choice(["ritz", "sx4", "ciaz", "wagon r", "swift", "vitara brezza", "s cross", "alto 800"], n),
        "Year": np.random.randint(2003, 2018, n),
        "Selling_Price": np.random.uniform(0.1, 35.0, n),
        "Present_Price": np.random.uniform(0.3, 92.6, n),
        "Driven_kms": np.random.randint(500, 500000, n),
        "Fuel_Type": np.random.choice(["Petrol", "Diesel", "CNG"], n, p=[0.7, 0.28, 0.02]),
        "Seller_Type": np.random.choice(["Dealer", "Individual"], n),
        "Transmission": np.random.choice(["Manual", "Automatic"], n, p=[0.8, 0.2]),
        "Owner": np.random.choice([0, 1, 3], n, p=[0.9, 0.08, 0.02])
    }

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df
