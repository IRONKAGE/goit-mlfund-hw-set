# 1. Єдине джерело істини (Kaggle / Наші власні назви)
UA_COLUMNS = {
    "longitude": "Довгота",
    "latitude": "Широта",
    "housing_median_age": "Медіанний вік житла",
    "total_rooms": "Загальна к-сть кімнат",
    "total_bedrooms": "Загальна к-сть спалень",
    "population": "Населення",
    "households": "К-сть домогосподарств",
    "median_income": "Медіанний дохід",
    "median_house_value": "Медіанна вартість житла",
    "ocean_proximity": "Близькість до океану",
    "Dist_to_LA": "Відстань до Лос-Анджелеса (км)",
    "Dist_to_SF": "Відстань до Сан-Франциско (км)",
    "Dist_to_Hub": "Відстань до найближчого хабу (км)"
}

# 2. Мапа відповідності (Sklearn -> Kaggle)
SKLEARN_MAP = {
    "Longitude": "longitude",
    "Latitude": "latitude",
    "HouseAge": "housing_median_age",
    "AveRooms": "total_rooms",
    "AveBedrms": "total_bedrooms",
    "Population": "population",
    "AveOccup": "households",
    "MedInc": "median_income",
    "MedHouseVal": "median_house_value"
}

# 3. Динамічне злиття (Без дублювання тексту!)
for sk_col, kaggle_col in SKLEARN_MAP.items():
    if kaggle_col in UA_COLUMNS:
        UA_COLUMNS[sk_col] = UA_COLUMNS[kaggle_col]
