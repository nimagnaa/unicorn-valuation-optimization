from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA = PROJECT_ROOT / "data" / "processed" / "features_dataset.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

OUTPUT = PROJECT_ROOT / "outputs" / "metrics"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA)

TARGET = "Valuation_Million"

DROP = [

    TARGET,
    "Company",
    "Company_Key",
    "Country",
    "City",
    "Industry",
    "Date Joined",
    "Select Inverstors",
    "Financial Stage",
    "Latest_Layoff",

    # remove raw leakage
    "Valuation ($B)",
    "Total Raised"

]

existing = [c for c in DROP if c in df.columns]

X = df.drop(columns=existing)

# convert categoricals
X = pd.get_dummies(X, drop_first=True)

# fill missing
X = X.fillna(X.median(numeric_only=True))

y = df[TARGET]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42

)

# ==========================================================
# MODELS
# ==========================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ),

    "LightGBM":
        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ),

    "CatBoost":
        CatBoostRegressor(
            verbose=False,
            random_state=42
        )

}

results = []

print()

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    r2 = r2_score(y_test, pred)

    mape = np.mean(np.abs((y_test-pred)/y_test))*100

    results.append({

        "Model":name,
        "MAE":round(mae,2),
        "RMSE":round(rmse,2),
        "MAPE":round(mape,2),
        "R2":round(r2,4)

    })

    filename = name.lower().replace(" ","_") + ".pkl"

    joblib.dump(
        model,
        MODEL_DIR / filename
    )

# ==========================================================
# SAVE RESULTS
# ==========================================================

results = pd.DataFrame(results)

results = results.sort_values(
    "R2",
    ascending=False
)

results.to_csv(

    OUTPUT / "model_comparison.csv",

    index=False

)

print()

print("=" * 60)

print(results)

print("=" * 60)

print()

print("Best Model")

print(results.iloc[0]["Model"])

print()

print("Saved Models ->", MODEL_DIR)

print("Saved Metrics ->", OUTPUT)

print("=" * 60)