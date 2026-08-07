from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "regression"
    / "xgboost.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "optimization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "ml_optimization.csv"
)


# =========================================================
# HEADER
# =========================================================

print("=" * 60)
print("ML OPTIMIZATION ENGINE")
print("=" * 60)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print()
print("Dataset Shape :", df.shape)


# =========================================================
# LOAD XGBOOST MODEL
# =========================================================

model = joblib.load(MODEL_PATH)

print()
print("Loaded Model : XGBoost")


# =========================================================
# PREPARE FEATURES
# =========================================================

TARGET = "Expected_Valuation_12M"

DROP_COLUMNS = [

    TARGET,

    "Down_Round",

    "Company",
    "Company_Key",
    "Country",
    "City",
    "Industry",
    "Date Joined",
    "Select Inverstors",
    "Financial Stage",
    "Latest_Layoff",

    "Valuation ($B)",
    "Total Raised"

]

DROP_COLUMNS = [
    c
    for c in DROP_COLUMNS
    if c in df.columns
]


X = df.drop(
    columns=DROP_COLUMNS
)


# =========================================================
# ENCODE CATEGORICAL VARIABLES
# =========================================================

X = pd.get_dummies(
    X,
    drop_first=True
)


# =========================================================
# CLEAN VALUES
# =========================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

X = X.fillna(0)


# =========================================================
# ALIGN WITH TRAINING FEATURES
# =========================================================

if hasattr(model, "feature_names_in_"):

    model_features = list(
        model.feature_names_in_
    )

    X = X.reindex(
        columns=model_features,
        fill_value=0
    )


# =========================================================
# BASE PREDICTION
# =========================================================

df["ML_Predicted_Valuation"] = model.predict(X)


# =========================================================
# OPTIMIZATION FUNCTION
# =========================================================

def optimize_company(row):

    current_valuation = row[
        "ML_Predicted_Valuation"
    ]

    revenue = row.get(
        "Estimated_Revenue_Million",
        0
    )

    burn = row.get(
        "Estimated_Burn_Million",
        0
    )

    runway = row.get(
        "Estimated_Runway_Months",
        0
    )

    capital_utilization = row.get(
        "Capital_Utilization",
        0
    )

    roi = row.get(
        "Estimated_ROI",
        0
    )

    employee_growth = row.get(
        "Employee_Growth_Rate",
        0
    )


    # -----------------------------------------------------
    # Revenue improvement
    # -----------------------------------------------------

    revenue_target = revenue * 1.20


    # -----------------------------------------------------
    # Burn reduction
    # -----------------------------------------------------

    burn_target = burn * 0.82


    # -----------------------------------------------------
    # Runway improvement
    # -----------------------------------------------------

    funding = row.get(
        "Funding_Million",
        0
    )

    if burn_target > 0:

        optimized_runway = (
            funding / burn_target
        ) * 12

    else:

        optimized_runway = runway


    # -----------------------------------------------------
    # Capital utilization
    # -----------------------------------------------------

    optimized_capital_utilization = min(
        capital_utilization * 1.15,
        1.0
    )


    # -----------------------------------------------------
    # ROI
    # -----------------------------------------------------

    optimized_roi = (
        roi * 1.15
        if pd.notna(roi)
        else roi
    )


    # -----------------------------------------------------
    # Valuation improvement
    #
    # This is a scenario estimate, not a new
    # machine-learning prediction.
    # -----------------------------------------------------

    revenue_effect = 0.12

    burn_effect = 0.06

    utilization_effect = 0.05

    optimized_valuation = (
        current_valuation
        * (
            1
            + revenue_effect
            + burn_effect
            + utilization_effect
        )
    )


    # -----------------------------------------------------
    # Down-round indicator
    # -----------------------------------------------------

    current_risk = row.get(
        "Down_Round",
        0
    )

    if (
        burn_target < burn
        and revenue_target > revenue
        and optimized_runway > runway
    ):

        optimized_risk = 0

    else:

        optimized_risk = current_risk


    return pd.Series({

        "Current_Valuation_ML":
            current_valuation,

        "Optimized_Valuation":
            optimized_valuation,

        "Valuation_Change_Percent":
            (
                (
                    optimized_valuation
                    - current_valuation
                )
                / current_valuation
                * 100
            )
            if current_valuation != 0
            else 0,

        "Current_Revenue":
            revenue,

        "Recommended_Revenue":
            revenue_target,

        "Revenue_Change_Percent":
            (
                (
                    revenue_target
                    - revenue
                )
                / revenue
                * 100
            )
            if revenue != 0
            else 0,

        "Current_Burn":
            burn,

        "Recommended_Burn":
            burn_target,

        "Burn_Change_Percent":
            (
                (
                    burn_target
                    - burn
                )
                / burn
                * 100
            )
            if burn != 0
            else 0,

        "Current_Runway":
            runway,

        "Optimized_Runway":
            optimized_runway,

        "Runway_Change_Months":
            optimized_runway - runway,

        "Current_Capital_Utilization":
            capital_utilization,

        "Optimized_Capital_Utilization":
            optimized_capital_utilization,

        "Current_ROI":
            roi,

        "Optimized_ROI":
            optimized_roi,

        "Current_Down_Round":
            current_risk,

        "Optimized_Down_Round":
            optimized_risk,

        "Employee_Growth_Rate":
            employee_growth

    })


# =========================================================
# RUN OPTIMIZATION
# =========================================================

optimization = df.apply(
    optimize_company,
    axis=1
)


# =========================================================
# ADD COMPANY INFORMATION
# =========================================================

company_columns = [

    "Company",
    "Country",
    "City",
    "Industry",
    "Financial Stage"

]

company_columns = [
    c
    for c in company_columns
    if c in df.columns
]


result = pd.concat(
    [
        df[company_columns],
        optimization
    ],
    axis=1
)


# =========================================================
# ROUND RESULTS
# =========================================================

numeric_columns = result.select_dtypes(
    include=np.number
).columns

result[numeric_columns] = result[
    numeric_columns
].round(4)


# =========================================================
# SAVE
# =========================================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("Optimization Completed")

print()
print("Features Generated:")

features = [

    "Current_Valuation_ML",
    "Optimized_Valuation",
    "Valuation_Change_Percent",
    "Current_Revenue",
    "Recommended_Revenue",
    "Current_Burn",
    "Recommended_Burn",
    "Current_Runway",
    "Optimized_Runway",
    "Current_Capital_Utilization",
    "Optimized_Capital_Utilization",
    "Current_ROI",
    "Optimized_ROI",
    "Current_Down_Round",
    "Optimized_Down_Round"

]

for feature in features:

    print("✓", feature)


print()
print("Output Shape :", result.shape)

print()
print("Saved To")
print(OUTPUT_PATH)

print("=" * 60)