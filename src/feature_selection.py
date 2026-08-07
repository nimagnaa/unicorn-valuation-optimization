from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_regression
)

from sklearn.ensemble import RandomForestRegressor

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = PROJECT_ROOT / "data" / "processed" / "features_dataset.csv"

OUTPUT = PROJECT_ROOT / "outputs" / "feature_selection"

OUTPUT.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

df = pd.read_csv(INPUT)

print("\nDataset Shape :", df.shape)

# =====================================================
# TARGET
# =====================================================

TARGET = "Valuation_Million"

# =====================================================
# REMOVE NON NUMERIC COLUMNS
# =====================================================

drop_columns = [

    "Company",
    "Valuation ($B)",
    "Total Raised",
    "Company_Key",
    "Country",
    "City",
    "Industry",
    "Select Inverstors",
    "Date Joined",
    "Financial Stage",
    "Latest_Layoff"

]

existing = [c for c in drop_columns if c in df.columns]

X = df.drop(columns=existing + [TARGET])

X = pd.get_dummies(
    X,
    drop_first=True
)

y = df[TARGET]

# =====================================================
# MISSING VALUES
# =====================================================

X = X.fillna(X.median(numeric_only=True))

# =====================================================
# VARIANCE THRESHOLD
# =====================================================

selector = VarianceThreshold(
    threshold=0.01
)

selector.fit(X)

variance_features = X.columns[
    selector.get_support()
]

X = X[variance_features]

print("\nFeatures After Variance Filter :", X.shape[1])

# =====================================================
# CORRELATION FILTER
# =====================================================

corr = X.corr().abs()

upper = corr.where(

    np.triu(
        np.ones(corr.shape),
        k=1
    ).astype(bool)

)

remove = [

    column

    for column in upper.columns

    if any(upper[column] > 0.95)

]

X = X.drop(columns=remove)

print("Features After Correlation Filter :", X.shape[1])

# =====================================================
# MUTUAL INFORMATION
# =====================================================

mi = mutual_info_regression(
    X,
    y,
    random_state=42
)

mi_df = pd.DataFrame({

    "Feature": X.columns,
    "Mutual_Information": mi

})

mi_df = mi_df.sort_values(

    "Mutual_Information",
    ascending=False

)

# =====================================================
# RANDOM FOREST IMPORTANCE
# =====================================================

model = RandomForestRegressor(

    n_estimators=300,
    random_state=42,
    n_jobs=-1

)

model.fit(X, y)

rf_df = pd.DataFrame({

    "Feature": X.columns,
    "RF_Importance": model.feature_importances_

})

rf_df = rf_df.sort_values(

    "RF_Importance",
    ascending=False

)

# =====================================================
# COMBINE SCORES
# =====================================================

ranking = mi_df.merge(

    rf_df,

    on="Feature"

)

ranking["Combined_Score"] = (

    ranking["Mutual_Information"] +
    ranking["RF_Importance"]

)

ranking = ranking.sort_values(

    "Combined_Score",
    ascending=False

)

# =====================================================
# SAVE
# =====================================================

ranking.to_csv(

    OUTPUT / "feature_ranking.csv",

    index=False

)

selected = ranking.head(30)

selected.to_csv(

    OUTPUT / "top30_features.csv",

    index=False

)

# =====================================================
# SUMMARY
# =====================================================

print("\nTop 20 Features\n")

print(

    ranking.head(20)

)

print()

print("=" * 60)
print("FEATURE SELECTION COMPLETED")
print("=" * 60)

print()

print("Total Features :", len(ranking))

print()

print("Top 30 Saved")

print()

print(OUTPUT)

print("=" * 60)