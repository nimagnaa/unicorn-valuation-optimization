from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ======================================================
# PATHS
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT / "data" / "processed" / "forecast_dataset.csv"

MODEL = PROJECT_ROOT / "models" / "tuned" / "tuned_xgboost.pkl"

OUTPUT = PROJECT_ROOT / "outputs" / "explainability"

OUTPUT.mkdir(parents=True, exist_ok=True)

print("="*60)
print("SHAP EXPLAINABILITY")
print("="*60)

# ======================================================

df = pd.read_csv(DATA)

TARGET = "Expected_Valuation_12M"

DROP = [
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

DROP = [c for c in DROP if c in df.columns]

X = df.drop(columns=DROP)

X = pd.get_dummies(X, drop_first=True)

X = X.fillna(X.median(numeric_only=True))

model = joblib.load(MODEL)

X = X.reindex(columns=model.feature_names_in_, fill_value=0)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": abs(shap_values).mean(axis=0)

})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    OUTPUT/"feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,7))

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT/"shap_summary.png",
    dpi=300
)

plt.close()

print()
print("Top Features")
print(importance.head(20))

print()
print("Saved to")
print(OUTPUT)