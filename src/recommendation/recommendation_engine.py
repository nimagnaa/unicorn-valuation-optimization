from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT / "data" / "processed" / "forecast_dataset.csv"

MODEL = PROJECT_ROOT / "models" / "tuned" / "tuned_xgboost.pkl"

OUTPUT = PROJECT_ROOT / "outputs" / "recommendations"

OUTPUT.mkdir(parents=True, exist_ok=True)

print("="*60)
print("AI RECOMMENDATION ENGINE")
print("="*60)

# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(DATA)

model = joblib.load(MODEL)

feature_columns = list(model.feature_names_in_)

recommendations = []

# ==========================================================
# SEARCH BEST BUSINESS STRATEGY
# ==========================================================

for _, row in df.iterrows():

    best_prediction = -1
    best = None

    base_revenue = row["Estimated_Revenue_Million"]
    base_burn = row["Estimated_Burn_Million"]
    base_emp = row["Estimated_Employees"]

    for revenue_factor in np.arange(1.00,1.31,0.05):

        for burn_factor in np.arange(0.75,1.01,0.05):

            for hiring_factor in np.arange(1.00,1.26,0.05):

                sample = row.copy()

                sample["Estimated_Revenue_Million"] = (
                    base_revenue * revenue_factor
                )

                sample["Estimated_Burn_Million"] = (
                    base_burn * burn_factor
                )

                sample["Estimated_Employees"] = (
                    base_emp * hiring_factor
                )

                sample["Employee_Growth_Rate"] = (
                    sample["Estimated_Employees"] /
                    max(sample["Company Age"],1)
                )

                sample["Cash_Efficiency"] = (
                    sample["Estimated_Revenue_Million"] /
                    max(sample["Estimated_Burn_Million"],1)
                )

                sample["Burn_Efficiency"] = (
                    sample["Estimated_Profit_Million"] /
                    max(sample["Estimated_Burn_Million"],1)
                )

                sample["Capital_Utilization"] = (
                    sample["Estimated_Revenue_Million"] /
                    max(sample["Funding_Million"],1)
                )

                sample = pd.DataFrame([sample])

                sample = sample.drop(

                    columns=[

                        "Company",
                        "Expected_Valuation_12M",
                        "Down_Round"

                    ],

                    errors="ignore"

                )

                sample = pd.get_dummies(sample)

                sample = sample.reindex(

                    columns=feature_columns,

                    fill_value=0

                )

                prediction = model.predict(sample)[0]

                if prediction > best_prediction:

                    best_prediction = prediction

                    best = {

                        "Revenue_Factor": revenue_factor,

                        "Burn_Factor": burn_factor,

                        "Hiring_Factor": hiring_factor,

                        "Predicted_Valuation": prediction

                    }

    recommendations.append(best)