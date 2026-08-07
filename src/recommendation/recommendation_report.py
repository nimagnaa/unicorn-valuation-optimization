from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = PROJECT_ROOT / "outputs" / "optimization" / "ml_optimization.csv"

OUTPUT = PROJECT_ROOT / "outputs" / "recommendations"

OUTPUT.mkdir(parents=True, exist_ok=True)

print("="*60)
print("EXECUTIVE RECOMMENDATION ENGINE")
print("="*60)

df = pd.read_csv(INPUT)

# ==========================================================
# RECOMMENDATIONS
# ==========================================================

df["Revenue_Increase_Million"] = (
    df["Recommended_Revenue"] -
    df["Estimated_Revenue_Million"]
)

df["Burn_Reduction_Million"] = (
    df["Estimated_Burn_Million"] -
    df["Recommended_Burn"]
)

df["Hiring_Increase"] = (
    df["Recommended_Employees"] -
    df["Estimated_Employees"]
)

df["Hiring_Growth_%"] = (
    df["Hiring_Increase"] /
    df["Estimated_Employees"] * 100
)

df["Revenue_Growth_%"] = (
    df["Revenue_Increase_Million"] /
    df["Estimated_Revenue_Million"] * 100
)

df["Burn_Reduction_%"] = (
    df["Burn_Reduction_Million"] /
    df["Estimated_Burn_Million"] * 100
)

# ==========================================================
# ROI
# ==========================================================

df["Expected_Return_Million"] = (
    df["Predicted_Valuation"] -
    df["Expected_Valuation_12M"]
)

df["ROI_Multiple"] = (
    df["Predicted_Valuation"] /
    df["Funding_Million"]
)

df["Payback_Years"] = (
    df["Funding_Million"] /
    (
        df["Recommended_Revenue"] * 0.20
    )
)

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

summary = []

for _, row in df.iterrows():

    text = f"""
Current Valuation : ${row['Expected_Valuation_12M']:.1f} Million

Recommended Revenue : ${row['Recommended_Revenue']:.1f} Million

Revenue Increase Required : ${row['Revenue_Increase_Million']:.1f} Million

Recommended Burn : ${row['Recommended_Burn']:.1f} Million

Burn Reduction Required : ${row['Burn_Reduction_Million']:.1f} Million

Recommended Employees : {int(row['Recommended_Employees'])}

Expected Valuation : ${row['Predicted_Valuation']:.1f} Million

Expected Return : ${row['Expected_Return_Million']:.1f} Million

ROI : {row['ROI_Multiple']:.2f}×

Payback : {row['Payback_Years']:.1f} Years
"""

    summary.append(text)

df["Executive_Report"] = summary

SAVE = OUTPUT / "executive_recommendations.csv"

df.to_csv(SAVE, index=False)

print()
print("Recommendation Report Created")
print()
print("Saved to")
print(SAVE)
print("="*60)