from pathlib import Path
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = PROJECT_ROOT / "outputs" / "optimization" / "optimization_results.csv"

OUTPUT = PROJECT_ROOT / "outputs" / "optimization"

df = pd.read_csv(INPUT)

print("="*60)
print("KPI IMPROVEMENT ENGINE")
print("="*60)


# ==========================================================
# HELPERS
# ==========================================================

def percent_change(current, new):

    if current == 0:
        return 0

    return round(((new-current)/current)*100,2)


# ==========================================================
# CURRENT METRICS
# ==========================================================

df["Current_Valuation"] = df["Expected_Valuation_12M"]

df["Current_Revenue"] = df["Estimated_Revenue_Million"]

df["Current_Burn"] = df["Estimated_Burn_Million"]

df["Current_Runway"] = df["Estimated_Runway_Months"]

df["Current_ROI"] = df["Estimated_ROI"]


# ==========================================================
# KPI IMPROVEMENTS
# ==========================================================

df["Valuation_Improvement_%"] = df.apply(

    lambda x: percent_change(

        x["Current_Valuation"],
        x["Expected_Valuation_After_Optimization"]

    ),

    axis=1

)

df["Revenue_Improvement_%"] = df.apply(

    lambda x: percent_change(

        x["Current_Revenue"],
        x["Recommended_Revenue_Million"]

    ),

    axis=1

)

df["Burn_Reduction_%"] = df.apply(

    lambda x: percent_change(

        x["Current_Burn"],
        x["Recommended_Burn_Million"]

    ),

    axis=1

)

df["Runway_Improvement_%"] = df.apply(

    lambda x: percent_change(

        x["Current_Runway"],
        x["Recommended_Runway"]

    ),

    axis=1

)

df["ROI_Improvement_%"] = df.apply(

    lambda x: percent_change(

        x["Current_ROI"],
        x["Optimized_ROI"]

    ),

    axis=1

)

# ==========================================================
# SAVE
# ==========================================================

SAVE = OUTPUT / "kpi_results.csv"

df.to_csv(SAVE, index=False)

print()
print("KPI Engine Completed")
print()

print(df[[

    "Company",

    "Current_Valuation",

    "Expected_Valuation_After_Optimization",

    "Valuation_Improvement_%",

    "Revenue_Improvement_%",

    "Burn_Reduction_%",

    "ROI_Improvement_%"

]].head())

print()

print("Saved")

print(SAVE)

print("="*60)