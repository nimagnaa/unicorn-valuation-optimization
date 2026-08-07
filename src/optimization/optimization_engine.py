from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = PROJECT_ROOT / "data" / "processed" / "forecast_dataset.csv"

OUTPUT = PROJECT_ROOT / "outputs" / "optimization"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

print("="*60)
print("AI BUSINESS OPTIMIZATION ENGINE")
print("="*60)

# ==========================================================
# TARGETS
# ==========================================================

TARGET_RUNWAY = 30

TARGET_BURN_RATIO = 0.25

TARGET_ROI = 0.35

TARGET_REVENUE_GROWTH = 0.20

TARGET_EMPLOYEE_GROWTH = 0.15

# ==========================================================
# OUTPUT LISTS
# ==========================================================

recommended_revenue=[]

recommended_burn=[]

recommended_runway=[]

recommended_hiring=[]

recommended_investment=[]

expected_roi=[]

optimized_valuation=[]

revenue_gap=[]

burn_gap=[]

valuation_gain=[]

# ==========================================================
# OPTIMIZATION
# ==========================================================

for _,row in df.iterrows():

    revenue=row["Estimated_Revenue_Million"]

    burn=row["Estimated_Burn_Million"]

    runway=row["Estimated_Runway_Months"]

    valuation=row["Expected_Valuation_12M"]

    funding=row["Funding_Million"]

    employees=row["Estimated_Employees"]

    # ---------------- Revenue ----------------

    target_revenue=revenue*(1+TARGET_REVENUE_GROWTH)

    # ---------------- Burn ----------------

    target_burn=max(

        burn*0.82,

        15

    )

    # ---------------- Hiring ----------------

    target_hiring=employees*(1+TARGET_EMPLOYEE_GROWTH)

    # ---------------- Runway ----------------

    target_runway=max(

        runway,

        TARGET_RUNWAY

    )

    # ---------------- Investment ----------------

    extra=max(

        0,

        target_burn*target_runway/12-funding

    )

    # ---------------- ROI ----------------

    roi=(

        target_revenue*0.25

    )/(

        funding+extra

    )

    # ---------------- Optimized Valuation ----------------

    new_value=(

        valuation

        *1.18

        *(target_revenue/revenue)

        *(burn/target_burn)

    )

    recommended_revenue.append(round(target_revenue,2))

    recommended_burn.append(round(target_burn,2))

    recommended_runway.append(round(target_runway,1))

    recommended_hiring.append(round(target_hiring))

    recommended_investment.append(round(extra,2))

    expected_roi.append(round(roi,2))

    optimized_valuation.append(round(new_value,2))

    revenue_gap.append(round(target_revenue-revenue,2))

    burn_gap.append(round(burn-target_burn,2))

    valuation_gain.append(round(new_value-valuation,2))

# ==========================================================
# SAVE
# ==========================================================

df["Recommended_Revenue_Million"]=recommended_revenue

df["Revenue_Increase_Required"]=revenue_gap

df["Recommended_Burn_Million"]=recommended_burn

df["Burn_Reduction_Required"]=burn_gap

df["Recommended_Runway"]=recommended_runway

df["Recommended_Employees"]=recommended_hiring

df["Additional_Investment_Required"]=recommended_investment

df["Optimized_ROI"]=expected_roi

df["Expected_Valuation_After_Optimization"]=optimized_valuation

df["Valuation_Gain"]=valuation_gain

SAVE = OUTPUT/"optimization_results.csv"

df.to_csv(SAVE,index=False)

print()

print("Optimization Completed")

print()

print(df[[

"Company",

"Expected_Valuation_12M",

"Expected_Valuation_After_Optimization",

"Recommended_Revenue_Million",

"Recommended_Burn_Million",

"Recommended_Runway",

"Optimized_ROI"

]].head())

print()

print("Saved")

print(SAVE)

print("="*60)