from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# LOAD DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = PROJECT_ROOT / "data" / "processed" / "features_dataset.csv"

OUTPUT = PROJECT_ROOT / "data" / "processed" / "forecast_dataset.csv"

df = pd.read_csv(INPUT)

print("="*65)
print("AI FORECASTING ENGINE")
print("="*65)

# ============================================================
# MACRO VARIABLES
# ============================================================

GDP_GROWTH = 0.035

INFLATION = 0.042

INTEREST_RATE = 0.061

# ============================================================
# INDUSTRY GROWTH
# ============================================================

industry_growth = {

    "Artificial intelligence":1.34,
    "Cybersecurity":1.24,
    "Fintech":1.18,
    "Health":1.16,
    "Data management & analytics":1.26,
    "Internet software & services":1.21,
    "Supply chain":1.10,
    "Consumer & retail":1.08,
    "Mobile & telecommunications":1.12

}

DEFAULT_GROWTH = 1.12

# ============================================================
# MARKET GROWTH
# ============================================================

market_growth=[]

for _,row in df.iterrows():

    factor=industry_growth.get(
        row["Industry"],
        DEFAULT_GROWTH
    )

    market_growth.append(factor)

df["Market_Growth"]=market_growth

# ============================================================
# EXPANSION RATE
# ============================================================

df["Expansion_Rate"]=(
      0.30*df["Funding_Velocity"]
    + 0.25*df["Capital_Utilization"]
    + 0.20*df["Cash_Efficiency"]
    + 0.25*df["Market_Growth"]
)

# ============================================================
# BUSINESS STRENGTH
# ============================================================

df["Business_Strength"]=(
      0.22*df["Funding_Efficiency"]
    + 0.18*df["Cash_Efficiency"]
    + 0.16*df["Burn_Efficiency"]
    + 0.16*df["Capital_Utilization"]
    + 0.14*df["Expansion_Rate"]
    + 0.14*df["Financial_Health_Index"]
)

# ============================================================
# EXPECTED REVENUE
# ============================================================

df["Expected_Revenue_12M"]=(
    df["Estimated_Revenue_Million"]
    *
    (
        1
        +0.08*df["Business_Strength"]
        +0.04*df["Market_Growth"]
        +GDP_GROWTH
        -INFLATION
    )
)

# ============================================================
# EXPECTED PROFIT
# ============================================================

df["Expected_Profit_12M"]=(
    df["Expected_Revenue_12M"]
    *
    df["Profit_Margin"]
)

# ============================================================
# EXPECTED BURN
# ============================================================

df["Expected_Burn_12M"]=(
    df["Estimated_Burn_Million"]
    *
    (
        1
        -0.10*df["Burn_Efficiency"]
    )
)

df["Expected_Burn_12M"]=df["Expected_Burn_12M"].clip(lower=10)

# ============================================================
# EXPECTED RUNWAY
# ============================================================

df["Expected_Runway_12M"]=(
    df["Funding_Million"]
    /
    df["Expected_Burn_12M"]
)*12

# ============================================================
# INVESTOR PREMIUM
# ============================================================

df["Investor_Premium"]=(
    np.log1p(df["Investors Count"])*45
)

# ============================================================
# MARKET PREMIUM
# ============================================================

df["Market_Premium"]=(
    df["Market_Growth"]*180
)

# ============================================================
# LAYOFF PENALTY
# ============================================================

df["Layoff_Penalty"]=(
      df["Layoff_Events"]*60
    + df["Average_Layoff_Percentage"]*2
)

# ============================================================
# FUTURE VALUATION
# ============================================================

df["Expected_Valuation_12M"]=(
      df["Expected_Revenue_12M"]*8
    + df["Expected_Profit_12M"]*5
    + df["Funding_Million"]*1.6
    + df["Investor_Premium"]
    + df["Market_Premium"]
    - df["Layoff_Penalty"]
)

df["Expected_Valuation_12M"]=(
    df["Expected_Valuation_12M"]
    .clip(lower=100)
)

# ============================================================
# DOWN ROUND
# ============================================================

risk=(
      0.30*(df["Estimated_Burn_Million"]/df["Funding_Million"])
    + 0.25*(df["Layoff_Events"]/5)
    + 0.20*(1/df["Funding_Efficiency"])
    + 0.15*(INFLATION)
    + 0.10*(INTEREST_RATE)
)

df["Down_Round"]=(risk>0.60).astype(int)

# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT,index=False)

print()
print("Forecast Features Added")
print()

features=[

"Market_Growth",
"Expansion_Rate",
"Business_Strength",
"Expected_Revenue_12M",
"Expected_Profit_12M",
"Expected_Burn_12M",
"Expected_Runway_12M",
"Expected_Valuation_12M",
"Down_Round"

]

for i in features:

    print("✓",i)

print()

print("Dataset Shape :",df.shape)

print()

print("Saved To")

print(OUTPUT)

print("="*65)