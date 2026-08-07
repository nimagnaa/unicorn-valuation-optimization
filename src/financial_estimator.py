import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================================
# LOAD DATA
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = PROJECT_ROOT / "data" / "processed" / "master_unicorn_dataset.csv"

OUTPUT = PROJECT_ROOT / "data" / "processed" / "financial_dataset.csv"

df = pd.read_csv(INPUT)
# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

numeric_fill = {

    "Funding_Million": df["Funding_Million"].median(),

    "Investors Count": df["Investors Count"].median(),

    "Company Age": df["Company Age"].median(),

    "Layoff_Events": 0,

    "Average_Layoff_Percentage": 0

}

for column, value in numeric_fill.items():

    if column in df.columns:

        df[column] = df[column].fillna(value)

CURRENT_YEAR = 2026

print("=" * 60)
print("FINANCIAL ESTIMATION ENGINE")
print("=" * 60)

# ==========================================================
# HELPER
# ==========================================================

def safe_divide(a, b):

    if pd.isna(a) or pd.isna(b):
        return np.nan

    if b == 0:
        return np.nan

    return a / b


# ==========================================================
# BUSINESS ASSUMPTIONS
# ==========================================================

industry_revenue_factor = {

    "Fintech": 1.45,
    "Artificial intelligence": 1.70,
    "Cybersecurity": 1.55,
    "Health": 1.25,
    "Data management & analytics": 1.60,
    "Internet software & services": 1.40,
    "Supply chain": 1.18,
    "Consumer & retail": 1.10,
    "Mobile & telecommunications": 1.22

}

industry_profit_margin = {

    "Fintech": 0.24,
    "Artificial intelligence": 0.30,
    "Cybersecurity": 0.28,
    "Health": 0.17,
    "Data management & analytics": 0.27,
    "Internet software & services": 0.23,
    "Supply chain": 0.14,
    "Consumer & retail": 0.11,
    "Mobile & telecommunications": 0.18

}

country_growth_factor = {

    "United States": 1.12,
    "India": 1.22,
    "United Kingdom": 1.09,
    "China": 1.08,
    "Germany": 1.06,
    "France": 1.05

}

DEFAULT_REVENUE_FACTOR = 1.20
DEFAULT_MARGIN = 0.18
DEFAULT_COUNTRY = 1.05

# ==========================================================
# ESTIMATE REVENUE
# ==========================================================

estimated_revenue = []

for _, row in df.iterrows():

    funding = row["Funding_Million"]

    investors = row["Investors Count"]

    company_age = max(row["Company Age"], 1)

    layoff_events = row["Layoff_Events"]

    industry = row["Industry"]

    country = row["Country"]

    revenue_factor = industry_revenue_factor.get(
        industry,
        DEFAULT_REVENUE_FACTOR
    )

    country_factor = country_growth_factor.get(
        country,
        DEFAULT_COUNTRY
    )

    investor_bonus = np.log1p(investors) * 10

    maturity_bonus = np.sqrt(company_age) * 14

    funding_component = funding * revenue_factor

    layoff_penalty = layoff_events * 15

    revenue = (

        funding_component

        + investor_bonus

        + maturity_bonus

        - layoff_penalty

    )

    revenue = revenue * country_factor

    revenue = max(revenue, 25)

    estimated_revenue.append(round(revenue, 2))

df["Estimated_Revenue_Million"] = estimated_revenue

# ==========================================================
# ESTIMATED EMPLOYEES
# ==========================================================






estimated_employees = []

for _, row in df.iterrows():

    funding = row["Funding_Million"]

    investors = row["Investors Count"]

    company_age = max(row["Company Age"], 1)

    layoffs = row["Layoff_Events"]

    employees = (

        funding * 4.5

        + investors * 8

        + company_age * 18

        - layoffs * 35

    )

    employees = max(employees, 50)

    if pd.isna(employees):
        print("NaN Found:")
        print(row[[
        "Company",
        "Funding_Million",
        "Investors Count",
        "Company Age",
        "Layoff_Events"
    ]])


    estimated_employees.append(round(employees))

df["Estimated_Employees"] = estimated_employees

# ==========================================================
# EMPLOYEE GROWTH RATE
# ==========================================================

df["Employee_Growth_Rate"] = (

    df["Estimated_Employees"]

    / df["Company Age"]

)

# ==========================================================
# ESTIMATED PROFIT
# ==========================================================

estimated_profit = []

profit_margin = []

operating_cost = []

for _, row in df.iterrows():

    industry = row["Industry"]

    margin = industry_profit_margin.get(
        industry,
        DEFAULT_MARGIN
    )

    revenue = row["Estimated_Revenue_Million"]

    cost = revenue * (1 - margin)

    profit = revenue - cost

    estimated_profit.append(round(profit, 2))

    operating_cost.append(round(cost, 2))

    profit_margin.append(round(margin, 3))

df["Operating_Cost_Million"] = operating_cost

df["Profit_Margin"] = profit_margin

df["Estimated_Profit_Million"] = estimated_profit

print()
print("Revenue, Employee and Profit Estimation Completed")
print()

# ==========================================================
# ESTIMATED BURN RATE
# ==========================================================

burn_rate = []

for _, row in df.iterrows():

    operating_cost = row["Operating_Cost_Million"]

    layoffs = row["Layoff_Events"]

    expansion_factor = 1 + (0.04 * row["Company Age"])

    burn = (

        operating_cost
        * expansion_factor

        + layoffs * 8

    )

    burn = max(burn, 15)

    burn_rate.append(round(burn, 2))

df["Estimated_Burn_Million"] = burn_rate


# ==========================================================
# ESTIMATED RUNWAY
# ==========================================================

df["Estimated_Runway_Months"] = (

    df["Funding_Million"]

    / df["Estimated_Burn_Million"]

) * 12

df["Estimated_Runway_Months"] = (

    df["Estimated_Runway_Months"]

    .clip(lower=3, upper=72)

)


# ==========================================================
# ROI
# ==========================================================

df["Estimated_ROI"] = (

    df["Estimated_Profit_Million"]

    /

    df["Funding_Million"]

)

df["Estimated_ROI"] = df["Estimated_ROI"].clip(lower=-2, upper=8)


# ==========================================================
# CASH EFFICIENCY
# ==========================================================

df["Cash_Efficiency"] = (

    df["Estimated_Revenue_Million"]

    /

    df["Estimated_Burn_Million"]

)


# ==========================================================
# BURN EFFICIENCY
# ==========================================================

df["Burn_Efficiency"] = (

    df["Estimated_Profit_Million"]

    /

    df["Estimated_Burn_Million"]

)


# ==========================================================
# CAPITAL UTILIZATION
# ==========================================================

df["Capital_Utilization"] = (

    df["Estimated_Revenue_Million"]

    /

    df["Funding_Million"]

)


# ==========================================================
# REVENUE PER EMPLOYEE
# ==========================================================

df["Revenue_Per_Employee"] = (

    df["Estimated_Revenue_Million"]

    /

    df["Estimated_Employees"]

)


# ==========================================================
# PROFIT PER EMPLOYEE
# ==========================================================

df["Profit_Per_Employee"] = (

    df["Estimated_Profit_Million"]

    /

    df["Estimated_Employees"]

)


# ==========================================================
# FUNDING PER EMPLOYEE
# ==========================================================

df["Funding_Per_Employee"] = (

    df["Funding_Million"]

    /

    df["Estimated_Employees"]

)


# ==========================================================
# RUNWAY CATEGORY
# ==========================================================

conditions = [

    df["Estimated_Runway_Months"] < 12,

    df["Estimated_Runway_Months"].between(12,24),

    df["Estimated_Runway_Months"] > 24

]

choices = [

    "Critical",

    "Moderate",

    "Healthy"

]

df["Runway_Status"] = np.select(
    conditions,
    choices,
    default="Healthy"
)


# ==========================================================
# FINANCIAL HEALTH INDEX
# ==========================================================

df["Financial_Health_Index"] = (

      0.30 * df["Cash_Efficiency"]

    + 0.25 * df["Burn_Efficiency"]

    + 0.20 * df["Capital_Utilization"]

    + 0.15 * df["Estimated_ROI"]

    + 0.10 * (df["Estimated_Runway_Months"] / 24)

)

df["Financial_Health_Index"] = (

    df["Financial_Health_Index"]

    .clip(lower=0)

)


# ==========================================================
# CLEAN INF VALUES
# ==========================================================

numeric_columns = df.select_dtypes(include="number").columns

df[numeric_columns] = (

    df[numeric_columns]

    .replace([np.inf,-np.inf],np.nan)

)

df[numeric_columns] = (

    df[numeric_columns]

    .fillna(df[numeric_columns].median())

)


# ==========================================================
# SAVE
# ==========================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

print("="*60)
print("FINANCIAL ESTIMATION COMPLETED")
print("="*60)

print()

print("New Financial Features")

features = [

    "Estimated_Revenue_Million",

    "Estimated_Employees",

    "Employee_Growth_Rate",

    "Estimated_Profit_Million",

    "Operating_Cost_Million",

    "Estimated_Burn_Million",

    "Estimated_Runway_Months",

    "Estimated_ROI",

    "Cash_Efficiency",

    "Burn_Efficiency",

    "Capital_Utilization",

    "Revenue_Per_Employee",

    "Profit_Per_Employee",

    "Funding_Per_Employee",

    "Financial_Health_Index"

]

for feature in features:

    print(f"✓ {feature}")

print()

print("Dataset Shape :",df.shape)

print()

print("Saved To")

print(OUTPUT)

print("="*60)