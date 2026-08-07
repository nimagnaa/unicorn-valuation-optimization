import re
import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

RAW_DATA = Path("data/raw")
PROCESSED_DATA = Path("data/processed")

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading datasets...")

unicorn = pd.read_csv(RAW_DATA / "Unicorn_Companies.csv")
layoffs = pd.read_excel(RAW_DATA / "layoffs_2_13_2023.xlsx")

print(f"Unicorn Dataset : {unicorn.shape}")
print(f"Layoff Dataset  : {layoffs.shape}")

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def clean_money(value):
    """
    Converts:
    $140B -> 140000
    $7.44B -> 7440
    $950M -> 950

    Unit:
    Million USD
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    value = value.replace("$", "")
    value = value.replace(",", "")

    multiplier = 1

    if value.endswith("B"):
        multiplier = 1000
        value = value[:-1]

    elif value.endswith("M"):
        multiplier = 1
        value = value[:-1]

    try:
        return float(value) * multiplier
    except:
        return np.nan


def clean_percentage(value):

    if pd.isna(value):
        return np.nan

    value = str(value).replace("%", "").strip()

    try:
        return float(value)
    except:
        return np.nan


def normalize_company(company):

    if pd.isna(company):
        return np.nan

    company = str(company).lower()

    company = re.sub(r"[^a-z0-9 ]", "", company)

    company = company.strip()

    return company


# =====================================================
# CLEAN UNICORN DATASET
# =====================================================

print("\nCleaning Unicorn Dataset...")

unicorn.columns = unicorn.columns.str.strip()

unicorn["Valuation_Million"] = unicorn["Valuation ($B)"].apply(clean_money)

unicorn["Funding_Million"] = unicorn["Total Raised"].apply(clean_money)

unicorn["Date Joined"] = pd.to_datetime(
    unicorn["Date Joined"],
    errors="coerce"
)

CURRENT_YEAR = 2026

unicorn["Company Age"] = (
    CURRENT_YEAR -
    unicorn["Founded Year"]
)

unicorn["Years Since Unicorn"] = (
    CURRENT_YEAR -
    unicorn["Date Joined"].dt.year
)

unicorn["Company_Key"] = unicorn["Company"].apply(normalize_company)

# =====================================================
# CLEAN LAYOFF DATASET
# =====================================================

print("Cleaning Layoff Dataset...")

layoffs.columns = layoffs.columns.str.strip()

layoffs["Layoff_Percentage"] = layoffs["Percentage"].apply(clean_percentage)

layoffs["Company_Key"] = layoffs["Company"].apply(normalize_company)

layoffs["Total Laid Off"] = pd.to_numeric(
    layoffs["Total Laid Off"],
    errors="coerce"
)

layoffs["Date"] = pd.to_datetime(
    layoffs["Date"],
    errors="coerce"
)

# =====================================================
# AGGREGATE LAYOFF DATA
# =====================================================

print("Aggregating Layoff Information...")

layoff_summary = (
    layoffs
    .groupby("Company_Key")
    .agg(
        Total_Layoffs=("Total Laid Off", "sum"),
        Average_Layoff_Percentage=("Layoff_Percentage", "mean"),
        Layoff_Events=("Company_Key", "count"),
        Latest_Layoff=("Date", "max")
    )
    .reset_index()
)

# =====================================================
# MERGE DATASETS
# =====================================================

print("Merging datasets...")

master = unicorn.merge(
    layoff_summary,
    on="Company_Key",
    how="left"
)

# =====================================================
# CREATE BASIC FEATURES
# =====================================================

print("Creating initial business features...")

master["Layoff_Events"] = master["Layoff_Events"].fillna(0)

master["Total_Layoffs"] = master["Total_Layoffs"].fillna(0)

master["Average_Layoff_Percentage"] = master[
    "Average_Layoff_Percentage"
].fillna(0)

master["Layoff_Flag"] = (
    master["Layoff_Events"] > 0
).astype(int)

master["Funding_Efficiency"] = (
    master["Valuation_Million"] /
    master["Funding_Million"]
)

master["Capital_Raised_Per_Year"] = (
    master["Funding_Million"] /
    master["Company Age"]
)

master["Funding_Velocity"] = (
    master["Funding_Million"] /
    master["Years Since Unicorn"]
)

master["Investor_Density"] = (
    master["Investors Count"] /
    master["Company Age"]
)

master["Valuation_Per_Investor"] = (
    master["Valuation_Million"] /
    master["Investors Count"]
)

master.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# =====================================================
# SAVE DATA
# =====================================================

output_path = PROCESSED_DATA / "master_unicorn_dataset.csv"

master.to_csv(
    output_path,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(f"Rows    : {master.shape[0]}")
print(f"Columns : {master.shape[1]}")

print("\nSaved File")
print(output_path)

print("\nTop Features Created")

features = [
    "Valuation_Million",
    "Funding_Million",
    "Company Age",
    "Years Since Unicorn",
    "Funding_Efficiency",
    "Capital_Raised_Per_Year",
    "Funding_Velocity",
    "Investor_Density",
    "Valuation_Per_Investor",
    "Layoff_Flag",
    "Layoff_Events",
    "Average_Layoff_Percentage"
]

for feature in features:
    print("✓", feature)

print("=" * 60)