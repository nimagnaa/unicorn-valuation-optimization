from pathlib import Path
import pandas as pd

from src.feature_engineering.financial_features import create_financial_features
from src.feature_engineering.growth_features import create_growth_features

# -------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = PROJECT_ROOT / "data" / "processed" / "financial_dataset.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "features_dataset.csv"

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv(INPUT)

print(f"\nInitial Shape : {df.shape}")

df = create_financial_features(df)
df = create_growth_features(df)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)
print(f"\nFinal Shape : {df.shape}")
print(f"\nSaved To : {OUTPUT}")