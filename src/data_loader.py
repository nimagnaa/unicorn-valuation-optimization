import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw")


def load_datasets():
    unicorn_df = pd.read_csv(DATA_PATH / "Unicorn_Companies.csv")
    layoffs_df = pd.read_excel(DATA_PATH / "layoffs_2_13_2023.xlsx")
    return unicorn_df, layoffs_df


if __name__ == "__main__":

    unicorn_df, layoffs_df = load_datasets()

    print("=" * 70)
    print("UNICORN DATASET")
    print("=" * 70)
    print(unicorn_df.head())
    print("\nShape:", unicorn_df.shape)

    print("\nColumns:")
    print(unicorn_df.columns.tolist())

    print("\nMissing Values:")
    print(unicorn_df.isnull().sum())

    print("\n")

    print("=" * 70)
    print("LAYOFF DATASET")
    print("=" * 70)
    print(layoffs_df.head())
    print("\nShape:", layoffs_df.shape)

    print("\nColumns:")
    print(layoffs_df.columns.tolist())

    print("\nMissing Values:")
    print(layoffs_df.isnull().sum())