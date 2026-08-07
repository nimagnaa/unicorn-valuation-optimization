from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT / "data" / "processed" / "forecast_dataset.csv"


def load_data():

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

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    y = df[TARGET]

    return train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42

    )