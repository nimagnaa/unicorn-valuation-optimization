import numpy as np


CURRENT_YEAR = 2026


def create_growth_features(df):

    print("Creating Growth Features...")

    df["Revenue_Per_Year"] = (
        df["Estimated_Revenue_Million"] /
        df["Company Age"]
    )

    df["Funding_Per_Year"] = (
        df["Funding_Million"] /
        df["Company Age"]
    )

    df["Profit_Per_Year"] = (
        df["Estimated_Profit_Million"] /
        df["Company Age"]
    )

    df["Revenue_Per_Employee"] = (
        df["Estimated_Revenue_Million"] /
        df["Estimated_Employees"]
    )

    df["Profit_Per_Employee"] = (
        df["Estimated_Profit_Million"] /
        df["Estimated_Employees"]
    )

    df["Valuation_Per_Employee"] = (
        df["Valuation_Million"] /
        df["Estimated_Employees"]
    )

    df["Funding_Per_Employee"] = (
        df["Funding_Million"] /
        df["Estimated_Employees"]
    )

    df["Growth_Momentum"] = (
        df["Funding_Velocity"] *
        df["Funding_Efficiency"]
    )

    df["Company_Maturity"] = np.select(

        [
            df["Company Age"] <= 5,
            df["Company Age"].between(6,10),
            df["Company Age"] > 10
        ],

        [
            "Early",
            "Growth",
            "Mature"
        ],

        default="Unknown"

    )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    print("Growth Features Created")

    return df