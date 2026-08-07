import numpy as np


def create_financial_features(df):

    print("Creating Financial Features...")

    # -----------------------------
    # Revenue Efficiency
    # -----------------------------

    df["Revenue_Efficiency"] = (
        df["Estimated_Revenue_Million"] /
        df["Funding_Million"]
    )

    # -----------------------------
    # Profit Margin
    # -----------------------------

    df["Profit_Margin"] = (
        df["Estimated_Profit_Million"] /
        df["Estimated_Revenue_Million"]
    )

    # -----------------------------
    # Burn Ratio
    # -----------------------------

    df["Burn_Ratio"] = (
        df["Estimated_Burn_Million"] /
        df["Funding_Million"]
    )

    # -----------------------------
    # Burn To Revenue
    # -----------------------------

    df["Burn_to_Revenue"] = (
        df["Estimated_Burn_Million"] /
        df["Estimated_Revenue_Million"]
    )

    # -----------------------------
    # Burn To Profit
    # -----------------------------

    df["Burn_to_Profit"] = (
        df["Estimated_Burn_Million"] /
        df["Estimated_Profit_Million"]
    )

    # -----------------------------
    # Revenue Per Investor
    # -----------------------------

    df["Revenue_Per_Investor"] = (
        df["Estimated_Revenue_Million"] /
        df["Investors Count"]
    )

    # -----------------------------
    # Profit Per Investor
    # -----------------------------

    df["Profit_Per_Investor"] = (
        df["Estimated_Profit_Million"] /
        df["Investors Count"]
    )

    # -----------------------------
    # Funding Per Investor
    # -----------------------------

    df["Funding_Per_Investor"] = (
        df["Funding_Million"] /
        df["Investors Count"]
    )

    # -----------------------------
    # Revenue To Valuation
    # -----------------------------

    df["Revenue_to_Valuation"] = (
        df["Estimated_Revenue_Million"] /
        df["Valuation_Million"]
    )

    # -----------------------------
    # Profit To Valuation
    # -----------------------------

    df["Profit_to_Valuation"] = (
        df["Estimated_Profit_Million"] /
        df["Valuation_Million"]
    )

    # -----------------------------
    # Funding Utilization Score
    # -----------------------------

    df["Funding_Utilization_Score"] = (
        df["Funding_Efficiency"] *
        df["Capital_Utilization"]
    )

    # -----------------------------
    # Capital Productivity
    # -----------------------------

    df["Capital_Productivity"] = (
        df["Estimated_Profit_Million"] /
        df["Funding_Million"]
    )

    # -----------------------------
    # Replace infinities
    # -----------------------------

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    print("Financial Features Created")

    return df