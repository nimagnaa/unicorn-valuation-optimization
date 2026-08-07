from pathlib import Path

import streamlit as st
import pandas as pd


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Optimization",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "outputs"
    / "optimization"
    / "ml_optimization.csv"
)

CSS_PATH = (
    ROOT
    / "dashboard"
    / "style.css"
)


# =========================================================
# LOAD CSS
# =========================================================

if CSS_PATH.exists():

    with open(CSS_PATH, encoding="utf-8") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# =========================================================
# HEADER
# =========================================================

st.title("AI Optimization")

st.caption(
    "Data-driven comparison of current and optimized financial performance."
)

st.divider()


# =========================================================
# LOAD DATA
# =========================================================

if not DATA_PATH.exists():

    st.error(
        "Optimization results were not found."
    )

    st.code(
        str(DATA_PATH)
    )

    st.stop()


df = pd.read_csv(DATA_PATH)


# =========================================================
# COMPANY SELECTION
# =========================================================

companies = sorted(
    df["Company"]
    .dropna()
    .unique()
)

company = st.selectbox(
    "Select Company",
    companies
)


row = df[
    df["Company"] == company
].iloc[0]


# =========================================================
# COMPANY INFORMATION
# =========================================================

st.subheader("Company")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Company",
        company
    )

with c2:

    st.metric(
        "Industry",
        row.get("Industry", "N/A")
    )

with c3:

    st.metric(
        "Country",
        row.get("Country", "N/A")
    )

with c4:

    st.metric(
        "Financial Stage",
        row.get("Financial Stage", "N/A")
    )


st.divider()


# =========================================================
# KPI CALCULATIONS
# =========================================================

current_valuation = row[
    "Current_Valuation_ML"
]

optimized_valuation = row[
    "Optimized_Valuation"
]


current_revenue = row[
    "Current_Revenue"
]

optimized_revenue = row[
    "Recommended_Revenue"
]


current_burn = row[
    "Current_Burn"
]

optimized_burn = row[
    "Recommended_Burn"
]


current_runway = row[
    "Current_Runway"
]

optimized_runway = row[
    "Optimized_Runway"
]


current_utilization = row[
    "Current_Capital_Utilization"
]

optimized_utilization = row[
    "Optimized_Capital_Utilization"
]


current_roi = row[
    "Current_ROI"
]

optimized_roi = row[
    "Optimized_ROI"
]


# =========================================================
# MAIN KPI TABLE
# =========================================================

st.subheader("Current vs Optimized")

kpi_table = pd.DataFrame({

    "KPI": [

        "Expected Valuation",

        "Estimated Revenue",

        "Burn Rate",

        "Runway",

        "Capital Utilization",

        "Expected ROI"

    ],

    "Current": [

        f"${current_valuation:,.0f} M",

        f"${current_revenue:,.0f} M",

        f"${current_burn:,.0f} M",

        f"{current_runway:.1f} mo",

        f"{current_utilization:.2f}x",

        f"{current_roi:.2f}x"

    ],

    "Optimized": [

        f"${optimized_valuation:,.0f} M",

        f"${optimized_revenue:,.0f} M",

        f"${optimized_burn:,.0f} M",

        f"{optimized_runway:.1f} mo",

        f"{optimized_utilization:.2f}x",

        f"{optimized_roi:.2f}x"

    ],

    "Change": [

        f"{row['Valuation_Change_Percent']:+.1f}%",

        f"{row['Revenue_Change_Percent']:+.1f}%",

        f"{row['Burn_Change_Percent']:+.1f}%",

        f"{row['Runway_Change_Months']:+.1f} mo",

        f"{(
            optimized_utilization
            - current_utilization
        ):+.2f}x",

        f"{(
            (
                optimized_roi
                - current_roi
            )
            / current_roi
            * 100
        ):+.1f}%"
        if current_roi != 0
        else "N/A"

    ]

})


st.dataframe(
    kpi_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# LARGE KPI CARDS
# =========================================================

st.subheader("Optimization Impact")


c1, c2, c3 = st.columns(3)

with c1:

    st.metric(

        "Valuation Improvement",

        f"{row['Valuation_Change_Percent']:+.1f}%"

    )


with c2:

    st.metric(

        "Revenue Improvement",

        f"{row['Revenue_Change_Percent']:+.1f}%"

    )


with c3:

    st.metric(

        "Additional Runway",

        f"{row['Runway_Change_Months']:+.1f} months"

    )


c1, c2, c3 = st.columns(3)

with c1:

    st.metric(

        "Burn Reduction",

        f"{abs(row['Burn_Change_Percent']):.1f}%"

    )


with c2:

    st.metric(

        "Capital Utilization",

        f"{optimized_utilization:.2f}x"

    )


with c3:

    roi_change = (

        (
            optimized_roi
            - current_roi
        )
        / current_roi
        * 100
        if current_roi != 0
        else 0
    )

    st.metric(

        "ROI Improvement",

        f"{roi_change:+.1f}%"

    )


st.divider()


# =========================================================
# RECOMMENDATIONS
# =========================================================

st.subheader("Recommended Actions")


st.write(
    f"""
    **Revenue:** Target approximately
    **${optimized_revenue:,.0f} million** in revenue.

    **Burn:** Reduce estimated burn from
    **${current_burn:,.0f} million** to approximately
    **${optimized_burn:,.0f} million**.

    **Runway:** The optimized scenario increases estimated
    runway from **{current_runway:.1f} months** to
    **{optimized_runway:.1f} months**.

    **Capital:** Improve capital utilization from
    **{current_utilization:.2f}x** to
    **{optimized_utilization:.2f}x**.

    **ROI:** Expected ROI moves from
    **{current_roi:.2f}x** to
    **{optimized_roi:.2f}x**.
    """
)


# =========================================================
# DOWN ROUND
# =========================================================

st.divider()

st.subheader("Down-Round Assessment")


current_risk = row[
    "Current_Down_Round"
]

optimized_risk = row[
    "Optimized_Down_Round"
]


risk_table = pd.DataFrame({

    "Metric": [
        "Current",
        "Optimized"
    ],

    "Down-Round Indicator": [

        "Yes" if current_risk == 1 else "No",

        "Yes" if optimized_risk == 1 else "No"

    ]

})


st.dataframe(
    risk_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DOWNLOAD
# =========================================================

st.divider()

st.subheader("Export")

st.download_button(

    "Download Company Optimization",

    row.to_frame().T.to_csv(
        index=False
    ),

    file_name=(
        f"{company}_optimization.csv"
    ),

    mime="text/csv"

)