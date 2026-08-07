from pathlib import Path

import streamlit as st
import pandas as pd


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Executive Report",
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
# CSS
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

st.title("Executive Report")

st.caption(
    "Executive-level summary of valuation and financial optimization results."
)

st.divider()


# =========================================================
# LOAD DATA
# =========================================================

if not DATA_PATH.exists():

    st.error(
        "Optimization results are not available."
    )

    st.stop()


df = pd.read_csv(DATA_PATH)


# =========================================================
# COMPANY
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
# COMPANY HEADER
# =========================================================

st.subheader(company)

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.write("Industry")
    st.write(
        f"**{row.get('Industry', 'N/A')}**"
    )

with info2:
    st.write("Country")
    st.write(
        f"**{row.get('Country', 'N/A')}**"
    )

with info3:
    st.write("Financial Stage")
    st.write(
        f"**{row.get('Financial Stage', 'N/A')}**"
    )

with info4:
    st.write("City")
    st.write(
        f"**{row.get('City', 'N/A')}**"
    )


st.divider()


# =========================================================
# KPI VALUES
# =========================================================

current_valuation = row["Current_Valuation_ML"]
optimized_valuation = row["Optimized_Valuation"]

current_revenue = row["Current_Revenue"]
optimized_revenue = row["Recommended_Revenue"]

current_burn = row["Current_Burn"]
optimized_burn = row["Recommended_Burn"]

current_runway = row["Current_Runway"]
optimized_runway = row["Optimized_Runway"]

current_utilization = row[
    "Current_Capital_Utilization"
]

optimized_utilization = row[
    "Optimized_Capital_Utilization"
]

current_roi = row["Current_ROI"]
optimized_roi = row["Optimized_ROI"]


# =========================================================
# KPI CHANGE
# =========================================================

valuation_change = (
    row["Valuation_Change_Percent"]
)

revenue_change = (
    row["Revenue_Change_Percent"]
)

burn_change = (
    row["Burn_Change_Percent"]
)

runway_change = (
    row["Runway_Change_Months"]
)

utilization_change = (
    optimized_utilization
    - current_utilization
)

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


# =========================================================
# EXECUTIVE KPI SUMMARY
# =========================================================

st.subheader("Executive KPI Summary")

k1, k2, k3 = st.columns(3)

with k1:

    st.metric(
        "Expected Valuation",
        f"${optimized_valuation:,.0f} M",
        f"{valuation_change:+.1f}%"
    )

with k2:

    st.metric(
        "Estimated Revenue",
        f"${optimized_revenue:,.0f} M",
        f"{revenue_change:+.1f}%"
    )

with k3:

    st.metric(
        "Burn Rate",
        f"${optimized_burn:,.0f} M",
        f"{burn_change:+.1f}%"
    )


k1, k2, k3 = st.columns(3)

with k1:

    st.metric(
        "Runway",
        f"{optimized_runway:.1f} months",
        f"{runway_change:+.1f} months"
    )

with k2:

    st.metric(
        "Capital Utilization",
        f"{optimized_utilization:.2f}x",
        f"{utilization_change:+.2f}x"
    )

with k3:

    st.metric(
        "Expected ROI",
        f"{optimized_roi:.2f}x",
        f"{roi_change:+.1f}%"
    )


st.divider()


# =========================================================
# CURRENT VS OPTIMIZED
# =========================================================

st.subheader("Current vs Optimized Performance")

comparison = pd.DataFrame({

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
        f"{current_runway:.1f} months",
        f"{current_utilization:.2f}x",
        f"{current_roi:.2f}x"
    ],

    "Optimized": [
        f"${optimized_valuation:,.0f} M",
        f"${optimized_revenue:,.0f} M",
        f"${optimized_burn:,.0f} M",
        f"{optimized_runway:.1f} months",
        f"{optimized_utilization:.2f}x",
        f"{optimized_roi:.2f}x"
    ],

    "Change": [
        f"{valuation_change:+.1f}%",
        f"{revenue_change:+.1f}%",
        f"{burn_change:+.1f}%",
        f"{runway_change:+.1f} months",
        f"{utilization_change:+.2f}x",
        f"{roi_change:+.1f}%"
    ]

})


st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# BUSINESS INTERPRETATION
# =========================================================

st.subheader("Management Interpretation")

st.write(
    f"""
    The optimization scenario projects an increase in expected
    valuation from **${current_valuation:,.0f} million** to
    **${optimized_valuation:,.0f} million**, representing a
    **{valuation_change:+.1f}%** change.

    Estimated revenue moves from **${current_revenue:,.0f} million**
    to approximately **${optimized_revenue:,.0f} million**.

    At the same time, the recommended burn level decreases from
    **${current_burn:,.0f} million** to approximately
    **${optimized_burn:,.0f} million**.

    This increases projected runway from **{current_runway:.1f} months**
    to **{optimized_runway:.1f} months**.

    Capital utilization improves from **{current_utilization:.2f}x**
    to **{optimized_utilization:.2f}x**, while expected ROI moves from
    **{current_roi:.2f}x** to **{optimized_roi:.2f}x**.
    """
)


# =========================================================
# DOWN ROUND
# =========================================================

st.divider()

st.subheader("Down-Round Assessment")

current_down_round = row[
    "Current_Down_Round"
]

optimized_down_round = row[
    "Optimized_Down_Round"
]


d1, d2 = st.columns(2)

with d1:

    st.write("Current")

    if current_down_round == 1:
        st.error("Down-round indicator: Yes")
    else:
        st.success("Down-round indicator: No")


with d2:

    st.write("Optimized")

    if optimized_down_round == 1:
        st.error("Down-round indicator: Yes")
    else:
        st.success("Down-round indicator: No")


# =========================================================
# EXPORT
# =========================================================

st.divider()

st.subheader("Export Report")

report = comparison.to_csv(
    index=False
)

st.download_button(
    "Download Executive KPI Report",
    report,
    file_name=f"{company}_executive_report.csv",
    mime="text/csv"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Unicorn Valuation Optimization | Executive Decision Support"
)