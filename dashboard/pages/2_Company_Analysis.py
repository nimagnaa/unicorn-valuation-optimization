from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE
# -------------------------------------------------

st.set_page_config(
    page_title="Company Analysis",
    layout="wide"
)

# -------------------------------------------------
# CSS
# -------------------------------------------------

css = Path(__file__).parents[1] / "style.css"

if css.exists():
    with open(css, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

ROOT = Path(__file__).parents[2]

df = pd.read_csv(
    ROOT/"data"/"processed"/"forecast_dataset.csv"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("Company Analysis")

st.caption(
    "Explore the financial profile and AI forecast for each unicorn company."
)

st.divider()

# -------------------------------------------------
# SEARCH
# -------------------------------------------------

company = st.selectbox(

    "Select Company",

    sorted(df["Company"].unique())

)

row = df[df["Company"] == company].iloc[0]

# -------------------------------------------------
# KPI
# -------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Current Valuation",
    f"${row['Valuation_Million']:,.0f} M"
)

c2.metric(
    "Expected Valuation",
    f"${row['Expected_Valuation_12M']:,.0f} M"
)

c3.metric(
    "Revenue",
    f"${row['Estimated_Revenue_Million']:,.0f} M"
)

c4.metric(
    "Profit",
    f"${row['Estimated_Profit_Million']:,.0f} M"
)

c5, c6, c7, c8 = st.columns(4)

c5.metric(
    "Burn Rate",
    f"${row['Estimated_Burn_Million']:,.0f} M"
)

c6.metric(
    "Runway",
    f"{row['Estimated_Runway_Months']:.1f} Months"
)

c7.metric(
    "ROI",
    f"{row['Estimated_ROI']:.2f}x"
)

if "Financial_Health_Index" in df.columns:
    health = row["Financial_Health_Index"]
else:
    health = 0

c8.metric(
    "Health Index",
    f"{health:.1f}"
)

st.divider()

# -------------------------------------------------
# COMPANY DETAILS
# -------------------------------------------------

left, right = st.columns([1, 1])

with left:

    st.subheader("Company Profile")

    profile = pd.DataFrame({

        "Field":[
            "Country",
            "City",
            "Industry",
            "Funding",
            "Investors",
            "Company Age"
        ],

        "Value":[
            row["Country"],
            row["City"],
            row["Industry"],
            row["Funding_Million"],
            row["Investors Count"],
            row["Company Age"]
        ]

    })

    st.table(profile)

with right:

    st.subheader("Financial Summary")

    finance = pd.DataFrame({

        "Metric":[
            "Revenue",
            "Profit",
            "Burn",
            "Runway",
            "Capital Utilization",
            "Cash Efficiency"
        ],

        "Value":[
            round(row["Estimated_Revenue_Million"],2),
            round(row["Estimated_Profit_Million"],2),
            round(row["Estimated_Burn_Million"],2),
            round(row["Estimated_Runway_Months"],2),
            round(row["Capital_Utilization"],2),
            round(row["Cash_Efficiency"],2)
        ]

    })

    st.table(finance)

st.divider()

# -------------------------------------------------
# CURRENT VS FORECAST
# -------------------------------------------------

fig = go.Figure()

fig.add_bar(
    name="Current",
    x=["Revenue", "Valuation"],
    y=[
        row["Estimated_Revenue_Million"],
        row["Valuation_Million"]
    ]
)

fig.add_bar(
    name="Forecast",
    x=["Revenue", "Valuation"],
    y=[
        row["Expected_Revenue_12M"],
        row["Expected_Valuation_12M"]
    ]
)

fig.update_layout(
    title="Current vs AI Forecast",
    barmode="group",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# -------------------------------------------------
# RAW RECORD
# -------------------------------------------------

st.subheader("Dataset Record")

st.dataframe(
    row.to_frame().T,
    use_container_width=True
)