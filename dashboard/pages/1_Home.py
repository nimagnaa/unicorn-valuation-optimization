from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Home",
    layout="wide"
)

# -------------------------------
# CSS
# -------------------------------

css = Path(__file__).parents[1] / "style.css"

if css.exists():
    with open(css, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# -------------------------------
# LOAD DATA
# -------------------------------

ROOT = Path(__file__).parents[2]

df = pd.read_csv(
    ROOT / "data" / "processed" / "forecast_dataset.csv"
)

# -------------------------------
# HEADER
# -------------------------------

st.title("Dashboard Overview")

st.caption(
    "Overview of unicorn companies, valuation, revenue, growth and financial performance."
)

st.divider()

# -------------------------------
# KPIs
# -------------------------------

col1,col2,col3,col4,col5 = st.columns(5)

col1.metric(
    "Companies",
    f"{len(df):,}"
)

col2.metric(
    "Avg Valuation",
    f"${df['Valuation_Million'].mean():,.0f} M"
)

col3.metric(
    "Avg Revenue",
    f"${df['Estimated_Revenue_Million'].mean():,.0f} M"
)

col4.metric(
    "Avg ROI",
    f"{df['Estimated_ROI'].mean():.2f}x"
)

col5.metric(
    "Avg Runway",
    f"{df['Estimated_Runway_Months'].mean():.1f} Months"
)

st.divider()

# -------------------------------
# CHART ROW 1
# -------------------------------


def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="#172033",
            family="Arial"
        ),
        title_font=dict(
            color="#123b70",
            size=18
        ),
        xaxis=dict(
            title_font=dict(color="#172033"),
            tickfont=dict(color="#425466"),
            gridcolor="#d9e2ec",
            linecolor="#aebdcc"
        ),
        yaxis=dict(
            title_font=dict(color="#172033"),
            tickfont=dict(color="#425466"),
            gridcolor="#d9e2ec",
            linecolor="#aebdcc"
        ),
        legend=dict(
            font=dict(color="#172033")
        )
    )

    return fig

left,right = st.columns(2)

industry = (
    df["Industry"]
    .value_counts()
    .head(10)
    .reset_index()
)

industry.columns=["Industry","Companies"]

fig = px.bar(
    industry,
    x="Industry",
    y="Companies",
    color="Companies",
    title="Top Industries"
)

fig.update_layout(height=420)

left.plotly_chart(
    fig,
    use_container_width=True
)

country = (
    df["Country"]
    .value_counts()
    .head(10)
    .reset_index()
)

country.columns=["Country","Companies"]

fig = px.bar(
    country,
    x="Country",
    y="Companies",
    color="Companies",
    title="Top Countries"
)

fig.update_layout(height=420)

right.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------------
# CHART ROW 2
# -------------------------------

left,right = st.columns(2)

fig = px.scatter(
    df,
    x="Estimated_Revenue_Million",
    y="Expected_Valuation_12M",
    color="Industry",
    hover_name="Company",
    title="Revenue vs Expected Valuation"
)

fig.update_layout(height=420)

left.plotly_chart(
    fig,
    use_container_width=True
)

fig = px.histogram(
    df,
    x="Valuation_Million",
    nbins=35,
    title="Valuation Distribution"
)

fig.update_layout(height=420)

right.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------------
# TABLE
# -------------------------------

st.subheader("Top Companies")

top = (
    df.sort_values(
        "Expected_Valuation_12M",
        ascending=False
    )
    [["Company",
      "Country",
      "Industry",
      "Valuation_Million",
      "Expected_Valuation_12M",
      "Estimated_ROI"]]
    .head(20)
)

st.dataframe(
    top,
    use_container_width=True,
    hide_index=True
)