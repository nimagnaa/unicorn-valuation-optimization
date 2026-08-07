from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import shap


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Explainability",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "regression"
    / "xgboost.pkl"
)

CSS_PATH = (
    PROJECT_ROOT
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

st.title("Model Explainability")

st.caption(
    "Feature-level explanation of the XGBoost valuation model."
)

st.divider()


# =========================================================
# CHECK FILES
# =========================================================

if not DATA_PATH.exists():

    st.error(
        "Forecast dataset was not found."
    )

    st.stop()


if not MODEL_PATH.exists():

    st.error(
        "XGBoost model was not found."
    )

    st.write(
        "Expected location:"
    )

    st.code(
        str(MODEL_PATH)
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)


# =========================================================
# PREPARE FEATURES
# =========================================================

TARGET = "Expected_Valuation_12M"

DROP_COLUMNS = [

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


DROP_COLUMNS = [
    col
    for col in DROP_COLUMNS
    if col in df.columns
]


X = df.drop(
    columns=DROP_COLUMNS
)


# =========================================================
# ENCODE
# =========================================================

X = pd.get_dummies(
    X,
    drop_first=True
)


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

X = X.fillna(0)


# =========================================================
# ALIGN WITH MODEL FEATURES
# =========================================================

if hasattr(model, "feature_names_in_"):

    model_features = list(
        model.feature_names_in_
    )

    X = X.reindex(
        columns=model_features,
        fill_value=0
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader(
    "Global Feature Importance"
)

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance

})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(20)
)

importance_df["Importance"] = (
    importance_df["Importance"]
    .round(4)
)


fig = px.bar(

    importance_df.sort_values(
        "Importance",
        ascending=True
    ),

    x="Importance",

    y="Feature",

    orientation="h",

    title="Top 20 Features Driving Valuation Prediction"

)

fig.update_layout(

    height=650,

    plot_bgcolor="black",

    paper_bgcolor="black",

    yaxis_title="",

    xaxis_title="Model Importance"

)

st.plotly_chart(

    fig,

    use_container_width=True

)


st.divider()


# =========================================================
# COMPANY-LEVEL EXPLANATION
# =========================================================

st.subheader(
    "Company-Level Explanation"
)

company = st.selectbox(

    "Select Company",

    sorted(
        df["Company"].dropna().unique()
    )

)


company_index = df.index[
    df["Company"] == company
][0]


company_position = list(
    df.index
).index(company_index)


company_X = X.iloc[
    company_position:
    company_position + 1
]


# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(
    company_X
)[0]

actual_target = df.loc[
    company_index,
    TARGET
]


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(

        "Model Prediction",

        f"${prediction:,.0f} M"

    )


with c2:

    st.metric(

        "Forecast Dataset Value",

        f"${actual_target:,.0f} M"

    )


with c3:

    difference = (
        prediction - actual_target
    )

    st.metric(

        "Prediction Difference",

        f"${difference:,.0f} M"

    )


# =========================================================
# SHAP
# =========================================================

st.subheader(
    "Why the Model Made This Prediction"
)

try:

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        company_X
    )

    values = shap_values.values[0]

    shap_df = pd.DataFrame({

        "Feature": company_X.columns,

        "SHAP Value": values,

        "Feature Value": company_X.iloc[0].values

    })

    shap_df["Absolute Impact"] = (
        shap_df["SHAP Value"]
        .abs()
    )

    shap_df = (
        shap_df
        .sort_values(
            "Absolute Impact",
            ascending=False
        )
        .head(15)
    )

    shap_df["Impact"] = np.where(

        shap_df["SHAP Value"] >= 0,

        "Increases Prediction",

        "Decreases Prediction"

    )


    fig = px.bar(

        shap_df.sort_values(
            "SHAP Value"
        ),

        x="SHAP Value",

        y="Feature",

        color="Impact",

        orientation="h",

        title=f"Prediction Drivers — {company}"

    )

    fig.update_layout(

        height=600,

        plot_bgcolor="black",

        paper_bgcolor="black",

        yaxis_title="",

        xaxis_title="SHAP Contribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


    st.subheader(
        "Top Prediction Drivers"
    )

    display = shap_df[
        [
            "Feature",
            "Feature Value",
            "SHAP Value",
            "Impact"
        ]
    ].copy()

    display["SHAP Value"] = (
        display["SHAP Value"]
        .round(4)
    )

    st.dataframe(

        display,

        use_container_width=True,

        hide_index=True

    )


except Exception as e:

    st.warning(
        "SHAP explanation could not be generated for this model."
    )

    st.write(
        str(e)
    )


# =========================================================
# INTERPRETATION
# =========================================================

st.divider()

st.subheader(
    "Interpretation"
)

top_features = importance_df.head(5)

st.write(
    """
    The model uses multiple financial, growth and business
    characteristics to estimate future valuation. The feature
    importance analysis shows which variables contribute most
    strongly to the model's predictions.
    """
)

for _, r in top_features.iterrows():

    st.write(
        f"**{r['Feature']}** — "
        f"importance {r['Importance']:.4f}"
    )