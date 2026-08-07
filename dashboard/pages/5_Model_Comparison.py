from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSS_PATH = PROJECT_ROOT / "dashboard" / "style.css"

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "regression"
    / "regression_results.csv"
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
# LOAD RESULTS
# =========================================================

if not RESULTS_PATH.exists():

    st.error(
        "Model results file was not found."
    )

    st.code(
        "outputs/regression/regression_results.csv"
    )

    st.stop()


results = pd.read_csv(RESULTS_PATH)


# =========================================================
# HEADER
# =========================================================

st.title("Model Comparison")

st.caption(
    "Evaluation of regression models used for 12-month unicorn valuation forecasting."
)

st.divider()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

results.columns = results.columns.str.strip()


# =========================================================
# BEST MODEL
# =========================================================

if "R2" in results.columns:

    best_model = results.loc[
        results["R2"].idxmax(),
        "Model"
    ]

    best_r2 = results["R2"].max()

else:

    best_model = results.iloc[0]["Model"]
    best_r2 = None


if "CV_R2" in results.columns:

    best_cv_model = results.loc[
        results["CV_R2"].idxmax(),
        "Model"
    ]

    best_cv = results["CV_R2"].max()

else:

    best_cv_model = "Not Available"
    best_cv = None


# =========================================================
# KPI CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Best Test Model",
        best_model
    )


with c2:

    if best_r2 is not None:

        st.metric(
            "Best Test R²",
            f"{best_r2:.4f}"
        )

    else:

        st.metric(
            "Best Test R²",
            "N/A"
        )


with c3:

    st.metric(
        "Best Cross-Validation Model",
        best_cv_model
    )


with c4:

    if best_cv is not None:

        st.metric(
            "Best CV R²",
            f"{best_cv:.4f}"
        )

    else:

        st.metric(
            "Best CV R²",
            "N/A"
        )


st.divider()


# =========================================================
# MODEL PERFORMANCE TABLE
# =========================================================

st.subheader("Model Performance")

display_results = results.copy()

numeric_columns = [
    "MAE",
    "RMSE",
    "MAPE",
    "R2",
    "CV_R2"
]

for col in numeric_columns:

    if col in display_results.columns:

        display_results[col] = display_results[col].round(4)


st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# R2 COMPARISON
# =========================================================

st.subheader("Model Accuracy")

if "R2" in results.columns:

    r2_data = results.sort_values(
        "R2",
        ascending=True
    )

    fig = px.bar(
        r2_data,
        x="R2",
        y="Model",
        orientation="h",
        text="R2",
        title="Test R² by Model"
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig.update_layout(
        height=420,
        plot_bgcolor="black",
        paper_bgcolor="black",
        xaxis_title="R²",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# CROSS VALIDATION
# =========================================================

if "CV_R2" in results.columns:

    st.subheader("Cross-Validation Performance")

    cv_data = results.sort_values(
        "CV_R2",
        ascending=True
    )

    fig = px.bar(
        cv_data,
        x="CV_R2",
        y="Model",
        orientation="h",
        text="CV_R2",
        title="Cross-Validated R²"
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig.update_layout(
        height=420,
        plot_bgcolor="black",
        paper_bgcolor="black",
        xaxis_title="CV R²",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# ERROR COMPARISON
# =========================================================

st.subheader("Prediction Error")

left, right = st.columns(2)


with left:

    if "MAE" in results.columns:

        mae_data = results.sort_values(
            "MAE",
            ascending=True
        )

        fig = px.bar(
            mae_data,
            x="MAE",
            y="Model",
            orientation="h",
            title="Mean Absolute Error"
        )

        fig.update_layout(
            height=400,
            plot_bgcolor="black",
            paper_bgcolor="black"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


with right:

    if "RMSE" in results.columns:

        rmse_data = results.sort_values(
            "RMSE",
            ascending=True
        )

        fig = px.bar(
            rmse_data,
            x="RMSE",
            y="Model",
            orientation="h",
            title="Root Mean Squared Error"
        )

        fig.update_layout(
            height=400,
            plot_bgcolor="black",
            paper_bgcolor="black"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# MAPE
# =========================================================

if "MAPE" in results.columns:

    st.subheader("Percentage Error")

    mape_data = results.sort_values(
        "MAPE",
        ascending=True
    )

    fig = px.bar(
        mape_data,
        x="Model",
        y="MAPE",
        text="MAPE",
        title="Mean Absolute Percentage Error"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=420,
        plot_bgcolor="black",
        paper_bgcolor="black",
        yaxis_title="MAPE (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# INTERVIEW INSIGHT
# =========================================================

st.divider()

st.subheader("Model Selection Insight")

if best_cv_model == best_model:

    st.write(
        f"""
        **{best_model}** provides the strongest overall result based on both
        test-set performance and cross-validation performance.

        The model achieved a test R² of **{best_r2:.4f}**
        and a cross-validated R² of **{best_cv:.4f}**.
        """
    )

else:

    st.write(
        f"""
        **{best_model}** achieved the highest test-set R², while
        **{best_cv_model}** achieved the strongest cross-validation R².

        This comparison is useful for evaluating both predictive accuracy
        and model stability across different training splits.
        """
    )


# =========================================================
# DOWNLOAD RESULTS
# =========================================================

st.download_button(

    label="Download Model Results",

    data=results.to_csv(index=False),

    file_name="model_comparison.csv",

    mime="text/csv"

)