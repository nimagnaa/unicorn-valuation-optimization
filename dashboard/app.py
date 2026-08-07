from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="AI Unicorn Valuation Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

css_path = BASE_DIR / "style.css"

if css_path.exists():

    with open(css_path, encoding="utf-8") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

st.markdown(
"""
<div class="main-title">
AI Unicorn Valuation Optimization Platform
</div>

<div class="sub-title">
Machine Learning Based Financial Forecasting and Business Optimization
</div>
""",
unsafe_allow_html=True
)

st.write(
"""
Welcome to the AI Unicorn Valuation Platform.

Use the navigation panel on the left to explore:

- Home Dashboard
- Company Analysis
- AI Forecast
- AI Optimization
- Model Comparison
- Explainability
- Executive Report
"""
)