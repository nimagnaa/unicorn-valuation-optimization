\# Unicorn Valuation Optimization Platform



An end-to-end machine learning platform for analyzing, forecasting, and optimizing the financial performance of unicorn companies.



The project combines financial estimation, feature engineering, regression models, forecasting, explainability, and ML-based optimization into one decision-support platform.



\---



\## Project Overview



The objective of this project is to answer a practical business question:



> How can a unicorn company improve its financial performance and expected valuation using data-driven recommendations?



The platform analyzes company-level financial and operational information and produces quantitative recommendations for:



\- Expected valuation

\- Revenue

\- Burn rate

\- Runway

\- Capital utilization

\- ROI

\- Down-round risk

\- Financial sustainability



The final results are presented through an interactive Streamlit dashboard.



\---



\## Key Features



\### Financial Estimation



The system estimates financial indicators from available company-level information.



Key metrics include:



\- Estimated Revenue

\- Estimated Employees

\- Estimated Profit

\- Operating Cost

\- Burn Rate

\- Runway

\- ROI

\- Cash Efficiency

\- Burn Efficiency

\- Capital Utilization

\- Revenue per Employee

\- Profit per Employee

\- Funding per Employee



\---



\### Feature Engineering



Additional business and growth features are generated from the financial dataset.



Examples include:



\- Funding Efficiency

\- Funding Velocity

\- Capital Raised per Year

\- Valuation per Investor

\- Employee Growth

\- Growth Momentum

\- Layoff Indicators

\- Company Age

\- Years Since Unicorn



\---



\## Machine Learning



Five regression models were evaluated:



1\. Linear Regression

2\. Random Forest

3\. XGBoost

4\. LightGBM

5\. CatBoost



The models were evaluated using:



\- MAE

\- RMSE

\- MAPE

\- R²

\- Cross-Validation R²



\### Model Performance



| Model | MAE | RMSE | MAPE | R² | CV R² |

|---|---:|---:|---:|---:|---:|

| XGBoost | 603.14 | 2603.53 | 3.14% | 0.9920 | 0.8962 |

| Random Forest | 1171.05 | 4119.54 | 4.06% | 0.9799 | 0.8949 |

| CatBoost | 1351.27 | 6962.15 | 5.09% | 0.9426 | 0.8308 |

| Linear Regression | 3823.06 | 7146.52 | 45.76% | 0.9395 | 0.2593 |

| LightGBM | 2514.47 | 10736.63 | 9.17% | 0.8635 | 0.8094 |



XGBoost was selected as the final regression model based on the combination of test-set performance and cross-validation performance.



\---



\## Target Leakage Control



A major part of the modeling process was identifying and removing target leakage.



The prediction target was:



`Expected\_Valuation\_12M`



Features mathematically derived from the target were removed from the model input.



This was important because an artificially high R² caused by leakage would not represent genuine predictive performance.



The revised model was evaluated again using both a held-out test set and cross-validation.



\---



\## Hyperparameter Optimization



Optuna was used to tune:



\- XGBoost

\- LightGBM

\- CatBoost



The objective was to improve predictive performance while avoiding overfitting.



The project therefore compares:



\- Untuned models

\- Tuned models



This provides a quantitative view of the effect of hyperparameter optimization.



\---



\## Forecasting Engine



The forecasting engine generates 12-month projections including:



\- Expected Revenue

\- Expected Profit

\- Expected Burn

\- Expected Runway

\- Expected Valuation

\- Down-Round Indicator



The forecasting stage combines company characteristics with industry, country, and macroeconomic assumptions.



\---



\## ML Optimization Engine



The optimization engine uses the trained ML model to create a current-versus-optimized scenario.



For each company it generates:



\- Current Valuation

\- Optimized Valuation

\- Valuation Change

\- Current Revenue

\- Recommended Revenue

\- Current Burn

\- Recommended Burn

\- Current Runway

\- Optimized Runway

\- Current Capital Utilization

\- Optimized Capital Utilization

\- Current ROI

\- Optimized ROI

\- Current Down-Round Indicator

\- Optimized Down-Round Indicator



The objective is to translate model predictions into practical financial recommendations.



\---



\## Explainability



SHAP is used to understand which features contribute most strongly to model predictions.



This helps answer:



> Why did the model predict this valuation?



Instead of treating the ML model as a black box, the platform provides feature-level explanations.



\---



\## Dashboard



The final application is built using Streamlit.



The dashboard contains:



\- Home

\- Company Analysis

\- AI Forecast

\- AI Optimization

\- Model Comparison

\- Explainability

\- Executive Report

\- Project Overview



The interface uses a blue and white professional analytics theme.



\---



\## Project Architecture



```text

unicorn/

│

├── dashboard/

│   ├── app.py

│   ├── style.css

│   ├── utils.py

│   └── pages/

│

├── data/

│   └── raw/

│

├── src/

│   ├── feature\_engineering/

│   ├── explainability/

│   ├── optimization/

│   ├── recommendation/

│   ├── tuning/

│   ├── models/

│   ├── preprocessing.py

│   ├── financial\_estimator.py

│   └── forecasting\_engine.py

│

├── requirements.txt

├── README.md

└── .gitignore

