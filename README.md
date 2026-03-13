# Credit-risk-modelling-and-Loan-Default-Prediction

## Overview
This repository contains a quantitative model developed to assess consumer credit risk and optimize risk-adjusted returns using historical loan tape data from LendingClub. By transitioning from baseline linear forecasting to advanced ensemble methods, this project aims to identify mispriced credit risk and maximize portfolio yield.

## Project Structure
* **`notebooks/`**: Contains the quantitative research, feature engineering, and model training workflows.
* **`outputs/`**: Contains the final datasets with predicted Probability of Default (PD) and expected loan yields.

## Investment & Modeling Methodology
The analysis was conducted in three distinct phases, mirroring a real-world quantitative research workflow:

### 1. Baseline Yield Forecasting
* **Objective:** Establish baseline expected returns across the loan portfolio to identify initial macro-trends in borrower data.
* **Techniques:** Linear Regression, Ridge, Lasso, and ElasticNet.
* **File:** `Loan-Default-Prediction-Linear-Models.ipynb`

### 2. Credit Risk & Probability of Default (PD) Modeling
* **Objective:** Assess downside risk by predicting the binary outcome of a loan being "Charged Off" versus "Fully Paid."
* **Techniques:** Logistic Regression.
* **Quantitative Focus:** Mitigated extreme class imbalances inherent in credit default data using SMOTE, and applied optimal threshold tuning to ensure conservative risk estimates.
* **File:** `Credit-Risk-Modeling-Logistic-Regression-SMOTE.ipynb`

### 3. Risk-Adjusted Return Optimization
* **Objective:** Generate excess yield by incorporating the predicted Probability of Default (PD) as a core feature into advanced non-linear regression models to forecast the final expected return of each loan.
* **Techniques:** XGBoost, LightGBM, and Random Forest regressors.
* **Quantitative Focus:** Implemented an expanding-window regression framework with year-wise feature scaling to preserve temporal consistency, simulate realistic trading environments, and prevent look-ahead bias.
* **File:** `Loan-Return-Optimization-XGBoost-LightGBM.ipynb`

## Technologies Used
* **Languages & Environments:** Python, Google Colab
* **Quantitative Libraries:** Pandas, NumPy, Scikit-Learn, Statsmodels, LightGBM, XGBoost, Imbalanced-Learn

## Key Financial Insights & Results
* Engineered a robust Probability of Default (PD) model capable of isolating high-risk consumer loans within heavily imbalanced datasets.
* Optimized predictive accuracy for continuous loan yields by transitioning to gradient-boosting frameworks (LightGBM/XGBoost), allowing for more precise capital allocation.
* Demonstrated strict adherence to quantitative best practices by eliminating data leakage through expanding-window validation.
