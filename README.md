# 🏦 Bank Customer Churn Prediction & Analytics Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B.svg)

## 📌 Project Overview
Customer churn is a critical metric for financial institutions. This project implements an end-to-end machine learning pipeline to predict whether a bank customer will close their account. 

Beyond just providing a predictive score, this architecture utilizes **SHAP (Shapley Additive exPlanations)** to interpret the model's decisions, providing actionable business insights. The trained model is served via a **FastAPI** backend and consumed by an interactive **Streamlit** frontend dashboard.

## 🚀 Key Features
* **Advanced Preprocessing:** Handles heavy class imbalance (80/20 split) using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **High-Performance Modeling:** Utilizes an **XGBoost** classifier optimized for tabular data, achieving a strong ROC-AUC score.
* **Explainable AI (XAI):** Integrates SHAP to break down the exact feature contributions (e.g., Age, Active Membership) driving each individual churn prediction.
* **Decoupled Full-Stack Architecture:** 
  * A **FastAPI** REST endpoint for model inference.
  * A responsive **Streamlit** UI for real-time risk assessment by bank managers.

## 📂 Project Structure
```text
bank-churn-prediction/
├── data/
│   ├── raw/                 # Original churn.csv dataset
│   └── processed/           # Transformed datasets (optional)
├── models/
│   ├── xgboost_churn_model.pkl  # Serialized XGBoost model
│   └── standard_scaler.pkl      # Fitted feature scaler
├── src/
│   ├── churn_model.ipynb    # EDA, SMOTE, Training, and SHAP analysis
│   ├── app.py               # FastAPI backend server
│   └── frontend.py          # Streamlit user interface
├── requirements.txt         # Project dependencies
└── README.md
