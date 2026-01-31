# Customer Churn Prediction – XGBoost & SHAP

## 📌 Overview
This project builds an end-to-end machine learning solution to predict customer churn.
It combines predictive modeling with explainability and a Streamlit web application.

## 🎯 Business Problem
Customer churn is costly. Identifying at-risk customers early enables targeted
retention strategies and reduces revenue loss.

## 🧠 Approach
- Data preprocessing & feature engineering
- XGBoost baseline model
- SMOTE to handle class imbalance
- Model evaluation using Recall, F1, ROC-AUC, PR-AUC
- SHAP for global & individual explainability
- Streamlit app for deployment

## 📊 Model Performance (Churn Class = 1)

| Model | Accuracy | ROC-AUC | PR-AUC | Recall | F1 |
|------|----------|--------|-------|--------|----|
| XGBoost Baseline | 0.86 | 0.853 | 0.676 | 0.51 | 0.59 |
| XGBoost + SMOTE | 0.85 | 0.849 | 0.670 | 0.55 | 0.59 |

## 🔍 Key Drivers of Churn (SHAP)
- Age
- Number of Products
- Inactive Membership
- Tenure
- Balance

SHAP was used to explain **why** customers churn, both globally and at individual level.

## 🚀 Streamlit App
The app allows users to:
- Enter customer details
- Predict churn probability
- Adjust decision threshold
- View key drivers of churn

## 🛠 Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, SHAP, Streamlit, Matplotlib

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
python train_and_save.py
streamlit run app.py
