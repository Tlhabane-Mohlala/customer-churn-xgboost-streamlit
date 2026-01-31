# Customer Churn Prediction with XGBoost, SHAP & Streamlit

🚀 **End-to-end data science project** focused on predicting customer churn and explaining *why* customers leave using explainable AI, with a production-ready Streamlit application.

---

## 📌 Business Context

Customer churn has a direct impact on revenue and long-term growth.  
Organizations need not only accurate churn predictions, but also **clear explanations** to support retention strategies and executive decision-making.

This project delivers:
- Accurate churn prediction
- Transparent model explanations (no black boxes)
- A deployable application for business users

---

## 🎯 Objectives

- Predict customer churn using machine learning  
- Handle class imbalance effectively  
- Evaluate models using churn-focused metrics  
- Explain predictions at both global and individual levels  
- Deploy the solution as an interactive web application  

---

## 🧠 Methodology

### Data Preparation
- Feature selection and cleaning
- One-hot encoding of categorical variables
- Min-Max scaling of numerical features

### Modeling
- Baseline **XGBoost** classifier
- **SMOTE** applied to address class imbalance
- Threshold-based decision logic for business flexibility

### Evaluation Metrics
- Accuracy
- ROC-AUC
- Precision-Recall AUC
- Recall (Churn class)
- F1-Score (Churn class)

---

## 📊 Model Performance (Churn = 1)

| Model              | Accuracy | ROC-AUC | PR-AUC | Recall | F1 |
|--------------------|----------|---------|--------|--------|----|
| XGBoost Baseline   | 0.86     | 0.853   | 0.676  | 0.51   | 0.59 |
| XGBoost + SMOTE    | 0.85     | 0.849   | 0.670  | 0.55   | 0.59 |

📌 **Insight:**  
SMOTE improves churn recall, enabling the model to capture more at-risk customers with a minimal trade-off in accuracy.

---

## 🔍 Explainability with SHAP (Why Customers Churn)

Most churn models are difficult to interpret.  
This project uses **SHAP (SHapley Additive exPlanations)** to ensure full transparency.

### Global Explainability
Key drivers of churn:
- Age
- Number of Products
- Inactive Membership
- Tenure
- Account Balance

SHAP summary plots show how each feature influences churn risk across the customer base.

### Individual Explainability
SHAP waterfall plots explain:
- Why a specific customer is predicted to churn
- Which features increase or reduce churn risk

This level of explainability is critical for:
- Executive trust
- Regulatory environments
- Targeted retention actions

---

## 🖥️ Streamlit Application

A Streamlit web application allows users to:

- Input customer details
- View churn probability
- Adjust the decision threshold
- Interpret predictions using feature importance

The app makes advanced analytics accessible to non-technical stakeholders.

---
