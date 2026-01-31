import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

def load_pickle(path: str):
    if not os.path.exists(path):
        st.error(f"❌ Missing file: {path}")
        st.info("Fix: Run `python train_and_save.py` first.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_artifacts():
    model = load_pickle("artifacts/xgb_model.pkl")
    scaler = load_pickle("artifacts/scaler.pkl")
    model_columns = load_pickle("artifacts/model_columns.pkl")
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

st.sidebar.title("Customer Inputs")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 650)
age = st.sidebar.number_input("Age", 18, 100, 40)
tenure = st.sidebar.number_input("Tenure (years)", 0, 10, 3)
balance = st.sidebar.number_input("Balance", 0.0, 1_000_000.0, 60000.0)
num_products = st.sidebar.number_input("Num Of Products", 1, 4, 2)
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 1_000_000.0, 80000.0)

geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
has_crcard = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])

threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.01)

input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_crcard,
    "IsActiveMember": is_active,
    "EstimatedSalary": estimated_salary
}])

encoded = pd.get_dummies(
    input_df,
    columns=["Geography", "Gender", "HasCrCard", "IsActiveMember"],
    drop_first=True
)

encoded = encoded.reindex(columns=model_columns, fill_value=0)

scale_vars = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
encoded[scale_vars] = scaler.transform(encoded[scale_vars])

st.title("Customer Churn Prediction (XGBoost + SMOTE)")
st.write("Fill in the customer details on the left, then click **Predict Churn**.")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Input Data (Raw)")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("Prediction Result")
    if st.button("Predict Churn"):
        prob = model.predict_proba(encoded)[0][1]
        prediction = 1 if prob >= threshold else 0

        st.metric("Churn Probability", f"{prob:.2%}")
        st.caption(f"Threshold used: {threshold:.2f}")

        if prediction == 1:
            st.error("Prediction: Exited (High churn risk)")
        else:
            st.success("Prediction: Not Exited (Low churn risk)")

st.markdown("---")
st.subheader("Top Drivers (Feature Importance)")

importances = model.feature_importances_
fi_df = pd.DataFrame({"Feature": model_columns, "Importance": importances}) \
    .sort_values("Importance", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(fi_df["Feature"], fi_df["Importance"])
ax.set_ylabel("Importance")
ax.set_xticklabels(fi_df["Feature"], rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)
st.caption("Tip: Lower threshold (0.35–0.45) catches more churners but increases false alarms.")
