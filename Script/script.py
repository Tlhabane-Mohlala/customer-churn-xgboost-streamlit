# ==============================================================
# CHURN PREDICTION (XGBoost + SMOTE) + EVALUATION + SHAP
# Dataset: Churn_Modelling.csv
# ==============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import shap

%matplotlib inline


# -----------------------------
# Config
# -----------------------------
DATA_PATH = r"C:\Users\mmohlala5\Downloads\Churn_Modelling.csv"
RANDOM_STATE = 42


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Rows:", len(df))
print("Columns:", df.shape[1])
df.head()


# -----------------------------
# Quick profiling (optional)
# -----------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nUnique values (small-cardinality columns):")
for col in df.columns:
    uniq = df[col].fillna("0").unique()
    if len(uniq) < 12:
        print(f"{col}: {len(uniq)} -> {uniq}")
    else:
        print(f"{col}: {len(uniq)}")


# -----------------------------
# Target distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Exited")
plt.title("Target Balance (Exited)")
plt.xlabel("Exited")
plt.ylabel("Count")
plt.xticks([0, 1], ["Not Exited", "Exited"])
plt.tight_layout()
plt.show()


# -----------------------------
# Feature selection
# -----------------------------
selected_cols = [
    "Geography", "Gender", "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
]
df1 = df[selected_cols].copy()
df1.head()


# -----------------------------
# Visual checks (optional)
# -----------------------------
# Pairplot (can be slow on some machines)
sns.pairplot(
    df1[["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary","Exited"]],
    hue="Exited"
)
plt.show()

# Categorical/numeric comparisons vs target
features_to_plot = [
    "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

for f in features_to_plot:
    plt.figure(figsize=(9, 5))
    sns.countplot(x=f, data=df1, hue="Exited")
    plt.title(f"{f} vs Exited")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# Boxplots for numeric columns
num_cols = df1.select_dtypes(include=["int64", "float64"]).columns
for c in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df1[c])
    plt.title(f"{c} (median: {df1[c].median():.2f})")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Preprocessing: One-hot + scaling
# -----------------------------
df_model = pd.get_dummies(
    df1,
    columns=["Geography", "Gender", "HasCrCard", "IsActiveMember"],
    drop_first=False
)

scale_vars = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
scaler = MinMaxScaler()
df_model[scale_vars] = scaler.fit_transform(df_model[scale_vars])

X = df_model.drop("Exited", axis=1)
y = df_model["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain shape:", X_train.shape, "| Test shape:", X_test.shape)


# -----------------------------
# Train: Baseline XGBoost
# -----------------------------
xgb_model = XGBClassifier(
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

xgb_model.fit(X_train, y_train)

y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

print("\nBaseline XGBoost")
print("Training Accuracy:", f"{accuracy_score(y_train, y_train_pred):.4f}")
print("Testing Accuracy :", f"{accuracy_score(y_test, y_test_pred):.4f}")


# Feature importance (baseline)
fi_base = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop Feature Importances (Baseline):")
print(fi_base.head(15))


# -----------------------------
# Train: XGBoost + SMOTE
# -----------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

xgb_model_smote = XGBClassifier(
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

xgb_model_smote.fit(X_resampled, y_resampled)

y_train_pred_smote = xgb_model_smote.predict(X_train)
y_test_pred_smote = xgb_model_smote.predict(X_test)

print("\nXGBoost + SMOTE")
print("Training Accuracy:", f"{accuracy_score(y_train, y_train_pred_smote):.4f}")
print("Testing Accuracy :", f"{accuracy_score(y_test, y_test_pred_smote):.4f}")


# Feature importance plot (SMOTE model)
plt.figure(figsize=(10, 5))
plt.bar(X.columns, xgb_model_smote.feature_importances_, align="center")
plt.title("Feature Importance (XGBoost + SMOTE)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ==============================================================
# Evaluation utilities
# ==============================================================

def evaluate_classifier(model, X_eval, y_eval, y_pred, name="Model"):
    print("\n" + "=" * 70)
    print(f"EVALUATION: {name}")
    print("=" * 70)

    acc = accuracy_score(y_eval, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_eval, y_pred, target_names=["Not Exited (0)", "Exited (1)"]))

    cm = confusion_matrix(y_eval, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Exited", "Exited"])
    disp.plot()
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.show()

    if not hasattr(model, "predict_proba"):
        print("predict_proba not available — skipping ROC/PR curves.")
        return

    y_proba = model.predict_proba(X_eval)[:, 1]

    # ROC
    roc_auc = roc_auc_score(y_eval, y_proba)
    fpr, tpr, _ = roc_curve(y_eval, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_eval, y_proba)
    ap = average_precision_score(y_eval, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve — {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {ap:.4f}")


def summary_metrics(model, X_eval, y_eval, y_pred, name):
    y_proba = model.predict_proba(X_eval)[:, 1]
    report = classification_report(y_eval, y_pred, output_dict=True)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_eval, y_pred),
        "ROC_AUC": roc_auc_score(y_eval, y_proba),
        "PR_AUC": average_precision_score(y_eval, y_proba),
        "Churn_Recall(1)": report["1"]["recall"],
        "Churn_Precision(1)": report["1"]["precision"],
        "Churn_F1(1)": report["1"]["f1-score"],
    }


# -----------------------------
# Evaluate both models
# -----------------------------
evaluate_classifier(xgb_model, X_test, y_test, y_test_pred, "XGBoost (Baseline)")
evaluate_classifier(xgb_model_smote, X_test, y_test, y_test_pred_smote, "XGBoost + SMOTE")

summary_df = pd.DataFrame([
    summary_metrics(xgb_model, X_test, y_test, y_test_pred, "XGBoost (Baseline)"),
    summary_metrics(xgb_model_smote, X_test, y_test, y_test_pred_smote, "XGBoost + SMOTE"),
]).sort_values(by="Churn_Recall(1)", ascending=False)

print("\n" + "=" * 70)
print("SUMMARY COMPARISON (focus: churn recall/F1/ROC/PR)")
print("=" * 70)
summary_df


# ==============================================================
# SHAP Explainability (use SMOTE model by default)
# ==============================================================

print("\nX_test shape:", X_test.shape)
X_explain = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)

model = xgb_model_smote
explainer = shap.TreeExplainer(model)

# SHAP version compatibility: new API vs old API
try:
    shap_exp = explainer(X_explain)
    shap_values = shap_exp.values
except Exception:
    shap_values = explainer.shap_values(X_explain)
    shap_exp = None

print("SHAP values shape:", np.array(shap_values).shape)

# Global importance (bar)
plt.figure()
if shap_exp is not None:
    shap.plots.bar(shap_exp, max_display=15)
else:
    shap.summary_plot(shap_values, X_explain, plot_type="bar", max_display=15)

# Global summary (beeswarm)
plt.figure()
if shap_exp is not None:
    shap.plots.beeswarm(shap_exp, max_display=15)
else:
    shap.summary_plot(shap_values, X_explain, max_display=15)

# Local explanation: one customer
shap.initjs()
customer_index = 5
x_row = X_explain.iloc[customer_index:customer_index + 1]

if shap_exp is not None:
    shap.force_plot(
        shap_exp.base_values[customer_index],
        shap_exp.values[customer_index],
        X_explain.iloc[customer_index],
        matplotlib=False
    )
else:
    base_value = explainer.expected_value
    shap.force_plot(
        base_value,
        shap_values[customer_index],
        X_explain.iloc[customer_index],
        matplotlib=False
    )

# Waterfall plot
plt.figure()
if shap_exp is not None:
    shap.plots.waterfall(shap_exp[customer_index], max_display=15)
else:
    exp = shap.Explanation(
        values=shap_values[customer_index],
        base_values=explainer.expected_value,
        data=X_explain.iloc[customer_index].values,
        feature_names=X_explain.columns.tolist()
    )
    shap.plots.waterfall(exp, max_display=15)

# Dependence plot (choose a real feature name)
feature_name = X_explain.columns[0]  # change to a top feature from SHAP bar plot
plt.figure()
if shap_exp is not None:
    shap.dependence_plot(feature_name, shap_exp.values, X_explain, interaction_index=None)
else:
    shap.dependence_plot(feature_name, shap_values, X_explain, interaction_index=None)

# Export SHAP global importance
abs_shap = np.abs(np.array(shap_values))
mean_abs_shap = abs_shap.mean(axis=0)

shap_importance_df = pd.DataFrame({
    "Feature": X_explain.columns,
    "MeanAbsSHAP": mean_abs_shap
}).sort_values("MeanAbsSHAP", ascending=False)

print("\nTop SHAP importances:")
print(shap_importance_df.head(20))
# shap_importance_df.to_csv("shap_feature_importance.csv", index=False)
