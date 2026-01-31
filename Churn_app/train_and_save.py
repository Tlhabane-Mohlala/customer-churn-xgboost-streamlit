import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATA_PATH = r"C:\Users\User\Downloads\Churn_Modelling.csv"

df = pd.read_csv(DATA_PATH)

df1 = df[['Geography', 'Gender', 'CreditScore', 'Age', 'Tenure', 'Balance',
          'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']].copy()

df_enc = pd.get_dummies(
    df1,
    columns=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'],
    drop_first=True
)

scale_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
scaler = MinMaxScaler()
df_enc[scale_vars] = scaler.fit_transform(df_enc[scale_vars])

X = df_enc.drop('Exited', axis=1)
y = df_enc['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_res, y_res)

os.makedirs("artifacts", exist_ok=True)

with open("artifacts/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("artifacts/model_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ Saved artifacts in ./artifacts:")
print(os.listdir("artifacts"))
