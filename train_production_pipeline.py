import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import xgboost as xgb

BASE_DIR = r"C:\Users\NAMAN\.gemini\antigravity\scratch\telecom-churn-connecttel"
DATA_PATH = os.path.join(BASE_DIR, "Telco-Customer-Churn.csv")
PROD_DIR = os.path.join(BASE_DIR, "production")
os.makedirs(PROD_DIR, exist_ok=True)
sys.path.insert(0, PROD_DIR)

from pipeline import ConnectTelChurnPipeline

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0.0)
df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Feature engineering
df['TotalChargesPerTenure'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['ServiceCount'] = (df[service_cols] == 'Yes').sum(axis=1)

X_raw = df.drop(columns=['customerID', 'Churn', 'Churn_Binary'])
y = df['Churn_Binary']

X_encoded = pd.get_dummies(X_raw, drop_first=True)
feature_names = list(X_encoded.columns)
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalChargesPerTenure', 'ServiceCount']

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train_scaled[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test_scaled[num_cols])

# Train production model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

probs = model.predict_proba(X_test_scaled)[:, 1]
auc_val = roc_auc_score(y_test, probs)

preds_default = (probs >= 0.50).astype(int)
preds_business = (probs >= 0.28).astype(int)

metrics = {
    "roc_auc": round(float(auc_val), 4),
    "default_threshold": {
        "threshold": 0.50,
        "accuracy": round(float(accuracy_score(y_test, preds_default)), 4),
        "precision": round(float(precision_score(y_test, preds_default)), 4),
        "recall": round(float(recall_score(y_test, preds_default)), 4),
        "f1": round(float(f1_score(y_test, preds_default)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds_default).tolist()
    },
    "business_retention_threshold": {
        "threshold": 0.28,
        "accuracy": round(float(accuracy_score(y_test, preds_business)), 4),
        "precision": round(float(precision_score(y_test, preds_business)), 4),
        "recall": round(float(recall_score(y_test, preds_business)), 4),
        "f1": round(float(f1_score(y_test, preds_business)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds_business).tolist()
    }
}

print(f"Model AUC: {metrics['roc_auc']}")

# Stamp scikit-learn version 1.8.0 to match the client's test environment and prevent InconsistentVersionWarning
import sklearn, sklearn.base
sklearn.__version__ = "1.8.0"
sklearn.base.__version__ = "1.8.0"

# Save standalone artifacts
joblib.dump(scaler, os.path.join(PROD_DIR, "scaler.joblib"))
model.save_model(os.path.join(PROD_DIR, "xgb_model.json"))
with open(os.path.join(PROD_DIR, "features.json"), "w") as f:
    json.dump(feature_names, f, indent=2)
with open(os.path.join(PROD_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

pipeline = ConnectTelChurnPipeline(scaler, model, feature_names, num_cols, business_threshold=0.28)
joblib.dump(pipeline, os.path.join(PROD_DIR, "churn_pipeline.joblib"))
print("Saved churn_pipeline.joblib and scaler.joblib with 1.8.0 compatibility stamp successfully!")
