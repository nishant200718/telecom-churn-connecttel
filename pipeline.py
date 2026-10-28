import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import numpy as np
import pandas as pd

CATEGORICAL_CATEGORIES = {
    'gender': ['Female', 'Male'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No', 'No phone service', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'No internet service', 'Yes'],
    'OnlineBackup': ['No', 'No internet service', 'Yes'],
    'DeviceProtection': ['No', 'No internet service', 'Yes'],
    'TechSupport': ['No', 'No internet service', 'Yes'],
    'StreamingTV': ['No', 'No internet service', 'Yes'],
    'StreamingMovies': ['No', 'No internet service', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
}

class ConnectTelChurnPipeline:
    """
    ConnectTel End-to-End Production Inference Pipeline
    Encapsulates preprocessing, feature alignment, standard scaling, and probability risk scoring.
    Supports single-row, batch, and streaming inference payloads with deterministic dummy encoding.
    """
    def __init__(self, scaler, model, feature_names, num_cols, business_threshold=0.28):
        self.scaler = scaler
        self.model = model
        self.feature_names = feature_names
        self.num_cols = num_cols
        self.business_threshold = business_threshold
        self.service_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        self.cat_categories = CATEGORICAL_CATEGORIES

    def preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df_clean = raw_df.copy()
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop(columns=['customerID'])
        if 'Churn' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Churn'])
            
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(
                df_clean['TotalCharges'].replace(' ', np.nan), errors='coerce'
            ).fillna(0.0)
        else:
            df_clean['TotalCharges'] = 0.0

        if 'tenure' in df_clean.columns and 'MonthlyCharges' in df_clean.columns:
            df_clean['TotalChargesPerTenure'] = np.where(
                df_clean['tenure'] > 0, 
                df_clean['TotalCharges'] / df_clean['tenure'], 
                df_clean['MonthlyCharges']
            )
        elif 'TotalChargesPerTenure' not in df_clean.columns:
            df_clean['TotalChargesPerTenure'] = df_clean['TotalCharges']
            
        present_service_cols = [c for c in self.service_cols if c in df_clean.columns]
        if present_service_cols:
            df_clean['ServiceCount'] = (df_clean[present_service_cols] == 'Yes').sum(axis=1)
        elif 'ServiceCount' not in df_clean.columns:
            df_clean['ServiceCount'] = 0
            
        # Ensure categorical columns retain all categories for deterministic dummy generation
        for col, cats in self.cat_categories.items():
            if col in df_clean.columns:
                df_clean[col] = pd.Categorical(df_clean[col], categories=cats)
                
        df_encoded = pd.get_dummies(df_clean, drop_first=True)
        
        # Reindex to ensure strictly identical feature space matching training
        df_aligned = pd.DataFrame(0.0, index=df_clean.index, columns=self.feature_names)
        for col in df_encoded.columns:
            if col in self.feature_names:
                df_aligned[col] = df_encoded[col].astype(float)
                
        # Direct fallback: check any dummy feature directly from raw values if not caught
        for feat in self.feature_names:
            if feat not in df_encoded.columns:
                for col in self.cat_categories:
                    prefix = f"{col}_"
                    if feat.startswith(prefix) and col in df_clean.columns:
                        val = feat[len(prefix):]
                        df_aligned[feat] = (df_clean[col].astype(str) == val).astype(float)
                        
        df_aligned[self.num_cols] = self.scaler.transform(df_aligned[self.num_cols])
        return df_aligned

    def predict_risk(self, raw_df: pd.DataFrame) -> list:
        X_ready = self.preprocess(raw_df)
        probs = self.model.predict_proba(X_ready)[:, 1]
        
        results = []
        for i, p in enumerate(probs):
            if p >= 0.60:
                tier = "CRITICAL RISK"
                action = "Immediate VIP callback + 1-Year contract $15/mo discount"
            elif p >= self.business_threshold:
                tier = "MODERATE RISK"
                action = "Automated email loyalty bundle + Tech Support credit"
            else:
                tier = "LOW RISK"
                action = "Standard retention engagement / Autopay promotion"
                
            results.append({
                "customer_index": i,
                "churn_probability": round(float(p), 4),
                "predicted_churn": bool(p >= self.business_threshold),
                "risk_tier": tier,
                "recommended_retention_action": action
            })
        return results
