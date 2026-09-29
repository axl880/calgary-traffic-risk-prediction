"""
Script 4/5 - Train the risk classification model
Project: Calgary Traffic Collision Risk Prediction

IMPORTANT: only road-network features are used as predictors. Features
derived from the incidents themselves (pct_lane_blocked, pct_*_collision,
avg_vehicle_count, total_incidents) are deliberately EXCLUDED to avoid
data leakage.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

BASE_DIR = r"C:\Portafolio\calgary-traffic-risk"
INPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "grid_features_final.csv")
MODEL_OUT = os.path.join(BASE_DIR, "data", "processed", "random_forest_model.joblib")

df = pd.read_csv(INPUT_CSV)
print(f"Total cells: {len(df)}")

# ============================================================
# Define the problem: binary high-risk classification
# ============================================================
# 1 = cell with High or Very_high risk | 0 = everything else (Low, Medium, No_data)
df["high_risk"] = df["risk_level"].isin(["High", "Very_high"]).astype(int)
print(f"\nTarget distribution (high_risk):")
print(df["high_risk"].value_counts())
print(f"Positive class proportion: {df['high_risk'].mean()*100:.1f}%")

# ============================================================
# Features: ONLY independent variables (no data leakage)
# ============================================================
NUMERIC_FEATURES = [
    "total_road_length_m",
    "pct_arterial_road",
    "num_road_segments",
    "road_density",
]
CATEGORICAL_FEATURE = "quadrant"

X = df[NUMERIC_FEATURES].copy()
X = X.join(pd.get_dummies(df[CATEGORICAL_FEATURE], prefix="quad", drop_first=True))
y = df["high_risk"]

print(f"\nFeatures used: {list(X.columns)}")

# ============================================================
# Train/test split (stratified to preserve class proportions)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTraining: {len(X_train)} cells | Test: {len(X_test)} cells")

# ============================================================
# Model 1: Logistic Regression (baseline)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_scaled, y_train)
pred_log = log_reg.predict(X_test_scaled)
proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n" + "=" * 60)
print("MODEL 1: Logistic Regression (baseline)")
print("=" * 60)
print(classification_report(y_test, pred_log, target_names=["Not high risk", "High risk"]))
print(f"AUC-ROC: {roc_auc_score(y_test, proba_log):.3f}")

# ============================================================
# Model 2: Random Forest
# ============================================================
rf = RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_leaf=15, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)  # Random Forest does not need scaling
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("MODEL 2: Random Forest")
print("=" * 60)
print(classification_report(y_test, pred_rf, target_names=["Not high risk", "High risk"]))
print(f"AUC-ROC: {roc_auc_score(y_test, proba_rf):.3f}")

print("\nConfusion matrix (Random Forest):")
print(confusion_matrix(y_test, pred_rf))

# ============================================================
# Feature importance (Random Forest)
# ============================================================
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Feature importance (Random Forest) ===")
print(importances)

# ============================================================
# Save the best model
# ============================================================
joblib.dump({"model": rf, "features": list(X.columns)}, MODEL_OUT)
print(f"\nModel saved to: {MODEL_OUT}")
