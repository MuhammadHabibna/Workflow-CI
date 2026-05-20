"""
modelling.py
============
Training script untuk MLflow Project (Kriteria 3: Workflow CI).

Script ini dijalankan secara otomatis oleh GitHub Actions setiap kali
trigger terpantik. Melatih model Random Forest dengan Hyperparameter Tuning,
mencatat semua parameter dan metrik ke DagsHub via MLflow (Manual Logging),
serta menyimpan model secara lokal untuk keperluan Docker build.

Algoritma : Random Forest Classifier
Dataset   : Online Payments Fraud Detection (preprocessed)
Tracking  : MLflow Manual Logging + DagsHub (online)
Tuning    : GridSearchCV
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import os
import json
import shutil

# ============================================================
# KONFIGURASI
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'fraud_detection_preprocessing')
DAGSHUB_OWNER = "MuhammadHabibna"
DAGSHUB_REPO = "Eksperimen_SML_Muhammad-Habib-Nur-Aiman"

# ============================================================
# [1/7] LOAD DATA
# ============================================================
print("[1/7] Memuat dataset preprocessed...")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()

print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

# ============================================================
# [2/7] CONNECT TO DAGSHUB
# ============================================================
print("[2/7] Menghubungkan ke DagsHub...")
dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
print(f"  Terhubung ke: https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}")

mlflow.set_experiment("Fraud_Detection_CI")

# ============================================================
# [3/7] HYPERPARAMETER TUNING
# ============================================================
print("[3/7] Menjalankan Hyperparameter Tuning (GridSearchCV)...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(f"  Best Parameters: {best_params}")
print(f"  Best CV F1 Score: {grid_search.best_score_:.4f}")

# ============================================================
# [4/7] EVALUASI MODEL TERBAIK
# ============================================================
print("[4/7] Mengevaluasi model terbaik...")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  ROC-AUC  : {roc_auc:.4f}")

# ============================================================
# [5/7] MANUAL LOGGING KE MLFLOW (DAGSHUB)
# ============================================================
print("[5/7] Melakukan Manual Logging ke MLflow (DagsHub)...")

with mlflow.start_run(run_name="RF_CI_Pipeline"):
    # --- Log Parameters ---
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", best_params['n_estimators'])
    mlflow.log_param("max_depth", best_params.get('max_depth', 'None'))
    mlflow.log_param("min_samples_split", best_params['min_samples_split'])
    mlflow.log_param("tuning_method", "GridSearchCV")
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("scoring", "f1")

    # --- Log Metrics ---
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    # --- Log Model ke DagsHub ---
    mlflow.sklearn.log_model(best_model, "model")

    # --- Artefak 1: Confusion Matrix ---
    print("  [Artefak 1] Membuat Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'], ax=ax1)
    ax1.set_title('Confusion Matrix - Random Forest (CI Pipeline)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    plt.tight_layout()
    cm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')
    fig1.savefig(cm_path, dpi=150)
    plt.close(fig1)
    mlflow.log_artifact(cm_path)

    # --- Artefak 2: Feature Importance ---
    print("  [Artefak 2] Membuat Feature Importance Plot...")
    feature_names = X_train.columns.tolist()
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('viridis', len(feature_names))
    ax2.barh(range(len(feature_names)),
             importances[indices[::-1]],
             color=colors, edgecolor='black')
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax2.set_xlabel('Importance')
    ax2.set_title('Feature Importance - Random Forest (CI Pipeline)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fi_path = os.path.join(SCRIPT_DIR, 'feature_importance.png')
    fig2.savefig(fi_path, dpi=150)
    plt.close(fig2)
    mlflow.log_artifact(fi_path)

    # --- Artefak 3: Classification Report ---
    print("  [Artefak 3] Menyimpan Classification Report...")
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], output_dict=True)
    report_path = os.path.join(SCRIPT_DIR, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(report_path)

    # --- Simpan Run ID untuk CI Pipeline ---
    run_id = mlflow.active_run().info.run_id
    run_id_path = os.path.join(SCRIPT_DIR, 'run_id.txt')
    with open(run_id_path, 'w') as f:
        f.write(run_id)

    print(f"\n  Run ID: {run_id}")

# ============================================================
# [6/7] SIMPAN MODEL LOKAL UNTUK DOCKER BUILD
# ============================================================
print("[6/7] Menyimpan model secara lokal untuk Docker build...")
local_model_path = os.path.join(SCRIPT_DIR, 'saved_model')
if os.path.exists(local_model_path):
    shutil.rmtree(local_model_path)
mlflow.sklearn.save_model(best_model, local_model_path)
print(f"  Model tersimpan di: {local_model_path}")

# ============================================================
# [7/7] SELESAI
# ============================================================
print("\n[7/7] SELESAI!")
print("=" * 60)
print(f"  Dashboard: https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}")
print("  Buka link di atas -> klik tab 'Experiments' untuk melihat hasil.")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
