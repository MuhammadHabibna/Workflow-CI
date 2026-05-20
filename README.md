# Workflow-CI: Fraud Detection ML Pipeline

Repository ini berisi **MLflow Project** dan **GitHub Actions Workflow** untuk melakukan re-training model Machine Learning secara otomatis (Continuous Integration).

## Struktur
```
Workflow-CI/
├── .github/workflows/
│   └── ml-ci.yml                        ← GitHub Actions CI Pipeline
└── MLProject/
    ├── MLProject                        ← Konfigurasi MLflow Project
    ├── conda.yaml                       ← Dependencies
    ├── modelling.py                     ← Script training + tuning
    ├── dockerhub_link.txt               ← Link Docker Hub
    └── fraud_detection_preprocessing/   ← Dataset preprocessed
```

## Cara Kerja
1. Setiap kali ada perubahan di folder `MLProject/`, GitHub Actions otomatis berjalan.
2. Robot CI akan melatih model Random Forest dengan Hyperparameter Tuning.
3. Semua metrik dan artefak dicatat ke **DagsHub** via MLflow.
4. Model dibungkus menjadi **Docker Image** dan di-push ke **Docker Hub**.

## Docker Hub
```
docker pull muhammadhabibna/fraud-detection-model
```
