# FraudPaul

Streamlit app for fraud detection with behavioral biometrics. The app lets analysts upload/score transaction CSVs, review explainability, and monitor keystroke-based liveness/anomaly signals from enrolled users.

## What it does
- Scores card transactions with trained models (RandomForest/XGBoost) and shows metrics, flags, and SHAP insights.
- Provides keystroke enrollment (typing a target sentence) to train a per-user OneClass SVM profile.
- Runs passive keystroke monitoring across key pages (dashboard, monitoring) to surface anomalies and log security events.
- Offers a central dashboard view of recent security events and navigation to the key tools.

## How it works
- `app/pages/fraud_detection.py`: CSV upload, column validation, inference, metrics, flagged rows, and SHAP plots.
- `app/pages/keystroke_enrollment.py`: Collects 5 typing samples, trains/saves the OneClass SVM and profile JSON.
- `app/pages/keystroke_monitoring.py`: Uses the trained model to score passive keystroke streams and shows anomalies/logs.
- `app/pages/dashboard.py`: High-level cards, quick actions, and recent security events.
- Models and logs are expected under `app/models/` and `app/logs/`; large artifacts are ignored by `.gitignore` to keep the repo light.

## Run locally
1) Install dependencies (ensure Python 3.10+).  
2) From repo root: `streamlit run app/main.py`  
3) Place required model artifacts under `app/models/` (RF/XGB/scaler for fraud, keystroke SVM/profile) and logs under `app/logs/` as needed.

## Notes
- Large datasets/models (e.g., `creditcard.csv`, face data, pickles) are intentionally excluded via `.gitignore`; add your own copies locally before running.
- Security and anomaly events append to `app/logs/security_logs.csv`; the dashboard and monitoring views read from that file.
