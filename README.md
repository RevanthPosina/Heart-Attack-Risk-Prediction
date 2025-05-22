# Heart-Attack-Risk-Prediction

# 🫀 Heart Attack Risk Prediction – BRFSS 2023

A full-stack ML pipeline to identify heart attack risk using CDC’s 2023 BRFSS dataset. Built with a production-first mindset, this project integrates clean data engineering, interpretable ML, and real-time deployment using AWS services.

> **Project Outcome:** A transparent, explainable, and high-performance heart-risk scoring engine — powered by XGBoost, deployed on SageMaker, and monitored via CloudWatch — ready for insurer use cases like underwriting and population health management.

---

## 🎯 Problem & Objective

Heart attacks are preventable yet costly. Traditional risk scoring misses early behavioral and comorbidity signals. This solution leverages public health data to deliver accurate, explainable risk predictions.

- **Business Need:** Insurance underwriting & preventative care
- **Goal:** Predict risk of heart attack from lifestyle + clinical factors
- **End Product:** Interpretable ML model with <1s API latency

---

## 📊 Dataset

- **Source:** [CDC BRFSS 2023](https://www.cdc.gov/brfss/)
- ~430,000 responses, 350+ fields
- 33 curated features selected post codebook mapping
- Target variable: `cvdinfra4` (ever diagnosed with a heart attack)

---

## 🧱 Project Phases

### 📥 1. Data Extraction & Preprocessing
- Extracted `.xpt` survey files, filtered relevant columns
- Cleaned missing/ambiguous codes (`7=DK`, `9=Refused`)
- Standardized binary indicators, renamed columns, created modeling-ready dataset

### 🧹 2. Data Cleaning & Mapping
- Dropped columns with excessive nulls
- Applied log + z-score transformation on skewed features (e.g., `bmi_log_z`)
- Mapped categorical codes to readable labels using BRFSS codebook

### 🧠 3. Feature Insights
- Identified high-risk groups using positive rate profiling (e.g., widowed, low-income)
- Used Cohen’s d, t-tests, chi-squared tests to rank feature importance

### 📈 4. Exploratory Data Analysis
- Bivariate risk profiling for both categorical and numerical features
- Visual checks, outlier detection, and initial pruning
- Selected statistically significant features for modeling

### 🧪 5. Model Development (XGBoost)
- Pipeline: `log_z` numerics + binned features + binary flags
- Baseline AUROC: **0.9885**, Recall: **1.00**
- Optuna tuning: 150+ trials, minor lift (best `scale_pos_weight ≈ 1.2`)
- Compared with LGBM & Logistic Regression — XGBoost selected for performance + ecosystem

### 🧠 6. SHAP Interpretation & Feature Engineering
- Used SHAP for global and individual explanations
- Created engineered features: `chronic_burden`, `preventive_neglect`, `bmi_risk_code`
- Validated model robustness after removing dominant feature (`prev_chd_or_mi`)

### 🚀 7. SageMaker Deployment
- Model saved as `xgb_top25_shap.joblib` and deployed using built-in XGBoost 1.5-1 container
- REST API exposed via `ml.m5.large` endpoint
- Smoke tested with sample input/output
- Latency consistently <150ms (p95)

### 📉 8. Monitoring with CloudWatch
- Configured `InvocationLatency > 1s` alarm via Boto3
- Programmatically triggered and reset on deployment
- No threshold breaches observed in load testing

### 📊 9. Experiment Tracking with MLflow
- Auto-logged: metrics, parameters, Conda env, artifacts
- Compared baseline vs tuned runs
- Enabled one-click rollback & reproducibility
- Stored trained models + test splits in artifact store

### 🌐 10. Streamlit Web App
- CSV upload → real-time inference via deployed API
- SHAP visualizations (global + per-row with GPT insights)
- XGBoost vs Logistic Regression comparison toggle

### 🔁 11. Airflow Orchestration
- Parameterized DAG: data prep → training → deploy → monitor
- Papermill-executed notebooks, IAM-secured tasks
- Airflow DAG updates model + CloudWatch alarm dynamically

---

## 💡 Key Results

| Metric         | Value      |
|----------------|------------|
| AUROC          | 0.9884     |
| Recall         | 1.00       |
| Precision      | ~0.66      |
| Inference Time | < 150 ms   |
| SHAP Validated | ✅         |

---

## 🧰 Tech Stack

**Data & Modeling:** Python, Pandas, Scikit-learn, XGBoost, Optuna  
**Explainability:** SHAP, OpenAI GPT API  
**Deployment:** AWS SageMaker, Boto3  
**Monitoring:** AWS CloudWatch, SNS  
**Tracking:** MLflow  
**Orchestration:** Airflow + Papermill  
**Web UI:** Streamlit
