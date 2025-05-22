import boto3
import os
import tarfile
from pathlib import Path

# Setup paths
base_path = Path(__file__).resolve().parents[3] / "models"
joblib_path = base_path / "xgb_top25_shap.joblib"
bst_path = base_path / "xgb_top25_shap.bst"
tar_path = base_path / "model.tar.gz"

# Check files exist
print("Looking for joblib model at:", joblib_path)
assert joblib_path.exists(), "Joblib model file not found!"

print("Looking for .bst booster at:", bst_path)
assert bst_path.exists(), ".bst booster file not found!"

# Create model.tar.gz from .bst (required by SageMaker native XGBoost)
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(bst_path, arcname="xgboost-model") 

print("Created SageMaker-compatible model.tar.gz archive.")

# Upload to S3
s3 = boto3.client("s3")
bucket = "verikai-heart-risk-pipeline"

s3.upload_file(str(joblib_path), bucket, "deployment/xgb_top25_shap.joblib")
s3.upload_file(str(bst_path), bucket, "deployment/xgb_top25_shap.bst")
s3.upload_file(str(tar_path), bucket, "deployment/model.tar.gz")

print("All model files uploaded: .joblib, .bst, and model.tar.gz")
