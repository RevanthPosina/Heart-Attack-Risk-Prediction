# inference.py

import joblib
import os
import numpy as np
import pandas as pd

def model_fn(model_dir):
    """Load the trained model from S3."""
    model_path = os.path.join(model_dir, "xgb_top25_shap.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """Deserialize input data from JSON."""
    if request_content_type == "application/json":
        data = pd.DataFrame(request_body)
        return data
    elif request_content_type == "text/csv":
        return pd.read_csv(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Generate prediction."""
    return model.predict_proba(input_data)[:, 1]  # output probability of class 1

def output_fn(prediction, response_content_type):
    """Serialize prediction result to JSON."""
    if response_content_type == "application/json":
        return str(prediction.tolist())
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
