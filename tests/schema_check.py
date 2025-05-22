import json
from src.utils.schema import validate_features
def validate_features(input_df, feature_list_path):
    with open(feature_list_path, "r") as f:
        expected_features = json.load(f)

    actual_features = list(input_df.columns)

    missing = set(expected_features) - set(actual_features)
    extra = set(actual_features) - set(expected_features)

    if missing or extra:
        raise ValueError(f"Schema mismatch!\nMissing: {missing}\nExtra: {extra}")

    if actual_features != expected_features:
        print("Warning: Column order mismatch.")

    # Reindex to match training order and fill missing with 0
    input_df = input_df.reindex(columns=expected_features, fill_value=0)
    return input_df



X_enc = pd.get_dummies(X_val)
X_enc = validate_features(X_enc, "data/derived/feature_list_final_curated.json")
