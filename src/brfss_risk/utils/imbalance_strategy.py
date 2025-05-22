from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

def get_sampling_pipeline(strategy="smote", random_state=42):
    """
    Returns an imbalanced-learn pipeline for sampling.

    Parameters:
        strategy (str): One of ['smote', 'undersample']
        random_state (int): Random seed

    Returns:
        imblearn.pipeline.Pipeline
    """
    if strategy == "smote":
        sampler = SMOTE(random_state=random_state)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError("strategy must be 'smote' or 'undersample'")

    return Pipeline(steps=[("sampler", sampler)])

def compute_scale_pos_weight(y):
    """
    Computes XGBoost-compatible scale_pos_weight for binary imbalance.

    Parameters:
        y (pd.Series or np.array): Target vector (binary)

    Returns:
        float: scale_pos_weight = negative / positive
    """
    counter = Counter(y)
    return round(counter[0] / counter[1], 2)
