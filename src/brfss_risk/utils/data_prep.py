import numpy as np
import pandas as pd
from yaml import safe_load
from pathlib import Path
from brfss_risk.utils.logger import get_logger, log_event
from scipy.stats import skew

logger = get_logger()

AMBIG_CODES = [7, 9, 77, 99]

def load_maps(config_path="configs/brfss_2023_column_map.yaml"):
    with open(config_path, "r") as f:
        return safe_load(f)


def apply_domain_maps(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns=maps["rename_map"])
    for col, mapping in maps["binary_maps"].items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
        else:
            logger.warning(f"Column '{col}' not found in DataFrame for binary mapping.")
    logger.info("Applied rename & binary maps")
    return df

def clean_codes(df: pd.DataFrame,
                missing_codes=AMBIG_CODES,
                exclude=None) -> pd.DataFrame:
    exclude = exclude or []
    for col in df.columns.difference(exclude):
        bad_mask = df[col].isin(missing_codes)
        if bad_mask.any():
            df[col] = df[col].mask(df[col].isin(missing_codes))
    logger.info("Converted ambiguous BRFSS codes to NaN")
    return df

def get_selected_cols_from_map(map_path=None) -> list:
    """Extract original column names from the renaming map as the selected subset."""
    maps = load_maps(map_path) if map_path else load_maps()
    return list(maps["rename_map"].values())

def apply_standard_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize special numeric codes for alcohol and unhealthy days columns."""
    df = df.copy()
    
    # Alcohol (ALCDAY4)
    if "alcohol_days" in df.columns:
        df["alcohol_days"] = df["alcohol_days"].replace({
            888: 0,  # No drinks in 30 days
            777: np.nan, 999: np.nan  # Don't know / refused
        })
        df["alcohol_days"] = df["alcohol_days"].where(df["alcohol_days"] <= 365, np.nan)

    # Mental health (MENTHLTH)
    if "mental_unhealthy_days" in df.columns:
        df["mental_unhealthy_days"] = df["mental_unhealthy_days"].replace({
            88: 0, 77: np.nan, 99: np.nan
        })

    # Physical health (PHYSHLTH)
    if "physical_unhealthy_days" in df.columns:
        df["physical_unhealthy_days"] = df["physical_unhealthy_days"].replace({
            88: 0, 77: np.nan, 99: np.nan
        })

    logger.info("Standardized alcohol and *_unhealthy_days fields")
    return df


def generate_missing_report(df, threshold=0.01, log=False):
    """
    Generates a missing value report.
    Optionally logs top missing columns if `log=True`.

    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Minimum missing % to report
        log (bool): Whether to log summary

    Returns:
        pd.DataFrame: Dataframe with feature name and % missing
    """
    mv = (
        df.isna().mean()
          .reset_index()
          .rename(columns={'index': 'feature', 0: 'pct_missing'})
          .sort_values('pct_missing', ascending=False)
    )
    mv = mv[mv['pct_missing'] > threshold]

    if log and not mv.empty:
        log_event(
            phase="PhaseX",
            action="Missing Report",
            summary_dict={
                "top_missing_pct": mv.set_index('feature')['pct_missing'].round(4).to_dict()
            }
        )
    return mv

def add_missing_indicators(df, cols):
    """
    Adds binary missing indicator columns for specified features.

    Args:
        df (pd.DataFrame): Input dataframe
        cols (list): List of columns to create indicators for

    Returns:
        pd.DataFrame: Updated dataframe with _missing columns
    """
    for col in cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df





def transform_skewed_features(df, cols, threshold=1.0, log_prefix="_log", log_event=None, phase="Phase2"):
    """
    Auto-transforms skewed features using log1p where abs(skew) > threshold.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        cols (list): List of continuous numeric column names to check.
        threshold (float): Skewness threshold above which to apply log1p.
        log_prefix (str): Suffix to add to transformed column.
        log_event (func, optional): Logger function for tracking.
        phase (str): Pipeline phase name for logging context.

    Returns:
        df (pd.DataFrame): DataFrame with new log-transformed columns.
        skew_report (list): List of (column, skewness) tuples.
        log_transform_cols (list): Names of newly created log1p columns.
    """
    skew_report = []
    log_transform_cols = []

    for col in cols:
        valid = df[col].dropna()
        sk = skew(valid) if len(valid) >= 3 else float('nan')
        skew_report.append((col, sk))

        if abs(sk) > threshold:
            new_col = f"{col}{log_prefix}"
            if new_col not in df.columns:
                df[new_col] = np.log1p(df[col])
                log_transform_cols.append(new_col)

    if log_event:
        log_event(
            phase=phase,
            action="Applied log1p to skewed features",
            summary_dict={
                "log1p_cols": log_transform_cols
            }
        )

    return df, skew_report, log_transform_cols
