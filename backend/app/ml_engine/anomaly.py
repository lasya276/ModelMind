"""
ML Engine - Anomaly Detection Module
Identifies statistically unusual rows in datasets using Scikit-learn's Isolation Forest.
Features:
- Purely numeric feature extraction with automatic ID exclusion.
- Safe imputation of missing and infinite values for model fitting.
- Comprehensive JSON-serializable anomaly report.
- Guaranteed immutability of input DataFrame.
"""

from typing import Dict, Any, List, Optional, Union
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Regex heuristic to recognize identifier columns that should not be used as features
ID_NAME_PATTERN = re.compile(
    r"(^|_|-)(id|uuid|guid|key|code|pk|identifier)($|_|-)",
    re.IGNORECASE,
)


def _is_potential_id_column(col_name: str, series: pd.Series, total_rows: int) -> bool:
    """
    Checks if a numeric column is likely an entity identifier or sequential index.
    """
    if total_rows < 2:
        return False

    name_matches = bool(ID_NAME_PATTERN.search(col_name) or col_name.lower().endswith("id"))
    clean = series.dropna()
    if clean.empty:
        return False

    uniqueness_pct = (clean.nunique() / total_rows) * 100

    # Flag if name matches ID pattern and uniqueness is high
    if name_matches and uniqueness_pct >= 90.0:
        return True

    return False


def detect_anomalies(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    contamination: Union[float, str] = "auto",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Detects unusual rows in a dataset using Isolation Forest.
    Never modifies the original DataFrame.
    Never deletes or drops anomalous rows.

    Args:
        df: Input pandas DataFrame to analyze.
        feature_columns: Optional list of numeric columns to use. If None,
                         automatically selects numeric features while excluding ID columns.
        contamination: The proportion of outliers in the dataset. Can be 'auto'
                       or a float in the range (0.0, 0.5].
        random_state: Random seed for reproducibility.

    Returns:
        Structured, JSON-serializable anomaly detection report.

    Raises:
        ValueError: If df is None, feature_columns do not exist, or are non-numeric.
        TypeError: If df is not a pandas DataFrame.
    """
    # 1. Input Validation
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    total_rows = int(len(df))

    # Validate contamination parameter
    if isinstance(contamination, (int, float)):
        if not (0.0 < contamination <= 0.5):
            raise ValueError(f"Contamination must be in (0.0, 0.5] or 'auto', got {contamination}.")
    elif contamination != "auto":
        raise ValueError(f"Contamination must be 'auto' or a float between 0.0 and 0.5, got '{contamination}'.")

    # 2. Feature Column Selection and Validation
    if feature_columns is not None:
        if not isinstance(feature_columns, list) or len(feature_columns) == 0:
            raise ValueError("feature_columns must be a non-empty list of column names.")

        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Specified feature column(s) {missing_cols} do not exist in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        non_numeric = [
            c for c in feature_columns
            if not (pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]))
        ]
        if non_numeric:
            raise ValueError(
                f"Specified feature column(s) {non_numeric} are not numeric. "
                "Anomaly detection requires numeric features."
            )

        selected_features = [str(c) for c in feature_columns]
    else:
        # Automatically select numeric features, excluding potential ID columns
        selected_features = []
        for col in df.columns:
            col_str = str(col)
            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            if is_numeric:
                if not _is_potential_id_column(col_str, series, total_rows):
                    selected_features.append(col_str)

    # 3. Handle Empty or Insufficient Data Gracefully
    if total_rows == 0:
        return {
            "status": "empty_dataset",
            "total_rows_analyzed": 0,
            "anomaly_count": 0,
            "normal_count": 0,
            "anomaly_percentage": 0.0,
            "features_used": selected_features,
            "anomaly_labels": [],
            "anomaly_scores": [],
            "is_anomaly": [],
            "explanation": "The dataset is empty. Zero rows were analyzed.",
        }

    if len(selected_features) == 0:
        return {
            "status": "no_numeric_features",
            "total_rows_analyzed": total_rows,
            "anomaly_count": 0,
            "normal_count": total_rows,
            "anomaly_percentage": 0.0,
            "features_used": [],
            "anomaly_labels": [1] * total_rows,
            "anomaly_scores": [0.0] * total_rows,
            "is_anomaly": [False] * total_rows,
            "explanation": (
                "No suitable numeric features were found or selected for anomaly detection. "
                "All rows are considered normal."
            ),
        }

    if total_rows < 5:
        return {
            "status": "insufficient_data",
            "total_rows_analyzed": total_rows,
            "anomaly_count": 0,
            "normal_count": total_rows,
            "anomaly_percentage": 0.0,
            "features_used": selected_features,
            "anomaly_labels": [1] * total_rows,
            "anomaly_scores": [0.0] * total_rows,
            "is_anomaly": [False] * total_rows,
            "explanation": (
                f"Dataset contains only {total_rows} row(s), which is insufficient for reliable anomaly detection "
                "(minimum of 5 rows recommended). All rows are treated as normal."
            ),
        }

    # 4. Prepare Feature Matrix Safely Without Mutating Original DataFrame
    X = df[selected_features].copy(deep=True)

    # Replace infinities with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Impute missing values with column median (fallback to 0.0 for completely null columns)
    for col in selected_features:
        clean_col = X[col].dropna()
        if not clean_col.empty:
            median_val = float(clean_col.median())
            X[col] = X[col].fillna(median_val)
        else:
            X[col] = X[col].fillna(0.0)

    # 5. Fit Isolation Forest and Predict
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    raw_preds = iso_forest.fit_predict(X)
    raw_scores = iso_forest.decision_function(X)

    # 6. Format JSON-Serializable Output
    anomaly_labels = [int(p) for p in raw_preds]  # 1 for normal, -1 for anomaly
    is_anomaly = [bool(p == -1) for p in raw_preds]
    anomaly_scores = [round(float(s), 4) for s in raw_scores]

    anomaly_count = int(sum(1 for p in raw_preds if p == -1))
    normal_count = int(sum(1 for p in raw_preds if p == 1))
    anomaly_pct = round((anomaly_count / total_rows * 100), 2)

    explanation = (
        f"Analyzed {total_rows} rows across {len(selected_features)} numeric feature(s) ({', '.join(selected_features)}). "
        f"Detected {anomaly_count} unusual observation(s) ({anomaly_pct}% of the dataset). "
        "Important: In Isolation Forest, an anomaly represents a statistically rare or unusual observation in feature space; "
        "it does not necessarily indicate an error, defect, or fraud."
    )

    return {
        "status": "completed",
        "total_rows_analyzed": total_rows,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_percentage": anomaly_pct,
        "features_used": selected_features,
        "anomaly_labels": anomaly_labels,
        "anomaly_scores": anomaly_scores,
        "is_anomaly": is_anomaly,
        "explanation": explanation,
    }
