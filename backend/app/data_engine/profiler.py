"""
Data Engine - Profiler Module
Computes structural, statistical, and descriptive summaries for Pandas DataFrames.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def _safe_float(val: Any, decimals: int = 4) -> Optional[float]:
    """
    Safely converts a numerical value to a standard Python float rounded to `decimals`.
    Returns None if the value is NaN, infinite, or None, ensuring JSON serializability.
    """
    if val is None or pd.isna(val) or np.isinf(val):
        return None
    return round(float(val), decimals)


def _compute_numeric_summary(series: pd.Series) -> Dict[str, Optional[float]]:
    """
    Computes descriptive statistics (mean, median, std, min, max) for a numeric column.
    Safely handles empty series, single-value series, and series with NaN values.
    """
    clean_series = series.dropna()

    if clean_series.empty:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    # std on a single sample is undefined (degrees of freedom = 0), so it yields NaN -> None
    std_val = clean_series.std() if len(clean_series) > 1 else None

    return {
        "mean": _safe_float(clean_series.mean()),
        "median": _safe_float(clean_series.median()),
        "std": _safe_float(std_val),
        "min": _safe_float(clean_series.min()),
        "max": _safe_float(clean_series.max()),
    }


def _compute_categorical_summary(series: pd.Series) -> Dict[str, Any]:
    """
    Computes descriptive information for a categorical / non-numeric column.
    Returns unique value count, top value (mode), and frequency of the top value.
    """
    clean_series = series.dropna()
    unique_count = int(clean_series.nunique())

    if clean_series.empty:
        return {
            "unique_count": 0,
            "top_value": None,
            "top_frequency": 0,
        }

    # Determine top value (mode) and its frequency
    top_value = str(clean_series.mode().iloc[0])
    top_frequency = int(clean_series.value_counts().iloc[0])

    return {
        "unique_count": unique_count,
        "top_value": top_value,
        "top_frequency": top_frequency,
    }


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a comprehensive, JSON-serializable profile of a Pandas DataFrame.

    Metrics computed:
    - Total rows and columns.
    - Column names and data types.
    - Missing value count and percentage per column.
    - Unique value count per column.
    - Duplicate row count.
    - Numeric statistics: mean, median, standard deviation, min, max.
    - Categorical statistics: unique count, top value, top frequency.

    Args:
        df: Input pandas DataFrame to profile. Original DataFrame is not modified.

    Returns:
        Dict[str, Any] containing the profile results.

    Raises:
        ValueError: If input DataFrame is None.
        TypeError: If input is not a pandas DataFrame.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    total_rows = int(len(df))
    total_columns = int(len(df.columns))
    columns = [str(col) for col in df.columns]

    # Handle duplicate row detection safely
    duplicate_rows = int(df.duplicated().sum()) if total_rows > 0 and total_columns > 0 else 0

    column_types: Dict[str, str] = {}
    missing_values: Dict[str, Dict[str, Any]] = {}
    unique_values: Dict[str, int] = {}
    numeric_summary: Dict[str, Dict[str, Optional[float]]] = {}
    categorical_summary: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        col_str = str(col)
        series = df[col]

        # 1. Column Data Type
        column_types[col_str] = str(series.dtype)

        # 2. Missing Values (count & percentage)
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        missing_values[col_str] = {
            "count": missing_count,
            "percentage": missing_pct,
        }

        # 3. Unique Value Count (non-null)
        unique_values[col_str] = int(series.nunique(dropna=True))

        # 4. Summary Statistics: Numeric vs Categorical
        # Note: Booleans are treated as categorical rather than numerical
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

        if is_numeric:
            numeric_summary[col_str] = _compute_numeric_summary(series)
        else:
            categorical_summary[col_str] = _compute_categorical_summary(series)

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "columns": columns,
        "column_types": column_types,
        "missing_values": missing_values,
        "unique_values": unique_values,
        "duplicate_rows": duplicate_rows,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }
