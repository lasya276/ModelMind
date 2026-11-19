"""
Data Engine - Exploratory Data Analysis (EDA) Module
Generates comprehensive, chart-ready exploratory analysis:
1. Numerical distribution analysis and histogram generation.
2. Categorical frequency analysis and bar-chart-ready aggregation.
3. Pearson correlation matrix and pairwise relationship analysis.
4. Missing value visualization analysis.
5. General structural dataset summary.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def _safe_float(val: Any, decimals: int = 4) -> Optional[float]:
    """
    Safely converts a numeric value to a standard Python float.
    Returns None if value is NaN, infinite, or None to ensure strict JSON serializability.
    """
    if val is None or pd.isna(val) or (isinstance(val, (float, np.floating)) and np.isinf(val)):
        return None
    return round(float(val), decimals)


def _compute_numerical_eda(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """
    Computes summary statistics and histogram binning for numerical columns.
    Filters out missing and infinite values safely before computing metrics.
    """
    results: Dict[str, Any] = {}

    for col in numeric_cols:
        series = df[col]
        # Replace infinite values with NaN and drop NaNs for safe computation
        clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
        valid_count = int(len(clean_series))

        if clean_series.empty:
            results[col] = {
                "column": col,
                "valid_count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "histogram": {
                    "bin_edges": [],
                    "counts": [],
                    "bin_labels": [],
                },
            }
            continue

        min_val = float(clean_series.min())
        max_val = float(clean_series.max())
        std_val = float(clean_series.std()) if len(clean_series) > 1 else None

        # Histogram generation
        unique_vals_count = int(clean_series.nunique())
        if min_val == max_val:
            # Single value / zero variance column
            bin_edges = [_safe_float(min_val), _safe_float(max_val)]
            counts = [valid_count]
            bin_labels = [f"{min_val:.2f}"]
        else:
            num_bins = min(10, max(2, unique_vals_count))
            np_counts, np_edges = np.histogram(clean_series, bins=num_bins)
            counts = [int(c) for c in np_counts]
            bin_edges = [_safe_float(e) for e in np_edges]
            bin_labels = [
                f"{np_edges[i]:.2f} - {np_edges[i+1]:.2f}"
                for i in range(len(counts))
            ]

        results[col] = {
            "column": col,
            "valid_count": valid_count,
            "mean": _safe_float(clean_series.mean()),
            "median": _safe_float(clean_series.median()),
            "std": _safe_float(std_val),
            "min": _safe_float(min_val),
            "max": _safe_float(max_val),
            "histogram": {
                "bin_edges": bin_edges,
                "counts": counts,
                "bin_labels": bin_labels,
            },
        }

    return results


def _compute_categorical_eda(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Any]:
    """
    Computes frequency distributions, modes, and bar-chart-ready category counts.
    Limits categories to top 10 to avoid returning excessively large payloads.
    """
    results: Dict[str, Any] = {}

    for col in categorical_cols:
        series = df[col]
        clean_series = series.dropna().astype(str)
        valid_count = int(len(clean_series))
        unique_count = int(clean_series.nunique())

        if clean_series.empty:
            results[col] = {
                "column": col,
                "valid_count": 0,
                "unique_count": 0,
                "top_category": None,
                "top_frequency": 0,
                "bar_chart": {
                    "categories": [],
                    "counts": [],
                    "percentages": [],
                },
            }
            continue

        value_counts = clean_series.value_counts()
        top_category = str(value_counts.index[0])
        top_frequency = int(value_counts.iloc[0])

        # Limit to top 10 categories, bucket the rest under 'Other'
        top_10 = value_counts.head(10)
        categories = [str(k) for k in top_10.index]
        counts = [int(v) for v in top_10.values]

        if len(value_counts) > 10:
            other_count = int(value_counts.iloc[10:].sum())
            if other_count > 0:
                categories.append("Other")
                counts.append(other_count)

        percentages = [
            round((c / valid_count * 100), 2) if valid_count > 0 else 0.0
            for c in counts
        ]

        results[col] = {
            "column": col,
            "valid_count": valid_count,
            "unique_count": unique_count,
            "top_category": top_category,
            "top_frequency": top_frequency,
            "bar_chart": {
                "categories": categories,
                "counts": counts,
                "percentages": percentages,
            },
        }

    return results


def _compute_correlation_analysis(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """
    Computes Pearson correlation for numerical columns with sufficient variance.
    Returns correlation matrix, pairwise relationships, and heatmap-ready data.
    Explicitly notes that correlation does not imply causation.
    """
    disclaimer = (
        "Correlation measures linear association between variables and does not imply causation. "
        "A strong correlation indicates a statistical relationship, not a direct cause-and-effect mechanism."
    )

    # Filter out columns with zero variance (constant or empty)
    suitable_cols: List[str] = []
    for col in numeric_cols:
        clean = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) > 1 and int(clean.nunique()) > 1:
            suitable_cols.append(col)

    if len(suitable_cols) < 2 or len(df) < 2:
        return {
            "status": "insufficient_data",
            "columns": suitable_cols,
            "matrix": {},
            "pairs": [],
            "heatmap_ready": {
                "x": suitable_cols,
                "y": suitable_cols,
                "z": [],
            },
            "disclaimer": disclaimer,
        }

    # Calculate Pearson correlation matrix
    clean_numeric_df = df[suitable_cols].replace([np.inf, -np.inf], np.nan)
    corr_matrix = clean_numeric_df.corr(method="pearson")

    matrix_dict: Dict[str, Dict[str, Optional[float]]] = {}
    z_matrix: List[List[Optional[float]]] = []

    for r_col in suitable_cols:
        matrix_dict[r_col] = {}
        row_vals: List[Optional[float]] = []
        for c_col in suitable_cols:
            val = corr_matrix.loc[r_col, c_col]
            safe_v = _safe_float(val)
            matrix_dict[r_col][c_col] = safe_v
            row_vals.append(safe_v)
        z_matrix.append(row_vals)

    # Extract distinct pairs (excluding diagonal and duplicate symmetric pairs)
    pairs: List[Dict[str, Any]] = []
    for i in range(len(suitable_cols)):
        for j in range(i + 1, len(suitable_cols)):
            c1 = suitable_cols[i]
            c2 = suitable_cols[j]
            corr_val = _safe_float(corr_matrix.loc[c1, c2])

            if corr_val is not None:
                abs_val = abs(corr_val)
                if abs_val >= 0.7:
                    strength = "strong"
                elif abs_val >= 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"
            else:
                strength = "unknown"

            pairs.append({
                "column_1": c1,
                "column_2": c2,
                "correlation": corr_val,
                "strength": strength,
            })

    # Sort pairs by absolute correlation strength descending
    pairs.sort(key=lambda p: abs(p["correlation"]) if p["correlation"] is not None else 0.0, reverse=True)

    return {
        "status": "computed",
        "columns": suitable_cols,
        "matrix": matrix_dict,
        "pairs": pairs,
        "heatmap_ready": {
            "x": suitable_cols,
            "y": suitable_cols,
            "z": z_matrix,
        },
        "disclaimer": disclaimer,
    }


def _compute_missing_value_analysis(df: pd.DataFrame, total_rows: int, total_columns: int) -> Dict[str, Any]:
    """
    Computes missingness statistics per column and formats chart-ready visualization data.
    """
    column_missing: List[Dict[str, Any]] = []
    total_missing_cells = 0

    for col in df.columns:
        col_str = str(col)
        missing_count = int(df[col].isna().sum())
        total_missing_cells += missing_count
        pct = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        column_missing.append({
            "column": col_str,
            "missing_count": missing_count,
            "missing_percentage": pct,
        })

    # Sort descending by missing count for clear visual prioritization
    sorted_missing = sorted(column_missing, key=lambda x: x["missing_count"], reverse=True)

    total_cells = total_rows * total_columns
    overall_pct = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    return {
        "total_missing_cells": total_missing_cells,
        "total_cells": total_cells,
        "overall_missing_percentage": overall_pct,
        "columns": column_missing,
        "chart_data": {
            "columns": [c["column"] for c in sorted_missing],
            "missing_counts": [c["missing_count"] for c in sorted_missing],
            "missing_percentages": [c["missing_percentage"] for c in sorted_missing],
        },
    }


def analyze_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a structured Exploratory Data Analysis (EDA) report for a Pandas DataFrame.
    Never modifies the original DataFrame.
    Never trains ML models or applies cleaning transformations.

    Args:
        df: Input pandas DataFrame to analyze.

    Returns:
        Dict[str, Any] containing numerical, categorical, correlation,
        missingness, and dataset summary analysis.

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

    # Identify numerical and categorical columns
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        # Treat numeric dtypes (excluding boolean) as numerical
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_cols.append(col_str)
        else:
            categorical_cols.append(col_str)

    # 1. Dataset Summary
    dataset_summary = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "numeric_column_count": len(numeric_cols),
        "categorical_column_count": len(categorical_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "has_data": total_rows > 0 and total_columns > 0,
    }

    # 2. Numerical Analysis & Histogram Data
    numerical_analysis = _compute_numerical_eda(df, numeric_cols)

    # 3. Categorical Analysis & Bar Chart Data
    categorical_analysis = _compute_categorical_eda(df, categorical_cols)

    # 4. Correlation Analysis
    correlation_analysis = _compute_correlation_analysis(df, numeric_cols)

    # 5. Missing Value Analysis
    missing_value_analysis = _compute_missing_value_analysis(df, total_rows, total_columns)

    return {
        "dataset_summary": dataset_summary,
        "numerical_analysis": numerical_analysis,
        "categorical_analysis": categorical_analysis,
        "correlation_analysis": correlation_analysis,
        "missing_value_analysis": missing_value_analysis,
    }
