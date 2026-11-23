"""
Data Engine - Quality Module
Evaluates dataset quality and integrity:
1. Missing values per column
2. Duplicate rows
3. Constant and empty columns
4. Potential identifier columns (heuristics)
5. Inconsistent categorical values (casing & whitespace variations)
6. Invalid numerical values (positive & negative infinity)
"""

from typing import Dict, Any, List, Set
import re
import numpy as np
import pandas as pd


def _check_missing_values(df: pd.DataFrame, total_rows: int) -> Dict[str, Any]:
    """
    Identifies columns with missing (NaN/None) values and computes counts,
    percentages, and assigned severity.
    """
    affected_columns: List[Dict[str, Any]] = []

    for col in df.columns:
        col_str = str(col)
        missing_count = int(df[col].isna().sum())

        if missing_count > 0:
            percentage = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0

            # Assign severity based on percentage of missing data
            if percentage >= 50.0:
                severity = "high"
            elif percentage >= 15.0:
                severity = "medium"
            else:
                severity = "low"

            affected_columns.append({
                "column": col_str,
                "missing_count": missing_count,
                "percentage": percentage,
                "severity": severity,
                "explanation": (
                    f"Column '{col_str}' has {missing_count} missing value(s) "
                    f"({percentage}% of total rows)."
                ),
            })

    return {
        "total_affected_columns": len(affected_columns),
        "columns": affected_columns,
    }


def _check_duplicate_rows(df: pd.DataFrame, total_rows: int, total_columns: int) -> Dict[str, Any]:
    """
    Detects fully duplicated rows across all columns.
    """
    if total_rows == 0 or total_columns == 0:
        return {
            "duplicate_count": 0,
            "percentage": 0.0,
            "severity": "low",
            "explanation": "No duplicate rows detected.",
        }

    duplicate_count = int(df.duplicated().sum())
    percentage = round((duplicate_count / total_rows * 100), 2)

    if duplicate_count == 0:
        severity = "low"
        explanation = "No duplicate rows detected."
    elif percentage >= 10.0:
        severity = "high"
        explanation = f"Detected {duplicate_count} duplicate row(s) ({percentage}% of dataset)."
    else:
        severity = "medium"
        explanation = f"Detected {duplicate_count} duplicate row(s) ({percentage}% of dataset)."

    return {
        "duplicate_count": duplicate_count,
        "percentage": percentage,
        "severity": severity,
        "explanation": explanation,
    }


def _check_constant_and_empty_columns(df: pd.DataFrame, total_rows: int) -> Dict[str, Any]:
    """
    Identifies completely empty columns (all nulls) and constant columns (single unique value).
    Handles empty columns separately from constant columns.
    """
    empty_columns: List[Dict[str, Any]] = []
    constant_columns: List[Dict[str, Any]] = []

    if total_rows == 0:
        return {
            "empty_columns": empty_columns,
            "constant_columns": constant_columns,
        }

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        non_null_series = series.dropna()
        unique_non_null = int(non_null_series.nunique())

        if non_null_series.empty:
            # Completely empty column
            empty_columns.append({
                "column": col_str,
                "severity": "high",
                "explanation": f"Column '{col_str}' is completely empty (100% missing values).",
            })
        elif unique_non_null == 1 and total_rows > 1:
            # Constant column with exactly one distinct non-null value
            single_val = non_null_series.iloc[0]
            val_str = str(single_val) if not pd.isna(single_val) else "None"
            constant_columns.append({
                "column": col_str,
                "constant_value": val_str,
                "severity": "medium",
                "explanation": (
                    f"Column '{col_str}' contains only one distinct non-null value ('{val_str}'). "
                    "It offers zero variance."
                ),
            })

    return {
        "empty_columns": empty_columns,
        "constant_columns": constant_columns,
    }


def _check_potential_id_columns(df: pd.DataFrame, total_rows: int) -> List[Dict[str, Any]]:
    """
    Identifies columns that may represent entity identifiers using naming heuristics
    and high uniqueness metrics. Explicitly marks them as potential IDs, not confirmed IDs.
    """
    potential_ids: List[Dict[str, Any]] = []

    if total_rows < 2:
        return potential_ids

    # Regex heuristic matching common identifier column patterns
    id_name_pattern = re.compile(r"(^|_|-)(id|uuid|guid|key|code|pk|identifier)($|_|-)", re.IGNORECASE)

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        non_null_series = series.dropna()

        if non_null_series.empty:
            continue

        unique_count = int(non_null_series.nunique())
        uniqueness_ratio = unique_count / total_rows
        uniqueness_pct = round(uniqueness_ratio * 100, 2)

        name_matches_heuristic = bool(id_name_pattern.search(col_str) or col_str.lower().endswith("id"))
        is_high_uniqueness = uniqueness_pct >= 95.0

        # Flag if:
        # 1. Name matches ID pattern AND uniqueness is high (>= 90%)
        # 2. Or uniqueness is 100% on a dataset with at least 5 rows
        if (name_matches_heuristic and uniqueness_pct >= 90.0) or (uniqueness_pct == 100.0 and total_rows >= 5):
            potential_ids.append({
                "column": col_str,
                "unique_count": unique_count,
                "uniqueness_percentage": uniqueness_pct,
                "name_heuristic_matched": name_matches_heuristic,
                "severity": "info",
                "is_potential_id": True,
                "explanation": (
                    f"Column '{col_str}' has {uniqueness_pct}% unique values "
                    + ("and matches ID naming patterns. " if name_matches_heuristic else ". ")
                    + "Marked as a potential identifier (not confirmed)."
                ),
            })

    return potential_ids


def _check_inconsistent_categorical_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects possible formatting variations in categorical/text columns such as
    whitespace differences ('New York ' vs 'New York') and casing differences ('apple' vs 'Apple').
    Does not modify or merge categories automatically.
    """
    affected_columns: List[Dict[str, Any]] = []

    for col in df.columns:
        col_str = str(col)
        series = df[col]

        # Only evaluate non-numeric, non-boolean columns
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if is_numeric:
            continue

        non_null_series = series.dropna()
        if non_null_series.empty:
            continue

        # Map normalized string -> set of original raw strings
        normalized_map: Dict[str, Set[str]] = {}
        for raw_val in non_null_series:
            raw_str = str(raw_val)
            norm_key = raw_str.strip().lower()
            if norm_key not in normalized_map:
                normalized_map[norm_key] = set()
            normalized_map[norm_key].add(raw_str)

        # Look for normalized keys that correspond to multiple distinct raw representations
        inconsistencies: List[Dict[str, Any]] = []
        for norm_key, raw_set in normalized_map.items():
            if len(raw_set) > 1:
                raw_list = sorted(list(raw_set))
                # Determine specific issue type
                has_whitespace_diff = any(v != v.strip() for v in raw_list)
                has_case_diff = len(set(v.strip() for v in raw_list)) > 1

                if has_whitespace_diff and has_case_diff:
                    issue_type = "case_and_whitespace_differences"
                elif has_whitespace_diff:
                    issue_type = "whitespace_differences"
                else:
                    issue_type = "case_differences"

                inconsistencies.append({
                    "normalized_value": norm_key,
                    "variations": raw_list,
                    "variation_count": len(raw_list),
                    "issue_type": issue_type,
                })

        if inconsistencies:
            affected_columns.append({
                "column": col_str,
                "inconsistency_count": len(inconsistencies),
                "inconsistencies": inconsistencies,
                "severity": "medium",
                "explanation": (
                    f"Column '{col_str}' contains {len(inconsistencies)} set(s) of values "
                    "with casing or whitespace discrepancies."
                ),
            })

    return {
        "total_affected_columns": len(affected_columns),
        "columns": affected_columns,
    }


def _check_invalid_numerical_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies positive and negative infinity values (+inf, -inf) in numerical columns.
    Reports counts without modifying data.
    """
    affected_columns: List[Dict[str, Any]] = []

    for col in df.columns:
        col_str = str(col)
        series = df[col]

        # Only evaluate numeric columns
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if not is_numeric:
            continue

        pos_inf_count = int(np.isposinf(series).sum())
        neg_inf_count = int(np.isneginf(series).sum())
        total_inf = pos_inf_count + neg_inf_count

        if total_inf > 0:
            finite_values = series[np.isfinite(series)]
            value_counts = finite_values.value_counts()
            mode_available = not value_counts.empty and int(value_counts.eq(value_counts.max()).sum()) == 1
            mode_unavailable_reason = None
            if not mode_available:
                mode_unavailable_reason = (
                    "No finite values are available."
                    if finite_values.empty
                    else "The most frequent finite values are tied."
                )

            affected_columns.append({
                "column": col_str,
                "positive_infinity_count": pos_inf_count,
                "negative_infinity_count": neg_inf_count,
                "total_invalid_count": total_inf,
                "finite_value_count": int(len(finite_values)),
                "mode_available": mode_available,
                "mode_unavailable_reason": mode_unavailable_reason,
                "severity": "high",
                "explanation": (
                    f"Column '{col_str}' contains {total_inf} infinite numerical value(s) "
                    f"({pos_inf_count} +inf, {neg_inf_count} -inf)."
                ),
            })

    return {
        "total_affected_columns": len(affected_columns),
        "columns": affected_columns,
    }


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive data quality checks on a Pandas DataFrame.
    Never modifies the original DataFrame.

    Checks:
    1. Missing values
    2. Duplicate rows
    3. Constant & empty columns
    4. Potential identifier columns (heuristics)
    5. Inconsistent categorical values (casing & whitespace)
    6. Invalid numerical values (+inf / -inf)

    Args:
        df: Input pandas DataFrame to inspect.

    Returns:
        Structured, JSON-serializable quality report dictionary.

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

    # Execute modular quality checks
    missing_report = _check_missing_values(df, total_rows)
    duplicate_report = _check_duplicate_rows(df, total_rows, total_columns)
    const_empty_report = _check_constant_and_empty_columns(df, total_rows)
    potential_id_report = _check_potential_id_columns(df, total_rows)
    inconsistent_cat_report = _check_inconsistent_categorical_values(df)
    invalid_num_report = _check_invalid_numerical_values(df)

    # Determine if any quality issues were detected
    has_issues = bool(
        missing_report["total_affected_columns"] > 0
        or duplicate_report["duplicate_count"] > 0
        or len(const_empty_report["constant_columns"]) > 0
        or len(const_empty_report["empty_columns"]) > 0
        or len(potential_id_report) > 0
        or inconsistent_cat_report["total_affected_columns"] > 0
        or invalid_num_report["total_affected_columns"] > 0
    )

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "has_issues": has_issues,
        "issue_counts": {
            "missing_value_columns": missing_report["total_affected_columns"],
            "duplicate_rows": duplicate_report["duplicate_count"],
            "constant_columns": len(const_empty_report["constant_columns"]),
            "empty_columns": len(const_empty_report["empty_columns"]),
            "potential_id_columns": len(potential_id_report),
            "inconsistent_categorical_columns": inconsistent_cat_report["total_affected_columns"],
            "invalid_numerical_columns": invalid_num_report["total_affected_columns"],
        },
        "missing_values": missing_report,
        "duplicate_rows": duplicate_report,
        "constant_columns": const_empty_report["constant_columns"],
        "empty_columns": const_empty_report["empty_columns"],
        "potential_id_columns": potential_id_report,
        "inconsistent_categorical_values": inconsistent_cat_report,
        "invalid_numerical_values": invalid_num_report,
    }
