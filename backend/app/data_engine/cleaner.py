"""
Data Engine - Cleaner Module
Provides rule-based dataset cleaning based strictly on explicit user instructions:
1. Missing value imputation (numeric: mean, median, constant; categorical: mode, constant).
2. Duplicate row removal (only when explicitly requested).
3. Text consistency (whitespace trimming, lowercase/uppercase conversions).
4. Constant and empty column detection and optional removal.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def _validate_columns_exist(df: pd.DataFrame, columns: List[str], context: str) -> None:
    """
    Validates that all specified column names exist in the DataFrame.
    Raises ValueError with descriptive message if any column is missing.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} specified for {context} do not exist in the DataFrame. "
            f"Available columns are: {list(df.columns)}"
        )


def _safe_json_value(val: Any) -> Any:
    """
    Converts a value to a strictly JSON-serializable native Python type.
    """
    if val is None or pd.isna(val) or (isinstance(val, (float, np.floating)) and np.isinf(val)):
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return round(float(val), 4)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    return str(val)


def clean_dataset(
    df: pd.DataFrame,
    operations: Optional[Dict[str, Any]] = None,
    missing_values: Optional[Dict[str, Any]] = None,
    remove_duplicates: bool = False,
    duplicate_subset: Optional[List[str]] = None,
    trim_whitespace: Optional[Union[List[str], bool]] = None,
    casing: Optional[Dict[str, str]] = None,
    drop_empty_columns: Union[bool, List[str]] = False,
    drop_constant_columns: Union[bool, List[str]] = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies user-selected cleaning operations to a copy of the input DataFrame.
    Never modifies the original DataFrame.
    Never applies cleaning transformations without explicit instruction.

    Args:
        df: Input pandas DataFrame.
        operations: Optional dictionary consolidating cleaning directives.
        missing_values: Dict mapping column names to strategy ('mean', 'median', 'mode', or dict with constant value).
        remove_duplicates: Whether to remove duplicate rows.
        duplicate_subset: Optional subset of columns to consider when identifying duplicates.
        trim_whitespace: List of columns or True (for all text columns) to trim leading/trailing whitespace.
        casing: Dict mapping text column names to 'lower'/'lowercase' or 'upper'/'uppercase'.
        drop_empty_columns: True (drop all completely empty columns) or List of specific empty columns to drop.
        drop_constant_columns: True (drop all constant columns) or List of specific constant columns to drop.

    Returns:
        Tuple of (cleaned_df, cleaning_report).

    Raises:
        ValueError: If input DataFrame is None, strategy is invalid, or column names are not found.
        TypeError: If input is not a pandas DataFrame.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Merge configuration from 'operations' dict if provided
    if operations:
        if "missing_values" in operations and missing_values is None:
            missing_values = operations["missing_values"]
        if "remove_duplicates" in operations and not remove_duplicates:
            remove_duplicates = bool(operations["remove_duplicates"])
        if "duplicate_subset" in operations and duplicate_subset is None:
            duplicate_subset = operations["duplicate_subset"]
        if "trim_whitespace" in operations and trim_whitespace is None:
            trim_whitespace = operations["trim_whitespace"]
        if "casing" in operations and casing is None:
            casing = operations["casing"]
        if "drop_empty_columns" in operations and not drop_empty_columns:
            drop_empty_columns = operations["drop_empty_columns"]
        if "drop_constant_columns" in operations and not drop_constant_columns:
            drop_constant_columns = operations["drop_constant_columns"]

    # 1. Guarantee Immutability: work strictly on a deep copy
    cleaned_df = df.copy(deep=True)

    initial_rows = int(len(df))
    initial_columns = int(len(df.columns))

    operations_performed: List[Dict[str, Any]] = []
    missing_imputed_details: List[Dict[str, Any]] = []
    total_missing_imputed = 0
    trimmed_whitespace_details: List[Dict[str, Any]] = []
    casing_applied_details: List[Dict[str, Any]] = []
    dropped_empty_columns: List[str] = []
    dropped_constant_columns: List[str] = []

    # -------------------------------------------------------------
    # 2. Inconsistent Categorical Values: Trimming Whitespace
    # -------------------------------------------------------------
    if trim_whitespace:
        cols_to_trim: List[str] = []
        if isinstance(trim_whitespace, bool) and trim_whitespace:
            # Select all object / string columns
            cols_to_trim = [
                str(c) for c in cleaned_df.columns
                if pd.api.types.is_string_dtype(cleaned_df[c]) or pd.api.types.is_object_dtype(cleaned_df[c])
            ]
        elif isinstance(trim_whitespace, list):
            _validate_columns_exist(cleaned_df, trim_whitespace, "whitespace trimming")
            cols_to_trim = [str(c) for c in trim_whitespace]
        else:
            raise ValueError("trim_whitespace must be a boolean or a list of column names.")

        for col in cols_to_trim:
            series = cleaned_df[col]
            if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
                raise ValueError(
                    f"Cannot trim whitespace on non-text column '{col}' with dtype '{series.dtype}'."
                )

            # Count values having leading or trailing whitespace
            non_null = series.dropna()
            count_affected = int(sum(1 for val in non_null if isinstance(val, str) and val != val.strip()))

            cleaned_df[col] = series.apply(lambda v: v.strip() if isinstance(v, str) else v)

            if count_affected > 0:
                trimmed_whitespace_details.append({
                    "column": col,
                    "trimmed_count": count_affected,
                })
                operations_performed.append({
                    "operation": "trim_whitespace",
                    "column": col,
                    "affected_rows": count_affected,
                    "description": f"Trimmed leading/trailing whitespace on column '{col}' ({count_affected} values updated).",
                })

    # -------------------------------------------------------------
    # 3. Inconsistent Categorical Values: Consistent Casing
    # -------------------------------------------------------------
    if casing:
        if not isinstance(casing, dict):
            raise ValueError("casing must be a dictionary mapping column names to 'lower' or 'upper'.")

        _validate_columns_exist(cleaned_df, list(casing.keys()), "casing conversion")

        for col, case_type in casing.items():
            series = cleaned_df[col]
            if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
                raise ValueError(
                    f"Cannot apply casing to non-text column '{col}' with dtype '{series.dtype}'."
                )

            case_lower = str(case_type).lower().strip()
            if case_lower in ("lower", "lowercase"):
                target_case = "lower"
                func = lambda v: v.lower() if isinstance(v, str) else v
            elif case_lower in ("upper", "uppercase"):
                target_case = "upper"
                func = lambda v: v.upper() if isinstance(v, str) else v
            else:
                raise ValueError(
                    f"Invalid casing '{case_type}' for column '{col}'. Supported options are 'lower' or 'upper'."
                )

            non_null = series.dropna()
            count_affected = int(sum(1 for val in non_null if isinstance(val, str) and val != func(val)))

            cleaned_df[col] = series.apply(func)

            casing_applied_details.append({
                "column": col,
                "case_applied": target_case,
                "affected_count": count_affected,
            })
            operations_performed.append({
                "operation": "casing_conversion",
                "column": col,
                "target_case": target_case,
                "affected_rows": count_affected,
                "description": f"Converted values in column '{col}' to {target_case}case ({count_affected} values updated).",
            })

    # -------------------------------------------------------------
    # 4. Missing Value Imputation
    # -------------------------------------------------------------
    if missing_values:
        if not isinstance(missing_values, dict):
            raise ValueError("missing_values must be a dictionary mapping column names to strategies.")

        _validate_columns_exist(cleaned_df, list(missing_values.keys()), "missing value imputation")

        for col, spec in missing_values.items():
            series = cleaned_df[col]
            missing_count = int(series.isna().sum())

            # Parse strategy and optional constant value
            if isinstance(spec, str):
                strategy = spec.lower().strip()
                fill_value = None
            elif isinstance(spec, dict):
                strategy = str(spec.get("strategy", "")).lower().strip()
                fill_value = spec.get("value")
            else:
                # Direct constant value passed (e.g. {"age": 0} or {"status": "Unknown"})
                strategy = "constant"
                fill_value = spec

            is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

            # Validate strategy and column data type compatibility upfront
            if strategy in ("mean", "median"):
                if not is_numeric:
                    raise ValueError(
                        f"Strategy '{strategy}' can only be applied to numeric columns, but column '{col}' has dtype '{series.dtype}'."
                    )
                clean_num = series.dropna()
                fill_value = clean_num.mean() if strategy == "mean" and not clean_num.empty else (
                    clean_num.median() if not clean_num.empty else None
                )

            elif strategy == "mode":
                clean_series = series.dropna()
                if not clean_series.empty:
                    mode_res = clean_series.mode()
                    fill_value = mode_res.iloc[0] if not mode_res.empty else None
                else:
                    fill_value = None

            elif strategy == "constant":
                if fill_value is None:
                    raise ValueError(
                        f"Strategy 'constant' for column '{col}' requires a valid non-null 'value' parameter."
                    )
            else:
                raise ValueError(
                    f"Invalid missing value strategy '{strategy}' for column '{col}'. "
                    "Supported strategies are: 'mean', 'median', 'mode', 'constant'."
                )

            # Skip imputation if column currently has no missing values
            if missing_count == 0:
                continue

            # Apply imputation if fill_value was determined
            if fill_value is not None:
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                imputed_count = missing_count
                total_missing_imputed += imputed_count

                safe_val = _safe_json_value(fill_value)
                missing_imputed_details.append({
                    "column": col,
                    "strategy": strategy,
                    "imputed_value": safe_val,
                    "imputed_count": imputed_count,
                })
                operations_performed.append({
                    "operation": "impute_missing_values",
                    "column": col,
                    "strategy": strategy,
                    "imputed_value": safe_val,
                    "imputed_count": imputed_count,
                    "description": f"Imputed {imputed_count} missing value(s) in column '{col}' using strategy '{strategy}' (value: {safe_val}).",
                })

    # -------------------------------------------------------------
    # 5. Duplicate Rows Removal
    # -------------------------------------------------------------
    removed_duplicates_count = 0
    if remove_duplicates and len(cleaned_df) > 0:
        if duplicate_subset:
            _validate_columns_exist(cleaned_df, duplicate_subset, "duplicate removal subset")

        dups_count = int(cleaned_df.duplicated(subset=duplicate_subset).sum())
        if dups_count > 0:
            cleaned_df = cleaned_df.drop_duplicates(subset=duplicate_subset, keep="first").reset_index(drop=True)
            removed_duplicates_count = dups_count
            operations_performed.append({
                "operation": "remove_duplicates",
                "removed_rows": dups_count,
                "subset": duplicate_subset if duplicate_subset else "all_columns",
                "description": f"Removed {dups_count} duplicate row(s) (subset: {duplicate_subset if duplicate_subset else 'all columns'}).",
            })

    # -------------------------------------------------------------
    # 6. Constant and Empty Columns: Detection and Optional Removal
    # -------------------------------------------------------------
    detected_empty: List[str] = []
    detected_constant: List[str] = []

    if initial_rows > 0:
        for col in list(cleaned_df.columns):
            series = cleaned_df[col]
            non_null = series.dropna()
            if non_null.empty:
                detected_empty.append(str(col))
            elif int(non_null.nunique()) == 1 and len(cleaned_df) > 1:
                detected_constant.append(str(col))

    # Drop empty columns only when explicitly requested
    if drop_empty_columns:
        if isinstance(drop_empty_columns, bool) and drop_empty_columns:
            cols_to_drop = [c for c in detected_empty if c in cleaned_df.columns]
        elif isinstance(drop_empty_columns, list):
            _validate_columns_exist(cleaned_df, drop_empty_columns, "drop empty columns")
            cols_to_drop = [c for c in drop_empty_columns if c in detected_empty and c in cleaned_df.columns]
        else:
            raise ValueError("drop_empty_columns must be a boolean or a list of column names.")

        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            dropped_empty_columns = cols_to_drop
            operations_performed.append({
                "operation": "drop_empty_columns",
                "columns": cols_to_drop,
                "description": f"Dropped {len(cols_to_drop)} completely empty column(s): {cols_to_drop}.",
            })

    # Drop constant columns only when explicitly requested
    if drop_constant_columns:
        if isinstance(drop_constant_columns, bool) and drop_constant_columns:
            cols_to_drop = [c for c in detected_constant if c in cleaned_df.columns]
        elif isinstance(drop_constant_columns, list):
            _validate_columns_exist(cleaned_df, drop_constant_columns, "drop constant columns")
            cols_to_drop = [c for c in drop_constant_columns if c in detected_constant and c in cleaned_df.columns]
        else:
            raise ValueError("drop_constant_columns must be a boolean or a list of column names.")

        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            dropped_constant_columns = cols_to_drop
            operations_performed.append({
                "operation": "drop_constant_columns",
                "columns": cols_to_drop,
                "description": f"Dropped {len(cols_to_drop)} constant column(s): {cols_to_drop}.",
            })

    # Unmodified detected issues (reported without changing)
    unmodified_empty = [c for c in detected_empty if c in cleaned_df.columns]
    unmodified_constant = [c for c in detected_constant if c in cleaned_df.columns]
    remaining_missing = {
        str(col): int(cleaned_df[col].isna().sum())
        for col in cleaned_df.columns
        if int(cleaned_df[col].isna().sum()) > 0
    }

    # -------------------------------------------------------------
    # 7. Compile Final Cleaning Report
    # -------------------------------------------------------------
    report: Dict[str, Any] = {
        "initial_shape": {
            "rows": initial_rows,
            "columns": initial_columns,
        },
        "cleaned_shape": {
            "rows": int(len(cleaned_df)),
            "columns": int(len(cleaned_df.columns)),
        },
        "operations_performed": operations_performed,
        "operations_count": len(operations_performed),
        "missing_values_imputed": {
            "total_imputed": total_missing_imputed,
            "details": missing_imputed_details,
        },
        "duplicates_removed": {
            "removed_count": removed_duplicates_count,
            "subset": duplicate_subset if duplicate_subset else "all_columns",
        },
        "text_formatting": {
            "whitespace_trimmed": trimmed_whitespace_details,
            "casing_applied": casing_applied_details,
        },
        "columns_dropped": {
            "empty_columns": dropped_empty_columns,
            "constant_columns": dropped_constant_columns,
        },
        "detected_unmodified_issues": {
            "empty_columns": unmodified_empty,
            "constant_columns": unmodified_constant,
            "remaining_missing_values": remaining_missing,
        },
    }

    return cleaned_df, report
