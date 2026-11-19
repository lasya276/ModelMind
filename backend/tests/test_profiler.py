"""
Unit tests for the Dataset Profiling Engine (backend/app/data_engine/profiler.py).
"""

import json
import pytest
import numpy as np
import pandas as pd
from app.data_engine.profiler import profile_dataset


def test_profile_dataset_basic():
    """Verify basic profiling metrics on a standard dataset."""
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["Paris", "London", "Tokyo"],
    })
    result = profile_dataset(df)

    assert result["total_rows"] == 3
    assert result["total_columns"] == 2
    assert result["columns"] == ["age", "city"]
    assert "age" in result["column_types"]
    assert "city" in result["column_types"]
    assert result["duplicate_rows"] == 0


def test_profile_numeric_columns():
    """Verify computation of mean, median, std, min, and max for numeric data."""
    df = pd.DataFrame({
        "scores": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    result = profile_dataset(df)
    stats = result["numeric_summary"]["scores"]

    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["std"] is not None
    assert round(stats["std"], 2) == 15.81


def test_profile_categorical_columns():
    """Verify computation of unique count, top value, and top frequency for categorical data."""
    df = pd.DataFrame({
        "fruit": ["apple", "banana", "apple", "orange", "apple", "banana"]
    })
    result = profile_dataset(df)
    cat_stats = result["categorical_summary"]["fruit"]

    assert cat_stats["unique_count"] == 3
    assert cat_stats["top_value"] == "apple"
    assert cat_stats["top_frequency"] == 3


def test_profile_missing_values():
    """Verify count and percentage calculations for missing values."""
    df = pd.DataFrame({
        "num": [1, np.nan, 3, np.nan],        # 2 missing out of 4 (50%)
        "cat": ["a", "b", np.nan, "d"],       # 1 missing out of 4 (25%)
        "full": [10, 20, 30, 40],             # 0 missing out of 4 (0%)
    })
    result = profile_dataset(df)

    assert result["missing_values"]["num"]["count"] == 2
    assert result["missing_values"]["num"]["percentage"] == 50.0

    assert result["missing_values"]["cat"]["count"] == 1
    assert result["missing_values"]["cat"]["percentage"] == 25.0

    assert result["missing_values"]["full"]["count"] == 0
    assert result["missing_values"]["full"]["percentage"] == 0.0


def test_profile_duplicate_rows():
    """Verify duplicate row counting."""
    df_with_dups = pd.DataFrame({
        "x": [1, 2, 1, 1],
        "y": ["a", "b", "a", "a"],
    })
    result = profile_dataset(df_with_dups)
    # Row (1, 'a') appears 3 times total, so there are 2 duplicate rows
    assert result["duplicate_rows"] == 2

    df_no_dups = pd.DataFrame({
        "x": [1, 2, 3],
        "y": ["a", "b", "c"],
    })
    assert profile_dataset(df_no_dups)["duplicate_rows"] == 0


def test_profile_empty_dataframe():
    """Verify safe handling of empty DataFrames."""
    # Completely empty DataFrame
    empty_df = pd.DataFrame()
    result_empty = profile_dataset(empty_df)
    assert result_empty["total_rows"] == 0
    assert result_empty["total_columns"] == 0
    assert result_empty["columns"] == []
    assert result_empty["duplicate_rows"] == 0

    # DataFrame with columns but 0 rows
    cols_only_df = pd.DataFrame(columns=["col_a", "col_b"])
    result_cols = profile_dataset(cols_only_df)
    assert result_cols["total_rows"] == 0
    assert result_cols["total_columns"] == 2
    assert result_cols["missing_values"]["col_a"]["count"] == 0
    assert result_cols["missing_values"]["col_a"]["percentage"] == 0.0


def test_profile_single_row_and_all_nan():
    """Verify single-row dataset (std is None) and all-NaN numeric columns."""
    # Single row: standard deviation is undefined
    single_row_df = pd.DataFrame({"val": [42.0]})
    res_single = profile_dataset(single_row_df)
    assert res_single["numeric_summary"]["val"]["std"] is None
    assert res_single["numeric_summary"]["val"]["mean"] == 42.0

    # All-NaN numeric column
    all_nan_df = pd.DataFrame({"val": [np.nan, np.nan, np.nan]})
    res_nan = profile_dataset(all_nan_df)
    nan_stats = res_nan["numeric_summary"]["val"]
    assert nan_stats["mean"] is None
    assert nan_stats["median"] is None
    assert nan_stats["std"] is None
    assert nan_stats["min"] is None
    assert nan_stats["max"] is None


def test_profile_immutability():
    """Verify that profiling does not alter the original DataFrame."""
    original_df = pd.DataFrame({
        "a": [1, 2, np.nan],
        "b": ["x", "y", "x"],
    })
    copy_df = original_df.copy(deep=True)

    _ = profile_dataset(original_df)

    # DataFrame equality and shape verification
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_profile_json_serializability():
    """Verify that all output values are strictly JSON-serializable."""
    df = pd.DataFrame({
        "num": [1.5, 2.5, np.nan, 4.5],
        "int_col": [1, 2, 3, 4],
        "cat": ["red", "blue", "red", None],
    })
    result = profile_dataset(df)

    # Should serialize without TypeError or NaN exceptions
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["total_rows"] == 4


def test_profile_invalid_inputs():
    """Verify exception handling for invalid inputs."""
    with pytest.raises(ValueError, match="cannot be None"):
        profile_dataset(None)

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        profile_dataset("invalid_string_input")
