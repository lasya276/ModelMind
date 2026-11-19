"""
Unit tests for the Data Cleaning Engine (backend/app/data_engine/cleaner.py).
"""

import json
import pytest
import numpy as np
import pandas as pd
from app.data_engine.cleaner import clean_dataset


def test_cleaner_no_operations_preserves_data():
    """Verify that calling clean_dataset with no instructions performs zero transformations."""
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0],
        "cat": ["A ", "b", "A "],
        "dup": [10, 10, 10],
    })
    copy_df = df.copy(deep=True)

    cleaned_df, report = clean_dataset(df)

    # DataFrame values and shapes remain identical
    pd.testing.assert_frame_equal(cleaned_df, copy_df)
    assert report["operations_count"] == 0
    assert report["missing_values_imputed"]["total_imputed"] == 0
    assert report["duplicates_removed"]["removed_count"] == 0


def test_impute_numeric_mean_median_constant():
    """Verify mean, median, and constant imputation on numeric columns."""
    df = pd.DataFrame({
        "mean_col": [10.0, 20.0, 30.0, np.nan],  # mean = 20.0
        "median_col": [1.0, 2.0, 100.0, np.nan], # median = 2.0
        "const_col": [5.0, np.nan, 15.0, np.nan], # const = 0.0
    })

    cleaned_df, report = clean_dataset(
        df,
        missing_values={
            "mean_col": "mean",
            "median_col": "median",
            "const_col": {"strategy": "constant", "value": 0.0},
        }
    )

    assert cleaned_df["mean_col"].isna().sum() == 0
    assert cleaned_df["mean_col"].iloc[3] == 20.0

    assert cleaned_df["median_col"].isna().sum() == 0
    assert cleaned_df["median_col"].iloc[3] == 2.0

    assert cleaned_df["const_col"].isna().sum() == 0
    assert cleaned_df["const_col"].iloc[1] == 0.0
    assert cleaned_df["const_col"].iloc[3] == 0.0

    assert report["missing_values_imputed"]["total_imputed"] == 4


def test_impute_categorical_mode_and_constant():
    """Verify mode and constant string imputation on categorical columns."""
    df = pd.DataFrame({
        "mode_col": ["apple", "banana", "apple", np.nan],  # mode = 'apple'
        "const_col": ["active", np.nan, "inactive", np.nan],
    })

    cleaned_df, report = clean_dataset(
        df,
        missing_values={
            "mode_col": "mode",
            "const_col": {"strategy": "constant", "value": "Unknown"},
        }
    )

    assert cleaned_df["mode_col"].isna().sum() == 0
    assert cleaned_df["mode_col"].iloc[3] == "apple"

    assert cleaned_df["const_col"].isna().sum() == 0
    assert cleaned_df["const_col"].iloc[1] == "Unknown"
    assert cleaned_df["const_col"].iloc[3] == "Unknown"


def test_unspecified_missing_columns_remain_unchanged():
    """Verify that columns with missing values are left untouched if not specified in cleaning."""
    df = pd.DataFrame({
        "impute_me": [10.0, np.nan, 30.0],
        "leave_me": ["x", np.nan, "z"],
    })

    cleaned_df, report = clean_dataset(
        df,
        missing_values={"impute_me": "mean"}
    )

    assert cleaned_df["impute_me"].isna().sum() == 0
    assert cleaned_df["leave_me"].isna().sum() == 1
    assert "leave_me" in report["detected_unmodified_issues"]["remaining_missing_values"]


def test_remove_duplicate_rows():
    """Verify that duplicates are removed only when explicitly requested."""
    df = pd.DataFrame({
        "a": [1, 2, 1, 1],
        "b": ["x", "y", "x", "x"],
    })

    # 1. Not removed by default
    not_cleaned, rep1 = clean_dataset(df, remove_duplicates=False)
    assert len(not_cleaned) == 4
    assert rep1["duplicates_removed"]["removed_count"] == 0

    # 2. Removed when requested
    cleaned, rep2 = clean_dataset(df, remove_duplicates=True)
    assert len(cleaned) == 2
    assert rep2["duplicates_removed"]["removed_count"] == 2


def test_remove_duplicates_with_subset():
    """Verify duplicate removal scoped to a subset of columns."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "email": ["user@test.com", "user@test.com", "other@test.com"],
        "age": [25, 30, 40],
    })

    cleaned, report = clean_dataset(
        df,
        remove_duplicates=True,
        duplicate_subset=["email"]
    )
    assert len(cleaned) == 2
    assert report["duplicates_removed"]["removed_count"] == 1


def test_trim_whitespace_and_casing():
    """Verify whitespace trimming and casing transformations on text columns."""
    df = pd.DataFrame({
        "city": [" New York ", "London  ", "  Paris"],
        "status": ["Pending", "approved", "REJECTED"],
    })

    cleaned_df, report = clean_dataset(
        df,
        trim_whitespace=True,
        casing={"city": "upper", "status": "lower"},
    )

    assert list(cleaned_df["city"]) == ["NEW YORK", "LONDON", "PARIS"]
    assert list(cleaned_df["status"]) == ["pending", "approved", "rejected"]
    assert len(report["text_formatting"]["whitespace_trimmed"]) == 1
    assert len(report["text_formatting"]["casing_applied"]) == 2


def test_constant_and_empty_columns_reported_and_dropped_on_request():
    """Verify that constant and empty columns are not deleted unless explicitly requested."""
    df = pd.DataFrame({
        "valid": [1, 2, 3],
        "empty_col": [np.nan, np.nan, np.nan],
        "const_col": ["same", "same", "same"],
    })

    # Default: reported, but NOT dropped
    df_kept, rep1 = clean_dataset(df)
    assert "empty_col" in df_kept.columns
    assert "const_col" in df_kept.columns
    assert "empty_col" in rep1["detected_unmodified_issues"]["empty_columns"]
    assert "const_col" in rep1["detected_unmodified_issues"]["constant_columns"]

    # Explicit drop
    df_dropped, rep2 = clean_dataset(
        df,
        drop_empty_columns=True,
        drop_constant_columns=True,
    )
    assert "empty_col" not in df_dropped.columns
    assert "const_col" not in df_dropped.columns
    assert "valid" in df_dropped.columns
    assert rep2["columns_dropped"]["empty_columns"] == ["empty_col"]
    assert rep2["columns_dropped"]["constant_columns"] == ["const_col"]


def test_cleaner_validation_errors():
    """Verify validation errors for nonexistent columns, invalid strategies, and bad types."""
    df = pd.DataFrame({
        "num": [1, 2, np.nan],
        "cat": ["a", "b", "c"],
    })

    # Nonexistent column in missing_values
    with pytest.raises(ValueError, match="do not exist in the DataFrame"):
        clean_dataset(df, missing_values={"nonexistent": "mean"})

    # Nonexistent column in casing
    with pytest.raises(ValueError, match="do not exist in the DataFrame"):
        clean_dataset(df, casing={"unknown": "lower"})

    # Invalid strategy
    with pytest.raises(ValueError, match="Supported strategies are"):
        clean_dataset(df, missing_values={"num": "unsupported_strategy"})

    # Mean on non-numeric column
    with pytest.raises(ValueError, match="Strategy 'mean' can only be applied to numeric columns"):
        clean_dataset(df, missing_values={"cat": "mean"})

    # Invalid casing option
    with pytest.raises(ValueError, match="Supported options are 'lower' or 'upper'"):
        clean_dataset(df, casing={"cat": "capitalized"})

    # Missing constant value
    with pytest.raises(ValueError, match="requires a valid non-null 'value'"):
        clean_dataset(df, missing_values={"num": {"strategy": "constant"}})


def test_cleaner_immutability():
    """Verify that the original DataFrame is never modified."""
    original_df = pd.DataFrame({
        "num": [10.0, np.nan, 30.0],
        "text": [" apple ", " banana", "cherry "],
        "dup": [1, 1, 1],
    })
    copy_df = original_df.copy(deep=True)

    _ = clean_dataset(
        original_df,
        missing_values={"num": "median"},
        trim_whitespace=True,
        casing={"text": "upper"},
        remove_duplicates=True,
    )

    pd.testing.assert_frame_equal(original_df, copy_df)


def test_cleaner_json_serializability():
    """Verify that cleaning report is completely JSON-serializable."""
    df = pd.DataFrame({
        "a": [1.5, np.nan, 3.5],
        "b": [" X ", "Y", "Z"],
    })
    _, report = clean_dataset(
        df,
        missing_values={"a": "mean"},
        trim_whitespace=True,
        casing={"b": "lower"},
    )

    json_str = json.dumps(report)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["operations_count"] >= 1


def test_cleaner_empty_dataframe():
    """Verify safe handling of empty DataFrames."""
    empty_df = pd.DataFrame()
    cleaned_df, report = clean_dataset(
        empty_df,
        remove_duplicates=True,
        drop_empty_columns=True,
    )

    assert len(cleaned_df) == 0
    assert report["initial_shape"]["rows"] == 0
    assert report["cleaned_shape"]["rows"] == 0
    assert report["operations_count"] == 0


def test_cleaner_invalid_inputs():
    """Verify exception handling when non-DataFrame inputs are passed."""
    with pytest.raises(ValueError, match="cannot be None"):
        clean_dataset(None)

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        clean_dataset("not a dataframe")
