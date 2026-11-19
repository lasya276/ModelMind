"""
Unit tests for the Data Quality Engine (backend/app/data_engine/quality.py).
"""

import json
import pytest
import numpy as np
import pandas as pd
from app.data_engine.quality import check_data_quality


def test_quality_clean_dataset():
    """Verify that a clean dataset reports no issues."""
    df = pd.DataFrame({
        "fruit": ["apple", "banana", "orange"],
        "price": [1.2, 0.5, 0.8],
    })
    report = check_data_quality(df)

    assert report["total_rows"] == 3
    assert report["total_columns"] == 2
    assert report["missing_values"]["total_affected_columns"] == 0
    assert report["duplicate_rows"]["duplicate_count"] == 0
    assert len(report["constant_columns"]) == 0
    assert len(report["empty_columns"]) == 0
    assert report["inconsistent_categorical_values"]["total_affected_columns"] == 0
    assert report["invalid_numerical_values"]["total_affected_columns"] == 0


def test_quality_missing_values():
    """Verify detection of missing values with counts, percentages, and severity."""
    df = pd.DataFrame({
        "high_missing": [1, np.nan, np.nan, np.nan],  # 75% -> high
        "med_missing": [1, 2, 3, np.nan],             # 25% -> medium
        "no_missing": [1, 2, 3, 4],                   # 0%
    })
    report = check_data_quality(df)
    missing = report["missing_values"]

    assert missing["total_affected_columns"] == 2

    high_col = next(c for c in missing["columns"] if c["column"] == "high_missing")
    assert high_col["missing_count"] == 3
    assert high_col["percentage"] == 75.0
    assert high_col["severity"] == "high"

    med_col = next(c for c in missing["columns"] if c["column"] == "med_missing")
    assert med_col["missing_count"] == 1
    assert med_col["percentage"] == 25.0
    assert med_col["severity"] == "medium"


def test_quality_duplicate_rows():
    """Verify detection of duplicate rows."""
    df = pd.DataFrame({
        "a": [1, 2, 1, 1],
        "b": ["x", "y", "x", "x"],
    })
    report = check_data_quality(df)
    dups = report["duplicate_rows"]

    assert dups["duplicate_count"] == 2
    assert dups["percentage"] == 50.0
    assert dups["severity"] == "high"


def test_quality_constant_and_empty_columns():
    """Verify distinguishing constant columns from completely empty columns."""
    df = pd.DataFrame({
        "constant_col": ["active", "active", "active"],
        "empty_col": [np.nan, np.nan, np.nan],
        "normal_col": [1, 2, 3],
    })
    report = check_data_quality(df)

    # Constant column check
    assert len(report["constant_columns"]) == 1
    assert report["constant_columns"][0]["column"] == "constant_col"
    assert report["constant_columns"][0]["constant_value"] == "active"
    assert report["constant_columns"][0]["severity"] == "medium"

    # Empty column check
    assert len(report["empty_columns"]) == 1
    assert report["empty_columns"][0]["column"] == "empty_col"
    assert report["empty_columns"][0]["severity"] == "high"


def test_quality_potential_id_columns():
    """Verify identifier heuristic flags (naming + uniqueness)."""
    df = pd.DataFrame({
        "user_id": [101, 102, 103, 104, 105],
        "non_id_unique": ["alpha", "beta", "gamma", "delta", "epsilon"],
        "category": ["A", "A", "B", "B", "C"],
    })
    report = check_data_quality(df)
    pot_ids = report["potential_id_columns"]

    id_names = [p["column"] for p in pot_ids]
    assert "user_id" in id_names

    user_id_entry = next(p for p in pot_ids if p["column"] == "user_id")
    assert user_id_entry["name_heuristic_matched"] is True
    assert user_id_entry["uniqueness_percentage"] == 100.0
    assert user_id_entry["is_potential_id"] is True

    # 'category' should not be flagged as ID
    assert "category" not in id_names


def test_quality_inconsistent_categorical_values():
    """Verify detection of casing and whitespace inconsistencies in categorical columns."""
    df = pd.DataFrame({
        "city": ["New York", "New York ", "new york", "Paris"],
        "country": ["USA", "USA", "France", "France"],
    })
    report = check_data_quality(df)
    inconsistent = report["inconsistent_categorical_values"]

    assert inconsistent["total_affected_columns"] == 1
    city_report = inconsistent["columns"][0]
    assert city_report["column"] == "city"

    # "new york" should be the normalized value with 3 variations
    incon_entry = city_report["inconsistencies"][0]
    assert incon_entry["normalized_value"] == "new york"
    assert set(incon_entry["variations"]) == {"New York", "New York ", "new york"}
    assert incon_entry["issue_type"] == "case_and_whitespace_differences"


def test_quality_invalid_numerical_values():
    """Verify detection of positive and negative infinity values."""
    df = pd.DataFrame({
        "values": [1.0, np.inf, -np.inf, 2.5, 3.0],
        "normal": [10, 20, 30, 40, 50],
    })
    report = check_data_quality(df)
    invalid = report["invalid_numerical_values"]

    assert invalid["total_affected_columns"] == 1
    val_report = invalid["columns"][0]
    assert val_report["column"] == "values"
    assert val_report["positive_infinity_count"] == 1
    assert val_report["negative_infinity_count"] == 1
    assert val_report["total_invalid_count"] == 2
    assert val_report["severity"] == "high"


def test_quality_empty_dataframe():
    """Verify safe handling of empty DataFrames."""
    empty_df = pd.DataFrame()
    report = check_data_quality(empty_df)

    assert report["total_rows"] == 0
    assert report["total_columns"] == 0
    assert report["has_issues"] is False
    assert report["duplicate_rows"]["duplicate_count"] == 0
    assert len(report["constant_columns"]) == 0
    assert len(report["empty_columns"]) == 0

    # DataFrame with columns but 0 rows
    cols_only = pd.DataFrame(columns=["a", "b"])
    report_cols = check_data_quality(cols_only)
    assert report_cols["total_rows"] == 0
    assert report_cols["total_columns"] == 2
    assert report_cols["has_issues"] is False


def test_quality_immutability():
    """Verify that quality checks do not modify the input DataFrame."""
    original_df = pd.DataFrame({
        "num": [1.0, np.inf, np.nan],
        "cat": ["Apple ", "apple", "Banana"],
    })
    copy_df = original_df.copy(deep=True)

    _ = check_data_quality(original_df)

    pd.testing.assert_frame_equal(original_df, copy_df)


def test_quality_json_serializability():
    """Verify that quality report can be serialized to JSON without custom encoders."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "num": [10.5, np.inf, -np.inf, np.nan],
        "text": ["Item ", "item", "Other", "Other"],
        "constant": [99, 99, 99, 99],
    })
    report = check_data_quality(df)

    # Should serialize without TypeError
    json_str = json.dumps(report)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["total_rows"] == 4
    assert deserialized["has_issues"] is True


def test_quality_invalid_input():
    """Verify exception handling when non-DataFrame inputs are passed."""
    with pytest.raises(ValueError, match="cannot be None"):
        check_data_quality(None)

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        check_data_quality(["not", "a", "df"])
