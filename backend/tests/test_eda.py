"""
Unit tests for the Exploratory Data Analysis (EDA) Engine (backend/app/data_engine/eda.py).
"""

import json
import pytest
import numpy as np
import pandas as pd
from app.data_engine.eda import analyze_eda


def test_eda_dataset_summary():
    """Verify general dataset summary metrics."""
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "num2": [10, 20, 30],
        "cat1": ["a", "b", "c"],
        "bool1": [True, False, True],
    })
    report = analyze_eda(df)
    summary = report["dataset_summary"]

    assert summary["total_rows"] == 3
    assert summary["total_columns"] == 4
    assert summary["numeric_column_count"] == 2
    assert summary["categorical_column_count"] == 2
    assert "num1" in summary["numeric_columns"]
    assert "num2" in summary["numeric_columns"]
    assert "cat1" in summary["categorical_columns"]
    assert "bool1" in summary["categorical_columns"]
    assert summary["has_data"] is True


def test_eda_numerical_analysis_and_histograms():
    """Verify numerical statistics and histogram generation."""
    df = pd.DataFrame({
        "values": [10.0, 20.0, 30.0, 40.0, 50.0],
        "with_inf": [1.0, 2.0, np.inf, -np.inf, np.nan],
    })
    report = analyze_eda(df)
    num_analysis = report["numerical_analysis"]

    # Clean numeric column
    val_stats = num_analysis["values"]
    assert val_stats["mean"] == 30.0
    assert val_stats["median"] == 30.0
    assert val_stats["min"] == 10.0
    assert val_stats["max"] == 50.0
    assert val_stats["valid_count"] == 5
    assert len(val_stats["histogram"]["counts"]) > 0
    assert len(val_stats["histogram"]["bin_edges"]) == len(val_stats["histogram"]["counts"]) + 1
    assert len(val_stats["histogram"]["bin_labels"]) == len(val_stats["histogram"]["counts"])

    # Column with infinities and NaNs
    inf_stats = num_analysis["with_inf"]
    assert inf_stats["valid_count"] == 2  # Only 1.0 and 2.0 are finite numbers
    assert inf_stats["mean"] == 1.5
    assert inf_stats["min"] == 1.0
    assert inf_stats["max"] == 2.0


def test_eda_categorical_analysis_and_bar_chart():
    """Verify frequency distributions, top categories, and bucketed categories."""
    # 15 distinct categories to trigger the 'Other' bucket
    categories = [f"cat_{i}" for i in range(15)]
    data = categories + ["cat_0", "cat_0"]  # cat_0 appears 3 times
    df = pd.DataFrame({"category": data})

    report = analyze_eda(df)
    cat_stats = report["categorical_analysis"]["category"]

    assert cat_stats["unique_count"] == 15
    assert cat_stats["top_category"] == "cat_0"
    assert cat_stats["top_frequency"] == 3

    bar_data = cat_stats["bar_chart"]
    assert len(bar_data["categories"]) == 11  # Top 10 + 'Other'
    assert "Other" in bar_data["categories"]
    assert len(bar_data["counts"]) == 11
    assert len(bar_data["percentages"]) == 11
    assert round(sum(bar_data["percentages"]), 1) == 100.0


def test_eda_correlation_analysis_computed():
    """Verify Pearson correlation matrix, pairwise metrics, and heatmap data."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfect positive correlation (r = 1.0)
        "z": [5.0, 4.0, 3.0, 2.0, 1.0],  # Perfect negative correlation (r = -1.0)
    })
    report = analyze_eda(df)
    corr = report["correlation_analysis"]

    assert corr["status"] == "computed"
    assert "Correlation measures linear association" in corr["disclaimer"]
    assert "does not imply causation" in corr["disclaimer"]

    # Diagonal self-correlation
    assert corr["matrix"]["x"]["x"] == 1.0
    assert corr["matrix"]["x"]["y"] == 1.0
    assert corr["matrix"]["x"]["z"] == -1.0

    # Heatmap ready data
    assert corr["heatmap_ready"]["x"] == ["x", "y", "z"]
    assert len(corr["heatmap_ready"]["z"]) == 3

    # Pairwise list
    assert len(corr["pairs"]) == 3
    pair_xy = next(p for p in corr["pairs"] if set([p["column_1"], p["column_2"]]) == {"x", "y"})
    assert pair_xy["correlation"] == 1.0
    assert pair_xy["strength"] == "strong"


def test_eda_correlation_insufficient_data():
    """Verify safe fallback when dataset has fewer than 2 numeric columns or constant columns."""
    # Only 1 numeric column
    df_single_num = pd.DataFrame({
        "num": [1, 2, 3],
        "text": ["a", "b", "c"],
    })
    report_single = analyze_eda(df_single_num)
    assert report_single["correlation_analysis"]["status"] == "insufficient_data"

    # Constant numeric column (0 variance)
    df_const = pd.DataFrame({
        "const": [5.0, 5.0, 5.0],
        "normal": [1.0, 2.0, 3.0],
    })
    report_const = analyze_eda(df_const)
    assert report_const["correlation_analysis"]["status"] == "insufficient_data"


def test_eda_missing_value_analysis():
    """Verify missing value counts, percentages, and chart-ready sorting."""
    df = pd.DataFrame({
        "no_missing": [1, 2, 3, 4],
        "high_missing": [1, np.nan, np.nan, np.nan],  # 75%
        "med_missing": [1, 2, np.nan, 4],             # 25%
    })
    report = analyze_eda(df)
    missing = report["missing_value_analysis"]

    assert missing["total_missing_cells"] == 4
    assert missing["total_cells"] == 12
    assert missing["overall_missing_percentage"] == round(4 / 12 * 100, 2)

    chart_data = missing["chart_data"]
    # Sorted descending by missingness
    assert chart_data["columns"][0] == "high_missing"
    assert chart_data["missing_counts"][0] == 3
    assert chart_data["missing_percentages"][0] == 75.0


def test_eda_empty_dataframe():
    """Verify safe handling of completely empty DataFrames."""
    empty_df = pd.DataFrame()
    report = analyze_eda(empty_df)

    assert report["dataset_summary"]["total_rows"] == 0
    assert report["dataset_summary"]["total_columns"] == 0
    assert report["dataset_summary"]["has_data"] is False
    assert report["correlation_analysis"]["status"] == "insufficient_data"
    assert report["missing_value_analysis"]["total_missing_cells"] == 0


def test_eda_all_null_columns():
    """Verify handling of columns where all values are NaN."""
    df = pd.DataFrame({
        "null_num": pd.Series([np.nan, np.nan], dtype="float64"),
        "null_cat": pd.Series([np.nan, np.nan], dtype="object"),
    })
    report = analyze_eda(df)

    assert report["numerical_analysis"]["null_num"]["valid_count"] == 0
    assert report["numerical_analysis"]["null_num"]["mean"] is None
    assert report["categorical_analysis"]["null_cat"]["valid_count"] == 0
    assert report["categorical_analysis"]["null_cat"]["top_category"] is None


def test_eda_immutability():
    """Verify that analyzing EDA does not alter the input DataFrame."""
    original_df = pd.DataFrame({
        "num": [10.0, np.inf, np.nan],
        "text": [" apple ", "banana", "apple "],
    })
    copy_df = original_df.copy(deep=True)

    _ = analyze_eda(original_df)

    pd.testing.assert_frame_equal(original_df, copy_df)


def test_eda_json_serializability():
    """Verify that entire EDA report is strictly JSON-serializable."""
    df = pd.DataFrame({
        "a": [1.5, 2.5, 3.5, 4.5],
        "b": [10.0, 20.0, 30.0, 40.0],
        "c": ["red", "blue", "green", "red"],
    })
    report = analyze_eda(df)

    json_str = json.dumps(report)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["dataset_summary"]["total_rows"] == 4


def test_eda_invalid_inputs():
    """Verify exception handling when non-DataFrame inputs are passed."""
    with pytest.raises(ValueError, match="cannot be None"):
        analyze_eda(None)

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        analyze_eda(["invalid", "input"])
