"""
Unit tests for the Anomaly Detection Engine (backend/app/ml_engine/anomaly.py).
"""

import json
import pytest
import numpy as np
import pandas as pd
from app.ml_engine.anomaly import detect_anomalies


def test_detect_anomalies_standard():
    """Verify anomaly detection on a standard dataset with obvious outliers."""
    # 30 points clustered around 10.0, plus 2 extreme outliers at 1000.0
    normal_x = [10.0 + (i % 3) for i in range(30)]
    normal_y = [20.0 + (i % 4) for i in range(30)]
    outlier_x = [1000.0, -500.0]
    outlier_y = [1000.0, -500.0]

    df = pd.DataFrame({
        "x": normal_x + outlier_x,
        "y": normal_y + outlier_y,
    })

    result = detect_anomalies(df, contamination=0.1, random_state=42)

    assert result["status"] == "completed"
    assert result["total_rows_analyzed"] == 32
    assert result["anomaly_count"] >= 1
    assert result["normal_count"] + result["anomaly_count"] == 32
    assert result["anomaly_percentage"] > 0.0
    assert len(result["anomaly_labels"]) == 32
    assert len(result["anomaly_scores"]) == 32
    assert len(result["is_anomaly"]) == 32
    assert result["features_used"] == ["x", "y"]

    # The extreme outlier at index 30 should be flagged
    assert result["is_anomaly"][30] is True
    assert result["anomaly_labels"][30] == -1

    # Verify clarification note in explanation
    assert "not necessarily indicate an error, defect, or fraud" in result["explanation"]


def test_detect_anomalies_automatic_id_exclusion():
    """Verify that ID columns are excluded from automatic feature selection."""
    df = pd.DataFrame({
        "user_id": [1001 + i for i in range(25)],
        "val_1": [float(i) for i in range(25)],
        "val_2": [i * 2.0 for i in range(25)],
    })

    result = detect_anomalies(df)

    assert "user_id" not in result["features_used"]
    assert "val_1" in result["features_used"]
    assert "val_2" in result["features_used"]


def test_detect_anomalies_feature_columns_selection():
    """Verify explicit selection of feature columns."""
    df = pd.DataFrame({
        "f1": [float(i) for i in range(20)],
        "f2": [i * 3.0 for i in range(20)],
        "ignore_me": [1.0] * 20,
    })

    result = detect_anomalies(df, feature_columns=["f1", "f2"])

    assert result["features_used"] == ["f1", "f2"]
    assert "ignore_me" not in result["features_used"]


def test_detect_anomalies_with_missing_and_infinite_values():
    """Verify safe handling of NaNs, +inf, and -inf in feature columns."""
    df = pd.DataFrame({
        "a": [10.0, 11.0, np.nan, 12.0, 10.0, np.inf, 11.5, 10.2, 100.0, -np.inf],
        "b": [5.0, np.nan, 5.2, 5.1, -np.inf, 5.3, 5.0, 5.1, 50.0, np.inf],
    })

    result = detect_anomalies(df, contamination=0.2, random_state=42)

    assert result["status"] == "completed"
    assert result["total_rows_analyzed"] == 10
    assert len(result["anomaly_scores"]) == 10


def test_detect_anomalies_small_and_empty_datasets():
    """Verify safe handling of small (<5 rows) and empty datasets."""
    # Empty DataFrame
    empty_df = pd.DataFrame()
    res_empty = detect_anomalies(empty_df)
    assert res_empty["status"] == "empty_dataset"
    assert res_empty["total_rows_analyzed"] == 0

    # Dataset with 3 rows (insufficient for reliable isolation forest)
    small_df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [10.0, 20.0, 30.0],
    })
    res_small = detect_anomalies(small_df)
    assert res_small["status"] == "insufficient_data"
    assert res_small["total_rows_analyzed"] == 3
    assert res_small["anomaly_count"] == 0
    assert res_small["is_anomaly"] == [False, False, False]


def test_detect_anomalies_no_numeric_features():
    """Verify behavior when dataset contains no numeric features."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "category": ["A", "B", "C", "D", "E"],
    })
    result = detect_anomalies(df)
    assert result["status"] == "no_numeric_features"
    assert result["features_used"] == []
    assert result["anomaly_count"] == 0


def test_detect_anomalies_immutability():
    """Verify that detect_anomalies does not modify the input DataFrame."""
    original_df = pd.DataFrame({
        "val": [1.0, np.nan, np.inf, -np.inf, 10.0, 100.0],
        "name": ["a", "b", "c", "d", "e", "f"],
    })
    copy_df = original_df.copy(deep=True)

    _ = detect_anomalies(original_df)

    pd.testing.assert_frame_equal(original_df, copy_df)


def test_detect_anomalies_json_serializability():
    """Verify that the anomaly report can be serialized to JSON without custom encoders."""
    df = pd.DataFrame({
        "x": [float(i) for i in range(25)],
        "y": [i * 1.5 for i in range(25)],
    })
    result = detect_anomalies(df)

    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["status"] == "completed"
    assert deserialized["total_rows_analyzed"] == 25


def test_detect_anomalies_invalid_inputs():
    """Verify error handling for invalid arguments and parameters."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        "text": ["a", "b", "c", "d", "e"],
    })

    # None DataFrame
    with pytest.raises(ValueError, match="cannot be None"):
        detect_anomalies(None)

    # Non-DataFrame type
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        detect_anomalies({"not": "a df"})

    # Non-existent feature column
    with pytest.raises(ValueError, match="do not exist in DataFrame"):
        detect_anomalies(df, feature_columns=["non_existent"])

    # Non-numeric feature column
    with pytest.raises(ValueError, match="are not numeric"):
        detect_anomalies(df, feature_columns=["text"])

    # Empty feature columns list
    with pytest.raises(ValueError, match="non-empty list"):
        detect_anomalies(df, feature_columns=[])

    # Invalid contamination float
    with pytest.raises(ValueError, match="Contamination must be"):
        detect_anomalies(df, contamination=0.9)

    # Invalid contamination string
    with pytest.raises(ValueError, match="Contamination must be"):
        detect_anomalies(df, contamination="invalid_string")
