"""
Unit tests for Dataset Intelligence Engine (backend/app/data_engine/intelligence.py).
"""

import json
import pytest
import numpy as np
import pandas as pd
from app.data_engine.intelligence import analyze_dataset_intelligence


def test_column_role_detection():
    """Verify accurate detection of column roles: numeric, categorical, boolean, datetime, potential_id."""
    df = pd.DataFrame({
        "user_id": [f"ID_{i}" for i in range(100)],
        "age": [20 + (i % 50) for i in range(100)],
        "city": ["New York", "London", "Tokyo", "Paris"] * 25,
        "is_active": [True, False] * 50,
        "signup_date": pd.date_range("2024-01-01", periods=100, freq="D"),
    })
    result = analyze_dataset_intelligence(df)
    roles = result["column_roles"]

    assert roles["user_id"]["role"] == "potential_id"
    assert roles["user_id"]["is_potential_id"] is True

    assert roles["age"]["role"] == "numeric"
    assert roles["city"]["role"] == "categorical"
    assert roles["is_active"]["role"] == "boolean"
    assert roles["signup_date"]["role"] == "datetime"

    assert result["role_counts"]["numeric"] == 1
    assert result["role_counts"]["categorical"] == 1
    assert result["role_counts"]["boolean"] == 1
    assert result["role_counts"]["datetime"] == 1
    assert result["role_counts"]["potential_id"] == 1


def test_possible_targets_identification():
    """Verify that discrete columns are suggested for classification and continuous for regression."""
    df = pd.DataFrame({
        "id": list(range(100)),
        "feature_1": [i * 0.5 for i in range(100)],
        "churn": ["yes", "no"] * 50,  # Discrete binary target candidate
        "salary": [30000 + i * 500 for i in range(100)],  # Continuous numeric target candidate
    })
    result = analyze_dataset_intelligence(df)
    targets = result["possible_targets"]
    target_names = [t["column"] for t in targets]

    # "churn" and "salary" should be identified as possible targets
    assert "churn" in target_names
    assert "salary" in target_names

    churn_target = next(t for t in targets if t["column"] == "churn")
    assert churn_target["suggested_task"] == "classification"
    assert churn_target["target_type"] == "binary"
    assert churn_target["status"] == "candidate_only"

    salary_target = next(t for t in targets if t["column"] == "salary")
    assert salary_target["suggested_task"] == "regression"
    assert salary_target["target_type"] == "continuous_numeric"
    assert salary_target["status"] == "candidate_only"


def test_id_column_exclusion_from_targets():
    """Verify that potential ID columns are never suggested as target columns."""
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(50)],
        "uuid_code": [f"uuid-{i}" for i in range(50)],
        "outcome": ["win", "loss"] * 25,
        "score": [float(i) for i in range(50)],
    })
    result = analyze_dataset_intelligence(df)
    targets = result["possible_targets"]
    target_names = [t["column"] for t in targets]

    assert "customer_id" not in target_names
    assert "uuid_code" not in target_names
    assert "outcome" in target_names


def test_analysis_recommendations_full_dataset():
    """Verify that all relevant analyses (EDA, Anomaly, Classification, Regression, Clustering) are recommended."""
    df = pd.DataFrame({
        "id": list(range(50)),
        "feat_a": [float(i) for i in range(50)],
        "feat_b": [i * 2.5 for i in range(50)],
        "target_class": ["A", "B", "C", "D", "E"] * 10,
        "target_value": [100.0 + i for i in range(50)],
    })
    result = analyze_dataset_intelligence(df)
    recommendations = {r["analysis_type"]: r for r in result["recommended_analyses"]}

    assert recommendations["exploratory_data_analysis"]["is_recommended"] is True
    assert recommendations["exploratory_data_analysis"]["requires_target"] is False

    assert recommendations["anomaly_detection"]["is_recommended"] is True
    assert recommendations["anomaly_detection"]["requires_target"] is False

    assert recommendations["classification"]["is_recommended"] is True
    assert recommendations["classification"]["requires_target"] is True
    assert "target_class" in recommendations["classification"]["candidate_targets"]

    assert recommendations["regression"]["is_recommended"] is True
    assert recommendations["regression"]["requires_target"] is True
    assert "target_value" in recommendations["regression"]["candidate_targets"]

    # 3 numeric columns ("feat_a", "feat_b", "target_value") and 50 rows -> clustering recommended
    assert recommendations["clustering"]["is_recommended"] is True
    assert recommendations["clustering"]["requires_target"] is False


def test_clustering_not_recommended_when_insufficient_numeric_features():
    """Verify clustering is marked as not recommended when there are fewer than 2 numeric features."""
    df = pd.DataFrame({
        "category": ["A", "B", "C"] * 10,
        "single_numeric": [float(i) for i in range(30)],
    })
    result = analyze_dataset_intelligence(df)
    recommendations = {r["analysis_type"]: r for r in result["recommended_analyses"]}

    assert recommendations["clustering"]["is_recommended"] is False
    assert "Requires at least 2 numeric features" in recommendations["clustering"]["suitability_reason"]


def test_empty_and_small_datasets():
    """Verify safe handling of empty and very small DataFrames."""
    # Completely empty DataFrame
    empty_df = pd.DataFrame()
    res_empty = analyze_dataset_intelligence(empty_df)
    assert res_empty["total_rows"] == 0
    assert res_empty["total_columns"] == 0
    assert res_empty["possible_targets"] == []
    for r in res_empty["recommended_analyses"]:
        assert r["is_recommended"] is False

    # Small DataFrame with 3 rows
    small_df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })
    res_small = analyze_dataset_intelligence(small_df)
    assert res_small["total_rows"] == 3
    # Too small for target identification or ML models
    assert res_small["possible_targets"] == []
    eda = next(r for r in res_small["recommended_analyses"] if r["analysis_type"] == "exploratory_data_analysis")
    assert eda["is_recommended"] is True


def test_dataframe_immutability():
    """Verify that analyzing dataset intelligence never alters the input DataFrame."""
    original_df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "score": [10.5, 20.0, 30.5, 40.0, 50.5],
        "category": ["high", "low", "high", "medium", "low"],
    })
    copy_df = original_df.copy(deep=True)

    _ = analyze_dataset_intelligence(original_df)

    pd.testing.assert_frame_equal(original_df, copy_df)


def test_json_serializability():
    """Verify that intelligence output can be serialized to JSON without custom encoders."""
    df = pd.DataFrame({
        "uid": [f"U_{i}" for i in range(25)],
        "num": [float(i) for i in range(25)],
        "status": ["active", "inactive"] * 12 + ["active"],
    })
    result = analyze_dataset_intelligence(df)

    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["total_rows"] == 25
    assert len(deserialized["recommended_analyses"]) == 5


def test_invalid_inputs():
    """Verify exception handling when non-DataFrame inputs are passed."""
    with pytest.raises(ValueError, match="cannot be None"):
        analyze_dataset_intelligence(None)

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        analyze_dataset_intelligence({"not": "a dataframe"})
