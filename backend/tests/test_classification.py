"""
Unit tests for the Classification Engine (backend/app/ml_engine/classification.py).
"""

import json
import pytest
import numpy as np
import pandas as pd
from app.ml_engine.classification import train_classification_model


def _create_deterministic_dataset(n_samples: int = 60, seed: int = 42) -> pd.DataFrame:
    """Helper to create a reproducible dataset with numeric and categorical features."""
    rng = np.random.RandomState(seed)
    feature_num1 = rng.normal(10.0, 2.0, n_samples)
    feature_num2 = rng.uniform(0.0, 50.0, n_samples)
    feature_cat = ["dept_A" if i % 2 == 0 else "dept_B" for i in range(n_samples)]
    # Target linearly correlated with feature_num1
    target = ["Pass" if val > 10.0 else "Fail" for val in feature_num1]

    return pd.DataFrame({
        "num1": feature_num1,
        "num2": feature_num2,
        "dept": feature_cat,
        "result": target,
    })


# 1. Binary classification
def test_binary_classification():
    """Verify training on a binary target with correct type detection and metrics."""
    df = _create_deterministic_dataset(60)
    res = train_classification_model(df, target_column="result", random_state=42)

    assert res["status"] == "completed"
    assert res["classification_type"] == "binary"
    assert res["target_column"] == "result"
    assert len(res["models"]) == 3
    for model_key in ["logistic_regression", "decision_tree", "random_forest"]:
        m = res["models"][model_key]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1_score"] <= 1.0
        assert m["averaging_method"] == "weighted"


# 2. Multiclass classification
def test_multiclass_classification():
    """Verify training on a 3-class target with confusion matrix shape matching class count."""
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "grade": ["A" if i % 3 == 0 else ("B" if i % 3 == 1 else "C") for i in range(n)],
    })
    res = train_classification_model(df, target_column="grade", random_state=42)

    assert res["classification_type"] == "multiclass"
    assert len(res["class_distribution"]) == 3
    # Check confusion matrices have 3x3 dimensions
    for model_key in ["logistic_regression", "decision_tree", "random_forest"]:
        cm_data = res["confusion_matrices"][model_key]
        assert len(cm_data["labels"]) == 3
        assert len(cm_data["matrix"]) == 3
        assert len(cm_data["matrix"][0]) == 3


# 3. Numerical features only
def test_numerical_features_only():
    """Verify classification when all feature columns are numeric."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "f1": rng.normal(0, 1, 50),
        "f2": rng.normal(10, 2, 50),
        "target": ["Yes" if i % 2 == 0 else "No" for i in range(50)],
    })
    res = train_classification_model(df, target_column="target", random_state=42)

    assert res["categorical_features"] == []
    assert set(res["numerical_features"]) == {"f1", "f2"}
    assert res["status"] == "completed"


# 4. Categorical features only
def test_categorical_features_only():
    """Verify classification when all feature columns are categorical strings."""
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "red"] * 15,
        "size": ["small", "medium", "large", "small"] * 15,
        "label": ["high", "low", "medium", "high"] * 15,
    })
    res = train_classification_model(df, target_column="label", random_state=42)

    assert res["numerical_features"] == []
    assert set(res["categorical_features"]) == {"color", "size"}
    assert res["status"] == "completed"


# 5. Mixed numerical and categorical features
def test_mixed_numerical_and_categorical():
    """Verify classification with both numeric and categorical columns present."""
    df = _create_deterministic_dataset(60)
    res = train_classification_model(df, target_column="result", random_state=42)

    assert set(res["numerical_features"]) == {"num1", "num2"}
    assert res["categorical_features"] == ["dept"]


# 6. Missing numerical values
def test_missing_numerical_values():
    """Verify that median imputer handles missing numeric values without raising errors."""
    df = _create_deterministic_dataset(60)
    df.loc[0:5, "num1"] = np.nan
    df.loc[10:15, "num2"] = np.nan

    res = train_classification_model(df, target_column="result", random_state=42)
    assert res["status"] == "completed"
    assert res["total_rows"] == 60


# 7. Missing categorical values
def test_missing_categorical_values():
    """Verify that mode imputer handles missing categorical values safely."""
    df = _create_deterministic_dataset(60)
    df.loc[0:5, "dept"] = None

    res = train_classification_model(df, target_column="result", random_state=42)
    assert res["status"] == "completed"
    assert res["total_rows"] == 60


# 8. Infinite numerical values
def test_infinite_numerical_values():
    """Verify that +inf and -inf in numerical features are replaced and imputed safely."""
    df = _create_deterministic_dataset(60)
    df.loc[2, "num1"] = np.inf
    df.loc[3, "num2"] = -np.inf

    res = train_classification_model(df, target_column="result", random_state=42)
    assert res["status"] == "completed"


# 9. User-selected feature columns
def test_user_selected_feature_columns():
    """Verify that only the requested subset of feature columns is trained on."""
    df = _create_deterministic_dataset(60)
    res = train_classification_model(
        df,
        target_column="result",
        feature_columns=["num1", "dept"],
        random_state=42,
    )

    assert res["selected_feature_columns"] == ["num1", "dept"]
    assert "num2" not in res["selected_feature_columns"]


# 10. Invalid target column
def test_invalid_target_column():
    """Verify ValueError is raised if target column is not found in the DataFrame."""
    df = _create_deterministic_dataset(30)
    with pytest.raises(ValueError, match="does not exist in the DataFrame"):
        train_classification_model(df, target_column="non_existent_column")


# 11. Target column excluded from features
def test_target_column_excluded_from_features():
    """Verify that even if user includes target in feature_columns, it is excluded to prevent leakage."""
    df = _create_deterministic_dataset(40)
    res = train_classification_model(
        df,
        target_column="result",
        feature_columns=["num1", "result"],
        random_state=42,
    )

    assert "result" not in res["selected_feature_columns"]
    assert res["selected_feature_columns"] == ["num1"]


# 12. Single-class target
def test_single_class_target():
    """Verify ValueError when target column contains only 1 distinct class."""
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "outcome": ["same", "same", "same", "same", "same"],
    })
    with pytest.raises(ValueError, match="requires at least 2 distinct classes"):
        train_classification_model(df, target_column="outcome")


# 13. Empty DataFrame
def test_empty_dataframe():
    """Verify ValueError when passing an empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        train_classification_model(empty_df, target_column="target")


# 14. Insufficient rows
def test_insufficient_rows():
    """Verify ValueError when dataset has fewer than 5 rows."""
    small_df = pd.DataFrame({
        "x": [1, 2, 3],
        "target": ["A", "B", "A"],
    })
    with pytest.raises(ValueError, match="insufficient for train/test splitting"):
        train_classification_model(small_df, target_column="target")


# 15. Rare class / stratification edge cases
def test_rare_class_stratification_edge_cases():
    """Verify graceful fallback to non-stratified split when a rare class has only 1 sample."""
    rng = np.random.RandomState(42)
    n = 30
    targets = ["A"] * 15 + ["B"] * 14 + ["Rare_C"] * 1  # Rare_C only appears once
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "target": targets,
    })
    res = train_classification_model(df, target_column="target", random_state=42)

    assert res["status"] == "completed"
    assert res["is_stratified"] is False  # Successfully fell back safely


# 16. Original DataFrame immutability
def test_dataframe_immutability():
    """Verify that training classification models never modifies the original DataFrame."""
    df = _create_deterministic_dataset(50)
    df.loc[0, "num1"] = np.nan
    df.loc[1, "num2"] = np.inf
    copy_df = df.copy(deep=True)

    _ = train_classification_model(df, target_column="result", random_state=42)

    pd.testing.assert_frame_equal(df, copy_df)


# 17. JSON serializability
def test_json_serializability():
    """Verify that the entire classification report is strictly JSON-serializable."""
    df = _create_deterministic_dataset(50)
    res = train_classification_model(df, target_column="result", random_state=42)

    json_str = json.dumps(res)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["status"] == "completed"
    assert "best_model" in deserialized


# 18. Same train/test split across models
def test_same_train_test_split_across_models():
    """Verify all 3 models evaluate on the exact same test sample size."""
    df = _create_deterministic_dataset(50)
    res = train_classification_model(df, target_column="result", test_size=0.25, random_state=42)

    expected_test_rows = res["test_rows"]
    assert expected_test_rows > 0

    # The sum of elements in each model's confusion matrix must equal test_rows
    for model_key in ["logistic_regression", "decision_tree", "random_forest"]:
        matrix = res["confusion_matrices"][model_key]["matrix"]
        total_test_in_cm = sum(sum(row) for row in matrix)
        assert total_test_in_cm == expected_test_rows


# 19. No target leakage
def test_no_target_leakage():
    """Verify target column is never included in features or preprocessor inputs."""
    df = _create_deterministic_dataset(40)
    res = train_classification_model(df, target_column="result", random_state=42)

    assert "result" not in res["selected_feature_columns"]
    assert "result" not in res["numerical_features"]
    assert "result" not in res["categorical_features"]


# 20. Best model selection based on weighted F1-score
def test_best_model_selection_weighted_f1():
    """Verify best model corresponds to the model with the highest weighted F1-score."""
    df = _create_deterministic_dataset(60)
    res = train_classification_model(df, target_column="result", random_state=42)

    best_key = res["best_model"]["model_key"]
    best_f1 = res["best_model"]["f1_score"]

    for model_key, model_eval in res["models"].items():
        assert model_eval["f1_score"] <= best_f1

    assert res["models"][best_key]["f1_score"] == best_f1
    assert res["best_model"]["selection_metric"] == "f1_score (weighted)"
