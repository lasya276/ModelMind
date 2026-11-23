"""
Unit tests for the Regression Engine (backend/app/ml_engine/regression.py).
"""

import json
import pytest
import numpy as np
import pandas as pd
from app.ml_engine.regression import train_regression_model


def _create_deterministic_regression_dataset(n_samples: int = 60, seed: int = 42) -> pd.DataFrame:
    """Helper to create a reproducible dataset with numeric and categorical features and a continuous target."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(10.0, 2.0, n_samples)
    x2 = rng.uniform(0.0, 50.0, n_samples)
    cat = ["dept_A" if i % 2 == 0 else "dept_B" for i in range(n_samples)]
    # Target is linear function of x1 and x2 plus small noise
    target = 2.5 * x1 + 0.8 * x2 + rng.normal(0, 0.5, n_samples)

    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "dept": cat,
        "target": target,
    })


# 1. Basic regression with numerical features
def test_basic_regression_numerical_features():
    """Verify regression when all features are numerical."""
    df = _create_deterministic_regression_dataset(60)
    res = train_regression_model(
        df,
        target_column="target",
        feature_columns=["x1", "x2"],
        random_state=42,
    )

    assert res["status"] == "completed"
    assert res["categorical_features"] == []
    assert set(res["numerical_features"]) == {"x1", "x2"}
    assert len(res["models"]) == 3

    for model_key in ["linear_regression", "decision_tree", "random_forest"]:
        m = res["models"][model_key]
        assert m["mae"] is not None and m["mae"] >= 0.0
        assert m["mse"] is not None and m["mse"] >= 0.0
        assert m["rmse"] is not None and m["rmse"] >= 0.0
        assert m["r2_score"] is not None


# 2. Categorical features
def test_categorical_features_only():
    """Verify regression when all feature columns are categorical."""
    df = pd.DataFrame({
        "region": ["North", "South", "East", "West"] * 15,
        "tier": ["Level_1", "Level_2", "Level_3"] * 20,
        "target": [float(i) * 1.5 for i in range(60)],
    })
    res = train_regression_model(df, target_column="target", random_state=42)

    assert res["status"] == "completed"
    assert res["numerical_features"] == []
    assert set(res["categorical_features"]) == {"region", "tier"}
    assert len(res["models"]) == 3


# 3. Mixed numerical and categorical features
def test_mixed_numerical_and_categorical():
    """Verify regression with both numerical and categorical features."""
    df = _create_deterministic_regression_dataset(60)
    res = train_regression_model(df, target_column="target", random_state=42)

    assert set(res["numerical_features"]) == {"x1", "x2"}
    assert res["categorical_features"] == ["dept"]
    assert res["status"] == "completed"


# 4. Missing numerical feature values
def test_missing_numerical_feature_values():
    """Verify that missing values in numeric features are median-imputed safely."""
    df = _create_deterministic_regression_dataset(60)
    df.loc[0:5, "x1"] = np.nan
    df.loc[10:15, "x2"] = np.nan

    res = train_regression_model(df, target_column="target", random_state=42)
    assert res["status"] == "completed"
    assert res["total_rows"] == 60


# 5. Missing categorical feature values
def test_missing_categorical_feature_values():
    """Verify that missing values in categorical features are mode-imputed safely."""
    df = _create_deterministic_regression_dataset(60)
    df.loc[0:5, "dept"] = None

    res = train_regression_model(df, target_column="target", random_state=42)
    assert res["status"] == "completed"
    assert res["total_rows"] == 60


# 6. Infinite numerical feature values
def test_infinite_numerical_feature_values():
    """Verify that +inf and -inf in numerical features are converted and imputed safely."""
    df = _create_deterministic_regression_dataset(60)
    df.loc[2, "x1"] = np.inf
    df.loc[3, "x2"] = -np.inf

    res = train_regression_model(df, target_column="target", random_state=42)
    assert res["status"] == "completed"


# 7. User-selected feature columns
def test_user_selected_feature_columns():
    """Verify that only the requested subset of feature columns is trained on."""
    df = _create_deterministic_regression_dataset(60)
    res = train_regression_model(
        df,
        target_column="target",
        feature_columns=["x1", "dept"],
        random_state=42,
    )

    assert res["selected_feature_columns"] == ["x1", "dept"]
    assert "x2" not in res["selected_feature_columns"]


# 8. Invalid target column
def test_invalid_target_column():
    """Verify ValueError when target column does not exist in the DataFrame."""
    df = _create_deterministic_regression_dataset(30)
    with pytest.raises(ValueError, match="does not exist in the DataFrame"):
        train_regression_model(df, target_column="non_existent_column")


# 9. Non-numerical target
def test_non_numerical_target():
    """Verify ValueError when target column contains non-numerical values."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": ["low", "medium", "high", "low", "medium", "high"],
    })
    with pytest.raises(ValueError, match="must contain valid numerical values"):
        train_regression_model(df, target_column="target")


# 10. Missing target values
def test_missing_target_values():
    """Verify ValueError when target column contains missing values."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0],
    })
    with pytest.raises(ValueError, match="contains 1 missing value"):
        train_regression_model(df, target_column="target")


# 11. Infinite target values
def test_infinite_target_values():
    """Verify ValueError when target column contains infinite values."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [10.0, np.inf, 30.0, 40.0, 50.0, 60.0],
    })
    with pytest.raises(ValueError, match="contains infinite values"):
        train_regression_model(df, target_column="target")


# 12. Empty DataFrame
def test_empty_dataframe():
    """Verify ValueError when passing an empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        train_regression_model(empty_df, target_column="target")


# 13. Insufficient rows
def test_insufficient_rows():
    """Verify ValueError when dataset has fewer than 5 rows."""
    small_df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "target": [10.0, 20.0, 30.0],
    })
    with pytest.raises(ValueError, match="insufficient for train/test splitting"):
        train_regression_model(small_df, target_column="target")


# 14. Invalid test size
def test_invalid_test_size():
    """Verify ValueError when test_size is outside [0.05, 0.5]."""
    df = _create_deterministic_regression_dataset(30)
    with pytest.raises(ValueError, match="test_size must be a float between 0.05 and 0.5"):
        train_regression_model(df, target_column="target", test_size=0.9)


# 15. Target column excluded from features
def test_target_column_excluded_from_features():
    """Verify that target column is excluded from features even if specified by user."""
    df = _create_deterministic_regression_dataset(40)
    res = train_regression_model(
        df,
        target_column="target",
        feature_columns=["x1", "target"],
        random_state=42,
    )

    assert "target" not in res["selected_feature_columns"]
    assert res["selected_feature_columns"] == ["x1"]


# 16. Original DataFrame immutability
def test_dataframe_immutability():
    """Verify that training regression models never modifies the original DataFrame."""
    df = _create_deterministic_regression_dataset(50)
    df.loc[0, "x1"] = np.nan
    df.loc[1, "x2"] = np.inf
    copy_df = df.copy(deep=True)

    _ = train_regression_model(df, target_column="target", random_state=42)

    pd.testing.assert_frame_equal(df, copy_df)


# 17. JSON serializability
def test_json_serializability():
    """Verify that entire regression report can be serialized with standard json.dumps."""
    df = _create_deterministic_regression_dataset(50)
    res = train_regression_model(df, target_column="target", random_state=42)

    json_str = json.dumps(res)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["status"] == "completed"
    assert "best_model" in deserialized


# 18. Same train/test split across all models
def test_same_train_test_split_across_models():
    """Verify all 3 models evaluate on the exact same test rows count."""
    df = _create_deterministic_regression_dataset(50)
    res = train_regression_model(df, target_column="target", test_size=0.25, random_state=42)

    expected_test_rows = res["test_rows"]
    assert expected_test_rows > 0
    assert res["train_rows"] + expected_test_rows == 50
    assert len({
        tuple(preview["row_indices"])
        for preview in res["predictions_preview"].values()
    }) == 1


def test_prediction_preview_values_align_with_source_rows_and_are_bounded():
    df = _create_deterministic_regression_dataset(1200)
    result = train_regression_model(df, target_column="target", test_size=0.5)

    previews = result["predictions_preview"]
    assert set(previews) == {"linear_regression", "decision_tree", "random_forest"}
    for preview in previews.values():
        assert preview["returned_rows"] == 500
        assert preview["total_rows"] == result["test_rows"]
        assert preview["truncated"] is True
        assert len(preview["row_indices"]) == 500
        assert len(preview["actual_values"]) == 500
        assert len(preview["predicted_values"]) == 500
        assert preview["actual_values"] == [
            float(df.iloc[row_index]["target"])
            for row_index in preview["row_indices"]
        ]


# 19. No target leakage
def test_no_target_leakage():
    """Verify target column is never included in features or preprocessor inputs."""
    df = _create_deterministic_regression_dataset(40)
    res = train_regression_model(df, target_column="target", random_state=42)

    assert "target" not in res["selected_feature_columns"]
    assert "target" not in res["numerical_features"]
    assert "target" not in res["categorical_features"]


# 20. Best model selection based on lowest RMSE
def test_best_model_selection_lowest_rmse():
    """Verify best model corresponds to the model with the lowest RMSE score."""
    df = _create_deterministic_regression_dataset(60)
    res = train_regression_model(df, target_column="target", random_state=42)

    best_key = res["best_model"]["model_key"]
    best_rmse = res["best_model"]["best_rmse"]

    for model_key, model_eval in res["models"].items():
        assert model_eval["rmse"] >= best_rmse

    assert res["models"][best_key]["rmse"] == best_rmse
    assert res["best_model"]["selection_metric"] == "rmse (lowest)"


# 21. Constant target values
def test_constant_target_values():
    """Verify safe execution and handling when all target values are identical."""
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "x2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "target": [50.0] * 10,  # Constant target
    })
    res = train_regression_model(df, target_column="target", random_state=42)

    assert res["status"] == "completed"
    assert res["target_summary"]["std"] == 0.0
    for model_key in ["linear_regression", "decision_tree", "random_forest"]:
        m = res["models"][model_key]
        assert m["rmse"] is not None
        # R2 score on constant target is safely handled (0.0 or None)
        assert m["r2_score"] is None or isinstance(m["r2_score"], float)


# 22. Unseen categorical values in the test set
def test_unseen_categorical_values_in_test_set():
    """Verify OneHotEncoder handle_unknown='ignore' handles new categories in test set without crashing."""
    # 20 rows of Category A, 20 rows of Category B, and 1 row of Category C
    cats = ["Cat_A"] * 25 + ["Cat_B"] * 25
    targets = [float(i) for i in range(50)]
    df = pd.DataFrame({
        "category": cats,
        "target": targets,
    })
    # Replace test set's last row with a novel category
    df.loc[49, "category"] = "Novel_Cat_Z"

    res = train_regression_model(df, target_column="target", test_size=0.2, random_state=42)
    assert res["status"] == "completed"
    assert len(res["models"]) == 3


# 23. Invalid feature column names
def test_invalid_feature_column_names():
    """Verify ValueError when nonexistent column names are provided in feature_columns."""
    df = _create_deterministic_regression_dataset(30)
    with pytest.raises(ValueError, match="Specified feature column.*do not exist"):
        train_regression_model(
            df,
            target_column="target",
            feature_columns=["x1", "fake_col_1", "fake_col_2"],
        )
