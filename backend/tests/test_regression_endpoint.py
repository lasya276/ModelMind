"""
API tests for the Regression endpoint (backend/app/routes/regression.py).
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data_engine.storage import get_dataset

client = TestClient(app)


def _make_regression_csv_bytes(n_rows: int = 30) -> bytes:
    """Helper to generate CSV bytes with numerical features and a numerical target."""
    lines = ["sqft,bedrooms,age,category,price"]
    for i in range(n_rows):
        sqft = 800 + i * 50
        bedrooms = 1 + (i % 4)
        age = 5 + i
        cat = "A" if i % 2 == 0 else "B"
        price = 100000 + sqft * 100 + bedrooms * 5000
        lines.append(f"{sqft},{bedrooms},{age},{cat},{price}")
    return "\n".join(lines).encode("utf-8")


# 1. Successful direct CSV upload with regression
def test_regression_successful_direct_upload():
    """Verify regression on directly uploaded CSV with numerical target."""
    csv_content = _make_regression_csv_bytes(30)
    response = client.post(
        "/regression",
        files={"file": ("housing.csv", csv_content, "text/csv")},
        data={"target_column": "price"},
    )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["filename"] == "housing.csv"
    assert "dataset_id" in res

    data = res["data"]
    assert data["status"] == "completed"
    assert data["target_column"] == "price"
    assert data["total_rows"] == 30
    assert data["train_rows"] + data["test_rows"] == 30
    assert "linear_regression" in data["models"]
    assert "decision_tree" in data["models"]
    assert "random_forest" in data["models"]

    # Verify metric keys
    for model_key in ["linear_regression", "decision_tree", "random_forest"]:
        model = data["models"][model_key]
        assert "mae" in model
        assert "mse" in model
        assert "rmse" in model
        assert "r2_score" in model
        preview = data["predictions_preview"][model_key]
        assert preview["returned_rows"] == data["test_rows"]
        assert len(preview["row_indices"]) == len(preview["actual_values"])
        assert len(preview["actual_values"]) == len(preview["predicted_values"])

    # Verify best model
    assert data["best_model"]["model_key"] in ["linear_regression", "decision_tree", "random_forest"]
    assert data["best_model"]["best_rmse"] is not None
    assert len(data["model_comparison"]) == 3

    # Verify target summary
    assert "target_summary" in data
    assert "mean" in data["target_summary"]
    assert "std" in data["target_summary"]
    assert "median" in data["target_summary"]


# 2. Feature columns specification
def test_regression_with_feature_columns():
    """Verify regression with explicitly specified feature columns."""
    csv_content = _make_regression_csv_bytes(30)
    response = client.post(
        "/regression",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "price", "feature_columns": "sqft,bedrooms"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert "sqft" in data["selected_feature_columns"]
    assert "bedrooms" in data["selected_feature_columns"]
    assert "age" not in data["selected_feature_columns"]


# 3. JSON feature columns format
def test_regression_json_feature_columns():
    """Verify feature_columns specified as JSON array."""
    csv_content = _make_regression_csv_bytes(30)
    response = client.post(
        "/regression",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={
            "target_column": "price",
            "feature_columns": json.dumps(["sqft", "age"]),
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["selected_feature_columns"] == ["sqft", "age"]


# 4. Via dataset_id query parameter
def test_regression_via_dataset_id_query():
    """Verify regression using existing dataset_id query parameter."""
    csv_content = _make_regression_csv_bytes(30)

    upload_res = client.post(
        "/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(
        f"/regression?dataset_id={dataset_id}",
        json={"target_column": "price"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert res.json()["dataset_id"] == dataset_id


# 5. Via JSON body
def test_regression_via_json_body():
    """Verify regression using dataset_id and parameters in JSON body."""
    csv_content = _make_regression_csv_bytes(30)

    upload_res = client.post(
        "/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(
        "/regression",
        json={
            "dataset_id": dataset_id,
            "target_column": "price",
            "feature_columns": ["sqft", "bedrooms"],
        },
    )
    assert res.status_code == 200
    assert res.json()["data"]["selected_feature_columns"] == ["sqft", "bedrooms"]


# 6. Missing target_column
def test_regression_missing_target_column():
    """Verify 400 Bad Request when target_column is not provided."""
    csv_content = _make_regression_csv_bytes(30)
    res = client.post(
        "/regression",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    assert res.status_code == 400
    assert "target_column" in res.json()["detail"].lower()


# 7. Nonexistent target column
def test_regression_nonexistent_target():
    """Verify 400 Bad Request when specified target column does not exist."""
    csv_content = _make_regression_csv_bytes(30)
    res = client.post(
        "/regression",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "nonexistent_column"},
    )
    assert res.status_code == 400
    assert "does not exist" in res.json()["detail"].lower()


# 8. Non-numeric target column
def test_regression_non_numeric_target():
    """Verify 400 Bad Request when target column is not numeric."""
    csv_content = _make_regression_csv_bytes(30)
    res = client.post(
        "/regression",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "category"},
    )
    assert res.status_code == 400
    assert "numerical" in res.json()["detail"].lower()


# 9. Nonexistent feature columns
def test_regression_nonexistent_feature_columns():
    """Verify 400 Bad Request when specified feature column does not exist."""
    csv_content = _make_regression_csv_bytes(30)
    res = client.post(
        "/regression",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "price", "feature_columns": "sqft,fake_col"},
    )
    assert res.status_code == 400
    assert "do not exist" in res.json()["detail"].lower()


# 10. Insufficient data (too few rows)
def test_regression_insufficient_data():
    """Verify 400 Bad Request when dataset has too few rows for train/test split."""
    lines = ["x,y"]
    for i in range(3):
        lines.append(f"{i},{i*2}")
    csv_content = "\n".join(lines).encode("utf-8")
    res = client.post(
        "/regression",
        files={"file": ("tiny.csv", csv_content, "text/csv")},
        data={"target_column": "y"},
    )
    assert res.status_code == 400
    assert "insufficient" in res.json()["detail"].lower()


# 11. Empty CSV
def test_regression_empty_csv():
    """Verify 400 Bad Request when an empty CSV is uploaded."""
    res = client.post(
        "/regression",
        files={"file": ("empty.csv", b"", "text/csv")},
        data={"target_column": "x"},
    )
    assert res.status_code == 400


# 12. Invalid file format
def test_regression_invalid_file_format():
    """Verify 400 Bad Request when non-CSV file is uploaded."""
    res = client.post(
        "/regression",
        files={"file": ("readme.txt", b"some text", "text/plain")},
        data={"target_column": "x"},
    )
    assert res.status_code == 400
    assert "Invalid file type" in res.json()["detail"]


# 13. Missing file and dataset_id
def test_regression_missing_inputs():
    """Verify 400 Bad Request when neither file nor dataset_id is provided."""
    res = client.post(
        "/regression",
        json={"target_column": "price"},
    )
    assert res.status_code == 400
    assert "No file or dataset_id provided" in res.json()["detail"]


# 14. Nonexistent dataset_id
def test_regression_nonexistent_dataset_id():
    """Verify 404 Not Found for a valid UUID that does not exist in storage."""
    fake_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    res = client.post(
        f"/regression?dataset_id={fake_uuid}",
        json={"target_column": "price"},
    )
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


# 15. Oversized file rejected
def test_regression_oversized_file_rejected(monkeypatch):
    """Verify 413 Content Too Large when uploaded file exceeds size limit."""
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)
    large_content = b"x,y,price\n" + (b"12345,67890,999999\n" * 150)
    res = client.post(
        "/regression",
        files={"file": ("large.csv", large_content, "text/csv")},
        data={"target_column": "price"},
    )
    assert res.status_code == 413


# 16. JSON serialization
def test_regression_json_serialization():
    """Verify regression response is strictly JSON-serializable."""
    csv_content = _make_regression_csv_bytes(30)
    res = client.post(
        "/regression",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "price"},
    )
    assert res.status_code == 200
    dumped = json.dumps(res.json())
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"


# 17. Original dataset remains unchanged
def test_regression_original_dataset_remains_unchanged():
    """Verify original dataset stored bytes are strictly preserved after regression."""
    original_csv = _make_regression_csv_bytes(30)

    upload_res = client.post(
        "/upload",
        files={"file": ("raw_data.csv", original_csv, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(
        f"/regression?dataset_id={dataset_id}",
        json={"target_column": "price"},
    )
    assert res.status_code == 200

    stored_bytes, stored_filename = get_dataset(dataset_id)
    assert stored_bytes == original_csv
    assert stored_filename == "raw_data.csv"
