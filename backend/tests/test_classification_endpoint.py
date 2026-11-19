"""
API tests for the Classification endpoint (backend/app/routes/classification.py).
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data_engine.storage import get_dataset

client = TestClient(app)


def _make_classification_csv_bytes(n_per_class: int = 15) -> bytes:
    """Helper to generate CSV bytes with binary classification data."""
    lines = ["age,income,score,category,target"]
    for i in range(n_per_class):
        lines.append(f"{25 + i},{30000 + i * 1000},{60 + i},A,yes")
    for i in range(n_per_class):
        lines.append(f"{40 + i},{50000 + i * 1000},{40 + i},B,no")
    return "\n".join(lines).encode("utf-8")


def _make_multiclass_csv_bytes(n_per_class: int = 10) -> bytes:
    """Helper to generate CSV bytes with multiclass classification data."""
    lines = ["x,y,label"]
    for i in range(n_per_class):
        lines.append(f"{1.0 + i * 0.1},{2.0 + i * 0.2},class_a")
    for i in range(n_per_class):
        lines.append(f"{5.0 + i * 0.1},{6.0 + i * 0.2},class_b")
    for i in range(n_per_class):
        lines.append(f"{9.0 + i * 0.1},{10.0 + i * 0.2},class_c")
    return "\n".join(lines).encode("utf-8")


# 1. Successful direct CSV upload with classification
def test_classify_successful_direct_upload():
    """Verify classification on directly uploaded CSV with binary target."""
    csv_content = _make_classification_csv_bytes(15)
    response = client.post(
        "/classify",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "target"},
    )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["filename"] == "data.csv"
    assert "dataset_id" in res

    data = res["data"]
    assert data["status"] == "completed"
    assert data["target_column"] == "target"
    assert data["classification_type"] == "binary"
    assert data["total_rows"] == 30
    assert data["train_rows"] + data["test_rows"] == 30
    assert "logistic_regression" in data["models"]
    assert "decision_tree" in data["models"]
    assert "random_forest" in data["models"]
    assert data["best_model"]["model_key"] in ["logistic_regression", "decision_tree", "random_forest"]
    assert data["best_model"]["f1_score"] is not None
    assert len(data["model_comparison"]) == 3

    # Verify confusion matrices
    for model_key in ["logistic_regression", "decision_tree", "random_forest"]:
        cm = data["confusion_matrices"][model_key]
        assert "labels" in cm
        assert "matrix" in cm
        assert len(cm["labels"]) == 2  # binary


# 2. Multiclass classification
def test_classify_multiclass():
    """Verify classification on multiclass target."""
    csv_content = _make_multiclass_csv_bytes(10)
    response = client.post(
        "/classify",
        files={"file": ("multi.csv", csv_content, "text/csv")},
        data={"target_column": "label"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["classification_type"] == "multiclass"
    assert len(data["confusion_matrices"]["logistic_regression"]["labels"]) == 3


# 3. Feature columns specification
def test_classify_with_feature_columns():
    """Verify classification with explicitly specified feature columns."""
    csv_content = _make_classification_csv_bytes(15)
    response = client.post(
        "/classify",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "target", "feature_columns": "age,income"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert "age" in data["selected_feature_columns"]
    assert "income" in data["selected_feature_columns"]
    assert "score" not in data["selected_feature_columns"]


# 4. JSON feature columns format
def test_classify_json_feature_columns():
    """Verify feature_columns specified as JSON array."""
    csv_content = _make_classification_csv_bytes(15)
    response = client.post(
        "/classify",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={
            "target_column": "target",
            "feature_columns": json.dumps(["age", "score"]),
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["selected_feature_columns"] == ["age", "score"]


# 5. Via dataset_id query parameter
def test_classify_via_dataset_id_query():
    """Verify classification using existing dataset_id query parameter."""
    csv_content = _make_classification_csv_bytes(15)

    upload_res = client.post(
        "/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(
        f"/classify?dataset_id={dataset_id}",
        json={"target_column": "target"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert res.json()["dataset_id"] == dataset_id


# 6. Via JSON body
def test_classify_via_json_body():
    """Verify classification using dataset_id and parameters in JSON body."""
    csv_content = _make_classification_csv_bytes(15)

    upload_res = client.post(
        "/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(
        "/classify",
        json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "feature_columns": ["age", "income"],
        },
    )
    assert res.status_code == 200
    assert res.json()["data"]["selected_feature_columns"] == ["age", "income"]


# 7. Missing target_column
def test_classify_missing_target_column():
    """Verify 400 Bad Request when target_column is not provided."""
    csv_content = _make_classification_csv_bytes(15)
    res = client.post(
        "/classify",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    assert res.status_code == 400
    assert "target_column" in res.json()["detail"].lower()


# 8. Nonexistent target column
def test_classify_nonexistent_target():
    """Verify 400 Bad Request when specified target column does not exist."""
    csv_content = _make_classification_csv_bytes(15)
    res = client.post(
        "/classify",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "nonexistent_column"},
    )
    assert res.status_code == 400
    assert "does not exist" in res.json()["detail"].lower()


# 9. Nonexistent feature columns
def test_classify_nonexistent_feature_columns():
    """Verify 400 Bad Request when specified feature column does not exist."""
    csv_content = _make_classification_csv_bytes(15)
    res = client.post(
        "/classify",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "target", "feature_columns": "age,fake_col"},
    )
    assert res.status_code == 400
    assert "do not exist" in res.json()["detail"].lower()


# 10. Single class target (insufficient for classification)
def test_classify_single_class_target():
    """Verify 400 Bad Request when target has only one unique class."""
    lines = ["x,y,label"]
    for i in range(20):
        lines.append(f"{i},{i*2},same_class")
    csv_content = "\n".join(lines).encode("utf-8")
    res = client.post(
        "/classify",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "label"},
    )
    assert res.status_code == 400
    assert "at least 2" in res.json()["detail"].lower()


# 11. Empty CSV
def test_classify_empty_csv():
    """Verify 400 Bad Request when an empty CSV is uploaded."""
    res = client.post(
        "/classify",
        files={"file": ("empty.csv", b"", "text/csv")},
        data={"target_column": "x"},
    )
    assert res.status_code == 400


# 12. Invalid file format
def test_classify_invalid_file_format():
    """Verify 400 Bad Request when non-CSV file is uploaded."""
    res = client.post(
        "/classify",
        files={"file": ("script.sh", b"echo hello", "text/x-sh")},
        data={"target_column": "x"},
    )
    assert res.status_code == 400
    assert "Invalid file type" in res.json()["detail"]


# 13. Missing file and dataset_id
def test_classify_missing_inputs():
    """Verify 400 Bad Request when neither file nor dataset_id is provided."""
    res = client.post(
        "/classify",
        json={"target_column": "x"},
    )
    assert res.status_code == 400
    assert "No file or dataset_id provided" in res.json()["detail"]


# 14. Nonexistent dataset_id
def test_classify_nonexistent_dataset_id():
    """Verify 404 Not Found for a valid UUID that does not exist in storage."""
    fake_uuid = "11111111-2222-3333-4444-555555555555"
    res = client.post(
        f"/classify?dataset_id={fake_uuid}",
        json={"target_column": "x"},
    )
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


# 15. Oversized file rejected
def test_classify_oversized_file_rejected(monkeypatch):
    """Verify 413 Content Too Large when uploaded file exceeds size limit."""
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)
    large_content = b"x,y,target\n" + (b"12345,67890,yes\n" * 150)
    res = client.post(
        "/classify",
        files={"file": ("large.csv", large_content, "text/csv")},
        data={"target_column": "target"},
    )
    assert res.status_code == 413


# 16. JSON serialization
def test_classify_json_serialization():
    """Verify classification response is strictly JSON-serializable."""
    csv_content = _make_classification_csv_bytes(15)
    res = client.post(
        "/classify",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"target_column": "target"},
    )
    assert res.status_code == 200
    dumped = json.dumps(res.json())
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"


# 17. Original dataset remains unchanged
def test_classify_original_dataset_remains_unchanged():
    """Verify original dataset stored bytes are strictly preserved after classification."""
    original_csv = _make_classification_csv_bytes(15)

    upload_res = client.post(
        "/upload",
        files={"file": ("raw_data.csv", original_csv, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(
        f"/classify?dataset_id={dataset_id}",
        json={"target_column": "target"},
    )
    assert res.status_code == 200

    stored_bytes, stored_filename = get_dataset(dataset_id)
    assert stored_bytes == original_csv
    assert stored_filename == "raw_data.csv"
