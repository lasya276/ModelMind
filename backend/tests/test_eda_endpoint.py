"""
API tests for the Exploratory Data Analysis (EDA) endpoint (backend/app/routes/eda.py).
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data_engine.storage import get_dataset

client = TestClient(app)


# 1. Successful direct CSV upload
def test_eda_successful_direct_upload():
    """Verify comprehensive EDA report generated from directly uploaded CSV."""
    csv_content = (
        b"x,y,category\n"
        b"1.0,10.0,Alpha\n"
        b"2.0,20.0,Beta\n"
        b"3.0,30.0,Alpha\n"
        b"4.0,40.0,Gamma\n"
        b"5.0,50.0,Beta\n"
    )
    response = client.post(
        "/eda",
        files={"file": ("dataset.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["filename"] == "dataset.csv"
    assert "dataset_id" in res

    data = res["data"]
    assert "dataset_summary" in data
    assert "numerical_analysis" in data
    assert "categorical_analysis" in data
    assert "correlation_analysis" in data
    assert "missing_value_analysis" in data

    # Verify numerical analysis histograms
    assert "x" in data["numerical_analysis"]
    assert "histogram" in data["numerical_analysis"]["x"]
    assert data["numerical_analysis"]["x"]["mean"] == 3.0

    # Verify categorical bar chart
    assert "category" in data["categorical_analysis"]
    assert "bar_chart" in data["categorical_analysis"]["category"]

    # Verify correlation matrix
    assert data["correlation_analysis"]["status"] == "computed"
    assert len(data["correlation_analysis"]["pairs"]) >= 1


# 2. Successful EDA check via dataset_id (query param)
def test_eda_via_dataset_id_query():
    """Verify EDA analysis using an existing dataset_id query parameter."""
    csv_content = b"feature_1,feature_2\n10,100\n20,200\n30,300\n"

    upload_res = client.post(
        "/upload",
        files={"file": ("sample.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["data"]["dataset_id"]

    eda_res = client.post(f"/eda?dataset_id={dataset_id}")
    assert eda_res.status_code == 200
    res = eda_res.json()
    assert res["status"] == "success"
    assert res["dataset_id"] == dataset_id
    assert res["data"]["dataset_summary"]["total_rows"] == 3
    assert res["data"]["dataset_summary"]["total_columns"] == 2


# 3. Successful EDA check via JSON body
def test_eda_via_dataset_id_json():
    """Verify EDA analysis using dataset_id provided in JSON body."""
    csv_content = b"a,b\n1,2\n3,4\n"

    upload_res = client.post(
        "/upload",
        files={"file": ("pairs.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    eda_res = client.post("/eda", json={"dataset_id": dataset_id})
    assert eda_res.status_code == 200
    assert eda_res.json()["data"]["dataset_summary"]["total_rows"] == 2


# 4. Empty CSV
def test_eda_empty_csv():
    """Verify 400 Bad Request when an empty CSV is uploaded."""
    response = client.post(
        "/eda",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# 5. Invalid file format
def test_eda_invalid_file_format():
    """Verify 400 Bad Request when non-CSV file is uploaded."""
    response = client.post(
        "/eda",
        files={"file": ("report.pdf", b"%PDF-1.4...", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


# 6. Corrupted CSV
def test_eda_corrupted_csv():
    """Verify 400 Bad Request when unparseable non-UTF-8 bytes are sent."""
    corrupted_bytes = b"\xff\xfe\x00\x00bad\x80\x81\x82bytes"
    response = client.post(
        "/eda",
        files={"file": ("corrupt.csv", corrupted_bytes, "text/csv")},
    )
    assert response.status_code == 400


# 7. Missing file and dataset_id
def test_eda_missing_inputs():
    """Verify 400 Bad Request when neither file nor dataset_id is provided."""
    response = client.post("/eda")
    assert response.status_code == 400
    assert "No file or dataset_id provided" in response.json()["detail"]


# 8. Nonexistent dataset_id
def test_eda_nonexistent_dataset_id():
    """Verify 404 Not Found for a valid UUID that does not exist in storage."""
    fake_uuid = "44444444-5555-6666-7777-888888888888"
    response = client.post(f"/eda?dataset_id={fake_uuid}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# 9. Invalid dataset_id format
def test_eda_invalid_dataset_id_format():
    """Verify 400 Bad Request when dataset_id format is invalid."""
    response = client.post("/eda?dataset_id=../../traversal")
    assert response.status_code == 400
    assert "invalid dataset_id format" in response.json()["detail"].lower()


# 10. Oversized file rejected
def test_eda_oversized_file_rejected(monkeypatch):
    """Verify 413 Content Too Large when uploaded file exceeds size limit."""
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)
    large_content = b"a,b\n" + (b"12345,67890\n" * 150)
    response = client.post(
        "/eda",
        files={"file": ("large.csv", large_content, "text/csv")},
    )
    assert response.status_code == 413


# 11. JSON serialization
def test_eda_json_serialization():
    """Verify EDA response is strictly JSON-serializable."""
    csv_content = b"val1,val2\n1.5,NaN\nNaN,3.2\n"
    response = client.post(
        "/eda",
        files={"file": ("nan_data.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    dumped = json.dumps(response.json())
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"


# 12. Original dataset remains unchanged after EDA
def test_eda_original_dataset_remains_unchanged():
    """Verify original dataset stored bytes are strictly preserved after EDA analysis."""
    original_csv = b"col_1,col_2\n10,20\n30,40\n"

    upload_res = client.post(
        "/upload",
        files={"file": ("data_preserve.csv", original_csv, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    eda_res = client.post(f"/eda?dataset_id={dataset_id}")
    assert eda_res.status_code == 200

    stored_bytes, stored_filename = get_dataset(dataset_id)
    assert stored_bytes == original_csv
    assert stored_filename == "data_preserve.csv"
