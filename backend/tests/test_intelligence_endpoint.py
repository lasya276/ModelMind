"""
API tests for the Dataset Intelligence endpoint (backend/app/routes/intelligence.py).
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# 1. Successful direct CSV upload with structural inference
def test_intelligence_successful_direct_upload():
    """Verify dataset intelligence analysis on direct CSV upload."""
    csv_content = (
        b"user_id,age,income,churn,active\n"
        b"101,25,50000,Yes,True\n"
        b"102,30,60000,No,True\n"
        b"103,35,75000,Yes,False\n"
        b"104,40,90000,No,True\n"
        b"105,45,105000,No,False\n"
        b"106,50,120000,Yes,True\n"
        b"107,55,135000,No,True\n"
        b"108,60,150000,No,False\n"
        b"109,65,165000,Yes,True\n"
        b"110,70,180000,No,False\n"
    )
    response = client.post(
        "/intelligence",
        files={"file": ("customers.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["filename"] == "customers.csv"
    assert "dataset_id" in res

    data = res["data"]
    assert data["total_rows"] == 10
    assert data["total_columns"] == 5
    assert "role_counts" in data
    assert "column_roles" in data
    assert "possible_targets" in data
    assert "recommended_analyses" in data

    # Verify ID column detection
    assert data["column_roles"]["user_id"]["role"] == "potential_id"

    # Verify candidate targets exist and are strictly candidate-only (not automatically selected)
    for target in data["possible_targets"]:
        assert target["status"] == "candidate_only"
        assert "column" in target
        assert "suggested_task" in target

    # Verify analysis recommendations contain prerequisites
    analysis_types = [rec["analysis_type"] for rec in data["recommended_analyses"]]
    assert "exploratory_data_analysis" in analysis_types
    assert "classification" in analysis_types
    assert "regression" in analysis_types
    assert "anomaly_detection" in analysis_types
    assert "clustering" in analysis_types


# 2. Successful intelligence check via dataset_id (query param)
def test_intelligence_via_dataset_id_query():
    """Verify dataset intelligence using an existing dataset_id query parameter."""
    csv_content = b"feature1,feature2,target\n1,10,0\n2,20,1\n3,30,0\n4,40,1\n5,50,0\n"

    upload_res = client.post(
        "/upload",
        files={"file": ("features.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    intel_res = client.post(f"/intelligence?dataset_id={dataset_id}")
    assert intel_res.status_code == 200
    res = intel_res.json()
    assert res["status"] == "success"
    assert res["dataset_id"] == dataset_id
    assert res["data"]["total_rows"] == 5
    assert res["data"]["total_columns"] == 3


# 3. Successful intelligence check via JSON body
def test_intelligence_via_dataset_id_json():
    """Verify dataset intelligence using dataset_id provided in JSON body."""
    csv_content = b"x,y\n1,2\n3,4\n5,6\n"

    upload_res = client.post(
        "/upload",
        files={"file": ("coords.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    intel_res = client.post("/intelligence", json={"dataset_id": dataset_id})
    assert intel_res.status_code == 200
    assert intel_res.json()["data"]["total_rows"] == 3


# 4. Empty CSV
def test_intelligence_empty_csv():
    """Verify 400 Bad Request when an empty CSV is uploaded."""
    response = client.post(
        "/intelligence",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# 5. Invalid file format
def test_intelligence_invalid_file_format():
    """Verify 400 Bad Request when non-CSV file is uploaded."""
    response = client.post(
        "/intelligence",
        files={"file": ("doc.docx", b"word doc", "application/vnd.openxmlformats")},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


# 6. Corrupted CSV
def test_intelligence_corrupted_csv():
    """Verify 400 Bad Request when corrupted bytes are uploaded."""
    corrupted_bytes = b"\xff\xfe\x00\x00non_utf8\x80\x81"
    response = client.post(
        "/intelligence",
        files={"file": ("corrupt.csv", corrupted_bytes, "text/csv")},
    )
    assert response.status_code == 400


# 7. Missing file and dataset_id
def test_intelligence_missing_inputs():
    """Verify 400 Bad Request when no file or dataset_id is sent."""
    response = client.post("/intelligence")
    assert response.status_code == 400
    assert "No file or dataset_id provided" in response.json()["detail"]


# 8. Nonexistent dataset_id
def test_intelligence_nonexistent_dataset_id():
    """Verify 404 Not Found for a valid UUID that does not exist in storage."""
    fake_uuid = "22222222-3333-4444-5555-666666666666"
    response = client.post(f"/intelligence?dataset_id={fake_uuid}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# 9. Invalid dataset_id format
def test_intelligence_invalid_dataset_id_format():
    """Verify 400 Bad Request when dataset_id format is invalid."""
    response = client.post("/intelligence?dataset_id=invalid_id_value")
    assert response.status_code == 400
    assert "invalid dataset_id format" in response.json()["detail"].lower()


# 10. Oversized file rejected
def test_intelligence_oversized_file_rejected(monkeypatch):
    """Verify 413 Content Too Large when uploaded file exceeds size limit."""
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)
    large_content = b"a,b\n" + (b"12345,67890\n" * 150)
    response = client.post(
        "/intelligence",
        files={"file": ("large.csv", large_content, "text/csv")},
    )
    assert response.status_code == 413


# 11. JSON serialization
def test_intelligence_json_serialization():
    """Verify intelligence report is strictly JSON-serializable."""
    csv_content = b"a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n13,14,15\n"
    response = client.post(
        "/intelligence",
        files={"file": ("serial.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    dumped = json.dumps(response.json())
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"
