"""
API tests for the Data Cleaning and Download endpoints (backend/app/routes/clean.py).
"""

import io
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.data_engine.storage import get_dataset

client = TestClient(app)


# 1. Successful direct CSV upload with cleaning operations
def test_clean_successful_direct_upload():
    """Verify cleaning operations applied to directly uploaded CSV."""
    csv_content = (
        b"name,age,city\n"
        b"Alice ,30,new york\n"
        b"Bob,NaN,San Francisco\n"
        b"Alice ,30,new york\n"  # Duplicate row
    )
    operations = {
        "remove_duplicates": True,
        "trim_whitespace": ["name"],
        "casing": {"city": "upper"},
        "missing_values": {"age": "mean"},
    }
    response = client.post(
        "/clean",
        files={"file": ("raw_data.csv", csv_content, "text/csv")},
        data={"operations": json.dumps(operations)},
    )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "cleaned_dataset_id" in res
    assert "original_dataset_id" in res
    assert "download_url" in res
    assert res["download_url"] == f"/clean/download/{res['cleaned_dataset_id']}"

    report = res["data"]
    assert report["initial_shape"]["rows"] == 3
    assert report["cleaned_shape"]["rows"] == 2  # Duplicate removed
    assert report["duplicates_removed"]["removed_count"] == 1
    assert report["missing_values_imputed"]["total_imputed"] == 1


# 2. Successful cleaning via dataset_id with JSON body
def test_clean_via_dataset_id_json_body():
    """Verify cleaning a stored dataset using dataset_id and operations in JSON body."""
    csv_content = b"x,y\n1,10\n1,10\n2,20\n"

    # Step 1: Upload
    upload_res = client.post(
        "/upload",
        files={"file": ("points.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["data"]["dataset_id"]

    # Step 2: Clean via JSON body
    clean_res = client.post(
        "/clean",
        json={
            "dataset_id": dataset_id,
            "operations": {
                "remove_duplicates": True,
            },
        },
    )
    assert clean_res.status_code == 200
    data = clean_res.json()
    assert data["status"] == "success"
    assert data["original_dataset_id"] == dataset_id
    assert data["data"]["cleaned_shape"]["rows"] == 2


# 3. Successful cleaning via query param dataset_id
def test_clean_via_dataset_id_query_param():
    """Verify cleaning a stored dataset using query parameter for dataset_id."""
    csv_content = b"col_a,col_b\napple,1\nbanana,2\n"

    upload_res = client.post(
        "/upload",
        files={"file": ("fruits.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    clean_res = client.post(
        f"/clean?dataset_id={dataset_id}",
        json={
            "operations": {
                "casing": {"col_a": "upper"},
            }
        },
    )
    assert clean_res.status_code == 200
    assert clean_res.json()["status"] == "success"


# 4. Cleaned CSV download endpoint
def test_clean_download_endpoint():
    """Verify downloading the cleaned CSV via /clean/download/{dataset_id}."""
    csv_content = b"name,val\nAlice ,10\nBob ,20\n"
    response = client.post(
        "/clean",
        files={"file": ("users.csv", csv_content, "text/csv")},
        data={"operations": json.dumps({"trim_whitespace": ["name"]})},
    )
    assert response.status_code == 200
    cleaned_id = response.json()["cleaned_dataset_id"]

    # Download using path parameter
    download_res = client.get(f"/clean/download/{cleaned_id}")
    assert download_res.status_code == 200
    assert "text/csv" in download_res.headers["content-type"]
    assert "attachment" in download_res.headers["content-disposition"]

    # Verify content parsed from downloaded CSV has whitespace trimmed
    df_downloaded = pd.read_csv(io.BytesIO(download_res.content))
    assert list(df_downloaded["name"]) == ["Alice", "Bob"]

    # Download using query parameter fallback
    download_query_res = client.get(f"/clean/download?dataset_id={cleaned_id}")
    assert download_query_res.status_code == 200
    assert download_query_res.content == download_res.content


# 5. Original dataset remains unchanged after cleaning
def test_original_dataset_remains_unchanged():
    """Verify original dataset stored bytes are strictly preserved after cleaning."""
    original_csv = b"item,price\nWidget ,100\nWidget ,100\n"

    upload_res = client.post(
        "/upload",
        files={"file": ("inventory.csv", original_csv, "text/csv")},
    )
    orig_id = upload_res.json()["data"]["dataset_id"]

    # Apply deduplication and whitespace trimming
    clean_res = client.post(
        f"/clean?dataset_id={orig_id}",
        json={"operations": {"remove_duplicates": True, "trim_whitespace": ["item"]}},
    )
    assert clean_res.status_code == 200

    # Verify original dataset in storage is 100% identical to the uploaded bytes
    orig_bytes, orig_filename = get_dataset(orig_id)
    assert orig_bytes == original_csv
    assert orig_filename == "inventory.csv"


# 6. Reject unsupported cleaning operations
def test_clean_reject_unsupported_operations():
    """Verify 400 Bad Request when unsupported operation keys are supplied."""
    csv_content = b"a,b\n1,2\n"
    response = client.post(
        "/clean",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"operations": json.dumps({"unsupported_fake_op": True})},
    )
    assert response.status_code == 400
    assert "Unsupported cleaning operation" in response.json()["detail"]


# 7. Reject invalid cleaning directive values
def test_clean_reject_invalid_directive():
    """Verify 400 Bad Request when cleaning strategy is invalid."""
    csv_content = b"text_col,val\nhello,1\n"
    # Cannot calculate mean of a text column
    response = client.post(
        "/clean",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"operations": json.dumps({"missing_values": {"text_col": "mean"}})},
    )
    assert response.status_code == 400


# 8. Malformed operations JSON string
def test_clean_malformed_operations_json():
    """Verify 400 Bad Request when operations is an unparseable JSON string."""
    csv_content = b"a,b\n1,2\n"
    response = client.post(
        "/clean",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"operations": "{malformed_json: bad}"},
    )
    assert response.status_code == 400
    assert "Invalid JSON string" in response.json()["detail"]


# 9. Download nonexistent dataset
def test_clean_download_nonexistent_dataset():
    """Verify 404 Not Found when downloading nonexistent dataset ID."""
    fake_uuid = "33333333-4444-5555-6666-777777777777"
    response = client.get(f"/clean/download/{fake_uuid}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# 10. Missing inputs
def test_clean_missing_inputs():
    """Verify 400 Bad Request when neither file nor dataset_id is provided."""
    response = client.post("/clean")
    assert response.status_code == 400
    assert "No file or dataset_id provided" in response.json()["detail"]


# 11. Oversized file rejected
def test_clean_oversized_file_rejected(monkeypatch):
    """Verify 413 Content Too Large when uploaded file exceeds size limit."""
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)
    large_content = b"a,b\n" + (b"12345,67890\n" * 150)
    response = client.post(
        "/clean",
        files={"file": ("large.csv", large_content, "text/csv")},
    )
    assert response.status_code == 413


# 12. JSON serialization
def test_clean_json_serialization():
    """Verify clean response is strictly JSON-serializable."""
    csv_content = b"a,b\n1,2\n3,4\n"
    response = client.post(
        "/clean",
        files={"file": ("simple.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    dumped = json.dumps(response.json())
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"
