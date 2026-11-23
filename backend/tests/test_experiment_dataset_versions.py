import json

from fastapi.testclient import TestClient

from app.data_engine.storage import get_dataset
from app.main import app


client = TestClient(app)


def test_experiment_endpoints_use_the_selected_cleaned_dataset_id():
    rows = ["x,y,label"]
    for index in range(30):
        label = "A " if index % 2 == 0 else "B"
        rows.append(f"{index},{index * 2 + 1},{label}")
    original_csv = ("\n".join(rows) + "\n").encode("utf-8")

    upload_response = client.post(
        "/upload",
        files={"file": ("source.csv", original_csv, "text/csv")},
    )
    assert upload_response.status_code == 200
    original_id = upload_response.json()["data"]["dataset_id"]

    clean_response = client.post(
        "/clean",
        json={
            "dataset_id": original_id,
            "operations": {"trim_whitespace": True},
        },
    )
    assert clean_response.status_code == 200
    cleaned_id = clean_response.json()["cleaned_dataset_id"]
    assert cleaned_id != original_id
    assert get_dataset(original_id)[0] == original_csv

    requests = [
        ("/classify", {"target_column": "label", "feature_columns": ["x"]}),
        ("/regression", {"target_column": "y", "feature_columns": ["x"]}),
        ("/cluster", {"feature_columns": ["x", "y"], "n_clusters": 2}),
        ("/anomaly", {"feature_columns": ["x", "y"], "contamination": 0.1}),
    ]

    for endpoint, parameters in requests:
        response = client.post(
            endpoint,
            json={"dataset_id": cleaned_id, **parameters},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["dataset_id"] == cleaned_id
        assert result["data"]["status"] == "completed"
        if endpoint == "/classify":
            assert set(result["data"]["class_distribution"]) == {"A", "B"}
