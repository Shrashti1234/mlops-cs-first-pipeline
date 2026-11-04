import pytest
from fastapi.testclient import TestClient

from main import create_app, MODEL_NAME
from pathlib import Path
import joblib


@pytest.fixture
def client():
    model_path = Path(__file__).resolve().parent / "models" / "iris_model.joblib"
    app = create_app(
        model_loader=lambda: joblib.load(model_path),
        version_loader=lambda: "test-version",
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy", "model_name": MODEL_NAME, "model_version": "test-version"
    }


def test_predict_setosa(client):
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": 0, "class_name": "setosa"}


@pytest.mark.parametrize("invalid_length", [0, -1])
def test_predict_rejects_nonpositive_measurements(client, invalid_length):
    response = client.post("/predict", json={
        "sepal_length": invalid_length,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    })
    assert response.status_code == 422


def test_import_from_another_directory(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    project = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; sys.path.insert(0, sys.argv[1]); import main; '
         'assert callable(main.create_app)',
         str(project)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_injected_loaders_do_not_contact_registry(monkeypatch):
    import main

    def unexpected_registry_call(*args, **kwargs):
        raise AssertionError("Test application contacted MLflow")

    monkeypatch.setattr(main.mlflow, "MlflowClient", unexpected_registry_call)

    class FakeModel:
        def predict(self, features):
            assert features == [[5.1, 3.5, 1.4, 0.2]]
            return [0]

    app = create_app(model_loader=FakeModel, version_loader=lambda: "7")
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == "7"
        response = client.post("/predict", json={
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        assert response.json() == {"prediction": 0, "class_name": "setosa"}


def test_startup_failure_is_clear():
    def broken_loader():
        raise ConnectionError("Tracking server unavailable")

    with pytest.raises(RuntimeError, match="Start the tracking server"):
        with TestClient(create_app(model_loader=broken_loader)):
            pass
