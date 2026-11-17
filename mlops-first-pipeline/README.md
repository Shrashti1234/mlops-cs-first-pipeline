# Iris ML pipeline

Train a Random Forest on Iris data, track metrics and register the model in MLflow,
and serve predictions through FastAPI. Use Python 3.14.

## Setup (PowerShell)

Run these commands from the `mlops-first-pipeline` project folder.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive API. GET `/health` checks
readiness. POST `/predict` accepts positive `sepal_length`, `sepal_width`,
`petal_length`, and `petal_width` measurements and returns `prediction` and
`class_name`.

Training defaults to the tracking server at http://127.0.0.1:6001. Start it in another terminal with `python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 6001`.
To use a different tracking server, set `$env:MLFLOW_TRACKING_URI` before
training (for example, `http://127.0.0.1:6001`). Training now registers
`iris-classifier` and promotes its `champion` alias, matching upstream.
Existing `iris-claassifier` versions remain untouched. Run training once
before starting the updated API.
If you already have a local Iris MLflow store, start the tracking server from
the directory containing that database and its artifacts to keep using your runs. To retain the old model name, set
`$env:MODEL_NAME="iris-claassifier"` in both training and API terminals.
Training publishes `models/iris_model.joblib` after successful registration and
requires test accuracy of at least 0.9. Retrain after changing scikit-learn
versions; saved models must use the same version as the serving environment.

## Docker

With Docker Desktop running in Linux container mode:

```powershell
docker build -t iris-ml-api .
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:6001 iris-ml-api
```

The API loads the registered champion from MLflow at startup. For Docker Desktop,
start MLflow with this command before running training and the container:

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 6001 --workers 1 --allowed-hosts "localhost,127.0.0.1,host.docker.internal,host.docker.internal:6001"
```

GitHub Actions runs API tests and builds the image for pushes and pull requests
to main. Tests inject model/version loaders and require no tracking server.

## Upstream update

Adapted from https://github.com/sqb-codes/KU-CS-MLOPS at commit
`432f2aeb1fafcb63503ab9016690e3287e130db8` on 2026-11-04.
Includes application factory, startup model loading, injectable loaders,
model name/version in `/health`, quality gate and champion promotion.
Deterministic training settings and pinned dependencies are preserved.
Restart the API after changing the champion alias to load the new version.
`MODEL_URI` can override the registry URI; custom URIs report version `custom`.
Course notes are saved in `upstream-notes.txt`; personal `notes.txt` is preserved.
