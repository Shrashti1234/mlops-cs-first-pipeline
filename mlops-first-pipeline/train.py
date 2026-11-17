from pathlib import Path
import os
import sys
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# MLflow prints Unicode run links, including in redirected Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
#configure mlflow

MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
project_dir = Path(__file__).resolve().parent
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:6001"))
#group realted traing runs under one experiment
mlflow.set_experiment("iris_classification_experiment")
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

n_estimators = 100
random_state = 42
#model settings we want to track in mlflow
model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.2
}

#start a tracked training run
with mlflow.start_run() as run:
    mlflow.log_params(model_parameters)
    #build classifier using selected parameters
    model=RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    MIN_TEST_ACCURACY = 0.9
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected: test accuracy: {accuracy:.2f} is below {MIN_TEST_ACCURACY}"
        )
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })

    print("MLflow run ID:", run.info.run_id)
    print("Test Accuracy:",accuracy)
    print("Test Precision:",precision)
    print("Test Recall:",recall)


# Save the trained model as reusable artifact
    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_model",
        registered_model_name=MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"
        ]

    )
    client = MlflowClient()
    current_run_id = mlflow.active_run().info.run_id
    registered_versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    current_version = next(
        (version for version in registered_versions if version.run_id == current_run_id),
        None,
    )
    if current_version is None:
        raise RuntimeError("Model was logged but its registered version could not be found")
    client.set_registered_model_alias(
        name=MODEL_NAME, alias=MODEL_ALIAS, version=current_version.version
    )
    print("Registered Model:", MODEL_NAME)
    print("Champion version:", current_version.version)
    joblib.dump(model, model_dir / "iris_model.joblib")
    print(f"Model saved to {model_dir / 'iris_model.joblib'}")
