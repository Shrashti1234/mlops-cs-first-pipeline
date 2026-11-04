from contextlib import asynccontextmanager
import os
from typing import Callable

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:6001")
MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


def load_model_version():
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


def create_app(model_loader: Callable = None, version_loader: Callable = None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            if model_loader is None and version_loader is None and MODEL_URI == f"models:/{MODEL_NAME}@{MODEL_ALIAS}":
                # Resolve the alias once so model and reported version stay consistent.
                version = load_model_version()
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                app.state.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
                app.state.model_version = version
            else:
                app.state.model = (model_loader or load_mlflow_model)()
                app.state.model_version = (
                    version_loader() if version_loader else
                    ("custom" if model_loader or MODEL_URI != f"models:/{MODEL_NAME}@{MODEL_ALIAS}" else load_model_version())
                )
        except Exception as ex:
            raise RuntimeError(
                f"Failed to load MLflow model {MODEL_URI}. "
                "Start the tracking server and run python train.py first."
            ) from ex
        yield

    app = FastAPI(
        title="Iris Classification API",
        description="Iris prediction API powered by MLflow",
        version="3.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def root():
        return {"message": "API is ready to serve..."}

    @app.get("/health")
    def health_check(request: Request):
        return {
            "status": "healthy",
            "model_name": MODEL_NAME,
            "model_version": request.app.state.model_version,
        }

    @app.post("/predict")
    def predict(data: IrisRequest, request: Request):
        features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
        prediction = int(request.app.state.model.predict(features)[0])
        return {
            "prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction],
        }

    return app


app = create_app()
