import pandas as pd
from fastapi import FastAPI

from app.schemas import BatchPredictionRequest, PredictionRequest
from src.inference.predictor import load_model, predict
from src.preprocessing.transformer import load_preprocessor
from src.utils.config import load_config


config = load_config()

app = FastAPI(
    title="Delivery Late Prediction API",
    description="API for predicting whether an order will be delivered late.",
    version="1.0.0",
)

preprocessor = load_preprocessor(
    config["paths"]["preprocessor"]
)

model = load_model(
    config["model"]["name"],
    config["model"]["version"],
)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model")
def model_info():
    return {
        "model": config["model"]["name"],
        "version": config["model"]["version"],
    }


@app.post("/predict")
def predict_order(request: PredictionRequest):
    order = request.model_dump()

    order["order_purchase_timestamp"] = pd.Timestamp(
        order["order_purchase_timestamp"]
    )
    order["order_estimated_delivery_date"] = pd.Timestamp(
        order["order_estimated_delivery_date"]
    )

    order_df = pd.DataFrame([order])

    result = predict(
        model=model,
        preprocessor=preprocessor,
        order=order_df,
        model_name=config["model"]["name"],
        model_version=config["model"]["version"],
    )

    return {
        "prediction": result["prediction"],
        "probability": result["probability"],
        "model": config["model"]["name"],
        "version": config["model"]["version"],
    }

@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    results = []

    for order_request in request.orders:
        order = order_request.model_dump()

        order["order_purchase_timestamp"] = pd.Timestamp(
            order["order_purchase_timestamp"]
        )
        order["order_estimated_delivery_date"] = pd.Timestamp(
            order["order_estimated_delivery_date"]
        )

        order_df = pd.DataFrame([order])

        result = predict(
            model=model,
            preprocessor=preprocessor,
            order=order_df,
            model_name=config["model"]["name"],
            model_version=config["model"]["version"],
        )

        results.append({
            "prediction": result["prediction"],
            "probability": result["probability"],
            "model": config["model"]["name"],
            "version": config["model"]["version"],
        })

    return {"predictions": results}