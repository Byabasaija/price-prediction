from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from huggingface_hub import hf_hub_download

from features import compute_features
from models import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = hf_hub_download(
        repo_id="byabasaija/uganda-food-prices-model",
        filename="maize_price_model_v2.pkl",
    )
    app.state.model = joblib.load(model_path)
    app.state.cpi = pd.read_csv(
        "food_cpi_monthly.csv",
        dtype={"year": int, "month": int, "food_cpi": float},
    ).set_index(["year", "month"])["food_cpi"]
    yield


app = FastAPI(title="Maize Price Prediction API", lifespan=lifespan)

FEATURE_COLS = [
    "price_lag_1", "price_lag_2", "price_lag_3",
    "rolling_mean_3", "rolling_mean_6",
    "month", "year", "food_cpi",
]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    features = compute_features(request.prices, request.month, request.year)
    key = (request.year, request.month)
    features["food_cpi"] = (
        app.state.cpi.get(key)
        or app.state.cpi.iloc[-1]  # fall back to latest known CPI
    )
    X = pd.DataFrame([[features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    prediction = app.state.model.predict(X)[0]
    return PredictResponse(predicted_price=round(float(prediction), 2))
