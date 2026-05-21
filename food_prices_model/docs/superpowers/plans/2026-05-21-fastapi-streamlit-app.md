# Maize Price Prediction API — FastAPI + Streamlit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI service that serves the trained Random Forest model via a `/predict` endpoint, with a Streamlit demo UI on top.

**Architecture:** FastAPI loads `maize_price_model.pkl` on startup and exposes a `/predict` endpoint. The endpoint accepts the last 6 months of prices plus the target month/year, computes the 7 features internally, and returns the predicted price. Streamlit is a separate app in the same repo that calls the FastAPI endpoint and renders a simple UI for demos.

**Tech Stack:** fastapi, uvicorn, pydantic, joblib, scikit-learn, numpy, streamlit, pytest, httpx

---

## File Map

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — loads model, defines /predict endpoint |
| `features.py` | Feature computation logic (lag, rolling mean, calendar) |
| `models.py` | Pydantic request/response schemas |
| `streamlit_app.py` | Streamlit demo UI — calls FastAPI and displays prediction |
| `tests/test_features.py` | Unit tests for feature computation |
| `tests/test_api.py` | Integration tests for FastAPI endpoints |
| `maize_price_model.pkl` | Trained model — copy from training project |
| `pyproject.toml` | Project dependencies |
| `Dockerfile` | Container definition for FastAPI app |

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `maize_price_model.pkl` (copy from training project)

- [ ] **Step 1: Create the project directory and initialise with uv**

```bash
mkdir maize-price-api
cd maize-price-api
uv init
uv add fastapi uvicorn pydantic joblib scikit-learn numpy streamlit
uv add --dev pytest httpx
```

Expected: `.venv` created, `pyproject.toml` populated.

- [ ] **Step 2: Copy the trained model**

```bash
cp ../machine-learning/maize_price_model.pkl .
```

Expected: `maize_price_model.pkl` present in project root.

- [ ] **Step 3: Create a .gitignore**

```
.venv/
__pycache__/
*.pyc
.env
```

- [ ] **Step 4: Initialise git and commit**

```bash
git init
git add pyproject.toml uv.lock .gitignore maize_price_model.pkl
git commit -m "chore: project setup with dependencies and trained model"
```

---

## Task 2: Feature Computation

**Files:**
- Create: `features.py`
- Create: `tests/test_features.py`

**ML context:** The model was trained on 7 features computed from historical prices. In production we compute these on the fly from the last 6 months of prices provided in the request.

- [ ] **Step 1: Create tests directory and write failing tests**

```bash
mkdir tests
touch tests/__init__.py
```

Create `tests/test_features.py`:

```python
import numpy as np
from features import compute_features

def test_lag_features():
    prices = [700.0, 720.0, 750.0, 780.0, 800.0, 820.0]
    month, year = 4, 2024
    features = compute_features(prices, month, year)
    assert features["price_lag_1"] == 820.0
    assert features["price_lag_2"] == 800.0
    assert features["price_lag_3"] == 780.0

def test_rolling_mean_3():
    prices = [700.0, 720.0, 750.0, 780.0, 800.0, 820.0]
    month, year = 4, 2024
    features = compute_features(prices, month, year)
    assert features["rolling_mean_3"] == round((780.0 + 800.0 + 820.0) / 3, 4)

def test_rolling_mean_6():
    prices = [700.0, 720.0, 750.0, 780.0, 800.0, 820.0]
    month, year = 4, 2024
    features = compute_features(prices, month, year)
    assert features["rolling_mean_6"] == round(sum(prices) / 6, 4)

def test_calendar_features():
    prices = [700.0, 720.0, 750.0, 780.0, 800.0, 820.0]
    features = compute_features(prices, month=7, year=2025)
    assert features["month"] == 7
    assert features["year"] == 2025

def test_returns_correct_keys():
    prices = [700.0, 720.0, 750.0, 780.0, 800.0, 820.0]
    features = compute_features(prices, month=1, year=2024)
    expected_keys = {
        "price_lag_1", "price_lag_2", "price_lag_3",
        "rolling_mean_3", "rolling_mean_6", "month", "year"
    }
    assert set(features.keys()) == expected_keys

def test_requires_at_least_6_prices():
    with pytest.raises(ValueError, match="at least 6"):
        compute_features([700.0, 720.0], month=1, year=2024)

import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_features.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `features` does not exist yet.

- [ ] **Step 3: Implement features.py**

```python
import numpy as np


def compute_features(prices: list[float], month: int, year: int) -> dict:
    if len(prices) < 6:
        raise ValueError("Need at least 6 months of prices to compute features")

    last6 = prices[-6:]

    return {
        "price_lag_1": last6[-1],
        "price_lag_2": last6[-2],
        "price_lag_3": last6[-3],
        "rolling_mean_3": round(float(np.mean(last6[-3:])), 4),
        "rolling_mean_6": round(float(np.mean(last6)), 4),
        "month": month,
        "year": year,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_features.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add features.py tests/
git commit -m "feat: feature computation with full test coverage"
```

---

## Task 3: Pydantic Schemas

**Files:**
- Create: `models.py`

- [ ] **Step 1: Create models.py**

```python
from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    prices: list[float]
    month: int
    year: int

    @field_validator("prices")
    @classmethod
    def prices_must_have_six_values(cls, v):
        if len(v) < 6:
            raise ValueError("prices must contain at least 6 values (last 6 months)")
        return v

    @field_validator("month")
    @classmethod
    def month_must_be_valid(cls, v):
        if not 1 <= v <= 12:
            raise ValueError("month must be between 1 and 12")
        return v


class PredictResponse(BaseModel):
    predicted_price: float
    currency: str = "UGX"
    unit: str = "per kg"
    commodity: str = "Maize"
    market: str = "Owino, Kampala"
```

- [ ] **Step 2: Commit**

```bash
git add models.py
git commit -m "feat: pydantic request and response schemas"
```

---

## Task 4: FastAPI App

**Files:**
- Create: `main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200():
    payload = {
        "prices": [700.0, 720.0, 750.0, 780.0, 800.0, 820.0],
        "month": 4,
        "year": 2024,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_returns_expected_fields():
    payload = {
        "prices": [700.0, 720.0, 750.0, 780.0, 800.0, 820.0],
        "month": 4,
        "year": 2024,
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    assert "predicted_price" in data
    assert "currency" in data
    assert data["currency"] == "UGX"
    assert isinstance(data["predicted_price"], float)


def test_predict_rejects_too_few_prices():
    payload = {
        "prices": [700.0, 720.0],
        "month": 4,
        "year": 2024,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_month():
    payload = {
        "prices": [700.0, 720.0, 750.0, 780.0, 800.0, 820.0],
        "month": 13,
        "year": 2024,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_api.py -v
```

Expected: `ImportError` — `main` does not exist yet.

- [ ] **Step 3: Implement main.py**

```python
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI

from features import compute_features
from models import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("maize_price_model.pkl")
    yield


app = FastAPI(title="Maize Price Prediction API", lifespan=lifespan)

FEATURE_COLS = [
    "price_lag_1", "price_lag_2", "price_lag_3",
    "rolling_mean_3", "rolling_mean_6",
    "month", "year",
]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    features = compute_features(request.prices, request.month, request.year)
    X = np.array([[features[col] for col in FEATURE_COLS]])
    prediction = app.state.model.predict(X)[0]
    return PredictResponse(predicted_price=round(float(prediction), 2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run the server manually and test it**

```bash
uv run uvicorn main:app --reload
```

Open browser at `http://localhost:8000/docs` — FastAPI's auto-generated Swagger UI. Test the `/predict` endpoint with:

```json
{
  "prices": [700, 720, 750, 780, 800, 820],
  "month": 6,
  "year": 2024
}
```

Expected: a JSON response with `predicted_price`.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: fastapi predict endpoint with health check"
```

---

## Task 5: Streamlit Demo

**Files:**
- Create: `streamlit_app.py`

- [ ] **Step 1: Create streamlit_app.py**

```python
import streamlit as st
import requests

st.set_page_config(page_title="Maize Price Predictor", page_icon="🌽")

st.title("🌽 Maize Price Predictor")
st.caption("Owino Market, Kampala — Random Forest Model")
st.markdown("Enter the last 6 months of Maize prices (UGX/kg), oldest first.")

col1, col2 = st.columns(2)

prices = []
labels = ["6 months ago", "5 months ago", "4 months ago", "3 months ago", "2 months ago", "Last month"]
for i, label in enumerate(labels):
    col = col1 if i < 3 else col2
    price = col.number_input(label, min_value=100, max_value=5000, value=800, step=10)
    prices.append(float(price))

st.divider()

col3, col4 = st.columns(2)
month = col3.selectbox("Target month", list(range(1, 13)), index=5,
                       format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                              "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
year = col4.number_input("Target year", min_value=2020, max_value=2030, value=2024)

if st.button("Predict price", type="primary"):
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"prices": prices, "month": month, "year": year},
        )
        if response.status_code == 200:
            data = response.json()
            st.success(f"Predicted price: **{data['predicted_price']:,.0f} {data['currency']}/kg**")
            st.caption(f"{data['commodity']} · {data['market']}")
        else:
            st.error(f"API error: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
```

- [ ] **Step 2: Run Streamlit and test manually**

In a second terminal (keep FastAPI running in the first):

```bash
uv run streamlit run streamlit_app.py
```

Expected: browser opens at `http://localhost:8501`. Enter 6 prices, click Predict, see the predicted price.

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: streamlit demo UI connecting to fastapi"
```

---

## Task 6: Docker

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Build and run the container**

```bash
docker build -t maize-price-api .
docker run -p 8000:8000 maize-price-api
```

Expected: API running at `http://localhost:8000/health` returns `{"status": "ok"}`.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "chore: add dockerfile for fastapi service"
```

---

## Done

After all tasks complete you should have:
- A running FastAPI service at `http://localhost:8000` with `/health` and `/predict` endpoints
- Auto-generated API docs at `http://localhost:8000/docs`
- A Streamlit demo at `http://localhost:8501`
- A Docker image ready to deploy to any cloud provider
- Full test coverage on feature computation and API endpoints
