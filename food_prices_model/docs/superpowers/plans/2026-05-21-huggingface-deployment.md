# Hugging Face Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload the trained model to Hugging Face Model Hub and deploy the Streamlit demo to Hugging Face Spaces so both are publicly accessible.

**Architecture:** The model gets its own HF repository with a model card documenting performance and usage. The Streamlit app gets deployed as a HF Space — a free hosted Streamlit environment. The Space calls the model directly (no FastAPI needed in this deployment — we load the model inside Streamlit for simplicity on HF).

**Tech Stack:** huggingface_hub, Hugging Face Spaces (Streamlit SDK), Git LFS

**Prerequisite:** Complete `2026-05-21-fastapi-streamlit-app.md` first. You need the trained model and Streamlit app from that project.

---

## File Map

| File | Purpose |
|------|---------|
| `maize_price_model.pkl` | Trained model — uploaded to HF Model Hub |
| `README.md` (model repo) | Model card — documents performance, usage, dataset |
| `app.py` | Streamlit app adapted for HF Spaces (loads model directly) |
| `requirements.txt` | HF Spaces dependency file (Spaces uses pip, not uv) |

---

## Task 1: Create a Hugging Face Account and Install CLI

- [ ] **Step 1: Create account**

Go to [huggingface.co](https://huggingface.co) and sign up if you don't have an account.

- [ ] **Step 2: Install the HF CLI**

In your `maize-price-api` project:

```bash
uv add huggingface_hub
```

- [ ] **Step 3: Log in**

```bash
uv run huggingface-cli login
```

Expected: prompts for an access token. Generate one at huggingface.co → Settings → Access Tokens → New token (write access).

---

## Task 2: Upload Model to HF Model Hub

- [ ] **Step 1: Create the model repository on HF**

```bash
uv run python -c "
from huggingface_hub import create_repo
create_repo('maize-price-predictor-uganda', repo_type='model', private=False)
print('Repo created')
"
```

Expected: prints `Repo created`. Repo URL will be `https://huggingface.co/<your-username>/maize-price-predictor-uganda`.

- [ ] **Step 2: Upload the model file**

```bash
uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='maize_price_model.pkl',
    path_in_repo='maize_price_model.pkl',
    repo_id='<your-username>/maize-price-predictor-uganda',
    repo_type='model',
)
print('Model uploaded')
"
```

Replace `<your-username>` with your HF username. Expected: prints `Model uploaded`.

- [ ] **Step 3: Create the model card (README.md)**

Create a file `model_card.md` locally:

```markdown
---
language:
- en
license: mit
tags:
- sklearn
- tabular-regression
- food-prices
- uganda
- east-africa
datasets:
- wfp-food-prices-uganda
metrics:
- mae
---

# Maize Price Predictor — Owino Market, Kampala

Predicts next month's Maize wholesale price (UGX/kg) at Owino market, Kampala, Uganda.

Trained on 16 years of WFP food price data (2006–2019). Tested on 2020–2022 (COVID era).

## Model Details

- **Algorithm:** Random Forest Regressor (100 trees)
- **Features:** price_lag_1, price_lag_2, price_lag_3, rolling_mean_3, rolling_mean_6, month, year
- **Train set:** 2006–2019 (175 months)
- **Test set:** 2020–2022 (20 months, COVID era)

## Performance

| Metric | Value |
|--------|-------|
| Train MAE | 57 UGX/kg |
| Test MAE | 193 UGX/kg |
| Avg test price | 932 UGX/kg |
| Error rate | 20.7% |

Note: the test period covers COVID-19 disruptions — an unusually hard prediction window.
Pre-COVID performance is expected to be significantly better.

## Usage

```python
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="<your-username>/maize-price-predictor-uganda",
    filename="maize_price_model.pkl"
)
model = joblib.load(model_path)

# Last 6 months of prices (oldest first), target month and year
prices = [700, 720, 750, 780, 800, 820]
features = np.array([[
    prices[-1],                        # price_lag_1
    prices[-2],                        # price_lag_2
    prices[-3],                        # price_lag_3
    sum(prices[-3:]) / 3,             # rolling_mean_3
    sum(prices) / 6,                   # rolling_mean_6
    6,                                 # month
    2024,                              # year
]])

predicted_price = model.predict(features)[0]
print(f"Predicted price: {predicted_price:,.0f} UGX/kg")
```

## Data

Source: [WFP Food Prices for Uganda](https://data.humdata.org/dataset/wfp-food-prices-for-uganda)

## Limitations

- Trained on one commodity (Maize) and one market (Owino, Kampala)
- Cannot predict prices during unprecedented disruptions (pandemics, conflicts)
- Retraining recommended annually as inflation shifts the price distribution
```

- [ ] **Step 4: Upload the model card**

```bash
uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='model_card.md',
    path_in_repo='README.md',
    repo_id='<your-username>/maize-price-predictor-uganda',
    repo_type='model',
)
print('Model card uploaded')
"
```

Expected: prints `Model card uploaded`. Visit `https://huggingface.co/<your-username>/maize-price-predictor-uganda` — model card should display.

- [ ] **Step 5: Commit**

```bash
git add model_card.md
git commit -m "docs: add hugging face model card"
```

---

## Task 3: Deploy Streamlit App to HF Spaces

HF Spaces runs Streamlit apps directly. No FastAPI needed here — we load the model directly inside Streamlit (simpler for a hosted environment).

- [ ] **Step 1: Create the Space on HF**

```bash
uv run python -c "
from huggingface_hub import create_repo
create_repo('maize-price-demo', repo_type='space', space_sdk='streamlit', private=False)
print('Space created')
"
```

Expected: Space created at `https://huggingface.co/spaces/<your-username>/maize-price-demo`.

- [ ] **Step 2: Create app.py for HF Spaces**

Create `hf_space/app.py`:

```python
import joblib
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Maize Price Predictor", page_icon="🌽")


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="<your-username>/maize-price-predictor-uganda",
        filename="maize_price_model.pkl",
    )
    return joblib.load(model_path)


model = load_model()

FEATURE_COLS = [
    "price_lag_1", "price_lag_2", "price_lag_3",
    "rolling_mean_3", "rolling_mean_6",
    "month", "year",
]

st.title("🌽 Maize Price Predictor")
st.caption("Owino Market, Kampala — Random Forest Model trained on 16 years of WFP data")
st.markdown("Enter the last 6 months of Maize wholesale prices (UGX/kg), oldest first.")

col1, col2 = st.columns(2)
prices = []
labels = ["6 months ago", "5 months ago", "4 months ago", "3 months ago", "2 months ago", "Last month"]
for i, label in enumerate(labels):
    col = col1 if i < 3 else col2
    price = col.number_input(label, min_value=100, max_value=5000, value=800, step=10)
    prices.append(float(price))

st.divider()

col3, col4 = st.columns(2)
month = col3.selectbox(
    "Predict for month",
    list(range(1, 13)),
    index=5,
    format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
)
year = col4.number_input("Predict for year", min_value=2020, max_value=2030, value=2025)

if st.button("Predict price", type="primary"):
    features = np.array([[
        prices[-1], prices[-2], prices[-3],
        float(np.mean(prices[-3:])),
        float(np.mean(prices)),
        month, year,
    ]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted price: **{prediction:,.0f} UGX/kg**")
    st.caption("Maize · Owino Market, Kampala · Random Forest (MAE ~193 UGX/kg on 2020–2022 data)")
    st.info("Note: accuracy is lower for years with major disruptions (COVID, conflict, drought).")
```

Replace `<your-username>` with your HF username.

- [ ] **Step 3: Create requirements.txt for the Space**

Create `hf_space/requirements.txt`:

```
scikit-learn>=1.8.0
joblib
numpy
huggingface_hub
```

- [ ] **Step 4: Push the Space files to HF**

```bash
uv run python -c "
from huggingface_hub import HfApi
api = HfApi()

api.upload_file(
    path_or_fileobj='hf_space/app.py',
    path_in_repo='app.py',
    repo_id='<your-username>/maize-price-demo',
    repo_type='space',
)

api.upload_file(
    path_or_fileobj='hf_space/requirements.txt',
    path_in_repo='requirements.txt',
    repo_id='<your-username>/maize-price-demo',
    repo_type='space',
)
print('Space files uploaded — building now')
"
```

Expected: prints `Space files uploaded`. HF will automatically build the Space (takes 1–3 minutes).

- [ ] **Step 5: Verify the Space is live**

Visit `https://huggingface.co/spaces/<your-username>/maize-price-demo`.

Expected: Streamlit app loads, enter prices, click Predict, see predicted price.

- [ ] **Step 6: Commit**

```bash
git add hf_space/
git commit -m "feat: hugging face spaces streamlit deployment"
```

---

## Done

After all tasks complete you should have:
- Model publicly available at `https://huggingface.co/<username>/maize-price-predictor-uganda`
- Live Streamlit demo at `https://huggingface.co/spaces/<username>/maize-price-demo`
- A shareable link you can post on Twitter, LinkedIn, and your portfolio

## Updating the model later

When you retrain with new features (rainfall, fuel prices):

```bash
# re-upload model
api.upload_file(path_or_fileobj='maize_price_model.pkl', ...)

# update model card with new MAE numbers
api.upload_file(path_or_fileobj='model_card.md', ...)
```

The Space picks up the new model automatically on next cold start.
