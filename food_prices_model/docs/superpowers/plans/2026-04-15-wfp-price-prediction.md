# WFP Uganda Food Price Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a Random Forest model on historical WFP data to predict next month's Maize price in Kampala's Owino market, while learning core ML terminology hands-on.

**Architecture:** Raw CSV → filter to one market/commodity → engineer lag + rolling + calendar features → time-based train/test split → train RandomForestRegressor → evaluate with MAE → visualize predicted vs actual + feature importance. All work lives in focused Jupyter notebooks, one per stage.

**Tech Stack:** pandas, numpy, scikit-learn (RandomForestRegressor, mean_absolute_error), matplotlib, seaborn

---

## File Map

| File | Purpose |
|------|---------|
| `wfp_food_prices_uga.csv` | Raw data — do not modify |
| `exploration.ipynb` | Already started — extend with market/commodity analysis |
| `02_feature_engineering.ipynb` | Filter data, build features, save engineered CSV |
| `03_model_training.ipynb` | Load features, train/test split, train model, evaluate, visualize |
| `data/maize_owino_features.csv` | Engineered dataset (created in Task 3) |
| `price_prediction_results.png` | Output chart (created in Task 4) |

---

## Task 1: Install Dependencies

**Files:**
- Modify: `pyproject.toml` (uv handles this automatically)

- [ ] **Step 1: Add required packages**

Run in terminal:
```bash
uv add numpy scikit-learn matplotlib seaborn
```

Expected output: packages installed into `.venv`

- [ ] **Step 2: Verify all imports work in a notebook cell**

Open `exploration.ipynb`, add a new cell and run:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("All imports OK")
```

Expected output: `All imports OK` — no errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add ml dependencies (scikit-learn, matplotlib, seaborn, numpy)"
```

---

## Task 2: Extend Data Exploration

**Files:**
- Modify: `exploration.ipynb` — add sections below the existing cells

**ML Terms introduced in this task:**
- **Dataset** — the raw table of observations
- **Feature** — a column used as input to the model (date, market, price)
- **Target variable** — what we're trying to predict (next month's price)
- **Time series** — data where order matters because each row depends on past rows
- **Distribution** — how values are spread (min, max, average, shape of histogram)

- [ ] **Step 1: Add a markdown cell explaining the goal**

Add this markdown cell after the existing `df.info()` cell:
```markdown
## What we're building

A **supervised regression** model that predicts `price` (UGX/kg) for the *next* month.

**Key terms:**
- **Supervised learning** — we train on examples where we already know the answer (historical prices)
- **Regression** — the output is a continuous number (price), not a category
- **Target variable** — `price` shifted one month forward (what the model must predict)
- **Features** — everything the model sees as input: past prices, rolling averages, month, year
```

- [ ] **Step 2: Check what Maize data exists per market**

Add and run this cell:
```python
maize_df = df[df["commodity"] == "Maize"].copy()

print("=== Maize rows per market ===")
print(maize_df.groupby("market")["price"].agg(["count", "min", "max", "mean"]).round(0))
```

Expected output: table showing Owino with ~202 rows, price range roughly 200–2500 UGX/kg.

- [ ] **Step 3: Plot raw Maize price over time for Owino**

Add and run this cell:
```python
owino = maize_df[maize_df["market"] == "Owino"].copy()
owino["date"] = pd.to_datetime(owino["date"])
owino = owino.sort_values("date")

plt.figure(figsize=(14, 4))
plt.plot(owino["date"], owino["price"], color="steelblue", linewidth=1.5)
plt.title("Maize Price Over Time — Owino Market, Kampala (UGX/kg)")
plt.xlabel("Date")
plt.ylabel("Price (UGX/kg)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Rows: {len(owino)}")
print(f"Date range: {owino['date'].min().date()} to {owino['date'].max().date()}")
```

Expected: a line chart from 2006 to 2022 showing upward trend with seasonal dips/peaks.

- [ ] **Step 4: Add seasonality explanation markdown cell**

```markdown
## What the chart tells us

The price is not random — it has structure:
- **Trend** — prices generally rise over time (inflation, population growth)
- **Seasonality** — prices drop after harvest (Oct–Jan) and climb before harvest (Mar–Jul)
- **Noise** — random variation the model won't capture (droughts, border closures, COVID)

This is why our features will include **month** (captures seasonality) and **year** (captures trend).
```

- [ ] **Step 5: Check for missing months**

Add and run this cell:
```python
owino_sorted = owino.set_index("date").sort_index()
expected_months = pd.date_range(
    start=owino_sorted.index.min(),
    end=owino_sorted.index.max(),
    freq="MS"
)
actual_months = owino_sorted.index.normalize()
missing = expected_months[~expected_months.isin(actual_months)]
print(f"Missing months: {len(missing)}")
if len(missing) > 0:
    print(missing.tolist())
```

This tells us if any months have gaps (which breaks lag calculations). Note the output — we handle gaps in the feature engineering step.

- [ ] **Step 6: Commit**

```bash
git add exploration.ipynb
git commit -m "feat: extend exploration with maize/owino analysis and seasonality notes"
```

---

## Task 3: Feature Engineering Notebook

**Files:**
- Create: `02_feature_engineering.ipynb`
- Create: `data/maize_owino_features.csv`

**ML Terms introduced:**
- **Feature engineering** — transforming raw data into inputs the model can learn from
- **Lag feature** — "what was the value N periods ago?" (lag_1 = last month's price)
- **Rolling window** — average over the past N months (smooths out noise, captures trend)
- **Train/test split** — dividing data so the model never sees test examples during training
- **Data leakage** — accidentally using future information to predict the past (a fatal error)

- [ ] **Step 1: Create the notebook and load data**

Create `02_feature_engineering.ipynb`. Add and run:
```python
import pandas as pd
import numpy as np
import os

df = pd.read_csv("wfp_food_prices_uga.csv", parse_dates=["date"])

# Filter to Maize in Owino (Kampala)
owino = df[(df["commodity"] == "Maize") & (df["market"] == "Owino")].copy()
owino = owino.sort_values("date").reset_index(drop=True)

print(f"Rows: {len(owino)}")
print(f"Date range: {owino['date'].min().date()} to {owino['date'].max().date()}")
owino[["date", "price"]].head(8)
```

Expected: 202 rows, 2006-02-15 to 2022-05-15.

- [ ] **Step 2: Add a markdown cell explaining the feature plan**

```markdown
## Feature Engineering Plan

Each row will represent one month. The model sees:

| Feature | Description | Why useful |
|---------|-------------|------------|
| `price_lag_1` | Price last month | Strong predictor — prices are sticky |
| `price_lag_2` | Price 2 months ago | Captures short-term momentum |
| `price_lag_3` | Price 3 months ago | Captures short-term momentum |
| `rolling_mean_3` | Avg price over last 3 months | Smooths noise, shows recent trend |
| `rolling_mean_6` | Avg price over last 6 months | Captures medium-term trend |
| `month` | Calendar month (1–12) | Captures harvest seasonality |
| `year` | Calendar year | Captures long-term inflation |

**Target:** `price` shifted forward by 1 month — i.e., "next month's price"

**Important — avoid data leakage:**
All rolling/lag calculations use `.shift(1)` before `.rolling()` so we never include
the current month's price in the rolling window. The model only sees the past.
```

- [ ] **Step 3: Build the feature engineering function**

Add and run:
```python
def build_features(price_series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """
    Takes a sorted price series and returns a DataFrame of features + target.
    All features use only past data (no leakage).
    """
    df = pd.DataFrame({"date": dates, "price": price_series}).reset_index(drop=True)

    # Lag features: look back N months
    df["price_lag_1"] = df["price"].shift(1)
    df["price_lag_2"] = df["price"].shift(2)
    df["price_lag_3"] = df["price"].shift(3)

    # Rolling averages over lagged series (shift first to avoid leakage)
    df["rolling_mean_3"] = df["price"].shift(1).rolling(window=3).mean()
    df["rolling_mean_6"] = df["price"].shift(1).rolling(window=6).mean()

    # Calendar features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # Target: next month's price
    df["target"] = df["price"].shift(-1)

    # Drop rows where we can't compute all features (first 6 rows) or target (last row)
    df = df.dropna().reset_index(drop=True)

    return df

features_df = build_features(owino["price"], owino["date"])
print(f"Shape after feature engineering: {features_df.shape}")
print(f"Date range: {features_df['date'].min().date()} to {features_df['date'].max().date()}")
features_df.head(4)
```

Expected: ~195 rows (202 minus ~7 dropped for NaN), all 9 columns present.

- [ ] **Step 4: Verify no data leakage**

Add and run:
```python
# Sanity check: for row 0, price_lag_1 should equal the price from the previous month
row = features_df.iloc[0]
print(f"Row 0 date: {row['date'].date()}")
print(f"Row 0 price_lag_1: {row['price_lag_1']:.2f}")

# Find that date in the original owino df
prev_idx = owino[owino["date"] < row["date"]].index[-1]
print(f"Actual previous month price: {owino.loc[prev_idx, 'price']:.2f}")
print("Match:", abs(row['price_lag_1'] - owino.loc[prev_idx, 'price']) < 0.01)
```

Expected: `Match: True` — confirms the lag is pointing to the right historical value.

- [ ] **Step 5: Show the train/test split**

Add and run:
```python
FEATURE_COLS = [
    "price_lag_1", "price_lag_2", "price_lag_3",
    "rolling_mean_3", "rolling_mean_6",
    "month", "year"
]

train = features_df[features_df["year"] <= 2019].copy()
test  = features_df[features_df["year"] >= 2020].copy()

print(f"Train: {len(train)} rows  ({train['year'].min()}–{train['year'].max()})")
print(f"Test:  {len(test)} rows  ({test['year'].min()}–{test['year'].max()})")
print()
print("Train target stats:")
print(train["target"].describe().round(0))
```

Expected: Train ~162 rows (2006–2019), Test ~33 rows (2020–2022).

- [ ] **Step 6: Add markdown explaining the split**

```markdown
## Why we split by time (not randomly)

In a normal ML problem you might randomly shuffle rows into train/test.
For time series, that would be **data leakage** — the model would be trained on
data from 2021 and tested on data from 2010. That's cheating: in the real world
you can only predict the future, not the past.

**Rule:** the test set must always be the most recent period.
We use 2020–2022 as the test set because it contains COVID-era price shocks,
which are a realistic challenge for the model.
```

- [ ] **Step 7: Save engineered data**

Add and run:
```python
os.makedirs("data", exist_ok=True)
features_df.to_csv("data/maize_owino_features.csv", index=False)
print("Saved to data/maize_owino_features.csv")
```

- [ ] **Step 8: Commit**

```bash
git add 02_feature_engineering.ipynb data/maize_owino_features.csv
git commit -m "feat: feature engineering notebook — lag, rolling, calendar features for Owino maize"
```

---

## Task 4: Model Training and Evaluation Notebook

**Files:**
- Create: `03_model_training.ipynb`
- Create: `price_prediction_results.png`

**ML Terms introduced:**
- **Model** — the mathematical function that maps features → predictions
- **Training** — the process of fitting the model to the training data
- **Random Forest** — an ensemble of many decision trees, each trained on a random sample
- **Decision tree** — a model that splits data by asking yes/no questions on features
- **Ensemble** — combining many weak models to get a stronger prediction
- **Hyperparameter** — a setting you choose before training (e.g., `n_estimators=100`)
- **MAE (Mean Absolute Error)** — on average, how many UGX is the prediction off by
- **Overfitting** — model memorizes training data but fails on new data
- **Feature importance** — how much each feature contributed to the model's decisions

- [ ] **Step 1: Create the notebook and load engineered data**

Create `03_model_training.ipynb`. Add and run:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

features_df = pd.read_csv("data/maize_owino_features.csv", parse_dates=["date"])

FEATURE_COLS = [
    "price_lag_1", "price_lag_2", "price_lag_3",
    "rolling_mean_3", "rolling_mean_6",
    "month", "year"
]

train = features_df[features_df["year"] <= 2019].copy()
test  = features_df[features_df["year"] >= 2020].copy()

X_train = train[FEATURE_COLS]
y_train = train["target"]
X_test  = test[FEATURE_COLS]
y_test  = test["target"]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
```

Expected: X_train (~162, 7), X_test (~33, 7).

- [ ] **Step 2: Add a markdown cell explaining Random Forest**

```markdown
## Why Random Forest?

A **Random Forest** is one of the most reliable models for tabular data.
Here's how it works:

1. Build 100 **decision trees**, each trained on a random subset of the training rows
2. Each tree learns rules like: *"if lag_1 > 800 AND month is March → predict 950"*
3. To predict, run the input through all 100 trees and average the results

**Why this is better than one tree:**
One tree tends to **overfit** — it memorises the training data perfectly but
fails on new data. Averaging 100 trees cancels out individual errors. This is
called an **ensemble**.

**Hyperparameters we're setting:**
- `n_estimators=100` — number of trees (more = more stable, slower to train)
- `random_state=42` — makes results reproducible (same shuffle every run)
```

- [ ] **Step 3: Train the model**

Add and run:
```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Training complete.")
print(f"Number of trees: {model.n_estimators}")
print(f"Features the model knows about: {FEATURE_COLS}")
```

Expected: instant output (small dataset). No errors.

- [ ] **Step 4: Evaluate on training data (check for overfitting)**

Add and run:
```python
y_train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
print(f"Train MAE: {train_mae:,.0f} UGX/kg")
```

Expected: train MAE will be very low (~10–50 UGX) because the model has memorised training data.

- [ ] **Step 5: Evaluate on test data (the real score)**

Add and run:
```python
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
avg_test_price = y_test.mean()

print(f"Test MAE:  {test_mae:,.0f} UGX/kg")
print(f"Avg test price: {avg_test_price:,.0f} UGX/kg")
print(f"Error as % of avg price: {test_mae / avg_test_price * 100:.1f}%")
```

Expected: test MAE significantly higher than train MAE (that gap = overfitting). A realistic test MAE might be 100–300 UGX/kg.

- [ ] **Step 6: Add a markdown cell explaining the MAE gap**

```markdown
## Reading the results

| Metric | Value | What it means |
|--------|-------|---------------|
| Train MAE | very low | Model memorised training data |
| Test MAE | higher | Real-world performance on unseen data |

The gap between train and test MAE is called **overfitting**.
The model learned the training data too well and struggles to generalise.

**Is this test MAE acceptable?**
If test MAE is 200 UGX/kg and average price is 1000 UGX/kg, the model is
off by ~20% on average. For a first model, that's a reasonable baseline.
We could reduce overfitting by limiting tree depth (`max_depth=5`) — but
for this learning project, the baseline result is the goal.
```

- [ ] **Step 7: Plot predicted vs actual**

Add and run:
```python
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# --- Chart 1: Predicted vs Actual ---
axes[0].plot(test["date"], y_test.values, label="Actual price",
             color="steelblue", linewidth=2, marker="o", markersize=4)
axes[0].plot(test["date"], y_test_pred, label="Predicted price",
             color="darkorange", linewidth=2, linestyle="--", marker="s", markersize=4)
axes[0].fill_between(test["date"], y_test.values, y_test_pred,
                     alpha=0.15, color="red", label="Error")
axes[0].set_title("Maize Price: Predicted vs Actual — Owino Market, Kampala (2020–2022)",
                  fontsize=13, fontweight="bold")
axes[0].set_ylabel("Price (UGX/kg)")
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].annotate(f"Test MAE: {test_mae:,.0f} UGX/kg ({test_mae/avg_test_price*100:.1f}% of avg price)",
                 xy=(0.02, 0.93), xycoords="axes fraction", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

# --- Chart 2: Feature Importance ---
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
colors = ["#2196F3" if i >= len(importance) - 3 else "#90CAF9" for i in range(len(importance))]
importance.plot(kind="barh", ax=axes[1], color=colors)
axes[1].set_title("Feature Importance — Which inputs matter most?", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Importance score (higher = model relies on this feature more)")
axes[1].axvline(x=1/len(FEATURE_COLS), color="red", linestyle="--", alpha=0.5, label="Equal weight baseline")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout(pad=2)
plt.savefig("price_prediction_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to price_prediction_results.png")
```

Expected: two charts — line chart of actual vs predicted prices (2020–2022), and a horizontal bar chart of feature importances. `price_lag_1` and `rolling_mean_3` should dominate.

- [ ] **Step 8: Add feature importance explanation**

```markdown
## Reading the Feature Importance chart

**Feature importance** tells you how much each input influenced the model's decisions
across all 100 trees. Higher = the model relied on this feature more.

Typical result for price prediction:
- `price_lag_1` dominates — last month's price is the strongest signal
- `rolling_mean_3` / `rolling_mean_6` come next — trend matters
- `month` ranks above `year` — seasonality is a real pattern in the data
- `price_lag_2` and `price_lag_3` are weaker — the model mostly cares about recent history

This confirms our intuition: **price momentum + seasonality = most of the story**.
```

- [ ] **Step 9: Print a human-readable summary**

Add and run:
```python
print("=" * 50)
print("MODEL SUMMARY")
print("=" * 50)
print(f"Commodity:  Maize")
print(f"Market:     Owino (Kampala)")
print(f"Train set:  2006–2019  ({len(X_train)} months)")
print(f"Test set:   2020–2022  ({len(X_test)} months)")
print(f"Algorithm:  Random Forest ({model.n_estimators} trees)")
print()
print(f"Train MAE:  {train_mae:,.0f} UGX/kg")
print(f"Test MAE:   {test_mae:,.0f} UGX/kg")
print(f"Avg price:  {avg_test_price:,.0f} UGX/kg")
print(f"Error rate: {test_mae / avg_test_price * 100:.1f}%")
print()
print("Top feature by importance:")
top = pd.Series(model.feature_importances_, index=FEATURE_COLS).idxmax()
print(f"  {top}")
```

- [ ] **Step 10: Commit everything**

```bash
git add 03_model_training.ipynb price_prediction_results.png data/
git commit -m "feat: model training notebook — random forest with MAE evaluation and feature importance chart"
```

---

## ML Glossary Reference (added to exploration.ipynb)

- [ ] **Add a final glossary cell to exploration.ipynb**

```markdown
## ML Glossary — Terms used in this project

| Term | Plain English |
|------|--------------|
| **Supervised learning** | Training on examples where the answer is known |
| **Regression** | Predicting a number (vs classification = predicting a category) |
| **Feature** | An input column the model uses to make predictions |
| **Target variable** | The column we're trying to predict |
| **Feature engineering** | Creating new features from raw data to help the model |
| **Lag feature** | "What was the value N months ago?" |
| **Rolling mean** | Average over a sliding window of past values |
| **Train/test split** | Dividing data so the model is evaluated on unseen examples |
| **Data leakage** | Accidentally using future data during training — invalidates results |
| **Overfitting** | Model memorises training data, fails to generalise |
| **Random Forest** | An ensemble of 100+ decision trees, averaged for better predictions |
| **Decision tree** | A model that makes predictions by asking yes/no questions |
| **Ensemble** | Combining many models to reduce error |
| **Hyperparameter** | A setting you choose before training (not learned from data) |
| **MAE** | Mean Absolute Error — average absolute difference between predicted and actual |
| **Feature importance** | How much each feature influenced the model's decisions |
| **Baseline** | The simplest possible model to beat (e.g., "always predict last month's price") |
```

- [ ] **Commit**

```bash
git add exploration.ipynb
git commit -m "docs: add ml glossary to exploration notebook"
```

---

## Done

After all tasks complete you should have:
- `exploration.ipynb` — extended with market analysis, seasonality chart, ML glossary
- `02_feature_engineering.ipynb` — lag features, rolling means, calendar features, leakage-free split
- `03_model_training.ipynb` — trained Random Forest, MAE scores, predicted-vs-actual chart, feature importance
- `price_prediction_results.png` — the LinkedIn-ready chart
- `data/maize_owino_features.csv` — saved engineered dataset
