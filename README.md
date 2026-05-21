# East Africa Food Price Prediction

An end-to-end machine learning project predicting food prices for traders, NGOs, and agricultural planners in East Africa.

Built as a hands-on learning project — raw data to trained model to production API.

## Structure

```
food_prices_model/    # Data exploration, feature engineering, model training
food_prices_api/      # FastAPI service + Streamlit demo (coming soon)
```

## food_prices_model

Trains a Random Forest model on 16 years of WFP Uganda food price data to predict next month's Maize price at Owino market, Kampala.

**Results:** Test MAE of 193 UGX/kg (~20% of average price) on 2020–2022 COVID-era data.

See [food_prices_model/](food_prices_model/) for full details.

## food_prices_api

FastAPI service wrapping the trained model, with a Streamlit demo UI.

Coming soon.

## Goal

A production API useful to traders, NGOs, and agricultural planners across East Africa.
Roadmap: expand to more commodities, more markets, and additional features (rainfall, fuel prices, conflict data).
