# Equity Forecasting (ARIMA + GARCH)

Time-series forecasting of equities: ARIMA models for the conditional mean and GARCH models for the conditional volatility, with a small harness to fit, compare, and export results across tickers.

## Contents

- `arima_models.py` — ARIMA mean models (order selection, fit, forecast).
- `garch_model.py` — GARCH volatility models for conditional variance / vol clustering.
- `main.py` — runs the pipeline across a set of tickers.
- `stock_analysis.py` — quick multi-horizon price retrieval and summary stats for a single ticker.
- `model_results.csv`, `model_summary.csv`, `ticker_results/` — saved outputs.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Note
Research and learning code — not investment advice. Prices pulled live via yfinance.
