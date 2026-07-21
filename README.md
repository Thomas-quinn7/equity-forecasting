# Equity Forecasting (ARIMA + GARCH)

Time-series forecasting of equities: ARIMA models for the conditional mean and GJR-GARCH models for the conditional volatility, with a harness to fit, compare, and export results across tickers, and a walk-forward out-of-sample backtest that checks whether the volatility forecasts actually beat standard baselines.

## Contents

- `garch_model.py` - GJR-GARCH fitting (AIC/BIC order selection over a small grid), multi-step volatility forecasts, and a hand-rolled residual-bootstrap confidence interval for the forecast path.
- `arima_models.py` - ARIMA grid search on log returns, return forecasts, and the compounded price path implied by them.
- `main.py` - runs both models across a set of tickers and appends results to CSV (idempotent per ticker/model/day).
- `backtest_vol.py` - out-of-sample validation (see below).
- `tests/` - offline test suite on simulated GJR-GARCH data; no network needed. Includes a no-lookahead guard test on the backtest.
- `model_results.csv`, `model_summary.csv`, `ticker_results/`, `backtest_results/` - saved outputs.

## Out-of-sample validation (`backtest_vol.py`)

An in-sample GARCH fit says nothing about forecasting skill, so the backtest re-fits on an expanding window every 21 days and produces 1-step-ahead variance forecasts where the forecast for day t only ever sees returns through day t-1 (enforced by a corruption test in `tests/`). Forecasts are scored against squared daily returns with QLIKE (robust to the noisy proxy; Patton 2011) and MSE, compared against two baselines every practitioner would reach for first - RiskMetrics EWMA (lambda = 0.94) and a 21-day rolling window - plus a Mincer-Zarnowitz regression and a Diebold-Mariano test of GARCH vs EWMA.

Result on AAPL/MSFT/GOOGL, 5y history, ~750 OOS days each (see `backtest_results/vol_backtest.csv`): GJR-GARCH has the best QLIKE on all three tickers and the ranking GARCH < EWMA < rolling is consistent, but the Diebold-Mariano test only reaches 5% significance on two of the three (AAPL p = 0.02, GOOGL p = 0.04, MSFT p = 0.50), and the Mincer-Zarnowitz R-squared is small everywhere - daily squared returns are a noisy target. Honest summary: the GJR fit adds real but modest forecasting value over EWMA, and none of these models would impress on MSE alone.

```bash
python backtest_vol.py --tickers AAPL MSFT GOOGL --period 5y
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                # fit + forecast the default tickers
python -m pytest tests/ -q    # offline test suite (~40s)
```

## Note

Research and learning code - not investment advice. Prices pulled live via yfinance.
