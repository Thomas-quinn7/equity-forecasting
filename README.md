# Equity Forecasting (ARIMA + GARCH)

![tests](https://github.com/Thomas-quinn7/equity-forecasting/actions/workflows/tests.yml/badge.svg)

Time-series forecasting of equities: ARIMA models for the conditional mean and GJR-GARCH models for the conditional volatility, with a harness to fit, compare, and export results across tickers, and a walk-forward out-of-sample backtest that checks whether the volatility forecasts actually beat standard baselines.

## Contents

- `garch_model.py` - GJR-GARCH fitting (AIC/BIC order selection over a small grid), multi-step volatility forecasts, and a hand-rolled residual-bootstrap confidence interval for the forecast path.
- `arima_models.py` - ARIMA grid search on log returns, return forecasts, and the compounded price path implied by them.
- `main.py` - runs both models across a set of tickers and appends results to CSV (idempotent per ticker/model/day).
- `backtest_vol.py` - out-of-sample validation of the volatility forecasts (see below).
- `backtest_arima.py` - out-of-sample validation of the ARIMA return forecasts (see below).
- `tests/` - offline test suite on simulated GJR-GARCH data; no network needed. Includes a no-lookahead guard test on the backtest.
- `model_results.csv`, `model_summary.csv`, `ticker_results/`, `backtest_results/` - saved outputs.

## Out-of-sample validation (`backtest_vol.py`)

An in-sample GARCH fit says nothing about forecasting skill, so the backtest re-fits on an expanding window every 21 days and produces 1-step-ahead variance forecasts where the forecast for day t only ever sees returns through day t-1 (enforced by a corruption test in `tests/`). Forecasts are scored against squared daily returns with QLIKE (robust to the noisy proxy; Patton 2011) and MSE, compared against two baselines every practitioner would reach for first - RiskMetrics EWMA (lambda = 0.94) and a 21-day rolling window - plus a Mincer-Zarnowitz regression and a Diebold-Mariano test of GARCH vs EWMA.

Result on AAPL/MSFT/GOOGL, 5y history, ~750 OOS days each (see `backtest_results/vol_backtest.csv`): GJR-GARCH has the best QLIKE on all three tickers and the ranking GARCH < EWMA < rolling is consistent, but the Diebold-Mariano test only reaches 5% significance on two of the three (AAPL p = 0.02, GOOGL p = 0.04, MSFT p = 0.50), and the Mincer-Zarnowitz R-squared is small everywhere - daily squared returns are a noisy target. Honest summary: the GJR fit adds real but modest forecasting value over EWMA, and none of these models would impress on MSE alone.

```bash
python backtest_vol.py --tickers AAPL MSFT GOOGL --period 5y
```

## Out-of-sample validation of the mean model (`backtest_arima.py`)

The same discipline applied to the ARIMA side, which previously produced
forecasts nothing ever graded. Every 63 days the AIC grid is re-run on the
expanding training window only — order selection happens *inside* the walk,
because a full-sample AIC pick is itself a look-ahead — and 1-step-ahead
return forecasts are scored (MSE, MAE, directional hit rate) against the two
null models any return forecast must beat: the zero forecast (random-walk
prices) and the expanding historical mean, with a Diebold-Mariano test of
ARIMA vs zero. A corruption test in `tests/` enforces causality, and a
synthetic-AR(1) test proves the harness detects real signal when it exists.

Result on AAPL/MSFT/GOOGL, 5y history, 753 OOS days each (see
`backtest_results/arima_backtest.csv`): ARIMA fails to beat the zero forecast
on every ticker — hit rate 48-49% (below a coin flip), and the Diebold-Mariano
test says its squared-error loss is *significantly worse* than zero's on AAPL
(p = 0.04) and MSFT (p = 0.05). Fitting AR structure to daily equity returns
adds noise, not signal. That is the finding, not a failure of the harness —
on synthetic AR(1) data the same pipeline beats zero decisively — and it is
why the repo's headline model is the volatility forecaster, where the GJR fit
does add value.

```bash
python backtest_arima.py --tickers AAPL MSFT GOOGL --period 5y
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
