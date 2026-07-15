import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def fetch_prices(ticker: str, period: str = "3y", interval: str = "1d") -> pd.Series:
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data.empty or "Close" not in data.columns:
        raise ValueError("No price data fetched. Check ticker, period, or internet connection.")
    px = data["Close"].astype(float).copy()
    px.name = ticker
    return px


def log_returns(prices: pd.Series) -> pd.Series:
    lr = np.log(prices).diff().dropna()
    lr.name = f"{prices.name}_logret"
    return lr


def _import_arima():
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        return ARIMA
    except Exception as exc:  # pragma: no cover
        import sys
        raise ImportError(
            "Failed to import 'statsmodels'. Install into this interpreter:\n"
            f"{sys.executable} -m pip install statsmodels\n"
            f"Original error: {exc}"
        ) from exc


@dataclass
class ArimaResult:
    order: Tuple[int, int, int]
    aic: float
    model: object


def fit_arima_grid(
    series: pd.Series,
    p_max: int = 3,
    d_grid: Iterable[int] = (0, 1),
    q_max: int = 3,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    verbose: bool = False,
    freq: Optional[str] = None,
) -> ArimaResult:
    """Simple ARIMA grid search minimizing AIC.

    Fits ARIMA(p,d,q) for small orders; returns the best by AIC.
    Expects a (roughly) stationary series; pass log returns for equities.
    """
    ARIMA = _import_arima()

    best: Optional[ArimaResult] = None
    y = series.dropna().astype(float)

    # Try to provide frequency info to avoid statsmodels ValueWarning
    if freq is None:
        try:
            freq = pd.infer_freq(y.index) or "B"
        except Exception:
            freq = "B"

    for d in d_grid:
        for p in range(0, p_max + 1):
            for q in range(0, q_max + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    m = ARIMA(
                        y,
                        order=(p, d, q),
                        dates=y.index,
                        freq=freq,
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility,
                    ).fit(method_kwargs={"disp": 0})
                    aic = float(m.aic)
                    if verbose:
                        print(f"ARIMA{(p, d, q)} AIC={aic:.2f}")
                    if best is None or aic < best.aic:
                        best = ArimaResult(order=(p, d, q), aic=aic, model=m)
                except Exception as exc:
                    if verbose:
                        print(f"Skip ARIMA{(p, d, q)}: {exc}")
                    continue

    if best is None:
        raise RuntimeError("No ARIMA model could be fit. Try different ranges or check data.")
    return best


def forecast_returns(best: ArimaResult, steps: int = 5) -> pd.Series:
    f = best.model.get_forecast(steps=steps)
    return f.predicted_mean


def forecast_prices_from_returns(
    last_price: float, ret_forecast: pd.Series
) -> pd.Series:
    # Convert forecasted log returns into price path (compounded)
    cum_ret = ret_forecast.cumsum()
    prices = last_price * np.exp(cum_ret)
    prices.name = "price_forecast"
    return prices


def run_for_tickers(
    tickers: Iterable[str],
    period: str = "3y",
    steps: int = 5,
    verbose: bool = False,
) -> None:
    for t in tickers:
        try:
            px = fetch_prices(t, period=period)
            rets = log_returns(px)
            best = fit_arima_grid(rets, verbose=verbose)
            r_fc = forecast_returns(best, steps=steps)
            p_fc = forecast_prices_from_returns(float(px.iloc[-1]), r_fc)
            print(f"Ticker: {t}")
            print(f"  Best ARIMA order: {best.order}  (AIC={best.aic:.2f})")
            print("  Return forecast:")
            print(r_fc.to_string())
            print("  Price forecast:")
            print(p_fc.to_string())
        except Exception as exc:
            print(f"[WARN] {t}: {exc}")


if __name__ == "__main__":
    # Quick demo
    run_for_tickers(["AAPL", "MSFT", "GOOGL"], period="3y", steps=5, verbose=False)
