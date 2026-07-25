"""Walk-forward out-of-sample evaluation of ARIMA return forecasts.

The AIC grid in `arima_models.py` selects an order on whatever sample it is
given, and nothing ever grades the forecast that comes out. This module is the
missing report card, built the same way `backtest_vol.py` grades the GARCH fit:

- Walk-forward 1-step-ahead mean-return forecasts: every `refit_every` days
  the AIC grid is re-run on the expanding training window ONLY (order
  selection happens inside the walk — a full-sample AIC pick is itself a
  look-ahead), then the fitted model filters forward causally so the forecast
  for day t uses returns through day t-1.
- Two null models: the zero forecast (log prices are a random walk — the
  efficient-markets default for daily equity returns) and the expanding
  historical mean. If ARIMA cannot beat the zero forecast, the fit is
  decoration.
- Losses: MSE and MAE on returns, directional hit rate, and a Diebold-Mariano
  test of ARIMA vs the zero forecast on squared-error loss.

Core functions take a return series directly so everything is testable
offline; the CLI fetches prices via yfinance.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from arima_models import fetch_prices, fit_arima_grid, log_returns
from backtest_vol import diebold_mariano, mse_loss


# ---------------------------------------------------------------------------
# Forecasters — each returns a Series of 1-step-ahead DAILY RETURN forecasts
# aligned to the day being forecast (forecast for index t uses data <= t-1).
# ---------------------------------------------------------------------------


def arima_walkforward_mean(
    returns: pd.Series,
    min_train: int = 500,
    refit_every: int = 63,
    p_max: int = 2,
    q_max: int = 2,
    d_grid=(0,),
) -> pd.Series:
    """Walk-forward 1-step-ahead return forecasts from an AIC-selected ARIMA.

    Every `refit_every` days the small AIC grid is re-run on the expanding
    training window and the winning model is refit — so both the ORDER and the
    parameters at any point in the walk are functions of past data only.
    Between refits the fitted model is extended over the block with
    ``append(refit=False)`` and one-step-ahead (``dynamic=False``) predictions
    are taken, so no forecast conditions on its own day or later.

    Returns are modelled directly (default d_grid=(0,): log returns are
    already the differenced series).
    """
    y = returns.dropna().astype(float)
    vals = y.values
    n = vals.size
    if n <= min_train:
        raise ValueError(f"Need more than min_train={min_train} observations, got {n}.")

    forecasts = np.full(n, np.nan)
    for start in range(min_train, n, refit_every):
        end = min(start + refit_every, n)
        # Integer-indexed slices: dates carry no information for the fit and
        # statsmodels' date/freq plumbing chokes on irregular calendars.
        train = pd.Series(vals[:start])
        best = fit_arima_grid(train, p_max=p_max, d_grid=d_grid, q_max=q_max)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            ext = best.model.append(
                pd.Series(vals[start:end], index=pd.RangeIndex(start, end)),
                refit=False,
            )
            pred = ext.get_prediction(start=start, end=end - 1, dynamic=False)
        forecasts[start:end] = np.asarray(pred.predicted_mean)

    return pd.Series(forecasts[min_train:], index=y.index[min_train:], name="arima_ret")


def zero_forecast(returns: pd.Series, min_train: int = 500) -> pd.Series:
    """Random-walk null: tomorrow's expected log return is zero."""
    y = returns.dropna().astype(float)
    return pd.Series(0.0, index=y.index[min_train:], name="zero_ret")


def expanding_mean_forecast(returns: pd.Series, min_train: int = 500) -> pd.Series:
    """Expanding historical mean: forecast for day t is mean(r[:t])."""
    y = returns.dropna().astype(float)
    # shift(1) so the mean for day t ends at t-1 — no same-day information.
    mean = y.expanding().mean().shift(1)
    return mean.iloc[min_train:].rename("mean_ret")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def hit_rate(realized: pd.Series, forecast: pd.Series) -> float:
    """Directional accuracy on days where the forecast takes a side."""
    df = pd.concat([realized, forecast], axis=1).dropna()
    took_side = df.iloc[:, 1] != 0.0
    if not took_side.any():
        return float("nan")
    r = np.sign(df.loc[took_side].iloc[:, 0])
    f = np.sign(df.loc[took_side].iloc[:, 1])
    return float((r == f).mean())


def backtest_arima_returns(
    returns: pd.Series,
    min_train: int = 500,
    refit_every: int = 63,
    p_max: int = 2,
    q_max: int = 2,
) -> pd.DataFrame:
    """Score walk-forward ARIMA vs null models on one return series."""
    realized = returns.dropna().astype(float).iloc[min_train:].rename("realized_ret")

    models = {
        "ARIMA(walk-forward)": arima_walkforward_mean(
            returns, min_train=min_train, refit_every=refit_every,
            p_max=p_max, q_max=q_max,
        ),
        "Zero(random walk)": zero_forecast(returns, min_train=min_train),
        "ExpandingMean": expanding_mean_forecast(returns, min_train=min_train),
    }

    losses = {}
    rows = []
    for name, fc in models.items():
        aligned = pd.concat([realized, fc], axis=1).dropna()
        rv, f = aligned.iloc[:, 0], aligned.iloc[:, 1]
        sq = mse_loss(rv, f)
        losses[name] = sq
        rows.append(
            {
                "model": name,
                "n_oos": len(aligned),
                "mse_ret": float(sq.mean()),
                "mae_ret": float((rv - f).abs().mean()),
                "hit_rate": hit_rate(rv, f),
            }
        )

    df = pd.DataFrame(rows).set_index("model")
    arima_name = "ARIMA(walk-forward)"
    zero_name = "Zero(random walk)"
    try:
        stat, pval = diebold_mariano(losses[arima_name], losses[zero_name])
        df.loc[arima_name, "dm_vs_zero_stat"] = stat
        df.loc[arima_name, "dm_vs_zero_p"] = pval
    except ValueError:
        pass
    return df


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def run_backtest(
    tickers: List[str],
    period: str = "5y",
    min_train: int = 500,
    refit_every: int = 63,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            rets = log_returns(fetch_prices(t, period=period))
            df = backtest_arima_returns(rets, min_train=min_train, refit_every=refit_every)
            df.insert(0, "ticker", t)
            frames.append(df)
            print(f"\n=== {t} ({len(rets)} obs, {int(df['n_oos'].iloc[0])} OOS days) ===")
            with pd.option_context("display.width", 160, "display.float_format", "{:.6g}".format):
                print(df.drop(columns="ticker").to_string())
        except Exception as exc:
            print(f"[WARN] {t}: {exc}")
    if not frames:
        raise RuntimeError("No ticker produced a backtest result.")
    result = pd.concat(frames)
    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        result.to_csv(out_csv)
        print(f"\nSaved: {out_csv}")
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Out-of-sample ARIMA return backtest")
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "GOOGL"])
    ap.add_argument("--period", default="5y", help="History period, e.g. 3y, 5y, 10y")
    ap.add_argument("--min-train", type=int, default=500, help="Initial training window (days)")
    ap.add_argument("--refit-every", type=int, default=63, help="Refit/order-reselection cadence (days)")
    default_csv = os.path.join(os.path.dirname(__file__), "backtest_results", "arima_backtest.csv")
    ap.add_argument("--out-csv", default=default_csv)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    run_backtest(
        tickers=ns.tickers,
        period=ns.period,
        min_train=ns.min_train,
        refit_every=ns.refit_every,
        out_csv=ns.out_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
