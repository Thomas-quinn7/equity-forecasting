"""Walk-forward out-of-sample evaluation of GJR-GARCH volatility forecasts.

The in-sample fit in `garch_model.py` says nothing about whether the model
predicts volatility it has not seen. This module answers that question:

- Walk-forward 1-step-ahead variance forecasts: refit on an expanding window
  every `refit_every` days, then filter the GJR recursion causally so the
  forecast for day t uses returns through day t-1 only.
- Two standard baselines: RiskMetrics EWMA (lambda = 0.94) and a rolling
  realized-vol window. If GARCH cannot beat these, the fit is decoration.
- Losses: QLIKE (robust to noisy volatility proxies, Patton 2011) and MSE on
  variance, scored against squared daily returns as the realized proxy.
- Mincer-Zarnowitz regression (realized on forecast: slope ~ 1, intercept ~ 0
  for an unbiased forecast) and a Diebold-Mariano test of GARCH vs EWMA.

Core functions take a return series directly so everything is testable
offline; the CLI fetches prices via yfinance.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from garch_model import fetch_prices, fit_garch, log_returns


# ---------------------------------------------------------------------------
# Forecasters — each returns a Series of 1-step-ahead DAILY VARIANCE forecasts
# aligned to the day being forecast (forecast for index t uses data <= t-1).
# ---------------------------------------------------------------------------


def _gjr_causal_filter(
    y: np.ndarray,
    mu: float,
    omega: float,
    alphas: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Run the GJR-GARCH variance recursion over `y` and return sigma2.

    sigma2[t] is built from returns up to t-1 only, so it IS the one-step
    ahead variance forecast for day t. Initialized at the sample variance.
    """
    n = y.size
    p, o, q = len(alphas), len(gammas), len(betas)
    eps = y - mu
    sigma2 = np.full(n, np.var(y), dtype=float)
    for t in range(1, n):
        s2 = omega
        for i in range(1, p + 1):
            if t - i >= 0:
                s2 += alphas[i - 1] * eps[t - i] ** 2
        for j in range(1, o + 1):
            if t - j >= 0 and eps[t - j] < 0.0:
                s2 += gammas[j - 1] * eps[t - j] ** 2
        for k in range(1, q + 1):
            if t - k >= 0:
                s2 += betas[k - 1] * sigma2[t - k]
        sigma2[t] = max(s2, 1e-12)
    return sigma2


def _extract_params(fitted, p: int, o: int, q: int) -> tuple:
    params = fitted.params
    mu = float(params.get("mu", 0.0))
    omega = float(params.get("omega", 0.0))

    def coefs(prefix: str, count: int) -> np.ndarray:
        return np.array(
            [float(params.get(f"{prefix}[{i}]", 0.0)) for i in range(1, count + 1)]
        )

    return mu, omega, coefs("alpha", p), coefs("gamma", o), coefs("beta", q)


def garch_walkforward_variance(
    returns: pd.Series,
    min_train: int = 500,
    refit_every: int = 21,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
) -> pd.Series:
    """Walk-forward 1-step-ahead variance forecasts from GJR-GARCH(p,o,q).

    Refits on the expanding window every `refit_every` days; between refits
    the last fitted parameters filter the causal recursion forward, so no
    forecast ever conditions on data from its own day or later.

    Returns daily-variance forecasts (original return units) indexed to the
    out-of-sample days `returns.index[min_train:]`.
    """
    y_all = returns.dropna().astype(float).values * 100.0  # arch-style scaling
    idx = returns.dropna().index
    n = y_all.size
    if n <= min_train:
        raise ValueError(f"Need more than min_train={min_train} observations, got {n}.")

    forecasts = np.full(n, np.nan)
    for start in range(min_train, n, refit_every):
        end = min(start + refit_every, n)
        train = pd.Series(y_all[:start], index=idx[:start])
        fitted = fit_garch(train, p=p, o=o, q=q, dist=dist, rescale=False)
        mu, omega, alphas, gammas, betas = _extract_params(fitted, p, o, q)
        # Filter over history + this block; keep only the block's forecasts.
        sigma2 = _gjr_causal_filter(y_all[:end], mu, omega, alphas, gammas, betas)
        forecasts[start:end] = sigma2[start:end]

    out = pd.Series(forecasts[min_train:], index=idx[min_train:], name="garch_var")
    return out / (100.0**2)


def ewma_variance(
    returns: pd.Series, min_train: int = 500, lam: float = 0.94
) -> pd.Series:
    """RiskMetrics EWMA 1-step-ahead variance: s2_t = lam*s2_{t-1} + (1-lam)*r_{t-1}^2."""
    r = returns.dropna().astype(float).values
    idx = returns.dropna().index
    n = r.size
    sigma2 = np.full(n, np.nan)
    sigma2[0] = np.var(r[: max(min_train, 2)])
    for t in range(1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return pd.Series(sigma2[min_train:], index=idx[min_train:], name="ewma_var")


def rolling_variance(
    returns: pd.Series, min_train: int = 500, window: int = 21
) -> pd.Series:
    """Trailing `window`-day sample variance as the forecast for the next day."""
    r = returns.dropna().astype(float)
    # shift(1) so the window for day t ends at t-1 — no same-day information.
    var = r.rolling(window).var().shift(1)
    return var.iloc[min_train:].rename("rolling_var")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def qlike_loss(realized_var: pd.Series, forecast_var: pd.Series) -> pd.Series:
    """QLIKE loss, per day: rv/f - log(rv/f) - 1 (>= 0, zero iff f == rv).

    Robust to noise in the realized proxy (Patton 2011); heavily penalizes
    under-prediction of variance, which is the expensive error for risk.
    """
    rv = realized_var.clip(lower=1e-12)
    f = forecast_var.clip(lower=1e-12)
    ratio = rv / f
    return (ratio - np.log(ratio) - 1.0).rename("qlike")


def mse_loss(realized_var: pd.Series, forecast_var: pd.Series) -> pd.Series:
    return ((realized_var - forecast_var) ** 2).rename("mse")


@dataclass
class MincerZarnowitz:
    intercept: float
    slope: float
    r2: float


def mincer_zarnowitz(realized_var: pd.Series, forecast_var: pd.Series) -> MincerZarnowitz:
    """OLS of realized on forecast variance. Unbiased forecast: a=0, b=1."""
    df = pd.concat([realized_var, forecast_var], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return MincerZarnowitz(intercept=float(beta[0]), slope=float(beta[1]), r2=r2)


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series) -> tuple[float, float]:
    """DM test on the loss differential (a - b). Negative stat favors model a.

    Plain (lag-0) variance is appropriate for 1-step-ahead forecasts.
    Returns (statistic, two-sided p-value under the normal approximation).
    """
    from scipy import stats

    d = (loss_a - loss_b).dropna().values
    n = d.size
    if n < 30:
        raise ValueError("Too few observations for a DM test.")
    stat = float(d.mean() / np.sqrt(d.var(ddof=1) / n))
    pval = float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
    return stat, pval


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def backtest_ticker_returns(
    returns: pd.Series,
    min_train: int = 500,
    refit_every: int = 21,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
    ewma_lambda: float = 0.94,
    rolling_window: int = 21,
) -> pd.DataFrame:
    """Score GARCH vs baselines on one return series. One row per model."""
    realized = (returns.dropna() ** 2).iloc[min_train:].rename("realized_var")

    models = {
        "GJR-GARCH": garch_walkforward_variance(
            returns, min_train=min_train, refit_every=refit_every, p=p, o=o, q=q, dist=dist
        ),
        f"EWMA({ewma_lambda})": ewma_variance(returns, min_train=min_train, lam=ewma_lambda),
        f"Rolling{rolling_window}d": rolling_variance(
            returns, min_train=min_train, window=rolling_window
        ),
    }

    losses = {}
    rows = []
    for name, fvar in models.items():
        aligned = pd.concat([realized, fvar], axis=1).dropna()
        rv, f = aligned.iloc[:, 0], aligned.iloc[:, 1]
        ql = qlike_loss(rv, f)
        losses[name] = ql
        mz = mincer_zarnowitz(rv, f)
        rows.append(
            {
                "model": name,
                "n_oos": len(aligned),
                "qlike": float(ql.mean()),
                "mse_var": float(mse_loss(rv, f).mean()),
                "mz_intercept": mz.intercept,
                "mz_slope": mz.slope,
                "mz_r2": mz.r2,
                "ann_vol_forecast_mean": float(np.sqrt(f.mean() * 252.0)),
                "ann_vol_realized_mean": float(np.sqrt(rv.mean() * 252.0)),
            }
        )

    df = pd.DataFrame(rows).set_index("model")
    garch_name = "GJR-GARCH"
    ewma_name = f"EWMA({ewma_lambda})"
    try:
        stat, pval = diebold_mariano(losses[garch_name], losses[ewma_name])
        df.loc[garch_name, "dm_vs_ewma_stat"] = stat
        df.loc[garch_name, "dm_vs_ewma_p"] = pval
    except ValueError:
        pass
    return df


def run_backtest(
    tickers: List[str],
    period: str = "5y",
    min_train: int = 500,
    refit_every: int = 21,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            rets = log_returns(fetch_prices(t, period=period))
            df = backtest_ticker_returns(rets, min_train=min_train, refit_every=refit_every)
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
    ap = argparse.ArgumentParser(description="Out-of-sample GARCH vol backtest")
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "GOOGL"])
    ap.add_argument("--period", default="5y", help="History period, e.g. 3y, 5y, 10y")
    ap.add_argument("--min-train", type=int, default=500, help="Initial training window (days)")
    ap.add_argument("--refit-every", type=int, default=21, help="Refit cadence (days)")
    default_csv = os.path.join(os.path.dirname(__file__), "backtest_results", "vol_backtest.csv")
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
