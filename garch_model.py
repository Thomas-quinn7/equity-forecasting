import warnings
from typing import Optional, Tuple, Iterable
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


def fetch_prices(ticker: str, period: str = "3y", interval: str = "1d") -> pd.Series:
    """Fetch adjusted close prices for a ticker from Yahoo Finance.

    Args:
        ticker: Ticker symbol.
        period: Data period, e.g., "1y", "3y", "5y", "max".
        interval: Data interval, e.g., "1d", "1h".

    Returns:
        Series of adjusted close prices indexed by date.
    """
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data.empty or "Close" not in data.columns:
        raise ValueError("No price data fetched. Check ticker, period, or internet connection.")
    px = data["Close"].astype(float).copy()
    px.name = ticker
    return px


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from price series."""
    lr = np.log(prices).diff().dropna()
    lr.name = f"{prices.name}_logret"
    return lr


def _import_arch():
    try:
        from arch import arch_model  # type: ignore
        return arch_model
    except Exception as exc:  # pragma: no cover
        import sys
        raise ImportError(
            "Failed to import 'arch'. Install into this interpreter:\n"
            f"{sys.executable} -m pip install arch\n"
            f"Original error: {exc}"
        ) from exc


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    rescale: bool = True,
    show_output: bool = False,
):
    """Fit a GJR-GARCH(p, o, q) model to returns.

    Args:
        returns: Return series.
        p: ARCH order.
        o: Asymmetric (leverage) order for GJR term.
        q: GARCH order.
        mean: Mean model, e.g., "Zero", "Constant".
        dist: Innovation dist, e.g., "normal", "t", "skewt", "ged".
        rescale: Whether to rescale returns by 100 for numerical stability.
        show_output: Whether to print the fit summary.

    Returns:
        Fitted model result object.
    """
    arch_model = _import_arch()

    y = returns.copy()
    scale = 100.0 if rescale else 1.0
    y = y * scale

    # Using vol="GARCH" with o>0 produces a GJR-GARCH specification
    am = arch_model(y, vol="GARCH", p=p, o=o, q=q, mean=mean, dist=dist)
    res = am.fit(disp="off")
    if show_output:
        print(res.summary())
    return res


def select_gjr_orders(
    returns: pd.Series,
    p_grid: Iterable[int] = (1, 2),
    o_grid: Iterable[int] = (0, 1),
    q_grid: Iterable[int] = (1, 2),
    mean: str = "Constant",
    dist: str = "t",
    criterion: str = "aic",
    rescale: bool = True,
    show_output: bool = False,
):
    """Grid search GJR-GARCH(p,o,q) by AIC/BIC and return best orders and fit.

    Returns a tuple: (p_best, o_best, q_best, best_result)
    """
    best_val = float("inf")
    best_tuple: Optional[Tuple[int, int, int]] = None
    best_res = None

    for p in p_grid:
        for o in o_grid:
            for q in q_grid:
                if p == 0 and q == 0:
                    continue
                try:
                    res = fit_garch(
                        returns,
                        p=p,
                        o=o,
                        q=q,
                        mean=mean,
                        dist=dist,
                        rescale=rescale,
                        show_output=False,
                    )
                    val = float(res.bic) if criterion.lower() == "bic" else float(res.aic)
                    if val < best_val:
                        best_val = val
                        best_tuple = (p, o, q)
                        best_res = res
                except Exception:
                    continue

    if best_tuple is None or best_res is None:
        raise RuntimeError("Model selection failed for all candidate orders.")
    if show_output:
        print(f"Selected GJR-GARCH orders (p,o,q)={best_tuple} by {criterion.upper()}={best_val:.2f}")
    return best_tuple[0], best_tuple[1], best_tuple[2], best_res


def forecast_volatility(
    fitted_model,
    horizon: int = 20,
    unscale: bool = True,
    return_ci: bool = False,
    ci: float = 0.95,
    n_sims: int = 2000,
    random_state: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series] | Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Forecast variance and volatility for given horizon.

    If return_ci=True, also returns a simulation-based (1±alpha) interval
    for the volatility using bootstrap of standardized residuals under the
    fitted GJR-GARCH model.

    Args:
        fitted_model: Result from fit_garch.
        horizon: Steps ahead to forecast.
        unscale: If returns were scaled by 100, unscale the volatility.
        return_ci: Whether to return CI bounds for volatility.
        ci: Confidence level for interval (e.g., 0.95).
        n_sims: Number of bootstrap simulations for CI.
        random_state: Seed for reproducibility.

    Returns:
        (var, vol) or (var, vol, vol_lo, vol_hi)
    """
    f = fitted_model.forecast(horizon=horizon)
    var = f.variance.iloc[-1]
    vol = np.sqrt(var)

    if unscale:
        var = var / (100.0**2)
        vol = vol / 100.0
    var.name = "forecast_variance"
    vol.name = "forecast_volatility"

    if not return_ci:
        return var, vol

    # Simulation-based CI for volatility
    q_lo = (1.0 - ci) / 2.0
    q_hi = 1.0 - q_lo

    # Extract orders and parameters
    model = fitted_model.model
    try:
        p = int(getattr(model.volatility, "p", 1))
        o = int(getattr(model.volatility, "o", 0))
        q = int(getattr(model.volatility, "q", 1))
    except Exception:
        p, o, q = 1, 1, 1

    params = getattr(fitted_model, "params", None)
    if params is None:
        raise RuntimeError("Fitted model has no params; cannot simulate CIs.")

    def _coef_list(prefix: str, count: int) -> list[float]:
        vals = []
        for i in range(1, count + 1):
            key = f"{prefix}[{i}]"
            if key in params.index:
                vals.append(float(params[key]))
            else:
                vals.append(0.0)
        return vals

    omega = float(params.get("omega", 0.0))
    alphas = _coef_list("alpha", p)
    betas = _coef_list("beta", q)
    gammas = _coef_list("gamma", o)

    # Histories for recursion
    eps = fitted_model.resid.dropna().values
    sig = fitted_model.conditional_volatility.dropna().values
    L_eps = max(p, o)
    eps_hist0 = list(reversed(eps[-L_eps:].tolist())) if L_eps > 0 else []
    sig2_hist0 = list(reversed((sig[-q:] ** 2).tolist())) if q > 0 else []

    std_resid = getattr(fitted_model, "std_resid", None)
    if std_resid is None:
        raise RuntimeError("Fitted model has no standardized residuals for simulation.")
    std_resid = std_resid.dropna().values
    if std_resid.size == 0:
        raise RuntimeError("Standardized residuals are empty; cannot simulate CIs.")

    rng = np.random.default_rng(random_state)
    vol_paths = np.empty((horizon, n_sims), dtype=float)

    for s in range(n_sims):
        eps_hist = eps_hist0.copy()
        sig2_hist = sig2_hist0.copy()
        for h in range(horizon):
            s2 = omega
            # ARCH and GJR terms
            for i in range(p):
                if i < len(eps_hist):
                    e = eps_hist[i]
                    s2 += alphas[i] * (e * e)
            for j in range(o):
                if j < len(eps_hist):
                    e = eps_hist[j]
                    if e < 0.0:
                        s2 += gammas[j] * (e * e)
            # GARCH terms
            for k in range(q):
                if k < len(sig2_hist):
                    s2 += betas[k] * sig2_hist[k]

            # Store simulated volatility for this horizon
            vol_paths[h, s] = np.sqrt(max(s2, 0.0))

            # Generate next shock for recursion
            z = rng.choice(std_resid)
            e_next = z * vol_paths[h, s]

            # Update histories
            if L_eps > 0:
                eps_hist.insert(0, e_next)
                if len(eps_hist) > L_eps:
                    eps_hist.pop()
            if q > 0:
                sig2_hist.insert(0, s2)
                if len(sig2_hist) > q:
                    sig2_hist.pop()

    vol_lo = np.quantile(vol_paths, q_lo, axis=1)
    vol_hi = np.quantile(vol_paths, q_hi, axis=1)

    # Build series aligned with var/vol index
    idx = vol.index
    vol_lo_s = pd.Series(vol_lo, index=idx, name="forecast_volatility_lo")
    vol_hi_s = pd.Series(vol_hi, index=idx, name="forecast_volatility_hi")

    if unscale:
        vol_lo_s = vol_lo_s / 100.0
        vol_hi_s = vol_hi_s / 100.0

    return var, vol, vol_lo_s, vol_hi_s


def plot_conditional_volatility(fitted_model, title: Optional[str] = None) -> None:
    """Plot in-sample conditional volatility."""
    cond_vol = fitted_model.conditional_volatility
    cond_vol = cond_vol / 100.0
    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(cond_vol.index, cond_vol.values, label="Conditional Vol (daily)")
    plt.ylabel("Volatility")
    plt.xlabel("Date")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    plt.close()


def run_example(
    ticker: str = "AAPL",
    period: str = "3y",
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
    horizon: int = 20,
    show_output: bool = True,
):
    """End-to-end example: fetch prices, fit GARCH, forecast, and plot."""
    px = fetch_prices(ticker, period=period)
    rets = log_returns(px)
    fitted = fit_garch(rets, p=p, o=o, q=q, dist=dist, show_output=show_output)
    var, vol = forecast_volatility(fitted, horizon=horizon)

    print(f"Next {horizon} days volatility forecast for {ticker} (daily):")
    print(vol)
    plot_conditional_volatility(
        fitted, title=f"{ticker} GJR-GARCH({p},{o},{q}) Conditional Volatility"
    )


if __name__ == "__main__":
    run_example()
