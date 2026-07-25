"""Offline tests — synthetic data only, no network, no yfinance.

The GJR-GARCH simulator below generates returns with known parameters so the
fitting, filtering, and backtest plumbing can be validated without market data.
"""

import numpy as np
import pandas as pd
import pytest

from arima_models import fit_arima_grid, forecast_prices_from_returns, forecast_returns
from backtest_arima import (
    arima_walkforward_mean,
    backtest_arima_returns,
    expanding_mean_forecast,
    hit_rate,
    zero_forecast,
)
from backtest_vol import (
    _gjr_causal_filter,
    backtest_ticker_returns,
    ewma_variance,
    garch_walkforward_variance,
    mincer_zarnowitz,
    mse_loss,
    qlike_loss,
    rolling_variance,
)
from garch_model import fit_garch, forecast_volatility, log_returns


def simulate_gjr(
    n: int = 800,
    mu: float = 0.0005,
    omega: float = 0.05,
    alpha: float = 0.05,
    gamma: float = 0.08,
    beta: float = 0.88,
    seed: int = 7,
) -> pd.Series:
    """Simulate GJR-GARCH(1,1,1) daily returns in PERCENT units."""
    rng = np.random.default_rng(seed)
    burn = 200
    z = rng.standard_normal(n + burn)
    r = np.empty(n + burn)
    s2 = omega / (1.0 - alpha - gamma / 2.0 - beta)
    eps_prev = 0.0
    s2_prev = s2
    for t in range(n + burn):
        s2 = omega + alpha * eps_prev**2 + beta * s2_prev
        if eps_prev < 0:
            s2 += gamma * eps_prev**2
        eps = np.sqrt(s2) * z[t]
        r[t] = mu + eps
        eps_prev, s2_prev = eps, s2
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(r[burn:] / 100.0, index=idx, name="sim_ret")


@pytest.fixture(scope="module")
def sim_returns() -> pd.Series:
    return simulate_gjr()


def test_log_returns_math():
    px = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3))
    px.name = "X"
    lr = log_returns(px)
    assert len(lr) == 2
    assert lr.iloc[0] == pytest.approx(np.log(110.0 / 100.0))
    assert lr.iloc[1] == pytest.approx(np.log(99.0 / 110.0))


def test_fit_garch_recovers_persistence(sim_returns):
    fitted = fit_garch(sim_returns, p=1, o=1, q=1, dist="normal")
    params = fitted.params
    persistence = (
        float(params["alpha[1]"]) + float(params["gamma[1]"]) / 2.0 + float(params["beta[1]"])
    )
    # True persistence is 0.05 + 0.08/2 + 0.88 = 0.97; allow estimation noise.
    assert 0.85 < persistence < 1.0


def test_forecast_volatility_shapes_and_ci(sim_returns):
    fitted = fit_garch(sim_returns, p=1, o=1, q=1, dist="normal")
    var, vol, lo, hi = forecast_volatility(
        fitted, horizon=10, return_ci=True, n_sims=300, random_state=0
    )
    assert len(var) == len(vol) == len(lo) == len(hi) == 10
    assert (vol.values > 0).all()
    assert (lo.values <= hi.values).all()
    # Daily vol on ~1%-vol returns should come back in sane daily units.
    assert vol.iloc[0] < 0.1


def test_causal_filter_matches_arch_conditional_vol(sim_returns):
    """Our hand-rolled recursion must agree with arch's own filtered variance."""
    y = sim_returns.values * 100.0
    fitted = fit_garch(pd.Series(y, index=sim_returns.index), rescale=False, dist="normal")
    p_ = fitted.params
    sigma2 = _gjr_causal_filter(
        y,
        mu=float(p_["mu"]),
        omega=float(p_["omega"]),
        alphas=np.array([float(p_["alpha[1]"])]),
        gammas=np.array([float(p_["gamma[1]"])]),
        betas=np.array([float(p_["beta[1]"])]),
    )
    arch_sigma2 = fitted.conditional_volatility.values ** 2
    # Initial conditions differ (arch backcasts); compare after burn-in.
    rel_err = np.abs(sigma2[100:] - arch_sigma2[100:]) / arch_sigma2[100:]
    assert np.median(rel_err) < 0.02


def test_walkforward_no_lookahead(sim_returns):
    """Corrupting future returns must not change forecasts up to that day."""
    min_train, refit_every = 500, 63
    base = garch_walkforward_variance(
        sim_returns, min_train=min_train, refit_every=refit_every, dist="normal"
    )
    corrupt_at = 700  # inside the OOS window
    corrupted = sim_returns.copy()
    corrupted.iloc[corrupt_at:] = corrupted.iloc[corrupt_at:] * 10.0
    redo = garch_walkforward_variance(
        corrupted, min_train=min_train, refit_every=refit_every, dist="normal"
    )
    # Forecast for day t uses returns through t-1, so every forecast up to and
    # including day `corrupt_at` must be bit-identical.
    cutoff = sim_returns.index[corrupt_at]
    pd.testing.assert_series_equal(base.loc[:cutoff], redo.loc[:cutoff])
    # ...and the corruption must actually move later forecasts (sanity).
    after = sim_returns.index[corrupt_at + 5]
    assert (base.loc[after:] - redo.loc[after:]).abs().max() > 0


def test_ewma_recursion_and_causality():
    idx = pd.bdate_range("2024-01-01", periods=6)
    r = pd.Series([0.01, -0.02, 0.005, 0.01, -0.01, 0.03], index=idx)
    f = ewma_variance(r, min_train=2, lam=0.9)
    # sigma2[0] = var(r[:2]); sigma2[t] = 0.9*sigma2[t-1] + 0.1*r[t-1]^2
    s2 = np.var(r.values[:2])
    expected = []
    for t in range(1, 6):
        s2 = 0.9 * s2 + 0.1 * r.values[t - 1] ** 2
        if t >= 2:
            expected.append(s2)
    np.testing.assert_allclose(f.values, expected)
    # Causality: changing the last return must not change its own forecast.
    r2 = r.copy()
    r2.iloc[-1] = 0.5
    f2 = ewma_variance(r2, min_train=2, lam=0.9)
    assert f.iloc[-1] == f2.iloc[-1]


def test_rolling_variance_is_shifted():
    idx = pd.bdate_range("2024-01-01", periods=10)
    r = pd.Series(np.linspace(-0.02, 0.02, 10), index=idx)
    f = rolling_variance(r, min_train=5, window=3)
    # Forecast for day 5 = variance of days 2,3,4 (ends the day before).
    assert f.iloc[0] == pytest.approx(np.var(r.values[2:5], ddof=1))


def test_qlike_minimized_at_truth():
    rv = pd.Series([1e-4, 2e-4, 5e-4])
    assert qlike_loss(rv, rv).abs().max() < 1e-12
    assert qlike_loss(rv, rv * 2.0).mean() > 0
    assert qlike_loss(rv, rv * 0.5).mean() > 0
    # Under-prediction costs more than over-prediction by the same factor.
    assert qlike_loss(rv, rv * 0.5).mean() > qlike_loss(rv, rv * 2.0).mean()
    assert mse_loss(rv, rv).max() == 0.0


def test_mincer_zarnowitz_exact_line():
    x = pd.Series(np.linspace(1e-4, 5e-4, 50))
    y = 1e-5 + 2.0 * x
    mz = mincer_zarnowitz(y, x)
    assert mz.intercept == pytest.approx(1e-5, rel=1e-6)
    assert mz.slope == pytest.approx(2.0, rel=1e-6)
    assert mz.r2 == pytest.approx(1.0)


def test_backtest_ticker_returns_end_to_end(sim_returns):
    df = backtest_ticker_returns(sim_returns, min_train=500, refit_every=200)
    assert len(df) == 3  # GARCH + EWMA + rolling
    assert (df["qlike"] > 0).all()
    assert (df["n_oos"] > 250).all()
    # On data generated BY a GJR process, GARCH should beat the rolling window.
    assert df.loc["GJR-GARCH", "qlike"] < df.loc["Rolling21d", "qlike"]


def test_gjr_seed_uses_training_window_only(sim_returns):
    """The recursion seed must come from training data, not the forecast block.

    Corrupt returns from the first day of a later refit block onward: the
    forecast FOR that exact day conditions on returns through the prior day
    only, so it must be bit-identical — which fails if the filter seeds its
    initial variance from a window that includes the corrupted block.
    """
    min_train, refit_every = 500, 63
    corrupt_at = min_train + 2 * refit_every  # first day of the third block
    base = garch_walkforward_variance(
        sim_returns, min_train=min_train, refit_every=refit_every, dist="normal"
    )
    corrupted = sim_returns.copy()
    corrupted.iloc[corrupt_at:] = corrupted.iloc[corrupt_at:] * 10.0
    redo = garch_walkforward_variance(
        corrupted, min_train=min_train, refit_every=refit_every, dist="normal"
    )
    day = sim_returns.index[corrupt_at]
    assert base.loc[day] == redo.loc[day]  # exact, not approximate


def simulate_ar1(n=600, phi=0.4, sd=0.01, seed=21) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    r[0] = 0.0
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0.0, sd)
    return pd.Series(r, index=pd.bdate_range("2021-01-01", periods=n), name="ar1_ret")


def test_arima_walkforward_no_lookahead():
    """Corrupting future returns must not change forecasts up to that day."""
    rets = simulate_ar1()
    min_train, refit_every = 300, 150
    common = dict(min_train=min_train, refit_every=refit_every, p_max=1, q_max=1)
    base = arima_walkforward_mean(rets, **common)
    corrupt_at = 380  # inside the OOS window, mid-block
    corrupted = rets.copy()
    corrupted.iloc[corrupt_at:] = corrupted.iloc[corrupt_at:] * 10.0
    redo = arima_walkforward_mean(corrupted, **common)
    cutoff = rets.index[corrupt_at]
    pd.testing.assert_series_equal(base.loc[:cutoff], redo.loc[:cutoff])
    assert (base.loc[cutoff:] - redo.loc[cutoff:]).abs().max() > 0


def test_arima_walkforward_beats_zero_on_ar1():
    """On genuinely autocorrelated returns, ARIMA must beat the random-walk null."""
    rets = simulate_ar1(n=600, phi=0.4)
    df = backtest_arima_returns(rets, min_train=300, refit_every=150, p_max=1, q_max=1)
    assert len(df) == 3
    assert df.loc["ARIMA(walk-forward)", "mse_ret"] < df.loc["Zero(random walk)", "mse_ret"]
    assert df.loc["ARIMA(walk-forward)", "hit_rate"] > 0.55
    # DM test ran and favors ARIMA (negative statistic = lower loss).
    assert df.loc["ARIMA(walk-forward)", "dm_vs_zero_stat"] < 0


def test_return_baselines_are_causal():
    rets = simulate_ar1(n=320, seed=22)
    zero = zero_forecast(rets, min_train=300)
    assert (zero == 0.0).all()
    mean = expanding_mean_forecast(rets, min_train=300)
    # Forecast for the first OOS day is the mean of the training window only.
    assert mean.iloc[0] == pytest.approx(rets.iloc[:300].mean())
    # Changing the last return must not change its own forecast.
    rets2 = rets.copy()
    rets2.iloc[-1] = 0.5
    mean2 = expanding_mean_forecast(rets2, min_train=300)
    assert mean.iloc[-1] == mean2.iloc[-1]


def test_hit_rate_ignores_zero_forecasts():
    realized = pd.Series([0.01, -0.02, 0.01, 0.03])
    forecast = pd.Series([0.005, 0.01, 0.0, 0.01])
    # Scored days: 1st (right), 2nd (wrong), 4th (right) -> 2/3.
    assert hit_rate(realized, forecast) == pytest.approx(2.0 / 3.0)
    assert np.isnan(hit_rate(realized, pd.Series([0.0, 0.0, 0.0, 0.0])))


def test_arima_grid_and_price_path_offline():
    rng = np.random.default_rng(3)
    n = 300
    r = np.empty(n)
    r[0] = 0.0
    for t in range(1, n):  # AR(1) in returns
        r[t] = 0.4 * r[t - 1] + rng.normal(0, 0.01)
    rets = pd.Series(r, index=pd.bdate_range("2022-01-01", periods=n))
    best = fit_arima_grid(rets, p_max=2, q_max=1)
    assert best.order[0] >= 1  # should find some AR structure
    fc = forecast_returns(best, steps=5)
    assert len(fc) == 5
    px = forecast_prices_from_returns(100.0, fc)
    assert px.iloc[0] == pytest.approx(100.0 * np.exp(fc.iloc[0]))
    assert px.iloc[-1] == pytest.approx(100.0 * np.exp(fc.cumsum().iloc[-1]))
