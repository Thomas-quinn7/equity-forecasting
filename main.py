import argparse
import sys
from typing import List, Optional

import os
import re
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Support both package execution (python -m Stock_pred.garch_main)
# and direct script execution (python Stock_pred/garch_main.py)
try:
    from .garch_model import (
        fetch_prices,
        log_returns,
        fit_garch,
        forecast_volatility,
        plot_conditional_volatility,
        select_gjr_orders,
    )
    from .arima_models import (
        fit_arima_grid,
        forecast_returns,
        forecast_prices_from_returns,
    )
except Exception:  # pragma: no cover - fallback for direct script run
    import os

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from Stock_pred.garch_model import (  # type: ignore
        fetch_prices,
        log_returns,
        fit_garch,
        forecast_volatility,
        plot_conditional_volatility,
        select_gjr_orders,
    )
    from Stock_pred.arima_models import (  # type: ignore
        fit_arima_grid,
        forecast_returns,
        forecast_prices_from_returns,
    )


def _append_records(out_csv: str, records: list[dict]) -> None:
    if not records:
        return
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df_new = pd.DataFrame.from_records(records)
    if os.path.exists(out_csv):
        try:
            df_old = pd.read_csv(out_csv)
            cols = list(df_old.columns)
            for c in df_new.columns:
                if c not in cols:
                    cols.append(c)
            df_old = df_old.reindex(columns=cols)
            df_new = df_new.reindex(columns=cols)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all.to_csv(out_csv, index=False)
            return
        except Exception:
            # Fallback to simple append with header preserved
            pass
    # If file doesn't exist or fallback case
    df_new.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)


def _per_ticker_path(base_dir: Optional[str], ticker: str) -> Optional[str]:
    if not base_dir:
        return None
    os.makedirs(base_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
    return os.path.join(base_dir, f"{safe}.csv")


def _has_run_for(out_csv: Optional[str], ticker: str, run_date: str, model: str) -> bool:
    """Return True if the CSV already contains rows for (ticker, run_date, model).

    This prevents double-counting results when running multiple times in a day.
    """
    if not out_csv or not os.path.exists(out_csv):
        return False
    try:
        # Chunked read to handle large files efficiently
        for chunk in pd.read_csv(
            out_csv,
            usecols=["ticker", "run_date", "model"],
            dtype=str,
            chunksize=50000,
        ):
            mask = (
                (chunk["ticker"] == ticker)
                & (chunk["run_date"] == run_date)
                & (chunk["model"] == model)
            )
            if mask.any():
                return True
        return False
    except Exception:
        # Fallback if columns missing or file malformed
        try:
            df = pd.read_csv(out_csv)
            if {"ticker", "run_date", "model"}.issubset(df.columns):
                return (
                    (df["ticker"].astype(str) == ticker)
                    & (df["run_date"].astype(str) == run_date)
                    & (df["model"].astype(str) == model)
                ).any()
            return False
        except Exception:
            return False


def run_garch_for_tickers(
    tickers: List[str],
    period: str = "3y",
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
    horizon: int = 20,
    plot: bool = False,
    out_csv: Optional[str] = None,
    mean: str = "Constant",
    select_orders: bool = False,
    pmax: int = 2,
    omax: int = 1,
    qmax: int = 2,
    order_criterion: str = "aic",
    day_count: int = 252,
    agg_days: int = 0,
    summary_csv: Optional[str] = None,
    per_ticker_dir: Optional[str] = None,
) -> None:
    run_dt = datetime.now()
    run_date = run_dt.date().isoformat()
    for t in tickers:
        try:
            px = fetch_prices(t, period=period)
            rets = log_returns(px)
            # Optionally select orders via AIC/BIC over small grid
            fitted = None
            if select_orders:
                p_grid = range(1, max(1, int(pmax)) + 1)
                o_grid = range(0, max(0, int(omax)) + 1)
                q_grid = range(1, max(1, int(qmax)) + 1)
                p, o, q, fitted = select_gjr_orders(
                    rets,
                    p_grid=p_grid,
                    o_grid=o_grid,
                    q_grid=q_grid,
                    mean=mean,
                    dist=dist,
                    criterion=order_criterion,
                )
            if fitted is None:
                fitted = fit_garch(rets, p=p, o=o, q=q, mean=mean, dist=dist, show_output=False)

            # Forecast horizon covers aggregation window if requested
            f_h = max(horizon, agg_days) if agg_days and agg_days > 0 else horizon
            var, vol, vol_lo, vol_hi = forecast_volatility(
                fitted, horizon=f_h, return_ci=True, ci=0.95, n_sims=2000
            )

            ann_factor = np.sqrt(float(day_count))
            # Limit what we display to 'horizon' steps
            vol_disp = vol.iloc[:horizon]
            ann_vol_disp = (vol * ann_factor).iloc[:horizon]
            ann_vol = vol * ann_factor
            ann_lo = vol_lo * ann_factor
            ann_hi = vol_hi * ann_factor
            print(f"Ticker: {t}")
            print(f"  Next {horizon} days daily vol forecast:")
            print(vol_disp.to_string())
            print(f"  Annualized vol (sqrt(252)) forecast:")
            print(ann_vol_disp.to_string())
            # Aggregated H-day vol for maturity matching
            agg_ann_vol = agg_ann_vol_lo = agg_ann_vol_hi = None
            if agg_days and agg_days > 0:
                h = min(int(agg_days), len(var))
                if h > 0:
                    agg_var = float(var.iloc[:h].sum())
                    agg_ann_vol = float(np.sqrt((float(day_count) / h) * agg_var))
                    # Approximate bounds via per-step bounds
                    agg_var_lo = float((vol_lo.iloc[:h] ** 2).sum())
                    agg_var_hi = float((vol_hi.iloc[:h] ** 2).sum())
                    agg_ann_vol_lo = float(np.sqrt((float(day_count) / h) * agg_var_lo))
                    agg_ann_vol_hi = float(np.sqrt((float(day_count) / h) * agg_var_hi))
                    print(f"  Aggregated {h}d annualized vol: {agg_ann_vol:.6f} (95% ~ [{agg_ann_vol_lo:.6f}, {agg_ann_vol_hi:.6f}])")
            if plot:
                plot_conditional_volatility(
                    fitted, title=f"{t} GJR-GARCH({p},{o},{q}) Conditional Volatility"
                )

            # Skip appending if already logged today for this ticker/model
            already_logged = _has_run_for(out_csv, t, run_date, "GJR-GARCH")
            if out_csv and not already_logged:
                # Extract GJR-GARCH asymmetry parameters
                gamma = gamma_t = gamma_p = alpha1 = beta1 = None
                try:
                    params = getattr(fitted, "params", None)
                    tvals = getattr(fitted, "tvalues", None)
                    pvals = getattr(fitted, "pvalues", None)
                    if params is not None:
                        gamma_keys = [k for k in params.index if str(k).startswith("gamma")]
                        alpha_keys = [k for k in params.index if str(k).startswith("alpha")]
                        beta_keys = [k for k in params.index if str(k).startswith("beta")]
                        if gamma_keys:
                            gamma = float(params[gamma_keys[0]])
                            if tvals is not None:
                                gamma_t = float(tvals[gamma_keys[0]])
                            if pvals is not None:
                                gamma_p = float(pvals[gamma_keys[0]])
                        if alpha_keys:
                            alpha1 = float(params[alpha_keys[0]])
                        if beta_keys:
                            beta1 = float(params[beta_keys[0]])
                except Exception:
                    pass

                # For a +/-1% return shock in original units, variance difference is gamma * 0.01^2
                asym_var_diff_1pct = (gamma / 10000.0) if (gamma is not None) else None

                # Full parameter vectors as JSON strings
                params_json = params_t_json = params_p_json = None
                try:
                    if params is not None:
                        params_json = json.dumps({k: float(params[k]) for k in params.index})
                    if tvals is not None:
                        params_t_json = json.dumps({k: float(tvals[k]) for k in tvals.index})
                    if pvals is not None:
                        params_p_json = json.dumps({k: float(pvals[k]) for k in pvals.index})
                except Exception:
                    pass
                records = []
                for step_idx, (d_vol, a_vol, d_lo, d_hi, a_lo, a_hi) in enumerate(
                    zip(
                        vol.iloc[:horizon].values,
                        (vol * ann_factor).iloc[:horizon].values,
                        vol_lo.iloc[:horizon].values,
                        vol_hi.iloc[:horizon].values,
                        (vol_lo * ann_factor).iloc[:horizon].values,
                        (vol_hi * ann_factor).iloc[:horizon].values,
                    ),
                    start=1,
                ):
                    records.append(
                        {
                            "run_datetime": run_dt.isoformat(timespec="seconds"),
                            "run_date": run_date,
                            "ticker": t,
                            "model": "GJR-GARCH",
                            "p": p,
                            "o": o,
                            "q": q,
                            "dist": dist,
                            "horizon": horizon,
                            "step": step_idx,
                            "daily_vol": float(d_vol),
                            "annualized_vol": float(a_vol),
                            "daily_vol_lo95": float(d_lo),
                            "daily_vol_hi95": float(d_hi),
                            "annualized_vol_lo95": float(a_lo),
                            "annualized_vol_hi95": float(a_hi),
                            "gamma": gamma,
                            "gamma_t": gamma_t,
                            "gamma_p": gamma_p,
                            "alpha1": alpha1,
                            "beta1": beta1,
                            "asym_var_diff_1pct": asym_var_diff_1pct,
                            "params": params_json,
                            "params_t": params_t_json,
                            "params_p": params_p_json,
                            "best_order_p": p if select_orders else None,
                            "best_order_d": None,
                            "best_order_q": q if select_orders else None,
                            "aic": float(getattr(fitted, "aic", np.nan)),
                            "ret_forecast": None,
                            "price_forecast": None,
                            "agg_days": int(agg_days) if agg_days else None,
                            "agg_ann_vol": agg_ann_vol,
                            "agg_ann_vol_lo95": agg_ann_vol_lo,
                            "agg_ann_vol_hi95": agg_ann_vol_hi,
                        }
                    )
                _append_records(out_csv, records)
                # Also write a one-line summary row (per run) to optional files
                summary_row = {
                    "run_datetime": run_dt.isoformat(timespec="seconds"),
                    "run_date": run_date,
                    "ticker": t,
                    "model": "GJR-GARCH",
                    "p": p,
                    "o": o,
                    "q": q,
                    "mean": mean,
                    "dist": dist,
                    "day_count": int(day_count),
                    "agg_days": int(agg_days) if agg_days else None,
                    "agg_ann_vol": agg_ann_vol,
                    "agg_ann_vol_lo95": agg_ann_vol_lo,
                    "agg_ann_vol_hi95": agg_ann_vol_hi,
                    "aic": float(getattr(fitted, "aic", np.nan)),
                }
                if summary_csv and not _has_run_for(summary_csv, t, run_date, "GJR-GARCH"):
                    _append_records(summary_csv, [summary_row])
                per_t_path = _per_ticker_path(per_ticker_dir, t)
                if per_t_path and not _has_run_for(per_t_path, t, run_date, "GJR-GARCH"):
                    _append_records(per_t_path, [summary_row])
            elif out_csv and already_logged:
                print(f"[INFO] GJR-GARCH results already logged for {t} on {run_date}; skipping CSV append.")
        except Exception as exc:
            print(f"[WARN] {t}: {exc}")


def run_arima_for_tickers(
    tickers: List[str],
    period: str = "3y",
    steps: int = 5,
    verbose: bool = False,
    out_csv: Optional[str] = None,
    summary_csv: Optional[str] = None,
    per_ticker_dir: Optional[str] = None,
) -> None:
    run_dt = datetime.now()
    run_date = run_dt.date().isoformat()
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

            # Skip appending if already logged today for this ticker/model
            already_logged = _has_run_for(out_csv, t, run_date, "ARIMA")
            if out_csv and not already_logged:
                order_p, order_d, order_q = best.order
                records = []
                for step_idx in range(1, steps + 1):
                    ret_val = float(r_fc.iloc[step_idx - 1])
                    price_val = float(p_fc.iloc[step_idx - 1])
                    records.append(
                        {
                            "run_datetime": run_dt.isoformat(timespec="seconds"),
                            "run_date": run_date,
                            "ticker": t,
                            "model": "ARIMA",
                            "p": None,
                            "q": None,
                            "dist": None,
                            "horizon": steps,
                            "step": step_idx,
                            "daily_vol": None,
                            "annualized_vol": None,
                            "best_order_p": int(order_p),
                            "best_order_d": int(order_d),
                            "best_order_q": int(order_q),
                            "aic": float(best.aic),
                            "ret_forecast": ret_val,
                            "price_forecast": price_val,
                        }
                    )
                _append_records(out_csv, records)
                # Summary (one line per run) for ARIMA
                arima_summary = {
                    "run_datetime": run_dt.isoformat(timespec="seconds"),
                    "run_date": run_date,
                    "ticker": t,
                    "model": "ARIMA",
                    "best_order_p": int(order_p),
                    "best_order_d": int(order_d),
                    "best_order_q": int(order_q),
                    "aic": float(best.aic),
                    "steps": int(steps),
                    "last_ret_forecast": float(r_fc.iloc[-1]),
                    "last_price_forecast": float(p_fc.iloc[-1]),
                }
                if summary_csv and not _has_run_for(summary_csv, t, run_date, "ARIMA"):
                    _append_records(summary_csv, [arima_summary])
                per_t_path = _per_ticker_path(per_ticker_dir, t)
                if per_t_path and not _has_run_for(per_t_path, t, run_date, "ARIMA"):
                    _append_records(per_t_path, [arima_summary])
            elif out_csv and already_logged:
                print(f"[INFO] ARIMA results already logged for {t} on {run_date}; skipping CSV append.")
        except Exception as exc:
            print(f"[WARN] {t}: {exc}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run GARCH fits for tickers")
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "GOOGL"], help="Tickers list") # Default tickers
    ap.add_argument("--period", default="3y", help="History period, e.g. 1y, 3y, 5y")
    ap.add_argument("--p", type=int, default=1, help="ARCH order (p)")
    ap.add_argument("--o", type=int, default=1, help="GJR asymmetry order (o)")
    ap.add_argument("--q", type=int, default=1, help="GARCH order (q)")
    ap.add_argument("--mean", default="Constant", choices=["Zero", "Constant"], help="Mean model")
    ap.add_argument("--dist", default="t", choices=["normal", "t", "skewt", "ged"], help="Innovation distribution")
    ap.add_argument("--horizon", type=int, default=20, help="Forecast horizon (days)")
    ap.add_argument("--agg-days", type=int, default=0, help="Aggregate forward days for annualized vol (0 disables)")
    ap.add_argument("--day-count", type=int, default=252, help="Annualization day-count (e.g., 252 or 365)")
    ap.add_argument("--plot", action="store_true", help="Plot conditional volatility")
    ap.add_argument(
        "--models",
        choices=["garch", "arima", "both"],
        default="both",
        help="Which models to run",
    )
    ap.add_argument("--select-orders", action="store_true", help="Grid-search (p,o,q) by criterion")
    ap.add_argument("--order-criterion", choices=["aic", "bic"], default="aic", help="Model selection criterion")
    ap.add_argument("--pmax", type=int, default=2, help="Max p for selection")
    ap.add_argument("--omax", type=int, default=1, help="Max o for selection")
    ap.add_argument("--qmax", type=int, default=2, help="Max q for selection")
    ap.add_argument("--steps", type=int, default=5, help="ARIMA forecast steps")
    ap.add_argument("--arima-verbose", action="store_true", help="Verbose ARIMA grid search logs")
    default_csv = os.path.join(os.path.dirname(__file__), "model_results.csv")
    default_summary = os.path.join(os.path.dirname(__file__), "model_summary.csv")
    default_ticker_dir = os.path.join(os.path.dirname(__file__), "ticker_results")
    ap.add_argument("--out-csv", default=default_csv, help="Path to append detailed results CSV")
    ap.add_argument("--summary-csv", default=default_summary, help="Path to append one-line summary CSV")
    ap.add_argument("--per-ticker-dir", default=default_ticker_dir, help="Directory for per-ticker summary CSVs")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = parse_args(argv or [])

    if ns.models in ("garch", "both"):
        run_garch_for_tickers(
            tickers=ns.tickers,
            period=ns.period,
            p=ns.p,
            o=ns.o,
            q=ns.q,
            dist=ns.dist,
            horizon=ns.horizon,
            plot=ns.plot,
            out_csv=ns.out_csv,
            mean=ns.mean,
            select_orders=ns.select_orders,
            pmax=ns.pmax,
            omax=ns.omax,
            qmax=ns.qmax,
            order_criterion=ns.order_criterion,
            day_count=ns.day_count,
            agg_days=ns.agg_days,
            summary_csv=ns.summary_csv,
            per_ticker_dir=ns.per_ticker_dir,
        )

    if ns.models in ("arima", "both"):
        run_arima_for_tickers(
            tickers=ns.tickers,
            period=ns.period,
            steps=ns.steps,
            verbose=ns.arima_verbose,
            out_csv=ns.out_csv,
            summary_csv=ns.summary_csv,
            per_ticker_dir=ns.per_ticker_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
