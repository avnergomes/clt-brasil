"""
CLT-BRASIL — shared analysis engine for the Novo CAGED (MTE) formal-employment series.

Provides:
  * data loading (national + per-UF panel)
  * stationarity / seasonality / heteroscedasticity diagnostics
  * a multi-model forecasting engine (SARIMA, Holt-Winters/ETS, Seasonal Naive)
    with backtesting metrics and forecasts-to-horizon with 95% confidence intervals.

The "saldo" (net balance) = admissions - dismissals of formal (CLT) jobs.
It can be negative, so multiplicative/log transforms are not applied to the level;
heteroscedasticity is handled by SARIMA differencing and assessed explicitly.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

REGIOES = {
    "Norte": ["RO", "AC", "AM", "RR", "PA", "AP", "TO"],
    "Nordeste": ["MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"],
    "Sudeste": ["MG", "ES", "RJ", "SP"],
    "Sul": ["PR", "SC", "RS"],
    "Centro-Oeste": ["MS", "MT", "GO", "DF"],
}
UF_NOME = {
    "RO": "Rondônia", "AC": "Acre", "AM": "Amazonas", "RR": "Roraima", "PA": "Pará",
    "AP": "Amapá", "TO": "Tocantins", "MA": "Maranhão", "PI": "Piauí", "CE": "Ceará",
    "RN": "Rio Grande do Norte", "PB": "Paraíba", "PE": "Pernambuco", "AL": "Alagoas",
    "SE": "Sergipe", "BA": "Bahia", "MG": "Minas Gerais", "ES": "Espírito Santo",
    "RJ": "Rio de Janeiro", "SP": "São Paulo", "PR": "Paraná", "SC": "Santa Catarina",
    "RS": "Rio Grande do Sul", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso",
    "GO": "Goiás", "DF": "Distrito Federal",
}
HORIZON_END = "2027-06-01"


# ----------------------------------------------------------------------------- data
def load_panel() -> pd.DataFrame:
    """Long panel: date, uf_code, uf, regiao, saldo (only valid 27 UFs)."""
    df = pd.read_csv(os.path.join(RAW, "caged_uf_panel.csv"), parse_dates=["date"])
    df = df[df["uf"].isin(UF_NOME)].copy()
    return df.sort_values(["uf", "date"]).reset_index(drop=True)


def load_brasil() -> pd.Series:
    df = pd.read_csv(os.path.join(RAW, "caged_brasil.csv"), parse_dates=["date"])
    s = df.set_index("date")["saldo"].asfreq("MS")
    s.name = "Brasil"
    return s


def uf_series(panel: pd.DataFrame, uf: str) -> pd.Series:
    s = panel[panel["uf"] == uf].set_index("date")["saldo"].asfreq("MS")
    s.name = uf
    return s


def all_series(panel: pd.DataFrame) -> dict[str, pd.Series]:
    out = {"Brasil": load_brasil()}
    for uf in UF_NOME:
        out[uf] = uf_series(panel, uf)
    return out


# ------------------------------------------------------------------------ diagnostics
def adf_test(s: pd.Series) -> dict:
    from statsmodels.tsa.stattools import adfuller
    r = adfuller(s.dropna(), autolag="AIC")
    return {"stat": r[0], "pvalue": r[1], "stationary": r[1] < 0.05}


def kpss_test(s: pd.Series) -> dict:
    from statsmodels.tsa.stattools import kpss
    stat, p, *_ = kpss(s.dropna(), regression="c", nlags="auto")
    return {"stat": stat, "pvalue": p, "stationary": p > 0.05}


def arch_test(s: pd.Series, lags: int = 12) -> dict:
    """Engle ARCH-LM test for conditional heteroscedasticity on residual fluctuation."""
    from statsmodels.stats.diagnostic import het_arch
    x = s.dropna().diff().dropna()
    stat, p, _, _ = het_arch(x, nlags=lags)
    return {"stat": stat, "pvalue": p, "heteroscedastic": p < 0.05}


def ljung_box(resid: pd.Series, lags: int = 12) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    r = acorr_ljungbox(resid.dropna(), lags=[lags], return_df=True)
    p = float(r["lb_pvalue"].iloc[0])
    return {"stat": float(r["lb_stat"].iloc[0]), "pvalue": p, "white_noise": p > 0.05}


def stl_decompose(s: pd.Series):
    from statsmodels.tsa.seasonal import STL
    return STL(s.dropna(), period=12, robust=True).fit()


def seasonal_strength(s: pd.Series) -> float:
    """Hyndman seasonal strength in [0,1]."""
    res = stl_decompose(s)
    var_r = np.var(res.resid)
    var_sr = np.var(res.seasonal + res.resid)
    return float(max(0.0, 1 - var_r / var_sr)) if var_sr > 0 else 0.0


# -------------------------------------------------------------------------- modeling
def _metrics(y_true, y_pred, y_train) -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    # MASE vs seasonal-naive in-sample (period=12)
    yt = np.asarray(y_train, float)
    d = np.abs(yt[12:] - yt[:-12])
    scale = np.mean(d) if len(d) and np.mean(d) > 0 else np.nan
    mase = float(np.mean(np.abs(err)) / scale) if scale and not np.isnan(scale) else np.nan
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    smape = float(np.nanmean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MASE": mase, "sMAPE": smape}


def _future_index(s: pd.Series, end=HORIZON_END) -> pd.DatetimeIndex:
    start = s.index[-1] + pd.offsets.MonthBegin(1)
    return pd.date_range(start, end, freq="MS")


def fit_sarima(s: pd.Series, steps: int):
    """auto_arima seasonal SARIMA; returns (mean, lower, upper, aic, order, sorder, resid)."""
    from pmdarima import auto_arima
    m = auto_arima(s, seasonal=True, m=12, stepwise=True, suppress_warnings=True,
                   error_action="ignore", max_p=3, max_q=3, max_P=2, max_Q=2,
                   d=None, D=1, information_criterion="aic")
    fc, ci = m.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
    resid = pd.Series(m.resid(), index=s.index[-len(m.resid()):])
    return (np.asarray(fc), ci[:, 0], ci[:, 1], float(m.aic()),
            m.order, m.seasonal_order, resid)


def fit_ets(s: pd.Series, steps: int):
    """Holt-Winters additive trend+season with 95% CI via simulation."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=12,
                                 initialization_method="estimated").fit()
    fc = model.forecast(steps)
    sims = model.simulate(steps, repetitions=1000, error="add")
    lo = sims.quantile(0.025, axis=1).values
    hi = sims.quantile(0.975, axis=1).values
    return np.asarray(fc), lo, hi, float(model.aic), model.resid


def fit_snaive(s: pd.Series, steps: int):
    """Seasonal naive baseline: repeat last year; CI from in-sample seasonal errors."""
    vals = s.values
    last_year = vals[-12:]
    fc = np.array([last_year[i % 12] for i in range(steps)])
    err = vals[12:] - vals[:-12]
    sd = np.std(err)
    z = 1.95996
    lo = fc - z * sd
    hi = fc + z * sd
    return fc, lo, hi


def backtest(s: pd.Series, fitter, h: int = 12) -> dict:
    """Hold out last h months; refit; score."""
    train, test = s.iloc[:-h], s.iloc[-h:]
    try:
        out = fitter(train, h)
        pred = out[0]
        return _metrics(test.values, pred, train.values)
    except Exception:
        return {"RMSE": np.nan, "MAE": np.nan, "MASE": np.nan, "sMAPE": np.nan}


def forecast_series(s: pd.Series, end=HORIZON_END) -> dict:
    """
    Fit SARIMA, ETS, Seasonal-Naive on a series; backtest (12m holdout) and
    forecast to `end` with 95% CI. Returns dict with per-model results + best model.
    """
    s = s.dropna()
    fidx = _future_index(s, end)
    steps = len(fidx)
    results = {"index": fidx, "history": s, "models": {}}

    fitters = {
        "SARIMA": fit_sarima,
        "ETS (Holt-Winters)": fit_ets,
        "Seasonal Naive": fit_snaive,
    }
    for name, fitter in fitters.items():
        try:
            out = fitter(s, steps)
            mean, lo, hi = out[0], out[1], out[2]
            bt = backtest(s, fitter, h=12)
            extra = {}
            if name == "SARIMA":
                extra = {"order": out[4], "seasonal_order": out[5], "aic": out[3]}
            elif name == "ETS (Holt-Winters)":
                extra = {"aic": out[3]}
            results["models"][name] = {
                "mean": pd.Series(mean, index=fidx),
                "lower": pd.Series(lo, index=fidx),
                "upper": pd.Series(hi, index=fidx),
                "metrics": bt, **extra,
            }
        except Exception as e:
            results["models"][name] = {"error": str(e)}

    scored = {k: v["metrics"]["RMSE"] for k, v in results["models"].items()
              if "metrics" in v and not np.isnan(v["metrics"]["RMSE"])}
    results["best"] = min(scored, key=scored.get) if scored else None
    return results
