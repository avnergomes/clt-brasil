"""
Headless forecast runner: fit every model on Brasil + 27 UFs and (re)write the
artifacts the report consumes. Mirrors the modeling cells of the analysis
notebook so the CSVs can be regenerated without opening Jupyter.

Models: SARIMA, ETS (Holt-Winters), Seasonal Naive, Random Forest, LightGBM
(defined in src/clt.py). Best model per series = lowest 12-month backtest RMSE.

Outputs:
  data/processed/previsoes.csv        serie, modelo, data, previsao, ic_inf, ic_sup
  data/processed/metricas_por_uf.csv  serie, melhor_modelo, prev_jun2027, IC, RMSE,
                                      MASE, diagnostics (ADF/KPSS/ARCH/seasonality)
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import clt


def main():
    panel = clt.load_panel()
    series_all = clt.all_series(panel)
    print(f"{len(series_all)} series; horizon -> {clt.HORIZON_END}", flush=True)

    results, diagnostics = {}, {}
    t0 = time.time()
    for i, (nome, s) in enumerate(series_all.items(), 1):
        ts = time.time()
        results[nome] = clt.forecast_series(s)
        adf = clt.adf_test(s); kp = clt.kpss_test(s); ar = clt.arch_test(s)
        diagnostics[nome] = {
            "adf_p": adf["pvalue"], "adf_stationary": adf["stationary"],
            "kpss_p": kp["pvalue"], "arch_p": ar["pvalue"],
            "heteroscedastic": ar["heteroscedastic"],
            "seasonal_strength": clt.seasonal_strength(s),
        }
        print(f"[{i:>2}/{len(series_all)}] {nome:<7} best={results[nome]['best']:<16} "
              f"({time.time()-ts:.0f}s)", flush=True)
    print(f"all fitted in {time.time()-t0:.0f}s", flush=True)

    # forecasts (long format) — every model that produced a mean
    linhas = []
    for nome, res in results.items():
        for mname, m in res["models"].items():
            if "mean" not in m:
                continue
            for dt_, mu in m["mean"].items():
                linhas.append({"serie": nome, "modelo": mname, "data": dt_.date(),
                               "previsao": round(mu), "ic_inf": round(m["lower"][dt_]),
                               "ic_sup": round(m["upper"][dt_])})
    os.makedirs(clt.PROC, exist_ok=True)
    pd.DataFrame(linhas).to_csv(os.path.join(clt.PROC, "previsoes.csv"), index=False)

    # best-model metrics + diagnostics per series
    met_rows = []
    for nome, res in results.items():
        b = res["best"]; m = res["models"].get(b, {})
        met = m.get("metrics", {})
        met_rows.append({
            "serie": nome, "melhor_modelo": b,
            "prev_jun2027": round(m["mean"].iloc[-1]) if "mean" in m else None,
            "ic_inf": round(m["lower"].iloc[-1]) if "lower" in m else None,
            "ic_sup": round(m["upper"].iloc[-1]) if "upper" in m else None,
            "RMSE": round(met.get("RMSE", float("nan"))),
            "MASE": round(met.get("MASE", float("nan")), 3),
            **{k: round(v, 3) if isinstance(v, float) else v
               for k, v in diagnostics[nome].items()}})
    pd.DataFrame(met_rows).to_csv(os.path.join(clt.PROC, "metricas_por_uf.csv"), index=False)

    # quick summary of which model won where
    win = pd.Series([r["best"] for r in results.values()]).value_counts()
    print("\nbest-model wins:\n" + win.to_string(), flush=True)
    print(f"\nsaved -> {clt.PROC}", flush=True)


if __name__ == "__main__":
    main()
