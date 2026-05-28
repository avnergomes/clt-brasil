"""
CLT-BRASIL — self-contained interactive HTML report builder.

Consumes forecast results from clt.forecast_series and renders a single
professional, offline HTML file (plotly.js embedded inline once).
"""
from __future__ import annotations
import os, datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

from clt import UF_NOME, REGIOES, ROOT

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    "SARIMA": "#2563eb",
    "ETS (Holt-Winters)": "#16a34a",
    "Seasonal Naive": "#9333ea",
    "hist": "#0f172a",
}
CI_FILL = {
    "SARIMA": "rgba(37,99,235,0.15)",
    "ETS (Holt-Winters)": "rgba(22,163,74,0.15)",
    "Seasonal Naive": "rgba(147,51,234,0.12)",
}


def _fmt(v, dec=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.{dec}f}".replace(",", " ")


def _layout(title, h=460):
    return dict(
        title=dict(text=title, font=dict(size=16, color="#0f172a")),
        template="plotly_white", height=h, hovermode="x unified",
        margin=dict(l=60, r=24, t=56, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=True, gridcolor="#eef2f7"),
        yaxis=dict(showgrid=True, gridcolor="#eef2f7", zeroline=True,
                   zerolinecolor="#cbd5e1", title="Saldo de empregos (CLT)"),
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color="#334155"),
    )


def fig_full(name, res) -> go.Figure:
    """History + all model forecasts with CI bands."""
    fig = go.Figure()
    hist = res["history"]
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Histórico",
                             mode="lines", line=dict(color=PALETTE["hist"], width=2.2)))
    for mname, m in res["models"].items():
        if "mean" not in m:
            continue
        c = PALETTE.get(mname, "#888")
        fig.add_trace(go.Scatter(
            x=list(m["upper"].index) + list(m["lower"].index[::-1]),
            y=list(m["upper"].values) + list(m["lower"].values[::-1]),
            fill="toself", fillcolor=CI_FILL.get(mname, "rgba(0,0,0,0.08)"),
            line=dict(width=0), hoverinfo="skip", showlegend=False,
            name=f"{mname} IC95%"))
        fig.add_trace(go.Scatter(x=m["mean"].index, y=m["mean"].values, name=mname,
                                 mode="lines", line=dict(color=c, width=2, dash="solid")))
    fig.update_layout(**_layout(f"{name} — saldo mensal e previsões até jun/2027 (IC 95%)", 480))
    return fig


def fig_explorer(uf_results: dict) -> go.Figure:
    """One figure with a dropdown to switch UF (history + best model + CI)."""
    fig = go.Figure()
    ufs = list(uf_results.keys())
    traces_per = 3  # CI band, mean, history
    for i, uf in enumerate(ufs):
        res = uf_results[uf]
        best = res.get("best")
        vis = (i == 0)
        hist = res["history"]
        m = res["models"].get(best, {})
        if "mean" in m:
            fig.add_trace(go.Scatter(
                x=list(m["upper"].index) + list(m["lower"].index[::-1]),
                y=list(m["upper"].values) + list(m["lower"].values[::-1]),
                fill="toself", fillcolor="rgba(37,99,235,0.15)", line=dict(width=0),
                hoverinfo="skip", showlegend=False, visible=vis))
            fig.add_trace(go.Scatter(x=m["mean"].index, y=m["mean"].values,
                                     name=f"Previsão ({best})", line=dict(color="#2563eb", width=2),
                                     visible=vis))
        else:
            fig.add_trace(go.Scatter(x=[], y=[], visible=vis, showlegend=False))
            fig.add_trace(go.Scatter(x=[], y=[], visible=vis, showlegend=False))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Histórico",
                                 line=dict(color="#0f172a", width=2), visible=vis))
    buttons = []
    n = len(ufs)
    for i, uf in enumerate(ufs):
        mask = [False] * (n * traces_per)
        for k in range(traces_per):
            mask[i * traces_per + k] = True
        label = f"{uf} — {UF_NOME[uf]}" if uf in UF_NOME else uf
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": mask},
                                  {"title": f"{label} — saldo CLT e previsão até jun/2027 (IC 95%)"}]))
    lay = _layout(f"{ufs[0]} — {UF_NOME.get(ufs[0], ufs[0])} — saldo CLT e previsão até jun/2027 (IC 95%)", 480)
    lay["updatemenus"] = [dict(buttons=buttons, direction="down", showactive=True,
                               x=1.0, xanchor="right", y=1.16, yanchor="top",
                               bgcolor="#ffffff", bordercolor="#cbd5e1")]
    fig.update_layout(**lay)
    return fig


def fig_regional(panel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#0891b2"]
    for c, (reg, ufs) in zip(colors, REGIOES.items()):
        sub = panel[panel["uf"].isin(ufs)].groupby("date")["saldo"].sum()
        roll = sub.rolling(12).sum()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name=reg,
                                 line=dict(color=c, width=2)))
    lay = _layout("Saldo acumulado em 12 meses por região", 420)
    lay["yaxis"]["title"] = "Saldo acumulado 12m"
    fig.update_layout(**lay)
    return fig


def fig_heatmap(panel: pd.DataFrame) -> go.Figure:
    p = panel.copy()
    p["mes"] = p["date"].dt.month
    order = [u for r in REGIOES.values() for u in r]
    piv = p.groupby(["uf", "mes"])["saldo"].mean().unstack().reindex(order)
    meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    fig = go.Figure(go.Heatmap(z=piv.values, x=meses, y=piv.index,
                               colorscale="RdYlGn", zmid=0,
                               colorbar=dict(title="Saldo médio")))
    lay = _layout("Sazonalidade — saldo médio por mês e UF", 640)
    lay["yaxis"]["title"] = "UF"
    lay["hovermode"] = "closest"
    fig.update_layout(**lay)
    return fig


def _kpi_cards(nat, uf_results) -> str:
    hist = nat["history"]
    last = hist.iloc[-1]; last_date = hist.index[-1].strftime("%b/%Y")
    acc12 = hist.iloc[-12:].sum()
    best = nat["best"]
    fc_end = nat["models"][best]["mean"].iloc[-1] if best else np.nan
    lo_end = nat["models"][best]["lower"].iloc[-1] if best else np.nan
    hi_end = nat["models"][best]["upper"].iloc[-1] if best else np.nan
    # best/worst UF by accumulated 12m
    acc_uf = {uf: r["history"].iloc[-12:].sum() for uf, r in uf_results.items()}
    top = max(acc_uf, key=acc_uf.get); bot = min(acc_uf, key=acc_uf.get)
    cards = [
        ("Saldo Brasil — último mês", _fmt(last), last_date, "blue"),
        ("Saldo acumulado 12 meses", _fmt(acc12), "Brasil", "green"),
        ("Previsão jun/2027 (Brasil)", _fmt(fc_end), f"IC95%: {_fmt(lo_end)} a {_fmt(hi_end)} · {best}", "violet"),
        ("Maior geração 12m (UF)", f"{top}", f"{_fmt(acc_uf[top])} vagas", "green"),
        ("Menor geração 12m (UF)", f"{bot}", f"{_fmt(acc_uf[bot])} vagas", "red"),
    ]
    html = '<div class="kpi-grid">'
    for title, val, sub, tone in cards:
        html += f'''<div class="kpi card tone-{tone}">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-sub">{sub}</div></div>'''
    return html + "</div>"


def _metrics_table(nat) -> str:
    rows = ""
    for mname, m in nat["models"].items():
        if "metrics" not in m:
            continue
        met = m["metrics"]
        order = ""
        if "order" in m:
            order = f"SARIMA{m['order']}x{m['seasonal_order']}"
        elif "aic" in m:
            order = "add. trend+season"
        else:
            order = "baseline m=12"
        star = " ★" if mname == nat["best"] else ""
        rows += f'''<tr class="{ 'best-row' if mname==nat['best'] else ''}">
          <td>{mname}{star}</td><td class="mono">{order}</td>
          <td class="num">{_fmt(met['RMSE'])}</td><td class="num">{_fmt(met['MAE'])}</td>
          <td class="num">{_fmt(met['MASE'],2)}</td><td class="num">{_fmt(met['sMAPE'],1)}%</td>
          <td class="num">{_fmt(m.get('aic'),1)}</td></tr>'''
    return f'''<table class="tbl"><thead><tr>
      <th>Modelo</th><th>Especificação</th><th>RMSE</th><th>MAE</th>
      <th>MASE</th><th>sMAPE</th><th>AIC</th></tr></thead><tbody>{rows}</tbody></table>'''


def _uf_table(uf_results, diagnostics) -> str:
    order = [u for r in REGIOES.values() for u in r]
    rows = ""
    for uf in order:
        res = uf_results[uf]; best = res.get("best")
        m = res["models"].get(best, {})
        fc = m["mean"].iloc[-1] if "mean" in m else np.nan
        lo = m["lower"].iloc[-1] if "lower" in m else np.nan
        hi = m["upper"].iloc[-1] if "upper" in m else np.nan
        met = m.get("metrics", {})
        dg = diagnostics.get(uf, {})
        acc = res["history"].iloc[-12:].sum()
        rows += f'''<tr><td class="mono">{uf}</td><td>{UF_NOME[uf]}</td>
          <td class="num">{_fmt(acc)}</td><td>{best or '—'}</td>
          <td class="num">{_fmt(fc)}</td><td class="num small">{_fmt(lo)} a {_fmt(hi)}</td>
          <td class="num">{_fmt(met.get('RMSE'))}</td>
          <td class="num">{_fmt(met.get('MASE'),2)}</td>
          <td class="num">{_fmt(dg.get('seasonal_strength'),2)}</td>
          <td>{'sim' if dg.get('heteroscedastic') else 'não'}</td></tr>'''
    return f'''<table class="tbl sortable"><thead><tr>
      <th>UF</th><th>Estado</th><th>Saldo 12m</th><th>Melhor modelo</th>
      <th>Prev. jun/27</th><th>IC 95%</th><th>RMSE</th><th>MASE</th>
      <th>Força sazonal</th><th>Heterosced.</th></tr></thead><tbody>{rows}</tbody></table>'''


CSS = """
:root{--bg:#f6f8fb;--card:#fff;--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;--blue:#2563eb;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,'Segoe UI',system-ui,sans-serif;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 80px}
header.hero{background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff;padding:48px 0 40px;margin-bottom:28px}
.hero .wrap{padding-bottom:0}
.hero h1{font-size:30px;margin:0 0 6px;font-weight:700;letter-spacing:-.5px}
.hero p{margin:4px 0;color:#cbd5e1;font-size:14px}
.badge{display:inline-block;background:rgba(255,255,255,.12);padding:4px 10px;border-radius:999px;
font-size:12px;margin-top:10px;color:#e2e8f0}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:24px;margin:22px 0;box-shadow:0 1px 3px rgba(15,23,42,.04)}
h2{font-size:20px;margin:0 0 4px;font-weight:650}
h2 .num{color:var(--blue)}
.lead{color:var(--mut);font-size:14px;margin:0 0 18px;max-width:80ch}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:6px 0 4px}
.kpi{padding:16px 18px;border-radius:12px;border:1px solid var(--line);background:#fff}
.kpi-title{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.kpi-value{font-size:26px;font-weight:700;margin:6px 0 2px}
.kpi-sub{font-size:12px;color:var(--mut)}
.tone-blue{border-top:3px solid #2563eb}.tone-green{border-top:3px solid #16a34a}
.tone-violet{border-top:3px solid #9333ea}.tone-red{border-top:3px solid #dc2626}
.tbl{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
.tbl th,.tbl td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}
.tbl th{background:#f1f5f9;font-weight:600;color:#334155;cursor:pointer;user-select:none}
.tbl td.num{text-align:right;font-variant-numeric:tabular-nums}
.tbl td.mono,.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.tbl td.small{font-size:11px;color:var(--mut)}
.best-row{background:#eff6ff}
.note{font-size:12.5px;color:var(--mut);margin-top:12px}
footer{color:var(--mut);font-size:12px;text-align:center;padding:30px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:860px){.grid2{grid-template-columns:1fr}}
"""

SORT_JS = """
document.querySelectorAll('table.sortable th').forEach((th,idx)=>{
 th.addEventListener('click',()=>{
  const tb=th.closest('table').querySelector('tbody');
  const rows=[...tb.querySelectorAll('tr')];
  const asc=th._asc=!th._asc;
  rows.sort((a,b)=>{
   const x=a.children[idx].innerText.replace(/[^0-9.\\-]/g,'');
   const y=b.children[idx].innerText.replace(/[^0-9.\\-]/g,'');
   const nx=parseFloat(x),ny=parseFloat(y);
   if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;
   return asc?a.children[idx].innerText.localeCompare(b.children[idx].innerText)
             :b.children[idx].innerText.localeCompare(a.children[idx].innerText);
  });
  rows.forEach(r=>tb.appendChild(r));
 });
});
"""


def build_report(nat, uf_results, panel, diagnostics, out_path=None):
    out_path = out_path or os.path.join(OUT, "CLT-BRASIL_relatorio.html")
    plotlyjs = pyo.get_plotlyjs()

    def div(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": True, "displaylogo": False,
                                   "responsive": True})

    d_nat = div(fig_full("Brasil", nat))
    d_exp = div(fig_explorer(uf_results))
    d_reg = div(fig_regional(panel))
    d_heat = div(fig_heatmap(panel))

    hist = nat["history"]
    span = f"{hist.index[0].strftime('%b/%Y')} – {hist.index[-1].strftime('%b/%Y')}"
    nat_dg = diagnostics.get("Brasil", {})
    gen = dt.datetime.now().strftime("%d/%m/%Y %H:%M")

    diag_txt = (
        f"Série mensal de {len(hist)} observações ({span}). "
        f"ADF p-valor = {_fmt(nat_dg.get('adf_p'),3)} "
        f"({'estacionária' if nat_dg.get('adf_stationary') else 'não estacionária'}); "
        f"KPSS p-valor = {_fmt(nat_dg.get('kpss_p'),3)}; "
        f"força sazonal = {_fmt(nat_dg.get('seasonal_strength'),2)}; "
        f"teste ARCH-LM p-valor = {_fmt(nat_dg.get('arch_p'),3)} "
        f"({'há' if nat_dg.get('heteroscedastic') else 'sem evidência de'} heterocedasticidade condicional)."
    )

    html = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLT-BRASIL — Emprego Formal (Novo CAGED/MTE)</title>
<style>{CSS}</style>
<script>{plotlyjs}</script></head><body>
<header class="hero"><div class="wrap">
  <h1>CLT-BRASIL · Emprego Formal no Brasil</h1>
  <p>Análise de série temporal e previsão do saldo de empregos celetistas (Novo CAGED — Ministério do Trabalho e Emprego)</p>
  <p>Cobertura: {span} · 27 UFs + agregado nacional · Horizonte de previsão: até jun/2027</p>
  <span class="badge">Fonte: PDET/MTE — microdados Novo CAGED · Gerado em {gen}</span>
</div></header>
<div class="wrap">

<section>
  <h2>Panorama</h2>
  <p class="lead">Indicadores-síntese do mercado de trabalho formal. O saldo é a diferença
  entre admissões e desligamentos com carteira assinada (CLT) declarados ao Novo CAGED.</p>
  {_kpi_cards(nat, uf_results)}
</section>

<section>
  <h2>Brasil — série histórica e previsões</h2>
  <p class="lead">Saldo mensal observado e projeções de três modelos até junho de 2027,
  com intervalos de confiança de 95%. {diag_txt}</p>
  {d_nat}
  <h3 style="margin:22px 0 2px;font-size:15px">Desempenho dos modelos (validação 12 meses)</h3>
  <p class="note">Métricas calculadas em backtest com retenção dos últimos 12 meses. ★ = modelo selecionado (menor RMSE).</p>
  {_metrics_table(nat)}
</section>

<section>
  <h2>Explorador por estado</h2>
  <p class="lead">Selecione a UF para visualizar a série histórica e a previsão do melhor
  modelo (menor RMSE em validação) com intervalo de confiança de 95%.</p>
  {d_exp}
</section>

<section>
  <div class="grid2">
    <div><h2>Regiões</h2><p class="lead">Saldo acumulado em 12 meses por região.</p>{d_reg}</div>
    <div><h2>Sazonalidade</h2><p class="lead">Saldo médio por mês do ano e UF — verde indica geração líquida; vermelho, perdas.</p></div>
  </div>
  {d_heat}
</section>

<section>
  <h2>Síntese por estado</h2>
  <p class="lead">Previsão para jun/2027, intervalo de confiança, métrica de erro e diagnósticos
  por UF. Clique nos cabeçalhos para ordenar.</p>
  {_uf_table(uf_results, diagnostics)}
  <p class="note">Força sazonal (0–1, Hyndman): proporção da variância explicada pelo componente sazonal.
  Heterocedasticidade: teste ARCH-LM (5%) sobre a série diferenciada.</p>
</section>

<footer>
  CLT-BRASIL · Elaborado a partir dos microdados do Novo CAGED (PDET/Ministério do Trabalho e Emprego).<br>
  Metodologia: saldo = Σ(saldomovimentação) por competência; modelos SARIMA, ETS (Holt-Winters) e Seasonal Naive.<br>
  Relatório autocontido — gerado em {gen}.
</footer>
</div>
<script>{SORT_JS}</script>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
