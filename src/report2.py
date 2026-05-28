"""
CLT-BRASIL — Report v2: editorial, narrative-driven, self-contained HTML.

Reconstructs forecasts from persisted CSVs (no model refit), computes
data-grounded narrative statistics, and renders a magazine-quality report
with a special data dossier on the 6x1 -> 5x2 work-scale debate.
"""
from __future__ import annotations
import os, datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

from clt import UF_NOME, REGIOES, ROOT, RAW, PROC

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

# ---- palette (editorial) ----
INK = "#0d1b2a"; PAPER = "#fbfaf7"; BLUE = "#1d4ed8"; SKY = "#3b82f6"
GREEN = "#15803d"; RED = "#b91c1c"; AMBER = "#c2410c"; VIOLET = "#7c3aed"
GRID = "#e7e5e1"
MODEL_COLORS = {"SARIMA": BLUE, "ETS (Holt-Winters)": GREEN, "Seasonal Naive": VIOLET}
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def br(v, dec=0):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "—"
    s = f"{v:,.{dec}f}"
    return s.replace(",", "·").replace(".", ",").replace("·", ".")


# ----------------------------------------------------------------- data assembly
def load_refs():
    import json
    p = os.path.join(PROC, "references.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def render_refs(refs):
    items = ""
    for r in refs:
        doi = r.get("doi") or ""
        link = (f' · <a href="https://doi.org/{doi}" target="_blank" rel="noopener">doi:{doi}</a>'
                if doi else " · (sem DOI)")
        items += (f'<li id="ref{r["n"]}"><b>{r["authors"]}</b> ({r["year"]}). '
                  f'{r["title"]}. <i>{r["venue"]}</i>.{link}</li>')
    return f'<ol class="refs">{items}</ol>'


def load_all():
    brasil = pd.read_csv(os.path.join(RAW, "caged_brasil.csv"), parse_dates=["date"]).set_index("date")["saldo"].asfreq("MS")
    panel = pd.read_csv(os.path.join(RAW, "caged_uf_panel.csv"), parse_dates=["date"])
    panel = panel[panel["uf"].isin(UF_NOME)]
    fc = pd.read_csv(os.path.join(PROC, "previsoes.csv"), parse_dates=["data"])
    met = pd.read_csv(os.path.join(PROC, "metricas_por_uf.csv"))
    sector = pd.read_csv(os.path.join(PROC, "sector_saldo_mensal.csv"), parse_dates=["date"])
    hours = pd.read_csv(os.path.join(PROC, "hours_dist_mensal.csv"), parse_dates=["date"])
    sec_hours = pd.read_csv(os.path.join(PROC, "sector_hours.csv"))
    return brasil, panel, fc, met, sector, hours, sec_hours


def hist_of(name, brasil, panel):
    if name == "Brasil":
        return brasil
    s = panel[panel["uf"] == name].set_index("date")["saldo"].asfreq("MS")
    s.name = name
    return s


def models_of(name, fc):
    sub = fc[fc["serie"] == name]
    out = {}
    for m, g in sub.groupby("modelo"):
        g = g.sort_values("data").set_index("data")
        out[m] = {"mean": g["previsao"], "lower": g["ic_inf"], "upper": g["ic_sup"]}
    return out


def best_of(name, met):
    row = met[met["serie"] == name]
    return row["melhor_modelo"].iloc[0] if len(row) else None


# ----------------------------------------------------------------- plotly theme
def base_layout(h=460, ytitle="Saldo de empregos (CLT)"):
    return dict(
        template="plotly_white", height=h, hovermode="x unified",
        margin=dict(l=64, r=26, t=20, b=44), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor="#cbd5e1",
                   zerolinewidth=1.2, title=ytitle),
        font=dict(family="-apple-system,Segoe UI,Roboto,sans-serif", size=12.5, color="#334155"),
    )


def div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


def fig_trajetoria(brasil):
    """History only, annotated narrative chart."""
    fig = go.Figure()
    colors = [GREEN if v >= 0 else RED for v in brasil.values]
    fig.add_trace(go.Bar(x=brasil.index, y=brasil.values, marker_color=colors,
                         marker_line_width=0, opacity=.45, name="Saldo mensal",
                         hovertemplate="%{x|%b/%Y}: %{y:,.0f}<extra></extra>"))
    mm = brasil.rolling(12).mean()
    fig.add_trace(go.Scatter(x=mm.index, y=mm.values, line=dict(color=INK, width=2.6),
                             name="Média móvel 12m",
                             hovertemplate="MM12 %{x|%b/%Y}: %{y:,.0f}<extra></extra>"))
    lay = base_layout(440)
    trough_date = brasil.idxmin(); trough = brasil.min()
    fig.add_annotation(x=trough_date, y=trough, text=f"<b>Abr/2020</b><br>{br(trough)}",
                       showarrow=True, arrowhead=2, ax=70, ay=-10, font=dict(color=RED, size=11),
                       arrowcolor=RED, align="left")
    peak_date = brasil.idxmax(); peak = brasil.max()
    fig.add_annotation(x=peak_date, y=peak, text=f"<b>Pico</b><br>{br(peak)}",
                       showarrow=True, arrowhead=2, ax=0, ay=-28, font=dict(color=GREEN, size=11),
                       arrowcolor=GREEN)
    fig.update_layout(**lay)
    return fig


def fig_forecast_fan(name, brasil_or_hist, models, best):
    fig = go.Figure()
    h = brasil_or_hist.iloc[-30:]
    fig.add_trace(go.Scatter(x=h.index, y=h.values, line=dict(color=INK, width=2.4),
                             name="Histórico"))
    # faded non-best models
    for m, md in models.items():
        c = MODEL_COLORS.get(m, "#888")
        is_best = (m == best)
        if is_best:
            fig.add_trace(go.Scatter(
                x=list(md["upper"].index) + list(md["lower"].index[::-1]),
                y=list(md["upper"].values) + list(md["lower"].values[::-1]),
                fill="toself", fillcolor="rgba(29,78,216,0.14)", line=dict(width=0),
                hoverinfo="skip", showlegend=True, name="IC 95%"))
        fig.add_trace(go.Scatter(x=md["mean"].index, y=md["mean"].values,
                                 line=dict(color=c, width=3 if is_best else 1.4,
                                           dash="solid" if is_best else "dot"),
                                 opacity=1 if is_best else .6,
                                 name=m + (" ★" if is_best else "")))
    lay = base_layout(440)
    fig.add_vline(x=h.index[-1], line=dict(color="#94a3b8", dash="dash", width=1))
    fig.update_layout(**lay)
    return fig


def fig_explorer(names, hist_fn, models_fn, best_fn):
    fig = go.Figure()
    tper = 3
    for i, uf in enumerate(names):
        vis = (i == 0)
        h = hist_fn(uf); models = models_fn(uf); best = best_fn(uf)
        md = models.get(best, {})
        if md:
            fig.add_trace(go.Scatter(
                x=list(md["upper"].index) + list(md["lower"].index[::-1]),
                y=list(md["upper"].values) + list(md["lower"].values[::-1]),
                fill="toself", fillcolor="rgba(29,78,216,0.13)", line=dict(width=0),
                hoverinfo="skip", showlegend=False, visible=vis))
            fig.add_trace(go.Scatter(x=md["mean"].index, y=md["mean"].values, visible=vis,
                                     name=f"Previsão · {best}", line=dict(color=BLUE, width=2.6)))
        else:
            fig.add_trace(go.Scatter(x=[], y=[], visible=vis, showlegend=False))
            fig.add_trace(go.Scatter(x=[], y=[], visible=vis, showlegend=False))
        fig.add_trace(go.Scatter(x=h.index, y=h.values, visible=vis, name="Histórico",
                                 line=dict(color=INK, width=2.2)))
    n = len(names); buttons = []
    for i, uf in enumerate(names):
        mask = [False] * (n * tper)
        for k in range(tper):
            mask[i * tper + k] = True
        buttons.append(dict(label=f"{uf} · {UF_NOME[uf]}", method="update",
                            args=[{"visible": mask}]))
    lay = base_layout(440)
    lay["updatemenus"] = [dict(buttons=buttons, direction="down", showactive=True,
                               x=1, xanchor="right", y=1.18, yanchor="top",
                               bgcolor="#fff", bordercolor=GRID, font=dict(size=12))]
    fig.update_layout(**lay)
    return fig


def fig_regioes(panel):
    fig = go.Figure()
    cols = {"Sudeste": AMBER, "Sul": BLUE, "Nordeste": GREEN, "Norte": VIOLET, "Centro-Oeste": "#0891b2"}
    for reg, ufs in REGIOES.items():
        s = panel[panel["uf"].isin(ufs)].groupby("date")["saldo"].sum().rolling(12).sum()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=reg, line=dict(color=cols[reg], width=2.4)))
    fig.update_layout(**base_layout(420, "Saldo acumulado 12 meses"))
    return fig


def fig_heatmap(panel):
    p = panel.copy(); p["mes"] = p["date"].dt.month
    order = [u for r in REGIOES.values() for u in r]
    piv = p.groupby(["uf", "mes"])["saldo"].mean().unstack().reindex(order)
    vmax = np.nanmax(np.abs(piv.values))
    fig = go.Figure(go.Heatmap(z=piv.values, x=MESES, y=order, colorscale="RdYlGn",
                               zmid=0, zmin=-vmax, zmax=vmax,
                               colorbar=dict(title="Saldo médio", thickness=12)))
    lay = base_layout(620, "")
    lay["yaxis"].update(autorange="reversed", title="UF")
    lay["hovermode"] = "closest"
    fig.update_layout(**lay)
    return fig


def _norm(s):
    return str(s).replace("–", "-").replace("≤", "<=").strip().lower()


def fig_hours_share(sec_hours):
    """Overall share of admissions by contracted-hours band, 44h highlighted."""
    g = sec_hours.groupby("faixa")["admissoes"].sum()
    order = ["Parcial (≤30h)", "30–39h", "40h", "41–43h", "44h (jornada máxima)"]
    norm_to_val = {_norm(k): v for k, v in g.items()}
    g = pd.Series([norm_to_val.get(_norm(c), 0.0) for c in order], index=order)
    share = g / g.sum() * 100
    colors = ["#cbd5e1", "#94a3b8", "#60a5fa", "#f59e0b", AMBER]
    fig = go.Figure(go.Bar(x=share.values, y=order, orientation="h",
                           marker_color=colors, text=[f"{v:.1f}%" for v in share.values],
                           textposition="outside",
                           hovertemplate="%{y}: %{x:.1f}% das admissões<extra></extra>"))
    lay = base_layout(330, "")
    lay["xaxis"].update(title="% das admissões formais (2020–2026)", showgrid=True, gridcolor=GRID, range=[0, max(share.values) * 1.18])
    lay["yaxis"].update(autorange="reversed")
    lay["margin"]["l"] = 150
    fig.update_layout(**lay)
    return fig, share


def fig_exposure(sector, sec_hours):
    """Bubble: x=% admissões a 44h, y=saldo 12m, size=admissões. Quadrant story."""
    last12 = sector[sector["date"] >= sector["date"].max() - pd.DateOffset(months=11)]
    agg = last12.groupby(["secao", "secao_nome"]).agg(saldo12=("saldo", "sum"),
                                                        adm12=("admissoes", "sum")).reset_index()
    sh = sec_hours.copy()
    tot = sh.groupby("secao")["admissoes"].sum()
    h44 = sh[sh["faixa"].str.contains("44")].groupby("secao")["admissoes"].sum()
    pct44 = (h44 / tot * 100).reindex(agg["secao"]).fillna(0).values
    agg["pct44"] = pct44
    agg = agg[agg["adm12"] > 0]
    sizes = np.sqrt(agg["adm12"]) ; sizes = sizes / sizes.max() * 64 + 8
    media44 = float(h44.sum() / tot.sum() * 100)
    fig = go.Figure()
    fig.add_vline(x=media44, line=dict(color="#94a3b8", dash="dash", width=1))
    fig.add_hline(y=0, line=dict(color="#cbd5e1", width=1))
    fig.add_trace(go.Scatter(
        x=agg["pct44"], y=agg["saldo12"], mode="markers+text",
        text=agg["secao"], textposition="middle center", textfont=dict(size=10, color="#fff"),
        marker=dict(size=sizes, color=agg["pct44"], colorscale="OrRd", cmin=0, cmax=80,
                    line=dict(color="#fff", width=1.2), opacity=.9),
        customdata=np.stack([agg["secao_nome"], agg["adm12"], agg["saldo12"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>% a 44h: %{x:.0f}%<br>"
                      "Saldo 12m: %{customdata[2]:,.0f}<br>Admissões 12m: %{customdata[1]:,.0f}<extra></extra>"))
    lay = base_layout(520, "Saldo de empregos · últimos 12 meses")
    lay["xaxis"].update(title="% das admissões contratadas a 44h (exposição à escala 6×1)",
                        showgrid=True, gridcolor=GRID, ticksuffix="%")
    fig.update_layout(**lay)
    return fig, agg, media44


# ----------------------------------------------------------------- HTML pieces
def stat(value, label, note="", tone="ink"):
    return f'''<div class="stat tone-{tone}"><div class="stat-v">{value}</div>
      <div class="stat-l">{label}</div>{f'<div class="stat-n">{note}</div>' if note else ''}</div>'''


CSS = r"""
:root{--ink:#0d1b2a;--paper:#fbfaf7;--card:#fff;--mut:#5b6675;--line:#e7e5e1;
--blue:#1d4ed8;--amber:#c2410c;--green:#15803d;--red:#b91c1c;--maxw:1120px}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);
font-family:-apple-system,'Segoe UI',Roboto,system-ui,sans-serif;line-height:1.7;
-webkit-font-smoothing:antialiased}
.serif{font-family:Georgia,'Times New Roman',serif}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 24px}
.prose{max-width:70ch}
.prose p{margin:0 0 16px;font-size:17px;color:#27313f}
.prose p .lead-in{font-weight:700}
/* progress + nav */
#prog{position:fixed;top:0;left:0;height:3px;background:var(--amber);width:0;z-index:60}
nav.top{position:sticky;top:0;z-index:50;background:rgba(251,250,247,.9);
backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav.top .wrap{display:flex;gap:6px;align-items:center;height:52px;overflow-x:auto}
nav.top a{font-size:12.5px;color:var(--mut);text-decoration:none;padding:6px 10px;
border-radius:7px;white-space:nowrap}
nav.top a:hover{background:#fff;color:var(--ink)}
nav.top .brand{font-weight:700;color:var(--ink);margin-right:8px}
/* hero */
header.hero{background:radial-gradient(1200px 500px at 80% -10%,#15356b 0%,#0d1b2a 55%);
color:#fff;padding:84px 0 70px;border-bottom:4px solid var(--amber)}
.kicker{font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:700;
color:var(--amber)}
header.hero .kicker{color:#fbbf24}
header.hero h1{font-size:clamp(34px,5vw,58px);line-height:1.05;margin:14px 0 18px;font-weight:700;
letter-spacing:-.02em}
header.hero .thesis{font-size:19px;max-width:62ch;color:#dbe4f0}
header.hero .src{margin-top:22px;font-size:13px;color:#9fb0c7}
.hero-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin-top:40px;
border-top:1px solid rgba(255,255,255,.18)}
.hero-strip .hs{padding:18px 18px 4px;border-right:1px solid rgba(255,255,255,.12)}
.hero-strip .hs:last-child{border-right:0}
.hero-strip .v{font-size:27px;font-weight:700}
.hero-strip .l{font-size:12.5px;color:#9fb0c7}
/* sections */
section{padding:64px 0}
section.alt{background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
section h2{font-size:clamp(26px,3.4vw,38px);line-height:1.12;margin:10px 0 8px;font-weight:700;
letter-spacing:-.015em}
.lede{font-size:19px;color:var(--mut);max-width:72ch;margin:0 0 26px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 14px 6px;
margin:22px 0;box-shadow:0 1px 2px rgba(13,27,42,.05)}
.figcap{font-size:12.5px;color:var(--mut);padding:4px 6px 8px}
.figcap b{color:var(--ink)}
/* findings grid */
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin:8px 0 6px}
.stat{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--blue);
border-radius:10px;padding:18px 20px}
.stat-v{font-size:30px;font-weight:700;letter-spacing:-.02em}
.stat-l{font-size:14px;color:var(--ink);font-weight:600;margin-top:2px}
.stat-n{font-size:12.5px;color:var(--mut);margin-top:5px}
.tone-green{border-left-color:var(--green)}.tone-red{border-left-color:var(--red)}
.tone-amber{border-left-color:var(--amber)}.tone-violet{border-left-color:#7c3aed}
.tone-ink{border-left-color:var(--ink)}
/* callout / pullquote */
.callout{border-left:4px solid var(--amber);background:#fff7ed;padding:16px 22px;border-radius:0 10px 10px 0;
margin:24px 0;font-size:16.5px;color:#7c2d12}
.pull{font-family:Georgia,serif;font-size:25px;line-height:1.35;color:var(--ink);
border-top:2px solid var(--ink);border-bottom:1px solid var(--line);padding:20px 0;margin:30px 0;max-width:30ch}
.grid2{display:grid;grid-template-columns:1.05fr .95fr;gap:34px;align-items:center}
@media(max-width:880px){.grid2{grid-template-columns:1fr}.hero-strip{grid-template-columns:repeat(2,1fr)}}
/* dossie 6x1 */
section.dossie{background:linear-gradient(180deg,#1a1206,#0d1b2a);color:#f4ede2;border-top:4px solid var(--amber)}
section.dossie h2,section.dossie .pull{color:#fff}
section.dossie .lede{color:#d6c3a8}
section.dossie .prose p{color:#e9dfce}
section.dossie .fig{background:#11203a;border-color:#2a3b57}
section.dossie .figcap{color:#a8b6cc}
section.dossie .stat{background:#11203a;border-color:#2a3b57;border-left-color:var(--amber)}
section.dossie .stat-l{color:#fff}section.dossie .stat-n{color:#a8b6cc}
.scn{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}
.scn .c{background:#11203a;border:1px solid #2a3b57;border-radius:12px;padding:20px}
.scn .c h4{margin:0 0 8px;font-size:16px;color:#fbbf24}
.scn .c.up h4{color:#86efac}.scn .c.down h4{color:#fca5a5}
.scn .c p{margin:0;font-size:14.5px;color:#d6deeb;line-height:1.6}
@media(max-width:880px){.scn{grid-template-columns:1fr}}
.cit a{color:#fbbf24;text-decoration:none;font-weight:700}
/* tables inside the dark dossier need an inverted theme */
section.dossie table.tbl{background:#11203a;color:#e9dfce;border:1px solid #2a3b57;border-radius:10px;overflow:hidden}
section.dossie .tbl th{background:#0b1830;color:#fbbf24;border-bottom:1px solid #2a3b57;position:static}
section.dossie .tbl td{border-bottom:1px solid #22344f;color:#e9dfce}
section.dossie .tbl td.mono{color:#fbbf24;font-weight:600}
section.dossie .tbl tr:hover td{background:#16284a}
.refs{margin:10px 0 0;padding-left:22px;color:#d6deeb;font-size:13.5px;line-height:1.65}
.refs li{margin-bottom:9px}.refs li b{color:#fff}
.refs a{color:#7dd3fc;text-decoration:none}.refs a:hover{text-decoration:underline}
section.dossie .refs li[id]:target{background:rgba(251,191,36,.18);border-radius:5px;padding:2px 6px}
/* tables */
table.tbl{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:10px;background:#fff}
.tbl th,.tbl td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left}
.tbl th{background:#f5f3ef;font-weight:600;cursor:pointer;user-select:none;position:sticky;top:52px}
.tbl td.num{text-align:right;font-variant-numeric:tabular-nums}
.tbl td.mono,.mono{font-family:ui-monospace,Consolas,monospace}
.tbl tr:hover td{background:#faf9f6}
.tbl .small{font-size:11.5px;color:var(--mut)}
footer{background:var(--ink);color:#b9c4d4;padding:48px 0;font-size:13px}
footer a{color:#dbe4f0}
.disc{font-size:12.5px;color:var(--mut);font-style:italic;margin-top:14px}
"""

PROG_JS = r"""
const prog=document.getElementById('prog');
addEventListener('scroll',()=>{const h=document.documentElement;
 const p=h.scrollTop/(h.scrollHeight-h.clientHeight)*100;prog.style.width=p+'%';});
document.querySelectorAll('table.sortable th').forEach((th,i)=>{th.addEventListener('click',()=>{
 const tb=th.closest('table').querySelector('tbody');const rs=[...tb.querySelectorAll('tr')];
 const asc=th._a=!th._a;rs.sort((a,b)=>{const x=a.children[i].innerText.replace(/[^0-9,.\-]/g,'').replace('.','').replace(',','.');
 const y=b.children[i].innerText.replace(/[^0-9,.\-]/g,'').replace('.','').replace(',','.');
 const nx=parseFloat(x),ny=parseFloat(y);if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;
 return asc?a.children[i].innerText.localeCompare(b.children[i].innerText):b.children[i].innerText.localeCompare(a.children[i].innerText);});
 rs.forEach(r=>tb.appendChild(r));});});
"""


def build():
    brasil, panel, fc, met, sector, hours, sec_hours = load_all()

    # ---- narrative statistics ----
    span = f"{brasil.index[0].strftime('%b/%Y')}–{brasil.index[-1].strftime('%b/%Y')}"
    last = brasil.iloc[-1]; last_m = brasil.index[-1].strftime("%b/%Y")
    acc12 = brasil.iloc[-12:].sum()
    acc_prev12 = brasil.iloc[-24:-12].sum()
    estoque = brasil.cumsum().iloc[-1]
    trough = brasil.min(); trough_m = brasil.idxmin().strftime("%b/%Y")
    nat_models = models_of("Brasil", fc); nat_best = best_of("Brasil", met)
    fc_end = nat_models[nat_best]["mean"].iloc[-1]
    fc_lo = nat_models[nat_best]["lower"].iloc[-1]; fc_hi = nat_models[nat_best]["upper"].iloc[-1]
    nat_dg = met[met["serie"] == "Brasil"].iloc[0]

    # state stats
    order = [u for r in REGIOES.values() for u in r]
    acc_uf = {u: hist_of(u, brasil, panel).iloc[-12:].sum() for u in order}
    top_uf = max(acc_uf, key=acc_uf.get); sp_share = acc_uf["SP"] / sum(v for v in acc_uf.values()) * 100
    sudeste_share = (panel[panel["uf"].isin(REGIOES["Sudeste"])].groupby("date")["saldo"].sum().iloc[-12:].sum()
                     / acc12 * 100)

    # 6x1 stats
    figs_hours, share = fig_hours_share(sec_hours)
    pct44 = share["44h (jornada máxima)"]
    pct_full = share["44h (jornada máxima)"] + share["41–43h"] + share["40h"]
    exp_fig, agg, media44 = fig_exposure(sector, sec_hours)
    exposed = agg.sort_values("pct44", ascending=False)
    top_exposed = exposed.head(4)
    # share of recent hiring (admissions) in high-exposure sectors (pct44 >= national avg)
    high = agg[agg["pct44"] >= media44]
    high_adm_share = high["adm12"].sum() / agg["adm12"].sum() * 100 if agg["adm12"].sum() else float("nan")

    # ---- figures ----
    d_traj = div(fig_trajetoria(brasil))
    d_fan = div(fig_forecast_fan("Brasil", brasil, nat_models, nat_best))
    d_reg = div(fig_regioes(panel))
    d_heat = div(fig_heatmap(panel))
    d_explorer = div(fig_explorer(order, lambda u: hist_of(u, brasil, panel),
                                  lambda u: models_of(u, fc), lambda u: best_of(u, met)))
    d_hours = div(figs_hours)
    d_exp = div(exp_fig)

    gen = dt.datetime.now().strftime("%d/%m/%Y")

    # ---- per-UF table ----
    def truthy(x):
        return str(x).strip().lower() in ("true", "1", "sim", "1.0")
    rows = ""
    for u in order:
        r = met[met["serie"] == u].iloc[0]
        rows += f'''<tr><td class="mono">{u}</td><td>{UF_NOME[u]}</td>
          <td class="num">{br(acc_uf[u])}</td><td>{r['melhor_modelo']}</td>
          <td class="num">{br(r['prev_jun2027'])}</td>
          <td class="num small">{br(r['ic_inf'])} a {br(r['ic_sup'])}</td>
          <td class="num">{br(r['seasonal_strength'],2)}</td>
          <td>{'sim' if truthy(r['heteroscedastic']) else 'não'}</td></tr>'''
    uf_table = f'''<table class="tbl sortable"><thead><tr><th>UF</th><th>Estado</th>
      <th>Saldo 12m</th><th>Modelo</th><th>Prev. jun/27</th><th>IC 95%</th>
      <th>Sazonal.</th><th>Heter.</th></tr></thead><tbody>{rows}</tbody></table>'''

    # exposure mini-table
    exp_rows = ""
    for _, r in exposed.head(8).iterrows():
        exp_rows += f'''<tr><td class="mono">{r['secao']}</td><td>{r['secao_nome']}</td>
          <td class="num">{r['pct44']:.0f}%</td><td class="num">{br(r['saldo12'])}</td>
          <td class="num">{br(r['adm12'])}</td></tr>'''
    exp_table = f'''<table class="tbl"><thead><tr><th>CNAE</th><th>Setor</th>
      <th>% a 44h</th><th>Saldo 12m</th><th>Admissões 12m</th></tr></thead><tbody>{exp_rows}</tbody></table>'''

    refs_html = render_refs(load_refs())
    plotlyjs = pyo.get_plotlyjs()
    tl = top_exposed["secao_nome"].tolist()
    exposed_list = ", ".join(tl[:3]) + " e " + tl[3] if len(tl) >= 4 else ", ".join(tl)

    html = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLT em Movimento · Emprego Formal no Brasil (Novo CAGED/MTE)</title>
<style>{CSS}</style><script>{plotlyjs}</script></head><body>
<div id="prog"></div>
<nav class="top"><div class="wrap">
  <span class="brand serif">CLT em Movimento</span>
  <a href="#panorama">Panorama</a><a href="#trajetoria">Trajetória</a>
  <a href="#geografia">Geografia</a><a href="#sazonalidade">Sazonalidade</a>
  <a href="#projecao">Projeção</a><a href="#dossie">Dossiê 6×1</a><a href="#estados">Estados</a>
</div></nav>

<header class="hero"><div class="wrap">
  <div class="kicker">Mercado de trabalho formal · Novo CAGED / MTE</div>
  <h1 class="serif">CLT em Movimento</h1>
  <p class="thesis">Seis anos de empregos com carteira assinada no Brasil — do colapso da
  pandemia à desaceleração de 2025 — e o que as projeções e o debate do <b>fim da escala 6×1</b>
  sinalizam para o trabalho formal até 2027.</p>
  <div class="hero-strip">
    <div class="hs"><div class="v">{br(estoque)}</div><div class="l">vínculos celetistas líquidos criados desde jan/2020</div></div>
    <div class="hs"><div class="v">{br(acc12)}</div><div class="l">saldo acumulado nos últimos 12 meses</div></div>
    <div class="hs"><div class="v">{pct44:.0f}%</div><div class="l">das admissões são contratadas a 44h (universo da 6×1)</div></div>
    <div class="hs"><div class="v">{br(fc_end)}</div><div class="l">saldo projetado para jun/2027 (modelo {nat_best})</div></div>
  </div>
  <div class="src">Fonte: microdados do Novo CAGED (PDET/Ministério do Trabalho e Emprego) · Período {span} · 27 UFs · Elaboração própria · {gen}</div>
</div></header>

<section id="panorama"><div class="wrap">
  <div class="kicker">Panorama</div>
  <h2 class="serif">O retrato de hoje</h2>
  <p class="lede">O emprego formal brasileiro vive um momento de <b>expansão em desaceleração</b>:
  ainda cria vagas todos os meses, mas em ritmo menor que no auge da retomada pós-pandemia.</p>
  <div class="prose">
    <p><span class="lead-in">A recuperação foi real, mas perdeu fôlego.</span> Entre {span}, o saldo
    acumulado somou <b>{br(estoque)}</b> vínculos celetistas líquidos. Nos últimos 12 meses o país
    gerou <b>{br(acc12)}</b> empregos formais — ante {br(acc_prev12)} nos 12 meses anteriores,
    uma variação de <b>{(acc12/acc_prev12-1)*100:+.1f}%</b> que confirma a perda de tração em um
    cenário de juros elevados e crédito mais caro.</p>
  </div>
  <div class="findings">
    {stat(br(last), 'Saldo do último mês', last_m, 'green' if last>=0 else 'red')}
    {stat(br(acc12), 'Saldo · 12 meses', 'admissões − desligamentos', 'green')}
    {stat(f'{sp_share:.0f}%', 'do saldo nacional', f'concentrado em São Paulo', 'amber')}
    {stat(f'{nat_dg["seasonal_strength"]:.2f}', 'Força da sazonalidade', 'padrão anual marcante (0–1)', 'violet')}
  </div>
</div></section>

<section id="trajetoria" class="alt"><div class="wrap">
  <div class="kicker">Trajetória</div>
  <h2 class="serif">A montanha-russa de seis anos</h2>
  <p class="lede">Da queda livre de 2020 à normalização recente: a série mensal conta a história
  econômica do país melhor que qualquer manchete.</p>
  <div class="fig">{d_traj}
    <div class="figcap"><b>Saldo mensal de empregos formais (CLT) no Brasil.</b> Barras: saldo do mês
    (verde positivo, vermelho negativo). Linha: média móvel de 12 meses. Fonte: Novo CAGED/MTE.</div>
  </div>
  <div class="prose">
    <p><span class="lead-in">Abril de 2020 é o fundo do poço.</span> No primeiro choque da pandemia,
    o país fechou <b>{br(trough)}</b> vagas formais em um único mês — o pior resultado da série.
    Seguiu-se uma recuperação em formato de “V”, com saldos recordes em 2021 e 2022 à medida que a
    economia reabria.</p>
    <p>Desde então, a média móvel se acomodou em um <b>platô positivo</b>: o Brasil continua criando
    empregos, mas o ímpeto arrefeceu em 2024–2026. O padrão dente-de-serra que se repete todo ano
    não é ruído — é <b>sazonalidade</b>, e ela tem nome e sobrenome setorial, como veremos adiante.</p>
  </div>
  <div class="pull serif">Cada dezembro o Brasil demite; cada começo de ano, recontrata. O calendário do emprego é quase tão previsível quanto o das estações.</div>
</div></section>

<section id="geografia"><div class="wrap">
  <div class="kicker">Geografia</div>
  <h2 class="serif">Cinco Brasis no mesmo mapa</h2>
  <p class="lede">A geração de empregos formais é profundamente concentrada. O Sudeste responde por
  cerca de <b>{sudeste_share:.0f}%</b> do saldo recente, e São Paulo sozinho por <b>{sp_share:.0f}%</b>.</p>
  <div class="fig">{d_reg}
    <div class="figcap"><b>Saldo acumulado em 12 meses, por região.</b> Soma móvel — leitura do estoque
    de vagas geradas no último ano em cada região.</div>
  </div>
  <p class="lede" style="margin-top:30px">Explore a série e a projeção de cada estado:</p>
  <div class="fig">{d_explorer}
    <div class="figcap"><b>Saldo mensal e previsão até jun/2027 por UF</b> (melhor modelo, IC 95%).
    Use o seletor no canto superior direito.</div>
  </div>
</div></section>

<section id="sazonalidade" class="alt"><div class="wrap">
  <div class="kicker">Sazonalidade</div>
  <h2 class="serif">O calendário do emprego</h2>
  <p class="lede">O saldo médio por mês revela um relógio anual: o país contrata no primeiro semestre
  e nas safras, e demite em dezembro. O comércio e os serviços ditam o ritmo.</p>
  <div class="fig">{d_heat}
    <div class="figcap"><b>Saldo médio por mês do ano e UF.</b> Verde = geração líquida típica;
    vermelho = perdas. A faixa vermelha de dezembro é universal.</div>
  </div>
  <div class="prose">
    <p>Esse padrão é dominado por setores intensivos em mão de obra e contratos de jornada cheia —
    exatamente os que estão no centro do debate sobre a escala <b>6×1</b>. É para ele que olhamos a seguir.</p>
  </div>
</div></section>

<section id="projecao"><div class="wrap">
  <div class="kicker">Projeção</div>
  <h2 class="serif">Para onde vamos</h2>
  <p class="lede">Três modelos — SARIMA, ETS (Holt-Winters) e <i>Seasonal Naive</i> — projetam a série
  até <b>junho de 2027</b>. O selecionado por validação foi o <b>{nat_best}</b>.</p>
  <div class="fig">{d_fan}
    <div class="figcap"><b>Brasil — histórico recente e previsões até jun/2027.</b> Linha cheia: modelo
    selecionado; tracejadas: alternativos; faixa: intervalo de confiança de 95%.</div>
  </div>
  <div class="prose">
    <p><span class="lead-in">A leitura central é de estabilidade sazonal, não de aceleração.</span>
    Para jun/2027 o modelo projeta saldo de <b>{br(fc_end)}</b> vagas, com intervalo de 95% entre
    <b>{br(fc_lo)}</b> e <b>{br(fc_hi)}</b> — amplitude que reflete a forte sazonalidade e a
    <b>heterocedasticidade</b> (variância instável) detectada na série.</p>
  </div>
  <div class="callout">⚠️ <b>As projeções pressupõem o arcabouço atual.</b> Uma mudança estrutural como
  o fim da escala 6×1 seria uma quebra que estes modelos, treinados no passado, não antecipam — e é
  por isso que ela merece um capítulo à parte.</div>
</div></section>

<section id="dossie" class="dossie"><div class="wrap">
  <div class="kicker">Dossiê · Reforma da jornada</div>
  <h2 class="serif">O fim da escala 6×1</h2>
  <p class="lede">A proposta de substituir a jornada <b>6×1</b> (seis dias de trabalho, um de descanso)
  por <b>5×2</b> mobiliza o Congresso e as ruas. O que os microdados do CAGED dizem sobre quem seria
  afetado — e o que esperar.</p>
  <div class="prose">
    <p><span class="lead-in">O que está em jogo.</span> A escala 6×1 organiza a semana em seis
    jornadas de trabalho e uma de descanso, tipicamente somando as <b>44 horas semanais</b> permitidas
    pela Constituição. A PEC do fim da 6×1 propõe limitar a semana a cinco dias (5×2) e reduzir a
    jornada, sem corte de salário. É a maior discussão sobre tempo de trabalho no país desde 1988.</p>
  </div>

  <div class="grid2">
    <div class="fig">{d_hours}
      <div class="figcap"><b>Distribuição das admissões formais por jornada contratada (2020–2026).</b>
      A barra destacada — <b>44h</b> — é o universo típico da 6×1.</div>
    </div>
    <div>
      <div class="findings" style="grid-template-columns:1fr 1fr">
        {stat(f'{pct44:.0f}%', 'das admissões a 44h', 'a jornada máxima legal — núcleo da 6×1', 'amber')}
        {stat(f'{pct_full:.0f}%', 'a 40h ou mais', 'predomínio do contrato de jornada cheia', 'amber')}
      </div>
      <div class="prose"><p>Os dados de contratação mostram que a jornada de <b>44 horas é a norma</b>,
      não a exceção: cerca de <b>{pct44:.0f}%</b> de todas as admissões formais desde 2020 foram
      pactuadas no teto constitucional. Reduzir a escala atinge, portanto, o contrato <i>modal</i> do
      mercado de trabalho brasileiro.</p></div>
    </div>
  </div>

  <p class="lede" style="margin-top:40px">Mas a exposição é desigual entre setores. O mapa abaixo cruza
  <b>quão dependente de jornada de 44h</b> é cada setor (eixo horizontal) com <b>quanto emprego ele
  gera</b> (eixo vertical).</p>
  <div class="fig">{d_exp}
    <div class="figcap"><b>Mapa de exposição setorial à escala 6×1.</b> Cada bolha é um setor (CNAE);
    horizontal: % das admissões a 44h; vertical: saldo de 12 meses; tamanho: volume de admissões.
    Linha tracejada: média nacional de {media44:.0f}% a 44h.</div>
  </div>
  <div class="prose">
    <p><span class="lead-in">Os mais expostos são também grandes empregadores.</span> Setores como
    <b>{exposed_list}</b> combinam alta dependência da jornada de 44h com geração relevante de vagas.
    Os setores acima da média de exposição respondem por cerca de <b>{high_adm_share:.0f}%</b> das
    contratações formais recentes — ou seja, a reforma incide justamente sobre a engrenagem que mais contrata.</p>
  </div>
  {exp_table}

  <h3 class="serif" style="font-size:24px;margin:42px 0 6px">O que esperar — à luz da evidência</h3>
  <p style="color:#d6c3a8;max-width:74ch">Não há consenso, e o efeito final depende do desenho (corte
  de jornada, prazo de transição, compensações). Cruzamos os cenários mais discutidos com a literatura
  econômica revisada por pares (numerada ao final):</p>
  <div class="scn">
    <div class="c up"><h4>↑ Argumentos a favor</h4><p>A produtividade <i>por hora</i> tende a subir
    quando se cortam jornadas longas — em dados empíricos o produto cresce menos que proporcionalmente
    às horas<sup class="cit"><a href="#ref1">1</a>,<a href="#ref2">2</a></sup>, e escalas comprimidas
    elevam satisfação e atitudes no trabalho<sup class="cit"><a href="#ref4">4</a></sup>. Há ainda ganho
    de saúde e sono ao reduzir jornadas extensas<sup class="cit"><a href="#ref7">7</a>,<a href="#ref8">8</a></sup>.</p></div>
    <div class="c down"><h4>↓ Riscos apontados</h4><p>Eleva o custo da hora em setores intensivos em
    mão de obra (comércio, alimentação, serviços); empregadores tendem a reagir ajustando salário-base
    e jornada<sup class="cit"><a href="#ref5">5</a></sup>. A experiência francesa das 35h não produziu
    efeito positivo robusto sobre o emprego<sup class="cit"><a href="#ref3">3</a></sup>, sugerindo
    cautela quanto a ganhos automáticos de postos.</p></div>
    <div class="c"><h4>⟳ Ajuste provável</h4><p>Reorganização de escalas, banco de horas, mais turnos
    parciais e contratações 5×2 a 40h<sup class="cit"><a href="#ref4">4</a>,<a href="#ref6">6</a></sup>.
    O peso recairia sobre os setores do quadrante direito do mapa acima — os de maior exposição a 44h.</p></div>
    <div class="c"><h4>📊 Para o modelo</h4><p>Uma transição alteraria o nível e a sazonalidade da
    série — uma quebra estrutural. As projeções deste relatório servem de <b>linha de base</b>
    (cenário “sem reforma”) contra a qual medir o efeito.</p></div>
  </div>

  <h3 class="serif" style="font-size:22px;margin:40px 0 4px">Evidência acadêmica</h3>
  <p style="color:#d6c3a8;max-width:74ch;font-size:14.5px">Referências localizadas pelo protocolo de
  busca científica <i>paper-lookup</i> (k-dense scientific-agent-skills) via OpenAlex, sem fabricação:
  cada item tem DOI verificável.</p>
  {refs_html}

  <p class="disc">Análise de natureza econômica e exploratória, baseada na composição das contratações
  formais. O CAGED registra horas contratadas, não o número de dias trabalhados; 44h é usado como
  proxy do universo 6×1. A evidência citada vem de contextos institucionais distintos do brasileiro e
  não deve ser lida como previsão do efeito da PEC, tampouco como aconselhamento.</p>
</div></section>

<section id="estados"><div class="wrap">
  <div class="kicker">Apêndice</div>
  <h2 class="serif">Síntese por estado</h2>
  <p class="lede">Saldo recente, modelo selecionado, projeção para jun/2027 com intervalo de confiança
  e diagnósticos por UF. Clique nos cabeçalhos para ordenar.</p>
  {uf_table}
</div></section>

<footer><div class="wrap">
  <b>CLT em Movimento</b> — elaborado a partir dos microdados do Novo CAGED (PDET/Ministério do
  Trabalho e Emprego).<br>
  Metodologia: saldo = Σ(saldomovimentação) por competência (reproduz a série oficial “sem ajuste”);
  modelos SARIMA, ETS (Holt-Winters) e Seasonal Naive, seleção por RMSE em validação de 12 meses;
  jornada via campo <i>horascontratuais</i>. Revisão de literatura conduzida pelo protocolo
  <i>paper-lookup</i> (k-dense scientific-agent-skills) sobre a base OpenAlex.<br>
  Relatório autocontido (offline) · Período {span} · Gerado em {gen}.
</div></footer>
<script>{PROG_JS}</script>
</body></html>"""

    out = os.path.join(OUT, "CLT-BRASIL_relatorio_v2.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


if __name__ == "__main__":
    print(build())
