"""
CLT-BRASIL — Report v2: editorial, narrative-driven, self-contained HTML.

Reconstructs forecasts from persisted CSVs (no model refit), computes
data-grounded narrative statistics, and renders a magazine-quality report
in the portfolio "Midnight" aesthetic (Fraunces / Inter / JetBrains Mono,
dusk-blue paper + amber accent), with a special data dossier on the
6x1 -> 5x2 work-scale debate.

Trilingual: builds Portuguese, English and Spanish editions from a single
translation layer. Run `python src/report2.py` to emit all three into output/.
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

# ---- palette (portfolio "Midnight"; hex tuned for Plotly on dark) ----
INK = "#e7ecf3"        # primary light ink (history line, annotations baseline)
INK2 = "#c2ccd9"       # secondary ink (axis/font)
PAPER = "#0d1320"      # midnight paper
BLUE = "#6ea8fe"
SKY = "#7cc4ff"
GREEN = "#56d49b"
RED = "#f08a7f"
AMBER = "#e3a857"      # warm amber accent
VIOLET = "#b89cff"
GRID = "rgba(255,255,255,0.09)"
ZERO = "rgba(255,255,255,0.22)"
MODEL_COLORS = {"SARIMA": BLUE, "ETS (Holt-Winters)": GREEN, "Seasonal Naive": VIOLET}

MESES = {
    "pt": ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"],
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "es": ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"],
}


def br(v, dec=0, lang="pt"):
    """Localized number formatting. pt/es: 1.234,5 · en: 1,234.5"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"{v:,.{dec}f}"  # always 1,234.5 first
    if lang == "en":
        return s
    return s.replace(",", "·").replace(".", ",").replace("·", ".")


def pct(v, dec=0, lang="pt"):
    s = f"{v:.{dec}f}"
    if lang != "en":
        s = s.replace(".", ",")
    return s + "%"


def signed_pct(v, dec=1, lang="pt"):
    s = f"{v:+.{dec}f}"
    if lang != "en":
        s = s.replace(".", ",")
    return s + "%"


def fmt_my(ts, lang="pt"):
    return f"{MESES[lang][ts.month - 1]}/{ts.year}"


def fmt_gen(now, lang="pt"):
    if lang == "en":
        return f"{MESES['en'][now.month - 1]} {now.day}, {now.year}"
    return now.strftime("%d/%m/%Y")


# ----------------------------------------------------------------- data assembly
def load_refs():
    import json
    p = os.path.join(PROC, "references.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def render_refs(refs, no_doi):
    items = ""
    for r in refs:
        doi = r.get("doi") or ""
        link = (f' · <a href="https://doi.org/{doi}" target="_blank" rel="noopener">doi:{doi}</a>'
                if doi else f" · ({no_doi})")
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
def base_layout(lang, h=460, ytitle=""):
    seps = ".," if lang == "en" else ",."
    return dict(
        template="plotly_dark", height=h, hovermode="x unified", separators=seps,
        margin=dict(l=64, r=26, t=20, b=44), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=12, color=INK2), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID,
                   tickfont=dict(color=INK2)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=ZERO,
                   zerolinewidth=1.2, title=dict(text=ytitle, font=dict(color=INK2)),
                   tickfont=dict(color=INK2)),
        font=dict(family="Inter,-apple-system,Segoe UI,Roboto,sans-serif", size=12.5, color=INK2),
    )


def div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


def fig_trajetoria(brasil, lang, S):
    fig = go.Figure()
    colors = [GREEN if v >= 0 else RED for v in brasil.values]
    fig.add_trace(go.Bar(x=brasil.index, y=brasil.values, marker_color=colors,
                         marker_line_width=0, opacity=.5, name=S["lg_monthly"],
                         hovertemplate="%{x|%b/%Y}: %{y:,.0f}<extra></extra>"))
    mm = brasil.rolling(12).mean()
    fig.add_trace(go.Scatter(x=mm.index, y=mm.values, line=dict(color=INK, width=2.6),
                             name=S["lg_ma12"],
                             hovertemplate="MM12 %{x|%b/%Y}: %{y:,.0f}<extra></extra>"))
    lay = base_layout(lang, 440, S["yt_saldo"])
    trough_date = brasil.idxmin(); trough = brasil.min()
    fig.add_annotation(x=trough_date, y=trough, text=f"<b>{fmt_my(trough_date, lang)}</b><br>{br(trough, 0, lang)}",
                       showarrow=True, arrowhead=2, ax=70, ay=-10, font=dict(color=RED, size=11),
                       arrowcolor=RED, align="left")
    peak_date = brasil.idxmax(); peak = brasil.max()
    fig.add_annotation(x=peak_date, y=peak, text=f"<b>{S['an_peak']}</b><br>{br(peak, 0, lang)}",
                       showarrow=True, arrowhead=2, ax=0, ay=-28, font=dict(color=GREEN, size=11),
                       arrowcolor=GREEN)
    fig.update_layout(**lay)
    return fig


def fig_forecast_fan(name, brasil_or_hist, models, best, lang, S):
    fig = go.Figure()
    h = brasil_or_hist.iloc[-30:]
    fig.add_trace(go.Scatter(x=h.index, y=h.values, line=dict(color=INK, width=2.4),
                             name=S["lg_hist"]))
    for m, md in models.items():
        c = MODEL_COLORS.get(m, "#888")
        is_best = (m == best)
        if is_best:
            fig.add_trace(go.Scatter(
                x=list(md["upper"].index) + list(md["lower"].index[::-1]),
                y=list(md["upper"].values) + list(md["lower"].values[::-1]),
                fill="toself", fillcolor="rgba(110,168,254,0.16)", line=dict(width=0),
                hoverinfo="skip", showlegend=True, name=S["lg_ci"]))
        fig.add_trace(go.Scatter(x=md["mean"].index, y=md["mean"].values,
                                 line=dict(color=c, width=3 if is_best else 1.4,
                                           dash="solid" if is_best else "dot"),
                                 opacity=1 if is_best else .6,
                                 name=m + (" ★" if is_best else "")))
    lay = base_layout(lang, 440, S["yt_saldo"])
    fig.add_vline(x=h.index[-1], line=dict(color="#94a3b8", dash="dash", width=1))
    fig.update_layout(**lay)
    return fig


def fig_explorer(names, hist_fn, models_fn, best_fn, lang, S):
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
                fill="toself", fillcolor="rgba(110,168,254,0.15)", line=dict(width=0),
                hoverinfo="skip", showlegend=False, visible=vis))
            fig.add_trace(go.Scatter(x=md["mean"].index, y=md["mean"].values, visible=vis,
                                     name=f"{S['lg_forecast']} · {best}", line=dict(color=BLUE, width=2.6)))
        else:
            fig.add_trace(go.Scatter(x=[], y=[], visible=vis, showlegend=False))
            fig.add_trace(go.Scatter(x=[], y=[], visible=vis, showlegend=False))
        fig.add_trace(go.Scatter(x=h.index, y=h.values, visible=vis, name=S["lg_hist"],
                                 line=dict(color=INK, width=2.2)))
    n = len(names); buttons = []
    for i, uf in enumerate(names):
        mask = [False] * (n * tper)
        for k in range(tper):
            mask[i * tper + k] = True
        buttons.append(dict(label=f"{uf} · {UF_NOME[uf]}", method="update",
                            args=[{"visible": mask}]))
    lay = base_layout(lang, 440, S["yt_saldo"])
    lay["updatemenus"] = [dict(buttons=buttons, direction="down", showactive=True,
                               x=1, xanchor="right", y=1.18, yanchor="top",
                               bgcolor="#16213a", bordercolor="#2a3b57",
                               font=dict(size=12, color=INK2))]
    fig.update_layout(**lay)
    return fig


def fig_regioes(panel, lang, S):
    fig = go.Figure()
    cols = {"Sudeste": AMBER, "Sul": BLUE, "Nordeste": GREEN, "Norte": VIOLET, "Centro-Oeste": SKY}
    for reg, ufs in REGIOES.items():
        s = panel[panel["uf"].isin(ufs)].groupby("date")["saldo"].sum().rolling(12).sum()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=S["reg"][reg], line=dict(color=cols[reg], width=2.4)))
    fig.update_layout(**base_layout(lang, 420, S["yt_acc12"]))
    return fig


def fig_heatmap(panel, lang, S):
    p = panel.copy(); p["mes"] = p["date"].dt.month
    order = [u for r in REGIOES.values() for u in r]
    piv = p.groupby(["uf", "mes"])["saldo"].mean().unstack().reindex(order)
    vmax = np.nanmax(np.abs(piv.values))
    midnight_div = [
        [0.00, "#ef4444"],   # vivid red — strongest losses
        [0.22, "#e3733f"],   # amber-orange
        [0.42, "#5a4636"],   # warm dark transition
        [0.50, "#1b2942"],   # neutral (blends with the dark card)
        [0.58, "#274a48"],   # cool dark transition
        [0.78, "#3fae7a"],   # green
        [1.00, "#56d49b"],   # mint — strongest gains
    ]
    fig = go.Figure(go.Heatmap(z=piv.values, x=MESES[lang], y=order, colorscale=midnight_div,
                               zmid=0, zmin=-vmax, zmax=vmax,
                               colorbar=dict(title=dict(text=S["cb_saldo"], font=dict(color=INK2)),
                                             thickness=12, tickfont=dict(color=INK2))))
    lay = base_layout(lang, 620, "")
    lay["yaxis"].update(autorange="reversed", title="UF")
    lay["hovermode"] = "closest"
    fig.update_layout(**lay)
    return fig


def _norm(s):
    return str(s).replace("–", "-").replace("≤", "<=").strip().lower()


def fig_hours_share(sec_hours, lang, S):
    g = sec_hours.groupby("faixa")["admissoes"].sum()
    keys = ["Parcial (≤30h)", "30–39h", "40h", "41–43h", "44h (jornada máxima)"]
    labels = S["hours_bands"]
    norm_to_val = {_norm(k): v for k, v in g.items()}
    g = pd.Series([norm_to_val.get(_norm(c), 0.0) for c in keys], index=labels)
    share = g / g.sum() * 100
    colors = ["#7c8aa0", "#9aa7bb", BLUE, "#f1b24a", AMBER]
    fig = go.Figure(go.Bar(x=share.values, y=labels, orientation="h",
                           marker_color=colors, text=[f"{v:.1f}%" for v in share.values],
                           textposition="outside", textfont=dict(color=INK2),
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    lay = base_layout(lang, 330, "")
    lay["xaxis"].update(title=S["xt_share"], showgrid=True, gridcolor=GRID,
                        range=[0, max(share.values) * 1.18])
    lay["yaxis"].update(autorange="reversed")
    lay["margin"]["l"] = 168
    fig.update_layout(**lay)
    return fig, share


def fig_exposure(sector, sec_hours, lang, S):
    last12 = sector[sector["date"] >= sector["date"].max() - pd.DateOffset(months=11)]
    agg = last12.groupby(["secao", "secao_nome"]).agg(saldo12=("saldo", "sum"),
                                                        adm12=("admissoes", "sum")).reset_index()
    sh = sec_hours.copy()
    tot = sh.groupby("secao")["admissoes"].sum()
    h44 = sh[sh["faixa"].str.contains("44")].groupby("secao")["admissoes"].sum()
    pct44 = (h44 / tot * 100).reindex(agg["secao"]).fillna(0).values
    agg["pct44"] = pct44
    agg = agg[agg["adm12"] > 0]
    sizes = np.sqrt(agg["adm12"]); sizes = sizes / sizes.max() * 64 + 8
    media44 = float(h44.sum() / tot.sum() * 100)
    fig = go.Figure()
    fig.add_vline(x=media44, line=dict(color="#94a3b8", dash="dash", width=1))
    fig.add_hline(y=0, line=dict(color=ZERO, width=1))
    fig.add_trace(go.Scatter(
        x=agg["pct44"], y=agg["saldo12"], mode="markers+text",
        text=agg["secao"], textposition="middle center", textfont=dict(size=10, color="#0d1320"),
        marker=dict(size=sizes, color=agg["pct44"], colorscale="OrRd", cmin=0, cmax=80,
                    line=dict(color="#0d1320", width=1.2), opacity=.92),
        customdata=np.stack([agg["secao_nome"], agg["adm12"], agg["saldo12"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>" + S["hv_p44"] + ": %{x:.0f}%<br>"
                      + S["hv_saldo12"] + ": %{customdata[2]:,.0f}<br>"
                      + S["hv_adm12"] + ": %{customdata[1]:,.0f}<extra></extra>"))
    lay = base_layout(lang, 520, S["yt_saldo12"])
    lay["xaxis"].update(title=S["xt_exposure"], showgrid=True, gridcolor=GRID, ticksuffix="%")
    fig.update_layout(**lay)
    return fig, agg, media44


# ----------------------------------------------------------------- HTML pieces
def stat(value, label, note="", tone="ink"):
    return f'''<div class="stat tone-{tone}"><div class="stat-v">{value}</div>
      <div class="stat-l">{label}</div>{f'<div class="stat-n">{note}</div>' if note else ''}</div>'''


CSS = r"""
:root{
--paper:oklch(14% 0.025 250);--paper-2:oklch(18% 0.03 250);--paper-3:oklch(22% 0.035 250);
--ink:oklch(96% 0.01 250);--ink-2:oklch(82% 0.015 250);--mut:oklch(60% 0.025 250);
--line:oklch(28% 0.03 250);--line-soft:oklch(22% 0.025 250);
--accent:oklch(74% 0.14 60);--accent-2:oklch(65% 0.18 30);--accent-soft:oklch(74% 0.14 60 / 0.12);
--blue:#6ea8fe;--green:oklch(72% 0.15 150);--red:oklch(68% 0.17 25);--violet:#b89cff;
--maxw:1120px;
--font-display:"Fraunces","Iowan Old Style",Georgia,serif;
--font-body:"Inter",-apple-system,'Segoe UI',Roboto,system-ui,sans-serif;
--font-mono:"JetBrains Mono",ui-monospace,Consolas,monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font-body);line-height:1.7;
-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;position:relative}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
background:radial-gradient(ellipse 70% 50% at 85% 3%,oklch(74% 0.14 60 / 0.06),transparent 60%),
 radial-gradient(ellipse 65% 40% at 5% 92%,oklch(55% 0.18 240 / 0.09),transparent 60%)}
body>*{position:relative;z-index:1}
.serif{font-family:var(--font-display);font-style:italic;font-weight:300}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 24px}
.prose{max-width:72ch}
.prose p{margin:0 0 16px;font-size:17px;color:var(--ink-2)}
.prose p .lead-in{font-weight:600;color:var(--ink)}
/* progress + nav */
#prog{position:fixed;top:0;left:0;height:3px;background:var(--accent);width:0;z-index:60}
nav.top{position:sticky;top:0;z-index:50;background:oklch(12% 0.022 250 / 0.82);
backdrop-filter:saturate(140%) blur(14px);border-bottom:1px solid var(--line)}
nav.top .wrap{display:flex;gap:6px;align-items:center;height:54px;overflow-x:auto}
nav.top a{font-size:12.5px;color:var(--ink-2);text-decoration:none;padding:6px 10px;
border-radius:8px;white-space:nowrap;font-family:var(--font-mono);letter-spacing:.01em;
transition:background .2s,color .2s}
nav.top a:hover{background:var(--paper-3);color:var(--ink)}
nav.top .brand{font-family:var(--font-display);font-style:italic;font-weight:400;
font-size:16px;color:var(--ink);margin-right:10px}
nav.top .langs{margin-left:auto;display:flex;gap:2px;flex-shrink:0}
nav.top .langs a{font-family:var(--font-mono);font-size:11px;letter-spacing:.06em;
padding:5px 10px;border-radius:999px;color:var(--mut)}
nav.top .langs a.on{background:var(--accent);color:oklch(18% 0.04 60);font-weight:600}
/* hero */
header.hero{padding:96px 0 70px;border-bottom:1px solid var(--line);position:relative;overflow:hidden}
header.hero::before{content:"";position:absolute;inset:0;z-index:-1;pointer-events:none;
background:radial-gradient(ellipse 50% 60% at 82% 12%,oklch(74% 0.14 60 / 0.10),transparent 60%),
 radial-gradient(ellipse 45% 55% at 8% 80%,oklch(60% 0.18 240 / 0.10),transparent 60%)}
.kicker{font-family:var(--font-mono);font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;
font-weight:500;color:var(--accent)}
header.hero h1{font-family:var(--font-display);font-style:italic;font-weight:300;
font-size:clamp(40px,7vw,76px);line-height:1.0;margin:16px 0 20px;letter-spacing:-.03em}
header.hero .thesis{font-size:20px;max-width:64ch;color:var(--ink-2);line-height:1.5}
header.hero .src{margin-top:24px;font-family:var(--font-mono);font-size:12px;color:var(--mut);letter-spacing:.01em}
.hero-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin-top:44px;
border-top:1px solid var(--line)}
.hero-strip .hs{padding:20px 18px 6px;border-right:1px solid var(--line)}
.hero-strip .hs:last-child{border-right:0}
.hero-strip .v{font-family:var(--font-display);font-style:italic;font-weight:400;font-size:30px;
color:var(--accent);letter-spacing:-.02em}
.hero-strip .l{font-size:12.5px;color:var(--mut);margin-top:4px}
/* sections */
section{padding:72px 0}
section.alt{background:oklch(16% 0.027 250);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
section h2{font-family:var(--font-display);font-style:italic;font-weight:300;
font-size:clamp(28px,4vw,46px);line-height:1.06;margin:10px 0 10px;letter-spacing:-.025em}
.lede{font-size:19px;color:var(--ink-2);max-width:74ch;margin:0 0 26px;line-height:1.55}
.fig{background:var(--paper-2);border:1px solid var(--line);border-radius:16px;padding:14px 14px 6px;
margin:24px 0}
.figcap{font-family:var(--font-mono);font-size:12px;color:var(--mut);padding:6px 6px 10px;line-height:1.5}
.figcap b{color:var(--ink-2);font-weight:500}
/* findings grid */
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin:8px 0 6px}
.stat{background:linear-gradient(180deg,var(--paper-2),var(--paper));border:1px solid var(--line);
border-left:3px solid var(--blue);border-radius:12px;padding:18px 20px}
.stat-v{font-family:var(--font-display);font-style:italic;font-weight:400;font-size:32px;letter-spacing:-.02em;color:var(--ink)}
.stat-l{font-size:14px;color:var(--ink);font-weight:500;margin-top:4px}
.stat-n{font-family:var(--font-mono);font-size:11.5px;color:var(--mut);margin-top:6px;line-height:1.45}
.tone-green{border-left-color:var(--green)}.tone-red{border-left-color:var(--red)}
.tone-amber{border-left-color:var(--accent)}.tone-violet{border-left-color:var(--violet)}
.tone-ink{border-left-color:var(--ink-2)}
.tone-amber .stat-v,.tone-green .stat-v,.tone-red .stat-v,.tone-violet .stat-v{color:var(--accent)}
.tone-green .stat-v{color:var(--green)}.tone-red .stat-v{color:var(--red)}.tone-violet .stat-v{color:var(--violet)}
/* callout / pullquote */
.callout{border-left:3px solid var(--accent);background:var(--accent-soft);padding:16px 22px;
border-radius:0 12px 12px 0;margin:26px 0;font-size:16.5px;color:var(--ink-2)}
.callout b{color:var(--ink)}
.pull{font-family:var(--font-display);font-style:italic;font-weight:300;font-size:27px;line-height:1.3;
color:var(--ink);border-top:1px solid var(--accent);border-bottom:1px solid var(--line);
padding:22px 0;margin:34px 0;max-width:32ch}
.grid2{display:grid;grid-template-columns:1.05fr .95fr;gap:36px;align-items:center}
@media(max-width:880px){.grid2{grid-template-columns:1fr}.hero-strip{grid-template-columns:repeat(2,1fr)}}
/* dossie 6x1 — warm-tinted midnight */
section.dossie{background:linear-gradient(180deg,oklch(17% 0.04 60),oklch(14% 0.03 280));
border-top:1px solid var(--accent)}
section.dossie .kicker{color:var(--accent)}
.scn{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}
.scn .c{background:oklch(20% 0.035 260 / 0.7);border:1px solid var(--line);border-radius:14px;padding:20px}
.scn .c h4{margin:0 0 8px;font-family:var(--font-display);font-style:italic;font-weight:400;
font-size:18px;color:var(--accent)}
.scn .c.up h4{color:var(--green)}.scn .c.down h4{color:var(--red)}
.scn .c p{margin:0;font-size:14.5px;color:var(--ink-2);line-height:1.6}
@media(max-width:880px){.scn{grid-template-columns:1fr}}
.cit a{color:var(--accent);text-decoration:none;font-weight:600}
.refs{margin:10px 0 0;padding-left:22px;color:var(--ink-2);font-size:13.5px;line-height:1.65}
.refs li{margin-bottom:9px}.refs li b{color:var(--ink)}
.refs a{color:var(--blue);text-decoration:none}.refs a:hover{text-decoration:underline}
.refs li[id]:target{background:var(--accent-soft);border-radius:5px;padding:2px 6px}
/* tables */
table.tbl{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:12px;
background:var(--paper-2);border:1px solid var(--line);border-radius:12px;overflow:hidden;
font-family:var(--font-mono)}
.tbl th,.tbl td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}
.tbl th{background:var(--paper-3);color:var(--ink);font-weight:500;cursor:pointer;user-select:none;
position:sticky;top:54px;letter-spacing:.02em;text-transform:uppercase;font-size:11px}
.tbl td{color:var(--ink-2)}
.tbl td.num{text-align:right;font-variant-numeric:tabular-nums}
.tbl td.mono,.mono{font-family:var(--font-mono)}
.tbl td.mono{color:var(--accent)}
.tbl tr:hover td{background:var(--paper-3)}
.tbl .small{font-size:11px;color:var(--mut)}
footer{background:oklch(11% 0.02 250);color:var(--mut);padding:54px 0;font-size:13px;
border-top:1px solid var(--line);font-family:var(--font-mono);line-height:1.7}
footer b{font-family:var(--font-display);font-style:italic;font-weight:400;color:var(--ink);font-size:16px}
footer a{color:var(--blue)}
.disc{font-size:12px;color:var(--mut);font-style:italic;margin-top:16px;font-family:var(--font-body)}
/* glossary tooltips for untranslatable Brazilian terms */
.term{border-bottom:1px dotted var(--accent);cursor:help;position:relative;color:var(--ink);
font-weight:500;font-style:inherit}
.term::after{content:attr(data-tip);position:absolute;left:0;bottom:150%;width:max-content;max-width:300px;
background:var(--paper-3);color:var(--ink-2);border:1px solid var(--line);border-radius:12px;padding:11px 13px;
font-family:var(--font-body);font-style:normal;font-weight:400;font-size:13px;line-height:1.5;white-space:normal;
box-shadow:0 12px 34px rgba(0,0,0,.5);opacity:0;visibility:hidden;transform:translateY(5px);
transition:opacity .18s ease,transform .18s ease;z-index:40;pointer-events:none}
.term:hover::after,.term:focus::after,.term:focus-visible::after{opacity:1;visibility:visible;transform:translateY(0)}
.term:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
@media(max-width:560px){.term::after{max-width:74vw}}
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

# ----------------------------------------------------------------- translations
LANGS = ("pt", "en", "es")
LANG_NAME = {"pt": "Português", "en": "English", "es": "Español"}
FILENAME = {"pt": "CLT-BRASIL_relatorio_pt.html",
            "en": "CLT-BRASIL_relatorio_en.html",
            "es": "CLT-BRASIL_relatorio_es.html"}

STR_ALL = {
    "pt": {
        "html_lang": "pt-br",
        "title": "CLT em Movimento · Emprego Formal no Brasil (Novo CAGED/MTE)",
        # nav
        "nav_panorama": "Panorama", "nav_traj": "Trajetória", "nav_geo": "Geografia",
        "nav_seas": "Sazonalidade", "nav_proj": "Projeção", "nav_dossie": "Dossiê 6×1",
        "nav_states": "Estados",
        "brand": "CLT em Movimento",
        # hero
        "hero_kicker": "Mercado de trabalho formal · Novo CAGED / {t_mte}",
        "hero_title": "CLT em Movimento",
        "hero_thesis": ("Seis anos de empregos com carteira assinada no Brasil, do colapso da "
                        "pandemia à desaceleração de 2025, e o que as projeções e o debate do "
                        "<b>fim da escala {t_6x1}</b> sinalizam para o trabalho formal até 2027."),
        "hs1": "vínculos celetistas líquidos criados desde jan/2020",
        "hs2": "saldo acumulado nos últimos 12 meses",
        "hs3": "das admissões são contratadas a 44h (universo da 6×1)",
        "hs4": "saldo projetado para jun/2027 (modelo {nat_best})",
        "src": ("Fonte: microdados do {t_caged} ({t_pdet}/Ministério do Trabalho e Emprego) · "
                "Período {span} · 27 UFs · Elaboração própria · {gen}"),
        # panorama
        "pan_kicker": "Panorama", "pan_h2": "O retrato de hoje",
        "pan_lede": ("O emprego formal brasileiro vive um momento de <b>expansão em desaceleração</b>: "
                     "ainda cria vagas todos os meses, mas em ritmo menor que no auge da retomada pós-pandemia."),
        "pan_p1": ("<span class=\"lead-in\">A recuperação foi real, mas perdeu fôlego.</span> Entre {span}, o saldo "
                   "acumulado somou <b>{estoque}</b> vínculos celetistas líquidos. Nos últimos 12 meses o país "
                   "gerou <b>{acc12}</b> empregos formais, ante {acc_prev12} nos 12 meses anteriores, "
                   "uma variação de <b>{var_pct}</b> que confirma a perda de tração em um "
                   "cenário de juros elevados e crédito mais caro."),
        "f_last": "Saldo do último mês", "f_acc12": "Saldo · 12 meses",
        "f_acc12_n": "admissões − desligamentos", "f_share": "do saldo nacional",
        "f_share_n": "concentrado em São Paulo", "f_seas": "Força da sazonalidade",
        "f_seas_n": "padrão anual marcante (0–1)",
        # trajetoria
        "tr_kicker": "Trajetória", "tr_h2": "A montanha-russa de seis anos",
        "tr_lede": ("Da queda livre de 2020 à normalização recente: a série mensal conta a história "
                    "econômica do país melhor que qualquer manchete."),
        "tr_cap": ("<b>Saldo mensal de empregos formais ({t_clt}) no Brasil.</b> Barras: saldo do mês "
                   "(verde positivo, vermelho negativo). Linha: média móvel de 12 meses. Fonte: Novo CAGED/MTE."),
        "tr_p1": ("<span class=\"lead-in\">{trough_m} é o fundo do poço.</span> No primeiro choque da pandemia, "
                  "o país fechou <b>{trough}</b> vagas formais em um único mês, o pior resultado da série. "
                  "Seguiu-se uma recuperação em formato de “V”, com saldos recordes em 2021 e 2022 à medida que a "
                  "economia reabria."),
        "tr_p2": ("Desde então, a média móvel se acomodou em um <b>platô positivo</b>: o Brasil continua criando "
                  "empregos, mas o ímpeto arrefeceu em 2024–2026. O padrão dente-de-serra que se repete todo ano "
                  "não é ruído; é <b>sazonalidade</b>, e ela tem nome e sobrenome setorial, como veremos adiante."),
        "tr_pull": ("Cada dezembro o Brasil demite; cada começo de ano, recontrata. O calendário do emprego "
                    "é quase tão previsível quanto o das estações."),
        # geografia
        "geo_kicker": "Geografia", "geo_h2": "Cinco Brasis no mesmo mapa",
        "geo_lede": ("A geração de empregos formais é profundamente concentrada. O Sudeste responde por "
                     "cerca de <b>{sudeste_share}</b> do saldo recente, e São Paulo sozinho por <b>{sp_share}</b>."),
        "geo_cap": ("<b>Saldo acumulado em 12 meses, por região.</b> Soma móvel, leitura do estoque "
                    "de vagas geradas no último ano em cada região."),
        "geo_explore": "Explore a série e a projeção de cada estado:",
        "geo_cap2": ("<b>Saldo mensal e previsão até jun/2027 por UF</b> (melhor modelo, IC 95%). "
                     "Use o seletor no canto superior direito."),
        # sazonalidade
        "se_kicker": "Sazonalidade", "se_h2": "O calendário do emprego",
        "se_lede": ("O saldo médio por mês revela um relógio anual: o país contrata no primeiro semestre "
                    "e nas safras, e demite em dezembro. O comércio e os serviços ditam o ritmo."),
        "se_cap": ("<b>Saldo médio por mês do ano e UF.</b> Verde = geração líquida típica; "
                   "vermelho = perdas. A faixa vermelha de dezembro é universal."),
        "se_p1": ("Esse padrão é dominado por setores intensivos em mão de obra e contratos de jornada cheia, "
                  "exatamente os que estão no centro do debate sobre a escala <b>6×1</b>. É para ele que olhamos a seguir."),
        # projeção
        "pr_kicker": "Projeção", "pr_h2": "Para onde vamos",
        "pr_lede": ("Três modelos, SARIMA, ETS (Holt-Winters) e <i>Seasonal Naive</i>, projetam a série "
                    "até <b>junho de 2027</b>. O selecionado por validação foi o <b>{nat_best}</b>."),
        "pr_cap": ("<b>Brasil, histórico recente e previsões até jun/2027.</b> Linha cheia: modelo "
                   "selecionado; tracejadas: alternativos; faixa: intervalo de confiança de 95%."),
        "pr_p1": ("<span class=\"lead-in\">A leitura central é de estabilidade sazonal, não de aceleração.</span> "
                  "Para jun/2027 o modelo projeta saldo de <b>{fc_end}</b> vagas, com intervalo de 95% entre "
                  "<b>{fc_lo}</b> e <b>{fc_hi}</b>, amplitude que reflete a forte sazonalidade e a "
                  "<b>heterocedasticidade</b> (variância instável) detectada na série."),
        "pr_callout": ("⚠️ <b>As projeções pressupõem o arcabouço atual.</b> Uma mudança estrutural como "
                       "o fim da escala 6×1 seria uma quebra que estes modelos, treinados no passado, não antecipam, e é "
                       "por isso que ela merece um capítulo à parte."),
        # dossiê
        "do_kicker": "Dossiê · Reforma da jornada", "do_h2": "O fim da escala 6×1",
        "do_lede": ("A proposta de substituir a jornada <b>{t_6x1}</b> (seis dias de trabalho, um de descanso) "
                    "por <b>{t_5x2}</b> mobiliza o Congresso e as ruas. O que os microdados do CAGED dizem sobre quem seria "
                    "afetado, e o que esperar."),
        "do_p1": ("<span class=\"lead-in\">O que está em jogo.</span> A escala 6×1 organiza a semana em seis "
                  "jornadas de trabalho e uma de descanso, tipicamente somando as <b>44 horas semanais</b> permitidas "
                  "pela Constituição. A PEC do fim da 6×1 propõe limitar a semana a cinco dias (5×2) e reduzir a "
                  "jornada, sem corte de salário. É a maior discussão sobre tempo de trabalho no país desde 1988."),
        "do_cap_hours": ("<b>Distribuição das admissões formais por jornada contratada (2020–2026).</b> "
                         "A barra destacada, <b>44h</b>, é o universo típico da 6×1."),
        "do_f_p44": "das admissões a 44h", "do_f_p44_n": "a jornada máxima legal, núcleo da 6×1",
        "do_f_full": "a 40h ou mais", "do_f_full_n": "predomínio do contrato de jornada cheia",
        "do_p2": ("Os dados de contratação mostram que a jornada de <b>44 horas é a norma</b>, "
                  "não a exceção: cerca de <b>{pct44}</b> de todas as admissões formais desde 2020 foram "
                  "pactuadas no teto constitucional. Reduzir a escala atinge, portanto, o contrato <i>modal</i> do "
                  "mercado de trabalho brasileiro."),
        "do_lede2": ("Mas a exposição é desigual entre setores. O mapa abaixo cruza "
                     "<b>quão dependente de jornada de 44h</b> é cada setor (eixo horizontal) com <b>quanto emprego ele "
                     "gera</b> (eixo vertical)."),
        "do_cap_exp": ("<b>Mapa de exposição setorial à escala 6×1.</b> Cada bolha é um setor ({t_cnae}); "
                       "horizontal: % das admissões a 44h; vertical: saldo de 12 meses; tamanho: volume de admissões. "
                       "Linha tracejada: média nacional de {media44} a 44h."),
        "do_p3": ("<span class=\"lead-in\">Os mais expostos são também grandes empregadores.</span> Setores como "
                  "<b>{exposed_list}</b> combinam alta dependência da jornada de 44h com geração relevante de vagas. "
                  "Os setores acima da média de exposição respondem por cerca de <b>{high_adm_share}</b> das "
                  "contratações formais recentes, ou seja, a reforma incide justamente sobre a engrenagem que mais contrata."),
        "do_h3a": "O que esperar, à luz da evidência",
        "do_h3a_p": ("Não há consenso, e o efeito final depende do desenho (corte de jornada, prazo de "
                     "transição, compensações). Cruzamos os cenários mais discutidos com a literatura econômica "
                     "revisada por pares (numerada ao final):"),
        "scn_up_h": "↑ Argumentos a favor",
        "scn_up_p": ("A produtividade <i>por hora</i> tende a subir quando se cortam jornadas longas: em dados "
                     "empíricos o produto cresce menos que proporcionalmente às horas<sup class=\"cit\">"
                     "<a href=\"#ref1\">1</a>,<a href=\"#ref2\">2</a></sup>, e escalas comprimidas elevam satisfação "
                     "e atitudes no trabalho<sup class=\"cit\"><a href=\"#ref4\">4</a></sup>. Há ainda ganho de saúde "
                     "e sono ao reduzir jornadas extensas<sup class=\"cit\"><a href=\"#ref7\">7</a>,"
                     "<a href=\"#ref8\">8</a></sup>."),
        "scn_down_h": "↓ Riscos apontados",
        "scn_down_p": ("Eleva o custo da hora em setores intensivos em mão de obra (comércio, alimentação, "
                       "serviços); empregadores tendem a reagir ajustando salário-base e jornada<sup class=\"cit\">"
                       "<a href=\"#ref5\">5</a></sup>. A experiência francesa das 35h não produziu efeito positivo "
                       "robusto sobre o emprego<sup class=\"cit\"><a href=\"#ref3\">3</a></sup>, sugerindo cautela "
                       "quanto a ganhos automáticos de postos."),
        "scn_adj_h": "⟳ Ajuste provável",
        "scn_adj_p": ("Reorganização de escalas, banco de horas, mais turnos parciais e contratações 5×2 a 40h"
                      "<sup class=\"cit\"><a href=\"#ref4\">4</a>,<a href=\"#ref6\">6</a></sup>. O peso recairia "
                      "sobre os setores do quadrante direito do mapa acima, os de maior exposição a 44h."),
        "scn_mod_h": "📊 Para o modelo",
        "scn_mod_p": ("Uma transição alteraria o nível e a sazonalidade da série, uma quebra estrutural. As "
                      "projeções deste relatório servem de <b>linha de base</b> (cenário “sem reforma”) contra a "
                      "qual medir o efeito."),
        "do_h3b": "Evidência acadêmica",
        "do_h3b_p": ("Referências localizadas pelo protocolo de busca científica <i>paper-lookup</i> "
                     "(k-dense scientific-agent-skills) via OpenAlex, sem fabricação: cada item tem DOI verificável."),
        "no_doi": "sem DOI",
        "do_disc": ("Análise de natureza econômica e exploratória, baseada na composição das contratações "
                    "formais. O CAGED registra horas contratadas, não o número de dias trabalhados; 44h é usado como "
                    "proxy do universo 6×1. A evidência citada vem de contextos institucionais distintos do brasileiro "
                    "e não deve ser lida como previsão do efeito da PEC, tampouco como aconselhamento."),
        # estados
        "st_kicker": "Apêndice", "st_h2": "Síntese por estado",
        "st_lede": ("Saldo recente, modelo selecionado, projeção para jun/2027 com intervalo de confiança "
                    "e diagnósticos por UF. Clique nos cabeçalhos para ordenar."),
        "th_uf": "UF", "th_state": "Estado", "th_saldo12": "Saldo 12m", "th_model": "Modelo",
        "th_prev": "Prev. jun/27", "th_ci": "IC 95%", "th_seas": "Sazonal.", "th_het": "Heter.",
        "yes": "sim", "no": "não",
        "th_cnae": "CNAE", "th_sector": "Setor", "th_p44": "% a 44h",
        "th_adm12": "Admissões 12m",
        # footer
        "ft_body": ("<b>CLT em Movimento</b> — elaborado a partir dos microdados do Novo CAGED (PDET/Ministério "
                    "do Trabalho e Emprego).<br>Metodologia: saldo = Σ(saldomovimentação) por competência (reproduz "
                    "a série oficial “sem ajuste”); modelos SARIMA, ETS (Holt-Winters) e Seasonal Naive, seleção por "
                    "RMSE em validação de 12 meses; jornada via campo <i>horascontratuais</i>. Revisão de literatura "
                    "conduzida pelo protocolo <i>paper-lookup</i> (k-dense scientific-agent-skills) sobre a base "
                    "OpenAlex.<br>Relatório autocontido (offline) · Período {span} · Gerado em {gen}."),
        # plotly labels
        "lg_monthly": "Saldo mensal", "lg_ma12": "Média móvel 12m", "lg_hist": "Histórico",
        "lg_ci": "IC 95%", "lg_forecast": "Previsão", "an_peak": "Pico",
        "yt_saldo": "Saldo de empregos (CLT)", "yt_acc12": "Saldo acumulado 12 meses",
        "yt_saldo12": "Saldo de empregos · últimos 12 meses",
        "cb_saldo": "Saldo médio",
        "xt_share": "% das admissões formais (2020–2026)",
        "xt_exposure": "% das admissões contratadas a 44h (exposição à escala 6×1)",
        "hv_p44": "% a 44h", "hv_saldo12": "Saldo 12m", "hv_adm12": "Admissões 12m",
        "hours_bands": ["Parcial (≤30h)", "30–39h", "40h", "41–43h", "44h (jornada máxima)"],
        "reg": {"Norte": "Norte", "Nordeste": "Nordeste", "Sudeste": "Sudeste",
                "Sul": "Sul", "Centro-Oeste": "Centro-Oeste"},
    },
    "en": {
        "html_lang": "en",
        "title": "CLT in Motion · Formal Employment in Brazil (Novo CAGED/MTE)",
        "nav_panorama": "Overview", "nav_traj": "Trajectory", "nav_geo": "Geography",
        "nav_seas": "Seasonality", "nav_proj": "Forecast", "nav_dossie": "6×1 Dossier",
        "nav_states": "States",
        "brand": "CLT in Motion",
        "hero_kicker": "Formal labor market · Novo CAGED / {t_mte}",
        "hero_title": "CLT in Motion",
        "hero_thesis": ("Six years of formally registered jobs in Brazil, from the pandemic collapse to the "
                        "2025 slowdown, and what the forecasts and the debate over <b>ending the {t_6x1} work "
                        "schedule</b> signal for formal employment through 2027."),
        "hs1": "net formal (CLT) jobs created since Jan/2020",
        "hs2": "net balance over the last 12 months",
        "hs3": "of hires are contracted at 44h (the 6×1 universe)",
        "hs4": "balance projected for Jun/2027 ({nat_best} model)",
        "src": ("Source: {t_caged} microdata ({t_pdet} / Ministry of Labor and Employment) · "
                "Period {span} · 27 states · Author's analysis · {gen}"),
        "pan_kicker": "Overview", "pan_h2": "Today's snapshot",
        "pan_lede": ("Brazil's formal employment is in a phase of <b>decelerating expansion</b>: it still adds "
                     "jobs every month, but at a slower pace than at the peak of the post-pandemic rebound."),
        "pan_p1": ("<span class=\"lead-in\">The recovery was real, but it ran out of steam.</span> Between {span}, the "
                   "cumulative balance reached <b>{estoque}</b> net formal jobs. Over the last 12 months the country "
                   "created <b>{acc12}</b> formal jobs, versus {acc_prev12} in the previous 12 months, "
                   "a change of <b>{var_pct}</b> that confirms the loss of traction amid high interest rates "
                   "and costlier credit."),
        "f_last": "Latest monthly balance", "f_acc12": "Balance · 12 months",
        "f_acc12_n": "hires − separations", "f_share": "of the national balance",
        "f_share_n": "concentrated in São Paulo", "f_seas": "Seasonal strength",
        "f_seas_n": "marked annual pattern (0–1)",
        "tr_kicker": "Trajectory", "tr_h2": "A six-year roller coaster",
        "tr_lede": ("From the free fall of 2020 to the recent normalization: the monthly series tells the country's "
                    "economic story better than any headline."),
        "tr_cap": ("<b>Monthly balance of formal ({t_clt}) jobs in Brazil.</b> Bars: monthly balance "
                   "(green positive, red negative). Line: 12-month moving average. Source: Novo CAGED/MTE."),
        "tr_p1": ("<span class=\"lead-in\">{trough_m} is the bottom.</span> In the first shock of the pandemic, the "
                  "country shed <b>{trough}</b> formal jobs in a single month, the worst result in the series. "
                  "A V-shaped recovery followed, with record balances in 2021 and 2022 as the economy reopened."),
        "tr_p2": ("Since then the moving average has settled into a <b>positive plateau</b>: Brazil keeps creating "
                  "jobs, but momentum cooled in 2024–2026. The sawtooth pattern that recurs every year is not noise; "
                  "it is <b>seasonality</b>, and it has a sectoral signature, as we will see."),
        "tr_pull": ("Every December Brazil lays off; every start of the year, it rehires. The employment calendar "
                    "is almost as predictable as the seasons."),
        "geo_kicker": "Geography", "geo_h2": "Five Brazils on one map",
        "geo_lede": ("Formal job creation is deeply concentrated. The Southeast accounts for about "
                     "<b>{sudeste_share}</b> of the recent balance, and São Paulo alone for <b>{sp_share}</b>."),
        "geo_cap": ("<b>Cumulative 12-month balance, by region.</b> Rolling sum, a read on the stock of jobs "
                    "created over the past year in each region."),
        "geo_explore": "Explore each state's series and forecast:",
        "geo_cap2": ("<b>Monthly balance and forecast to Jun/2027 by state</b> (best model, 95% CI). "
                     "Use the selector in the top-right corner."),
        "se_kicker": "Seasonality", "se_h2": "The employment calendar",
        "se_lede": ("The average balance per month reveals an annual clock: the country hires in the first half "
                    "and during harvests, and lays off in December. Retail and services set the rhythm."),
        "se_cap": ("<b>Average balance by month of the year and state.</b> Green = typical net creation; "
                   "red = losses. The red December band is universal."),
        "se_p1": ("This pattern is dominated by labor-intensive sectors and full-time contracts, precisely those at "
                  "the center of the debate over the <b>6×1</b> schedule. That is what we turn to next."),
        "pr_kicker": "Forecast", "pr_h2": "Where we are heading",
        "pr_lede": ("Three models, SARIMA, ETS (Holt-Winters) and <i>Seasonal Naive</i>, project the series "
                    "through <b>June 2027</b>. The one selected by validation was <b>{nat_best}</b>."),
        "pr_cap": ("<b>Brazil, recent history and forecasts to Jun/2027.</b> Solid line: selected model; "
                   "dashed: alternatives; band: 95% confidence interval."),
        "pr_p1": ("<span class=\"lead-in\">The central reading is seasonal stability, not acceleration.</span> "
                  "For Jun/2027 the model projects a balance of <b>{fc_end}</b> jobs, with a 95% interval between "
                  "<b>{fc_lo}</b> and <b>{fc_hi}</b>, a width that reflects the strong seasonality and the "
                  "<b>heteroskedasticity</b> (unstable variance) detected in the series."),
        "pr_callout": ("⚠️ <b>The forecasts assume the current framework.</b> A structural change such as ending "
                       "the 6×1 schedule would be a break that these models, trained on the past, do not anticipate, "
                       "which is why it deserves a chapter of its own."),
        "do_kicker": "Dossier · Working-time reform", "do_h2": "Ending the 6×1 schedule",
        "do_lede": ("The proposal to replace the <b>{t_6x1}</b> schedule (six days of work, one of rest) with "
                    "<b>{t_5x2}</b> is mobilizing Congress and the streets. What the CAGED microdata say about who would "
                    "be affected, and what to expect."),
        "do_p1": ("<span class=\"lead-in\">What is at stake.</span> The 6×1 schedule organizes the week into six "
                  "workdays and one rest day, typically adding up to the <b>44 weekly hours</b> allowed by the "
                  "Constitution. The constitutional amendment to end 6×1 proposes capping the week at five days "
                  "(5×2) and reducing hours, with no pay cut. It is the biggest debate over working time in the "
                  "country since 1988."),
        "do_cap_hours": ("<b>Distribution of formal hires by contracted weekly hours (2020–2026).</b> "
                         "The highlighted bar, <b>44h</b>, is the typical 6×1 universe."),
        "do_f_p44": "of hires at 44h", "do_f_p44_n": "the legal maximum, the core of 6×1",
        "do_f_full": "at 40h or more", "do_f_full_n": "full-time contracts predominate",
        "do_p2": ("Hiring data show that the <b>44-hour week is the norm</b>, not the exception: about "
                  "<b>{pct44}</b> of all formal hires since 2020 were agreed at the constitutional ceiling. "
                  "Cutting the schedule therefore hits the <i>modal</i> contract of the Brazilian labor market."),
        "do_lede2": ("But exposure is uneven across sectors. The map below crosses <b>how dependent on a 44h week</b> "
                     "each sector is (horizontal axis) with <b>how much employment it generates</b> (vertical axis)."),
        "do_cap_exp": ("<b>Map of sectoral exposure to the 6×1 schedule.</b> Each bubble is a sector ({t_cnae}); "
                       "horizontal: % of hires at 44h; vertical: 12-month balance; size: hiring volume. "
                       "Dashed line: national average of {media44} at 44h."),
        "do_p3": ("<span class=\"lead-in\">The most exposed are also large employers.</span> Sectors such as "
                  "<b>{exposed_list}</b> combine heavy dependence on the 44h week with significant job creation. "
                  "Sectors above the average exposure account for about <b>{high_adm_share}</b> of recent formal "
                  "hiring; in other words, the reform falls precisely on the engine that hires the most."),
        "do_h3a": "What to expect, in light of the evidence",
        "do_h3a_p": ("There is no consensus, and the final effect depends on the design (size of the cut, "
                     "transition period, offsets). We cross the most-discussed scenarios with peer-reviewed economic "
                     "literature (numbered at the end):"),
        "scn_up_h": "↑ Arguments in favor",
        "scn_up_p": ("Productivity <i>per hour</i> tends to rise when long shifts are cut: in empirical data output "
                     "grows less than proportionally to hours<sup class=\"cit\"><a href=\"#ref1\">1</a>,"
                     "<a href=\"#ref2\">2</a></sup>, and compressed schedules raise satisfaction and work attitudes"
                     "<sup class=\"cit\"><a href=\"#ref4\">4</a></sup>. There are also health and sleep gains from "
                     "reducing long shifts<sup class=\"cit\"><a href=\"#ref7\">7</a>,<a href=\"#ref8\">8</a></sup>."),
        "scn_down_h": "↓ Risks raised",
        "scn_down_p": ("It raises the hourly cost in labor-intensive sectors (retail, food service, services); "
                       "employers tend to react by adjusting base pay and hours<sup class=\"cit\">"
                       "<a href=\"#ref5\">5</a></sup>. France's 35-hour experience did not produce a robust positive "
                       "effect on employment<sup class=\"cit\"><a href=\"#ref3\">3</a></sup>, suggesting caution about "
                       "automatic job gains."),
        "scn_adj_h": "⟳ Likely adjustment",
        "scn_adj_p": ("Reorganization of shifts, hour banks, more part-time shifts and 5×2 hires at 40h"
                      "<sup class=\"cit\"><a href=\"#ref4\">4</a>,<a href=\"#ref6\">6</a></sup>. The burden would fall "
                      "on the sectors in the right-hand quadrant of the map above, those with the greatest 44h exposure."),
        "scn_mod_h": "📊 For the model",
        "scn_mod_p": ("A transition would change the level and seasonality of the series, a structural break. The "
                      "forecasts in this report serve as a <b>baseline</b> (the “no reform” scenario) against which to "
                      "measure the effect."),
        "do_h3b": "Academic evidence",
        "do_h3b_p": ("References located by the <i>paper-lookup</i> scientific search protocol "
                     "(k-dense scientific-agent-skills) via OpenAlex, with no fabrication: each item has a verifiable DOI."),
        "no_doi": "no DOI",
        "do_disc": ("An economic and exploratory analysis based on the composition of formal hiring. CAGED records "
                    "contracted hours, not the number of days worked; 44h is used as a proxy for the 6×1 universe. The "
                    "cited evidence comes from institutional contexts different from Brazil's and should not be read as "
                    "a forecast of the amendment's effect, nor as advice."),
        "st_kicker": "Appendix", "st_h2": "State-by-state summary",
        "st_lede": ("Recent balance, selected model, forecast for Jun/2027 with confidence interval and diagnostics "
                    "by state. Click the headers to sort."),
        "th_uf": "State", "th_state": "Name", "th_saldo12": "Balance 12m", "th_model": "Model",
        "th_prev": "Forecast Jun/27", "th_ci": "95% CI", "th_seas": "Seas.", "th_het": "Heter.",
        "yes": "yes", "no": "no",
        "th_cnae": "CNAE", "th_sector": "Sector", "th_p44": "% at 44h",
        "th_adm12": "Hires 12m",
        "ft_body": ("<b>CLT in Motion</b> — built from Novo CAGED microdata (PDET / Ministry of Labor and "
                    "Employment).<br>Methodology: balance = Σ(saldomovimentação) per period (reproduces the official "
                    "“unadjusted” series); SARIMA, ETS (Holt-Winters) and Seasonal Naive models, selected by RMSE in a "
                    "12-month validation; working time via the <i>horascontratuais</i> field. Literature review "
                    "conducted with the <i>paper-lookup</i> protocol (k-dense scientific-agent-skills) over the "
                    "OpenAlex base.<br>Self-contained report (offline) · Period {span} · Generated on {gen}."),
        "lg_monthly": "Monthly balance", "lg_ma12": "12m moving avg", "lg_hist": "History",
        "lg_ci": "95% CI", "lg_forecast": "Forecast", "an_peak": "Peak",
        "yt_saldo": "Job balance (CLT)", "yt_acc12": "Cumulative 12-month balance",
        "yt_saldo12": "Job balance · last 12 months",
        "cb_saldo": "Avg. balance",
        "xt_share": "% of formal hires (2020–2026)",
        "xt_exposure": "% of hires contracted at 44h (exposure to the 6×1 schedule)",
        "hv_p44": "% at 44h", "hv_saldo12": "Balance 12m", "hv_adm12": "Hires 12m",
        "hours_bands": ["Part-time (≤30h)", "30–39h", "40h", "41–43h", "44h (legal maximum)"],
        "reg": {"Norte": "North", "Nordeste": "Northeast", "Sudeste": "Southeast",
                "Sul": "South", "Centro-Oeste": "Central-West"},
    },
    "es": {
        "html_lang": "es",
        "title": "CLT en Movimiento · Empleo Formal en Brasil (Novo CAGED/MTE)",
        "nav_panorama": "Panorama", "nav_traj": "Trayectoria", "nav_geo": "Geografía",
        "nav_seas": "Estacionalidad", "nav_proj": "Proyección", "nav_dossie": "Dosier 6×1",
        "nav_states": "Estados",
        "brand": "CLT en Movimiento",
        "hero_kicker": "Mercado de trabajo formal · Novo CAGED / {t_mte}",
        "hero_title": "CLT en Movimiento",
        "hero_thesis": ("Seis años de empleo formal registrado en Brasil, del colapso de la pandemia a la "
                        "desaceleración de 2025, y lo que las proyecciones y el debate sobre el <b>fin de la "
                        "jornada {t_6x1}</b> señalan para el empleo formal hasta 2027."),
        "hs1": "vínculos formales (CLT) netos creados desde ene/2020",
        "hs2": "saldo acumulado en los últimos 12 meses",
        "hs3": "de las contrataciones son a 44h (el universo de la 6×1)",
        "hs4": "saldo proyectado para jun/2027 (modelo {nat_best})",
        "src": ("Fuente: microdatos del {t_caged} ({t_pdet} / Ministerio de Trabajo y Empleo) · "
                "Período {span} · 27 estados · Elaboración propia · {gen}"),
        "pan_kicker": "Panorama", "pan_h2": "La foto de hoy",
        "pan_lede": ("El empleo formal brasileño vive un momento de <b>expansión en desaceleración</b>: todavía "
                     "crea empleos cada mes, pero a un ritmo menor que en el auge de la recuperación pospandemia."),
        "pan_p1": ("<span class=\"lead-in\">La recuperación fue real, pero perdió fuelle.</span> Entre {span}, el "
                   "saldo acumulado sumó <b>{estoque}</b> vínculos formales netos. En los últimos 12 meses el país "
                   "generó <b>{acc12}</b> empleos formales, frente a {acc_prev12} en los 12 meses anteriores, "
                   "una variación de <b>{var_pct}</b> que confirma la pérdida de tracción en un escenario de tasas "
                   "altas y crédito más caro."),
        "f_last": "Saldo del último mes", "f_acc12": "Saldo · 12 meses",
        "f_acc12_n": "contrataciones − desvinculaciones", "f_share": "del saldo nacional",
        "f_share_n": "concentrado en São Paulo", "f_seas": "Fuerza de la estacionalidad",
        "f_seas_n": "patrón anual marcado (0–1)",
        "tr_kicker": "Trayectoria", "tr_h2": "La montaña rusa de seis años",
        "tr_lede": ("De la caída libre de 2020 a la normalización reciente: la serie mensual cuenta la historia "
                    "económica del país mejor que cualquier titular."),
        "tr_cap": ("<b>Saldo mensual de empleos formales ({t_clt}) en Brasil.</b> Barras: saldo del mes "
                   "(verde positivo, rojo negativo). Línea: media móvil de 12 meses. Fuente: Novo CAGED/MTE."),
        "tr_p1": ("<span class=\"lead-in\">{trough_m} es el fondo.</span> En el primer choque de la pandemia, el "
                  "país destruyó <b>{trough}</b> empleos formales en un solo mes, el peor resultado de la serie. "
                  "Siguió una recuperación en forma de “V”, con saldos récord en 2021 y 2022 a medida que la economía "
                  "reabría."),
        "tr_p2": ("Desde entonces la media móvil se asentó en una <b>meseta positiva</b>: Brasil sigue creando "
                  "empleo, pero el impulso se enfrió en 2024–2026. El patrón de dientes de sierra que se repite cada "
                  "año no es ruido; es <b>estacionalidad</b>, y tiene una firma sectorial, como veremos."),
        "tr_pull": ("Cada diciembre Brasil despide; cada inicio de año, recontrata. El calendario del empleo es "
                    "casi tan previsible como el de las estaciones."),
        "geo_kicker": "Geografía", "geo_h2": "Cinco Brasiles en un mismo mapa",
        "geo_lede": ("La creación de empleo formal está profundamente concentrada. El Sudeste responde por cerca de "
                     "<b>{sudeste_share}</b> del saldo reciente, y São Paulo solo por <b>{sp_share}</b>."),
        "geo_cap": ("<b>Saldo acumulado en 12 meses, por región.</b> Suma móvil, una lectura del stock de empleos "
                    "creados en el último año en cada región."),
        "geo_explore": "Explore la serie y la proyección de cada estado:",
        "geo_cap2": ("<b>Saldo mensual y previsión hasta jun/2027 por estado</b> (mejor modelo, IC 95%). "
                     "Use el selector en la esquina superior derecha."),
        "se_kicker": "Estacionalidad", "se_h2": "El calendario del empleo",
        "se_lede": ("El saldo medio por mes revela un reloj anual: el país contrata en el primer semestre y en las "
                    "cosechas, y despide en diciembre. El comercio y los servicios marcan el ritmo."),
        "se_cap": ("<b>Saldo medio por mes del año y estado.</b> Verde = creación neta típica; rojo = pérdidas. "
                   "La franja roja de diciembre es universal."),
        "se_p1": ("Este patrón está dominado por sectores intensivos en mano de obra y contratos de jornada completa, "
                  "justamente los que están en el centro del debate sobre la jornada <b>6×1</b>. Es lo que miramos a "
                  "continuación."),
        "pr_kicker": "Proyección", "pr_h2": "Hacia dónde vamos",
        "pr_lede": ("Tres modelos, SARIMA, ETS (Holt-Winters) y <i>Seasonal Naive</i>, proyectan la serie hasta "
                    "<b>junio de 2027</b>. El seleccionado por validación fue <b>{nat_best}</b>."),
        "pr_cap": ("<b>Brasil, historia reciente y previsiones hasta jun/2027.</b> Línea continua: modelo "
                   "seleccionado; discontinuas: alternativos; banda: intervalo de confianza del 95%."),
        "pr_p1": ("<span class=\"lead-in\">La lectura central es de estabilidad estacional, no de aceleración.</span> "
                  "Para jun/2027 el modelo proyecta un saldo de <b>{fc_end}</b> empleos, con un intervalo del 95% entre "
                  "<b>{fc_lo}</b> y <b>{fc_hi}</b>, una amplitud que refleja la fuerte estacionalidad y la "
                  "<b>heterocedasticidad</b> (varianza inestable) detectada en la serie."),
        "pr_callout": ("⚠️ <b>Las proyecciones suponen el marco actual.</b> Un cambio estructural como el fin de la "
                       "jornada 6×1 sería una ruptura que estos modelos, entrenados con el pasado, no anticipan, y por "
                       "eso merece un capítulo aparte."),
        "do_kicker": "Dosier · Reforma de la jornada", "do_h2": "El fin de la jornada 6×1",
        "do_lede": ("La propuesta de sustituir la jornada <b>{t_6x1}</b> (seis días de trabajo, uno de descanso) por "
                    "<b>{t_5x2}</b> moviliza al Congreso y a las calles. Lo que los microdatos del CAGED dicen sobre "
                    "quién se vería afectado, y qué esperar."),
        "do_p1": ("<span class=\"lead-in\">Lo que está en juego.</span> La jornada 6×1 organiza la semana en seis "
                  "días de trabajo y uno de descanso, sumando normalmente las <b>44 horas semanales</b> permitidas por "
                  "la Constitución. La enmienda para terminar con la 6×1 propone limitar la semana a cinco días (5×2) y "
                  "reducir la jornada, sin recorte salarial. Es el mayor debate sobre tiempo de trabajo en el país "
                  "desde 1988."),
        "do_cap_hours": ("<b>Distribución de las contrataciones formales por jornada contratada (2020–2026).</b> "
                         "La barra destacada, <b>44h</b>, es el universo típico de la 6×1."),
        "do_f_p44": "de las contrataciones a 44h", "do_f_p44_n": "la jornada máxima legal, núcleo de la 6×1",
        "do_f_full": "a 40h o más", "do_f_full_n": "predominio del contrato de jornada completa",
        "do_p2": ("Los datos de contratación muestran que la jornada de <b>44 horas es la norma</b>, no la excepción: "
                  "cerca del <b>{pct44}</b> de todas las contrataciones formales desde 2020 se pactaron en el techo "
                  "constitucional. Reducir la jornada afecta, por tanto, al contrato <i>modal</i> del mercado laboral "
                  "brasileño."),
        "do_lede2": ("Pero la exposición es desigual entre sectores. El mapa de abajo cruza <b>cuán dependiente de la "
                     "jornada de 44h</b> es cada sector (eje horizontal) con <b>cuánto empleo genera</b> (eje vertical)."),
        "do_cap_exp": ("<b>Mapa de exposición sectorial a la jornada 6×1.</b> Cada burbuja es un sector ({t_cnae}); "
                       "horizontal: % de contrataciones a 44h; vertical: saldo de 12 meses; tamaño: volumen de "
                       "contrataciones. Línea discontinua: media nacional de {media44} a 44h."),
        "do_p3": ("<span class=\"lead-in\">Los más expuestos son también grandes empleadores.</span> Sectores como "
                  "<b>{exposed_list}</b> combinan alta dependencia de la jornada de 44h con una generación relevante de "
                  "empleo. Los sectores por encima de la media de exposición concentran cerca del <b>{high_adm_share}</b> "
                  "de las contrataciones formales recientes; es decir, la reforma recae justo sobre el engranaje que "
                  "más contrata."),
        "do_h3a": "Qué esperar, a la luz de la evidencia",
        "do_h3a_p": ("No hay consenso, y el efecto final depende del diseño (tamaño del recorte, plazo de transición, "
                     "compensaciones). Cruzamos los escenarios más discutidos con la literatura económica revisada por "
                     "pares (numerada al final):"),
        "scn_up_h": "↑ Argumentos a favor",
        "scn_up_p": ("La productividad <i>por hora</i> tiende a subir cuando se recortan jornadas largas: en datos "
                     "empíricos el producto crece menos que proporcionalmente a las horas<sup class=\"cit\">"
                     "<a href=\"#ref1\">1</a>,<a href=\"#ref2\">2</a></sup>, y las jornadas comprimidas elevan la "
                     "satisfacción y las actitudes en el trabajo<sup class=\"cit\"><a href=\"#ref4\">4</a></sup>. "
                     "También hay mejoras de salud y sueño al reducir jornadas extensas<sup class=\"cit\">"
                     "<a href=\"#ref7\">7</a>,<a href=\"#ref8\">8</a></sup>."),
        "scn_down_h": "↓ Riesgos señalados",
        "scn_down_p": ("Eleva el coste de la hora en sectores intensivos en mano de obra (comercio, alimentación, "
                       "servicios); los empleadores tienden a reaccionar ajustando salario base y jornada"
                       "<sup class=\"cit\"><a href=\"#ref5\">5</a></sup>. La experiencia francesa de las 35h no produjo "
                       "un efecto positivo robusto sobre el empleo<sup class=\"cit\"><a href=\"#ref3\">3</a></sup>, lo "
                       "que sugiere cautela frente a ganancias automáticas de puestos."),
        "scn_adj_h": "⟳ Ajuste probable",
        "scn_adj_p": ("Reorganización de turnos, bolsa de horas, más turnos parciales y contrataciones 5×2 a 40h"
                      "<sup class=\"cit\"><a href=\"#ref4\">4</a>,<a href=\"#ref6\">6</a></sup>. El peso recaería sobre "
                      "los sectores del cuadrante derecho del mapa anterior, los de mayor exposición a 44h."),
        "scn_mod_h": "📊 Para el modelo",
        "scn_mod_p": ("Una transición alteraría el nivel y la estacionalidad de la serie, una ruptura estructural. "
                      "Las proyecciones de este informe sirven de <b>línea base</b> (escenario “sin reforma”) contra "
                      "la cual medir el efecto."),
        "do_h3b": "Evidencia académica",
        "do_h3b_p": ("Referencias localizadas mediante el protocolo de búsqueda científica <i>paper-lookup</i> "
                     "(k-dense scientific-agent-skills) vía OpenAlex, sin fabricación: cada ítem tiene un DOI verificable."),
        "no_doi": "sin DOI",
        "do_disc": ("Análisis de naturaleza económica y exploratoria, basado en la composición de las contrataciones "
                    "formales. El CAGED registra horas contratadas, no el número de días trabajados; 44h se usa como "
                    "proxy del universo 6×1. La evidencia citada proviene de contextos institucionales distintos del "
                    "brasileño y no debe leerse como una previsión del efecto de la enmienda, ni como asesoramiento."),
        "st_kicker": "Apéndice", "st_h2": "Síntesis por estado",
        "st_lede": ("Saldo reciente, modelo seleccionado, proyección para jun/2027 con intervalo de confianza y "
                    "diagnósticos por estado. Haga clic en los encabezados para ordenar."),
        "th_uf": "Estado", "th_state": "Nombre", "th_saldo12": "Saldo 12m", "th_model": "Modelo",
        "th_prev": "Prev. jun/27", "th_ci": "IC 95%", "th_seas": "Estac.", "th_het": "Heter.",
        "yes": "sí", "no": "no",
        "th_cnae": "CNAE", "th_sector": "Sector", "th_p44": "% a 44h",
        "th_adm12": "Contrat. 12m",
        "ft_body": ("<b>CLT en Movimiento</b> — elaborado a partir de los microdatos del Novo CAGED (PDET / "
                    "Ministerio de Trabajo y Empleo).<br>Metodología: saldo = Σ(saldomovimentação) por período "
                    "(reproduce la serie oficial “sin ajuste”); modelos SARIMA, ETS (Holt-Winters) y Seasonal Naive, "
                    "selección por RMSE en validación de 12 meses; jornada vía el campo <i>horascontratuais</i>. "
                    "Revisión de literatura realizada con el protocolo <i>paper-lookup</i> (k-dense "
                    "scientific-agent-skills) sobre la base OpenAlex.<br>Informe autocontenido (offline) · "
                    "Período {span} · Generado el {gen}."),
        "lg_monthly": "Saldo mensual", "lg_ma12": "Media móvil 12m", "lg_hist": "Histórico",
        "lg_ci": "IC 95%", "lg_forecast": "Previsión", "an_peak": "Pico",
        "yt_saldo": "Saldo de empleos (CLT)", "yt_acc12": "Saldo acumulado 12 meses",
        "yt_saldo12": "Saldo de empleos · últimos 12 meses",
        "cb_saldo": "Saldo medio",
        "xt_share": "% de las contrataciones formales (2020–2026)",
        "xt_exposure": "% de las contrataciones a 44h (exposición a la jornada 6×1)",
        "hv_p44": "% a 44h", "hv_saldo12": "Saldo 12m", "hv_adm12": "Contrat. 12m",
        "hours_bands": ["Parcial (≤30h)", "30–39h", "40h", "41–43h", "44h (jornada máxima)"],
        "reg": {"Norte": "Norte", "Nordeste": "Nordeste", "Sudeste": "Sudeste",
                "Sul": "Sur", "Centro-Oeste": "Centro-Oeste"},
    },
}


GLOSSARY = {
    "pt": {
        "clt": ("CLT (Consolidação das Leis do Trabalho): o regime de emprego formal brasileiro, "
                "com carteira assinada e plenos direitos trabalhistas."),
        "caged": ("Novo CAGED (Cadastro Geral de Empregados e Desempregados): registro administrativo "
                  "mensal de admissões e desligamentos formais, mantido pelo Ministério do Trabalho."),
        "pdet": ("PDET (Programa de Disseminação das Estatísticas do Trabalho): o braço do Ministério "
                 "do Trabalho que publica os microdados do CAGED."),
        "mte": "MTE: Ministério do Trabalho e Emprego, órgão federal responsável pela política trabalhista.",
        "s6x1": ("Escala 6×1: jornada de seis dias de trabalho para um de descanso, comum no comércio "
                 "e nos serviços e tipicamente somando as 44 horas semanais."),
        "s5x2": "Escala 5×2: cinco dias de trabalho e dois de descanso por semana, o modelo proposto na reforma.",
        "cnae": ("CNAE (Classificação Nacional de Atividades Econômicas): o sistema oficial brasileiro de "
                 "classificação setorial das empresas."),
    },
    "en": {
        "clt": ("CLT (Consolidação das Leis do Trabalho, the Consolidation of Labor Laws): Brazil's "
                "formal, fully-registered employment regime with full labor rights."),
        "caged": ("Novo CAGED (Cadastro Geral de Empregados e Desempregados, the General Registry of "
                  "Employed and Unemployed): Brazil's monthly administrative record of formal hires and "
                  "separations, run by the Ministry of Labor."),
        "pdet": ("PDET (Programa de Disseminação das Estatísticas do Trabalho, the Labor Statistics "
                 "Dissemination Program): the Ministry of Labor arm that publishes the CAGED microdata."),
        "mte": "MTE: Brazil's Ministry of Labor and Employment, the federal body for labor policy.",
        "s6x1": ("6×1 schedule: a roster of six workdays for every rest day, common in Brazilian retail "
                 "and services and typically adding up to the 44-hour legal week."),
        "s5x2": "5×2 schedule: five workdays and two rest days per week, the model proposed by the reform.",
        "cnae": ("CNAE (Classificação Nacional de Atividades Econômicas, the National Classification of "
                 "Economic Activities): Brazil's official system for classifying business sectors."),
    },
    "es": {
        "clt": ("CLT (Consolidação das Leis do Trabalho, la Consolidación de las Leyes del Trabajo): el "
                "régimen de empleo formal registrado de Brasil, con plenos derechos laborales."),
        "caged": ("Novo CAGED (Cadastro Geral de Empregados e Desempregados, el Registro General de "
                  "Empleados y Desempleados): el registro administrativo mensual de contrataciones y "
                  "desvinculaciones formales del Ministerio de Trabajo de Brasil."),
        "pdet": ("PDET (Programa de Disseminação das Estatísticas do Trabalho, el Programa de Difusión de "
                 "Estadísticas Laborales): el área del Ministerio de Trabajo que publica los microdatos del CAGED."),
        "mte": "MTE: el Ministerio de Trabajo y Empleo de Brasil, órgano federal de la política laboral.",
        "s6x1": ("Jornada 6×1: turno de seis días de trabajo por uno de descanso, común en el comercio y "
                 "los servicios y que suele sumar las 44 horas semanales."),
        "s5x2": "Jornada 5×2: cinco días de trabajo y dos de descanso por semana, el modelo propuesto por la reforma.",
        "cnae": ("CNAE (Classificação Nacional de Atividades Econômicas, la Clasificación Nacional de "
                 "Actividades Económicas): el sistema oficial brasileño de clasificación sectorial de empresas."),
    },
}


def term(key, label, lang):
    tip = GLOSSARY[lang][key].replace('"', "&quot;")
    return f'<span class="term" tabindex="0" data-tip="{tip}">{label}</span>'


def lang_switch(cur):
    items = ""
    for lg in LANGS:
        cls = " class=\"on\"" if lg == cur else ""
        items += f'<a href="{FILENAME[lg]}"{cls}>{lg.upper()}</a>'
    return f'<span class="langs">{items}</span>'


def build(lang):
    S = STR_ALL[lang]
    brasil, panel, fc, met, sector, hours, sec_hours = load_all()

    # ---- narrative statistics ----
    span = f"{fmt_my(brasil.index[0], lang)}–{fmt_my(brasil.index[-1], lang)}"
    last = brasil.iloc[-1]; last_m = fmt_my(brasil.index[-1], lang)
    acc12 = brasil.iloc[-12:].sum()
    acc_prev12 = brasil.iloc[-24:-12].sum()
    estoque = brasil.cumsum().iloc[-1]
    trough = brasil.min(); trough_m = fmt_my(brasil.idxmin(), lang)
    nat_models = models_of("Brasil", fc); nat_best = best_of("Brasil", met)
    fc_end = nat_models[nat_best]["mean"].iloc[-1]
    fc_lo = nat_models[nat_best]["lower"].iloc[-1]; fc_hi = nat_models[nat_best]["upper"].iloc[-1]
    nat_dg = met[met["serie"] == "Brasil"].iloc[0]
    var_pct = signed_pct((acc12 / acc_prev12 - 1) * 100, 1, lang)

    order = [u for r in REGIOES.values() for u in r]
    acc_uf = {u: hist_of(u, brasil, panel).iloc[-12:].sum() for u in order}
    sp_share = acc_uf["SP"] / sum(v for v in acc_uf.values()) * 100
    sudeste_share = (panel[panel["uf"].isin(REGIOES["Sudeste"])].groupby("date")["saldo"].sum().iloc[-12:].sum()
                     / acc12 * 100)

    figs_hours, share = fig_hours_share(sec_hours, lang, S)
    pct44 = share.iloc[-1]
    pct_full = share.iloc[-1] + share.iloc[-2] + share.iloc[-3]
    exp_fig, agg, media44 = fig_exposure(sector, sec_hours, lang, S)
    exposed = agg.sort_values("pct44", ascending=False)
    top_exposed = exposed.head(4)
    high = agg[agg["pct44"] >= media44]
    high_adm_share = high["adm12"].sum() / agg["adm12"].sum() * 100 if agg["adm12"].sum() else float("nan")

    # ---- figures ----
    d_traj = div(fig_trajetoria(brasil, lang, S))
    d_fan = div(fig_forecast_fan("Brasil", brasil, nat_models, nat_best, lang, S))
    d_reg = div(fig_regioes(panel, lang, S))
    d_heat = div(fig_heatmap(panel, lang, S))
    d_explorer = div(fig_explorer(order, lambda u: hist_of(u, brasil, panel),
                                  lambda u: models_of(u, fc), lambda u: best_of(u, met), lang, S))
    d_hours = div(figs_hours)
    d_exp = div(exp_fig)

    now = dt.datetime.now()
    gen = fmt_gen(now, lang)

    # ---- per-UF table ----
    def truthy(x):
        return str(x).strip().lower() in ("true", "1", "sim", "1.0")
    rows = ""
    for u in order:
        r = met[met["serie"] == u].iloc[0]
        rows += f'''<tr><td class="mono">{u}</td><td>{UF_NOME[u]}</td>
          <td class="num">{br(acc_uf[u], 0, lang)}</td><td>{r['melhor_modelo']}</td>
          <td class="num">{br(r['prev_jun2027'], 0, lang)}</td>
          <td class="num small">{br(r['ic_inf'], 0, lang)} a {br(r['ic_sup'], 0, lang)}</td>
          <td class="num">{br(r['seasonal_strength'], 2, lang)}</td>
          <td>{S['yes'] if truthy(r['heteroscedastic']) else S['no']}</td></tr>'''
    uf_table = f'''<table class="tbl sortable"><thead><tr><th>{S['th_uf']}</th><th>{S['th_state']}</th>
      <th>{S['th_saldo12']}</th><th>{S['th_model']}</th><th>{S['th_prev']}</th><th>{S['th_ci']}</th>
      <th>{S['th_seas']}</th><th>{S['th_het']}</th></tr></thead><tbody>{rows}</tbody></table>'''

    exp_rows = ""
    for _, r in exposed.head(8).iterrows():
        exp_rows += f'''<tr><td class="mono">{r['secao']}</td><td>{r['secao_nome']}</td>
          <td class="num">{r['pct44']:.0f}%</td><td class="num">{br(r['saldo12'], 0, lang)}</td>
          <td class="num">{br(r['adm12'], 0, lang)}</td></tr>'''
    exp_table = f'''<table class="tbl"><thead><tr><th>{S['th_cnae']}</th><th>{S['th_sector']}</th>
      <th>{S['th_p44']}</th><th>{S['th_saldo12']}</th><th>{S['th_adm12']}</th></tr></thead>
      <tbody>{exp_rows}</tbody></table>'''

    refs_html = render_refs(load_refs(), S["no_doi"])
    plotlyjs = pyo.get_plotlyjs()
    tl = top_exposed["secao_nome"].tolist()
    sep = {"pt": " e ", "en": " and ", "es": " y "}[lang]
    exposed_list = ", ".join(tl[:3]) + sep + tl[3] if len(tl) >= 4 else ", ".join(tl)

    ctx = dict(
        span=span, estoque=br(estoque, 0, lang), acc12=br(acc12, 0, lang),
        acc_prev12=br(acc_prev12, 0, lang), var_pct=var_pct, nat_best=nat_best,
        trough=br(trough, 0, lang), trough_m=trough_m, fc_end=br(fc_end, 0, lang),
        fc_lo=br(fc_lo, 0, lang), fc_hi=br(fc_hi, 0, lang),
        sp_share=pct(sp_share, 0, lang), sudeste_share=pct(sudeste_share, 0, lang),
        pct44=pct(pct44, 0, lang), pct_full=pct(pct_full, 0, lang),
        media44=pct(media44, 0, lang), exposed_list=exposed_list,
        high_adm_share=pct(high_adm_share, 0, lang), gen=gen,
        t_clt=term("clt", "CLT", lang), t_caged=term("caged", "Novo CAGED", lang),
        t_pdet=term("pdet", "PDET", lang), t_mte=term("mte", "MTE", lang),
        t_6x1=term("s6x1", "6×1", lang), t_5x2=term("s5x2", "5×2", lang),
        t_cnae=term("cnae", "CNAE", lang),
    )

    def T(key):
        return S[key].format(**ctx)

    html = f"""<!doctype html><html lang="{S['html_lang']}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{S['title']}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style><script>{plotlyjs}</script></head><body>
<div id="prog"></div>
<nav class="top"><div class="wrap">
  <span class="brand">{S['brand']}</span>
  <a href="#panorama">{S['nav_panorama']}</a><a href="#trajetoria">{S['nav_traj']}</a>
  <a href="#geografia">{S['nav_geo']}</a><a href="#sazonalidade">{S['nav_seas']}</a>
  <a href="#projecao">{S['nav_proj']}</a><a href="#dossie">{S['nav_dossie']}</a>
  <a href="#estados">{S['nav_states']}</a>
  {lang_switch(lang)}
</div></nav>

<header class="hero"><div class="wrap">
  <div class="kicker">{T('hero_kicker')}</div>
  <h1>{S['hero_title']}</h1>
  <p class="thesis">{T('hero_thesis')}</p>
  <div class="hero-strip">
    <div class="hs"><div class="v">{ctx['estoque']}</div><div class="l">{S['hs1']}</div></div>
    <div class="hs"><div class="v">{ctx['acc12']}</div><div class="l">{S['hs2']}</div></div>
    <div class="hs"><div class="v">{ctx['pct44']}</div><div class="l">{S['hs3']}</div></div>
    <div class="hs"><div class="v">{ctx['fc_end']}</div><div class="l">{S['hs4'].format(**ctx)}</div></div>
  </div>
  <div class="src">{T('src')}</div>
</div></header>

<section id="panorama"><div class="wrap">
  <div class="kicker">{S['pan_kicker']}</div>
  <h2>{S['pan_h2']}</h2>
  <p class="lede">{S['pan_lede']}</p>
  <div class="prose"><p>{T('pan_p1')}</p></div>
  <div class="findings">
    {stat(br(last, 0, lang), S['f_last'], last_m, 'green' if last>=0 else 'red')}
    {stat(ctx['acc12'], S['f_acc12'], S['f_acc12_n'], 'green')}
    {stat(ctx['sp_share'], S['f_share'], S['f_share_n'], 'amber')}
    {stat(br(nat_dg['seasonal_strength'], 2, lang), S['f_seas'], S['f_seas_n'], 'violet')}
  </div>
</div></section>

<section id="trajetoria" class="alt"><div class="wrap">
  <div class="kicker">{S['tr_kicker']}</div>
  <h2>{S['tr_h2']}</h2>
  <p class="lede">{S['tr_lede']}</p>
  <div class="fig">{d_traj}<div class="figcap">{T('tr_cap')}</div></div>
  <div class="prose">
    <p>{T('tr_p1')}</p>
    <p>{T('tr_p2')}</p>
  </div>
  <div class="pull">{S['tr_pull']}</div>
</div></section>

<section id="geografia"><div class="wrap">
  <div class="kicker">{S['geo_kicker']}</div>
  <h2>{S['geo_h2']}</h2>
  <p class="lede">{T('geo_lede')}</p>
  <div class="fig">{d_reg}<div class="figcap">{S['geo_cap']}</div></div>
  <p class="lede" style="margin-top:30px">{S['geo_explore']}</p>
  <div class="fig">{d_explorer}<div class="figcap">{S['geo_cap2']}</div></div>
</div></section>

<section id="sazonalidade" class="alt"><div class="wrap">
  <div class="kicker">{S['se_kicker']}</div>
  <h2>{S['se_h2']}</h2>
  <p class="lede">{S['se_lede']}</p>
  <div class="fig">{d_heat}<div class="figcap">{S['se_cap']}</div></div>
  <div class="prose"><p>{S['se_p1']}</p></div>
</div></section>

<section id="projecao"><div class="wrap">
  <div class="kicker">{S['pr_kicker']}</div>
  <h2>{S['pr_h2']}</h2>
  <p class="lede">{T('pr_lede')}</p>
  <div class="fig">{d_fan}<div class="figcap">{S['pr_cap']}</div></div>
  <div class="prose"><p>{T('pr_p1')}</p></div>
  <div class="callout">{S['pr_callout']}</div>
</div></section>

<section id="dossie" class="dossie"><div class="wrap">
  <div class="kicker">{S['do_kicker']}</div>
  <h2>{S['do_h2']}</h2>
  <p class="lede">{T('do_lede')}</p>
  <div class="prose"><p>{S['do_p1']}</p></div>

  <div class="grid2">
    <div class="fig">{d_hours}<div class="figcap">{S['do_cap_hours']}</div></div>
    <div>
      <div class="findings" style="grid-template-columns:1fr 1fr">
        {stat(ctx['pct44'], S['do_f_p44'], S['do_f_p44_n'], 'amber')}
        {stat(ctx['pct_full'], S['do_f_full'], S['do_f_full_n'], 'amber')}
      </div>
      <div class="prose"><p>{T('do_p2')}</p></div>
    </div>
  </div>

  <p class="lede" style="margin-top:40px">{S['do_lede2']}</p>
  <div class="fig">{d_exp}<div class="figcap">{T('do_cap_exp')}</div></div>
  <div class="prose"><p>{T('do_p3')}</p></div>
  {exp_table}

  <h2 style="font-size:clamp(22px,3vw,30px);margin:44px 0 6px">{S['do_h3a']}</h2>
  <p class="lede" style="max-width:78ch">{S['do_h3a_p']}</p>
  <div class="scn">
    <div class="c up"><h4>{S['scn_up_h']}</h4><p>{S['scn_up_p']}</p></div>
    <div class="c down"><h4>{S['scn_down_h']}</h4><p>{S['scn_down_p']}</p></div>
    <div class="c"><h4>{S['scn_adj_h']}</h4><p>{S['scn_adj_p']}</p></div>
    <div class="c"><h4>{S['scn_mod_h']}</h4><p>{S['scn_mod_p']}</p></div>
  </div>

  <h2 style="font-size:clamp(20px,2.6vw,26px);margin:42px 0 4px">{S['do_h3b']}</h2>
  <p class="lede" style="max-width:78ch;font-size:14.5px">{S['do_h3b_p']}</p>
  {refs_html}

  <p class="disc">{S['do_disc']}</p>
</div></section>

<section id="estados"><div class="wrap">
  <div class="kicker">{S['st_kicker']}</div>
  <h2>{S['st_h2']}</h2>
  <p class="lede">{S['st_lede']}</p>
  {uf_table}
</div></section>

<footer><div class="wrap">{T('ft_body')}</div></footer>
<script>{PROG_JS}</script>
</body></html>"""

    out = os.path.join(OUT, FILENAME[lang])
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


def build_all():
    outs = [build(lang) for lang in LANGS]
    # keep legacy filename pointing at the Portuguese edition
    legacy = os.path.join(OUT, "CLT-BRASIL_relatorio_v2.html")
    with open(os.path.join(OUT, FILENAME["pt"]), encoding="utf-8") as f:
        pt_html = f.read()
    with open(legacy, "w", encoding="utf-8") as f:
        f.write(pt_html)
    outs.append(legacy)
    return outs


if __name__ == "__main__":
    for o in build_all():
        print(o)
