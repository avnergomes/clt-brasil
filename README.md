# CLT em Movimento — Emprego Formal no Brasil

Análise de série temporal e previsão do **saldo de empregos formais (celetistas / CLT)** no Brasil
e nas 27 Unidades da Federação, a partir dos **microdados do CAGED** do Ministério do Trabalho
e Emprego (MTE). A análise e a modelagem se apoiam no **Novo CAGED (eSocial, 2020 em diante)**, e o
contexto histórico de longo prazo é traçado com o **CAGED Antigo (Lei 4.923/65, 2007–2019)**, com
projeção até **junho de 2027** e um dossiê sobre o debate da **escala 6×1 → 5×2**.

### 🔗 Relatório online (GitHub Pages)

- **Relatório interativo (v2, editorial):** https://avnergomes.github.io/clt-brasil/
- Acesso direto ao HTML: [`output/CLT-BRASIL_relatorio_v2.html`](https://avnergomes.github.io/clt-brasil/output/CLT-BRASIL_relatorio_v2.html)
- Notebook documentado: [`notebooks/CLT-BRASIL_analise.ipynb`](notebooks/CLT-BRASIL_analise.ipynb)

---

## Principais resultados

- **10,0 milhões** de vínculos celetistas líquidos criados desde jan/2020.
- **1,21 milhão** de empregos formais no saldo acumulado dos últimos 12 meses (expansão em desaceleração).
- O choque da pandemia levou o saldo a **−902 mil** em abril de 2020 — o pior mês da série.
- **São Paulo** concentra cerca de **30%** do saldo nacional recente; o Sudeste, ~60%.
- Forte **sazonalidade** (contratações no 1º semestre, demissões em dezembro) e **heterocedasticidade**.
- Previsão para **jun/2027**: ~**166 mil** (modelo selecionado por validação), IC 95% [−200 mil; 534 mil].
- **81%** das admissões formais são contratadas a **44h** (88% a ≥40h) — o universo típico da escala 6×1.

## Fonte de dados e metodologia

- **Fonte (núcleo da análise):** microdados do *Novo CAGED* (PDET/MTE), `ftp.mtps.gov.br`,
  arquivos `CAGEDMOV<AAAAMM>.7z`, de **jan/2020** em diante (base eSocial).
- **Fonte (contexto histórico):** microdados do *CAGED Antigo* (Lei 4.923/65), `CAGEDEST_<MMAAAA>.7z`,
  cobrindo **2007–2019**. Usado apenas para traçar a tendência de longo prazo (média móvel de 12 meses).
  Alguns meses têm arquivos corrompidos no FTP do MTE e são preenchidos por **interpolação temporal**,
  sinalizados no relatório.
- **Quebra metodológica (jan/2020):** a troca das declarações da Lei 4.923/65 pelo eSocial torna as duas
  eras **não estritamente comparáveis**; o relatório marca essa ruptura e mantém a modelagem/previsão
  restrita ao Novo CAGED (2020+).
- **Saldo** de cada competência/UF = `Σ(saldomovimentação)` (cada admissão `+1`, cada desligamento `−1`).
  Essa agregação **reproduz exatamente** a série oficial *“Novo CAGED sem ajuste”* (validado contra o
  IPEADATA: jun/2025 = 166.621 em ambas as fontes).
- **Diagnósticos:** decomposição STL, ADF/KPSS (estacionariedade), ARCH-LM (heterocedasticidade),
  força sazonal de Hyndman.
- **Modelos:** SARIMA (`auto_arima`), ETS (Holt-Winters) e *Seasonal Naive*; seleção pelo menor
  **RMSE** em *backtest* (retenção dos últimos 12 meses); previsão com **IC de 95%**.

## Dossiê: o fim da escala 6×1

A seção especial cruza, com os microdados, **a exposição setorial à jornada de 44h** (proxy do
universo 6×1) com a **geração de empregos** de cada setor (CNAE). A interpretação é ancorada em
**literatura econômica revisada por pares**, localizada pelo protocolo de busca científica
[`paper-lookup`](https://github.com/k-dense-ai/scientific-agent-skills) via **OpenAlex** (sem
fabricação — cada referência tem DOI verificável): Pencavel (2014), Collewet & Sauermann (2017),
Chemin & Wasmer (2009), Baltes et al. (1999), Trejo (1991), Deakin & Wilkinson (1988), Afonso (2017)
e Harrington (2001).

> ⚠️ A análise do 6×1 é de natureza econômica e exploratória. O CAGED registra horas contratadas,
> não dias trabalhados; 44h é usado como proxy. Não constitui previsão de aprovação da PEC nem
> aconselhamento.

## Estrutura do projeto

```
CLT-BRASIL/
├── src/
│   ├── clt.py             # motor de análise: dados, testes, modelagem/forecast
│   ├── report.py          # gerador do relatório v1
│   └── report2.py         # gerador do relatório v2 (editorial + dossiê 6×1)
├── scripts/
│   ├── ingest_caged.py        # ingestão Novo CAGED (2020+): saldo mensal por UF (FTP → CSV)
│   ├── ingest_caged_legacy.py # ingestão CAGED Antigo (2007–2019): saldo nacional histórico
│   ├── ingest_sector_hours.py # ingestão: saldo por setor + distribuição de jornada
│   ├── run_forecasts.py       # roda os modelos de previsão por série
│   ├── build_notebook.py      # monta o notebook documentado
│   └── search_papers.py       # busca de literatura (OpenAlex)
├── notebooks/CLT-BRASIL_analise.ipynb
├── data/
│   ├── raw/                # séries por UF e nacional
│   └── processed/          # previsões, métricas, setores/jornada, referências
└── output/                 # relatórios HTML autocontidos (v1 e v2)
```

## Como executar

Requisitos: **Python 3.12+**. Instale as dependências:

```bash
pip install -r requirements.txt
```

Pipeline completo (a ingestão baixa ~4 GB de microdados do FTP do MTE e leva alguns minutos):

```bash
# 1. Ingestão dos microdados → séries por UF e nacional
python scripts/ingest_caged.py

# 2. Ingestão de setor (CNAE) + distribuição de jornada contratada
python scripts/ingest_sector_hours.py

# 3. (Re)montar e executar o notebook documentado
python scripts/build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/CLT-BRASIL_analise.ipynb

# 4. Gerar o relatório editorial v2
python -c "import sys; sys.path.insert(0,'src'); import report2; print(report2.build())"
```

Os relatórios são **HTML autocontidos** (Plotly embutido) — abrem offline, sem dependências externas.

## Tecnologias

`pandas` · `numpy` · `statsmodels` · `pmdarima` · `scikit-learn` · `plotly` · `matplotlib` · `py7zr` · `jupyter`
