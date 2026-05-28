"""
Enriched extraction from Novo CAGED microdata:
  * sector (CNAE seção) monthly saldo + admissions
  * contracted-hours distribution of ADMISSIONS (44h = jornada máxima / marca da 6x1)
  * sector x hours table (full period) for 6x1 exposure analysis

Run `python ingest_sector_hours.py test` to process a single month only.
Outputs to data/processed/.
"""
import ftplib, io, py7zr, tempfile, os, sys, time
import numpy as np
import pandas as pd

BASE = "/pdet/microdados/NOVO CAGED"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)
LOG = os.path.join(PROC, "ingest_sector.log")

SEC_NOME = {
    "A": "Agropecuária", "B": "Indústria extrativa", "C": "Indústria de transformação",
    "D": "Eletricidade e gás", "E": "Água, esgoto e resíduos", "F": "Construção",
    "G": "Comércio e reparação", "H": "Transporte e armazenagem",
    "I": "Alojamento e alimentação", "J": "Informação e comunicação",
    "K": "Atividades financeiras", "L": "Atividades imobiliárias",
    "M": "Atividades profissionais e científicas", "N": "Atividades administrativas e serviços",
    "O": "Administração pública", "P": "Educação", "Q": "Saúde humana e serviço social",
    "R": "Artes, cultura e esporte", "S": "Outros serviços",
    "T": "Serviços domésticos", "U": "Organismos internacionais",
}
FAIXAS = ["Parcial (≤30h)", "30–39h", "40h", "41–43h", "44h (jornada máxima)"]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def faixa_horas(h):
    cond = [h <= 30, h < 40, h == 40, h < 44, h >= 44]
    return np.select(cond, FAIXAS, default=FAIXAS[-1])


def connect():
    f = ftplib.FTP("ftp.mtps.gov.br", timeout=120)
    f.encoding = "latin-1"; f.login()
    return f


def process_month(f, year, ym):
    f.cwd(f"{BASE}/{year}/{ym}")
    buf = io.BytesIO(); f.retrbinary(f"RETR CAGEDMOV{ym}.7z", buf.write); buf.seek(0)
    td = tempfile.mkdtemp()
    with py7zr.SevenZipFile(buf) as z:
        z.extractall(td)
    path = os.path.join(td, f"CAGEDMOV{ym}.txt")
    df = pd.read_csv(path, sep=";", encoding="latin-1", header=0,
                     usecols=[4, 6, 11], names=["secao", "saldo", "horas"],
                     dtype={"secao": str, "horas": str})
    try:
        os.remove(path); os.rmdir(td)
    except OSError:
        pass
    df["saldo"] = pd.to_numeric(df["saldo"], errors="coerce").fillna(0).astype(int)
    df["horas"] = pd.to_numeric(df["horas"].str.replace(",", ".", regex=False),
                                errors="coerce").fillna(0.0)
    date = f"{ym[:4]}-{ym[4:]}-01"
    sec_saldo = df.groupby("secao")["saldo"].sum()
    adm = df[df["saldo"] == 1].copy()
    sec_adm = adm.groupby("secao").size()
    adm["faixa"] = faixa_horas(adm["horas"].values)
    hours_nat = adm.groupby("faixa").size()
    sec_hours = adm.groupby(["secao", "faixa"]).size()
    return date, sec_saldo, sec_adm, hours_nat, sec_hours


def main():
    test = len(sys.argv) > 1 and sys.argv[1] == "test"
    f = connect(); f.cwd(BASE)
    years = sorted(y for y in f.nlst() if y.isdigit())
    sec_rows, hours_rows = [], []
    sec_hours_tot = {}
    for y in years:
        f.cwd(f"{BASE}/{y}")
        months = sorted(m for m in f.nlst() if m.isdigit() and len(m) == 6)
        for ym in months:
            t0 = time.time()
            try:
                date, sec_saldo, sec_adm, hours_nat, sec_hours = process_month(f, y, ym)
            except Exception as e:
                log(f"WARN {ym}: {type(e).__name__} {e}; reconnect")
                try: f.quit()
                except Exception: pass
                f = connect()
                date, sec_saldo, sec_adm, hours_nat, sec_hours = process_month(f, y, ym)
            for sec in sec_saldo.index:
                if sec in SEC_NOME:
                    sec_rows.append((date, sec, SEC_NOME[sec],
                                     int(sec_saldo[sec]), int(sec_adm.get(sec, 0))))
            for fx, n in hours_nat.items():
                hours_rows.append((date, fx, int(n)))
            for (sec, fx), n in sec_hours.items():
                if sec in SEC_NOME:
                    sec_hours_tot[(sec, fx)] = sec_hours_tot.get((sec, fx), 0) + int(n)
            log(f"{ym}: saldo={int(sec_saldo.sum()):>8}, adm={int(sec_adm.sum()):>8} ({time.time()-t0:.1f}s)")
            if test:
                f.quit()
                pd.DataFrame(sec_rows, columns=["date","secao","secao_nome","saldo","admissoes"]).to_csv(
                    os.path.join(PROC, "TEST_sector.csv"), index=False)
                log("TEST done"); return

    f.quit()
    pd.DataFrame(sec_rows, columns=["date","secao","secao_nome","saldo","admissoes"]).to_csv(
        os.path.join(PROC, "sector_saldo_mensal.csv"), index=False)
    pd.DataFrame(hours_rows, columns=["date","faixa","admissoes"]).to_csv(
        os.path.join(PROC, "hours_dist_mensal.csv"), index=False)
    sh = [(s, SEC_NOME[s], fx, n) for (s, fx), n in sec_hours_tot.items()]
    pd.DataFrame(sh, columns=["secao","secao_nome","faixa","admissoes"]).to_csv(
        os.path.join(PROC, "sector_hours.csv"), index=False)
    log("DONE: sector_saldo_mensal.csv, hours_dist_mensal.csv, sector_hours.csv")


if __name__ == "__main__":
    main()
