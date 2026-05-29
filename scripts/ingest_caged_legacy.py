"""
Ingest CAGED Antigo microdata (2007-2019, PDET/MTE) and rebuild every panel the
analysis uses, so the series can be extended back before the Novo CAGED era.

Source : ftp.mtps.gov.br /pdet/microdados/CAGED/<ano>/CAGEDEST_<MMYYYY>.7z
Format : ; delimited, latin-1, 1 header line. Relevant columns (0-indexed):
           0  Admitidos/Desligados   01 = admissao, 02 = desligamento
           6  CNAE 2.0 Classe        5-digit; first 2 = divisao -> secao (A..U)
          10  Qtd Hora Contrat       contracted weekly hours (44 = nucleo da 6x1)
          23  UF                     2-digit IBGE state code

METHODOLOGICAL BREAK: the CAGED Antigo (employer monthly declarations, Lei
4.923/65) is NOT directly comparable to the Novo CAGED (eSocial, broader
coverage, different timing). Every output row is tagged fonte="CAGED Antigo" so
the report can draw the Jan/2020 break instead of pretending the series is
continuous. CAGEDEST excludes late declarations (CAGED_AJUSTES), so saldos run
slightly below the official "com ajustes" figures; the trajectory is preserved.

Outputs (suffix _legacy.csv, kept separate from the Novo CAGED files):
  data/raw/caged_brasil_legacy.csv         date, saldo, fonte
  data/raw/caged_uf_panel_legacy.csv       date, uf_code, uf, regiao, saldo, fonte
  data/processed/sector_saldo_mensal_legacy.csv  date, secao, secao_nome, saldo, admissoes, fonte
  data/processed/hours_dist_mensal_legacy.csv    date, faixa, admissoes, fonte
  data/processed/sector_hours_legacy.csv         secao, secao_nome, faixa, admissoes, fonte

Run `python ingest_caged_legacy.py test` to process only 2019-12 and write
TEST_*.csv for validation before launching the full 2007-2019 backfill.
"""
import ftplib, io, py7zr, tempfile, os, sys, time, glob, json, subprocess, shutil, csv
import numpy as np
import pandas as pd

# Some CAGED Antigo archives on the MTE FTP are partially corrupt: py7zr's
# extractall() aborts with LZMAError, but 7-Zip salvages the data (it reports a
# non-zero exit but still writes the .txt). Prefer the 7-Zip CLI when available.
SEVENZIP = next((p for p in (
    r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe",
    shutil.which("7z"), shutil.which("7za")) if p and os.path.exists(p)), None)

BASE = "/pdet/microdados/CAGED"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
CACHE = os.path.join(RAW, "legacy_cache")   # per-month JSON checkpoints (resumable)
os.makedirs(RAW, exist_ok=True)
os.makedirs(PROC, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
LOG = os.path.join(RAW, "ingest_legacy.log")
FONTE = "CAGED Antigo"
MAX_TRIES = 5

UF = {
 "11":("RO","Norte"),"12":("AC","Norte"),"13":("AM","Norte"),"14":("RR","Norte"),
 "15":("PA","Norte"),"16":("AP","Norte"),"17":("TO","Norte"),
 "21":("MA","Nordeste"),"22":("PI","Nordeste"),"23":("CE","Nordeste"),"24":("RN","Nordeste"),
 "25":("PB","Nordeste"),"26":("PE","Nordeste"),"27":("AL","Nordeste"),"28":("SE","Nordeste"),
 "29":("BA","Nordeste"),
 "31":("MG","Sudeste"),"32":("ES","Sudeste"),"33":("RJ","Sudeste"),"35":("SP","Sudeste"),
 "41":("PR","Sul"),"42":("SC","Sul"),"43":("RS","Sul"),
 "50":("MS","Centro-Oeste"),"51":("MT","Centro-Oeste"),"52":("GO","Centro-Oeste"),
 "53":("DF","Centro-Oeste"),
}

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


def divisao_to_secao(div):
    """CNAE 2.0 division (1-99) -> section letter (A..U)."""
    if div <= 0:
        return None
    if div <= 3:  return "A"
    if div <= 9:  return "B"
    if div <= 33: return "C"
    if div == 35: return "D"
    if div <= 39: return "E"
    if div <= 43: return "F"
    if div <= 47: return "G"
    if div <= 53: return "H"
    if div <= 56: return "I"
    if div <= 63: return "J"
    if div <= 66: return "K"
    if div == 68: return "L"
    if div <= 75: return "M"
    if div <= 82: return "N"
    if div == 84: return "O"
    if div == 85: return "P"
    if div <= 88: return "Q"
    if div <= 93: return "R"
    if div <= 96: return "S"
    if div == 97: return "T"
    if div == 99: return "U"
    return None


def faixa_horas(h):
    cond = [h <= 30, h < 40, h == 40, h < 44, h >= 44]
    return np.select(cond, FAIXAS, default=FAIXAS[-1])


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def connect():
    f = ftplib.FTP("ftp.mtps.gov.br", timeout=180)
    f.encoding = "latin-1"; f.login()
    return f


def _download_verified(f, year, ym, dest):
    """Download CAGEDEST_<ym>.7z to `dest` on disk, verifying the byte count
    against the FTP SIZE. Extracting from a real file (not BytesIO) avoids a
    py7zr streaming bug that raises spurious LZMAError on some archives."""
    f.cwd(f"{BASE}/{year}")
    fname = f"CAGEDEST_{ym}.7z"
    try:
        expected = f.size(fname)
    except Exception:
        expected = None
    with open(dest, "wb") as fh:
        f.retrbinary(f"RETR {fname}", fh.write)
    got = os.path.getsize(dest)
    if expected is not None and got != expected:
        raise IOError(f"size mismatch {got}/{expected} for {fname}")


def _extract_txt(archive_path, td):
    """Extract the single .txt from the archive into `td`. Uses 7-Zip when
    present (recovers partially-corrupt archives), else py7zr. Returns
    (txt_path, partial) where partial flags a salvaged-with-errors extraction."""
    if SEVENZIP:
        r = subprocess.run([SEVENZIP, "x", archive_path, f"-o{td}", "-y"],
                           capture_output=True, text=True)
        txt = glob.glob(os.path.join(td, "**", "*.txt"), recursive=True)
        if txt and os.path.getsize(txt[0]) > 0:
            return txt[0], (r.returncode != 0)
    with py7zr.SevenZipFile(archive_path) as z:   # fallback / no 7-Zip installed
        z.extractall(td)
    txt = glob.glob(os.path.join(td, "**", "*.txt"), recursive=True)
    return txt[0], False


def _read_txt(path):
    # CAGED Antigo is ;-delimited with NO quoted fields -> QUOTE_NONE prevents a
    # stray quote in a salvaged file from triggering "EOF inside string"; a
    # truncated trailing row is dropped by on_bad_lines="skip".
    return pd.read_csv(path, sep=";", encoding="latin-1", header=0,
                       usecols=[0, 6, 10, 23], names=["mov", "cnae", "horas", "uf"],
                       dtype={"mov": str, "cnae": str, "horas": str, "uf": str},
                       on_bad_lines="skip", quoting=csv.QUOTE_NONE)


def _aggregate(archive_path):
    """Extract the .7z file and read one month. Returns (df, partial); df is None
    when a partially-corrupt archive cannot be parsed (month becomes a gap)."""
    td = tempfile.mkdtemp()
    try:
        path, partial = _extract_txt(archive_path, td)
        if partial:
            log(f"  WARN salvaged partially-corrupt archive: {os.path.basename(archive_path)}")
        try:
            df = _read_txt(path)
        except Exception as e:
            if partial:
                log(f"  unparseable salvaged archive ({type(e).__name__}); month -> gap")
                return None, True
            raise
    finally:
        for leftover in glob.glob(os.path.join(td, "**", "*"), recursive=True):
            try: os.remove(leftover)
            except OSError: pass
        try: os.rmdir(td)
        except OSError: pass
    return df, partial


def _summarise(df):
    """Aggregate one month's dataframe -> dict of plain dicts."""

    mov = pd.to_numeric(df["mov"], errors="coerce")
    df["saldo"] = np.where(mov == 1, 1, np.where(mov == 2, -1, 0)).astype(int)
    div = pd.to_numeric(df["cnae"].str[:2], errors="coerce").fillna(0).astype(int)
    df["secao"] = div.map(divisao_to_secao)
    df["uf"] = df["uf"].str.zfill(2)
    df["horas"] = pd.to_numeric(df["horas"].str.replace(",", ".", regex=False),
                                errors="coerce").fillna(0.0)

    adm = df[df["saldo"] == 1]
    af = adm.assign(faixa=faixa_horas(adm["horas"].values))
    return {
        "by_uf": {k: int(v) for k, v in df.groupby("uf")["saldo"].sum().items()},
        "sec_saldo": {k: int(v) for k, v in df.groupby("secao")["saldo"].sum().items() if k},
        "sec_adm": {k: int(v) for k, v in adm.groupby("secao").size().items() if k},
        "hours_nat": {k: int(v) for k, v in af.groupby("faixa").size().items()},
        "sec_hours": {f"{s}|{fx}": int(v)
                      for (s, fx), v in af.groupby(["secao", "faixa"]).size().items() if s},
    }


def get_month(f, year, ym):
    """Return aggregates for a month, using the on-disk cache when present.
    Robust to corrupt/truncated downloads: verifies size and retries."""
    cpath = os.path.join(CACHE, f"{ym}.json")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as fh:
            return json.load(fh), f, True
    last = None
    arch = os.path.join(tempfile.gettempdir(), f"CAGEDEST_{ym}.7z")
    for attempt in range(1, MAX_TRIES + 1):
        try:
            _download_verified(f, year, ym, arch)
            df, partial = _aggregate(arch)
            agg = {"_partial": True} if df is None else {**_summarise(df), "_partial": bool(partial)}
            with open(cpath, "w", encoding="utf-8") as fh:
                json.dump(agg, fh)
            return agg, f, False
        except Exception as e:
            last = e
            log(f"  retry {attempt}/{MAX_TRIES} {ym}: {type(e).__name__} {e}")
            try: f.quit()
            except Exception: pass
            time.sleep(2 * attempt)
            f = connect()
        finally:
            try: os.remove(arch)
            except OSError: pass
    raise RuntimeError(f"{ym} failed after {MAX_TRIES} tries: {last}")


def main():
    test = len(sys.argv) > 1 and sys.argv[1] == "test"
    f = connect(); f.cwd(BASE)
    years = sorted(y for y in f.nlst() if y.isdigit())
    if test:
        years = ["2019"]

    uf_rows, sec_rows, hours_rows = [], [], []
    sec_hours_tot = {}
    gaps = []
    for y in years:
        f.cwd(f"{BASE}/{y}")
        files = sorted(n for n in f.nlst() if n.startswith("CAGEDEST_") and n.endswith(".7z"))
        months = [n[len("CAGEDEST_"):-len(".7z")] for n in files]  # MMYYYY
        if test:
            months = ["122019"]
        for ym in months:
            mm, yyyy = ym[:2], ym[2:]
            date = f"{yyyy}-{mm}-01"
            t0 = time.time()
            agg, f, cached = get_month(f, y, ym)

            if agg.get("_partial"):
                gaps.append(date)
                log(f"{date}: SKIP (source archive corrupt, only partially "
                    f"recoverable) -> left as gap")
                continue

            by_uf = agg["by_uf"]
            for code, s in by_uf.items():
                if code in UF:
                    name, reg = UF[code]
                    uf_rows.append((date, code, name, reg, int(s), FONTE))
            for sec, v in agg["sec_saldo"].items():
                if sec in SEC_NOME:
                    sec_rows.append((date, sec, SEC_NOME[sec],
                                     int(v), int(agg["sec_adm"].get(sec, 0)), FONTE))
            for fx, n in agg["hours_nat"].items():
                hours_rows.append((date, fx, int(n), FONTE))
            for key, n in agg["sec_hours"].items():
                sec, fx = key.split("|", 1)
                if sec in SEC_NOME:
                    sec_hours_tot[(sec, fx)] = sec_hours_tot.get((sec, fx), 0) + int(n)

            nat = sum(int(s) for c, s in by_uf.items() if c in UF)
            tag = "cache" if cached else f"{time.time()-t0:.1f}s"
            log(f"{date}: UF={len(by_uf)}, saldo BR={nat:>9}, "
                f"adm={sum(agg['sec_adm'].values()):>8} ({tag})")

    f.quit()

    # national from UF rows
    uf_df = pd.DataFrame(uf_rows, columns=["date","uf_code","uf","regiao","saldo","fonte"])
    uf_df.sort_values(["date","uf_code"]).to_csv(
        os.path.join(RAW, "caged_uf_panel_legacy.csv"), index=False)
    br = uf_df.groupby("date")["saldo"].sum().reset_index()
    br["fonte"] = FONTE
    br.to_csv(os.path.join(RAW, "caged_brasil_legacy.csv"), index=False)

    pd.DataFrame(sec_rows, columns=["date","secao","secao_nome","saldo","admissoes","fonte"]).to_csv(
        os.path.join(PROC, "sector_saldo_mensal_legacy.csv"), index=False)
    pd.DataFrame(hours_rows, columns=["date","faixa","admissoes","fonte"]).to_csv(
        os.path.join(PROC, "hours_dist_mensal_legacy.csv"), index=False)
    sh = [(s, SEC_NOME[s], fx, n, FONTE) for (s, fx), n in sec_hours_tot.items()]
    pd.DataFrame(sh, columns=["secao","secao_nome","faixa","admissoes","fonte"]).to_csv(
        os.path.join(PROC, "sector_hours_legacy.csv"), index=False)
    log(f"DONE: {uf_df['date'].nunique()} months "
        f"({uf_df['date'].min()}..{uf_df['date'].max()})")
    if gaps:
        log(f"GAPS ({len(gaps)}, corrupt source archives left out): {', '.join(gaps)}")


if __name__ == "__main__":
    main()
