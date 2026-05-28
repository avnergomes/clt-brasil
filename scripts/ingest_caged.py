"""
Ingest Novo CAGED microdata (PDET/MTE) and build a per-UF monthly saldo panel.

Source : ftp.mtps.gov.br /pdet/microdados/NOVO CAGED/<ano>/<anomes>/CAGEDMOV<anomes>.7z
Method : saldo(UF, mes) = sum(saldomovimentacao) over CAGEDMOV  -- reproduces the
         official "Novo CAGED sem ajuste" series exactly (validated vs IPEADATA).
Output : data/raw/caged_uf_panel.csv   (long: date, uf_code, uf, regiao, saldo)
         data/raw/caged_brasil.csv     (date, saldo)
"""
import ftplib, io, py7zr, tempfile, os, csv, sys, time, json

BASE = "/pdet/microdados/NOVO CAGED"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)
LOG  = os.path.join(RAW, "ingest_progress.log")

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

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def connect():
    f = ftplib.FTP("ftp.mtps.gov.br", timeout=120)
    f.encoding = "latin-1"
    f.login()
    return f

def aggregate_month(f, year, ym):
    f.cwd(f"{BASE}/{year}/{ym}")
    buf = io.BytesIO()
    f.retrbinary(f"RETR CAGEDMOV{ym}.7z", buf.write)
    buf.seek(0)
    td = tempfile.mkdtemp()
    with py7zr.SevenZipFile(buf) as z:
        z.extractall(td)
    path = os.path.join(td, f"CAGEDMOV{ym}.txt")
    byuf = {}
    with open(path, encoding="latin-1") as fh:
        rd = csv.reader(fh, delimiter=";")
        next(rd)
        for row in rd:
            uf = row[2]; s = int(row[6])
            byuf[uf] = byuf.get(uf, 0) + s
    try:
        os.remove(path); os.rmdir(td)
    except OSError:
        pass
    return byuf

def main():
    f = connect()
    f.cwd(BASE)
    years = [y for y in f.nlst() if y.isdigit()]
    years.sort()
    rows = []
    for y in years:
        f.cwd(f"{BASE}/{y}")
        months = sorted(m for m in f.nlst() if m.isdigit() and len(m) == 6)
        for ym in months:
            t0 = time.time()
            try:
                byuf = aggregate_month(f, y, ym)
            except Exception as e:
                log(f"WARN {ym}: {type(e).__name__} {e}; reconnecting")
                try: f.quit()
                except Exception: pass
                f = connect()
                byuf = aggregate_month(f, y, ym)
            date = f"{ym[:4]}-{ym[4:]}-01"
            for code, s in byuf.items():
                name, reg = UF.get(code, (f"ND{code}", "ND"))
                rows.append((date, code, name, reg, s))
            tot = sum(byuf.values())
            log(f"{ym}: {len(byuf)} UF, saldo BR={tot:>8} ({round(time.time()-t0,1)}s)")
    f.quit()

    panel = os.path.join(RAW, "caged_uf_panel.csv")
    with open(panel, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "uf_code", "uf", "regiao", "saldo"])
        w.writerows(sorted(rows))
    # national
    br = {}
    for date, code, name, reg, s in rows:
        if code in UF:  # only valid UFs in national aggregate
            br[date] = br.get(date, 0) + s
    bras = os.path.join(RAW, "caged_brasil.csv")
    with open(bras, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["date", "saldo"])
        for d in sorted(br): w.writerow([d, br[d]])
    log(f"DONE: {len(rows)} rows -> {panel}; national -> {bras}")

if __name__ == "__main__":
    main()
