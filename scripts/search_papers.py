"""
Literature search following the k-dense paper-lookup protocol.
Primary database: OpenAlex (broadest, all fields incl. economics) + polite pool.
Topic: effects of working-time / workweek reduction (4-day week, 35h, 6x1->5x2)
on employment, productivity and well-being.

Returns VERIFIABLE metadata only (title, authors, year, venue, DOI, citations).
No citation is invented — every entry comes from the live API response.
"""
import urllib.request, urllib.parse, json, time

OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": "http://localhost:3128", "https": "http://localhost:3128"}))
MAILTO = "avnerpaesgomes@gmail.com"

QUERIES = [
    "four-day work week",
    "four-day week trial",
    "working time reduction",
    "workweek reduction employment",
    "35-hour workweek France",
    "reduction in working hours productivity",
    "shorter working hours well-being",
    "work sharing unemployment",
    "compressed workweek",
    "working hours labor supply reform",
    "overtime hours employment effects",
    "productivity of working hours",
    "has work-sharing worked hours reduction",
    "mandated reduction standard working hours employment",
    "four-day week Iceland public sector",
    "four-day week United Kingdom trial",
    "shorter workweek productivity experiment",
    "standard hours reduction employment France",
]

# a work is kept only if its title/abstract is topically on-point
TOPIC_KW = [
    "working time", "work time", "working hours", "work hours", "hours of work",
    "workweek", "work week", "work-week", "four-day", "four day", "4-day",
    "35-hour", "35 hour", "working-time", "hours reduction", "reduction in hours",
    "reduction of hours", "shorter hours", "work sharing", "work-sharing",
    "worksharing", "compressed", "overtime", "labour supply", "labor supply",
    "hours worked", "weekly hours",
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lit-review/1.0"})
    return json.loads(OPENER.open(req, timeout=60).read())


def abstract(inv):
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))[:600]


def search(q, per=40):
    base = "https://api.openalex.org/works?"
    params = {
        "filter": f"title_and_abstract.search:{q},type:article,is_paratext:false",
        "sort": "cited_by_count:desc",
        "per_page": per,
        "mailto": MAILTO,
        "select": "id,doi,title,publication_year,cited_by_count,"
                  "authorships,primary_location,abstract_inverted_index",
    }
    return get(base + urllib.parse.urlencode(params)).get("results", [])


def on_topic(d):
    txt = ((d["title"] or "") + " " + (d["abstract"] or "")).lower()
    return any(k in txt for k in TOPIC_KW)


def main():
    seen = {}
    for q in QUERIES:
        try:
            res = search(q)
        except Exception as e:
            print("ERR", q, e); continue
        for w in res:
            wid = w["id"]
            if wid in seen:
                continue
            auth = [a["author"]["display_name"] for a in w.get("authorships", [])][:4]
            venue = (w.get("primary_location") or {}).get("source") or {}
            seen[wid] = {
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "cites": w.get("cited_by_count"),
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "authors": auth,
                "venue": venue.get("display_name") if venue else None,
                "abstract": abstract(w.get("abstract_inverted_index")),
            }
        time.sleep(0.3)

    ranked = sorted(seen.values(), key=lambda d: d["cites"] or 0, reverse=True)
    out = [d for d in ranked if d["title"] and on_topic(d)]
    with open("data/processed/papers.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    def safe(s):
        return str(s).encode("ascii", "replace").decode("ascii")
    print(f"{len(out)} unique works. Top 30 by citations:\n")
    for d in out[:30]:
        a = (d["authors"][0] if d["authors"] else "?") + (" et al." if len(d["authors"]) > 1 else "")
        print(f"[{d['cites']:>6}] {d['year']} {safe(a)} - {safe(d['title'])[:78]}")
        print(f"         {safe(d['venue'])[:58]} | doi: {d['doi'] or '-'}")


if __name__ == "__main__":
    main()
