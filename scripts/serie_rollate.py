"""Serie continue dei futures come le vive chi tiene un CFD Fineco.

Il CFD Fineco su un future segue un contratto alla volta e non viene rinnovato in automatico:
nell'ultimo giorno di negoziazione del CFD la posizione aperta viene chiusa d'ufficio, e per
restare dentro bisogna riaprirla sul contratto successivo pagando di nuovo lo spread. Chi resta
posizionato guadagna o perde quanto il contratto che tiene. La serie continua di Yahoo, invece,
nel giorno del cambio salta da un contratto all'altro senza correzione: quel salto non è un
rendimento di nessuno, e sul gas vale spesso più di una σ giornaliera.

Qui si ricostruisce il rendimento di chi rinnova:
- storico: prezzi di regolamento NYMEX del primo e del secondo contratto pubblicati dall'EIA
  (aggiornati una volta alla settimana);
- ultime sedute dopo l'ultimo dato EIA: chiusure dei singoli contratti su Yahoo;
- calendario: ultimo giorno di negoziazione del future secondo le regole del CME, anticipato
  per il CFD quando Fineco lo chiude prima (petrolio: due giorni lavorativi prima);
- rettifica proporzionale all'indietro: l'ultimo valore della serie è il prezzo del contratto
  che si tiene da domani, i rendimenti giornalieri sono quelli di chi rinnova.
"""
from __future__ import annotations

import datetime as dt
import io
import math
import time
import urllib.request

import numpy as np
import pandas as pd

CODICI_MESE = "FGHJKMNQUVXZ"

ROLLATE = {
    "NG=F": {"eia": "https://www.eia.gov/dnav/ng/hist_xls/RNGC{n}d.xls", "radice": "NG", "borsa": ".NYM",
             "scadenza": "ng", "anticipo": 0,
             "regola": "il future scade il terzultimo giorno lavorativo del mese prima della consegna; "
                       "il CFD Fineco si chiude lo stesso giorno"},
    "CL=F": {"eia": "https://www.eia.gov/dnav/pet/hist_xls/RCLC{n}d.xls", "radice": "CL", "borsa": ".NYM",
             "scadenza": "cl", "anticipo": 2,
             "regola": "il future scade tre giorni lavorativi prima del 25 del mese prima della consegna; "
                       "il CFD Fineco si chiude due giorni lavorativi prima"},
}


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------
def _osservato(d: dt.date) -> dt.date:
    """Festa che cade di sabato: venerdì; di domenica: lunedì."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def _ennesimo(anno: int, mese: int, giorno_sett: int, n: int) -> dt.date:
    d = dt.date(anno, mese, 1)
    d += dt.timedelta(days=(giorno_sett - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _ultimo(anno: int, mese: int, giorno_sett: int) -> dt.date:
    d = dt.date(anno + (mese == 12), mese % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - giorno_sett) % 7)


def _pasqua(anno: int) -> dt.date:
    a, b, c = anno % 19, anno // 100, anno % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    mese = (h + l_ - 7 * m + 114) // 31
    giorno = (h + l_ - 7 * m + 114) % 31 + 1
    return dt.date(anno, mese, giorno)


def festivi_usa(anno: int) -> set:
    """Giorni di chiusura del NYMEX (calendario delle borse USA). Serve solo per i mesi che i dati
    non coprono ancora; per il passato valgono i giorni presenti nei dati."""
    f = {
        _ennesimo(anno, 1, 0, 3),                 # Martin Luther King
        _ennesimo(anno, 2, 0, 3),                 # Presidents' Day
        _pasqua(anno) - dt.timedelta(days=2),     # Venerdì santo
        _ultimo(anno, 5, 0),                      # Memorial Day
        _osservato(dt.date(anno, 7, 4)),
        _ennesimo(anno, 9, 0, 1),                 # Labor Day
        _ennesimo(anno, 11, 3, 4),                # Thanksgiving
        _osservato(dt.date(anno, 12, 25)),
    }
    capodanno = dt.date(anno, 1, 1)
    if capodanno.weekday() != 5:                  # di sabato non si recupera il venerdì prima
        f.add(_osservato(capodanno))
    if anno >= 2022:
        f.add(_osservato(dt.date(anno, 6, 19)))   # Juneteenth
    return f


def giorni_di_borsa(date_dati, fino_a: pd.Timestamp) -> list:
    """Giorni dei dati, poi giorni feriali senza festività fino a fino_a."""
    giorni = sorted(pd.Timestamp(d).normalize() for d in date_dati)
    d = giorni[-1] + pd.Timedelta(days=1) if giorni else pd.Timestamp("2000-01-03")
    festivi = {}
    while d <= fino_a:
        if d.weekday() < 5:
            if d.year not in festivi:
                festivi[d.year] = festivi_usa(d.year)
            if d.date() not in festivi[d.year]:
                giorni.append(d)
        d += pd.Timedelta(days=1)
    return giorni


def calendario(giorni: list, regola: str, anticipo: int) -> list:
    """Per ogni mese coperto: (mese di consegna, ultimo giorno del future, chiusura del CFD).

    ng: il future di consegna M+1 scade il terzultimo giorno lavorativo del mese M.
    cl: scade tre giorni lavorativi prima del 25 del mese M; se il 25 non è lavorativo,
        quattro giorni lavorativi prima del 25.
    Il CFD si chiude `anticipo` giorni lavorativi prima della scadenza del future."""
    per_mese = {}
    for g in giorni:
        per_mese.setdefault((g.year, g.month), []).append(g)
    indice = {g: i for i, g in enumerate(giorni)}
    eventi = []
    chiavi = sorted(per_mese)
    for (a, m) in chiavi[:-1] if len(chiavi) > 1 else []:   # l'ultimo mese può essere incompleto
        gm = per_mese[(a, m)]
        if regola == "ng":
            if len(gm) < 3:
                continue
            scad = gm[-3]
        else:
            prima25 = [g for g in gm if g.day < 25]
            venticinque = [g for g in gm if g.day == 25]
            k = 3 if venticinque else 4
            if len(prima25) < k:
                continue
            scad = prima25[-k]
        i = indice[scad] - anticipo
        if i < 0:
            continue
        consegna = f"{a + (m == 12)}-{m % 12 + 1:02d}"
        eventi.append({"consegna": consegna, "scadenza": scad, "roll": giorni[i]})
    return eventi


def simbolo_yahoo(cfg: dict, consegna: str) -> str:
    a, m = map(int, consegna.split("-"))
    return f"{cfg['radice']}{CODICI_MESE[m - 1]}{str(a)[2:]}{cfg['borsa']}"


# ---------------------------------------------------------------------------
# Dati EIA
# ---------------------------------------------------------------------------
def scarica_eia(url: str, tentativi: int = 3) -> pd.Series:
    """Foglio "Data 1" del file XLS dell'EIA: colonna date e colonna prezzi."""
    errore = None
    for i in range(tentativi):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mercati-e-code; GitHub Actions)"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dati = resp.read()
            grezzo = pd.read_excel(io.BytesIO(dati), sheet_name="Data 1", header=None)
            date = pd.to_datetime(grezzo.iloc[:, 0], errors="coerce")
            prezzi = pd.to_numeric(grezzo.iloc[:, 1], errors="coerce")
            s = pd.Series(prezzi.to_numpy(), index=date).dropna()
            s = s[s.index.notna()]
            s.index = pd.DatetimeIndex(s.index).normalize()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if len(s) < 1000:
                raise ValueError(f"solo {len(s)} righe")
            return s
        except Exception as e:
            errore = e
            if i < tentativi - 1:
                time.sleep(5 * 2 ** i)
    raise RuntimeError(f"EIA {url.rsplit('/', 1)[-1]}: {type(errore).__name__}: {errore}")


# ---------------------------------------------------------------------------
# Serie rinnovata
# ---------------------------------------------------------------------------
def _ln(a, b):
    return math.log(a / b) if a is not None and b is not None and a > 0 and b > 0 else float("nan")


def rendimenti_rinnovati(c1: pd.Series, c2: pd.Series, eventi: list) -> pd.DataFrame:
    """Rendimento giornaliero di chi tiene il contratto del CFD e lo rinnova alla chiusura del CFD.

    Etichette EIA: C1 è il contratto con la consegna più vicina ancora quotato (anche nel suo ultimo
    giorno), C2 il successivo. Dopo la chiusura del CFD (giorno r) e fino alla scadenza (giorno e)
    si tiene C2; il giorno dopo la scadenza il vecchio C2 diventa C1."""
    df = pd.concat({"c1": c1, "c2": c2}, axis=1, join="inner").sort_index()
    date = list(df.index)
    pos = {d: i for i, d in enumerate(date)}
    n = len(date)
    tiene2 = np.zeros(n, dtype=bool)          # durante la seduta t si tiene il contratto etichettato C2
    rietichetta = np.zeros(n, dtype=bool)     # t è il primo giorno dopo una scadenza
    roll = np.zeros(n, dtype=bool)            # alla chiusura di t il CFD scade: si chiude e si riapre
    for ev in eventi:
        e, r = pos.get(ev["scadenza"]), pos.get(ev["roll"])
        if e is None or r is None:
            continue
        roll[r] = True
        tiene2[r + 1:e + 1] = True
        if e + 1 < n:
            rietichetta[e + 1] = True
    v1, v2 = df["c1"].to_numpy(float), df["c2"].to_numpy(float)
    r = np.full(n, np.nan)
    for t in range(1, n):
        if rietichetta[t]:
            r[t] = _ln(v1[t], v2[t - 1])
        elif tiene2[t]:
            r[t] = _ln(v2[t], v2[t - 1])
        else:
            r[t] = _ln(v1[t], v1[t - 1])
    sp = np.array([_ln(b, a) for a, b in zip(v1, v2)])
    # dopo la chiusura del giorno t si tiene C2 se t sta tra il roll del CFD e la scadenza
    dopo2 = np.zeros(n, dtype=bool)
    for ev in eventi:
        e, rr = pos.get(ev["scadenza"]), pos.get(ev["roll"])
        if e is not None and rr is not None:
            dopo2[rr:e + 1] = True
    return pd.DataFrame({"r": r, "sp": sp, "roll": roll, "rietichetta": rietichetta, "dopo2": dopo2,
                         "c1": v1, "c2": v2}, index=df.index)


def verifica_calendario(df: pd.DataFrame) -> dict:
    """Il giorno dopo la scadenza C1 deve ripartire dal vecchio C2, non dal vecchio C1. Si contano i
    giorni in cui è così, quando lo spread tra i due contratti supera il 2%, con il calendario
    calcolato e con lo stesso calendario spostato di una seduta prima e dopo: se le scadenze sono
    giuste, la quota è molto più alta senza spostamento. Non arriva al 100% perché un movimento
    del giorno contrario allo spread può confondere il confronto."""
    v1, v2 = df["c1"].to_numpy(float), df["c2"].to_numpy(float)
    base = np.flatnonzero(df["rietichetta"].to_numpy())
    out = {}
    for nome, sp in (("prima", -1), ("calcolato", 0), ("dopo", 1)):
        ok = tot = 0
        for t in base + sp:
            if t < 1 or t >= len(v1):
                continue
            s = _ln(v2[t - 1], v1[t - 1])
            if not math.isfinite(s) or abs(s) < 0.02:
                continue
            tot += 1
            ok += abs(_ln(v1[t], v2[t - 1])) < abs(_ln(v1[t], v1[t - 1]))
        out[nome] = round(ok / tot, 3) if tot else None
        if sp == 0:
            out["controllati"] = tot
    return out


def serie_rinnovata(simbolo: str, yahoo: pd.DataFrame, scarica_rapido, pulisci, ieri: pd.Timestamp) -> tuple:
    """Restituisce (DataFrame con Close rettificato, informazioni) oppure solleva un'eccezione."""
    cfg = ROLLATE[simbolo]
    c1 = scarica_eia(cfg["eia"].format(n=1))
    c2 = scarica_eia(cfg["eia"].format(n=2))
    comuni = c1.index.intersection(c2.index)
    giorni = giorni_di_borsa(comuni, ieri + pd.Timedelta(days=120))
    eventi = calendario(giorni, cfg["scadenza"], cfg["anticipo"])
    base = rendimenti_rinnovati(c1, c2, eventi)
    controllo = verifica_calendario(base)
    ultima_eia = base.index[-1]

    # contratto tenuto dopo la chiusura di t: il primo il cui CFD si chiude dopo t
    def tenuto_dopo(t):
        return next(ev["consegna"] for ev in eventi if ev["roll"] > t)

    def primo_secondo(t):                       # C1 e C2 secondo le etichette EIA
        vivi = [ev["consegna"] for ev in eventi if ev["scadenza"] >= t]
        return vivi[0], vivi[1]

    coda = [d for d in yahoo.index if d > ultima_eia]
    tenuti = {tenuto_dopo(ultima_eia)} | {tenuto_dopo(d) for d in coda}
    per_spread = set()
    for d in coda:
        per_spread |= set(primo_secondo(d))
    prezzi = {}
    for consegna in sorted(tenuti | per_spread):
        sim = simbolo_yahoo(cfg, consegna)
        df = scarica_rapido(sim, "3mo")
        time.sleep(0.4)
        if df is not None:
            prezzi[consegna] = pulisci(df)["Close"]
        elif consegna in tenuti:
            raise RuntimeError(f"contratto {sim} non disponibile su Yahoo")

    def prezzo(consegna, d, serve=True):
        s = prezzi.get(consegna)
        if s is None or d not in s.index:
            if serve:
                raise RuntimeError(f"manca la chiusura di {simbolo_yahoo(cfg, consegna)} del {d:%Y-%m-%d}")
            return None
        return float(s.loc[d])

    righe = []
    prec = ultima_eia
    for d in coda:
        x = tenuto_dopo(prec)
        f1, f2 = primo_secondo(d)
        righe.append({"data": d, "r": _ln(prezzo(x, d), prezzo(x, prec)),
                      "sp": _ln(prezzo(f2, d, False), prezzo(f1, d, False)),   # lo spread è facoltativo
                      "roll": any(ev["roll"] == d for ev in eventi)})
        prec = d
    ultima = coda[-1] if coda else ultima_eia
    tenuto = tenuto_dopo(ultima)
    if coda:
        ancora = prezzo(tenuto, ultima)
    else:
        ancora = float(base["c2"].iloc[-1] if base["dopo2"].iloc[-1] else base["c1"].iloc[-1])

    r = pd.concat([base["r"], pd.Series({x["data"]: x["r"] for x in righe}, dtype=float)])
    sp = pd.concat([base["sp"], pd.Series({x["data"]: x["sp"] for x in righe}, dtype=float)])
    roll = pd.concat([base["roll"], pd.Series({x["data"]: x["roll"] for x in righe}, dtype=bool)])
    buchi = int(r.iloc[1:].isna().sum())
    r = r.fillna(0.0)
    r.iloc[0] = 0.0
    # rettifica proporzionale all'indietro: il livello di oggi è il prezzo del contratto tenuto
    livelli = ancora * np.exp(-(r[::-1].cumsum()[::-1].to_numpy() - r.to_numpy()))
    out = pd.DataFrame({"Close": livelli}, index=r.index)

    prossimo = next((ev for ev in eventi if ev["roll"] > ultima), None)
    info = {
        "fonte": f"EIA (regolamenti NYMEX) fino al {ultima_eia:%Y-%m-%d}" + (f", poi contratti Yahoo" if coda else ""),
        "regola": cfg["regola"],
        "anticipo": cfg["anticipo"],
        "contratto": {"consegna": tenuto, "simbolo": simbolo_yahoo(cfg, tenuto)},
        "prossimo_roll": ({"data": prossimo["roll"].strftime("%Y-%m-%d"), "consegna": prossimo["consegna"],
                           "nuovo": next(ev["consegna"] for ev in eventi if ev["roll"] > prossimo["roll"])}
                          if prossimo else None),
        "verifica": controllo,
        "buchi": buchi,
        "roll_date": [d for d, v in roll.items() if v],
        "spread": sp,
    }
    return out, info


def confronto_rendimenti(yahoo: pd.DataFrame, rinnovata: pd.DataFrame) -> dict | None:
    """Rendimento annuo di chi è sempre long: serie continua di Yahoo contro serie rinnovata, tra le
    stesse due date. La differenza è quanto è costato (o reso) il passaggio da un contratto all'altro.
    Si usano solo i prezzi agli estremi, così le chiusure negative del WTI nel 2020 non contano."""
    a, b = yahoo["Close"], rinnovata["Close"]
    comuni = a.index.intersection(b.index)
    comuni = comuni[(a.loc[comuni] > 0).to_numpy() & (b.loc[comuni] > 0).to_numpy()]
    if len(comuni) < 500:
        return None
    d0, d1 = comuni[0], comuni[-1]
    anni = (d1 - d0).days / 365.25
    ga, gb = math.log(a.loc[d1] / a.loc[d0]), math.log(b.loc[d1] / b.loc[d0])
    return {"dal": d0.strftime("%Y-%m-%d"), "al": d1.strftime("%Y-%m-%d"), "anni": round(anni, 1),
            "grezza": round(math.expm1(ga / anni), 4), "rinnovata": round(math.expm1(gb / anni), 4)}


def demo(df: pd.DataFrame, simbolo: str) -> dict:
    """Calendario e spread inventati per provare l'app con --demo."""
    cfg = ROLLATE[simbolo]
    eventi = calendario(giorni_di_borsa(df.index, df.index[-1] + pd.Timedelta(days=120)), cfg["scadenza"], cfg["anticipo"])
    t = np.arange(len(df))
    sp = pd.Series(0.03 * np.sin(2 * np.pi * t / 252) + 0.01, index=df.index)
    ultima = df.index[-1]
    prossimo = next((ev for ev in eventi if ev["roll"] > ultima), None)
    return {"fonte": "simulazione", "regola": cfg["regola"], "anticipo": cfg["anticipo"],
            "contratto": {"consegna": "demo", "simbolo": "DEMO"},
            "prossimo_roll": ({"data": prossimo["roll"].strftime("%Y-%m-%d"), "consegna": prossimo["consegna"], "nuovo": "demo"}
                              if prossimo else None),
            "verifica": None, "buchi": 0,
            "roll_date": [ev["roll"] for ev in eventi if ev["roll"] <= ultima], "spread": sp}
