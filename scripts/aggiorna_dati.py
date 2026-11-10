#!/usr/bin/env python3
"""
Mercati e code: aggiornamento giornaliero dei dati.

Scarica da Yahoo Finance le chiusure giornaliere di indici azionari e
commodities, calcola tre stime di volatilità (realizzata a 21 sedute, EWMA
RiskMetrics, GJR-GARCH(1,1) con innovazioni t di Student) e le statistiche
della distribuzione dei rendimenti, poi scrive un unico file JSON che la web
app (docs/index.html) legge e disegna.

Uso:
    python scripts/aggiorna_dati.py --out docs/data.json
    python scripts/aggiorna_dati.py --out docs/data.json --demo   # dati simulati, solo per prova

Se un mercato non si scarica, resta in elenco con i dati del giorno prima
(stato "vecchio"). Lo script termina con errore solo se nessun mercato è stato
aggiornato, così GitHub ti avvisa per email.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Mercati seguiti (simboli di Yahoo Finance). Per aggiungerne uno basta una
# riga in più. "implicita" è l'indice di volatilità implicita a 30 giorni da
# confrontare con la previsione GARCH a 21 sedute; None se non ce n'è uno.
# ---------------------------------------------------------------------------
ASSET = [
    {"simbolo": "^GSPC", "nome": "S&P 500", "gruppo": "Indici azionari", "implicita": "^VIX"},
    {"simbolo": "^NDX", "nome": "Nasdaq 100", "gruppo": "Indici azionari", "implicita": "^VXN"},
    {"simbolo": "^STOXX50E", "nome": "Euro Stoxx 50", "gruppo": "Indici azionari", "implicita": None},
    {"simbolo": "^GDAXI", "nome": "DAX", "gruppo": "Indici azionari", "implicita": None},
    {"simbolo": "FTSEMIB.MI", "nome": "FTSE MIB", "gruppo": "Indici azionari", "implicita": None},
    {"simbolo": "^FTSE", "nome": "FTSE 100", "gruppo": "Indici azionari", "implicita": None},
    {"simbolo": "^N225", "nome": "Nikkei 225", "gruppo": "Indici azionari", "implicita": None},
    {"simbolo": "CL=F", "nome": "Petrolio WTI", "gruppo": "Energia", "implicita": "^OVX"},
    {"simbolo": "BZ=F", "nome": "Petrolio Brent", "gruppo": "Energia", "implicita": None},
    {"simbolo": "NG=F", "nome": "Gas naturale Henry Hub", "gruppo": "Energia", "implicita": None},
    {"simbolo": "TTF=F", "nome": "Gas TTF Amsterdam", "gruppo": "Energia", "implicita": None},
    {"simbolo": "GC=F", "nome": "Oro", "gruppo": "Metalli", "implicita": "^GVZ"},
    {"simbolo": "SI=F", "nome": "Argento", "gruppo": "Metalli", "implicita": None},
    {"simbolo": "HG=F", "nome": "Rame", "gruppo": "Metalli", "implicita": None},
]
NOMI_IMPLICITA = {"^VIX": "VIX", "^VXN": "VXN", "^OVX": "OVX", "^GVZ": "GVZ"}

INIZIO = "2000-01-01"          # inizio del campione per tutti i mercati
GIORNI_ANNO = 252              # sedute per anno, per annualizzare
GIORNI_SERIE = 252             # sedute mostrate nei grafici temporali
LAMBDA_EWMA = 0.94             # RiskMetrics, dati giornalieri
FINESTRA_REALIZZATA = 21       # sedute della volatilità realizzata
ORIZZONTE_MESE = 21            # sedute della previsione "a un mese"
BORDI = np.round(np.arange(-8.0, 8.0 + 1e-9, 0.25), 2)   # classi dell'istogramma, in σ
SOGLIE = [2, 3, 4, 5]          # soglie delle code, in σ
QUOTE_HILL = {"q1": 0.01, "q25": 0.025, "q5": 0.05}       # quota del campione nella coda
PAUSA = 2.0                    # secondi tra un download e l'altro
TENTATIVI = 4


# ---------------------------------------------------------------------------
# Utilità di arrotondamento: niente NaN nel JSON
# ---------------------------------------------------------------------------
def num(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def sig(x, n=6):
    """Arrotonda a n cifre significative (prezzi di ordini di grandezza diversi)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return 0.0 if x == 0 else float(f"{x:.{n}g}")


def lista(valori, nd):
    return [num(v, nd) for v in valori]


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------
def scarica(simbolo: str, periodo: str = "max") -> pd.DataFrame:
    import yfinance as yf

    ultimo_errore = None
    for i in range(TENTATIVI):
        try:
            df = yf.Ticker(simbolo).history(
                period=periodo, interval="1d", auto_adjust=False,
                actions=False, raise_errors=True,
            )
            if df is None or df.empty or "Close" not in df.columns:
                raise ValueError("nessun dato restituito")
            return df
        except Exception as e:  # rete, limiti di Yahoo, simbolo sbagliato
            ultimo_errore = e
            if i < TENTATIVI - 1:
                time.sleep(10 * 2 ** i)
    raise RuntimeError(f"{type(ultimo_errore).__name__}: {ultimo_errore}")


def pulisci(df: pd.DataFrame) -> pd.DataFrame:
    """Indice a data di borsa, niente duplicati, niente barra della seduta in corso."""
    df = df.copy()
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)          # resta la data locale della borsa
    df.index = pd.DatetimeIndex(idx).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[[c for c in ("Close", "Volume") if c in df.columns]]
    df = df[df["Close"].notna()]
    oggi = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
    return df[(df.index >= pd.Timestamp(INIZIO)) & (df.index < oggi)]


def rendimenti_log(df: pd.DataFrame) -> pd.Series:
    """Rendimenti logaritmici tra chiusure consecutive positive.

    Una chiusura <= 0 (il WTI ad aprile 2020) non viene scavalcata: i due
    rendimenti che la toccano sono esclusi invece di inventare un rendimento
    che nessuno ha avuto. Si escludono anche i giorni fermi (variazione zero e
    volume zero), che Yahoo a volte inserisce nei giorni di festa.
    """
    p = df["Close"].to_numpy(dtype=float)
    r = np.full(p.shape, np.nan)
    ok = (p[1:] > 0) & (p[:-1] > 0)
    r[1:][ok] = np.log(p[1:][ok] / p[:-1][ok])
    if "Volume" in df.columns:
        fermo = (r == 0) & (df["Volume"].fillna(0).to_numpy() == 0)
        r[fermo] = np.nan
    return pd.Series(r, index=df.index).dropna()


# ---------------------------------------------------------------------------
# Volatilità
# ---------------------------------------------------------------------------
def varianza_ewma(r: np.ndarray, lam: float = LAMBDA_EWMA, n0: int = 30) -> np.ndarray:
    """v[t] = previsione della varianza del giorno t con i dati fino a t-1.
    L'ultimo elemento, v[len(r)], è la previsione per la prossima seduta."""
    v = np.empty(len(r) + 1)
    v[0] = np.mean(r[:n0] ** 2)
    for t in range(len(r)):
        v[t + 1] = lam * v[t] + (1 - lam) * r[t] ** 2
    return v


def stima_garch(r: pd.Series):
    """GJR-GARCH(1,1) con media costante e innovazioni t di Student.

    I rendimenti sono moltiplicati per 100 (scala consigliata da arch). Il
    segno di gamma è libero: arch impone solo alpha + gamma >= 0, quindi
    l'asimmetria rovesciata tipica delle commodity può emergere.
    """
    from arch import arch_model

    am = arch_model(100 * r, mean="Constant", vol="GARCH", p=1, o=1, q=1,
                    dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 1000})
    if res.convergence_flag != 0:
        res = am.fit(disp="off", show_warning=False, starting_values=res.params.to_numpy(),
                     options={"maxiter": 3000})
    p = res.params
    omega, alpha, gamma, beta = (float(p[k]) for k in ("omega", "alpha[1]", "gamma[1]", "beta[1]"))
    nu, mu = float(p["nu"]), float(p["mu"])
    pers = alpha + gamma / 2 + beta            # P(innovazione < 0) = 1/2 con la t simmetrica
    var_futura = res.forecast(horizon=ORIZZONTE_MESE, reindex=False).variance.to_numpy()[-1]  # in %^2
    info = {
        "omega": num(omega, 6), "alpha": num(alpha, 4), "gamma": num(gamma, 4),
        "beta": num(beta, 4), "nu": num(nu, 2), "mu": num(mu, 4),
        "persistenza": num(pers, 4),
        "emivita": num(math.log(0.5) / math.log(pers), 1) if 0 < pers < 1 else None,
        "vol_lungo": num(math.sqrt(GIORNI_ANNO * omega / (1 - pers)), 2) if pers < 1 else None,
        "convergenza": bool(res.convergence_flag == 0),
    }
    return info, var_futura, res.conditional_volatility, res.std_resid


# ---------------------------------------------------------------------------
# Distribuzione
# ---------------------------------------------------------------------------
def istogramma(x: np.ndarray) -> dict:
    conteggi, _ = np.histogram(x, bins=BORDI)
    return {"conteggi": conteggi.astype(int).tolist(),
            "sotto": int((x < BORDI[0]).sum()), "sopra": int((x > BORDI[-1]).sum())}


def code(x: np.ndarray) -> list:
    return [{"k": k, "sinistra": int((x < -k).sum()), "destra": int((x > k).sum())} for k in SOGLIE]


def hill(valori: np.ndarray, k: int):
    """Stimatore di Hill dell'indice di coda alfa sui k valori più grandi."""
    v = np.sort(valori[valori > 0])[::-1]
    if k < 10 or len(v) <= k:
        return None
    lv = np.log(v[: k + 1])
    g = float(np.mean(lv[:k] - lv[k]))
    return 1.0 / g if g > 0 else None


def statistiche(r: pd.Series, z: pd.Series | None, nu_garch: float | None) -> dict:
    x = r.to_numpy()
    n = len(x)
    media, dev = float(x.mean()), float(x.std(ddof=1))
    xs = (x - media) / dev
    df_t, loc_t, scala_t = stats.t.fit(xs)

    d4 = (x - media) ** 4
    i_max = int(np.argmax(d4))
    q01 = float(np.quantile(x, 0.01))

    out = {
        "n": n, "dal": r.index[0].strftime("%Y-%m-%d"),
        "media": num(media, 6), "dev_std": num(dev, 6),
        "asimmetria": num(stats.skew(x), 3), "curtosi": num(stats.kurtosis(x), 2),
        "var1": num(-(math.exp(q01) - 1), 5),
        "es1": num(-(float(np.mean(np.exp(x[x <= q01]))) - 1), 5),
        "hill": {
            lato: {nome: num(hill(v, max(10, round(q * n))), 2) for nome, q in QUOTE_HILL.items()}
            for lato, v in (("sinistra", -x), ("destra", x))
        },
        "max_somma": {"quota": num(d4[i_max] / d4.sum(), 4),
                      "data": r.index[i_max].strftime("%Y-%m-%d"),
                      "rendimento": num(math.exp(x[i_max]) - 1, 5)},
        "grezzi": {**istogramma(xs), "code": code(xs),
                   "t": {"df": num(df_t, 3), "loc": num(loc_t, 4), "scala": num(scala_t, 4)},
                   # probabilità di coda attese con la t stimata sui rendimenti standardizzati
                   "code_t": [{"k": k,
                               "sinistra": float(stats.t.cdf((-k - loc_t) / scala_t, df_t)),
                               "destra": float(stats.t.sf((k - loc_t) / scala_t, df_t))} for k in SOGLIE]},
        "garch": None,
    }
    if z is not None:
        zz = z.dropna().to_numpy()
        out["garch"] = {**istogramma(zz), "code": code(zz), "n": int(len(zz)), "nu": num(nu_garch, 2),
                        "code_t": None}
        if nu_garch and nu_garch > 2:
            s = math.sqrt((nu_garch - 2) / nu_garch)   # t con varianza unitaria, come nel GARCH
            p = {k: float(stats.t.sf(k / s, nu_garch)) for k in SOGLIE}
            out["garch"]["code_t"] = [{"k": k, "sinistra": p[k], "destra": p[k]} for k in SOGLIE]
    return out


def previsione(prezzo: float, sigma: float, nu: float | None) -> dict:
    """Quantili della chiusura di domani, senza drift."""
    quantili = (0.01, 0.05, 0.5, 0.95, 0.99)
    out = {"prezzo": sig(prezzo, 7), "sigma": num(sigma, 6), "nu": num(nu, 2), "t": None,
           "n": {str(round(q * 100)): sig(prezzo * math.exp(stats.norm.ppf(q) * sigma), 7) for q in quantili}}
    if nu and nu > 2:
        s = math.sqrt((nu - 2) / nu)           # t con varianza unitaria
        out["t"] = {str(round(q * 100)): sig(prezzo * math.exp(stats.t.ppf(q, nu) * s * sigma), 7)
                    for q in quantili}
    return out


# ---------------------------------------------------------------------------
# Analisi di un mercato
# ---------------------------------------------------------------------------
def analizza(cfg: dict, df: pd.DataFrame, impl: pd.DataFrame | None) -> tuple[dict, dict]:
    """Restituisce la voce per data.json e lo storico completo per il laboratorio."""
    r = rendimenti_log(df)
    if len(r) < 500:
        raise ValueError(f"storico troppo corto ({len(r)} rendimenti)")
    x = r.to_numpy()

    v_ewma = varianza_ewma(x)
    ewma_serie = pd.Series(np.sqrt(GIORNI_ANNO * v_ewma[:-1]) * 100, index=r.index)
    real_serie = np.sqrt(GIORNI_ANNO * (r ** 2).rolling(FINESTRA_REALIZZATA).mean()) * 100

    garch, var_futura, cond_vol, z = None, None, None, None
    try:
        garch, var_futura, cond_vol, z = stima_garch(r)
    except Exception as e:  # il GARCH non deve bloccare il resto
        print(f"    GARCH non stimato per {cfg['simbolo']}: {e}", file=sys.stderr)

    data_ult = r.index[-1]
    chiusura = float(df.loc[data_ult, "Close"])
    if garch is not None:
        sigma1 = math.sqrt(var_futura[0]) / 100
        z_ult = float(z.iloc[-1])
        nu = garch["nu"]
        vol21 = math.sqrt(var_futura[:ORIZZONTE_MESE].sum() * GIORNI_ANNO / ORIZZONTE_MESE)
    else:
        sigma1 = math.sqrt(v_ewma[-1])
        z_ult = float(x[-1] / math.sqrt(v_ewma[-2]))
        nu, vol21 = None, None

    implicita = None
    if impl is not None and cfg.get("implicita"):
        impl = impl[impl.index <= data_ult]
        if len(impl):
            valore = float(impl["Close"].iloc[-1])
            implicita = {"simbolo": cfg["implicita"], "nome": NOMI_IMPLICITA.get(cfg["implicita"], cfg["implicita"]),
                         "valore": num(valore, 2), "data": impl.index[-1].strftime("%Y-%m-%d"),
                         "differenza": num(valore - vol21, 2) if vol21 is not None else None}

    coda = r.index[-GIORNI_SERIE:]
    serie = {
        "date": [d.strftime("%Y-%m-%d") for d in coda],
        "chiusura": [sig(v) for v in df["Close"].reindex(coda).to_numpy()],
        "rendimento": lista(np.exp(r.loc[coda].to_numpy()) - 1, 5),
        "realizzata": lista(real_serie.loc[coda].to_numpy(), 2),
        "ewma": lista(ewma_serie.loc[coda].to_numpy(), 2),
        "garch": lista(cond_vol.loc[coda].to_numpy() * math.sqrt(GIORNI_ANNO), 2) if cond_vol is not None else None,
        "z": lista(z.loc[coda].to_numpy(), 2) if z is not None else None,
    }

    # Storico completo per il laboratorio di strategie. Tutto è allineato alle date
    # dei rendimenti: z[t] è il rendimento del giorno t diviso per la σ prevista per
    # quel giorno; s[t] è la σ prevista per la seduta DOPO t, nota alla chiusura di t.
    # Così chi entra alla chiusura di t usa solo informazioni disponibili in quel momento.
    storia = {
        "simbolo": cfg["simbolo"],
        "date": [d.strftime("%Y-%m-%d") for d in r.index],
        "c": [sig(v) for v in df["Close"].reindex(r.index).to_numpy()],
        "ze": lista(x / np.sqrt(v_ewma[:-1]), 2),
        "se": lista(np.sqrt(v_ewma[1:]) * 100, 3),            # % giornaliera
        "zg": None, "sg": None,
    }
    if garch is not None:
        cv = cond_vol.to_numpy()                                # σ GARCH del giorno t, in %
        storia["zg"] = lista(z.to_numpy(), 2)
        storia["sg"] = lista(np.append(cv[1:], math.sqrt(var_futura[0])), 3)

    return {
        "simbolo": cfg["simbolo"], "nome": cfg["nome"], "gruppo": cfg["gruppo"],
        "stato": "ok", "errore": None,
        "ultimo": {"data": data_ult.strftime("%Y-%m-%d"), "chiusura": sig(chiusura, 7),
                   "var": num(math.exp(x[-1]) - 1, 5), "z": num(z_ult, 2),
                   "z_fonte": "garch" if garch is not None else "ewma"},
        "vol": {
            "realizzata": num(real_serie.iloc[-1], 2),
            "ewma": num(math.sqrt(GIORNI_ANNO * v_ewma[-1]) * 100, 2),
            "garch_domani": num(sigma1 * math.sqrt(GIORNI_ANNO) * 100, 2) if garch is not None else None,
            "garch_mese": num(vol21, 2),
            "garch_lungo": garch["vol_lungo"] if garch is not None else None,
        },
        "garch": garch,
        "implicita": implicita,
        "domani": previsione(chiusura, sigma1, nu),
        "distribuzione": statistiche(r, z, nu),
        "serie": serie,
    }, storia


def nome_file_storia(simbolo: str) -> str:
    return "storia/" + re.sub(r"[^A-Za-z0-9]+", "_", simbolo).strip("_") + ".json"


# ---------------------------------------------------------------------------
# Dati simulati (--demo): GJR-GARCH-t con parametri plausibili, per provare l'app
# ---------------------------------------------------------------------------
PARAM_DEMO = {
    "Indici azionari": dict(omega=0.02, alpha=0.02, gamma=0.13, beta=0.89, nu=7.0, mu=0.03),
    "Energia": dict(omega=0.06, alpha=0.06, gamma=-0.02, beta=0.925, nu=4.5, mu=0.0),
    "Metalli": dict(omega=0.02, alpha=0.04, gamma=-0.01, beta=0.95, nu=5.5, mu=0.02),
}
PREZZI_DEMO = {"^GSPC": 6800, "^NDX": 25000, "^STOXX50E": 5600, "^GDAXI": 24000, "FTSEMIB.MI": 43000,
               "^FTSE": 9700, "^N225": 50000, "CL=F": 60, "BZ=F": 64, "NG=F": 4.3, "TTF=F": 31,
               "GC=F": 4000, "SI=F": 48, "HG=F": 5.0}


def simula(cfg: dict, seme: int) -> pd.DataFrame:
    rng = np.random.default_rng(seme)
    p = PARAM_DEMO[cfg["gruppo"]].copy()
    if cfg["simbolo"] in ("NG=F", "TTF=F"):
        p.update(omega=0.25, alpha=0.08, gamma=-0.04, beta=0.90, nu=4.0)
    ieri = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date()) - pd.Timedelta(days=1)
    date = pd.bdate_range(INIZIO, ieri)
    n = len(date)
    e_std = rng.standard_t(p["nu"], n) * math.sqrt((p["nu"] - 2) / p["nu"])
    h = p["omega"] / (1 - p["alpha"] - p["gamma"] / 2 - p["beta"])
    r = np.empty(n)
    e_prec = 0.0
    for t in range(n):
        h = p["omega"] + (p["alpha"] + p["gamma"] * (e_prec < 0)) * e_prec ** 2 + p["beta"] * h
        e_prec = math.sqrt(h) * e_std[t]
        r[t] = p["mu"] + e_prec
    prezzi = np.exp(np.cumsum(r / 100))
    prezzi *= PREZZI_DEMO.get(cfg["simbolo"], 100) / prezzi[-1]
    return pd.DataFrame({"Close": prezzi, "Volume": 1.0}, index=date)


def implicita_demo(df: pd.DataFrame, seme: int) -> pd.DataFrame:
    rng = np.random.default_rng(1000 + seme)
    r = rendimenti_log(df).to_numpy()[-63:]
    livello = float(np.sqrt(GIORNI_ANNO * np.mean(r ** 2)) * 100 * 1.15 + rng.normal(0, 1))
    return pd.DataFrame({"Close": [livello]}, index=[df.index[-1]])


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Aggiorna i dati della web app Mercati e code.")
    ap.add_argument("--out", default="docs/data.json", help="file JSON da scrivere")
    ap.add_argument("--demo", action="store_true", help="usa dati simulati invece di Yahoo Finance")
    args = ap.parse_args()
    out = Path(args.out)

    precedenti = {}
    if out.exists() and not args.demo:
        try:
            vecchi = json.loads(out.read_text("utf-8"))
            if not vecchi.get("demo"):
                precedenti = {a["simbolo"]: a for a in vecchi.get("asset", [])
                              if a.get("stato") in ("ok", "vecchio")}
        except Exception as e:
            print(f"File precedente illeggibile, si riparte da zero: {e}", file=sys.stderr)

    risultati, aggiornati = [], 0
    for i, cfg in enumerate(ASSET):
        try:
            if args.demo:
                df = simula(cfg, i)
            else:
                if i:
                    time.sleep(PAUSA)
                df = pulisci(scarica(cfg["simbolo"]))
            impl = None
            if cfg.get("implicita"):
                try:
                    impl = implicita_demo(df, i) if args.demo else pulisci(scarica(cfg["implicita"], "3mo"))
                except Exception as e:
                    print(f"    implicita {cfg['implicita']} non disponibile: {e}", file=sys.stderr)
            voce, storia = analizza(cfg, df, impl)
            # lo storico del laboratorio va in un file a parte, letto solo quando serve
            voce["storia"] = nome_file_storia(cfg["simbolo"])
            file_storia = out.parent / voce["storia"]
            file_storia.parent.mkdir(parents=True, exist_ok=True)
            file_storia.write_text(json.dumps(storia, separators=(",", ":"), allow_nan=False), "utf-8")
            aggiornati += 1
            print(f"ok   {cfg['simbolo']:<11} {voce['ultimo']['data']}  chiusura {voce['ultimo']['chiusura']}")
        except Exception as e:
            msg = (str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}")[:200]
            print(f"ERR  {cfg['simbolo']:<11} {msg}", file=sys.stderr)
            voce = dict(precedenti.get(cfg["simbolo"], {}))
            voce["stato"] = "vecchio" if voce else "errore"
            voce["errore"] = msg
            voce.pop("storia", None)   # lo storico di oggi non c'è: il laboratorio resta spento
        voce.update({k: cfg[k] for k in ("simbolo", "nome", "gruppo")})
        risultati.append(voce)

    dati = {
        "versione": 1,
        "generato": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fonte": "Simulazione GJR-GARCH-t (dati di prova)" if args.demo else "Yahoo Finance",
        "demo": bool(args.demo),
        "inizio": INIZIO,
        "bordi": BORDI.tolist(),
        "soglie": SOGLIE,
        "gaussiana_code": {str(k): float(stats.norm.sf(k)) for k in SOGLIE},
        "parametri": {"lambda_ewma": LAMBDA_EWMA, "finestra_realizzata": FINESTRA_REALIZZATA,
                      "orizzonte_mese": ORIZZONTE_MESE, "giorni_anno": GIORNI_ANNO,
                      "quote_hill": QUOTE_HILL},
        "asset": risultati,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(dati, ensure_ascii=False, separators=(",", ":"), allow_nan=False), "utf-8")
    tmp.replace(out)
    print(f"\n{aggiornati}/{len(ASSET)} mercati aggiornati -> {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0 if aggiornati else 1


if __name__ == "__main__":
    sys.exit(main())
