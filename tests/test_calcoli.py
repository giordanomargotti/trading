#!/usr/bin/env python3
"""
Controlli dei calcoli su dati con proprietà note. Non serve la rete.

    python tests/test_calcoli.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aggiorna_dati as ad  # noqa: E402

errori = []


def controlla(nome, condizione, dettaglio=""):
    print(f"{'ok ' if condizione else 'NO '} {nome}  {dettaglio}")
    if not condizione:
        errori.append(nome)


def gjr(n, omega, alpha, gamma, beta, nu, mu, seme):
    rng = np.random.default_rng(seme)
    e_std = rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
    h = omega / (1 - alpha - gamma / 2 - beta)
    r, e = np.empty(n), 0.0
    for t in range(n):
        h = omega + (alpha + gamma * (e < 0)) * e * e + beta * h
        e = math.sqrt(h) * e_std[t]
        r[t] = mu + e
    return pd.Series(r / 100, index=pd.bdate_range("1990-01-01", periods=n))


# 1. Il GJR-GARCH-t recupera i parametri, anche con gamma negativo (commodity)
for nome, vero in {
    "azionario": dict(omega=0.02, alpha=0.02, gamma=0.12, beta=0.90, nu=6.0, mu=0.03),
    "commodity": dict(omega=0.05, alpha=0.08, gamma=-0.04, beta=0.90, nu=5.0, mu=0.0),
}.items():
    info, var_futura, cond_vol, z = ad.stima_garch(gjr(30000, seme=1, **vero))
    for k, tol in (("alpha", 0.02), ("gamma", 0.03), ("beta", 0.02), ("nu", 1.0)):
        controlla(f"GARCH {nome}: {k}", abs(info[k] - vero[k]) < tol, f"stimato {info[k]} vero {vero[k]}")
    pers = vero["alpha"] + vero["gamma"] / 2 + vero["beta"]
    controlla(f"GARCH {nome}: persistenza", abs(info["persistenza"] - pers) < 0.01,
              f"stimata {info['persistenza']} vera {pers:.4f}")
    controlla(f"GARCH {nome}: residui standardizzati con varianza circa 1",
              abs(float(z.var()) - 1) < 0.05, f"{float(z.var()):.3f}")

# 2. Hill: Pareto con alfa noto e t di Student con nu gradi di libertà
rng = np.random.default_rng(7)
pareto = (rng.pareto(2.5, 200_000) + 1)
a = ad.hill(pareto, 5000)
controlla("Hill su Pareto alfa 2,5", abs(a - 2.5) < 0.1, f"{a:.3f}")
t3 = rng.standard_t(3, 200_000)
a = ad.hill(t3, 2000)
controlla("Hill sulla coda destra di una t(3)", abs(a - 3) < 0.35, f"{a:.3f}")

# 3. Code: con dati gaussiani il rapporto osservato/gaussiana è circa 1
x = rng.standard_normal(1_000_000)
for c in ad.code(x)[:2]:
    k = c["k"]
    attesi = stats.norm.sf(k) * len(x)
    controlla(f"Code gaussiane oltre {k}σ", abs(c["sinistra"] / attesi - 1) < 0.1 and abs(c["destra"] / attesi - 1) < 0.1,
              f"sinistra {c['sinistra']} destra {c['destra']} attesi {attesi:.0f}")

# 4. EWMA uguale alla ricorsione di pandas
r = rng.standard_normal(500) / 100
v = ad.varianza_ewma(r)
s = pd.Series(np.r_[np.mean(r[:30] ** 2), r ** 2]).ewm(alpha=1 - ad.LAMBDA_EWMA, adjust=False).mean().to_numpy()
controlla("EWMA uguale a pandas ewm", np.allclose(v, s), f"max diff {np.max(np.abs(v - s)):.2e}")

# 5. Prezzo negativo non scavalcato e giorni fermi esclusi
idx = pd.bdate_range("2020-04-15", periods=6)
df = pd.DataFrame({"Close": [20.0, 18.27, -37.63, 10.01, 10.01, 12.0], "Volume": [1, 1, 1, 1, 0, 1]}, index=idx)
rr = ad.rendimenti_log(df)
controlla("WTI: niente rendimenti attorno alla chiusura negativa",
          list(rr.index) == [idx[1], idx[5]], f"date tenute {[d.date().isoformat() for d in rr.index]}")
controlla("WTI: il primo rendimento è corretto", abs(rr.iloc[0] - math.log(18.27 / 20)) < 1e-12)

# 6. Pulizia: fuso orario della borsa, duplicati, seduta in corso esclusa
oggi = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
idx = pd.DatetimeIndex([oggi - pd.Timedelta(days=d) for d in (3, 2, 2, 0)]).tz_localize("America/New_York")
grezzo = pd.DataFrame({"Close": [1.0, 2.0, 2.5, 3.0], "Volume": [1, 1, 1, 1]}, index=idx)
pulito = ad.pulisci(grezzo)
controlla("Pulizia: duplicato tenuto una volta, barra di oggi tolta",
          len(pulito) == 2 and pulito["Close"].iloc[-1] == 2.5, f"{pulito['Close'].tolist()}")

# 7. Quantili di domani: la t con varianza unitaria ha code più larghe della gaussiana al 99%
p = ad.previsione(100.0, 0.01, 4.0)
controlla("Previsione: t più larga della gaussiana all'1%", p["t"]["1"] < p["n"]["1"], f"t {p['t']['1']} n {p['n']['1']}")
controlla("Previsione: t più stretta della gaussiana al 5%", p["t"]["5"] > p["n"]["5"], f"t {p['t']['5']} n {p['n']['5']}")

# 8. Storico del laboratorio: niente informazioni dal futuro negli ingressi
cfg = {"simbolo": "TEST", "nome": "Test", "gruppo": "Energia", "implicita": None}
df = ad.simula(cfg, 3)
voce, st = ad.analizza(cfg, df, None)
rr = ad.rendimenti_log(df)
info, var_futura, cond_vol, z = ad.stima_garch(rr)
cv = cond_vol.to_numpy()
v = ad.varianza_ewma(rr.to_numpy())
controlla("Storico: tutte le serie della stessa lunghezza",
          len({len(st[k]) for k in ("date", "c", "zg", "sg", "ze", "se")}) == 1, f"{len(st['date'])} sedute")
controlla("Storico: σ GARCH è quella della seduta dopo",
          np.allclose(st["sg"][:-1], np.round(cv[1:], 3)) and abs(st["sg"][-1] - round(math.sqrt(var_futura[0]), 3)) < 1e-9)
controlla("Storico: σ EWMA è quella della seduta dopo", np.allclose(st["se"], np.round(np.sqrt(v[1:]) * 100, 3)))
controlla("Storico: z EWMA usa la σ prevista per quel giorno",
          np.allclose(st["ze"], np.round(rr.to_numpy() / np.sqrt(v[:-1]), 2)))
controlla("Storico: nome del file", ad.nome_file_storia("TTF=F") == "storia/TTF_F.json" and
          ad.nome_file_storia("^GSPC") == "storia/GSPC.json")

# 9. Serie rinnovata a ogni scadenza: con contratti che si muovono tutti come lo spot e differiscono
#    solo per una base fissa, chi rinnova guadagna esattamente il rendimento dello spot; la serie
#    continua grezza invece salta della differenza di base nel giorno del cambio.
import serie_rollate as sr  # noqa: E402


def contratti_sintetici(regola, anticipo, seme):
    rng = np.random.default_rng(seme)
    tutti = [g for g in sr.giorni_di_borsa([], pd.Timestamp("2013-06-30")) if g >= pd.Timestamp("2005-01-03")]
    eventi = sr.calendario(tutti, regola, anticipo)          # il calendario va oltre l'ultimo prezzo
    giorni = [g for g in tutti if g <= pd.Timestamp("2012-12-31")]
    n = len(giorni)
    spot = np.exp(np.cumsum(rng.normal(0, 0.03, n))) * 4.0
    # base del contratto con consegna nel mese m: stagionale e diversa da mese a mese
    base = {ev["consegna"]: 0.08 * math.sin(int(ev["consegna"][5:]) / 12 * 2 * math.pi) + rng.normal(0, 0.03) for ev in eventi}
    prezzo = {c: pd.Series(spot * math.exp(b), index=giorni) for c, b in base.items()}
    eventi = [ev for ev in eventi if ev["scadenza"] <= giorni[-1] or ev is next(e for e in eventi if e["scadenza"] > giorni[-1])] + \
        [ev for ev in eventi if ev["scadenza"] > giorni[-1]][1:]
    c1, c2 = [], []
    for t in giorni:
        vivi = [ev["consegna"] for ev in eventi if ev["scadenza"] >= t]
        c1.append(prezzo[vivi[0]][t]); c2.append(prezzo[vivi[1]][t])
    return giorni, spot, eventi, prezzo, pd.Series(c1, index=giorni), pd.Series(c2, index=giorni)


for regola, anticipo in (("ng", 0), ("cl", 2)):
    giorni, spot, eventi, prezzo, c1, c2 = contratti_sintetici(regola, anticipo, 11)
    tab = sr.rendimenti_rinnovati(c1, c2, eventi)
    vero = np.diff(np.log(spot))
    ok = np.allclose(tab["r"].to_numpy()[1:], vero, atol=1e-12)
    grezzo = np.diff(np.log(c1.to_numpy()))
    salti = int((np.abs(grezzo - vero) > 1e-9).sum())
    controlla(f"Rinnovo {regola}: rendimenti uguali allo spot", ok, f"salti nella serie grezza {salti}")
    ver = sr.verifica_calendario(tab)
    controlla(f"Rinnovo {regola}: la verifica riconosce il calendario giusto",
              ver["calcolato"] > 0.75 and ver["calcolato"] > max(ver["prima"], ver["dopo"]) + 0.3, str(ver))
    # chiusura del CFD: tra il roll e la scadenza si tiene il secondo contratto
    ev = eventi[30]
    i_r, i_e = giorni.index(ev["roll"]), giorni.index(ev["scadenza"])
    controlla(f"Rinnovo {regola}: {anticipo} sedute tra chiusura del CFD e scadenza", i_e - i_r == anticipo)

# scadenze note. NG dicembre 2026: terzultimo giorno lavorativo di novembre, il 25 (il 26 è Thanksgiving).
# CL gennaio 2027: il 25 dicembre è festa, quindi quattro giorni lavorativi prima, il 21; CFD chiuso il 17.
# CL gennaio 2026: scaduto il 19 dicembre 2025 (stessa regola, il 25 era giovedì).
giorni = sr.giorni_di_borsa([], pd.Timestamp("2027-03-31"))
ng = {ev["consegna"]: ev for ev in sr.calendario(giorni, "ng", 0)}
cl = {ev["consegna"]: ev for ev in sr.calendario(giorni, "cl", 2)}
controlla("Calendario NG: dicembre 2026 scade il 25/11/2026", ng["2026-12"]["scadenza"] == pd.Timestamp("2026-11-25"),
          str(ng["2026-12"]["scadenza"].date()))
controlla("Calendario CL: gennaio 2027 scade il 21/12/2026, CFD chiuso il 17/12",
          cl["2027-01"]["scadenza"] == pd.Timestamp("2026-12-21") and cl["2027-01"]["roll"] == pd.Timestamp("2026-12-17"),
          f"{cl['2027-01']['scadenza'].date()} / {cl['2027-01']['roll'].date()}")
controlla("Calendario CL: gennaio 2026 scaduto il 19/12/2025", cl["2026-01"]["scadenza"] == pd.Timestamp("2025-12-19"))
controlla("Calendario CL: maggio 2020 scade il 21/04/2020, CFD chiuso il 17/04",
          {ev["consegna"]: ev for ev in sr.calendario(sr.giorni_di_borsa([], pd.Timestamp("2020-06-30")), "cl", 2)}["2020-05"]["roll"]
          == pd.Timestamp("2020-04-17"))

# coda con i contratti di Yahoo dopo l'ultimo dato EIA: stesso risultato della serie tutta EIA
giorni, spot, eventi, prezzo, c1, c2 = contratti_sintetici("ng", 0, 5)
taglio = giorni[-40]
originale = sr.scarica_eia
sr.scarica_eia = lambda url: (c1 if "C1d" in url else c2)[lambda s: s.index <= taglio]
yahoo = pd.DataFrame({"Close": c1})
simboli = {sr.simbolo_yahoo(sr.ROLLATE["NG=F"], c): c for c in prezzo}
fake_rapido = lambda sim, periodo: pd.DataFrame({"Close": prezzo[simboli[sim]]}) if sim in simboli else None
try:
    df_r, info = sr.serie_rinnovata("NG=F", yahoo, fake_rapido, lambda d: d, giorni[-1])
finally:
    sr.scarica_eia = originale
r_r = np.diff(np.log(df_r["Close"].to_numpy()))
tenuto = info["contratto"]["consegna"]
controlla("Rinnovo con coda Yahoo: rendimenti uguali allo spot", np.allclose(r_r, np.diff(np.log(spot)), atol=1e-10))
controlla("Rinnovo con coda Yahoo: l'ultimo valore è il prezzo del contratto tenuto",
          abs(df_r["Close"].iloc[-1] - prezzo[tenuto].iloc[-1]) < 1e-9, f"{tenuto}")
controlla("Rinnovo con coda Yahoo: date di chiusura del CFD nella coda",
          sum(1 for d in info["roll_date"] if d > taglio) == sum(1 for ev in eventi if taglio < ev["roll"] <= giorni[-1]))

print("\nTutto ok" if not errori else f"\n{len(errori)} controlli falliti: {errori}")
sys.exit(1 if errori else 0)
