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

print("\nTutto ok" if not errori else f"\n{len(errori)} controlli falliti: {errori}")
sys.exit(1 if errori else 0)
