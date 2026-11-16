"""Densità reale del rendimento a scadenza con il GJR-GARCH(1,1) a innovazioni t di Student, per simulazione.
Modulo condiviso da bt_garch.py (strike scelti per probabilità) e bt_ev.py (combinazione scelta per valore atteso).

Ipotesi dichiarate:
- origine della previsione = chiusura della seduta di ingresso: varianza prevista per la seduta successiva (garch.parquet)
  e parametri dell'ultima stima disponibile a quella data (garch_parametri.csv);
- deriva nulla: il rendimento logaritmico giornaliero simulato è sigma_t * z_t, senza costante;
- orizzonte in sedute di borsa tra l'ingresso e la scadenza;
- stessi numeri casuali per le sedute che condividono la stessa stima (il parametro nu cambia solo alla ristima): a parità
  di insieme di orizzonti richiesti, i due backtest usano esattamente le stesse traiettorie.
"""
import numpy as np, pandas as pd

def sedute(ud, t, E):
    """Sedute di borsa dopo t fino alla scadenza E compresa (ud = date delle sedute, ordinate)."""
    return int(np.searchsorted(ud, np.datetime64(E), side='right') - np.searchsorted(ud, np.datetime64(t), side='right'))

def orizzonti(want, ud):
    """Da {scadenza: [(data, obiettivo)]} a {data: insieme degli orizzonti in sedute}; scadenze oltre l'ultima seduta escluse."""
    need = {}
    for E, v in want.items():
        if ud[-1] < np.datetime64(E): continue
        for t, D in v:
            h = sedute(ud, t, E)
            if h >= 1: need.setdefault(pd.Timestamp(t), set()).add(h)
    return need

def simula(g, pp, need, riduci, nsim=20000, seed=20261114):
    """need: {data: insieme di orizzonti in sedute}. riduci(data, h, campione): funzione applicata al campione dei
    rendimenti logaritmici cumulati (in %) a ogni orizzonte richiesto. Restituisce {(data, h): riduci(...)}."""
    Q = {}
    gg = g.dropna(subset=['s2next'])
    blocco = np.searchsorted(pp.index.values, gg.index.values, side='right') - 1
    rng = np.random.default_rng(seed)
    hmax = max(max(v) for v in need.values())
    for b in np.unique(blocco):
        om, al, ga, be, nu = pp.iloc[b][['omega', 'alpha', 'gamma', 'beta', 'nu']]
        Z = rng.standard_t(nu, size=(hmax, nsim)) * np.sqrt((nu - 2) / nu)        # innovazioni con varianza unitaria
        for t in gg.index[blocco == b]:
            hs = need.get(t)
            if not hs: continue
            s2 = np.full(nsim, gg.at[t, 's2next']); cum = np.zeros(nsim)
            for k in range(max(hs)):
                r = np.sqrt(s2) * Z[k]; cum += r
                if k + 1 in hs: Q[(t, k + 1)] = riduci(t, k + 1, cum)
                s2 = om + (al + ga * (r < 0)) * r * r + be * s2
    return Q
