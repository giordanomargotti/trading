"""Funzioni comuni: Black-76 senza sconto (r=0 sul forward implicito), inversione IV, HAC,
rendimenti delle operazioni, scala mensile a capitale costante, statistiche mensili."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr

def b76(F, K, T, s, call):
    """Prezzo Black-76 non scontato. call: array booleano."""
    v = s * np.sqrt(T)
    d1 = np.log(F / K) / v + 0.5 * v
    d2 = d1 - v
    c = F * ndtr(d1) - K * ndtr(d2)
    return np.where(call, c, c - (F - K))

def iv_b76(price, F, K, T, call, lo=0.01, hi=4.0, it=60):
    """Inversione per bisezione, vettoriale. NaN se il prezzo e' fuori dai limiti di non arbitraggio."""
    price = np.asarray(price, float)
    intr = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    ok = np.isfinite(price) & (price > intr + 1e-8) & (T > 0) & (price < np.where(call, F, K))
    a = np.full(price.shape, lo); b = np.full(price.shape, hi)
    ok &= (b76(F, K, T, a, call) <= price) & (b76(F, K, T, b, call) >= price)
    for _ in range(it):
        m = 0.5 * (a + b)
        up = b76(F, K, T, m, call) < price
        a = np.where(up, m, a); b = np.where(up, b, m)
    out = 0.5 * (a + b)
    out[~ok] = np.nan
    return out

def delta_b76(F, K, T, s, call):
    v = s * np.sqrt(T)
    d1 = np.log(F / K) / v + 0.5 * v
    return np.where(call, ndtr(d1), ndtr(d1) - 1.0)

def hac_mean(x, lag):
    """Media, errore standard Newey-West (Bartlett) e statistica t."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); m = x.mean(); e = x - m
    g = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        g += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(g, 1e-300) / n)
    return m, se, m / se

def hac_ols(y, X, lag):
    """OLS con matrice di covarianza Newey-West. X include la costante."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    Z = X * u[:, None]
    S = Z.T @ Z / n
    for j in range(1, min(lag, n - 1) + 1):
        G = Z[j:].T @ Z[:-j] / n
        S += (1 - j / (lag + 1)) * (G + G.T)
    V = n * XtXi @ S @ XtXi
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return b, np.sqrt(np.diag(V)), V, r2

def rendimenti_operazioni(tr, fee):
    """Rendimento di ogni operazione sul margine e P&L in dollari per contratto, commissioni incluse.
    a = P&L a metà prezzo / margine; b = metà spread bid-ask pagata su apertura e chiusura / margine;
    f = gambe negoziate / (100 · margine), da moltiplicare per la commissione per gamba.
    rho0: a metà prezzo; rho1: pagando metà spread a ogni gamba (prezzi bid-ask). Restituisce una copia di tr."""
    tr = tr.copy()
    tr['a'] = tr.pnl_mid / tr.margine; tr['b'] = tr.hs / tr.margine; tr['f'] = (tr.gambe + tr.gambe_chiusura) / (100 * tr.margine)
    tr['rho1'] = tr.a - tr.b - fee * tr.f; tr['rho0'] = tr.a - fee * tr.f
    tr['usd1'] = 100 * (tr.pnl_mid - tr.hs) - fee * (tr.gambe + tr.gambe_chiusura); tr['usd0'] = 100 * tr.pnl_mid - fee * (tr.gambe + tr.gambe_chiusura)
    tr['exit'] = tr.t + pd.to_timedelta(tr.giorni, unit='D')
    tr['mese'] = tr.exit.dt.to_period('M')
    return tr

def scala_mensile(g, col, D, mesi):
    """Rendimento mensile di una scala a capitale costante: somma dei rendimenti delle operazioni chiuse nel mese,
    divisa per le posizioni aperte in parallelo (un ingresso per seduta per D giorni di calendario = D·252/365)."""
    slots = D * 252 / 365
    return (g.groupby('mese')[col].sum() / slots).reindex(mesi, fill_value=0.0).values

def rendimenti_mensili_spx(u):
    """Rendimenti mensili di prezzo dell'indice (fine mese su fine mese); il primo mese parte dalla prima chiusura."""
    um = u.resample('ME').last(); r = um.pct_change(); r.iloc[0] = um.iloc[0] / u.iloc[0] - 1
    return r

def statistiche_mensili(R):
    """Statistiche di una serie mensile: media e volatilità annualizzate, Sharpe, t di Newey-West (3 ritardi),
    drawdown massimo sulla somma cumulata, mese peggiore, CVaR al 5%, asimmetria, quota di mesi positivi."""
    R = np.asarray(R, float)
    m, se, t = hac_mean(R, 3)
    cum = np.cumsum(R); dd = float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max())
    sd = R.std(ddof=1)
    q = np.quantile(R, .05)
    return dict(media_annua=float(12 * m), vol_annua=float(np.sqrt(12) * sd), sharpe=float(np.sqrt(12) * m / sd) if sd > 0 else 0.0, t=float(t),
                max_dd=dd, mese_peggiore=float(R.min()), cvar5=float(R[R <= q].mean()), asimmetria=float(stats.skew(R)), mesi_positivi=float((R > 0).mean()))
