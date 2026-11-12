"""Funzioni comuni: Black-76 senza sconto (r=0 sul forward implicito), inversione IV, HAC."""
import numpy as np
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
