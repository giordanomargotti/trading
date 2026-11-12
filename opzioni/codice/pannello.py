"""Pannello (data, scadenza): IV ATM, varianza implicita model-free, varianza realizzata fino a scadenza, skew."""
import sys, glob, numpy as np, pandas as pd
out = sys.argv[1]
u = pd.read_parquet(out + '/sottostante.parquet').close
lr = np.log(u).diff()
cs2 = (lr ** 2).fillna(0).cumsum()          # somma cumulata dei quadrati dei rendimenti logaritmici
dates = u.index.values
rows = []
for f in sorted(glob.glob(out + '/c2/*.parquet')):
    ch = pd.read_parquet(f, columns=['QUOTE_DATE', 'EXPIRE_DATE', 'DTE', 'STRIKE', 'F', 'UNDERLYING_LAST',
                                     'C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'P_IVX', 'C_IVX', 'P_DX', 'C_DX'])
    ch = ch[ch.DTE >= 5]
    for (t, E), g in ch.groupby(['QUOTE_DATE', 'EXPIRE_DATE'], sort=False):
        K = g.STRIKE.values; F = g.F.values[0]; T = g.DTE.values[0] / 365.0
        if len(K) < 8: continue
        put = K < F
        ivo = np.where(put, g.P_IVX.values, g.C_IVX.values).astype(float)
        m = np.isfinite(ivo)
        if m.sum() < 6 or not (K[m].min() < F < K[m].max()): continue
        iv_atm = float(np.interp(F, K[m], ivo[m]))
        # varianza model-free (formula CBOE): opzioni OTM al prezzo medio, troncamento a due bid nulli consecutivi
        bid = np.where(put, g.P_BID.values, g.C_BID.values); ask = np.where(put, g.P_ASK.values, g.C_ASK.values)
        q = 0.5 * (bid + ask)
        i0 = np.searchsorted(K, F) - 1          # K0: primo strike <= F
        if i0 < 1 or i0 >= len(K) - 2: continue
        keep = np.zeros(len(K), bool)
        z = 0
        for i in range(i0, -1, -1):             # lato put
            if bid[i] <= 0:
                z += 1
                if z >= 2: break
                continue
            z = 0; keep[i] = True
        z = 0
        for i in range(i0 + 1, len(K)):         # lato call
            if bid[i] <= 0:
                z += 1
                if z >= 2: break
                continue
            z = 0; keep[i] = True
        Kk = K[keep]; qk = q[keep].copy()
        if len(Kk) < 6: continue
        j0 = np.where(Kk == K[i0])[0]
        if len(j0):                              # a K0 si usa la media put/call
            qk[j0[0]] = 0.5 * (0.5 * (g.P_BID.values[i0] + g.P_ASK.values[i0]) + 0.5 * (g.C_BID.values[i0] + g.C_ASK.values[i0]))
        dK = np.gradient(Kk)
        mf = 2.0 / T * np.sum(dK / Kk ** 2 * qk) - (F / K[i0] - 1.0) ** 2 / T
        # skew: IV della put delta 0,25 meno IV della call delta 0,25; IV put delta 0,10
        pdx = g.P_DX.values.astype(float); cdx = g.C_DX.values.astype(float)
        def iv_at(dx, iv, target):
            mm = np.isfinite(dx) & np.isfinite(iv)
            if mm.sum() < 3: return np.nan
            o = np.argsort(dx[mm]); x = dx[mm][o]; y = iv[mm][o]
            if not (x[0] <= target <= x[-1]): return np.nan
            return float(np.interp(target, x, y))
        p25 = iv_at(pdx, g.P_IVX.values.astype(float), -0.25); c25 = iv_at(cdx, g.C_IVX.values.astype(float), 0.25)
        p10 = iv_at(pdx, g.P_IVX.values.astype(float), -0.10)
        # varianza realizzata da t alla scadenza (ultima chiusura disponibile <= scadenza)
        it = np.searchsorted(dates, np.datetime64(t)); ie = np.searchsorted(dates, np.datetime64(E), side='right') - 1
        if ie <= it or ie >= len(dates):
            rv = np.nan; n = 0; ST = np.nan; complete = False
        else:
            complete = (dates[-1] >= np.datetime64(E))
            n = ie - it; rv = (cs2.values[ie] - cs2.values[it]) / T; ST = u.values[ie]
        rows.append((t, E, g.DTE.values[0], g.UNDERLYING_LAST.values[0], F, iv_atm, mf, rv if complete else np.nan, n,
                     ST if complete else np.nan, p25, c25, p10, int(keep.sum())))
    print(f, len(rows), flush=True)
p = pd.DataFrame(rows, columns=['t', 'E', 'dte', 'S', 'F', 'iv_atm', 'mfvar', 'rvar', 'n', 'ST', 'iv_p25', 'iv_c25', 'iv_p10', 'nstrikes'])
p.to_parquet(out + '/pannello.parquet', index=False)
print(p.describe().T[['count', 'mean', '50%', 'min', 'max']])
