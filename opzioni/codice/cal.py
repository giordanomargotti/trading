"""Calendar lungo su put ATM: vendita della scadenza vicina, acquisto della scadenza successiva sullo stesso strike,
chiusura alla scadenza della gamba venduta (gamba superstite valutata al prezzo medio di chiusura di quel giorno)."""
import sys, glob, numpy as np, pandas as pd
out = sys.argv[1]
u = pd.read_parquet(out + '/sottostante.parquet').close
ud = u.index.values
cols = ['QUOTE_DATE', 'EXPIRE_DATE', 'DTE', 'STRIKE', 'F', 'UNDERLYING_LAST', 'P_BID', 'P_ASK']
ch = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(out + '/c2/*.parquet'))], ignore_index=True)
ch = ch[(ch.STRIKE / ch.UNDERLYING_LAST - 1).abs() < 0.12]
ch = ch[(ch.P_BID > 0) & (ch.P_ASK >= ch.P_BID)]
ch['mid'] = 0.5 * (ch.P_BID + ch.P_ASK); ch['hs'] = 0.5 * (ch.P_ASK - ch.P_BID)
ix = ch.set_index(['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE'])[['mid', 'hs']].sort_index()
exp_by_date = ch.groupby('QUOTE_DATE').EXPIRE_DATE.unique()
rec = []
for D, (lo, hi, tgt) in {30: (21, 42, 30), 7: (5, 10, 7)}.items():
    s = pd.read_parquet(out + f'/serie_{D}.parquet')
    for t, E, F, S in zip(s.index, s.E, s.F, s.S):
        ie = np.searchsorted(ud, np.datetime64(E), side='right') - 1
        if ud[-1] < np.datetime64(E): continue
        tE = pd.Timestamp(ud[ie]); ST = u.values[ie]
        cand = [e for e in exp_by_date.get(t, []) if lo <= (e - E).days <= hi]
        if not cand: continue
        E2 = min(cand, key=lambda e: abs((e - E).days - tgt))
        try:
            front = ix.loc[(t, E)]; back = ix.loc[(t, E2)]
        except KeyError:
            continue
        ks = front.index.intersection(back.index)
        if len(ks) == 0: continue
        K = ks[np.argmin(np.abs(ks.values - F))]
        if abs(K - F) > 0.01 * S: continue
        try:
            ex = ix.loc[(tE, E2, K)]
        except KeyError:
            continue
        v0 = back.loc[K, 'mid'] - front.loc[K, 'mid']
        if v0 <= 0: continue
        h = front.loc[K, 'hs'] + back.loc[K, 'hs'] + ex.hs
        pnl = ex.mid - max(K - ST, 0.0) - v0
        rec.append(('CAL', 0.5, '-', D, t, E, 2, v0, -v0, S, ST, 'hold', pnl, h, 1, (E - t).days, 0))
c = pd.DataFrame(rec, columns=['fam', 'par', 'ala', 'D', 't', 'E', 'gambe', 'margine', 'credito', 'S', 'ST', 'gest', 'pnl_mid', 'hs', 'gambe_chiusura', 'giorni', 'uscita'])
tr = pd.read_parquet(out + '/operazioni.parquet')
tr = pd.concat([tr[tr.fam != 'CAL'], c], ignore_index=True)
tr.to_parquet(out + '/operazioni.parquet', index=False)
print(c.groupby('D').agg(n=('pnl_mid', 'size'), debito=('margine', 'mean'), pnl=('pnl_mid', 'mean'), hs=('hs', 'mean'), vinc=('pnl_mid', lambda x: (x > 0).mean())))
