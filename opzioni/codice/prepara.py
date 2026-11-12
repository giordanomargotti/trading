"""Costruisce: serie del sottostante, catena compatta (DTE<=75), controlli di qualita'."""
import glob, sys, json
import pandas as pd, numpy as np
pq, out = sys.argv[1], sys.argv[2]
parts, und, qual = [], [], []
for f in sorted(glob.glob(pq + '/*.parquet')):
    d = pd.read_parquet(f)
    n0 = len(d)
    und.append(d.groupby('QUOTE_DATE').UNDERLYING_LAST.median())
    d['DTE'] = (d.EXPIRE_DATE - d.QUOTE_DATE).dt.days.astype('int16')
    qual.append(dict(file=f.split('/')[-1], righe=n0, giorni=int(d.QUOTE_DATE.nunique()),
                     scadenze=int(d.EXPIRE_DATE.nunique()),
                     bid_ask_incrociati=int(((d.C_BID > d.C_ASK) | (d.P_BID > d.P_ASK)).sum()),
                     iv_put_mancanti=int(d.P_IV.isna().sum()), iv_call_mancanti=int(d.C_IV.isna().sum())))
    d = d[(d.DTE >= 0) & (d.DTE <= 75)]
    parts.append(d)
ch = pd.concat(parts, ignore_index=True).drop_duplicates(['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE'])
ch = ch.sort_values(['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE']).reset_index(drop=True)
ch.to_parquet(out + '/catena.parquet', index=False)
u = pd.concat(und).groupby(level=0).median().sort_index()
u.name = 'close'
u.to_frame().to_parquet(out + '/sottostante.parquet')
json.dump(qual, open(out + '/qualita.json', 'w'))
print(len(ch), ch.QUOTE_DATE.min(), ch.QUOTE_DATE.max(), ch.QUOTE_DATE.nunique(), 'giorni')
print(u.head(3), u.tail(3), len(u))
r = np.log(u).diff()
print('rendimenti estremi:', r.nsmallest(4).round(4).to_dict(), r.nlargest(3).round(4).to_dict())
g = u.index.to_series().diff().dt.days
print('buchi > 5 giorni:', g[g > 5].to_dict())
print(pd.DataFrame(qual)[['righe','giorni','bid_ask_incrociati','iv_put_mancanti']].sum())
