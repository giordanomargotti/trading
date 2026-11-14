"""Distanza dal forward della put con delta 0,16 a 30 giorni in funzione della volatilità implicita ATM all'ingresso.
Mostra che gli strike scelti per delta si allargano da soli quando la volatilità implicita sale.
Scrive la chiave distanza_strike in filtro_vol.json (terzili di IV, correlazione, retta dei minimi quadrati).
Uso: python distanza_strike.py <cartella out>"""
import sys, glob, json, numpy as np, pandas as pd
out = sys.argv[1]
s = pd.read_parquet(out + '/serie_30.parquet')[['E', 'iv_atm']].reset_index()
s.columns = ['QUOTE_DATE', 'EXPIRE_DATE', 'iv']
cols = ['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'F', 'P_BID', 'P_ASK', 'P_DX']
ch = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(out + '/c2/*.parquet'))], ignore_index=True)
ch = ch.merge(s, on=['QUOTE_DATE', 'EXPIRE_DATE'])
ch = ch[(ch.P_BID > 0) & (ch.P_ASK >= ch.P_BID) & np.isfinite(ch.P_DX) & (ch.STRIKE < ch.F)]
ch['err'] = (ch.P_DX + 0.16).abs()
x = ch.loc[ch.groupby('QUOTE_DATE').err.idxmin()]
x = x[x.err < 0.05].dropna(subset=['iv'])
x['dist'] = 1 - x.STRIKE / x.F
q = x.iv.quantile([1 / 3, 2 / 3]).values
x['classe'] = np.where(x.iv <= q[0], 'bassa', np.where(x.iv <= q[1], 'media', 'alta'))
b, a = np.polyfit(x.iv, x.dist, 1)
o = dict(terzili=[dict(classe=c, iv=float(g.iv.mean()), dist=float(g.dist.mean()), n=int(len(g))) for c in ('bassa', 'media', 'alta') for g in [x[x.classe == c]]],
         corr=float(np.corrcoef(x.iv, x.dist)[0, 1]), pendenza=float(b), costante=float(a), minimo=float(x.dist.min()), massimo=float(x.dist.max()))
print(json.dumps(o, indent=1))
if len(sys.argv) > 2 and sys.argv[2] == '--scrivi':
    W = json.load(open(out + '/filtro_vol.json')); W['distanza_strike'] = o
    json.dump(W, open(out + '/filtro_vol.json', 'w'), separators=(',', ':'))
