"""Aggiunge alla catena: forward implicito per put-call parity, IV propria (Black-76 su prezzo medio OTM), delta."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2])
from lib import iv_b76, delta_b76, b76
out = sys.argv[1]
import os
year = int(sys.argv[3])
os.makedirs(out + '/c2', exist_ok=True)
ch = pd.read_parquet(out + '/catena.parquet', filters=[('QUOTE_DATE', '>=', pd.Timestamp(year, 1, 1)), ('QUOTE_DATE', '<', pd.Timestamp(year + 1, 1, 1))])
ch = ch.drop(columns=['C_GAMMA','C_THETA','P_GAMMA','P_THETA','C_VEGA','P_VEGA','C_DELTA','P_DELTA','C_VOLUME','P_VOLUME'])
ch = ch[ch.DTE >= 1].copy()
for c in ['C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'STRIKE']:
    ch[c] = ch[c].astype('float64')
cm = 0.5 * (ch.C_BID + ch.C_ASK); pm = 0.5 * (ch.P_BID + ch.P_ASK)
valid = (ch.C_BID > 0) & (ch.P_BID > 0) & (ch.C_ASK >= ch.C_BID) & (ch.P_ASK >= ch.P_BID)
ch['absd'] = np.where(valid, (cm - pm).abs(), np.inf)
g = ch.groupby(['QUOTE_DATE', 'EXPIRE_DATE'], sort=False)
idx = g.absd.idxmin()
fw = ch.loc[idx, ['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'absd']].copy()
fw['F'] = fw.STRIKE.values + (cm.loc[idx].values - pm.loc[idx].values)
fw = fw[np.isfinite(fw.absd)]
ch = ch.merge(fw[['QUOTE_DATE', 'EXPIRE_DATE', 'F']], on=['QUOTE_DATE', 'EXPIRE_DATE'], how='inner').drop(columns='absd')
T = ch.DTE.values / 365.0
F = ch.F.values; K = ch.STRIKE.values
# IV per lato: put e call separatamente, dal prezzo medio (solo quotazioni con bid > 0)
for side, call in (('P', False), ('C', True)):
    bid = ch[side + '_BID'].values; ask = ch[side + '_ASK'].values
    mid = np.where((bid > 0) & (ask >= bid), 0.5 * (bid + ask), np.nan)
    callv = np.full(len(ch), call)
    iv = iv_b76(mid, F, K, T, callv)
    ch[side + '_IVX'] = iv.astype('float32')
    ch[side + '_DX'] = delta_b76(F, K, T, iv, callv).astype('float32')
ch.to_parquet(out + f'/c2/{year}.parquet', index=False)
# controlli di integrita'
s = ch.sample(min(200000, len(ch)), random_state=1)
Ts = s.DTE.values / 365.0
rt = b76(s.F.values, s.STRIKE.values, Ts, s.P_IVX.values.astype(float), np.zeros(len(s), bool))
mid = 0.5 * (s.P_BID + s.P_ASK).values
m = np.isfinite(rt)
print('righe', len(ch), 'gruppi', len(fw))
print('round-trip IV put: errore assoluto massimo', np.nanmax(np.abs(rt[m] - mid[m])))
atm = ch[(ch.STRIKE - ch.F).abs() < 3]
print('ATM: IV put - IV call (mediana, 5%, 95%)', (atm.P_IVX - atm.C_IVX).quantile([.5, .05, .95]).round(4).tolist())
print('ATM: delta call - delta put (deve essere 1):', (atm.C_DX - atm.P_DX).quantile([.05, .5, .95]).round(4).tolist())
print('confronto con IV del fornitore (put OTM, mediana differenza):', (ch.P_IVX - ch.P_IV)[ch.STRIKE < ch.F].median())
print(year, 'costo di carry implicito annuo a 30 giorni (mediana):')
x = ch[(ch.DTE.between(25, 35))].drop_duplicates(['QUOTE_DATE', 'EXPIRE_DATE'])
print(((x.F / x.UNDERLYING_LAST - 1) * 365 / x.DTE).groupby(x.QUOTE_DATE.dt.year).median().round(4).to_dict())
