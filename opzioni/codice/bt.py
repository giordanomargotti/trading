"""Backtest delle famiglie di strategie sulle catene SPX: una operazione per seduta e per scadenza obiettivo.
Prezzi: medio (mid) e semi-spread denaro-lettera per gamba, cosi' che il P&L sia lineare nella quota di spread pagata.
Regole di uscita valutate sui prezzi medi di chiusura giornaliera."""
import sys, glob, numpy as np, pandas as pd
out = sys.argv[1]
TARGETS = {7: 2, 14: 3, 30: 5, 45: 7}
DELTAS = [0.10, 0.16, 0.20, 0.30]
u = pd.read_parquet(out + '/sottostante.parquet').close
cols = ['QUOTE_DATE', 'EXPIRE_DATE', 'DTE', 'STRIKE', 'F', 'UNDERLYING_LAST', 'C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'P_DX', 'C_DX']
ch = pd.concat([pd.read_parquet(f, columns=cols).astype({c: 'float32' for c in cols[3:] if c != 'UNDERLYING_LAST'}) for f in sorted(glob.glob(out + '/c2/*.parquet'))], ignore_index=True)
ch = ch.sort_values(['EXPIRE_DATE', 'QUOTE_DATE', 'STRIKE']).reset_index(drop=True)
# mappa (scadenza) -> [(data, obiettivo)]
want = {}
for D in TARGETS:
    s = pd.read_parquet(out + f'/serie_{D}.parquet')
    for t, E in zip(s.index, s.E):
        want.setdefault(E, []).append((t, D))
print('scadenze usate', len(want), 'operazioni base', sum(len(v) for v in want.values()), flush=True)
ch = ch[ch.EXPIRE_DATE.isin(list(want))]
rec = []
def nearest(delta_row, target, valid):
    d = np.where(valid, np.abs(delta_row - target), np.inf)
    j = int(np.argmin(d))
    return j if d[j] < 0.05 else -1
for E, g in ch.groupby('EXPIRE_DATE', sort=False):
    dts = np.sort(g.QUOTE_DATE.unique()); Ks = np.sort(g.STRIKE.unique())
    ri = np.searchsorted(dts, g.QUOTE_DATE.values); ci = np.searchsorted(Ks, g.STRIKE.values)
    M = {}
    for c in ['C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'P_DX', 'C_DX']:
        a = np.full((len(dts), len(Ks)), np.nan, np.float64); a[ri, ci] = g[c].values; M[c] = a
    Frow = np.full(len(dts), np.nan); Frow[ri] = g.F.values
    Srow = np.full(len(dts), np.nan); Srow[ri] = g.UNDERLYING_LAST.values
    ok_p = (M['P_BID'] > 0) & (M['P_ASK'] >= M['P_BID']); ok_c = (M['C_BID'] > 0) & (M['C_ASK'] >= M['C_BID'])
    mid = {'P': np.where(ok_p, 0.5 * (M['P_BID'] + M['P_ASK']), np.nan), 'C': np.where(ok_c, 0.5 * (M['C_BID'] + M['C_ASK']), np.nan)}
    hsp = {'P': np.where(ok_p, 0.5 * (M['P_ASK'] - M['P_BID']), np.nan), 'C': np.where(ok_c, 0.5 * (M['C_ASK'] - M['C_BID']), np.nan)}
    ie = np.searchsorted(u.index.values, np.datetime64(E), side='right') - 1
    if u.index.values[-1] < np.datetime64(E): continue
    ST = u.values[ie]
    for (t, D) in want[E]:
        i = int(np.searchsorted(dts, np.datetime64(t)))
        if i >= len(dts) or dts[i] != np.datetime64(t): continue
        F = Frow[i]; S = Srow[i]
        vp = np.isfinite(mid['P'][i]) & np.isfinite(M['P_DX'][i]) & (Ks < F)
        vc = np.isfinite(mid['C'][i]) & np.isfinite(M['C_DX'][i]) & (Ks > F)
        P = {d: nearest(M['P_DX'][i], -d, vp) for d in DELTAS + [0.05]}
        C = {d: nearest(M['C_DX'][i], d, vc) for d in DELTAS + [0.05]}
        both = np.isfinite(mid['P'][i]) & np.isfinite(mid['C'][i])
        if not both.any(): continue
        jatm = int(np.argmin(np.where(both, np.abs(Ks - F), np.inf)))
        def wing(j, side, rule):
            if j < 0: return -1
            if rule == 'd05':
                w = (P if side == 'P' else C)[0.05]
            else:
                tgt = Ks[j] - 0.01 * S if side == 'P' else Ks[j] + 0.01 * S
                v = np.isfinite(mid[side][i]) & ((Ks < Ks[j]) if side == 'P' else (Ks > Ks[j]))
                if not v.any(): return -1
                w = int(np.argmin(np.where(v, np.abs(Ks - tgt), np.inf)))
            if w < 0: return -1
            if side == 'P' and not Ks[w] < Ks[j]: return -1
            if side == 'C' and not Ks[w] > Ks[j]: return -1
            return w
        cfgs = []   # (famiglia, parametro, ala, gambe [(lato, colonna, quantita)], tipo margine)
        for d in DELTAS:
            jp, jc = P[d], C[d]
            if jp >= 0: cfgs.append(('SP', d, '-', [('P', jp, -1)], 'nudo'))
            if jp >= 0 and jc >= 0: cfgs.append(('STRG', d, '-', [('P', jp, -1), ('C', jc, -1)], 'nudo'))
            for rule in ('d05', 'w1'):
                wp, wc = wing(jp, 'P', rule), wing(jc, 'C', rule)
                if wp >= 0: cfgs.append(('PCS', d, rule, [('P', jp, -1), ('P', wp, 1)], 'definito'))
                if wc >= 0: cfgs.append(('CCS', d, rule, [('C', jc, -1), ('C', wc, 1)], 'definito'))
                if wp >= 0 and wc >= 0: cfgs.append(('IC', d, rule, [('P', jp, -1), ('P', wp, 1), ('C', jc, -1), ('C', wc, 1)], 'definito'))
        cfgs.append(('STRD', 0.5, '-', [('P', jatm, -1), ('C', jatm, -1)], 'nudo'))
        for wd in (0.10, 0.05):
            if P[wd] >= 0 and C[wd] >= 0 and Ks[P[wd]] < Ks[jatm] < Ks[C[wd]]:
                cfgs.append(('IFLY', wd, '-', [('P', jatm, -1), ('C', jatm, -1), ('P', P[wd], 1), ('C', C[wd], 1)], 'definito'))
        for d in (0.10, 0.30):
            if P[d] >= 0: cfgs.append(('LP', d, '-', [('P', P[d], 1)], 'debito'))
        if P[0.30] >= 0 and P[0.10] >= 0 and Ks[P[0.10]] < Ks[P[0.30]]:
            cfgs.append(('DPS', 0.30, '-', [('P', P[0.30], 1), ('P', P[0.10], -1)], 'debito'))
        for fam, par, rule, legs, mtype in cfgs:
            v0 = sum(q * mid[s][i, j] for s, j, q in legs)        # valore della posizione al prezzo medio (negativo = credito)
            h0 = sum(hsp[s][i, j] for s, j, q in legs)
            if not np.isfinite(v0) or not np.isfinite(h0): continue
            pay = sum(q * (max(Ks[j] - ST, 0.0) if s == 'P' else max(ST - Ks[j], 0.0)) for s, j, q in legs)
            credit = -v0
            if mtype == 'definito':
                wP = [Ks[j] for s, j, q in legs if s == 'P']; wC = [Ks[j] for s, j, q in legs if s == 'C']
                width = max((max(wP) - min(wP)) if len(wP) > 1 else 0.0, (max(wC) - min(wC)) if len(wC) > 1 else 0.0)
                margin = width - credit
                if credit <= 0 or margin <= 0: continue
            elif mtype == 'nudo':
                margin = 0.20 * S
                if credit <= 0: continue
            else:
                margin = v0
                if v0 <= 0: continue
            kshort = [Ks[j] for s, j, q in legs if q < 0]
            base = (fam, par, rule, D, t, E, len(legs), margin, credit, S, ST)
            rec.append(base + ('hold', pay - v0, h0, 0, int((np.datetime64(E) - np.datetime64(t)) / np.timedelta64(1, 'D')), 0))
            if mtype == 'debito' or i + 1 >= len(dts): continue
            vpath = sum(q * mid[s][i + 1:, j] for s, j, q in legs)
            hpath = sum(hsp[s][i + 1:, j] for s, j, q in legs)
            pnl = vpath - v0
            okm = np.isfinite(pnl) & np.isfinite(hpath)
            for name, tp, sl in (('tp50', 0.5, None), ('tp50sl200', 0.5, 2.0)):
                hit = okm & (pnl >= tp * credit)
                if sl is not None: hit |= okm & (pnl <= -sl * credit)
                if hit.any():
                    k = int(np.argmax(hit))
                    rec.append(base + (name, pnl[k], h0 + hpath[k], len(legs), int((dts[i + 1 + k] - np.datetime64(t)) / np.timedelta64(1, 'D')), 1 if pnl[k] > 0 else 2))
                else:
                    rec.append(base + (name, pay - v0, h0, 0, int((np.datetime64(E) - np.datetime64(t)) / np.timedelta64(1, 'D')), 0))
tr = pd.DataFrame(rec, columns=['fam', 'par', 'ala', 'D', 't', 'E', 'gambe', 'margine', 'credito', 'S', 'ST', 'gest', 'pnl_mid', 'hs', 'gambe_chiusura', 'giorni', 'uscita'])
tr.to_parquet(out + '/operazioni.parquet', index=False)
print(len(tr), tr.groupby(['fam', 'gest']).size().unstack().to_string())
