"""Backtest con strike scelti dalla densità GJR-GARCH invece che dal delta di mercato.

Regola "garch": lo strike della put è il quantile p della densità reale del prezzo a scadenza, cioè lo strike con
probabilità p di finire in the money secondo il modello; per la call il quantile 1-p. Le wing di put credit spread e
iron condor sono al quantile 5% (95% per le call). Regola "delta": strike con delta di mercato più vicino a p, wing a
delta 0,05 (stessa regola di bt.py), ricalcolata qui sulle stesse sedute per un confronto appaiato.

Densità: simulazione del GJR-GARCH(1,1) con innovazioni t di Student standardizzate, 20.000 traiettorie, a partire
dalla varianza prevista per la seduta successiva (garch.parquet) e dai parametri dell'ultima stima disponibile
(garch_parametri.csv): solo informazioni note alla chiusura della seduta di ingresso. Ipotesi dichiarate:
- la densità è centrata sul forward implicito della seduta di ingresso (origine = ultima osservazione);
- deriva nulla: il rendimento logaritmico giornaliero simulato è sigma_t * z_t, senza costante, così che la regola non
  favorisca la vendita di put o di call;
- orizzonte in sedute di borsa tra l'ingresso e la scadenza, non in giorni di calendario;
- stessi numeri casuali per tutte le sedute che condividono la stessa stima (il parametro nu cambia solo alla ristima).

Famiglie: short put (SP), short call (SC), long put (LP), long call (LC), put credit spread (PCS), iron condor (IC);
p = 0,10, 0,16, 0,20, 0,30; scadenze a 7, 14, 30 e 45 giorni; posizioni tenute a scadenza.
Prezzi, margine e commissioni come in bt.py e stat_bt.py. Controllo: per SP, LP, PCS e IC la regola delta deve
riprodurre operazioni.parquet.
Uso: python bt_garch.py <cartella out> <cartella src>
"""
import sys, glob, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, sys.argv[2])
from lib import hac_mean, rendimenti_operazioni, scala_mensile, statistiche_mensili

out = sys.argv[1]
FEE = 1.0
LIV = [0.10, 0.16, 0.20, 0.30]          # probabilità ITM (regola garch) o delta (regola delta) della gamba principale
ALA = 0.05                              # probabilità ITM o delta delle wing
TOL_K = 0.01                            # distanza massima, in quota del forward, tra strike quotato e strike obiettivo
NSIM, SEED = 20000, 20261114
QL = sorted(set([ALA] + LIV + [1 - ALA] + [1 - p for p in LIV]))

u = pd.read_parquet(out + '/sottostante.parquet').close
g = pd.read_parquet(out + '/garch.parquet')
pp = pd.read_csv(out + '/garch_parametri.csv', parse_dates=['data']).set_index('data')
ud = u.index.values

# ---------------------------------------------------------------- operazioni richieste: (scadenza) -> [(data, obiettivo)]
want = {}
for D in (7, 14, 30, 45):
    s = pd.read_parquet(out + f'/serie_{D}.parquet')
    for t, E in zip(s.index, s.E):
        want.setdefault(E, []).append((t, D))

# ---------------------------------------------------------------- quantili della densità GARCH per (data, sedute a scadenza)
def sedute(t, E):
    """Sedute di borsa dopo t fino alla scadenza E compresa."""
    return int(np.searchsorted(ud, np.datetime64(E), side='right') - np.searchsorted(ud, np.datetime64(t), side='right'))

need = {}
for E, v in want.items():
    if ud[-1] < np.datetime64(E): continue
    for t, D in v:
        h = sedute(t, E)
        if h >= 1: need.setdefault(pd.Timestamp(t), set()).add(h)
Q = {}                                   # (data, h) -> quantili del rendimento logaritmico cumulato in % ai livelli QL
gg = g.dropna(subset=['s2next'])
stima = pp.index.values
blocco = np.searchsorted(stima, gg.index.values, side='right') - 1
rng = np.random.default_rng(SEED)
HMAX = max(max(v) for v in need.values())
for b in np.unique(blocco):
    om, al, ga, be, nu = pp.iloc[b][['omega', 'alpha', 'gamma', 'beta', 'nu']]
    Z = rng.standard_t(nu, size=(HMAX, NSIM)) * np.sqrt((nu - 2) / nu)        # innovazioni con varianza unitaria
    for t in gg.index[blocco == b]:
        hs = need.get(t)
        if not hs: continue
        s2 = np.full(NSIM, gg.at[t, 's2next']); cum = np.zeros(NSIM)
        for k in range(max(hs)):
            r = np.sqrt(s2) * Z[k]; cum += r
            if k + 1 in hs: Q[(t, k + 1)] = np.quantile(cum, QL)
            s2 = om + (al + ga * (r < 0)) * r * r + be * s2
print('densità simulate', len(Q), flush=True)
iq = {round(p, 2): i for i, p in enumerate(QL)}

# ---------------------------------------------------------------- backtest
cols = ['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'F', 'UNDERLYING_LAST', 'C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'P_DX', 'C_DX']
ch = pd.concat([pd.read_parquet(f, columns=cols).astype({c: 'float32' for c in cols[2:] if c != 'UNDERLYING_LAST'}) for f in sorted(glob.glob(out + '/c2/*.parquet'))], ignore_index=True)
ch = ch[ch.EXPIRE_DATE.isin(list(want))].sort_values(['EXPIRE_DATE', 'QUOTE_DATE', 'STRIKE']).reset_index(drop=True)

def vicino(valori, obiettivo, validi, tol):
    d = np.where(validi, np.abs(valori - obiettivo), np.inf)
    j = int(np.argmin(d))
    return j if d[j] < tol else -1

rec = []
for E, gE in ch.groupby('EXPIRE_DATE', sort=False):
    if ud[-1] < np.datetime64(E): continue
    dts = np.sort(gE.QUOTE_DATE.unique()); Ks = np.sort(gE.STRIKE.unique()).astype(float)
    ri = np.searchsorted(dts, gE.QUOTE_DATE.values); ci = np.searchsorted(Ks, gE.STRIKE.values)
    M = {}
    for c in ['C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'P_DX', 'C_DX']:
        a = np.full((len(dts), len(Ks)), np.nan); a[ri, ci] = gE[c].values; M[c] = a
    Frow = np.full(len(dts), np.nan); Frow[ri] = gE.F.values
    Srow = np.full(len(dts), np.nan); Srow[ri] = gE.UNDERLYING_LAST.values
    ok_p = (M['P_BID'] > 0) & (M['P_ASK'] >= M['P_BID']); ok_c = (M['C_BID'] > 0) & (M['C_ASK'] >= M['C_BID'])
    mid = {'P': np.where(ok_p, 0.5 * (M['P_BID'] + M['P_ASK']), np.nan), 'C': np.where(ok_c, 0.5 * (M['C_BID'] + M['C_ASK']), np.nan)}
    hsp = {'P': np.where(ok_p, 0.5 * (M['P_ASK'] - M['P_BID']), np.nan), 'C': np.where(ok_c, 0.5 * (M['C_ASK'] - M['C_BID']), np.nan)}
    ST = u.values[np.searchsorted(ud, np.datetime64(E), side='right') - 1]
    for (t, D) in want[E]:
        i = int(np.searchsorted(dts, np.datetime64(t)))
        if i >= len(dts) or dts[i] != np.datetime64(t): continue
        q = Q.get((pd.Timestamp(t), sedute(t, E)))
        if q is None: continue                       # confronto solo dove la densità GARCH esiste (dal 2012)
        F = Frow[i]; S = Srow[i]
        vp = np.isfinite(mid['P'][i]) & np.isfinite(M['P_DX'][i]) & (Ks < F)
        vc = np.isfinite(mid['C'][i]) & np.isfinite(M['C_DX'][i]) & (Ks > F)
        J = {'delta': {}, 'garch': {}}
        for p in LIV + [ALA]:
            J['delta'][p] = (vicino(M['P_DX'][i], -p, vp, 0.05), vicino(M['C_DX'][i], p, vc, 0.05))
            kp = F * np.exp(q[iq[round(p, 2)]] / 100); kc = F * np.exp(q[iq[round(1 - p, 2)]] / 100)
            J['garch'][p] = (vicino(Ks, kp, vp, TOL_K * F), vicino(Ks, kc, vc, TOL_K * F))
        for regola in ('delta', 'garch'):
            wp, wc = J[regola][ALA]
            for p in LIV:
                jp, jc = J[regola][p]
                cfgs = []
                if jp >= 0:
                    cfgs.append(('SP', [('P', jp, -1)], 'nudo')); cfgs.append(('LP', [('P', jp, 1)], 'debito'))
                    if wp >= 0 and Ks[wp] < Ks[jp]: cfgs.append(('PCS', [('P', jp, -1), ('P', wp, 1)], 'definito'))
                if jc >= 0:
                    cfgs.append(('SC', [('C', jc, -1)], 'nudo')); cfgs.append(('LC', [('C', jc, 1)], 'debito'))
                if jp >= 0 and jc >= 0 and wp >= 0 and wc >= 0 and Ks[wp] < Ks[jp] and Ks[wc] > Ks[jc]:
                    cfgs.append(('IC', [('P', jp, -1), ('P', wp, 1), ('C', jc, -1), ('C', wc, 1)], 'definito'))
                for fam, legs, mtype in cfgs:
                    v0 = sum(n * mid[s][i, j] for s, j, n in legs); h0 = sum(hsp[s][i, j] for s, j, n in legs)
                    if not np.isfinite(v0) or not np.isfinite(h0): continue
                    pay = sum(n * (max(Ks[j] - ST, 0.0) if s == 'P' else max(ST - Ks[j], 0.0)) for s, j, n in legs)
                    credit = -v0
                    if mtype == 'definito':
                        wP = [Ks[j] for s, j, n in legs if s == 'P']; wC = [Ks[j] for s, j, n in legs if s == 'C']
                        width = max((max(wP) - min(wP)) if len(wP) > 1 else 0.0, (max(wC) - min(wC)) if len(wC) > 1 else 0.0)
                        margin = width - credit
                        if credit <= 0 or margin <= 0: continue
                    elif mtype == 'nudo':
                        margin = 0.20 * S
                        if credit <= 0: continue
                    else:
                        margin = v0
                        if v0 <= 0: continue
                    s0, j0, _ = legs[0] if fam != 'IC' else legs[0]
                    kput = next((Ks[j] for s, j, n in legs if s == 'P' and (n < 0 or fam == 'LP')), np.nan)
                    kcall = next((Ks[j] for s, j, n in legs if s == 'C' and (n < 0 or fam == 'LC')), np.nan)
                    dput = next((abs(M['P_DX'][i, j]) for s, j, n in legs if s == 'P' and (n < 0 or fam == 'LP')), np.nan)
                    dcall = next((abs(M['C_DX'][i, j]) for s, j, n in legs if s == 'C' and (n < 0 or fam == 'LC')), np.nan)
                    rec.append((regola, fam, p, D, t, E, len(legs), margin, credit, S, F, ST, pay - v0, h0, 0,
                                int((np.datetime64(E) - np.datetime64(t)) / np.timedelta64(1, 'D')), kput, kcall, dput, dcall))
tr = pd.DataFrame(rec, columns=['regola', 'fam', 'par', 'D', 't', 'E', 'gambe', 'margine', 'credito', 'S', 'F', 'ST', 'pnl_mid', 'hs', 'gambe_chiusura',
                                'giorni', 'k_put', 'k_call', 'delta_put', 'delta_call'])
tr.to_parquet(out + '/operazioni_garch.parquet', index=False)
print('operazioni', len(tr), flush=True)

# ---------------------------------------------------------------- controllo: la regola delta riproduce operazioni.parquet
rif = pd.read_parquet(out + '/operazioni.parquet')
rif = rif[(rif.gest == 'hold') & rif.fam.isin(['SP', 'LP', 'PCS', 'IC']) & rif.ala.isin(['-', 'd05'])]
a = tr[(tr.regola == 'delta') & tr.fam.isin(['SP', 'LP', 'PCS', 'IC'])].merge(rif, on=['fam', 'par', 'D', 't', 'E'], suffixes=('', '_rif'))
att = tr[(tr.regola == 'delta') & tr.fam.isin(['SP', 'PCS', 'IC'])].shape[0] + tr[(tr.regola == 'delta') & (tr.fam == 'LP') & tr.par.isin([0.10, 0.30])].shape[0]
ctrl = dict(confrontate=int(len(a)), attese=int(att), scarto_max_pnl=float((a.pnl_mid - a.pnl_mid_rif).abs().max()), scarto_max_margine=float((a.margine - a.margine_rif).abs().max()))
print('controllo con operazioni.parquet:', ctrl, flush=True)

# ---------------------------------------------------------------- statistiche appaiate sulle sedute comuni alle due regole
tr = rendimenti_operazioni(tr, FEE)
mesi = pd.period_range(tr.mese.min(), tr.mese.max(), freq='M')
def itm(d, fam):
    """Frequenza con cui la gamba principale finisce in the money (per l'iron condor: una delle due gambe vendute)."""
    p = d.ST < d.k_put; c = d.ST > d.k_call
    return float((p | c).mean()) if fam == 'IC' else float(p.mean() if fam in ('SP', 'LP', 'PCS') else c.mean())
C = {}
for (fam, par, D), d in tr.groupby(['fam', 'par', 'D']):
    comuni = set(d[d.regola == 'delta'].t) & set(d[d.regola == 'garch'].t)
    o = dict(n=len(comuni)); R = {}
    for regola in ('delta', 'garch'):
        x = d[(d.regola == regola) & d.t.isin(comuni)]
        R[regola] = scala_mensile(x, 'rho1', D, mesi)
        m = statistiche_mensili(R[regola])
        o[regola] = dict(sharpe=m['sharpe'], media_annua=m['media_annua'], max_dd=m['max_dd'], mese_peggiore=m['mese_peggiore'], totale=float(R[regola].sum()),
                         sharpe_mid=statistiche_mensili(scala_mensile(x, 'rho0', D, mesi))['sharpe'],
                         media_rho=float(x.rho1.mean()), vincenti=float((x.rho1 > 0).mean()), peggiore=float(x.rho1.min()), asimmetria=float(stats.skew(x.rho1)),
                         curtosi_ecc=float(stats.kurtosis(x.rho1)), credito_su_margine=float((x.credito / x.margine).mean()), itm=itm(x, fam),
                         dist_put=float((1 - x.k_put / x.F).mean()), dist_call=float((x.k_call / x.F - 1).mean()),
                         delta_put=float(x.delta_put.mean()), delta_call=float(x.delta_call.mean()))
    dm, se, tt = hac_mean(R['garch'] - R['delta'], 3)
    o['diff'] = dict(media_annua=float(12 * dm), t=float(tt))
    C[f'{fam}|{par:.2f}|{D}'] = o
FAMS = ['SP', 'SC', 'PCS', 'IC', 'LP', 'LC']
Fm = {}
for fam in FAMS:
    v = [o for k, o in C.items() if k.startswith(fam + '|')]
    med = lambda f: float(np.median([f(o) for o in v]))
    Fm[fam] = dict(n=len(v), sharpe_delta=med(lambda o: o['delta']['sharpe']), sharpe_garch=med(lambda o: o['garch']['sharpe']),
                   sharpe_mid_delta=med(lambda o: o['delta']['sharpe_mid']), sharpe_mid_garch=med(lambda o: o['garch']['sharpe_mid']),
                   quota_garch_migliore=float(np.mean([o['garch']['sharpe'] > o['delta']['sharpe'] for o in v])),
                   quota_diff_signif=float(np.mean([abs(o['diff']['t']) > 1.96 for o in v])),
                   quota_dd_minore=float(np.mean([o['garch']['max_dd'] < o['delta']['max_dd'] for o in v])))
# calibrazione: frequenza ITM realizzata della gamba principale per livello (short put e short call, tutte le scadenze)
cal = {}
for fam in ('SP', 'SC'):
    for p in LIV:
        v = [o for k, o in C.items() if k.startswith(f'{fam}|{p:.2f}|')]
        cal[f'{fam}|{p:.2f}'] = dict(delta=float(np.mean([o['delta']['itm'] for o in v])), garch=float(np.mean([o['garch']['itm'] for o in v])),
                                     delta_mercato_garch=float(np.mean([o['garch']['delta_put' if fam == 'SP' else 'delta_call'] for o in v])))
tot = dict(n=len(C), quota_garch_migliore=float(np.mean([o['garch']['sharpe'] > o['delta']['sharpe'] for o in C.values()])),
           quota_diff_signif=float(np.mean([abs(o['diff']['t']) > 1.96 for o in C.values()])),
           dal=str(tr.t.min().date()), al=str(tr.t.max().date()), nsim=NSIM, tol_k=TOL_K, operazioni=int(len(tr)))
json.dump(dict(config=C, famiglie=Fm, calibrazione=cal, totale=tot, controllo=ctrl), open(out + '/garch_strike.json', 'w'), separators=(',', ':'))
pd.set_option('display.width', 250)
print(pd.DataFrame(Fm).T.round(3).to_string()); print(pd.DataFrame(cal).T.round(3).to_string()); print(tot)
