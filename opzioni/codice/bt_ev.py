"""Backtest con la combinazione scelta per massimo valore atteso sotto la densità GJR-GARCH.

Per ogni seduta e per ogni scadenza obiettivo (7, 14, 30, 45 giorni: le stesse scadenze di bt.py) si valutano tutte le
combinazioni di una famiglia costruibili con le opzioni candidate; si apre quella con il valore atteso più alto, se
positivo. Posizioni tenute a scadenza.

Opzioni candidate: tutte le opzioni out-of-the-money rispetto al forward (put con strike sotto il forward, call sopra)
con quotazioni valide (bid > 0, ask >= bid) e delta di mercato in valore assoluto almeno pari a DMIN, che esclude le
opzioni con quotazioni quasi nulle. Le opzioni in the money non aggiungono esposizioni: per la put-call parity sono
opzioni out-of-the-money dell'altro tipo più un forward, che sotto la densità martingala ha valore atteso nullo.

Valore atteso (EV) di una combinazione, in punti indice per contratto:
  EV = E_GARCH[payoff a scadenza] - costo di apertura ai prezzi bid-ask - commissioni (1 dollaro per gamba = 0,01 punti)
cioè il valore atteso, sotto la densità GARCH, del P&L ai prezzi bid-ask usato in bt.py e stat_bt.py (rho1 * margine).
Densità: densita_garch.simula con gli stessi parametri, le stesse traiettorie e le stesse sedute di bt_garch.py. Il prezzo
a scadenza simulato F * exp(r) è normalizzato in modo che la sua media sia il forward F: densità martingala, nessuna
deriva, nessun valore atteso da una posizione nel forward. Il valore atteso nasce solo dalla differenza tra i prezzi di
mercato e la forma della densità (volatilità, asimmetria, code).

Criterio: si sceglie la combinazione con l'EV per contratto più alto e la si apre se l'EV è positivo. L'EV non è
diviso per il margine: per put credit spread e iron condor il margine (larghezza meno credito al prezzo medio) può
avvicinarsi a zero sulle combinazioni strette con credito quasi pari alla larghezza, e il rapporto diverge su
incoerenze delle quotazioni. Il rendimento atteso sul margine (er = EV / margine) si registra per la calibrazione.

Famiglie: short put (SP), short call (SC), long put (LP), long call (LC), put credit spread (PCS), iron condor (IC, gambe
scelte liberamente tra le candidate, larghezze diverse ammesse sui due lati, margine = larghezza massima meno credito
medio). L'EV dell'iron condor è la somma degli EV dei due vertical spread, quindi il massimo si ottiene scegliendo il
migliore spread su ciascun lato; se quella coppia avesse margine non positivo la seduta si conta e si esclude.

Confronto: regola a delta di bt_garch.py (operazioni_garch.parquet), stesse scadenze e stesso periodo.
Uso: python bt_ev.py <cartella out> <cartella src>
"""
import sys, glob, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, sys.argv[2])
from lib import hac_mean, rendimenti_operazioni, scala_mensile, statistiche_mensili
from densita_garch import simula, orizzonti, sedute

out = sys.argv[1]
FEE = 1.0                         # dollari per gamba e per contratto, come in stat_bt.py
FEEP = FEE / 100                  # la stessa commissione in punti indice
DMIN = 0.02                       # delta di mercato minimo, in valore assoluto, delle opzioni candidate
NSIM, SEED = 20000, 20261114      # come bt_garch.py: stesse traiettorie
FAMS = ['SP', 'SC', 'PCS', 'IC', 'LP', 'LC']
LIV = [0.10, 0.16, 0.20, 0.30]
RIF = {'SP': (0.16, 45), 'SC': (0.16, 30), 'PCS': (0.20, 30), 'IC': (0.16, 30), 'LP': (0.10, 30), 'LC': (0.16, 30)}  # rappresentative della sezione 5.7

u = pd.read_parquet(out + '/sottostante.parquet').close
g = pd.read_parquet(out + '/garch.parquet')
pp = pd.read_csv(out + '/garch_parametri.csv', parse_dates=['data']).set_index('data')
ud = u.index.values
t0 = g.dropna(subset=['s2next']).index[0]
giorno = lambda t: int(np.datetime64(t, 'D').astype('int64'))

# ---------------------------------------------------------------- operazioni richieste, come in bt_garch.py
want = {}
for D in (7, 14, 30, 45):
    s = pd.read_parquet(out + f'/serie_{D}.parquet')
    for t, E in zip(s.index, s.E):
        want.setdefault(E, []).append((t, D))
need = orizzonti(want, ud)       # stesso insieme di orizzonti di bt_garch.py: stesse traiettorie simulate
coppie = pd.DataFrame([(t, E, D) for E, v in want.items() if ud[-1] >= np.datetime64(E) for t, D in v if pd.Timestamp(t) >= t0],
                      columns=['QUOTE_DATE', 'EXPIRE_DATE', 'D'])
for c in ('QUOTE_DATE', 'EXPIRE_DATE'): coppie[c] = pd.to_datetime(coppie[c]).astype('datetime64[ns]')
obiettivi = {(giorno(t), giorno(E)): list(v) for (t, E), v in coppie.groupby(['QUOTE_DATE', 'EXPIRE_DATE']).D}

# ---------------------------------------------------------------- opzioni candidate per (seduta, scadenza)
cols = ['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'F', 'UNDERLYING_LAST', 'C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'P_DX', 'C_DX']
chiavi = coppie[['QUOTE_DATE', 'EXPIRE_DATE']].drop_duplicates()
parti = []
for f in sorted(glob.glob(out + '/c2/*.parquet')):
    x = pd.read_parquet(f, columns=cols)
    for c in ('QUOTE_DATE', 'EXPIRE_DATE'): x[c] = x[c].astype('datetime64[ns]')
    parti.append(x.merge(chiavi, on=['QUOTE_DATE', 'EXPIRE_DATE']))
ch = pd.concat(parti, ignore_index=True); del parti
cand = {}
for (t, E), x in ch.groupby(['QUOTE_DATE', 'EXPIRE_DATE'], sort=False):
    F = float(x.F.iloc[0]); S = float(x.UNDERLYING_LAST.iloc[0])
    p = x[(x.STRIKE < F) & (x.P_BID > 0) & (x.P_ASK >= x.P_BID) & (-x.P_DX >= DMIN)].sort_values('STRIKE')
    c = x[(x.STRIKE > F) & (x.C_BID > 0) & (x.C_ASK >= x.C_BID) & (x.C_DX >= DMIN)].sort_values('STRIKE')
    cand[(giorno(t), giorno(E))] = dict(t=t, E=E, F=F, S=S,
        Kp=p.STRIKE.to_numpy(float), bp=p.P_BID.to_numpy(float), ap=p.P_ASK.to_numpy(float), dp=(-p.P_DX).to_numpy(float),
        Kc=c.STRIKE.to_numpy(float), bc=c.C_BID.to_numpy(float), ac=c.C_ASK.to_numpy(float), dc=c.C_DX.to_numpy(float))
del ch
np_ = np.array([len(c['Kp']) for c in cand.values()]); nc_ = np.array([len(c['Kc']) for c in cand.values()])
print(f'coppie (seduta, scadenza) {len(cand)} | put candidate: mediana {np.median(np_):.0f}, massimo {np_.max()} | call candidate: mediana {np.median(nc_):.0f}, massimo {nc_.max()}', flush=True)

# ---------------------------------------------------------------- valore atteso del payoff di ogni opzione candidata
th = {}
for k, c in cand.items():
    h = sedute(ud, c['t'], c['E'])
    if h >= 1: th.setdefault((k[0], h), []).append(k)

def attese(t, h, cum):
    """E[(K - S_T)^+] e E[(S_T - K)^+] per le candidate di tutte le scadenze a h sedute dalla data t."""
    ks = th.get((giorno(t), h))
    if not ks: return None
    x = np.exp(cum / 100); x /= x.mean()                 # prezzo a scadenza / forward, con media 1
    xs = np.sort(x); cs = np.concatenate([[0.0], np.cumsum(xs)]); N = len(xs)
    for k in ks:
        c = cand[k]; F = c['F']
        kp = c['Kp'] / F; i = np.searchsorted(xs, kp)
        c['fp'] = F * (kp * i - cs[i]) / N
        kc = c['Kc'] / F; j = np.searchsorted(xs, kc)
        c['fc'] = F * ((cs[N] - cs[j]) - kc * (N - j)) / N
    return None
simula(g, pp, need, attese, NSIM, SEED)
print('valori attesi calcolati per', sum('fp' in c for c in cand.values()), 'coppie', flush=True)

# ---------------------------------------------------------------- scelta della combinazione
def spread(K, b, a, f, lato):
    """Tutti i vertical spread a credito di un lato: gamba venduta piu' vicina al forward, gamba comprata piu' lontana.
    Restituisce indici (venduta, comprata), EV ai prezzi bid-ask, credito al prezzo medio, larghezza; solo spread con
    credito medio e margine positivi."""
    n = len(K)
    if n < 2: return None
    I, J = np.tril_indices(n, -1) if lato == 'P' else np.triu_indices(n, 1)   # put: venduta I > J; call: venduta I < J
    m = 0.5 * (b + a)
    ev = (b[I] - f[I]) - (a[J] - f[J]) - 2 * FEEP
    cr = m[I] - m[J]; w = np.abs(K[I] - K[J])
    ok = (cr > 0) & (w - cr > 0)
    if not ok.any(): return None
    return I[ok], J[ok], ev[ok], cr[ok], w[ok]

ESCLUSE = dict(ic_margine_non_positivo=0)
RAPPORTO_PCS = []                # diagnostica: max EV/margine tra i put credit spread di ogni (seduta, scadenza), criterio non usato
def scegli(c):
    """{famiglia: (EV, gambe [(lato, strike, bid, ask, delta, quantita)], margine)} con la combinazione a EV massimo."""
    S = c['S']
    gp = lambda i, q: ('P', c['Kp'][i], c['bp'][i], c['ap'][i], c['dp'][i], q)
    gc = lambda i, q: ('C', c['Kc'][i], c['bc'][i], c['ac'][i], c['dc'][i], q)
    R = {}
    for lato in ('P', 'C'):
        if len(c['K' + lato.lower()]) == 0: continue
        b, a, f = c['b' + lato.lower()], c['a' + lato.lower()], c['f' + lato.lower()]
        gg = gp if lato == 'P' else gc
        ev = b - f - FEEP; i = int(np.argmax(ev))                                  # vendita scoperta
        R['SP' if lato == 'P' else 'SC'] = (ev[i], [gg(i, -1)], 0.20 * S)
        ev = f - a - FEEP; i = int(np.argmax(ev))                                  # acquisto: margine = premio medio
        R['LP' if lato == 'P' else 'LC'] = (ev[i], [gg(i, 1)], 0.5 * (b[i] + a[i]))
    P = spread(c['Kp'], c['bp'], c['ap'], c['fp'], 'P') if len(c['Kp']) else None
    if P is not None:
        I, J, ev, cr, w = P; i = int(np.argmax(ev))
        RAPPORTO_PCS.append(float(np.max(ev / (w - cr))))
        R['PCS'] = (ev[i], [gp(I[i], -1), gp(J[i], 1)], w[i] - cr[i])
    C = spread(c['Kc'], c['bc'], c['ac'], c['fc'], 'C') if len(c['Kc']) else None
    if P is not None and C is not None:
        i, j = int(np.argmax(P[2])), int(np.argmax(C[2]))
        mg = max(P[4][i], C[4][j]) - P[3][i] - C[3][j]
        if mg > 0:
            R['IC'] = (P[2][i] + C[2][j], [gp(P[0][i], -1), gp(P[1][i], 1), gc(C[0][j], -1), gc(C[1][j], 1)], mg)
        else:
            ESCLUSE['ic_margine_non_positivo'] += 1
    return R

rec, opp = [], []
for n_, (k, c) in enumerate(cand.items()):
    if 'fp' not in c: continue
    E = c['E']; ST = u.values[np.searchsorted(ud, np.datetime64(E), side='right') - 1]
    giorni = int((np.datetime64(E) - np.datetime64(c['t'])) / np.timedelta64(1, 'D'))
    R = scegli(c)
    for D in obiettivi[k]:
        for fam in FAMS:
            if fam not in R:
                opp.append((fam, D, c['t'], False, np.nan)); continue
            ev, legs, mg = R[fam]
            opp.append((fam, D, c['t'], True, float(ev)))
            if not ev > 0: continue
            v0 = sum(q * 0.5 * (b + a) for s, K, b, a, d, q in legs); h0 = sum(0.5 * (a - b) for s, K, b, a, d, q in legs)
            pay = sum(q * (max(K - ST, 0.0) if s == 'P' else max(ST - K, 0.0)) for s, K, b, a, d, q in legs)
            main = [l for l in legs if (l[5] < 0) or fam in ('LP', 'LC')]
            kp = next((l[1] for l in main if l[0] == 'P'), np.nan); kc = next((l[1] for l in main if l[0] == 'C'), np.nan)
            dp = next((l[4] for l in main if l[0] == 'P'), np.nan); dc = next((l[4] for l in main if l[0] == 'C'), np.nan)
            wing = [l[4] for l in legs if l[5] > 0 and fam in ('PCS', 'IC')]
            ws = [abs(legs[0][1] - legs[1][1])] if fam == 'PCS' else ([abs(legs[0][1] - legs[1][1]), abs(legs[2][1] - legs[3][1])] if fam == 'IC' else [np.nan])
            # gamba principale sull'opzione candidata più vicina al forward; wing sull'opzione candidata più lontana (delta minimo)
            vicina = (np.isfinite(dp) and dp == c['dp'].max()) or (np.isfinite(dc) and dc == c['dc'].max())
            lontana = any(d == (c['dp'].min() if l[0] == 'P' else c['dc'].min()) for l, d in zip([l for l in legs if l[5] > 0 and fam in ('PCS', 'IC')], wing))
            rec.append((fam, D, c['t'], E, len(legs), mg, -v0, c['S'], c['F'], ST, pay - v0, h0, 0, giorni, ev, ev / mg,
                        kp, kc, dp, dc, max(ws), float(np.mean(wing)) if wing else np.nan, bool(vicina), bool(lontana)))
    if n_ % 2000 == 0: print('  coppie elaborate', n_, flush=True)
tr = pd.DataFrame(rec, columns=['fam', 'D', 't', 'E', 'gambe', 'margine', 'credito', 'S', 'F', 'ST', 'pnl_mid', 'hs', 'gambe_chiusura', 'giorni',
                                'ev', 'er', 'k_put', 'k_call', 'delta_put', 'delta_call', 'larghezza', 'delta_wing', 'vicina_forward', 'wing_lontana'])
tr.to_parquet(out + '/operazioni_ev.parquet', index=False)
op = pd.DataFrame(opp, columns=['fam', 'D', 't', 'fattibile', 'ev_max'])
print('operazioni', len(tr), '| escluse', ESCLUSE, flush=True)

# ---------------------------------------------------------------- controllo: EV ex ante = media sotto la densità del P&L realizzato
# (verifica di coerenza delle unità: rho1 * margine deve avere come valore atteso l'EV; qui si controlla solo l'identità dei costi)
tr = rendimenti_operazioni(tr, FEE)
costo = (tr.rho1 * tr.margine - (tr.pnl_mid - tr.hs - FEEP * tr.gambe)).abs().max()
print('controllo costi: scarto massimo tra rho1 * margine e P&L bid-ask meno commissioni', costo, flush=True)

# ---------------------------------------------------------------- statistiche
dl = pd.read_parquet(out + '/operazioni_garch.parquet')
dl = rendimenti_operazioni(dl[dl.regola == 'delta'], FEE)
mesi = pd.period_range(min(tr.mese.min(), dl.mese.min()), max(tr.mese.max(), dl.mese.max()), freq='M')
def sintesi(x, D):
    """Serie mensile del ladder ai prezzi bid-ask e statistiche di una serie di operazioni con scadenza obiettivo D."""
    R = scala_mensile(x, 'rho1', D, mesi); m = statistiche_mensili(R)
    return R, dict(sharpe=m['sharpe'], media_annua=m['media_annua'], max_dd=m['max_dd'], mese_peggiore=m['mese_peggiore'],
                   sharpe_mid=statistiche_mensili(scala_mensile(x, 'rho0', D, mesi))['sharpe'], operazioni=int(len(x)),
                   rho=float(x.rho1.mean()) if len(x) else np.nan, vincenti=float((x.rho1 > 0).mean()) if len(x) else np.nan,
                   peggiore=float(x.rho1.min()) if len(x) else np.nan, asimmetria=float(stats.skew(x.rho1)) if len(x) > 2 else np.nan)
DEL = {(fam, round(par, 2), D): sintesi(x, D) for (fam, par, D), x in dl.groupby(['fam', 'par', 'D'])}
SC = (7, 14, 30, 45)
C = {}
for fam in FAMS:
    put = fam in ('SP', 'PCS', 'IC', 'LP'); call = fam in ('SC', 'IC', 'LC')
    for D in SC:
        o = op[(op.fam == fam) & (op.D == D)]; nf = int(o.fattibile.sum())
        x = tr[(tr.fam == fam) & (tr.D == D)]
        R, s = sintesi(x, D)
        lv = RIF[fam][0]; Rd, sd = DEL[(fam, lv, D)]
        # regola delta di riferimento sulle sole sedute in cui la regola EV apre un'operazione: separa la scelta della seduta
        # (EV positivo) dalla scelta della combinazione
        _, sdt = sintesi(dl[(dl.fam == fam) & np.isclose(dl.par, lv) & (dl.D == D) & dl.t.isin(set(x.t))], D)
        dm, se, tt = hac_mean(R - Rd, 3)
        s.update(sedute=nf, quota=len(x) / nf if nf else np.nan, er=float(x.er.mean()),
                 delta_put=float(x.delta_put.mean()) if put else np.nan, delta_call=float(x.delta_call.mean()) if call else np.nan,
                 dist_put=float((1 - x.k_put / x.F).mean()) if put else np.nan, dist_call=float((x.k_call / x.F - 1).mean()) if call else np.nan,
                 larghezza_pct=float((x.larghezza / x.F).mean() * 100) if fam in ('PCS', 'IC') else np.nan,
                 delta_wing=float(x.delta_wing.mean()) if fam in ('PCS', 'IC') else np.nan,
                 vicina_forward=float(x.vicina_forward.mean()), wing_lontana=float(x.wing_lontana.mean()) if fam in ('PCS', 'IC') else np.nan,
                 delta_rif=dict(livello=lv, **sd), delta_rif_stesse_sedute=sdt, diff_t=float(tt), diff_media_annua=float(12 * dm),
                 delta_sharpe_livelli=[DEL[(fam, l, D)][1]['sharpe'] for l in LIV])
        C[f'{fam}|{D}'] = s
Fm = {}
for fam in FAMS:
    x = tr[tr.fam == fam]; v = [C[f'{fam}|{D}'] for D in SC]
    sh_d = {D: [DEL[(fam, lv, D)][1]['sharpe'] for lv in LIV if (fam, lv, D) in DEL] for D in SC}
    tutti = [a for D in SC for a in sh_d[D]]
    Fm[fam] = dict(sharpe_ev_med=float(np.median([o['sharpe'] for o in v])), sharpe_ev_mid_med=float(np.median([o['sharpe_mid'] for o in v])),
                   sharpe_delta_med=float(np.median(tutti)), sharpe_delta_max=float(np.max(tutti)),
                   sharpe_delta_mid_med=float(np.median([DEL[(fam, lv, D)][1]['sharpe_mid'] for lv in LIV for D in SC if (fam, lv, D) in DEL])),
                   sopra_mediana_delta=int(sum(o['sharpe'] > np.median(sh_d[D]) for o, D in zip(v, SC))),
                   sopra_max_delta=int(sum(o['sharpe'] > np.max(sh_d[D]) for o, D in zip(v, SC))),
                   quota=float(np.mean([o['quota'] for o in v])), er=float(x.er.mean()), rho=float(x.rho1.mean()),
                   corr=float(stats.spearmanr(x.er, x.rho1).statistic), vicina_forward=float(x.vicina_forward.mean()),
                   delta_put=float(x.delta_put.mean()) if fam in ('SP', 'PCS', 'IC', 'LP') else np.nan,
                   delta_call=float(x.delta_call.mean()) if fam in ('SC', 'IC', 'LC') else np.nan,
                   larghezza_pct=float((x.larghezza / x.F).mean() * 100) if fam in ('PCS', 'IC') else np.nan,
                   delta_wing=float(x.delta_wing.mean()) if fam in ('PCS', 'IC') else np.nan,
                   wing_lontana=float(x.wing_lontana.mean()) if fam in ('PCS', 'IC') else np.nan)
# calibrazione: quintili del rendimento atteso sul margine, rendimento realizzato medio per quintile (tutte le scadenze)
cal = {}
for fam in FAMS:
    x = tr[tr.fam == fam]; q = pd.qcut(x.er.rank(method='first'), 5, labels=False)
    cal[fam] = dict(er=[float(v) for v in x.groupby(q).er.mean()], rho=[float(v) for v in x.groupby(q).rho1.mean()])
rp = np.array(RAPPORTO_PCS)
diag = dict(n=int(len(rp)), mediana=float(np.median(rp)), quota_sopra_1=float((rp > 1).mean()), quota_sopra_10=float((rp > 10).mean()), massimo=float(rp.max()))
tot = dict(dal=str(tr.t.min().date()), al=str(tr.t.max().date()), operazioni=int(len(tr)), coppie=int(len(cand)), dmin=DMIN, nsim=NSIM, rapporto_pcs=diag,
           escluse=ESCLUSE, controllo_costi=float(costo), n_config=len(C),
           sopra_mediana_delta=int(sum(o['sopra_mediana_delta'] for o in Fm.values())), sopra_max_delta=int(sum(o['sopra_max_delta'] for o in Fm.values())),
           diff_signif=int(sum(abs(o['diff_t']) > 1.96 for o in C.values())), diff_signif_pos=int(sum(o['diff_t'] > 1.96 for o in C.values())))
json.dump(dict(config=C, famiglie=Fm, calibrazione=cal, totale=tot), open(out + '/garch_ev.json', 'w'), separators=(',', ':'),
          default=lambda v: None if (isinstance(v, float) and not np.isfinite(v)) else v)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(pd.DataFrame(Fm).T.round(3).to_string())
rows = [dict(cfg=k, quota=o['quota'], er=o['er'], rho=o['rho'], vinc=o['vincenti'], sh=o['sharpe'], sh_mid=o['sharpe_mid'], ann=o['media_annua'], dd=o['max_dd'],
             peg=o['mese_peggiore'], dput=o['delta_put'], dcall=o['delta_call'], dwing=o['delta_wing'], larg=o['larghezza_pct'], vic=o['vicina_forward'],
             wlon=o['wing_lontana'], sh_drif=o['delta_rif']['sharpe'], sh_drif_ss=o['delta_rif_stesse_sedute']['sharpe'], t=o['diff_t']) for k, o in C.items()]
print(pd.DataFrame(rows).round(3).to_string())
print({k: (np.round(v['er'], 3).tolist(), np.round(v['rho'], 3).tolist()) for k, v in cal.items()})
print(tot)
