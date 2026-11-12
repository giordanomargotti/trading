"""Statistiche sul premio per il rischio di volatilita' (VRP) sulle catene SPX 2010-2023."""
import sys, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, sys.argv[2])
from lib import hac_mean, hac_ols
out = sys.argv[1]
p = pd.read_parquet(out + '/pannello.parquet')
g = pd.read_parquet(out + '/garch.parquet')
u = pd.read_parquet(out + '/sottostante.parquet').close
dates = u.index.values
p = p[(p.mfvar > 0)].copy()
p['mfvol'] = np.sqrt(p.mfvar); p['rvol'] = np.sqrt(p.rvar)
# previsione GARCH della varianza totale fino a scadenza (n sedute), annualizzata sul tempo di calendario
gi = g.reindex(p.t.values)
pe = gi.pers.values; n = p.n.values.astype(float)
geo = np.where(np.abs(1 - pe) > 1e-9, (1 - pe ** n) / (1 - pe), n)
tot = n * gi.s2lr.values + (gi.s2next.values - gi.s2lr.values) * geo       # (%)^2
p['gvol'] = np.sqrt(tot / 1e4 / (p.dte.values / 365.0))
p['hv21'] = gi.hv21.values
TARGETS = {7: 2, 14: 3, 21: 4, 30: 5, 45: 7, 60: 10}
R = {'obiettivi': {}, 'meta': {}}
series = {}
def pick(D, tol):
    q = p[(p.dte - D).abs() <= tol].copy()
    q['d'] = (q.dte - D).abs()
    q = q.sort_values(['t', 'd', 'dte']).drop_duplicates('t')
    return q.set_index('t').sort_index()
def desc(x, lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    m, se, t = hac_mean(x, lag)
    return dict(n=int(len(x)), media=m, mediana=float(np.median(x)), dev_std=float(x.std(ddof=1)), asimmetria=float(stats.skew(x)),
                curtosi=float(stats.kurtosis(x, fisher=False)), quota_positiva=float((x > 0).mean()), t_hac=t, es_hac=se,
                p5=float(np.quantile(x, .05)), p1=float(np.quantile(x, .01)), minimo=float(x.min()), massimo=float(x.max()))
for D, tol in TARGETS.items():
    q = pick(D, tol).dropna(subset=['rvar'])
    lag = int(round(D * 252 / 365 * 1.5))
    q['vrp_var'] = q.mfvar - q.rvar; q['vrp_vol'] = q.iv_atm - q.rvol; q['vrp_mfvol'] = q.mfvol - q.rvol
    q['logratio'] = np.log(q.mfvar / q.rvar)
    series[D] = q
    o = dict(giorni=int(len(q)), dal=str(q.index[0].date()), al=str(q.index[-1].date()), dte_medio=float(q.dte.mean()),
             iv_atm_media=float(q.iv_atm.mean()), mfvol_media=float(q.mfvol.mean()), rvol_media=float(q.rvol.mean()),
             vrp_vol=desc(q.vrp_vol, lag), vrp_mfvol=desc(q.vrp_mfvol, lag), vrp_var=desc(q.vrp_var, lag), logratio=desc(q.logratio, lag))
    # campione senza sovrapposizione: un'osservazione ogni D giorni di calendario
    idx = []; last = None
    for t in q.index:
        if last is None or (t - last).days >= q.loc[t, 'dte']:
            idx.append(t); last = t
    qn = q.loc[idx]
    o['non_sovrapposto'] = dict(n=int(len(qn)), vrp_vol=desc(qn.vrp_vol, 1), vrp_var=desc(qn.vrp_var, 1))
    # regressione di Mincer-Zarnowitz: rvol = a + b * iv
    for name, col in (('mz_atm', 'iv_atm'), ('mz_mf', 'mfvol')):
        X = np.column_stack([np.ones(len(q)), q[col].values])
        b, se, V, r2 = hac_ols(q.rvol.values, X, lag)
        d = b - np.array([0.0, 1.0]); wald = float(d @ np.linalg.inv(V) @ d)
        o[name] = dict(a=b[0], b=b[1], es_a=se[0], es_b=se[1], r2=r2, wald=wald, p_wald=float(stats.chi2.sf(wald, 2)),
                       t_b_uguale_1=float((b[1] - 1) / se[1]))
    X = np.column_stack([np.ones(len(q)), np.log(q.iv_atm.values)])
    b, se, V, r2 = hac_ols(np.log(q.rvol.values), X, lag)
    o['mz_log'] = dict(a=b[0], b=b[1], es_a=se[0], es_b=se[1], r2=r2)
    # confronto delle previsioni (campione con GARCH disponibile)
    qq = q.dropna(subset=['gvol', 'hv21'])
    fc = {}
    for name, col in (('IV ATM', 'iv_atm'), ('Model-free', 'mfvol'), ('GJR-GARCH-t', 'gvol'), ('HV21', 'hv21')):
        f = qq[col].values; y = qq.rvol.values
        ql = (y ** 2 / f ** 2 - np.log(y ** 2 / f ** 2) - 1).mean()
        fc[name] = dict(errore_medio=float((f - y).mean()), rmse=float(np.sqrt(((f - y) ** 2).mean())), mae=float(np.abs(f - y).mean()),
                        qlike=float(ql), corr=float(np.corrcoef(f, y)[0, 1]))
    o['previsioni'] = dict(n=int(len(qq)), dal=str(qq.index[0].date()), modelli=fc)
    X = np.column_stack([np.ones(len(qq)), qq.iv_atm.values, qq.gvol.values, qq.hv21.values])
    b, se, V, r2 = hac_ols(qq.rvol.values, X, lag)
    o['encompassing'] = dict(coef=b.tolist(), es=se.tolist(), t=(b / se).tolist(), r2=r2, nomi=['costante', 'IV ATM', 'GJR-GARCH-t', 'HV21'])
    # segnale: differenza IV - GARCH all'ingresso e VRP successivo
    qq = qq.copy(); qq['gap'] = qq.iv_atm - qq.gvol
    qq['q'] = pd.qcut(qq.gap, 5, labels=False)
    o['segnale_gap'] = [dict(quintile=int(k) + 1, gap_medio=float(v.gap.mean()), vrp_vol=float(v.vrp_vol.mean()), vrp_var=float(v.vrp_var.mean()),
                             quota_positiva=float((v.vrp_vol > 0).mean()), p5=float(v.vrp_vol.quantile(.05)), n=int(len(v))) for k, v in qq.groupby('q')]
    X = np.column_stack([np.ones(len(qq)), qq.gap.values])
    b, se, V, r2 = hac_ols(qq.vrp_vol.values, X, lag)
    o['segnale_gap_reg'] = dict(a=b[0], b=b[1], t_b=float(b[1] / se[1]), r2=r2)
    qq['qi'] = pd.qcut(qq.iv_atm, 5, labels=False)
    o['per_livello_iv'] = [dict(quintile=int(k) + 1, iv_media=float(v.iv_atm.mean()), vrp_vol=float(v.vrp_vol.mean()), vrp_var=float(v.vrp_var.mean()),
                                quota_positiva=float((v.vrp_vol > 0).mean()), p5=float(v.vrp_vol.quantile(.05)), n=int(len(v))) for k, v in qq.groupby('qi')]
    o['per_anno'] = [dict(anno=int(y), iv=float(v.iv_atm.mean()), rv=float(v.rvol.mean()), vrp_vol=float(v.vrp_vol.mean()), vrp_var=float(v.vrp_var.mean()),
                          quota_positiva=float((v.vrp_vol > 0).mean()), minimo=float(v.vrp_vol.min()), n=int(len(v))) for y, v in q.groupby(q.index.year)]
    per = {'2010-2014': (2010, 2014), '2015-2019': (2015, 2019), '2020-2023': (2020, 2023)}
    o['per_periodo'] = {k: desc(q.vrp_vol[(q.index.year >= a) & (q.index.year <= b)], lag) for k, (a, b) in per.items()}
    o['skew'] = dict(rr25=float((q.iv_p25 - q.iv_c25).mean()), p10_meno_atm=float((q.iv_p10 - q.iv_atm).mean()))
    R['obiettivi'][str(D)] = o
    print(D, o['giorni'], 'IV', round(o['iv_atm_media'], 4), 'RV', round(o['rvol_media'], 4), 'VRP vol', round(o['vrp_vol']['media'], 4), 't', round(o['vrp_vol']['t_hac'], 2),
          'pos', round(o['vrp_vol']['quota_positiva'], 3), 'MZ b', round(o['mz_atm']['b'], 3), 'skew', round(o['vrp_vol']['asimmetria'], 2))
# serie per i grafici (obiettivo 30 giorni): campionamento settimanale
q = series[30]
w = q[['iv_atm', 'mfvol', 'rvol', 'vrp_vol', 'gvol', 'hv21']].resample('W-FRI').last().dropna(subset=['iv_atm'])
R['serie30'] = dict(date=[str(d.date()) for d in w.index], **{c: [None if not np.isfinite(v) else round(float(v), 4) for v in w[c]] for c in w.columns})
h, e = np.histogram(q.vrp_vol.values * 100, bins=np.arange(-60, 22, 2))
R['isto30'] = dict(bordi=e.tolist(), conteggi=h.tolist())
R['dispersione30'] = dict(iv=[round(float(v), 4) for v in q.iv_atm.values[::5]], rv=[round(float(v), 4) for v in q.rvol.values[::5]])
ts = {str(D): dict(iv=float(series[D].iv_atm.mean()), mf=float(series[D].mfvol.mean()), rv=float(series[D].rvol.mean())) for D in TARGETS}
R['struttura'] = ts
R['meta'] = dict(sedute=int(len(u)), dal=str(u.index[0].date()), al=str(u.index[-1].date()), coppie=int(len(p)),
                 buchi=int((pd.Series(u.index).diff().dt.days > 5).sum()))
for D in TARGETS: series[D].to_parquet(out + f'/serie_{D}.parquet')
json.dump(R, open(out + '/vrp.json', 'w'))
o = R['obiettivi']['30']
print(json.dumps({k: o[k] for k in ['mz_atm', 'mz_mf', 'previsioni', 'encompassing', 'segnale_gap', 'segnale_gap_reg', 'per_livello_iv', 'per_periodo', 'non_sovrapposto', 'skew']}, indent=0)[:6000])
print(pd.DataFrame(o['per_anno']).round(4).to_string())
