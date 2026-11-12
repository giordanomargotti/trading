"""Sintesi dei backtest: statistiche per configurazione, serie mensili a scala costante, regimi, test multipli."""
import sys, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, sys.argv[2])
from lib import hac_mean, rendimenti_operazioni, scala_mensile, rendimenti_mensili_spx, statistiche_mensili
out = sys.argv[1]
FEE = 1.0                                   # dollari per gamba e per contratto (apertura e chiusura anticipata)
tr = pd.read_parquet(out + '/operazioni.parquet')
u = pd.read_parquet(out + '/sottostante.parquet').close
tag = []
for D in (7, 14, 30, 45):
    s = pd.read_parquet(out + f'/serie_{D}.parquet')[['iv_atm', 'gvol', 'mfvol']].reset_index(); s['D'] = D; tag.append(s)
tag = pd.concat(tag)
s30 = pd.read_parquet(out + '/serie_30.parquet').iv_atm; s60 = pd.read_parquet(out + '/serie_60.parquet').iv_atm
slope = (s60 - s30).dropna().rename('pend')
tr = tr.merge(tag, on=['t', 'D'], how='left').merge(slope, left_on='t', right_index=True, how='left')
tr['gap'] = tr.iv_atm - tr.gvol
tr = rendimenti_operazioni(tr, FEE)
months = pd.period_range('2010-01', '2023-12', freq='M')
ivq = {D: tag[tag.D == D].iv_atm.quantile([1 / 3, 2 / 3]).values for D in (7, 14, 30, 45)}
mstats = statistiche_mensili
def ladder(g, col, D):
    return scala_mensile(g, col, D, months)
rb = rendimenti_mensili_spx(u).values
def capm(R):
    X = np.column_stack([np.ones(len(rb)), rb])
    from lib import hac_ols
    b, se, V, r2 = hac_ols(np.asarray(R, float), X, 3)
    return dict(alfa_annuo=float(12 * b[0]), beta=float(b[1]), t_alfa=float(b[0] / se[0]), r2=float(r2))
C = []; SER = {}
for (fam, par, ala, D, gest), g in tr.groupby(['fam', 'par', 'ala', 'D', 'gest'], sort=True):
    key = f'{fam}|{par:.2f}|{ala}|{D}|{gest}'
    o = dict(key=key, fam=fam, par=float(par), ala=ala, D=int(D), gest=gest, n=int(len(g)), dal=str(g.t.min().date()))
    X = g[['a', 'b', 'f']].values
    o['m'] = X.mean(0).tolist(); cv = np.cov(X.T); o['cov'] = [cv[0, 0], cv[1, 1], cv[2, 2], cv[0, 1], cv[0, 2], cv[1, 2]]
    for x in (0, 1):
        r = g[f'rho{x}'].values; d = g[f'usd{x}'].values
        o[f't{x}'] = dict(media_usd=float(d.mean()), mediana_usd=float(np.median(d)), vincenti=float((d > 0).mean()), media_rho=float(r.mean()),
                          peggiore_usd=float(d.min()), asimmetria=float(stats.skew(r)), vincita_media=float(d[d > 0].mean()) if (d > 0).any() else 0.0,
                          perdita_media=float(d[d <= 0].mean()) if (d <= 0).any() else 0.0)
        R = ladder(g, f'rho{x}', D); o[f'p{x}'] = mstats(R); o[f'capm{x}'] = capm(R)
        if x == 1:
            o['h1'] = mstats(R[:84]); o['h2'] = mstats(R[84:])
            o['anni'] = [float(v) for v in R.reshape(14, 12).sum(1)]
        SER[f'{key}|{x}'] = [int(round(v * 1e4)) for v in R]
    o['credito_medio'] = float(g.credito.mean()); o['margine_medio'] = float(100 * g.margine.mean()); o['spread_su_credito'] = float(g.hs.sum() / g.credito.abs().sum())
    o['giorni_medi'] = float(g.giorni.mean()); o['uscite'] = [float((g.uscita == k).mean()) for k in (0, 1, 2)]
    q1, q2 = ivq[D]
    o['reg_iv'] = [float(g.rho1[g.iv_atm <= q1].mean()), float(g.rho1[(g.iv_atm > q1) & (g.iv_atm <= q2)].mean()), float(g.rho1[g.iv_atm > q2].mean())]
    gg = g.dropna(subset=['gap'])
    o['reg_gap'] = [float(gg.rho1[gg.gap <= 0].mean()), float(gg.rho1[gg.gap > 0].mean()), int((gg.gap <= 0).sum()), int((gg.gap > 0).sum())]
    gs = g.dropna(subset=['pend'])
    o['reg_pend'] = [float(gs.rho1[gs.pend < 0].mean()) if (gs.pend < 0).any() else None, float(gs.rho1[gs.pend >= 0].mean()), int((gs.pend < 0).sum()), int((gs.pend >= 0).sum())]
    # filtri sull'ingresso (prezzi denaro-lettera): serie mensile e statistiche dal 2012
    msk = (months >= pd.Period('2012-02', 'M'))
    base = ladder(gg, 'rho1', D)[msk]
    o['filtri'] = {'nessuno': mstats(base)}
    for name, cond in (('gap>0', gg.gap > 0), ('gap>2pt', gg.gap > 0.02), ('gap<=0', gg.gap <= 0), ('iv_alta', gg.iv_atm > q2), ('iv_bassa', gg.iv_atm <= q1)):
        sub = gg[cond]
        if len(sub) < 30: continue
        Rf = ladder(sub, 'rho1', D)[msk]
        o['filtri'][name] = dict(mstats(Rf), n=int(len(sub)), media_rho=float(sub.rho1.mean()))
        if name == 'gap>0': SER[f'{key}|gap'] = [0] * int((~msk).sum()) + [int(round(v * 1e4)) for v in Rf]
    C.append(o)
df = pd.DataFrame([dict(key=o['key'], fam=o['fam'], gest=o['gest'], D=o['D'], sh0=o['p0']['sharpe'], sh1=o['p1']['sharpe'], t1=o['p1']['t'], h1=o['h1']['sharpe'], h2=o['h2']['sharpe'],
                        m1=o['p1']['media_annua'], dd1=o['p1']['max_dd'], alfa1=o['capm1']['alfa_annuo'], beta1=o['capm1']['beta'], talfa1=o['capm1']['t_alfa'], talfa0=o['capm0']['t_alfa']) for o in C])
credit = df[~df.fam.isin(['LP', 'DPS', 'CAL'])]
N = len(df)
# Sharpe deflazionato (Bailey e Lopez de Prado, 2014) per la migliore configurazione, su base mensile
best = df.sort_values('sh1').iloc[-1]
srm = df.sh1.values / np.sqrt(12); V = srm.var(ddof=1); em = 0.5772156649
sr0 = np.sqrt(V) * ((1 - em) * stats.norm.ppf(1 - 1 / N) + em * stats.norm.ppf(1 - 1 / (N * np.e)))
Rb = np.array(SER[best.key + '|1']) / 1e4; srb = Rb.mean() / Rb.std(ddof=1); T = len(Rb)
g3 = stats.skew(Rb); g4 = stats.kurtosis(Rb, fisher=False)
dsr = stats.norm.cdf((srb - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * srb + (g4 - 1) / 4 * srb ** 2))
psr0 = stats.norm.cdf(srb * np.sqrt(T - 1) / np.sqrt(1 - g3 * srb + (g4 - 1) / 4 * srb ** 2))
top = credit.sort_values('h1', ascending=False).head(20)
MT = dict(n_config=int(N), quota_sharpe_pos_mid=float((df.sh0 > 0).mean()), quota_sharpe_pos_ba=float((df.sh1 > 0).mean()), quota_t2_ba=float((df.t1 > 2).mean()), quota_talfa2_ba=float((df.talfa1 > 2).mean()), quota_talfa2_mid=float((df.talfa0 > 2).mean()), quota_talfa_neg2_ba=float((df.talfa1 < -2).mean()),
          sharpe_mediano_mid=float(df.sh0.median()), sharpe_mediano_ba=float(df.sh1.median()), migliore=best.key, sharpe_migliore=float(best.sh1),
          sharpe_atteso_massimo_sotto_h0=float(sr0 * np.sqrt(12)), dsr=float(dsr), psr=float(psr0),
          corr_meta=float(stats.spearmanr(credit.h1, credit.h2)[0]), top20_h1_media_h1=float(top.h1.mean()), top20_h1_media_h2=float(top.h2.mean()),
          mediana_h2=float(credit.h2.median()), top20_h1_quota_h2_positiva=float((top.h2 > 0).mean()),
          isto_sh1=np.histogram(df.sh1, bins=np.arange(-2, 2.01, 0.2))[0].tolist(), isto_sh0=np.histogram(df.sh0, bins=np.arange(-2, 2.01, 0.2))[0].tolist())
# riferimento: indice S&P 500 (solo prezzo), rendimenti mensili
BM = mstats(rb)
# prezzo dell'assicurazione: perdita a scadenza della put venduta / premio incassato, per delta e scadenza
ins = []
for (D, par), g in tr[(tr.fam == 'SP') & (tr.gest == 'hold')].groupby(['D', 'par']):
    loss = g.credito - g.pnl_mid
    ins.append(dict(D=int(D), delta=float(par), rapporto=float(loss.sum() / g.credito.sum()), premio_medio=float(g.credito.mean()), quota_itm=float((loss > 0).mean()),
                    rapporto_ba=float(loss.sum() / (g.credito - g.hs).sum()), n=int(len(g))))
json.dump(dict(config=C, serie=SER, mesi=[str(m) for m in months], test_multipli=MT, riferimento=BM, assicurazione=ins, fee=FEE), open(out + '/bt.json', 'w'))
pd.set_option('display.width', 250)
print(json.dumps(MT, indent=0)[:1500]); print('SPX', BM)
v = df[(df.gest == 'hold')].copy(); v['cfg'] = v.key.str.replace('|hold', '')
print(v.sort_values('sh1', ascending=False)[['cfg', 'sh0', 'sh1', 't1', 'm1', 'dd1', 'h1', 'h2', 'alfa1', 'beta1', 'talfa1', 'talfa0']].round(2).head(16).to_string())
print(v[v.fam=='CAL'].round(2).to_string())
print(df.groupby(['fam', 'gest'])[['sh0', 'sh1', 'm1', 'dd1', 'talfa1', 'talfa0', 'beta1']].median().round(2).to_string())
print(df.groupby(['D'])[['sh0', 'sh1']].median().round(2).to_string())
