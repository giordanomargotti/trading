"""Filtro operativo sulla volatilita' realizzata: sospensione degli ingressi quando la volatilita' storica supera una soglia,
ripresa quando rientra. Confronto con l'assenza di filtro, prezzi bid-ask, scala mensile a capitale costante."""
import sys, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, sys.argv[2])
from lib import rendimenti_operazioni, scala_mensile, statistiche_mensili
out = sys.argv[1]; FEE = 1.0
tr = pd.read_parquet(out + '/operazioni.parquet')
g = pd.read_parquet(out + '/garch.parquet')
hv = g.hv21.dropna()
rank = hv.expanding(252).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)     # percentile su dati gia' osservati
tr = tr.merge(g[['hv21', 'hv10']], left_on='t', right_index=True, how='left').merge(rank.rename('rk'), left_on='t', right_index=True, how='left')
tr = rendimenti_operazioni(tr, FEE)
months = pd.period_range('2011-03', '2023-12', freq='M')     # da quando il percentile espandente e' definito
tr = tr[tr.rk.notna() & ~tr.fam.isin(['LP', 'DPS', 'CAL'])]
F = {'nessuno': lambda d: np.ones(len(d), bool), 'hv21<=20': lambda d: d.hv21 <= 0.20, 'hv21<=25': lambda d: d.hv21 <= 0.25, 'hv21<=30': lambda d: d.hv21 <= 0.30,
     'hv21<=p90': lambda d: d.rk <= 0.90, 'hv10<=25': lambda d: d.hv10 <= 0.25, 'solo_hv21>25': lambda d: d.hv21 > 0.25}
def ms(R):
    x = statistiche_mensili(R)
    return dict(sharpe=x['sharpe'], ann=x['media_annua'], dd=x['max_dd'], peggiore=x['mese_peggiore'], cvar5=x['cvar5'])
rows = []
for (fam, par, ala, D, gest), d in tr.groupby(['fam', 'par', 'ala', 'D', 'gest']):
    for name, f in F.items():
        s = d[np.asarray(f(d))]
        R = scala_mensile(s, 'rho1', D, months)
        rows.append(dict(fam=fam, par=par, ala=ala, D=D, gest=gest, filtro=name, quota=len(s) / len(d), rho=float(s.rho1.mean()) if len(s) else np.nan, **ms(R)))
r = pd.DataFrame(rows)
piv = lambda col: r.pivot_table(index=['fam', 'par', 'ala', 'D', 'gest'], columns='filtro', values=col)
sh, dd, an, pg = piv('sharpe'), piv('dd'), piv('ann'), piv('peggiore')
res = {'filtri': {}, 'famiglie': {}, 'selezione': {}, 'periodo': '2011-03 / 2023-12'}
for name in F:
    if name == 'nessuno':
        res['filtri'][name] = dict(sharpe_mediano=float(sh[name].median()), dd_mediano=float(dd[name].median()), quota_operazioni=1.0, ann_mediano=float(an[name].median()), peggiore_mediano=float(pg[name].median()))
        continue
    res['filtri'][name] = dict(sharpe_mediano=float(sh[name].median()), dd_mediano=float(dd[name].median()), ann_mediano=float(an[name].median()), peggiore_mediano=float(pg[name].median()),
                               quota_operazioni=float(r[r.filtro == name].quota.mean()), quota_sharpe_migliore=float((sh[name] > sh['nessuno']).mean()),
                               quota_dd_minore=float((dd[name] < dd['nessuno']).mean()), quota_ann_maggiore=float((an[name] > an['nessuno']).mean()))
for fam in ['SP', 'STRG', 'PCS', 'STRD', 'IC', 'IFLY', 'CCS']:
    res['famiglie'][fam] = {name: dict(sharpe=float(sh.loc[fam][name].median()), dd=float(dd.loc[fam][name].median())) for name in F}
for key in [('SP', 0.16, '-', 45, 'hold'), ('PCS', 0.20, 'd05', 30, 'hold'), ('IC', 0.16, 'd05', 30, 'hold'), ('STRD', 0.5, '-', 30, 'hold'), ('STRG', 0.16, '-', 30, 'hold')]:
    x = r.set_index(['fam', 'par', 'ala', 'D', 'gest']).loc[key].set_index('filtro')
    res['selezione']['|'.join(map(str, key))] = {n: dict(sharpe=float(x.loc[n, 'sharpe']), ann=float(x.loc[n, 'ann']), dd=float(x.loc[n, 'dd']), peggiore=float(x.loc[n, 'peggiore']), rho=float(x.loc[n, 'rho'])) for n in F}
# premio successivo per livello di volatilita' realizzata all'ingresso (orizzonte 30 giorni)
s = pd.read_parquet(out + '/serie_30.parquet').join(g.hv10).dropna(subset=['hv21', 'vrp_vol'])
s['mf'] = s.mfvol - s.rvol
b = pd.cut(s.hv21, [0, 0.10, 0.15, 0.20, 0.25, 0.30, 2.0])
res['vrp_per_hv'] = [dict(classe=str(k), n=int(len(v)), iv=float(v.iv_atm.mean()), rv=float(v.rvol.mean()), vrp_atm=float(v.vrp_vol.mean()), vrp_mf=float(v.mf.mean()), quota_pos=float((v.vrp_vol > 0).mean()),
                          p5=float(v.vrp_vol.quantile(.05)), minimo=float(v.vrp_vol.min()), t=float(hac_mean(v.vrp_vol.values, 30)[2])) for k, v in s.groupby(b, observed=True)]
hi = s[s.hv21 > 0.25]; lo = s[s.hv21 <= 0.25]
res['vrp_soglia25'] = dict(alta=dict(n=int(len(hi)), vrp=float(hi.vrp_vol.mean()), sd=float(hi.vrp_vol.std()), p5=float(hi.vrp_vol.quantile(.05))), bassa=dict(n=int(len(lo)), vrp=float(lo.vrp_vol.mean()), sd=float(lo.vrp_vol.std()), p5=float(lo.vrp_vol.quantile(.05))))
# ampiezza degli strike scelti per delta in funzione della volatilita' implicita
o = pd.read_parquet(out + '/operazioni.parquet'); o = o[(o.fam == 'SP') & (o.par == 0.16) & (o.D == 30) & (o.gest == 'hold')].merge(pd.read_parquet(out + '/serie_30.parquet')[['iv_atm']], left_on='t', right_index=True)
json.dump(res, open(out + '/filtro_vol.json', 'w'))
pd.set_option('display.width', 220)
print(pd.DataFrame(res['filtri']).T.round(3).to_string())
print(pd.DataFrame({f: {n: v['sharpe'] for n, v in d.items()} for f, d in res['famiglie'].items()}).T.round(2).to_string())
print(pd.DataFrame({f: {n: v['dd'] for n, v in d.items()} for f, d in res['famiglie'].items()}).T.round(2).to_string())
for k, v in res['selezione'].items(): print(k); print(pd.DataFrame(v).T.round(3).to_string())
print(pd.DataFrame(res['vrp_per_hv']).round(4).to_string()); print(res['vrp_soglia25'])
