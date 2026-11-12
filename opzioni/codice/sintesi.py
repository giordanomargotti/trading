"""Dati compatti per il sito e figure (italiano e inglese) per il documento."""
import sys, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out = sys.argv[1]
from lib import rendimenti_mensili_spx
V = json.load(open(out + '/vrp.json')); B = json.load(open(out + '/bt.json'))
u = pd.read_parquet(out + '/sottostante.parquet').close
rbm = rendimenti_mensili_spx(u)
def r(x, d=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return None
    return round(float(x), d)
def rr(o, d=4):
    if isinstance(o, dict): return {k: rr(v, d) for k, v in o.items()}
    if isinstance(o, list): return [rr(v, d) for v in o]
    if isinstance(o, float): return r(o, d)
    return o
cfg = []
for o in B['config']:
    c = {k: o[k] for k in ('key', 'fam', 'par', 'ala', 'D', 'gest', 'n', 'dal')}
    for k in ('t0', 't1', 'p0', 'p1', 'h1', 'h2', 'capm0', 'capm1', 'reg_iv', 'reg_gap', 'reg_pend', 'credito_medio', 'margine_medio', 'spread_su_credito', 'giorni_medi', 'uscite'):
        c[k] = rr(o[k])
    c['filtri'] = {k: {kk: r(vv) for kk, vv in v.items() if kk in ('media_annua', 'sharpe', 'max_dd', 't', 'n', 'media_rho')} for k, v in o['filtri'].items()}
    c['s0'] = B['serie'][o['key'] + '|0']; c['s1'] = B['serie'][o['key'] + '|1']; c['sg'] = B['serie'].get(o['key'] + '|gap')
    cfg.append(c)
Dz = json.load(open(out + '/distribuzioni.json'))
S = dict(vrp=rr(V), config=cfg, mesi=B['mesi'], spx=[int(round(v * 1e4)) for v in rbm.values], test=rr(B['test_multipli']),
         riferimento=rr(B['riferimento']), assicurazione=rr(B['assicurazione']), fee=B['fee'], fvol=rr(json.load(open(out + '/filtro_vol.json'))),
         code=rr(Dz))
json.dump(S, open(out + '/sito.json', 'w'), separators=(',', ':'))
print('sito.json', round(len(open(out + '/sito.json').read()) / 1e6, 2), 'MB')
# ---------------------------------------------------------------- figure
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e4e0', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'figure.dpi': 120})
L = {'it': dict(iv='IV ATM', rv='Volatilità realizzata fino a scadenza', vol='Volatilità annualizzata (%)', vrp='IV ATM − volatilità realizzata (punti)',
                freq='Numero di sedute', anno='Anno', oriz='Giorni alla scadenza', mf='Volatilità implicita model-free', bis='Bisettrice', mz='Retta di regressione',
                ivx='IV ATM all\'ingresso (%)', rvy='Volatilità realizzata successiva (%)', quint='Quintile di IV − GARCH all\'ingresso', vrpm='VRP medio successivo (punti)',
                sh='Indice di Sharpe annualizzato', ncfg='Numero di configurazioni', mid='Mid price', ba='Bid-ask', hvx='Volatilità storica a 21 sedute all\'ingresso (%)', cum='Rendimento cumulato sul margine (somma aritmetica)',
                h1='Sharpe 2010-2016', h2='Sharpe 2017-2023', ratio='Perdita a scadenza / premio incassato', delta='Delta della short put', gg='giorni',
                spx='S&P 500 (solo prezzo)', zx='Rendimento in deviazioni standard dalla media', cnt='Operazioni (scala logaritmica)',
                norm='Normale di pari media e varianza', qqx='Quantili della normale standard', qqy='Quantili empirici standardizzati'),
     'en': dict(iv='ATM IV', rv='Realized volatility to expiry', vol='Annualized volatility (%)', vrp='ATM IV − realized volatility (points)',
                freq='Number of sessions', anno='Year', oriz='Days to expiry', mf='Model-free implied volatility', bis='45° line', mz='Regression line',
                ivx='ATM IV at entry (%)', rvy='Subsequent realized volatility (%)', quint='Quintile of IV − GARCH at entry', vrpm='Mean subsequent VRP (points)',
                sh='Annualized Sharpe ratio', ncfg='Number of configurations', mid='Mid price', ba='Bid-ask', hvx='21-session historical volatility at entry (%)', cum='Cumulative return on margin (arithmetic sum)',
                h1='Sharpe 2010-2016', h2='Sharpe 2017-2023', ratio='Loss at expiry / premium collected', delta='Delta of the short put', gg='days',
                spx='S&P 500 (price only)', zx='Return in standard deviations from the mean', cnt='Trades (log scale)',
                norm='Normal with equal mean and variance', qqx='Standard normal quantiles', qqy='Standardized empirical quantiles')}
o30 = V['obiettivi']['30']; s = V['serie30']; d = pd.to_datetime(s['date'])
by = {c['key']: c for c in cfg}
names = {'it': {'SP': 'Short put', 'PCS': 'Put credit spread', 'CCS': 'Call credit spread', 'IC': 'Iron condor', 'STRD': 'Short straddle', 'STRG': 'Short strangle'},
         'en': {'SP': 'Short put', 'PCS': 'Put credit spread', 'CCS': 'Call credit spread', 'IC': 'Iron condor', 'STRD': 'Short straddle', 'STRG': 'Short strangle'}}
sel = [('SP|0.16|-|45|hold', 'SP', 'Δ0.16, 45'), ('PCS|0.20|d05|30|hold', 'PCS', 'Δ0.20/0.05, 30'), ('IC|0.16|d05|30|hold', 'IC', 'Δ0.16/0.05, 30'),
       ('STRD|0.50|-|30|hold', 'STRD', 'ATM, 30'), ('CCS|0.16|d05|30|hold', 'CCS', 'Δ0.16/0.05, 30')]
for lang, T in L.items():
    def save(name):
        plt.tight_layout(); plt.savefig(f'{out}/fig/{name}_{lang}.pdf'); plt.close()
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.plot(d, np.array(s['rvol'], float) * 100, color=C[1], lw=1.0, label=T['rv']); ax.plot(d, np.array(s['iv_atm'], float) * 100, color=C[0], lw=1.0, label=T['iv'])
    ax.set_ylabel(T['vol']); ax.legend(loc='upper left'); save('serie')
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    e = np.array(V['isto30']['bordi']); ax.bar(e[:-1] + 1, V['isto30']['conteggi'], width=1.7, color=C[0]); ax.axvline(0, color='#52514e', lw=0.8)
    ax.set_xlabel(T['vrp']); ax.set_ylabel(T['freq']); save('isto')
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    a = pd.DataFrame(o30['per_anno']); ax.bar(a.anno, a.vrp_vol * 100, color=[C[0] if v >= 0 else C[1] for v in a.vrp_vol], width=0.7); ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xticks(a.anno); ax.tick_params(axis='x', labelsize=8); ax.set_ylabel(T['vrp']); save('anni')
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    Ds = [7, 14, 21, 30, 45, 60]; st = V['struttura']
    ax.plot(Ds, [st[str(D)]['mf'] * 100 for D in Ds], 'o-', color=C[2], label=T['mf']); ax.plot(Ds, [st[str(D)]['iv'] * 100 for D in Ds], 'o-', color=C[0], label=T['iv'])
    ax.plot(Ds, [st[str(D)]['rv'] * 100 for D in Ds], 'o-', color=C[1], label=T['rv']); ax.set_xticks(Ds); ax.set_xlabel(T['oriz']); ax.set_ylabel(T['vol']); ax.legend(fontsize=8); save('struttura')
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    x = np.array(V['dispersione30']['iv']) * 100; y = np.array(V['dispersione30']['rv']) * 100
    ax.scatter(x, y, s=5, color=C[0], alpha=0.45, lw=0); xx = np.array([5, 85]); ax.plot(xx, xx, color='#52514e', lw=0.8, ls='--', label=T['bis'])
    ax.plot(xx, o30['mz_atm']['a'] * 100 + o30['mz_atm']['b'] * xx, color=C[1], lw=1.2, label=T['mz']); ax.set_xlabel(T['ivx']); ax.set_ylabel(T['rvy']); ax.legend(fontsize=8); save('mz')
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    q = o30['segnale_gap']; ax.bar([k['quintile'] for k in q], [k['vrp_vol'] * 100 for k in q], color=C[0], width=0.6); ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xlabel(T['quint']); ax.set_ylabel(T['vrpm']); save('gap')
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    bins = np.arange(-3, 2.01, 0.2); sh0 = [c['p0']['sharpe'] for c in cfg]; sh1 = [c['p1']['sharpe'] for c in cfg]
    ax.hist(np.clip(sh0, -3, 2), bins=bins, color=C[0], alpha=0.75, label=T['mid']); ax.hist(np.clip(sh1, -3, 2), bins=bins, color=C[1], alpha=0.65, label=T['ba'])
    ax.axvline(0, color='#52514e', lw=0.8); ax.set_xlabel(T['sh']); ax.set_ylabel(T['ncfg']); ax.legend(); save('sharpe')
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    md = pd.PeriodIndex(B['mesi'], freq='M').to_timestamp('M')
    for k, (key, fam, lab) in enumerate(sel):
        ax.plot(md, np.cumsum(by[key]['s1']) / 1e4, color=C[k], lw=1.2, label=f"{names[lang][fam]} ({lab} {T['gg']})")
    ax.plot(md, np.cumsum(rbm.values), color='#52514e', lw=1.0, ls='--', label=T['spx']); ax.axhline(0, color='#52514e', lw=0.6)
    ax.set_ylabel(T['cum'], fontsize=8); ax.legend(fontsize=7.5, ncol=1, loc='lower left'); save('curve')
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ins = pd.DataFrame(B['assicurazione'])
    for k, D in enumerate([7, 14, 30, 45]):
        g = ins[ins.D == D]; ax.plot(g.delta, g.rapporto, 'o-', color=C[k], label=f"{D} {T['gg']}")
    ax.axhline(1, color='#52514e', lw=0.8, ls='--'); ax.set_ylim(0, 1.1); ax.set_xticks([0.10, 0.16, 0.20, 0.30]); ax.set_xlabel(T['delta']); ax.set_ylabel(T['ratio']); ax.legend(fontsize=8); save('assicurazione')
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cr = [c for c in cfg if c['fam'] not in ('LP', 'DPS', 'CAL')]
    ax.scatter([c['h1']['sharpe'] for c in cr], [c['h2']['sharpe'] for c in cr], s=7, color=C[0], alpha=0.6, lw=0); ax.axhline(0, color='#52514e', lw=0.6); ax.axvline(0, color='#52514e', lw=0.6)
    lim = [-4, 5]; ax.plot(lim, lim, color='#52514e', lw=0.8, ls='--', label=T['bis']); ax.set_xlabel(T['h1']); ax.set_ylabel(T['h2']); ax.legend(fontsize=8); save('meta')
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    hb = S['fvol']['vrp_per_hv']; xs = ['≤10', '10–15', '15–20', '20–25', '25–30', '>30']
    ax.bar(xs, [b['vrp_atm'] * 100 for b in hb], color=[C[0] if b['vrp_atm'] >= 0 else C[1] for b in hb], width=0.6); ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xlabel(T['hvx']); ax.set_ylabel(T['vrpm']); save('hv')
    # distribuzione dei rendimenti per operazione: istogrammi in unità di deviazione standard e QQ plot
    pan = [('SP|0.16|-|45|hold', 'Short put Δ0.16, 45'), ('STRG|0.16|-|30|hold', 'Short strangle Δ0.16, 30'),
           ('PCS|0.20|d05|30|hold', 'Put credit spread Δ0.20/0.05, 30'), ('IC|0.16|w1|30|hold', 'Iron condor Δ0.16, 1%, 30'),
           ('LP|0.10|-|30|hold', 'Long put Δ0.10, 30'), ('D30', 'S&P 500, 30 ' + T['gg'])]
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.4))
    for ax, (k, tit) in zip(axs.ravel(), pan):
        h = Dz['indice'][k]['isto'] if k == 'D30' else Dz['config'][k]['isto']
        cen = h['lo'] + 0.25 + 0.5 * np.arange(len(h['n'])); nrm = np.array(h['normale'])
        ax.bar(cen, h['n'], width=0.5, color=C[5] if k == 'D30' else C[0], linewidth=0)
        ok = nrm >= 0.01; ax.plot(cen[ok], nrm[ok], color=C[1], lw=1.1)
        ax.set_yscale('log'); ax.set_ylim(0.6, None); ax.set_title(tit.replace(' Δ', '\nΔ', 1), fontsize=7.5); ax.tick_params(labelsize=7)
    fig.supxlabel(T['zx'], fontsize=8); fig.supylabel(T['cnt'], fontsize=8)
    fig.legend(handles=[plt.Line2D([], [], color=C[1], lw=1.1)], labels=[T['norm']], loc='upper center', ncol=1, fontsize=7.5)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(f'{out}/fig/code_{lang}.pdf'); plt.close()
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    qn = np.array(Dz['qn'])
    for j, (k, tit) in enumerate([pan[0], pan[2], pan[3], pan[4], pan[5]]):
        q = Dz['indice'][k]['isto']['qq'] if k == 'D30' else Dz['config'][k]['isto']['qq']
        ax.plot(qn, q, 'o-', ms=2.5, lw=0.9, color=C[[0, 1, 2, 4, 5][j]], label=tit)
    ax.plot([-3.5, 3.5], [-3.5, 3.5], color='#52514e', lw=0.8, ls='--'); ax.set_xlabel(T['qqx']); ax.set_ylabel(T['qqy'])
    ax.set_ylim(-16, 16); ax.legend(fontsize=6.5, loc='upper left'); save('qq')
print('figure scritte')
