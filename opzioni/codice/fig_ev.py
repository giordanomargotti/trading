"""Figura della sezione sulla combinazione scelta per massimo valore atteso sotto la densità GARCH (bt_ev.py):
calibrazione del rendimento atteso e confronto degli Sharpe con la regola a delta. Uso: python fig_ev.py <cartella out>"""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
out = sys.argv[1]
G = json.load(open(out + '/garch_ev.json'))
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e4e0', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'figure.dpi': 120})
FAM = {'SP': 'Short put', 'SC': 'Short call', 'PCS': 'Put credit spread', 'IC': 'Iron condor', 'LP': 'Long put', 'LC': 'Long call'}
COL = dict(zip(FAM, C))
L = {'it': dict(x='Rendimento atteso GARCH (%)', y='Rendimento realizzato (%)', a='Vendita di volatilità', b='Acquisto di opzioni',
                sx='Sharpe, regola delta (mediana)', sy='Sharpe con regola EV', bis='Bisettrice'),
     'en': dict(x='GARCH expected return (%)', y='Realized return (%)', a='Volatility selling', b='Option buying',
                sx='Sharpe ratio, delta rule (median)', sy='Sharpe ratio, EV rule', bis='45° line')}
for lang, T in L.items():
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.6))
    for ax, fams, tit in ((axs[0], ['SP', 'SC', 'PCS', 'IC'], T['a']), (axs[1], ['LP', 'LC'], T['b'])):
        lo, hi = 0, 0
        for f in fams:
            c = G['calibrazione'][f]; x = 100 * np.array(c['er']); y = 100 * np.array(c['rho'])
            ax.plot(x, y, 'o-', color=COL[f], ms=3, lw=1.1, label=FAM[f]); lo = min(lo, x.min(), y.min()); hi = max(hi, x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color='#52514e', lw=0.8, ls=':', label=T['bis'])
        ax.axhline(0, color='#52514e', lw=0.6); ax.set_title(tit, fontsize=8); ax.set_xlabel(T['x'], fontsize=7.5); ax.legend(fontsize=6)
    axs[0].set_ylabel(T['y'], fontsize=7.5)
    ax = axs[2]
    for k, o in G['config'].items():
        f = k.split('|')[0]; v = o['delta_sharpe_livelli']
        ax.errorbar(np.median(v), o['sharpe'], xerr=[[np.median(v) - min(v)], [max(v) - np.median(v)]], fmt='o', ms=3, color=COL[f], lw=0.7, capsize=0)
    lim = [-1.4, 1.2]; ax.plot(lim, lim, color='#52514e', lw=0.8, ls=':'); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(T['sx'], fontsize=7.5); ax.set_ylabel(T['sy'], fontsize=7.5)
    plt.tight_layout(); plt.savefig(f'{out}/fig/garch_ev_{lang}.pdf'); plt.close()
print('figura scritta')
