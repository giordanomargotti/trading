"""Figura della sezione sugli strike scelti dalla densità GARCH: calibrazione e confronto degli Sharpe.
Uso: python fig_garch.py <cartella out>"""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
out = sys.argv[1]
G = json.load(open(out + '/garch_strike.json'))
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e4e0', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'figure.dpi': 120})
L = {'it': dict(x='Livello obiettivo: delta o probabilità GARCH (%)', y='Scadenze in the money (%)', pd='Put, regola delta', pg='Put, regola GARCH',
                cd='Call, regola delta', cg='Call, regola GARCH', bis='Bisettrice', sx='Sharpe con regola delta', sy='Sharpe con regola GARCH'),
     'en': dict(x='Target level: delta or GARCH probability (%)', y='Expiries in the money (%)', pd='Put, delta rule', pg='Put, GARCH rule',
                cd='Call, delta rule', cg='Call, GARCH rule', bis='45° line', sx='Sharpe ratio, delta rule', sy='Sharpe ratio, GARCH rule')}
FAM = {'SP': 'Short put', 'SC': 'Short call', 'PCS': 'Put credit spread', 'IC': 'Iron condor', 'LP': 'Long put', 'LC': 'Long call'}
liv = [0.10, 0.16, 0.20, 0.30]
for lang, T in L.items():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    x = [100 * p for p in liv]; cal = G['calibrazione']
    for fam, regola, col, ls, lab in (('SP', 'delta', C[0], '-', T['pd']), ('SP', 'garch', C[0], '--', T['pg']), ('SC', 'delta', C[1], '-', T['cd']), ('SC', 'garch', C[1], '--', T['cg'])):
        a.plot(x, [100 * cal[f'{fam}|{p:.2f}'][regola] for p in liv], 'o' + ls, color=col, ms=3.5, lw=1.2, label=lab)
    a.plot([5, 35], [5, 35], color='#52514e', lw=0.8, ls=':', label=T['bis']); a.set_xlabel(T['x'], fontsize=8); a.set_ylabel(T['y'], fontsize=8); a.legend(fontsize=6.5)
    for j, fam in enumerate(FAM):
        v = [o for k, o in G['config'].items() if k.startswith(fam + '|')]
        b.scatter([o['delta']['sharpe'] for o in v], [o['garch']['sharpe'] for o in v], s=12, color=C[j], label=FAM[fam])
    lim = [-1.6, 1.6]; b.plot(lim, lim, color='#52514e', lw=0.8, ls=':'); b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel(T['sx'], fontsize=8); b.set_ylabel(T['sy'], fontsize=8); b.legend(fontsize=6.5, loc='lower right')
    plt.tight_layout(); plt.savefig(f'{out}/fig/garch_strike_{lang}.pdf'); plt.close()
print('figura scritta')
