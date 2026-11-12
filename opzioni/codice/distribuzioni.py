"""Distribuzione dei rendimenti delle strategie: le code sono grasse?

Due unità di analisi:
- operazione: rendimento sul margine a prezzi bid-ask, commissioni incluse (rho1 di lib.rendimenti_operazioni),
  un ingresso per seduta, quindi operazioni sovrapposte e dipendenti tra loro;
- mese: scala a capitale costante (serie mensili già calcolate da stat_bt.py e salvate in bt.json).

Statistiche:
- asimmetria e curtosi in eccesso (zero per la normale);
- frequenza dei rendimenti oltre 3 deviazioni standard dalla media, rapportata a quella della normale (0,135% per lato);
- expected shortfall all'1% in unità di deviazione standard, rapportato a quello della normale (2,665): misura di coda
  che non dipende da un'ipotesi parametrica e risente meno della curtosi campionaria dei pochi episodi estremi;
- CVaR al 5% e all'1% (media dei rendimenti sotto il quantile);
- test di Jarque-Bera solo sulle serie mensili: sulle operazioni giornaliere, sovrapposte, l'ipotesi di indipendenza
  del test non vale e il p-value sarebbe troppo piccolo;
- per le strutture a credito con rischio definito la perdita è limitata dal margine (larghezza meno credito): si misura
  la frequenza delle perdite oltre il 90% del margine e la si confronta con quella di una normale di pari media e varianza.

Indice di coda di Hill: solo sui rendimenti dell'indice. Per le strutture tenute a scadenza la perdita oltre lo strike è
lineare nel prezzo finale, quindi la coda delle perdite ha lo stesso indice di coda del sottostante. Sui rendimenti delle
strategie lo stimatore dipende dalla soglia (corpo della distribuzione schiacciato sul premio, operazioni sovrapposte che
replicano lo stesso episodio): per la short put 0,16 a 45 giorni varia da 0,5 a 1,7 tra k = 1% e k = 5%, e non si riporta.

Confronto: rendimenti dell'indice sullo stesso orizzonte di calendario D (uno per seduta, sovrapposti) e mensili.
Uso: python distribuzioni.py <cartella out> <cartella src>
"""
import sys, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, sys.argv[2])
from lib import rendimenti_operazioni, rendimenti_mensili_spx

out = sys.argv[1]
FEE = 1.0                                        # dollari per gamba e per contratto, come in stat_bt.py
P3 = float(stats.norm.cdf(-3))                   # probabilità normale oltre 3 deviazioni standard, per lato
ES1_NORM = float(stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01)   # expected shortfall all'1% della normale standard: 2,665
CREDITO_DEFINITO = ('PCS', 'CCS', 'IC', 'IFLY')  # perdita massima = margine
K_HILL = (0.01, 0.025, 0.05)                     # quota del campione nella coda per lo stimatore di Hill
PQ = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
      0.95, 0.975, 0.99, 0.995, 0.9975, 0.999, 0.9995]


def hill(x, quota):
    """Stimatore di Hill dell'indice di coda sulle k = quota·n osservazioni più grandi del campione (k almeno 10).
    Richiede che le k+1 osservazioni più grandi siano positive; altrimenti restituisce None."""
    n = len(x); x = np.sort(x)[::-1]
    k = max(10, int(round(quota * n)))
    if n <= k or x[k] <= 0:
        return None
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def descrivi(x, jb=False):
    """Momenti, quantili di coda e frequenze oltre 3 deviazioni standard di un campione."""
    x = np.asarray(x, float); m = x.mean(); sd = x.std(ddof=1)
    q01, q05 = np.quantile(x, [0.01, 0.05])
    o = dict(n=int(len(x)), media=float(m), sd=float(sd), asimmetria=float(stats.skew(x)), curtosi_ecc=float(stats.kurtosis(x)),
             q01=float(q01), q05=float(q05), cvar5=float(x[x <= q05].mean()), cvar1=float(x[x <= q01].mean()),
             minimo=float(x.min()), massimo=float(x.max()),
             rapp_3s_sx=float((x < m - 3 * sd).mean() / P3), rapp_3s_dx=float((x > m + 3 * sd).mean() / P3),
             rapp_es1=float((m - x[x <= q01].mean()) / sd / ES1_NORM))
    if jb:
        r = stats.jarque_bera(x); o['jb'] = float(r.statistic); o['jb_p'] = float(r.pvalue)
    return o


def istogramma(x):
    """Istogramma in unità di deviazione standard (classi di 0,5), conteggi attesi sotto la normale di pari media
    e varianza, quantili standardizzati per il QQ plot."""
    x = np.asarray(x, float); z = (x - x.mean()) / x.std(ddof=1)
    lo = np.floor(z.min() * 2) / 2; hi = np.ceil(z.max() * 2) / 2
    edges = np.arange(lo, hi + 0.25, 0.5)
    cnt = np.histogram(z, bins=edges)[0]
    attesi = len(z) * np.diff(stats.norm.cdf(edges))
    return dict(lo=float(lo), n=[int(c) for c in cnt], normale=[round(float(a), 3) for a in attesi],
                qq=[round(float(q), 3) for q in np.quantile(z, PQ)])


def indice_su_orizzonte(u, D):
    """Rendimenti dell'indice da ogni seduta alla prima chiusura disponibile dopo D giorni di calendario."""
    idx = u.index.searchsorted(u.index + pd.Timedelta(days=D))
    ok = idx < len(u)
    return u.values[idx[ok]] / u.values[ok] - 1.0


tr = rendimenti_operazioni(pd.read_parquet(out + '/operazioni.parquet'), FEE)
B = json.load(open(out + '/bt.json'))
u = pd.read_parquet(out + '/sottostante.parquet').close

C = {}
for (fam, par, ala, D, gest), g in tr.groupby(['fam', 'par', 'ala', 'D', 'gest'], sort=True):
    key = f'{fam}|{par:.2f}|{ala}|{D}|{gest}'
    x = g.rho1.values
    o = dict(op=descrivi(x), isto=istogramma(x))
    o['mese'] = descrivi(np.array(B['serie'][key + '|1']) / 1e4, jb=True)
    if fam in CREDITO_DEFINITO:
        m, sd = x.mean(), x.std(ddof=1)
        o['perdita_90'] = dict(osservata=float((x <= -0.9).mean()), normale=float(stats.norm.cdf((-0.9 - m) / sd)))
    C[key] = o

# configurazione a miglior rendimento cumulato per famiglia: somma dei rendimenti mensili del ladder ai prezzi bid-ask.
# Selezione a posteriori sullo stesso campione: premia la leva sul margine e sconta il problema di multiple testing.
MIG = {}
for c in B['config']:
    k = c['key']; fam = k.split('|')[0]
    tot = float(np.sum(B['serie'][k + '|1']) / 1e4)
    if fam not in MIG or tot > MIG[fam]['totale']:
        MIG[fam] = dict(key=k, totale=tot, totale_mid=float(np.sum(B['serie'][k + '|0']) / 1e4))

IND = {}
for D in (7, 14, 30, 45):
    r = indice_su_orizzonte(u, D)
    IND[f'D{D}'] = descrivi(r)
    a = [hill(-r, q) for q in K_HILL]
    IND[f'D{D}']['isto'] = istogramma(r)
    IND[f'D{D}']['hill_sx'] = a[1]; IND[f'D{D}']['hill_min'] = min(a); IND[f'D{D}']['hill_max'] = max(a)
IND['mensile'] = descrivi(rendimenti_mensili_spx(u).values, jb=True)

# sintesi per famiglia su tutte le configurazioni (nessuna selezione a posteriori)
F = {}
for fam in sorted({k.split('|')[0] for k in C}):
    v = [o for k, o in C.items() if k.startswith(fam + '|')]
    med = lambda f: float(np.median([f(o) for o in v]))
    F[fam] = dict(n=len(v), asim_op=med(lambda o: o['op']['asimmetria']), curt_op=med(lambda o: o['op']['curtosi_ecc']),
                  rapp_3s_sx=med(lambda o: o['op']['rapp_3s_sx']), rapp_3s_dx=med(lambda o: o['op']['rapp_3s_dx']), rapp_es1=med(lambda o: o['op']['rapp_es1']),
                  asim_mese=med(lambda o: o['mese']['asimmetria']), curt_mese=med(lambda o: o['mese']['curtosi_ecc']),
                  quota_jb_1=float(np.mean([o['mese']['jb_p'] < 0.01 for o in v])))
    if fam in CREDITO_DEFINITO:
        F[fam]['perdita_90_oss'] = med(lambda o: o['perdita_90']['osservata']); F[fam]['perdita_90_norm'] = med(lambda o: o['perdita_90']['normale'])
TOT = dict(n=len(C), quota_jb_1=float(np.mean([o['mese']['jb_p'] < 0.01 for o in C.values()])),
           quota_curt_mese_sopra_spx=float(np.mean([o['mese']['curtosi_ecc'] > IND['mensile']['curtosi_ecc'] for o in C.values()])),
           quota_asim_op_negativa=float(np.mean([o['op']['asimmetria'] < 0 for o in C.values()])))

json.dump(dict(config=C, indice=IND, famiglie=F, totale=TOT, migliori=MIG, pq=PQ, qn=[float(stats.norm.ppf(p)) for p in PQ], p3=P3, es1_normale=ES1_NORM,
               k_hill=K_HILL), open(out + '/distribuzioni.json', 'w'), separators=(',', ':'))

# riepilogo a schermo: configurazioni della tabella di selezione del paper e indice
SEL = ['SP|0.10|-|7|hold', 'SP|0.16|-|45|hold', 'STRG|0.16|-|30|hold', 'STRD|0.50|-|30|hold', 'PCS|0.20|d05|30|hold', 'PCS|0.20|w1|30|hold',
       'PCS|0.20|d05|30|tp50', 'PCS|0.20|d05|30|tp50sl200', 'IC|0.16|d05|30|hold', 'IC|0.16|w1|30|hold', 'IFLY|0.10|-|30|hold',
       'CCS|0.16|d05|30|hold', 'CAL|0.50|-|30|hold', 'LP|0.10|-|30|hold', 'DPS|0.30|-|30|hold']
rows = []
for k in SEL:
    o = C[k]; r = dict(cfg=k, n=o['op']['n'], asim=o['op']['asimmetria'], curt=o['op']['curtosi_ecc'], r3sx=o['op']['rapp_3s_sx'], r3dx=o['op']['rapp_3s_dx'],
                      es1=o['op']['rapp_es1'], cvar1=o['op']['cvar1'], minimo=o['op']['minimo'], asim_m=o['mese']['asimmetria'], curt_m=o['mese']['curtosi_ecc'],
                      jb_p=o['mese']['jb_p'])
    if 'perdita_90' in o: r['p90'] = o['perdita_90']['osservata']; r['p90n'] = o['perdita_90']['normale']
    rows.append(r)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(3).to_string())
print(pd.DataFrame({k: {c: v[c] for c in v if c != 'isto'} for k, v in IND.items()}).T[
    ['n', 'asimmetria', 'curtosi_ecc', 'rapp_3s_sx', 'rapp_3s_dx', 'rapp_es1', 'cvar1', 'minimo', 'hill_sx', 'hill_min', 'hill_max']].round(3).to_string())
print(pd.DataFrame(F).T.round(3).to_string())
print(TOT)
print({f: (m['key'], round(m['totale'], 3)) for f, m in MIG.items()})
