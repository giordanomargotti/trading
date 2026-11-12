"""Numeri e tabelle per i documenti LaTeX, nelle due lingue."""
import json, numpy as np
def load(out):
    S = json.load(open(out + '/sito.json')); S['fvol'] = json.load(open(out + '/filtro_vol.json')); return S
def fmt(x, d, lang, sign=False):
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    s = s.replace('-', '$-$')
    return s.replace('.', '{,}') if lang == 'it' else s
def integer(n, lang):
    s = f'{int(n):,}'
    return s.replace(',', '.') if lang == 'it' else s
_F = {'SP': 'Short put', 'STRG': 'Short strangle', 'STRD': 'Short straddle ATM', 'PCS': 'Put credit spread', 'CCS': 'Call credit spread', 'IC': 'Iron condor',
      'IFLY': 'Iron butterfly', 'LP': 'Long put', 'DPS': 'Debit put spread', 'CAL': 'Long calendar ATM'}
FAM = {'it': _F, 'en': dict(_F, STRD='Short ATM straddle', CAL='Long ATM calendar')}
GEST_TXT = {'it': {'hold': 'a scadenza', 'tp50': 'take profit 50%', 'tp50sl200': 'take profit 50% e stop loss 200%'},
            'en': {'hold': 'held to expiry', 'tp50': '50% take profit', 'tp50sl200': '50% take profit and 200% stop loss'}}

def etichetta(key, lang, tex=False, nome=True):
    """Descrizione di una configurazione (famiglia, delta o ATM, wing, giorni, gestione) per tabelle, testo e figure.
    Il calendar vende la scadenza D e compra quella a circa 2D giorni."""
    fam, par, ala, D, gest = key.split('|'); p = float(par)
    d = r'$\Delta$' if tex else 'Δ'
    if fam in ('STRD', 'CAL'):
        ps = ''                                     # il nome della famiglia contiene già ATM
    elif fam == 'IFLY':
        ps = f'ATM/{d}{p:.2f}'
    else:
        ps = f'{d}{p:.2f}' + ('/0.05' if ala == 'd05' else (', wing 1%' if ala == 'w1' else ''))
    gg = 'gg' if lang == 'it' else 'd'
    dur = f'{D}/{2 * int(D)} {gg}' if fam == 'CAL' else f'{D} {gg}'
    s = ', '.join(x for x in (ps, dur, GEST_TXT[lang][gest]) if x)
    if tex:
        s = s.replace('%', r'\%')
    if lang == 'it':
        s = s.replace('0.', '0{,}' if tex else '0,')
    return (FAM[lang][fam] + ' ' + s) if nome else s
def numbers(S, lang):
    V = S['vrp']; T = S['test']; o = V['obiettivi']; K = {}
    f = lambda x, d=2, sign=False: fmt(x, d, lang, sign)
    for D in ('7', '14', '21', '30', '45', '60'):
        a = o[D]
        K[f'iv{D}'] = f(a['iv_atm_media'] * 100); K[f'mf{D}'] = f(a['mfvol_media'] * 100); K[f'rv{D}'] = f(a['rvol_media'] * 100)
        K[f'vrp{D}'] = f(a['vrp_vol']['media'] * 100, 2, True); K[f'vrpt{D}'] = f(a['vrp_vol']['t_hac']); K[f'vrpmed{D}'] = f(a['vrp_vol']['mediana'] * 100, 2, True)
        K[f'vrppos{D}'] = f(a['vrp_vol']['quota_positiva'] * 100, 1); K[f'vrpsk{D}'] = f(a['vrp_vol']['asimmetria'], 1); K[f'vrpmin{D}'] = f(a['vrp_vol']['minimo'] * 100, 1)
        K[f'mfvrp{D}'] = f(a['vrp_mfvol']['media'] * 100, 2, True); K[f'mfvrpt{D}'] = f(a['vrp_mfvol']['t_hac']); K[f'mfpos{D}'] = f(a['vrp_mfvol']['quota_positiva'] * 100, 1)
        K[f'var{D}'] = f(a['vrp_var']['media'] * 1e4, 0, True); K[f'vart{D}'] = f(a['vrp_var']['t_hac'])
        K[f'mza{D}'] = f(a['mz_atm']['a'] * 100, 2); K[f'mzb{D}'] = f(a['mz_atm']['b'], 3); K[f'mzse{D}'] = f(a['mz_atm']['es_b'], 3); K[f'mzr{D}'] = f(a['mz_atm']['r2'], 2)
        K[f'mzp{D}'] = f(a['mz_atm']['p_wald'], 3); K[f'mzbmf{D}'] = f(a['mz_mf']['b'], 3); K[f'n{D}'] = integer(a['giorni'], lang); K[f'nn{D}'] = integer(a['non_sovrapposto']['n'], lang)
        K[f'nnt{D}'] = f(a['non_sovrapposto']['vrp_vol']['t_hac'])
    a = o['30']
    for k, nm in (('2010-2014', 'pa'), ('2015-2019', 'pb'), ('2020-2023', 'pc')):
        K[nm] = f(a['per_periodo'][k]['media'] * 100, 2, True); K[nm + 't'] = f(a['per_periodo'][k]['t_hac'])
    pr = a['previsioni']['modelli']
    for k, nm in (('IV ATM', 'fiv'), ('Model-free', 'fmf'), ('GJR-GARCH-t', 'fga'), ('HV21', 'fhv')):
        K[nm + 'rmse'] = f(pr[k]['rmse'] * 100); K[nm + 'ql'] = f(pr[k]['qlike'], 3); K[nm + 'bias'] = f(pr[k]['errore_medio'] * 100, 2, True); K[nm + 'corr'] = f(pr[k]['corr'], 2)
    K['fn'] = integer(a['previsioni']['n'], lang)
    e = a['encompassing']
    for i, nm in enumerate(('ec', 'eiv', 'ega', 'ehv')):
        K[nm] = f(e['coef'][i], 2, True); K[nm + 't'] = f(e['t'][i])
    K['er2'] = f(e['r2'], 2)
    for q in a['segnale_gap']:
        K[f"gq{q['quintile']}"] = f(q['vrp_vol'] * 100, 2, True); K[f"gg{q['quintile']}"] = f(q['gap_medio'] * 100, 1, True)
    K['gapb'] = f(a['segnale_gap_reg']['b'], 2); K['gapt'] = f(a['segnale_gap_reg']['t_b']); K['gapr2'] = f(a['segnale_gap_reg']['r2'] * 100, 1)
    K['rr25'] = f(a['skew']['rr25'] * 100); K['p10atm'] = f(a['skew']['p10_meno_atm'] * 100)
    K['sedute'] = integer(V['meta']['sedute'], lang); K['coppie'] = integer(V['meta']['coppie'], lang)
    K['ncfg'] = str(T['n_config']); K['shmid'] = f(T['sharpe_mediano_mid'], 2, True); K['shba'] = f(T['sharpe_mediano_ba'], 2, True)
    K['posmid'] = f(T['quota_sharpe_pos_mid'] * 100, 0); K['posba'] = f(T['quota_sharpe_pos_ba'] * 100, 0)
    K['amid'] = f(T['quota_talfa2_mid'] * 100, 1); K['aba'] = f(T['quota_talfa2_ba'] * 100, 1); K['aneg'] = f(T['quota_talfa_neg2_ba'] * 100, 1)
    K['best'] = f(T['sharpe_migliore']); K['sr0'] = f(T['sharpe_atteso_massimo_sotto_h0']); K['dsr'] = f(T['dsr'], 3)
    K['rho'] = f(T['corr_meta']); K['toph1'] = f(T['top20_h1_media_h1']); K['toph2'] = f(T['top20_h1_media_h2']); K['medh2'] = f(T['mediana_h2'], 2, True)
    R = S['riferimento']; K['spxsh'] = f(R['sharpe']); K['spxann'] = f(R['media_annua'] * 100, 1); K['spxdd'] = f(R['max_dd'] * 100, 0)
    by = {c['key']: c for c in S['config']}
    K['ntrade'] = integer(sum(c['n'] for c in S['config']), lang)
    for nm, key in (('sp', 'SP|0.16|-|45|hold'), ('pcs', 'PCS|0.20|d05|30|hold'), ('pcsw', 'PCS|0.20|w1|30|hold'), ('ic', 'IC|0.16|d05|30|hold'), ('strd', 'STRD|0.50|-|30|hold'), ('ccs', 'CCS|0.16|d05|30|hold')):
        c = by[key]
        K[nm + 'win'] = f(c['t1']['vincenti'] * 100, 0); K[nm + 'usd'] = integer(round(c['t1']['media_usd']), lang).replace('-', '$-$'); K[nm + 'worst'] = integer(round(-c['t1']['peggiore_usd']), lang)
        K[nm + 'sh0'] = f(c['p0']['sharpe'], 2, True); K[nm + 'sh1'] = f(c['p1']['sharpe'], 2, True); K[nm + 'alfa'] = f(c['capm1']['alfa_annuo'] * 100, 1, True); K[nm + 'ta'] = f(c['capm1']['t_alfa'], 2, True)
        K[nm + 'beta'] = f(c['capm1']['beta']); K[nm + 'dd'] = f(c['p1']['max_dd'] * 100, 0); K[nm + 'sc'] = f(c['spread_su_credito'] * 100, 0)
        K[nm + 'fg'] = f(c['filtri'].get('gap>0', {}).get('sharpe', float('nan')), 2, True); K[nm + 'fn'] = f(c['filtri']['nessuno']['sharpe'], 2, True)
    ins = {(i['D'], i['delta']): i for i in S['assicurazione']}
    for D in (7, 14, 30, 45):
        for dl in (0.10, 0.30):
            K[f'ins{D}_{int(dl * 100)}'] = f(ins[(D, dl)]['rapporto'], 2)
    cr = [c for c in S['config'] if c['fam'] not in ('LP', 'DPS', 'CAL')]
    fn = np.array([c['filtri']['nessuno']['sharpe'] for c in cr]); fg = np.array([c['filtri'].get('gap>0', {}).get('sharpe', np.nan) for c in cr])
    K['filtq'] = f(np.nanmean(fg > fn) * 100, 0); K['filtn'] = f(np.median(fn), 2, True); K['filtg'] = f(np.nanmedian(fg), 2, True)
    for g in ('hold', 'tp50', 'tp50sl200'):
        x = [c for c in cr if c['gest'] == g]
        K['g0' + g] = f(np.median([c['p0']['sharpe'] for c in x]), 2, True); K['g1' + g] = f(np.median([c['p1']['sharpe'] for c in x]), 2, True)
    W = S['fvol']; fl = W['filtri']
    for nm, k in (('f20', 'hv21<=20'), ('f25', 'hv21<=25'), ('f30', 'hv21<=30'), ('f90', 'hv21<=p90'), ('fhi', 'solo_hv21>25')):
        K[nm + 'sh'] = f(fl[k]['sharpe_mediano'], 2, True); K[nm + 'q'] = f(fl[k]['quota_operazioni'] * 100, 0); K[nm + 'better'] = f(fl[k]['quota_sharpe_migliore'] * 100, 0); K[nm + 'dd'] = f(fl[k]['quota_dd_minore'] * 100, 0)
    K['f0sh'] = f(fl['nessuno']['sharpe_mediano'], 2, True)
    sp = W['selezione']['SP|0.16|-|45|hold']
    K['fsp0'] = f(sp['nessuno']['sharpe'], 2, True); K['fsp25'] = f(sp['hv21<=25']['sharpe'], 2, True); K['fsphi'] = f(sp['solo_hv21>25']['sharpe'], 2, True)
    K['fspw0'] = f(-sp['nessuno']['peggiore'] * 100, 1); K['fspw25'] = f(-sp['hv21<=25']['peggiore'] * 100, 1)
    K['fsprho0'] = f(sp['nessuno']['rho'] * 100, 1); K['fsprhohi'] = f(sp['solo_hv21>25']['rho'] * 100, 1)
    ic = W['selezione']['IC|0.16|d05|30|hold']; K['fic0'] = f(ic['nessuno']['sharpe'], 2, True); K['fic25'] = f(ic['hv21<=25']['sharpe'], 2, True)
    ds = W['distanza_strike']
    for i, nm in enumerate(('dlo', 'dmi', 'dhi')):
        K[nm] = f(ds['terzili'][i]['dist'] * 100, 1); K[nm + 'iv'] = f(ds['terzili'][i]['iv'] * 100, 1)
    K['dcorr'] = f(ds['corr'], 2)
    K.update(numeri_code(S, lang)); K.update(numeri_migliori(S, lang))
    hb = W['vrp_per_hv']; K['hvlo'] = f(hb[0]['vrp_atm'] * 100, 2, True); K['hvhi'] = f(hb[-1]['vrp_atm'] * 100, 2, True); K['hvmid'] = f(hb[4]['vrp_atm'] * 100, 2, True)
    return K
def tab_fvol(S, lang):
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    lab = {'it': {'nessuno': 'Nessun filtro', 'hv21<=20': r'HV21 $\le$ 20\%', 'hv21<=25': r'HV21 $\le$ 25\%', 'hv21<=30': r'HV21 $\le$ 30\%', 'hv21<=p90': r'HV21 $\le$ 90° percentile', 'hv10<=25': r'HV10 $\le$ 25\%', 'solo_hv21>25': r'Solo HV21 $>$ 25\%'},
           'en': {'nessuno': 'No filter', 'hv21<=20': r'HV21 $\le$ 20\%', 'hv21<=25': r'HV21 $\le$ 25\%', 'hv21<=30': r'HV21 $\le$ 30\%', 'hv21<=p90': r'HV21 $\le$ 90th percentile', 'hv10<=25': r'HV10 $\le$ 25\%', 'solo_hv21>25': r'Only HV21 $>$ 25\%'}}[lang]
    rows = []
    for k, v in S['fvol']['filtri'].items():
        g = lambda n, d=0: f(v[n] * 100, d) if n in v else '--'
        rows.append(' & '.join([lab[k], f(v['quota_operazioni'] * 100, 0), f(v['sharpe_mediano'], 2, True), g('quota_sharpe_migliore'), f(v['dd_mediano'] * 100, 0), g('quota_dd_minore'), f(v['peggiore_mediano'] * 100, 0)]) + r' \\')
    return '\n'.join(rows)
def tab_hv(S, lang):
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    names = [r'$\le$ 10', '10--15', '15--20', '20--25', '25--30', r'$>$ 30']
    return '\n'.join(' & '.join([names[i], integer(b['n'], lang), f(b['iv'] * 100), f(b['rv'] * 100), f(b['vrp_atm'] * 100, 2, True), f(b['t']), f(b['vrp_mf'] * 100, 2, True), f(b['quota_pos'] * 100, 0), f(b['p5'] * 100, 1), f(b['minimo'] * 100, 0)]) + r' \\'
                     for i, b in enumerate(S['fvol']['vrp_per_hv']))
def tab_orizzonti(S, lang):
    o = S['vrp']['obiettivi']; f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    rows = []
    for D in ('7', '14', '21', '30', '45', '60'):
        a = o[D]
        rows.append(' & '.join([D, integer(a['giorni'], lang), f(a['iv_atm_media'] * 100), f(a['mfvol_media'] * 100), f(a['rvol_media'] * 100),
                                f(a['vrp_vol']['media'] * 100, 2, True), f(a['vrp_vol']['t_hac']), f(a['vrp_vol']['quota_positiva'] * 100, 0),
                                f(a['vrp_mfvol']['media'] * 100, 2, True), f(a['vrp_mfvol']['t_hac']), f(a['vrp_vol']['asimmetria'], 1), f(a['vrp_vol']['minimo'] * 100, 0)]) + r' \\')
    return '\n'.join(rows)
def tab_mz(S, lang):
    o = S['vrp']['obiettivi']; f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    rows = []
    for D in ('7', '14', '21', '30', '45', '60'):
        a = o[D]; m = a['mz_atm']; g = a['mz_mf']
        rows.append(' & '.join([D, f(m['a'] * 100, 2, True), f(m['b'], 3), f(m['es_b'], 3), f(m['r2'], 2), f(m['p_wald'], 3), f(g['b'], 3), f(g['p_wald'], 3)]) + r' \\')
    return '\n'.join(rows)
def tab_anni(S, lang):
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    return '\n'.join(' & '.join([str(a['anno']), str(a['n']), f(a['iv'] * 100), f(a['rv'] * 100), f(a['vrp_vol'] * 100, 2, True), f(a['quota_positiva'] * 100, 0), f(a['minimo'] * 100, 1)]) + r' \\'
                     for a in S['vrp']['obiettivi']['30']['per_anno'])
def tab_famiglie(S, lang):
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    rows = []
    order = [('SP', '-'), ('STRG', '-'), ('PCS', 'd05'), ('PCS', 'w1'), ('STRD', '-'), ('IC', 'd05'), ('IC', 'w1'), ('IFLY', '-'), ('CCS', 'd05'), ('CCS', 'w1'), ('CAL', '-'), ('DPS', '-'), ('LP', '-')]
    wl = {'it': {'d05': r', wing $\Delta$ 0{,}05', 'w1': r', wing 1\% dello spot', '-': ''}, 'en': {'d05': r', $\Delta$ 0.05 wing', 'w1': r', 1\% of spot wing', '-': ''}}[lang]
    for fam, ala in order:
        x = [c for c in S['config'] if c['fam'] == fam and c['ala'] == ala and c['gest'] == 'hold']
        med = lambda g: float(np.median([g(c) for c in x]))
        rows.append(' & '.join([FAM[lang][fam] + wl[ala], str(len(x)), f(med(lambda c: c['p0']['sharpe']), 2, True), f(med(lambda c: c['p1']['sharpe']), 2, True),
                                f(med(lambda c: c['capm1']['beta']), 2, True), f(med(lambda c: c['capm0']['t_alfa']), 2, True), f(med(lambda c: c['capm1']['t_alfa']), 2, True),
                                f(med(lambda c: c['spread_su_credito']) * 100, 0), f(med(lambda c: c['t1']['vincenti']) * 100, 0)]) + r' \\')
    return '\n'.join(rows)
# configurazioni rappresentative: tabella di selezione e tabella delle code
SEL_RAPPR = [('SP|0.10|-|7|hold', r'$\Delta$0.10, 7'), ('SP|0.16|-|45|hold', r'$\Delta$0.16, 45'), ('STRG|0.16|-|30|hold', r'$\Delta$0.16, 30'), ('PCS|0.20|d05|30|hold', r'$\Delta$0.20/0.05, 30'),
             ('PCS|0.20|w1|30|hold', r'$\Delta$0.20, 1\%, 30'), ('PCS|0.20|d05|30|tp50', r'$\Delta$0.20/0.05, 30, TP50'), ('PCS|0.20|d05|30|tp50sl200', r'$\Delta$0.20/0.05, 30, TP50/SL200'),
             ('STRD|0.50|-|30|hold', 'ATM, 30'), ('IC|0.16|d05|30|hold', r'$\Delta$0.16/0.05, 30'), ('IC|0.16|w1|30|hold', r'$\Delta$0.16, 1\%, 30'), ('IFLY|0.10|-|30|hold', r'ATM/$\Delta$0.10, 30'),
             ('CCS|0.16|d05|30|hold', r'$\Delta$0.16/0.05, 30'), ('CAL|0.50|-|30|hold', 'ATM, 30/60'), ('LP|0.10|-|30|hold', r'$\Delta$0.10, 30')]
def tab_selezione(S, lang):
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    by = {c['key']: c for c in S['config']}
    gg = 'gg' if lang == 'it' else 'd'
    rows = []
    for key, lab in SEL_RAPPR:
        c = by[key]
        if lang == 'it': lab = lab.replace('0.', '0{,}')
        rows.append(' & '.join([FAM[lang][c['fam']], lab + ' ' + gg, f(c['t1']['vincenti'] * 100, 0), integer(round(c['t1']['media_usd']), lang).replace('-', '$-$'), integer(round(c['t1']['peggiore_usd']), lang).replace('-', '$-$'),
                                f(c['p0']['sharpe'], 2, True), f(c['p1']['sharpe'], 2, True), f(c['p1']['max_dd'] * 100, 0), f(c['capm1']['beta'], 2, True), f(c['capm1']['alfa_annuo'] * 100, 1, True), f(c['capm1']['t_alfa'], 2, True)]) + r' \\')
    return '\n'.join(rows)
def tab_ins(S, lang):
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    ins = {(i['D'], i['delta']): i for i in S['assicurazione']}
    return '\n'.join(' & '.join([str(D)] + [f(ins[(D, dl)]['rapporto']) for dl in (0.10, 0.16, 0.20, 0.30)] + [f(ins[(D, dl)]['quota_itm'] * 100, 1) for dl in (0.10, 0.30)]) + r' \\' for D in (7, 14, 30, 45))
def tab_filtri(S, lang):
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    cr = [c for c in S['config'] if c['fam'] not in ('LP', 'DPS', 'CAL')]
    rows = []
    for fam in ('SP', 'STRG', 'PCS', 'STRD', 'IC', 'IFLY', 'CCS'):
        x = [c for c in cr if c['fam'] == fam]
        med = lambda k: float(np.nanmedian([c['filtri'].get(k, {}).get('sharpe', np.nan) for c in x]))
        rows.append(' & '.join([FAM[lang][fam], str(len(x)), f(med('nessuno'), 2, True), f(med('gap>0'), 2, True), f(med('gap<=0'), 2, True), f(med('iv_alta'), 2, True), f(med('iv_bassa'), 2, True)]) + r' \\')
    return '\n'.join(rows)

def numeri_code(S, lang):
    """Numeri del testo sulla distribuzione dei rendimenti (sezione sulle code)."""
    f = lambda x, d=2, sign=False: fmt(x, d, lang, sign)
    Z = S['code']; C = Z['config']; I = Z['indice']; T = Z['totale']
    sp, pcs, pcsl, icw, lp, cal = (C[k] for k in ('SP|0.16|-|45|hold', 'PCS|0.20|d05|30|hold', 'PCS|0.20|d05|30|tp50sl200', 'IC|0.16|w1|30|hold', 'LP|0.10|-|30|hold', 'CAL|0.50|-|30|hold'))
    return dict(csp_sk=f(sp['op']['asimmetria'], 1), csp_ku=f(sp['op']['curtosi_ecc'], 0), csp_r3=f(sp['op']['rapp_3s_sx'], 1), csp_es=f(sp['op']['rapp_es1'], 1),
                csp_msk=f(sp['mese']['asimmetria'], 1), csp_mku=f(sp['mese']['curtosi_ecc'], 0),
                ci45_sk=f(I['D45']['asimmetria'], 1), ci45_ku=f(I['D45']['curtosi_ecc'], 1), ci30_es=f(I['D30']['rapp_es1'], 1),
                cim_sk=f(I['mensile']['asimmetria'], 2), cim_ku=f(I['mensile']['curtosi_ecc'], 2), cim_jbp=f(I['mensile']['jb_p'], 2),
                cq_asneg=f(T['quota_asim_op_negativa'] * 100, 0), cq_jb=f(T['quota_jb_1'] * 100, 0), cq_kuspx=f(T['quota_curt_mese_sopra_spx'] * 100, 0),
                chill30=f(I['D30']['hill_sx'], 1), chillmin=f(I['D30']['hill_min'], 1), chillmax=f(I['D30']['hill_max'], 1),
                cpcs_ku=f(pcs['op']['curtosi_ecc'], 1), cpcs_r3=f(pcs['op']['rapp_3s_sx'], 1), cpcsl_ku=f(pcsl['op']['curtosi_ecc'], 1),
                cicw_p90=f(icw['perdita_90']['osservata'] * 100, 1), cicw_p90n=f(icw['perdita_90']['normale'] * 100, 1), cicw_ku=f(icw['op']['curtosi_ecc'], 2),
                clp_sk=f(lp['op']['asimmetria'], 1), clp_ku=f(lp['op']['curtosi_ecc'], 0), ccal_jb=f(cal['mese']['jb_p'], 2))

def tab_code(S, lang):
    """Tabella della distribuzione dei rendimenti: configurazioni rappresentative e indice."""
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    Z = S['code']; C = Z['config']; I = Z['indice']
    gg = 'gg' if lang == 'it' else 'd'
    pj = lambda p: '$<$' + f(0.001, 3) if p < 0.001 else f(p, 3)
    rows = []
    for key, lab in SEL_RAPPR:
        o = C[key]; fam = key.split('|')[0]
        if lang == 'it': lab = lab.replace('0.', '0{,}')
        rows.append(' & '.join([FAM[lang][fam], lab + ' ' + gg, f(o['op']['asimmetria'], 2), f(o['op']['curtosi_ecc'], 1), f(o['op']['rapp_3s_sx'], 1), f(o['op']['rapp_es1'], 2),
                                f(o['op']['minimo'] * 100, 0), f(o['mese']['asimmetria'], 2), f(o['mese']['curtosi_ecc'], 1), pj(o['mese']['jb_p'])]) + r' \\')
    i30, im = I['D30'], I['mensile']
    rows.append(r'\midrule')
    rows.append(' & '.join([r'S\&P 500', '30 ' + gg, f(i30['asimmetria'], 2), f(i30['curtosi_ecc'], 1), f(i30['rapp_3s_sx'], 1), f(i30['rapp_es1'], 2), f(i30['minimo'] * 100, 0),
                            f(im['asimmetria'], 2), f(im['curtosi_ecc'], 1), pj(im['jb_p'])]) + r' \\')
    return '\n'.join(rows)

ORD_FAM = ['SP', 'STRG', 'STRD', 'PCS', 'IC', 'IFLY', 'CCS', 'CAL', 'DPS', 'LP']

def numeri_migliori(S, lang):
    """Numeri del testo sulle configurazioni a miglior rendimento cumulato (put credit spread, iron condor, short put, long put)."""
    f = lambda x, d=2, sign=False: fmt(x, d, lang, sign)
    MG = S['code']['migliori']; by = {c['key']: c for c in S['config']}; Z = S['code']['config']; K = {}
    for fam, p in (('PCS', 'mpcs'), ('IC', 'mic'), ('SP', 'msp'), ('LP', 'mlp')):
        k = MG[fam]['key']; c = by[k]; o = Z[k]
        K.update({p + '_cfg': etichetta(k, lang, tex=True, nome=False), p + '_tot': f(MG[fam]['totale'] * 100, 0, True), p + '_ann': f(c['p1']['media_annua'] * 100, 1, True),
                  p + '_sh': f(c['p1']['sharpe'], 2, True), p + '_dd': f(c['p1']['max_dd'] * 100, 0), p + '_wm': f(c['p1']['mese_peggiore'] * 100, 0, True),
                  p + '_sk': f(o['op']['asimmetria'], 1, True), p + '_ku': f(o['op']['curtosi_ecc'], 0), p + '_min': f(o['op']['minimo'] * 100, 0, True),
                  p + '_max': f(o['op']['massimo'] * 100, 0, True), p + '_vinc': f(c['t1']['vincenti'] * 100, 0)})
    return K

def tab_migliori(S, lang):
    """Configurazione a miglior rendimento cumulato (ladder, prezzi bid-ask) per ogni famiglia."""
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    MG = S['code']['migliori']; by = {c['key']: c for c in S['config']}; Z = S['code']['config']
    rows = []
    for fam in ORD_FAM:
        k = MG[fam]['key']; c = by[k]; o = Z[k]
        rows.append(' & '.join([FAM[lang][fam], etichetta(k, lang, tex=True, nome=False), f(MG[fam]['totale'] * 100, 0, True), f(c['p1']['media_annua'] * 100, 1, True),
                                f(c['p1']['sharpe'], 2, True), f(c['p1']['max_dd'] * 100, 0), f(c['p1']['mese_peggiore'] * 100, 0, True), f(c['t1']['vincenti'] * 100, 0),
                                f(o['op']['asimmetria'], 2, True), f(o['op']['curtosi_ecc'], 1), f(o['mese']['asimmetria'], 2, True), f(o['mese']['curtosi_ecc'], 1)]) + r' \\')
    return '\n'.join(rows)
