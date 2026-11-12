"""Numeri e tabelle per i documenti LaTeX, nelle due lingue."""
import json, numpy as np
def load(out):
    return json.load(open(out + '/sito.json'))
def fmt(x, d, lang, sign=False):
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    s = s.replace('-', '$-$')
    return s.replace('.', '{,}') if lang == 'it' else s
def integer(n, lang):
    s = f'{int(n):,}'
    return s.replace(',', '.') if lang == 'it' else s
FAM = {'it': {'SP': 'Put venduta', 'STRG': 'Strangle venduto', 'STRD': 'Straddle venduto ATM', 'PCS': 'Put credit spread', 'CCS': 'Call credit spread', 'IC': 'Iron condor',
              'IFLY': 'Iron butterfly', 'LP': 'Put comprata', 'DPS': 'Debit put spread', 'CAL': 'Calendar lungo ATM'},
       'en': {'SP': 'Short put', 'STRG': 'Short strangle', 'STRD': 'Short ATM straddle', 'PCS': 'Put credit spread', 'CCS': 'Call credit spread', 'IC': 'Iron condor',
              'IFLY': 'Iron butterfly', 'LP': 'Long put', 'DPS': 'Debit put spread', 'CAL': 'Long ATM calendar'}}
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
    return K
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
    wl = {'it': {'d05': r', ala $\Delta$ 0{,}05', 'w1': r', ala 1\% dello spot', '-': ''}, 'en': {'d05': r', $\Delta$ 0.05 wing', 'w1': r', 1\% of spot wing', '-': ''}}[lang]
    for fam, ala in order:
        x = [c for c in S['config'] if c['fam'] == fam and c['ala'] == ala and c['gest'] == 'hold']
        med = lambda g: float(np.median([g(c) for c in x]))
        rows.append(' & '.join([FAM[lang][fam] + wl[ala], str(len(x)), f(med(lambda c: c['p0']['sharpe']), 2, True), f(med(lambda c: c['p1']['sharpe']), 2, True),
                                f(med(lambda c: c['capm1']['beta']), 2, True), f(med(lambda c: c['capm0']['t_alfa']), 2, True), f(med(lambda c: c['capm1']['t_alfa']), 2, True),
                                f(med(lambda c: c['spread_su_credito']) * 100, 0), f(med(lambda c: c['t1']['vincenti']) * 100, 0)]) + r' \\')
    return '\n'.join(rows)
def tab_selezione(S, lang):
    f = lambda x, d=2, s=False: fmt(x, d, lang, s)
    by = {c['key']: c for c in S['config']}
    gg = 'gg' if lang == 'it' else 'd'
    sel = [('SP|0.10|-|7|hold', r'$\Delta$0.10, 7'), ('SP|0.16|-|45|hold', r'$\Delta$0.16, 45'), ('STRG|0.16|-|30|hold', r'$\Delta$0.16, 30'), ('PCS|0.20|d05|30|hold', r'$\Delta$0.20/0.05, 30'),
           ('PCS|0.20|w1|30|hold', r'$\Delta$0.20, 1\%, 30'), ('PCS|0.20|d05|30|tp50', r'$\Delta$0.20/0.05, 30, TP50'), ('PCS|0.20|d05|30|tp50sl200', r'$\Delta$0.20/0.05, 30, TP50/SL200'),
           ('STRD|0.50|-|30|hold', 'ATM, 30'), ('IC|0.16|d05|30|hold', r'$\Delta$0.16/0.05, 30'), ('IC|0.16|w1|30|hold', r'$\Delta$0.16, 1\%, 30'), ('IFLY|0.10|-|30|hold', r'ATM/$\Delta$0.10, 30'),
           ('CCS|0.16|d05|30|hold', r'$\Delta$0.16/0.05, 30'), ('CAL|0.50|-|30|hold', 'ATM, 30/60'), ('LP|0.10|-|30|hold', r'$\Delta$0.10, 30')]
    rows = []
    for key, lab in sel:
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
