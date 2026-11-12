"""Compila i quattro documenti LaTeX (paper e sintesi, italiano e inglese) a partire dai modelli e dai risultati."""
import sys, os, re, shutil, subprocess, glob
sys.path.insert(0, os.path.dirname(__file__))
import paper_num as P
out, tpl, dst = sys.argv[1], sys.argv[2], sys.argv[3]
S = P.load(out)
os.makedirs(dst + '/figure', exist_ok=True)
for f in glob.glob(out + '/fig/*.pdf'): shutil.copy(f, dst + '/figure/')
for lang in ('it', 'en'):
    K = P.numbers(S, lang)
    K.update(TAB_ORIZ=P.tab_orizzonti(S, lang), TAB_MZ=P.tab_mz(S, lang), TAB_ANNI=P.tab_anni(S, lang), TAB_FAM=P.tab_famiglie(S, lang),
             TAB_SEL=P.tab_selezione(S, lang), TAB_INS=P.tab_ins(S, lang), TAB_FILT=P.tab_filtri(S, lang), TAB_FVOL=P.tab_fvol(S, lang), TAB_HV=P.tab_hv(S, lang))
    for src, name in ((f'tpl_{lang}.tex', f'paper_{lang}.tex'), (f'tpl_abs_{lang}.tex', f'abstract_{lang}.tex')):
        s = open(f'{tpl}/{src}', encoding='utf-8').read()
        miss = set()
        def rep(m):
            k = m.group(1)
            if k not in K: miss.add(k); return '??'
            return K[k]
        s = re.sub(r'@@([A-Za-z0-9_]+)@@', rep, s)
        if miss: print('MANCANTI', name, sorted(miss))
        open(f'{dst}/{name}', 'w', encoding='utf-8').write(s)
        for _ in range(2):
            r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name], cwd=dst, capture_output=True, text=True)
        log = open(f'{dst}/{name[:-4]}.log', errors='replace').read()
        pages = re.search(r'Output written on .*\((\d+) pages?', log)
        print(name, 'pagine', pages.group(1) if pages else 'ERRORE', '| overfull', log.count('Overfull \\hbox'), '| undefined', log.count('undefined'))
        if not pages: print(r.stdout[-1500:])
for ext in ('aux', 'log', 'out'):
    for f in glob.glob(f'{dst}/*.{ext}'): os.remove(f)
