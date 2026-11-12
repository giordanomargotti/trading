"""Assembla il sito autonomo: modello HTML + dati dei risultati + Chart.js in linea."""
import sys
tpl, data, chartjs, dst = sys.argv[1:5]
s = open(tpl, encoding='utf-8').read()
d = open(data, encoding='utf-8').read().replace('</', '<\\/')
c = open(chartjs, encoding='utf-8').read().replace('</script', '<\\/script')
assert '/*DATA*/' in s and '/*CHARTJS*/' in s
s = s.replace('/*CHARTJS*/', c).replace('/*DATA*/', d)
open(dst, 'w', encoding='utf-8').write(s)
print(dst, round(len(s) / 1e6, 2), 'MB')
