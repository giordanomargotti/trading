# Volatility risk premium e strategie in opzioni su S&P 500

Risultati della ricerca sulle option chain end-of-day delle opzioni SPX dal 2010 al 2023 (fonte: [OptionsDX](https://www.optionsdx.com/), dataset SPX Option Chains; opzioni su indice S&P 500 di stile europeo, cash-settled, quotate al Cboe): esistenza del volatility risk premium, backtest di dieci famiglie di strategie non direzionali, rassegna della letteratura.

## Contenuto

| Percorso | Contenuto |
|---|---|
| `index.html` | sito interattivo autonomo: si apre con un doppio clic, senza server e senza rete |
| `paper/paper_it.pdf`, `paper/paper_en.pdf` | documento completo in italiano e in inglese (sorgenti `.tex` accanto) |
| `paper/abstract_it.pdf`, `paper/abstract_en.pdf` | sintesi di due pagine con grafici (sorgenti `.tex` accanto) |
| `paper/figure/` | figure in PDF, nelle due lingue |
| `dati/` | risultati: `vrp.json`, `sito.json` (configurazioni e serie mensili), `distribuzioni.json` (forma della distribuzione dei rendimenti), parametri GARCH, controlli di qualità |
| `codice/` | pipeline Python completa e modelli dei documenti |

## Risultati principali

- Volatilità implicita model-free a 30 giorni meno realizzata: +3,45 punti (t 6,48). Volatilità at-the-money meno realizzata: +0,72 punti (t 1,36). Il premio è nelle wing.
- Premio at-the-money significativo nel 2010–2014, nullo nel 2015–2019 e nel 2020–2023.
- 434 configurazioni, 1.310.509 operazioni simulate. Sharpe mediano +0,39 ai prezzi mid e −0,40 ai prezzi bid-ask.
- Configurazioni con alfa positivo significativo rispetto all'indice: 16,4% ai prezzi mid, 2,5% ai prezzi bid-ask.
- Filtro di sospensione sulla volatilità realizzata (HV21 oltre il 25%): Sharpe mediano da −0,30 a −0,34, mese peggiore invariato. Gli strike scelti per delta si allargano già con la volatilità implicita.
- Distribuzione dei rendimenti: skewness per operazione negativa nel 96% delle configurazioni, normalità della serie mensile rifiutata (Jarque-Bera, 1%) nel 92%. Short put delta 0,16 a 45 giorni: skewness −9,5 ed excess kurtosis 108 per operazione, contro −1,1 e 4,8 dell'indice sullo stesso orizzonte.
- Configurazioni a rendimento cumulato più alto per famiglia (ladder, prezzi bid-ask): put credit spread Δ0,30/0,05 a 7 giorni +798% (Sharpe 0,43, mese peggiore −147% del capitale impegnato), short put Δ0,30 a 30 giorni +298% (Sharpe 0,85), iron condor Δ0,30/0,05 a 45 giorni con take profit 50% +105%, long put Δ0,30 a 45 giorni −4.409%. Nessuna ha distribuzione normale.

## Riproduzione

Dati di partenza: archivi `spx_eod_*.7z` del repository `finance-thinking/options_strategy`, branch `strategies_backtest`, cartella `data/option_chain_data/SPX_OPTION_DATA_ZIP`.

```bash
pip install numpy pandas scipy pyarrow py7zr arch matplotlib
S=codice; mkdir -p pq raw out/fig
python $S/converti.py <cartella degli archivi 7z> pq raw      # 7z -> Parquet mensili
python $S/prepara.py pq out                                   # catena compatta, serie del sottostante
for y in $(seq 2010 2023); do python $S/arricchisci.py out $S $y; done   # forward, IV, delta
python $S/pannello.py out                                     # IV ATM, varianza model-free, varianza realizzata
python $S/garch.py out                                        # GJR-GARCH-t senza informazioni future
python $S/vrp.py out $S                                       # statistiche sul premio
python $S/bt.py out && python $S/cal.py out                   # backtest
python $S/stat_bt.py out $S && python $S/filtro_vol.py out $S # sintesi, filtro sulla volatilità realizzata
python $S/distribuzioni.py out $S                             # distribuzione dei rendimenti e code
python $S/sintesi.py out                                      # dati del sito, figure
python $S/costruisci_paper.py out $S/modelli build            # quattro documenti LaTeX e PDF
python $S/costruisci_sito.py $S/modelli/tpl.html out/sito.json <chart.umd.min.js> index.html
```

Tempo complessivo: circa mezz'ora; memoria necessaria circa 6 GB.

## Autori

Giordano Margotti, Alex Crescentini, Alessandro Celani, Giulio Sciacchitano (gruppo di lavoro finance-thinking).

## Fonti non incluse

La option chain XSP (IVolatility, 2014–2026) del repository `AlexCrescentini/backtest_put_credit_spread` è archiviata con Git LFS e non è inclusa nell'archivio ZIP esaminato. Il materiale della cartella Google Drive del progetto (script, snapshot giornalieri, documenti) non è integrato in questa versione.

Documento di ricerca. Non costituisce consulenza in materia di investimenti.
