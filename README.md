# Fat Tail

Web app da installare sul telefono. Ogni mattina mostra le chiusure del giorno prima di 14 indici azionari e commodities, la loro volatilità, le code delle distribuzioni e cosa dice oggi la regola di trading seguita.

https://giordanomargotti.github.io/trading/

## Le tre pagine

- **Oggi**: la regola seguita, con la soglia d'ingresso per la prossima chiusura o, se una posizione è segnata come aperta, il livello d'uscita; dimensione della posizione rispetto al capitale e al lotto Fineco; prossima chiusura d'ufficio del CFD; diario delle operazioni chiuse. Sotto, i mercati con una seduta oltre 2σ.
- **Mercati**: per ogni mercato quattro numeri chiave e sei riquadri da aprire:
  - chiusura di domani: distribuzione prevista, intervalli al 90% e al 98%;
  - volatilità: realizzata a 21 sedute, EWMA (λ 0,94), GJR-GARCH(1,1) con t di Student, implicita (VIX, VXN, OVX, GVZ);
  - grappoli di volatilità: rendimenti con la fascia di ±2σ prevista, autocorrelazione dell'ampiezza e della direzione;
  - curva e scadenze: prossimi contratti, contango o backwardation, roll, rinnovo del CFD (petrolio, gas, metalli);
  - distribuzione e code: istogramma con gaussiana e t, sedute oltre 2-5σ, indice di coda α di Hill, VaR ed ES;
  - prezzo dell'ultimo anno.
- **Strategia**: contrarian, trend following (su shock in σ o su rottura) o combinata secondo la fase di mercato, con stop, trailing, target, durata, spread, finanziamento e filtro sul verso della curva. Risultati storici in R, per fase di mercato e per volatilità all'ingresso, 200 operazioni simulate; un pulsante porta la regola nella pagina Oggi.

Il nome in alto apre una scheda su convessità e code grasse. Ogni riquadro ha un «Come si legge». Le scelte restano salvate sul telefono.

Un workflow di GitHub Actions scarica i dati da Yahoo Finance (e dall'EIA lo storico dei contratti NYMEX) dal martedì al sabato alle 05:17 UTC, ricalcola tutto e pubblica il sito su GitHub Pages. È tutto gratuito con un repository pubblico.

## Attivazione

1. Il repository deve essere **pubblico**: con l'account gratuito GitHub Pages non pubblica i repository privati.
2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
3. **Actions → Aggiorna dati → Run workflow.** Il primo giro dura qualche minuto.
4. Aprire https://giordanomargotti.github.io/trading/

Da lì in poi si aggiorna da solo. Se un giorno il download fallisce per tutti i mercati, GitHub manda una email e l'app continua a mostrare i dati del giorno prima.

## Installazione sul telefono

- **Android (Chrome):** aprire l'indirizzo, menu ⋮ → *Installa app* oppure *Aggiungi a schermata Home*.
- **iPhone (Safari):** aprire l'indirizzo, tasto Condividi → *Aggiungi alla schermata Home*.

L'app si apre anche senza rete, con gli ultimi dati scaricati.

## Aggiungere o togliere mercati

Modificare la lista `ASSET` in cima a `scripts/aggiorna_dati.py`: una riga per mercato, con il simbolo di Yahoo Finance. Al giro successivo compare nell'app.

## Prova in locale

```bash
pip install -r requirements.txt
python tests/test_calcoli.py                                  # controlli dei calcoli, senza rete
node tests/test_motore.js                                     # controlli del motore delle simulazioni
python scripts/aggiorna_dati.py --out docs/data.json --demo   # dati simulati
python -m http.server -d docs 8000                            # poi aprire http://localhost:8000
```

Senza `--demo` lo script scarica i dati veri da Yahoo. I dati simulati sono segnalati nell'app con un avviso; il primo aggiornamento vero li sostituisce.

## Struttura

| File | Cosa fa |
|---|---|
| `scripts/aggiorna_dati.py` | download, pulizia, volatilità, statistiche, curve, scrittura di `docs/data.json` |
| `scripts/serie_rollate.py` | calendario delle scadenze e serie rinnovate di gas e WTI (EIA fino al 5 aprile 2024, poi Yahoo) |
| `scripts/ricerca.js` | studio fuori campione delle regole, fuori dall'app, da eseguire a mano; dati dello studio del 10/11/2026 nel ramo `dati-ricerca` |
| `tests/test_calcoli.py` | controlli su dati con proprietà note (GARCH, Hill, code, EWMA, pulizia, scadenze, rinnovo) |
| `tests/test_motore.js` | controlli del motore delle simulazioni |
| `docs/index.html` | l'app: legge `data.json` e disegna tutto |
| `docs/motore.js` | motore delle simulazioni, lo stesso per l'app e per lo studio |
| `docs/curve.json` | curve dei futures salvate ogni giorno |
| `docs/storia/*.json` | storico completo di ogni mercato: rigenerato a ogni giro e pubblicato solo su Pages, non salvato nel repository |
| `docs/sw.js`, `docs/manifest.webmanifest` | installazione e funzionamento offline |
| `docs/vendor/chart.umd.min.js` | Chart.js 4.6.1 (licenza MIT) |
| `.github/workflows/aggiorna.yml` | aggiornamento giornaliero e pubblicazione |

## Limiti

- Brent, TTF e metalli usano la serie del future più vicino senza rettifica per il cambio di contratto: nei giorni di roll compaiono salti che non sono rendimenti veri. Gas Henry Hub e WTI sono rinnovati, ma dopo il 5 aprile 2024 (fine dei dati EIA) il rendimento del giorno del cambio è messo a zero quando il contratto non è più quotato su Yahoo.
- Yahoo Finance non ha un'API ufficiale: a volte limita le richieste o cambia formato. Lo script ritenta, e in caso di errore tiene i dati del giorno prima.
- OVX e GVZ sono calcolati sulle opzioni degli ETF USO e GLD, non dei futures.
- È uno strumento di studio, non un consiglio di investimento.
