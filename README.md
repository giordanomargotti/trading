# Mercati e code

Web app da installare sul telefono. Ogni mattina mostra le chiusure del giorno prima di indici azionari e commodities, con:

- **volatilità** con tre metodi: realizzata a 21 sedute, EWMA (λ 0,94) e GJR-GARCH(1,1) con t di Student;
- **distribuzione vera dei rendimenti**: istogramma con gaussiana e t di Student sovrapposte, in scala lineare o logaritmica, grezza o divisa per la σ del GARCH;
- **code**: quanto spesso il prezzo ha superato 2, 3, 4 e 5σ, confrontato con la gaussiana e con la t; indice di coda α di Hill; peso del giorno più estremo sulla curtosi;
- **domani**: distribuzione prevista della prossima chiusura, con gli intervalli al 90% e al 98%;
- **movimenti in σ** dell'ultimo anno e l'elenco delle sedute oltre ±2σ;
- **volatilità implicita** (VIX, VXN, OVX, GVZ) confrontata con la previsione GARCH a 21 sedute;
- **curva dei futures** per petrolio, gas e metalli: prossimi contratti mensili, contango o backwardation, roll annualizzato, confronto con un mese prima;
- **serie rinnovate** per gas Henry Hub e WTI: il rendimento di chi tiene un CFD Fineco e lo riapre a ogni scadenza, senza il salto della serie continua nel giorno del cambio di contratto;
- **laboratorio di strategie**: contrarian, trend following (su shock in σ o su rottura) o combinata secondo la fase di mercato, con stop, trailing, target, durata massima, spread, finanziamento e filtro sul verso della curva. Mostra la distribuzione dei risultati in R, le fasi di mercato, un Monte Carlo di 200 operazioni future e cosa fare oggi con le regole scelte, compresa la dimensione della posizione;
- **ricerca**: una griglia di 512 regole scelte sulla prima metà della storia di gas e WTI e verificate sulla seconda metà, su Brent e su TTF.

Un workflow di GitHub Actions scarica i dati da Yahoo Finance (e dall'EIA lo storico dei contratti NYMEX) dal martedì al sabato alle 05:17 UTC, ricalcola tutto e pubblica il sito su GitHub Pages. La ricerca si avvia a mano: **Actions → Ricerca strategie → Run workflow**. È tutto gratuito con un repository pubblico.

## Attivazione

1. Il repository deve essere **pubblico**: con l'account gratuito GitHub Pages non pubblica i repository privati.
2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
3. **Actions → Aggiorna dati → Run workflow.** Il primo giro dura qualche minuto.
4. Apri https://giordanomargotti.github.io/trading/

Da lì in poi si aggiorna da solo. Se un giorno il download fallisce per tutti i mercati, GitHub ti manda una email e l'app continua a mostrare i dati del giorno prima.

## Installazione sul telefono

- **Android (Chrome):** apri l'indirizzo, menu ⋮ → *Installa app* oppure *Aggiungi a schermata Home*.
- **iPhone (Safari):** apri l'indirizzo, tasto Condividi → *Aggiungi alla schermata Home*.

L'app si apre anche senza rete, con gli ultimi dati scaricati.

## Aggiungere o togliere mercati

Modifica la lista `ASSET` in cima a `scripts/aggiorna_dati.py`: una riga per mercato, con il simbolo di Yahoo Finance. Al giro successivo compare nell'app.

## Prova in locale

```bash
pip install -r requirements.txt
python tests/test_calcoli.py                                  # controlli dei calcoli, senza rete
python scripts/aggiorna_dati.py --out docs/data.json --demo   # dati simulati
python -m http.server -d docs 8000                            # poi apri http://localhost:8000
```

Senza `--demo` lo script scarica i dati veri da Yahoo. I dati simulati sono segnalati nell'app con un avviso; il primo aggiornamento vero li sostituisce.

## Struttura

| File | Cosa fa |
|---|---|
| `scripts/aggiorna_dati.py` | download, pulizia, volatilità, statistiche, curve, scrittura di `docs/data.json` |
| `scripts/serie_rollate.py` | calendario delle scadenze e serie rinnovate di gas e WTI (EIA fino al 5 aprile 2024, poi Yahoo) |
| `scripts/ricerca.js` | ricerca con protocollo fuori campione, scrive `docs/ricerca.json` |
| `tests/test_calcoli.py` | controlli su dati con proprietà note (GARCH, Hill, code, EWMA, pulizia, scadenze, rinnovo) |
| `docs/index.html` | l'app: legge `data.json` e disegna tutto, compreso il laboratorio |
| `docs/motore.js` | motore delle simulazioni, lo stesso per l'app e per la ricerca |
| `docs/curve.json` | curve dei futures salvate ogni giorno |
| `docs/storia/*.json` | storico completo per il laboratorio, uno per mercato: rigenerato a ogni giro e pubblicato solo su Pages, non salvato nel repository |
| `docs/sw.js`, `docs/manifest.webmanifest` | installazione e funzionamento offline |
| `docs/vendor/chart.umd.min.js` | Chart.js 4.6.1 (licenza MIT) |
| `.github/workflows/aggiorna.yml` | aggiornamento giornaliero e pubblicazione |
| `.github/workflows/ricerca.yml` | ricerca, da avviare a mano |

## Limiti

- Brent, TTF e metalli usano la serie del future più vicino senza rettifica per il cambio di contratto: nei giorni di roll compaiono salti che non sono rendimenti veri. Gas Henry Hub e WTI sono rinnovati, ma dopo il 5 aprile 2024 (fine dei dati EIA) il rendimento del giorno del cambio è messo a zero quando il contratto non è più quotato su Yahoo.
- Yahoo Finance non ha un'API ufficiale: a volte limita le richieste o cambia formato. Lo script ritenta, e in caso di errore tiene i dati del giorno prima.
- OVX e GVZ sono calcolati sulle opzioni degli ETF USO e GLD, non dei futures.
- È uno strumento di studio, non un consiglio di investimento.
