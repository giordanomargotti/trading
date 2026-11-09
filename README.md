# Mercati e code

Web app da installare sul telefono. Ogni mattina mostra le chiusure del giorno prima di indici azionari e commodities, con:

- **volatilità** con tre metodi: realizzata a 21 sedute, EWMA (λ 0,94) e GJR-GARCH(1,1) con t di Student;
- **distribuzione vera dei rendimenti**: istogramma con gaussiana e t di Student sovrapposte, in scala lineare o logaritmica, grezza o divisa per la σ del GARCH;
- **code**: quanto spesso il prezzo ha superato 2, 3, 4 e 5σ, confrontato con la gaussiana e con la t; indice di coda α di Hill; peso del giorno più estremo sulla curtosi;
- **domani**: distribuzione prevista della prossima chiusura, con gli intervalli al 90% e al 98%;
- **movimenti in σ** dell'ultimo anno e l'elenco delle sedute oltre ±2σ;
- **volatilità implicita** (VIX, VXN, OVX, GVZ) confrontata con la previsione GARCH a 21 sedute.

Un workflow di GitHub Actions scarica i dati da Yahoo Finance dal martedì al sabato alle 05:17 UTC, ricalcola tutto e pubblica il sito su GitHub Pages. È tutto gratuito con un repository pubblico.

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
| `scripts/aggiorna_dati.py` | download, pulizia, volatilità, statistiche, scrittura di `docs/data.json` |
| `tests/test_calcoli.py` | controlli su dati con proprietà note (GARCH, Hill, code, EWMA, pulizia) |
| `docs/index.html` | l'app: legge `data.json` e disegna tutto |
| `docs/sw.js`, `docs/manifest.webmanifest` | installazione e funzionamento offline |
| `docs/vendor/chart.umd.min.js` | Chart.js 4.6.1 (licenza MIT) |
| `.github/workflows/aggiorna.yml` | aggiornamento giornaliero e pubblicazione |

## Limiti

- Le commodity usano la serie del future più vicino senza rettifica per il cambio di contratto: nei giorni di roll compaiono salti che non sono rendimenti veri, soprattutto sul gas. Alzano le code osservate.
- Yahoo Finance non ha un'API ufficiale: a volte limita le richieste o cambia formato. Lo script ritenta, e in caso di errore tiene i dati del giorno prima.
- OVX e GVZ sono calcolati sulle opzioni degli ETF USO e GLD, non dei futures.
- È uno strumento di studio, non un consiglio di investimento.
