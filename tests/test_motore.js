/* Controlli del motore delle simulazioni su una serie sintetica con proprietà note.
   Uso: node tests/test_motore.js */
"use strict";
const M = require("../docs/motore.js");
let errori = 0;
function controlla(nome, ok, dettaglio = "") {
  console.log(`${ok ? "ok  " : "ERR "} ${nome}${dettaglio ? " — " + dettaglio : ""}`);
  if (!ok) errori++;
}

/* passeggiata casuale con volatilità che cambia nel tempo e qualche seduta tripla; σ EWMA e z come
   li scrive aggiorna_dati.py: ze[t] = rendimento di t diviso per la σ prevista la sera prima,
   se[t] = σ prevista per t+1, in % */
function serie(n, seme) {
  const rng = M.mulberry32(seme), gauss = () => Math.sqrt(-2 * Math.log(rng() || 1e-12)) * Math.cos(2 * Math.PI * rng());
  const c = [100], date = [], ze = [null], se = [2];
  let v = 4, giorno = Date.UTC(2010, 0, 4);
  for (let t = 0; t < n; t++) {
    while ([0, 6].includes(new Date(giorno).getUTCDay())) giorno += 864e5;
    date.push(new Date(giorno).toISOString().slice(0, 10)); giorno += 864e5;
    if (t === 0) continue;
    const vera = 1.5 * (1 + 0.5 * Math.sin(2 * Math.PI * t / 500));
    const r = vera * gauss() * (rng() < 0.03 ? 3 : 1);                  /* in % */
    c.push(c[t - 1] * Math.exp(r / 100));
    ze.push(r / Math.sqrt(v));
    v = 0.94 * v + 0.06 * r * r;
    se.push(Math.sqrt(v));
  }
  return {date, c, ze, se, roll: Array.from({length: Math.floor(n / 21)}, (_, k) => 20 + 21 * k)};
}

const st = serie(4000, 7);
const base = {sigma: "ewma", periodo: "tutto", costo: 0.3, finanz: 5, finanzShort: 1, N: 60, indicatore: "forza", fase: "tutte", dirTrend: "tutte", Nr: 55};
const regole = {
  "trend con trailing": {strategia: "trend", ingresso: "shock", lato: "entrambi", k: 1.5, S: 2, T: 3, G: 0, H: 0},
  "contrarian con target e durata": {strategia: "contrarian", ingresso: "shock", lato: "long", k: 1.5, S: 4, T: 0, G: 1, H: 10},
  "rottura": {strategia: "trend", ingresso: "rottura", Nr: 20, lato: "short", k: 2, S: 2, T: 4, G: 0, H: 0},
};

/* 1. Ingresso forzato alla stessa chiusura e nello stesso verso di un'operazione della regola:
      deve dare la stessa uscita e lo stesso risultato. */
for (const [nome, r] of Object.entries(regole)) {
  const p = {...base, ...r}, ris = M.simulaStrategia(st, p);
  let uguali = 0;
  for (const o of ris.ops) {
    const f = M.simulaStrategia(st, {...p, ingressoForzato: {i: o.ie, lato: o.lato}}).ops;
    if (f.length === 1 && f[0].iu === o.iu && f[0].motivo === o.motivo && Math.abs(f[0].R - o.R) < 1e-12 && f[0].rinnovi === o.rinnovi) uguali++;
  }
  controlla(`Ingresso forzato, ${nome}: stesse uscite della regola`, ris.ops.length > 30 && uguali === ris.ops.length, `${uguali} su ${ris.ops.length}`);
}

/* 2. Ingresso forzato all'ultima chiusura: posizione aperta da zero sedute, costa lo spread,
      stop a S σ dal prezzo d'ingresso. */
{
  const p = {...base, ...regole["trend con trailing"]}, n = st.c.length, s = st.se[n - 1] / 100;
  const f = M.simulaStrategia(st, {...p, ingressoForzato: {i: n - 1, lato: -1}}).ops;
  const ok = f.length === 1 && f[0].motivo === "aperta" && f[0].durata === 0 && f[0].rinnovi === 0 &&
    Math.abs(f[0].R + 0.003 / (2 * s)) < 1e-12 && Math.abs(f[0].livello - st.c[n - 1] * Math.exp(2 * s)) < 1e-9;
  controlla("Ingresso forzato all'ultima chiusura: aperta, stop a 2σ sopra per lo short", ok, f.length ? `R ${f[0].R.toFixed(4)}, livello ${f[0].livello.toFixed(3)}` : "nessuna");
}

/* 3. Ingresso forzato in mezzo alla storia, uscita verificata a mano: prima chiusura oltre lo stop
      iniziale o oltre il trailing dal punto migliore. */
{
  const p = {...base, ...regole["trend con trailing"]}, i = 1500, lato = 1, s = st.se[i] / 100;
  const f = M.simulaStrategia(st, {...p, ingressoForzato: {i, lato}}).ops[0];
  let best = 0, atteso = null;
  for (let t = i + 1; t < st.c.length; t++) {
    const x = lato * Math.log(st.c[t] / st.c[i]);
    if (x <= Math.max(-p.S * s, best - p.T * s)) { atteso = t; break; }
    if (x > best) best = x;
  }
  controlla("Ingresso forzato: uscita alla prima chiusura oltre stop o trailing", f && f.iu === atteso, `uscita ${f && st.date[f.iu]}, attesa ${st.date[atteso]}`);
}

/* 4. Segnale dell'ultima chiusura per chi è senza posizione */
{
  const n = st.c.length, copia = {...st, ze: st.ze.slice()};
  copia.ze[n - 1] = -3;
  const trend = {...base, ...regole["trend con trailing"]};
  controlla("Segnale all'ultima chiusura: trend, seduta a −3σ → short", M.simulaStrategia(copia, trend).segnaleUltima === -1);
  controlla("Segnale all'ultima chiusura: solo long → nessuno", M.simulaStrategia(copia, {...trend, lato: "long"}).segnaleUltima === 0);
  controlla("Segnale all'ultima chiusura: contrarian → long", M.simulaStrategia(copia, {...trend, strategia: "contrarian"}).segnaleUltima === 1);
  copia.ze[n - 1] = 0.5;
  controlla("Segnale all'ultima chiusura: seduta normale → nessuno", M.simulaStrategia(copia, trend).segnaleUltima === 0);
  /* la soglia per la prossima seduta è il prezzo a k σ dall'ultima chiusura */
  const z = M.simulaStrategia(copia, {...trend, lato: "short"}).daZero, s = st.se[n - 1] / 100;
  controlla("Soglia per la prossima seduta: short sotto k σ", z.short && !z.short.sopra && Math.abs(z.short.prezzo - st.c[n - 1] * Math.exp(-1.5 * s)) < 1e-9 && !z.long);
}

console.log(errori ? `\n${errori} controlli falliti` : "\nTutto ok");
process.exit(errori ? 1 : 0);
