/* Motore del laboratorio di strategie.
   Lo usano sia l'app (index.html) sia la ricerca (scripts/ricerca.js), così i numeri coincidono.
   Ogni decisione usa solo dati noti alla chiusura del giorno: si entra e si esce alle chiusure. */
"use strict";

const SOGLIA_LATERALE = 0.5, SOGLIA_TREND = 1, Q_VR = 5, SEDUTE_ANNO = 252;
const cacheSoglieVR = {};

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const quantile = (ordinati, p) => ordinati[Math.min(ordinati.length - 1, Math.max(0, Math.round(p * (ordinati.length - 1))))];

/* Forza del trend: rendimento delle ultime N sedute diviso per quello che farebbe in media una
   passeggiata casuale con la stessa volatilità (σ stimata dalla media dei |rendimenti|).
   Su una passeggiata casuale |F| è circa |N(0,1)|: un terzo delle sedute sotto 0,5, un terzo
   tra 0,5 e 1, un terzo sopra 1. */
function forzaTrend(st, N) {
  if (st._forza && st._forza.N === N) return st._forza.F;
  const n = st.c.length, F = new Array(n).fill(null), r = new Float64Array(n);
  for (let t = 1; t < n; t++) r[t] = Math.log(st.c[t] / st.c[t - 1]);
  let s = 0, a = 0;
  for (let t = 1; t < n; t++) {
    s += r[t]; a += Math.abs(r[t]);
    if (t > N) { s -= r[t - N]; a -= Math.abs(r[t - N]); }
    if (t >= N && a > 0) F[t] = s / (a / N * Math.sqrt(Math.PI / 2) * Math.sqrt(N));
  }
  st._forza = {N, F};
  return F;
}
const faseDi = f => f == null ? null : Math.abs(f) < SOGLIA_LATERALE ? "laterale" : Math.abs(f) < SOGLIA_TREND ? "intermedia" : "trend";

/* Autocorrelazione con il variance ratio di Lo e MacKinlay: varianza dei rendimenti a Q sedute
   divisa per Q volte la varianza giornaliera, sulle ultime W sedute. Le soglie sono i terzili
   ottenuti su una passeggiata casuale gaussiana con la stessa finestra: fisse, non scelte
   guardando il mercato. */
function serieVR(P, W) {
  const n = P.length, V = new Array(n).fill(null), q = Q_VR;
  for (let t = W; t < n; t++) {
    const mu = (P[t] - P[t - W]) / W;
    let v1 = 0, vq = 0, m = 0;
    for (let i = t - W + 1; i <= t; i++) { const d = P[i] - P[i - 1] - mu; v1 += d * d; }
    for (let j = t - W + q; j <= t; j++) { const d = P[j] - P[j - q] - q * mu; vq += d * d; m++; }
    v1 /= W; vq /= m;
    V[t] = v1 > 0 ? vq / (q * v1) : null;
  }
  return V;
}
function soglieVR(W) {
  if (cacheSoglieVR[W]) return cacheSoglieVR[W];
  const rng = mulberry32(99 + W), n = 30000, P = new Float64Array(n);
  for (let t = 1; t < n; t++) { let u = 0; while (!u) u = rng(); P[t] = P[t - 1] + Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); }
  const V = serieVR(P, W).filter(v => v != null).sort((a, b) => a - b);
  return (cacheSoglieVR[W] = [quantile(V, 1 / 3), quantile(V, 2 / 3)]);
}
function fasiSerie(st, p) {
  const chiave = p.indicatore + ":" + p.N;
  if (st._fasi && st._fasi.chiave === chiave) return st._fasi.fasi;
  let fasi;
  if (p.indicatore === "autocorrelazione") {
    const n = st.c.length, P = new Float64Array(n);
    for (let t = 1; t < n; t++) P[t] = P[t - 1] + Math.log(st.c[t] / st.c[t - 1]);
    const V = serieVR(P, p.N), [s1, s2] = soglieVR(p.N);
    fasi = V.map(v => v == null ? null : v < s1 ? "laterale" : v > s2 ? "trend" : "intermedia");
  } else {
    fasi = forzaTrend(st, p.N).map(faseDi);
  }
  st._fasi = {chiave, fasi};
  return fasi;
}

/* Massimo e minimo delle N chiusure precedenti, esclusa quella del giorno: servono all'ingresso
   a rottura. La chiusura di t rompe il massimo se è più alta di tutte le N precedenti. */
function estremi(st, N) {
  if (st._estremi && st._estremi.N === N) return st._estremi;
  const n = st.c.length, max = new Array(n).fill(null), min = new Array(n).fill(null);
  for (let t = N; t < n; t++) {
    let a = -Infinity, b = Infinity;
    for (let i = t - N; i < t; i++) { if (st.c[i] > a) a = st.c[i]; if (st.c[i] < b) b = st.c[i]; }
    max[t] = a; min[t] = b;
  }
  st._estremi = {N, max, min};
  return st._estremi;
}

/* Chiusure in cui il CFD scade (st.roll, solo per le serie rinnovate): chi è dentro viene chiuso
   d'ufficio e per restare in posizione paga di nuovo lo spread. cum[t] = scadenze fino a t compresa. */
function rollCumulati(st) {
  if (st._rollCum) return st._rollCum;
  const n = st.c.length, cum = new Int32Array(n), giorni = new Set(st.roll || []);
  let k = 0;
  for (let t = 0; t < n; t++) { if (giorni.has(t)) k++; cum[t] = k; }
  return (st._rollCum = cum);
}

/* Verso pagato dalla curva il giorno t: in contango (secondo contratto più caro del primo) il roll
   paga chi è short, in backwardation chi è long. 0 se lo spread non c'è. */
const versoCurva = (st, t) => !st.sp || st.sp[t] == null || st.sp[t] === 0 ? 0 : st.sp[t] > 0 ? -1 : 1;

/* Parametri p:
   strategia  contrarian | trend | combinata (contrarian nella prima fase, trend nella terza)
   ingresso   shock (seduta oltre k σ) | rottura (chiusura oltre il massimo o il minimo di Nr sedute)
   lato       long | short | entrambi | curva (solo nel verso pagato dal roll)
   k, Nr      soglia in σ e finestra della rottura
   S, T, G, H stop iniziale, trailing e target in σ; durata massima in sedute (0 = spento)
   costo      spread di andata e ritorno, in % del controvalore; si paga di nuovo a ogni scadenza del CFD
   finanz, finanzShort   finanziamento overnight di long e short, in % annuo del controvalore
              (negativo = accredito)
   sigma      garch | ewma; periodo tutto | prima | seconda
   N, indicatore, fase, dirTrend   misura e filtri delle fasi di mercato */
function simulaStrategia(st, p) {
  const usaG = p.sigma === "garch" && st.zg;
  const z = usaG ? st.zg : st.ze, sg = usaG ? st.sg : st.se, n = st.c.length;
  let i0 = 0, i1 = n - 1;
  if (p.periodo && p.periodo !== "tutto") {
    const meta = (Date.parse(st.date[0]) + Date.parse(st.date[n - 1])) / 2;
    const k = st.date.findIndex(d => Date.parse(d) >= meta);
    if (p.periodo === "prima") i1 = k - 1; else i0 = k;
  }
  const lnc = new Float64Array(n);
  for (let t = 0; t < n; t++) lnc[t] = Math.log(st.c[t]);
  const costo = (p.costo || 0) / 100, rollCum = rollCumulati(st);
  const finLong = (p.finanz || 0) / 100 / SEDUTE_ANNO, finShort = (p.finanzShort ?? p.finanz ?? 0) / 100 / SEDUTE_ANNO;
  const N = p.N || 60, F = forzaTrend(st, N), FS = fasiSerie(st, {...p, N});
  const rottura = p.ingresso === "rottura", E = rottura ? estremi(st, p.Nr || 55) : null;
  const filtraFase = p.strategia !== "combinata" && p.fase && p.fase !== "tutte";
  const filtraDir = p.dirTrend && p.dirTrend !== "tutte";
  const ops = [];
  let pos = 0, ie = 0, se = 0, best = 0, fIn = null, faseIn = null;

  function segnale(regola, t) {
    if (rottura) {
      if (E.max[t] == null) return 0;
      const su = st.c[t] > E.max[t], giu = st.c[t] < E.min[t];
      return regola === "trend" ? (su ? 1 : giu ? -1 : 0) : (giu ? 1 : su ? -1 : 0);
    }
    const zt = z[t];
    if (zt == null) return 0;
    return regola === "contrarian" ? (zt <= -p.k ? 1 : zt >= p.k ? -1 : 0) : (zt >= p.k ? 1 : zt <= -p.k ? -1 : 0);
  }
  /* regola da usare in t, dopo i filtri di fase; null se in t non si opera */
  function regolaIn(t) {
    const fase = FS[t];
    if ((filtraFase || filtraDir || p.strategia === "combinata") && fase == null) return null;
    if (filtraFase && fase !== p.fase) return null;
    return p.strategia === "combinata" ? (fase === "laterale" ? "contrarian" : fase === "trend" ? "trend" : null) : p.strategia;
  }
  function ammesso(d, t) {
    if ((d === 1 && p.lato === "short") || (d === -1 && p.lato === "long")) return false;
    if (p.lato === "curva" && versoCurva(st, t) !== d) return false;
    if (filtraDir && ((Math.sign(F[t]) === d) !== (p.dirTrend === "favore"))) return false;
    return true;
  }

  for (let t = i0; t <= i1; t++) {
    if (pos !== 0) {
      const x = pos * (lnc[t] - lnc[ie]);                   /* risultato in log dal prezzo d'ingresso */
      const livStop = -p.S * se, livTrail = p.T > 0 ? best - p.T * se : -Infinity;
      let motivo = null;
      if (x <= Math.max(livStop, livTrail)) motivo = livTrail > livStop ? "trailing" : "stop";
      else if (p.G > 0 && x >= p.G * se) motivo = "target";
      else if (p.H > 0 && t - ie >= p.H) motivo = "durata";
      else if (t === i1) motivo = i1 === n - 1 ? "aperta" : "fine periodo";
      if (motivo) {
        const durata = t - ie, rinnovi = rollCum[t - 1] - rollCum[ie];      /* scadenze tra ingresso e uscita */
        const costi = costo * (1 + rinnovi) + (pos === 1 ? finLong : finShort) * durata;
        const pct = (pos === 1 ? Math.exp(x) - 1 : 1 - Math.exp(-x)) - costi;
        const op = {ie, iu: t, lato: pos, R: (x - costi) / (p.S * se), pct, durata, motivo, f: fIn, fase: faseIn, se, rinnovi};
        if (motivo === "aperta") {
          const b = Math.max(best, x), liv = Math.max(livStop, p.T > 0 ? b - p.T * se : -Infinity);
          op.livello = st.c[ie] * Math.exp(pos * liv);       /* stop valido dalla prossima seduta */
          op.target = p.G > 0 ? st.c[ie] * Math.exp(pos * p.G * se) : null;
          op.restanti = p.H > 0 ? p.H - durata : null;
        }
        ops.push(op);
        pos = 0;
        continue;                                           /* niente nuovo ingresso nella stessa seduta */
      }
      if (x > best) best = x;
      continue;
    }
    if (t === i1) break;
    const s = sg[t];
    if (s == null || !(s > 0)) continue;
    const regola = regolaIn(t);
    if (!regola) continue;
    const d = segnale(regola, t);
    if (d && ammesso(d, t)) { pos = d; ie = t; se = s / 100; best = 0; fIn = F[t]; faseIn = FS[t]; }
  }

  /* livelli di ingresso per la seduta dopo t, per chi è senza posizione */
  function livelliIngresso(t) {
    const s = sg[t] / 100, regola = regolaIn(t), P = st.c[t];
    const out = {fase: FS[t], regola, long: null, short: null};
    if (regola && s > 0) {
      let su, giu;
      if (rottura) {
        let a = -Infinity, b = Infinity;
        for (let i = t - (p.Nr || 55) + 1; i <= t; i++) { if (st.c[i] > a) a = st.c[i]; if (st.c[i] < b) b = st.c[i]; }
        su = a; giu = b;
      } else {
        su = P * Math.exp(p.k * s); giu = P * Math.exp(-p.k * s);
      }
      const longSopra = regola === "trend";               /* trend: long sopra, short sotto */
      const okDir = d => !filtraDir || (Math.sign(F[t]) === d) === (p.dirTrend === "favore");
      const okCurva = d => p.lato !== "curva" || versoCurva(st, t) === d;
      if (p.lato !== "short" && okDir(1) && okCurva(1)) out.long = {sopra: longSopra, prezzo: longSopra ? su : giu};
      if (p.lato !== "long" && okDir(-1) && okCurva(-1)) out.short = {sopra: !longSopra, prezzo: longSopra ? giu : su};
      out.sigma = s;
    }
    return out;
  }
  /* a fine storico: se la regola è senza posizione, i livelli per la prossima seduta (prossima);
     in ogni caso, quelli di chi parte oggi senza posizione (daZero) */
  let prossima = null, daZero = null;
  if (i1 === n - 1) {
    const ultima = ops.length ? ops[ops.length - 1] : null;
    daZero = livelliIngresso(n - 1);
    if (!(ultima && ultima.motivo === "aperta")) prossima = daZero;
  }
  const anni = (Date.parse(st.date[i1]) - Date.parse(st.date[i0])) / (365.25 * 864e5);
  return {ops, i0, i1, anni, prossima, daZero};
}

function statisticheOps(ops, anni) {
  const n = ops.length;
  if (!n) return null;
  const R = ops.map(o => o.R), somma = a => a.reduce((x, y) => x + y, 0);
  const vinc = R.filter(r => r > 0), perd = R.filter(r => r <= 0);
  const media = somma(R) / n;
  const sd = Math.sqrt(somma(R.map(r => (r - media) ** 2)) / Math.max(1, n - 1));
  const skew = n > 2 && sd > 0 ? somma(R.map(r => ((r - media) / sd) ** 3)) * n / ((n - 1) * (n - 2)) : null;
  const sV = somma(vinc), sP = -somma(perd);
  let cum = 0, picco = 0, dd = 0;
  const curva = R.map(r => { cum += r; picco = Math.max(picco, cum); dd = Math.max(dd, picco - cum); return cum; });
  return {n, perAnno: anni > 0 ? n / anni : null, vincenti: vinc.length / n, media, sd, t: sd > 0 ? media / (sd / Math.sqrt(n)) : 0,
    mediaPct: somma(ops.map(o => o.pct)) / n, mediaV: vinc.length ? sV / vinc.length : 0, mediaP: perd.length ? -sP / perd.length : 0,
    pf: sP > 0 ? sV / sP : Infinity, peggiore: Math.min(...R), migliore: Math.max(...R),
    durata: somma(ops.map(o => o.durata)) / n, skew, ddR: dd, curva, totale: cum};
}

function perFase(ops) {
  const gruppi = {laterale: [], intermedia: [], trend: [], favore: [], contro: []};
  for (const o of ops) {
    if (o.fase == null) continue;
    gruppi[o.fase].push(o.R);
    gruppi[Math.sign(o.f) === o.lato ? "favore" : "contro"].push(o.R);
  }
  const out = {};
  for (const [k, v] of Object.entries(gruppi)) {
    const n = v.length, media = n ? v.reduce((a, b) => a + b, 0) / n : null;
    const varianza = n > 1 ? v.reduce((a, r) => a + (r - media) ** 2, 0) / (n - 1) : null;
    out[k] = {n, media, es: varianza != null ? Math.sqrt(varianza / n) : null};   /* es = errore standard della media */
  }
  return out;
}

/* 200 operazioni future: ricampionamento con reinserimento delle operazioni passate */
function montecarlo(R, f, nOp = 200, nPerc = 4000) {
  const rng = mulberry32(20261110), m = R.length, w = nOp + 1;
  const eq = new Float32Array(nPerc * w), finali = new Float64Array(nPerc), dd = new Float64Array(nPerc);
  let meta = 0;
  for (let p = 0; p < nPerc; p++) {
    let E = 1, picco = 1, maxdd = 0, minE = 1;
    eq[p * w] = 1;
    for (let k = 1; k <= nOp; k++) {
      E *= 1 + f * R[Math.floor(rng() * m)];
      if (E < 0) E = 0;
      if (E > picco) picco = E;
      maxdd = Math.max(maxdd, picco > 0 ? 1 - E / picco : 1);
      minE = Math.min(minE, E);
      eq[p * w + k] = E;
    }
    finali[p] = E; dd[p] = maxdd;
    if (minE <= 0.5) meta++;
  }
  const bande = {p5: [], p25: [], p50: [], p75: [], p95: []}, col = new Float64Array(nPerc);
  for (let k = 0; k < w; k++) {
    for (let p = 0; p < nPerc; p++) col[p] = eq[p * w + k];
    col.sort();
    for (const [nome, q] of [["p5", .05], ["p25", .25], ["p50", .5], ["p75", .75], ["p95", .95]]) bande[nome].push(quantile(col, q) * 100);
  }
  let perdita = 0;
  for (const e of finali) if (e < 1) perdita++;
  finali.sort(); dd.sort();
  return {bande, finali, dd, pMeta: meta / nPerc, pPerdita: perdita / nPerc};
}

/* istogramma con classi di larghezza "tonda", al massimo una quarantina */
function classi(valori, larghezze) {
  let lo = Infinity, hi = -Infinity;
  for (const v of valori) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const w = larghezze.find(l => (hi - lo) / l <= 40) || larghezze[larghezze.length - 1];
  const inizio = Math.floor(lo / w) * w, nb = Math.max(1, Math.floor((hi - inizio) / w) + 1);
  const conteggi = new Array(nb).fill(0);
  for (const v of valori) conteggi[Math.min(nb - 1, Math.floor((v - inizio) / w))]++;
  return conteggi.map((k, i) => ({x: inizio + (i + 0.5) * w, y: k, da: inizio + i * w, a: inizio + (i + 1) * w}));
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {SOGLIA_LATERALE, SOGLIA_TREND, Q_VR, SEDUTE_ANNO, mulberry32, quantile, forzaTrend, faseDi, serieVR, soglieVR,
    fasiSerie, estremi, rollCumulati, versoCurva, simulaStrategia, statisticheOps, perFase, montecarlo, classi};
}
