#!/usr/bin/env node
/* Ricerca di una regola con un protocollo fissato prima di guardare i risultati.

   - Griglia di regole: trend su shock, trend su rottura, contrarian con trailing, contrarian
     classico, combinata (contrarian nella fase di ritorno alla media o laterale, trend nella fase di
     persistenza o di trend). Ognuna long, short, su entrambi i lati, oppure solo nel verso pagato
     dalla curva dei futures (short in contango, long in backwardation).
   - σ EWMA, che non usa informazioni future. Costi del CFD Fineco: spread di andata e ritorno, pagato
     di nuovo a ogni scadenza del contratto per chi resta dentro; finanziamento Euribor 1 mese + 2,95%
     sui long, 2,95% − Euribor sugli short.
   - Scelta sulla PRIMA metà della storia rinnovata di gas Henry Hub e petrolio WTI: punteggio = media
     dei t statistici del risultato medio in R sui due mercati.
   - Verifica sulla SECONDA metà degli stessi mercati, che la selezione non vede, e su Brent e TTF,
     che hanno solo la serie continua di Yahoo non rinnovata.
   - La regola scelta è la prima per punteggio dentro il campione, qualunque cosa faccia fuori.
   - Riferimenti senza parametri: sempre long, sempre short, e la regola della curva (short in
     contango, long in backwardation), in % annuo sul controvalore.

   Uso: node scripts/ricerca.js [cartella docs]   (legge storia/*.json, scrive ricerca.json) */
"use strict";
const fs = require("fs");
const path = require("path");
const M = require("../docs/motore.js");

const DOCS = path.resolve(process.argv[2] || path.join(__dirname, "..", "docs"));
const SELEZIONE = ["NG=F", "CL=F"];
const VERIFICA = ["BZ=F", "TTF=F"];
const NOMI = {"NG=F": "Gas Henry Hub", "CL=F": "Petrolio WTI", "BZ=F": "Petrolio Brent", "TTF=F": "Gas TTF"};
/* spread di andata e ritorno in %: KID Fineco sul gas 19 € per lato su 10.000 €; mark-up 0,10% per
   lato sul petrolio più lo spread del mercato. Il TTF non è tra i CFD Fineco: stesso costo del gas. */
const COSTO = {"NG=F": 0.4, "CL=F": 0.25, "BZ=F": 0.25, "TTF=F": 0.4};
const EURIBOR = 2.2, SPREAD_FIN = 2.95;
const FIN_LONG = EURIBOR + SPREAD_FIN, FIN_SHORT = SPREAD_FIN - EURIBOR;
const MIN_OPS_NG = 30, MIN_OPS = 15, TOP = 10;

const fileStoria = s => path.join(DOCS, "storia", s.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "") + ".json");

function griglia() {
  const out = [];
  for (const lato of ["long", "short", "entrambi", "curva"]) {
    for (const k of [1.5, 2, 2.5]) for (const S of [1.5, 2, 3]) for (const T of [2, 3, 4])
      out.push({famiglia: "Trend su shock", strategia: "trend", ingresso: "shock", lato, k, S, T, G: 0, H: 0});
    for (const Nr of [20, 55, 100]) for (const S of [2, 3]) for (const T of [3, 4, 6])
      out.push({famiglia: "Trend su rottura", strategia: "trend", ingresso: "rottura", Nr, lato, k: 2, S, T, G: 0, H: 0});
    for (const k of [1.5, 2, 2.5]) for (const S of [1.5, 2, 3]) for (const T of [1.5, 2, 3])
      out.push({famiglia: "Contrarian con trailing", strategia: "contrarian", ingresso: "shock", lato, k, S, T, G: 0, H: 0});
    for (const k of [1.5, 2, 2.5]) for (const S of [3, 4]) for (const G of [1, 1.5]) for (const H of [5, 10])
      out.push({famiglia: "Contrarian classico", strategia: "contrarian", ingresso: "shock", lato, k, S, T: 0, G, H});
    for (const indicatore of ["forza", "autocorrelazione"]) for (const N of [60, 120]) for (const k of [1.5, 2]) for (const S of [2, 3]) for (const T of [2, 3])
      out.push({famiglia: "Combinata", strategia: "combinata", ingresso: "shock", indicatore, N, lato, k, S, T, G: 0, H: 0});
  }
  return out.map((c, id) => ({id, ...c}));
}

const parametri = (cfg, simbolo, periodo) => ({sigma: "ewma", periodo, costo: COSTO[simbolo], finanz: FIN_LONG,
  finanzShort: FIN_SHORT, N: 60, indicatore: "forza", fase: "tutte", dirTrend: "tutte", Nr: 55, ...cfg});

function valuta(st, cfg, simbolo, periodo) {
  const r = M.simulaStrategia(st, parametri(cfg, simbolo, periodo)), s = M.statisticheOps(r.ops, r.anni);
  if (!s) return {n: 0, media: 0, t: 0, pf: null, perAnno: 0, vincenti: null, peggiore: null, skew: null, mediaPct: null};
  return {n: s.n, media: s.media, t: s.t, pf: isFinite(s.pf) ? s.pf : null, perAnno: s.perAnno,
    vincenti: s.vincenti, peggiore: s.peggiore, skew: s.skew, mediaPct: s.mediaPct};
}

/* intervallo di sedute di un periodo, con la stessa regola del motore */
function intervallo(st, periodo) {
  const n = st.c.length;
  if (periodo === "tutto") return [0, n - 1];
  const meta = (Date.parse(st.date[0]) + Date.parse(st.date[n - 1])) / 2;
  const k = st.date.findIndex(d => Date.parse(d) >= meta);
  return periodo === "prima" ? [0, k - 1] : [k, n - 1];
}

/* Riferimenti senza parametri, in % annuo sul controvalore tenuto costante:
   verso(t) = posizione tenuta dalla chiusura di t alla chiusura di t+1. */
function riferimento(st, simbolo, periodo, verso) {
  const [i0, i1] = intervallo(st, periodo), costo = COSTO[simbolo] / 100, rollCum = M.rollCumulati(st);
  let pos = 0, logEq = 0, picco = 0, dd = 0, somma = 0, somma2 = 0, giorni = 0, cambi = 0, tempo = 0;
  for (let t = i0; t <= i1; t++) {
    if (t > i0) {
      const r = Math.log(st.c[t] / st.c[t - 1]);
      let g = pos ? pos * (Math.exp(r) - 1) : 0;
      g -= pos === 1 ? FIN_LONG / 100 / M.SEDUTE_ANNO : pos === -1 ? FIN_SHORT / 100 / M.SEDUTE_ANNO : 0;
      const nuovo = t === i1 ? 0 : verso(t);
      if (nuovo !== pos) { g -= Math.abs(nuovo - pos) * costo / 2; cambi++; }
      else if (pos && rollCum[t] - rollCum[t - 1]) g -= costo;      /* scadenza del CFD: chiudi e riapri */
      logEq += Math.log(Math.max(1e-9, 1 + g));
      somma += g; somma2 += g * g; giorni++; if (pos) tempo++;
      picco = Math.max(picco, logEq); dd = Math.max(dd, 1 - Math.exp(logEq - picco));
      pos = nuovo;
    } else {
      pos = verso(t);
      if (pos) { logEq += Math.log(1 - costo / 2); cambi++; }
    }
  }
  const anni = (Date.parse(st.date[i1]) - Date.parse(st.date[i0])) / (365.25 * 864e5);
  const media = somma / giorni, sd = Math.sqrt(Math.max(0, somma2 / giorni - media * media));
  return {annuo: Math.expm1(logEq / anni), vol: sd * Math.sqrt(M.SEDUTE_ANNO), rapporto: sd > 0 ? media / sd * Math.sqrt(M.SEDUTE_ANNO) : null,
    discesa: dd, cambiAnno: cambi / anni, investito: tempo / giorni};
}

function spearman(a, b) {
  const rango = v => { const o = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]), r = new Array(v.length); o.forEach(([, i], k) => { r[i] = k; }); return r; };
  const ra = rango(a), rb = rango(b), n = a.length, ma = (n - 1) / 2;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - ma); da += (ra[i] - ma) ** 2; db += (rb[i] - ma) ** 2; }
  return num / Math.sqrt(da * db);
}

const arr = (x, d = 3) => x == null || !isFinite(x) ? null : Math.round(x * 10 ** d) / 10 ** d;
const compatto = v => ({n: v.n, media: arr(v.media), t: arr(v.t, 2), pf: arr(v.pf, 2), perAnno: arr(v.perAnno, 1),
  vincenti: arr(v.vincenti, 3), peggiore: arr(v.peggiore, 2), skew: arr(v.skew, 2), mediaPct: arr(v.mediaPct, 5)});
const compattoRif = v => ({annuo: arr(v.annuo, 4), vol: arr(v.vol, 4), rapporto: arr(v.rapporto, 2), discesa: arr(v.discesa, 3),
  cambiAnno: arr(v.cambiAnno, 1), investito: arr(v.investito, 2)});

function main() {
  const storie = {}, mercati = {};
  for (const s of [...SELEZIONE, ...VERIFICA]) {
    try {
      const st = storie[s] = JSON.parse(fs.readFileSync(fileStoria(s), "utf8"));
      const [, i1p] = intervallo(st, "prima");
      const quotaSp = per => { if (!st.sp) return 0; const [a, b] = intervallo(st, per); let k = 0; for (let t = a; t <= b; t++) if (st.sp[t] != null) k++; return arr(k / (b - a + 1), 3); };
      mercati[s] = {nome: NOMI[s], dal: st.date[0], al: st.date[st.date.length - 1], meta: st.date[i1p + 1],
        sedute: st.date.length, rinnovata: !!st.rinnovo, fineEia: st.fine_eia || null, curva: !!st.sp,
        spreadPrima: quotaSp("prima"), spreadSeconda: quotaSp("seconda"), scadenze: (st.roll || []).length, costo: COSTO[s]};
    } catch (e) {
      console.error(`storico mancante per ${s}: ${e.message}`);
    }
  }
  if (!SELEZIONE.every(s => storie[s])) { console.error("Mancano gli storici della selezione: ricerca annullata."); process.exit(1); }

  /* riferimenti */
  const riferimenti = {};
  for (const s of Object.keys(storie)) {
    const st = storie[s], periodi = SELEZIONE.includes(s) ? ["prima", "seconda", "tutto"] : ["tutto"];
    riferimenti[s] = {};
    for (const per of periodi) {
      const r = {long: compattoRif(riferimento(st, s, per, () => 1)), short: compattoRif(riferimento(st, s, per, () => -1))};
      if (st.sp) r.curva = compattoRif(riferimento(st, s, per, t => M.versoCurva(st, t)));
      riferimenti[s][per] = r;
    }
  }

  /* griglia */
  const configs = griglia(), righe = [];
  for (const cfg of configs) {
    const is = {}, oos = {}, ver = {};
    for (const s of SELEZIONE) { is[s] = valuta(storie[s], cfg, s, "prima"); oos[s] = valuta(storie[s], cfg, s, "seconda"); }
    for (const s of VERIFICA) if (storie[s] && (cfg.lato !== "curva" || storie[s].sp)) ver[s] = valuta(storie[s], cfg, s, "tutto");
    const ammessa = is["NG=F"].n >= MIN_OPS_NG && is["CL=F"].n >= MIN_OPS;
    const media = o => { const v = SELEZIONE.filter(s => o[s].n >= MIN_OPS).map(s => o[s].t); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0; };
    righe.push({cfg, is, oos, ver, ammessa, punteggioIS: media(is), punteggioOOS: media(oos)});
  }
  const ammesse = righe.filter(r => r.ammessa).sort((a, b) => b.punteggioIS - a.punteggioIS);
  const rho = ammesse.length > 2 ? spearman(ammesse.map(r => r.punteggioIS), ammesse.map(r => r.punteggioOOS)) : null;

  /* famiglie: media di tutte le regole della famiglia, senza scegliere */
  const famiglie = {};
  for (const r of ammesse) {
    const k = r.cfg.famiglia + "|" + r.cfg.lato;
    const f = famiglie[k] = famiglie[k] || {famiglia: r.cfg.famiglia, lato: r.cfg.lato, regole: 0, sIS: 0, sOOS: 0, pos: 0, sNG: 0, sCL: 0, sV: {}, nV: {}};
    f.regole++; f.sIS += r.punteggioIS; f.sOOS += r.punteggioOOS; if (r.punteggioOOS > 0) f.pos++;
    f.sNG += r.oos["NG=F"].media; f.sCL += r.oos["CL=F"].media;
    for (const [s, v] of Object.entries(r.ver)) if (v.n >= MIN_OPS) { f.sV[s] = (f.sV[s] || 0) + v.media; f.nV[s] = (f.nV[s] || 0) + 1; }
  }
  const tabFamiglie = Object.values(famiglie).map(f => ({famiglia: f.famiglia, lato: f.lato, regole: f.regole,
    tIS: arr(f.sIS / f.regole, 2), tOOS: arr(f.sOOS / f.regole, 2), quotaPositiveOOS: arr(f.pos / f.regole, 2),
    mediaOOS: {"NG=F": arr(f.sNG / f.regole), "CL=F": arr(f.sCL / f.regole)},
    mediaVerifica: Object.fromEntries(VERIFICA.map(s => [s, f.nV[s] ? arr(f.sV[s] / f.nV[s]) : null]))}))
    .sort((a, b) => b.tOOS - a.tOOS);

  const riassunto = r => ({cfg: r.cfg, punteggioIS: arr(r.punteggioIS, 2), punteggioOOS: arr(r.punteggioOOS, 2),
    is: Object.fromEntries(Object.entries(r.is).map(([k, v]) => [k, compatto(v)])),
    oos: Object.fromEntries(Object.entries(r.oos).map(([k, v]) => [k, compatto(v)])),
    verifica: Object.fromEntries(Object.entries(r.ver).map(([k, v]) => [k, compatto(v)]))});
  const migliori = ammesse.slice(0, TOP).map(riassunto);

  const esito = {
    generato: new Date().toISOString().replace(/\.\d+Z$/, "Z"),
    protocollo: {selezione: SELEZIONE, verifica: VERIFICA, costo: COSTO, euribor: EURIBOR, finanzLong: FIN_LONG, finanzShort: FIN_SHORT,
      sigma: "ewma", minOperazioniNG: MIN_OPS_NG, minOperazioni: MIN_OPS, regole: configs.length,
      punteggio: "media dei t statistici del risultato medio in R sulla prima metà di gas Henry Hub e petrolio WTI"},
    mercati, riferimenti, regoleAmmesse: ammesse.length, correlazioneSpearman: arr(rho, 3),
    quotaRegoleConOOSPositivo: arr(ammesse.filter(r => r.punteggioOOS > 0).length / Math.max(1, ammesse.length), 3),
    migliori, topOOSpositive: migliori.filter(m => m.punteggioOOS > 0).length, famiglie: tabFamiglie, scelta: migliori[0] || null,
  };
  fs.writeFileSync(path.join(DOCS, "ricerca.json"), JSON.stringify(esito));

  /* riassunto leggibile nei log */
  const f2 = (x, d = 2) => (x == null ? "–" : (x >= 0 ? "+" : "") + x.toFixed(d)).padStart(6);
  const p1 = x => x == null ? "   –  " : ((x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%").padStart(7);
  const descr = c => `${c.famiglia}, ${c.lato}` + (c.ingresso === "rottura" ? `, rottura ${c.Nr}` : `, k ${c.k}`) +
    `, S ${c.S}` + (c.T ? `, T ${c.T}` : "") + (c.G ? `, G ${c.G}` : "") + (c.H ? `, H ${c.H}` : "") + (c.strategia === "combinata" ? `, ${c.indicatore} ${c.N}` : "");
  const righeLog = [];
  const log = (...x) => { const t = x.join(" "); righeLog.push(t); console.log(t); };
  log(`Regole: ${configs.length}, ammesse: ${ammesse.length}`);
  for (const [s, m] of Object.entries(mercati))
    log(`${s} ${m.dal}..${m.al}, metà ${m.meta}, ${m.rinnovata ? `serie rinnovata (EIA fino al ${m.fineEia}), ${m.scadenze} scadenze` : "serie continua NON rinnovata"}` +
      (m.curva ? `, spread di curva nel ${Math.round(m.spreadPrima * 100)}% della prima metà e nel ${Math.round(m.spreadSeconda * 100)}% della seconda` : ""));
  log("\nRiferimenti, % annuo (annuo | vol | rapporto | discesa massima):");
  for (const [s, per] of Object.entries(riferimenti)) for (const [nomePer, r] of Object.entries(per)) for (const [k, v] of Object.entries(r))
    log(`${s.padEnd(6)} ${nomePer.padEnd(7)} ${k.padEnd(6)} ${p1(v.annuo)} ${p1(v.vol)} ${f2(v.rapporto)} ${p1(-v.discesa)}  investito ${Math.round(v.investito * 100)}%`);
  log(`\nCorrelazione di Spearman tra punteggio dentro e fuori campione: ${rho == null ? "–" : rho.toFixed(3)}`);
  log(`Quota di regole con punteggio fuori campione positivo: ${(esito.quotaRegoleConOOSPositivo * 100).toFixed(0)}%`);
  log("\nLe 10 migliori dentro il campione (t IS | t OOS | R medio OOS NG, n | CL, n | Brent | TTF):");
  for (const m of migliori) {
    const ng = m.oos["NG=F"], cl = m.oos["CL=F"], bz = m.verifica["BZ=F"], ttf = m.verifica["TTF=F"];
    log(`${f2(m.punteggioIS)} | ${f2(m.punteggioOOS)} | ${f2(ng.media)} ${String(ng.n).padStart(4)} | ${f2(cl.media)} ${String(cl.n).padStart(4)} | ${bz ? f2(bz.media) : "    –"} | ${ttf ? f2(ttf.media) : "    –"}  ${descr(m.cfg)}`);
  }
  log("\nFamiglie (t medio IS | t medio OOS | quota OOS positivi | R medio OOS NG | CL | Brent | TTF):");
  for (const f of tabFamiglie)
    log(`${f2(f.tIS)} | ${f2(f.tOOS)} | ${String(Math.round(f.quotaPositiveOOS * 100)).padStart(3)}% | ${f2(f.mediaOOS["NG=F"])} | ${f2(f.mediaOOS["CL=F"])} | ${f2(f.mediaVerifica["BZ=F"])} | ${f2(f.mediaVerifica["TTF=F"])}  ${f.famiglia}, ${f.lato}`);
  if (process.env.GITHUB_STEP_SUMMARY) fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, "```\n" + righeLog.join("\n") + "\n```\n");
}

main();
