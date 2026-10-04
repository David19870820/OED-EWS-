// tv_ar_protocol.mjs
// =====================================================================
// Dry-run of the TRANS-ID P2 protocol on synthetic OED data.
// Verifies each preregistered analysis step works on known dynamics:
//
//   RQ1a moving-window AR(1)+SD EWS with phase-randomized surrogates
//   RQ1b piecewise TV-AR vs constant AR(1) via BIC
//   H2a  directionality: E(PA) vs D(NA) channel EWS (improvement -> E leads)
//   RQ2  recovery nonlinearity: linear vs logistic BIC
//
// Two conditions (sensitivity):
//   A ideal: nBeeps=120, seed=42 (matches node_demo 5/5 demo)
//   B realistic: nBeeps=300, seed=123 (longer, more events/noise)
// =====================================================================
import { writeFileSync } from 'node:fs';
import { simulateEMA, buildDerived } from '../node_demo/simulate_ema.mjs';

function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN; }
function sd(a) { if (a.length < 2) return NaN; const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); }
function detrend(x) {
  const n = x.length;
  const t = Array.from({ length: n }, (_, i) => i);
  const mt = mean(t), mx = mean(x);
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (t[i] - mt) ** 2; sxy += (t[i] - mt) * (x[i] - mx); }
  const b = sxx ? sxy / sxx : 0, a = mx - b * mt;
  return x.map((v, i) => v - (a + b * t[i]));
}
function rollingAR1(x, w) {
  const out = new Array(x.length).fill(NaN);
  for (let i = w; i < x.length; i++) {
    const seg = x.slice(i - w, i);
    const y = seg.slice(1), X = seg.slice(0, -1);
    const mX = mean(X), mY = mean(y);
    let sxx = 0, sxy = 0;
    for (let j = 0; j < X.length; j++) { sxx += (X[j] - mX) ** 2; sxy += (X[j] - mX) * (y[j] - mY); }
    out[i] = sxx < 1e-12 ? NaN : sxy / sxx;
  }
  return out;
}
function rollingSD(x, w) {
  const out = new Array(x.length).fill(NaN);
  for (let i = w; i < x.length; i++) {
    const seg = x.slice(i - w, i);
    const m = mean(seg);
    out[i] = Math.sqrt(seg.reduce((s, v) => s + (v - m) ** 2, 0) / (seg.length - 1));
  }
  return out;
}
function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }
function phaseRandomize(x, rng) {
  // FIXED: zero-pad to power of 2 (iterative FFT requires pow2 length);
  // no double scaling (inverse FFT already divides by N).
  const n = x.length;
  const N = nextPow2(n);
  const mx = mean(x);
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < n; i++) re[i] = x[i] - mx;
  fft(re, im, false);
  const half = Math.floor(N / 2) + 1;
  const amp = [], ph = [];
  for (let k = 0; k < half; k++) {
    amp.push(Math.hypot(re[k], im[k]));
    ph.push(k === 0 || (N % 2 === 0 && k === half - 1) ? 0 : rng() * 2 * Math.PI);
  }
  const re2 = new Float64Array(N), im2 = new Float64Array(N);
  for (let k = 0; k < half; k++) {
    const c = Math.cos(ph[k]) * amp[k], s = Math.sin(ph[k]) * amp[k];
    re2[k] = c; im2[k] = s;
    if (k > 0 && k < half - (N % 2 === 0 ? 1 : 0)) { re2[N - k] = c; im2[N - k] = -s; }
  }
  fft(re2, im2, true);
  const out = [];
  for (let i = 0; i < n; i++) out.push(re2[i] + mx);
  return out;
}
function fft(re, im, invert) {
  const n = re.length;
  if (n <= 1) return;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = 2 * Math.PI / len * (invert ? 1 : -1);
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm;
        const nRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nRe;
      }
    }
  }
  if (invert) { for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
}

function fitAR1Seg(y, X) {
  const n = y.length;
  const mX = mean(X.map((r) => r[1])), my = mean(y);
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (X[i][1] - mX) ** 2; sxy += (X[i][1] - mX) * (y[i] - my); }
  const phi = sxx < 1e-12 ? 0 : sxy / sxx;
  const c = my - phi * mX;
  let rss = 0;
  for (let i = 0; i < n; i++) { const e = y[i] - c - phi * X[i][1]; rss += e * e; }
  return { phi, rss };
}
function bicAR1(x, segs) {
  const n = x.length;
  const ys = [], Xs = [];
  for (let t = 1; t < n; t++) { ys.push(x[t]); Xs.push([1, x[t - 1]]); }
  const m = ys.length;
  if (m < 6) return NaN;
  if (segs === 1) {
    const f = fitAR1Seg(ys, Xs);
    const k = 2;
    const ll = -m / 2 * (Math.log(2 * Math.PI * f.rss / m) + 1);
    return { bic: k * Math.log(m) - 2 * ll, k, phi: f.phi };
  }
  const per = Math.floor(m / segs);
  let rss = 0;
  const phis = [];
  for (let s = 0; s < segs; s++) {
    const from = s * per, to = (s === segs - 1) ? m : from + per;
    if (to - from < 4) return NaN;
    const f = fitAR1Seg(ys.slice(from, to), Xs.slice(from, to));
    rss += f.rss;
    phis.push(f.phi);
  }
  const k = 2 * segs;
  const ll = -m / 2 * (Math.log(2 * Math.PI * rss / m) + 1);
  return { bic: k * Math.log(m) - 2 * ll, k, phis };
}

function ewsTest(rows, varCol, t0, window, pre, nSurrogates, seed, minVar = 0) {
  let s = seed >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const xRaw = rows.map((r) => r[varCol]);
  const n = xRaw.length;
  if (t0 - pre < window || t0 < 1) return null;
  const xm0 = mean(xRaw);
  const x = xRaw.map((v) => v - xm0); // demean only; detrend destroys EWS (validated)
  const m0 = mean(x);
  const v0 = x.reduce((a, b) => a + (b - m0) ** 2, 0) / n;
  if (v0 < minVar) return null;
  const ar = rollingAR1(x, window);
  const sdd = rollingSD(x, window);
  const baseTo = Math.min(Math.floor(n * 0.3), t0);
  const baseAr = mean(ar.slice(0, baseTo).filter((v) => !Number.isNaN(v)));
  const preAr = mean(ar.slice(t0 - pre, t0).filter((v) => !Number.isNaN(v)));
  const baseSd = mean(sdd.slice(0, baseTo).filter((v) => !Number.isNaN(v)));
  const preSd = mean(sdd.slice(t0 - pre, t0).filter((v) => !Number.isNaN(v)));
  const obsDiffAr = preAr - baseAr;
  const obsDiffSd = preSd - baseSd;
  if (!Number.isFinite(obsDiffAr) || !Number.isFinite(obsDiffSd)) return null;
  let countAr = 0, countSd = 0;
  for (let k = 0; k < nSurrogates; k++) {
    const xs = phaseRandomize(x, rng);
    const es = rollingAR1(xs, window);
    const ss = rollingSD(xs, window);
    const bm = mean(es.slice(0, baseTo).filter((v) => !Number.isNaN(v)));
    const ps = mean(es.slice(t0 - pre, t0).filter((v) => !Number.isNaN(v)));
    if (ps - bm >= obsDiffAr) countAr++;
    const bm2 = mean(ss.slice(0, baseTo).filter((v) => !Number.isNaN(v)));
    const ps2 = mean(ss.slice(t0 - pre, t0).filter((v) => !Number.isNaN(v)));
    if (ps2 - bm2 >= obsDiffSd) countSd++;
  }
  return { diffAr: obsDiffAr, pAr: (countAr + 1) / (nSurrogates + 1), diffSd: obsDiffSd, pSd: (countSd + 1) / (nSurrogates + 1) };
}

function recoveryNonlinearity(rows, t0, varCol) {
  const seg = rows.slice(t0).map((r) => r[varCol]);
  if (seg.length < 8) return null;
  const y = seg, t = Array.from({ length: y.length }, (_, i) => i);
  const n = y.length;
  const my = mean(y);
  const mt = mean(t);
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (t[i] - mt) ** 2; sxy += (t[i] - mt) * (y[i] - my); }
  const b = sxx ? sxy / sxx : 0, a = my - b * mt;
  let rssLin = 0;
  for (let i = 0; i < n; i++) rssLin += (y[i] - (a + b * t[i])) ** 2;
  const kLin = 2;
  const llLin = -n / 2 * (Math.log(2 * Math.PI * rssLin / n) + 1);
  const bicLin = kLin * Math.log(n) - 2 * llLin;
  const yMin = Math.min(...y), yMax = Math.max(...y);
  const c = yMax - yMin + 1e-6;
  let bestRss = Infinity;
  for (let aa = -3; aa <= 3.01; aa += 0.25) {
    for (let bb = -1.5; bb <= 1.51; bb += 0.1) {
      let rss = 0;
      for (let i = 0; i < n; i++) {
        const pred = c / (1 + Math.exp(-(aa + bb * t[i])));
        rss += (y[i] - pred) ** 2;
      }
      if (rss < bestRss) bestRss = rss;
    }
  }
  const kLog = 3;
  const llLog = -n / 2 * (Math.log(2 * Math.PI * bestRss / n) + 1);
  const bicLog = kLog * Math.log(n) - 2 * llLog;
  return { bicLin, bicLog, logisticWins: bicLog < bicLin };
}

function runCondition(label, cfg) {
  const { nPeople, nBeeps, seed, window, pre, nSurrogates } = cfg;
  const people = simulateEMA({ nPeople, nBeeps, seed, depressionProp: 0.5, obsNoise: 0.25 });
  const derived = people.map((p) => ({ ...p, rows: buildDerived(p) }));
  const rows = [];
  for (const p of derived) {
    if (!p.dep) continue;
    const t0 = p.transitionBeep !== undefined && p.transitionBeep !== null ? p.transitionBeep : Math.floor(p.rows.length / 2);
    const rec = { id: p.id, recovering: p.recovering ? 1 : 0, t0, n: p.rows.length };
    const eD = ewsTest(p.rows, 'U_true', t0, window, pre, nSurrogates, p.id * 7 + 1, 1e-6);
    if (eD) { rec.diffAr = eD.diffAr; rec.pAr = eD.pAr; rec.diffSd = eD.diffSd; rec.pSd = eD.pSd; }
    const eE = ewsTest(p.rows, 'recover', t0, window, pre, nSurrogates, p.id * 7 + 2, 1e-6);
    if (eE) { rec.diffArE = eE.diffAr; rec.pArE = eE.pAr; rec.diffSdE = eE.diffSd; rec.pSdE = eE.pSd; }
    const x = detrend(p.rows.map((r) => r.D));
    const bicC = bicAR1(x, 1);
    const nSegs = Math.min(6, Math.max(2, Math.floor((x.length - 1) / 40)));
    const bicT = bicAR1(x, nSegs);
    if (bicC && bicT) { rec.bicC = bicC.bic; rec.bicT = bicT.bic; rec.tvARwins = bicT.bic < bicC.bic; }
    if (p.recovering) {
      const rq2 = recoveryNonlinearity(p.rows, t0, 'E');
      if (rq2) { rec.bicLin = rq2.bicLin; rec.bicLog = rq2.bicLog; rec.logisticWins = rq2.logisticWins; }
    }
    rows.push(rec);
  }
  const recov = rows.filter((r) => r.recovering === 1 && Number.isFinite(r.pAr));
  const non = rows.filter((r) => r.recovering === 0 && Number.isFinite(r.pAr));
  const sigR = recov.filter((r) => r.pAr < 0.05);
  const sigN = non.filter((r) => r.pAr < 0.05);
  const out = {
    label,
    cfg,
    nRecov: recov.length, nNon: non.length,
    sigRecov: sigR.length, sigNon: sigN.length,
    meanDiffArRecov: mean(recov.map((r) => r.diffAr)),
    meanDiffArNon: mean(non.map((r) => r.diffAr)),
    sigSdRecov: recov.filter((r) => Number.isFinite(r.pSd) && r.pSd < 0.05).length,
    tvARwinsRecov: recov.filter((r) => r.tvARwins === true).length,
    tvARwinsNon: non.filter((r) => r.tvARwins === true).length,
    sigErecov: recov.filter((r) => Number.isFinite(r.pArE) && r.pArE < 0.05).length,
    meanDiffArErecov: mean(recov.filter((r) => Number.isFinite(r.diffArE)).map((r) => r.diffArE)),
    logisticWins: rows.filter((r) => r.recovering === 1 && r.logisticWins === true).length,
    nRQ2: rows.filter((r) => r.recovering === 1 && Number.isFinite(r.bicLog)).length,
    meanBICdiffRecov: mean(recov.filter((r) => Number.isFinite(r.bicT) && Number.isFinite(r.bicC)).map((r) => r.bicT - r.bicC)),
    meanBICdiffNon: mean(non.filter((r) => Number.isFinite(r.bicT) && Number.isFinite(r.bicC)).map((r) => r.bicT - r.bicC)),
    per: rows,
  };
  console.log('--- ' + label + ' ---');
  console.log('RQ1a EWS(U-channel): recovering sig=' + sigR.length + '/' + recov.length + ' (' + (100 * sigR.length / Math.max(recov.length, 1)).toFixed(0) + '%) mean diff=' + out.meanDiffArRecov.toFixed(4) + ' | non sig=' + sigN.length + '/' + non.length + ' (' + (100 * sigN.length / Math.max(non.length, 1)).toFixed(0) + '%) mean diff=' + out.meanDiffArNon.toFixed(4));
  console.log('        SD-EWS recovering sig=' + out.sigSdRecov + '/' + recov.length);
  console.log('RQ1b TV-AR wins: recovering ' + out.tvARwinsRecov + '/' + recov.length + ' | non ' + out.tvARwinsNon + '/' + non.length + ' | mean dBIC recov=' + out.meanBICdiffRecov.toFixed(1) + ' non=' + out.meanBICdiffNon.toFixed(1));
  console.log('H2a  direction: E-channel sig=' + out.sigErecov + '/' + recov.length + ' (mean diff E=' + out.meanDiffArErecov.toFixed(4) + ') vs D sig=' + sigR.length + '/' + recov.length + ' (mean diff D=' + out.meanDiffArRecov.toFixed(4) + ')');
  console.log('RQ2  nonlinear: logistic wins=' + out.logisticWins + '/' + out.nRQ2 + ' recovering');
  console.log('');
  return out;
}

const condA = runCondition('A ideal (nBeeps=120, seed=42)', { nPeople: 20, nBeeps: 120, seed: 42, window: 25, pre: 8, nSurrogates: 100 });
const condB = runCondition('B realistic (nBeeps=300, seed=123)', { nPeople: 40, nBeeps: 300, seed: 123, window: 30, pre: 15, nSurrogates: 100 });

writeFileSync('tv_ar_protocol_result.json', JSON.stringify({ condA, condB, note: 'Synthetic dry-run of TRANS-ID P2 protocol. Sensitivity: condition A matches node_demo demo (short, fewer events); condition B is longer with more event noise.' }, null, 2));