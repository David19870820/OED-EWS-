// validate_ews2.mjs - improved baseline definition (early segment as baseline)
function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN; }
function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }
function phaseRandomize(x, rng) {
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
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = re2[i] + mx;
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
function detrend(x) {
  const n = x.length;
  const t = Array.from({ length: n }, (_, i) => i);
  const mt = mean(t), mx = mean(x);
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (t[i] - mt) ** 2; sxy += (t[i] - mt) * (x[i] - mx); }
  const b = sxx ? sxy / sxx : 0, a = mx - b * mt;
  return x.map((v, i) => v - (a + b * t[i]));
}
function genCSD(n, seed, { phi0 = 0.2, phi1 = 0.95, shift = 0, tShift = null, rampStartF = 0.4, rampEndF = 0.75 } = {}) {
  let s = seed >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const x = new Array(n);
  let v = 0;
  const rampStart = Math.floor(n * rampStartF), rampEnd = Math.floor(n * rampEndF);
  const t0 = tShift || Math.floor(n * rampEndF);
  for (let i = 0; i < n; i++) {
    let phi = phi0;
    if (i >= rampStart && i < rampEnd) phi = phi0 + (phi1 - phi0) * ((i - rampStart) / (rampEnd - rampStart));
    else if (i >= rampEnd) phi = phi1;
    const level = i >= t0 ? shift : 0;
    v = phi * v + (rng() - 0.5) * 0.3 + level;
    x[i] = v;
  }
  return { x, t0 };
}
function ewsTest(x, t0, window, pre, nSurrogates, seed, baseFrom = 0, baseTo = null) {
  let s = seed >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const n = x.length;
  if (t0 - pre < window || t0 < 1) return null;
  const xd = detrend(x);
  const ar = rollingAR1(xd, window);
  const sdd = rollingSD(xd, window);
  const bt = baseTo || Math.floor(n * 0.3);
  const baseAr = mean(ar.slice(baseFrom, Math.min(bt, t0)).filter((v) => !Number.isNaN(v)));
  const preAr = mean(ar.slice(t0 - pre, t0).filter((v) => !Number.isNaN(v)));
  const baseSd = mean(sdd.slice(baseFrom, Math.min(bt, t0)).filter((v) => !Number.isNaN(v)));
  const preSd = mean(sdd.slice(t0 - pre, t0).filter((v) => !Number.isNaN(v)));
  const obsDiffAr = preAr - baseAr;
  const obsDiffSd = preSd - baseSd;
  let countAr = 0, countSd = 0;
  for (let k = 0; k < nSurrogates; k++) {
    const xs = phaseRandomize(xd, rng);
    const es = rollingAR1(xs, window);
    const ss = rollingSD(xs, window);
    const bm = mean(es.slice(baseFrom, Math.min(bt, t0)).filter((v) => !Number.isNaN(v)));
    const ps = mean(es.slice(t0 - pre, t0).filter((v) => !Number.isNaN(v)));
    if (ps - bm >= obsDiffAr) countAr++;
    const bm2 = mean(ss.slice(baseFrom, Math.min(bt, t0)).filter((v) => !Number.isNaN(v)));
    const ps2 = mean(ss.slice(t0 - pre, t0).filter((v) => !Number.isNaN(v)));
    if (ps2 - bm2 >= obsDiffSd) countSd++;
  }
  return { diffAr: obsDiffAr, pAr: (countAr + 1) / (nSurrogates + 1), diffSd: obsDiffSd, pSd: (countSd + 1) / (nSurrogates + 1) };
}

const n = 200, window = 25, pre = 10, nSurr = 100;
console.log('=== Validation (baseline = early 30%) ===');
let nSigAr = 0, nSigSd = 0, nTotal = 0;
for (let rep = 0; rep < 20; rep++) {
  const { x, t0 } = genCSD(n, 1000 + rep, { phi0: 0.2, phi1: 0.95, shift: 1.2 });
  const r = ewsTest(x, t0, window, pre, nSurr, rep * 13 + 1);
  if (!r) continue;
  nTotal++;
  if (r.pAr < 0.05) nSigAr++;
  if (r.pSd < 0.05) nSigSd++;
  if (rep < 3) console.log('  rep=' + rep + ' diffAr=' + r.diffAr.toFixed(3) + ' pAr=' + r.pAr.toFixed(3) + ' diffSd=' + r.diffSd.toFixed(3) + ' pSd=' + r.pSd.toFixed(3));
}
console.log('CSD signal: AR1-EWS sig ' + nSigAr + '/' + nTotal + ' | SD-EWS sig ' + nSigSd + '/' + nTotal);
console.log('');
console.log('=== Control (stationary, early baseline) ===');
let nFPAr = 0, nFPSd = 0, nCtl = 0;
for (let rep = 0; rep < 20; rep++) {
  const { x } = genCSD(n, 5000 + rep, { phi0: 0.3, phi1: 0.3, shift: 0 });
  const r = ewsTest(x, Math.floor(n * 0.75), window, pre, nSurr, rep * 17 + 1);
  if (!r) continue;
  nCtl++;
  if (r.pAr < 0.05) nFPAr++;
  if (r.pSd < 0.05) nFPSd++;
}
console.log('control: AR1 FP ' + nFPAr + '/' + nCtl + ' | SD FP ' + nFPSd + '/' + nCtl);