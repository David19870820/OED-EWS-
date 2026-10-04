// Verify: does phaseRandomize produce NaN for non-power-of-2 lengths?
import { simulateEMA, buildDerived } from '../node_demo/simulate_ema.mjs';
function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN; }
function detrend(x) {
  const n = x.length;
  const t = Array.from({ length: n }, (_, i) => i);
  const mt = mean(t), mx = mean(x);
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (t[i] - mt) ** 2; sxy += (t[i] - mt) * (x[i] - mx); }
  const b = sxx ? sxy / sxx : 0, a = mx - b * mt;
  return x.map((v, i) => v - (a + b * t[i]));
}
function phaseRandomize(x, rng) {
  const n = x.length;
  const re = new Float64Array(n), im = new Float64Array(n);
  const mx = mean(x);
  for (let i = 0; i < n; i++) re[i] = x[i] - mx;
  fft(re, im, false);
  const half = Math.floor(n / 2) + 1;
  const amp = [], ph = [];
  for (let k = 0; k < half; k++) {
    amp.push(Math.hypot(re[k], im[k]));
    ph.push(k === 0 || (n % 2 === 0 && k === half - 1) ? 0 : rng() * 2 * Math.PI);
  }
  const re2 = new Float64Array(n), im2 = new Float64Array(n);
  for (let k = 0; k < half; k++) {
    const c = Math.cos(ph[k]) * amp[k], s = Math.sin(ph[k]) * amp[k];
    re2[k] = c; im2[k] = s;
    if (k > 0 && k < half - (n % 2 === 0 ? 1 : 0)) { re2[n - k] = c; im2[n - k] = -s; }
  }
  fft(re2, im2, true);
  return Array.from(re2, (v) => v / n).map((v) => v + mx);
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

let s = 42 >>> 0;
const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };

// Test 1: power of 2 (128)
const x128 = Array.from({ length: 128 }, () => rng());
const y128 = phaseRandomize(x128, rng);
console.log('n=128 (power of 2): NaNs = ' + y128.filter((v) => Number.isNaN(v)).length + ' | range=[' + Math.min(...y128).toFixed(3) + ',' + Math.max(...y128).toFixed(3) + ']');

// Test 2: non-power of 2 (120)
const x120 = Array.from({ length: 120 }, () => rng());
const y120 = phaseRandomize(x120, rng);
console.log('n=120 (non-power): NaNs = ' + y120.filter((v) => Number.isNaN(v)).length + ' | finite=' + y120.filter((v) => Number.isFinite(v)).length);

// Test 3: actual data length
const people = simulateEMA({ nPeople: 2, nBeeps: 120, seed: 42, depressionProp: 0.5 });
const derived = people.map((p) => ({ ...p, rows: buildDerived(p) }));
const x = detrend(derived[0].rows.map((r) => r.D));
console.log('real data length = ' + x.length + ', finite = ' + x.filter(Number.isFinite).length);
const yreal = phaseRandomize(x, rng);
console.log('phaseRandomize(real): NaNs = ' + yreal.filter((v) => Number.isNaN(v)).length + ' | finite = ' + yreal.filter(Number.isFinite).length);