function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN; }
function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }
// fixed: no double scaling, zero-padded to pow2
function phaseRandomizeFixed(x, rng) {
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
  fft(re2, im2, true);   // already divides by N inside
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

let s = 42 >>> 0;
const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
const x = Array.from({ length: 121 }, (_, i) => Math.sin(i / 5) + 0.3 * (Math.random() - 0.5));
const xm = mean(x);
const varOrig = x.reduce((a, v) => a + (v - xm) ** 2, 0) / x.length;
const surrs = [];
for (let k = 0; k < 50; k++) {
  const y = phaseRandomizeFixed(x, rng);
  const ym = mean(y);
  surrs.push(y.reduce((a, v) => a + (v - ym) ** 2, 0) / y.length);
}
console.log('orig var = ' + varOrig.toFixed(4));
console.log('surrogate var mean = ' + (surrs.reduce((a, b) => a + b, 0) / 50).toFixed(4));
console.log('all finite: ' + phaseRandomizeFixed(x, rng).every(Number.isFinite));
// Also: AR(1) structure should be destroyed while var preserved
function ar1(seq) {
  const y = seq.slice(1), X = seq.slice(0, -1);
  const mX = mean(X), mY = mean(y);
  let sxx = 0, sxy = 0;
  for (let j = 0; j < X.length; j++) { sxx += (X[j] - mX) ** 2; sxy += (X[j] - mX) * (y[j] - mY); }
  return sxx < 1e-12 ? NaN : sxy / sxx;
}
const xAC = Array.from({ length: 256 }, (_, i) => { let v = 0; for (let k = 0; k <= i; k++) v = 0.8 * v + (Math.random() - 0.5); return v; });
console.log('AR(1) original = ' + ar1(xAC).toFixed(3) + ' | surrogate = ' + ar1(phaseRandomizeFixed(xAC, rng)).toFixed(3) + ' (should be ~0)');