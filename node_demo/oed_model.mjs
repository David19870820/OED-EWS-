// oed_model.mjs —— OED 主链方程的纯 JS 参考实现（无依赖）
// 与 python/oed_empirical/model.py 逻辑一致。

export const DEFAULT_PARAMS = {
  A: 1.0, A_baseline: 0.6, lam: 1.5, delta_max: 1.0, k: 2.0,
  delta_target: 0.3, kappa: 3.0, theta_S0: 0.4, eta: 1.2, h0: 1.0,
  alpha_delta: 0.8, beta_delta: 0.6, gamma_self: 0.15,
  gamma: 0.5, lambda_H: 0.3, alpha_O: 0.6, lambda_O: 0.4,
  gamma_des: 0.3, A_actual: 0.15,
};

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));
const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

export function gFunction(M, delta, deltaTarget, deltaMax = 1.0) {
  const f = clip(delta / deltaMax, 0, 1);
  return Math.abs(M) * f * Math.sign(deltaTarget);
}

export function EFrom(M, delta, deltaTarget, p = DEFAULT_PARAMS) {
  const g = gFunction(M, delta, deltaTarget, p.delta_max);
  return p.A * g * Math.sign(deltaTarget); // Θ = 1
}

export function DFrom(M, delta, deltaTarget, p = DEFAULT_PARAMS) {
  const g = gFunction(M, delta, deltaTarget, p.delta_max);
  return p.A * (1 - g);
}

export function DeltaFrom(M, delta, deltaTarget, p = DEFAULT_PARAMS) {
  return EFrom(M, delta, deltaTarget, p) - p.lam * DFrom(M, delta, deltaTarget, p);
}

export function UFrom(M, delta, deltaTarget, p = DEFAULT_PARAMS) {
  const d = DeltaFrom(M, delta, deltaTarget, p);
  return Math.sign(deltaTarget) * sigmoid(p.k * Math.abs(d - deltaTarget));
}

export function SFrom(delta, H, p = DEFAULT_PARAMS) {
  const thetaS = p.theta_S0 + p.eta * (H / (H + p.h0));
  return sigmoid(p.kappa * (thetaS - delta));
}

// 欧拉积分：输入 M 序列，返回各状态/派生序列（长度 n+1）
// 支持时变 Δ_target：p.delta_target 可为数值或与 M 等长的数组
// （对应"修复过程中 Δ_target 被逐步修正"的动力学——计算框架模拟2）
export function simulateEuler(M, dt = 1.0, x0 = null, p = DEFAULT_PARAMS) {
  const n = M.length;
  const dTarget = (i) => (Array.isArray(p.delta_target) ? p.delta_target[i] : p.delta_target);
  let delta = x0 ? x0[0] : Math.max(dTarget(0), 0.05);
  let H = x0 ? x0[1] : 0.0;
  let mA = x0 ? x0[2] : p.A_actual;

  const out = { t: [], delta: [], H: [], mA: [], E: [], D: [], Delta: [], U: [], S: [], M: [] };
  const push = (t, Mv, dT) => {
    out.t.push(t); out.M.push(Mv); out.delta.push(delta);
    out.H.push(H); out.mA.push(mA);
    out.E.push(EFrom(Mv, delta, dT, p));
    out.D.push(DFrom(Mv, delta, dT, p));
    out.Delta.push(DeltaFrom(Mv, delta, dT, p));
    out.U.push(UFrom(Mv, delta, dT, p));
    out.S.push(SFrom(delta, H, p));
  };

  push(0, M[0], dTarget(0));
  for (let i = 0; i < n; i++) {
    const Mv = clip(M[i], 0.02, 1.0);
    const dT = dTarget(i);
    const E = EFrom(Mv, delta, dT, p);
    const D = DFrom(Mv, delta, dT, p);
    const U = UFrom(Mv, delta, dT, p);
    const S = SFrom(delta, H, p);
    const Dblocked = D * S;
    const dDelta = p.alpha_delta * (Mv - delta) - p.beta_delta * H
      - p.gamma_self * U * (p.A - p.A_baseline);
    const dH = p.gamma * Dblocked - p.lambda_H * H;
    const dmA = p.alpha_O * (1 - Mv) * (1 - delta) * H
      - p.lambda_O * mA * E - p.gamma_des * (mA - p.A_actual);
    delta = Math.max(0, delta + dDelta * dt);
    H = Math.max(0, H + dH * dt);
    mA = Math.max(0, mA + dmA * dt);
    push((i + 1) * dt, M[i], dT);
  }
  return out;
}

// 简单的 LCG 伪随机（确定性，便于复现）
export function makeRng(seed = 42) {
  let s = seed >>> 0;
  const next = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  return {
    next,
    normal() {
      let u = 0, v = 0;
      while (u === 0) u = next();
      while (v === 0) v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
  };
}
