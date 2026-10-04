// simulate_ema.mjs —— 从 OED 方程生成合成 EMA 数据（Node 版）
// 含"抄后路式恢复相变"：抑郁组中一半个体在序列中点附近经历
// M 与 Δ_target 的渐变修正（修复过程），用于演示 P2 临界慢降检测。
import { DEFAULT_PARAMS, simulateEuler, makeRng } from './oed_model.mjs';

// 理论变量 → 1..5 问卷条目（sigmoid 压缩 + 噪声）
function toItems(vals, noise, rng) {
  return vals.map((v) => {
    const x = v + rng.normal() * noise;
    const s = 1 + 4 / (1 + Math.exp(-2.2 * (x - 0.5)));
    return Math.max(1, Math.min(5, s));
  });
}

export function simulateEMA({
  nPeople = 20, nBeeps = 120, dt = 1.0, obsNoise = 0.25,
  processNoise = 0.02, seed = 42, depressionProp = 0.5,
  eventRate = 0.22, eventImpact = 0.5,
  recoverProp = 0.5,   // 抑郁组中经历恢复相变的比例
  rampLen = 24,        // Δ_target 修正的渐变时长（beep）
} = {}) {
  const rng = makeRng(seed);
  const nDep = Math.round(nPeople * depressionProp);
  const people = [];
  const t0 = Math.floor(nBeeps / 2);

  for (let i = 0; i < nPeople; i++) {
    const dep = i < nDep;
    const recovering = dep && (i - 0) % 2 === 0;      // 抑郁组交替：一半恢复
    const p = { ...DEFAULT_PARAMS };
    if (dep) {
      p.delta_target = -0.4; p.theta_S0 = 0.65; p.A_baseline = 0.35;
    } else {
      p.delta_target = 0.3; p.theta_S0 = 0.4; p.A_baseline = 0.6;
    }

    // 恢复组：Δ_target 从 -0.4 渐变修正至 +0.2（抄后路：期待修正）
    // 相变（符号翻转）发生在 Δ_target 穿越 0 的时刻——这是 P2 的"转换点"
    let transitionBeep = null;
    const rampLenLoc = recovering ? Math.max(30, Math.min(Math.floor(nBeeps * 0.4), 90)) : 0; // slower ramp -> real CSD window
    if (recovering) {
      const baseDT = p.delta_target, targetDT = 0.25;
      p.delta_target = Array.from({ length: nBeeps }, (_, j) => {
        if (j < t0) return baseDT;
        if (j >= t0 + rampLenLoc) return targetDT;
        return baseDT + (targetDT - baseDT) * ((j - t0) / rampLenLoc);
      });
      // 符号翻转点：|base|/(|base|+target) 比例处
      const crossFrac = Math.abs(baseDT) / (Math.abs(baseDT) + Math.abs(targetDT));
      transitionBeep = t0 + Math.min(rampLenLoc, Math.floor(rampLenLoc * crossFrac));
    }

    // M(t)：基线 + 波动；恢复组在中点后 M 渐变升高（新安全客体在场）
    const Mbase = dep ? 0.25 : 0.72;
    const M = [];
    const evPos = [], evNeg = [];
    for (let j = 0; j < nBeeps; j++) {
      let m = Mbase + 0.15 * rng.normal();
      if (recovering && j >= t0 && j < t0 + rampLenLoc) {
        m += (0.60 - Mbase) * ((j - t0) / rampLenLoc);
      } else if (recovering && j >= t0 + rampLenLoc) {
        m += (0.60 - Mbase);
      }
      const ep = rng.next() < eventRate ? 1 : 0;
      const en = rng.next() < eventRate ? 1 : 0;
      evPos.push(ep); evNeg.push(en);
      m = m + eventImpact * ep * (1 - m);
      M.push(Math.max(0.02, Math.min(1, m)));
    }

    const sim = simulateEuler(M, dt, null, p);
    const N = sim.t.length;

    // 事件注入（体现公理三 λ>1）
    const evPosFull = (j) => (evPos[j] || 0);
    const evNegFull = (j) => (evNeg[j] || 0);
    const injE = sim.E.map((e, j) => e + 0.4 * eventImpact * evPosFull(j));
    const injD = sim.D.map((d, j) => d + 0.4 * p.lam * eventImpact * evNegFull(j));

    // 观测条目
    const pa = toItems(injE, obsNoise, rng);
    const na = toItems(injD, obsNoise, rng);
    const understood = toItems(sim.M, obsNoise, rng);
    const expect = toItems(sim.delta, obsNoise, rng);
    const ruminat = toItems(sim.H, obsNoise, rng);
    const suppress = toItems(sim.S, obsNoise, rng);
    const hopeless = toItems(sim.mA, obsNoise, rng);
    const recover = toItems(sim.U, obsNoise, rng);

    const meanNA = na.reduce((a, b) => a + b, 0) / N;
    const phq9 = dep
      ? Math.round(3 + 18 * (0.2 + 0.8 * (meanNA / 5)))
      : Math.round(4 * meanNA / 5);

    const rows = [];
    for (let j = 0; j < N; j++) {
      rows.push({
        id: i, beep: j, t: sim.t[j], dep: dep ? 1 : 0, phq9,
        recovering: recovering ? 1 : 0,
        transition: (recovering && j === t0) ? 1 : 0,
        E_true: sim.E[j], D_true: sim.D[j], M_true: sim.M[j],
        delta_true: sim.delta[j], H_true: sim.H[j], mA_true: sim.mA[j],
        U_true: sim.U[j], S_true: sim.S[j],
        pa1: pa[j], pa2: pa[j], pa3: pa[j], pa4: pa[j], pa5: pa[j],
        na1: na[j], na2: na[j], na3: na[j], na4: na[j], na5: na[j],
        understood: understood[j], lonely: 6 - understood[j],
        expect: expect[j], ruminat: ruminat[j], suppress: suppress[j],
        hopeless: hopeless[j], recover: recover[j],
        event_pos: evPos[j], event_neg: evNeg[j],
      });
    }
    people.push({ id: i, dep, recovering, transitionBeep, params: p, rows });
  }
  return people;
}

// 聚合条目 → 理论变量（等价于真实数据的 mapping 步骤）
export function buildDerived(person) {
  return person.rows.map((r) => {
    const pa = (r.pa1 + r.pa2 + r.pa3 + r.pa4 + r.pa5) / 5;
    const na = (r.na1 + r.na2 + r.na3 + r.na4 + r.na5) / 5;
    return {
      ...r,
      E: pa, D: na,
      M: (r.understood + (6 - r.lonely)) / 2,
      delta: r.expect, H: r.ruminat, S: r.suppress, mA: r.hopeless, U: r.recover,
      intensity: 1.0,
      event_type: r.event_neg === 1 ? 1 : 0,
    };
  });
}
