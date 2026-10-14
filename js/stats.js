/*
 * stats.js — a small, dependency-free statistics engine for multiple linear regression.
 * Runs in the browser (window.MLR) and in Node (module.exports) so the math can be unit-tested.
 *
 * Contents
 *   - seeded RNG + train/test split + k-fold assignment
 *   - dense linear algebra (Gram matrices, Gauss-Jordan inverse)
 *   - OLS with full inference (SE, t, p, 95% CI, F-test, AIC/BIC, VIF)
 *   - distributions (incomplete beta -> Student t, F)
 *   - 5 feature-selection algorithms that each return a nested subset for every k
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.MLR = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------------------------------------------------------- RNG / splitting
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffledIndices(n, seed) {
    const rand = mulberry32(seed);
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }

  function trainTestSplit(n, testRatio, seed) {
    const idx = shuffledIndices(n, seed);
    const nTest = Math.max(1, Math.round(n * testRatio));
    return { test: idx.slice(0, nTest).sort((a, b) => a - b), train: idx.slice(nTest).sort((a, b) => a - b) };
  }

  function foldAssignment(n, k, seed) {
    const idx = shuffledIndices(n, seed + 7919);
    const fold = new Array(n);
    idx.forEach((row, i) => (fold[row] = i % k));
    return fold;
  }

  // ---------------------------------------------------------------- linear algebra
  /** Inverse of a symmetric positive (semi)definite matrix; returns null when singular. */
  function invert(A) {
    const n = A.length;
    const M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
    const scale = Math.max(1e-300, ...A.map((r, i) => Math.abs(r[i])));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (Math.abs(M[piv][c]) < 1e-12 * scale) return null;
      [M[c], M[piv]] = [M[piv], M[c]];
      const p = M[c][c];
      for (let j = 0; j < 2 * n; j++) M[c][j] /= p;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
      }
    }
    return M.map((row) => row.slice(n));
  }

  /** Design matrix with leading intercept column, restricted to `cols`. */
  function design(X, cols) {
    return X.map((row) => [1, ...cols.map((c) => row[c])]);
  }

  function gram(Z, y) {
    const p = Z[0].length;
    const ZtZ = Array.from({ length: p }, () => new Array(p).fill(0));
    const Zty = new Array(p).fill(0);
    for (let i = 0; i < Z.length; i++) {
      const z = Z[i];
      for (let a = 0; a < p; a++) {
        Zty[a] += z[a] * y[i];
        for (let b = a; b < p; b++) ZtZ[a][b] += z[a] * z[b];
      }
    }
    for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) ZtZ[a][b] = ZtZ[b][a];
    return { ZtZ, Zty };
  }

  const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

  // ---------------------------------------------------------------- distributions
  function lnGamma(x) {
    const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x;
    let tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (const c of g) ser += c / ++y;
    return -tmp + Math.log((2.5066282746310005 * ser) / x);
  }

  function betacf(a, b, x) {
    const MAXIT = 300, EPS = 3e-16, FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  /** Regularized incomplete beta I_x(a, b). */
  function ibeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
  }

  /** Two-sided p-value of a t statistic. */
  const tPValue = (t, df) => (Number.isFinite(t) ? ibeta(df / (df + t * t), df / 2, 0.5) : 0);
  /** Upper-tail p-value of an F statistic. */
  const fPValue = (F, d1, d2) => (Number.isFinite(F) && F > 0 ? ibeta(d2 / (d2 + d1 * F), d2 / 2, d1 / 2) : F > 0 ? 0 : 1);

  /** Two-sided critical value t* such that P(|T| > t*) = alpha. */
  function tCritical(alpha, df) {
    let lo = 0, hi = 1000;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (tPValue(mid, df) > alpha) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  // ---------------------------------------------------------------- OLS
  /**
   * Ordinary least squares with an intercept on the columns `cols` of X.
   * Returns coefficients and full inferential statistics, or null if X'X is singular.
   */
  function ols(X, y, cols, { inference = true } = {}) {
    const n = y.length;
    const Z = design(X, cols);
    const p = cols.length + 1;
    const { ZtZ, Zty } = gram(Z, y);
    const inv = invert(ZtZ);
    if (!inv) return null;
    const beta = inv.map((row) => dot(row, Zty));
    const fitted = Z.map((z) => dot(z, beta));
    const resid = y.map((v, i) => v - fitted[i]);
    const ybar = mean(y);
    const sse = resid.reduce((s, r) => s + r * r, 0);
    const sst = y.reduce((s, v) => s + (v - ybar) ** 2, 0);
    const df = n - p;
    const r2 = sst > 0 ? 1 - sse / sst : 0;
    const model = {
      cols, beta, n, p, df, sse, sst, r2,
      adjR2: df > 0 ? 1 - ((1 - r2) * (n - 1)) / df : NaN,
      rmse: Math.sqrt(sse / n),
      fitted, resid,
      predictRow: (row) => beta[0] + cols.reduce((s, c, j) => s + beta[j + 1] * row[c], 0),
    };
    if (!inference) return model;

    const sigma2 = df > 0 ? sse / df : NaN;
    const se = inv.map((row, i) => Math.sqrt(Math.max(0, sigma2 * row[i])));
    const tStat = beta.map((b, i) => b / se[i]);
    const pVal = tStat.map((t) => (df > 0 ? tPValue(t, df) : NaN));
    const tc = df > 0 ? tCritical(0.05, df) : NaN;
    const k = p - 1;
    const F = k > 0 && df > 0 ? ((sst - sse) / k) / sigma2 : NaN;
    const ll = -(n / 2) * (Math.log(2 * Math.PI) + Math.log(sse / n) + 1);
    return Object.assign(model, {
      sigma: Math.sqrt(sigma2), se, tStat, pVal, xtxInv: inv, tCrit: tc,
      ciLow: beta.map((b, i) => b - tc * se[i]),
      ciHigh: beta.map((b, i) => b + tc * se[i]),
      F, Fp: k > 0 && df > 0 ? fPValue(F, k, df) : NaN,
      logLik: ll, aic: -2 * ll + 2 * p, bic: -2 * ll + Math.log(n) * p,
    });
  }

  /** Variance inflation factor of every column in `cols`. */
  function vif(X, cols) {
    if (cols.length < 2) return cols.map(() => 1);
    return cols.map((c) => {
      const xc = X.map((row) => row[c]);
      const m = ols(X, xc, cols.filter((o) => o !== c), { inference: false });
      return m && m.r2 < 1 ? 1 / (1 - m.r2) : Infinity;
    });
  }

  function metrics(yTrue, yPred) {
    const n = yTrue.length;
    const ybar = mean(yTrue);
    let sse = 0, sst = 0, sae = 0;
    for (let i = 0; i < n; i++) {
      const e = yTrue[i] - yPred[i];
      sse += e * e; sae += Math.abs(e); sst += (yTrue[i] - ybar) ** 2;
    }
    return { rmse: Math.sqrt(sse / n), mae: sae / n, r2: sst > 0 ? 1 - sse / sst : NaN };
  }

  function cvRmse(X, y, cols, folds, k) {
    let sse = 0, cnt = 0;
    for (let f = 0; f < k; f++) {
      const tr = [], te = [];
      folds.forEach((g, i) => (g === f ? te : tr).push(i));
      const m = ols(tr.map((i) => X[i]), tr.map((i) => y[i]), cols, { inference: false });
      if (!m) return Infinity;
      for (const i of te) { const e = y[i] - m.predictRow(X[i]); sse += e * e; cnt++; }
    }
    return Math.sqrt(sse / cnt);
  }

  // ---------------------------------------------------------------- standardization / Lasso
  function standardize(X) {
    const p = X[0].length;
    const mu = Array.from({ length: p }, (_, j) => mean(X.map((r) => r[j])));
    const sd = mu.map((m, j) => Math.sqrt(mean(X.map((r) => (r[j] - m) ** 2))) || 1);
    return { Xs: X.map((r) => r.map((v, j) => (v - mu[j]) / sd[j])), mu, sd };
  }

  /**
   * Lasso / ElasticNet regularization path via coordinate descent on standardized data.
   * Ranks features by the lambda at which they first become non-zero.
   */
  function penalizedRanking(X, y, l1Ratio) {
    const { Xs } = standardize(X);
    const n = Xs.length, p = Xs[0].length;
    const ybar = mean(y);
    const yc = y.map((v) => v - ybar);
    const corr = Array.from({ length: p }, (_, j) => Math.abs(Xs.reduce((s, r, i) => s + r[j] * yc[i], 0)) / n);
    const lamMax = Math.max(...corr) / l1Ratio;
    const steps = 120;
    const b = new Array(p).fill(0);
    const r = yc.slice();
    const entered = new Map();
    for (let s = 0; s < steps; s++) {
      const lam = lamMax * Math.pow(1e-4, s / (steps - 1));
      for (let it = 0; it < 500; it++) {
        let maxDelta = 0;
        for (let j = 0; j < p; j++) {
          let rho = 0;
          for (let i = 0; i < n; i++) rho += Xs[i][j] * (r[i] + Xs[i][j] * b[j]);
          rho /= n;
          const soft = Math.sign(rho) * Math.max(0, Math.abs(rho) - lam * l1Ratio);
          const nb = soft / (1 + lam * (1 - l1Ratio));
          const delta = nb - b[j];
          if (delta !== 0) {
            for (let i = 0; i < n; i++) r[i] -= Xs[i][j] * delta;
            b[j] = nb;
            maxDelta = Math.max(maxDelta, Math.abs(delta));
          }
        }
        if (maxDelta < 1e-9) break;
      }
      // features entering at the same lambda are ordered by coefficient magnitude
      const fresh = b.map((v, j) => [j, Math.abs(v)]).filter(([j, v]) => v > 1e-12 && !entered.has(j)).sort((a, c) => c[1] - a[1]);
      for (const [j] of fresh) entered.set(j, s);
      if (entered.size === p) break;
    }
    const order = [...entered.keys()];
    for (let j = 0; j < p; j++) if (!entered.has(j)) order.push(j);
    return order;
  }

  // ---------------------------------------------------------------- feature selection
  const nestedFromOrder = (order) => order.map((_, k) => order.slice(0, k + 1));

  function forwardSelection(X, y, folds, nFolds) {
    const p = X[0].length;
    const chosen = [], subsets = [];
    const remaining = new Set(Array.from({ length: p }, (_, j) => j));
    while (remaining.size) {
      let best = null, bestScore = Infinity;
      for (const j of remaining) {
        const s = cvRmse(X, y, [...chosen, j], folds, nFolds);
        if (s < bestScore) { bestScore = s; best = j; }
      }
      if (best === null) best = remaining.values().next().value;
      chosen.push(best); remaining.delete(best);
      subsets.push(chosen.slice());
    }
    return subsets;
  }

  function backwardElimination(X, y, folds, nFolds) {
    const p = X[0].length;
    let current = Array.from({ length: p }, (_, j) => j);
    const subsets = [current.slice()];
    while (current.length > 1) {
      let drop = null, bestScore = Infinity;
      for (const j of current) {
        const s = cvRmse(X, y, current.filter((c) => c !== j), folds, nFolds);
        if (s < bestScore) { bestScore = s; drop = j; }
      }
      if (drop === null) drop = current[current.length - 1];
      current = current.filter((c) => c !== drop);
      subsets.unshift(current.slice());
    }
    return subsets;
  }

  /** Recursive feature elimination: repeatedly drop the smallest |standardized coefficient|. */
  function rfe(X, y) {
    const { Xs } = standardize(X);
    let current = Array.from({ length: X[0].length }, (_, j) => j);
    const subsets = [current.slice()];
    while (current.length > 1) {
      const m = ols(Xs, y, current, { inference: false });
      let drop;
      if (!m) drop = current[current.length - 1];
      else {
        let minAbs = Infinity;
        current.forEach((c, i) => { if (Math.abs(m.beta[i + 1]) < minAbs) { minAbs = Math.abs(m.beta[i + 1]); drop = c; } });
      }
      current = current.filter((c) => c !== drop);
      subsets.unshift(current.slice());
    }
    return subsets;
  }

  /** Best-subset search: for every k, the subset with the smallest training SSE (uses one Gram matrix). */
  function exhaustive(X, y) {
    const p = X[0].length;
    const { ZtZ, Zty } = gram(design(X, Array.from({ length: p }, (_, j) => j)), y);
    const yty = dot(y, y);
    const best = new Array(p + 1).fill(null).map(() => ({ sse: Infinity, cols: null }));
    for (let mask = 1; mask < 1 << p; mask++) {
      const cols = [];
      for (let j = 0; j < p; j++) if (mask & (1 << j)) cols.push(j);
      const idx = [0, ...cols.map((c) => c + 1)];
      const inv = invert(idx.map((a) => idx.map((b) => ZtZ[a][b])));
      if (!inv) continue;
      const zy = idx.map((a) => Zty[a]);
      const beta = inv.map((row) => dot(row, zy));
      const sse = yty - dot(beta, zy);
      if (sse < best[cols.length].sse) best[cols.length] = { sse, cols };
    }
    return best.slice(1).map((b, k) => b.cols || Array.from({ length: k + 1 }, (_, j) => j));
  }

  const EXHAUSTIVE_MAX = 16;

  const ALGORITHMS = [
    { id: "forward", name: "Forward Selection (CV)", run: (X, y, f, k) => forwardSelection(X, y, f, k) },
    { id: "backward", name: "Backward Elimination (CV)", run: (X, y, f, k) => backwardElimination(X, y, f, k) },
    { id: "rfe", name: "RFE (std. coefficients)", run: (X, y) => rfe(X, y) },
    { id: "lasso", name: "Lasso Path (L1)", run: (X, y) => nestedFromOrder(penalizedRanking(X, y, 1)) },
    { id: "exhaustive", name: "Exhaustive Best Subset", run: (X, y) => exhaustive(X, y), maxP: EXHAUSTIVE_MAX },
  ];

  /**
   * Runs every algorithm on the training set and scores every (algorithm, k) on the test set.
   * Returns { results: [{ id, name, skipped, steps: [{ k, cols, trainR2, adjR2, cvRmse, testRmse, testR2, testMae }] }] }
   */
  function runFeatureSelection(Xtr, ytr, Xte, yte, { nFolds = 5, seed = 42 } = {}) {
    const p = Xtr[0].length;
    const folds = foldAssignment(ytr.length, nFolds, seed);
    return ALGORITHMS.map((alg) => {
      if (alg.maxP && p > alg.maxP) return { id: alg.id, name: alg.name, skipped: `skipped: p = ${p} > ${alg.maxP}`, steps: [] };
      const t0 = Date.now();
      const subsets = alg.run(Xtr, ytr, folds, nFolds);
      const steps = subsets.map((cols) => {
        const m = ols(Xtr, ytr, cols, { inference: false });
        if (!m) return { k: cols.length, cols, invalid: true };
        const te = metrics(yte, Xte.map((r) => m.predictRow(r)));
        return {
          k: cols.length, cols, trainR2: m.r2, adjR2: m.adjR2, trainRmse: m.rmse,
          cvRmse: cvRmse(Xtr, ytr, cols, folds, nFolds),
          testRmse: te.rmse, testR2: te.r2, testMae: te.mae,
        };
      });
      return { id: alg.id, name: alg.name, ms: Date.now() - t0, steps };
    });
  }

  return {
    mulberry32, trainTestSplit, foldAssignment, invert, ols, vif, metrics, cvRmse, standardize,
    penalizedRanking, forwardSelection, backwardElimination, rfe, exhaustive, runFeatureSelection,
    tPValue, fPValue, tCritical, ibeta, mean, ALGORITHMS, EXHAUSTIVE_MAX,
  };
});
