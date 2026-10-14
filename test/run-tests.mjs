// Sanity tests for the regression engine against published reference values.
// Usage: node test/run-tests.mjs   (needs network access to fetch the two datasets)
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const MLR = require("../js/stats.js");
const DataPrep = require("../js/data.js");

let failures = 0;
function check(label, actual, expected, tol) {
  const ok = Math.abs(actual - expected) <= tol;
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}: ${actual.toFixed(6)} (expected ${expected} ± ${tol})`);
}

async function load(url, target) {
  const ds = DataPrep.parseCSV(await (await fetch(url)).text());
  const profile = DataPrep.profileColumns(ds);
  const preds = profile.filter((c) => c.name !== target).map((c) => c.name);
  return DataPrep.buildMatrix(ds, profile, target, preds);
}

// Distributions (reference values from scipy.stats / numerical integration)
check("t two-sided p (t=2.0, df=10)", MLR.tPValue(2, 10), 0.0733880, 1e-6);
check("t critical 95% (df=44)", MLR.tCritical(0.05, 44), 2.015368, 1e-5);
check("F upper p (F=3, 4, 20)", MLR.fPValue(3, 4, 20), 0.0432010, 1e-6);

// 50 Startups — full-sample OLS (statsmodels reference)
const s = await load("https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv", "Profit");
console.log("50 Startups features:", s.features.map((f) => f.name).join(", "));
const sm = MLR.ols(s.X, s.y, s.features.map((_, j) => j));
check("50S R²", sm.r2, 0.9508, 5e-4);
check("50S adj R²", sm.adjR2, 0.9452, 5e-4);
check("50S β R&D Spend", sm.beta[1], 0.8060, 5e-4);
check("50S β Marketing", sm.beta[3], 0.0270, 5e-4);
check("50S F-statistic", sm.F, 169.9, 0.1);

// Boston Housing — full-sample OLS (classic reference: R² 0.7406, intercept 36.4595)
const b = await load("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv", "medv");
const bm = MLR.ols(b.X, b.y, b.features.map((_, j) => j));
check("Boston R²", bm.r2, 0.7406, 5e-4);
check("Boston adj R²", bm.adjR2, 0.7338, 5e-4);
check("Boston intercept", bm.beta[0], 36.4595, 1e-3);
check("Boston F-statistic", bm.F, 108.1, 0.1);
const v = MLR.vif(b.X, b.features.map((_, j) => j));
check("Boston VIF(tax)", v[b.features.findIndex((f) => f.name === "tax")], 9.0086, 1e-3);

// Feature selection end-to-end
for (const [name, d] of [["50 Startups", s], ["Boston", b]]) {
  const split = MLR.trainTestSplit(d.y.length, 0.2, 42);
  const pick = (idx, arr) => idx.map((i) => arr[i]);
  const t0 = Date.now();
  const res = MLR.runFeatureSelection(pick(split.train, d.X), pick(split.train, d.y), pick(split.test, d.X), pick(split.test, d.y));
  console.log(`\n${name}: feature selection in ${Date.now() - t0} ms`);
  for (const r of res) {
    if (r.skipped) { console.log(`  ${r.name}: ${r.skipped}`); continue; }
    const best = r.steps.reduce((a, c) => (c.testRmse < a.testRmse ? c : a));
    console.log(`  ${r.name.padEnd(28)} best k=${best.k} testRMSE=${best.testRmse.toFixed(3)} R²=${best.testR2.toFixed(4)}  [${best.cols.map((c) => d.features[c].name).join(", ")}]`);
    if (r.steps.length !== d.features.length || r.steps.some((st, i) => st.cols.length !== i + 1)) { failures++; console.log("  FAIL: subsets not k = 1..p"); }
  }
}

console.log(failures ? `\n${failures} check(s) failed` : "\nAll checks passed");
process.exit(failures ? 1 : 0);
