/*
 * data.js — CSV parsing and preprocessing (type inference, one-hot encoding, missing rows).
 * Works in the browser (window.DataPrep) and in Node (module.exports).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DataPrep = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const MAX_CATEGORIES = 20;

  /** RFC 4180 CSV parser; detects ',', ';', or tab delimiters. */
  function parseCSV(text) {
    text = text.replace(/^﻿/, "");
    const firstLine = text.split(/\r?\n/, 1)[0];
    const delim = [",", ";", "\t"].map((d) => [d, firstLine.split(d).length]).sort((a, b) => b[1] - a[1])[0][0];
    const rows = [];
    let row = [], field = "", inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQuotes) {
        if (ch === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; } else inQuotes = false;
        } else field += ch;
      } else if (ch === '"') inQuotes = true;
      else if (ch === delim) { row.push(field); field = ""; }
      else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(field); rows.push(row); row = []; field = "";
      } else field += ch;
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    const nonEmpty = rows.filter((r) => r.some((v) => v.trim() !== ""));
    if (nonEmpty.length < 2) throw new Error("CSV needs a header row and at least one data row.");
    const header = nonEmpty[0].map((h, i) => h.trim() || `col_${i + 1}`);
    const data = nonEmpty.slice(1).map((r) => header.map((_, i) => (r[i] ?? "").trim()));
    return { header, rows: data };
  }

  const MISSING = new Set(["", "na", "n/a", "nan", "null", "none", "?", "-"]);
  const isMissing = (v) => MISSING.has(String(v).trim().toLowerCase());
  const toNumber = (v) => {
    const s = String(v).trim().replace(/,(?=\d{3}\b)/g, "");
    return s === "" ? NaN : Number(s);
  };

  /** Classify each column as numeric, categorical (<= MAX_CATEGORIES levels) or ignored text. */
  function profileColumns({ header, rows }) {
    return header.map((name, j) => {
      const vals = rows.map((r) => r[j]).filter((v) => !isMissing(v));
      const numeric = vals.filter((v) => Number.isFinite(toNumber(v)));
      const missing = rows.length - vals.length;
      if (vals.length && numeric.length / vals.length >= 0.95) {
        const nums = numeric.map(toNumber);
        const levels = new Set(nums);
        return { name, index: j, kind: "numeric", missing, binary: levels.size === 2 };
      }
      const levels = [...new Set(vals)].sort();
      if (levels.length >= 2 && levels.length <= MAX_CATEGORIES) return { name, index: j, kind: "categorical", levels, missing };
      return { name, index: j, kind: "text", missing, reason: levels.length < 2 ? "constant" : `${levels.length} distinct values` };
    });
  }

  /**
   * Build the numeric design matrix.
   * Categorical predictors are one-hot encoded with the first (alphabetical) level as the baseline.
   * Rows with a missing value in the target or any used predictor are dropped.
   */
  function buildMatrix(dataset, profile, targetName, predictorNames) {
    const target = profile.find((c) => c.name === targetName);
    if (!target || target.kind !== "numeric") throw new Error("Target must be a numeric column.");
    const predictors = profile.filter((c) => predictorNames.includes(c.name) && c.name !== targetName && c.kind !== "text");
    const features = [];
    for (const c of predictors) {
      if (c.kind === "numeric") features.push({ name: c.name, source: c.name, type: "numeric" });
      else c.levels.slice(1).forEach((lv) => features.push({ name: `${c.name}_${lv}`, source: c.name, type: "dummy", level: lv, baseline: c.levels[0] }));
    }
    const X = [], y = [], kept = [];
    dataset.rows.forEach((r, i) => {
      const yt = toNumber(r[target.index]);
      if (!Number.isFinite(yt) || isMissing(r[target.index])) return;
      const row = [];
      for (const c of predictors) {
        const v = r[c.index];
        if (isMissing(v)) return;
        if (c.kind === "numeric") {
          const num = toNumber(v);
          if (!Number.isFinite(num)) return;
          row.push(num);
        } else c.levels.slice(1).forEach((lv) => row.push(v === lv ? 1 : 0));
      }
      X.push(row); y.push(yt); kept.push(i);
    });
    // drop zero-variance encoded features (e.g. a level that vanished with missing rows)
    const keep = features.map((_, j) => X.length > 0 && X.some((r) => r[j] !== X[0][j]));
    return {
      X: X.map((r) => r.filter((_, j) => keep[j])),
      y, kept,
      features: features.filter((_, j) => keep[j]),
      predictors,
      target,
      dropped: dataset.rows.length - X.length,
    };
  }

  return { parseCSV, profileColumns, buildMatrix, toNumber, isMissing, MAX_CATEGORIES };
});
