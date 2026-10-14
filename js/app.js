/* app.js — UI wiring for Regression Lab. Depends on Chart.js, MLR (stats.js) and DataPrep (data.js). */
(function () {
  "use strict";

  const PRESETS = {
    startups: {
      url: "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv",
      target: "Profit",
      problem:
        "<b>Kaggle · 50 Startups</b>：創投想知道一間新創公司的 <b>Profit（利潤）</b> 主要由哪些支出決定。" +
        "資料包含 50 家公司的 R&amp;D Spend（研發）、Administration（行政）、Marketing Spend（行銷）與所在州 State。" +
        "目標：建立多元線性迴歸，找出最精簡、泛化能力最好的特徵組合，避免共線性與過度擬合。",
    },
    boston: {
      url: "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
      target: "medv",
      problem:
        "<b>Boston Housing</b>：預測波士頓各區自有住宅的中位數房價 <b>medv</b>（單位：千美元），共 506 筆、13 個特徵。" +
        "主要欄位：crim 犯罪率、zn 大坪數住宅用地比例、indus 非零售商業用地比例、chas 是否臨 Charles River、nox 一氧化氮濃度、" +
        "rm 平均房間數、age 1940 年前建成比例、dis 到就業中心距離、rad 公路可及性、tax 房產稅率、ptratio 師生比、b 族群指標、lstat 低收入人口比例。" +
        "目標：比較不同特徵選擇方法，找出最佳的特徵數量 k。",
    },
  };

  const $ = (id) => document.getElementById(id);
  const state = { ds: null, profile: null, preset: "startups", sourceLabel: "", excluded: new Set(), algo: "all", k: 1, charts: {}, inputs: {} };

  // ------------------------------------------------------------------ helpers
  function fmt(v, digits = 4) {
    if (v === Infinity) return "∞";
    if (v == null || !Number.isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e4) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
    if (a >= 100) return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
    if (a < 1e-4) return v.toExponential(2);
    return Number(v.toPrecision(digits)).toString();
  }
  const fmtP = (p) => (!Number.isFinite(p) ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const seriesColor = (i) => css(`--s${(i % 5) + 1}`);
  const POINT_STYLES = ["circle", "rect", "triangle", "rectRot", "rectRounded"];
  const pick = (idx, arr) => idx.map((i) => arr[i]);

  function setStatus(msg, isError = false) {
    $("status").innerHTML = msg;
    $("status").classList.toggle("error", isError);
  }

  /** github.com/.../blob/... → raw.githubusercontent.com/... */
  function normalizeUrl(url) {
    const m = url.trim().match(/^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/blob\/(.+)$/);
    return m ? `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${m[3]}` : url.trim();
  }

  async function fetchText(url) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.text();
    } catch (direct) {
      // Most likely a CORS block: retry through the Vercel serverless proxy.
      let r;
      try {
        r = await fetch(`/api/proxy?url=${encodeURIComponent(url)}`);
      } catch {
        throw new Error(`無法下載（${direct.message}）`);
      }
      if (!r.ok) throw new Error((await r.text().catch(() => "")) || `proxy HTTP ${r.status}`);
      return r.text();
    }
  }

  // ------------------------------------------------------------------ loading
  async function loadFromUrl() {
    const url = normalizeUrl($("urlInput").value);
    if (!/^https?:\/\//i.test(url)) return setStatus("請輸入以 http(s):// 開頭的 CSV 網址。", true);
    $("urlInput").value = url;
    $("loadBtn").disabled = true;
    setStatus("下載資料中…");
    try {
      const text = await fetchText(url);
      if (/^\s*<(!doctype|html)/i.test(text)) throw new Error("這個網址回傳的是 HTML 網頁而不是 CSV（GitHub 請使用 Raw 連結）。");
      const preset = Object.keys(PRESETS).find((k) => PRESETS[k].url === url) || "custom";
      $("presetSelect").value = preset;
      state.preset = preset;
      onDataset(DataPrep.parseCSV(text), url, preset === "custom" ? null : PRESETS[preset].target);
      const q = preset === "custom" ? `?url=${encodeURIComponent(url)}` : `?dataset=${preset}`;
      history.replaceState(null, "", q);
    } catch (e) {
      setStatus(`載入失敗：${esc(e.message)}`, true);
    } finally {
      $("loadBtn").disabled = false;
    }
  }

  function loadFromFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        $("presetSelect").value = "custom";
        state.preset = "custom";
        onDataset(DataPrep.parseCSV(reader.result), file.name, null);
      } catch (e) {
        setStatus(`解析失敗：${esc(e.message)}`, true);
      }
    };
    reader.readAsText(file);
  }

  function onDataset(ds, sourceLabel, presetTarget) {
    state.ds = ds;
    state.profile = DataPrep.profileColumns(ds);
    state.sourceLabel = sourceLabel;
    state.excluded = new Set();
    state.inputs = {};
    state.algo = "all";
    const numeric = state.profile.filter((c) => c.kind === "numeric");
    if (!numeric.length) throw new Error("找不到任何數值欄位，無法做迴歸。");
    const target = numeric.find((c) => c.name === presetTarget) || numeric[numeric.length - 1];
    $("targetSelect").innerHTML = numeric.map((c) => `<option ${c === target ? "selected" : ""}>${esc(c.name)}</option>`).join("");
    $("content").hidden = false;
    renderPredictorChips();
    runAnalysis();
  }

  // ------------------------------------------------------------------ data understanding
  const target = () => $("targetSelect").value;

  function renderPredictorChips() {
    $("predictorList").innerHTML = state.profile
      .filter((c) => c.name !== target())
      .map((c) => {
        const disabled = c.kind === "text";
        const tag = c.kind === "numeric" ? "num" : c.kind === "categorical" ? `cat · ${c.levels.length}` : `略過：${c.reason}`;
        const checked = !disabled && !state.excluded.has(c.name);
        return `<label class="chip ${disabled ? "disabled" : ""}"><input type="checkbox" value="${esc(c.name)}" ${checked ? "checked" : ""} ${disabled ? "disabled" : ""}/>${esc(c.name)} <span class="tag">${esc(tag)}</span></label>`;
      })
      .join("");
  }

  function renderDataUnderstanding(prep) {
    const { ds, profile } = state;
    const missingCells = profile.reduce((s, c) => s + c.missing, 0);
    const kinds = (k) => profile.filter((c) => c.kind === k).length;
    $("dataTiles").innerHTML = tiles([
      ["Rows", ds.rows.length, `使用 ${prep.y.length} 列（移除 ${prep.dropped} 列缺值）`],
      ["Columns", ds.header.length, `${kinds("numeric")} 數值 · ${kinds("categorical")} 類別 · ${kinds("text")} 略過`],
      ["Encoded features (p)", prep.features.length, "one-hot 後的特徵數"],
      ["Missing cells", missingCells, `${((missingCells / (ds.rows.length * ds.header.length)) * 100).toFixed(1)}%`],
      ["Target", esc(target()), `mean ${fmt(MLR.mean(prep.y))}`],
    ]);

    const head = `<thead><tr>${ds.header.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>`;
    $("previewTable").innerHTML = head + `<tbody>${ds.rows.slice(0, 8).map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody>`;

    const rows = profile.map((c) => {
      if (c.kind !== "numeric") {
        const info = c.kind === "categorical" ? c.levels.map(esc).join(" / ") : `略過（${esc(c.reason)}）`;
        return `<tr><td>${esc(c.name)}</td><td>${ds.rows.length - c.missing}</td><td colspan="5" style="text-align:left">${info}</td><td>${c.missing}</td></tr>`;
      }
      const v = ds.rows.map((r) => DataPrep.toNumber(r[c.index])).filter(Number.isFinite).sort((a, b) => a - b);
      const m = MLR.mean(v);
      const sd = Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, v.length - 1));
      const med = v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
      return `<tr><td>${esc(c.name)}</td><td>${v.length}</td><td>${fmt(m)}</td><td>${fmt(sd)}</td><td>${fmt(v[0])}</td><td>${fmt(med)}</td><td>${fmt(v[v.length - 1])}</td><td>${c.missing}</td></tr>`;
    });
    $("describeTable").innerHTML = `<thead><tr><th>Column</th><th>count</th><th>mean</th><th>std</th><th>min</th><th>median</th><th>max</th><th>missing</th></tr></thead><tbody>${rows.join("")}</tbody>`;

    renderCorrChart(prep);
  }

  function pearson(a, b) {
    const ma = MLR.mean(a), mb = MLR.mean(b);
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < a.length; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
    return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
  }

  function renderCorrChart(prep) {
    const corr = prep.features.map((f, j) => ({ name: f.name, r: pearson(prep.X.map((row) => row[j]), prep.y) })).sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
    $("corrBox").style.height = `${Math.max(180, corr.length * 26 + 60)}px`;
    const pos = css("--pos"), neg = css("--neg");
    makeChart("corrChart", {
      type: "bar",
      data: {
        labels: corr.map((c) => c.name),
        datasets: [{ data: corr.map((c) => c.r), backgroundColor: corr.map((c) => (c.r >= 0 ? pos : neg)), borderRadius: 4, borderSkipped: false, maxBarThickness: 18, categoryPercentage: 0.8 }],
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `r = ${c.raw.toFixed(3)}` } } },
        scales: { x: { min: -1, max: 1, grid: { color: (c) => (c.tick.value === 0 ? css("--axis") : css("--grid")) } }, y: { grid: { display: false } } },
      },
    });
  }

  // ------------------------------------------------------------------ analysis
  function runAnalysis() {
    const checked = [...$("predictorList").querySelectorAll("input:checked")].map((i) => i.value);
    state.excluded = new Set(state.profile.filter((c) => c.name !== target() && c.kind !== "text" && !checked.includes(c.name)).map((c) => c.name));
    let prep;
    try {
      prep = DataPrep.buildMatrix(state.ds, state.profile, target(), checked);
    } catch (e) {
      return setStatus(esc(e.message), true);
    }
    const p = prep.features.length;
    if (!p) return setStatus("請至少選擇一個預測變數。", true);
    const testRatio = Number($("testRatio").value) / 100;
    const nTrain = prep.y.length - Math.max(1, Math.round(prep.y.length * testRatio));
    const folds = Number($("foldsSelect").value);
    if (nTrain < p + 2 || nTrain < folds * 2) return setStatus(`訓練資料太少（${nTrain} 列）無法擬合 ${p} 個特徵，請減少特徵或降低 test size。`, true);

    state.prep = prep;
    renderDataUnderstanding(prep);
    $("problemText").innerHTML =
      state.preset in PRESETS
        ? PRESETS[state.preset].problem
        : `<b>自訂資料集</b>（${esc(state.sourceLabel)}）：以 ${prep.predictors.length} 個欄位預測數值目標 <b>${esc(target())}</b>，比較 5 種特徵選擇方法，找出測試誤差最低、最精簡的多元線性迴歸模型。`;

    setStatus(`訓練中… (n = ${prep.y.length}, p = ${p})`);
    $("runBtn").disabled = true;
    setTimeout(() => {
      try {
        const t0 = performance.now();
        const seed = Number($("seedInput").value) || 0;
        const split = MLR.trainTestSplit(prep.y.length, testRatio, seed);
        state.split = split;
        state.Xtr = pick(split.train, prep.X); state.ytr = pick(split.train, prep.y);
        state.Xte = pick(split.test, prep.X); state.yte = pick(split.test, prep.y);
        state.results = MLR.runFeatureSelection(state.Xtr, state.ytr, state.Xte, state.yte, { nFolds: folds, seed });

        const best = globalBest();
        if (state.algo !== "all" && !state.results.find((r) => r.id === state.algo && !r.skipped)) state.algo = "all";
        state.k = best ? best.step.k : 1;
        $("prepSummary").innerHTML =
          `Train ${split.train.length} 列 / Test ${split.test.length} 列 · ${p} 個特徵：` +
          prep.features.map((f) => esc(f.name)).join(", ") +
          (prep.features.some((f) => f.type === "dummy") ? ` · 類別基準：${[...new Set(prep.features.filter((f) => f.type === "dummy").map((f) => `${esc(f.source)}=${esc(f.baseline)}`))].join(", ")}` : "");
        renderModeling();
        const skipped = state.results.filter((r) => r.skipped).map((r) => r.name);
        setStatus(
          `✓ 已載入 <b>${esc(state.sourceLabel)}</b> · ${prep.y.length} 列 × ${p} 特徵 · 計算 ${Math.round(performance.now() - t0)} ms` +
            (best ? ` · 最佳：${esc(best.alg.name)}, k = ${best.step.k}, Test RMSE ${fmt(best.step.testRmse)}` : "") +
            (skipped.length ? ` · 已略過 ${skipped.join(", ")}（p > ${MLR.EXHAUSTIVE_MAX}）` : "")
        );
      } catch (e) {
        console.error(e);
        setStatus(`分析失敗：${esc(e.message)}`, true);
      } finally {
        $("runBtn").disabled = false;
      }
    }, 30);
  }

  const validSteps = (r) => (r.skipped ? [] : r.steps.filter((s) => !s.invalid));

  function globalBest() {
    let best = null;
    for (const alg of state.results) for (const step of validSteps(alg)) if (!best || step.testRmse < best.step.testRmse - 1e-12) best = { alg, step };
    return best;
  }

  /** The (algorithm, step) shown in Evaluation: the chosen algorithm, or the best one at k in compare mode. */
  function activeModel() {
    const k = state.k;
    if (state.algo !== "all") {
      const alg = state.results.find((r) => r.id === state.algo);
      return { alg, step: alg.steps[k - 1] };
    }
    let best = null;
    for (const alg of state.results) {
      const step = validSteps(alg).find((s) => s.k === k);
      if (step && (!best || step.testRmse < best.step.testRmse - 1e-12)) best = { alg, step };
    }
    return best;
  }

  // ------------------------------------------------------------------ modeling
  function renderModeling() {
    const p = state.prep.features.length;
    $("algoSelect").innerHTML =
      `<option value="all">Compare all 5 algorithms</option>` +
      state.results.map((r) => `<option value="${r.id}" ${r.skipped ? "disabled" : ""}>${esc(r.name)}${r.skipped ? ` (${esc(r.skipped)})` : ""}</option>`).join("");
    $("algoSelect").value = state.algo;
    $("kSlider").max = p;
    $("kSlider").value = state.k;
    renderSelection();
  }

  function renderSelection() {
    $("kVal").textContent = state.k;
    const scope = state.algo === "all" ? null : state.results.find((r) => r.id === state.algo);
    const best = scope ? validSteps(scope).reduce((a, s) => (!a || s.testRmse < a.testRmse ? s : a), null) : globalBest()?.step;
    $("kHint").textContent = best ? `（Test RMSE 最低在 k = ${best.k}）` : "";
    renderModelingCharts();
    renderCompareTable();
    renderModel();
  }

  function selectedKPlugin() {
    return {
      id: "selectedK",
      afterDatasetsDraw(chart) {
        const x = chart.scales.x.getPixelForValue(state.k - 1);
        const { top, bottom } = chart.chartArea;
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = css("--muted");
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
        ctx.restore();
      },
    };
  }

  function renderModelingCharts() {
    const p = state.prep.features.length;
    const labels = Array.from({ length: p }, (_, i) => i + 1);
    const datasets = (metric) =>
      state.results.map((r, i) => {
        if (r.skipped) return null;
        const focus = state.algo === "all" || state.algo === r.id;
        const color = seriesColor(i);
        return {
          label: r.name,
          data: r.steps.map((s) => (s.invalid ? null : s[metric])),
          borderColor: focus ? color : color + "40",
          backgroundColor: focus ? color : color + "40",
          borderWidth: state.algo === r.id ? 3 : 2,
          pointStyle: POINT_STYLES[i],
          pointRadius: 4, pointHoverRadius: 6,
          pointBorderColor: css("--surface"), pointBorderWidth: 1,
          tension: 0, order: state.algo === r.id ? 0 : 1,
        };
      }).filter(Boolean);
    const common = (yTitle, digits) => ({
      interaction: { mode: "index", intersect: false },
      onClick: (_, els, chart) => {
        const x = chart.scales.x.getValueForPixel(_.x);
        if (Number.isFinite(x)) { state.k = Math.min(p, Math.max(1, Math.round(x) + 1)); $("kSlider").value = state.k; renderSelection(); }
      },
      plugins: {
        legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8 } },
        tooltip: { callbacks: { title: (items) => `k = ${items[0].label}`, label: (c) => ` ${c.dataset.label}: ${fmt(c.raw, digits)}` } },
      },
      scales: { x: { title: { display: true, text: "Number of features (k)" }, grid: { display: false } }, y: { title: { display: true, text: yTitle } } },
    });
    makeChart("rmseChart", { type: "line", data: { labels, datasets: datasets("testRmse") }, options: common("Test RMSE", 5), plugins: [selectedKPlugin()] });
    makeChart("r2Chart", { type: "line", data: { labels, datasets: datasets("testR2") }, options: common("Test R²", 4), plugins: [selectedKPlugin()] });
  }

  function renderCompareTable() {
    const p = state.prep.features.length;
    const active = activeModel();
    const head = `<thead><tr><th>Algorithm</th>${Array.from({ length: p }, (_, i) => `<th>k=${i + 1}</th>`).join("")}<th style="text-align:left">最佳子集</th></tr></thead>`;
    const body = state.results.map((r, i) => {
      if (r.skipped) return `<tr class="dim"><td><span class="swatch" style="background:${seriesColor(i)}"></span>${esc(r.name)}</td><td colspan="${p + 1}" style="text-align:left">${esc(r.skipped)}</td></tr>`;
      const best = validSteps(r).reduce((a, s) => (!a || s.testRmse < a.testRmse ? s : a), null);
      const cells = r.steps.map((s) => {
        const cls = [s === best ? "best" : "", active && active.alg === r && active.step === s ? "sel" : ""].join(" ");
        return `<td class="${cls}" data-algo="${r.id}" data-k="${s.k}" title="${esc(s.cols.map((c) => state.prep.features[c].name).join(", "))}">${s.invalid ? "—" : fmt(s.testRmse, 5)}</td>`;
      });
      const dim = state.algo !== "all" && state.algo !== r.id ? "dim" : "";
      return `<tr class="${dim}"><td><span class="swatch" style="background:${seriesColor(i)}"></span>${esc(r.name)}</td>${cells.join("")}<td style="text-align:left">${best ? best.cols.map((c) => esc(state.prep.features[c].name)).join(", ") : "—"}</td></tr>`;
    });
    $("compareTable").innerHTML = head + `<tbody>${body.join("")}</tbody>`;
  }

  // ------------------------------------------------------------------ evaluation
  function renderModel() {
    const active = activeModel();
    if (!active || !active.step || active.step.invalid) {
      $("modelLabel").textContent = "此組合的設計矩陣為奇異矩陣（完全共線），無法擬合。";
      return;
    }
    const { alg, step } = active;
    const { features } = state.prep;
    const cols = step.cols;
    const m = MLR.ols(state.Xtr, state.ytr, cols);
    const vifs = MLR.vif(state.Xtr, cols);
    const sdY = Math.sqrt(MLR.mean(state.ytr.map((v) => (v - MLR.mean(state.ytr)) ** 2)));
    const sdX = cols.map((c) => { const col = state.Xtr.map((r) => r[c]); const mu = MLR.mean(col); return Math.sqrt(MLR.mean(col.map((v) => (v - mu) ** 2))); });
    const predTe = state.Xte.map((r) => m.predictRow(r));
    const te = MLR.metrics(state.yte, predTe);
    state.model = m;

    $("modelLabel").innerHTML =
      `${esc(alg.name)}${state.algo === "all" ? ` <span class="muted small">（k = ${step.k} 時表現最佳的演算法）</span>` : ""} · k = ${step.k}<br>` +
      cols.map((c) => `<span class="pill">${esc(features[c].name)}</span>`).join("");

    $("metricTiles").innerHTML = tiles([
      ["Train R²", fmt(m.r2), `Adj. R² ${fmt(m.adjR2)}`],
      ["Test R²", fmt(te.r2), "out-of-sample"],
      ["Test RMSE", fmt(te.rmse), `Train RMSE ${fmt(m.rmse)}`],
      ["Test MAE", fmt(te.mae), `CV RMSE ${fmt(step.cvRmse)}`],
      ["F-statistic", fmt(m.F), `p ${fmtP(m.Fp)}`],
      ["AIC / BIC", fmt(m.aic), `BIC ${fmt(m.bic)}`],
    ]);

    const terms = cols.map((c, j) => {
      const b = m.beta[j + 1];
      return `<span>${b < 0 ? "−" : "+"} ${fmt(Math.abs(b))} × ${esc(features[c].name)}</span>`;
    });
    $("equation").innerHTML = `ŷ<sub>${esc(target())}</sub> = ${fmt(m.beta[0])} ${terms.join(" ")}`;

    const names = ["(Intercept)", ...cols.map((c) => features[c].name)];
    const stars = (p) => (p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "");
    const vifCell = (v) => (v > 10 ? `<span class="flag-bad">${fmt(v)} ▲</span>` : v > 5 ? `<span class="flag-warn">${fmt(v)} △</span>` : fmt(v));
    $("coefTable").innerHTML =
      `<thead><tr><th>Term</th><th>β</th><th>Std. β</th><th>Std. Error</th><th>t</th><th>p-value</th><th>95% CI low</th><th>95% CI high</th><th>VIF</th></tr></thead><tbody>` +
      names.map((n, i) => `<tr><td>${esc(n)}</td><td>${fmt(m.beta[i])}</td><td>${i ? fmt((m.beta[i] * sdX[i - 1]) / sdY, 3) : "—"}</td><td>${fmt(m.se[i])}</td><td>${fmt(m.tStat[i], 3)}</td>` +
        `<td class="${m.pVal[i] < 0.05 ? "sig" : ""}">${fmtP(m.pVal[i])} ${stars(m.pVal[i])}</td><td>${fmt(m.ciLow[i])}</td><td>${fmt(m.ciHigh[i])}</td><td>${i ? vifCell(vifs[i - 1]) : "—"}</td></tr>`).join("") +
      `</tbody>`;

    renderModelCharts(m, predTe);

    const rows = state.split.test.map((ri, i) => ({ row: state.prep.kept[ri] + 1, actual: state.yte[i], pred: predTe[i], res: state.yte[i] - predTe[i] })).sort((a, b) => Math.abs(b.res) - Math.abs(a.res)).slice(0, 10);
    $("residTable").innerHTML =
      `<thead><tr><th>Row #</th><th>Actual</th><th>Predicted</th><th>Residual</th><th>% error</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${r.row}</td><td>${fmt(r.actual)}</td><td>${fmt(r.pred)}</td><td>${fmt(r.res)}</td><td>${r.actual ? ((Math.abs(r.res) / Math.abs(r.actual)) * 100).toFixed(1) + "%" : "—"}</td></tr>`).join("") +
      `</tbody>`;

    renderPredictor(m, cols);
  }

  function renderModelCharts(m, predTe) {
    if (!m) return;
    const c1 = seriesColor(0), c2 = seriesColor(1);
    const surface = css("--surface");
    const point = (color) => ({ backgroundColor: color + "cc", borderColor: surface, borderWidth: 1, pointRadius: 4, pointHoverRadius: 6 });
    const all = [...state.ytr, ...state.yte, ...m.fitted, ...predTe];
    const lo = Math.min(...all), hi = Math.max(...all);
    const tooltip = (xl, yl) => ({ callbacks: { label: (c) => ` ${c.dataset.label}: ${xl} ${fmt(c.raw.x)}, ${yl} ${fmt(c.raw.y)}` } });
    makeChart("avpChart", {
      type: "scatter",
      data: {
        datasets: [
          { label: "Train", data: state.ytr.map((y, i) => ({ x: y, y: m.fitted[i] })), ...point(c1), pointStyle: "circle" },
          { label: "Test", data: state.yte.map((y, i) => ({ x: y, y: predTe[i] })), ...point(c2), pointStyle: "rect" },
          { label: "y = x", type: "line", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: css("--muted"), borderDash: [5, 5], borderWidth: 1.5, pointRadius: 0 },
        ],
      },
      options: {
        plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8 } }, tooltip: tooltip("actual", "pred") },
        scales: { x: { title: { display: true, text: `Actual ${target()}` } }, y: { title: { display: true, text: `Predicted ${target()}` } } },
      },
    });
    makeChart("residChart", {
      type: "scatter",
      data: {
        datasets: [
          { label: "Train", data: m.fitted.map((f, i) => ({ x: f, y: m.resid[i] })), ...point(c1), pointStyle: "circle" },
          { label: "Test", data: predTe.map((f, i) => ({ x: f, y: state.yte[i] - f })), ...point(c2), pointStyle: "rect" },
        ],
      },
      options: {
        plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8 } }, tooltip: tooltip("fitted", "resid") },
        scales: {
          x: { title: { display: true, text: "Fitted value" } },
          y: { title: { display: true, text: "Residual" }, grid: { color: (c) => (c.tick.value === 0 ? css("--axis") : css("--grid")), lineWidth: (c) => (c.tick.value === 0 ? 1.5 : 1) } },
        },
      },
    });
  }

  // ------------------------------------------------------------------ deployment / prediction
  function renderPredictor(m, cols) {
    const { features, predictors } = state.prep;
    const sources = predictors.filter((p) => cols.some((c) => features[c].source === p.name));
    const html = sources.map((p) => {
      const id = `in_${p.index}`;
      if (p.kind === "categorical") {
        const saved = state.inputs[p.name] ?? p.levels[0];
        return `<label class="field"><span class="label">${esc(p.name)}</span><select id="${id}" data-src="${esc(p.name)}">${p.levels.map((l) => `<option ${l === saved ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></label>`;
      }
      const j = features.findIndex((f) => f.source === p.name);
      const col = state.Xtr.map((r) => r[j]);
      const mu = MLR.mean(col);
      const saved = state.inputs[p.name] ?? Number(mu.toPrecision(6));
      return `<label class="field"><span class="label">${esc(p.name)} <span class="muted">(${fmt(Math.min(...col))} – ${fmt(Math.max(...col))})</span></span><input id="${id}" type="number" step="any" value="${saved}" data-src="${esc(p.name)}" /></label>`;
    });
    $("predictForm").innerHTML = html.join("") || `<p class="muted">此模型沒有特徵。</p>`;
    $("predLabel").textContent = `預測 ${target()}`;
    updatePrediction();
  }

  function updatePrediction() {
    const m = state.model;
    if (!m) return;
    const { features } = state.prep;
    const values = {};
    $("predictForm").querySelectorAll("[data-src]").forEach((el) => {
      values[el.dataset.src] = el.tagName === "SELECT" ? el.value : Number(el.value);
      state.inputs[el.dataset.src] = el.tagName === "SELECT" ? el.value : el.value;
    });
    const row = features.map((f) => (f.type === "dummy" ? (values[f.source] === f.level ? 1 : 0) : values[f.source] ?? 0));
    const yhat = m.predictRow(row);
    const x0 = [1, ...m.cols.map((c) => row[c])];
    const lev = x0.reduce((s, a, i) => s + a * x0.reduce((t, b, j) => t + m.xtxInv[i][j] * b, 0), 0);
    const half = m.tCrit * m.sigma * Math.sqrt(1 + lev);
    $("predValue").textContent = fmt(yhat);
    $("predInterval").textContent = Number.isFinite(half) ? `95% 預測區間：${fmt(yhat - half)} ~ ${fmt(yhat + half)}` : "";

    const parts = m.cols.map((c, j) => ({ name: features[c].name, v: m.beta[j + 1] * row[c] }));
    const maxAbs = Math.max(1e-12, ...parts.map((p) => Math.abs(p.v)));
    $("contrib").innerHTML =
      `<div class="contrib-row"><span class="name">(Intercept)</span><span class="bar"></span><span class="val">${fmt(m.beta[0])}</span></div>` +
      parts.map((p) => {
        const w = (Math.abs(p.v) / maxAbs) * 50;
        const style = p.v >= 0 ? `left:50%;width:${w}%;background:var(--pos)` : `right:50%;width:${w}%;background:var(--neg)`;
        return `<div class="contrib-row"><span class="name" title="${esc(p.name)}">${esc(p.name)}</span><span class="bar"><span style="${style}"></span></span><span class="val">${p.v >= 0 ? "+" : "−"}${fmt(Math.abs(p.v))}</span></div>`;
      }).join("");
  }

  // ------------------------------------------------------------------ charts / tiles
  function tiles(items) {
    return items.map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d ?? ""}</div></div>`).join("");
  }

  function makeChart(id, config) {
    if (state.charts[id]) state.charts[id].destroy();
    Chart.defaults.font.family = css("--font");
    Chart.defaults.color = css("--text-2");
    Chart.defaults.borderColor = css("--grid");
    config.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: { duration: 250 } }, config.options);
    state.charts[id] = new Chart($(id), config);
  }

  function rerenderCharts() {
    if (!state.prep || !state.results) return;
    renderCorrChart(state.prep);
    renderSelection();
  }

  // ------------------------------------------------------------------ theme
  function applyTheme(t) {
    if (t) document.documentElement.dataset.theme = t;
    else delete document.documentElement.dataset.theme;
  }
  try { applyTheme(localStorage.getItem("theme")); } catch {}
  $("themeToggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("theme", next); } catch {}
    rerenderCharts();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerenderCharts);

  // ------------------------------------------------------------------ events
  $("presetSelect").addEventListener("change", (e) => {
    const key = e.target.value;
    if (key === "custom") {
      $("urlInput").value = "";
      $("urlInput").focus();
      setStatus("貼上任何公開 CSV 的網址，然後按「載入並分析」。");
      return;
    }
    $("urlInput").value = PRESETS[key].url;
    loadFromUrl();
  });
  $("loadBtn").addEventListener("click", loadFromUrl);
  $("urlInput").addEventListener("keydown", (e) => { if (e.key === "Enter") loadFromUrl(); });
  $("fileInput").addEventListener("change", (e) => { if (e.target.files[0]) loadFromFile(e.target.files[0]); e.target.value = ""; });
  $("targetSelect").addEventListener("change", () => { state.inputs = {}; renderPredictorChips(); runAnalysis(); });
  $("predictorList").addEventListener("change", runAnalysis);
  $("runBtn").addEventListener("click", runAnalysis);
  $("testRatio").addEventListener("input", (e) => ($("testRatioVal").textContent = `${e.target.value}%`));
  $("testRatio").addEventListener("change", runAnalysis);
  $("seedInput").addEventListener("change", runAnalysis);
  $("foldsSelect").addEventListener("change", runAnalysis);
  $("algoSelect").addEventListener("change", (e) => {
    state.algo = e.target.value;
    renderSelection();
  });
  $("kSlider").addEventListener("input", (e) => {
    state.k = Number(e.target.value);
    renderSelection();
  });
  $("compareTable").addEventListener("click", (e) => {
    const td = e.target.closest("td[data-algo]");
    if (!td) return;
    state.algo = td.dataset.algo;
    state.k = Number(td.dataset.k);
    $("algoSelect").value = state.algo;
    $("kSlider").value = state.k;
    renderSelection();
  });
  $("predictForm").addEventListener("input", updatePrediction);

  // ------------------------------------------------------------------ boot
  const params = new URLSearchParams(location.search);
  if (params.get("url")) {
    $("presetSelect").value = "custom";
    $("urlInput").value = params.get("url");
  } else {
    const key = PRESETS[params.get("dataset")] ? params.get("dataset") : "startups";
    $("presetSelect").value = key;
    $("urlInput").value = PRESETS[key].url;
  }
  loadFromUrl();
})();
