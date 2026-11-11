// Runs the detector entirely in the browser. Mirrors src/data.py (preprocessing) and src/predict.py (verdict).
// Preprocessing rules: honour the phone rotation flag, 3-channel RGB, NATIVE-resolution 224x224 crops
// (centre + 4 corners, small images reflect-padded at the bottom/right), ImageNet normalisation,
// average the sigmoid of the five logits.
const CROP = 224, MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const MAX_MB = 25;
// FIXED VERDICT ZONES (score = probability the image is AI-generated, 0 to 1).
//   below REAL_BELOW            -> Likely real
//   REAL_BELOW up to AI_FROM    -> Uncertain
//   AI_FROM and above           -> Likely AI-generated
// The decision line (manifest.threshold, 0.55) sits in the middle of the uncertain zone. It is only used for the
// plain two-way confusion matrix, so the page can be compared with the training notebook.
// The 10-point width is a judgement call: on the validation set it gave about 80% right answers outside the zone
// (wider zones were more accurate but left nearly half the images uncertain). The page shows the test-set numbers.
let REAL_BELOW = 0.45, AI_FROM = 0.65;
let session = null, manifest = null;

const $ = (id) => document.getElementById(id);

async function loadModel() {
  manifest = await (await fetch("model/manifest.json")).json();
  if (manifest.run === "untrained-test") $("testbanner").hidden = false;
  const bufs = []; let got = 0;
  for (const p of manifest.parts) {
    const r = await fetch("model/" + p);
    if (!r.ok) throw new Error("Could not download " + p);
    const b = new Uint8Array(await r.arrayBuffer());
    bufs.push(b); got += b.length;
    $("loadbar").style.width = Math.round(100 * got / manifest.bytes) + "%";
  }
  const all = new Uint8Array(got); let o = 0;
  for (const b of bufs) { all.set(b, o); o += b.length; }
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;   // must be an absolute URL
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(all, { executionProviders: ["wasm"] });
  $("loader").hidden = true; $("app").hidden = false;
  if (manifest.zones) { REAL_BELOW = manifest.zones.real_below; AI_FROM = manifest.zones.ai_from; }
  drawMeter(); $("batch").hidden = false;
}

async function decode(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none" });
  } catch (e) {
    return await createImageBitmap(file);
  }
}

function readPixels(bmp, sx, sy, w, h) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(bmp, sx, sy, w, h, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

// five crops as RGBA arrays, same boxes as src/data.py five_boxes()
function fiveCrops(bmp) {
  const W = bmp.width, H = bmp.height;
  const padW = Math.max(0, CROP - W), padH = Math.max(0, CROP - H);
  const PW = W + padW, PH = H + padH;
  const boxes = [[Math.floor((PW - CROP) / 2), Math.floor((PH - CROP) / 2)],
                 [0, 0], [PW - CROP, 0], [0, PH - CROP], [PW - CROP, PH - CROP]];
  if (!padW && !padH) return boxes.map(([x, y]) => readPixels(bmp, x, y, CROP, CROP));
  // small image: pad bottom/right (reflect, or edge when the image is too small to reflect), like numpy.pad
  const src = readPixels(bmp, 0, 0, W, H);
  const reflect = H > padH && W > padW;
  const map = (i, n) => (i < n ? i : (reflect ? 2 * n - 2 - i : n - 1));
  return boxes.map(([bx, by]) => {
    const out = new Uint8ClampedArray(CROP * CROP * 4);
    for (let y = 0; y < CROP; y++) {
      const sy = map(by + y, H);
      for (let x = 0; x < CROP; x++) {
        const sx = map(bx + x, W), s = (sy * W + sx) * 4, d = (y * CROP + x) * 4;
        out[d] = src[s]; out[d + 1] = src[s + 1]; out[d + 2] = src[s + 2]; out[d + 3] = 255;
      }
    }
    return out;
  });
}

function toTensor(crops) {
  const N = CROP * CROP, data = new Float32Array(crops.length * 3 * N);
  crops.forEach((px, k) => {
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) data[k * 3 * N + c * N + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  });
  return new ort.Tensor("float32", data, [crops.length, 3, CROP, CROP]);
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const pct = (x) => (100 * x).toFixed(0) + "%";
function verdictFor(p) { return p < REAL_BELOW ? "real" : (p >= AI_FROM ? "AI-generated" : "uncertain"); }

// the three-zone meter: green, amber, red, with a pin at the score
function drawMeter() {
  $("zreal").style.width = (100 * REAL_BELOW) + "%";
  $("zunsure").style.width = (100 * (AI_FROM - REAL_BELOW)) + "%";
  $("zai").style.width = (100 * (1 - AI_FROM)) + "%";
  $("tick1").style.left = (100 * REAL_BELOW) + "%"; $("tick1").textContent = pct(REAL_BELOW);
  $("tick2").style.left = (100 * AI_FROM) + "%"; $("tick2").textContent = pct(AI_FROM);
  document.querySelectorAll(".zr").forEach((e) => (e.textContent = pct(REAL_BELOW)));
  document.querySelectorAll(".za").forEach((e) => (e.textContent = pct(AI_FROM)));
}

async function analyseFile(file) {
  const bmp = await decode(file);
  const W = bmp.width, H = bmp.height;
  const out = await session.run({ input: toTensor(fiveCrops(bmp)) });
  const logits = Array.from(out.logit.data);
  bmp.close && bmp.close();
  const p = logits.map(sigmoid).reduce((a, b) => a + b, 0) / logits.length;
  const thr = manifest.threshold, notes = [];
  if (Math.min(W, H) < CROP) notes.push(`This image is small (${W} x ${H}); the model works on 224 x 224 pixel crops, so the result is less reliable.`);
  if (/\.jpe?g$/i.test(file.name) && file.size / (W * H) < 0.25)
    notes.push("This image looks heavily compressed (for example sent through WhatsApp). Compression can hide the traces the model looks for.");
  const label = verdictFor(p);
  if (label === "uncertain") notes.push(`The score is between ${pct(REAL_BELOW)} and ${pct(AI_FROM)}, the uncertain zone. On our test images the model was close to a coin toss in this zone, so do not rely on it.`);
  return { p, logits, label, notes, width: W, height: H, threshold: thr };
}

function show(r, file) {
  $("error").hidden = true;
  const cls = { "real": "real", "AI-generated": "ai", "uncertain": "unsure" }[r.label];
  const text = { "real": "Likely real", "AI-generated": "Likely AI-generated", "uncertain": "Uncertain" }[r.label];
  $("verdict").textContent = text; $("verdict").className = cls;
  $("prob").textContent = `AI score: ${(100 * r.p).toFixed(1)}% (the model's estimated chance that this image is AI-generated)`;
  $("pin").style.left = (100 * r.p) + "%";
  $("notes").innerHTML = "";
  (r.notes.length ? r.notes : ["No warnings for this image."]).forEach((n) => {
    const li = document.createElement("li"); li.textContent = n; $("notes").appendChild(li);
  });
  $("result").hidden = false;
}

async function onFile(file) {
  if (!file) return;
  $("result").hidden = true; $("error").hidden = true;
  if (file.size > MAX_MB * 1024 * 1024) { fail(`That file is larger than ${MAX_MB} MB. Please use a smaller one.`); return; }
  $("preview").src = URL.createObjectURL(file); $("preview").hidden = false;
  $("droptext").textContent = "Checking...";
  try {
    show(await analyseFile(file), file);
  } catch (e) {
    console.error(e);
    fail("Sorry, I could not read that file as an image. Try a JPEG, PNG or WebP photo (iPhone HEIC photos are not supported by browsers).");
  }
  $("droptext").textContent = "Choose another photo";
}
function fail(msg) { $("error").textContent = msg; $("error").hidden = false; $("droptext").textContent = "Choose a photo, or drop one here"; }

$("file").addEventListener("change", (e) => onFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => onFile(e.dataTransfer.files[0]));

window.__ready = loadModel().catch((e) => {
  console.error(e);
  $("loadtext").textContent = "Sorry, the model could not be loaded. Please refresh the page or try another browser.";
});
window.__analyse = analyseFile;

/* ====================================================================================================
   BATCH TEST: take in many images (for example the notebook's test set), score every one automatically,
   and show the confusion matrix, zone table and a downloadable results file.
   Labels come from (1) a labels.csv chosen together with the images, or (2) folder names real/ and ai/.
   ==================================================================================================== */
const IMG_EXT = /\.(jpe?g|png|webp|bmp|gif)$/i;
let batchItems = [], batchResults = [], batchStop = false, batchBusy = false;

function normLabel(v) {
  v = String(v == null ? "" : v).trim().toLowerCase();
  if (["ai", "fake", "generated", "ai-generated", "synthetic", "1", "true"].includes(v)) return "ai";
  if (["real", "human", "authentic", "0", "false"].includes(v)) return "real";
  return null;
}
function labelFromPath(path) {
  const segs = path.toLowerCase().split("/").slice(0, -1);
  for (let i = segs.length - 1; i >= 0; i--) { const l = normLabel(segs[i]); if (l && isNaN(+segs[i])) return l; }
  return null;
}
function parseCsv(text) {
  const rows = []; let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) { if (ch === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += ch; }
    else if (ch === '"') q = true;
    else if (ch === ",") { row.push(cur); cur = ""; }
    else if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(cur); cur = ""; if (row.length > 1 || row[0] !== "") rows.push(row); row = []; }
    else cur += ch;
  }
  if (cur !== "" || row.length) { row.push(cur); rows.push(row); }
  const head = rows.shift().map((h) => h.trim().toLowerCase());
  return rows.map((r) => Object.fromEntries(head.map((h, i) => [h, (r[i] || "").trim()])));
}
function seededShuffle(arr, seed) {           // mulberry32, so the same N images are picked every time
  const a = arr.slice(); let t = seed >>> 0;
  const rnd = () => { t += 0x6D2B79F5; let x = t; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
const baseName = (n) => n.replace(/^.*\//, "").replace(/\.[^.]+$/, "");

async function onBatchPick(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const csvFile = files.find((f) => /\.csv$/i.test(f.name));
  const meta = {};
  if (csvFile) {
    try {
      for (const row of parseCsv(await csvFile.text())) {
        const key = row.image_id || row.filename || row.file || row.path || row.id;
        if (key) { meta[key.toLowerCase()] = row; meta[baseName(key).toLowerCase()] = row; }
      }
    } catch (e) { console.error(e); }
  }
  batchItems = files.filter((f) => IMG_EXT.test(f.name)).map((f) => {
    const path = f.webkitRelativePath || f.name;
    const m = meta[f.name.toLowerCase()] || meta[baseName(f.name).toLowerCase()] || null;
    const label = (m && normLabel(m.label)) || labelFromPath(path);
    return { file: f, path, label, role: m && m.role || "", generator: m && m.generator || "", ref: m && m.p_fake !== undefined && m.p_fake !== "" ? +m.p_fake : (m && m.notebook_p_fake ? +m.notebook_p_fake : null) };
  });
  const n = batchItems.length, R = batchItems.filter((i) => i.label === "real").length, A = batchItems.filter((i) => i.label === "ai").length;
  $("bfound").hidden = false;
  $("bfound").textContent = n
    ? `Found ${n} images: ${R} real, ${A} AI-generated, ${n - R - A} without a label.` + (csvFile ? ` Labels read from ${csvFile.name}.` : (R + A ? " Labels read from the folder names." : " No labels found, so you will get scores but no confusion matrix."))
    : "No images found in that selection.";
  $("blimit").max = n || 1; $("blimit").value = n || 1;
  $("brun").disabled = !n;
  $("bresults").hidden = true;
}

function fmtTime(sec) { sec = Math.max(0, Math.round(sec)); return sec >= 60 ? `${Math.floor(sec / 60)} min ${sec % 60} s` : `${sec} s`; }

async function runBatch() {
  if (batchBusy) return;
  batchBusy = true; batchStop = false; batchResults = [];
  $("brun").hidden = true; $("bstop").hidden = false; $("bprog").hidden = false; $("bresults").hidden = true;
  let todo = batchItems;
  const lim = Math.max(1, Math.min(todo.length, parseInt($("blimit").value, 10) || todo.length));
  if (lim < todo.length) todo = seededShuffle(todo, 42).slice(0, lim);
  const t0 = performance.now();
  for (let i = 0; i < todo.length && !batchStop; i++) {
    const it = todo[i];
    try {
      const r = await analyseFile(it.file);
      batchResults.push({ ...it, p: r.p, verdict: r.label, w: r.width, h: r.height, ok: true });
    } catch (e) {
      batchResults.push({ ...it, p: NaN, verdict: "unreadable", w: 0, h: 0, ok: false });
    }
    const done = i + 1, el = (performance.now() - t0) / 1000;
    $("bbar").style.width = (100 * done / todo.length) + "%";
    $("btext").textContent = `Checked ${done} of ${todo.length}. About ${fmtTime(el / done * (todo.length - done))} left.`;
    if (done % 5 === 0 || done === todo.length) renderBatch(false);
    await new Promise((r) => setTimeout(r, 0));
  }
  $("btext").textContent = (batchStop ? "Stopped. " : "Done. ") + `${batchResults.length} images checked in ${fmtTime((performance.now() - t0) / 1000)}.`;
  renderBatch(true);
  batchBusy = false; $("bstop").hidden = true; $("brun").hidden = false;
}

function aucOf(rows) {                          // rank-based ROC-AUC with tie handling
  const pos = rows.filter((r) => r.label === "ai"), neg = rows.filter((r) => r.label === "real");
  if (!pos.length || !neg.length) return null;
  const all = rows.map((r) => ({ p: r.p, y: r.label === "ai" })).sort((a, b) => a.p - b.p);
  let i = 0, sumPos = 0;
  while (i < all.length) { let j = i; while (j < all.length && all[j].p === all[i].p) j++; const rank = (i + 1 + j) / 2; for (let k = i; k < j; k++) if (all[k].y) sumPos += rank; i = j; }
  return (sumPos - pos.length * (pos.length + 1) / 2) / (pos.length * neg.length);
}
function f1(tp, fp, fn) { const p = tp + fp ? tp / (tp + fp) : 0, r = tp + fn ? tp / (tp + fn) : 0; return p + r ? 2 * p * r / (p + r) : 0; }

function stats(rows) {
  const thr = manifest.threshold;
  let rr = 0, ra = 0, ar = 0, aa = 0;            // two-way at the decision line
  const z = { real: { real: 0, uncertain: 0, "AI-generated": 0 }, ai: { real: 0, uncertain: 0, "AI-generated": 0 } };
  for (const r of rows) {
    const aiPred = r.p >= thr;
    if (r.label === "real") { aiPred ? ra++ : rr++; } else { aiPred ? aa++ : ar++; }
    z[r.label][r.verdict]++;
  }
  const n = rows.length;
  const acc = n ? (rr + aa) / n : 0;
  const macro = (f1(aa, ra, ar) + f1(rr, ar, ra)) / 2;
  const answered = rows.filter((r) => r.verdict !== "uncertain");
  const rightAnswered = answered.filter((r) => (r.verdict === "AI-generated") === (r.label === "ai")).length;
  return { n, rr, ra, ar, aa, z, acc, macro, auc: aucOf(rows), unc: n - answered.length, answeredAcc: answered.length ? rightAnswered / answered.length : null };
}

function matrix(el, cols, rowsData) {
  const max = Math.max(1, ...rowsData.flatMap((r) => r.cells));
  let h = "<table class='cm'><thead><tr><th></th>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
  rowsData.forEach((r) => {
    h += `<tr><th>${r.name}</th>` + r.cells.map((v, i) => `<td class="${r.good.includes(i) ? "good" : "bad"}" style="--a:${(v / max).toFixed(2)}">${v}</td>`).join("") + "</tr>";
  });
  el.innerHTML = h + "</tbody></table>";
}

let shownRows = 0;
function renderBatch(final) {
  $("bresults").hidden = false;
  const ok = batchResults.filter((r) => r.ok), lab = ok.filter((r) => r.label === "real" || r.label === "ai");
  const bad = batchResults.length - ok.length;
  if (lab.length) {
    const s = stats(lab); const thr = manifest.threshold;
    $("bsum").innerHTML =
      `<div><b>${(100 * s.acc).toFixed(1)}%</b><span>accuracy at the ${thr} decision line</span></div>` +
      `<div><b>${s.macro.toFixed(3)}</b><span>macro-F1 (same measure as the notebook)</span></div>` +
      `<div><b>${s.auc == null ? "n/a" : s.auc.toFixed(3)}</b><span>ROC-AUC</span></div>` +
      `<div><b>${(100 * s.unc / s.n).toFixed(0)}%</b><span>of images fall in the uncertain zone</span></div>` +
      `<div><b>${s.answeredAcc == null ? "n/a" : (100 * s.answeredAcc).toFixed(1) + "%"}</b><span>correct when the page does give an answer</span></div>`;
    $("bcm").hidden = false;
    matrix($("cm2"), ["Predicted real", "Predicted AI"], [
      { name: "True real", cells: [s.rr, s.ra], good: [0] }, { name: "True AI", cells: [s.ar, s.aa], good: [1] }]);
    matrix($("cm3"), ["Likely real", "Uncertain", "Likely AI"], [
      { name: "True real", cells: [s.z.real.real, s.z.real.uncertain, s.z.real["AI-generated"]], good: [0] },
      { name: "True AI", cells: [s.z.ai.real, s.z.ai.uncertain, s.z.ai["AI-generated"]], good: [2] }]);
    // per-group table when the labels file has a role column
    const roles = [...new Set(lab.map((r) => r.role).filter(Boolean))];
    let g = "";
    if (roles.length) {
      g = "<table class='gt'><thead><tr><th>Test set</th><th>Images</th><th>Macro-F1</th><th>ROC-AUC</th><th>Uncertain</th></tr></thead><tbody>";
      roles.forEach((role) => { const t = stats(lab.filter((r) => r.role === role)); g += `<tr><td>${role}</td><td>${t.n}</td><td>${t.macro.toFixed(3)}</td><td>${t.auc == null ? "n/a" : t.auc.toFixed(3)}</td><td>${(100 * t.unc / t.n).toFixed(0)}%</td></tr>`; });
      g += "</tbody></table>";
    }
    $("bgroups").innerHTML = g;
  } else {
    $("bsum").innerHTML = `<div><b>${ok.length}</b><span>images scored (no labels, so no confusion matrix)</span></div>`; $("bcm").hidden = true; $("bgroups").innerHTML = "";
  }
  // parity with the notebook, when the labels file carries the notebook scores
  const ref = ok.filter((r) => r.ref != null && !isNaN(r.ref));
  if (ref.length) {
    const d = ref.map((r) => Math.abs(r.p - r.ref)), mean = d.reduce((a, b) => a + b, 0) / d.length;
    $("bparity").hidden = false;
    $("bparity").textContent = `Match with the notebook: across ${ref.length} images the website's score differs from the notebook's by ${mean.toFixed(4)} on average (largest ${Math.max(...d).toFixed(3)}). Tiny differences are normal because browsers and Python decode JPEG files slightly differently.`;
  } else $("bparity").hidden = true;
  if (bad) { $("bunread").hidden = false; $("bunread").textContent = `${bad} file(s) could not be read as images and were skipped.`; } else $("bunread").hidden = true;
  // table
  const onlyWrong = $("bwrong").checked;
  const rows = batchResults.filter((r) => !onlyWrong || (r.ok && (r.label === "real" || r.label === "ai") && (r.verdict === "uncertain" || (r.verdict === "AI-generated") !== (r.label === "ai"))));
  const show = rows.slice(0, 300);
  $("btable").innerHTML = "<thead><tr><th>Image</th><th>True</th><th>AI score</th><th>Verdict</th><th>Result</th></tr></thead><tbody>" +
    show.map((r) => {
      let res = "";
      if (r.ok && r.label) res = r.verdict === "uncertain" ? "uncertain" : (((r.verdict === "AI-generated") === (r.label === "ai")) ? "correct" : "wrong");
      const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
      return `<tr class="${res}"><td>${esc(r.path)}</td><td>${r.label || "?"}</td><td>${r.ok ? (100 * r.p).toFixed(1) + "%" : "-"}</td><td>${r.verdict}</td><td>${res || "-"}</td></tr>`;
    }).join("") + "</tbody>";
  $("bmore").textContent = rows.length > 300 ? `Showing the first 300 of ${rows.length} rows. The download has all of them.` : "";
}

function downloadCsv() {
  const thr = manifest.threshold, q = (t) => '"' + String(t).replace(/"/g, '""') + '"';
  const head = "filename,true_label,role,generator,ai_score,verdict,decision_at_" + thr + ",correct,width,height,notebook_p_fake";
  const lines = batchResults.map((r) => {
    const dec = r.ok ? (r.p >= thr ? "ai" : "real") : "";
    const correct = r.ok && r.label ? (dec === r.label ? "yes" : "no") : "";
    return [q(r.path), r.label || "", q(r.role), q(r.generator), r.ok ? r.p.toFixed(6) : "", r.verdict, dec, correct, r.w, r.h, r.ref != null && !isNaN(r.ref) ? r.ref : ""].join(",");
  });
  const blob = new Blob([[head, ...lines].join("\n")], { type: "text/csv" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "mazimacheck_batch_results.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

$("bfolder").addEventListener("change", (e) => onBatchPick(e.target.files));
$("bfiles").addEventListener("change", (e) => onBatchPick(e.target.files));
$("brun").addEventListener("click", runBatch);
$("bstop").addEventListener("click", () => { batchStop = true; $("btext").textContent = "Stopping after this image..."; });
$("bwrong").addEventListener("change", () => renderBatch(true));
$("bdl").addEventListener("click", downloadCsv);
window.__batchResults = () => batchResults;
