// MazimaCheck v2: page behaviour. The model code is in engine.js, the batch test in batch.js.
const VCLASS = { "real": "real", "AI-generated": "ai", "uncertain": "unsure" };
const VTEXT = { "real": "Likely real", "AI-generated": "Likely AI-generated", "uncertain": "Uncertain" };
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let current = null;            // the photo on the desk: { file, bmp, result, map }
let modelReady = null;

/* ---------------- views ---------------- */
function show(view) {
  document.querySelectorAll(".view").forEach((s) => (s.hidden = s.id !== "v-" + view));
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === view)));
  if (view === "history") renderHistory();
  if (view === "lab") { $("labneed").hidden = !!current; $("labgo").hidden = !current; }
  if (location.hash !== "#" + view) history.replaceState(null, "", "#" + view);
  window.scrollTo(0, 0);
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));

/* ---------------- model state ---------------- */
function startModel() {
  modelReady = loadModel((frac, fromDevice) => {
    $("statefill").style.width = Math.round(100 * frac) + "%";
    $("modelstate").textContent = fromDevice ? "Opening the saved model" : `Downloading the model, ${Math.round(100 * frac)}%`;
  }).then((info) => {
    $("modelstate").textContent = info.fromDevice ? "Model ready (saved on this device)" : "Model ready and saved for next time";
    $("statefill").style.width = "100%";
    drawGauge();
    document.querySelectorAll(".zr").forEach((e) => (e.textContent = pct(REAL_BELOW)));
    document.querySelectorAll(".za").forEach((e) => (e.textContent = pct(AI_FROM)));
  });
  modelReady.catch((e) => {
    console.error(e);
    $("modelstate").textContent = e.message === "offline-no-model"
      ? "You are offline and the model has not been saved yet. Connect once to download it."
      : "The model could not be loaded. Refresh the page to try again.";
  });
}

/* ---------------- gauge ---------------- */
function drawGauge() {
  $("zreal").style.width = (100 * REAL_BELOW) + "%";
  $("zunsure").style.width = (100 * (AI_FROM - REAL_BELOW)) + "%";
  $("zai").style.width = (100 * (1 - AI_FROM)) + "%";
  let h = "";
  for (let i = 0; i <= 20; i++) h += `<span class="${i % 2 ? "" : "major"}" style="left:${i * 5}%"></span>`;
  [REAL_BELOW, AI_FROM].forEach((v) => (h += `<em style="left:${100 * v}%">${pct(v)}</em>`));
  $("ticks").innerHTML = h;
}

/* ---------------- photo ---------------- */
function photoError(msg) { $("error").textContent = msg; $("error").hidden = false; $("droptext").textContent = "You can also drop a photo here or paste one."; }

async function onPhoto(file) {
  if (!file) return;
  $("error").hidden = true;
  if (!/^image\//.test(file.type) && !/\.(jpe?g|png|webp|bmp|gif)$/i.test(file.name || "")) return photoError("That file is not a picture. Choose a JPG, PNG or WebP photo.");
  if (file.size > MAX_MB * 1024 * 1024) return photoError(`That photo is larger than ${MAX_MB} MB. Choose a smaller copy.`);
  show("photo");
  $("droptext").textContent = "Checking the photo";
  try {
    await modelReady;
    const result = await analyseFile(file, true);
    if (current && current.bmp && current.bmp.close) current.bmp.close();
    current = { file, bmp: result.bmp, result, map: null, name: file.name || "pasted photo" };
    renderPhoto();
    fileFacts(file).then((f) => renderFacts(f)).catch(() => renderFacts({ exif: {}, credentials: false }));
    saveHistory({ kind: "photo", name: current.name, p: result.p, verdict: result.label, w: result.width, h: result.height, thumb: thumbOf(result.bmp) });
  } catch (e) {
    console.error(e);
    photoError(session ? "That photo could not be read. Try saving it again as JPG or PNG." : "The model is not ready yet. Wait for the bar to finish, then choose the photo again.");
  }
}

function renderPhoto() {
  const r = current.result;
  $("p-empty").hidden = true; $("p-result").hidden = false;
  const img = $("preview");
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  img.dataset.url = img.src = URL.createObjectURL(current.file);
  clearMap();
  $("verdictbox").className = "verdict " + VCLASS[r.label];
  $("verdict").textContent = VTEXT[r.label];
  $("score").textContent = (100 * r.p).toFixed(1) + "%";
  $("vwhy").textContent = r.label === "real" ? `Scores below ${pct(REAL_BELOW)} were real 79% of the time in our tests.`
    : r.label === "AI-generated" ? `Scores of ${pct(AI_FROM)} and above were AI 81% of the time in our tests.`
    : "In this range the model was right barely more often than a coin toss.";
  $("pin").style.left = (100 * r.p) + "%"; $("pinval").textContent = (100 * r.p).toFixed(0) + "%";
  $("notes").innerHTML = r.notes.map((n) => `<li>${esc(n)}</li>`).join("");
  $("facts").innerHTML = "";
  $("droptext").textContent = "You can also drop a photo here or paste one.";
}

function renderFacts(f) {
  const r = current.result, file = current.file, mp = r.width * r.height / 1e6;
  const kb = file.size / 1024, rows = [];
  rows.push(["Size", `${r.width} x ${r.height} pixels (${mp.toFixed(1)} megapixels)`]);
  rows.push(["File", `${(file.type || "unknown type").replace("image/", "").toUpperCase()}, ${kb > 1024 ? (kb / 1024).toFixed(1) + " MB" : kb.toFixed(0) + " KB"}`]);
  const e = f.exif || {};
  if (e.make || e.model) rows.push(["Camera", [e.make, e.model].filter(Boolean).join(" ")]);
  if (e.taken || e.modified) rows.push(["Date in file", e.taken || e.modified]);
  if (e.software) rows.push(["Software", e.software]);
  if (!e.make && !e.model && !e.taken && !e.software) rows.push(["Camera details", "None in the file. Chat apps and screenshots strip them, so this alone proves nothing."]);
  rows.push(["Content Credentials", f.credentials ? "A label is embedded in this file. MazimaCheck does not verify it yet." : "No label found"]);
  $("facts").innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
}

/* ---------------- suspicion map ---------------- */
function tileColour(p) { return p < REAL_BELOW ? "31,143,118" : (p >= AI_FROM ? "207,74,56" : "242,169,59"); }
function clearMap() {
  const c = $("mapcanvas"); c.width = c.height = 0;
  $("mapbtn").textContent = "Show the suspicion map"; $("mapbtn").disabled = false;
  $("mapstate").textContent = ""; $("maplegend").hidden = true;
}
function drawMap() {
  const m = current && current.map, img = $("preview"), c = $("mapcanvas");
  if (!m || !img.clientWidth) return;
  const w = img.clientWidth, h = img.clientHeight, dpr = window.devicePixelRatio || 1;
  c.style.width = w + "px"; c.style.height = h + "px"; c.style.left = img.offsetLeft + "px"; c.style.top = img.offsetTop + "px";
  c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  const g = c.getContext("2d"); g.scale(dpr, dpr);
  const tw = w / m.cols, th = h / m.rows, k = w / m.W;
  g.font = "700 12px Figtree, sans-serif"; g.textBaseline = "top";
  for (const t of m.tiles) {
    const col = tileColour(t.p), a = 0.18 + 0.42 * Math.abs(t.p - 0.5) * 2;
    g.fillStyle = `rgba(${col},${a.toFixed(2)})`; g.fillRect(t.c * tw, t.r * th, tw, th);
    g.strokeStyle = "rgba(255,255,255,.55)"; g.lineWidth = 1; g.strokeRect(t.c * tw + .5, t.r * th + .5, tw - 1, th - 1);
    g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.strokeRect(t.x * k, t.y * k, CROP * k, CROP * k);
    if (tw > 44) { g.fillStyle = "rgba(15,30,54,.85)"; g.fillRect(t.c * tw + 4, t.r * th + 4, 34, 18); g.fillStyle = "#fff"; g.fillText((100 * t.p).toFixed(0) + "%", t.c * tw + 8, t.r * th + 7); }
  }
}
async function toggleMap() {
  if (!current) return;
  const c = $("mapcanvas");
  if (current.map && c.width) { c.width = c.height = 0; $("mapbtn").textContent = "Show the suspicion map"; $("maplegend").hidden = true; return; }
  if (!current.map) {
    const mine = current;
    $("mapbtn").disabled = true;
    const m = await suspicionMap(mine.bmp, (d, n) => ($("mapstate").textContent = `Reading sample ${d} of ${n}`));
    $("mapbtn").disabled = false;
    if (mine !== current) return;
    if (!m) { $("mapstate").textContent = "This photo is too small for a map."; return; }
    mine.map = m;
    const hot = m.tiles.filter((t) => t.p >= AI_FROM).length;
    $("mapstate").textContent = `${m.tiles.length} samples, average ${(100 * m.mean).toFixed(0)}%. ${hot} of ${m.tiles.length} read as likely AI.`;
  }
  drawMap(); $("mapbtn").textContent = "Hide the suspicion map"; $("maplegend").hidden = false;
}
$("mapbtn").addEventListener("click", () => toggleMap().catch((e) => { console.error(e); $("mapbtn").disabled = false; $("mapstate").textContent = "The map could not be drawn for this photo."; }));
window.addEventListener("resize", () => { if ($("mapcanvas").width) drawMap(); });

["file", "cam", "file2"].forEach((id) => $(id).addEventListener("change", (e) => { onPhoto(e.target.files[0]); e.target.value = ""; }));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
document.addEventListener("drop", (e) => {
  const f = e.dataTransfer && e.dataTransfer.files[0]; if (!f) return;
  if (/^video\//.test(f.type)) onVideo(f); else if (!$("v-lab").hidden && e.dataTransfer.files.length > 1) return; else onPhoto(f);
});
document.addEventListener("paste", (e) => { const it = [...(e.clipboardData || {}).items || []].find((i) => i.type.startsWith("image/")); if (it) onPhoto(it.getAsFile()); });
$("tolab").addEventListener("click", () => show("lab"));

/* ---------------- lab: chat-app stress test ---------------- */
const STEPS = [
  { name: "As you gave it", longest: Infinity, q: null },
  { name: "Light: 2048 px, quality 85", longest: 2048, q: 0.85 },
  { name: "Typical chat: 1600 px, quality 75", longest: 1600, q: 0.75 },
  { name: "Heavy: 1280 px, quality 60", longest: 1280, q: 0.60 },
  { name: "Forwarded many times: 800 px, quality 45", longest: 800, q: 0.45 },
];
let labBusy = false;
async function runLab() {
  if (!current || labBusy) return;
  labBusy = true; $("labrun").disabled = true; $("labout").innerHTML = ""; $("labsum").textContent = "";
  const mine = current, ps = [];
  try {
    for (let i = 0; i < STEPS.length; i++) {
      const s = STEPS[i];
      $("labstate").textContent = `Step ${i + 1} of ${STEPS.length}`;
      let p, info;
      if (s.q == null) { p = mine.result.p; info = `${mine.result.width} x ${mine.result.height}`; }
      else { const d = await degrade(mine.bmp, s.longest, s.q); p = (await scoreBitmap(d.bmp)).p; info = `${d.w} x ${d.h}, ${(d.bytes / 1024).toFixed(0)} KB`; d.bmp.close && d.bmp.close(); }
      ps.push(p);
      const v = verdictFor(p);
      $("labout").insertAdjacentHTML("beforeend", `<div class="step"><div><b>${esc(s.name)}</b><br><span class="small">${info}</span></div><div class="track"><b style="width:${(100 * p).toFixed(1)}%;background:rgb(${tileColour(p)})"></b></div><div class="num">${(100 * p).toFixed(1)}%</div><div class="tag ${VCLASS[v]}">${VTEXT[v].replace("-generated", "")}</div></div>`);
    }
    const verdicts = new Set(ps.map(verdictFor)), swing = 100 * (Math.max(...ps) - Math.min(...ps));
    $("labsum").textContent = verdicts.size === 1
      ? `The verdict held across all five copies. The score moved by ${swing.toFixed(0)} points.`
      : `The verdict changed as the photo was shrunk and compressed, and the score moved by ${swing.toFixed(0)} points. A copy of this photo that has travelled through chat apps could get a different answer from the original.`;
    $("labstate").textContent = "";
  } catch (e) { console.error(e); $("labstate").textContent = "The stress test could not finish on this photo."; }
  labBusy = false; $("labrun").disabled = false;
}
$("labrun").addEventListener("click", runLab);

/* ---------------- video ---------------- */
let vRun = 0, vPoints = [];
function seekTo(video, t) {
  return new Promise((res, rej) => {
    const done = () => { video.removeEventListener("seeked", done); res(); };
    video.addEventListener("seeked", done);
    setTimeout(() => { video.removeEventListener("seeked", done); rej(new Error("seek timeout")); }, 8000);
    video.currentTime = t;
  });
}
async function onVideo(file) {
  if (!file) return;
  show("video"); $("verror").hidden = true;
  const run = ++vRun, video = $("vplayer");
  if (video.dataset.url) URL.revokeObjectURL(video.dataset.url);
  video.dataset.url = video.src = URL.createObjectURL(file);
  $("vwork").hidden = false; $("vframes").innerHTML = ""; vPoints = []; drawChart();
  $("vverdictbox").className = "verdict"; $("vverdict").textContent = "Checking frames"; $("vscore").textContent = ""; $("vvwhy").textContent = "";
  try {
    await modelReady;
    await new Promise((res, rej) => { if (video.readyState >= 1) return res(); video.onloadedmetadata = res; video.onerror = () => rej(new Error("unreadable")); });
    const dur = video.duration;
    if (!isFinite(dur) || dur <= 0 || !video.videoWidth) throw new Error("unreadable");
    const n = Math.max(2, Math.min(parseInt($("vcount").value, 10), Math.floor(dur * 4) || 2));
    const c = document.createElement("canvas"); c.width = video.videoWidth; c.height = video.videoHeight;
    const g = c.getContext("2d", { willReadFrequently: true });
    $("vstop").hidden = false;
    for (let i = 0; i < n && run === vRun; i++) {
      const t = dur * (i + 0.5) / n;
      await seekTo(video, t);
      g.drawImage(video, 0, 0);
      const bmp = await createImageBitmap(c);
      const { p } = await scoreBitmap(bmp);
      if (run !== vRun) { bmp.close && bmp.close(); break; }
      const pt = { t, p, thumb: thumbOf(bmp, 184) }; bmp.close && bmp.close();
      vPoints.push(pt); drawChart(dur);
      const v = verdictFor(p), b = document.createElement("button");
      b.className = VCLASS[v]; b.innerHTML = `<img src="${pt.thumb}" alt="Frame at ${t.toFixed(1)} seconds">${t.toFixed(1)} s, ${(100 * p).toFixed(0)}%`;
      b.addEventListener("click", () => { video.currentTime = t; });
      $("vframes").appendChild(b);
      $("vstate").textContent = `Checked frame ${i + 1} of ${n}`;
    }
    $("vstop").hidden = true;
    if (!vPoints.length) return;
    const ps = vPoints.map((x) => x.p), mean = ps.reduce((a, b) => a + b, 0) / ps.length, hot = ps.filter((p) => p >= AI_FROM).length;
    const v = verdictFor(mean);
    $("vverdictbox").className = "verdict " + VCLASS[v];
    $("vverdict").textContent = { "real": "Frame scores are mostly low", "AI-generated": "Frame scores are mostly high", "uncertain": "Frame scores are mixed" }[v];
    $("vscore").innerHTML = `<span>${(100 * mean).toFixed(1)}%</span> average AI score over ${ps.length} frames`;
    $("vvwhy").textContent = `${hot} of ${ps.length} frames read as likely AI. Highest frame ${(100 * Math.max(...ps)).toFixed(0)}%, lowest ${(100 * Math.min(...ps)).toFixed(0)}%.`;
    $("vstate").textContent = run === vRun ? "Done" : "Stopped";
    saveHistory({ kind: "video", name: file.name || "recorded clip", p: mean, verdict: v, w: video.videoWidth, h: video.videoHeight, thumb: vPoints[Math.floor(vPoints.length / 2)].thumb });
  } catch (e) {
    console.error(e); $("vstop").hidden = true;
    $("verror").textContent = e.message === "unreadable" ? "This browser could not open that video. MP4 files recorded on a phone work best." : "The video check stopped early. Try a shorter clip.";
    $("verror").hidden = false;
  }
}
function drawChart(dur) {
  const c = $("vchart"), g = c.getContext("2d"), W = c.width, H = c.height, L = 52, R = 16, T = 14, B = 32;
  g.clearRect(0, 0, W, H);
  const y = (p) => T + (1 - p) * (H - T - B), D = dur || (vPoints.length ? vPoints[vPoints.length - 1].t * 1.05 : 1), x = (t) => L + (t / D) * (W - L - R);
  [[0, REAL_BELOW, "31,143,118"], [REAL_BELOW, AI_FROM, "242,169,59"], [AI_FROM, 1, "207,74,56"]].forEach(([a, b, col]) => { g.fillStyle = `rgba(${col},.16)`; g.fillRect(L, y(b), W - L - R, y(a) - y(b)); });
  g.fillStyle = "#52627A"; g.font = "600 15px Figtree, sans-serif"; g.textAlign = "right"; g.textBaseline = "middle";
  [0, REAL_BELOW, AI_FROM, 1].forEach((p) => g.fillText(pct(p), L - 6, y(p)));
  g.textAlign = "center"; g.textBaseline = "top"; g.fillText("0 s", L, H - B + 8); g.fillText(D.toFixed(0) + " s", W - R - 10, H - B + 8);
  if (!vPoints.length) return;
  g.strokeStyle = "#0F1E36"; g.lineWidth = 2; g.beginPath();
  vPoints.forEach((pt, i) => (i ? g.lineTo(x(pt.t), y(pt.p)) : g.moveTo(x(pt.t), y(pt.p)))); g.stroke();
  vPoints.forEach((pt) => { g.fillStyle = `rgb(${tileColour(pt.p)})`; g.beginPath(); g.arc(x(pt.t), y(pt.p), 5, 0, 7); g.fill(); g.strokeStyle = "#0F1E36"; g.lineWidth = 1.5; g.stroke(); });
  c._x = x;
}
$("vchart").addEventListener("click", (e) => {
  const c = $("vchart"); if (!vPoints.length || !c._x) return;
  const px = (e.clientX - c.getBoundingClientRect().left) * c.width / c.clientWidth;
  const near = vPoints.reduce((a, b) => (Math.abs(c._x(b.t) - px) < Math.abs(c._x(a.t) - px) ? b : a));
  $("vplayer").currentTime = near.t;
});
["vfile", "vcam"].forEach((id) => $(id).addEventListener("change", (e) => { onVideo(e.target.files[0]); e.target.value = ""; }));
$("vstop").addEventListener("click", () => { vRun++; $("vstate").textContent = "Stopped"; $("vstop").hidden = true; });

/* ---------------- history (kept on this device only) ---------------- */
const HKEY = "mz-history-v1";
function thumbOf(bmp, size = 128) {
  const k = size / Math.max(bmp.width, bmp.height), c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(bmp.width * k)); c.height = Math.max(1, Math.round(bmp.height * k));
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.6);
}
function readHistory() { try { return JSON.parse(localStorage.getItem(HKEY) || "[]"); } catch (e) { return []; } }
function saveHistory(item) {
  try {
    const list = readHistory(); list.unshift({ ...item, at: new Date().toISOString() });
    localStorage.setItem(HKEY, JSON.stringify(list.slice(0, 60)));
  } catch (e) { /* storage full or blocked: history is a convenience, so carry on */ }
}
function renderHistory() {
  const list = readHistory();
  $("hempty").hidden = !!list.length;
  $("hlist").innerHTML = list.map((h) => `<div class="hitem ${VCLASS[h.verdict] || ""}"><img src="${h.thumb}" alt=""><div><b>${esc(h.name)}</b><span>${h.kind === "video" ? "Video" : "Photo"}, ${h.w} x ${h.h}, ${new Date(h.at).toLocaleString()}</span><br><span class="tag ${VCLASS[h.verdict] || ""}">${VTEXT[h.verdict] || ""}</span></div><strong>${(100 * h.p).toFixed(0)}%</strong></div>`).join("");
}
$("hclear").addEventListener("click", () => { try { localStorage.removeItem(HKEY); } catch (e) {} renderHistory(); });
$("hexport").addEventListener("click", () => {
  const q = (t) => '"' + String(t).replace(/"/g, '""') + '"';
  const lines = readHistory().map((h) => [h.at, h.kind, q(h.name), h.p.toFixed(4), h.verdict, h.w, h.h].join(","));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([["checked_at,kind,name,ai_score,verdict,width,height", ...lines].join("\n")], { type: "text/csv" }));
  a.download = "mazimacheck_history.csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
});

/* ---------------- install and offline ---------------- */
let installEvent = null;
window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); installEvent = e; $("install").hidden = false; });
$("install").addEventListener("click", async () => { if (!installEvent) return; installEvent.prompt(); await installEvent.userChoice; installEvent = null; $("install").hidden = true; });
if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("sw.js").catch((e) => console.warn(e));

/* ---------------- start ---------------- */
const first = location.hash.slice(1);
show(["photo", "video", "lab", "history", "about"].includes(first) ? first : "photo");
startModel();
window.__mz = { onPhoto, onVideo, runLab, toggleMap, get current() { return current; }, get vPoints() { return vPoints; }, ready: () => modelReady };
