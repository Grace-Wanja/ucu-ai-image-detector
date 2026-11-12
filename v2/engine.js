// MazimaCheck engine: everything that touches the model. No page layout code lives here.
// Preprocessing mirrors training (src/data.py): honour the phone rotation flag, RGB, NATIVE-resolution
// 224 x 224 crops (centre + 4 corners; small images reflect-padded bottom/right), ImageNet normalisation,
// and the photo's score is the average of the sigmoid of the five crop logits.
const CROP = 224, MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const MAX_MB = 25;
// Fixed verdict zones. Score = estimated probability the image is AI-generated.
let REAL_BELOW = 0.45, AI_FROM = 0.65;
let session = null, manifest = null;
const MODEL_BASE = "../model/";

const $ = (id) => document.getElementById(id);
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const pct = (x) => (100 * x).toFixed(0) + "%";
function verdictFor(p) { return p < REAL_BELOW ? "real" : (p >= AI_FROM ? "AI-generated" : "uncertain"); }

// ---- model download, saved on the device after the first visit (Cache Storage) -------------------------
async function loadModel(onProgress) {
  const store = ("caches" in window) ? await caches.open("mz-model").catch(() => null) : null;
  const murl = new URL(MODEL_BASE + "manifest.json", document.baseURI).href;
  let mres = await fetch(murl, { cache: "no-cache" }).catch(() => null);
  if (mres && mres.ok) { if (store) store.put(murl, mres.clone()).catch(() => {}); }
  else mres = store ? await store.match(murl) : null;                  // offline: use the saved copy
  if (!mres) throw new Error("offline-no-model");
  manifest = await mres.json();
  const tag = "?v=" + manifest.run + "-" + manifest.bytes;             // a new model gets a new key
  const bufs = []; let got = 0, fromDevice = true;
  for (const p of manifest.parts) {
    const url = new URL(MODEL_BASE + p, document.baseURI).href + tag;
    let r = store ? await store.match(url) : null;
    if (!r) {
      fromDevice = false;
      r = await fetch(url);
      if (!r.ok) throw new Error("Could not download " + p);
      if (store) store.put(url, r.clone()).catch(() => {});
    }
    const b = new Uint8Array(await r.arrayBuffer());
    bufs.push(b); got += b.length;
    onProgress && onProgress(got / manifest.bytes, fromDevice);
  }
  if (store) {                                                          // drop parts of older models
    for (const k of await store.keys()) if (k.url.includes("model.onnx") && !k.url.endsWith(tag)) store.delete(k);
  }
  const all = new Uint8Array(got); let o = 0;
  for (const b of bufs) { all.set(b, o); o += b.length; }
  ort.env.wasm.wasmPaths = new URL("../ort/", document.baseURI).href;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(all, { executionProviders: ["wasm"] });
  if (manifest.zones) { REAL_BELOW = manifest.zones.real_below; AI_FROM = manifest.zones.ai_from; }
  return { fromDevice };
}

// ---- pixels ---------------------------------------------------------------------------------------------
async function decode(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none" });
  } catch (e) { return await createImageBitmap(file); }
}
function readPixels(bmp, sx, sy, w, h) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(bmp, sx, sy, w, h, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}
// five crops, same boxes as src/data.py five_boxes()
function fiveCrops(bmp) {
  const W = bmp.width, H = bmp.height;
  const padW = Math.max(0, CROP - W), padH = Math.max(0, CROP - H);
  const PW = W + padW, PH = H + padH;
  const boxes = [[Math.floor((PW - CROP) / 2), Math.floor((PH - CROP) / 2)], [0, 0], [PW - CROP, 0], [0, PH - CROP], [PW - CROP, PH - CROP]];
  if (!padW && !padH) return boxes.map(([x, y]) => readPixels(bmp, x, y, CROP, CROP));
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
    for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) data[k * 3 * N + c * N + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  });
  return new ort.Tensor("float32", data, [crops.length, 3, CROP, CROP]);
}

// ---- inference: one job at a time, because the runtime cannot run two at once ----------------------------
let queue = Promise.resolve();
function runModel(crops) {
  const job = queue.then(async () => Array.from((await session.run({ input: toTensor(crops) })).logit.data));
  queue = job.catch(() => {});
  return job;
}
async function scoreBitmap(bmp) {
  const logits = await runModel(fiveCrops(bmp));
  return { p: logits.map(sigmoid).reduce((a, b) => a + b, 0) / logits.length, logits };
}

// Used by the Photo view and the batch test.
async function analyseFile(file, keep = false) {
  const bmp = await decode(file);
  const W = bmp.width, H = bmp.height;
  const { p, logits } = await scoreBitmap(bmp);
  const notes = [];
  if (Math.min(W, H) < CROP) notes.push(`This image is small (${W} x ${H}). The model reads 224 x 224 pixel samples, so the result is less reliable.`);
  if (/\.jpe?g$/i.test(file.name || "") && file.size / (W * H) < 0.25)
    notes.push("This image looks heavily compressed, as happens on WhatsApp. Compression can hide the traces the model looks for.");
  if (W * H > 6e6) notes.push("This is a large photo. The five samples cover a very small part of it, so open the suspicion map for a wider look.");
  const label = verdictFor(p);
  if (label === "uncertain") notes.push(`The score sits between ${pct(REAL_BELOW)} and ${pct(AI_FROM)}. On our test images the model was close to a coin toss in this range, so do not rely on it.`);
  if (!keep && bmp.close) bmp.close();
  return { p, logits, label, notes, width: W, height: H, threshold: manifest.threshold, bmp: keep ? bmp : null };
}

// ---- suspicion map: the same model on a grid of native-resolution samples spread over the whole photo -----
// Each tile is coloured by ONE 224-pixel sample cut from its centre. It shows where the model's own scores are
// high. It is not a proof of editing, and the verdict above still comes from the five standard samples.
function gridPlan(W, H, maxTiles = 24) {
  let rows = Math.max(1, Math.round(Math.sqrt(maxTiles * H / W)));
  let cols = Math.max(1, Math.floor(maxTiles / rows));
  cols = Math.max(1, Math.min(cols, Math.floor(W / CROP)));
  rows = Math.max(1, Math.min(rows, Math.floor(H / CROP)));
  return { cols, rows };
}
async function suspicionMap(bmp, onProgress, maxTiles = 24) {
  const W = bmp.width, H = bmp.height;
  if (W < CROP || H < CROP) return null;
  const { cols, rows } = gridPlan(W, H, maxTiles);
  const tiles = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const x = Math.round(Math.min(W - CROP, Math.max(0, (c + 0.5) * W / cols - CROP / 2)));
    const y = Math.round(Math.min(H - CROP, Math.max(0, (r + 0.5) * H / rows - CROP / 2)));
    tiles.push({ c, r, x, y, p: NaN });
  }
  for (let i = 0; i < tiles.length; i += 4) {
    const part = tiles.slice(i, i + 4);
    const logits = await runModel(part.map((t) => readPixels(bmp, t.x, t.y, CROP, CROP)));
    part.forEach((t, k) => (t.p = sigmoid(logits[k])));
    onProgress && onProgress(Math.min(tiles.length, i + 4), tiles.length);
  }
  const mean = tiles.reduce((a, t) => a + t.p, 0) / tiles.length;
  return { cols, rows, tiles, mean, W, H };
}

// ---- re-save a picture the way a chat app would (shrink, then JPEG), and hand back a new bitmap ---------
async function degrade(bmp, longest, quality) {
  const k = Math.min(1, longest / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * k)), h = Math.max(1, Math.round(bmp.height * k));
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d"); ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, w, h);
  const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", quality));
  return { bmp: await createImageBitmap(blob), bytes: blob.size, w, h };
}

// ---- file facts: read from the file itself, no model involved ---------------------------------------------
async function fileFacts(file) {
  const head = new Uint8Array(await file.slice(0, 768 * 1024).arrayBuffer());
  const facts = { exif: {}, credentials: false };
  const find = (s) => { outer: for (let i = 0; i + s.length <= head.length; i++) { for (let j = 0; j < s.length; j++) if (head[i + j] !== s.charCodeAt(j)) continue outer; return i; } return -1; };
  facts.credentials = find("c2pa") >= 0;                                  // a Content Credentials label is embedded
  if (head[0] === 0xFF && head[1] === 0xD8) {                              // JPEG: look for the Exif block
    let o = 2;
    while (o + 4 < head.length && head[o] === 0xFF) {
      const marker = head[o + 1], len = (head[o + 2] << 8) | head[o + 3];
      if (marker === 0xE1 && String.fromCharCode(...head.slice(o + 4, o + 8)) === "Exif") { try { facts.exif = readExif(head, o + 10); } catch (e) {} break; }
      if (marker === 0xDA) break;
      o += 2 + len;
    }
  }
  return facts;
}
function readExif(b, t) {                                                 // t = start of the TIFF header
  const le = b[t] === 0x49;
  const u16 = (o) => le ? b[o] | (b[o + 1] << 8) : (b[o] << 8) | b[o + 1];
  const u32 = (o) => (le ? (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) : ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3])) >>> 0;
  const out = {}, names = { 0x010F: "make", 0x0110: "model", 0x0131: "software", 0x0132: "modified", 0x9003: "taken" };
  const ifd = (off) => {
    const n = u16(t + off);
    for (let i = 0; i < n && i < 200; i++) {
      const e = t + off + 2 + i * 12, tag = u16(e), type = u16(e + 2), count = u32(e + 4);
      if (tag === 0x8769) { ifd(u32(e + 8)); continue; }
      if (names[tag] && type === 2 && count < 200) {
        const at = count > 4 ? t + u32(e + 8) : e + 8;
        out[names[tag]] = String.fromCharCode(...b.slice(at, at + count)).replace(/\0+$/, "").trim();
      }
    }
  };
  ifd(u32(t + 4));
  return out;
}
