// MazimaCheck faces: finds faces with YuNet (a small pretrained detector, 233 KB, MIT licence). No training here.
// The model wants a 640 x 640 picture, so a photo is shrunk to fit with its shape kept, and the boxes are scaled back.
// Decoding follows OpenCV's FaceDetectorYN and was checked against it (see faces/README.txt).
const FACE_IN = 640, FACE_MIN = 0.7, FACE_NMS = 0.3;
let faceSession = null;

async function loadFaces() {
  const url = new URL("faces/yunet.onnx", document.baseURI).href;
  const store = ("caches" in window) ? await caches.open("mz-model").catch(() => null) : null;
  let r = store ? await store.match(url) : null;
  if (!r) {
    r = await fetch(url);
    if (!r.ok) throw new Error("face model download failed");
    if (store) store.put(url, r.clone()).catch(() => {});
  }
  faceSession = await ort.InferenceSession.create(new Uint8Array(await r.arrayBuffer()), { executionProviders: ["wasm"] });
}

function faceInput(bmp) {
  const sc = Math.min(FACE_IN / bmp.width, FACE_IN / bmp.height);
  const w = Math.max(1, Math.round(bmp.width * sc)), h = Math.max(1, Math.round(bmp.height * sc));
  const c = document.createElement("canvas"); c.width = c.height = FACE_IN;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.fillStyle = "#000"; g.fillRect(0, 0, FACE_IN, FACE_IN);
  g.imageSmoothingQuality = "high"; g.drawImage(bmp, 0, 0, w, h);
  const px = g.getImageData(0, 0, FACE_IN, FACE_IN).data, N = FACE_IN * FACE_IN, data = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) { data[i] = px[i * 4 + 2]; data[N + i] = px[i * 4 + 1]; data[2 * N + i] = px[i * 4]; }   // blue, green, red
  return { tensor: new ort.Tensor("float32", data, [1, 3, FACE_IN, FACE_IN]), sc };
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y), x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return i / (a.w * a.h + b.w * b.h - i || 1);
}

// Returns faces in the photo's own pixel coordinates: [{ x, y, w, h, score }], best first. Empty if the model is missing.
async function detectFaces(bmp, minScore = FACE_MIN) {
  if (!faceSession) return [];
  const { tensor, sc } = faceInput(bmp);
  const out = await enqueue(() => faceSession.run({ input: tensor }));
  const found = [];
  for (const s of [8, 16, 32]) {
    const cols = FACE_IN / s, cls = out["cls_" + s].data, obj = out["obj_" + s].data, bb = out["bbox_" + s].data;
    for (let i = 0; i < cls.length; i++) {
      const score = Math.sqrt(Math.min(1, Math.max(0, cls[i])) * Math.min(1, Math.max(0, obj[i])));
      if (score < minScore) continue;
      const r = Math.floor(i / cols), c = i % cols;
      const cx = (c + bb[4 * i]) * s, cy = (r + bb[4 * i + 1]) * s, w = Math.exp(bb[4 * i + 2]) * s, h = Math.exp(bb[4 * i + 3]) * s;
      found.push({ x: cx - w / 2, y: cy - h / 2, w, h, score });
    }
  }
  found.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const f of found) if (keep.every((k) => iou(k, f) <= FACE_NMS)) keep.push(f);
  return keep.map((f) => {                                                  // back to the photo's pixels, kept inside it
    const x = Math.max(0, f.x / sc), y = Math.max(0, f.y / sc);
    return { x, y, w: Math.min(bmp.width - x, f.w / sc), h: Math.min(bmp.height - y, f.h / sc), score: f.score };
  }).filter((f) => f.w > 2 && f.h > 2);
}

// ---- forged-face check: only active once a trained model is published at faces/forged_manifest.json -------------
// The manifest says which ONNX file to load, how a face is cut out (padding, size) and where the verdict zones are.
let forgedNet = null, forgedInfo = null;
async function cachedGet(url) {
  const store = ("caches" in window) ? await caches.open("mz-model").catch(() => null) : null;
  let r = await fetch(url, { cache: "no-cache" }).catch(() => null);
  if (r && r.ok) { if (store) store.put(url, r.clone()).catch(() => {}); return r; }
  return store ? await store.match(url) : null;
}
async function loadForged() {
  const m = await cachedGet(new URL("faces/forged_manifest.json", document.baseURI).href);
  if (!m) return;                                                           // no model published yet: no per-face verdict
  forgedInfo = await m.json();
  const url = new URL("faces/" + forgedInfo.onnx, document.baseURI).href;
  const store = ("caches" in window) ? await caches.open("mz-model").catch(() => null) : null;
  let r = store ? await store.match(url) : null;
  if (!r) { r = await fetch(url); if (!r.ok) throw new Error("forged-face model download failed"); if (store) store.put(url, r.clone()).catch(() => {}); }
  forgedNet = await ort.InferenceSession.create(new Uint8Array(await r.arrayBuffer()), { executionProviders: ["wasm"] });
}
function faceVerdict(p) {
  if (p == null || !forgedInfo) return null;
  return p < forgedInfo.zones.real_below ? "real" : (p >= forgedInfo.zones.forged_from ? "forged" : "uncertain");
}
// Cut the face the way training did and return the S x S canvas.
// square: a square around the face centre with `pad` extra on each side, black where it falls outside the photo (current model).
// otherwise: the older box-plus-padding squashed to a square.
function cropFaceCanvas(bmp, f, S, pad, square) {
  const c = document.createElement("canvas"); c.width = c.height = S;
  const g = c.getContext("2d", { willReadFrequently: true }); g.imageSmoothingQuality = "high";
  g.fillStyle = "#000"; g.fillRect(0, 0, S, S);
  if (!square) {
    const x0 = Math.max(0, f.x - pad * f.w), y0 = Math.max(0, f.y - pad * f.h);
    const x1 = Math.min(bmp.width, f.x + f.w + pad * f.w), y1 = Math.min(bmp.height, f.y + f.h + pad * f.h);
    g.drawImage(bmp, x0, y0, x1 - x0, y1 - y0, 0, 0, S, S); return c;
  }
  const cx = f.x + f.w / 2, cy = f.y + f.h / 2, side = Math.max(f.w, f.h) * (1 + 2 * pad);
  const bx0 = Math.round(cx - side / 2), by0 = Math.round(cy - side / 2), bx1 = Math.round(cx + side / 2), by1 = Math.round(cy + side / 2);
  const bw = bx1 - bx0, bh = by1 - by0;
  const sx0 = Math.max(0, bx0), sy0 = Math.max(0, by0), sx1 = Math.min(bmp.width, bx1), sy1 = Math.min(bmp.height, by1);
  if (sx1 > sx0 && sy1 > sy0) g.drawImage(bmp, sx0, sy0, sx1 - sx0, sy1 - sy0, (sx0 - bx0) * S / bw, (sy0 - by0) * S / bh, (sx1 - sx0) * S / bw, (sy1 - sy0) * S / bh);
  return c;
}
// P(forged) for one face, or null when no face model is published.
async function scoreFace(bmp, f) {
  if (!forgedNet) return null;
  const S = forgedInfo.crop.size, c = cropFaceCanvas(bmp, f, S, forgedInfo.crop.pad, !!forgedInfo.crop.square);
  const px = c.getContext("2d").getImageData(0, 0, S, S).data, N = S * S, data = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) data[k * N + i] = (px[i * 4 + k] / 255 - MEAN[k]) / STD[k];
  const out = await enqueue(() => forgedNet.run({ input: new ort.Tensor("float32", data, [1, 3, S, S]) }));
  return sigmoid(out.logit.data[0]);
}
async function scoreFaces(bmp, faces, max = 12) {
  for (const f of faces.slice(0, max)) { try { f.p = await scoreFace(bmp, f); } catch (e) { console.warn(e); f.p = null; } }
}
