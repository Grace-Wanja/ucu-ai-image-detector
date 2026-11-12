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
