// Runs the detector entirely in the browser. Mirrors src/data.py (preprocessing) and src/predict.py (verdict).
// Preprocessing rules: honour the phone rotation flag, 3-channel RGB, NATIVE-resolution 224x224 crops
// (centre + 4 corners, small images reflect-padded at the bottom/right), ImageNet normalisation,
// average the sigmoid of the five logits.
const CROP = 224, MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const UNCERTAIN_BAND = 0.10, MAX_MB = 25;
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
  $("thrmark").style.left = (100 * manifest.threshold) + "%";
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
  const unsure = Math.abs(p - thr) < UNCERTAIN_BAND;
  if (unsure) notes.push("The score is close to the decision line, so treat this as uncertain.");
  const label = unsure ? "uncertain" : (p >= thr ? "AI-generated" : "real");
  return { p, logits, label, notes, width: W, height: H, threshold: thr };
}

function show(r, file) {
  $("error").hidden = true;
  const cls = { "real": "real", "AI-generated": "ai", "uncertain": "unsure" }[r.label];
  const text = { "real": "Likely real", "AI-generated": "Likely AI-generated", "uncertain": "Uncertain" }[r.label];
  $("verdict").textContent = text; $("verdict").className = cls;
  $("prob").textContent = `Probability that this image is AI-generated: ${Math.round(r.p * 100)}%`;
  $("probbar").style.width = Math.round(r.p * 100) + "%";
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
