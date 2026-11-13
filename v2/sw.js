// Keeps the app shell available offline. The scoring model is saved separately by engine.js (Cache Storage "mz-model").
const SHELL = "mz-shell-v13";
const FILES = ["./", "index.html", "style.css", "engine.js", "faces.js", "app.js", "batch.js", "manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png",
  "faces/yunet.onnx", "../ort/ort.wasm.min.js", "../ort/ort-wasm-simd-threaded.mjs", "../ort/ort-wasm-simd-threaded.wasm"];
self.addEventListener("install", (e) => { e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())); });
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k.startsWith("mz-shell-") && k !== SHELL).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
// Network first, so updates show up straight away; fall back to the saved copy when offline.
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin || u.pathname.includes("/model/")) return;
  e.respondWith(fetch(e.request, { cache: "no-cache" }).then((r) => {
    if (r.ok) { const copy = r.clone(); caches.open(SHELL).then((c) => c.put(e.request, copy)); }
    return r;
  }).catch(() => caches.match(e.request, { ignoreSearch: true })));
});
