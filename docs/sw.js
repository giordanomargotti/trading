// Service worker: l'app si apre anche senza rete con gli ultimi dati scaricati.
// Pagina e dati: prima la rete, poi la copia salvata. Libreria e icone: prima la copia salvata.
const CACHE = "mercati-v3";
const GUSCIO = ["./", "index.html", "motore.js", "manifest.webmanifest", "vendor/chart.umd.min.js",
  "icons/icon-192.png", "icons/icon-512.png", "icons/apple-touch-icon.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(GUSCIO)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys()
    .then(chiavi => Promise.all(chiavi.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== location.origin) return;
  const statico = /\/(vendor|icons)\//.test(new URL(req.url).pathname);
  if (statico) {
    e.respondWith(caches.match(req).then(c => c || fetch(req)));
    return;
  }
  e.respondWith(
    fetch(req).then(r => {
      if (r.ok) {
        const copia = r.clone();
        caches.open(CACHE).then(c => c.put(req, copia));
      }
      return r;
    }).catch(() => caches.match(req, {ignoreSearch: true}))
  );
});
