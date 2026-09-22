/* Offline shell + explicit data refresh.
 *
 * Game stores have bad wifi, so the app must work with no connection at all:
 * the shell and the last-seen card data are both cached. Data is network-first
 * so a refresh picks up new grades, falling back to cache the moment the
 * network is unavailable.
 */
const SHELL = "fra-shell-v3";
const DATA = "fra-data-v1";
const SHELL_FILES = [
  "./", "./index.html", "./signals.html", "./reviews.html", "./ideas.html", "./cardpreview.js",
  "./manifest.webmanifest", "./icon-192.png", "./icon-512.png"
];
const DATA_FILES = ["grades.json", "signals.json", "validation.json",
                    "calibration.json", "setinfo.json", "reviewdata.json", "ideas.json"];

self.addEventListener("install", e => {
  e.waitUntil((async () => {
    const c = await caches.open(SHELL);
    await Promise.allSettled(SHELL_FILES.map(f => c.add(f)));
    self.skipWaiting();
  })());
});

self.addEventListener("activate", e => {
  e.waitUntil((async () => {
    const keep = [SHELL, DATA];
    for (const k of await caches.keys()) if (!keep.includes(k)) await caches.delete(k);
    self.clients.claim();
  })());
});

const isData = url => DATA_FILES.some(f => url.pathname.endsWith(f));

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;

  if (isData(url)) {
    // network-first: a refresh should actually see new grades
    e.respondWith((async () => {
      try {
        const r = await fetch(e.request, { cache: "no-store" });
        if (r && r.ok) (await caches.open(DATA)).put(e.request, r.clone());
        return r;
      } catch (err) {
        const hit = await caches.match(e.request);
        if (hit) return hit;
        throw err;
      }
    })());
    return;
  }

  if (url.origin === self.location.origin) {
    // cache-first for the app shell
    e.respondWith((async () => {
      const hit = await caches.match(e.request);
      if (hit) return hit;
      try {
        const r = await fetch(e.request);
        if (r && r.ok) (await caches.open(SHELL)).put(e.request, r.clone());
        return r;
      } catch (err) {
        return caches.match("./index.html");
      }
    })());
  }
  // card images on cards.scryfall.io fall through to the network
});

self.addEventListener("message", e => {
  if (e.data === "refresh-data") {
    caches.delete(DATA).then(() => e.source && e.source.postMessage("data-cleared"));
  }
});
