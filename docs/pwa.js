/* Install the service worker, and drive the Refresh button.
 *
 * The heavy pipeline (scraping the set review, retraining, regrading) runs on a
 * real machine and publishes JSON. This end just pulls the published JSON and
 * tells you when it last managed to.
 */
(function () {
  const STAMP = "fra-data-stamp";

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () =>
      navigator.serviceWorker.register("./sw.js").catch(() => {}));
  }

  function ago(iso) {
    if (!iso) return "not yet refreshed";
    const s = (Date.now() - new Date(iso)) / 1000;
    if (s < 90) return "updated just now";
    if (s < 5400) return `updated ${Math.round(s / 60)} min ago`;
    if (s < 172800) return `updated ${Math.round(s / 3600)} h ago`;
    return `updated ${Math.round(s / 86400)} days ago`;
  }

  function paint(el, msg) { if (el) el.textContent = msg; }

  window.FRA = {
    mountRefresh(btn, status) {
      paint(status, ago(localStorage.getItem(STAMP)));
      btn.addEventListener("click", async () => {
        const was = btn.textContent;
        btn.disabled = true; btn.textContent = "Refreshing…";
        try {
          if (navigator.serviceWorker && navigator.serviceWorker.controller) {
            navigator.serviceWorker.controller.postMessage("refresh-data");
          }
          const files = ["grades.json", "signals.json", "validation.json",
                         "calibration.json", "setinfo.json"];
          const got = await Promise.all(files.map(f =>
            fetch(f + "?t=" + Date.now(), { cache: "no-store" })
              .then(r => r.ok ? r.json() : null).catch(() => null)));
          if (!got[0]) throw new Error("no data");
          try { localStorage.setItem(STAMP, new Date().toISOString()); } catch (e) {}
          paint(status, "updated just now — reloading");
          setTimeout(() => location.reload(), 500);
        } catch (e) {
          paint(status, "no connection — showing the last data saved on this phone");
          btn.disabled = false; btn.textContent = was;
        }
      });
    }
  };
})();
