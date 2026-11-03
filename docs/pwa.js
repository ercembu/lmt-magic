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

  // Which set this build grades (set.json, written by build_pwa.sh). Pages mark
  // the words that depend on it with data-set="name|code|cards|train_sets";
  // some of those are inserted by page scripts after load, hence the observer.
  let SET = null;
  function fillSet(root) {
    if (!SET) return;
    root.querySelectorAll("[data-set]").forEach(el => {
      const v = SET[el.dataset.set];
      if (v != null && el.textContent !== String(v)) el.textContent = v;
    });
  }
  fetch(LMT.D("set.json")).then(r => r.ok ? r.json() : null).then(s => {
    if (!s) return;
    SET = s;
    document.title = document.title.replace(/^[^—]*—/, `${s.name} —`);
    fillSet(document);
    new MutationObserver(() => fillSet(document))
      .observe(document.body, { childList: true, subtree: true });
  }).catch(() => {});

  // set picker at the start of the nav: one <select>, reloads the page on the chosen set
  function mountPicker() {
    const nav = document.querySelector("nav");
    if (!nav || !window.LMT || LMT.sets.length < 2) return;
    const sel = document.createElement("select");
    sel.setAttribute("aria-label", "Set");
    sel.style.cssText = "background:var(--ink-2);color:var(--text);border:1px solid var(--edge);" +
      "border-radius:3px;padding:2px 6px;font:inherit;font-size:12.5px;cursor:pointer";
    for (const s of LMT.sets) {
      const o = document.createElement("option");
      o.value = s.code; o.textContent = `${s.code} · ${s.name}`;
      if (s.code === LMT.code) o.selected = true;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => LMT.choose(sel.value));
    nav.insertBefore(sel, nav.firstChild);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountPicker);
  else mountPicker();

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
          const files = ["grades.json", "set.json", "signals.json", "setinfo.json", "reviewdata.json"]
            .map(LMT.D).concat(["validation.json", "calibration.json", "sets.js"]);
          const got = await Promise.all(files.map(f =>
            fetch(f + "?t=" + Date.now(), { cache: "no-store" })
              .then(r => r.ok ? r.text() : null).catch(() => null)));
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
