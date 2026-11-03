/* Which set the pages show. Loaded in <head>, after sets.js, before any page script.
 *
 * Order of precedence: ?set=XXX in the address, then the set last picked on this
 * device, then the site's default (sets.js, written by build_pwa.sh). Page scripts
 * fetch per-set data through LMT.D("grades.json") -> "data/XXX/grades.json".
 */
(function () {
  const S = window.LMT_SETS || { default: null, sets: [] };
  const codes = S.sets.map(s => s.code);
  const KEY = "lmt-set";
  let code = (new URLSearchParams(location.search).get("set") || "").toUpperCase();
  if (!codes.includes(code)) {
    try { code = localStorage.getItem(KEY); } catch (e) { code = null; }
  }
  if (!codes.includes(code)) code = S.default || codes[0] || null;
  try { if (code) localStorage.setItem(KEY, code); } catch (e) {}

  window.LMT = {
    code,
    sets: S.sets,
    D: f => code ? `data/${code}/${f}` : f,
    choose(c) {
      try { localStorage.setItem(KEY, c); } catch (e) {}
      const u = new URL(location.href);
      u.searchParams.set("set", c);
      location.href = u.toString();
    }
  };
})();
