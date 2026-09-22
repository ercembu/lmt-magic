/* Every card element on every page must resolve to an image.
 *
 * Two invariants, checked against the built pages and their real data:
 *   1. any element that stands for a card names it (data-card or data-img);
 *      relying on link text broke three pages, because they put the grade
 *      badge inside the link.
 *   2. every card the page can name is in the registry with an image.
 */
const fs = require("fs");
const DOCS = "docs";

function openTags(js) {
  const out = [];
  const re = /<(a|button|div|span)\b([^>]*?)>/g;
  let m;
  while ((m = re.exec(js))) out.push({ tag: m[1], attrs: m[2], at: m.index });
  return out;
}

const pages = {
  "index.html": ["grades.json", d => d.map(c => [c.name, c.image])],
  "signals.html": ["signals.json", d => {
    const o = [];
    for (const col of d.colors) for (const c of d.pulls[col]) o.push([c.name, c.image]);
    for (const k in d.pairs) for (const g of ["gold", "key_cards", "top"])
      for (const c of d.pairs[k][g]) o.push([c.name, c.image]);
    return o;
  }],
  "reviews.html": ["reviewdata.json", d => d.cards.map(c => [c.name, c.image])],
  "ideas.html": ["ideas.json", d => {
    const o = [];
    const add = (l, nk = "name", ik = "image") => (l || []).forEach(c => o.push([c[nk], c[ik]]));
    d.mechanics.forEach(m => { add(m.top); add(m.quotes); });
    d.themes.forEach(t => add(t.quotes));
    d.synergies.archetypes.forEach(a => add(a.quotes));
    add(d.verdicts.high); add(d.verdicts.low); add(d.comparisons);
    add(d.comparisons, "compared_to", "compared_image");
    return o;
  }],
};

let bad = 0;
for (const [page, [dataFile, extract]] of Object.entries(pages)) {
  const html = fs.readFileSync(`${DOCS}/${page}`, "utf8");
  const js = html.slice(html.indexOf("<script>"));
  if (!html.includes('src="cardpreview.js"')) { console.log(`${page}: NOT WIRED`); bad++; continue; }

  const cardTags = openTags(js).filter(t =>
    /href="\$\{[^"]*(scryfall_uri|\.uri|compared_uri)/.test(t.attrs) ||
    /data-card=|data-img=/.test(t.attrs));
  const unnamed = cardTags.filter(t => !/data-card=|data-img=|data-nopreview/.test(t.attrs));

  const recs = extract(JSON.parse(fs.readFileSync(`${DOCS}/${dataFile}`, "utf8")));
  const named = recs.filter(([n]) => n);
  const noImage = named.filter(([, i]) => !i);
  const uniq = new Set(named.map(([n]) => n));

  console.log(`${page}: ${cardTags.length} card elements, ${unnamed.length} unnamed | ` +
              `${uniq.size} cards registered, ${noImage.length} without an image`);
  unnamed.forEach(t => console.log(`   UNNAMED <${t.tag} ${t.attrs.trim().slice(0, 80)}>`));
  [...new Set(noImage.map(([n]) => n))].slice(0, 6)
    .forEach(n => console.log(`   NO IMAGE: ${n}`));
  if (unnamed.length || noImage.length) bad++;
}
console.log(bad ? "\nFAIL" : "\nall card elements resolve");
process.exit(bad ? 1 : 0);
