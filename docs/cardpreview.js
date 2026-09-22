/* Floating card image for any link to a card.
 *
 * Reading a grade or a quote without the card in front of you is guesswork, and
 * every page here is a list of card names. So: hover on a desktop, tap on a
 * phone, and the card appears.
 *
 * Attaches itself to every Scryfall link on the page rather than asking each
 * template to opt in -- a link to a card is exactly the thing that should show
 * one. An element can also name its image directly with data-img, or name the
 * card with data-card and let the registry resolve it.
 *
 * Touch rules, which are not the same as hover:
 *   tap a card        show it, and do not follow the link yet
 *   tap it again      now follow the link
 *   tap the image     nothing; it stays up
 *   tap anywhere else close it
 */
(function () {
  const REG = Object.create(null);      // card name -> image url
  const HOVER = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  const box = document.createElement("div");
  box.id = "cardpv";
  box.hidden = true;
  box.innerHTML = '<img alt="">';
  const img = box.firstChild;

  const css = document.createElement("style");
  css.textContent = `
    #cardpv{position:fixed;z-index:9999;pointer-events:none;
      filter:drop-shadow(0 12px 28px rgba(0,0,0,.6))}
    #cardpv img{display:block;width:264px;max-width:72vw;border-radius:4.75% / 3.5%;
      background:#211c33}
    #cardpv.touch{pointer-events:auto}
    #cardpv.touch img{width:min(74vw,330px)}
    @media (prefers-reduced-motion:no-preference){
      #cardpv{animation:cardpv-in .09s ease-out}
      @keyframes cardpv-in{from{opacity:0}to{opacity:1}}
    }`;

  let current = null;                   // the element the preview belongs to

  function ready() {
    document.head.appendChild(css);
    document.body.appendChild(box);
  }
  if (document.body) ready(); else document.addEventListener("DOMContentLoaded", ready);

  function imageFor(el) {
    if (el.dataset.img) return el.dataset.img;
    let key = el.dataset.card || el.textContent.trim();
    // Several lists put the grade inside the link ("A+ Overwrite the Multiverse"),
    // so falling back to link text needs the badge stripped off first.
    const bare = key.replace(/^(?:[A-F][+-]?|guest|rated \S+)\s+/i, "");
    return REG[key] || REG[key.split(" // ")[0]]
        || REG[bare] || REG[bare.split(" // ")[0]] || null;
  }

  /** The nearest thing under the pointer that stands for a card. */
  function cardEl(node) {
    const el = node && node.closest &&
      node.closest('[data-img],[data-card],a[href*="scryfall.com"]');
    // data-nopreview is for a deliberate "open this in Scryfall" button sitting
    // next to a card image that is already on screen: previewing it would be
    // redundant, and on touch it would eat the tap the button exists for.
    if (!el || el.hasAttribute("data-nopreview")) return null;
    return imageFor(el) ? el : null;
  }

  function place(x, y) {
    const w = box.offsetWidth || 264, h = box.offsetHeight || 368;
    const pad = 14;
    let left = x + 18, top = y + 18;
    if (left + w + pad > innerWidth) left = x - w - 18;
    if (left < pad) left = pad;
    if (top + h + pad > innerHeight) top = innerHeight - h - pad;
    if (top < pad) top = pad;
    box.style.left = left + "px";
    box.style.top = top + "px";
  }

  function centre() {
    const w = box.offsetWidth || 300, h = box.offsetHeight || 420;
    box.style.left = Math.max(8, (innerWidth - w) / 2) + "px";
    box.style.top = Math.max(8, (innerHeight - h) / 2) + "px";
  }

  function show(el, x, y) {
    const src = imageFor(el);
    if (!src) return;
    current = el;
    box.classList.toggle("touch", !HOVER);
    // only reveal once the bitmap is there, so no empty frame flashes
    if (img.getAttribute("src") !== src) {
      img.hidden = true;
      img.onload = () => {
        img.hidden = false;
        if (current === el) HOVER ? place(x, y) : centre();
      };
      img.src = src;
    } else {
      img.hidden = false;
    }
    img.alt = el.dataset.card || el.textContent.trim();
    box.hidden = false;
    if (!img.hidden) HOVER ? place(x, y) : centre();
  }

  function hide() {
    box.hidden = true;
    current = null;
  }

  if (HOVER) {
    document.addEventListener("mouseover", e => {
      const el = cardEl(e.target);
      if (el && el !== current) show(el, e.clientX, e.clientY);
      else if (!el && current) hide();
    });
    document.addEventListener("mousemove", e => {
      if (current && !img.hidden) place(e.clientX, e.clientY);
    });
    addEventListener("scroll", hide, { passive: true });
  } else {
    document.addEventListener("click", e => {
      if (box.contains(e.target)) return;          // the image itself stays put
      const el = cardEl(e.target);
      if (!el) { hide(); return; }
      if (current === el) return;                  // second tap: follow the link
      e.preventDefault();
      show(el);
    });
  }
  addEventListener("keydown", e => { if (e.key === "Escape") hide(); });

  window.CardPreview = {
    /** Teach it a set of cards: {name: imageUrl}. Safe to call repeatedly. */
    register(map) {
      for (const k in map) if (map[k]) REG[k] = map[k];
    },
    /** Same, from a list of records carrying name + image. */
    registerCards(list, nameKey = "name", imgKey = "image") {
      for (const c of list || []) if (c && c[imgKey]) REG[c[nameKey]] = c[imgKey];
    },
    hide,
  };
})();
