# lmt-magic — a limited grader for sets that haven't come out yet

[limited-grades](https://github.com/youssefm/limited-grades) shows you how cards in a
Magic set actually performed. It's excellent, and it's useless the week you need it
most: the week before release, when you're sitting down to a prerelease with a card
pool nobody has ever played.

This grades **Reality Fracture** (`FRA`, released 2026-10-02) with a model that has
never seen a game of it.

## How it works

The split that makes this possible:

| | source | available for FRA? |
|---|---|---|
| **Features** — what a card *is* | Scryfall | yes, fully spoiled |
| **Labels** — how a card *performed* | 17Lands | no, and that's the point |

A gradient-boosted model learns the mapping `card text → win rate` from 31 released
sets, then applies it to 280 FRA cards it has never seen. FRA never appears in
training; it is only ever the prediction target.

### Labels

17Lands' aggregate API is patron-gated for anonymous callers — `has_patron_access:
false`, date filters silently ignored, win rates stripped for anything but the current
set. So labels are computed from the **public game-data dumps** instead: one row per
game, per-card columns for opening hand / drawn / tutored / deck. ~5.7M games across
21 sets.

Doing it ourselves is better anyway: consistent metric definitions across sets, and
far more data. Spot-checked against the 18 cards 17Lands *did* expose:

```
mean |difference| = 0.015   (their n≈780, ours n≈49,000)
```

Within sampling noise on their end.

The target is the **within-set z-score of GIH win rate**. Sets differ in baseline win
rate (0.541–0.565) and spread, and grading is relative to a set's own pool anyway.

### How a grade is calculated

`explain.py <card>` prints this chain for any card.

1. **Features** — 98 numeric, set-agnostic (cost, body, types, keywords, effect classes).
2. **Text** — reminder text stripped, card name → `~`, numbers → `numval`; TF-IDF over
   2,500 word 1–3-grams learned from released sets only.
3. **Model** — LightGBM over the 2,598 combined inputs → a raw predicted z.
4. **Rank** — cards sorted by raw score. Regression predictions are shrunk toward the
   mean, so their *magnitudes* are discarded; **only the ordering survives**.
5. **Grade** — ranks mapped onto the grade distribution released sets actually have:

   ```
   A+ 1.8%  A 3.5%  A- 3.7%  B+ 4.0%  B 8.2%  B- 7.8%  C+ 11.6%
   C 14.3%  C- 10.5%  D+ 9.7%  D 10.3%  D- 10.2%  F 4.3%
   ```

   Earlier this forced predictions onto a normal. That is wrong in a measurable way:
   real win rates are fat-tailed (kurtosis +0.2 to +2.8), so a normal under-produces
   A+ — 61 across the 21 sets where limited-grades' own method gives 101. Using the
   measured curve instead moved exact-grade agreement 16%→17% and mean error 2.20→2.15.
   The mapping is monotonic, so it cannot change the ordering, only where the
   boundaries fall.
6. **Guards** — advisory only; they do not move the grade (see below).

### Honest validation

Leave-one-set-out: hold out a whole set, train on the other 20, predict it cold. The
text vectoriser is refit inside each fold so held-out cards never leak vocabulary.
Nothing about the target set enters training — `train.py` imports only `dataset` and
`textfeat`, never `guards`, `setinfo` or `precedent`.

Ranking ability:

| predictor | mean Spearman |
|---|---|
| rarity alone | 0.268 |
| hand-written heuristic | 0.292 |
| card features only | 0.471 |
| features + oracle text | 0.535 |
| + ridge ensemble | 0.545 |
| + 10 more training sets | 0.566 |
| **+ published expert ratings, where a review exists** | **0.622** |

**But Spearman flatters it.** `validate_grades.py` scores predictions against the real
grades of all 21 sets, using limited-grades' exact algorithm (fit a normal to the set's
GIH win rates, cut percentiles at their thresholds — `src/lib/CardGrader.ts`), so these
are directly comparable to what limited-grades shows:

| | exact grade | mean error | within 1 step |
|---|---|---|---|
| noise floor — replay the same games | 77% | 0.23 | 100% |
| **this model** | **16%** | **2.13** | **44%** |
| rarity only | ~0% | 2.89 | 32% |

The labels are not the problem: median 20,135 games per card, so a set's real grades are
stable to ±0.23 steps. The gap is the model.

It is much better at the ends than the middle:

| actually | model puts in same tier |
|---|---|
| A-tier (A+/A/A-) | 51% |
| B-tier | 40% |
| C-tier | 38% |
| D/F-tier | 58% |

`holdout_demo.py <SET>` shows this concretely — on Bloomburrow the model's top 10 landed
at the 93rd actual percentile and its bottom 10 at the 22nd, and even on the two *worst*
folds the top 10 still hit the 91st–95th. It finds bombs and duds; the B-to-C range is
close to a coin flip.

**Sort a pool with it. Don't settle an argument with it.**

## Set mechanics as guard rails

The model is deliberately set-agnostic — only features present in every set, which is
what lets it read a new one. `guards.py` supplies set knowledge from WotC's published
prerelease material. Precedent is *measured*, not assumed: `precedent_scan.py` counts
mechanically-equivalent cards across the 21 training sets and how they performed.

| FRA mechanic | precedents | mean z | verdict |
|---|---|---|---|
| Cadet tokens | 529 | +0.53 | model knows this |
| Heartwood (mana tokens) | 124 | −0.01 | model knows this |
| planeswalker-matters | 113 | +0.48 | model knows this |
| Prepare | 44 | −0.13 | precedent in SOS + WOE Adventure |
| Threshold | 30 | +0.22 | thin |
| Behold | 19 | +0.46 | thin |
| **Empower** | **11** | **+0.94** | **model is guessing** |

That table corrected two things I would have gotten wrong by hand: Prepare is *not* new
(Secrets of Strixhaven has 36, identically worded), and Empower genuinely is.

### Is regression even the right model?

Step 4 discards predicted magnitudes and keeps only the ordering, which suggests a
ranking objective should fit better. It doesn't. `model_compare.py`, identical folds:

| objective | Spearman | grade err |
|---|---|---|
| **L2 regression** | **0.533** | **2.176** |
| Huber | 0.530 | 2.180 |
| MAE | 0.502 | 2.270 |
| LambdaRank (NDCG) | 0.413 | 2.503 |
| rank_xendcg | 0.366 | 2.604 |
| Ridge on same inputs | 0.498 | 2.298 |

NDCG is top-heavy by construction — it optimises the head of the list and largely
ignores the middle, while Spearman weights every position equally. Wrong tool.

Ridge getting 0.498 says most of the signal is close to linear, which makes averaging
the two worthwhile: a 0.7/0.3 rank-average of GBM and ridge gives **+0.009 Spearman,
paired t p=0.020, Wilcoxon p=0.019** across 21 folds. Small, but it survives a paired
test, so it ships. Hyperparameter changes (`num_leaves=63`) did not (p=0.10).

### The guard layer: what failed, and what replaced it

Three attempts, in order:

**1. Numeric nudges — failed.** `guard_ablation.py` holds out a released set and grades
it with and without them, scoring both against that set's real grades. Across all
**11 sets with validated setinfo**:

| | value | p |
|---|---|---|
| mean Δ Spearman | +0.0005 | 0.543 |
| mean Δ grade error | −0.0015 | 0.812 |

Helped 3 sets, hurt 2, no change on 6. Noise. On FRA these nudges would have touched
42% of cards and moved 66 grades, so they are disabled
(`guards.APPLY_ADJUSTMENTS = False`).

An earlier version of this table ran on 3 sets and one of them (`MKM`) was scraped from
a soft-404 page, so it was really FRA's archetypes applied to MKM cards. The conclusion
survived the correction; the evidence for it is now much better.

**2. Guard signals as model features — no effect.** Archetype breadth/depth and text
novelty were added to the feature matrix. They rank **#4 and #5 by gain** — the model
splits on them constantly — and changed Spearman by −0.000. They are redundant with
what the text features already encode. A useful reminder that feature importance
measures how often a model splits on something, not whether it added information.

**3. Calibrated uncertainty — works, and ships.** The one thing novelty genuinely
predicts is *the model's own error*. `calibrate.py` measures this out-of-fold:

| familiarity band | n | mean grade error | within 1 |
|---|---|---|---|
| 1 (most familiar wording) | 1109 | 1.90 | 46% |
| 2 | 1109 | 1.95 | 47% |
| 3 | 1108 | 2.17 | 45% |
| 4 | 1109 | 2.32 | 42% |
| 5 (rarest wording) | 1109 | 2.40 | 40% |

Monotonic across all five bands. So every card now carries a measured error band
rather than a hand-assigned "low/medium/high", and the UI states it plainly: *band 4
of 5; cards like this missed by 2.32 grades on average*. 106 FRA cards sit in bands
4–5.

One intuition that did **not** survive testing: I expected counting out-of-vocabulary
words as maximally unfamiliar (so Empower cards would score as novel) to calibrate
better. It calibrated worse (+0.072 vs +0.080), so `mean_idf` stayed. Cards leaning on
new mechanics still get a separate, explicit precedent note.

### Does any of this generalise, or is it FRA-shaped?

`fetch_guides.py` pulls WotC's prerelease guide for every set that has one and parses
archetypes out of it. Slugs come from the news archive, not guesswork — the suffixes
are inconsistent (`...-prerelease-guide`, `...-prerelease-primer`,
`duskmourn-house-of-horror-prerelease-and-draft-guide`,
`the-guide-to-your-first-prerelease-with-aetherdrift`).

Guides use at least three layouts for the same information, all now handled:

| layout | example | seen in |
|---|---|---|
| colours before the name | `<h3>White-Blue Fatehold Surveil Aggro` | FRA, EOE, TLA |
| colours in parentheses | `<h3>Birdfolk (White-Blue)` | BLB, DSK |
| colours as mana-symbol images | `<h3>Silverquill Repartee Aggro <img alt="White mana symbol">` | SOS, MSH |

**Validated setinfo for 12 sets:** FRA, HOB, MSH, SOS, TMT, ECL, TLA, EOE, FIN, DFT,
DSK, BLB. TDM and OTJ genuinely have no archetype section (TDM is organised by single
colour, OTJ by narrative), and no guide was published at all for MKM, LCI, WOE, MOM,
ONE, BRO, DMU or LTR. Sets with five archetypes rather than ten (HOB, SOS, TMT, ECL)
are correct — those sets really do ship five.

#### Two traps worth knowing about

magic.wizards.com serves **soft-404s**: HTTP 200 with a "PAGE NOT FOUND" body, and some
misses redirect to the Daily MTG index, which embeds whichever guide is currently
featured. An early version of this scraped that page for `MKM` and wrote a setinfo file
containing Reality Fracture's archetypes and removal list — 0 of its 34 "MKM removal
spells" were MKM cards, and a guard ablation was run against it before the mistake was
caught.

So `fetch_guides.py` now checks twice: the page must be the article we asked for (title
matches the slug, not a 404 or the index), and **at least 80% of the parsed card names
must be cards in that set**. Anything else is rejected rather than written. Run
`fetch_guides.py` rather than `parse_guide.py` directly, so both checks apply.

## Layout

```
scripts/
  fetch_scryfall.py    card data for every set
  download_labels.sh   17lands public dumps
  aggregate_labels.py  raw games -> per-card GIH/GND/IWD
  features.py          card -> 98 set-agnostic features
  textfeat.py          oracle text normalisation + TF-IDF
  dataset.py           join cards to labels, z-score per set
  train.py             LightGBM + leave-one-set-out CV
  fetch_guides.py      download + validate guides for every set that has one
  parse_guide.py       WotC prerelease guide -> archetypes + removal
  precedent_scan.py    measure precedent for a set's mechanics
  guards.py            set knowledge; adjustments off by default (see below)
  predict.py           grade a set
  explain.py           trace one card from features to grade
  model_compare.py     regression vs ranking vs linear objectives
  calibrate.py         measure error-vs-novelty, build the trust bands
  signals.py           colour depth, commit-worthy cards, pair strength
  validate_signals.py  are those colour/pair rankings real? (mostly not)
  holdout_demo.py      show the damage on a set we know the answer to
  validate_grades.py   predicted vs limited-grades' real grades, all 21 sets
  guard_ablation.py    does the guard layer help? (it does not)
  validate_removal.py  regex detector vs WotC's official list
web/
  index.html           the full set, sorted into colour piles
  signals.html         what to commit to: colour depth and pair strength
```

### Aggregating grades does not make them safer

It is tempting to assume that averaging 15 card grades into a "colour depth" score cancels
the errors out. `validate_signals.py` tests that across all 21 released sets, building the
rankings from predictions and from real win rates and comparing:

| ranking | rank correlation | called it exactly | chance |
|---|---|---|---|
| colour depth (5 per set) | **+0.22** | best colour 9/31, worst colour 8/31 | 6/31 |
| pair strength (10 per set) | **+0.35** | best pair 9/31; inside real top 3 17/31 | 3/31, 9/31 |

The colour ordering is effectively noise and the worst-colour call is exactly chance. The
reason is measurable rather than mysterious:

```
real spread between colours (sd)  0.184
our error on a colour's score     0.180   -> error is 1.0x the spread
real spread between pairs   (sd)  0.137
our error on a pair's score       0.132   -> error is 1.0x the spread
```

Colours within a set are genuinely close together, and our error is the same size as the
gap we are trying to measure. The errors are also *correlated* — the model misjudges
synergy-dependent cards in a consistent direction — so averaging does not shrink them the
way independent noise would.

### Why the colour ranking can't be fixed with features

Worth writing down, because the obvious fixes are all wrong.

**It is not a ceiling problem.** Replaying the same games reproduces a set's card order at
spearman ~0.99 within every rarity — commons have a median of 48,491 games behind them. The
gap is the model, with plenty of headroom.

**It is not a colour bias.** No colour is consistently over- or under-rated across sets
(means +0.08 to −0.06 against sd 0.16–0.27; each colour is over-rated in roughly half the
sets). Within-colour ranking accuracy is uniform too: 0.544–0.569 for all five.

**It is not commons, even though commons are the model's weak spot** (spearman 0.443 vs
0.608 on mythics). A colour's best 15 cards are 38% rare, 34% uncommon, 17% mythic and only
**11% common** — depth is set by the top end, which is where the model is strongest.

**What it actually is:** the error on a colour's depth score (0.180) is the same size as the
real gap between colours (0.184), and the errors cluster by colour ~1.3x more than chance.
Feed the same card-level accuracy back in with *independent* errors and colour-depth rho
jumps from 0.13 to ~0.50 — so the structure of the error, not its size, is what kills it.

Two attempts to fix it, both tested leave-one-set-out:

| attempt | card spearman | colour-depth rho | colour error clustering |
|---|---|---|---|
| baseline | 0.555 | 0.133 | 1.30x |
| + set-context features | 0.547 (+0.002, p=0.36) | — | — |
| + per-colour context features | 0.545 (−0.011, p=0.09) | **0.057** | **1.59x** |

The second result is structural, not bad luck. **A feature that is constant within a colour
can only add colour-correlated error**: the model learns something like "colours with thin
removal underperform" from 21 sets, that relationship does not generalise, and the mistake
is then applied to every card of that colour at once — precisely the failure it was meant to
cure. Both are reverted; `context.py` is kept so the experiment is reproducible.

The only route that works is generic card-level accuracy, and the bar is high. Corrupting
true win rates with independent noise to hit a target card-level accuracy gives:

| card spearman | colour-depth rho | best colour called right |
|---|---|---|
| 0.55 (today) | 0.13 *(measured)* | 6/21 |
| 0.70 | ~0.65 | 13/21 |
| 0.83 | ~0.79 | 14/21 |

Going from 0.55 to 0.70 card-level is the price of a colour ladder worth reading. That is a
much bigger lift than better features on the same data — it likely needs a different kind of
signal altogether (human pick orders, card images, explicit synergy modelling), not another
column.

Both rankings stay on the page, each labelled with the numbers above, because what the
model thinks is still worth seeing. Neither should drive a decision. The per-colour card
lists, which come straight from individual grades, are the part of that page to trust.

(Checked once for a scaling artifact: predictions are rank-normalised while raw win rates
are fat-tailed, and a mean-of-top-15 is sensitive to tail shape. Putting both sides on the
same scale changed nothing — +0.138 to +0.133.)

### What did and didn't move the needle

Everything below was tested leave-one-set-out with a paired test.

| idea | result |
|---|---|
| **published expert ratings as a feature** | **+0.076, p<0.001 — shipped** |
| **10 more training sets (21 -> 31)** | **+0.018, p<0.001 — shipped** |
| ridge ensemble | +0.009, p=0.02 — shipped |
| set-context features (curve, removal density, burn) | +0.002, p=0.36 |
| LSA components over TF-IDF | +0.001, p=0.84 |
| sealed labels instead of draft | +0.001, p=0.81 |
| sentence embeddings (MiniLM) alongside TF-IDF | -0.000, p=0.93 |
| distilling expert ratings for sets without a review | -0.020, p<0.001 |
| per-colour context features | -0.011, p=0.09 |
| sentence embeddings replacing TF-IDF | -0.028, p<0.001 |
| IWD instead of GIH WR as target | -0.088, p<0.001 |
| LambdaRank / rank_xendcg objectives | -0.12 to -0.17 |

The pattern is consistent: **more information helps, cleverer processing of the same
information does not.** The two wins add data the model did not have. Every attempt to
squeeze more out of the existing card text — semantic embeddings, set context, LSA,
different objectives — landed within noise or hurt.

That also explains why expert ratings cannot be distilled. Training a model to imitate the
expert from card text scores 0.43 alone and *hurts* when blended (-0.020): what the expert
contributes is precisely the part that is not in the text, so no function of the text can
reconstruct it.

### Does 49% actually mean anything?

"Within one grade step" is a harsh proxy for a prerelease tool: it counts C vs C+ as a
miss, a distinction worth about 2% win rate that changes nothing about which 23 cards you
sleeve. `pool_sim.py` measures the real task instead — open six packs, build the best
two-colour deck by the model's ranking, and compare it to the best deck that pool could
have produced:

| | mean card quality of the deck built |
|---|---|
| perfect (knowing the real win rates) | +0.099 |
| **this model** | **-0.026** |
| a published set review | -0.058 |
| random picks | -0.215 |

The model recovers **60%** of the distance from random to perfect; a published set review
recovers 50%. Using it costs roughly 0.5% win rate per card against an oracle nobody has
on prerelease day.

Two things follow. Grade letters are cosmetic next to ordering — shrinking predictions
toward the middle improves every grade metric (mean error 1.99 -> 1.89) while changing the
card order not at all, so `predict.SHRINK` stays at 1.0 and keeps the top of the scale
readable. And the honest pitch for the tool is not "49% accurate" but "sorts your pool
about as well as a good set review, before one exists".

### What the set review exposed

When the Reality Fracture review landed it moved 62% of the blind grades, but with **no
systematic bias** (mean movement +0.00) — corrections went both ways. The blind ordering
already agreed with the reviewer at 0.567 spearman, about as well as the reviewer agrees
with reality.

The instructive failure was `Emrakul, the Exigent Doom`: graded **B+** blind, rated
**0/10** by the reviewer, whose reasoning was one line — *"ten mana is, well, untenable."*

Two things came out of chasing that.

**`text_len` is gone.** Removing it and `n_lines` costs -0.0014 spearman (p=0.22), i.e.
nothing, and "more words means better card" is precisely how a {10} Eldrazi with 269
characters of rules text reached B+. A proxy that contributes nothing measurable and
produces indefensible reasoning is not worth keeping.

**A cost cliff does not work, and the reason matters.** Adding `cmc>=7` / `cmc>=9`
features made Emrakul *worse* (B+ → A-). It is not a missing feature — it is the target.
GIH win rate is conditional on a card being in your deck and drawn, so the only ten-drops
the model can learn from are the ones somebody chose to play, and those had ramp behind
them:

| mana value | printed | reach training | mean GIH WR |
|---|---|---|---|
| 0-5 | 7563 | 93% | 0.5530 |
| 6-7 | 584 | 88% | 0.5620 |
| 8+ | 105 | 88% | **0.5537** |

Expensive cards that get played perform *fine*. "Unplayable" shows up as **not being
played**, which this metric cannot express at all.

The fix would be a different signal: **maindeck rate** — of the players who opened this
card, how many actually ran it. That is in the same public dumps (`deck_` and `sideboard_`
columns) and would separate "good when played" from "nobody plays it". Not implemented.

## Two pages

**`index.html`** — all 280 cards sorted into colour piles, best to worst, with a detail
panel per card (grade, percentile, trust band, set-mechanic notes, archetype fit).

**`signals.html`** — the sealed decision rather than the card list:

* *Which colours are deep* — colours ranked by the average grade of their best 15
  cards, which is roughly what you actually run of your deeper colour. A colour full of
  marginal playables with no top end is a worse place to be than a thinner one with real
  bombs, so ranking on count would mislead.
* *Cards that decide a colour on their own* — the best mono-coloured cards per colour.
  Mono-coloured, because a gold bomb commits both colours at once while a mono one
  leaves your second colour open.
* *The ten pairs, ranked* — by the quality of the best 23 cards castable in each pair
  (a sealed deck is ~23 spells + 17 lands), with each pair's signpost gold cards and
  WotC's stated payoffs.

That last section surfaces the model's clearest blind spot rather than hiding it: WotC's
archetype payoffs frequently grade C or lower, because the model reads every card alone
and cannot see that a card is good *because* the rest of the archetype is in the deck.
Where most of a pair's payoffs grade poorly, the page says so and tells you to trust
WotC over the grade.

## On your phone

The grader installs as a PWA — an app icon, offline, no store or APK needed. GitHub Pages
hosts it, which also gives the HTTPS that service workers require.

**One-time setup:** in the repo on GitHub, Settings → Pages → Source: *Deploy from a
branch*, branch `main`, folder `/docs`. After a minute it is live at
`https://ercembu.github.io/lmt-magic/`.

**Install it:** open that URL in Chrome on the phone → menu → *Add to Home screen*.

It then works with no connection at all, which matters — game store wifi is unreliable and
the whole point is to use this while building a sealed pool. The app shell and the last
card data you loaded are both cached on the device.

### Refreshing without touching a computer

Retraining is LightGBM over 8,273 cards and ~2,600 features across 31 sets, so it cannot
run on the phone, and the Refresh button in the app deliberately does not try: it only
pulls already-published grades. Triggering a build would need a GitHub token, and a public
static page cannot hold a secret without leaking it.

So the pipeline runs in GitHub Actions instead (`.github/workflows/refresh.yml`), which
means no PC is involved at all:

* **daily** — checks whether a set review has been published; retrains only if something
  actually changed, so most days cost seconds
* **from your phone** — GitHub mobile app → Actions → *Refresh grades* → *Run workflow*.
  That is the phone-triggered full retrain. Takes ~6 minutes, then hit Refresh in the app.
* **on push** — rebuilds the published site when the front end changes

Card data is cached rather than committed (57MB, regenerable), so a cache miss just
re-fetches it from Scryfall.

Locally, `./scripts/refresh_fra.sh` does the same thing, and `./scripts/build_pwa.sh FRA`
rebuilds the published site without retraining.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install pandas scikit-learn lightgbm scipy
./run.sh FRA        # ~15 min, mostly the 1.2GB download (cached after)
./web/serve.sh      # http://localhost:8731/web/
```

## Grading another set

When the next set is spoiled:

1. Add it to `SETS` in `fetch_scryfall.py`.
2. Add its guide slug to `SLUGS` in `fetch_guides.py` (find it in the news archive,
   don't guess — suffixes vary) and run it. Archetypes are read off the guide's own
   headings in whichever of the three layouts it uses, and the result is rejected
   unless the named cards actually belong to the set.
3. Describe its mechanics in `SET_MECHANICS` in `guards.py` — each needs a `pattern`
   (matches this set) and an `analogue` (matches equivalent older cards, so precedent
   counts mean something; literal token names always score zero).
4. `./run.sh <CODE>`, then `guard_ablation.py <a released set>` before trusting any
   guard change you make.

Add the previous set to the training pool once it has ~3 weeks of 17Lands data.

## What this can't do

Grades are a curve *within* the set — roughly as many A's here as in any set, so an A+
means "among the best in Reality Fracture", not "among the best in Magic".

The model reads mana cost, body, rules text and rarity. It cannot read the room. It
does not know that a format is fast, that a common is a build-around, or that two cards
combo. It systematically underrates cards whose value is purely synergistic, and its
worst misses are cards that need the format's context to make sense — on Bloomburrow it
put `Patchwork Banner` in F, and it finished at the 86th percentile.

Card data from [Scryfall](https://scryfall.com). Performance labels computed from
[17Lands](https://www.17lands.com) public datasets. Neither endorses this.
