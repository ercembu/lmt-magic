# MageZero self-play on FRA sealed

One AlphaZero-style agent that plays any FRA sealed deck, trained by self-play, so card values can come
from games played well rather than from XMage's stock AI. Engine: upstream XMage with MageZero's AI
ported onto it (`mage-patches/`). Search and network: [MageZero](https://github.com/WillWroble/MageZero)
(`magezero.patch`). The loop is `scripts/mz_loop.py`; the viewer is `sim/viewer.html`.

## Continue on a fresh checkout

```sh
sim/mzport/setup.sh [snapshot.tar]      # toolchain, XMage fork, MageZero, bundle, venv, decks (~20 min)
cd <repo> && LMT_SIM=~/pers/lmt-magic-sim ~/pers/lmt-magic-sim/.venv-ml/bin/python \
    scripts/mz_loop.py --gens 6 --games 120 --budget 150
```

Without a snapshot the loop starts again at gen 0, which takes about an hour to reach a network that
beats search alone. The snapshot (`snapshot.sh out.tar`) holds the newest model, `state.json` and every
generation's training files. It's too big for git, so keep it on a drive or a GitHub release.
`--replays` adds the games for the viewer.

Run the viewer on :8743 with `selfplay_refresh.sh` and `python3 -m http.server 8743` in `$LMT_SIM/selfplay_view`.

Every heavy step runs under a systemd memory cap (`systemd-run --user`). Without the caps, the laptop
(32 GB) crashed once.

## What a generation does

1. **Gate** (from gen 1 on): v*g* plays 60 games as seat A against search alone (seat B). It leads
   self-play only at 55% or more; otherwise the generation plays search-only.
2. **Play** 120 games between random deck pairs from `simdecks.tgz`. Search-only runs on 14 threads,
   CPU-bound. A network-led generation runs on 24 threads.
3. **Train** v*g+1* from v*g* on the last 8 generations, for 8 epochs. Both seats train. The newest
   generation's last 10% of games (both seats) is held out for validation, and the best validation
   epoch is deployed.

## Results so far (2026-11-02)

| | gate vs search alone | value loss (held-out) |
|---|---|---|
| v1, seat B held out, 2 epochs | 5 / 60 | 0.344 (worse than guessing the mean, 0.30) |
| v1, by-game split, 8 epochs | not run | 0.051 |
| v2 (gens 0–1) | **36 / 59 = 61%**: leads gen 2 | 0.115 |

Priority accuracy is about 81%, but only about 25% on states where the search did something other
than pass. That's the number to watch.

## Mistakes that cost time (don't repeat them)

- **Players skipped land drops** (8–11% of drops; some players went 4+ turns with lands in hand). The
  search-only heuristic values one land at about +0.06 of [-1, 1], which is less than the noise from
  150 simulations. Also "pass now, land in main 2" transposes into the same states, so pass won the tie
  twice. Fix: the MCTS never offers Pass while a land drop is available in its own main phase
  (`-Dlmt.skipLandDrops=true` restores the old behaviour). The copy-on-write speed-up was suspected
  first and ruled out by an A/B run (`-Dlmt.noPlayableCow=true`).
- **Validating on seat B of the same games.** Those are the same games from the other side, with
  flipped labels. The network memorized which cards won each game, and validation looked like "overfit
  after 2 epochs". The real problem was too little training: 2 epochs of 64-state batches is about 430
  steps.
- **More model-server processes are slower.** Separate CUDA contexts time-slice the GPU. A batch costs
  about 25 ms from 1 state to 128 (kernel launches and syncs), so the fix is more states in flight per
  batch: game threads park instead of spinning on `Thread.yield()`, requests carry 32 states, and
  there are more game threads.
- **X spells were always cast with X=0** (fixed 2026-11-02, from gen 3's self-play on). MageZero's X cap counts the
  mana already in the pool, which assumes manual tapping; the loop auto-taps, so the pool was empty and the cap came
  out at 0 without ever reaching the search. Guiding Hydra died on entry with 6 lands out. Gens 0–2 and gen 3's gate
  were played with the bug, so the four FRA X cards look worse in that data than they are.
- **A draw trained as a win for seat B.** B's states were labeled `!playerAWon`, so a game that hit the turn
  limit was a loss for A and a win for B. Each seat is now labeled by its own win, so a draw is a loss for both
  (fixed 2026-11-02, from gen 3's self-play on; the turn limit went from 30 to 50 at the same time).
- The **gen-1 data from before the land fix** taught the first network that pass was the best move with
  lands in hand. Data from an engine bug has to go, so `pool/FRA_SEALED_preLandFix` is kept aside and
  never trained on.

## Files

- `mage-patches/`: 19 commits on upstream XMage `6b07e46d8d` (branch `lmt-ai-port`). They cover
  MageZero's AI port, deterministic ids, the TRK/FRA cards, the recorder, copy-on-write, the land rule
  parked waits, X under auto-tap, and draws labeled as losses.
- `magezero.patch`: on MageZero `fb6ee21`. The server shares a port (SO_REUSEPORT) and has quieter
  logs; the trainer prints progress lines and per-epoch accuracy.
- `mz-xmage.sh`: the runner launcher (ParallelGC; heap set by `MZ_XMX`, default 11g).
- `selfplay_refresh.sh`: viewer upkeep (stats every 10 s, game index every minute).
- `log4j-debug.properties`: pass it with `MZ_JAVA_OPTS=-Dlog4j.configuration=file:...` to see each
  search decision (`actions: [Pass score … count …]`).
- `setup.sh`, `snapshot.sh`: rebuild a workspace, and pack its state.
