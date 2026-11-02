# lmt-magic: notes for Claude

The goal is a **sealed card grader for sets that aren't out yet**: it grades every card of the next set (TRK now)
before the prerelease. Self-play (MageZero on XMage) exists to produce card values from games played well;
those values only matter if they add information the blind model lacks, measured on FRA's real win rates.
Judge self-play changes by what they do to card values, not to the agent's strength alone.

## Commits

Conventional Commits: `type(scope): summary` (lowercase imperative, no period, <= ~72 chars), an optional
short body saying why, and one logical change per commit. Scopes: `sim` (XMage fork, agent, mage-patches),
`selfplay` (`scripts/mz_loop.py` and its data), `viewer`, `model`, `web`, `data`. The same applies to commits in
the XMage fork that become `sim/mzport/mage-patches/*.patch`.

## Self-play

- Setup and the history of what went wrong: `sim/mzport/README.md`. Engine changes are patches on upstream
  XMage: commit in the fork (`$LMT_SIM/mzport/mage`), then `git format-patch` into `sim/mzport/mage-patches/`.
- Rebuild only what changed: `mvn -o -pl <module>[,Mage.MageZero] install`, then copy the changed jars from
  `Mage.MageZero/target/mage-magezero.zip` into `magezero-py/xmage/lib/`. While a runner is up, replace a jar
  by renaming a new copy over it, never by writing into it.
- Run the loop and the viewer as `systemd-run --user` units (`lmt-mzloop`, `lmt-viewer-http`,
  `lmt-viewer-refresh`). On a 16 GB machine: `LMT_PLAY_MEM=8G MZ_XMX=6g LMT_SERVER_MEM=2500M LMT_TRAIN_MEM=7G`,
  `--threads 10 --threads-online 16`. AMD GPUs: `TORCH_INDEX=https://download.pytorch.org/whl/rocm7.1`.
- Restarting a generation mid-play: stop `lmt-mzloop` and `lmt-mzplay`, move the generation's play files into
  `pool/FRA_SEALED/aborted_genN_runK/` (training skips it, the viewer still lists it), keep `gate/`.
- Check fixes in the replays (`*/replays/*.json.gz`: per-step snapshots and the game log), not in `play.log`:
  the AI's decision lines are filtered out of it.
- Don't `pkill -f` a pattern that also appears in the shell command running it: it kills that shell.

## Open issues (deferred to the round after gens 3-5)

- **Instants and flash are cast at the first chance; tricks almost never in combat.** In 42 gen-3 games, 13% of
  casts were in the caster's own upkeep or draw step and under 1% in any combat step. Search-only play is greedy
  on the board heuristic. Candidate fix: pass on your own upkeep/draw with an empty stack. Measure the timing
  table again on network-led gens first: it may be learned.
- **The search sees the opponent's hand** (the network doesn't), so games are open-handed and tricks, flash and
  counterspells are likely undervalued. Fix it if FRA's real win rates show that bias.
