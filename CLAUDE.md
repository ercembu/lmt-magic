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

## Open issues

- **Combat tricks are almost never cast in combat** (under 1% of casts in gen 3). Since gen 4's gate the search
  hides the opponent's hand and passes in its own upkeep, which should help; measure the casting-timing table
  again on gens 4-5 before doing more (it may be learned, or need a deeper search).
