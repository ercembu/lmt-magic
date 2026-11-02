#!/bin/sh
# Pack the self-play state needed to continue on another checkout: the newest model version, state.json, and
# each generation's training files and logs (replays only with --replays). Restore with setup.sh <tar>.
#
#   sim/mzport/snapshot.sh out.tar [--replays]
#
# Too big for git (each generation ~280 MB of training data, each model ~146 MB): keep it on a drive or
# attach it to a GitHub release.
set -eu
OUT=$(realpath "$1"); REPLAYS=${2:-}
M=${LMT_SIM:-$HOME/pers/lmt-magic-sim}/mzport/magezero-py
cd $M
LATEST=$(ls -d models/FRA_SEALED/ver* | grep -E 'ver[0-9]+$' | sort -V | tail -1)
FILES="$LATEST pool/FRA_SEALED/state.json"
for g in pool/FRA_SEALED/gen*; do
  for f in player_a.hdf5 player_b.hdf5 pairs.tsv game.yml play.log train.log gate/play.log gate/pairs.tsv gate/game.yml; do
    [ -e $g/$f ] && FILES="$FILES $g/$f"
  done
  [ -n "$REPLAYS" ] && [ -d $g/replays ] && FILES="$FILES $g/replays"
  [ -n "$REPLAYS" ] && [ -d $g/gate/replays ] && FILES="$FILES $g/gate/replays"
done
tar cf "$OUT" $FILES
echo "$OUT: $(du -h "$OUT" | cut -f1), model $LATEST, $(ls -d pool/FRA_SEALED/gen* | wc -l) generations"
