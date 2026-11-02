#!/bin/sh
# Self-play viewer upkeep: stats.json every 10 s (live training and gate numbers); once a minute, link every
# replay into games/ (named by generation; gate games as genNgate) and re-index. Links whose target has
# moved are dropped, so the list never shows a game that can't load.
SIM=${LMT_SIM:-/home/ahmet/pers/lmt-magic-sim}
REPO=${LMT_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
POOL=$SIM/mzport/magezero-py/pool/FRA_SEALED
V=$SIM/selfplay_view
i=0
while true; do
  if [ $((i % 6)) -eq 0 ]; then
    for d in $POOL/gen*/replays $POOL/gen*/gate/replays $POOL/aborted_gen*/replays; do
      [ -d "$d" ] || continue
      case $d in */gate/replays) g=$(basename $(dirname $(dirname $d)))gate;; *) g=$(basename $(dirname $d) | sed 's/^aborted_\(gen[0-9]*\)_\(run[0-9]*\)$/\1\2/');; esac
      for f in $d/*.json.gz; do [ -e "$f" ] && ln -sf "$f" "$V/games/${g}_$(basename $f)"; done
    done
    find $V/games -xtype l -delete
    cd $REPO && .venv/bin/python scripts/sim_index.py $V $SIM/simwork/card_values.json FRA > $V/index.log 2>&1
  fi
  cd $REPO && .venv/bin/python scripts/selfplay_stats.py $POOL $V/stats.json 120 > $V/stats.log 2>&1
  i=$((i + 1))
  sleep 10
done
