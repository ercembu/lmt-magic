#!/bin/bash
# Full pipeline, start to finish. Each stage caches, so re-runs are cheap.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
TARGET="${1:-TRK}"

echo "==> 1/7  card data from Scryfall"
$PY scripts/fetch_scryfall.py

echo "==> 2/7  17lands public game data (~1.2GB, cached)"
scripts/download_labels.sh

echo "==> 3/7  aggregating per-card win rates"
$PY scripts/aggregate_labels.py FRA HOB MSH SOS TMT ECL TLA EOE FIN TDM DFT DSK \
    BLB OTJ MKM LCI WOE LTR MOM ONE BRO DMU

echo "==> 3b/7 combining the published reviews"
$PY scripts/mtgazone.py "$TARGET" 2>/dev/null || true
$PY scripts/consensus.py "$TARGET" || true

echo "==> 4/7  measuring precedent for $TARGET's mechanics"
$PY scripts/precedent_scan.py "$TARGET"

echo "==> 5/7  training + leave-one-set-out validation"
$PY scripts/train.py

echo "==> 6/7  calibrating trust bands from out-of-fold error"
$PY scripts/calibrate.py

echo "==> 7/7  grading $TARGET"
$PY scripts/predict.py "$TARGET"

$PY scripts/signals.py "$TARGET"
./scripts/build_pwa.sh "$TARGET"
echo
echo "Done. Run  ./web/serve.sh  and open http://localhost:8731/"
