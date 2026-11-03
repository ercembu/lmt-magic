#!/bin/bash
# Run once a set review is published for the target set (default TRK). Adds the
# expert rating feature -- worth +0.15 spearman on FRA, the largest single input
# the model has -- and regrades.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T="${1:-TRK}"
$PY scripts/expert_grades.py $T
if [ ! -f data/expert/$T.json ]; then
  echo "No $T review published yet — nothing to do."; exit 0
fi
$PY scripts/mtgazone.py $T 2>/dev/null || true
$PY scripts/consensus.py $T || true
$PY scripts/train.py
$PY scripts/calibrate.py
$PY scripts/predict.py $T
$PY scripts/signals.py $T
$PY scripts/reviews_page.py $T || true
$PY scripts/ideas.py $T || true
echo "Regraded with expert ratings."

# publish: rebuild the Pages site and push, so the phone can pull it
./scripts/build_pwa.sh $T
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add -A docs data/grades_$T.json data/signals_$T.json \
              data/calibration.json data/validation.json data/expert 2>/dev/null || true
  git commit -m "Regrade $T with expert ratings" >/dev/null 2>&1 \
    && git push && echo "Pushed. Hit Refresh on your phone." \
    || echo "Nothing to commit, or push failed — check 'git status'."
fi
