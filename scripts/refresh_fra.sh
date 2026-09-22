#!/bin/bash
# Run once Draftsim publishes the Reality Fracture set review (expected before
# the 2026-09-25 prerelease). Adds the expert rating feature, which measured
# +0.076 spearman across 25 sets, and regrades.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/expert_grades.py FRA
if [ ! -f data/expert/FRA.json ]; then
  echo "No FRA review published yet — nothing to do."; exit 0
fi
$PY scripts/mtgazone.py FRA 2>/dev/null || true
$PY scripts/consensus.py FRA || true
$PY scripts/train.py
$PY scripts/calibrate.py
$PY scripts/predict.py FRA
$PY scripts/signals.py FRA
cp -f data/grades_FRA.json web/grades.json
cp -f data/signals_FRA.json web/signals.json
cp -f data/calibration.json web/calibration.json
cp -f data/validation.json web/validation.json
echo "Regraded with expert ratings."

# publish: rebuild the Pages site and push, so the phone can pull it
./scripts/build_pwa.sh FRA
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add -A docs data/grades_FRA.json data/signals_FRA.json \
              data/calibration.json data/validation.json data/expert 2>/dev/null || true
  git commit -m "Regrade FRA with expert ratings" >/dev/null 2>&1 \
    && git push && echo "Pushed. Hit Refresh on your phone." \
    || echo "Nothing to commit, or push failed — check 'git status'."
fi
