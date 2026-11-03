#!/bin/bash
# Assemble docs/ — what GitHub Pages serves. Keeps web/ as the source and copies
# in the generated data of every graded set, so the published site is a snapshot
# of the last pipeline run for each of them.
#
#   scripts/build_pwa.sh [DEFAULT_SET]      (default TRK: the set the site opens on)
#
# Each set's data lives in docs/data/<SET>/; docs/sets.js lists the sets for the
# picker in the nav. A set is published when data/grades_<SET>.json and
# data/signals_<SET>.json exist. validation.json and calibration.json describe the
# model, not a set, so they stay at the top.
set -e
cd "$(dirname "$0")/.."
DEFAULT="${1:-TRK}"
PY="${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"
rm -rf docs && mkdir -p docs/data

cp web/index.html web/signals.html web/reviews.html web/ideas.html web/pwa.js web/lmtset.js \
   web/cardpreview.js web/sw.js web/manifest.webmanifest web/icon-192.png web/icon-512.png docs/
cp data/validation.json  docs/validation.json
cp data/calibration.json docs/calibration.json

SETS=()
for g in data/grades_*.json; do
  s=$(basename "$g" .json); s=${s#grades_}
  case "$s" in *.*) continue;; esac          # grades_FRA.prerelease.json is an archive, not a set
  [ -f "data/signals_$s.json" ] || continue
  d=docs/data/$s; mkdir -p "$d"
  cp "data/grades_$s.json"  "$d/grades.json"
  cp "data/signals_$s.json" "$d/signals.json"
  cp "data/setinfo/$s.json"     "$d/setinfo.json"    2>/dev/null || true
  cp "data/reviewdata_$s.json"  "$d/reviewdata.json" 2>/dev/null || true
  cp "data/ideas_$s.json"       "$d/ideas.json"      2>/dev/null || true
  $PY scripts/setmeta.py "$s" "$d/set.json" >/dev/null
  SETS+=("$s")
done

# the picker's list: the default set first, then the rest newest first (by release date)
$PY - "$DEFAULT" "${SETS[@]}" > docs/sets.js <<'EOF'
import json, sys
default, codes = sys.argv[1], sys.argv[2:]
sets = [json.load(open(f"docs/data/{c}/set.json")) for c in codes]
for s in sets:
    cards = json.load(open(f"data/cards/{s['code']}.json"))
    s["released"] = min(c.get("released_at", "") for c in cards)
sets.sort(key=lambda s: (s["code"] != default, [-ord(ch) for ch in s["released"]]))
if default not in codes:
    default = sets[0]["code"]
print("window.LMT_SETS = " + json.dumps({"default": default, "sets": sets}) + ";")
EOF

# Pages serves this as a plain static dir; the .nojekyll stops Jekyll eating files
touch docs/.nojekyll
DATA_HASH=$(cat docs/data/*/grades.json docs/data/*/signals.json | sha1sum | cut -c1-12)
printf '{"sets":"%s","default":"%s","data":"%s"}\n' "${SETS[*]}" "$DEFAULT" "$DATA_HASH" > docs/version.json

echo "docs/ built: ${SETS[*]} (opens on $DEFAULT)"
du -sh docs
