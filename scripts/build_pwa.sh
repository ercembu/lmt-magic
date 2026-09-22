#!/bin/bash
# Assemble docs/ — what GitHub Pages serves. Keeps web/ as the source and copies
# in the current generated data, so the published site is always a snapshot of
# the last pipeline run.
set -e
cd "$(dirname "$0")/.."
TARGET="${1:-FRA}"
rm -rf docs && mkdir -p docs

cp web/index.html web/signals.html web/reviews.html web/ideas.html web/pwa.js web/cardpreview.js web/sw.js \
   web/manifest.webmanifest web/icon-192.png web/icon-512.png docs/

cp "data/grades_$TARGET.json"  docs/grades.json
cp "data/signals_$TARGET.json" docs/signals.json
cp data/validation.json        docs/validation.json
cp data/calibration.json       docs/calibration.json
cp "data/setinfo/$TARGET.json" docs/setinfo.json 2>/dev/null || true
cp "data/reviewdata_$TARGET.json" docs/reviewdata.json 2>/dev/null || true
cp "data/ideas_$TARGET.json"      docs/ideas.json      2>/dev/null || true

# Pages serves this as a plain static dir; the .nojekyll stops Jekyll eating files
touch docs/.nojekyll
DATA_HASH=$(cat docs/grades.json docs/signals.json | sha1sum | cut -c1-12)
printf '{"set":"%s","data":"%s"}\n' "$TARGET" "$DATA_HASH" > docs/version.json

echo "docs/ built for $TARGET:"
du -sh docs; ls docs
