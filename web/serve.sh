#!/bin/bash
# Serve the grader locally, exactly as GitHub Pages will: build docs/ (every graded set,
# opening on $TARGET, default TRK) and serve it. file:// won't work (the data is fetched).
cd "$(dirname "$0")/.." || exit 1
./scripts/build_pwa.sh "${TARGET:-TRK}" || exit 1
PORT="${1:-8731}"
echo "Grader running at http://localhost:$PORT/"
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory docs
