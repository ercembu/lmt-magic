#!/bin/bash
# Serve the grader locally. Card data is fetched, so file:// won't work (CORS).
cd "$(dirname "$0")/.." || exit 1
TARGET="${TARGET:-TRK}"
cp -f data/grades_$TARGET.json web/grades.json
cp -f data/signals_$TARGET.json web/signals.json
cp -f data/validation.json web/validation.json 2>/dev/null
cp -f data/precedent_$TARGET.json web/precedent.json 2>/dev/null
cp -f data/setinfo/$TARGET.json web/setinfo.json 2>/dev/null
.venv/bin/python scripts/setmeta.py "$TARGET" web/set.json >/dev/null
PORT="${1:-8731}"
echo "Grader running at http://localhost:$PORT/web/"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
