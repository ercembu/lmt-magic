#!/bin/bash
# Serve the grader locally. Card data is fetched, so file:// won't work (CORS).
cd "$(dirname "$0")/.." || exit 1
cp -f data/grades_FRA.json web/grades.json
cp -f data/signals_FRA.json web/signals.json
cp -f data/validation.json web/validation.json 2>/dev/null
cp -f data/precedent_FRA.json web/precedent.json 2>/dev/null
cp -f data/setinfo/FRA.json web/setinfo.json 2>/dev/null
PORT="${1:-8731}"
echo "Grader running at http://localhost:$PORT/web/"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
