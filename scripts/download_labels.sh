#!/bin/bash
# Download 17lands public game-data dumps (one per set). Resumable.
cd "$(dirname "$0")/../data/raw" || exit 1
SETS="HOB MSH SOS TMT ECL TLA EOE FIN TDM DFT DSK BLB OTJ MKM LCI WOE LTR MOM ONE BRO DMU"
for S in $SETS; do
  F="game_data_public.$S.PremierDraft.csv.gz"
  if [ -f "$F" ]; then
    # verify size matches remote; skip if complete
    remote=$(curl -sI -m 30 "https://17lands-public.s3.amazonaws.com/analysis_data/game_data/$F" | grep -i '^content-length' | tr -d '\r' | awk '{print $2}')
    local=$(stat -c%s "$F")
    [ "$remote" = "$local" ] && { echo "$S: complete"; continue; }
  fi
  echo "$S: downloading..."
  curl -s --retry 3 -C - -m 1800 "https://17lands-public.s3.amazonaws.com/analysis_data/game_data/$F" -o "$F"
done
echo "ALL DOWNLOADS DONE"
ls -la
