#!/usr/bin/env bash
# Publish the crates of the workspace on crates.io one by one, in the order of the
# layers (every crate after the crates it depends on), with a pause between two crates
# and a retry when crates.io rate-limits the publication of new crate names (HTTP 429:
# the error names the delay to wait). A crate whose version is already on crates.io is
# skipped, so the script resumes where a previous run stopped.
#
#   tools/publish_crates.sh            # publish (needs CARGO_REGISTRY_TOKEN or cargo login)
#   DRY_RUN=1 tools/publish_crates.sh  # print the commands only
#   PAUSE=60 tools/publish_crates.sh   # the pause between two crates (s), 30 by default
set -euo pipefail

CRATES=(
  comnum-core comnum-math
  comnum-dsp
  comnum-fec comnum-mapping comnum-mimo comnum-waveform comnum-channel comnum-hardware comnum-metrics
  comnum-sim
  comnum-systems comnum-spec
  comnum
)
PAUSE="${PAUSE:-30}"
DRY_RUN="${DRY_RUN:-}"

version() {
  cargo metadata --no-deps --format-version 1 | python3 -c "
import json, sys
name = sys.argv[1]
for p in json.load(sys.stdin)['packages']:
    if p['name'] == name:
        print(p['version'])
" "$1"
}

wait_seconds() {
  # the delay a rate-limit error of crates.io names: "try again after <HTTP date>" (the
  # form of 2026) or "after N seconds", plus 10 s of margin; 600 s if it names none. It
  # never fails: under `set -euo pipefail` a grep that finds nothing would end the script
  # instead of waiting (the first run of v0.2.0 stopped so after five crates).
  python3 - "$1" <<'PY' || echo 600
import re, sys, time
from email.utils import parsedate_to_datetime
text = sys.argv[1]
m = re.search(r"after ([A-Z][a-z]{2}, \d{1,2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2} GMT)", text)
if m:
    print(max(10, int(parsedate_to_datetime(m.group(1)).timestamp() - time.time()) + 10))
else:
    m = re.search(r"after (\d+) second", text)
    print(int(m.group(1)) + 10 if m else 600)
PY
}

published() {
  # whether crate $1 at version $2 is on crates.io
  local body
  body=$(curl -sf -A "comnum-release (vincent.choqueuse@gmail.com)" "https://crates.io/api/v1/crates/$1/$2" 2>/dev/null) || return 1
  python3 -c "import json, sys; sys.exit(0 if 'version' in json.loads(sys.argv[1]) else 1)" "$body"
}

for crate in "${CRATES[@]}"; do
  v=$(version "$crate")
  if published "$crate" "$v"; then
    echo "== $crate $v: already on crates.io, skipped"
    continue
  fi
  attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "== $crate $v: publishing (attempt $attempt)"
    if [ -n "$DRY_RUN" ]; then
      echo "   cargo publish -p $crate --no-verify"
      break
    fi
    if out=$(cargo publish -p "$crate" --no-verify 2>&1); then
      echo "$out" | tail -2
      break
    fi
    echo "$out" | tail -5
    if echo "$out" | grep -q "429"; then
      wait_s=$(wait_seconds "$out")
      echo "   rate-limited: waiting ${wait_s}s"
      sleep "$wait_s"
      continue
    fi
    if echo "$out" | grep -q "already uploaded\|already exists"; then
      echo "   already uploaded: skipped"
      break
    fi
    if [ "$attempt" -ge 3 ]; then
      echo "   giving up on $crate" >&2
      exit 1
    fi
    sleep "$PAUSE"
  done
  # let the index of crates.io see the new crate before the next one depends on it
  echo "   pause ${PAUSE}s"
  [ -n "$DRY_RUN" ] || sleep "$PAUSE"
done
echo "== every crate is on crates.io"
