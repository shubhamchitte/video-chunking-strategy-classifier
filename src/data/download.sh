#!/usr/bin/env bash
# Download a sampled subset (10%) of the public Video-MME benchmark into ./data/.
# Rerunnable and resumable: files that already pass their checksum are skipped,
# partial files are resumed with `curl -C -`. Run from the repo root.
#
#   bash src/data/download.sh            # metadata + 90 videos + 78 subtitles
#   JOBS=2 bash src/data/download.sh     # fewer parallel downloads
#
# Outputs (all gitignored, never commit):
#   data/meta/{test.jsonl,manifest.json,checksums.sha256,sample_ids.txt}
#   data/videos/{video_id}.mp4
#   data/subtitles/{video_id}.srt
#   data/download_failures.txt   (persistent failures, empty if none)
set -uo pipefail

BASE="https://huggingface.co/datasets/aintropy-ai/video-mme-sample10/resolve/main"
DATA="data"
JOBS="${JOBS:-4}"
MIN_FREE_GB=25

if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"
elif command -v shasum >/dev/null 2>&1; then SHA="shasum -a 256"
else echo "ERROR: need sha256sum or shasum" >&2; exit 1; fi
export SHA BASE DATA

mkdir -p "$DATA/meta" "$DATA/videos" "$DATA/subtitles"

# 1. Free disk space (KB available on the filesystem holding data/).
free_kb=$(df -Pk "$DATA" | awk 'NR==2 {print $4}')
free_gb=$((free_kb / 1024 / 1024))
echo "Free disk: ${free_gb} GB"
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  echo "ERROR: less than ${MIN_FREE_GB} GB free, aborting" >&2; exit 1
fi

# 2. Metadata (small; always refreshed).
for f in qa/test.jsonl manifest.json checksums.sha256 sample_ids.txt; do
  curl -sSL --fail --retry 5 -o "$DATA/meta/$(basename "$f")" "$BASE/$f" \
    || { echo "ERROR: could not fetch $f" >&2; exit 1; }
done

# Expected "<sha>  <repo path>" lines for videos and subtitles only.
EXPECTED="$DATA/meta/expected_media.sha256"
grep -E '  (videos/.*\.mp4|subtitles/.*\.srt)$' "$DATA/meta/checksums.sha256" > "$EXPECTED"
echo "Media entries in checksums.sha256: $(wc -l < "$EXPECTED" | tr -d ' ')"

# Video ids from manifest.json must all have a checksum entry.
python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['video_ids']))" \
  "$DATA/meta/manifest.json" > "$DATA/meta/manifest_ids.txt"
while read -r id; do
  grep -q "  videos/${id}.mp4$" "$EXPECTED" || echo "WARN: no checksum for videos/${id}.mp4" >&2
done < "$DATA/meta/manifest_ids.txt"

# fetch_one "<sha>  <repo path>": skip if valid, else download (resume) and verify.
fetch_one() {
  local sha="${1%%  *}" path="${1#*  }"
  local dest="$DATA/$path"   # videos/x.mp4 -> data/videos/x.mp4
  if [ -f "$dest" ] && [ "$($SHA "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "ok (cached) $path"; return 0
  fi
  curl -sSL -C - --retry 5 --fail -o "$dest" "$BASE/$path"
  if [ -f "$dest" ] && [ "$($SHA "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "ok $path"; return 0
  fi
  echo "FAIL $path"; return 1
}
export -f fetch_one

run_pass() {  # $1 = list file; prints FAIL lines into $2
  tr '\n' '\0' < "$1" | xargs -0 -P "$JOBS" -I{} bash -c 'fetch_one "$1"' _ {} \
    | tee "$2.log" | grep '^FAIL ' | sed 's/^FAIL //' > "$2" || true
}

# 3. First pass, then one retry for failures (corrupt partials are deleted first).
run_pass "$EXPECTED" "$DATA/meta/pass1_failed.txt"
if [ -s "$DATA/meta/pass1_failed.txt" ]; then
  echo "Retrying $(wc -l < "$DATA/meta/pass1_failed.txt" | tr -d ' ') failed file(s) once"
  grep -F -f "$DATA/meta/pass1_failed.txt" "$EXPECTED" > "$DATA/meta/retry.sha256"
  while read -r p; do rm -f "$DATA/$p"; done < "$DATA/meta/pass1_failed.txt"
  run_pass "$DATA/meta/retry.sha256" "$DATA/download_failures.txt"
else
  : > "$DATA/download_failures.txt"
fi

# 4. Final full verification of everything on disk.
(cd "$DATA" && $SHA -c --quiet meta/expected_media.sha256) \
  > "$DATA/meta/final_verify.txt" 2>&1
n_bad=$(grep -c 'FAILED' "$DATA/meta/final_verify.txt" || true)
n_vid=$(ls "$DATA/videos"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
n_srt=$(ls "$DATA/subtitles"/*.srt 2>/dev/null | wc -l | tr -d ' ')
echo "Videos on disk: $n_vid  Subtitles on disk: $n_srt  Checksum failures: $n_bad"
echo "Persistent failures: $(wc -l < "$DATA/download_failures.txt" | tr -d ' ') (see $DATA/download_failures.txt)"
[ "$n_bad" -eq 0 ] && [ ! -s "$DATA/download_failures.txt" ]
