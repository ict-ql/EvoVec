#!/usr/bin/env bash
set -euo pipefail

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

prompt="$tmpdir/prompt.txt"
last="$tmpdir/last.txt"
log="$tmpdir/codex.log"
cat > "$prompt"

if ! codex exec \
  --skip-git-repo-check \
  --ephemeral \
  --sandbox read-only \
  --output-last-message "$last" \
  - < "$prompt" > "$log" 2>&1; then
  cat "$log" >&2
  exit 1
fi

cat "$last"
