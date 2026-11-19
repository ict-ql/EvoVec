#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

MANIFEST="${MANIFEST:-runs/manifests/tsvc_kernels.jsonl}"
TSVC_ROOT="${TSVC_ROOT:-runs/tsvc_kernels/v1}"
LLM_CONFIG="${LLM_CONFIG:-config/llm.yaml}"
LLVM_PROJECT_ROOT="${LLVM_PROJECT_ROOT:-/path/to/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$LLVM_PROJECT_ROOT/build}"
ALIVE_TV="${ALIVE_TV:-/path/to/alive2/build/alive-tv}"
DECLARATIONS="${DECLARATIONS:-runs/ablation/intel_intrinsic_declarations.json}"
JOBS="${JOBS:-8}"
FUZZ_RUNS="${FUZZ_RUNS:-10000}"

BM25_RECOMMENDATIONS="${BM25_RECOMMENDATIONS:-runs/ablation/bm25/recommendations.jsonl}"
RANDOM_SEED0_RECOMMENDATIONS="${RANDOM_SEED0_RECOMMENDATIONS:-runs/ablation/random_seed0/recommendations.jsonl}"
RANDOM_SEED1_RECOMMENDATIONS="${RANDOM_SEED1_RECOMMENDATIONS:-runs/ablation/random_seed1/recommendations.jsonl}"
RANDOM_SEED2_RECOMMENDATIONS="${RANDOM_SEED2_RECOMMENDATIONS:-runs/ablation/random_seed2/recommendations.jsonl}"
FULL_RECOMMENDATIONS="${FULL_RECOMMENDATIONS:-runs/ablation/full_list/recommendations.jsonl}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    echo "missing executable: $1" >&2
    exit 1
  fi
}

require_file "$MANIFEST"
require_file "$LLM_CONFIG"
require_file "$BM25_RECOMMENDATIONS"
require_file "$RANDOM_SEED0_RECOMMENDATIONS"
require_file "$RANDOM_SEED1_RECOMMENDATIONS"
require_file "$RANDOM_SEED2_RECOMMENDATIONS"
require_file "$FULL_RECOMMENDATIONS"
require_executable "$ALIVE_TV"
require_executable "$LLVM_BUILD_DIR/bin/clang"
require_executable "$LLVM_BUILD_DIR/bin/llvm-config"
require_executable "$LLVM_BUILD_DIR/bin/llvm-link"

FULL_TOP_K="$(
  python3 - "$FULL_RECOMMENDATIONS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    first = json.loads(next(source))
print(len(first["recommendations"]))
PY
)"

if [[ ! -s "$DECLARATIONS" || "${FORCE_DECLARATIONS:-0}" == "1" ]]; then
  echo "[setup] resolving declarations for $FULL_TOP_K intrinsics"
  python3 scripts/resolve_llvm_intrinsic_declarations.py \
    --llvm-project-root "$LLVM_PROJECT_ROOT" \
    --llvm-config "$LLVM_BUILD_DIR/bin/llvm-config" \
    --recommendations-jsonl "$FULL_RECOMMENDATIONS" \
    --recommendation-top-k "$FULL_TOP_K" \
    --out-json "$DECLARATIONS"
else
  echo "[setup] reusing declarations: $DECLARATIONS"
fi

run_pass1() {
  local name="$1"
  local recommendations="$2"
  local top_k="$3"
  local out_dir="runs/tsvc_ir_opt/ablation_${name}_pass1"

  echo "[$name] starting pass@1 with top-k=$top_k"
  python3 scripts/run_tsvc_ir_optimization_experiment.py \
    --manifest "$MANIFEST" \
    --tsvc-root "$TSVC_ROOT" \
    --recommendations-jsonl "$recommendations" \
    --intrinsic-declarations-json "$DECLARATIONS" \
    --out-dir "$out_dir" \
    --skip-existing \
    --resume-completed \
    --skip-scalar-ir-eval \
    --modes guided \
    --limit 0 \
    --max-iters 1 \
    --recommendation-top-k "$top_k" \
    --alive2 \
    --alive-tv "$ALIVE_TV" \
    --differential-fuzz \
    --llvm-link "$LLVM_BUILD_DIR/bin/llvm-link" \
    --clang "$LLVM_BUILD_DIR/bin/clang" \
    --fuzz-runs "$FUZZ_RUNS" \
    --fuzz-input-timeout 5 \
    --fuzz-timeout 1800 \
    --run-timeout 300 \
    --jobs "$JOBS" \
    --llm-config "$LLM_CONFIG"
}

run_pass1 bm25 "$BM25_RECOMMENDATIONS" 20
run_pass1 random_seed0 "$RANDOM_SEED0_RECOMMENDATIONS" 20
run_pass1 random_seed1 "$RANDOM_SEED1_RECOMMENDATIONS" 20
run_pass1 random_seed2 "$RANDOM_SEED2_RECOMMENDATIONS" 20
run_pass1 full_list "$FULL_RECOMMENDATIONS" "$FULL_TOP_K"

echo "All retriever pass@1 experiments completed."
