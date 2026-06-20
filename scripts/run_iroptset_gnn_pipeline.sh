#!/usr/bin/env bash
set -euo pipefail

# End-to-end IR-OptSet instruction-retrieval pipeline.
#
# Examples:
#   DATA_SOURCE=huggingface USE_HF_TOKEN=0 bash scripts/run_iroptset_gnn_pipeline.sh
#   DATA_SOURCE=parquet PARQUET_DIR=/data/IR-OptSet/data bash scripts/run_iroptset_gnn_pipeline.sh

DATA_SOURCE="${DATA_SOURCE:-huggingface}"          # huggingface or parquet
DATASET="${DATASET:-YangziResearch/IR-OptSet}"
PARQUET_DIR="${PARQUET_DIR:-}"
OUT_TAG="${OUT_TAG:-pilot_10k_v1}"
USE_HF_TOKEN="${USE_HF_TOKEN:-0}"
PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX:-}"

TRAIN_SIZE="${TRAIN_SIZE:-10000}"
VAL_SIZE="${VAL_SIZE:-2000}"
TEST_SIZE="${TEST_SIZE:-2000}"
SEED="${SEED:-17}"

LABEL_IR_FIELD="${LABEL_IR_FIELD:-o3_ir}"
INTRINSIC_INCLUDE_PREFIXES="${INTRINSIC_INCLUDE_PREFIXES:-}"
INTRINSIC_EXCLUDE_PREFIXES="${INTRINSIC_EXCLUDE_PREFIXES:-}"

MAX_TOKEN_VOCAB="${MAX_TOKEN_VOCAB:-20000}"
MAX_INSTRUCTION_VOCAB="${MAX_INSTRUCTION_VOCAB:-512}"
MAX_NODES="${MAX_NODES:-512}"
MAX_EDGES="${MAX_EDGES:-4096}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-3}"
DROPOUT="${DROPOUT:-0.15}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
EVAL_KS="${EVAL_KS:-10,20,50}"
DEVICE="${DEVICE:-auto}"

SPLIT_DIR="runs/iroptset/splits/${OUT_TAG}"
LABEL_DIR="runs/iroptset/instruction_labels/${OUT_TAG}"
GRAPH_DIR="runs/iroptset/graphs/${OUT_TAG}"
MODEL_DIR="runs/iroptset/models/gnn_${OUT_TAG}"

if [[ -n "$PYTHONPATH_PREFIX" ]]; then
  export PYTHONPATH="$PYTHONPATH_PREFIX${PYTHONPATH:+:$PYTHONPATH}"
fi

split_args=(
  python3 scripts/prepare_iroptset_splits.py
  --source "$DATA_SOURCE"
  --out-dir "$SPLIT_DIR"
  --train-size "$TRAIN_SIZE"
  --val-size "$VAL_SIZE"
  --test-size "$TEST_SIZE"
  --seed "$SEED"
  --ir-fields preprocessed_ir,o3_ir
)

if [[ "$DATA_SOURCE" == "huggingface" ]]; then
  split_args+=(--dataset "$DATASET")
  if [[ "$USE_HF_TOKEN" == "1" ]]; then
    split_args+=(--hf-token)
  fi
elif [[ "$DATA_SOURCE" == "parquet" ]]; then
  if [[ -z "$PARQUET_DIR" ]]; then
    echo "PARQUET_DIR is required when DATA_SOURCE=parquet" >&2
    exit 2
  fi
  split_args+=(--parquet-dir "$PARQUET_DIR")
else
  echo "DATA_SOURCE must be huggingface or parquet" >&2
  exit 2
fi

echo "==> Creating splits in ${SPLIT_DIR}"
"${split_args[@]}"

echo "==> Generating instruction labels in ${LABEL_DIR}"
label_args=(
  python3 scripts/label_iroptset_instructions.py
  --split-dir "$SPLIT_DIR"
  --out-dir "$LABEL_DIR"
  --ir-field "$LABEL_IR_FIELD"
)

if [[ -n "$INTRINSIC_INCLUDE_PREFIXES" ]]; then
  label_args+=(--include-prefix "$INTRINSIC_INCLUDE_PREFIXES")
fi
if [[ -n "$INTRINSIC_EXCLUDE_PREFIXES" ]]; then
  label_args+=(--exclude-prefix "$INTRINSIC_EXCLUDE_PREFIXES")
fi

"${label_args[@]}"

echo "==> Building graph dataset in ${GRAPH_DIR}"
python3 scripts/build_iroptset_graph_dataset.py \
  --split-dir "$SPLIT_DIR" \
  --label-dir "$LABEL_DIR" \
  --out-dir "$GRAPH_DIR" \
  --ir-field preprocessed_ir \
  --max-token-vocab "$MAX_TOKEN_VOCAB" \
  --max-instruction-vocab "$MAX_INSTRUCTION_VOCAB" \
  --max-nodes "$MAX_NODES" \
  --max-edges "$MAX_EDGES"

echo "==> Training GNN in ${MODEL_DIR}"
python3 scripts/train_instruction_gnn.py \
  --graph-dir "$GRAPH_DIR" \
  --out-dir "$MODEL_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --layers "$LAYERS" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --device "$DEVICE" \
  --eval-ks "$EVAL_KS"

echo "==> Done"
echo "Splits:  ${SPLIT_DIR}"
echo "Labels:  ${LABEL_DIR}"
echo "Graphs:  ${GRAPH_DIR}"
echo "Model:   ${MODEL_DIR}"
