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
INSTRUCTION_SPEC_JSONL="${INSTRUCTION_SPEC_JSONL:-}"

MAX_TOKEN_VOCAB="${MAX_TOKEN_VOCAB:-20000}"
MAX_INSTRUCTION_VOCAB="${MAX_INSTRUCTION_VOCAB:-512}"
MAX_NODES="${MAX_NODES:-512}"
MAX_EDGES="${MAX_EDGES:-4096}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INSTRUCTION_BATCH_SIZE="${INSTRUCTION_BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-3}"
DROPOUT="${DROPOUT:-0.15}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
EVAL_KS="${EVAL_KS:-10,20,50}"
DEVICE="${DEVICE:-auto}"
TRAINER="${TRAINER:-classifier}"                 # classifier or dual

SPLIT_DIR="runs/iroptset/splits/${OUT_TAG}"
LABEL_DIR="runs/iroptset/instruction_labels/${OUT_TAG}"
GRAPH_DIR="runs/iroptset/graphs/${OUT_TAG}"
INSTRUCTION_GRAPH_DIR="runs/iroptset/instruction_graphs/${OUT_TAG}"
if [[ "$TRAINER" == "dual" ]]; then
  MODEL_DIR="runs/iroptset/models/dual_gnn_${OUT_TAG}"
else
  MODEL_DIR="runs/iroptset/models/gnn_${OUT_TAG}"
fi

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

if [[ -n "$INSTRUCTION_SPEC_JSONL" ]]; then
  echo "==> Building instruction semantic graphs in ${INSTRUCTION_GRAPH_DIR}"
  python3 scripts/build_instruction_semantic_graphs.py \
    --spec-jsonl "$INSTRUCTION_SPEC_JSONL" \
    --out-dir "$INSTRUCTION_GRAPH_DIR"
fi

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
if [[ "$TRAINER" == "dual" ]]; then
  if [[ -z "$INSTRUCTION_SPEC_JSONL" ]]; then
    echo "INSTRUCTION_SPEC_JSONL is required when TRAINER=dual" >&2
    exit 2
  fi
  python3 scripts/train_instruction_dual_gnn.py \
    --program-graph-dir "$GRAPH_DIR" \
    --instruction-graph-dir "$INSTRUCTION_GRAPH_DIR" \
    --out-dir "$MODEL_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --instruction-batch-size "$INSTRUCTION_BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --device "$DEVICE" \
    --eval-ks "$EVAL_KS"
else
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
fi

echo "==> Done"
echo "Splits:  ${SPLIT_DIR}"
echo "Labels:  ${LABEL_DIR}"
echo "Graphs:  ${GRAPH_DIR}"
if [[ -n "$INSTRUCTION_SPEC_JSONL" ]]; then
  echo "Instr:   ${INSTRUCTION_GRAPH_DIR}"
fi
echo "Model:   ${MODEL_DIR}"
