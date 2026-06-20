# IR-OptSet Instruction Retrieval Pilot

This plan uses IR-OptSet as the training source for an instruction-retrieval
module while keeping TSVC as the held-out evaluation set.

The goal is not to train the final GNN immediately. The first milestone is to
build IR-derived intrinsic instruction labels and measure whether a simple
retriever can predict the instruction-like intrinsics used by an unseen IR
function.

## Data Split

Use a small repo-disjoint pilot split first:

- train: 10,000 functions
- validation: 2,000 functions
- internal test: 2,000 functions
- final held-out benchmark: TSVC only, never used for training or tuning

The split script assigns whole repositories to only one split and filters rows
whose repository, file path, or function name looks TSVC-related.

```bash
python3 scripts/prepare_iroptset_splits.py \
  --source huggingface \
  --dataset YangziResearch/IR-OptSet \
  --out-dir runs/iroptset/splits/pilot_10k_v1 \
  --train-size 10000 \
  --val-size 2000 \
  --test-size 2000 \
  --seed 17 \
  --ir-fields preprocessed_ir,o3_ir
```

For a local parquet mirror of the Hugging Face dataset, use:

```bash
python3 scripts/prepare_iroptset_splits.py \
  --source parquet \
  --parquet-dir /path/to/IR-OptSet/data \
  --out-dir runs/iroptset/splits/pilot_10k_v1 \
  --train-size 10000 \
  --val-size 2000 \
  --test-size 2000 \
  --seed 17 \
  --ir-fields preprocessed_ir,o3_ir
```

Required Python packages for split preparation:

```bash
python3 -m pip install datasets pyarrow
```

`datasets` is needed for direct Hugging Face streaming. `pyarrow` is needed for
local parquet input.

## Instruction Labels

After creating the split, scan each selected optimized IR function and extract
LLVM intrinsic calls. These labels are taken from the middle-end IR directly:
we do not need to lower with `llc` just to discover instruction-like operations
that already appear as intrinsics.

```bash
python3 scripts/label_iroptset_instructions.py \
  --split-dir runs/iroptset/splits/pilot_10k_v1 \
  --out-dir runs/iroptset/instruction_labels/pilot_10k_v1 \
  --ir-field o3_ir
```

To restrict labels to a target intrinsic family, add an include prefix:

```bash
python3 scripts/label_iroptset_instructions.py \
  --split-dir runs/iroptset/splits/pilot_10k_v1 \
  --out-dir runs/iroptset/instruction_labels/pilot_10k_v1 \
  --ir-field o3_ir \
  --include-prefix llvm.x86.
```

The default extractor keeps all `llvm.*` call/invoke/callbr intrinsics except
debug, lifetime, assumption, and other metadata/control-only intrinsics. Use
`--include-prefix` and `--exclude-prefix` when a run should focus on a narrower
instruction family.

The label records include:

- repository and function metadata
- IR field used for extraction
- label kind (`llvm_intrinsic`)
- per-function intrinsic multiset, also stored in the existing
  `instruction_counts` field for downstream compatibility
- unique intrinsic list, also stored in the existing `instructions` field
- failure information for rows with missing IR

## First Model Baseline

Before building a full GNN, train a contrastive retriever:

```text
program encoder(preprocessed_ir) -> embedding
instruction encoder(instruction metadata / semantic graph) -> embedding
score = dot(program_embedding, instruction_embedding)
```

Evaluate on the IR-OptSet validation/internal-test splits with:

- `Recall@10`
- `Recall@20`
- `Recall@50`
- mean reciprocal rank for IR-used intrinsics

Then inject the top-k retrieved intrinsics into the Codex assembly prompt as an
instruction palette and evaluate only on TSVC.

## Held-Out Rule

Do not train, tune prompts, choose hyperparameters, or select checkpoints using
TSVC results. TSVC should only be used after the retriever is frozen.
