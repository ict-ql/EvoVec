# IR-OptSet Instruction Retrieval Pilot

This plan uses IR-OptSet as the training source for an instruction-retrieval
module while keeping TSVC as the held-out evaluation set.

The goal is not to train the final GNN immediately. The first milestone is to
build compiler-derived instruction labels and measure whether a simple retriever
can predict the instructions a compiler would use for an unseen IR function.

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

After creating the split, compile each selected optimized IR function to
assembly and extract the emitted instruction mnemonics:

```bash
python3 scripts/label_iroptset_instructions.py \
  --split-dir runs/iroptset/splits/pilot_10k_v1 \
  --out-dir runs/iroptset/instruction_labels/pilot_10k_v1 \
  --compile-ir-field o3_ir \
  --llc llc \
  --target x86_64-pc-linux-gnu \
  --mcpu native \
  --opt-level=-O3 \
  --timeout 60
```

Preferred system tool:

```text
llc
```

Install LLVM tools or pass an explicit `--llc /path/to/llc`.
If system LLVM is unavailable, install `llvmlite` and use:

```bash
python3 scripts/label_iroptset_instructions.py \
  --split-dir runs/iroptset/splits/pilot_10k_v1 \
  --out-dir runs/iroptset/instruction_labels/pilot_10k_v1 \
  --compile-ir-field o3_ir \
  --backend llvmlite \
  --target x86_64-pc-linux-gnu \
  --mcpu generic \
  --timeout 60
```

The `llvmlite` backend uses LLVM's Python bindings to emit assembly and avoids
the need for a separate `llc` executable. It is useful for bootstrapping, but a
system `llc` run is still preferable for paper-grade reproducibility because
the compiler version and command-line flags are easier to record exactly.

The label records include:

- repository and function metadata
- target and CPU settings
- per-function instruction multiset
- unique instruction list
- compiler command or backend used to emit assembly
- failure information for functions that do not compile

Use `--keep-artifacts` to preserve generated assembly files under the label
output directory.

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
- mean reciprocal rank for compiler-used instructions

Then inject the top-k retrieved instructions into the Codex assembly prompt as
an instruction palette and evaluate only on TSVC.

## Held-Out Rule

Do not train, tune prompts, choose hyperparameters, or select checkpoints using
TSVC results. TSVC should only be used after the retriever is frozen.
