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

## Instruction Semantic Graphs

The instruction side should not be represented as a bare intrinsic string. Each
instruction/intrinsic gets a small semantic graph built from:

- intrinsic or compiler builtin spelling
- assembly instruction mnemonic/form
- ISA architecture and extension
- CPUID or feature flags
- operand roles and types
- normalized Operation pseudocode as micro-operations

Example input record:

```json
{
  "label": "llvm.x86.sse2.padd.w",
  "architecture": "x86",
  "extension": "SSE2",
  "intrinsic": "_mm_add_epi16",
  "instruction": "paddw xmm, xmm",
  "cpuid": ["SSE2"],
  "operands": [
    {"name": "dst", "role": "dst", "type": "v8i16"},
    {"name": "a", "role": "src", "type": "v8i16"},
    {"name": "b", "role": "src", "type": "v8i16"}
  ],
  "operation": [
    "for lane in 0..7",
    "dst.i16[lane] = a.i16[lane] + b.i16[lane]"
  ],
  "split": "train"
}
```

Build semantic instruction graphs with:

```bash
python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl examples/instruction_semantics/instruction_specs.jsonl \
  --out-dir runs/iroptset/instruction_graphs/pilot_10k_v1
```

The graph builder creates instruction, intrinsic, assembly, ISA, CPUID, operand,
type, micro-op, operator, constant, and predicate nodes. The micro-op graph is
deliberately normalized: it should capture the operation's dataflow, lane
structure, masks, reductions, extension operations, and accumulation behavior
without depending on backend lowering.

For x86, the raw material should come from the Intel Intrinsics Guide or its
offline data export. For RVV, use RVV intrinsic/spec tables. Store only the
normalized JSONL records in this repo so the model sees a stable schema rather
than page-specific formatting.

## ISA Split Policy

The first experiment should be explicit about the instruction distribution:

- x86 training: SSE, SSE2, SSE3, SSSE3, SSE4.x, AVX, F16C, FMA, AVX2
- x86 validation: AVX-VNNI and AVX-VNNI-INT8, if enough records are available
- x86 held-out test: AVX-512 VNNI, BF16, and FP16-style instructions
- cross-ISA training: RVV records, kept with `architecture=riscv` and
  `extension=RVV`

The point of the AVX-512/VNNI holdout is to test whether the model can use the
micro-op semantics of newer dot-product and mixed-precision instructions rather
than only memorizing x86 intrinsic names. The point of the RVV records is to
teach vector lane, mask, and `vl` semantics in a cross-ISA way while keeping ISA
metadata explicit.

The machine-readable policy lives in:

```text
config/instruction_semantic_splits.json
```

## First Model Baseline

The semantic path trains a dual-graph contrastive retriever:

```text
program encoder(preprocessed_ir computation graph) -> embedding
instruction encoder(instruction micro-op semantic graph) -> embedding
score = dot(program_embedding, instruction_embedding)
```

```bash
python3 scripts/train_instruction_dual_gnn.py \
  --program-graph-dir runs/iroptset/graphs/pilot_10k_v1 \
  --instruction-graph-dir runs/iroptset/instruction_graphs/pilot_10k_v1 \
  --out-dir runs/iroptset/models/dual_gnn_pilot_10k_v1
```

The older classifier path is still available for ablations, but it should be
treated as a label-only baseline because it does not encode instruction
micro-op graphs.

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
