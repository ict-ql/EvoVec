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
instruction/intrinsic gets a `mini-programl` semantic graph built from:

- intrinsic or compiler builtin spelling
- assembly instruction mnemonic/form
- ISA architecture and extension
- CPUID or feature flags
- operand roles and types
- Vegen-style parsed Operation pseudocode as a variable-level data/control graph

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
    "FOR j := 0 to 7",
    "i := j*16",
    "dst[i+15:i] := a[i+15:i] + b[i+15:i]",
    "ENDFOR"
  ],
  "split": "train"
}
```

Build semantic instruction graphs with:

```bash
python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl examples/instruction_semantics/instruction_specs.jsonl \
  --split-policy config/instruction_semantic_splits.json \
  --out-dir runs/iroptset/instruction_graphs/pilot_10k_v1
```

The dataset builder is a wrapper around the same standalone parser used at
inference time:

```bash
python3 scripts/build_mini_programl_graph.py \
  --mode intrinsic \
  --spec-jsonl examples/instruction_semantics/instruction_specs.jsonl \
  --out runs/debug/intrinsic_graphs.jsonl
```

The graph builder creates ProGraML-style instruction, variable, constant, and
type nodes. ISA, CPUID, intrinsic spelling, assembly form, and other metadata
are graph or node features rather than extra fake nodes. Operation text is first
parsed into a Vegen-style AST (`For`, `Update`, `BitSlice`, `BinaryExpr`,
`Var`, `Number`, and friends), then lowered into mini-ProGraML. Operation
symbols such as `j`, `i`, `dst`, `a`, and `b` become variable nodes.
Arithmetic/effectful expressions and bit slices become instruction nodes. A
slice such as `a[i+15:i]` has explicit `base`, `hi`, and `lo` data edges, so it
is distinguishable from `a[i+32:i+15]`. A static loop such as
`FOR j := 0 to 7` gets a `control` self-loop on the loop node; its body has
`iteration_count=8`, while the self-loop/backedge has `execution_count=7`.

For x86, the raw material should come from the Intel Intrinsics Guide offline
XML data. The Vegen `data-latest.xml` file follows that shape and can be
converted directly:

```bash
python3 scripts/import_intel_intrinsics_xml.py \
  --xml /path/to/data-latest.xml \
  --llvm-alias-map runs/instruction_semantics/x86_llvm_aliases.json \
  --out runs/instruction_semantics/intel_intrinsics.jsonl
```

Build the alias map with Vegen-style C wrappers and clang-emitted LLVM IR:

```bash
python3 scripts/build_x86_intrinsic_alias_map.py \
  --xml /path/to/data-latest.xml \
  --emit-wrapper-c runs/instruction_semantics/x86_wrappers.c \
  --compile-ir runs/instruction_semantics/x86_wrappers.ll \
  --out runs/instruction_semantics/x86_llvm_aliases.json
```

For RVV, use RVV intrinsic/spec tables. Store only the normalized JSONL records
in this repo so the model sees a stable schema rather than page-specific
formatting.

## Mini-ProGraML Schema

Both program graphs and instruction semantic graphs use the same compact
ProGraML-inspired relation family:

- `control`: execution/order structure, including instruction-to-next-
  instruction, terminator-to-successor-entry edges, and intrinsic loop
  self-loops
- `data`: value use edges from SSA variables or instruction operands to
  consuming operations, plus operation-to-defined-value edges with `role=def`;
  intrinsic bit slices use `role=base`, `role=hi`, and `role=lo`
- `type`: interned type nodes to variable/constant/instruction nodes
- edge `position`: operand or successor order, following ProGraML's use of
  positional edges
- edge features: loop and operand-role metadata such as `backedge`,
  `loop_header`, `iteration_count`, `execution_count`, `role=constant`, or
  `role=def`
- compact node features retain `data_type` only on `type` nodes. Typed
  variables/constants/instructions must connect to the interned type node with
  `type` edges rather than duplicating the type as a feature on every node.
- SSA identifiers are debug-only. Graphs must canonicalize SSA variables to
  stable tokens such as `var`, so `%x + %y` and `%1 + %2` produce the same
  trainable graph when their structure, opcodes, constants, and types match.

Call edges are intentionally omitted for this experiment.

Both modes go through `scripts/mini_programl_graphs.py`:

```text
parse_mini_programl(mode="program", source=llvm_ir)
parse_mini_programl(mode="intrinsic", source=intrinsic_spec_record)
```

For a one-off program graph, use:

```bash
python3 scripts/build_mini_programl_graph.py \
  --mode program \
  --ir-file examples/program_graph/scale_sum.ll \
  --out runs/debug/scale_sum.program_graph.json
```

## ISA Split Policy

The first experiment should be explicit about the instruction distribution:

- x86 training: SSE, SSE2, SSE3, SSSE3, SSE4.x, AVX, F16C, FMA, AVX2
- x86 test: every x86 extension not listed above, including AVX-VNNI,
  AVX-512, AVX-512 VNNI, BF16, FP16, AMX, ADX, BMI, AES, SHA, and future
  extensions
- cross-ISA test: RVV records, kept with `architecture=riscv` and
  `extension=RVV`

AVX-512 VNNI is just one important test family, not the only x86 test target.
The broader holdout tests whether the model can use Operation-graph semantics for
newer dot-product, mixed-precision, matrix/tile, crypto, bit-manipulation, and
other non-training operations rather than memorizing SSE/AVX names. RVV is a
cross-ISA generalization test for vector lane, mask, and `vl` semantics.

The machine-readable policy lives in:

```text
config/instruction_semantic_splits.json
```

When `scripts/build_iroptset_graph_dataset.py` receives
`--instruction-graph-dir`, it filters the program train split to labels that
exist in `train.instruction_graphs.jsonl`. This guarantees that RVV and
non-SSE/AVX x86 extensions do not enter the training labels. Use
`--max-train-examples-per-instruction` to greedily cap examples per train label
so common instructions do not dominate the training split.

## Diff-Supervised Region Model

The paper path trains on changed scalar regions extracted from before/after IR
graph diffs, not on function-level bags. The model has a shared program-region
encoder and two heads:

```text
program encoder(changed scalar region) -> region embedding
changeability head(region embedding) -> should_change score
instruction encoder(instruction Operation semantic graph) -> instruction embedding
match score(region, instruction) = dot(region embedding, instruction embedding)
```

Training positives come from scalar regions that disappear or are rewritten in
the optimized/vectorized graph and are aligned to newly introduced intrinsic or
vector operations. Training negatives come from candidate regions that remain
unchanged under the same diff. At inference time, candidate regions are proposed
from store-rooted backward slices plus loop/backedge slices; the model first
scores whether a region should change, then ranks the instruction graphs for
that region. The older producer-consumer chain proposer is retained only for
debugging or ablation because it often cuts vectorizable computations too
finely. Multiple-instance learning is intentionally not part of the main
method.

Generic vector memory labels, such as `llvm.vector.load.*` and
`llvm.vector.store.*`, are graph-internal operations rather than useful
top-level recommendations. They are excluded from supervised target labels and
from inference rankings by default, while load/store nodes remain inside the
region and instruction graphs.

Use the LLVM vectorizer tests as the first clean training source:

```bash
python3 scripts/build_llvm_vectorize_region_dataset.py \
  --llvm-project-root /path/to/llvm-project \
  --opt /path/to/llvm-project/build/bin/opt \
  --out-dir runs/llvm_vectorize_regions/x86_v1
```

The collector reads `llvm/test/Transforms/LoopVectorize/X86` and
`llvm/test/Transforms/SLPVectorizer/X86`, runs the corresponding vectorizer pass,
and writes:

```text
runs/llvm_vectorize_regions/x86_v1/train.regions.jsonl
runs/llvm_vectorize_regions/x86_v1/val.regions.jsonl
runs/llvm_vectorize_regions/x86_v1/test.regions.jsonl
runs/llvm_vectorize_regions/x86_v1/vector_instruction_specs.jsonl
```

Build instruction graphs for the observed vector and LLVM intrinsic labels:

```bash
python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl runs/llvm_vectorize_regions/x86_v1/vector_instruction_specs.jsonl \
  --out-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs
```

Add unsupported or weakly supported target instructions as zero-shot candidates:

```bash
python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl runs/llvm_vectorize_regions/x86_v1/vector_instruction_specs.jsonl \
  --candidate-spec-jsonl runs/instruction_semantics/amx_instruction_specs.jsonl \
  --out-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs
```

The generated synthetic specs cover generic vector ops, reductions, vector libm
labels, and overflow/masked intrinsic labels. When the region labels include
real `llvm.x86.*` intrinsics, include the imported Intel/Vegen intrinsic spec
JSONL in the same `--spec-jsonl` invocation so those intrinsic labels can use
real target-instruction semantics during training.

For zero-shot instruction retrieval, including unsupported or weakly supported
target intrinsics such as AMX, instruction graph root tokens are generic by
default. The instruction name stays in metadata only; the encoder matches the
program region against the micro-operation graph. Records passed through
`--candidate-spec-jsonl` are forced to split `candidate`, so they contribute
vocabulary and inference candidates without being treated as negative labels
during supervised training.

Then train the region model:

```bash
python3 scripts/train_region_intrinsic_model.py \
  --region-dir runs/llvm_vectorize_regions/x86_v1 \
  --instruction-graph-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs \
  --out-dir runs/llvm_vectorize_regions/models/region_intrinsic_x86_v1
```

Ubuntu training servers can run the same commands from the repository root. The
pipeline assumes only Python, PyTorch, and an LLVM `opt` binary:

```bash
python3 --version
python3 -m pip install torch
/path/to/llvm-project/build/bin/opt --version
```

Use the `opt` built from the same LLVM checkout that provides the test
directories when possible. Collection failures are non-fatal per file and are
summarized in `runs/llvm_vectorize_regions/x86_v1/summary.json`.

Evaluate region detection and instruction matching with:

- change-region accuracy, precision, and recall
- `Recall@1`, `Recall@5`, and `Recall@10` over positive changed regions

Then inject the top-k retrieved intrinsics into the Codex assembly prompt as an
instruction palette and evaluate only on TSVC.

## Held-Out Rule

Do not train, tune prompts, choose hyperparameters, or select checkpoints using
TSVC results. TSVC should only be used after the retriever is frozen.
