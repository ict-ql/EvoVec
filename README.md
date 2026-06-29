# LLM Compiler Replacement Pipeline

This project scaffolds an experiment where an LLM or agent acts as the compiler
under test:

```text
source code or LLVM IR -> LLM compiler agent -> target assembly -> assembler/linker -> tests
```

Conventional compilers are used only for baselines, reference behavior, optional
LLVM IR normalization, and assembly/link/test tooling. They are not the code
generator in the experimental path.

## Research Modes

1. **Full compiler replacement**
   - Input: C or Fortran source.
   - Output: target assembly.
   - Claim: the LLM replaces frontend analysis, optimization, and backend code
     generation for the selected benchmark scope.

2. **Backend replacement**
   - Input: LLVM IR.
   - Output: target assembly.
   - Claim: the LLM replaces instruction selection, register allocation, and
     target lowering. This is the more controlled first milestone.

## Initial Benchmarks

- **NPB**: NAS Parallel Benchmarks for Fortran/HPC kernels and whole programs.
- **TSVC**: vectorization kernels for loop optimization and SIMD-sensitive code.

Use local copies of the suites so licensing and versioning stay explicit.

```bash
python3 scripts/prepare_benchmarks.py \
  --suite tsvc \
  --root /path/to/TSVC \
  --out runs/manifests/tsvc.jsonl

python3 scripts/prepare_benchmarks.py \
  --suite npb \
  --root /path/to/NPB3.x \
  --out runs/manifests/npb.jsonl
```

## Source Computation Graph Example

The GNN input graph is built from LLVM IR, not from generated assembly. The
default representation is `mini-programl`: a small ProGraML-style directed
multigraph with `INSTRUCTION`, `VARIABLE`, `CONSTANT`, and `TYPE` nodes.
Edges have ProGraML-like `control`, `data`, or `type` flow, plus numeric
positions and lightweight features such as `backedge`. We omit call edges
because this experiment is function/intrinsic retrieval oriented.

For a small function:

```llvm
define i32 @scale_sum(i32 %a, i32 %b) {
entry:
  %x = add i32 %a, %b
  %y = mul i32 %x, 4
  ret i32 %y
}
```

`scripts/build_mini_programl_graph.py` turns it into a symbolic graph. The
checked-in example is:

```text
examples/program_graph/scale_sum.graph.json
```

Regenerate it with:

```bash
python3 scripts/inspect_program_graph.py \
  --ir-file examples/program_graph/scale_sum.ll \
  --out examples/program_graph/scale_sum.graph.json
```

The same standalone parser is the inference-time entry point:

```bash
python3 scripts/build_mini_programl_graph.py \
  --mode program \
  --ir-file examples/program_graph/scale_sum.ll \
  --vocab runs/iroptset/graphs/pilot_10k_v1/vocab.json \
  --out runs/inference/scale_sum.program_graph.json
```

Compact symbolic output interns each normalized type as one `type` node. Only
that node keeps `data_type`; typed variables, constants, and instructions point
to it with `type` edges. SSA names such as `%x`, `%y`, `%1`, and `%2` are
debug-only text, while trainable variable tokens stay canonical as `var`.

Readable sketch:

```text
nodes
  0  variable     %a
  1  variable     %b
  2  type         i32
  3  instruction  add
  4  variable     %x
  5  instruction  mul
  6  variable     %y
  7  constant     4
  8  instruction  ret

edges
  i32 -> %a                type
  i32 -> %b                type
  i32 -> add               type
  add -> %x                data(position=0, role=def)
  i32 -> %x                type
  %a -> add                data(position=0, role=use)
  %b -> add                data(position=1, role=use)
  add -> mul               control
  i32 -> mul               type
  mul -> %y                data(position=0, role=def)
  i32 -> %y                type
  %x -> mul                data(position=0, role=use)
  i32 -> 4                 type
  4 -> mul                 data(role=constant)
  mul -> ret               control
  i32 -> ret               type
  %y -> ret                data(position=0, role=use)
```

The encoded dataset stores token IDs, node type IDs, typed directed edges, and
edge positions. Loops are visible as control-flow cycles; backedges are marked
in edge features when a terminator branches to an earlier block.

Training currently tensorizes only these graph fields into the GNN encoder:

```text
node_token_ids      -> node_tokens
node_type_ids       -> node_types
edges[:, 0]         -> edge_src
edges[:, 1]         -> edge_dst
edges[:, 2]         -> edge_types
edges[:, 3]         -> edge_positions
edge_feature_ids    -> edge_features
edge_counts         -> edge_counts
```

`graph_ids` is created during batching for graph-level pooling, and `label_ids`
is used as retrieval supervision. Symbolic inspection fields such as
`node_features`, `graph_features`, `node_tokens`, `node_types`, and
`edge_features` are not consumed by the trainer unless they are first encoded
into the tensor fields above.

For the paper path, do not train instruction recommendation from function-level
bags. Use before/after graph diffs to extract changed scalar regions, then train
a changed-region detector and an intrinsic matcher on those regions. The
function-level dual-graph trainer is kept only as a pilot baseline until the
diff-supervised region dataset is built.

Start with LLVM's focused X86 vectorizer tests rather than IR-OptSet:

```bash
python3 scripts/build_llvm_vectorize_region_dataset.py \
  --llvm-project-root /path/to/llvm-project \
  --opt /path/to/llvm-project/build/bin/opt \
  --out-dir runs/llvm_vectorize_regions/x86_v1
```

The script reads:

```text
llvm/test/Transforms/LoopVectorize/X86
llvm/test/Transforms/SLPVectorizer/X86
```

It runs `opt`, extracts changed scalar regions from before/after IR graph
diffs, emits train/val/test `.regions.jsonl`, and writes synthetic instruction
specs for observed vector IR operations and LLVM intrinsic labels:

```text
runs/llvm_vectorize_regions/x86_v1/vector_instruction_specs.jsonl
```

Region proposals are store-rooted by default: each store instruction seeds a
backward data slice that pulls in the address/value computation feeding the
write. When the slice reaches a `phi`, the graph includes that `phi` node's
control-flow predecessor edges, so loop backedges can appear inside the
store-rooted region without creating a separate loop-level candidate. The older
producer-consumer and whole loop/backedge proposers are still available for
debugging or ablation, but they are not the default because they either cut
computations too finely or make regions too coarse.

Generic `llvm.vector.load.*` and `llvm.vector.store.*` labels are excluded from
supervised targets by default. They remain as graph nodes inside regions and
instruction micro-operation graphs, but they are not useful top-level
recommendations. Pass `--keep-memory-target-labels` only for debugging the
closed-set vectorizer behavior.

Build instruction graphs for those observed labels:

```bash
python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl runs/llvm_vectorize_regions/x86_v1/vector_instruction_specs.jsonl \
  --out-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs
```

Do not include zero-shot target instructions, such as AMX, in the training
instruction graph directory. Keep training on observed vectorizer labels only.
For inference, build a separate instruction graph directory that appends AMX to
the candidate set:

```bash
python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl \
    runs/llvm_vectorize_regions/x86_v1/vector_instruction_specs.jsonl \
    examples/instruction_semantics/amx_instruction_specs.jsonl \
  --out-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs_with_amx
```

The synthetic specs cover generic LLVM vector ops, reductions, and common
fallback intrinsic labels such as vector libm and overflow intrinsics. If the
collected labels include real `llvm.x86.*` intrinsics, pass the imported
Intel/Vegen intrinsic spec JSONL in the same `--spec-jsonl` list so those labels
use real target-instruction semantics instead of a generic fallback.

The semantic retrieval path uses generic instruction root tokens by default. It
prevents the instruction label itself, such as `llvm.vector.add.v4xi32` or an
AMX intrinsic name, from becoming a trainable node token. The label remains in
metadata, while the GNN sees the operation graph. This is important for
zero-shot candidate instructions whose labels never appear as training targets.
Use `--root-token-mode label` only for closed-set debugging.

Then train the changed-region detector plus intrinsic/vector-op matcher:

```bash
python3 scripts/train_region_intrinsic_model.py \
  --region-dir runs/llvm_vectorize_regions/x86_v1 \
  --instruction-graph-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs \
  --out-dir runs/llvm_vectorize_regions/models/region_intrinsic_x86_v1
```

To inspect recommendations on pre-vectorization TSVC IR, first emit scalar IR
with LLVM vectorizers disabled, then run the retriever:

```bash
clang -S -emit-llvm -O3 -fno-vectorize -fno-slp-vectorize \
  -I runs/tsvc_kernels/v1/support \
  runs/tsvc_kernels/v1/kernels/s000/reference_kernel.c \
  -o runs/tsvc_prevec_ir/s000.ll

python3 scripts/recommend_region_intrinsics.py \
  --ir-file runs/tsvc_prevec_ir/s000.ll \
  --checkpoint runs/llvm_vectorize_regions/models/region_intrinsic_x86_v1/best.pt \
  --instruction-graph-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs \
  --out-jsonl runs/llvm_vectorize_regions/recommendations/s000.jsonl \
  --out-md runs/llvm_vectorize_regions/recommendations/s000.md \
  --top-k 10 \
  --device cuda
```

For all split TSVC kernels, use the batch wrapper. It emits scalar IR for each
kernel in `runs/manifests/tsvc_kernels.jsonl`, then keeps the top 20
recommendations per candidate region. On a remote Ubuntu server, regenerate the
manifest on that server so the absolute paths point to the remote checkout; if
you copied a manifest from another machine, pass `--tsvc-root` with the remote
`runs/tsvc_kernels/v1` path.

```bash
python3 scripts/prepare_tsvc_kernels.py \
  --src-dir third_party/TSVC_2/src \
  --out-dir runs/tsvc_kernels/v1 \
  --manifest runs/manifests/tsvc_kernels.jsonl \
  --kernels all

python3 scripts/run_tsvc_region_inference.py \
  --manifest runs/manifests/tsvc_kernels.jsonl \
  --tsvc-root runs/tsvc_kernels/v1 \
  --checkpoint runs/llvm_vectorize_regions/models/region_intrinsic_x86_v1/best.pt \
  --instruction-graph-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs \
  --out-dir runs/tsvc_region_recommendations/top20 \
  --top-k 20 \
  --device cuda
```

To test whether recommendation-guided IR optimization improves TSVC runtime,
run the baseline and guided LLM paths side by side. The baseline prompt receives
only pre-vectorization LLVM IR; the guided prompt receives the same IR plus the
top recommendation rows from the retriever. Both paths must return complete
LLVM IR, which is linked into the existing TSVC harness and checked by checksum:

```bash
python3 scripts/run_tsvc_ir_optimization_experiment.py \
  --manifest runs/manifests/tsvc_kernels.jsonl \
  --tsvc-root runs/tsvc_kernels/v1 \
  --recommendations-jsonl runs/tsvc_region_recommendations/top20/recommendations.jsonl \
  --out-dir runs/tsvc_ir_opt/gpt5_top20 \
  --llm-config config/llm.example.yaml \
  --model gpt-5 \
  --max-tokens 16000 \
  --temperature 0.2 \
  --modes baseline,guided \
  --limit 10 \
  --target-cpu auto \
  --clang clang \
  --cc cc
```

The exact prompts, model outputs, compile/run logs, and per-kernel records are
written under the output directory. `summary.json` reports success counts, mean
times, and paired guided-vs-baseline speedup when both modes pass correctness.
The runner records target details in `target.json`. It first asks
`clang -march=native -###` for the LLVM target CPU, then falls back to
`gcc -Q --help=target -march=native` and CPU-model heuristics when clang reports
only `native`; pass `--target-cpu znver4` or a similar LLVM CPU name to override
auto-detection.
The same LLM client is available directly as `python3 scripts/ask_llm.py`, which
reads from stdin, `-p`, or `--file` and uses the same config format. Keep API
keys out of git; set `OPENAI_API_KEY` or pass a private config file.

For a quick AMX smoke test, emit scalar GEMM IR and run the no-AMX checkpoint
against the inference-time instruction graph directory that contains AMX:

```bash
mkdir -p runs/gemm_prevec_ir runs/gemm_recommendations

clang -S -emit-llvm -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops \
  examples/gemm/gemm_scalar.c \
  -o runs/gemm_prevec_ir/gemm_scalar.ll

python3 scripts/recommend_region_intrinsics.py \
  --ir-file runs/gemm_prevec_ir/gemm_scalar.ll \
  --checkpoint runs/llvm_vectorize_regions/models/region_intrinsic_x86_v1/best.pt \
  --instruction-graph-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs_with_amx \
  --out-jsonl runs/gemm_recommendations/gemm_amx.jsonl \
  --out-md runs/gemm_recommendations/gemm_amx.md \
  --top-k 20 \
  --store-slice-depth 8 \
  --store-slice-max-nodes 256 \
  --device cuda

grep -Ein "amx|tdpb|tile" runs/gemm_recommendations/gemm_amx.md
```

The recommender re-encodes instruction graph rows with the checkpoint's existing
instruction vocabulary. That lets the inference directory contain AMX even
though the checkpoint was trained without AMX; tokens never seen during training
fall back to `<unk>`.

To train continuously while checking the GEMM AMX zero-shot probe every few
epochs, use the training graph directory without AMX and the probe graph
directory with AMX:

```bash
REGION_DIR=runs/llvm_vectorize_regions/x86_store_phi_nomem_v3
MODEL_DIR=runs/llvm_vectorize_regions/models/region_intrinsic_x86_store_phi_nomem_v3_amx_probe

mkdir -p runs/gemm_prevec_ir
clang -S -emit-llvm -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops \
  examples/gemm/gemm_scalar.c \
  -o runs/gemm_prevec_ir/gemm_scalar.ll

python3 scripts/train_region_intrinsic_model.py \
  --region-dir "$REGION_DIR" \
  --instruction-graph-dir "$REGION_DIR/instruction_graphs" \
  --out-dir "$MODEL_DIR" \
  --epochs 60 \
  --batch-size 32 \
  --instruction-batch-size 128 \
  --hidden-dim 128 \
  --layers 3 \
  --dropout 0.15 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --temperature 0.07 \
  --match-loss-weight 1.0 \
  --max-pos-weight 20.0 \
  --device cuda \
  --probe-every-epochs 5 \
  --probe-ir-file runs/gemm_prevec_ir/gemm_scalar.ll \
  --probe-instruction-graph-dir "$REGION_DIR/instruction_graphs_with_amx" \
  --probe-out-dir "$MODEL_DIR/amx_probe" \
  --probe-top-k 20 \
  --probe-store-slice-depth 8 \
  --probe-store-slice-max-nodes 256
```

The trainer writes normal training checkpoints, a probe report for every probe
epoch, and one candidate checkpoint for every probe epoch whose top-k
recommendations include an AMX/tdpb/tile target:

```text
$MODEL_DIR/best.pt
$MODEL_DIR/last.pt
$MODEL_DIR/amx_probe_candidates/epoch_*.pt
$MODEL_DIR/amx_probe_candidates/epoch_*.json
$MODEL_DIR/amx_probe/probe_history.jsonl
```

The output is inference-only: each candidate region gets a change probability
and a ranked list of intrinsic/vector labels. It does not require after-IR or
diff labels. Generic vector load/store labels are hidden from the ranking by
default; pass `--include-memory-labels` only when inspecting raw closed-set
behavior.

On Ubuntu, run these commands from the repository root. The scripts are portable
Python and do not require Windows paths or PowerShell, but the server must have:

```bash
python3 --version
python3 -m pip install torch
/path/to/llvm-project/build/bin/opt --version
```

The LLVM `opt` binary should come from the same LLVM checkout as the test
directories when possible. If the first collection run fails, inspect
`runs/llvm_vectorize_regions/x86_v1/summary.json`; failed tests are recorded
there with the `opt` stderr.

The instruction side uses the same `mini-programl` idea. Each instruction or
intrinsic gets metadata plus a Vegen-style Operation graph. The Operation text
is parsed into statements and expressions such as `For`, `Update`,
`BitSlice`, `BinaryExpr`, `Var`, and `Number`; those AST nodes are then lowered
to mini-ProGraML. Symbols in the pseudocode, including temporaries and loop
indices, become variable nodes; arithmetic/effectful expressions and bit slices
become instruction nodes; static loops get a control self-loop with
execution-count features.

```json
{
  "label": "llvm.x86.sse2.padd.w",
  "architecture": "x86",
  "extension": "SSE2",
  "intrinsic": "_mm_add_epi16",
  "instruction": "paddw xmm, xmm",
  "cpuid": ["SSE2"],
  "operation": [
    "FOR j := 0 to 7",
    "i := j*16",
    "dst[i+15:i] := a[i+15:i] + b[i+15:i]",
    "ENDFOR"
  ]
}
```

This produces variable nodes such as `j`, `i`, `dst`, `a`, and `b`. The loop
node has a `control` self-loop; for `0 to 7`, the loop body has
`iteration_count=8` and the self-loop/backedge has `execution_count=7`.

Readable intrinsic sketch:

```text
nodes
  instruction  for
  variable     j
  instruction  mul
  variable     i
  instruction  add
  instruction  slice       a[i+15:i]
  instruction  slice       b[i+15:i]
  instruction  slice       dst[i+15:i]
  variable     dst
  variable     a
  variable     b

edges
  for -> for        control(backedge=true, iteration_count=8, execution_count=7)
  for -> j          data(role=induction_def)
  j -> mul          data(role=use, execution_count=8)
  16 -> mul         data(role=constant, execution_count=8)
  mul -> i          data(role=def, execution_count=8)
  a -> slice        data(role=base, execution_count=8)
  i + 15 -> slice   data(role=hi, execution_count=8)
  i -> slice        data(role=lo, execution_count=8)
  slice(a) -> add   data(role=lhs, execution_count=8)
  slice(b) -> add   data(role=rhs, execution_count=8)
  dst -> slice      data(role=base, execution_count=8)
  add -> slice(dst) data(role=def, execution_count=8)
```

Build those instruction Operation graphs with:

```bash
python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl examples/instruction_semantics/instruction_specs.jsonl \
  --split-policy config/instruction_semantic_splits.json \
  --out-dir runs/instruction_semantics/sample
```

For one-off inspection or inference-time construction of an intrinsic graph,
use the same standalone parser with intrinsic mode:

```bash
python3 scripts/build_mini_programl_graph.py \
  --mode intrinsic \
  --spec-jsonl examples/instruction_semantics/instruction_specs.jsonl \
  --out runs/inference/intrinsic_graphs.jsonl
```

To convert Intel/Vegen Intrinsics Guide XML into the spec JSONL format:

```bash
python3 scripts/import_intel_intrinsics_xml.py \
  --xml /path/to/data-latest.xml \
  --llvm-alias-map runs/instruction_semantics/x86_llvm_aliases.json \
  --out runs/instruction_semantics/intel_intrinsics.jsonl
```

Build that alias map with Vegen-style wrappers:

```bash
python3 scripts/build_x86_intrinsic_alias_map.py \
  --xml /path/to/data-latest.xml \
  --emit-wrapper-c runs/instruction_semantics/x86_wrappers.c \
  --compile-ir runs/instruction_semantics/x86_wrappers.ll \
  --out runs/instruction_semantics/x86_llvm_aliases.json
```

For the dual-graph retriever, train with a program graph directory and an
instruction graph directory:

```bash
python3 scripts/train_instruction_dual_gnn.py \
  --program-graph-dir runs/iroptset/graphs/pilot_10k_v1 \
  --instruction-graph-dir runs/iroptset/instruction_graphs/pilot_10k_v1 \
  --out-dir runs/iroptset/models/dual_gnn_pilot_10k_v1
```

When using `scripts/run_iroptset_gnn_pipeline.sh`, set
`INSTRUCTION_SPEC_JSONL=...`. The pipeline always trains the dual-graph
retriever. It filters train labels through `train.instruction_graphs.jsonl` and
uses `MAX_TRAIN_EXAMPLES_PER_INSTRUCTION` to cap very common labels.


## GPT-5.5 Pass@1

The fixed prompt template for fast assembly generation is:

```text
prompts/source_to_asm_fast_v1.txt
```

Run one GPT-5.5 candidate per task with:

```bash
OPENAI_API_KEY=... python3 scripts/run_pass_at_1.py \
  --manifest runs/manifests/tsvc.jsonl \
  --out-dir runs/pass_at_1/gpt-5.5/tsvc_v1 \
  --model gpt-5.5 \
  --target x86_64-pc-linux-gnu \
  --mode source-to-asm \
  --limit 10
```

The first automatic score is `pass_at_1_assemble`. Link-and-run correctness
should be added per TSVC/NPB harness before claiming semantic correctness.

## Agentic Compiler Condition

For the paper claim "an agent replaces the compiler", use a fixed generation
template, fixed verifier, fixed repair template, and fixed iteration budget:

```bash
python3 scripts/run_agentic_compile.py \
  --manifest runs/manifests/tsvc.jsonl \
  --out-dir runs/agentic/gpt-5.5/tsvc_v1 \
  --agent-cmd "your-agent-command" \
  --target x86_64-pc-linux-gnu \
  --mode source-to-asm \
  --limit 10 \
  --max-iters 5
```

This reports within-budget agentic success. Keep it separate from pass@1.

## Minimal Agent Loop

Create a prompt for one benchmark task and ask an external agent command to emit
assembly:

```bash
python3 scripts/llm_compile.py \
  --manifest runs/manifests/tsvc.jsonl \
  --index 0 \
  --target x86_64-pc-linux-gnu \
  --mode source-to-asm \
  --agent-cmd "python3 examples/mock_agent.py" \
  --out runs/candidates/task0.s
```

The mock agent intentionally returns invalid placeholder assembly. Replace
`--agent-cmd` with the real LLM compiler agent. The command receives the prompt
on stdin and must write assembly to stdout.

Then check whether the generated assembly can at least assemble:

```bash
python3 scripts/evaluate_asm.py \
  --asm runs/candidates/task0.s \
  --target x86_64-pc-linux-gnu \
  --workdir runs/eval/task0
```

For paper-grade evaluation, add a harness and run command so the object is linked
and executed against benchmark-specific correctness checks.

## Evaluation Metrics

- assembly success rate
- link success rate
- benchmark verifier pass rate
- differential correctness against compiler-built reference binaries
- fuzz/property-test pass rate for isolated kernels
- runtime versus `clang/gcc/gfortran -O0/-O2/-O3`
- code size versus compiler baseline
- pass@k after iterative feedback
- agent cost: tokens, retries, wall time

See [docs/research_plan.md](docs/research_plan.md) for the proposed paper path
and [docs/benchmark_protocol.md](docs/benchmark_protocol.md) for the benchmark
rules behind the compiler-replacement claim. The iterative-agent protocol is in
[docs/agentic_compiler_protocol.md](docs/agentic_compiler_protocol.md).
