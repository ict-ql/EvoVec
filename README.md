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

The GNN input graph is built from LLVM IR, not from generated assembly. For a
small function:

```llvm
define i32 @scale_sum(i32 %a, i32 %b) {
entry:
  %x = add i32 %a, %b
  %y = mul i32 %x, 4
  ret i32 %y
}
```

`scripts/build_iroptset_graph_dataset.py` turns it into a symbolic graph. The
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

Readable sketch:

```text
nodes
  0  function     function
  1  argument     arg
  2  argument     arg
  3  block        block_entry
  4  instruction  add
  5  instruction  mul
  6  instruction  ret

edges
  function -> block_entry  contains
  block_entry -> add       contains
  block_entry -> mul       contains
  block_entry -> ret       contains
  arg -> add               def_use
  arg -> add               def_use
  add -> mul               next
  add -> mul               def_use
  mul -> ret               next
  mul -> ret               def_use
```

The encoded dataset stores token IDs, node type IDs, and typed directed edges.
This keeps the source-side graph small and compiler-middle-end oriented: blocks,
IR operations, SSA definitions, uses, operands, and branch structure.

The instruction side is a separate semantic graph. Each instruction or intrinsic
gets metadata plus normalized micro-operations from its Operation pseudocode:

```json
{
  "label": "llvm.x86.sse2.padd.w",
  "architecture": "x86",
  "extension": "SSE2",
  "intrinsic": "_mm_add_epi16",
  "instruction": "paddw xmm, xmm",
  "cpuid": ["SSE2"],
  "operation": [
    "for lane in 0..7",
    "dst.i16[lane] = a.i16[lane] + b.i16[lane]"
  ]
}
```

Build those instruction micro-op graphs with:

```bash
python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl examples/instruction_semantics/instruction_specs.jsonl \
  --split-policy config/instruction_semantic_splits.json \
  --out-dir runs/instruction_semantics/sample
```

To convert Intel/Vegen Intrinsics Guide XML into the spec JSONL format:

```bash
python3 scripts/import_intel_intrinsics_xml.py \
  --xml /path/to/data-latest.xml \
  --out runs/instruction_semantics/intel_intrinsics.jsonl
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
retriever.


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
