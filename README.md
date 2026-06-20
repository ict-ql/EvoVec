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
