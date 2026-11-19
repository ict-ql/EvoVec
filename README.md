# EvoVec

EvoVec is a retrieval-augmented LLM vectorization framework for evolving
instruction sets. A dual-encoder retriever matches scalar LLVM IR regions to
instruction semantic graphs. Selected intrinsic names and exact signatures guide
the LLM to generate optimized LLVM IR; Clang compiles and links the result.

```text
LLVM vectorization tests --> diff-supervised region graphs --> retriever training
ISL instruction descriptions --> instruction semantic graphs ----^

scalar C --> scalar LLVM IR --> region extraction --> instruction retrieval
         --> LLM IR optimization --> validation --> TSVC timing
                                    |
                                    +--> validator feedback --> IR repair
```

Instruction semantics are used by the retriever. The generation prompt supplies
selected intrinsic names and signatures.

## Requirements and setup

Experiments target Linux with Bash and Python 3.10 or newer. Install PyTorch
(with CUDA support for GPU training), the `openai` package for generation, and
`PyYAML` for YAML configuration:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch openai pyyaml

export LLVM_PROJECT_ROOT=/path/to/llvm-project
export LLVM_BUILD_DIR="$LLVM_PROJECT_ROOT/build"
export PATH="$LLVM_BUILD_DIR/bin:$PATH"
export ALIVE_TV=/path/to/alive2/build/alive-tv
export OPENAI_API_KEY='your-api-key'
cp config/llm.example.yaml config/llm.yaml
```

Use LLVM 19.1.0 tools (`clang`, `opt`, `llvm-link`, `llvm-config`), LLVM
sources and development libraries, a C++ compiler for declaration resolution,
and GCC for reference binaries. ICX is optional for compiler comparisons.
Formal validation needs a compatible Alive2 build; differential testing needs
Clang's libFuzzer and AddressSanitizer runtimes.

Install the PyTorch build appropriate for your GPU. Dependencies are not pinned
to an archived release environment. All commands below run from the repository
root.

The example configuration selects `gpt-5`. Blank credentials and base URL defer
to `OPENAI_API_KEY` and `OPENAI_BASE_URL`. Private `config/llm.yaml` is ignored
by Git. Generation commands incur API costs; training, retrieval, saved-candidate
validation, and timing do not call the LLM.

## Repository layout

| Path | Purpose |
| --- | --- |
| `scripts/build_llvm_vectorize_region_dataset.py` | Diff-supervised region dataset |
| `scripts/build_instruction_semantic_graphs.py` | Instruction semantic graphs |
| `scripts/train_region_intrinsic_model.py` | Region-change and instruction-matching training |
| `scripts/run_tsvc_region_inference.py` | TSVC scalar IR and recommendations |
| `scripts/resolve_llvm_intrinsic_declarations.py` | Exact intrinsic signatures |
| `scripts/run_tsvc_ir_optimization_experiment.py` | IR generation, validation, and repair |
| `scripts/benchmark_tsvc_correct_candidates.py` | TSVC timing of accepted candidates |
| `scripts/run_tsvc_compiler_baselines.py` | GCC, LLVM, and ICX baselines |
| `scripts/build_tsvc_text_retrieval_recommendations.py` | BM25, random, and full-list hints |
| `scripts/rebuild_tsvc_records.py` | Rebuild records from saved results |
| `prompts/` | English IR generation and repair templates |
| `src/` | Shared LLM, IR, target, and fuzzing utilities |
| `examples/` | LLVM IR, GEMM, and instruction-semantic examples |
| `third_party/TSVC_2/src/` | TSVC sources used to construct harnesses |
| `runs/` | Generated datasets, checkpoints, candidates, and logs; Git-ignored |

Shared utilities are stored directly in `src/`. Script entry points add this
directory to the Python import path automatically. Shared graph encoders and
intrinsic-label extraction utilities are in `scripts/graph_encoder.py` and
`scripts/intrinsic_labels.py`, respectively.

## 1. Build data and train the retriever

```bash
python3 scripts/build_llvm_vectorize_region_dataset.py \
  --llvm-project-root "$LLVM_PROJECT_ROOT" \
  --opt "$LLVM_BUILD_DIR/bin/opt" \
  --out-dir runs/llvm_vectorize_regions/x86_v1

python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl runs/llvm_vectorize_regions/x86_v1/vector_instruction_specs.jsonl \
  --out-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs

python3 scripts/train_region_intrinsic_model.py \
  --region-dir runs/llvm_vectorize_regions/x86_v1 \
  --instruction-graph-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs \
  --out-dir runs/llvm_vectorize_regions/models/x86_v1 \
  --epochs 100 --batch-size 32 --layers 3 --hidden-dim 128 \
  --dropout 0.15 --lr 0.001 --weight-decay 0.0001
```

The builder uses the LLVM LoopVectorize and SLPVectorizer X86 test directories
by default. Before/after IR differences provide region-change labels and target
instructions. Splitting uses source path and function name: all regions from a
function stay together. All positive regions and at most eight negative regions
per function are retained; this caps negatives without enforcing class balance.

Generated instruction specs cover observed vector operations and fallback
intrinsic labels. For supported x86 intrinsics, import external Intel/Vegen XML
using `scripts/import_intel_intrinsics_xml.py` and include the resulting specs
alongside the generated JSONL in `--spec-jsonl`. Consult its `--help` for options.
External XML data, generated datasets, and trained checkpoints are not bundled.

For the held-out AMX case study, build a separate inference candidate directory:

```bash
python3 scripts/build_instruction_semantic_graphs.py \
  --spec-jsonl \
    runs/llvm_vectorize_regions/x86_v1/vector_instruction_specs.jsonl \
    examples/instruction_semantics/amx_instruction_specs.jsonl \
  --out-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs_with_amx
```

Keep AMX out of the training graphs. Held out refers to retriever training, not
LLM pretraining. The paper used 313 candidate graphs (311 entries plus two AMX
instructions); rebuilding with different specs can change that count.

## 2. Prepare TSVC and retrieve hints

```bash
python3 scripts/prepare_tsvc_kernels.py \
  --src-dir third_party/TSVC_2/src \
  --out-dir runs/tsvc_kernels/v1 \
  --manifest runs/manifests/tsvc_kernels.jsonl \
  --kernels all --iterations 100000

python3 scripts/run_tsvc_region_inference.py \
  --manifest runs/manifests/tsvc_kernels.jsonl \
  --tsvc-root runs/tsvc_kernels/v1 \
  --checkpoint runs/llvm_vectorize_regions/models/x86_v1/last.pt \
  --instruction-graph-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs \
  --out-dir runs/tsvc_region_recommendations/top20 \
  --clang "$LLVM_BUILD_DIR/bin/clang" --top-k 20 --device cuda

python3 scripts/resolve_llvm_intrinsic_declarations.py \
  --llvm-project-root "$LLVM_PROJECT_ROOT" \
  --llvm-config "$LLVM_BUILD_DIR/bin/llvm-config" \
  --recommendations-jsonl runs/tsvc_region_recommendations/top20/recommendations.jsonl \
  --recommendation-top-k 20 \
  --out-json runs/llvm_intrinsics/tsvc_top20_declarations.json
```

Use the same prepared kernel tree for all comparisons. Preparation defaults to
four iterations for smoke checks; specify `--iterations 100000` for the paper's
setting. Each kernel retains its original loop formula based on this parameter.
Regenerate manifests on the execution machine or use `--tsvc-root` to relocate
kernel paths.

## 3. Generate, validate, and repair IR

This command compares unguided GPT-5 (`baseline`) with retrieval guidance
(`guided`) under Pass@1. Set `--limit 1` for a smoke run; `--limit 0` runs all
kernels.

```bash
python3 scripts/run_tsvc_ir_optimization_experiment.py \
  --manifest runs/manifests/tsvc_kernels.jsonl \
  --tsvc-root runs/tsvc_kernels/v1 \
  --recommendations-jsonl runs/tsvc_region_recommendations/top20/recommendations.jsonl \
  --intrinsic-declarations-json runs/llvm_intrinsics/tsvc_top20_declarations.json \
  --out-dir runs/tsvc_ir_opt/gpt5_top20_pass1 \
  --llm-config config/llm.yaml --model gpt-5 --max-tokens 16000 \
  --modes baseline,guided --limit 0 --max-iters 1 \
  --alive2 --alive-tv "$ALIVE_TV" \
  --differential-fuzz --llvm-link "$LLVM_BUILD_DIR/bin/llvm-link" \
  --clang "$LLVM_BUILD_DIR/bin/clang" --cc gcc \
  --fuzz-runs 10000 --fuzz-input-timeout 5 --fuzz-timeout 1800 \
  --run-timeout 300 --jobs 8 \
  --skip-scalar-ir-eval --skip-existing --resume-completed
```

Templates are `prompts/llvm_ir_vectorize_fast_v1.txt` and
`prompts/llvm_ir_self_correction_fast_v1.txt`. Temperature is omitted by default.
Each attempt saves prompts, candidates, postprocessing reports, and validator
logs. `records.jsonl` and `summary.json` collect results.

With both semantic validators enabled, acceptance requires compilation and the
TSVC check, plus a pass from Alive2 or differential fuzzing. A concrete
counterexample from either validator prevents acceptance. Unsupported features
and tool errors are inconclusive. Without semantic-validation flags, the runner
can accept TSVC-only results; enable both for the paper's correctness protocol.

Differential fuzzing links renamed reference and candidate implementations into
one LLVM module, initializes identical state, and compares observable outputs.
During fuzzing only, implementations stop after three `dummy()` calls. Matching
kernel-local `exit()` behavior is checked too. The timing path uses the full TSVC
iteration setting. Finite fuzzing does not prove equivalence for all inputs.

For self-correction, use the command above with `--max-iters 5` and
`--out-dir runs/tsvc_ir_opt/gpt5_top20_self_correct`. This allows one initial
generation and up to four repairs using original IR, the previous candidate, and
structured validation feedback. Accepted candidates stop repair. Inconclusive
validation alone does not trigger additional repair spending.

To reuse saved Pass@1 generations in that separate experiment, add:

```bash
--seed-results-dir runs/tsvc_ir_opt/gpt5_top20_pass1
```

Keep `--skip-existing --resume-completed` on resumptions. Completed successful
mode records are reused and successful saved LLM calls are not repeated.
Seeding a new output directory can rerun validation of the reused IR; subsequent
repair attempts can still make new API calls.

## 4. Measure performance

```bash
python3 scripts/benchmark_tsvc_correct_candidates.py \
  --experiment-dir runs/tsvc_ir_opt/gpt5_top20_self_correct \
  --tsvc-root runs/tsvc_kernels/v1 \
  --output-dir runs/tsvc_performance/gpt5_top20_self_correct \
  --gcc gcc --clang "$LLVM_BUILD_DIR/bin/clang" \
  --warmups 0 --repetitions 1 --run-timeout 300

python3 scripts/run_tsvc_compiler_baselines.py \
  --manifest runs/manifests/tsvc_kernels.jsonl \
  --tsvc-root runs/tsvc_kernels/v1 \
  --out-dir runs/tsvc_compiler_baselines/native_o3 \
  --compiler gcc=gcc --compiler llvm19="$LLVM_BUILD_DIR/bin/clang" \
  --compiler icx=icx --reference gcc --march native --opt-level 3 \
  --limit 151 --run-timeout 300 --jobs 1
```

Use the actual experiment directory, including the Pass@1 directory when timing
that condition. Omit ICX if unavailable. Baseline and guided candidates are
selected independently: the timing script requires TSVC and at least one
semantic validator to pass. `--repetitions` controls complete executable
invocations, not the internal TSVC iteration parameter. Avoid concurrent timing
workloads.

Use one consistent GCC reference for every per-kernel speedup:
`GCC -O3 time / candidate time`. The paper reports a geometric mean over all 151
kernels, assigning `0.1x` to unsuccessful LLM outputs. Accepted-only timing
averages are a different quantity. Counts at `>=1.0x`, `>=1.2x`, `>=1.5x`, and
`>=2.0x` are cumulative.

## 5. Retrieval ablations

Prepare an external Intel Intrinsics Guide XML, an Intel-to-LLVM alias map, and
per-kernel IR summaries. `scripts/build_x86_intrinsic_alias_map.py --help`
describes wrapper generation and alias extraction.

```bash
python3 scripts/build_tsvc_text_retrieval_recommendations.py \
  --xml /path/to/data-latest.xml \
  --alias-map runs/ablation/intel_to_llvm_aliases.json \
  --summaries runs/ablation/IR_summaries.zip \
  --method bm25 --top-k 20 \
  --out-jsonl runs/ablation/bm25/recommendations.jsonl
```

Use `--method random --seed 0` or `--method full` with separate output paths for
the other strategies. The builder makes no LLM calls. BM25 summaries must
already exist; random and full-list modes use that input to enumerate kernels.
Vendor descriptions support textual ranking; generation receives names and
signatures.

Evaluate each recommendation set through the IR runner with
`--modes guided --max-iters 1`. For the full list, set
`--recommendation-top-k` to the corpus size to avoid the default top-20 cap.
The wrapper `scripts/run_tsvc_retriever_ablation_pass1.sh` automates BM25, three
random seeds, and full-list generation. Set the environment variables at the top
of that script for your recommendation files, LLVM installation, Alive2, and
private LLM configuration. Full-list prompts can be substantially larger.

## Single-program example

```bash
python3 scripts/run_single_program_ir_optimization_pipeline.py \
  --program examples/gemm/gemm_scalar.c \
  --checkpoint runs/llvm_vectorize_regions/models/x86_v1/last.pt \
  --instruction-graph-dir runs/llvm_vectorize_regions/x86_v1/instruction_graphs_with_amx \
  --llvm-project-root "$LLVM_PROJECT_ROOT" \
  --llvm-config "$LLVM_BUILD_DIR/bin/llvm-config" \
  --clang "$LLVM_BUILD_DIR/bin/clang" \
  --out-dir runs/single_ir_opt/gemm --model gpt-5 --max-tokens 16000
```

This wrapper defaults to generation only. An arbitrary program needs its own
correctness and timing harness before the result can be considered an accepted
optimization. AMX execution requires supporting hardware and operating-system
support.
