# Research Plan

## Thesis

An LLM can be evaluated as a compiler replacement when its generated assembly is
assembled, linked, executed, and scored with the same discipline used for normal
compiler research.

The strongest first claim is not that the LLM beats mature compilers. It is that
an agentic LLM compiler can close the loop from source or LLVM IR to executable
assembly with measurable correctness and performance on established scientific
and vectorization benchmarks.

## Scope

### Phase 1: Kernel-Level Backend Replacement

- Input: LLVM IR extracted from TSVC and small NPB kernels.
- Output: assembly for one target ISA, initially x86_64.
- Baselines: `clang -O0`, `clang -O2`, `clang -O3`.
- Correctness: assemble, link into a small harness, compare outputs.

This avoids asking the model to solve parsing, interprocedural source semantics,
Fortran runtime behavior, and whole-program linking all at once.

### Phase 2: Source-to-Assembly Compiler Replacement

- Input: C and Fortran source.
- Output: assembly.
- Correctness: benchmark-native verification plus differential execution.
- Baselines: `gcc`, `clang`, `gfortran`, `flang` where available.

This is the stronger "replace the compiler" claim.

### Phase 3: Whole-Benchmark Replacement

- Input: complete NPB or TSVC benchmark translation units.
- Output: all assembly objects needed for the executable.
- Correctness: official benchmark verifier.
- Performance: wall-clock runtime and code size against compiler baselines.

## Benchmarks

### TSVC

TSVC is the best early target because it contains focused loop kernels with clear
optimization patterns. It lets us study whether the LLM can produce correct and
fast scalar or vector assembly without the full complexity of large applications.

Recommended use:

- isolate one kernel at a time
- generate assembly for the kernel body/function
- link with a fixed C harness
- compare exact or tolerance-based outputs
- measure scalar/SIMD speedups

### NPB

NPB is the best scientific/HPC benchmark target. It stresses Fortran, arrays,
loop nests, memory behavior, and benchmark-native verification.

Recommended use:

- start with serial NPB kernels and small classes
- use benchmark verifier output as the primary correctness signal
- evaluate Fortran source-to-assembly separately from LLVM-IR-to-assembly
- report performance by benchmark class and compiler baseline

## Experimental Conditions

Use separate rows for:

- one-shot LLM generation
- LLM plus assembler feedback
- LLM plus linker feedback
- LLM plus runtime verifier feedback
- LLM plus performance feedback
- LLM plus decomposition into smaller functions or loop kernels

The paper should show where feedback helps and where it fails.

## Key Paper Claim

A good title-level claim could be:

> Towards LLM Compiler Replacement: Verified Source/IR-to-Assembly Generation on
> Scientific and Vectorization Benchmarks

The important contribution is the verified pipeline and benchmark analysis, not
only the model prompt.
