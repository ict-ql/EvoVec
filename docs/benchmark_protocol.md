# Benchmark Protocol

## Principle

The LLM path must not call a compiler to generate code. A compiler may be used
only outside the experimental path:

- to build reference binaries
- to provide baseline runtime/code-size numbers
- to assemble LLM-emitted assembly
- to link objects with a fixed harness
- to generate LLVM IR only for the backend-replacement condition

Report these roles explicitly in the paper.

## TSVC First

Use TSVC as the first correctness and performance target.

Recommended unit:

```text
one TSVC kernel function -> LLM assembly -> fixed harness -> differential test
```

Why:

- kernels are small enough for current LLM context windows
- expected behavior is easier to isolate
- vectorization and loop transformations are central compiler tasks
- scalar correctness and SIMD performance can be separated

Minimum table:

| condition | assemble | link | correct | runtime | code size |
| --- | --- | --- | --- | --- | --- |
| LLM one-shot |  |  |  |  |  |
| LLM + assembler feedback |  |  |  |  |  |
| LLM + runtime feedback |  |  |  |  |  |
| clang/gcc -O0 |  |  |  |  |  |
| clang/gcc -O3 |  |  |  |  |  |

## NPB Second

Use NPB after the kernel path is stable.

Recommended progression:

1. serial kernels extracted from NPB
2. full serial benchmarks at small classes
3. OpenMP/MPI variants only after the serial path works

For full NPB programs, correctness should come from the official benchmark
verifier. For kernel-level NPB tasks, use differential tests against a
compiler-built reference implementation.

## Input Conditions

Use separate input conditions in the experiment:

- C source -> assembly
- Fortran source -> assembly
- LLVM IR -> assembly

The first two are full compiler-replacement conditions. The LLVM IR condition is
a backend-replacement condition and should not be described as replacing the
entire compiler frontend.
