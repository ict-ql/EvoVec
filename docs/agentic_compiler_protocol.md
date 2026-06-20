# Agentic Compiler Protocol

## Framing

An iterative LLM agent may be evaluated as a compiler replacement when the
experimental unit is a fixed source-to-assembly procedure rather than a single
sample. The agent receives a program fragment, emits target assembly, observes
only predeclared verifier feedback, and revises the same candidate until it
passes or reaches a fixed budget. This is distinct from pass@1: pass@1 measures
the correctness of the first sampled assembly; the agentic compiler condition
measures a reproducible closed-loop translation protocol.

The LLM path must not call a compiler to generate assembly, object code, LLVM
IR, optimized source, or repair patches. Compilers are permitted only as
baselines, reference oracles, assemblers, and linkers. These roles must be
declared in the methods section and kept outside the generation loop.

## Fixed Artifacts

Each experiment must specify immutable artifacts before any benchmark task is
run:

- prompt template: the exact source-to-assembly instruction template
- verifier: the exact assemble, link, execute, and differential or benchmark
  verification procedure
- self-correction template: the exact repair instruction used after a failed
  attempt
- maximum iterations: the maximum number of LLM attempts per task
- target: ISA, ABI, assembler syntax, operating system, and harness interface

Changing any artifact creates a new protocol version and must be reported as a
separate condition.

## Initial Prompt Template

Use one fixed template for every task in a condition:

```text
You are replacing a compiler for this task.
Translate the input program fragment to target assembly.

Target:
{target_triple}
ABI:
{abi_summary}
Assembler syntax:
{assembler_syntax}
Function interface:
{function_signature_and_calling_convention}
Allowed external symbols:
{allowed_symbols}

Input program:
{source_or_ir}

Output only complete assembly for the requested function or translation unit.
Do not emit C, LLVM IR, pseudocode, explanations, or build commands.
```

The template may include task metadata required by the ABI or harness, but it
must not include per-task hints derived from compiler output.

## Fixed Verifier

The verifier is the only feedback channel. It should run the same ordered stages
for every attempt:

1. assemble the emitted assembly with a fixed assembler invocation
2. link the object with a fixed harness
3. execute deterministic tests
4. compare outputs with a reference oracle or official benchmark verifier
5. optionally measure runtime and code size after correctness passes

Assembler and linker diagnostics may be returned verbatim or normalized, but the
normalization rule must be fixed. Runtime feedback should report only predefined
signals such as exit status, failing test id, observed value, expected value,
tolerance, timeout, and sanitizer or trap summaries.

## Self-Correction Template

Use one fixed repair template after each failed attempt:

```text
The previous assembly failed the fixed verifier.

Original input:
{source_or_ir}

Previous assembly:
{previous_assembly}

Verifier result:
{verifier_feedback}

Revise the assembly so that it satisfies the same target, ABI, and function
interface. Output only complete replacement assembly. Do not call or imitate a
compiler, and do not emit explanations.
```

The agent may use its own prior assembly and verifier feedback, but it must not
receive compiler-generated candidate assembly or task-specific compiler repair
advice.

## Iteration Budget and Stopping Rules

Set the maximum number of attempts before the run begins; a typical initial
budget is five attempts per task, including the first attempt. Stop early when
the candidate assembles, links, executes, and passes the fixed correctness
verifier. Stop with failure when the maximum attempt count is reached, output is
malformed, the verifier times out under the fixed timeout policy, or the agent
violates the output contract in a way the harness cannot consume.

Performance feedback should not be used for further repair unless the protocol
defines a separate optimization condition with its own fixed template, budget,
and stopping rule.

## Metrics

Report both per-attempt and per-task outcomes:

- total tasks and excluded tasks, with reasons
- assemble rate by attempt and within budget
- link rate by attempt and within budget
- correctness rate by attempt and within budget
- median attempts to first correct candidate
- verifier timeout and malformed-output rates
- runtime for correct candidates, normalized to compiler baselines
- code size for correct candidates, normalized to compiler baselines
- wall-clock agent cost, token count, and number of model calls
- failure taxonomy: syntax, ABI, link, wrong answer, nondeterminism, timeout,
  illegal instruction, crash, and output-contract violation

For comparison with pass@1, report the first-attempt correctness separately from
the within-budget agentic correctness. Compiler baselines such as `clang`,
`gcc`, `gfortran`, or `flang` should be reported as ordinary baselines or
oracles, not as part of LLM generation.
