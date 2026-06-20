# GPT-5.5 Pass@1 Protocol

## Fixed Prompt

All source-to-assembly pass@1 runs must use:

```text
prompts/source_to_asm_fast_v1.txt
```

Do not tune the prompt per benchmark task. If the prompt changes, increment the
template version and report results separately.

## Model

Default model id:

```text
gpt-5.5
```

The runner uses the Responses API and sends exactly one request per task. That
single response is the pass@1 candidate.

## Command

```bash
OPENAI_API_KEY=... python3 scripts/run_pass_at_1.py \
  --manifest runs/manifests/tsvc.jsonl \
  --out-dir runs/pass_at_1/gpt-5.5/tsvc_v1 \
  --model gpt-5.5 \
  --target x86_64-pc-linux-gnu \
  --mode source-to-asm \
  --limit 10
```

## Scoring

The initial automated score is assembly success:

```text
pass@1_assemble = assembled_tasks / total_tasks
```

This is not enough for a paper. The next score must link each candidate into a
fixed harness and run correctness tests:

```text
pass@1_correct = verifier_passing_tasks / total_tasks
```

For TSVC, use per-kernel differential harnesses. For NPB, use the official
benchmark verifier once whole-program generation is attempted.
