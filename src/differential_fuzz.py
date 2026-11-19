from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable


CommandRunner = Callable[..., dict[str, Any]]
LLVM_SYMBOL_CHARS = r"A-Za-z0-9_$.-"
DEFINE_PATTERN = re.compile(
    rf"^\s*define\b[^@]*@(?P<name>[{LLVM_SYMBOL_CHARS}]+)\s*\(",
    re.MULTILINE,
)


def defined_functions(ir_text: str) -> list[str]:
    return list(dict.fromkeys(match.group("name") for match in DEFINE_PATTERN.finditer(ir_text)))


def prefix_defined_functions(ir_text: str, prefix: str) -> tuple[str, dict[str, str]]:
    names = defined_functions(ir_text)
    mapping = {name: f"{prefix}{name}" for name in names}
    rewritten = ir_text
    for name in sorted(names, key=len, reverse=True):
        pattern = re.compile(rf"@{re.escape(name)}(?=[^{LLVM_SYMBOL_CHARS}]|$)")
        rewritten = pattern.sub(f"@{mapping[name]}", rewritten)
    return rewritten, mapping


def rename_llvm_symbol(ir_text: str, old_name: str, new_name: str) -> str:
    pattern = re.compile(rf"@{re.escape(old_name)}(?=[^{LLVM_SYMBOL_CHARS}]|$)")
    return pattern.sub(f"@{new_name}", ir_text)


def infer_tsvc_dimensions(ir_text: str) -> tuple[int, int]:
    one_dimensional = re.search(
        r"^@(?:a|x)\s*=.*?\[(\d+) x (?:float|double)\]",
        ir_text,
        re.MULTILINE,
    )
    two_dimensional = re.search(
        r"^@aa\s*=.*?\[(\d+) x \[(\d+) x (?:float|double)\]\]",
        ir_text,
        re.MULTILINE,
    )
    if not one_dimensional or not two_dimensional:
        raise ValueError("could not infer TSVC dimensions from scalar IR globals")
    len_1d = int(one_dimensional.group(1))
    outer_len = int(two_dimensional.group(1))
    inner_len = int(two_dimensional.group(2))
    if outer_len != inner_len:
        raise ValueError(f"expected square TSVC 2D arrays, found {outer_len}x{inner_len}")
    return len_1d, outer_len


def render_fuzz_harness(kernel: str, arg_expr: str, len_1d: int, len_2d: int) -> str:
    template = r'''#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "array_defs.h"

__attribute__((aligned(ARRAY_ALIGNMENT))) real_t flat_2d_array[LEN_2D * LEN_2D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t x[LEN_1D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t a[LEN_1D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t b[LEN_1D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t c[LEN_1D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t d[LEN_1D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t e[LEN_1D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t aa[LEN_2D][LEN_2D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t bb[LEN_2D][LEN_2D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t cc[LEN_2D][LEN_2D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t tt[LEN_2D][LEN_2D];
__attribute__((aligned(ARRAY_ALIGNMENT))) int indx[LEN_1D];
static __attribute__((aligned(ARRAY_ALIGNMENT))) real_t xx_storage[LEN_1D];
static __attribute__((aligned(ARRAY_ALIGNMENT))) real_t yy_storage[LEN_1D];
real_t * __restrict__ xx = xx_storage;
real_t *yy = yy_storage;

extern real_t __comagent_ref_@@KERNEL@@(struct args_t *);
extern real_t __comagent_candidate_@@KERNEL@@(struct args_t *);

static const uint8_t *fuzz_data;
static size_t fuzz_size;
static size_t fuzz_offset;
static jmp_buf exit_jump;
static int exit_armed;
static int captured_exit_status;
static int dummy_call_count;
static uint32_t dummy_scalar_bits[3];

static uint32_t next_u32(void) {
    uint32_t value = 0x9e3779b9u ^ (uint32_t)fuzz_offset;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        value ^= (uint32_t)fuzz_data[fuzz_offset % fuzz_size] << shift;
        fuzz_offset++;
    }
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    return value;
}

static real_t next_real(void) {
    int32_t scaled = (int32_t)(next_u32() % 4095u) - 2047;
    if (scaled >= -8 && scaled <= 8) {
        scaled += scaled < 0 ? -9 : 9;
    }
    return (real_t)scaled / (real_t)128.0;
}

static void fill_real(real_t *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        values[index] = next_real();
    }
}

int initialise_arrays(const char *name) {
    (void)name;
    fuzz_offset = 0;
    fill_real(flat_2d_array, LEN_2D * LEN_2D);
    fill_real(x, LEN_1D);
    fill_real(a, LEN_1D);
    fill_real(b, LEN_1D);
    fill_real(c, LEN_1D);
    fill_real(d, LEN_1D);
    fill_real(e, LEN_1D);
    fill_real((real_t *)aa, LEN_2D * LEN_2D);
    fill_real((real_t *)bb, LEN_2D * LEN_2D);
    fill_real((real_t *)cc, LEN_2D * LEN_2D);
    fill_real((real_t *)tt, LEN_2D * LEN_2D);
    fill_real(xx_storage, LEN_1D);
    fill_real(yy_storage, LEN_1D);
    for (size_t index = 0; index < LEN_1D; index++) {
        indx[index] = 1 + (int)(next_u32() % (LEN_2D - 1));
    }
    xx = xx_storage;
    yy = yy_storage;
    return 0;
}

int dummy(
    real_t av[LEN_1D], real_t bv[LEN_1D], real_t cv[LEN_1D],
    real_t dv[LEN_1D], real_t ev[LEN_1D],
    real_t aav[LEN_2D][LEN_2D], real_t bbv[LEN_2D][LEN_2D],
    real_t ccv[LEN_2D][LEN_2D], real_t scalar
) {
    (void)av;
    (void)bv;
    (void)cv;
    (void)dv;
    (void)ev;
    (void)aav;
    (void)bbv;
    (void)ccv;
    if (dummy_call_count < 3) {
        memcpy(&dummy_scalar_bits[dummy_call_count], &scalar, sizeof(scalar));
    }
    dummy_call_count++;
    if (exit_armed && dummy_call_count >= 3) {
        longjmp(exit_jump, 2);
    }
    return 0;
}

real_t calc_checksum(const char *name) {
    (void)name;
    real_t sum = 0;
    const real_t *blocks[] = {
        flat_2d_array, x, a, b, c, d, e,
        (const real_t *)aa, (const real_t *)bb, (const real_t *)cc,
        (const real_t *)tt, xx_storage, yy_storage,
    };
    const size_t counts[] = {
        LEN_2D * LEN_2D, LEN_1D, LEN_1D, LEN_1D, LEN_1D, LEN_1D,
        LEN_1D, LEN_2D * LEN_2D, LEN_2D * LEN_2D, LEN_2D * LEN_2D,
        LEN_2D * LEN_2D, LEN_1D, LEN_1D,
    };
    for (size_t block = 0; block < sizeof(counts) / sizeof(counts[0]); block++) {
        for (size_t index = 0; index < counts[block]; index++) {
            sum += blocks[block][index];
        }
    }
    return sum;
}

_Noreturn void __comagent_fuzz_exit(int status) {
    if (exit_armed) {
        captured_exit_status = status;
        longjmp(exit_jump, 1);
    }
    abort();
}

struct observed_state {
    real_t flat_2d_array[LEN_2D * LEN_2D];
    real_t x[LEN_1D];
    real_t a[LEN_1D];
    real_t b[LEN_1D];
    real_t c[LEN_1D];
    real_t d[LEN_1D];
    real_t e[LEN_1D];
    real_t aa[LEN_2D * LEN_2D];
    real_t bb[LEN_2D * LEN_2D];
    real_t cc[LEN_2D * LEN_2D];
    real_t tt[LEN_2D * LEN_2D];
    real_t xx[LEN_1D];
    real_t yy[LEN_1D];
    int indx[LEN_1D];
    uintptr_t xx_pointer;
    uintptr_t yy_pointer;
};

struct run_result {
    int terminated;
    int exit_status;
    int benchmark_iterations;
    uint32_t dummy_scalar_bits[3];
    uint32_t return_bits;
    int n1;
    int n3;
    uint32_t s1_bits;
    uint32_t s2_bits;
    int ip[LEN_1D];
};

static struct observed_state reference_state;

static void capture_state(struct observed_state *state) {
    memset(state, 0, sizeof(*state));
    memcpy(state->flat_2d_array, flat_2d_array, sizeof(flat_2d_array));
    memcpy(state->x, x, sizeof(x));
    memcpy(state->a, a, sizeof(a));
    memcpy(state->b, b, sizeof(b));
    memcpy(state->c, c, sizeof(c));
    memcpy(state->d, d, sizeof(d));
    memcpy(state->e, e, sizeof(e));
    memcpy(state->aa, aa, sizeof(aa));
    memcpy(state->bb, bb, sizeof(bb));
    memcpy(state->cc, cc, sizeof(cc));
    memcpy(state->tt, tt, sizeof(tt));
    memcpy(state->xx, xx_storage, sizeof(xx_storage));
    memcpy(state->yy, yy_storage, sizeof(yy_storage));
    memcpy(state->indx, indx, sizeof(indx));
    state->xx_pointer = (uintptr_t)xx;
    state->yy_pointer = (uintptr_t)yy;
}

typedef real_t (*kernel_function)(struct args_t *);

static struct run_result run_kernel(kernel_function function) {
    fuzz_offset = 0;
    int n1 = 1 + (int)(next_u32() % (LEN_2D / 2));
    int n3 = 1 + (int)(next_u32() % (LEN_2D / 2));
    real_t s1 = next_real();
    real_t s2 = next_real();
    int ip_storage[LEN_1D];
    for (size_t index = 0; index < LEN_1D; index++) {
        ip_storage[index] = 1 + (int)(next_u32() % (LEN_2D - 1));
    }
    int *ip = ip_storage;
    struct args_t arguments = {.arg_info = @@ARG_EXPR@@};
    struct run_result summary;
    memset(&summary, 0, sizeof(summary));
    summary.n1 = n1;
    summary.n3 = n3;
    memcpy(&summary.s1_bits, &s1, sizeof(s1));
    memcpy(&summary.s2_bits, &s2, sizeof(s2));
    memcpy(summary.ip, ip_storage, sizeof(ip_storage));
    captured_exit_status = 0;
    dummy_call_count = 0;
    memset(dummy_scalar_bits, 0, sizeof(dummy_scalar_bits));
    exit_armed = 1;
    int jump_reason = setjmp(exit_jump);
    if (jump_reason != 0) {
        exit_armed = 0;
        summary.terminated = jump_reason == 1;
        summary.exit_status = jump_reason == 1 ? captured_exit_status : 0;
        summary.benchmark_iterations = dummy_call_count;
        memcpy(summary.dummy_scalar_bits, dummy_scalar_bits, sizeof(dummy_scalar_bits));
        return summary;
    }
    real_t result = function(&arguments);
    exit_armed = 0;
    summary.benchmark_iterations = dummy_call_count;
    memcpy(summary.dummy_scalar_bits, dummy_scalar_bits, sizeof(dummy_scalar_bits));
    memcpy(&summary.return_bits, &result, sizeof(result));
    return summary;
}

static void fail_difference(const char *kind, size_t offset, unsigned ref_byte, unsigned candidate_byte) {
    fprintf(
        stderr,
        "COMAGENT_DIFFERENTIAL_MISMATCH kind=%s byte_offset=%zu reference=%u candidate=%u\n",
        kind,
        offset,
        ref_byte,
        candidate_byte
    );
    abort();
}

static void compare_bytes(const char *kind, const void *reference, const void *candidate, size_t size) {
    const unsigned char *reference_bytes = reference;
    const unsigned char *candidate_bytes = candidate;
    for (size_t offset = 0; offset < size; offset++) {
        if (reference_bytes[offset] != candidate_bytes[offset]) {
            fail_difference(kind, offset, reference_bytes[offset], candidate_bytes[offset]);
        }
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0) {
        return 0;
    }
    fuzz_data = data;
    fuzz_size = size;
    struct run_result reference_result = run_kernel(__comagent_ref_@@KERNEL@@);
    capture_state(&reference_state);
    struct run_result candidate_result = run_kernel(__comagent_candidate_@@KERNEL@@);
    compare_bytes(
        "termination-kind",
        &reference_result.terminated,
        &candidate_result.terminated,
        sizeof(reference_result.terminated)
    );
    if (reference_result.terminated) {
        compare_bytes(
            "exit-status",
            &reference_result.exit_status,
            &candidate_result.exit_status,
            sizeof(reference_result.exit_status)
        );
        return 0;
    }
    compare_bytes("return-and-arguments", &reference_result, &candidate_result, sizeof(reference_result));
    struct observed_state candidate_state;
    capture_state(&candidate_state);
    compare_bytes("global-state", &reference_state, &candidate_state, sizeof(reference_state));
    return 0;
}
'''
    if len_1d <= 0 or len_2d <= 1:
        raise ValueError("TSVC dimensions must satisfy LEN_1D > 0 and LEN_2D > 1")
    return (
        template.replace("@@KERNEL@@", kernel)
        .replace("@@ARG_EXPR@@", arg_expr)
        .replace("#include \"common.h\"", f"#include \"common.h\"\n_Static_assert(LEN_1D == {len_1d}, \"LEN_1D mismatch\");\n_Static_assert(LEN_2D == {len_2d}, \"LEN_2D mismatch\");")
    )


def run_differential_fuzz(
    *,
    source_ir_path: Path,
    candidate_ir_path: Path,
    task: dict[str, Any],
    support_dir: Path,
    out_dir: Path,
    clang: str,
    llvm_link: str,
    march: str,
    runs: int,
    max_len: int,
    per_input_timeout: int,
    timeout: int,
    sanitizers: str,
    extra_clang_args: list[str],
    command_runner: CommandRunner,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_ir = source_ir_path.read_text(encoding="utf-8", errors="replace")
    candidate_ir = candidate_ir_path.read_text(encoding="utf-8", errors="replace")
    try:
        len_1d, len_2d = infer_tsvc_dimensions(source_ir)
    except ValueError as exc:
        return {
            "enabled": True,
            "ok": False,
            "status": "inconclusive",
            "error": str(exc),
            "stage": "inspect_source",
        }
    prefixed_source, source_mapping = prefix_defined_functions(source_ir, "__comagent_ref_")
    prefixed_candidate, candidate_mapping = prefix_defined_functions(candidate_ir, "__comagent_candidate_")
    prefixed_source = rename_llvm_symbol(prefixed_source, "exit", "__comagent_fuzz_exit")
    prefixed_candidate = rename_llvm_symbol(prefixed_candidate, "exit", "__comagent_fuzz_exit")
    kernel = str(task["kernel"])
    if kernel not in source_mapping:
        return {
            "enabled": True,
            "ok": False,
            "status": "inconclusive",
            "error": f"reference IR does not define {kernel}",
            "stage": "rename",
        }
    if kernel not in candidate_mapping:
        return {
            "enabled": True,
            "ok": False,
            "status": "fail",
            "error": f"candidate IR does not define {kernel}",
            "stage": "rename",
        }

    source_prefixed_path = out_dir / "reference.prefixed.ll"
    candidate_prefixed_path = out_dir / "candidate.prefixed.ll"
    harness_c_path = out_dir / "differential_fuzz_harness.c"
    harness_ir_path = out_dir / "differential_fuzz_harness.ll"
    source_verified_path = out_dir / "reference.verified.ll"
    candidate_verified_path = out_dir / "candidate.verified.ll"
    combined_ir_path = out_dir / "combined.ll"
    executable_path = out_dir / "differential_fuzzer"
    source_prefixed_path.write_text(prefixed_source, encoding="utf-8")
    candidate_prefixed_path.write_text(prefixed_candidate, encoding="utf-8")
    harness_c_path.write_text(
        render_fuzz_harness(
            kernel,
            str(task.get("arg_expr") or "NULL"),
            len_1d,
            len_2d,
        ),
        encoding="utf-8",
    )

    stages: dict[str, Any] = {}
    stages["compile_harness"] = command_runner(
        [
            clang,
            "-S",
            "-emit-llvm",
            "-O1",
            "-g",
            "-I",
            str(support_dir),
            str(harness_c_path),
            "-o",
            str(harness_ir_path),
        ],
        timeout=timeout,
    )
    if not stages["compile_harness"]["ok"]:
        return _fuzz_result(False, "compile_harness", stages, combined_ir_path, executable_path)

    stages["verify_reference_ir"] = command_runner(
        [llvm_link, str(source_prefixed_path), "-S", "-o", str(source_verified_path)],
        timeout=timeout,
    )
    if not stages["verify_reference_ir"]["ok"]:
        return _fuzz_result(False, "verify_reference_ir", stages, combined_ir_path, executable_path)

    stages["verify_candidate_ir"] = command_runner(
        [llvm_link, str(candidate_prefixed_path), "-S", "-o", str(candidate_verified_path)],
        timeout=timeout,
    )
    if not stages["verify_candidate_ir"]["ok"]:
        return _fuzz_result(
            False,
            "verify_candidate_ir",
            stages,
            combined_ir_path,
            executable_path,
            forced_status="fail",
            counterexample_kind="invalid-candidate-ir",
        )

    stages["link_ir"] = command_runner(
        [
            llvm_link,
            str(source_verified_path),
            str(candidate_verified_path),
            str(harness_ir_path),
            "-S",
            "-o",
            str(combined_ir_path),
        ],
        timeout=timeout,
    )
    if not stages["link_ir"]["ok"]:
        return _fuzz_result(False, "link_ir", stages, combined_ir_path, executable_path)

    sanitizer_flag = f"-fsanitize={sanitizers}"
    stages["build_fuzzer"] = command_runner(
        [
            clang,
            "-O1",
            "-g",
            sanitizer_flag,
            "-fno-omit-frame-pointer",
            f"-march={march}",
            *extra_clang_args,
            str(combined_ir_path),
            "-lm",
            "-o",
            str(executable_path),
        ],
        timeout=timeout,
    )
    if not stages["build_fuzzer"]["ok"]:
        return _fuzz_result(False, "build_fuzzer", stages, combined_ir_path, executable_path)

    stages["fuzz"] = command_runner(
        [
            str(executable_path),
            f"-runs={runs}",
            f"-max_len={max_len}",
            f"-timeout={per_input_timeout}",
            "-detect_leaks=0",
        ],
        timeout=timeout,
    )
    fuzz_ok = bool(stages["fuzz"]["ok"])
    return _fuzz_result(
        fuzz_ok,
        "complete" if fuzz_ok else "fuzz",
        stages,
        combined_ir_path,
        executable_path,
    )


def _fuzz_result(
    ok: bool,
    stage: str,
    stages: dict[str, Any],
    combined_ir_path: Path,
    executable_path: Path,
    forced_status: str | None = None,
    counterexample_kind: str | None = None,
) -> dict[str, Any]:
    fuzz_stage = stages.get("fuzz") or {}
    fuzz_output = f"{fuzz_stage.get('stdout', '')}\n{fuzz_stage.get('stderr', '')}"
    is_input_timeout = "libFuzzer: timeout" in fuzz_output
    has_mismatch = "COMAGENT_DIFFERENTIAL_MISMATCH" in fuzz_output
    candidate_runtime_failure = (
        not is_input_timeout
        and "__comagent_candidate_" in fuzz_output
        and any(marker in fuzz_output for marker in ("AddressSanitizer", "UndefinedBehaviorSanitizer", "runtime error:"))
    )
    has_counterexample = has_mismatch or candidate_runtime_failure or forced_status == "fail"
    if counterexample_kind is None:
        if has_mismatch:
            counterexample_kind = "state-mismatch"
        elif candidate_runtime_failure:
            counterexample_kind = "candidate-runtime-failure"
    status = forced_status or ("pass" if ok else ("fail" if has_counterexample else "inconclusive"))
    return {
        "enabled": True,
        "ok": ok,
        "status": status,
        "counterexample": has_counterexample,
        "counterexample_kind": counterexample_kind,
        "stage": stage,
        "stages": stages,
        "combined_ir": str(combined_ir_path),
        "executable": str(executable_path),
    }
