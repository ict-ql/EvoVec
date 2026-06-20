#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_KERNELS = ["s000", "s111", "s1111"]
LOCAL_HELPERS = {
    "s151": ["s151s"],
    "s152": ["s152s"],
    "s31111": ["test"],
    "s471": ["s471s"],
    "s4121": ["f"],
}


def parse_time_function_calls(source: str) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    pattern = re.compile(r"time_function\(&([A-Za-z_][A-Za-z0-9_]*),\s*(.*?)\);")
    for match in pattern.finditer(source):
        calls.append((match.group(1), match.group(2).strip()))
    return calls


def find_function(source: str, name: str) -> str:
    pattern = re.compile(rf"(?:real_t|void|int)\s+{re.escape(name)}\s*\([^)]*\)\s*\{{")
    match = pattern.search(source)
    if not match:
        raise ValueError(f"could not find function {name}")
    brace = source.find("{", match.start())
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[match.start() : index + 1]
    raise ValueError(f"could not find end of function {name}")


def patched_common_h(text: str, iterations: int, len_1d: int, len_2d: int) -> str:
    replacements = {
        r"#define iterations\s+\d+": f"#define iterations {iterations}",
        r"#define LEN_1D\s+\d+": f"#define LEN_1D {len_1d}",
        r"#define LEN_2D\s+\d+": f"#define LEN_2D {len_2d}",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text


def write_globals(path: Path) -> None:
    path.write_text(
        """#include "common.h"
#include "array_defs.h"

__attribute__((aligned(ARRAY_ALIGNMENT))) real_t flat_2d_array[LEN_2D*LEN_2D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t x[LEN_1D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t a[LEN_1D], b[LEN_1D], c[LEN_1D], d[LEN_1D], e[LEN_1D],
    aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D], tt[LEN_2D][LEN_2D];
__attribute__((aligned(ARRAY_ALIGNMENT))) int indx[LEN_1D];
real_t* __restrict__ xx;
real_t* yy;
""",
        encoding="utf-8",
    )


def harness_source(kernel: str, arg_expr: str) -> str:
    return f"""#include <stdio.h>
#include <stdlib.h>
#include "common.h"

extern real_t {kernel}(struct args_t *func_args);

int main(void) {{
    int n1 = 1;
    int n3 = 1;
    int *ip = NULL;
    real_t s1 = 0.0f;
    real_t s2 = 0.0f;
    init(&ip, &s1, &s2);
    (void)n1;
    (void)n3;
    (void)ip;
    (void)s1;
    (void)s2;

    printf("Loop \\tTime(sec) \\tChecksum\\n");
    struct args_t func_args = {{.arg_info = {arg_expr}}};
    real_t result = {kernel}(&func_args);
    double tic = func_args.t1.tv_sec + (func_args.t1.tv_usec / 1000000.0);
    double toc = func_args.t2.tv_sec + (func_args.t2.tv_usec / 1000000.0);
    printf("%10.6f\\t%.9g\\n", toc - tic, (double)result);
    return EXIT_SUCCESS;
}}
"""


def prompt_input(common_h: str, array_defs_h: str, kernel_source: str) -> str:
    return f"""#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <sys/time.h>

/* common.h */
{common_h}

/* array_defs.h */
{array_defs_h}

/* Externally linked helper implementations are provided by TSVC common.c. */
/* Generate assembly only for the following kernel function. */
{kernel_source}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Split TSVC_2 into isolated kernel tasks.")
    parser.add_argument("--src-dir", type=Path, default=Path("third_party/TSVC_2/src"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/tsvc_kernels/v1"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/manifests/tsvc_kernels.jsonl"))
    parser.add_argument(
        "--kernels",
        default=",".join(DEFAULT_KERNELS),
        help="Comma-separated kernel names, or 'all' to use every TSVC main time_function call.",
    )
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--len-1d", type=int, default=320)
    parser.add_argument("--len-2d", type=int, default=32)
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    tsvc_c = (src_dir / "tsvc.c").read_text(encoding="utf-8")
    common_c = (src_dir / "common.c").read_text(encoding="utf-8")
    dummy_c = (src_dir / "dummy.c").read_text(encoding="utf-8")
    common_h = patched_common_h(
        (src_dir / "common.h").read_text(encoding="utf-8"),
        args.iterations,
        args.len_1d,
        args.len_2d,
    )
    array_defs_h = (src_dir / "array_defs.h").read_text(encoding="utf-8")

    out_dir = args.out_dir.resolve()
    support_dir = out_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    (support_dir / "common.c").write_text(common_c, encoding="utf-8")
    (support_dir / "dummy.c").write_text(dummy_c, encoding="utf-8")
    (support_dir / "common.h").write_text(common_h, encoding="utf-8")
    (support_dir / "array_defs.h").write_text(array_defs_h, encoding="utf-8")
    write_globals(support_dir / "globals.c")

    main_calls = parse_time_function_calls(tsvc_c)
    arg_by_kernel = {kernel: arg_expr for kernel, arg_expr in main_calls}
    if args.kernels.strip().lower() == "all":
        kernel_items = main_calls
    else:
        requested = [item.strip() for item in args.kernels.split(",") if item.strip()]
        kernel_items = [(kernel, arg_by_kernel.get(kernel, "NULL")) for kernel in requested]

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as manifest:
        for kernel, arg_expr in kernel_items:
            helper_names = LOCAL_HELPERS.get(kernel, [])
            helper_source = "\n\n".join(find_function(tsvc_c, helper) for helper in helper_names)
            kernel_source = find_function(tsvc_c, kernel)
            task_source = (helper_source + "\n\n" + kernel_source).strip() if helper_source else kernel_source
            kernel_dir = out_dir / "kernels" / kernel
            kernel_dir.mkdir(parents=True, exist_ok=True)
            input_path = kernel_dir / "input.c"
            ref_path = kernel_dir / "reference_kernel.c"
            harness_path = kernel_dir / "harness.c"

            input_path.write_text(
                prompt_input(common_h, array_defs_h, task_source),
                encoding="utf-8",
            )
            ref_path.write_text(
                "#include <time.h>\n"
                "#include <stdio.h>\n"
                "#include <stdlib.h>\n"
                "#include <stddef.h>\n"
                "#include <math.h>\n"
                "#include <sys/time.h>\n"
                '#include "common.h"\n'
                '#include "array_defs.h"\n\n'
                + task_source
                + "\n",
                encoding="utf-8",
            )
            harness_path.write_text(harness_source(kernel, arg_expr), encoding="utf-8")

            record = {
                "suite": "tsvc",
                "task_id": f"tsvc:{kernel}",
                "kernel": kernel,
                "language": "c",
                "granularity": "kernel_function",
                "root": str(out_dir),
                "path": str(input_path.relative_to(out_dir)),
                "kernel_dir": str(kernel_dir),
                "support_dir": str(support_dir),
                "reference_kernel": str(ref_path),
                "harness": str(harness_path),
                "iterations": args.iterations,
                "len_1d": args.len_1d,
                "len_2d": args.len_2d,
                "arg_expr": arg_expr,
                "helpers": helper_names,
                "required_symbols": [kernel, *helper_names],
            }
            manifest.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"wrote {len(kernel_items)} TSVC kernel tasks to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
