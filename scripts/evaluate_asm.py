#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def run(cmd: list[str], cwd: Path | None = None) -> dict[str, object]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "cmd": cmd,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "ok": False,
        }
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "ok": proc.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble, optionally link, and optionally run candidate assembly."
    )
    parser.add_argument("--asm", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cc", default="cc")
    parser.add_argument(
        "--link",
        nargs="*",
        default=None,
        help="Extra files/flags used to link the object into an executable.",
    )
    parser.add_argument(
        "--run-command",
        help="Command to run after linking. Use {exe} as the executable placeholder.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    obj = args.workdir / "candidate.o"
    exe = args.workdir / "candidate.exe"

    result: dict[str, object] = {
        "asm": str(args.asm),
        "target": args.target,
        "object": str(obj),
    }

    assemble_cmd = [args.cc]
    if "clang" in os.path.basename(args.cc):
        assemble_cmd.extend(["-target", args.target])
    assemble_cmd.extend(["-c", str(args.asm), "-o", str(obj)])
    assemble = run(assemble_cmd)
    result["assemble"] = assemble

    if assemble["ok"] and args.link is not None:
        link_cmd = [args.cc, str(obj), *args.link, "-o", str(exe)]
        link = run(link_cmd)
        result["link"] = link

        if link["ok"] and args.run_command:
            command = args.run_command.format(exe=str(exe))
            result["run"] = run(shlex.split(command), cwd=args.workdir)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))

    assemble_ok = bool(result["assemble"]["ok"])  # type: ignore[index]
    link_ok = "link" not in result or bool(result["link"]["ok"])  # type: ignore[index]
    run_ok = "run" not in result or bool(result["run"]["ok"])  # type: ignore[index]
    return 0 if assemble_ok and link_ok and run_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
