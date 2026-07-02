#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any


DEF_RE = re.compile(r"\bdef\s+int_([A-Za-z0-9_]+)\b")
LLVM_INTRINSIC_ALIASES = {
    "_tile_dpbssd": "llvm.x86.tdpbssd.internal",
    "_tile_dpbusd": "llvm.x86.tdpbusd.internal",
    "amx.tile.tdpbssd": "llvm.x86.tdpbssd.internal",
    "amx.tile.tdpbusd": "llvm.x86.tdpbusd.internal",
}


def run_cmd(cmd: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": 127, "stdout": "", "stderr": str(exc), "cmd": cmd}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "cmd": cmd,
            "error": "timeout",
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "cmd": cmd,
    }


def intrinsic_base_from_td_def(def_name: str) -> str:
    return "llvm." + def_name.replace("_", ".")


def default_td_files(llvm_project_root: Path) -> list[Path]:
    candidates = [
        llvm_project_root / "llvm" / "include" / "llvm" / "IR",
        llvm_project_root / "include" / "llvm" / "IR",
    ]
    include_dir = next((path for path in candidates if path.exists()), None)
    if include_dir is None:
        raise SystemExit(f"could not find llvm/include/llvm/IR under {llvm_project_root}")
    names = ["Intrinsics.td", "IntrinsicsX86.td"]
    return [include_dir / name for name in names if (include_dir / name).exists()]


def parse_td_intrinsics(paths: list[Path]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in DEF_RE.finditer(text):
            enum_name = match.group(1)
            entries[intrinsic_base_from_td_def(enum_name)] = enum_name
    return entries


def normalize_hint_name(text: str) -> str:
    text = text.strip()
    text = LLVM_INTRINSIC_ALIASES.get(text, text)
    if text.startswith("llvm.vector.") and not text.startswith("llvm.vector.reduce."):
        return "ir.vector." + text.removeprefix("llvm.vector.")
    return text


def hint_text(item: dict[str, Any]) -> str:
    intrinsic = str(item.get("intrinsic") or "").strip()
    label = str(item.get("label") or "").strip()
    return normalize_hint_name(intrinsic or label)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"expected JSON object in {path}:{line_number}")
            rows.append(row)
    return rows


def collect_recommendation_names(path: Path, top_k: int) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in load_jsonl(path):
        for item in row.get("recommendations", [])[:top_k]:
            if not isinstance(item, dict):
                continue
            name = hint_text(item)
            if name and name.startswith("llvm.") and name not in seen:
                names.append(name)
                seen.add(name)
    return names


def collect_names(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        name = normalize_hint_name(name)
        if name and name.startswith("llvm.") and name not in seen:
            names.append(name)
            seen.add(name)

    for name in args.name or []:
        add(name)
    for names_file in args.names_file or []:
        for line in names_file.read_text(encoding="utf-8", errors="replace").splitlines():
            add(line.strip())
    if args.recommendations_jsonl:
        for name in collect_recommendation_names(args.recommendations_jsonl, args.recommendation_top_k):
            add(name)
    return names


def find_base(name: str, entries: dict[str, str]) -> tuple[str, str] | None:
    best: tuple[str, str] | None = None
    for base, enum_name in entries.items():
        if name == base or name.startswith(base + "."):
            if best is None or len(base) > len(best[0]):
                best = (base, enum_name)
    return best


def find_llvm_config(args: argparse.Namespace) -> str:
    if args.llvm_config:
        return str(args.llvm_config)
    candidates: list[Path] = []
    if args.llvm_project_root:
        root = args.llvm_project_root
        candidates.extend(
            [
                root / "build" / "bin" / "llvm-config",
                root / "build-release" / "bin" / "llvm-config",
                root / "llvm-build" / "bin" / "llvm-config",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    for name in ("llvm-config", "llvm-config-19", "llvm-config-18", "llvm-config-17"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "could not find llvm-config; pass --llvm-config from the same LLVM build as --llvm-project-root"
    )


def llvm_config_args(llvm_config: str, args: list[str]) -> list[str]:
    result = run_cmd([llvm_config, *args], timeout=30)
    if not result["ok"]:
        raise SystemExit(f"llvm-config {' '.join(args)} failed:\n{result['stderr']}")
    return shlex.split(str(result["stdout"]))


def find_versioned_system_library(name: str) -> str | None:
    machine = platform.machine().lower()
    incompatible_parts: set[str] = set()
    if machine in {"x86_64", "amd64"}:
        incompatible_parts.update({"i386-linux-gnu", "i686-linux-gnu"})
    search_roots = [
        Path("/usr/lib"),
        Path("/usr/lib64"),
        Path("/usr/local/lib"),
        Path("/lib"),
        Path("/lib64"),
    ]
    matches: list[Path] = []
    for root in search_roots:
        if root.exists():
            matches.extend(root.glob(f"**/lib{name}.so*"))
            matches.extend(root.glob(f"**/lib{name}.a"))
    existing = sorted(
        {
            path.resolve()
            for path in matches
            if path.exists() and not incompatible_parts.intersection(path.parts)
        },
        key=lambda path: str(path),
    )
    for path in existing:
        if path.name == f"lib{name}.so" or path.name == f"lib{name}.a":
            return str(path)
    for path in existing:
        if re.fullmatch(rf"lib{re.escape(name)}\.so(?:\.\d+)+", path.name):
            return str(path)
    return None


def resolve_system_lib_args(args: list[str]) -> list[str]:
    versioned_fallback_libs = {"zstd"}
    resolved: list[str] = []
    for arg in args:
        if arg.startswith("-l") and len(arg) > 2 and arg[2:] in versioned_fallback_libs:
            library = find_versioned_system_library(arg[2:])
            resolved.append(library or arg)
        else:
            resolved.append(arg)
    return resolved


def cpp_string(text: str) -> str:
    return json.dumps(text)


def helper_source(entries: dict[str, str]) -> str:
    entry_lines = "\n".join(
        f"    {{{cpp_string(base)}, Intrinsic::{enum_name}}},"
        for base, enum_name in sorted(entries.items())
    )
    return f"""#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#if __has_include("llvm/IR/IntrinsicsX86.h")
#include "llvm/IR/IntrinsicsX86.h"
#endif
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

struct Entry {{
  const char *Base;
  Intrinsic::ID ID;
}};

static const Entry Entries[] = {{
{entry_lines}
}};

static std::string jsonEscape(StringRef Text) {{
  std::string Out;
  for (char C : Text) {{
    switch (C) {{
    case '\\\\':
      Out += "\\\\\\\\";
      break;
    case '"':
      Out += "\\\\\\"";
      break;
    case '\\n':
      Out += "\\\\n";
      break;
    case '\\r':
      Out += "\\\\r";
      break;
    case '\\t':
      Out += "\\\\t";
      break;
    default:
      if (static_cast<unsigned char>(C) < 0x20) {{
        char Buffer[8];
        std::snprintf(Buffer, sizeof(Buffer), "\\\\u%04x", C);
        Out += Buffer;
      }} else {{
        Out += C;
      }}
    }}
  }}
  return Out;
}}

static const Entry *findEntry(StringRef FullName) {{
  const Entry *Best = nullptr;
  for (const Entry &Item : Entries) {{
    StringRef Base(Item.Base);
    bool Match = FullName == Base ||
                 (FullName.starts_with(Base) && FullName.size() > Base.size() &&
                  FullName[Base.size()] == '.');
    if (Match && (!Best || Base.size() > StringRef(Best->Base).size()))
      Best = &Item;
  }}
  return Best;
}}

static Type *parseScalarType(LLVMContext &Context, StringRef Token,
                             std::string &Error) {{
  if (Token == "f16")
    return Type::getHalfTy(Context);
  if (Token == "bf16")
    return Type::getBFloatTy(Context);
  if (Token == "f32")
    return Type::getFloatTy(Context);
  if (Token == "f64")
    return Type::getDoubleTy(Context);
  if (Token.starts_with("i")) {{
    unsigned Bits = 0;
    if (!Token.drop_front(1).getAsInteger(10, Bits) && Bits > 0)
      return IntegerType::get(Context, Bits);
  }}
  if (Token.starts_with("p")) {{
    unsigned AddressSpace = 0;
    if (!Token.drop_front(1).getAsInteger(10, AddressSpace))
      return PointerType::get(Context, AddressSpace);
  }}
  Error = "unsupported LLVM intrinsic type token: " + Token.str();
  return nullptr;
}}

static Type *parseTypeToken(LLVMContext &Context, StringRef Token,
                            std::string &Error) {{
  if (Token.size() >= 3 && Token[0] == 'v' &&
      std::isdigit(static_cast<unsigned char>(Token[1]))) {{
    size_t Pos = 1;
    while (Pos < Token.size() &&
           std::isdigit(static_cast<unsigned char>(Token[Pos])))
      ++Pos;
    unsigned Lanes = 0;
    if (Token.slice(1, Pos).getAsInteger(10, Lanes) || Lanes == 0) {{
      Error = "invalid vector lane count in token: " + Token.str();
      return nullptr;
    }}
    Type *Element = parseScalarType(Context, Token.drop_front(Pos), Error);
    if (!Element)
      return nullptr;
    return FixedVectorType::get(Element, Lanes);
  }}
  return parseScalarType(Context, Token, Error);
}}

static void printRecord(StringRef Name, bool Ok, StringRef Declaration,
                        StringRef Error) {{
  outs() << "{{\\"name\\":\\"" << jsonEscape(Name) << "\\",";
  outs() << "\\"ok\\":" << (Ok ? "true" : "false") << ",";
  outs() << "\\"declaration\\":\\"" << jsonEscape(Declaration) << "\\",";
  outs() << "\\"error\\":\\"" << jsonEscape(Error) << "\\"}}\\n";
}}

static int resolveOne(StringRef FullName) {{
  const Entry *Item = findEntry(FullName);
  if (!Item) {{
    printRecord(FullName, false, "", "intrinsic base was not found in TableGen entries");
    return 0;
  }}

  LLVMContext Context;
  Module M("intrinsic_declarations", Context);
  SmallVector<Type *, 8> OverloadTypes;
  StringRef Suffix = FullName.drop_front(StringRef(Item->Base).size());
  if (Suffix.starts_with("."))
    Suffix = Suffix.drop_front(1);
  while (!Suffix.empty()) {{
    StringRef Token;
    std::tie(Token, Suffix) = Suffix.split('.');
    if (Token.empty())
      continue;
    std::string Error;
    Type *Ty = parseTypeToken(Context, Token, Error);
    if (!Ty) {{
      printRecord(FullName, false, "", Error);
      return 0;
    }}
    OverloadTypes.push_back(Ty);
  }}

  Function *F = Intrinsic::getDeclaration(&M, Item->ID, OverloadTypes);
  std::string Declaration;
  raw_string_ostream OS(Declaration);
  F->print(OS);
  OS.flush();
  printRecord(FullName, true, Declaration, "");
  return 0;
}}

int main(int argc, char **argv) {{
  if (argc < 2) {{
    std::cerr << "usage: intrinsic_decl_helper <llvm.intrinsic.name>...\\n";
    return 2;
  }}
  for (int I = 1; I < argc; ++I)
    resolveOne(argv[I]);
  return 0;
}}
"""


def compile_helper(
    *,
    build_dir: Path,
    entries: dict[str, str],
    llvm_config: str,
    cxx: str,
    timeout: int,
) -> Path:
    source = build_dir / "intrinsic_decl_helper.cpp"
    binary = build_dir / ("intrinsic_decl_helper.exe" if os.name == "nt" else "intrinsic_decl_helper")
    source.write_text(helper_source(entries), encoding="utf-8")
    cxxflags = llvm_config_args(llvm_config, ["--cxxflags"])
    ldflags = llvm_config_args(llvm_config, ["--ldflags"])
    libs = llvm_config_args(llvm_config, ["--libs", "core"])
    system_libs = resolve_system_lib_args(llvm_config_args(llvm_config, ["--system-libs"]))
    cmd = [cxx, *cxxflags, str(source), "-o", str(binary), *ldflags, *libs, *system_libs]
    result = run_cmd(cmd, timeout=timeout)
    if not result["ok"]:
        raise SystemExit(
            "failed to compile LLVM intrinsic declaration helper:\n"
            + " ".join(shlex.quote(part) for part in cmd)
            + "\n"
            + str(result["stderr"])
        )
    return binary


def run_helper(binary: Path, names: list[str], timeout: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in names:
        result = run_cmd([str(binary), name], timeout=timeout)
        if not result["ok"]:
            records.append(
                {
                    "name": name,
                    "ok": False,
                    "declaration": "",
                    "error": str(result["stderr"] or result.get("error") or "helper failed"),
                    "returncode": result["returncode"],
                }
            )
            continue
        for line in str(result["stdout"]).splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                row = {"name": name, "ok": False, "declaration": "", "error": str(exc)}
            records.append(row)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve exact LLVM intrinsic declarations from an llvm-project checkout "
            "and a matching LLVM build."
        )
    )
    parser.add_argument("--llvm-project-root", type=Path, required=True)
    parser.add_argument("--llvm-config", type=Path, help="Path to llvm-config from the matching LLVM build.")
    parser.add_argument("--td-file", type=Path, action="append", help="Override TableGen files to scan.")
    parser.add_argument("--name", action="append", help="Full intrinsic name to resolve. Can be repeated.")
    parser.add_argument("--names-file", type=Path, action="append", help="One full intrinsic name per line.")
    parser.add_argument("--recommendations-jsonl", type=Path, help="GNN recommendations JSONL to scan for hints.")
    parser.add_argument("--recommendation-top-k", type=int, default=20)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--build-dir", type=Path, help="Directory for the temporary C++ helper build.")
    parser.add_argument("--keep-build-dir", action="store_true")
    parser.add_argument("--compile-timeout", type=int, default=120)
    parser.add_argument("--resolve-timeout", type=int, default=30)
    args = parser.parse_args()

    if args.recommendation_top_k <= 0:
        parser.error("--recommendation-top-k must be positive")

    names = collect_names(args)
    if not names:
        raise SystemExit("no intrinsic names to resolve")

    td_files = args.td_file or default_td_files(args.llvm_project_root)
    td_entries = parse_td_intrinsics(td_files)
    selected_entries: dict[str, str] = {}
    skipped: list[dict[str, Any]] = []
    selected_names: list[str] = []
    for name in names:
        base = find_base(name, td_entries)
        if base is None:
            skipped.append(
                {
                    "name": name,
                    "ok": False,
                    "declaration": "",
                    "error": "intrinsic base was not found in scanned TableGen files",
                }
            )
            continue
        selected_entries[base[0]] = base[1]
        selected_names.append(name)

    records: list[dict[str, Any]] = [*skipped]
    llvm_config = find_llvm_config(args)
    build_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.build_dir:
        build_dir = args.build_dir
        build_dir.mkdir(parents=True, exist_ok=True)
    else:
        build_owner = tempfile.TemporaryDirectory(prefix="llvm-intrinsic-decls-")
        build_dir = Path(build_owner.name)

    try:
        if selected_names:
            binary = compile_helper(
                build_dir=build_dir,
                entries=selected_entries,
                llvm_config=llvm_config,
                cxx=args.cxx,
                timeout=args.compile_timeout,
            )
            records.extend(run_helper(binary, selected_names, args.resolve_timeout))
    finally:
        if build_owner is not None and not args.keep_build_dir:
            build_owner.cleanup()

    declarations = {
        str(record["name"]): str(record["declaration"]).strip()
        for record in records
        if record.get("ok") and str(record.get("declaration", "")).strip()
    }
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "llvm_project_root": str(args.llvm_project_root),
        "llvm_config": llvm_config,
        "td_files": [str(path) for path in td_files],
        "requested": names,
        "resolved": len(declarations),
        "failed": len([record for record in records if not record.get("ok")]),
        "declarations": declarations,
        "records": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "resolved": len(declarations), "failed": output["failed"]}))
    return 0 if declarations else 1


if __name__ == "__main__":
    raise SystemExit(main())
