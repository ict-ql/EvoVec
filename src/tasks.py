from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SOURCE_EXTENSIONS = {
    ".c": "c",
    ".cc": "c++",
    ".cpp": "c++",
    ".cxx": "c++",
    ".f": "fortran",
    ".for": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".ll": "llvm-ir",
}


@dataclass(frozen=True)
class CompileTask:
    suite: str
    task_id: str
    path: Path
    language: str
    granularity: str

    def to_json(self, root: Path) -> dict[str, str]:
        try:
            rel_path = self.path.relative_to(root)
        except ValueError:
            rel_path = self.path
        return {
            "suite": self.suite,
            "task_id": self.task_id,
            "path": str(rel_path),
            "language": self.language,
            "granularity": self.granularity,
        }


def detect_language(path: Path) -> str | None:
    return SOURCE_EXTENSIONS.get(path.suffix.lower())
