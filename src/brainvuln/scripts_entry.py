"""Console-script entrypoints (pyproject [project.scripts]).

Thin delegators so the packaged console commands stay import-light and the
argument parsing lives in one place under ``scripts/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _run(name: str) -> int:
    import runpy

    target = _SCRIPTS_DIR / name
    if not target.exists():  # pragma: no cover - only when installed without scripts/
        print(f"Script {name} not found at {target}; run from the repository checkout.",
              file=sys.stderr)
        return 1
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
    return 0


def fetch_gene_sets_main() -> int:
    return _run("fetch_gene_sets.py")


def build_expression_main() -> int:
    return _run("build_expression.py")


def run_pipeline_main() -> int:
    return _run("run_pipeline.py")


def make_score_map_main() -> int:
    return _run("make_score_map.py")
