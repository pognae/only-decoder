"""Compatibility shim for running modules without setting PYTHONPATH=src.

This package extends its module search path to include ``src/sllm`` so that
commands like ``uvicorn sllm.api.server:app`` work from the repository root.
"""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_repo_root = Path(__file__).resolve().parent.parent
_src_pkg = _repo_root / "src" / "sllm"
if _src_pkg.is_dir():
    __path__.append(str(_src_pkg))  # type: ignore[attr-defined]
