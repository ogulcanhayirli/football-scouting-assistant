"""The chat Lambda's code folder, built before ``cdk synth`` (prompt 20).

``build()`` makes ``infra/build/chat/``: the pinned dependencies of
``infra/chat/requirements.txt`` installed for Python 3.12 on arm64 (wheels
only, so no Docker is needed), then ``scout/__init__.py`` and
``scout/chat/*.py`` from ``src/``, nothing else of the package (the rest
needs pandas). The dependencies are installed again only when the
requirements change; the code is copied fresh every time.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

INFRA = Path(__file__).resolve().parents[1]
ROOT = INFRA.parent
REQUIREMENTS = INFRA / "chat" / "requirements.txt"
OUT = INFRA / "build" / "chat"
PLATFORM = "aarch64-manylinux_2_28"
PYTHON = "3.12"


def copy_code(out: Path, src: Path = ROOT / "src") -> list[Path]:
    """The chat package and the package's ``__init__.py``, into ``out/scout``."""
    target = Path(out) / "scout"
    shutil.rmtree(target, ignore_errors=True)
    (target / "chat").mkdir(parents=True)
    copied = [Path(shutil.copy2(src / "scout" / "__init__.py", target / "__init__.py"))]
    for path in sorted((src / "scout" / "chat").glob("*.py")):
        copied.append(Path(shutil.copy2(path, target / "chat" / path.name)))
    return copied


def install_dependencies(out: Path, requirements: Path = REQUIREMENTS) -> None:
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()[:16]
    marker = Path(out) / f".deps-{digest}"
    if marker.exists():
        return
    shutil.rmtree(out, ignore_errors=True)
    Path(out).mkdir(parents=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            str(out),
            "--python-version",
            PYTHON,
            "--python-platform",
            PLATFORM,
            "--only-binary",
            ":all:",
            "--no-deps",  # the file pins every package
            "-r",
            str(requirements),
        ],
        check=True,
    )
    shutil.rmtree(Path(out) / "bin", ignore_errors=True)  # console scripts, not needed
    marker.touch()


def build(out: Path = OUT) -> Path:
    install_dependencies(out)
    copy_code(out)
    return out
