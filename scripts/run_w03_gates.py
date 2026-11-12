"""Run the repository quality gates and record their outcome in reports/w03/quality_gates.json.

    uv run python scripts/run_w03_gates.py

Same commands as CI (.github/workflows): ruff, mypy, pytest, plus ``uv lock --check``.
The Week 3 report reads the recorded summary lines instead of hand-typed counts.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = {
    "ruff": [sys.executable, "-m", "ruff", "check", "."],
    "mypy": [sys.executable, "-m", "mypy", "src/", "fleet_simulator.py"],
    "pytest": [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings"],
    "pytest_w03": [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "tests/w03"],
    # uv is on PATH in CI; on the Windows dev machine it is run as ``py -3.11 -m uv``
    "uv_lock_check": (["uv"] if shutil.which("uv") else ["py", "-3.11", "-m", "uv"]) + ["lock", "--check"],
}


def last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def main() -> None:
    out: dict[str, object] = {"run_at_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=False, cwd=ROOT, capture_output=True, text=True)
    out["git_head"] = head.stdout.strip()
    for name, cmd in GATES.items():
        try:
            res = subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True, text=True, timeout=1800)
            summary = last_line(res.stdout + "\n" + res.stderr)
            out[name] = {"ok": res.returncode == 0, "summary": re.sub(r"\x1b\[[0-9;]*m", "", summary)}
        except (OSError, subprocess.TimeoutExpired) as exc:
            out[name] = {"ok": False, "summary": f"{type(exc).__name__}: {exc}"}
        print(name, out[name])
    path = ROOT / "reports" / "w03" / "quality_gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
