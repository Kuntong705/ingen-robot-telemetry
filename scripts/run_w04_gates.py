"""Run the Week 4 quality gates and record what actually happened.

Writes ``reports/w04/quality_gates.json``. A gate counts as passed only when its
command ran and exited 0. Gates that need Docker are skipped (and recorded as
skipped, not passed) when ``--no-docker`` is given or Docker is unavailable.

    python scripts/run_w04_gates.py                  # all gates (Docker ones take ~10-20 min)
    python scripts/run_w04_gates.py --no-docker
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
UV = ["uv"] if shutil.which("uv") else ["py", "-3.11", "-m", "uv"]
GITLEAKS_IMAGE = "zricethezav/gitleaks:v8.30.0"


def last_lines(text: str, n: int = 3) -> str:
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", ln).strip() for ln in text.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:])


def run(name: str, cmd: list[str], timeout: int = 3600, cwd: Path = ROOT) -> dict[str, object]:
    started = datetime.now(UTC)
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False,
                             encoding="utf-8", errors="replace")
        out = {"ok": res.returncode == 0, "exit_code": res.returncode, "summary": last_lines(res.stdout + "\n" + res.stderr)}
    except (OSError, subprocess.TimeoutExpired) as exc:
        out = {"ok": False, "exit_code": None, "summary": f"{type(exc).__name__}: {exc}"}
    # record a portable command line: no interpreter path or absolute repository path (no user names)
    out["command"] = " ".join(cmd).replace(PY, "python").replace(ROOT.as_posix(), ".").replace(str(ROOT), ".")
    out["seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    print(f"{name}: {'PASS' if out['ok'] else 'FAIL'} ({out['seconds']} s) {out['summary'][:200]}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-docker", action="store_true")
    parser.add_argument("--only", nargs="*", help="re-run only these gates; the other recorded results are kept")
    args = parser.parse_args()
    gates: dict[str, object] = {"run_at_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    gates["git_head"] = head.stdout.strip()
    # the gates test the working tree; they describe git_head only when the tree is clean
    gates["git_worktree_clean"] = dirty.stdout.strip() == ""
    if not gates["git_worktree_clean"]:
        print("WARNING: uncommitted changes; this record does not describe a commit", flush=True)
    specs: dict[str, tuple[str, list[str], int]] = {
        "ruff": ("ruff", [PY, "-m", "ruff", "check", "."], 3600),
        "mypy": ("mypy", [PY, "-m", "mypy", "src/", "fleet_simulator.py"], 3600),
        "pytest": ("pytest", [PY, "-m", "pytest", "-q", "-p", "no:warnings", "--cov=ingen_fleet_health",
                              "--cov-report=term"], 5400),
        "uv_lock_check": ("uv lock --check", [*UV, "lock", "--check"], 3600),
    }
    docker = shutil.which("docker") is not None and not args.no_docker
    if docker:
        specs["docker_build"] = ("docker compose build", ["docker", "compose", "build"], 5400)
        specs["docker_app_smoke"] = ("app smoke", ["docker", "run", "--rm", "ingen-fleet-health:app", "python", "-m",
                                                   "ingen_fleet_health.pipeline", "--help"], 3600)
        specs["airflow_dag_import"] = ("airflow DAG import", [
            "docker", "run", "--rm", "-e", "FLEET_PIPELINE_CONFIG=/opt/fleet/configs/pipeline.yaml",
            "ingen-fleet-health:airflow", "bash", "-c",
            ("airflow db migrate >/dev/null 2>&1; airflow dags reserialize >/dev/null 2>&1; "
            "airflow dags list-import-errors && airflow dags list | grep fleet_health_daily")], 3600)
        specs["gitleaks"] = ("gitleaks", ["docker", "run", "--rm", "-v", f"{ROOT.as_posix()}:/repo", GITLEAKS_IMAGE,
                                          "git", "/repo", "--redact", "--exit-code", "1"], 3600)
    out = ROOT / "reports" / "w04" / "quality_gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if args.only:
        if existing.get("git_head") != gates["git_head"]:
            raise SystemExit("--only would mix results from two commits; run all gates instead")
        gates = {**existing, **gates}
    for key, (name, cmd, timeout) in specs.items():
        if not args.only or key in args.only:
            gates[key] = run(name, cmd, timeout=timeout)
    if not docker:
        for g in ("docker_build", "docker_app_smoke", "airflow_dag_import", "gitleaks"):
            gates.setdefault(g, {"ok": None, "summary": "skipped (no Docker)"})
    out.write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
