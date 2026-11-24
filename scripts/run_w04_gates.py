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
    out["command"] = " ".join(cmd)
    out["seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    print(f"{name}: {'PASS' if out['ok'] else 'FAIL'} ({out['seconds']} s) {out['summary'][:200]}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-docker", action="store_true")
    args = parser.parse_args()
    gates: dict[str, object] = {"run_at_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    gates["git_head"] = head.stdout.strip()
    gates["ruff"] = run("ruff", [PY, "-m", "ruff", "check", "."])
    gates["mypy"] = run("mypy", [PY, "-m", "mypy", "src/", "fleet_simulator.py"])
    gates["pytest"] = run("pytest", [PY, "-m", "pytest", "-q", "-p", "no:warnings", "--cov=ingen_fleet_health",
                                     "--cov-report=term"], timeout=5400)
    gates["uv_lock_check"] = run("uv lock --check", [*UV, "lock", "--check"])
    docker = shutil.which("docker") is not None and not args.no_docker
    if docker:
        gates["docker_build"] = run("docker compose build", ["docker", "compose", "build"], timeout=5400)
        gates["docker_app_smoke"] = run("app smoke", ["docker", "run", "--rm", "ingen-fleet-health:app", "python", "-m",
                                                      "ingen_fleet_health.pipeline", "--help"])
        gates["airflow_dag_import"] = run("airflow DAG import", [
            "docker", "run", "--rm", "-e", "FLEET_PIPELINE_CONFIG=/opt/fleet/configs/pipeline.yaml",
            "ingen-fleet-health:airflow", "bash", "-c",
            "airflow db migrate >/dev/null 2>&1; airflow dags list-import-errors && airflow dags list | grep fleet_health_daily"])
        gates["gitleaks"] = run("gitleaks", ["docker", "run", "--rm", "-v", f"{ROOT.as_posix()}:/repo", GITLEAKS_IMAGE,
                                             "git", "/repo", "--redact", "--exit-code", "1"])
    else:
        for g in ("docker_build", "docker_app_smoke", "airflow_dag_import", "gitleaks"):
            gates[g] = {"ok": None, "summary": "skipped (no Docker)"}
    out = ROOT / "reports" / "w04" / "quality_gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    keep = {k: v for k, v in existing.items() if k.startswith("manual_")}   # results of the Docker/Airflow runbook
    out.write_text(json.dumps({**gates, **keep}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
