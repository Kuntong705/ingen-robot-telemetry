# Contributing

## Development setup

```bash
py -3.11 -m uv sync --locked          # Windows; on Linux/macOS: uv sync --locked
py -3.11 -m uv run ruff check .
py -3.11 -m uv run mypy src/ fleet_simulator.py
py -3.11 -m uv run python -m pytest -q
```

Python 3.11 only. Dependencies are managed with uv and pinned in `uv.lock`. Change them only with `uv add` or `uv remove`, and commit the lockfile.

## Rules that keep results trustworthy

1. **No leakage.** Features use only data at or before the prediction time. Test robots never enter training. Every metric is reported next to a naive baseline, from the unit-grouped, embargoed protocol (ADR-0004).
2. **No silent changes to reported results.** A change that moves a reported number regenerates the report from its JSON and says why in the commit message.
3. **Synthetic data stays labelled synthetic.** Any value that is not measured (costs, hardware parameters) is marked as an assumption in its config file.
4. **No secrets, no real data, no confidential material.** CI runs gitleaks on every push. Keys go in environment variables (`.env.example`). Generated data, `mlruns/` and `mlflow.db` are git-ignored.
5. **Documented data behaviour.** A new pipeline step states what it does when its input is late, missing, duplicated or implausible (ADR-0006), with a test.

## Pull requests

- Keep each pull request small enough to review in one sitting.
- Describe what changed and why.
- Leave CI green: ruff, mypy, pytest with coverage, gitleaks, and the Docker build.
- Record architectural decisions as a new ADR in `docs/adr/`. Revise an existing ADR with a dated "revision" note; never by rewriting it.
