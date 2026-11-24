# W04 — Final readout live demo (about 8 minutes of the 30–40 minute readout)

All robots, faults and numbers are **synthetic**. Every number named below is read from the screen or from `reports/w04/*.json`; none needs to be memorised. The deck is `docs/W04_Capstone_Deck.pdf`.

## Before the call (about 30 minutes, once)

```bash
docker compose build
docker compose up -d mlflow
docker compose run --rm trainer          # small fleet: simulate -> train + register v1 -> publish -> backfill
docker compose up -d airflow dashboard
```

1. Open http://localhost:8501 (dashboard) and http://localhost:5000 (MLflow).
2. Open http://localhost:8080 (Airflow). The admin password is printed in `docker compose logs airflow` (look for "Password for user").
3. Unpause `fleet_health_daily` and keep the browser tabs open in this order: dashboard, Airflow, MLflow.

## Live sequence

1. **Dashboard, "Pull list" (about 2 min).**
   - Show the count of robots to pull and read one robot's explanation aloud.
   - Point at the main driver and the battery health line.
   - Then show the risk ranking and the "No recent data" message (if any): missing data is shown as missing.
2. **Unit drill-down (about 1 min).**
   - Pick the top robot. Show the risk crossing the alert level, the alert marker, and the telemetry behind it, such as vibration or motor temperature.
3. **Alert queue (about 1 min).**
   - Open one alert and show the driver table and the automatic evidence check ("n statements checked, all supported").
   - Say why free-text LLM summaries are not used: 0/20 faithful in Week 3.
4. **Airflow (about 2 min).**
   - Show the DAG graph: two platform groups (wait → ingest → features → score → alerts → drift), then the weekly performance check.
   - Trigger a run, or show the last run, and show the task logs.
   - Re-run proof from a terminal:
     `docker compose exec airflow bash -c "sha256sum /data/quickstart/w04/predictions/platform=fari/date=<D>/predictions.parquet"` before and after re-running that date. The hashes are identical.
5. **Missing partition (about 1 min).**
   - Explain the policy (ADR-0006): wait up to 6 h, then fail loudly; the other platform still runs; nothing is written for the missing date.
   - Show it in `docker compose run --rm trainer python -m ingen_fleet_health.pipeline status`, or show the dashboard warning banner.
6. **MLflow (about 30 s).**
   - Show the registered models `fleet_early_warning` v1 and `fleet_soh_forecast` v1 with their run parameters. The pipeline pins these versions in `configs/pipeline.yaml`.

## Questions to be ready for (answers are in the documents)

| Question | Where the answer is |
|---|---|
| Why is the label "qualifying onset within 72 h"? | `docs/adr/0003-failure-label-definition.md` (rev1) |
| How do you know there is no leakage? | ADR-0004, leakage demo in `docs/W02_Evaluation_Protocol.docx` |
| Why not deploy the Isolation Forest, which ranks better on Rover? | ADR-0005, model card §4 |
| Does text help? | `docs/W03_Fusion_Results.docx`: no detectable gain at the assumed note rate |
| Where do the cost numbers come from? | `configs/cost_assumptions.yaml` (placeholders); model card §8: the break-even ratio uses measured counts only |
| What happens when data is late? | ADR-0006 |
