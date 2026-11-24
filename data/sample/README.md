# data/sample — small committed sample (SYNTHETIC)

Everything in this folder is **simulated**. It is not InGen telemetry, not customer data and not personal data.

| Platform | Robots | Days | Seed | Simulator | Config |
|---|---|---|---|---|---|
| Aido Rover | 2 | 1 (2026-10-26, simulated date) | 7 | v0.2 | `configs/aido_rover_v0_2.yaml` |
| Fari (synthetic profile) | 2 | 1 (2026-10-26, simulated date) | 7 | v0.2 | `configs/fari.yaml` |

The sample shows the file formats the pipeline reads, in the same layout as the full study data:

- `generated/fleet_telemetry.csv`: 1-minute telemetry, the input of the Week 1 ingestion (`ingest_pipeline.py`).
- `events/`: robot register and maintenance events.
- `ground_truth/`: fault events, battery truth, impairments and unit parameters. These are simulator truth and are never model inputs.
- `simulation_manifest.json`: simulator version, config hash, seed and row counts.

It is too small to train or evaluate anything, and it contains no fault onset. All other data is git-ignored and is regenerated from the configs (see the README sections "Quickstart" and "Regenerate the full study data and train").

Regenerate it (the tests check that the committed files match a fresh run):

```bash
python scripts/make_data_sample.py
```
