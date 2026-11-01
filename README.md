# ingen-robot-telemetry
Robotics telemetry and predictive health data engineering

## Week 1 Prototype — Rover Telemetry Simulator

This project explores synthetic robot fleet telemetry and predictive maintenance.

### Current Implementation

- One simulated patrol robot over 24 hours
- One-minute telemetry sampling (1,440 records)
- Simplified battery state-of-charge model
- Motor temperature and vibration simulation
- Synthetic bearing-wear fault injection
- Basic data quality checks
- Six automated pytest tests

All generated data is synthetic. Model parameters are illustrative assumptions, not verified InGen hardware specifications.

### Run the Simulator

From the project root:

```powershell
.\.venv\Scripts\python.exe simulate_rover.py
```

Generated outputs:

- `data/generated/telemetry.csv`
- `data/generated/fault_events.csv`
- `data/generated/quality_report.json`

### Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current result: 6 passed.

### Next Steps

- Extend simulation to multiple robots
- Improve physics-based battery and thermal models
- Implement independent schema and physical-range validation
- Add data ingestion and Parquet storage
- Add CI with GitHub Actions

