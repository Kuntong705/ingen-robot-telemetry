# Week 01 — Engineering Log

**Project:** Robot Fleet Telemetry Analytics & Predictive Health  
**Period:** October 26 – November 1, 2026  
**Status:** Initial prototype and engineering foundation completed

## 1. Work Completed

Following the October 26 onboarding, I reviewed the internship requirements and available reference materials, then established the development environment using Python 3.11, PyCharm, GitHub Desktop, and a private GitHub repository.

I implemented an initial synthetic telemetry simulator for an outdoor patrol robot. The first version generated 1,440 minute-level observations over 24 hours, including battery state of charge, electrical power, ambient temperature, motor temperature, and vibration measurements.

I then extended the simulator to support configurable fleet sizes and simulation durations. A three-robot, two-day test generated 8,640 telemetry records. The extended model includes simplified battery health degradation, internal resistance growth, terrain-dependent loads, and two synthetic fault scenarios: bearing wear and cooling degradation.

The fleet simulation was subsequently scaled to 20 robots over 60 days, generating 1,728,000 records. I also implemented a Raw-to-Validated-to-Curated Parquet pipeline and tested synthetic network dropouts and gradual temperature-sensor drift. The dropout experiment removed 8,486 records, all of which were accounted for in the pipeline's missing-minute report.

## 2. Technical Decisions

I drafted the Week 1 Data Contract defining the initial telemetry schema, validation rules, supporting event tables, and failure-labeling strategy.

Three architectural decisions were proposed:

- Use Parquet and DuckDB for efficient local analytical storage.
- Adopt a wide telemetry table with one row per robot per timestamp.
- Define a future 72-hour fault prediction target while separating ground-truth fault events from prediction-time features.

These are proposed design decisions; the complete storage pipeline and prediction-label generation have not yet been implemented.

## 3. Validation and Testing

I developed an independent CSV validation module to check required fields, timestamp ordering, numeric ranges, duplicate timestamps, and sampling gaps.

I also added corrupted-record tests to confirm that invalid battery values, malformed numerical fields, missing columns, and duplicate timestamps are detected.

Currently, 26 automated pytest tests pass. GitHub Actions CI has been configured to execute Ruff linting, mypy type checking, and pytest. The latest workflow completed successfully after resolving initial linting issues.

## 4. Challenges and Limitations

The main challenge is balancing physical plausibility with the limited availability of approved operational data. All simulation parameters are illustrative rather than calibrated against real hardware.

The validation module currently reports invalid records but does not yet implement a complete raw-to-validated-to-curated ingestion pipeline.

The current fleet simulation also lacks several planned mechanisms, including communication dropouts and sensor drift.

## 5. Next Steps

My priorities are to expand simulation coverage, strengthen data-quality reporting, implement Parquet ingestion, and complete the fleet data brief.

I will also review the proposed architecture and outstanding technical requirements during the November 2 progress meeting.

