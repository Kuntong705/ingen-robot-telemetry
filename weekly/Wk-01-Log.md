# Week 01 — Engineering Log

**Project:** Robot Fleet Telemetry Analytics & Predictive Health  
**Period:** October 26 – November 1, 2026  
**Status:** Engineering prototype and data foundation completed; further validation pending

## 1. Work Completed

I established the Python 3.11 development environment, GitHub repository, and automated testing workflow. I first developed a 24-hour single-rover simulator, then extended it to a configurable fleet. The full-scale run generated 1,728,000 minute-level records across 20 simulated robots and 60 days.

The simulator models battery state of charge and health, internal resistance, terrain-dependent power consumption, motor temperature, and vibration. It includes synthetic bearing-wear and cooling-degradation scenarios.

I implemented a Raw-to-Validated-to-Curated pipeline using DuckDB and Parquet, with invalid-record quarantine, daily summaries, and automated quality reporting. The clean full-scale dataset passed validation.

A separate seeded impairment tool introduces network dropouts and gradual temperature-sensor drift. In one experiment, 8,486 records were removed, and the pipeline identified exactly 8,486 missing minutes.

All 30 automated pytest tests passed, and the GitHub Actions workflow successfully completed Ruff, mypy, and pytest checks. I also added a cumulative odometer to the fleet telemetry schema, implemented non-decreasing odometer validation, and verified that records with decreasing odometer values are quarantined.

## 2. Failure-Label Decision

I proposed defining the prediction target as **a new qualifying fault onset within the next 72 hours**.

I considered two alternatives: detecting an abnormal condition at the present moment, and predicting the eventual failure time. Present-state detection provides little advance warning, while failure-time labeling depends on a consistent definition of operational failure.

The 72-hour onset-based target is intended to give fleet operators time to inspect or replace a robot before its next scheduled patrol.

I also proposed excluding the first 24 hours following maintenance to avoid misleading post-repair signals. Features must use information available at prediction time, and incomplete future windows should be censored rather than treated as negative examples.

These choices remain proposed design decisions. Prediction-label generation, maintenance exclusions, and operational suitability still require implementation and review.

## 3. Architecture Decisions

I selected DuckDB over local Parquet for reproducible, low-cost analysis and a wide telemetry table to simplify rolling-window feature engineering.

Ground-truth fault labels are maintained separately from prediction-time telemetry to reduce leakage risk.

## 4. Challenges and Limitations

All physical parameters are illustrative rather than calibrated against real robot hardware. The current simulator simplifies battery degradation, terrain effects, and fault progression.

Although sensor drift can be injected, the validation pipeline does not yet identify drift that remains within plausible physical ranges.

Additional work is needed on simulator parameter documentation, physical-model fidelity, schema coverage, and reproducible project packaging.

## 5. Next Steps

I will finalize the five-platform data brief, document simulation assumptions, strengthen validation rules, and review the proposed failure-label definition during the November 2 meeting.