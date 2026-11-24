# ADR-002 — Telemetry Table Layout

**Decision:** Adopt a wide telemetry table with one row per unit per timestamp.

**Rationale:** Related sensor measurements are available in the same record, simplifying cross-channel comparisons, temporal aggregation, and future feature engineering.

**Alternative considered:** Long-format event/value tables.

**Trade-off:** The wide layout is convenient for the relatively fixed Rover schema but less flexible for heterogeneous high-dimensional robotic platforms.

Platform-specific extension schemas may be needed for future fleet integration.

## Revision history

- Original: Week 1, `docs/W01_Data_Contract.md` §8 (text above copied verbatim; the data contract keeps its copy).
- Week 4 addendum (2026-11-24): the wide layout is kept; the daily pipeline partitions it by `platform=<p>/date=<YYYY-MM-DD>` (ADR-0006), which serves the 24 h / 72 h rolling windows by reading whole days.
