# ADR-001 — Analytical Storage

**Decision:** Use DuckDB with local Parquet files.

**Rationale:** Parquet supports compressed columnar storage, while DuckDB enables local SQL analysis without requiring a cloud database.

**Alternative considered:** BigQuery or another managed cloud analytical platform.

**Trade-off:** Local analysis is inexpensive and reproducible, but production-scale orchestration, multiuser access, and centralized governance would require additional infrastructure.

## Revision history

- Original: Week 1, `docs/W01_Data_Contract.md` §7 (text above copied verbatim; the data contract keeps its copy).
- None. Still accepted in Week 4: the daily pipeline writes one Parquet partition per (table, platform, date) and the dashboard reads them with pandas/DuckDB; no BigQuery sandbox or other paid service is used (ADR-0006).
