"""Platform profiles for the telemetry data contract (Week 3, Fari gap analysis).

Week 1 hard-coded three Aido Rover facts into ingestion: the ``mode`` enum
(``patrol`` / ``dock``), a 30-60 V pack window and ``power = 48 V x current``.
A platform profile makes those values data instead of code, and adds the
privacy rule Plan v3.0 asks for: on Fari no imagery or audio field may enter
analytics, which is *enforced* here (the file is refused), not just stated.

The profile is derived from the platform's simulator YAML, so there is one
source for each platform's numbers.
"""

from __future__ import annotations

from typing import Any

# Week 1 behaviour, unchanged: used whenever no profile is passed.
DEFAULT_PROFILE: dict[str, Any] = {
    "platform": "aido_rover",
    "allowed_modes": ("patrol", "dock"),
    "nominal_voltage_v": 48.0,
    "voltage_range_v": (30.0, 60.0),
    "forbidden_field_patterns": (),
    "not_applicable_fields": (),
}


class ContractViolation(ValueError):
    """The input breaks a schema-level rule; nothing from it may be stored."""


def platform_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Build the ingestion profile from a platform YAML (``configs/*.yaml``)."""
    modes = config.get("modes", {})
    contract = config.get("contract", {})
    lo, hi = contract.get("voltage_range_v", DEFAULT_PROFILE["voltage_range_v"])
    return {
        "platform": str(config.get("platform", DEFAULT_PROFILE["platform"])),
        "allowed_modes": (str(modes.get("active", "patrol")), str(modes.get("docked", "dock"))),
        "nominal_voltage_v": float(config["battery"]["nominal_voltage_v"]),
        "voltage_range_v": (float(lo), float(hi)),
        "forbidden_field_patterns": tuple(str(p).lower() for p in contract.get("forbidden_field_patterns", ())),
        "not_applicable_fields": tuple(str(f) for f in contract.get("not_applicable_fields", ())),
    }


def forbidden_fields(columns: list[str], profile: dict[str, Any]) -> list[str]:
    """Column names that match a forbidden pattern (case-insensitive substring)."""
    patterns = profile.get("forbidden_field_patterns", ())
    return sorted(c for c in columns if any(p in c.lower() for p in patterns))


def assert_no_forbidden_fields(columns: list[str], profile: dict[str, Any]) -> None:
    bad = forbidden_fields(columns, profile)
    if bad:
        raise ContractViolation(
            f"{profile.get('platform')}: forbidden field(s) {bad} - imagery/audio data must not enter analytics; "
            "the file was refused before any copy was stored"
        )
