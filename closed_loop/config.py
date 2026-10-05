"""Validated access to the repository's authoritative study parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARAMETERS_PATH = REPOSITORY_ROOT / "config" / "parameters.json"


def load_parameters(path: str | Path = PARAMETERS_PATH) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read study parameters: {source}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("study parameters must be a JSON object")
    if value.get("schema_version") != 5:
        raise RuntimeError("unsupported study-parameter schema")

    execution = value.get("execution")
    profiles = value.get("profiles")
    generation = value.get("mechanistic_generation")
    engineering = value.get("engineering")
    reporting = value.get("reporting")
    if not all(isinstance(section, dict) for section in (
        execution, profiles, generation, engineering, reporting,
    )):
        raise RuntimeError("study parameters omit a required section")

    default_profile = execution.get("default_profile")
    if default_profile != "article_full_10000" or default_profile not in profiles:
        raise RuntimeError("the default profile must be article_full_10000")
    profile = profiles[default_profile]
    if not isinstance(profile, dict):
        raise RuntimeError("the default profile must be a JSON object")
    if (profile.get("development_count"), profile.get("test_count")) != (8_000, 2_000):
        raise RuntimeError("the article profile must contain 8000/2000 LHS candidates")
    if profile.get("counts_are_candidate_rows") is not True:
        raise RuntimeError("article profile counts must denote attempted LHS candidates")
    if profile.get("replace_rejected_mechanistic_candidates") is not False:
        raise RuntimeError("the article profile must not replace rejected candidates")
    if generation.get("balance_tolerance") != 1.0e-6:
        raise RuntimeError("physical mass-conservation tolerance must be 1e-6")

    prohibited = {
        "srt_min_d", "srt_max_d", "sor_max_m_d", "sor_report_only_max_m_d",
        "slr_max_kg_m2_d",
    }
    present = prohibited.intersection(engineering)
    if present:
        raise RuntimeError(f"prohibited engineering guardrails remain: {sorted(present)}")
    if reporting.get("timing_protocol") != "primary_route_time_v1":
        raise RuntimeError("only the primary route Time protocol is supported")
    if reporting.get("timing_metric") != "Time" or reporting.get("timing_unit") != "s":
        raise RuntimeError("the timing metric must be Time in seconds")
    return value


def article_profile_parameters() -> dict[str, Any]:
    parameters = load_parameters()
    profile_name = parameters["execution"]["default_profile"]
    return dict(parameters["profiles"][profile_name])


def physical_balance_tolerance() -> float:
    return float(load_parameters()["mechanistic_generation"]["balance_tolerance"])
    
def engineering_parameters() -> dict[str, Any]:
    return dict(load_parameters()["engineering"])