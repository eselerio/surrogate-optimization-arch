"""Manuscript-v3 result aggregation and failure accounting.

The optimization phase writes one case at a time and can be interrupted between
atomic JSON checkpoints.  Reporting therefore treats every artifact as
optional: an absent case is represented explicitly, a partial case remains
``not_yet_adjudicated``, and no missing value is promoted to a successful
result.  The public :func:`build_reporting_tables` function is read-only.

Mass-conservation and non-negativity records are gathered from both the
untouched-test assessment and selected optimization decisions.  If a selected
response has been saved before its audit CSV, the audit is reconstructed with
the frozen development row scales used by the projection.

Generation reporting distinguishes all attempted candidates from accepted
development/test rows.  It preserves rejection causes, accepted-slot
provenance, migration evidence, and accepted-coordinate coverage.  Effective
accepted inputs and targets are preferred over the immutable round-0 legacy
artifacts when replacement generation has been activated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

from .manuscript_v3 import DECISION_LOWER, DECISION_UPPER, violation_record
from .model import (
    COMPOSITE_MATRIX,
    COMPONENTS,
    INVARIANT_MATRIX,
    INFLUENT_LOWER,
    INFLUENT_UPPER,
    N_COMPONENTS,
    N_STAGES,
    NOMINAL_INFLUENT,
    TSS_VECTOR,
)
from .projection import NetworkLayout, fit_network_row_scales


CONTROL_NAMES: tuple[str, ...] = (
    "H", "a_3", "a_4", "a_5", "r_I", "r_R", "w",
)
OBJECTIVE_COMPONENT_NAMES: tuple[str, ...] = (
    "quality", "hrt", "aeration", "internal_recycle", "return_sludge", "wasting",
)
ENGINEERING_QUANTITY_NAMES: tuple[str, ...] = (
    "solids_inventory",
    "external_solids_loss",
    "srt_d",
    "sor_m_d",
    "slr_kg_m2_d",
    "underflow_tss_g_m3",
    "feed_tss_g_m3",
    "effluent_cod",
    "effluent_tn",
    "effluent_tp",
    "effluent_tss",
)
TRUST_DIAGNOSTICS: tuple[str, ...] = (
    "correction", "regularized_leverage", "particulate_split",
    "reactor_residual", "clarifier_flux",
)
FAILURE_CLASSES: tuple[str, ...] = (
    "no accepted optimization start",
    "projection failure",
    "reference integration failure",
    "residual or reduced-stability failure",
    "engineering infeasibility",
    "smooth-branch disagreement",
    "upper-stationarity failure",
    "validated result",
)
PENDING_CLASS = "not yet adjudicated"
REQUIRED_PHYSICAL_METHODS: tuple[str, ...] = (
    "raw", "projected", "mechanistic", "smooth", "reference",
)
EXPECTED_STARTS = 1
LEGACY_EXPECTED_STARTS = 9
GENERATION_BLOCKS: tuple[str, ...] = ("development", "test")
GENERATION_REJECTION_FLAGS: tuple[tuple[str, str], ...] = (
    ("solver_exception", "rejected_solver_exception"),
    ("mass_or_residual", "rejected_mass_or_residual"),
    ("stability", "rejected_stability"),
    ("nonnegativity", "rejected_nonnegativity"),
    ("domain", "rejected_domain"),
    ("root_distance", "rejected_root_distance"),
    ("branch_disagreement", "rejected_branch_disagreement"),
    ("other_solver_rejection", "rejected_other_solver_rejection"),
)


@dataclass(frozen=True)
class StudyGeometry:
    """Geometry and frozen objective scales needed only for reporting."""

    layer_count: int
    layer_volume_m3: float
    fresh_flow_m3_d: float = 10_000.0
    clarifier_area_m2: float = 1_500.0
    underflow_tss_reference_g_m3: float = 15_000.0

    @property
    def response_count(self) -> int:
        return (N_STAGES + 3) * N_COMPONENTS + self.layer_count


@dataclass(frozen=True)
class RouteSnapshot:
    """Normalized view of one route, including incomplete checkpoints."""

    case: str
    route: str
    artifact_state: str
    outcome: str
    payload: Mapping[str, Any] | None
    starts: tuple[Mapping[str, Any], ...]
    selected_start: int | None
    selected: Mapping[str, Any] | None
    selected_arrays: Mapping[str, np.ndarray]
    equivalence: Mapping[str, Any] | None
    reference_arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ReportingBundle:
    """All deterministic manuscript tables from one filesystem snapshot."""

    run_directory: Path
    expected_cases: tuple[str, ...]
    tables: Mapping[str, pd.DataFrame]
    warnings: tuple[str, ...]

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def write(self, output_directory: str | Path | None = None) -> dict[str, Path]:
        """Write every table and a manifest atomically as CSV/JSON files."""

        destination = (
            Path(output_directory)
            if output_directory is not None
            else self.run_directory / "tables" / "manuscript_v3"
        )
        destination.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name, frame in self.tables.items():
            target = destination / f"{name}.csv"
            temporary = target.with_suffix(target.suffix + ".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(target)
            written[name] = target
        manifest = {
            "run_directory": str(self.run_directory),
            "expected_cases": list(self.expected_cases),
            "table_rows": {name: int(len(frame)) for name, frame in self.tables.items()},
            "warnings": list(self.warnings),
        }
        target = destination / "report_manifest.json"
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(target)
        written["manifest"] = target
        return written


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _safe_json(path: Path, warnings: list[str]) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, Mapping):
        warnings.append(f"Ignored non-object JSON artifact {path}.")
        return None
    return value


def _safe_json_list(path: Path, warnings: list[str]) -> tuple[Mapping[str, Any], ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return ()
    return _mapping_sequence(value)


def _safe_npz(path: Path, warnings: list[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: np.asarray(data[name]).copy() for name in data.files}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return {}


def _safe_csv(path: Path, warnings: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _effective_design(
    run_directory: Path,
    warnings: list[str],
) -> dict[str, np.ndarray]:
    """Load accepted-row inputs when available, with legacy-run fallback."""

    effective = run_directory / "datasets" / "effective_design.npz"
    path = effective if effective.is_file() else run_directory / "datasets" / "design.npz"
    return _safe_npz(path, warnings)


def _accepted_development(
    run_directory: Path,
    warnings: list[str],
) -> dict[str, np.ndarray]:
    """Load accepted mechanistic targets, falling back only for legacy runs."""

    accepted = (
        run_directory / "datasets" / "development" / "mechanistic_accepted_v3.npz"
    )
    path = (
        accepted
        if accepted.is_file()
        else run_directory / "datasets" / "development" / "mechanistic_rows_v3.npz"
    )
    return _safe_npz(path, warnings)


def _boolean_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a strict Boolean view of a possibly CSV-decoded column."""

    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _nearest_rank(values: Iterable[float], probability: float = 0.95) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return math.nan
    finite.sort()
    return float(finite[math.ceil(probability * len(finite)) - 1])


def _finite_summary(values: Iterable[Any]) -> dict[str, float | int]:
    data = np.asarray([_as_float(value) for value in values], dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return {
            "count": 0, "total": math.nan, "mean": math.nan, "median": math.nan,
            "iqr": math.nan, "p95_nearest_rank": math.nan, "maximum": math.nan,
        }
    q1, q3 = np.quantile(data, [0.25, 0.75])
    return {
        "count": int(len(data)),
        "total": float(np.sum(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "iqr": float(q3 - q1),
        "p95_nearest_rank": _nearest_rank(data),
        "maximum": float(np.max(data)),
    }


def _expected_cases(
    run_directory: Path,
    requested: Sequence[str] | None,
    warnings: list[str],
) -> tuple[str, ...]:
    if requested is not None:
        values = tuple(dict.fromkeys(str(value) for value in requested))
        if not values:
            raise ValueError("expected_cases cannot be empty.")
        return values
    design = _effective_design(run_directory, warnings)
    robustness = design.get("robustness_influents")
    count = int(robustness.shape[0]) if robustness is not None and robustness.ndim == 2 else 0
    inferred = ["nominal", *(f"robustness_{index + 1:02d}" for index in range(count))]
    optimization = run_directory / "optimization"
    if optimization.is_dir():
        for path in optimization.iterdir():
            if path.is_dir() and path.name not in inferred:
                inferred.append(path.name)
    return tuple(inferred or ["nominal"])


def _infer_geometry(run_directory: Path, warnings: list[str]) -> StudyGeometry:
    candidates: list[int] = []
    predictions = _safe_npz(run_directory / "predictions" / "untouched_test.npz", warnings)
    for value in predictions.values():
        if value.ndim == 2:
            candidates.append(int(value.shape[1]))
    development = _accepted_development(run_directory, warnings)
    targets = development.get("targets")
    if targets is not None and targets.ndim == 2:
        candidates.append(int(targets.shape[1]))
    for count in candidates:
        layers = count - (N_STAGES + 3) * N_COMPONENTS
        if layers >= 3:
            return StudyGeometry(layers, 6_000.0 / layers)
    warnings.append("Could not infer the clarifier layer count; defaulted to the 10-layer article geometry.")
    return StudyGeometry(10, 600.0)


def _selected_start(
    payload: Mapping[str, Any] | None,
) -> tuple[int | None, Mapping[str, Any] | None, tuple[Mapping[str, Any], ...]]:
    if payload is None:
        return None, None, ()
    starts = _mapping_sequence(payload.get("starts"))
    selected_index = _as_int(payload.get("selected_start"))
    if selected_index is None:
        selected_object = _mapping(payload.get("selected"))
        if selected_object is not None:
            selected_index = _as_int(selected_object.get("start_index"))
    selected = next(
        (item for item in starts if _as_int(item.get("start_index")) == selected_index),
        None,
    )
    return selected_index, selected, starts


def _route_snapshot(
    run_directory: Path,
    case: str,
    route: str,
    warnings: list[str],
) -> RouteSnapshot:
    directory = run_directory / "optimization" / case
    payload = _safe_json(directory / f"{route}.json", warnings)
    partial = _safe_json_list(directory / f"{route}_starts.partial.json", warnings)
    selected_index, selected, starts = _selected_start(payload)
    if not starts and partial:
        starts = partial
    selected_arrays = _safe_npz(directory / f"{route}_selected.npz", warnings)
    equivalence = _safe_json(directory / f"{route}_equivalence.json", warnings)
    reference_arrays = _safe_npz(directory / f"{route}_reference.npz", warnings)
    if payload is not None:
        artifact_state = "complete"
        outcome = str(payload.get("status", "status_unavailable"))
    elif partial or selected_arrays:
        artifact_state = "in_progress"
        outcome = "in_progress"
    else:
        artifact_state = "absent"
        outcome = "not_attempted"
    if selected_index is None and selected_arrays:
        # The selected NPZ is an authoritative endpoint artifact even if the
        # JSON snapshot was not visible during this read.
        selected_index = _as_int(selected_arrays.get("start_index"))
    return RouteSnapshot(
        case=case,
        route=route,
        artifact_state=artifact_state,
        outcome=outcome,
        payload=payload,
        starts=starts,
        selected_start=selected_index,
        selected=selected,
        selected_arrays=selected_arrays,
        equivalence=equivalence,
        reference_arrays=reference_arrays,
    )


def _surrogate_final(snapshot: RouteSnapshot) -> Mapping[str, Any] | None:
    return _mapping(snapshot.selected.get("final")) if snapshot.selected is not None else None


def _selected_theta(snapshot: RouteSnapshot) -> np.ndarray | None:
    stored = snapshot.selected_arrays.get("theta")
    if stored is not None and stored.shape == (len(CONTROL_NAMES),) and np.all(np.isfinite(stored)):
        return np.asarray(stored, dtype=float)
    source = _surrogate_final(snapshot) if snapshot.route == "surrogate" else snapshot.selected
    if source is None:
        return None
    values = np.asarray(source.get("theta", []), dtype=float)
    return values if values.shape == (len(CONTROL_NAMES),) and np.all(np.isfinite(values)) else None


def _selected_response_map(snapshot: RouteSnapshot, geometry: StudyGeometry) -> dict[str, np.ndarray]:
    responses: dict[str, np.ndarray] = {}

    def retain(name: str, value: Any) -> None:
        candidate = np.asarray(value, dtype=float)
        if candidate.shape == (geometry.response_count,) and np.all(np.isfinite(candidate)):
            responses[name] = candidate

    if snapshot.route == "surrogate":
        for name in ("raw", "projected", "smooth", "reference"):
            if name in snapshot.selected_arrays:
                retain(name, snapshot.selected_arrays[name])
        final = _surrogate_final(snapshot)
        if final is not None:
            for name in ("raw", "projected"):
                if name not in responses and name in final:
                    retain(name, final[name])
    else:
        if "response" in snapshot.selected_arrays:
            retain("smooth", snapshot.selected_arrays["response"])
        for name in ("raw", "projected", "smooth", "reference"):
            if name in snapshot.selected_arrays:
                retain(name, snapshot.selected_arrays[name])
    if "response" in snapshot.reference_arrays:
        retain("reference", snapshot.reference_arrays["response"])
    elif "reference" in snapshot.reference_arrays:
        retain("reference", snapshot.reference_arrays["reference"])
    return responses


def _start_feasible(snapshot: RouteSnapshot, start: Mapping[str, Any]) -> bool:
    direct = _as_bool(start.get("feasible"))
    if direct is not None:
        return direct
    final = _mapping(start.get("final"))
    feasibility = _mapping(final.get("feasibility")) if final is not None else None
    return bool(feasibility is not None and feasibility.get("feasible") is True)


def _start_stationary(snapshot: RouteSnapshot, start: Mapping[str, Any]) -> bool:
    direct = _as_bool(start.get("stationary"))
    if direct is not None:
        return direct
    final = _mapping(start.get("final"))
    stationarity = _mapping(final.get("stationarity")) if final is not None else None
    return bool(stationarity is not None and stationarity.get("stationary") is True)


def _selected_feasible(snapshot: RouteSnapshot) -> bool | None:
    if snapshot.selected is None:
        return True if snapshot.selected_arrays else None
    return _start_feasible(snapshot, snapshot.selected)


def _selected_stationary(snapshot: RouteSnapshot) -> bool | None:
    if snapshot.selected is None:
        return None
    return _start_stationary(snapshot, snapshot.selected)


def _stages(start: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return _mapping_sequence(start.get("stages"))


def _route_elapsed(snapshot: RouteSnapshot) -> float:
    if snapshot.payload is not None:
        elapsed = _as_float(snapshot.payload.get("elapsed_seconds"))
        if math.isfinite(elapsed):
            return elapsed
    values = [
        _as_float(stage.get("elapsed_seconds"))
        for start in snapshot.starts
        for stage in _stages(start)
    ]
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite)) if finite else math.nan


def _selected_active_constraints(snapshot: RouteSnapshot) -> int | None:
    if snapshot.selected is None:
        return None
    if snapshot.route == "direct":
        kkt = _mapping(snapshot.selected.get("kkt"))
        return _as_int(kkt.get("active_inequality_count")) if kkt is not None else None
    final = _surrogate_final(snapshot)
    projection = _mapping(final.get("projection")) if final is not None else None
    diagnostics = _mapping(projection.get("diagnostics")) if projection is not None else None
    return _as_int(diagnostics.get("active_inequality_count")) if diagnostics is not None else None


def _replay_status(snapshot: RouteSnapshot) -> str:
    if snapshot.equivalence is not None:
        replay = _mapping(snapshot.equivalence.get("reference_replay"))
        accepted = _as_bool(
            replay.get("accepted")
            if replay is not None else snapshot.equivalence.get("accepted")
        )
        return "accepted" if accepted else "failed"
    if snapshot.reference_arrays:
        return "reference_available_equivalence_pending"
    if _selected_theta(snapshot) is not None:
        return "pending"
    return "not_applicable"


def _route_status_table(snapshots: Sequence[RouteSnapshot]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        declared_attempts = None
        if snapshot.payload is not None:
            declared_attempts = _as_int(
                snapshot.payload.get(
                    "optimization_attempt_count",
                    snapshot.payload.get("article_start_count"),
                )
            )
        expected_starts = (
            declared_attempts
            if declared_attempts is not None and declared_attempts >= 0
            else LEGACY_EXPECTED_STARTS
        )
        stages = [stage for start in snapshot.starts for stage in _stages(start)]
        iterations = [
            _as_int(stage.get("iterations"))
            for stage in stages
            if _as_int(stage.get("iterations")) is not None
        ]
        feasible = sum(_start_feasible(snapshot, start) for start in snapshot.starts)
        stationary = sum(_start_stationary(snapshot, start) for start in snapshot.starts)
        rows.append({
            "case": snapshot.case,
            "route": snapshot.route,
            "artifact_state": snapshot.artifact_state,
            "outcome": snapshot.outcome,
            "selected": _selected_theta(snapshot) is not None,
            "selected_start": snapshot.selected_start,
            "selected_feasible": _selected_feasible(snapshot),
            "selected_stationary": _selected_stationary(snapshot),
            "starts_attempted": len(snapshot.starts),
            "starts_expected": expected_starts,
            "feasible_starts": feasible,
            "stationary_starts": stationary,
            "failed_starts": len(snapshot.starts) - feasible,
            "stages_attempted": len(stages),
            "iterations": sum(iterations),
            "active_constraint_count": _selected_active_constraints(snapshot),
            "elapsed_seconds": _route_elapsed(snapshot),
            "preflight_stage_wall_time_seconds": _as_float(
                snapshot.payload.get("preflight_stage_wall_time_seconds")
                if snapshot.payload is not None else None
            ),
            "replay_status": _replay_status(snapshot),
            "equivalence_accepted": (
                _as_bool(snapshot.equivalence.get("accepted"))
                if snapshot.equivalence is not None else None
            ),
        })
    return pd.DataFrame(rows)


def _active_constraint_table(snapshots: Sequence[RouteSnapshot]) -> pd.DataFrame:
    """Retain every named upper row plus lower/direct active-set counts."""

    engineering_names = (
        "srt_lower", "srt_upper", "external_solids_loss_guard", "slr_upper",
        "underflow_tss_upper", "feed_tss_lower", "sor_upper",
    )
    trust_names = (
        "correction", "regularized_leverage", "particulate_split",
        "reactor_residual", "clarifier_flux",
    )
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        count = _selected_active_constraints(snapshot)
        rows.append({
            "case": snapshot.case,
            "route": snapshot.route,
            "constraint_group": "projection_qp" if snapshot.route == "surrogate" else "direct_nlp",
            "constraint": "all_inequalities",
            "residual": math.nan,
            "active": None if count is None else count > 0,
            "active_count": count,
            "violated": None,
        })
        if snapshot.route != "surrogate":
            continue
        final = _surrogate_final(snapshot)
        if final is None:
            continue
        for group, names, key in (
            ("engineering", engineering_names, "engineering_rows"),
            ("trust", trust_names, "trust_rows"),
        ):
            try:
                values = np.asarray(final.get(key, []), dtype=float).reshape(-1)
            except (TypeError, ValueError):
                values = np.asarray([])
            for index, value in enumerate(values):
                name = names[index] if index < len(names) else f"row_{index + 1}"
                finite = math.isfinite(float(value))
                rows.append({
                    "case": snapshot.case,
                    "route": snapshot.route,
                    "constraint_group": group,
                    "constraint": name,
                    "residual": float(value) if finite else math.nan,
                    "active": bool(finite and value >= -1.0e-7),
                    "active_count": 1 if finite and value >= -1.0e-7 else 0,
                    "violated": bool(finite and value > 1.0e-6),
                })
    return pd.DataFrame(rows)


def _controls_table(snapshots: Sequence[RouteSnapshot]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        theta = _selected_theta(snapshot)
        row: dict[str, Any] = {
            "case": snapshot.case,
            "route": snapshot.route,
            "status": snapshot.outcome,
            "selected_start": snapshot.selected_start,
            "available": theta is not None,
        }
        row.update({name: math.nan for name in CONTROL_NAMES})
        if theta is not None:
            row.update(dict(zip(CONTROL_NAMES, theta, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _quality_scale(run_directory: Path, geometry: StudyGeometry, warnings: list[str]) -> np.ndarray:
    design = _effective_design(run_directory, warnings)
    development = _accepted_development(run_directory, warnings)
    decisions = design.get("development_decisions")
    targets = development.get("targets")
    if (
        decisions is None or targets is None or decisions.ndim != 2 or targets.ndim != 2
        or len(decisions) != len(targets) or decisions.shape[1] < 7
        or targets.shape[1] != geometry.response_count
    ):
        warnings.append("Development targets were unavailable for the four objective quality scales.")
        return np.full(4, math.nan)
    start = (N_STAGES + 1) * N_COMPONENTS
    g_e = targets[:, start : start + N_COMPONENTS]
    c_e = g_e / (1.0 - decisions[:, 6, None])
    scales = np.std(c_e @ COMPOSITE_MATRIX.T, axis=0, ddof=0)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        warnings.append("At least one objective quality scale was non-positive.")
        return np.full(4, math.nan)
    return scales


def _response_quantities(
    theta: np.ndarray,
    response: np.ndarray,
    geometry: StudyGeometry,
    quality_scale: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, float]:
    layer_start = (N_STAGES + 3) * N_COMPONENTS
    reactors = [
        response[(index + 1) * N_COMPONENTS : (index + 2) * N_COMPONENTS]
        for index in range(N_STAGES)
    ]
    g_e = response[(N_STAGES + 1) * N_COMPONENTS : (N_STAGES + 2) * N_COMPONENTS]
    g_u = response[(N_STAGES + 2) * N_COMPONENTS : layer_start]
    layers = response[layer_start:]
    hrt, r_r, waste = float(theta[0]), float(theta[5]), float(theta[6])
    q_c, q_u, q_e = 1.0 + r_r, r_r + waste, 1.0 - waste
    c_e, c_u = g_e / q_e, g_u / q_u
    feed_tss = float(TSS_VECTOR @ reactors[-1])
    underflow_tss = float(TSS_VECTOR @ c_u)
    effluent_tss_flow = float(TSS_VECTOR @ g_e)
    external_loss = effluent_tss_flow + waste * underflow_tss
    stage_volume = geometry.fresh_flow_m3_d * hrt / (24.0 * N_STAGES)
    inventory = stage_volume * sum(float(TSS_VECTOR @ reactor) for reactor in reactors)
    inventory += geometry.layer_volume_m3 * float(np.sum(layers))
    srt = inventory / (geometry.fresh_flow_m3_d * external_loss) if external_loss != 0.0 else math.nan
    sor = geometry.fresh_flow_m3_d * q_e / geometry.clarifier_area_m2
    slr = 1.0e-3 * geometry.fresh_flow_m3_d * q_c * feed_tss / geometry.clarifier_area_m2
    composites = COMPOSITE_MATRIX @ c_e
    quantities = {
        "solids_inventory": inventory,
        "external_solids_loss": external_loss,
        "srt_d": srt,
        "sor_m_d": sor,
        "slr_kg_m2_d": slr,
        "underflow_tss_g_m3": underflow_tss,
        "feed_tss_g_m3": feed_tss,
        **dict(zip(("effluent_cod", "effluent_tn", "effluent_tp", "effluent_tss"), composites, strict=True)),
    }
    quality = (
        float(np.mean(composites / quality_scale))
        if np.all(np.isfinite(quality_scale)) else math.nan
    )
    components = np.asarray([
        quality,
        (theta[0] - 6.0) / 30.0,
        theta[0] * float(np.sum(theta[1:4])) / (36.0 * 3.0),
        theta[4] / 4.0,
        (theta[5] - 0.25) / 1.0,
        theta[6] * underflow_tss / (0.05 * geometry.underflow_tss_reference_g_m3),
    ])
    objective = float(np.asarray([0.50, 0.15, 0.20, 0.05, 0.05, 0.05]) @ components)
    return quantities, components, objective


def _json_objective(snapshot: RouteSnapshot) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    source = _surrogate_final(snapshot) if snapshot.route == "surrogate" else snapshot.selected
    if source is None:
        return math.nan, None, None
    objective = _as_float(source.get("objective"))
    components = np.asarray(source.get("objective_components", []), dtype=float)
    engineering = np.asarray(source.get("engineering", []), dtype=float)
    if components.shape != (len(OBJECTIVE_COMPONENT_NAMES),) or not np.all(np.isfinite(components)):
        components = None
    if engineering.ndim != 1 or not len(engineering) or not np.all(np.isfinite(engineering)):
        engineering = None
    return objective, components, engineering


def _objective_engineering_tables(
    snapshots: Sequence[RouteSnapshot],
    geometry: StudyGeometry,
    quality_scale: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    objectives: list[dict[str, Any]] = []
    engineering_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        theta = _selected_theta(snapshot)
        responses = _selected_response_map(snapshot, geometry)
        solver_objective, solver_components, _ = _json_objective(snapshot)
        selected_method = "projected" if snapshot.route == "surrogate" else "smooth"
        row: dict[str, Any] = {
            "case": snapshot.case,
            "route": snapshot.route,
            "response_method": selected_method,
            "available": theta is not None and selected_method in responses,
            "solver_reported_objective": solver_objective,
            "recomputed_objective": math.nan,
        }
        row.update({name: math.nan for name in OBJECTIVE_COMPONENT_NAMES})
        if solver_components is not None:
            row.update(dict(zip(OBJECTIVE_COMPONENT_NAMES, solver_components, strict=True)))
        if theta is not None and selected_method in responses:
            quantities, components, objective = _response_quantities(
                theta, responses[selected_method], geometry, quality_scale,
            )
            row["recomputed_objective"] = objective
            row.update(dict(zip(OBJECTIVE_COMPONENT_NAMES, components, strict=True)))
            engineering_rows.append({
                "case": snapshot.case,
                "decision_route": snapshot.route,
                "response_method": selected_method,
                "available": True,
                **quantities,
            })
        else:
            engineering_rows.append({
                "case": snapshot.case,
                "decision_route": snapshot.route,
                "response_method": selected_method,
                "available": False,
                **{name: math.nan for name in ENGINEERING_QUANTITY_NAMES},
            })
        objectives.append(row)
        for method in ("raw", "projected", "smooth", "reference"):
            quality: dict[str, Any] = {
                "case": snapshot.case,
                "decision_route": snapshot.route,
                "response_method": method,
                "available": theta is not None and method in responses,
                "COD": math.nan,
                "TN": math.nan,
                "TP": math.nan,
                "TSS": math.nan,
                "objective": math.nan,
            }
            if theta is not None and method in responses:
                quantities, _, objective = _response_quantities(theta, responses[method], geometry, quality_scale)
                quality.update({
                    "COD": quantities["effluent_cod"],
                    "TN": quantities["effluent_tn"],
                    "TP": quantities["effluent_tp"],
                    "TSS": quantities["effluent_tss"],
                    "objective": objective,
                })
            quality_rows.append(quality)
    return pd.DataFrame(objectives), pd.DataFrame(engineering_rows), pd.DataFrame(quality_rows)


def _profile_rows(
    snapshots: Sequence[RouteSnapshot],
    geometry: StudyGeometry,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    composite_names = ("COD", "TN", "TP", "TSS")
    for snapshot in snapshots:
        theta = _selected_theta(snapshot)
        if theta is None:
            continue
        responses = _selected_response_map(snapshot, geometry)
        for method, response in responses.items():
            q_e, q_u = 1.0 - theta[6], theta[5] + theta[6]
            blocks: list[tuple[str, np.ndarray, tuple[str, ...]]] = []
            mixer = response[:N_COMPONENTS]
            blocks.append(("mixer", COMPOSITE_MATRIX @ mixer, composite_names))
            for stage in range(N_STAGES):
                reactor = response[(stage + 1) * N_COMPONENTS : (stage + 2) * N_COMPONENTS]
                reactor_values = np.asarray([
                    reactor[0],
                    COMPOSITE_MATRIX[1] @ reactor,
                    COMPOSITE_MATRIX[2] @ reactor,
                    COMPOSITE_MATRIX[3] @ reactor,
                ])
                blocks.append((f"reactor_{stage + 1}", reactor_values, ("DO", "TN", "TP", "TSS")))
            overflow_start = (N_STAGES + 1) * N_COMPONENTS
            underflow_start = (N_STAGES + 2) * N_COMPONENTS
            c_e = response[overflow_start : overflow_start + N_COMPONENTS] / q_e
            c_u = response[underflow_start : underflow_start + N_COMPONENTS] / q_u
            blocks.append(("clarifier_overflow", COMPOSITE_MATRIX @ c_e, composite_names))
            blocks.append(("clarifier_underflow", COMPOSITE_MATRIX @ c_u, composite_names))
            for location, values, quantities in blocks:
                for quantity, value in zip(quantities, values, strict=True):
                    rows.append({
                        "case": snapshot.case,
                        "decision_route": snapshot.route,
                        "response_method": method,
                        "location": location,
                        "quantity": quantity,
                        "value": float(value),
                    })
            layers = response[(N_STAGES + 3) * N_COMPONENTS :]
            for index, value in enumerate(layers):
                rows.append({
                    "case": snapshot.case,
                    "decision_route": snapshot.route,
                    "response_method": method,
                    "location": f"clarifier_layer_{index + 1}",
                    "quantity": "TSS",
                    "value": float(value),
                })
    return pd.DataFrame(
        rows,
        columns=("case", "decision_route", "response_method", "location", "quantity", "value"),
    )


def _equivalence_table(snapshots: Sequence[RouteSnapshot]) -> pd.DataFrame:
    fields = (
        "smooth_accepted", "reference_accepted", "accepted", "state_rms", "state_inf",
        "own_smooth_residual", "own_reference_residual", "cross_residual",
        "relative_objective_difference", "engineering_difference",
        "reference_root_difference_generation", "reference_root_difference_state_scale",
        "branch_agreement", "feasibility_agreement",
    )
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        row: dict[str, Any] = {
            "case": snapshot.case,
            "decision_route": snapshot.route,
            "artifact_state": (
                "available" if snapshot.equivalence is not None
                else "pending" if _selected_theta(snapshot) is not None
                else "not_applicable"
            ),
        }
        for field in fields:
            row[field] = snapshot.equivalence.get(field) if snapshot.equivalence is not None else None
        optimizer_root = (
            _mapping(snapshot.equivalence.get("optimizer_root_reproduction"))
            if snapshot.equivalence is not None else None
        )
        for field in (
            "applicable", "state_scaled_inf", "feed_tss_scaled_absolute",
            "maximum_scaled_difference", "branch_agreement", "accepted",
        ):
            row[f"optimizer_root_{field}"] = (
                optimizer_root.get(field) if optimizer_root is not None else None
            )
        derivative = (
            _mapping(snapshot.equivalence.get("derivative_audit"))
            if snapshot.equivalence is not None else None
        )
        derivative_result = (
            _mapping(derivative.get("result")) if derivative is not None else None
        )
        row["derivative_audit_status"] = (
            derivative.get("status") if derivative is not None else None
        )
        row["derivative_audit_passed"] = (
            derivative.get("passed") if derivative is not None else None
        )
        row["derivative_maximum_jacobian_discrepancy"] = (
            derivative_result.get("maximum_jacobian_discrepancy")
            if derivative_result is not None else None
        )
        row["derivative_maximum_hessian_discrepancy"] = (
            derivative_result.get("maximum_hessian_discrepancy")
            if derivative_result is not None else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _trust_table(run_directory: Path, snapshots: Sequence[RouteSnapshot], warnings: list[str]) -> pd.DataFrame:
    limits = _safe_json(run_directory / "metrics" / "trust_limits.json", warnings) or {}
    development = _safe_csv(run_directory / "metrics" / "trust_development_oof.csv", warnings)
    test = _safe_csv(run_directory / "metrics" / "trust_untouched_test.csv", warnings)
    aliases = {
        "correction": "correction",
        "regularized_leverage": "regularized_leverage",
        "particulate_split": "particulate_split",
        "reactor_residual": "reactor_residual",
        "clarifier_flux": "clarifier_flux",
    }
    selected_values: dict[str, list[float]] = {name: [] for name in TRUST_DIAGNOSTICS}
    raw_names = ("correction", "regularized_leverage", "particulate_split", "reactor_residual", "clarifier_flux")
    for snapshot in snapshots:
        if snapshot.route != "surrogate":
            continue
        final = _surrogate_final(snapshot)
        values = np.asarray(final.get("trust_values", []), dtype=float) if final is not None else np.asarray([])
        if values.shape == (5,) and np.all(np.isfinite(values)):
            converted = np.asarray([
                math.sqrt(max(0.0, values[0])), values[1],
                math.sqrt(max(0.0, values[2])), math.sqrt(max(0.0, values[3])),
                math.sqrt(max(0.0, values[4])),
            ])
            for name, value in zip(raw_names, converted, strict=True):
                selected_values[name].append(float(value))
    rows: list[dict[str, Any]] = []
    for name in TRUST_DIAGNOSTICS:
        column = aliases[name]
        development_values = development[column].to_numpy(dtype=float) if column in development else np.asarray([])
        test_values = test[column].to_numpy(dtype=float) if column in test else np.asarray([])
        rows.append({
            "diagnostic": name,
            "frozen_limit": _as_float(limits.get(name)),
            "development_count": int(np.count_nonzero(np.isfinite(development_values))),
            "development_p95": _nearest_rank(development_values),
            "test_count": int(np.count_nonzero(np.isfinite(test_values))),
            "test_p95": _nearest_rank(test_values),
            "selected_count": len(selected_values[name]),
            "largest_selected_value": (
                max(selected_values[name]) if selected_values[name] else math.nan
            ),
        })
    return pd.DataFrame(rows)


def _case_influents(
    run_directory: Path,
    expected_cases: Sequence[str],
    warnings: list[str],
) -> dict[str, np.ndarray]:
    result = {"nominal": np.asarray(NOMINAL_INFLUENT, dtype=float)}
    design = _effective_design(run_directory, warnings)
    values = design.get("robustness_influents")
    if values is not None and values.ndim == 2 and values.shape[1] == N_COMPONENTS:
        for index, row in enumerate(values):
            result[f"robustness_{index + 1:02d}"] = np.asarray(row, dtype=float)
            result[f"scenario_{index + 1:02d}"] = np.asarray(row, dtype=float)
    return {case: result[case] for case in expected_cases if case in result}


def _reconstruct_selected_violations(
    run_directory: Path,
    snapshots: Sequence[RouteSnapshot],
    expected_cases: Sequence[str],
    geometry: StudyGeometry,
    warnings: list[str],
) -> pd.DataFrame:
    design = _effective_design(run_directory, warnings)
    development = _accepted_development(run_directory, warnings)
    decisions = design.get("development_decisions")
    influents = design.get("development_influents")
    targets = development.get("targets")
    model = _safe_npz(run_directory / "models" / "ridge_surrogate.npz", warnings)
    state_scale = model.get("response_scale")
    if (
        decisions is None or influents is None or targets is None or state_scale is None
        or decisions.ndim != 2 or influents.ndim != 2 or targets.ndim != 2
        or decisions.shape != (len(targets), 7)
        or influents.shape != (len(targets), N_COMPONENTS)
        or targets.shape[1] != geometry.response_count
        or state_scale.shape != (geometry.response_count,)
    ):
        warnings.append("Selected-response physical audits could not be reconstructed from frozen scales.")
        return pd.DataFrame()
    layout = NetworkLayout(layer_count=geometry.layer_count)
    try:
        row_scales = fit_network_row_scales(
            targets,
            influents,
            internal_recycle=decisions[:, 4],
            return_recycle=decisions[:, 5],
            waste_fraction=decisions[:, 6],
            invariant_operator=INVARIANT_MATRIX,
            tss_weights=TSS_VECTOR,
            layout=layout,
            minimum_scale=1.0,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        warnings.append(f"Selected-response row-scale reconstruction failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()
    case_influents = _case_influents(run_directory, expected_cases, warnings)
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        theta = _selected_theta(snapshot)
        feed = case_influents.get(snapshot.case)
        if theta is None or feed is None:
            continue
        for method, response in _selected_response_map(snapshot, geometry).items():
            try:
                record = violation_record(
                    method,
                    f"{snapshot.case}:{snapshot.route}",
                    response,
                    theta,
                    feed,
                    layout,
                    row_scales.equality,
                    row_scales.inequality,
                    state_scale,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                warnings.append(
                    f"Physical audit failed for {snapshot.case}/{snapshot.route}/{method}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            rows.append(record)
    return pd.DataFrame(rows)


def _physical_detail(
    run_directory: Path,
    reconstructed: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    combined = _safe_csv(
        run_directory / "metrics" / "physical_violations_all_analysis.csv",
        warnings,
    )
    if not combined.empty:
        return combined
    assessment = _safe_csv(run_directory / "metrics" / "physical_violations_assessment.csv", warnings)
    equivalence = _safe_csv(
        run_directory / "metrics" / "physical_violations_equivalence_test.csv",
        warnings,
    )
    selected = _safe_csv(run_directory / "metrics" / "physical_violations_selected_cases.csv", warnings)
    frames: list[pd.DataFrame] = []
    if not assessment.empty:
        item = assessment.copy()
        item.insert(0, "analysis_scope", "untouched_test")
        frames.append(item)
    if not equivalence.empty:
        item = equivalence.copy()
        item.insert(0, "analysis_scope", "untouched_test_equivalence")
        frames.append(item)
    if not selected.empty:
        item = selected.copy()
        item.insert(0, "analysis_scope", "selected_decisions")
        frames.append(item)
    if not reconstructed.empty:
        item = reconstructed.copy()
        item.insert(0, "analysis_scope", "selected_decisions")
        if frames:
            existing = pd.concat(frames, ignore_index=True)
            keys = set(zip(existing.get("case", []), existing.get("method", []), strict=False))
            item = item[
                [(case, method) not in keys for case, method in zip(item["case"], item["method"], strict=True)]
            ]
        if not item.empty:
            frames.append(item)
    if not frames:
        return pd.DataFrame(columns=("analysis_scope", "case", "method"))
    return pd.concat(frames, ignore_index=True, sort=False)


def _maximum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else math.nan


def _sum(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return int(values.sum())


def _audited_physical_rows(
    group: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int, float]:
    """Return rows with an available audit and explicit coverage counts.

    Historical physical-audit tables predate ``audit_available``. Those rows
    contain computed residual columns and remain valid evidence, so an absent
    column (or the NaNs introduced when such a table is concatenated with a
    newer one) is treated as available. An explicit false value identifies a
    placeholder whose zero-valued count fields must not enter violation sums.
    """

    total = int(len(group))
    if total == 0:
        return group, 0, 0, math.nan
    if "audit_available" not in group:
        audited = group
    else:
        def available(value: Any) -> bool:
            if pd.isna(value):
                return True
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes"}:
                    return True
                if normalized in {"false", "0", "no"}:
                    return False
            if isinstance(value, (int, np.integer)) and value in (0, 1):
                return bool(value)
            return False

        mask = group["audit_available"].map(available).to_numpy(dtype=bool)
        audited = group.loc[mask]
    audited_count = int(len(audited))
    unavailable_count = total - audited_count
    return audited, audited_count, unavailable_count, audited_count / total


def _physical_summary(detail: pd.DataFrame) -> pd.DataFrame:
    families = (
        "mass_mixer_component_max", "mass_reactor_invariant_max",
        "mass_clarifier_component_max", "mass_soluble_passthrough_max",
        "mass_tss_endpoint_max", "mass_external_invariant_max",
        "mass_physical_residual_max", "nonlinear_balance_residual_max",
        "rate_nonnegativity_violation_max",
    )
    required = {
        "untouched_test": ("raw", "projected", "mechanistic"),
        "untouched_test_equivalence": ("smooth", "reference"),
        "selected_decisions": ("raw", "projected", "smooth", "reference"),
    }
    rows: list[dict[str, Any]] = []
    for scope, methods in required.items():
        for method in methods:
            group = (
                detail[(detail["analysis_scope"] == scope) & (detail["method"] == method)]
                if {"analysis_scope", "method"}.issubset(detail.columns)
                else pd.DataFrame()
            )
            audited, audited_count, unavailable_count, coverage = (
                _audited_physical_rows(group)
            )
            has_audit = audited_count > 0
            mass_count = (
                _sum(audited, "mass_conservation_violation_count")
                if has_audit else math.nan
            )
            negative_count = (
                _sum(audited, "nonnegativity_violation_count")
                if has_audit else math.nan
            )
            row = {
                "analysis_scope": scope,
                "method": method,
                "availability": (
                    "available" if audited_count == len(group) and audited_count
                    else "partially_available" if audited_count
                    else "not_available"
                ),
                "record_count": int(len(group)),
                "audited_record_count": audited_count,
                "unavailable_record_count": unavailable_count,
                "audit_coverage_fraction": coverage,
                "mass_conservation_tolerance": 1.0e-8,
                "mass_conservation_violation_max": _maximum(
                    audited, "mass_conservation_violation_max"
                ),
                "mass_conservation_violation_count": mass_count,
                "records_with_mass_violation": int(
                    np.count_nonzero(
                        pd.to_numeric(audited.get("mass_conservation_violation_count", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy() > 0
                    )
                ) if has_audit else math.nan,
                "nonnegativity_tolerance": 1.0e-10,
                "nonnegativity_violation_max": _maximum(
                    audited, "nonnegativity_violation_max"
                ),
                "nonnegativity_violation_count": negative_count,
                "records_with_nonnegativity_violation": int(
                    np.count_nonzero(
                        pd.to_numeric(audited.get("nonnegativity_violation_count", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy() > 0
                    )
                ) if has_audit else math.nan,
                "minimum_coordinate": (
                    float(pd.to_numeric(audited["minimum_coordinate"], errors="coerce").min())
                    if has_audit and "minimum_coordinate" in audited else math.nan
                ),
                "network_inequality_violation_max": _maximum(
                    audited, "network_inequality_violation_max"
                ),
                "network_inequality_violation_count": (
                    _sum(audited, "network_inequality_violation_count")
                    if has_audit else math.nan
                ),
            }
            row.update({column: _maximum(audited, column) for column in families})
            rows.append(row)
    for method in REQUIRED_PHYSICAL_METHODS:
        group = detail[detail["method"] == method] if "method" in detail else pd.DataFrame()
        audited, audited_count, unavailable_count, coverage = _audited_physical_rows(group)
        has_audit = audited_count > 0
        row = {
            "analysis_scope": "all_analysis",
            "method": method,
            "availability": (
                "available" if audited_count == len(group) and audited_count
                else "partially_available" if audited_count
                else "not_available"
            ),
            "record_count": int(len(group)),
            "audited_record_count": audited_count,
            "unavailable_record_count": unavailable_count,
            "audit_coverage_fraction": coverage,
            "mass_conservation_tolerance": 1.0e-8,
            "mass_conservation_violation_max": _maximum(
                audited, "mass_conservation_violation_max"
            ),
            "mass_conservation_violation_count": (
                _sum(audited, "mass_conservation_violation_count")
                if has_audit else math.nan
            ),
            "records_with_mass_violation": int(
                np.count_nonzero(
                    pd.to_numeric(audited.get("mass_conservation_violation_count", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy() > 0
                )
            ) if has_audit else math.nan,
            "nonnegativity_tolerance": 1.0e-10,
            "nonnegativity_violation_max": _maximum(
                audited, "nonnegativity_violation_max"
            ),
            "nonnegativity_violation_count": (
                _sum(audited, "nonnegativity_violation_count")
                if has_audit else math.nan
            ),
            "records_with_nonnegativity_violation": int(
                np.count_nonzero(
                    pd.to_numeric(audited.get("nonnegativity_violation_count", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy() > 0
                )
            ) if has_audit else math.nan,
            "minimum_coordinate": (
                float(pd.to_numeric(audited["minimum_coordinate"], errors="coerce").min())
                if has_audit and "minimum_coordinate" in audited else math.nan
            ),
            "network_inequality_violation_max": _maximum(
                audited, "network_inequality_violation_max"
            ),
            "network_inequality_violation_count": (
                _sum(audited, "network_inequality_violation_count")
                if has_audit else math.nan
            ),
        }
        row.update({column: _maximum(audited, column) for column in families})
        rows.append(row)
    return pd.DataFrame(rows)


def _projection_failed(snapshot: RouteSnapshot) -> bool:
    if snapshot.route != "surrogate":
        return False
    final = _surrogate_final(snapshot)
    projection = _mapping(final.get("projection")) if final is not None else None
    accepted = _as_bool(projection.get("accepted")) if projection is not None else None
    return accepted is False


def _residual_or_stability_failed(equivalence: Mapping[str, Any]) -> bool:
    tests = (
        ("state_rms", 1.0e-6),
        ("state_inf", 1.0e-5),
        ("own_smooth_residual", 1.0e-8),
        ("own_reference_residual", 1.0e-8),
        ("cross_residual", 1.0e-5),
        ("relative_objective_difference", 1.0e-6),
        ("engineering_difference", 1.0e-6),
        ("reference_root_difference_generation", 1.0e-6),
        ("reference_root_difference_state_scale", 1.0e-6),
    )
    if _as_bool(equivalence.get("smooth_accepted")) is False:
        return True
    return any(
        math.isfinite(_as_float(equivalence.get(name)))
        and _as_float(equivalence.get(name)) > tolerance
        for name, tolerance in tests
    )


def _disposition(snapshot: RouteSnapshot) -> str:
    if snapshot.artifact_state != "complete":
        return PENDING_CLASS
    if _selected_theta(snapshot) is None:
        return "no accepted optimization start"
    if _projection_failed(snapshot):
        return "projection failure"
    equivalence = snapshot.equivalence
    if equivalence is None:
        return PENDING_CLASS
    replay = _mapping(equivalence.get("reference_replay"))
    if (
        _as_bool(equivalence.get("reference_accepted")) is False
        or (replay is not None and _as_bool(replay.get("accepted")) is False)
    ):
        return "reference integration failure"
    if _residual_or_stability_failed(equivalence):
        return "residual or reduced-stability failure"
    if _selected_feasible(snapshot) is False or _as_bool(equivalence.get("feasibility_agreement")) is False:
        return "engineering infeasibility"
    if _as_bool(equivalence.get("branch_agreement")) is False:
        return "smooth-branch disagreement"
    optimizer_root = _mapping(equivalence.get("optimizer_root_reproduction"))
    if optimizer_root is not None and optimizer_root.get("applicable") is True:
        if _as_bool(optimizer_root.get("branch_agreement")) is False:
            return "smooth-branch disagreement"
        if _as_bool(optimizer_root.get("accepted")) is False:
            return "residual or reduced-stability failure"
    derivative = _mapping(equivalence.get("derivative_audit"))
    if derivative is not None and _as_bool(derivative.get("passed")) is False:
        return "upper-stationarity failure"
    if _selected_stationary(snapshot) is False:
        return "upper-stationarity failure"
    if _as_bool(equivalence.get("accepted")) is True:
        return "validated result"
    return PENDING_CLASS


def _case_and_failure_tables(
    snapshots: Sequence[RouteSnapshot],
    expected_cases: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = {(item.case, item.route): item for item in snapshots}
    rows: list[dict[str, Any]] = []
    for case in expected_cases:
        surrogate = lookup[(case, "surrogate")]
        direct = lookup[(case, "direct")]
        selected_count = sum(_selected_theta(item) is not None for item in (surrogate, direct))
        rows.append({
            "case": case,
            "surrogate_outcome": surrogate.outcome,
            "mechanistic_outcome": direct.outcome,
            "selected_n": selected_count,
            "surrogate_replay_outcome": _replay_status(surrogate),
            "mechanistic_replay_outcome": _replay_status(direct),
            "surrogate_disposition": _disposition(surrogate),
            "mechanistic_disposition": _disposition(direct),
        })
    case_status = pd.DataFrame(rows)
    failure_rows: list[dict[str, Any]] = []
    for route, column in (("surrogate", "surrogate_disposition"), ("direct", "mechanistic_disposition")):
        dispositions = case_status[column].tolist()
        for category in (*FAILURE_CLASSES, PENDING_CLASS):
            failure_rows.append({
                "route": route,
                "classification": category,
                "classification_type": "pending" if category == PENDING_CLASS else "adjudication",
                "count": dispositions.count(category),
                "denominator": len(expected_cases),
                "fraction": dispositions.count(category) / len(expected_cases),
            })
    return case_status, pd.DataFrame(failure_rows)


def _generation_tables(
    run_directory: Path,
    warnings: list[str],
) -> dict[str, pd.DataFrame]:
    """Summarize accepted-row replacement without hiding failed attempts."""

    summary_rows: list[dict[str, Any]] = []
    disposition_rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    attempts_frames: list[pd.DataFrame] = []
    provenance_frames: list[pd.DataFrame] = []
    migration_frames: list[pd.DataFrame] = []
    coordinate_names = (*CONTROL_NAMES, *COMPONENTS)
    lower_bounds = np.concatenate((DECISION_LOWER, INFLUENT_LOWER))
    upper_bounds = np.concatenate((DECISION_UPPER, INFLUENT_UPPER))

    for block in GENERATION_BLOCKS:
        directory = run_directory / "datasets" / block
        attempts = _safe_csv(directory / "all_attempts.csv", warnings)
        provenance = _safe_csv(directory / "accepted_provenance.csv", warnings)
        migration = _safe_csv(directory / "base_checkpoint_migration.csv", warnings)
        accepted_inputs = _safe_npz(directory / "accepted_inputs.npz", warnings)
        summary = _safe_json(directory / "replacement_summary.json", warnings) or {}
        available = any(
            path.is_file()
            for path in (
                directory / "all_attempts.csv",
                directory / "accepted_provenance.csv",
                directory / "accepted_inputs.npz",
                directory / "replacement_summary.json",
            )
        )

        accepted_mask = _boolean_series(attempts, "accepted")
        attempted_count = int(len(attempts))
        accepted_attempt_count = int(accepted_mask.sum())
        rejected_count = int(attempted_count - accepted_attempt_count)
        accepted_count = int(
            len(provenance)
            if len(provenance)
            else (_as_int(summary.get("accepted_count")) or 0)
        )
        required_count = _as_int(summary.get("requested_accepted_count"))
        if required_count is None:
            required_count = accepted_count
        slots = (
            pd.to_numeric(provenance["accepted_slot"], errors="coerce")
            if "accepted_slot" in provenance
            else pd.Series(dtype=float)
        )
        traced = bool(
            required_count >= 0
            and len(provenance) == required_count
            and slots.notna().all()
            and set(slots.astype(int).tolist()) == set(range(required_count))
            and (
                "source_candidate_id" in provenance
                and provenance["source_candidate_id"].astype(str).is_unique
            )
        )
        supplemental_attempts = _as_int(summary.get("supplemental_attempt_count"))
        if supplemental_attempts is None and "candidate_round" in attempts:
            rounds = pd.to_numeric(attempts["candidate_round"], errors="coerce")
            supplemental_attempts = int((rounds > 0).sum())
        supplemental_attempts = supplemental_attempts or 0
        summary_rows.append({
            "block": block,
            "availability": "available" if available else "not_available",
            "candidate_attempt_denominator": attempted_count,
            "accepted_row_denominator": accepted_count,
            "required_accepted_rows": required_count,
            "rejected_candidate_count": rejected_count,
            "attempt_acceptance_fraction": (
                accepted_attempt_count / attempted_count
                if attempted_count else math.nan
            ),
            "base_attempt_count": _as_int(summary.get("base_attempt_count")),
            "base_accepted_count": _as_int(summary.get("base_accepted_count")),
            "supplemental_attempt_count": supplemental_attempts,
            "supplemental_accepted_count": _as_int(
                summary.get("supplemental_accepted_count")
            ),
            "supplemental_round_count": _as_int(
                summary.get("supplemental_round_count")
            ),
            "accepted_slots_fully_traced": traced,
            "single_global_strength_one_lhs": (
                bool(supplemental_attempts == 0) if available else None
            ),
            "accepted_set_conditioned_on_mechanistic_acceptance": (
                True if available else None
            ),
        })

        primary_categories = ("accepted", *(name for name, _ in GENERATION_REJECTION_FLAGS))
        primary = attempts.get(
            "rejection_reason", pd.Series("", index=attempts.index, dtype=object),
        ).fillna("").astype(str)
        for category in primary_categories:
            if category == "accepted":
                count = accepted_attempt_count
            else:
                count = int(((~accepted_mask) & primary.eq(category)).sum())
            disposition_rows.append({
                "block": block,
                "disposition": category,
                "count": count,
                "candidate_attempt_denominator": attempted_count,
                "fraction_of_attempts": count / attempted_count if attempted_count else math.nan,
            })
        known_primary = {name for name, _ in GENERATION_REJECTION_FLAGS}
        unclassified = int(
            ((~accepted_mask) & ~primary.isin(known_primary)).sum()
        )
        disposition_rows.append({
            "block": block,
            "disposition": "unclassified_rejection",
            "count": unclassified,
            "candidate_attempt_denominator": attempted_count,
            "fraction_of_attempts": (
                unclassified / attempted_count if attempted_count else math.nan
            ),
        })
        for reason, column in GENERATION_REJECTION_FLAGS:
            count = int(((~accepted_mask) & _boolean_series(attempts, column)).sum())
            reason_rows.append({
                "block": block,
                "rejection_reason": reason,
                "count": count,
                "candidate_attempt_denominator": attempted_count,
                "rejected_candidate_denominator": rejected_count,
                "fraction_of_attempts": count / attempted_count if attempted_count else math.nan,
                "fraction_of_rejections": count / rejected_count if rejected_count else math.nan,
                "categories_are_nonexclusive": True,
            })

        decisions = accepted_inputs.get("decisions")
        influents = accepted_inputs.get("influents")
        if (
            decisions is not None
            and influents is not None
            and decisions.ndim == 2
            and influents.ndim == 2
            and decisions.shape[1] == len(CONTROL_NAMES)
            and influents.shape[1] == len(COMPONENTS)
            and len(decisions) == len(influents)
        ):
            values = np.hstack((decisions, influents))
            for index, name in enumerate(coordinate_names):
                coordinate = np.asarray(values[:, index], dtype=float)
                finite = coordinate[np.isfinite(coordinate)]
                span = float(upper_bounds[index] - lower_bounds[index])
                coverage_rows.append({
                    "block": block,
                    "coordinate": name,
                    "accepted_row_denominator": len(coordinate),
                    "finite_count": len(finite),
                    "declared_lower": float(lower_bounds[index]),
                    "declared_upper": float(upper_bounds[index]),
                    "minimum": float(np.min(finite)) if len(finite) else math.nan,
                    "p05": float(np.quantile(finite, 0.05)) if len(finite) else math.nan,
                    "median": float(np.median(finite)) if len(finite) else math.nan,
                    "p95": float(np.quantile(finite, 0.95)) if len(finite) else math.nan,
                    "maximum": float(np.max(finite)) if len(finite) else math.nan,
                    "mean": float(np.mean(finite)) if len(finite) else math.nan,
                    "population_sd": float(np.std(finite, ddof=0)) if len(finite) else math.nan,
                    "observed_span_fraction": (
                        float((np.max(finite) - np.min(finite)) / span)
                        if len(finite) and span > 0.0 else math.nan
                    ),
                    "below_declared_count": int(np.sum(finite < lower_bounds[index])),
                    "above_declared_count": int(np.sum(finite > upper_bounds[index])),
                })

        if len(attempts):
            frame = attempts.copy()
            frame.insert(0, "block", block)
            attempts_frames.append(frame)
        if len(provenance):
            frame = provenance.copy()
            frame.insert(0, "block", block)
            provenance_frames.append(frame)
        if len(migration):
            frame = migration.copy()
            frame.insert(0, "block", block)
            migration_frames.append(frame)

    return {
        "generation_summary": pd.DataFrame(summary_rows),
        "generation_attempt_disposition": pd.DataFrame(disposition_rows),
        "generation_rejection_reasons": pd.DataFrame(reason_rows),
        "generation_accepted_coverage": pd.DataFrame(coverage_rows),
        "generation_attempt_ledger": (
            pd.concat(attempts_frames, ignore_index=True, sort=False)
            if attempts_frames else pd.DataFrame()
        ),
        "generation_accepted_provenance": (
            pd.concat(provenance_frames, ignore_index=True, sort=False)
            if provenance_frames else pd.DataFrame()
        ),
        "generation_checkpoint_migration": (
            pd.concat(migration_frames, ignore_index=True, sort=False)
            if migration_frames else pd.DataFrame()
        ),
    }


def _timing_tables(
    run_directory: Path,
    snapshots: Sequence[RouteSnapshot],
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categories: dict[str, list[float]] = {
        "raw_inference": [],
        "qp_deployment": [],
        "surrogate_optimization": [],
        "smooth_mechanistic_nlp": [],
        "reference_replay": [],
        "fixed_input_equivalence": [],
        "mechanistic_generation": [],
        "training": [],
    }
    attempt_ledgers_found = False
    for block in GENERATION_BLOCKS:
        path = run_directory / "datasets" / block / "all_attempts.csv"
        if not path.is_file():
            continue
        attempt_ledgers_found = True
        attempts = _safe_csv(path, warnings)
        if "elapsed_seconds" in attempts:
            categories["mechanistic_generation"].extend(
                pd.to_numeric(attempts["elapsed_seconds"], errors="coerce").tolist()
            )
    if not attempt_ledgers_found:
        generation_path = run_directory / "metrics" / "mechanistic_generation_summary.csv"
        if not generation_path.is_file():
            generation_path = run_directory / "metrics" / "mechanistic_generation_audit.csv"
        generation = _safe_csv(generation_path, warnings)
        if "elapsed_seconds" in generation:
            categories["mechanistic_generation"].extend(
                pd.to_numeric(generation["elapsed_seconds"], errors="coerce").tolist()
            )
    ridge = _safe_npz(run_directory / "models" / "ridge_surrogate.npz", warnings)
    if "elapsed_seconds" in ridge:
        categories["training"].extend(
            np.asarray(ridge["elapsed_seconds"], dtype=float).reshape(-1).tolist()
        )
    benchmark_path = run_directory / "metrics" / "inference_timing_batches.csv"
    external = _safe_csv(
        benchmark_path
        if benchmark_path.is_file()
        else run_directory / "metrics" / "timing_events.csv",
        warnings,
    )
    if {"category", "elapsed_seconds"}.issubset(external.columns):
        for category, group in external.groupby("category"):
            # The inference protocol reports latency per test response; all
            # optimization/generation events remain complete-event seconds.
            value_column = (
                "per_response_latency_seconds"
                if str(category) in {"raw_inference", "qp_deployment"}
                and "per_response_latency_seconds" in group.columns
                else "elapsed_seconds"
            )
            categories.setdefault(str(category), []).extend(
                pd.to_numeric(group[value_column], errors="coerce").tolist()
            )
    start_times: dict[str, list[float]] = {"surrogate": [], "direct": []}
    case_times: dict[str, list[float]] = {"surrogate": [], "direct": []}
    reference_times: dict[str, list[float]] = {"surrogate": [], "direct": []}
    for snapshot in snapshots:
        elapsed = _route_elapsed(snapshot)
        if math.isfinite(elapsed):
            case_times[snapshot.route].append(elapsed)
            category = "surrogate_optimization" if snapshot.route == "surrogate" else "smooth_mechanistic_nlp"
            categories[category].append(elapsed)
        for start in snapshot.starts:
            value = sum(
                time for time in (_as_float(stage.get("elapsed_seconds")) for stage in _stages(start))
                if math.isfinite(time)
            )
            if _stages(start):
                start_times[snapshot.route].append(value)
        if snapshot.equivalence is not None:
            comparison = _as_float(snapshot.equivalence.get("elapsed_seconds"))
            if math.isfinite(comparison):
                categories["fixed_input_equivalence"].append(comparison)
            replay_payload = _mapping(snapshot.equivalence.get("reference_replay"))
            replay = _as_float(
                None if replay_payload is None else replay_payload.get("elapsed_seconds")
            )
            if math.isfinite(replay):
                reference_times[snapshot.route].append(replay)
                categories["reference_replay"].append(replay)
    summary = pd.DataFrame([
        {
            "category": category,
            "unit": (
                "seconds_per_response"
                if category in {"raw_inference", "qp_deployment"}
                else "seconds"
            ),
            **_finite_summary(values),
        }
        for category, values in categories.items()
    ])
    workload: list[dict[str, Any]] = []
    for route in ("surrogate", "direct"):
        route_snapshots = [item for item in snapshots if item.route == route]
        attempted = sum(len(item.starts) for item in route_snapshots)
        accepted = sum(sum(_start_feasible(item, start) for start in item.starts) for item in route_snapshots)
        workload.append({
            "route": route,
            "median_start_time_seconds": _finite_summary(start_times[route])["median"],
            "median_complete_case_time_seconds": _finite_summary(case_times[route])["median"],
            "p95_complete_case_time_seconds": _finite_summary(case_times[route])["p95_nearest_rank"],
            "accepted_starts": accepted,
            "attempted_starts": attempted,
            "reference_validation_time_seconds": _finite_summary(reference_times[route])["total"],
        })
    return summary, pd.DataFrame(workload)


def _scenario_comparison(
    snapshots: Sequence[RouteSnapshot],
    geometry: StudyGeometry,
    quality_scale: np.ndarray,
    response_scale: np.ndarray | None,
) -> pd.DataFrame:
    lookup = {(item.case, item.route): item for item in snapshots}
    rows: list[dict[str, Any]] = []
    cases = tuple(dict.fromkeys(item.case for item in snapshots))
    for case in cases:
        surrogate = lookup[(case, "surrogate")]
        direct = lookup[(case, "direct")]
        s_theta, d_theta = _selected_theta(surrogate), _selected_theta(direct)
        s_response = _selected_response_map(surrogate, geometry)
        d_response = _selected_response_map(direct, geometry)

        def objective(theta: np.ndarray | None, values: Mapping[str, np.ndarray]) -> float:
            if theta is None or "reference" not in values:
                return math.nan
            return _response_quantities(theta, values["reference"], geometry, quality_scale)[2]

        s_objective = objective(s_theta, s_response)
        d_objective = objective(d_theta, d_response)

        def error(values: Mapping[str, np.ndarray], method: str) -> float:
            if method not in values or "reference" not in values or response_scale is None:
                return math.nan
            return float(np.sqrt(np.mean(((values[method] - values["reference"]) / response_scale) ** 2)))

        s_time, d_time = _route_elapsed(surrogate), _route_elapsed(direct)
        rows.append({
            "case": case,
            "J_S_reference": s_objective,
            "J_M_reference": d_objective,
            "delta_J_S_minus_M": s_objective - d_objective,
            "projected_reference_nrmse_at_S": error(s_response, "projected"),
            "smooth_reference_nrmse_at_M": error(d_response, "smooth"),
            "surrogate_time_seconds": s_time,
            "mechanistic_time_seconds": d_time,
            "mechanistic_surrogate_time_ratio": (
                d_time / s_time if math.isfinite(s_time) and math.isfinite(d_time) and s_time > 0.0 else math.nan
            ),
        })
    return pd.DataFrame(rows)


def _model_response_scale(run_directory: Path, geometry: StudyGeometry, warnings: list[str]) -> np.ndarray | None:
    model = _safe_npz(run_directory / "models" / "ridge_surrogate.npz", warnings)
    scale = model.get("response_scale")
    if scale is None or scale.shape != (geometry.response_count,) or np.any(scale <= 0.0):
        return None
    return np.asarray(scale, dtype=float)


def build_reporting_tables(
    run_directory: str | Path,
    *,
    expected_cases: Sequence[str] | None = None,
) -> ReportingBundle:
    """Build every v3 result table from a non-mutating artifact snapshot.

    Parameters
    ----------
    run_directory:
        Root containing ``datasets/``, ``metrics/``, and ``optimization/``.
    expected_cases:
        Optional explicit denominator.  By default, the nominal case and every
        row in the effective design's ``robustness_influents`` array are retained.
    """

    run = Path(run_directory).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run}")
    warnings: list[str] = []
    cases = _expected_cases(run, expected_cases, warnings)
    geometry = _infer_geometry(run, warnings)
    snapshots = tuple(
        _route_snapshot(run, case, route, warnings)
        for case in cases
        for route in ("surrogate", "direct")
    )
    quality_scale = _quality_scale(run, geometry, warnings)
    response_scale = _model_response_scale(run, geometry, warnings)
    objectives, engineering, quality = _objective_engineering_tables(
        snapshots, geometry, quality_scale,
    )
    controls = _controls_table(snapshots)
    profiles = _profile_rows(snapshots, geometry)
    comparison = _scenario_comparison(
        snapshots, geometry, quality_scale, response_scale,
    )
    reconstructed = _reconstruct_selected_violations(
        run, snapshots, cases, geometry, warnings,
    )
    physical_detail = _physical_detail(run, reconstructed, warnings)
    case_status, failure_accounting = _case_and_failure_tables(snapshots, cases)
    generation = _generation_tables(run, warnings)
    timing_summary, timing_workload = _timing_tables(run, snapshots, warnings)
    ridge = _safe_csv(run / "metrics" / "ridge_cross_validation.csv", warnings)
    prediction = _safe_csv(run / "metrics" / "untouched_prediction_metrics.csv", warnings)
    projection = _safe_csv(run / "metrics" / "projection_qp_diagnostics.csv", warnings)
    test_equivalence = _safe_csv(
        run / "metrics" / "smooth_reference_equivalence_test.csv", warnings,
    )
    tables: dict[str, pd.DataFrame] = {
        **generation,
        "ridge_selection": ridge,
        "prediction_metrics": prediction,
        "projection_diagnostics": projection,
        "smooth_reference_equivalence_test": test_equivalence,
        "trust_diagnostics": _trust_table(run, snapshots, warnings),
        "route_status": _route_status_table(snapshots),
        "active_constraints": _active_constraint_table(snapshots),
        "selected_controls": controls,
        "nominal_controls": controls[controls["case"] == "nominal"].reset_index(drop=True),
        "scenario_controls": controls[controls["case"] != "nominal"].reset_index(drop=True),
        "objective_decomposition": objectives,
        "engineering_quantities": engineering,
        "selected_quality": quality,
        "nominal_quality": quality[quality["case"] == "nominal"].reset_index(drop=True),
        "scenario_quality": quality[quality["case"] != "nominal"].reset_index(drop=True),
        "process_profiles": profiles,
        "nominal_profiles": profiles[profiles["case"] == "nominal"].reset_index(drop=True),
        "smooth_reference_equivalence": _equivalence_table(snapshots),
        "nominal_comparison": comparison[comparison["case"] == "nominal"].reset_index(drop=True),
        "scenario_comparison": comparison[comparison["case"] != "nominal"].reset_index(drop=True),
        "case_status": case_status,
        "failure_accounting": failure_accounting,
        "timing_summary": timing_summary,
        "timing_workload": timing_workload,
        "physical_violation_detail": physical_detail,
        "physical_violation_summary": _physical_summary(physical_detail),
    }
    return ReportingBundle(run, cases, tables, tuple(dict.fromkeys(warnings)))


def write_reporting_tables(
    run_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
    expected_cases: Sequence[str] | None = None,
) -> ReportingBundle:
    """Build and atomically write a v3 reporting bundle."""

    bundle = build_reporting_tables(run_directory, expected_cases=expected_cases)
    bundle.write(output_directory)
    return bundle


__all__ = [
    "CONTROL_NAMES",
    "ENGINEERING_QUANTITY_NAMES",
    "FAILURE_CLASSES",
    "OBJECTIVE_COMPONENT_NAMES",
    "PENDING_CLASS",
    "REQUIRED_PHYSICAL_METHODS",
    "ReportingBundle",
    "StudyGeometry",
    "TRUST_DIAGNOSTICS",
    "build_reporting_tables",
    "write_reporting_tables",
]
