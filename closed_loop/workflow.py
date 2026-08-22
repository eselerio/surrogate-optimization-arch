"""Immutable, staged execution of the closed-loop simultaneous-NLP study.

The workflow deliberately keeps the development, calibration, and assessment
blocks separate.  Optimization uses one nine-start combined nonlinear program
and calls the nonsmoothed BDF model once for independent validation of each
selected point.  No deployment QP, DIRECT search, or mechanistic reference
search is part of this implementation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import threading
from time import perf_counter_ns, sleep
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from . import model as mechanism
from . import surrogate as surrogate_core
from .surrogate import (
    LeastSquaresDiagnostics,
    QuadraticFeatureMap,
    QuadraticSurrogate,
)


STAGES: tuple[str, ...] = (
    "static",
    "pilot",
    "dataset",
    "fit",
    "calibration",
    "assessment",
    "nlp_preflight",
    "optimization",
    "report",
    "complete",
)
WORKFLOW_SCHEMA_VERSION = 4
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_ATOMIC_REPLACE_RETRY = os.name == "nt"
_WINDOWS_TRANSIENT_REPLACE_ERRORS = frozenset((5, 32))
_WINDOWS_ATOMIC_REPLACE_DELAYS_SECONDS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)


class WorkflowError(RuntimeError):
    """Base error for a scientifically invalid or incomplete run."""


class ContractMismatchError(WorkflowError):
    """An existing run belongs to a different immutable contract."""


class ImmutableRunError(WorkflowError):
    """A sealed run was asked to execute again."""


class StageExecutionError(WorkflowError):
    """A stage failed its declared scientific acceptance contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_array_sha256(values: Any) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return sha256(canonical.tobytes(order="C")).hexdigest()


def _replace_atomic_bytes(temporary: Path, destination: Path) -> None:
    """Replace one completed byte artifact, retrying only transient Windows locks."""

    for attempt in range(len(_WINDOWS_ATOMIC_REPLACE_DELAYS_SECONDS) + 1):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as exc:
            transient_windows_lock = (
                _WINDOWS_ATOMIC_REPLACE_RETRY
                and getattr(exc, "winerror", None)
                in _WINDOWS_TRANSIENT_REPLACE_ERRORS
            )
            if not transient_windows_lock or attempt == len(
                _WINDOWS_ATOMIC_REPLACE_DELAYS_SECONDS
            ):
                raise
            sleep(_WINDOWS_ATOMIC_REPLACE_DELAYS_SECONDS[attempt])


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_atomic_bytes(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False)
    _atomic_bytes(path, payload.encode("utf-8") + b"\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise WorkflowError(f"{path} must contain a JSON object")
    return value


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    finite = np.sort(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must lie in (0, 1]")
    return float(finite[math.ceil(probability * finite.size) - 1])


def _maximin_robustness_indices(
    influents: np.ndarray,
    nominal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int = 10,
) -> np.ndarray:
    """Select the deterministic domain-spanning robustness preflight panel."""

    values = np.asarray(influents, dtype=np.float64)
    reference = np.asarray(nominal, dtype=np.float64)
    spans = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != reference.size or spans.shape != reference.shape:
        raise ValueError("maximin panel arrays have inconsistent shapes")
    if np.any(spans <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("maximin panel requires finite rows and positive spans")
    wanted = min(int(count), values.shape[0])
    if wanted <= 0:
        return np.empty(0, dtype=np.int64)
    standardized = (values - lower) / spans
    chosen_points = [((reference - lower) / spans).copy()]
    selected: list[int] = []
    available = np.ones(values.shape[0], dtype=bool)
    for _ in range(wanted):
        squared = np.stack(
            [np.sum(np.square(standardized - point[None, :]), axis=1) for point in chosen_points],
            axis=1,
        )
        score = np.min(squared, axis=1)
        score[~available] = -np.inf
        index = int(np.argmax(score))  # first occurrence is the stipulated lower-index tie rule
        selected.append(index)
        available[index] = False
        chosen_points.append(standardized[index])
    return np.asarray(selected, dtype=np.int64)


def _peak_resident_memory_bytes() -> int:
    try:
        import psutil

        memory = psutil.Process().memory_info()
        return int(getattr(memory, "peak_wset", memory.rss))
    except Exception:
        return 0


def _resident_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


class _StageMemoryMonitor:
    """Sample resident memory while one stage runs.

    ``resource.ru_maxrss`` and the Windows ``peak_wset`` value are process-wide
    high-water marks.  Reusing either value for every stage makes later stages
    inherit peaks caused by earlier work.  This small sampler instead records
    the maximum RSS actually observed between a stage's start and finish.
    """

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = float(interval_seconds)
        self.maximum_bytes = _resident_memory_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.maximum_bytes = max(self.maximum_bytes, _resident_memory_bytes())

    def start(self) -> "_StageMemoryMonitor":
        self._thread.start()
        return self

    def stop(self) -> int:
        self.maximum_bytes = max(self.maximum_bytes, _resident_memory_bytes())
        self._stop.set()
        self._thread.join()
        return int(self.maximum_bytes)


def _physical_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def _target_columns() -> tuple[str, ...]:
    names = [f"m:{name}" for name in mechanism.COMPONENTS]
    for stage in range(1, mechanism.N_STAGES + 1):
        names.extend(f"c{stage}:{name}" for name in mechanism.COMPONENTS)
    names.extend(f"gE:{name}" for name in mechanism.COMPONENTS)
    names.extend(f"gU:{name}" for name in mechanism.COMPONENTS)
    names.extend(f"layer:{index}" for index in range(1, mechanism.N_LAYERS + 1))
    return tuple(names)


DERIVED_ASSESSMENT_NAMES: tuple[str, ...] = (
    "overflow_cod",
    "overflow_tn",
    "overflow_tp",
    "overflow_tss",
    "underflow_cod",
    "underflow_tn",
    "underflow_tp",
    "underflow_tss",
    "normalized_clarifier_inventory",
)


def _derived_assessment_responses(
    decisions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Recover the nine concentration/inventory responses reported downstream."""

    theta = np.asarray(decisions, dtype=np.float64)
    chi = np.asarray(targets, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 5 or chi.shape != (theta.shape[0], 170):
        raise ValueError("derived responses require matched (n,5) and (n,170) arrays")
    q_effluent = 1.0 - theta[:, 4]
    q_underflow = theta[:, 3] + theta[:, 4]
    if np.any(q_effluent <= 0.0) or np.any(q_underflow <= 0.0):
        raise ValueError("derived outlet concentrations require positive outlet flows")
    overflow = chi[:, 120:140] / q_effluent[:, None]
    underflow = chi[:, 140:160] / q_underflow[:, None]
    overflow_composites = overflow @ mechanism.COMPOSITE_MATRIX.T
    underflow_composites = underflow @ mechanism.COMPOSITE_MATRIX.T
    normalized_inventory = (
        mechanism.CLARIFIER.layer_volume
        * np.sum(chi[:, 160:170], axis=1)
        / mechanism.CLARIFIER.fresh_flow
    )
    values = np.column_stack(
        (overflow_composites, underflow_composites, normalized_inventory)
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("derived assessment responses must be finite")
    return values


def _derived_metric_frame(
    truth: np.ndarray,
    prediction: np.ndarray,
    development_scale: np.ndarray,
    *,
    variance_relative_tolerance: float,
) -> pd.DataFrame:
    """Apply the manuscript's raw-response metric definitions to derived outputs."""

    observed = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    scale = np.asarray(development_scale, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("derived truth and prediction matrices must have matching shapes")
    if observed.shape[1] != len(DERIVED_ASSESSMENT_NAMES) or scale.shape != (
        len(DERIVED_ASSESSMENT_NAMES),
    ):
        raise ValueError("derived metric dimensions changed")
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("development-derived response scales must be finite and positive")
    error = predicted - observed
    squared = np.square(error)
    rmse = np.sqrt(np.mean(squared, axis=0))
    centered = observed - np.mean(observed, axis=0)
    denominator = np.sum(np.square(centered), axis=0)
    assessment_scale = np.sqrt(np.mean(np.square(centered), axis=0))
    reference = np.maximum(1.0, np.max(np.abs(observed), axis=0))
    defined = assessment_scale > variance_relative_tolerance * reference
    r_squared = np.full(scale.shape, np.nan, dtype=np.float64)
    r_squared[defined] = 1.0 - np.sum(squared[:, defined], axis=0) / denominator[defined]
    return pd.DataFrame(
        {
            "response": DERIVED_ASSESSMENT_NAMES,
            "development_scale": scale,
            "rmse": rmse,
            "mae": np.mean(np.abs(error), axis=0),
            "bias": np.mean(error, axis=0),
            "nrmse": rmse / scale,
            "r_squared": r_squared,
            "r_squared_defined": defined,
        }
    )


@dataclass(frozen=True)
class MechanisticRow:
    index: int
    decisions: np.ndarray
    influent: np.ndarray
    target: np.ndarray
    accepted: bool
    elapsed_seconds: float
    diagnostics: dict[str, Any]
    error: str | None = None


def _solution_diagnostics(result: Any) -> dict[str, Any]:
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        start=int(result.start),
        nfev=int(result.nfev),
        cost=float(result.cost),
        route=str(result.route),
        integration_time_days=float(result.integration_time_days),
        integration_steps=int(result.integration_steps),
    )
    return diagnostics


def _solve_payload(payload: tuple[int, np.ndarray, np.ndarray, dict[str, Any]]) -> MechanisticRow:
    index, decisions, influent, settings = payload
    started = perf_counter_ns()
    try:
        from threadpoolctl import threadpool_limits

        relaxation = settings.get("dynamic_relaxation", {})
        with threadpool_limits(limits=1):
            result = mechanism.solve_steady_state(
                mechanism.OperatingPoint(*[float(value) for value in decisions]),
                influent,
                max_nfev=int(settings["steady_max_residual_evaluations"]),
                tolerance=float(settings["steady_xtol"]),
                acceptance_tolerance=float(settings["scaled_derivative_acceptance_d_inv"]),
                minimum_relaxation_days=float(relaxation.get("minimum_horizon_d", 400.0)),
                solids_turnovers=float(relaxation.get("waste_sludge_turnovers", 50.0)),
                integration_rtol=float(relaxation.get("relative_tolerance", 1.0e-7)),
                integration_atol=float(relaxation.get("scaled_absolute_tolerance", 1.0e-9)),
            )
        accepted = bool(result.accepted)
        return MechanisticRow(
            index=index,
            decisions=np.asarray(decisions, dtype=np.float64).copy(),
            influent=np.asarray(influent, dtype=np.float64).copy(),
            target=np.asarray(result.target, dtype=np.float64),
            accepted=accepted,
            elapsed_seconds=(perf_counter_ns() - started) / 1.0e9,
            diagnostics=_json_value(_solution_diagnostics(result)),
            error=None if accepted else "mechanistic acceptance contract failed",
        )
    except Exception as exc:
        return MechanisticRow(
            index=index,
            decisions=np.asarray(decisions, dtype=np.float64).copy(),
            influent=np.asarray(influent, dtype=np.float64).copy(),
            target=np.full(mechanism.TARGET_SIZE, np.nan),
            accepted=False,
            elapsed_seconds=(perf_counter_ns() - started) / 1.0e9,
            diagnostics={},
            error=f"{type(exc).__name__}: {exc}",
        )


def save_surrogate_bundle(
    path: Path,
    model: QuadraticSurrogate,
    **extras: np.ndarray,
) -> None:
    feature = model.feature_map
    arrays = {
        "decision_center": feature.decision_center,
        "decision_scale": feature.decision_scale,
        "influent_center": feature.influent_center,
        "influent_scale": feature.influent_scale,
        "term_center": feature.term_center,
        "term_scale": feature.term_scale,
        "response_center": model.response_center,
        "response_scale": model.response_scale,
        "coefficients": model.coefficients,
        "feature_qr_upper": model.feature_qr_upper,
        "feature_qr_pivots": model.feature_qr_pivots,
        **{name: np.asarray(value) for name, value in extras.items()},
    }
    atomic_npz(path, **arrays)
    atomic_json(
        path.with_suffix(".json"),
        {
            "diagnostics": model.diagnostics.as_dict(),
            "feature_count": model.feature_map.feature_count,
            "response_count": model.response_count,
            "npz_sha256": sha256_file(path),
        },
    )


def load_surrogate_bundle(
    path: Path,
) -> tuple[QuadraticSurrogate, None, dict[str, np.ndarray]]:
    metadata = _load_json(path.with_suffix(".json"))
    if metadata.get("npz_sha256") != sha256_file(path):
        raise ContractMismatchError(f"surrogate bundle {path.name} failed its digest check")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    feature_map = QuadraticFeatureMap(
        decision_center=arrays.pop("decision_center"),
        decision_scale=arrays.pop("decision_scale"),
        influent_center=arrays.pop("influent_center"),
        influent_scale=arrays.pop("influent_scale"),
        term_center=arrays.pop("term_center"),
        term_scale=arrays.pop("term_scale"),
    )
    model = QuadraticSurrogate(
        feature_map=feature_map,
        response_center=arrays.pop("response_center"),
        response_scale=arrays.pop("response_scale"),
        coefficients=arrays.pop("coefficients"),
        diagnostics=LeastSquaresDiagnostics(**metadata["diagnostics"]),
        feature_qr_upper=arrays.pop("feature_qr_upper"),
        feature_qr_pivots=arrays.pop("feature_qr_pivots"),
    )
    if "equality_scale" in arrays or "inequality_scale" in arrays:
        raise ContractMismatchError(
            "surrogate bundle contains retired network-projection scales"
        )
    return model, None, arrays


class ClosedLoopWorkflow:
    """Run, resume, audit, and seal one configured scientific calculation."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        profile: str,
        run_id: str,
        repository_root: str | Path | None = None,
        results_root: str | Path | None = None,
        mechanistic_solver: Callable[[int, np.ndarray, np.ndarray], MechanisticRow] | None = None,
        nlp_backend: Any | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.repository_root = Path(
            repository_root or self.config_path.parent.parent
        ).resolve()
        self.config = _load_json(self.config_path)
        profiles = self.config.get("design", {}).get("profiles", {})
        if profile not in profiles:
            raise WorkflowError(f"unknown execution profile {profile!r}")
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise WorkflowError("run id must begin alphanumerically and contain only letters, digits, ._-")
        self.profile_name = profile
        self.profile = dict(profiles[profile])
        self.run_id = run_id
        configured_root = Path(self.config["execution"]["results_root"])
        root = Path(results_root) if results_root is not None else self.repository_root / configured_root
        self.results_root = root.resolve()
        self.run_root = self.results_root / run_id
        self.manifest_path = self.run_root / "manifest.json"
        self.completion_path = self.run_root / "COMPLETED.json"
        self.mechanistic_solver = mechanistic_solver
        self._nlp_backend = nlp_backend
        self._manifest: dict[str, Any] = {}
        self._ledger_next_sequence: int | None = None
        self._ledger_cache: list[dict[str, Any]] | None = None
        self._contract = self._build_contract()

    @property
    def sample_count(self) -> int:
        return sum(int(block["count"]) for block in self.profile["blocks"].values())

    @property
    def development_count(self) -> int:
        return int(self.profile["blocks"]["development"]["count"])

    @property
    def calibration_count(self) -> int:
        return int(self.profile["blocks"]["calibration"]["count"])

    @property
    def assessment_count(self) -> int:
        return int(self.profile["blocks"]["assessment"]["count"])

    def _nlp(self) -> Any:
        return self._nlp_backend or importlib.import_module("closed_loop.nlp")

    @property
    def _ledger_root(self) -> Path:
        return self.run_root / "optimization" / "invocation_ledger"

    def _ledger_events(self) -> list[dict[str, Any]]:
        if self._ledger_cache is not None:
            return self._ledger_cache
        events: list[dict[str, Any]] = []
        for expected, path in enumerate(sorted(self._ledger_root.glob("event_*.json")), start=1):
            event = _load_json(path)
            if int(event.get("sequence", -1)) != expected:
                raise ContractMismatchError("invocation ledger sequence is not contiguous")
            if event.get("contract_sha256") != self._contract["contract_sha256"]:
                raise ContractMismatchError("invocation ledger belongs to another contract")
            events.append(event)
        self._ledger_cache = events
        return self._ledger_cache

    def _ledger_event(
        self,
        kind: str,
        logical_key: str,
        state: str,
        **details: Any,
    ) -> dict[str, Any]:
        if state not in {"attempted", "completed", "reused", "interrupted"}:
            raise ValueError("invalid invocation-ledger state")
        self._ledger_root.mkdir(parents=True, exist_ok=True)
        if self._ledger_next_sequence is None:
            self._ledger_next_sequence = len(self._ledger_events()) + 1
        sequence = self._ledger_next_sequence
        event = {
            "sequence": sequence,
            "timestamp_utc": utc_now(),
            "contract_sha256": self._contract["contract_sha256"],
            "kind": str(kind),
            "logical_key": str(logical_key),
            "state": state,
            **_json_value(details),
        }
        path = self._ledger_root / f"event_{sequence:08d}.json"
        if path.exists():
            raise ContractMismatchError("invocation ledger would overwrite an existing event")
        atomic_json(path, event)
        if self._ledger_cache is None:
            self._ledger_cache = []
        self._ledger_cache.append(event)
        self._ledger_next_sequence += 1
        return event

    def _last_ledger_event(self, kind: str, logical_key: str) -> dict[str, Any] | None:
        for event in reversed(self._ledger_events()):
            if event["kind"] == kind and event["logical_key"] == logical_key:
                return event
        return None

    def _reconcile_interrupted_invocation(
        self, kind: str, logical_key: str, **details: Any,
    ) -> None:
        previous = self._last_ledger_event(kind, logical_key)
        if previous is not None and previous["state"] == "attempted":
            self._ledger_event(
                kind,
                logical_key,
                "interrupted",
                reason="attempt had no durable completion event",
                **details,
            )

    def _invocation_summary(self) -> dict[str, Any]:
        events = self._ledger_events()
        result: dict[str, Any] = {"event_count": len(events), "kinds": {}}
        for kind in sorted({str(event["kind"]) for event in events}):
            selected = [event for event in events if event["kind"] == kind]
            keys = {str(event["logical_key"]) for event in selected}
            result["kinds"][kind] = {
                "logical_calls": len(keys),
                **{
                    state: sum(event["state"] == state for event in selected)
                    for state in ("attempted", "completed", "reused", "interrupted")
                },
            }
        return result

    def _case_cache_identity(self, case: Any) -> dict[str, str]:
        parameter_sha256 = _float_array_sha256(case.parameter_vector())
        payload = {
            "contract_sha256": self._contract["contract_sha256"],
            "case_id": str(case.case_id),
            "case_parameter_sha256": parameter_sha256,
        }
        return {
            **payload,
            "case_cache_key": sha256_bytes(canonical_json(payload)),
        }

    def _start_cache_identity(
        self, case: Any, initial_point: np.ndarray, start_index: int,
    ) -> dict[str, Any]:
        case_identity = self._case_cache_identity(case)
        payload = {
            **case_identity,
            "start_index": int(start_index),
            "initial_point_sha256": _float_array_sha256(initial_point),
        }
        return {
            **payload,
            "start_cache_key": sha256_bytes(canonical_json(payload)),
        }

    def _exact_cache_identity(self, case: Any, decisions: np.ndarray) -> dict[str, str]:
        case_identity = self._case_cache_identity(case)
        payload = {
            **case_identity,
            "selected_controls_sha256": _float_array_sha256(decisions),
        }
        return {
            **payload,
            "exact_cache_key": sha256_bytes(canonical_json(payload)),
        }

    def _scientific_workloads(self, profile_name: str = "full") -> dict[str, int]:
        profile = self.config["design"]["profiles"][profile_name]
        sensitivity = len(self._resolved_sensitivity_cases(profile_name))
        robustness = int(profile["robustness"]["count"])
        sample_count = sum(int(block["count"]) for block in profile["blocks"].values())
        cases = 1 + robustness + sensitivity
        starts = int(self.config.get("optimization", {}).get("multistart", {}).get("count", 9))
        return {
            "dataset_bdf_invocations": sample_count,
            "selected_point_bdf_invocations_maximum": cases,
            "bdf_invocations_maximum": sample_count + cases,
            "combined_nlp_starts": starts * cases,
            "optimization_cases": cases,
            "robustness_cases": robustness,
            "sensitivity_cases": sensitivity,
            "physical_qp_evaluations": 0,
            "direct_evaluations": 0,
        }

    def _build_contract(self) -> dict[str, Any]:
        import casadi
        import scipy

        module_names = ("model", "surrogate", "design", "nlp", "workflow")
        module_hashes = {
            name: sha256_file(self.repository_root / "closed_loop" / f"{name}.py")
            for name in module_names
        }
        lock = self.repository_root / "uv.lock"
        if not lock.is_file():
            raise WorkflowError("uv.lock is required for the immutable numerical contract")
        casadi_root = Path(casadi.__file__).resolve().parent
        binary_patterns = (
            "_casadi.pyd",
            "libcasadi.dll",
            "libcasadi_nlpsol_ipopt.dll",
            "libipopt-*.dll",
            "libcoinmumps-*.dll",
        )
        binaries: dict[str, str] = {}
        for pattern in binary_patterns:
            for path in sorted(casadi_root.glob(pattern)):
                binaries[path.name] = sha256_file(path)
        if not any("ipopt" in name.lower() for name in binaries):
            raise WorkflowError("the CasADi/IPOPT binary could not be bound into the run contract")
        if not any("mumps" in name.lower() for name in binaries):
            raise WorkflowError("the IPOPT/MUMPS binary could not be bound into the run contract")
        value = {
            "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
            "profile": self.profile_name,
            "profile_settings": self.profile,
            "config_sha256": sha256_file(self.config_path),
            "module_sha256": module_hashes,
            "dependency_lock": {"path": "uv.lock", "sha256": sha256_file(lock)},
            "numerical_environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pandas": pd.__version__,
                "casadi": casadi.__version__,
            },
            "casadi_ipopt_binary_sha256": binaries,
            "host": {
                "node": platform.node(),
                "processor": platform.processor(),
                "platform": platform.platform(),
                "physical_memory_bytes": _physical_memory_bytes(),
            },
            "mechanistic_solver_override": (
                None
                if self.mechanistic_solver is None
                else f"{self.mechanistic_solver.__module__}.{getattr(self.mechanistic_solver, '__qualname__', type(self.mechanistic_solver).__qualname__)}"
            ),
            "ordered_blocks": {
                "development": [0, self.development_count],
                "calibration": [self.development_count, self.development_count + self.calibration_count],
                "assessment": [self.development_count + self.calibration_count, self.sample_count],
            },
        }
        value["contract_sha256"] = sha256_bytes(canonical_json(value))
        return value

    def _initialize(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        for directory in (
            "inputs", "checks", "datasets/chunks", "splits", "models", "predictions",
            "metrics", "optimization/cache", "optimization/invocation_ledger",
            "tables", "figures", "report", "timing",
        ):
            (self.run_root / directory).mkdir(parents=True, exist_ok=True)
        if self.completion_path.exists():
            raise ImmutableRunError(f"run {self.run_id!r} is already sealed and immutable")
        if self.manifest_path.exists():
            self._manifest = _load_json(self.manifest_path)
            previous = self._manifest.get("contract", {}).get("contract_sha256")
            if previous != self._contract["contract_sha256"]:
                raise ContractMismatchError("the run id belongs to a different configuration or implementation")
        else:
            self._manifest = {
                "schema_version": WORKFLOW_SCHEMA_VERSION,
                "run_id": self.run_id,
                "profile": self.profile_name,
                "article_eligible": bool(self.profile.get("article_eligible", False)),
                "status": "initialized",
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
                "contract": self._contract,
                "stages": {},
            }
            self._write_manifest()

    def _write_manifest(self) -> None:
        self._manifest["updated_utc"] = utc_now()
        atomic_json(self.manifest_path, self._manifest)

    def _stage_marker(self, stage: str) -> Path:
        return self.run_root / "checks" / f"stage_{stage}.json"

    def _artifact_hashes(self, paths: Iterable[Path]) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in sorted({value.resolve() for value in paths}):
            if not path.is_relative_to(self.run_root) or not path.is_file():
                raise StageExecutionError(f"required stage artifact is missing: {path}")
            result[path.relative_to(self.run_root).as_posix()] = sha256_file(path)
        if not result:
            raise StageExecutionError("a completed stage must bind at least one artifact")
        return result

    def _stage_artifacts(self, stage: str) -> tuple[Path, ...]:
        fixed: dict[str, tuple[str, ...]] = {
            "static": ("inputs/resolved_config.json", "inputs/contract.json", "datasets/design.npz", "splits/ordered_blocks.json", "checks/mechanistic_matrix_audit.json", "checks/scientific_workload.json"),
            "pilot": ("timing/pilot_summary.json",),
            "dataset": ("datasets/mechanistic_dataset.npz", "datasets/mechanistic_dataset.parquet", "checks/dataset_validation.json"),
            "fit": ("models/development_surrogate.npz", "models/development_surrogate.json", "models/development_assets.json"),
            "calibration": ("metrics/calibration.json", "predictions/calibration_scores.npz"),
            "assessment": ("metrics/assessment_summary.json", "metrics/assessment_coordinate_metrics.csv", "metrics/assessment_derived_metrics.csv", "predictions/assessment_predictions.npz"),
            "nlp_preflight": ("optimization/cases.json", "optimization/robustness_design.npz", "optimization/ordered_starts.npz", "optimization/ordered_starts.json", "timing/nlp_preflight.json"),
            "optimization": ("optimization/case_summary.parquet", "optimization/summary.json"),
            "report": ("report/summary.json", "timing/report_generation.json", "tables/workload.csv", "tables/timing_summary.csv", "tables/robustness_summary.csv", "tables/combined_route_outcomes.csv", "tables/bound_activity.csv", "tables/underflow_tss_sensitivity.csv", "tables/objective_weight_sensitivity.csv", "figures/assessment_parity.png", "figures/assessment_parity.pdf", "figures/optimization_objectives.png", "figures/optimization_objectives.pdf"),
            "complete": ("checks/terminal_replay.json",),
        }
        paths = [self.run_root / name for name in fixed.get(stage, ())]
        if stage == "pilot":
            pilot_count = min(
                int(self.config["execution"]["computational_feasibility"]["bdf_serial_preflight_rows"]),
                self.sample_count,
            )
            for path in sorted((self.run_root / "datasets" / "chunks").glob("*")):
                match = re.fullmatch(r"rows_(\d{6})_(\d{6})(?:\.diagnostics)?\.(?:npz|json)", path.name)
                if match and int(match.group(2)) <= pilot_count:
                    paths.append(path)
        if stage == "dataset":
            paths.extend(sorted((self.run_root / "datasets" / "chunks").glob("*")))
        if stage == "nlp_preflight":
            timing_path = self.run_root / "timing" / "nlp_preflight.json"
            if timing_path.is_file():
                for case_id in _load_json(timing_path).get("panel_case_ids", []):
                    root = self.run_root / "optimization" / "cache" / str(case_id)
                    paths.extend((root / "starts.npz", root / "starts.json"))
        if stage == "optimization":
            paths.extend(sorted((self.run_root / "optimization" / "cache").rglob("*")))
            paths.extend(sorted(self._ledger_root.glob("event_*.json")))
        return tuple(path for path in paths if path.is_file() or path in [self.run_root / name for name in fixed.get(stage, ())])

    def _is_stage_complete(self, stage: str) -> bool:
        record = self._manifest.get("stages", {}).get(stage, {})
        if record.get("status") != "complete":
            return False
        marker = self._stage_marker(stage)
        if not marker.is_file() or sha256_file(marker) != record.get("marker_sha256"):
            raise ContractMismatchError(f"completed stage {stage!r} lost its immutable marker")
        payload = _load_json(marker)
        if payload.get("contract_sha256") != self._contract["contract_sha256"]:
            raise ContractMismatchError(f"completed stage {stage!r} has a different contract")
        for relative, expected in payload.get("required_artifact_sha256", {}).items():
            path = (self.run_root / relative).resolve()
            if not path.is_relative_to(self.run_root) or not path.is_file() or sha256_file(path) != expected:
                raise ContractMismatchError(f"completed stage {stage!r} artifact changed: {relative}")
        return True

    def _mark_stage(self, stage: str, status: str, **details: Any) -> None:
        record = {"status": status, "updated_utc": utc_now(), **_json_value(details)}
        if status == "complete":
            artifacts = self._artifact_hashes(self._stage_artifacts(stage))
            marker_payload = {
                "stage": stage,
                "status": status,
                "contract_sha256": self._contract["contract_sha256"],
                "completed_utc": utc_now(),
                "required_artifact_sha256": artifacts,
                **_json_value(details),
            }
            atomic_json(self._stage_marker(stage), marker_payload)
            record["marker_sha256"] = sha256_file(self._stage_marker(stage))
            record["required_artifact_count"] = len(artifacts)
        self._manifest.setdefault("stages", {})[stage] = record
        self._manifest["status"] = "failed" if status == "failed" else f"through_{stage}"
        self._write_manifest()

    def run(self, *, through: str = "complete") -> dict[str, Any]:
        if through not in STAGES:
            raise WorkflowError(f"through must be one of {', '.join(STAGES)}")
        self._initialize()
        for stage in STAGES[: STAGES.index(through) + 1]:
            if self._is_stage_complete(stage):
                if stage == "complete" and not self.completion_path.exists():
                    self._finalize_seal()
                continue
            self._mark_stage(stage, "running")
            stage_started = perf_counter_ns()
            memory_monitor = _StageMemoryMonitor().start()
            try:
                details = getattr(self, f"_stage_{stage}")()
            except Exception as exc:
                stage_high_water = memory_monitor.stop()
                failure = exc if isinstance(exc, WorkflowError) else StageExecutionError(
                    f"stage {stage!r} failed with {type(exc).__name__}: {exc}"
                )
                self._mark_stage(
                    stage,
                    "failed",
                    error=f"{type(failure).__name__}: {failure}",
                    stage_high_water_resident_memory_bytes=stage_high_water,
                )
                if failure is exc:
                    raise
                raise failure from exc
            stage_high_water = memory_monitor.stop()
            completed_details = dict(details or {})
            completed_details.update(
                stage_wall_seconds=(perf_counter_ns() - stage_started) / 1.0e9,
                stage_resident_memory_bytes=_resident_memory_bytes(),
                stage_high_water_resident_memory_bytes=stage_high_water,
            )
            self._mark_stage(stage, "complete", **completed_details)
            if stage == "complete":
                self._finalize_seal()
        return self._manifest

    def _profile_seeds(self) -> tuple[int, int, int, int]:
        blocks = self.profile["blocks"]
        return (
            int(blocks["development"]["seed"]),
            int(blocks["calibration"]["seed"]),
            int(blocks["assessment"]["seed"]),
            int(self.profile["robustness"]["seed"]),
        )

    def _resolved_sensitivity_cases(self, profile_name: str | None = None) -> tuple[dict[str, Any], ...]:
        selected = profile_name or self.profile_name
        sensitivity = self.config.get("sensitivity", {})
        if not sensitivity:
            return ()
        cases: list[dict[str, Any]] = []
        for case in sensitivity["underflow_tss"]["cases"]:
            if not case.get("reuse_primary_case", False):
                cases.append({"case_id": str(case["case_id"]), "case_class": "sensitivity", "sensitivity_family": "underflow_tss", "underflow_tss_limit": float(case["limit_g_m3"]), "weights": None})
        objective = self.config["objective"]
        names = ("H_weight", "a_weight", "r_I_weight", "r_R_weight", "wasted_solids_weight")
        primary = {name: float(objective[name]) for name in names}
        quality = float(sensitivity["objective_weights"]["quality_weight_fixed"])
        total = float(sensitivity["objective_weights"]["nonquality_weight_total"])
        for case in sensitivity["objective_weights"]["cases"]:
            focal = str(case["focal_weight"])
            factor = float(case["factor"])
            focal_value = primary[focal] * factor
            other_scale = (total - focal_value) / (total - primary[focal])
            resolved = {name: (focal_value if name == focal else primary[name] * other_scale) for name in names}
            cases.append({"case_id": str(case["case_id"]), "case_class": "sensitivity", "sensitivity_family": "objective_weights", "underflow_tss_limit": float(self.config["upper_constraints"]["underflow_tss_max_g_m3"]), "weights": [quality, resolved["H_weight"], resolved["a_weight"], resolved["r_I_weight"], resolved["r_R_weight"], resolved["wasted_solids_weight"]], "focal_weight": focal, "factor": factor})
        if len({case["case_id"] for case in cases}) != len(cases):
            raise StageExecutionError("sensitivity case identifiers are not unique")
        return tuple(cases)

    def _stage_static(self) -> dict[str, Any]:
        if self.development_count + self.calibration_count + self.assessment_count != self.sample_count:
            raise StageExecutionError("development, calibration, and assessment counts must partition the design")
        if self.development_count < 351 or self.calibration_count < 1 or self.assessment_count < 1:
            raise StageExecutionError("the independent blocks do not support fitting and calibration")
        if tuple(self.config["process"]["component_order"]) != mechanism.COMPONENTS:
            raise StageExecutionError("configured and implemented component orders differ")
        expected_title = "Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate"
        if self.config["article"]["title"] != expected_title:
            raise StageExecutionError("configured article title differs from the requested title")
        if [self.profile["blocks"][name]["method"] for name in ("development", "calibration", "assessment")] != ["lhs", "iid_open_uniform", "iid_open_uniform"]:
            raise StageExecutionError("the profile must use one development LHS and two independent iid blocks")
        declared = self.config.get("workloads", {}).get(self.profile_name, {})
        computed = self._scientific_workloads(self.profile_name)
        workload_pairs = {
            "total_cases": "optimization_cases",
            "design_bdf_routes": "dataset_bdf_invocations",
            "exact_validation_bdf_routes_max": "selected_point_bdf_invocations_maximum",
            "bdf_routes_max": "bdf_invocations_maximum",
            "combined_nlp_starts": "combined_nlp_starts",
            "qp_evaluations": "physical_qp_evaluations",
            "direct_evaluations": "direct_evaluations",
        }
        for configured_name, computed_name in workload_pairs.items():
            if int(declared.get(configured_name, -1)) != int(computed[computed_name]):
                raise StageExecutionError(f"configured workload {configured_name} is inconsistent")
        dimensions = self.config["optimization"]["nlp"]
        if dimensions["combined_dimensions"] != {"variables": 115, "equalities": 110, "general_inequalities": 9}:
            raise StageExecutionError("combined-NLP dimensions changed")
        multistart = self.config["optimization"]["multistart"]
        if int(multistart["count"]) != 9 or int(multistart["lhs_start_count"]) != 8 or int(multistart["lhs_seed"]) != 271828:
            raise StageExecutionError("the fixed nine-start NLP design changed")
        legacy = " ".join(self.config["optimization"].get("disabled_legacy_methods", [])).lower()
        if "direct" not in legacy or "physical projection qp" not in legacy:
            raise StageExecutionError("legacy DIRECT and physical-QP routes must be explicitly disabled")
        audit = mechanism.audit_mechanistic_matrices()
        if not bool(audit["passed"]):
            raise StageExecutionError(f"mechanistic matrix audit failed: {audit}")
        design_module = importlib.import_module("closed_loop.design")
        development_seed, calibration_seed, assessment_seed, robustness_seed = self._profile_seeds()
        blocks = design_module.generate_study_design_blocks(
            self.development_count,
            self.calibration_count,
            self.assessment_count,
            development_seed=development_seed,
            calibration_seed=calibration_seed,
            assessment_seed=assessment_seed,
        )
        physical = blocks.physical
        unit = blocks.unit
        labels = np.concatenate((
            np.zeros(self.development_count, dtype=np.int8),
            np.ones(self.calibration_count, dtype=np.int8),
            np.full(self.assessment_count, 2, dtype=np.int8),
        ))
        atomic_npz(
            self.run_root / "datasets" / "design.npz",
            row=np.arange(self.sample_count, dtype=np.int64),
            block=labels,
            unit=unit,
            decisions=physical[:, :5],
            influents=physical[:, 5:],
        )
        block_metadata = []
        for name, block in zip(("development", "calibration", "assessment"), (blocks.development, blocks.calibration, blocks.assessment)):
            block_metadata.append({"name": name, "rows": block.n_points, "seed": block.seed, "final_state": block.final_state, "draw_count": block.draw_count, "sampling": "strength-1 Latin hypercube" if name == "development" else "row-major open-unit iid"})
        atomic_json(self.run_root / "splits" / "ordered_blocks.json", {"blocks": block_metadata, "block_codes": {"0": "development", "1": "calibration", "2": "assessment"}, "robustness_seed": robustness_seed})
        workload = {"profile": self._scientific_workloads(self.profile_name), "full": self._scientific_workloads("full")}
        atomic_json(self.run_root / "inputs" / "resolved_config.json", self.config)
        atomic_json(self.run_root / "inputs" / "contract.json", self._contract)
        atomic_json(self.run_root / "checks" / "mechanistic_matrix_audit.json", audit)
        atomic_json(self.run_root / "checks" / "scientific_workload.json", workload)
        return {"rows": self.sample_count, "block_counts": [self.development_count, self.calibration_count, self.assessment_count], "full_workload": workload["full"]}

    def _load_design(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.run_root / "datasets" / "design.npz", allow_pickle=False) as payload:
            return payload["decisions"].copy(), payload["influents"].copy(), payload["block"].copy()

    def _solve_row(self, index: int, decisions: np.ndarray, influent: np.ndarray) -> MechanisticRow:
        if self.mechanistic_solver is not None:
            return self.mechanistic_solver(index, decisions.copy(), influent.copy())
        return _solve_payload((index, decisions, influent, self.config["mechanistic_solver"]))

    def _chunk_path(self, start: int, stop: int) -> Path:
        return self.run_root / "datasets" / "chunks" / f"rows_{start:06d}_{stop:06d}.npz"

    def _save_chunk(self, start: int, stop: int, rows: Sequence[MechanisticRow]) -> None:
        if [row.index for row in rows] != list(range(start, stop)):
            raise StageExecutionError("mechanistic chunk returned rows out of order")
        atomic_npz(
            self._chunk_path(start, stop),
            row=np.asarray([row.index for row in rows], dtype=np.int64),
            decisions=np.vstack([row.decisions for row in rows]),
            influents=np.vstack([row.influent for row in rows]),
            targets=np.vstack([row.target for row in rows]),
            accepted=np.asarray([row.accepted for row in rows], dtype=bool),
            elapsed_seconds=np.asarray([row.elapsed_seconds for row in rows], dtype=np.float64),
        )
        atomic_json(
            self._chunk_path(start, stop).with_suffix(".diagnostics.json"),
            {"start": start, "stop": stop, "rows": [{"row": row.index, "accepted": row.accepted, "elapsed_seconds": row.elapsed_seconds, "error": row.error, "diagnostics": row.diagnostics} for row in rows]},
        )

    def _generate_chunk(self, start: int, stop: int, *, serial: bool) -> dict[str, Any]:
        path = self._chunk_path(start, stop)
        diagnostics_path = path.with_suffix(".diagnostics.json")
        if path.is_file() and diagnostics_path.is_file():
            with np.load(path, allow_pickle=False) as payload:
                accepted = payload["accepted"].copy()
                elapsed = payload["elapsed_seconds"].copy()
            if accepted.shape != (stop - start,) or not np.all(accepted):
                raise StageExecutionError(f"existing chunk {path.name} is incomplete or rejected")
            return {"start": start, "stop": stop, "rows": stop - start, "elapsed_seconds": float(np.sum(elapsed)), "resumed": True}
        decisions, influents, _ = self._load_design()
        payloads = [(index, decisions[index], influents[index], self.config["mechanistic_solver"]) for index in range(start, stop)]
        if serial or self.mechanistic_solver is not None or int(self.profile.get("parallel_workers", 1)) == 1:
            rows = [self._solve_row(index, decisions[index], influents[index]) for index in range(start, stop)]
        else:
            with ProcessPoolExecutor(max_workers=int(self.profile["parallel_workers"])) as executor:
                rows = list(executor.map(_solve_payload, payloads, chunksize=1))
        self._save_chunk(start, stop, rows)
        rejected = [row for row in rows if not row.accepted or not np.all(np.isfinite(row.target))]
        if rejected:
            first = rejected[0]
            raise StageExecutionError(f"mechanistic row {first.index} failed: {first.error}")
        return {"start": start, "stop": stop, "rows": stop - start, "elapsed_seconds": float(sum(row.elapsed_seconds for row in rows)), "resumed": False}

    def _generate_through(self, stop: int, *, pilot_serial: bool = False) -> list[dict[str, Any]]:
        configured = [int(value) for value in self.profile.get("generation_checkpoints", [])]
        boundaries = sorted({0, 4, 16, 64, 256, stop, *configured})
        boundaries = [value for value in boundaries if 0 <= value <= stop]
        if boundaries[-1] != stop:
            boundaries.append(stop)
        summaries = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            summaries.append(self._generate_chunk(start, end, serial=pilot_serial and end <= 256))
            atomic_json(self.run_root / "checks" / f"dataset_rows_{end:06d}.json", {"rows_completed": end, "checkpoint_utc": utc_now(), "chunks": len(summaries)})
        return summaries

    def _stage_pilot(self) -> dict[str, Any]:
        pilot_count = min(256, self.sample_count)
        self._generate_through(pilot_count, pilot_serial=True)
        elapsed: list[float] = []
        peak = 0
        for path in sorted((self.run_root / "datasets" / "chunks").glob("*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                rows = payload["row"]
                mask = rows < pilot_count
                elapsed.extend(payload["elapsed_seconds"][mask].tolist())
            diag_path = path.with_suffix(".diagnostics.json")
            if diag_path.is_file():
                for record in _load_json(diag_path).get("rows", []):
                    peak = max(peak, int(record.get("diagnostics", {}).get("peak_resident_memory_bytes", 0) or 0))
        summary = {
            "row_count": pilot_count,
            "execution": "uncontended serial complete-route BDF invocations",
            "p95_seconds": _nearest_rank(elapsed, 0.95),
            "mean_seconds": float(np.mean(elapsed)),
            "maximum_seconds": float(np.max(elapsed)),
            "maximum_resident_memory_bytes": max(peak, _peak_resident_memory_bytes()),
            "all_accepted": len(elapsed) == pilot_count,
        }
        atomic_json(self.run_root / "timing" / "pilot_summary.json", summary)
        return summary

    def _assemble_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        chunks = []
        for path in sorted((self.run_root / "datasets" / "chunks").glob("*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                chunks.append({name: payload[name].copy() for name in payload.files})
        if not chunks:
            raise StageExecutionError("no mechanistic chunks exist")
        rows = np.concatenate([chunk["row"] for chunk in chunks])
        order = np.argsort(rows, kind="stable")
        if not np.array_equal(rows[order], np.arange(self.sample_count)):
            raise StageExecutionError("mechanistic chunks do not cover each design row exactly once")
        decisions = np.vstack([chunk["decisions"] for chunk in chunks])[order]
        influents = np.vstack([chunk["influents"] for chunk in chunks])[order]
        targets = np.vstack([chunk["targets"] for chunk in chunks])[order]
        elapsed = np.concatenate([chunk["elapsed_seconds"] for chunk in chunks])[order]
        return decisions, influents, targets, elapsed

    def _stage_dataset(self) -> dict[str, Any]:
        self._generate_through(self.sample_count, pilot_serial=True)
        decisions, influents, targets, elapsed = self._assemble_dataset()
        if targets.shape != (self.sample_count, mechanism.TARGET_SIZE) or not np.all(np.isfinite(targets)):
            raise StageExecutionError("the assembled mechanistic response is incomplete or non-finite")
        _, _, blocks = self._load_design()
        atomic_npz(self.run_root / "datasets" / "mechanistic_dataset.npz", row=np.arange(self.sample_count), block=blocks, decisions=decisions, influents=influents, targets=targets, elapsed_seconds=elapsed)
        frame = pd.DataFrame(np.column_stack((np.arange(self.sample_count), blocks, decisions, influents, targets)), columns=("row", "block", *self.config["process"]["decision_bounds"].keys(), *mechanism.COMPONENTS, *_target_columns()))
        atomic_parquet(self.run_root / "datasets" / "mechanistic_dataset.parquet", frame)
        generation_by_block = {
            name: {
                "rows": int(np.sum(blocks == code)),
                "elapsed_seconds": float(np.sum(elapsed[blocks == code])),
            }
            for code, name in enumerate(("development", "calibration", "assessment"))
        }
        summary = {
            "rows": self.sample_count,
            "target_columns": mechanism.TARGET_SIZE,
            "all_finite": True,
            "elapsed_seconds": float(np.sum(elapsed)),
            "generation_by_block": generation_by_block,
            "dataset_sha256": sha256_file(
                self.run_root / "datasets" / "mechanistic_dataset.npz"
            ),
        }
        atomic_json(self.run_root / "checks" / "dataset_validation.json", summary)
        return summary

    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.run_root / "datasets" / "mechanistic_dataset.npz", allow_pickle=False) as payload:
            decisions = payload["decisions"].copy()
            influents = payload["influents"].copy()
            targets = payload["targets"].copy()
            blocks = payload["block"].copy()
        if decisions.shape != (self.sample_count, 5) or influents.shape != (self.sample_count, 20) or targets.shape != (self.sample_count, 170):
            raise StageExecutionError("stored mechanistic dataset dimensions changed")
        return decisions, influents, targets, blocks

    @staticmethod
    def _positive_population_scale(values: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        scale = np.std(matrix, axis=0, ddof=0)
        reference = np.maximum(1.0, np.max(np.abs(matrix), axis=0))
        if np.any(scale <= 1.0e-12 * reference) or not np.all(np.isfinite(scale)):
            bad = np.flatnonzero(scale <= 1.0e-12 * reference)
            raise StageExecutionError(f"development scale {name!r} is zero or negligible at coordinates {bad[:10].tolist()}")
        return scale

    @staticmethod
    def _quality_outputs(decisions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        q_effluent = 1.0 - decisions[:, 4]
        c_effluent = targets[:, 120:140] / q_effluent[:, None]
        return c_effluent @ mechanism.COMPOSITE_MATRIX.T

    @staticmethod
    def _mechanistic_state(targets: np.ndarray) -> np.ndarray:
        return np.concatenate((targets[:, 20:120], targets[:, 160:170]), axis=1)

    def _fit_development_assets(
        self,
        model: QuadraticSurrogate,
        decisions: np.ndarray,
        influents: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        nlp = self._nlp()
        quality_scale = nlp.fit_quality_scale(decisions, targets)
        state_center, state_scale = nlp.fit_mechanistic_state_scaling(targets)
        inventory_scale = nlp.fit_inventory_scale(decisions, targets)
        smoothing = nlp.fit_smoothing_scales(targets)
        residual_scales = nlp.fit_mechanistic_residual_scales(
            decisions, influents, targets, smoothing
        )
        derived_scale = self._positive_population_scale(
            _derived_assessment_responses(decisions, targets),
            "derived assessment responses",
        )
        if not np.allclose(derived_scale[:4], quality_scale, rtol=1.0e-13, atol=0.0):
            raise StageExecutionError(
                "overflow-composite scales disagree between fitting and assessment"
            )
        final_reactor = targets[:, 100:120]
        clarifier_feed_mass_scale = np.maximum(
            1.0,
            np.max(np.abs((1.0 + decisions[:, 3])[:, None] * final_reactor), axis=0),
        )
        leverage = np.asarray(model.leverage(decisions, influents), dtype=np.float64)
        arrays = {
            "quality_scale": quality_scale,
            "feature_qr_upper": model.feature_qr_upper,
            "feature_qr_pivots": model.feature_qr_pivots,
            "leverage_max": np.asarray(float(np.max(leverage))),
            "inventory_scale": np.asarray(inventory_scale),
            "mechanistic_state_center": state_center,
            "mechanistic_state_scale": state_scale,
            "mechanistic_residual_scale": residual_scales,
            "derived_assessment_scale": derived_scale,
            "clarifier_feed_mass_scale": clarifier_feed_mass_scale,
            "smoothing_scales": np.asarray([getattr(smoothing, field.name) for field in fields(smoothing)], dtype=np.float64),
        }
        metadata = {
            "development_rows": int(decisions.shape[0]),
            "leverage_max": float(arrays["leverage_max"]),
            "inventory_scale": inventory_scale,
            "quality_scale": quality_scale,
            "derived_assessment_scale": derived_scale,
            "smoothing": asdict(smoothing),
            "fit_information_boundary": "development only",
        }
        return arrays, metadata

    def _stage_fit(self) -> dict[str, Any]:
        decisions, influents, targets, blocks = self._load_dataset()
        development = blocks == 0
        if int(np.sum(development)) != self.development_count:
            raise StageExecutionError("development block membership changed")
        started = perf_counter_ns()
        model = QuadraticSurrogate.fit(
            decisions[development],
            influents[development],
            targets[development],
            variance_relative_tolerance=float(self.config["surrogate"]["variance_relative_floor"]),
            maximum_condition_number=float(self.config["surrogate"]["maximum_design_condition_number"]),
        )
        arrays, metadata = self._fit_development_assets(
            model, decisions[development], influents[development], targets[development]
        )
        elapsed = (perf_counter_ns() - started) / 1.0e9
        save_surrogate_bundle(
            self.run_root / "models" / "development_surrogate.npz",
            model,
            **arrays,
        )
        metadata.update(
            fit_seconds=elapsed,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
            diagnostics=model.diagnostics.as_dict(),
        )
        atomic_json(self.run_root / "models" / "development_assets.json", metadata)
        return metadata

    def _stage_calibration(self) -> dict[str, Any]:
        decisions, influents, targets, blocks = self._load_dataset()
        model, _, _ = load_surrogate_bundle(self.run_root / "models" / "development_surrogate.npz")
        calibration = blocks == 1
        started = perf_counter_ns()
        predicted = model.predict(decisions[calibration], influents[calibration])
        result = surrogate_core.calibrate_split_conformal(
            targets[calibration],
            predicted,
            model.response_scale,
            alpha=float(self.config["surrogate"].get("calibration", {}).get("alpha", 0.05)),
            maximum_delta=float(self.config["surrogate"].get("calibration", {}).get("delta_max_inclusive", 1.0)),
        )
        atomic_npz(
            self.run_root / "predictions" / "calibration_scores.npz",
            row=np.flatnonzero(calibration),
            truth=targets[calibration],
            raw=predicted,
            scores=result.scores,
        )
        summary = {
            **result.as_dict(),
            "elapsed_seconds": (perf_counter_ns() - started) / 1.0e9,
            "gate": "finite and 0 < delta <= 1",
            "passed": True,
        }
        atomic_json(self.run_root / "metrics" / "calibration.json", summary)
        return summary

    def _stage_assessment(self) -> dict[str, Any]:
        decisions, influents, targets, blocks = self._load_dataset()
        model, _, arrays = load_surrogate_bundle(
            self.run_root / "models" / "development_surrogate.npz"
        )
        calibration = _load_json(self.run_root / "metrics" / "calibration.json")
        assessment = blocks == 2
        started = perf_counter_ns()
        predicted = model.predict(decisions[assessment], influents[assessment])
        settings = self.config["surrogate"]["assessment"]
        metrics = surrogate_core.assess_raw_surrogate(
            model,
            decisions[assessment],
            influents[assessment],
            targets[assessment],
            delta=float(calibration["delta"]),
            complete_state_rmse_max=float(settings.get("complete_state_standardized_rmse_max_exclusive", 1.0)),
            minimum_coverage=float(settings.get("empirical_coverage_min_inclusive", 0.90)),
            variance_relative_tolerance=float(self.config["surrogate"]["variance_relative_floor"]),
        )
        coordinate = metrics.coordinate_metrics
        coordinate_frame = pd.DataFrame(
            {
                "coordinate": _target_columns(),
                "rmse": coordinate.rmse,
                "mae": coordinate.mae,
                "bias": coordinate.bias,
                "nrmse": coordinate.nrmse,
                "r_squared": coordinate.r_squared,
                "r_squared_defined": coordinate.r_squared_defined,
            }
        )
        atomic_csv(self.run_root / "metrics" / "assessment_coordinate_metrics.csv", coordinate_frame)
        derived_truth = _derived_assessment_responses(
            decisions[assessment], targets[assessment]
        )
        derived_prediction = _derived_assessment_responses(
            decisions[assessment], predicted
        )
        derived_frame = _derived_metric_frame(
            derived_truth,
            derived_prediction,
            arrays["derived_assessment_scale"],
            variance_relative_tolerance=float(
                self.config["surrogate"]["variance_relative_floor"]
            ),
        )
        atomic_csv(
            self.run_root / "metrics" / "assessment_derived_metrics.csv",
            derived_frame,
        )
        atomic_npz(
            self.run_root / "predictions" / "assessment_predictions.npz",
            row=np.flatnonzero(assessment), truth=targets[assessment], raw=predicted,
            scores=metrics.scores,
        )
        summary = {
            **metrics.as_dict(),
            "derived_response_count": len(DERIVED_ASSESSMENT_NAMES),
            "derived_development_scales_finite_positive": bool(
                np.all(np.isfinite(arrays["derived_assessment_scale"]))
                and np.all(arrays["derived_assessment_scale"] > 0.0)
            ),
            "elapsed_seconds": (perf_counter_ns() - started) / 1.0e9,
        }
        atomic_json(self.run_root / "metrics" / "assessment_summary.json", summary)
        if not metrics.passed:
            raise StageExecutionError(
                "untouched assessment gate failed: "
                f"finite={metrics.predictions_finite}, "
                f"standardized_rmse={metrics.complete_state_standardized_rmse:.6g}, "
                f"coverage={metrics.empirical_coverage:.6g}"
            )
        return summary

    def _load_nlp_assets(self) -> tuple[Any, QuadraticSurrogate]:
        nlp = self._nlp()
        model, _, arrays = load_surrogate_bundle(
            self.run_root / "models" / "development_surrogate.npz"
        )
        calibration = _load_json(self.run_root / "metrics" / "calibration.json")
        smoothing_names = [field.name for field in fields(nlp.SmoothingScales)]
        smoothing_values = np.asarray(arrays["smoothing_scales"], dtype=np.float64)
        if smoothing_values.shape != (len(smoothing_names),):
            raise StageExecutionError("stored smoothing-scale vector has changed dimensions")
        smoothing = nlp.SmoothingScales(
            **dict(zip(smoothing_names, smoothing_values.tolist(), strict=True))
        )
        assets = nlp.CombinedNLPAssets(
            model=model,
            fidelity_delta=float(calibration["delta"]),
            leverage_max=float(arrays["leverage_max"]),
            state_center=arrays["mechanistic_state_center"],
            state_scale=arrays["mechanistic_state_scale"],
            residual_scale=arrays["mechanistic_residual_scale"],
            quality_scale=arrays["quality_scale"],
            inventory_scale=float(arrays["inventory_scale"]),
            smoothing=smoothing,
        )
        return assets, model

    def _ipopt_settings(self) -> Any:
        nlp = self._nlp()
        configured = self.config["optimization"]["nlp"]
        derivatives = str(configured["derivatives"]).lower()
        if "exact" not in derivatives or "second" not in derivatives:
            raise StageExecutionError("the configured derivative contract must require exact second derivatives")
        return nlp.IPOPTSettings(
            primal_tolerance=float(configured["primal_tolerance"]),
            stationarity_tolerance=float(configured["stationarity_tolerance"]),
            dual_tolerance=float(configured["dual_tolerance"]),
            complementarity_tolerance=float(configured["complementarity_tolerance"]),
            physical_nonnegativity_tolerance=float(configured["physical_nonnegativity_tolerance"]),
            tol=float(configured["tol"]),
            constraint_violation_tolerance=float(configured["constraint_violation_tolerance"]),
            dual_infeasibility_tolerance=float(configured["dual_infeasibility_tolerance"]),
            ipopt_complementarity_tolerance=float(configured["complementarity_tolerance"]),
            maximum_iterations=int(configured["maximum_iterations"]),
            bound_relax_factor=float(configured["bound_relax_factor"]),
            linear_solver=str(configured["linear_solver"]).lower(),
            mu_strategy=str(configured["mu_strategy"]).lower(),
            hessian_approximation="exact",
            accepted_return_statuses=tuple(str(value) for value in configured["accepted_return_statuses"]),
        )

    def _build_nlp_problems(self, *, compile_solver: bool = True) -> tuple[Any, Any, QuadraticSurrogate]:
        nlp = self._nlp()
        assets, model = self._load_nlp_assets()
        settings = self._ipopt_settings()
        problem = nlp.build_combined_nlp(
            assets, settings=settings, compile_solver=compile_solver
        )
        return problem, assets, model

    def _primary_weights(self) -> list[float]:
        objective = self.config["objective"]
        return [
            float(objective["quality_weight"]),
            float(objective["H_weight"]),
            float(objective["a_weight"]),
            float(objective["r_I_weight"]),
            float(objective["r_R_weight"]),
            float(objective["wasted_solids_weight"]),
        ]

    def _case_records(self) -> list[dict[str, Any]]:
        design = importlib.import_module("closed_loop.design")
        _, _, _, robustness_seed = self._profile_seeds()
        robustness_count = int(self.profile["robustness"]["count"])
        generated = design.generate_robustness_design(robustness_count, seed=robustness_seed)
        nominal = np.asarray(self.config["process"]["nominal_influent"], dtype=np.float64)
        limit = float(self.config["upper_constraints"]["underflow_tss_max_g_m3"])
        primary = self._primary_weights()
        cases: list[dict[str, Any]] = [
            {"case_id": "nominal", "case_class": "nominal", "sensitivity_family": None, "influent": nominal, "weights": primary, "underflow_tss_limit": limit, "robustness_index": None}
        ]
        for index, influent in enumerate(generated.physical):
            cases.append({"case_id": f"robustness_{index + 1:03d}", "case_class": "robustness", "sensitivity_family": None, "influent": influent, "weights": primary, "underflow_tss_limit": limit, "robustness_index": index})
        for specification in self._resolved_sensitivity_cases():
            cases.append({**specification, "influent": nominal, "weights": specification["weights"] or primary, "robustness_index": None})
        atomic_npz(
            self.run_root / "optimization" / "robustness_design.npz",
            unit=generated.unit,
            influents=generated.physical,
            selected_preflight_indices=_maximin_robustness_indices(
                generated.physical,
                nominal,
                np.asarray([pair[0] for pair in self.config["process"]["influent_bounds"].values()]),
                np.asarray([pair[1] for pair in self.config["process"]["influent_bounds"].values()]),
                count=10,
            ),
        )
        metadata = {
            "seed": generated.seed,
            "final_state": generated.final_state,
            "draw_count": generated.draw_count,
            "cases": _json_value(cases),
        }
        atomic_json(self.run_root / "optimization" / "cases.json", metadata)
        return cases

    def _load_case_records(self) -> list[dict[str, Any]]:
        records = _load_json(self.run_root / "optimization" / "cases.json")["cases"]
        return [dict(record) for record in records]

    def _case_definition(self, record: Mapping[str, Any]) -> Any:
        nlp = self._nlp()
        weights = nlp.ObjectiveWeights(*[float(value) for value in record["weights"]])
        return nlp.CaseDefinition(
            influent=np.asarray(record["influent"], dtype=np.float64),
            weights=weights,
            underflow_tss_limit=float(record["underflow_tss_limit"]),
            case_id=str(record["case_id"]),
        )

    @staticmethod
    def _start_record(result: Any) -> dict[str, Any]:
        return {
            "start_index": int(result.start_index),
            "status": str(result.status),
            "solver_success": bool(result.solver_success),
            "accepted": bool(result.accepted),
            "objective": float(result.objective),
            "elapsed_seconds": float(result.elapsed_seconds),
            "iterations": int(result.iterations),
            "error": result.error,
            "diagnostics": dict(result.diagnostics),
            "kkt": asdict(result.kkt),
        }

    @staticmethod
    def _start_array_names() -> tuple[str, ...]:
        return (
            "primal", "equality_multipliers", "inequality_multipliers",
            "bound_multipliers", "equality", "inequality", "normalized_controls",
            "decisions", "state",
        )

    def _result_from_record(
        self, record: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    ) -> Any:
        nlp = self._nlp()
        kkt_values = {
            name: (
                bool(value)
                if name == "finite"
                else (math.inf if value is None else float(value))
            )
            for name, value in record["kkt"].items()
        }
        return nlp.NLPStartResult(
            start_index=int(record["start_index"]),
            status=str(record["status"]),
            solver_success=bool(record["solver_success"]),
            accepted=bool(record["accepted"]),
            objective=float(record["objective"]) if record["objective"] is not None else math.nan,
            primal=np.asarray(arrays["primal"], dtype=np.float64),
            equality_multipliers=np.asarray(arrays["equality_multipliers"], dtype=np.float64),
            inequality_multipliers=np.asarray(arrays["inequality_multipliers"], dtype=np.float64),
            bound_multipliers=np.asarray(arrays["bound_multipliers"], dtype=np.float64),
            equality=np.asarray(arrays["equality"], dtype=np.float64),
            inequality=np.asarray(arrays["inequality"], dtype=np.float64),
            normalized_controls=np.asarray(arrays["normalized_controls"], dtype=np.float64),
            decisions=np.asarray(arrays["decisions"], dtype=np.float64),
            state=np.asarray(arrays["state"], dtype=np.float64),
            diagnostics={
                str(name): (math.nan if value is None else float(value))
                for name, value in record.get("diagnostics", {}).items()
            },
            kkt=nlp.KKTDiagnostics(**kkt_values),
            elapsed_seconds=float(record["elapsed_seconds"]),
            iterations=int(record["iterations"]),
            error=record.get("error"),
        )

    def _start_checkpoint_paths(self, case_id: str, start_index: int) -> tuple[Path, Path]:
        root = self.run_root / "optimization" / "cache" / case_id / "start_checkpoints"
        return root / f"start_{start_index:02d}.npz", root / f"start_{start_index:02d}.json"

    def _save_start_checkpoint(
        self,
        identity: Mapping[str, Any],
        result: Any,
        complete_state: np.ndarray,
    ) -> None:
        npz_path, json_path = self._start_checkpoint_paths(
            str(identity["case_id"]), int(identity["start_index"])
        )
        complete = np.asarray(complete_state, dtype=np.float64)
        if complete.shape != (mechanism.TARGET_SIZE,):
            raise StageExecutionError("a start checkpoint complete state must have 170 entries")
        arrays = {
            name: np.asarray(getattr(result, name), dtype=np.float64)
            for name in self._start_array_names()
        }
        arrays["complete_state"] = complete
        metadata = {
            "schema_version": 2,
            "kind": "combined_nlp_start",
            **identity,
            "array_shapes": {
                **{name: list(value.shape) for name, value in arrays.items()},
                "record_json": [],
            },
            "array_sha256": {
                name: _float_array_sha256(value) for name, value in arrays.items()
            },
            "result": self._start_record(result),
        }
        arrays["record_json"] = np.asarray(
            json.dumps(
                _json_value(metadata),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        atomic_npz(npz_path, **arrays)
        atomic_json(json_path, {**metadata, "npz_sha256": sha256_file(npz_path)})

    def _load_start_checkpoint(
        self, identity: Mapping[str, Any],
    ) -> tuple[Any, np.ndarray] | None:
        npz_path, json_path = self._start_checkpoint_paths(
            str(identity["case_id"]), int(identity["start_index"])
        )
        if not npz_path.exists() and not json_path.exists():
            return None
        if not npz_path.is_file():
            raise ContractMismatchError(
                "start-checkpoint metadata exists without its atomic array cache"
            )
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        if "record_json" not in arrays or arrays["record_json"].shape != ():
            raise ContractMismatchError("start checkpoint has no authoritative record")
        try:
            metadata = json.loads(str(arrays["record_json"].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ContractMismatchError("start checkpoint record is unreadable") from exc
        for name, expected in identity.items():
            if metadata.get(name) != expected:
                raise ContractMismatchError(f"start checkpoint identity changed at {name}")
        if metadata.get("kind") != "combined_nlp_start" or metadata.get("schema_version") != 2:
            raise ContractMismatchError("start checkpoint failed its type contract")
        expected_shapes = {
            "primal": (115,), "equality_multipliers": (110,),
            "inequality_multipliers": (9,), "bound_multipliers": (115,),
            "equality": (110,), "inequality": (9,), "normalized_controls": (5,),
            "decisions": (5,), "state": (110,), "complete_state": (170,),
            "record_json": (),
        }
        if set(arrays) != set(expected_shapes) or any(
            arrays[name].shape != shape for name, shape in expected_shapes.items()
        ):
            raise ContractMismatchError("start checkpoint array dimensions changed")
        if metadata.get("array_shapes") != {
            name: list(shape) for name, shape in expected_shapes.items()
        }:
            raise ContractMismatchError("start checkpoint shape metadata changed")
        for name in self._start_array_names() + ("complete_state",):
            if metadata.get("array_sha256", {}).get(name) != _float_array_sha256(
                arrays[name]
            ):
                raise ContractMismatchError(
                    f"start checkpoint array digest changed for {name}"
                )
        public_metadata = {**metadata, "npz_sha256": sha256_file(npz_path)}
        if json_path.is_file():
            if _load_json(json_path) != _json_value(public_metadata):
                raise ContractMismatchError(
                    "start checkpoint JSON and atomic cache disagree"
                )
        else:
            atomic_json(json_path, public_metadata)
        result = self._result_from_record(metadata["result"], arrays)
        if result.start_index != int(identity["start_index"]):
            raise ContractMismatchError("start checkpoint result index changed")
        return result, arrays["complete_state"]

    def _save_nlp_case(
        self,
        case: Any,
        results: Sequence[Any],
        complete_states: np.ndarray,
        start_identities: Sequence[Mapping[str, Any]],
    ) -> None:
        if len(results) != 9:
            raise StageExecutionError("each combined case must retain exactly nine starts")
        complete = np.asarray(complete_states, dtype=np.float64)
        if complete.shape != (9, mechanism.TARGET_SIZE):
            raise StageExecutionError("combined start reconstructions must have shape (9,170)")
        case_identity = self._case_cache_identity(case)
        root = self.run_root / "optimization" / "cache" / str(case.case_id)
        root.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: np.stack(
                [np.asarray(getattr(result, name), dtype=np.float64) for result in results]
            )
            for name in self._start_array_names()
        }
        arrays["complete_state"] = complete
        metadata = {
            "schema_version": 3,
            "kind": "combined_nlp_case",
            **case_identity,
            "array_shapes": {
                **{name: list(value.shape) for name, value in arrays.items()},
                "record_json": [],
            },
            "array_sha256": {
                name: _float_array_sha256(value) for name, value in arrays.items()
            },
            "start_cache_keys": [
                str(identity["start_cache_key"]) for identity in start_identities
            ],
            "combined": [self._start_record(result) for result in results],
        }
        arrays["record_json"] = np.asarray(
            json.dumps(
                _json_value(metadata),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        atomic_npz(root / "starts.npz", **arrays)
        atomic_json(
            root / "starts.json",
            {**metadata, "npz_sha256": sha256_file(root / "starts.npz")},
        )

    def _load_nlp_case(
        self,
        case: Any,
        *,
        expected_start_identities: Sequence[Mapping[str, Any]] | None = None,
        record_reuse: bool = False,
    ) -> tuple[Any, ...]:
        case_identity = self._case_cache_identity(case)
        case_id = str(case.case_id)
        root = self.run_root / "optimization" / "cache" / case_id
        npz_path = root / "starts.npz"
        json_path = root / "starts.json"
        if not npz_path.is_file():
            raise ContractMismatchError(f"NLP cache for {case_id} has no atomic array cache")
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        if "record_json" not in arrays or arrays["record_json"].shape != ():
            raise ContractMismatchError(f"NLP cache for {case_id} has no authoritative record")
        try:
            metadata = json.loads(str(arrays["record_json"].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ContractMismatchError(f"NLP cache for {case_id} record is unreadable") from exc
        for name, expected in case_identity.items():
            if metadata.get(name) != expected:
                raise ContractMismatchError(f"NLP case-cache identity changed at {name}")
        if metadata.get("kind") != "combined_nlp_case" or metadata.get("schema_version") != 3:
            raise ContractMismatchError(f"NLP cache for {case_id} failed its type contract")
        expected_shapes = {
            "primal": (9, 115), "equality_multipliers": (9, 110),
            "inequality_multipliers": (9, 9), "bound_multipliers": (9, 115),
            "equality": (9, 110), "inequality": (9, 9),
            "normalized_controls": (9, 5), "decisions": (9, 5),
            "state": (9, 110), "complete_state": (9, 170),
            "record_json": (),
        }
        if set(arrays) != set(expected_shapes) or any(
            arrays[name].shape != shape for name, shape in expected_shapes.items()
        ) or metadata.get("array_shapes") != {
            name: list(shape) for name, shape in expected_shapes.items()
        }:
            raise ContractMismatchError(f"NLP cache for {case_id} has changed dimensions")
        if len(metadata.get("combined", ())) != 9 or len(metadata.get("start_cache_keys", ())) != 9:
            raise ContractMismatchError(f"NLP cache for {case_id} does not contain nine starts")
        for name in self._start_array_names() + ("complete_state",):
            if metadata.get("array_sha256", {}).get(name) != _float_array_sha256(
                arrays[name]
            ):
                raise ContractMismatchError(
                    f"NLP cache for {case_id} array digest changed for {name}"
                )
        public_metadata = {**metadata, "npz_sha256": sha256_file(npz_path)}
        if json_path.is_file():
            if _load_json(json_path) != _json_value(public_metadata):
                raise ContractMismatchError(
                    f"NLP cache for {case_id} JSON and atomic cache disagree"
                )
        else:
            atomic_json(json_path, public_metadata)
        if expected_start_identities is not None:
            expected_keys = [str(identity["start_cache_key"]) for identity in expected_start_identities]
            if metadata["start_cache_keys"] != expected_keys:
                raise ContractMismatchError(f"NLP cache for {case_id} has different initial points")
        results = []
        for row, record in enumerate(metadata["combined"]):
            result_arrays = {name: arrays[name][row] for name in self._start_array_names()}
            result = self._result_from_record(record, result_arrays)
            if result.start_index != row:
                raise ContractMismatchError(f"NLP cache for {case_id} has reordered starts")
            results.append(result)
        if record_reuse:
            for row, logical_key in enumerate(metadata["start_cache_keys"]):
                self._reconcile_interrupted_invocation(
                    "combined_nlp_start", logical_key, case_id=case_id, start_index=row,
                )
                self._ledger_event(
                    "combined_nlp_start", logical_key, "reused",
                    case_id=case_id, start_index=row, source="aggregate_case_cache",
                    verification="contract, case, initial point, dimensions, and SHA-256 passed",
                )
        return tuple(results)

    def _load_nlp_complete_states(self, case: Any) -> np.ndarray:
        self._load_nlp_case(case)
        root = self.run_root / "optimization" / "cache" / str(case.case_id)
        with np.load(root / "starts.npz", allow_pickle=False) as payload:
            values = payload["complete_state"].copy()
        if values.shape != (9, mechanism.TARGET_SIZE):
            raise ContractMismatchError(f"NLP cache for {case.case_id} has changed dimensions")
        return values

    def _solve_nlp_case(
        self,
        record: Mapping[str, Any],
        problem: Any,
        assets: Any,
        model: QuadraticSurrogate,
        decisions: np.ndarray,
        influents: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[Any, ...]:
        nlp = self._nlp()
        case = self._case_definition(record)
        normalized = nlp.ordered_normalized_starts()
        initial_points = [
            nlp.combined_initial_point(
                z, case.influent, decisions, influents, targets, assets
            )[0]
            for z in normalized
        ]
        identities = [
            self._start_cache_identity(case, point, index)
            for index, point in enumerate(initial_points)
        ]
        case_root = self.run_root / "optimization" / "cache" / str(case.case_id)
        aggregate_npz = case_root / "starts.npz"
        aggregate_json = case_root / "starts.json"
        if aggregate_npz.exists() or aggregate_json.exists():
            if not aggregate_npz.is_file():
                raise ContractMismatchError(
                    "aggregate NLP metadata exists without its atomic array cache"
                )
            return self._load_nlp_case(
                case,
                expected_start_identities=identities,
                record_reuse=True,
            )
        from threadpoolctl import threadpool_limits

        results: list[Any] = []
        complete_states: list[np.ndarray] = []
        for index, (point, identity) in enumerate(zip(initial_points, identities, strict=True)):
            logical_key = str(identity["start_cache_key"])
            checkpoint = self._load_start_checkpoint(identity)
            if checkpoint is not None:
                self._reconcile_interrupted_invocation(
                    "combined_nlp_start", logical_key,
                    case_id=str(case.case_id), start_index=index,
                )
                self._ledger_event(
                    "combined_nlp_start", logical_key, "reused",
                    case_id=str(case.case_id), start_index=index,
                    source="per_start_checkpoint",
                    verification="contract, case, initial point, dimensions, and SHA-256 passed",
                )
                result, complete_state = checkpoint
            else:
                self._reconcile_interrupted_invocation(
                    "combined_nlp_start", logical_key,
                    case_id=str(case.case_id), start_index=index,
                )
                self._ledger_event(
                    "combined_nlp_start", logical_key, "attempted",
                    case_id=str(case.case_id), start_index=index,
                )
                try:
                    with threadpool_limits(limits=1):
                        result = nlp.solve_nlp_start(
                            problem, case, point, start_index=index
                        )
                    complete_state = nlp.evaluate_problem(
                        problem, result.primal, case
                    )["complete_state"]
                    self._save_start_checkpoint(identity, result, complete_state)
                except BaseException as exc:
                    self._ledger_event(
                        "combined_nlp_start", logical_key, "interrupted",
                        case_id=str(case.case_id), start_index=index,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    raise
                self._ledger_event(
                    "combined_nlp_start", logical_key, "completed",
                    case_id=str(case.case_id), start_index=index,
                    accepted=bool(result.accepted),
                )
            results.append(result)
            complete_states.append(np.asarray(complete_state, dtype=np.float64))
        self._save_nlp_case(
            case, results, np.stack(complete_states), identities
        )
        return tuple(results)

    def _preflight_case_ids(self, cases: Sequence[Mapping[str, Any]]) -> list[str]:
        with np.load(self.run_root / "optimization" / "robustness_design.npz", allow_pickle=False) as payload:
            selected = set(int(value) for value in payload["selected_preflight_indices"])
        identifiers = ["nominal"]
        identifiers.extend(str(case["case_id"]) for case in cases if case["case_class"] == "sensitivity")
        identifiers.extend(str(case["case_id"]) for case in cases if case["case_class"] == "robustness" and int(case["robustness_index"]) in selected)
        return identifiers

    def _computational_projection(
        self,
        combined_seconds: Sequence[float],
        nlp_peak_memory: int,
    ) -> dict[str, Any]:
        pilot = _load_json(self.run_root / "timing" / "pilot_summary.json")
        fit = _load_json(self.run_root / "models" / "development_assets.json")
        workload = self._scientific_workloads("full")
        safety = float(self.config.get("execution", {}).get("computational_feasibility", {}).get("projection_safety_factor", 1.5))
        bdf_p95 = float(pilot["p95_seconds"])
        combined_p95 = _nearest_rank(combined_seconds, 0.95)
        projected_seconds = float(fit["fit_seconds"]) + safety * (
            workload["bdf_invocations_maximum"] * bdf_p95
            + workload["combined_nlp_starts"] * combined_p95
        )
        peak = max(
            int(pilot.get("maximum_resident_memory_bytes", 0)),
            int(fit.get("peak_resident_memory_bytes", 0)),
            int(nlp_peak_memory),
        )
        projected_memory = int(math.ceil(safety * peak))
        feasibility = self.config["execution"].get("computational_feasibility", {})
        maximum_core_days = float(feasibility.get("maximum_projected_core_days", 30.0))
        maximum_memory = float(feasibility.get("maximum_projected_resident_memory_gib", 25.0)) * 1024**3
        result = {
            "safety_factor": safety,
            "bdf_p95_seconds": bdf_p95,
            "combined_nlp_p95_seconds": combined_p95,
            "fit_seconds": float(fit["fit_seconds"]),
            "projected_core_seconds": projected_seconds,
            "projected_core_days": projected_seconds / 86400.0,
            "projected_resident_memory_bytes": projected_memory,
            "maximum_projected_core_days": maximum_core_days,
            "maximum_projected_resident_memory_bytes": int(maximum_memory),
            "projection_passed": bool(projected_seconds <= maximum_core_days * 86400 and projected_memory <= maximum_memory),
            "gate_enforced": True,
            "projection_scope": "full declared workload, enforced for every execution profile",
            "full_workload": workload,
        }
        if not result["projection_passed"]:
            raise StageExecutionError("the predeclared computational-feasibility gate failed")
        return result

    def _stage_nlp_preflight(self) -> dict[str, Any]:
        cases = self._case_records()
        if self.profile_name == "test_2000" and len(cases) != 23:
            raise StageExecutionError(f"test_2000 must contain the declared 23 cases; found {len(cases)}")
        decisions, influents, targets, blocks = self._load_dataset()
        development = blocks == 0
        problem, assets, model = self._build_nlp_problems()
        starts = self._nlp().ordered_normalized_starts()
        if starts.shape != (9, 5):
            raise StageExecutionError("the deterministic NLP start design must contain exactly nine rows")
        atomic_npz(self.run_root / "optimization" / "ordered_starts.npz", normalized_controls=starts)
        _, final_state, draw_count = importlib.import_module("closed_loop.design").unit_latin_hypercube(
            8, 5, seed=int(self.config["optimization"]["multistart"]["lhs_seed"])
        )
        atomic_json(
            self.run_root / "optimization" / "ordered_starts.json",
            {
                "center_start": starts[0],
                "lhs_seed": int(self.config["optimization"]["multistart"]["lhs_seed"]),
                "lhs_final_state": final_state,
                "lhs_draw_count": draw_count,
                "npz_sha256": sha256_file(self.run_root / "optimization" / "ordered_starts.npz"),
            },
        )
        wanted = set(self._preflight_case_ids(cases))
        combined_times: list[float] = []
        peak = _peak_resident_memory_bytes()
        for record in cases:
            if record["case_id"] not in wanted:
                continue
            results = self._solve_nlp_case(
                record, problem, assets, model,
                decisions[development],
                influents[development],
                targets[development],
            )
            combined_times.extend(result.elapsed_seconds for result in results)
            peak = max(peak, _peak_resident_memory_bytes())
        projection = self._computational_projection(combined_times, peak)
        summary = {
            "panel_case_ids": [case_id for case_id in self._preflight_case_ids(cases)],
            "panel_case_count": len(wanted),
            "combined_starts_cached": len(combined_times),
            "maximum_resident_memory_bytes": peak,
            "projection": projection,
        }
        atomic_json(self.run_root / "timing" / "nlp_preflight.json", summary)
        return summary

    def _engineering_from_target(
        self,
        decisions: np.ndarray,
        target: np.ndarray,
        weights: Sequence[float],
        underflow_limit: float,
        quality_scale: np.ndarray,
        inventory_scale: float,
        feed_mass_scale: np.ndarray | None = None,
    ) -> dict[str, Any]:
        theta = np.asarray(decisions, dtype=np.float64)
        chi = np.asarray(target, dtype=np.float64)
        q_c = 1.0 + theta[3]
        q_u = theta[3] + theta[4]
        q_e = 1.0 - theta[4]
        reactors = chi[20:120].reshape(mechanism.N_STAGES, mechanism.N_COMPONENTS)
        layers = chi[160:170]
        g_e = chi[120:140]
        g_u = chi[140:160]
        c_e = g_e / q_e
        c_u = g_u / q_u
        feed_tss = float(mechanism.TSS_VECTOR @ reactors[-1])
        effluent_tss = float(mechanism.TSS_VECTOR @ c_e)
        underflow_tss = float(mechanism.TSS_VECTOR @ c_u)
        boundary = q_e * effluent_tss + theta[4] * underflow_tss
        stage_volume = mechanism.CLARIFIER.fresh_flow * theta[0] / (24.0 * mechanism.N_STAGES)
        reactor_inventory = stage_volume * float(np.sum(reactors @ mechanism.TSS_VECTOR))
        clarifier_inventory = mechanism.CLARIFIER.layer_volume * float(np.sum(layers))
        inventory = reactor_inventory + clarifier_inventory
        composites = mechanism.COMPOSITE_MATRIX @ c_e
        underflow_composites = mechanism.COMPOSITE_MATRIX @ c_u
        feed_mass = q_c * reactors[-1]
        closure_numerator = np.abs(feed_mass - g_e - g_u)
        closure_denominator = np.maximum(1.0, np.abs(feed_mass) + np.abs(g_e) + np.abs(g_u))
        closure = closure_numerator / closure_denominator
        reporting_scale = (
            np.maximum(1.0, np.abs(feed_mass))
            if feed_mass_scale is None
            else np.asarray(feed_mass_scale, dtype=np.float64)
        )
        if reporting_scale.shape != (mechanism.N_COMPONENTS,) or np.any(reporting_scale <= 0.0):
            raise StageExecutionError("Clarifier feed-mass reporting scales changed dimensions")
        recovery_available = feed_mass > 1.0e-10 * reporting_scale
        recovery = np.full(mechanism.N_COMPONENTS, np.nan)
        concentration_ratio = np.full(mechanism.N_COMPONENTS, np.nan)
        recovery[recovery_available] = g_u[recovery_available] / feed_mass[recovery_available]
        positive_feed = recovery_available & (reactors[-1] > 0.0)
        concentration_ratio[positive_feed] = c_u[positive_feed] / reactors[-1, positive_feed]
        components = np.asarray(
            [
                float(np.mean(composites / quality_scale)),
                (theta[0] - 6.0) / 30.0,
                theta[1],
                theta[2] / 4.0,
                (theta[3] - 0.25) / 1.0,
                theta[4] * underflow_tss / (0.05 * 15_000.0),
            ],
            dtype=np.float64,
        )
        weight_array = np.asarray(weights, dtype=np.float64)
        sor = mechanism.CLARIFIER.fresh_flow * q_e / mechanism.CLARIFIER.area
        slr = mechanism.CLARIFIER.fresh_flow * q_c * feed_tss / (1000.0 * mechanism.CLARIFIER.area)
        srt = inventory / (mechanism.CLARIFIER.fresh_flow * boundary) if boundary > 0.0 else math.inf
        domain = np.asarray([1.0 - feed_tss, 1.0 - boundary])
        engineering = np.asarray(
            [
                (8.0 * mechanism.CLARIFIER.fresh_flow * boundary - inventory) / inventory_scale,
                (inventory - 30.0 * mechanism.CLARIFIER.fresh_flow * boundary) / inventory_scale,
                (sor - 20.0) / 20.0,
                (slr - 100.0) / 100.0,
                (underflow_tss - underflow_limit) / 15_000.0,
            ]
        )
        tolerance = float(self.config["upper_constraints"]["normalized_feasibility_tolerance"])
        return {
            "objective": float(weight_array @ components),
            "components": components,
            "weighted_components": weight_array * components,
            "feed_tss": feed_tss,
            "boundary_solids": boundary,
            "reactor_solids_inventory": reactor_inventory,
            "clarifier_solids_inventory": clarifier_inventory,
            "normalized_clarifier_inventory": clarifier_inventory / mechanism.CLARIFIER.fresh_flow,
            "inventory": inventory,
            "srt_days": srt,
            "surface_overflow_rate": sor,
            "solids_loading_rate": slr,
            "underflow_tss": underflow_tss,
            "effluent_components": c_e,
            "underflow_components": c_u,
            "effluent_composites": composites,
            "underflow_composites": underflow_composites,
            "clarifier_layers": layers,
            "component_mass_closure": closure,
            "maximum_component_mass_closure": float(np.max(closure)),
            "component_recovery_available": recovery_available,
            "component_underflow_recovery": recovery,
            "component_concentration_ratio": concentration_ratio,
            "reactor_dissolved_oxygen_profile": reactors[:, mechanism.COMPONENT_INDEX["S_O"]],
            "reactor_tn_profile": reactors @ mechanism.COMPOSITE_MATRIX[1],
            "reactor_tp_profile": reactors @ mechanism.COMPOSITE_MATRIX[2],
            "reactor_tss_profile": reactors @ mechanism.TSS_VECTOR,
            "domain_rows": domain,
            "engineering_rows": engineering,
            "domain_engineering_passed": bool(np.all(domain <= tolerance) and np.all(engineering <= tolerance)),
        }

    def _exact_fidelity_passed(self, normalized_fidelity: float) -> bool:
        tolerance = float(
            self.config["upper_constraints"]["normalized_feasibility_tolerance"]
        )
        return bool(
            np.isfinite(normalized_fidelity)
            and normalized_fidelity - 1.0 <= tolerance
        )

    def _load_exact_cache(
        self, case: Any, selected: Any,
    ) -> dict[str, Any] | None:
        root = self.run_root / "optimization" / "cache" / str(case.case_id)
        npz_path = root / "exact_combined.npz"
        json_path = root / "exact_combined.json"
        if not npz_path.exists() and not json_path.exists():
            return None
        if not npz_path.is_file():
            raise ContractMismatchError("exact replay metadata exists without its atomic array cache")
        identity = self._exact_cache_identity(case, selected.decisions)
        with np.load(npz_path, allow_pickle=False) as payload:
            required = {
                "decisions", "influent", "target", "selected_mechanistic_state",
                "selected_complete_state", "raw_surrogate_prediction", "record_json",
            }
            if set(payload.files) != required:
                raise ContractMismatchError("exact replay cache array schema changed")
            arrays = {name: payload[name].copy() for name in payload.files}
        shapes = {
            "decisions": (5,), "influent": (20,), "target": (170,),
            "selected_mechanistic_state": (110,), "selected_complete_state": (170,),
            "raw_surrogate_prediction": (170,), "record_json": (),
        }
        if any(arrays[name].shape != shape for name, shape in shapes.items()):
            raise ContractMismatchError("exact replay cache dimensions changed")
        try:
            record = json.loads(str(arrays["record_json"].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ContractMismatchError("exact replay cache record is unreadable") from exc
        for name, expected in identity.items():
            if record.get(name) != expected:
                raise ContractMismatchError(f"exact replay cache identity changed at {name}")
        if record.get("kind") != "exact_bdf_replay" or record.get("schema_version") != 2:
            raise ContractMismatchError("exact replay cache type changed")
        for name in required - {"record_json"}:
            if record.get("array_sha256", {}).get(name) != _float_array_sha256(
                arrays[name]
            ):
                raise ContractMismatchError(
                    f"exact replay cache array digest changed for {name}"
                )
        if not np.array_equal(arrays["decisions"], np.asarray(selected.decisions)):
            raise ContractMismatchError("exact replay cache selected controls changed")
        if not np.array_equal(arrays["influent"], np.asarray(case.influent)):
            raise ContractMismatchError("exact replay cache influent changed")
        public_record = {**record, "npz_sha256": sha256_file(npz_path)}
        if json_path.is_file():
            if _load_json(json_path) != _json_value(public_record):
                raise ContractMismatchError("exact replay JSON and atomic cache disagree")
        else:
            atomic_json(json_path, public_record)
        return public_record

    def _save_exact_cache(
        self,
        case: Any,
        selected: Any,
        selected_complete_state: np.ndarray,
        raw: np.ndarray,
        target: np.ndarray,
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        root = self.run_root / "optimization" / "cache" / str(case.case_id)
        root.mkdir(parents=True, exist_ok=True)
        npz_path = root / "exact_combined.npz"
        json_path = root / "exact_combined.json"
        arrays = {
            "decisions": np.asarray(selected.decisions, dtype=np.float64),
            "influent": np.asarray(case.influent, dtype=np.float64),
            "target": np.asarray(target, dtype=np.float64),
            "selected_mechanistic_state": np.asarray(selected.state, dtype=np.float64),
            "selected_complete_state": np.asarray(selected_complete_state, dtype=np.float64),
            "raw_surrogate_prediction": np.asarray(raw, dtype=np.float64),
        }
        stored_record = {
            **record,
            "array_sha256": {
                name: _float_array_sha256(value) for name, value in arrays.items()
            },
        }
        encoded = json.dumps(
            _json_value(stored_record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        atomic_npz(
            npz_path,
            **arrays,
            record_json=np.asarray(encoded),
        )
        public_record = {**stored_record, "npz_sha256": sha256_file(npz_path)}
        atomic_json(json_path, public_record)
        return dict(public_record)

    def _exact_replay(
        self,
        case_index: int,
        case: Mapping[str, Any],
        selected: Any | None,
        selected_complete_state: np.ndarray | None,
        model: QuadraticSurrogate,
        assets: Any,
        quality_scale: np.ndarray,
        inventory_scale: float,
        feed_mass_scale: np.ndarray,
    ) -> dict[str, Any]:
        if selected is None:
            return {
                "attempted": False,
                "accepted": False,
                "outcome": "no accepted NLP start",
            }
        if selected_complete_state is None:
            raise StageExecutionError("selected combined NLP result has no reconstructed complete state")
        case_definition = self._case_definition(case)
        identity = self._exact_cache_identity(case_definition, selected.decisions)
        logical_key = str(identity["exact_cache_key"])
        cached = self._load_exact_cache(case_definition, selected)
        if cached is not None:
            self._reconcile_interrupted_invocation(
                "exact_bdf_replay", logical_key, case_id=str(case_definition.case_id),
            )
            self._ledger_event(
                "exact_bdf_replay", logical_key, "reused",
                case_id=str(case_definition.case_id), source="atomic_exact_cache",
                verification="contract, case, selected controls, dimensions, and SHA-256 passed",
            )
            return cached

        self._reconcile_interrupted_invocation(
            "exact_bdf_replay", logical_key, case_id=str(case_definition.case_id),
        )
        self._ledger_event(
            "exact_bdf_replay", logical_key, "attempted",
            case_id=str(case_definition.case_id),
        )
        try:
            row = self._solve_row(
                1_000_000 + case_index,
                selected.decisions,
                case_definition.influent,
            )
            raw = model.predict(selected.decisions, case_definition.influent)
            target_finite = bool(
                row.target.shape == (mechanism.TARGET_SIZE,)
                and np.all(np.isfinite(row.target))
            )
            record: dict[str, Any] = {
                "schema_version": 2,
                "kind": "exact_bdf_replay",
                **identity,
                "attempted": True,
                "generator_accepted": bool(row.accepted),
                "elapsed_seconds": float(row.elapsed_seconds),
                "error": row.error,
                "diagnostics": row.diagnostics,
                "accepted": False,
            }
            if not target_finite:
                record["outcome"] = "exact integration failure"
            else:
                exact_y = np.concatenate((row.target[20:120], row.target[160:170]))
                physical = mechanism.diagnostics(
                    exact_y,
                    mechanism.OperatingPoint(*selected.decisions.tolist()),
                    case_definition.influent,
                    residual_tolerance=float(
                        self.config["mechanistic_solver"]
                        ["scaled_derivative_acceptance_d_inv"]
                    ),
                    check_stability=True,
                )
                exact = self._engineering_from_target(
                    selected.decisions, row.target, case["weights"],
                    float(case["underflow_tss_limit"]), quality_scale,
                    inventory_scale, feed_mass_scale,
                )
                selected_engineering = self._engineering_from_target(
                    selected.decisions, selected_complete_state, case["weights"],
                    float(case["underflow_tss_limit"]), quality_scale,
                    inventory_scale, feed_mass_scale,
                )
                score = float(
                    np.mean(np.square((row.target - raw) / model.response_scale))
                )
                normalized_fidelity = score / float(self._load_calibration_delta())
                branch_error = float(
                    np.max(np.abs((exact_y - selected.state) / assets.state_scale))
                )
                flow_difference = (
                    np.asarray(selected_complete_state[120:160]) - row.target[120:160]
                ).reshape(2, mechanism.N_COMPONENTS)
                composite_difference = np.vstack(
                    (
                        selected_engineering["effluent_composites"]
                        - exact["effluent_composites"],
                        selected_engineering["underflow_composites"]
                        - exact["underflow_composites"],
                    )
                )
                record.update(
                    engineering=exact,
                    selected_engineering=selected_engineering,
                    physical_diagnostics=physical,
                    residual_stability_passed=bool(
                        row.accepted and physical["passed"]
                    ),
                    fidelity_score=score,
                    normalized_fidelity=normalized_fidelity,
                    fidelity_constraint_row=normalized_fidelity - 1.0,
                    fidelity_passed=self._exact_fidelity_passed(normalized_fidelity),
                    branch_scaled_infinity=branch_error,
                    selected_state_scaled_rms=float(
                        np.sqrt(np.mean(np.square(
                            (row.target - selected_complete_state) / model.response_scale
                        )))
                    ),
                    selected_state_scaled_infinity=float(
                        np.max(np.abs(
                            (row.target - selected_complete_state) / model.response_scale
                        ))
                    ),
                    selected_state_objective_difference=(
                        selected_engineering["objective"] - exact["objective"]
                    ),
                    nlp_minus_bdf_outlet_component_flow=flow_difference,
                    nlp_minus_bdf_outlet_composite_concentration=composite_difference,
                    dynamic_balance_scaled_residual_inf=physical[
                        "scaled_residual_inf"
                    ],
                )
                branch_limit = float(
                    self.config["optimization"]["exact_validation"]
                    ["branch_scaled_infinity_tolerance"]
                )
                if not record["residual_stability_passed"]:
                    record["outcome"] = "residual/stability failure"
                elif not exact["domain_engineering_passed"]:
                    record["outcome"] = "domain/engineering failure"
                elif not record["fidelity_passed"]:
                    record["outcome"] = "statistical-fidelity-validation failure"
                elif branch_error > branch_limit:
                    record["outcome"] = "branch/smoothing-validation failure"
                else:
                    record["accepted"] = True
                    record["outcome"] = "validated combined recommendation"
            public_record = self._save_exact_cache(
                case_definition, selected, selected_complete_state, raw,
                row.target, record,
            )
        except BaseException as exc:
            self._ledger_event(
                "exact_bdf_replay", logical_key, "interrupted",
                case_id=str(case_definition.case_id),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._ledger_event(
            "exact_bdf_replay", logical_key, "completed",
            case_id=str(case_definition.case_id),
            accepted=bool(public_record["accepted"]),
            outcome=str(public_record["outcome"]),
        )
        return public_record

    def _load_calibration_delta(self) -> float:
        return float(_load_json(self.run_root / "metrics" / "calibration.json")["delta"])

    def _case_summary(
        self,
        record: Mapping[str, Any],
        results: Sequence[Any],
        problem: Any,
        model: QuadraticSurrogate,
        assets: Any,
        case_index: int,
        quality_scale: np.ndarray,
        inventory_scale: float,
        feed_mass_scale: np.ndarray,
    ) -> dict[str, Any]:
        nlp = self._nlp()
        case_definition = self._case_definition(record)
        selected = nlp.select_best_start(results)
        evaluation = (
            None
            if selected is None
            else nlp.evaluate_problem(problem, selected.primal, case_definition)
        )
        exact_replay = self._exact_replay(
            case_index,
            record,
            selected,
            None if evaluation is None else evaluation["complete_state"],
            model,
            assets,
            quality_scale,
            inventory_scale,
            feed_mass_scale,
        )
        summary = {
            "case_id": record["case_id"],
            "case_class": record["case_class"],
            "sensitivity_family": record.get("sensitivity_family"),
            "influent": record["influent"],
            "weights": record["weights"],
            "underflow_tss_limit": record["underflow_tss_limit"],
            "accepted_starts": sum(bool(result.accepted) for result in results),
            "selected_start": None if selected is None else int(selected.start_index),
            "selected_objective": None if selected is None else float(selected.objective),
            "selected_decisions": None if selected is None else selected.decisions,
            "nlp_diagnostics": None if evaluation is None else evaluation["diagnostics"],
            "exact": exact_replay,
            "bdf_replays_attempted": int(exact_replay["attempted"]),
        }
        atomic_json(
            self.run_root / "optimization" / "cache" / str(record["case_id"]) / "result.json",
            summary,
        )
        return summary

    @staticmethod
    def _flat_case_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
        row = {
            key: value
            for key, value in summary.items()
            if key not in {"influent", "weights", "nlp_diagnostics", "exact", "selected_decisions"}
        }
        decisions = summary.get("selected_decisions")
        for index, name in enumerate(("H", "a", "r_I", "r_R", "w")):
            row[f"selected_{name}"] = None if decisions is None else decisions[index]
        exact = summary["exact"]
        row["outcome"] = exact["outcome"]
        row["exact_valid"] = bool(exact.get("accepted", False))
        row["exact_seconds"] = exact.get("elapsed_seconds")
        for name in (
            "fidelity_score", "normalized_fidelity", "branch_scaled_infinity",
            "selected_state_scaled_rms", "selected_state_scaled_infinity",
            "selected_state_objective_difference", "fidelity_constraint_row",
            "dynamic_balance_scaled_residual_inf", "fidelity_passed",
            "residual_stability_passed", "generator_accepted",
        ):
            row[f"exact_{name}"] = exact.get(name)
        for name, value in exact.get("diagnostics", {}).items():
            if isinstance(value, (bool, int, float)):
                row[f"exact_diagnostic_{name}"] = value
        engineering = exact.get("engineering", {})
        for name in (
            "objective", "feed_tss", "boundary_solids", "reactor_solids_inventory",
            "clarifier_solids_inventory", "normalized_clarifier_inventory", "inventory",
            "srt_days", "surface_overflow_rate", "solids_loading_rate", "underflow_tss",
            "maximum_component_mass_closure",
        ):
            row[f"exact_{name}"] = engineering.get(name)
        for index, name in enumerate(("quality", "H", "a", "r_I", "r_R", "wasted_solids")):
            components = engineering.get("components")
            weighted = engineering.get("weighted_components")
            row[f"exact_component_{name}"] = None if components is None else components[index]
            row[f"exact_weighted_{name}"] = None if weighted is None else weighted[index]
        for prefix, key, names in (
            ("exact_effluent", "effluent_composites", ("cod", "tn", "tp", "tss")),
            ("exact_underflow", "underflow_composites", ("cod", "tn", "tp", "tss")),
            ("exact_domain", "domain_rows", ("feed_tss", "boundary_solids")),
            ("exact_engineering_row", "engineering_rows", ("srt_min", "srt_max", "sor", "slr", "underflow_tss")),
        ):
            values = engineering.get(key)
            for index, name in enumerate(names):
                row[f"{prefix}_{name}"] = None if values is None else values[index]
        flow_difference = exact.get("nlp_minus_bdf_outlet_component_flow")
        for route_index, route in enumerate(("effluent", "underflow")):
            for component_index, component in enumerate(mechanism.COMPONENTS):
                row[
                    f"exact_nlp_minus_bdf_{route}_component_flow_{component}"
                ] = (
                    None
                    if flow_difference is None
                    else flow_difference[route_index][component_index]
                )
        composite_difference = exact.get(
            "nlp_minus_bdf_outlet_composite_concentration"
        )
        for route_index, route in enumerate(("effluent", "underflow")):
            for composite_index, composite in enumerate(("cod", "tn", "tp", "tss")):
                row[
                    f"exact_nlp_minus_bdf_{route}_{composite}_concentration"
                ] = (
                    None
                    if composite_difference is None
                    else composite_difference[route_index][composite_index]
                )
        for index, value in enumerate(engineering.get("clarifier_layers", ())):
            row[f"exact_clarifier_layer_{index + 1}"] = value
        for index, component in enumerate(mechanism.COMPONENTS):
            for prefix, key in (
                ("exact_effluent_component", "effluent_components"),
                ("exact_underflow_component", "underflow_components"),
                ("exact_component_mass_closure", "component_mass_closure"),
                ("exact_underflow_recovery", "component_underflow_recovery"),
                ("exact_concentration_ratio", "component_concentration_ratio"),
            ):
                values = engineering.get(key)
                row[f"{prefix}_{component}"] = None if values is None else values[index]
        for prefix, key in (
            ("exact_reactor_do", "reactor_dissolved_oxygen_profile"),
            ("exact_reactor_tn", "reactor_tn_profile"),
            ("exact_reactor_tp", "reactor_tp_profile"),
            ("exact_reactor_tss", "reactor_tss_profile"),
        ):
            for index, value in enumerate(engineering.get(key, ())):
                row[f"{prefix}_{index + 1}"] = value
        nlp_diagnostic_names = (
            "component_quality", "component_H", "component_a", "component_r_I",
            "component_r_R", "component_wasted_solids", "weighted_quality",
            "weighted_H", "weighted_a", "weighted_r_I", "weighted_r_R",
            "weighted_wasted_solids", "feed_tss", "boundary_solids",
            "solids_inventory", "srt_days", "surface_overflow_rate",
            "solids_loading_rate", "underflow_tss", "effluent_cod",
            "effluent_tn", "effluent_tp", "effluent_tss",
            "engineering_objective", "fidelity", "normalized_fidelity",
            "leverage", "maximum_scaled_dynamic_residual",
        )
        for name in nlp_diagnostic_names:
            row[f"nlp_{name}"] = None
        for name, value in (summary.get("nlp_diagnostics") or {}).items():
            row[f"nlp_{name}"] = value
        for index, name in enumerate(("quality", "H", "a", "r_I", "r_R", "wasted_solids")):
            row[f"weight_{name}"] = summary["weights"][index]
        return _json_value(row)

    def _stage_optimization(self) -> dict[str, Any]:
        cases = self._load_case_records()
        decisions, influents, targets, blocks = self._load_dataset()
        development = blocks == 0
        problem, assets, model = self._build_nlp_problems()
        _, _, stored_assets = load_surrogate_bundle(
            self.run_root / "models" / "development_surrogate.npz"
        )
        feed_mass_scale = stored_assets["clarifier_feed_mass_scale"]
        summaries: list[dict[str, Any]] = []
        for case_index, record in enumerate(cases):
            results = self._solve_nlp_case(
                record, problem, assets, model,
                decisions[development], influents[development], targets[development],
            )
            summaries.append(
                self._case_summary(
                    record, results, problem, model, assets,
                    case_index, assets.quality_scale, assets.inventory_scale,
                    feed_mass_scale,
                )
            )
        frame = pd.DataFrame([self._flat_case_summary(summary) for summary in summaries])
        atomic_parquet(self.run_root / "optimization" / "case_summary.parquet", frame)
        invocation_summary = self._invocation_summary()
        nlp_invocations = invocation_summary["kinds"].get("combined_nlp_start", {})
        exact_invocations = invocation_summary["kinds"].get("exact_bdf_replay", {})
        realized = {
            "case_count": len(cases),
            "combined_nlp_starts": 9 * len(cases),
            "bdf_replays_realized": int(sum(summary["bdf_replays_attempted"] for summary in summaries)),
            "bdf_replays_maximum": len(cases),
            "combined_nlp_start_attempts": int(nlp_invocations.get("attempted", 0)),
            "combined_nlp_start_completions": int(nlp_invocations.get("completed", 0)),
            "combined_nlp_start_reuses": int(nlp_invocations.get("reused", 0)),
            "combined_nlp_start_interruptions": int(nlp_invocations.get("interrupted", 0)),
            "exact_bdf_replay_attempts": int(exact_invocations.get("attempted", 0)),
            "exact_bdf_replay_completions": int(exact_invocations.get("completed", 0)),
            "exact_bdf_replay_reuses": int(exact_invocations.get("reused", 0)),
            "exact_bdf_replay_interruptions": int(exact_invocations.get("interrupted", 0)),
            "physical_qp_evaluations": 0,
            "direct_evaluations": 0,
            "validated_combined_recommendations": int(sum(summary["exact"].get("accepted", False) for summary in summaries)),
            "invocation_ledger": invocation_summary,
        }
        atomic_json(self.run_root / "optimization" / "summary.json", realized)
        return realized

    @staticmethod
    def _outcome_table(
        frame: pd.DataFrame,
        column: str,
        categories: Sequence[str],
    ) -> pd.DataFrame:
        values = frame[column].astype(str) if column in frame else pd.Series(dtype=str)
        denominator = int(len(values))
        return pd.DataFrame(
            {
                "outcome": list(categories),
                "count": [int(np.sum(values == category)) for category in categories],
                "denominator": denominator,
                "fraction": [float(np.mean(values == category)) if denominator else math.nan for category in categories],
            }
        )

    @staticmethod
    def _validate_replayed_frame(
        stored: pd.DataFrame,
        current: pd.DataFrame,
        label: str,
    ) -> pd.DataFrame:
        """Validate a reconstructed table by column identity, then stored order.

        JSON sidecars are deliberately serialized with sorted keys, whereas
        Parquet/CSV tables retain the insertion order of the in-memory result
        dictionaries.  Column order therefore is not a scientific difference.
        Names, uniqueness, row count, and every value remain mandatory.
        """

        if (
            not stored.columns.is_unique
            or not current.columns.is_unique
            or len(stored) != len(current)
            or set(stored.columns) != set(current.columns)
        ):
            raise StageExecutionError(f"terminal replay schema changed for {label}")
        aligned = current.loc[:, stored.columns]
        try:
            pd.testing.assert_frame_equal(
                stored.reset_index(drop=True),
                aligned.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1.0e-11,
                atol=1.0e-13,
            )
        except AssertionError as exc:
            raise StageExecutionError(
                f"terminal replay values changed for {label}"
            ) from exc
        return aligned

    def _robustness_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        robust = frame.loc[frame["case_class"] == "robustness"]
        columns = [
            column for column in robust.columns
            if column.startswith(("selected_", "nlp_", "exact_"))
        ]
        records = []
        for column in columns:
            values = pd.to_numeric(robust[column], errors="coerce").dropna().to_numpy()
            q25 = _nearest_rank(values, 0.25)
            q75 = _nearest_rank(values, 0.75)
            records.append(
                {
                    "quantity": column,
                    "eligible_n": int(values.size),
                    "mean": float(np.mean(values)) if values.size else math.nan,
                    "q25": q25,
                    "median_q50": _nearest_rank(values, 0.50),
                    "q75": q75,
                    "interquartile_range": q75 - q25,
                    "p95": _nearest_rank(values, 0.95),
                    "maximum": float(np.max(values)) if values.size else math.nan,
                }
            )
        return pd.DataFrame(
            records,
            columns=(
                "quantity", "eligible_n", "mean", "q25", "median_q50",
                "q75", "interquartile_range", "p95", "maximum",
            ),
        )

    def _bound_activity(self, frame: pd.DataFrame) -> pd.DataFrame:
        robust = frame.loc[frame["case_class"] == "robustness"]
        cases = {
            str(record["case_id"]): self._case_definition(record)
            for record in self._load_case_records()
        }
        tolerance = float(self.config["reporting"]["activity"]["control_normalized_distance"])
        inequality_lower = float(self.config["reporting"]["activity"]["scaled_constraint_lower"])
        inequality_upper = float(self.config["reporting"]["activity"]["scaled_constraint_upper"])
        bounds = np.asarray(list(self.config["process"]["decision_bounds"].values()), dtype=np.float64)
        records: list[dict[str, Any]] = []
        eligible = robust["selected_start"].notna()
        for coordinate, name in enumerate(("H", "a", "r_I", "r_R", "w")):
            values = pd.to_numeric(robust.loc[eligible, f"selected_{name}"], errors="coerce").to_numpy()
            normalized = (values - bounds[coordinate, 0]) / np.diff(bounds[coordinate])[0]
            for side, active in (("lower", normalized <= tolerance), ("upper", 1.0 - normalized <= tolerance)):
                records.append({"route": "combined_nlp", "kind": "control_bound", "quantity": f"{name}_{side}", "eligible_n": int(values.size), "active_count": int(np.sum(active)), "active_fraction": float(np.mean(active)) if values.size else math.nan})
        groups = (
            ("domain_feed_tss", 0),
            ("domain_boundary_solids", 1),
            ("engineering_srt_min", 2),
            ("engineering_srt_max", 3),
            ("engineering_sor", 4),
            ("engineering_slr", 5),
            ("engineering_underflow_tss", 6),
            ("trust_fidelity", 7),
            ("trust_leverage", 8),
        )
        activity: dict[tuple[str, str], list[bool]] = {
            **{("combined_nlp", name): [] for name, _ in groups},
            **{("exact_bdf", name): [] for name, _ in groups[:7]},
        }
        for row in robust.to_dict(orient="records"):
            selected = row.get("selected_start")
            if selected is not None and not (isinstance(selected, float) and np.isnan(selected)):
                inequalities = self._load_nlp_case(cases[str(row["case_id"])])[int(selected)].inequality
                for name, index in groups:
                    value = inequalities[index]
                    activity[("combined_nlp", name)].append(
                        bool(inequality_lower <= value <= inequality_upper)
                    )
            exact_path = self.run_root / "optimization" / "cache" / str(row["case_id"]) / "exact_combined.json"
            if exact_path.is_file():
                exact = _load_json(exact_path).get("engineering")
                if exact:
                    exact_rows = [*exact["domain_rows"], *exact["engineering_rows"]]
                    for (name, _), value in zip(groups[:7], exact_rows, strict=True):
                        activity[("exact_bdf", name)].append(
                            bool(inequality_lower <= float(value) <= inequality_upper)
                        )
        for (route, name), active in sorted(activity.items()):
            records.append({"route": route, "kind": "scaled_constraint", "quantity": name, "eligible_n": len(active), "active_count": int(np.sum(active)), "active_fraction": float(np.mean(active)) if active else math.nan})
        return pd.DataFrame(
            records,
            columns=("route", "kind", "quantity", "eligible_n", "active_count", "active_fraction"),
        )

    def _timing_summary(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []

        def stage_memory(*stages: str) -> int:
            return max(
                (
                    int(
                        self._manifest.get("stages", {})
                        .get(stage, {})
                        .get("stage_high_water_resident_memory_bytes", 0)
                    )
                    for stage in stages
                ),
                default=0,
            )

        def add(name: str, values: Sequence[float], memory: int = 0) -> None:
            array = np.asarray(values, dtype=np.float64)
            array = array[np.isfinite(array)]
            if array.size == 0:
                return
            records.append(
                {
                    "stage": name,
                    "count": int(array.size),
                    "total_seconds": float(np.sum(array)),
                    "mean_seconds": float(np.mean(array)),
                    "median_seconds": _nearest_rank(array, 0.50),
                    "q1_seconds": _nearest_rank(array, 0.25),
                    "q3_seconds": _nearest_rank(array, 0.75),
                    "interquartile_range_seconds": _nearest_rank(array, 0.75) - _nearest_rank(array, 0.25),
                    "p95_seconds": _nearest_rank(array, 0.95),
                    "maximum_seconds": float(np.max(array)),
                    "maximum_resident_memory_bytes": int(memory),
                }
            )

        with np.load(self.run_root / "datasets" / "mechanistic_dataset.npz", allow_pickle=False) as payload:
            elapsed = payload["elapsed_seconds"]
            blocks = payload["block"]
        for code, name in enumerate(("development", "calibration", "assessment")):
            owning_stages = ("pilot", "dataset") if code == 0 else ("dataset",)
            add(
                f"mechanistic_data_generation_{name}",
                elapsed[blocks == code],
                stage_memory(*owning_stages),
            )
        times: list[float] = []
        for path in sorted((self.run_root / "optimization" / "cache").glob("*/starts.json")):
            times.extend(float(record["elapsed_seconds"]) for record in _load_json(path)["combined"])
        add("combined_nlp_start", times, stage_memory("nlp_preflight", "optimization"))
        exact_times = []
        for path in sorted((self.run_root / "optimization" / "cache").glob("*/exact_combined.json")):
            value = _load_json(path).get("elapsed_seconds")
            if value is not None:
                exact_times.append(float(value))
        add("exact_bdf_validation", exact_times, stage_memory("optimization"))
        fit = _load_json(self.run_root / "models" / "development_assets.json")
        add("development_fit_and_scaling", [float(fit["fit_seconds"])], stage_memory("fit"))
        calibration = _load_json(self.run_root / "metrics" / "calibration.json")
        assessment = _load_json(self.run_root / "metrics" / "assessment_summary.json")
        add("calibration", [float(calibration["elapsed_seconds"])], stage_memory("calibration"))
        add("assessment", [float(assessment["elapsed_seconds"])], stage_memory("assessment"))
        reporting_path = self.run_root / "timing" / "report_generation.json"
        if reporting_path.is_file():
            reporting = _load_json(reporting_path)
            add(
                "reporting",
                [float(reporting["elapsed_seconds"])],
                int(reporting["stage_high_water_resident_memory_bytes"]),
            )
        for stage in STAGES[: STAGES.index("report")]:
            stage_record = self._manifest.get("stages", {}).get(stage, {})
            duration = stage_record.get("stage_wall_seconds")
            if duration is not None:
                add(
                    f"checkpoint_{stage}",
                    [float(duration)],
                    int(stage_record.get("stage_high_water_resident_memory_bytes", 0)),
                )
        return pd.DataFrame(records)

    def _workload_table(
        self, frame: pd.DataFrame, realized: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Return declared logical work beside durable attempt/reuse counts."""

        workloads = self._scientific_workloads(self.profile_name)
        exact_logical = int(realized["bdf_replays_realized"])
        realized_workload = {
            "dataset_bdf_invocations": self.sample_count,
            "selected_point_bdf_invocations_maximum": exact_logical,
            "bdf_invocations_maximum": self.sample_count + exact_logical,
            "combined_nlp_starts": int(realized["combined_nlp_starts"]),
            "optimization_cases": int(realized["case_count"]),
            "robustness_cases": int(np.sum(frame["case_class"] == "robustness")),
            "sensitivity_cases": int(np.sum(frame["case_class"] == "sensitivity")),
            "physical_qp_evaluations": int(realized["physical_qp_evaluations"]),
            "direct_evaluations": int(realized["direct_evaluations"]),
        }
        exact_attempts = int(realized["exact_bdf_replay_attempts"])
        exact_completions = int(realized["exact_bdf_replay_completions"])
        exact_reuses = int(realized["exact_bdf_replay_reuses"])
        exact_interruptions = int(realized["exact_bdf_replay_interruptions"])
        nlp_attempts = int(realized["combined_nlp_start_attempts"])
        nlp_completions = int(realized["combined_nlp_start_completions"])
        nlp_reuses = int(realized["combined_nlp_start_reuses"])
        nlp_interruptions = int(realized["combined_nlp_start_interruptions"])
        attempt_counts = {
            "dataset_bdf_invocations": self.sample_count,
            "selected_point_bdf_invocations_maximum": exact_attempts,
            "bdf_invocations_maximum": self.sample_count + exact_attempts,
            "combined_nlp_starts": nlp_attempts,
        }
        reuse_counts = {
            "selected_point_bdf_invocations_maximum": exact_reuses,
            "bdf_invocations_maximum": exact_reuses,
            "combined_nlp_starts": nlp_reuses,
        }
        completion_counts = {
            "dataset_bdf_invocations": self.sample_count,
            "selected_point_bdf_invocations_maximum": exact_completions,
            "bdf_invocations_maximum": self.sample_count + exact_completions,
            "combined_nlp_starts": nlp_completions,
        }
        interruption_counts = {
            "selected_point_bdf_invocations_maximum": exact_interruptions,
            "bdf_invocations_maximum": exact_interruptions,
            "combined_nlp_starts": nlp_interruptions,
        }
        projection = _load_json(
            self.run_root / "timing" / "nlp_preflight.json"
        )["projection"]
        table = pd.DataFrame(
            [
                {
                    "quantity": key,
                    "planned_maximum": int(value),
                    "realized": int(realized_workload[key]),
                    "realized_logical_calls": int(realized_workload[key]),
                    "actual_solver_attempts": int(attempt_counts.get(key, 0)),
                    "completed_attempts": int(completion_counts.get(key, 0)),
                    "interrupted_attempts": int(interruption_counts.get(key, 0)),
                    "verified_cache_reuses": int(reuse_counts.get(key, 0)),
                    "projection_safety_factor": float(projection["safety_factor"]),
                    "projected_full_core_days": float(projection["projected_core_days"]),
                    "projected_full_resident_memory_bytes": int(
                        projection["projected_resident_memory_bytes"]
                    ),
                    "projection_passed": bool(projection["projection_passed"]),
                }
                for key, value in workloads.items()
            ]
        )
        if table.isna().any(axis=None):
            raise StageExecutionError("workload table contains an undefined count")
        return table

    def _stage_report(self) -> dict[str, Any]:
        reporting_started = perf_counter_ns()
        reporting_memory = _StageMemoryMonitor().start()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frame = pd.read_parquet(self.run_root / "optimization" / "case_summary.parquet")
        robustness = frame.loc[frame["case_class"] == "robustness"].reset_index(drop=True)
        categories = self.config["reporting"]["combined_outcome_classes"]
        atomic_csv(
            self.run_root / "tables" / "combined_route_outcomes.csv",
            self._outcome_table(robustness, "outcome", categories),
        )
        atomic_csv(self.run_root / "tables" / "robustness_summary.csv", self._robustness_summary(frame))
        atomic_csv(self.run_root / "tables" / "bound_activity.csv", self._bound_activity(frame))
        tss = pd.concat(
            [frame.loc[frame["case_id"] == "underflow_tss_12000"], frame.loc[frame["case_id"] == "nominal"], frame.loc[frame["case_id"] == "underflow_tss_20000"]],
            ignore_index=True,
        )
        atomic_csv(self.run_root / "tables" / "underflow_tss_sensitivity.csv", tss)
        weights = frame.loc[frame["sensitivity_family"] == "objective_weights"].copy()
        atomic_csv(self.run_root / "tables" / "objective_weight_sensitivity.csv", weights)
        realized = _load_json(self.run_root / "optimization" / "summary.json")
        workload_frame = self._workload_table(frame, realized)
        atomic_csv(self.run_root / "tables" / "workload.csv", workload_frame)

        with np.load(self.run_root / "predictions" / "assessment_predictions.npz", allow_pickle=False) as payload:
            truth = payload["truth"].ravel()
            raw = payload["raw"].ravel()
        if truth.size > 20_000:
            index = np.linspace(0, truth.size - 1, 20_000, dtype=np.int64)
            truth, raw = truth[index], raw[index]
        figure, axis = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)
        limits = (float(min(np.min(truth), np.min(raw))), float(max(np.max(truth), np.max(raw))))
        axis.scatter(truth, raw, s=3, alpha=0.2, rasterized=True)
        axis.plot(limits, limits, color="black", linewidth=1)
        axis.set(xlabel="Mechanistic state", ylabel="Raw surrogate state")
        figure.savefig(self.run_root / "figures" / "assessment_parity.png", dpi=220)
        figure.savefig(self.run_root / "figures" / "assessment_parity.pdf")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
        values = pd.to_numeric(frame["exact_objective"], errors="coerce")
        positions = np.arange(len(frame))
        finite = np.isfinite(values.to_numpy(dtype=np.float64))
        if np.any(finite):
            axis.bar(positions[finite], values.to_numpy()[finite], color="#4472C4")
        else:
            axis.text(
                0.5, 0.5, "No finite exact objectives",
                ha="center", va="center", transform=axis.transAxes,
            )
        axis.set(xlabel="Case", ylabel="Selected exact engineering objective")
        axis.set_xticks(positions, frame["case_id"], rotation=90)
        figure.savefig(self.run_root / "figures" / "optimization_objectives.png", dpi=220)
        figure.savefig(self.run_root / "figures" / "optimization_objectives.pdf")
        plt.close(figure)

        reporting_timing = {
            "elapsed_seconds": (perf_counter_ns() - reporting_started) / 1.0e9,
            "stage_high_water_resident_memory_bytes": reporting_memory.stop(),
            "scope": "table and figure construction through self-timing serialization",
        }
        atomic_json(
            self.run_root / "timing" / "report_generation.json",
            reporting_timing,
        )
        timing = self._timing_summary()
        atomic_csv(self.run_root / "tables" / "timing_summary.csv", timing)

        table_names = (
            "workload.csv", "timing_summary.csv", "robustness_summary.csv",
            "combined_route_outcomes.csv", "bound_activity.csv",
            "underflow_tss_sensitivity.csv", "objective_weight_sensitivity.csv",
        )
        table_inventory = {
            name: {
                "rows": int(len(pd.read_csv(self.run_root / "tables" / name))),
                "columns": list(pd.read_csv(self.run_root / "tables" / name, nrows=0).columns),
                "sha256": sha256_file(self.run_root / "tables" / name),
            }
            for name in table_names
        }
        report = {
            "run_id": self.run_id,
            "profile": self.profile_name,
            "article_eligible": bool(self.profile.get("article_eligible", False)),
            "release_status": "provisional_pending_terminal_replay",
            "release_authority": "COMPLETED.json is created only after terminal replay and sealing",
            "contract_sha256": self._contract["contract_sha256"],
            "contract_artifact": "inputs/contract.json",
            "numerical_environment": self._contract["numerical_environment"],
            "casadi_ipopt_binary_sha256": self._contract["casadi_ipopt_binary_sha256"],
            "host": self._contract["host"],
            "data_generation_worker_count": int(self.profile["parallel_workers"]),
            "information_blocks": {"development": self.development_count, "calibration": self.calibration_count, "assessment": self.assessment_count},
            "calibration": _load_json(self.run_root / "metrics" / "calibration.json"),
            "assessment": _load_json(self.run_root / "metrics" / "assessment_summary.json"),
            "nlp_preflight": _load_json(self.run_root / "timing" / "nlp_preflight.json"),
            "optimization": realized,
            "invocation_ledger": realized["invocation_ledger"],
            "reporting_timing": reporting_timing,
            "table_inventory": table_inventory,
            "generated_utc": utc_now(),
        }
        atomic_json(self.run_root / "report" / "summary.json", report)
        return {"tables": 7, "figures": 4, "report_sha256": sha256_file(self.run_root / "report" / "summary.json")}

    def _terminal_replay(self) -> dict[str, Any]:
        nlp = self._nlp()
        decisions, influents, targets, blocks = self._load_dataset()
        model, _, stored_assets = load_surrogate_bundle(
            self.run_root / "models" / "development_surrogate.npz"
        )

        maximum_difference = 0.0

        def compare(stored: Any, current: Any, label: str) -> None:
            """Compare a persisted scientific value, accepting JSON's null-for-NaN form."""

            nonlocal maximum_difference
            if isinstance(current, Mapping):
                if not isinstance(stored, Mapping) or set(stored) != set(current):
                    raise StageExecutionError(f"terminal replay keys changed for {label}")
                for name in current:
                    compare(stored[name], current[name], f"{label}.{name}")
                return
            if isinstance(current, (list, tuple, np.ndarray)):
                current_values = np.asarray(current).reshape(-1).tolist()
                if not isinstance(stored, (list, tuple, np.ndarray)):
                    raise StageExecutionError(f"terminal replay dimensions changed for {label}")
                stored_values = np.asarray(stored, dtype=object).reshape(-1).tolist()
                if len(stored_values) != len(current_values):
                    raise StageExecutionError(f"terminal replay dimensions changed for {label}")
                for index, (old, new) in enumerate(zip(stored_values, current_values, strict=True)):
                    compare(old, new, f"{label}[{index}]")
                return
            if isinstance(current, (bool, np.bool_)):
                if bool(stored) != bool(current):
                    raise StageExecutionError(f"terminal replay changed {label}")
                return
            if isinstance(current, str) or current is None:
                if stored != current:
                    raise StageExecutionError(f"terminal replay changed {label}")
                return
            new_value = float(current)
            if stored is None:
                if not math.isfinite(new_value):
                    return
                raise StageExecutionError(f"terminal replay changed {label}")
            old_value = float(stored)
            if not math.isfinite(old_value) or not math.isfinite(new_value):
                if (math.isnan(old_value) and math.isnan(new_value)) or old_value == new_value:
                    return
                raise StageExecutionError(f"terminal replay changed {label}")
            if math.isnan(old_value) and math.isnan(new_value):
                return
            difference = abs(old_value - new_value)
            maximum_difference = max(maximum_difference, difference)
            if difference > 1.0e-11 * (1.0 + abs(old_value)):
                raise StageExecutionError(f"terminal replay changed {label}")

        calibration_rows = blocks == 1
        with np.load(self.run_root / "predictions" / "calibration_scores.npz", allow_pickle=False) as payload:
            calibration_truth, calibration_raw = payload["truth"], payload["raw"]
        compare(calibration_truth, targets[calibration_rows], "calibration truth")
        replay_calibration_raw = model.predict(
            decisions[calibration_rows], influents[calibration_rows]
        )
        compare(calibration_raw, replay_calibration_raw, "calibration predictions")
        calibration = surrogate_core.calibrate_split_conformal(
            calibration_truth, replay_calibration_raw, model.response_scale,
            alpha=float(self.config["surrogate"]["calibration"]["alpha"]),
            maximum_delta=float(self.config["surrogate"]["calibration"]["delta_max_inclusive"]),
        )
        stored_calibration = _load_json(self.run_root / "metrics" / "calibration.json")
        for name, value in calibration.as_dict().items():
            compare(stored_calibration[name], value, f"calibration.{name}")

        assessment_rows = blocks == 2
        with np.load(self.run_root / "predictions" / "assessment_predictions.npz", allow_pickle=False) as payload:
            assessment_truth, assessment_raw = payload["truth"], payload["raw"]
        compare(assessment_truth, targets[assessment_rows], "assessment truth")
        replay_assessment_raw = model.predict(
            decisions[assessment_rows], influents[assessment_rows]
        )
        compare(assessment_raw, replay_assessment_raw, "assessment predictions")
        replay_assessment = surrogate_core.assess_raw_predictions(
            assessment_truth, replay_assessment_raw, model.response_scale,
            delta=calibration.delta,
            complete_state_rmse_max=float(self.config["surrogate"]["assessment"]["complete_state_standardized_rmse_max_exclusive"]),
            minimum_coverage=float(self.config["surrogate"]["assessment"]["empirical_coverage_min_inclusive"]),
            variance_relative_tolerance=float(self.config["surrogate"]["variance_relative_floor"]),
        )
        if not replay_assessment.passed:
            raise StageExecutionError("terminal assessment replay failed")
        stored_assessment = _load_json(self.run_root / "metrics" / "assessment_summary.json")
        for name, value in replay_assessment.as_dict().items():
            compare(stored_assessment[name], value, f"assessment.{name}")
        derived_frame = _derived_metric_frame(
            _derived_assessment_responses(
                decisions[assessment_rows], assessment_truth
            ),
            _derived_assessment_responses(
                decisions[assessment_rows], replay_assessment_raw
            ),
            stored_assets["derived_assessment_scale"],
            variance_relative_tolerance=float(
                self.config["surrogate"]["variance_relative_floor"]
            ),
        )
        stored_derived = pd.read_csv(
            self.run_root / "metrics" / "assessment_derived_metrics.csv"
        )
        try:
            pd.testing.assert_frame_equal(
                stored_derived,
                derived_frame,
                check_dtype=False,
                check_exact=False,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
        except AssertionError as exc:
            raise StageExecutionError("terminal derived-assessment replay changed") from exc

        problem, assets, _ = self._build_nlp_problems(compile_solver=False)
        case_records = self._load_case_records()
        kkt_count = 0
        algebraic_count = 0
        exact_count = 0
        reconstructed_summaries: list[dict[str, Any]] = []
        for case_index, record in enumerate(case_records):
            case = self._case_definition(record)
            case_id = str(record["case_id"])
            results = self._load_nlp_case(case)
            complete_states = self._load_nlp_complete_states(case)
            for row_index, result in enumerate(results):
                if np.all(np.isfinite(result.primal)):
                    evaluation = nlp.evaluate_problem(problem, result.primal, case)
                    compare(result.objective, evaluation["objective"], f"{case_id}.objective")
                    compare(result.equality, evaluation["equality"], f"{case_id}.equality")
                    compare(result.inequality, evaluation["inequality"], f"{case_id}.inequality")
                    compare(result.decisions, evaluation["decisions"], f"{case_id}.decisions")
                    compare(result.state, evaluation["state"], f"{case_id}.state")
                    compare(result.diagnostics, evaluation["diagnostics"], f"{case_id}.diagnostics")
                    compare(complete_states[row_index], evaluation["complete_state"], f"{case_id}.complete_state")
                    algebraic_count += 1
                replay_inputs = (
                    result.primal,
                    result.equality_multipliers,
                    result.inequality_multipliers,
                    result.bound_multipliers,
                )
                if not all(np.all(np.isfinite(value)) for value in replay_inputs):
                    if result.accepted:
                        raise StageExecutionError("an accepted NLP record has non-finite replay inputs")
                    continue
                replay = nlp.replay_kkt(
                    problem, result.primal, case, result.equality_multipliers,
                    result.inequality_multipliers, result.bound_multipliers,
                )
                compare(asdict(result.kkt), asdict(replay), f"{case_id}.kkt")
                kkt_count += 1

            selected = nlp.select_best_start(results)
            stored_result = _load_json(
                self.run_root / "optimization" / "cache" / case_id / "result.json"
            )
            reconstructed_summaries.append(stored_result)
            selected_index = None if selected is None else int(selected.start_index)
            compare(stored_result["selected_start"], selected_index, f"{case_id}.selection")
            if selected is None:
                expected_exact = {
                    "attempted": False,
                    "accepted": False,
                    "outcome": "no accepted NLP start",
                }
                compare(stored_result["exact"], expected_exact, f"{case_id}.exact")
                continue

            exact_root = self.run_root / "optimization" / "cache" / case_id
            exact_record = self._load_exact_cache(case, selected)
            if exact_record is None:
                raise ContractMismatchError(f"exact replay cache for {case_id} is missing")
            with np.load(exact_root / "exact_combined.npz", allow_pickle=False) as payload:
                exact_arrays = {name: payload[name].copy() for name in payload.files}
            compare(exact_arrays["decisions"], selected.decisions, f"{case_id}.exact decisions")
            compare(exact_arrays["influent"], case.influent, f"{case_id}.exact influent")
            compare(exact_arrays["selected_mechanistic_state"], selected.state, f"{case_id}.selected state")
            selected_evaluation = nlp.evaluate_problem(problem, selected.primal, case)
            compare(
                exact_arrays["selected_complete_state"],
                selected_evaluation["complete_state"],
                f"{case_id}.selected complete state",
            )
            raw = model.predict(selected.decisions, case.influent)
            compare(exact_arrays["raw_surrogate_prediction"], raw, f"{case_id}.exact prediction")
            exact_target = exact_arrays["target"]
            if exact_target.shape != (mechanism.TARGET_SIZE,) or not np.all(np.isfinite(exact_target)):
                outcome = "exact integration failure"
                exact_accepted = False
            else:
                exact_state = np.concatenate((exact_target[20:120], exact_target[160:170]))
                physical = mechanism.diagnostics(
                    exact_state,
                    mechanism.OperatingPoint(*selected.decisions.tolist()),
                    case.influent,
                    residual_tolerance=float(
                        self.config["mechanistic_solver"]["scaled_derivative_acceptance_d_inv"]
                    ),
                    check_stability=True,
                )
                engineering = self._engineering_from_target(
                    selected.decisions,
                    exact_target,
                    record["weights"],
                    float(record["underflow_tss_limit"]),
                    assets.quality_scale,
                    assets.inventory_scale,
                    stored_assets["clarifier_feed_mass_scale"],
                )
                score = float(
                    np.mean(np.square((exact_target - raw) / model.response_scale))
                )
                normalized_fidelity = score / calibration.delta
                fidelity_passed = self._exact_fidelity_passed(normalized_fidelity)
                branch = float(
                    np.max(np.abs((exact_state - selected.state) / assets.state_scale))
                )
                selected_complete = selected_evaluation["complete_state"]
                selected_engineering = self._engineering_from_target(
                    selected.decisions,
                    selected_complete,
                    record["weights"],
                    float(record["underflow_tss_limit"]),
                    assets.quality_scale,
                    assets.inventory_scale,
                    stored_assets["clarifier_feed_mass_scale"],
                )
                residual_stability_passed = bool(
                    exact_record["generator_accepted"] and physical["passed"]
                )
                flow_difference = (
                    selected_complete[120:160] - exact_target[120:160]
                ).reshape(2, mechanism.N_COMPONENTS)
                composite_difference = np.vstack(
                    (
                        selected_engineering["effluent_composites"]
                        - engineering["effluent_composites"],
                        selected_engineering["underflow_composites"]
                        - engineering["underflow_composites"],
                    )
                )
                exact_values = {
                    "engineering": engineering,
                    "selected_engineering": selected_engineering,
                    "physical_diagnostics": physical,
                    "residual_stability_passed": residual_stability_passed,
                    "fidelity_score": score,
                    "normalized_fidelity": normalized_fidelity,
                    "fidelity_constraint_row": normalized_fidelity - 1.0,
                    "fidelity_passed": fidelity_passed,
                    "branch_scaled_infinity": branch,
                    "selected_state_scaled_rms": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    (exact_target - selected_complete)
                                    / model.response_scale
                                )
                            )
                        )
                    ),
                    "selected_state_scaled_infinity": float(
                        np.max(
                            np.abs(
                                (exact_target - selected_complete)
                                / model.response_scale
                            )
                        )
                    ),
                    "selected_state_objective_difference": (
                        selected_engineering["objective"] - engineering["objective"]
                    ),
                    "nlp_minus_bdf_outlet_component_flow": flow_difference,
                    "nlp_minus_bdf_outlet_composite_concentration": composite_difference,
                    "dynamic_balance_scaled_residual_inf": physical[
                        "scaled_residual_inf"
                    ],
                }
                for name, value in exact_values.items():
                    compare(exact_record[name], value, f"{case_id}.exact {name}")
                if not residual_stability_passed:
                    outcome = "residual/stability failure"
                elif not engineering["domain_engineering_passed"]:
                    outcome = "domain/engineering failure"
                elif not fidelity_passed:
                    outcome = "statistical-fidelity-validation failure"
                elif branch > float(self.config["optimization"]["exact_validation"]["branch_scaled_infinity_tolerance"]):
                    outcome = "branch/smoothing-validation failure"
                else:
                    outcome = "validated combined recommendation"
                exact_accepted = outcome == "validated combined recommendation"
            compare(exact_record["outcome"], outcome, f"{case_id}.exact outcome")
            compare(exact_record["accepted"], exact_accepted, f"{case_id}.exact accepted")
            compare(stored_result["exact"], exact_record, f"{case_id}.result exact")
            exact_count += 1

        replay_frame = pd.DataFrame(
            [self._flat_case_summary(summary) for summary in reconstructed_summaries]
        )
        stored_frame = pd.read_parquet(
            self.run_root / "optimization" / "case_summary.parquet"
        )

        replay_frame = self._validate_replayed_frame(
            stored_frame, replay_frame, "optimization case summary"
        )
        robustness = replay_frame.loc[
            replay_frame["case_class"] == "robustness"
        ].reset_index(drop=True)
        realized = _load_json(self.run_root / "optimization" / "summary.json")
        expected_tables = {
            "combined_route_outcomes.csv": self._outcome_table(
                robustness,
                "outcome",
                self.config["reporting"]["combined_outcome_classes"],
            ),
            "robustness_summary.csv": self._robustness_summary(replay_frame),
            "bound_activity.csv": self._bound_activity(replay_frame),
            "underflow_tss_sensitivity.csv": pd.concat(
                [
                    replay_frame.loc[
                        replay_frame["case_id"] == "underflow_tss_12000"
                    ],
                    replay_frame.loc[replay_frame["case_id"] == "nominal"],
                    replay_frame.loc[
                        replay_frame["case_id"] == "underflow_tss_20000"
                    ],
                ],
                ignore_index=True,
            ),
            "objective_weight_sensitivity.csv": replay_frame.loc[
                replay_frame["sensitivity_family"] == "objective_weights"
            ].copy(),
            "workload.csv": self._workload_table(replay_frame, realized),
            "timing_summary.csv": self._timing_summary(),
        }
        for name, expected in expected_tables.items():
            self._validate_replayed_frame(
                pd.read_csv(self.run_root / "tables" / name),
                expected,
                f"table {name}",
            )

        report = _load_json(self.run_root / "report" / "summary.json")
        if report.get("run_id") != self.run_id or report.get("profile") != self.profile_name:
            raise StageExecutionError("terminal report refers to another run")
        if report.get("release_status") != "provisional_pending_terminal_replay":
            raise StageExecutionError("the report was not provisional before terminal sealing")
        compare(
            report.get("contract_sha256"),
            self._contract["contract_sha256"],
            "report contract",
        )
        compare(
            report.get("numerical_environment"),
            self._contract["numerical_environment"],
            "report numerical environment",
        )
        compare(
            report.get("casadi_ipopt_binary_sha256"),
            self._contract["casadi_ipopt_binary_sha256"],
            "report numerical binaries",
        )
        compare(report.get("host"), self._contract["host"], "report host")
        compare(
            report.get("data_generation_worker_count"),
            int(self.profile["parallel_workers"]),
            "report worker count",
        )
        compare(report.get("optimization"), realized, "report optimization summary")
        for name, expected in expected_tables.items():
            inventory = report.get("table_inventory", {}).get(name)
            if not isinstance(inventory, Mapping):
                raise StageExecutionError(f"report inventory is missing table {name}")
            compare(inventory["rows"], len(expected), f"report {name} row count")
            compare(
                inventory["columns"], list(expected.columns),
                f"report {name} columns",
            )
            compare(
                inventory["sha256"],
                sha256_file(self.run_root / "tables" / name),
                f"report {name} digest",
            )
        result = {
            "passed": True,
            "scope": "non-BDF replay only; exact BDF artifacts are immutable and are not rerun",
            "dataset_rows": int(decisions.shape[0]),
            "calibration_scores": int(calibration.scores.size),
            "assessment_rows": int(assessment_truth.shape[0]),
            "nlp_algebraic_records_replayed": algebraic_count,
            "kkt_records_replayed": kkt_count,
            "exact_records_replayed": exact_count,
            "report_tables_reconstructed": len(expected_tables),
            "case_summary_rows_reconstructed": len(replay_frame),
            "contract_sha256": self._contract["contract_sha256"],
            "report_release_status_before_seal": report["release_status"],
            "maximum_replay_difference": maximum_difference,
            "optimization_cases": len(case_records),
            "completed_utc": utc_now(),
        }
        atomic_json(self.run_root / "checks" / "terminal_replay.json", result)
        return result

    def _stage_complete(self) -> dict[str, Any]:
        missing = [stage for stage in STAGES[:-1] if not self._is_stage_complete(stage)]
        if missing:
            raise StageExecutionError(f"cannot seal run; incomplete stages: {missing}")
        return self._terminal_replay()

    def _artifact_inventory(self) -> pd.DataFrame:
        excluded = {"COMPLETED.json", "manifest.json", "manifest.sha256", "artifact_inventory.csv"}
        records = []
        for path in sorted(self.run_root.rglob("*")):
            if path.is_file() and path.relative_to(self.run_root).as_posix() not in excluded:
                records.append({"path": path.relative_to(self.run_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        return pd.DataFrame(records, columns=("path", "bytes", "sha256"))

    def _finalize_seal(self) -> None:
        inventory = self._artifact_inventory()
        atomic_csv(self.run_root / "artifact_inventory.csv", inventory)
        for record in inventory.to_dict(orient="records"):
            if sha256_file(self.run_root / record["path"]) != record["sha256"]:
                raise StageExecutionError(f"artifact changed during final seal: {record['path']}")
        inventory_digest = sha256_file(self.run_root / "artifact_inventory.csv")
        self._manifest["status"] = "complete"
        self._manifest["completed_utc"] = utc_now()
        self._manifest["artifact_count"] = len(inventory)
        self._manifest["inventory_sha256"] = inventory_digest
        self._write_manifest()
        manifest_digest = sha256_file(self.manifest_path)
        _atomic_bytes(self.run_root / "manifest.sha256", f"{manifest_digest}  manifest.json\n".encode())
        atomic_json(
            self.completion_path,
            {
                "run_id": self.run_id,
                "profile": self.profile_name,
                "status": "complete",
                "article_eligible": bool(self.profile.get("article_eligible", False)),
                "completed_utc": self._manifest["completed_utc"],
                "contract_sha256": self._contract["contract_sha256"],
                "report_release_status": "terminally_sealed",
                "report_summary_sha256": sha256_file(
                    self.run_root / "report" / "summary.json"
                ),
                "terminal_replay_sha256": sha256_file(
                    self.run_root / "checks" / "terminal_replay.json"
                ),
                "manifest_sha256": manifest_digest,
                "inventory_sha256": inventory_digest,
                "artifact_count": len(inventory),
            },
        )


__all__ = [
    "ContractMismatchError",
    "ClosedLoopWorkflow",
    "ImmutableRunError",
    "MechanisticRow",
    "STAGES",
    "StageExecutionError",
    "WorkflowError",
    "_derived_assessment_responses",
    "_derived_metric_frame",
    "_maximin_robustness_indices",
    "_nearest_rank",
    "load_surrogate_bundle",
    "save_surrogate_bundle",
    "sha256_file",
]
