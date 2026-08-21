"""Resumable orchestration for the closed-loop article calculation.

Each stage writes immutable scientific artifacts before its checkpoint is
marked complete.  Re-running the same run id resumes only when the resolved
configuration and implementation contract are unchanged; a sealed run is
never reopened.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from . import model as mechanism
from .surrogate import (
    LeastSquaresDiagnostics,
    NetworkLayout,
    NetworkRowScales,
    PhysicalProjector,
    ProjectionError,
    QuadraticFeatureMap,
    QuadraticSurrogate,
    SearchResult,
    SearchSettings,
    affine_projection,
    build_network_operators,
    deterministic_bounded_search,
    feasibility_first_merit,
    fit_network_row_scales,
)


STAGES: tuple[str, ...] = (
    "static",
    "pilot",
    "dataset",
    "assessment",
    "production",
    "optimization",
    "report",
    "complete",
)
WORKFLOW_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WorkflowError(RuntimeError):
    """Base error for a scientifically invalid or incomplete run."""


class ContractMismatchError(WorkflowError):
    """The run directory belongs to a different immutable contract."""


class ImmutableRunError(WorkflowError):
    """A completed run was asked to execute again."""


class StageExecutionError(WorkflowError):
    """A stage stopped without authorizing its downstream stages."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
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
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
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


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, np.ndarray):
        return _json_scalar(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _module_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise WorkflowError(f"{path} must contain a JSON object.")
    return value


def _peak_resident_memory_bytes() -> int:
    """Return the process high-water resident set when the platform exposes it."""

    try:
        import psutil

        memory = psutil.Process().memory_info()
        return int(getattr(memory, "peak_wset", memory.rss))
    except Exception:
        # Memory is a feasibility diagnostic, not a reason to obscure the
        # scientific failure that may have prompted this measurement.
        return 0


def _mechanistic_solution_diagnostics(result: Any) -> dict[str, Any]:
    """Serialize the stream, kinetic, and solids diagnostics used in reporting."""

    reactors = np.asarray(result.reactors, dtype=np.float64)
    effluent = np.asarray(result.effluent, dtype=np.float64)
    underflow = np.asarray(result.underflow, dtype=np.float64)
    operating = result.operating
    final_reactor = reactors[-1]
    clarifier_feed_mass = operating.q_clarifier * final_reactor
    underflow_mass = operating.q_underflow * underflow
    recoveries = np.full(mechanism.N_COMPONENTS, np.nan, dtype=np.float64)
    available = np.abs(clarifier_feed_mass) > 1.0e-12
    recoveries[available] = underflow_mass[available] / clarifier_feed_mass[available]
    feed_tss = float(mechanism.TSS_VECTOR @ final_reactor)
    underflow_tss = float(mechanism.TSS_VECTOR @ underflow)
    return {
        "effluent_concentration": effluent,
        "underflow_concentration": underflow,
        "effluent_composites": mechanism.COMPOSITE_MATRIX @ effluent,
        "underflow_composites": mechanism.COMPOSITE_MATRIX @ underflow,
        "component_recoveries": recoveries,
        "particulate_recoveries": recoveries[mechanism.PARTICULATE],
        "underflow_densification": (
            underflow_tss / feed_tss if feed_tss > 1.0e-12 else np.nan
        ),
        "clarifier_feed_tss_g_m3": feed_tss,
        "underflow_tss_g_m3": underflow_tss,
        "clarifier_layer_inventory_g_d_m3": (
            mechanism.CLARIFIER.layer_volume
            * float(np.sum(np.asarray(result.layers, dtype=np.float64)))
            / mechanism.CLARIFIER.fresh_flow
        ),
        "reactor_process_rates": np.vstack(
            [mechanism.process_rates(state) for state in reactors]
        ),
        "reactor_oxygen_transfer_rates": np.asarray(
            [
                mechanism.oxygen_transfer(state, stage, operating.aeration)
                for stage, state in enumerate(reactors)
            ],
            dtype=np.float64,
        ),
    }


def _qr_leverage(qr_upper: np.ndarray, feature: np.ndarray) -> float:
    """Evaluate phi (Phi'Phi)^-1 phi' through the production QR factor."""

    upper = np.asarray(qr_upper, dtype=np.float64)
    vector = np.asarray(feature, dtype=np.float64)
    if upper.ndim != 2 or upper.shape[0] != upper.shape[1] or vector.shape != (upper.shape[0],):
        raise WorkflowError("QR leverage factor and feature dimensions are inconsistent")
    whitened = solve_triangular(
        upper.T,
        vector,
        lower=True,
        check_finite=True,
    )
    return float(whitened @ whitened)


def _splitmix64_latin_hypercube(sample_count: int, dimension: int, seed: int) -> np.ndarray:
    """Exact SplitMix64/Fisher--Yates fallback for the manuscript design."""

    if sample_count < 1 or dimension < 1:
        raise ValueError("Latin-hypercube dimensions must be positive.")
    mask = (1 << 64) - 1
    state = int(seed) & mask

    def draw() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return (z ^ (z >> 31)) & mask

    design = np.empty((sample_count, dimension), dtype=np.float64)
    modulus = 1 << 64
    denominator = float(1 << 53)
    for coordinate in range(dimension):
        permutation = np.arange(sample_count, dtype=np.int64)
        for index in range(sample_count - 1, 0, -1):
            bound = index + 1
            limit = modulus - (modulus % bound)
            word = draw()
            while word >= limit:
                word = draw()
            swap = word % bound
            permutation[index], permutation[swap] = permutation[swap], permutation[index]
        for row in range(sample_count):
            jitter = (draw() >> 11) / denominator
            design[row, coordinate] = (float(permutation[row]) + jitter) / sample_count
    return design


def generate_latin_hypercube(sample_count: int, dimension: int, seed: int) -> np.ndarray:
    """Isolated adapter to the optional design module, with an exact fallback."""

    try:
        design_module = importlib.import_module("closed_loop.design")
    except ImportError:
        result = _splitmix64_latin_hypercube(sample_count, dimension, seed)
    else:
        if hasattr(design_module, "latin_hypercube"):
            generated = design_module.latin_hypercube(sample_count, dimension, seed)
        elif hasattr(design_module, "unit_latin_hypercube"):
            generated = design_module.unit_latin_hypercube(
                sample_count, dimension, seed=seed
            )
        else:
            generated = _splitmix64_latin_hypercube(sample_count, dimension, seed)
        if isinstance(generated, tuple):
            generated = generated[0]
        result = np.asarray(generated, dtype=np.float64)
    if result.shape != (sample_count, dimension):
        raise WorkflowError(
            f"Latin-hypercube generator returned {result.shape}; expected {(sample_count, dimension)}."
        )
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result >= 1.0):
        raise WorkflowError("Latin-hypercube coordinates must be finite and lie in [0, 1).")
    strata = np.floor(result * sample_count).astype(np.int64)
    expected = np.arange(sample_count, dtype=np.int64)
    for coordinate in range(dimension):
        if not np.array_equal(np.sort(strata[:, coordinate]), expected):
            raise WorkflowError(f"Latin-hypercube coordinate {coordinate} does not use every stratum once.")
    return result


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


def _solve_payload(payload: tuple[int, np.ndarray, np.ndarray, dict[str, Any]]) -> MechanisticRow:
    index, decisions, influent, settings = payload
    started = perf_counter()
    try:
        from threadpoolctl import threadpool_limits

        operating = mechanism.OperatingPoint(*[float(value) for value in decisions])
        relaxation = settings.get("dynamic_relaxation", {})
        # Each process owns one mechanistic solve.  Limiting its BLAS pools
        # avoids multiplying 12 worker processes by a second layer of threads.
        with threadpool_limits(limits=1):
            result = mechanism.solve_steady_state(
                operating,
                influent,
                max_nfev=int(settings["steady_max_residual_evaluations"]),
                tolerance=float(settings["steady_xtol"]),
                acceptance_tolerance=float(settings["scaled_derivative_acceptance_d_inv"]),
                minimum_relaxation_days=float(relaxation.get("minimum_horizon_d", 400.0)),
                solids_turnovers=float(relaxation.get("waste_sludge_turnovers", 40.0)),
                integration_rtol=float(relaxation.get("relative_tolerance", 1.0e-7)),
                integration_atol=float(relaxation.get("scaled_absolute_tolerance", 1.0e-9)),
            )
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "start": result.start,
                "nfev": result.nfev,
                "cost": result.cost,
                "route": result.route,
                "integration_time_days": result.integration_time_days,
                "integration_steps": result.integration_steps,
            }
        )
        diagnostics.update(_mechanistic_solution_diagnostics(result))
        diagnostics["peak_resident_memory_bytes"] = _peak_resident_memory_bytes()
        accepted = bool(result.accepted)
        target = result.target if accepted else np.full(mechanism.TARGET_SIZE, np.nan)
        return MechanisticRow(
            index=index,
            decisions=decisions.copy(),
            influent=influent.copy(),
            target=np.asarray(target, dtype=np.float64),
            accepted=accepted,
            elapsed_seconds=perf_counter() - started,
            diagnostics=_json_scalar(diagnostics),
            error=None if accepted else "mechanistic acceptance contract failed",
        )
    except Exception as exc:
        return MechanisticRow(
            index=index,
            decisions=decisions.copy(),
            influent=influent.copy(),
            target=np.full(mechanism.TARGET_SIZE, np.nan),
            accepted=False,
            elapsed_seconds=perf_counter() - started,
            diagnostics={"peak_resident_memory_bytes": _peak_resident_memory_bytes()},
            error=f"{type(exc).__name__}: {exc}",
        )


def _ordered_checkpoints(sample_count: int, configured: Sequence[int]) -> tuple[int, ...]:
    required = (4, 16, 64, 256, sample_count)
    values = {int(value) for value in (*configured, *required) if 0 < int(value) <= sample_count}
    values.add(sample_count)
    return tuple(sorted(values))


def _target_columns() -> tuple[str, ...]:
    columns: list[str] = []
    columns.extend(f"m:{name}" for name in mechanism.COMPONENTS)
    for stage in range(1, mechanism.N_STAGES + 1):
        columns.extend(f"c{stage}:{name}" for name in mechanism.COMPONENTS)
    columns.extend(f"gE:{name}" for name in mechanism.COMPONENTS)
    columns.extend(f"gU:{name}" for name in mechanism.COMPONENTS)
    columns.extend(f"layer:{layer}" for layer in range(1, mechanism.N_LAYERS + 1))
    return tuple(columns)


def save_surrogate_bundle(
    path: Path,
    model: QuadraticSurrogate,
    row_scales: NetworkRowScales,
    **extras: np.ndarray,
) -> None:
    feature_map = model.feature_map
    arrays = {
        "decision_center": feature_map.decision_center,
        "decision_scale": feature_map.decision_scale,
        "influent_center": feature_map.influent_center,
        "influent_scale": feature_map.influent_scale,
        "term_center": feature_map.term_center,
        "term_scale": feature_map.term_scale,
        "response_center": model.response_center,
        "response_scale": model.response_scale,
        "coefficients": model.coefficients,
        "equality_scale": row_scales.equality,
        "inequality_scale": row_scales.inequality,
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
) -> tuple[QuadraticSurrogate, NetworkRowScales, dict[str, np.ndarray]]:
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
    diagnostics = LeastSquaresDiagnostics(**metadata["diagnostics"])
    model = QuadraticSurrogate(
        feature_map=feature_map,
        response_center=arrays.pop("response_center"),
        response_scale=arrays.pop("response_scale"),
        coefficients=arrays.pop("coefficients"),
        diagnostics=diagnostics,
    )
    row_scales = NetworkRowScales(
        equality=arrays.pop("equality_scale"),
        inequality=arrays.pop("inequality_scale"),
    )
    return model, row_scales, arrays


class ClosedLoopWorkflow:
    """Execute, resume, validate, and seal one configured scientific run."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        profile: str,
        run_id: str,
        repository_root: str | Path | None = None,
        results_root: str | Path | None = None,
        mechanistic_solver: Callable[[int, np.ndarray, np.ndarray], MechanisticRow] | None = None,
    ) -> None:
        self.repository_root = Path(repository_root or Path(config_path).resolve().parent.parent).resolve()
        self.config_path = Path(config_path).resolve()
        self.config = _load_json(self.config_path)
        profiles = self.config.get("execution", {}).get("profiles", {})
        if profile not in profiles:
            raise WorkflowError(f"unknown execution profile {profile!r}")
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise WorkflowError("run id must begin alphanumerically and contain only letters, digits, ._-.")
        self.profile_name = profile
        self.profile = dict(profiles[profile])
        self.run_id = run_id
        configured_root = Path(self.config["execution"]["results_root"])
        resolved_results = Path(results_root) if results_root is not None else self.repository_root / configured_root
        self.results_root = resolved_results.resolve()
        self.run_root = self.results_root / run_id
        self.manifest_path = self.run_root / "manifest.json"
        self.completion_path = self.run_root / "COMPLETED.json"
        self.mechanistic_solver = mechanistic_solver
        self._layout = NetworkLayout()
        self._contract = self._build_contract()
        self._manifest: dict[str, Any] = {}

    @property
    def sample_count(self) -> int:
        return int(self.profile["sample_count"])

    @property
    def development_count(self) -> int:
        return int(self.profile["development_count"])

    @property
    def assessment_count(self) -> int:
        return int(self.profile["assessment_count"])

    def _scientific_workloads(self) -> dict[str, int]:
        """Return the immutable article workload, independent of run profile."""

        try:
            full = self.config["execution"]["profiles"]["full"]
            robustness_cases = int(full["robustness_cases"])
            final_case_replays = 1 + robustness_cases
            qp = (
                int(full["assessment_count"])
                + int(full["sample_count"])
                + final_case_replays * int(full["surrogate_search_budget"])
                + final_case_replays
            )
            mechanistic = (
                int(full["sample_count"])
                + int(full["nominal_mechanistic_search_budget"])
                + robustness_cases * int(full["robustness_mechanistic_search_budget"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StageExecutionError("the full scientific workload is not completely configured") from exc
        return {
            "qp_scientific_evaluations": qp,
            "mechanistic_scientific_evaluations": mechanistic,
            "final_cold_qp_replays": final_case_replays,
            "optimization_cases": final_case_replays,
        }

    def _artifact_hashes(self, paths: Iterable[Path]) -> dict[str, str]:
        records: dict[str, str] = {}
        for path in sorted({item.resolve() for item in paths}):
            if not path.is_file() or not path.is_relative_to(self.run_root):
                raise StageExecutionError(f"required stage artifact is missing or outside the run: {path}")
            relative = path.relative_to(self.run_root).as_posix()
            records[relative] = sha256_file(path)
        if not records:
            raise StageExecutionError("a completed stage must bind at least one scientific artifact")
        return records

    def _required_stage_artifacts(self, stage: str) -> tuple[Path, ...]:
        fixed: dict[str, tuple[str, ...]] = {
            "static": (
                "inputs/resolved_config.json", "inputs/contract.json",
                "checks/mechanistic_matrix_audit.json", "checks/scientific_workload.json",
                "datasets/design.npz", "datasets/design_generator.json",
                "datasets/design.parquet", "splits/ordered_split.json",
            ),
            "pilot": ("timing/pilot_summary.json",),
            "dataset": (
                "datasets/mechanistic_dataset.npz", "datasets/mechanistic_dataset.parquet",
                "datasets/mechanistic_diagnostics.parquet", "checks/dataset_validation.json",
            ),
            "assessment": (
                "models/development_surrogate.npz", "models/development_surrogate.json",
                "predictions/assessment_predictions.npz",
                "metrics/assessment_coordinate_metrics.csv",
                "metrics/assessment_block_metrics.csv", "metrics/assessment_derived_metrics.csv",
                "metrics/assessment_qp_diagnostics.parquet",
                "metrics/assessment_summary.json", "timing/computational_feasibility.json",
                "timing/qp_preflight_times.npz", "timing/qp_preflight_diagnostics.parquet",
            ),
            "production": (
                "models/production_surrogate.npz", "models/production_surrogate.json",
                "metrics/production_qp_diagnostics.parquet", "metrics/production_summary.json",
            ),
            "optimization": (
                "optimization/case_influents.npz", "optimization/robustness_generator.json",
                "optimization/case_summary.parquet", "optimization/summary.json",
                "tables/optimization_summary.csv",
            ),
            "report": (
                "report/summary.json", "tables/dataset_summary.csv",
                "tables/assessment_summary.csv", "figures/assessment_parity.png",
                "figures/assessment_parity.pdf", "figures/optimization_objectives.png",
                "figures/optimization_objectives.pdf",
            ),
            "complete": ("checks/terminal_replay.json",),
        }
        paths = [self.run_root / relative for relative in fixed.get(stage, ())]
        if stage in {"pilot", "dataset"}:
            paths.extend((self.run_root / "datasets" / "chunks").glob("*"))
            paths.extend((self.run_root / "checks").glob("dataset_rows_*.json"))
        if stage == "optimization":
            paths.extend((self.run_root / "optimization").rglob("*"))
        return tuple(path for path in paths if path.is_file() or path in {self.run_root / rel for rel in fixed.get(stage, ())})

    def _build_contract(self) -> dict[str, Any]:
        module_paths = {
            name: self.repository_root / "closed_loop" / f"{name}.py"
            for name in ("model", "surrogate", "design", "workflow")
        }
        value = {
            "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
            "profile": self.profile_name,
            "profile_settings": self.profile,
            "config_sha256": sha256_file(self.config_path),
            "module_sha256": {name: _module_hash(path) for name, path in module_paths.items()},
            "python": sys.version,
            "numpy": np.__version__,
            "mechanistic_solver_override": (
                None
                if self.mechanistic_solver is None
                else f"{self.mechanistic_solver.__module__}.{getattr(self.mechanistic_solver, '__qualname__', type(self.mechanistic_solver).__qualname__)}"
            ),
            "ordered_split": {
                "development": [0, self.development_count],
                "assessment": [self.development_count, self.sample_count],
            },
        }
        value["contract_sha256"] = sha256_bytes(canonical_json(value))
        return value

    def _initialize(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        for directory in (
            "inputs", "checks", "datasets/chunks", "splits", "models", "predictions",
            "metrics", "optimization", "tables", "figures", "report", "timing",
        ):
            (self.run_root / directory).mkdir(parents=True, exist_ok=True)
        if self.completion_path.exists():
            raise ImmutableRunError(f"run {self.run_id!r} is already sealed and immutable")
        if self.manifest_path.exists():
            self._manifest = _load_json(self.manifest_path)
            if self._manifest.get("contract", {}).get("contract_sha256") != self._contract["contract_sha256"]:
                raise ContractMismatchError(
                    "the existing run id was created by a different configuration or implementation"
                )
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

    def _is_stage_complete(self, stage: str) -> bool:
        record = self._manifest.get("stages", {}).get(stage, {})
        marker = self._stage_marker(stage)
        if record.get("status") != "complete":
            return False
        if not marker.is_file() or sha256_file(marker) != record.get("marker_sha256"):
            raise ContractMismatchError(f"completed stage {stage!r} lost its bound marker")
        try:
            payload = _load_json(marker)
        except (OSError, ValueError, WorkflowError) as exc:
            raise ContractMismatchError(f"completed stage {stage!r} has an unreadable marker") from exc
        if (
            payload.get("stage") != stage
            or payload.get("contract_sha256") != self._contract["contract_sha256"]
        ):
            raise ContractMismatchError(f"completed stage {stage!r} marker changed identity")
        artifact_hashes = payload.get("required_artifact_sha256")
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise ContractMismatchError(f"completed stage {stage!r} has no bound artifact set")
        for relative, expected in artifact_hashes.items():
            path = (self.run_root / str(relative)).resolve()
            if (
                not path.is_relative_to(self.run_root)
                or not path.is_file()
                or sha256_file(path) != expected
            ):
                raise ContractMismatchError(
                    f"completed stage {stage!r} artifact changed: {relative}"
                )
        return True

    def _mark_stage(self, stage: str, status: str, **details: Any) -> None:
        record = {"status": status, "updated_utc": utc_now(), **_json_scalar(details)}
        if status == "complete":
            marker = self._stage_marker(stage)
            artifact_hashes = self._artifact_hashes(self._required_stage_artifacts(stage))
            marker_payload = {
                "stage": stage,
                "status": "complete",
                "contract_sha256": self._contract["contract_sha256"],
                "completed_utc": utc_now(),
                "required_artifact_sha256": artifact_hashes,
                **_json_scalar(details),
            }
            atomic_json(marker, marker_payload)
            record["marker_sha256"] = sha256_file(marker)
            record["required_artifact_count"] = len(artifact_hashes)
            record["required_artifact_set_sha256"] = sha256_bytes(
                canonical_json(artifact_hashes)
            )
        self._manifest.setdefault("stages", {})[stage] = record
        self._manifest["status"] = status if status == "failed" else f"through_{stage}"
        self._write_manifest()

    def run(self, *, through: str = "complete") -> dict[str, Any]:
        if through not in STAGES:
            raise WorkflowError(f"through must be one of {', '.join(STAGES)}")
        self._initialize()
        last_index = STAGES.index(through)
        for stage in STAGES[: last_index + 1]:
            if self._is_stage_complete(stage):
                if stage == "complete" and not self.completion_path.exists():
                    try:
                        self._finalize_seal()
                    except WorkflowError:
                        raise
                    except Exception as exc:
                        raise StageExecutionError(
                            f"final run seal failed with {type(exc).__name__}: {exc}"
                        ) from exc
                continue
            self._mark_stage(stage, "running")
            try:
                details = getattr(self, f"_stage_{stage}")()
            except Exception as exc:
                failure = (
                    exc
                    if isinstance(exc, WorkflowError)
                    else StageExecutionError(
                        f"stage {stage!r} failed with {type(exc).__name__}: {exc}"
                    )
                )
                self._mark_stage(stage, "failed", error=f"{type(failure).__name__}: {failure}")
                if failure is exc:
                    raise
                raise failure from exc
            self._mark_stage(stage, "complete", **(details or {}))
            if stage == "complete":
                try:
                    self._finalize_seal()
                except WorkflowError:
                    raise
                except Exception as exc:
                    raise StageExecutionError(
                        f"final run seal failed with {type(exc).__name__}: {exc}"
                    ) from exc
        return self._manifest

    def _audit_static_configuration(self) -> dict[str, Any]:
        """Reject any mixed configuration/implementation/article contract."""

        def require_equal(actual: Any, expected: Any, label: str) -> None:
            actual_array = np.asarray(actual)
            expected_array = np.asarray(expected)
            same = actual_array.shape == expected_array.shape
            if same and np.issubdtype(actual_array.dtype, np.number) and np.issubdtype(
                expected_array.dtype, np.number
            ):
                same = bool(
                    np.allclose(
                        actual_array.astype(np.float64),
                        expected_array.astype(np.float64),
                        rtol=8.0 * np.finfo(np.float64).eps,
                        atol=8.0 * np.finfo(np.float64).eps,
                    )
                )
            elif same:
                same = bool(np.array_equal(actual_array, expected_array))
            if not same:
                raise StageExecutionError(
                    f"configured {label} differs from the implemented article contract: "
                    f"{actual!r} != {expected!r}"
                )

        process = self.config["process"]
        decision_bounds = {
            "H": [6.0, 36.0], "a": [0.0, 1.0], "r_I": [0.0, 4.0],
            "r_R": [0.25, 1.25], "w": [0.001, 0.05],
        }
        require_equal(list(process["decision_bounds"]), list(decision_bounds), "decision order")
        require_equal(list(process["decision_bounds"].values()), list(decision_bounds.values()), "decision bounds")
        require_equal(
            list(process["influent_bounds"]), list(mechanism.COMPONENTS), "influent-bound order"
        )
        require_equal(
            [pair[0] for pair in process["influent_bounds"].values()],
            mechanism.INFLUENT_LOWER,
            "influent lower bounds",
        )
        require_equal(
            [pair[1] for pair in process["influent_bounds"].values()],
            mechanism.INFLUENT_UPPER,
            "influent upper bounds",
        )
        require_equal(process["nominal_influent"], mechanism.NOMINAL_INFLUENT, "nominal influent")
        process_scalars = {
            "fresh_flow_m3_per_d": mechanism.CLARIFIER.fresh_flow,
            "soluble_count": int(mechanism.SOLUBLE.size),
            "reactor_count": mechanism.N_STAGES,
            "unaerated_reactors": [1, 2],
            "aerated_reactors": [3, 4, 5],
            "oxygen_saturation_g_m3": 8.5,
            "kla_per_aeration_d_inv": 47.0,
        }
        for key, expected in process_scalars.items():
            require_equal(process[key], expected, f"process.{key}")

        clarifier = self.config["clarifier"]
        clarifier_values = {
            "layer_count": mechanism.N_LAYERS,
            "feed_layer_one_based": mechanism.CLARIFIER.feed_layer + 1,
            "surface_area_m2": mechanism.CLARIFIER.area,
            "total_depth_m": mechanism.N_LAYERS * mechanism.CLARIFIER.layer_volume / mechanism.CLARIFIER.area,
            "layer_volume_m3": mechanism.CLARIFIER.layer_volume,
            "maximum_settling_speed_m_d": mechanism.CLARIFIER.maximum_settling_velocity,
            "settling_velocity_coefficient_m_d": mechanism.CLARIFIER.theoretical_settling_velocity,
            "hindered_coefficient_m3_g": mechanism.CLARIFIER.hindered_coefficient,
            "low_concentration_coefficient_m3_g": mechanism.CLARIFIER.low_concentration_coefficient,
            "non_settleable_fraction": mechanism.CLARIFIER.nonsettleable_fraction,
            "flux_limiting_threshold_g_m3": mechanism.CLARIFIER.flux_threshold,
        }
        for key, expected in clarifier_values.items():
            require_equal(clarifier[key], expected, f"clarifier.{key}")

        solver = self.config["mechanistic_solver"]
        solver_values = {
            "steady_xtol": 1.0e-9, "steady_ftol": 1.0e-9, "steady_gtol": 1.0e-9,
            "steady_max_residual_evaluations": 5000,
            "scaled_derivative_acceptance_d_inv": 1.0e-8,
            "minimum_state_acceptance": -1.0e-10,
            "external_balance_acceptance": 1.0e-8,
            "stability_eigenvalue_max_d_inv": 1.0e-8,
        }
        for key, expected in solver_values.items():
            require_equal(solver[key], expected, f"mechanistic_solver.{key}")
        relaxation = solver["dynamic_relaxation"]
        relaxation_values = {
            "method": "BDF", "minimum_horizon_d": 400.0,
            "waste_sludge_turnovers": 40.0, "relative_tolerance": 1.0e-7,
            "scaled_absolute_tolerance": 1.0e-9,
            "maximum_step_fraction_of_horizon": 0.01,
        }
        for key, expected in relaxation_values.items():
            require_equal(relaxation[key], expected, f"mechanistic_solver.dynamic_relaxation.{key}")
        initialization = solver["initialization"]
        require_equal(
            initialization["start_1_reactor_particulate_factors"], [1.0] * 5,
            "mechanistic_solver.initialization.start_1_reactor_particulate_factors",
        )
        require_equal(
            initialization["start_2_reactor_particulate_factors"], [1.5, 2.0, 2.5, 3.0, 3.5],
            "mechanistic_solver.initialization.start_2_reactor_particulate_factors",
        )
        require_equal(
            initialization["start_2_layer_factors"],
            [0.002, 0.005, 0.01, 0.03, 0.10, 0.50, 1.25, 2.25, 3.25, 4.00],
            "mechanistic_solver.initialization.start_2_layer_factors",
        )

        surrogate = self.config["surrogate"]
        require_equal(
            [surrogate["input_dimension"], surrogate["feature_dimension"], surrogate["target_dimension"]],
            [25, 351, mechanism.TARGET_SIZE],
            "surrogate dimensions",
        )
        deployment = surrogate["deployment_qp"]
        deployment_values = {
            "equality_rows": (
                self._layout.equality_count_without_invariants
                + self._layout.stage_count * mechanism.INVARIANT_MATRIX.shape[0]
            ),
            "inequality_rows_excluding_nonnegativity": self._layout.inequality_count,
            "absolute_tolerance": 1.0e-8, "relative_tolerance": 1.0e-8,
            "nonnegativity_tolerance": 1.0e-10, "maximum_iterations": 100000,
            "polish": True, "cold_retry": True,
        }
        for key, expected in deployment_values.items():
            require_equal(deployment[key], expected, f"surrogate.deployment_qp.{key}")

        feasibility = self.config["execution"]["computational_feasibility"]
        feasibility_values = {
            "mechanistic_preflight_rows": 256, "qp_preflight_rows": 1000,
            "production_fit_projection_factor": 2.25,
            "maximum_projected_core_days": 30.0, "memory_allocation_gib": 64.0,
            "maximum_projected_resident_memory_gib": 51.2,
        }
        for key, expected in feasibility_values.items():
            require_equal(feasibility[key], expected, f"execution.computational_feasibility.{key}")
        workloads = self._scientific_workloads()
        require_equal(
            [workloads["qp_scientific_evaluations"], workloads["mechanistic_scientific_evaluations"], workloads["final_cold_qp_replays"]],
            [2_549_101, 280_000, 101],
            "full scientific workloads",
        )
        return {
            "passed": True,
            "checked_process_fields": len(process_scalars) + 4,
            "checked_clarifier_fields": len(clarifier_values),
            "checked_solver_fields": len(solver_values) + len(relaxation_values) + 3,
            "checked_deployment_fields": len(deployment_values),
            "scientific_workloads": workloads,
        }

    def _stage_static(self) -> dict[str, Any]:
        if self.development_count + self.assessment_count != self.sample_count:
            raise StageExecutionError("development and assessment counts must exactly partition the design")
        if self.config["surrogate"]["feature_dimension"] != 351 or self.config["surrogate"]["target_dimension"] != 170:
            raise StageExecutionError("configured surrogate dimensions differ from the implemented contract")
        if tuple(self.config["process"]["component_order"]) != mechanism.COMPONENTS:
            raise StageExecutionError("configured and mechanistic component orders differ")
        audit = mechanism.audit_mechanistic_matrices()
        if not bool(audit["passed"]):
            raise StageExecutionError(f"mechanistic matrix audit failed: {audit}")
        configuration_audit = self._audit_static_configuration()

        atomic_json(self.run_root / "inputs" / "resolved_config.json", self.config)
        atomic_json(self.run_root / "inputs" / "contract.json", self._contract)
        atomic_json(self.run_root / "checks" / "mechanistic_matrix_audit.json", _json_scalar(audit))
        atomic_json(
            self.run_root / "checks" / "scientific_workload.json",
            configuration_audit,
        )

        random_design = self.config["random_design"]
        process = self.config["process"]
        decision_names = tuple(process["decision_bounds"].keys())
        influent_names = tuple(process["influent_bounds"].keys())
        if tuple(random_design["dimension_order"]) != decision_names + influent_names:
            raise StageExecutionError("random-design order differs from the declared physical bounds order")
        decision_bounds = np.asarray([process["decision_bounds"][name] for name in decision_names], dtype=float)
        influent_bounds = np.asarray([process["influent_bounds"][name] for name in influent_names], dtype=float)
        seed = int(random_design["design_seed"])
        try:
            design_module = importlib.import_module("closed_loop.design")
            generated_design = design_module.generate_training_design(self.sample_count, seed=seed)
            unit_design = np.asarray(generated_design.unit, dtype=np.float64)
            physical_design = np.asarray(generated_design.physical, dtype=np.float64)
            if tuple(generated_design.columns) != decision_names + influent_names:
                raise StageExecutionError("design module and configuration use different column orders")
            decisions = physical_design[:, :5]
            influents = physical_design[:, 5:]
            generator_metadata = {
                "seed": int(generated_design.seed),
                "final_state": int(generated_design.final_state),
                "draw_count": int(generated_design.draw_count),
                "generator": "SplitMix64",
            }
        except (ImportError, AttributeError):
            unit_design = generate_latin_hypercube(
                self.sample_count, len(random_design["dimension_order"]), seed
            )
            decisions = decision_bounds[:, 0] + unit_design[:, :5] * np.diff(
                decision_bounds, axis=1
            ).ravel()
            influents = influent_bounds[:, 0] + unit_design[:, 5:] * np.diff(
                influent_bounds, axis=1
            ).ravel()
            generator_metadata = {"seed": seed, "generator": "SplitMix64 fallback"}
        expected_decisions = decision_bounds[:, 0] + unit_design[:, :5] * np.diff(
            decision_bounds, axis=1
        ).ravel()
        expected_influents = influent_bounds[:, 0] + unit_design[:, 5:] * np.diff(
            influent_bounds, axis=1
        ).ravel()
        if not np.array_equal(decisions, expected_decisions) or not np.array_equal(
            influents, expected_influents
        ):
            raise StageExecutionError("design module physical mapping differs from configured bounds")
        atomic_npz(
            self.run_root / "datasets" / "design.npz",
            unit_design=unit_design,
            decisions=decisions,
            influents=influents,
        )
        atomic_json(self.run_root / "datasets" / "design_generator.json", generator_metadata)
        design_frame = pd.DataFrame(
            np.column_stack((np.arange(self.sample_count), decisions, influents)),
            columns=("row", *decision_names, *influent_names),
        )
        atomic_parquet(self.run_root / "datasets" / "design.parquet", design_frame)
        split = {
            "membership_rule": "immutable generation order",
            "development_rows_zero_based_half_open": [0, self.development_count],
            "assessment_rows_zero_based_half_open": [self.development_count, self.sample_count],
        }
        atomic_json(self.run_root / "splits" / "ordered_split.json", split)
        return {
            "sample_count": self.sample_count,
            "design_sha256": sha256_file(self.run_root / "datasets" / "design.npz"),
            "generator": generator_metadata,
            "matrix_audit": audit,
            "configuration_audit": configuration_audit,
        }

    def _checkpoints(self) -> tuple[int, ...]:
        return _ordered_checkpoints(
            self.sample_count, tuple(int(value) for value in self.profile["generation_checkpoints"])
        )

    def _chunk_path(self, start: int, stop: int) -> Path:
        return self.run_root / "datasets" / "chunks" / f"rows_{start:06d}_{stop:06d}.npz"

    def _chunk_check_path(self, start: int, stop: int) -> Path:
        return self.run_root / "checks" / f"dataset_rows_{start:06d}_{stop:06d}.json"

    def _solve_rows(
        self,
        start: int,
        stop: int,
        decisions: np.ndarray,
        influents: np.ndarray,
    ) -> list[MechanisticRow]:
        if self.mechanistic_solver is not None:
            return [
                self.mechanistic_solver(index, decisions[index].copy(), influents[index].copy())
                for index in range(start, stop)
            ]
        settings = dict(self.config["mechanistic_solver"])
        payloads = [
            (index, decisions[index].copy(), influents[index].copy(), settings)
            for index in range(start, stop)
        ]
        workers = min(int(self.profile.get("parallel_workers", 1)), stop - start)
        if workers <= 1:
            return [_solve_payload(payload) for payload in payloads]
        thread_variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        previous_environment = {name: os.environ.get(name) for name in thread_variables}
        try:
            for name in thread_variables:
                os.environ[name] = "1"
            with ProcessPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(_solve_payload, payloads, chunksize=1))
        finally:
            for name, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous

    def _validate_chunk(self, path: Path, start: int, stop: int) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            indices = payload["indices"]
            decisions = payload["decisions"]
            influents = payload["influents"]
            targets = payload["targets"]
            accepted = payload["accepted"]
            elapsed = payload["elapsed_seconds"]
        count = stop - start
        if not np.array_equal(indices, np.arange(start, stop, dtype=np.int64)):
            raise StageExecutionError(f"chunk {path.name} has broken row order")
        if decisions.shape != (count, 5) or influents.shape != (count, 20):
            raise StageExecutionError(f"chunk {path.name} has broken input dimensions")
        if targets.shape != (count, 170) or accepted.shape != (count,) or elapsed.shape != (count,):
            raise StageExecutionError(f"chunk {path.name} has broken output dimensions")
        finite_targets = np.all(np.isfinite(targets), axis=1)
        if not np.array_equal(finite_targets, accepted.astype(bool)):
            raise StageExecutionError(f"chunk {path.name} target finiteness disagrees with acceptance")
        return {
            "start": start,
            "stop": stop,
            "row_count": count,
            "accepted_count": int(np.count_nonzero(accepted)),
            "elapsed_seconds": float(np.sum(elapsed)),
            "p95_seconds": float(np.quantile(elapsed, 0.95, method="higher")),
            "sha256": sha256_file(path),
        }

    def _generate_chunk(
        self,
        start: int,
        stop: int,
        decisions: np.ndarray,
        influents: np.ndarray,
    ) -> dict[str, Any]:
        path = self._chunk_path(start, stop)
        check_path = self._chunk_check_path(start, stop)
        if path.exists() and check_path.exists():
            check = _load_json(check_path)
            if check.get("sha256") != sha256_file(path):
                raise ContractMismatchError(f"existing chunk {path.name} no longer matches its checkpoint")
            summary = self._validate_chunk(path, start, stop)
        elif check_path.exists():
            raise ContractMismatchError(f"chunk checkpoint {check_path.name} has lost its data artifact")
        else:
            # A data or diagnostics file without its final check marker was
            # never committed.  Recomputing and atomically replacing it is the
            # safe resume route after interruption between writes.
            rows = self._solve_rows(start, stop, decisions, influents)
            if [row.index for row in rows] != list(range(start, stop)):
                raise StageExecutionError("mechanistic worker results lost immutable row order")
            targets = np.vstack([row.target for row in rows])
            accepted = np.asarray([row.accepted for row in rows], dtype=bool)
            elapsed = np.asarray([row.elapsed_seconds for row in rows], dtype=np.float64)
            atomic_npz(
                path,
                indices=np.arange(start, stop, dtype=np.int64),
                decisions=decisions[start:stop],
                influents=influents[start:stop],
                targets=targets,
                accepted=accepted,
                elapsed_seconds=elapsed,
            )
            diagnostics_records = []
            for row in rows:
                record = {
                    "row": row.index,
                    "accepted": row.accepted,
                    "elapsed_seconds": row.elapsed_seconds,
                    "error": row.error,
                    **row.diagnostics,
                }
                diagnostics_records.append(_json_scalar(record))
            atomic_json(path.with_suffix(".diagnostics.json"), diagnostics_records)
            summary = self._validate_chunk(path, start, stop)
            summary["diagnostics_sha256"] = sha256_file(path.with_suffix(".diagnostics.json"))
            atomic_json(check_path, summary)
        if int(summary["accepted_count"]) != int(summary["row_count"]):
            raise StageExecutionError(
                f"mechanistic rows {start}:{stop} contain rejected states; inspect "
                f"{path.with_suffix('.diagnostics.json').relative_to(self.run_root)}. "
                "No replacement row is permitted; a method correction requires a new run id."
            )
        return summary

    def _generate_through(self, stop: int) -> list[dict[str, Any]]:
        design_path = self.run_root / "datasets" / "design.npz"
        with np.load(design_path, allow_pickle=False) as payload:
            decisions = payload["decisions"]
            influents = payload["influents"]
        boundaries = [0, *(value for value in self._checkpoints() if value <= stop)]
        if boundaries[-1] != stop:
            boundaries.append(stop)
        summaries: list[dict[str, Any]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            summaries.append(self._generate_chunk(start, end, decisions, influents))
        return summaries

    def _stage_pilot(self) -> dict[str, Any]:
        feasibility = self.config["execution"]["computational_feasibility"]
        pilot_count = min(int(feasibility["mechanistic_preflight_rows"]), self.sample_count)
        summaries = self._generate_through(pilot_count)
        elapsed = np.concatenate(
            [
                np.load(self._chunk_path(summary["start"], summary["stop"]), allow_pickle=False)[
                    "elapsed_seconds"
                ]
                for summary in summaries
            ]
        )
        p95 = float(np.quantile(elapsed, 0.95, method="higher"))
        workloads = self._scientific_workloads()
        projected_seconds = float(workloads["mechanistic_scientific_evaluations"] * p95)
        projected_core_days = projected_seconds / 86_400.0
        memory_observations = [_peak_resident_memory_bytes()]
        for summary in summaries:
            diagnostics_path = self._chunk_path(
                int(summary["start"]), int(summary["stop"])
            ).with_suffix(".diagnostics.json")
            for record in json.loads(diagnostics_path.read_text(encoding="utf-8")):
                memory_observations.append(int(record.get("peak_resident_memory_bytes", 0) or 0))
        mechanistic_peak_memory = max(memory_observations)
        article_gate_enforced = bool(self.profile.get("article_eligible", False))
        mechanism_time_passed = projected_core_days <= float(
            feasibility["maximum_projected_core_days"]
        )
        pilot_summary = {
            "pilot_count": pilot_count,
            "all_accepted": True,
            "mechanistic_p95_seconds": p95,
            "scientific_mechanistic_evaluations": workloads[
                "mechanistic_scientific_evaluations"
            ],
            "scientific_mechanistic_projected_seconds": projected_seconds,
            "scientific_mechanistic_projected_core_days": projected_core_days,
            "mechanistic_preflight_peak_resident_memory_bytes": mechanistic_peak_memory,
            "article_gate_enforced": article_gate_enforced,
            "mechanistic_time_component_passed": mechanism_time_passed,
            "chunks": summaries,
            "dynamic_fallback_available": True,
            "unresolved_failure_policy": "stop without replacement",
        }
        atomic_json(self.run_root / "timing" / "pilot_summary.json", pilot_summary)
        if article_gate_enforced and not mechanism_time_passed:
            raise StageExecutionError(
                "the mechanistic preflight alone exceeds the configured 30-core-day article envelope"
            )
        return pilot_summary

    def _consolidate_dataset(self) -> dict[str, Any]:
        targets: list[np.ndarray] = []
        decisions: list[np.ndarray] = []
        influents: list[np.ndarray] = []
        elapsed: list[np.ndarray] = []
        diagnostics: list[dict[str, Any]] = []
        boundaries = [0, *self._checkpoints()]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            path = self._chunk_path(start, stop)
            self._validate_chunk(path, start, stop)
            with np.load(path, allow_pickle=False) as payload:
                decisions.append(payload["decisions"])
                influents.append(payload["influents"])
                targets.append(payload["targets"])
                elapsed.append(payload["elapsed_seconds"])
            diagnostic_payload = json.loads(path.with_suffix(".diagnostics.json").read_text(encoding="utf-8"))
            diagnostics.extend(diagnostic_payload)
        decision_matrix = np.vstack(decisions)
        influent_matrix = np.vstack(influents)
        target_matrix = np.vstack(targets)
        elapsed_vector = np.concatenate(elapsed)
        if decision_matrix.shape != (self.sample_count, 5) or target_matrix.shape != (self.sample_count, 170):
            raise StageExecutionError("consolidated mechanistic dataset has an incorrect shape")
        path = self.run_root / "datasets" / "mechanistic_dataset.npz"
        atomic_npz(
            path,
            row=np.arange(self.sample_count, dtype=np.int64),
            decisions=decision_matrix,
            influents=influent_matrix,
            targets=target_matrix,
            elapsed_seconds=elapsed_vector,
        )
        columns = (
            "row",
            *self.config["process"]["decision_bounds"].keys(),
            *self.config["process"]["influent_bounds"].keys(),
            *_target_columns(),
        )
        frame = pd.DataFrame(
            np.column_stack(
                (np.arange(self.sample_count), decision_matrix, influent_matrix, target_matrix)
            ),
            columns=columns,
        )
        atomic_parquet(self.run_root / "datasets" / "mechanistic_dataset.parquet", frame)
        atomic_parquet(self.run_root / "datasets" / "mechanistic_diagnostics.parquet", pd.DataFrame(diagnostics))
        return {
            "rows": self.sample_count,
            "target_columns": target_matrix.shape[1],
            "all_finite": bool(np.all(np.isfinite(target_matrix))),
            "dataset_sha256": sha256_file(path),
            "elapsed_seconds": float(np.sum(elapsed_vector)),
        }

    def _stage_dataset(self) -> dict[str, Any]:
        summaries = self._generate_through(self.sample_count)
        result = self._consolidate_dataset()
        result["chunks"] = summaries
        atomic_json(self.run_root / "checks" / "dataset_validation.json", result)
        return result

    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self.run_root / "datasets" / "mechanistic_dataset.npz"
        with np.load(path, allow_pickle=False) as payload:
            decisions = payload["decisions"].copy()
            influents = payload["influents"].copy()
            targets = payload["targets"].copy()
        if (
            decisions.shape != (self.sample_count, 5)
            or influents.shape != (self.sample_count, 20)
            or targets.shape != (self.sample_count, 170)
            or not np.all(np.isfinite(targets))
        ):
            raise StageExecutionError("sealed mechanistic dataset failed reload validation")
        return decisions, influents, targets

    def _fit_response(
        self,
        decisions: np.ndarray,
        influents: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[QuadraticSurrogate, NetworkRowScales]:
        settings = self.config["surrogate"]
        response = QuadraticSurrogate.fit(
            decisions,
            influents,
            targets,
            variance_relative_tolerance=float(settings["variance_relative_floor"]),
            maximum_condition_number=float(settings["maximum_design_condition_number"]),
        )
        row_scales = fit_network_row_scales(
            targets,
            influents,
            internal_recycle=decisions[:, 2],
            return_recycle=decisions[:, 3],
            waste_fraction=decisions[:, 4],
            invariant_operator=mechanism.INVARIANT_MATRIX,
            tss_weights=mechanism.TSS_VECTOR,
            layout=self._layout,
        )
        return response, row_scales

    def _projector(
        self, model: QuadraticSurrogate, row_scales: NetworkRowScales
    ) -> PhysicalProjector:
        settings = self.config["surrogate"]["deployment_qp"]
        return PhysicalProjector(
            state_scale=model.response_scale,
            equality_scale=row_scales.equality,
            inequality_scale=row_scales.inequality,
            absolute_tolerance=float(settings["absolute_tolerance"]),
            relative_tolerance=float(settings["relative_tolerance"]),
            maximum_iterations=int(settings["maximum_iterations"]),
            polish=bool(settings["polish"]),
            equality_acceptance_tolerance=float(settings["absolute_tolerance"]),
            inequality_acceptance_tolerance=float(settings["relative_tolerance"]),
            nonnegativity_acceptance_tolerance=float(settings["nonnegativity_tolerance"]),
        )

    def _cold_qp_preflight(
        self,
        model: QuadraticSurrogate,
        row_scales: NetworkRowScales,
        decisions: np.ndarray,
        influents: np.ndarray,
    ) -> dict[str, Any]:
        """Cold-deploy evenly spaced development rows for the article gate."""

        configured = int(
            self.config["execution"]["computational_feasibility"]["qp_preflight_rows"]
        )
        count = min(configured, decisions.shape[0])
        indices = np.linspace(0, decisions.shape[0] - 1, count, dtype=np.int64)
        if np.unique(indices).size != count:
            raise StageExecutionError("QP preflight row selection is not one-to-one")
        projector = self._projector(model, row_scales)
        elapsed = np.empty(count, dtype=np.float64)
        diagnostics: list[dict[str, Any]] = []
        peak_memory = _peak_resident_memory_bytes()
        for local, row in enumerate(indices):
            started = perf_counter()
            raw = model.predict(decisions[row], influents[row])
            operating = decisions[row]
            operators = build_network_operators(
                influents[row],
                internal_recycle=float(operating[2]),
                return_recycle=float(operating[3]),
                waste_fraction=float(operating[4]),
                invariant_operator=mechanism.INVARIANT_MATRIX,
                tss_weights=mechanism.TSS_VECTOR,
                layout=self._layout,
            )
            try:
                result = projector.project(
                    raw,
                    operators.equality_matrix,
                    operators.equality_rhs,
                    operators.inequality_matrix,
                    warm_start=None,
                )
            except ProjectionError as exc:
                raise StageExecutionError(
                    f"cold QP preflight failed at development row {int(row)}: {exc}"
                ) from exc
            elapsed[local] = perf_counter() - started
            diagnostics.append({"row": int(row), **result.diagnostics.as_dict()})
            peak_memory = max(peak_memory, _peak_resident_memory_bytes())
        return {
            "row_indices": indices,
            "elapsed_seconds": elapsed,
            "diagnostics": diagnostics,
            "p95_seconds": float(np.quantile(elapsed, 0.95, method="higher")),
            "peak_resident_memory_bytes": int(peak_memory),
            "all_accepted": True,
        }

    def _evaluate_computational_feasibility(
        self,
        *,
        fit_seconds: float,
        fit_peak_resident_memory_bytes: int,
        qp_preflight: Mapping[str, Any],
    ) -> dict[str, Any]:
        settings = self.config["execution"]["computational_feasibility"]
        workloads = self._scientific_workloads()
        pilot = _load_json(self.run_root / "timing" / "pilot_summary.json")
        fit_factor = float(settings["production_fit_projection_factor"])
        mechanistic_p95 = float(pilot["mechanistic_p95_seconds"])
        qp_p95 = float(qp_preflight["p95_seconds"])
        projected_seconds = (
            fit_factor * float(fit_seconds)
            + workloads["qp_scientific_evaluations"] * qp_p95
            + workloads["mechanistic_scientific_evaluations"] * mechanistic_p95
        )
        projected_core_days = projected_seconds / 86_400.0
        fit_memory = int(fit_peak_resident_memory_bytes)
        qp_memory = int(qp_preflight["peak_resident_memory_bytes"])
        mechanistic_memory = int(pilot["mechanistic_preflight_peak_resident_memory_bytes"])
        projected_memory_bytes = int(
            max(1.25 * fit_memory, qp_memory, mechanistic_memory)
        )
        projected_memory_gib = projected_memory_bytes / float(1024**3)
        time_passed = projected_core_days <= float(settings["maximum_projected_core_days"])
        memory_passed = projected_memory_gib <= float(
            settings["maximum_projected_resident_memory_gib"]
        )
        preflights_passed = bool(pilot["all_accepted"] and qp_preflight["all_accepted"])
        projection_passed = bool(time_passed and memory_passed and preflights_passed)
        gate_enforced = bool(self.profile.get("article_eligible", False))
        result = {
            "profile": self.profile_name,
            "verification_profile": not gate_enforced,
            "article_gate_enforced": gate_enforced,
            "article_reporting_eligible": bool(gate_enforced and projection_passed),
            "projection_passed": projection_passed,
            "time_limit_passed": time_passed,
            "memory_limit_passed": memory_passed,
            "preflights_passed": preflights_passed,
            "scientific_workloads": workloads,
            "development_fit_seconds": float(fit_seconds),
            "production_fit_projection_factor": fit_factor,
            "mechanistic_p95_seconds": mechanistic_p95,
            "qp_p95_seconds": qp_p95,
            "projected_scientific_seconds": projected_seconds,
            "projected_scientific_core_days": projected_core_days,
            "maximum_projected_core_days": float(settings["maximum_projected_core_days"]),
            "fit_peak_resident_memory_bytes": fit_memory,
            "qp_peak_resident_memory_bytes": qp_memory,
            "mechanistic_peak_resident_memory_bytes": mechanistic_memory,
            "projected_peak_resident_memory_bytes": projected_memory_bytes,
            "projected_peak_resident_memory_gib": projected_memory_gib,
            "maximum_projected_resident_memory_gib": float(
                settings["maximum_projected_resident_memory_gib"]
            ),
            "memory_allocation_gib": float(settings["memory_allocation_gib"]),
            "qp_preflight_rows": int(len(qp_preflight["row_indices"])),
            "qp_preflight_row_indices": np.asarray(qp_preflight["row_indices"]).tolist(),
            "eligibility_note": (
                "full-profile article feasibility gate"
                if gate_enforced
                else "verification profile records the full-study projection but cannot establish article eligibility"
            ),
        }
        atomic_json(self.run_root / "timing" / "computational_feasibility.json", result)
        self._manifest["computational_feasibility"] = {
            "projection_passed": projection_passed,
            "article_gate_enforced": gate_enforced,
            "article_reporting_eligible": result["article_reporting_eligible"],
        }
        self._write_manifest()
        if gate_enforced and not projection_passed:
            raise StageExecutionError(
                "the full-profile computational feasibility gate failed before assessment"
            )
        return result

    def _deploy_rows(
        self,
        model: QuadraticSurrogate,
        row_scales: NetworkRowScales,
        decisions: np.ndarray,
        influents: np.ndarray,
        *,
        include_affine: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, list[dict[str, Any]]]:
        raw = model.predict(decisions, influents)
        deployed = np.empty_like(raw)
        affine = np.empty_like(raw) if include_affine else None
        diagnostics: list[dict[str, Any]] = []
        projector = self._projector(model, row_scales)
        for row in range(decisions.shape[0]):
            operating = decisions[row]
            operators = build_network_operators(
                influents[row],
                internal_recycle=float(operating[2]),
                return_recycle=float(operating[3]),
                waste_fraction=float(operating[4]),
                invariant_operator=mechanism.INVARIANT_MATRIX,
                tss_weights=mechanism.TSS_VECTOR,
                layout=self._layout,
            )
            if affine is not None:
                affine[row] = affine_projection(
                    raw[row],
                    operators.equality_matrix,
                    operators.equality_rhs,
                    model.response_scale,
                )
            try:
                result = projector.project(
                    raw[row],
                    operators.equality_matrix,
                    operators.equality_rhs,
                    operators.inequality_matrix,
                )
            except ProjectionError as exc:
                raise StageExecutionError(f"deployment QP failed at local row {row}: {exc}") from exc
            deployed[row] = result.state
            diagnostics.append({"row": row, **result.diagnostics.as_dict()})
        return deployed, affine, diagnostics

    @staticmethod
    def _coordinate_metrics(
        truth: np.ndarray,
        prediction: np.ndarray,
        development_scale: np.ndarray,
        form: str,
        coordinate_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        error = prediction - truth
        rmse = np.sqrt(np.mean(np.square(error), axis=0))
        mae = np.mean(np.abs(error), axis=0)
        bias = np.mean(error, axis=0)
        denominator = np.sum(np.square(truth - np.mean(truth, axis=0)), axis=0)
        r_squared = np.full(truth.shape[1], np.nan)
        denominator_scale = np.sqrt(denominator / truth.shape[0])
        denominator_reference = np.maximum(1.0, np.max(np.abs(truth), axis=0))
        variable = denominator_scale > 1.0e-12 * denominator_reference
        r_squared[variable] = 1.0 - np.sum(np.square(error[:, variable]), axis=0) / denominator[variable]
        return pd.DataFrame(
            {
                "form": form,
                "coordinate": tuple(coordinate_names or _target_columns()),
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "nrmse": rmse / development_scale,
                "r_squared": r_squared,
            }
        )

    def _stage_assessment(self) -> dict[str, Any]:
        decisions, influents, targets = self._load_dataset()
        development = slice(0, self.development_count)
        assessment = slice(self.development_count, self.sample_count)
        fit_memory_before = _peak_resident_memory_bytes()
        started = perf_counter()
        model, row_scales = self._fit_response(
            decisions[development], influents[development], targets[development]
        )
        fit_seconds = perf_counter() - started
        fit_peak_memory = max(fit_memory_before, _peak_resident_memory_bytes())
        model_path = self.run_root / "models" / "development_surrogate.npz"
        save_surrogate_bundle(model_path, model, row_scales)

        qp_preflight = self._cold_qp_preflight(
            model,
            row_scales,
            decisions[development],
            influents[development],
        )
        atomic_npz(
            self.run_root / "timing" / "qp_preflight_times.npz",
            row=np.asarray(qp_preflight["row_indices"], dtype=np.int64),
            elapsed_seconds=np.asarray(qp_preflight["elapsed_seconds"], dtype=np.float64),
        )
        atomic_parquet(
            self.run_root / "timing" / "qp_preflight_diagnostics.parquet",
            pd.DataFrame(qp_preflight["diagnostics"]),
        )
        feasibility = self._evaluate_computational_feasibility(
            fit_seconds=fit_seconds,
            fit_peak_resident_memory_bytes=fit_peak_memory,
            qp_preflight=qp_preflight,
        )

        raw = model.predict(decisions[assessment], influents[assessment])
        deployed, affine, qp_diagnostics = self._deploy_rows(
            model,
            row_scales,
            decisions[assessment],
            influents[assessment],
            include_affine=True,
        )
        assert affine is not None
        truth = targets[assessment]
        standardized_raw_error = (raw - truth) / model.response_scale
        standardized_deployed_error = (deployed - truth) / model.response_scale
        correction = (deployed - raw) / model.response_scale
        correction_rms = np.linalg.norm(correction, axis=1) / math.sqrt(truth.shape[1])
        state_error = float(np.sqrt(np.mean(np.square(standardized_deployed_error))))
        correction_ratio = float(
            np.sqrt(
                np.sum(np.square(correction))
                / max(1.0e-24, float(np.sum(np.square(standardized_raw_error))))
            )
        )
        correction_p95 = float(
            np.sort(correction_rms)[math.ceil(0.95 * correction_rms.size) - 1]
        )
        raw_norm_squared = np.sum(np.square(standardized_raw_error), axis=1)
        deployed_norm_squared = np.sum(np.square(standardized_deployed_error), axis=1)
        gates = self.config["surrogate"]["assessment"]
        relative_slack = float(gates["nonexpansive_relative_slack"])
        nonexpansive_limit = raw_norm_squared + relative_slack * (1.0 + raw_norm_squared)
        nonexpansive_pass = bool(np.all(deployed_norm_squared <= nonexpansive_limit))
        passed = bool(
            state_error < float(gates["deployed_state_standardized_rmse_max"])
            and correction_ratio <= float(gates["correction_to_raw_error_ratio_max"])
            and correction_p95 <= float(gates["correction_rms_p95_max"])
            and nonexpansive_pass
        )

        atomic_npz(
            self.run_root / "predictions" / "assessment_predictions.npz",
            row=np.arange(self.development_count, self.sample_count, dtype=np.int64),
            truth=truth,
            raw=raw,
            affine=affine,
            deployed=deployed,
            correction_rms=correction_rms,
        )
        metric_frame = pd.concat(
            [
                self._coordinate_metrics(truth, raw, model.response_scale, "raw"),
                self._coordinate_metrics(truth, affine, model.response_scale, "affine"),
                self._coordinate_metrics(truth, deployed, model.response_scale, "deployed"),
            ],
            ignore_index=True,
        )
        atomic_csv(self.run_root / "metrics" / "assessment_coordinate_metrics.csv", metric_frame)
        block_slices: dict[str, slice] = {"mixer": self._layout.mixer_slice}
        for stage in range(self._layout.stage_count):
            block_slices[f"reactor_{stage + 1}"] = self._layout.reactor_slice(stage)
        block_slices.update(
            {
                "overflow_flow": self._layout.overflow_flow_slice,
                "underflow_flow": self._layout.underflow_flow_slice,
                "clarifier_layers": self._layout.layer_slice,
                "complete_state": slice(0, self._layout.state_size),
            }
        )
        block_records = []
        for form, prediction in (("raw", raw), ("affine", affine), ("deployed", deployed)):
            standardized_error = (prediction - truth) / model.response_scale
            for block, block_slice in block_slices.items():
                block_records.append(
                    {
                        "form": form,
                        "block": block,
                        "standardized_rmse": float(
                            np.sqrt(np.mean(np.square(standardized_error[:, block_slice])))
                        ),
                    }
                )
        atomic_csv(
            self.run_root / "metrics" / "assessment_block_metrics.csv",
            pd.DataFrame(block_records),
        )

        development_qe = 1.0 - decisions[development, 4]
        development_qu = decisions[development, 3] + decisions[development, 4]
        assessment_qe = 1.0 - decisions[assessment, 4]
        assessment_qu = decisions[assessment, 3] + decisions[assessment, 4]

        def derived_response(values: np.ndarray, qe: np.ndarray, qu: np.ndarray) -> np.ndarray:
            overflow_concentration = values[:, self._layout.overflow_flow_slice] / qe[:, None]
            underflow_concentration = values[:, self._layout.underflow_flow_slice] / qu[:, None]
            normalized_clarifier_inventory = (
                float(self.config["clarifier"]["layer_volume_m3"])
                / float(self.config["process"]["fresh_flow_m3_per_d"])
                * np.sum(values[:, self._layout.layer_slice], axis=1)
            )
            return np.column_stack(
                (
                    overflow_concentration @ mechanism.COMPOSITE_MATRIX.T,
                    underflow_concentration @ mechanism.COMPOSITE_MATRIX.T,
                    normalized_clarifier_inventory,
                )
            )

        development_derived = derived_response(
            targets[development], development_qe, development_qu
        )
        derived_scale = np.std(development_derived, axis=0, ddof=0)
        derived_reference = np.maximum(1.0, np.max(np.abs(development_derived), axis=0))
        if np.any(derived_scale <= 1.0e-12 * derived_reference):
            raise StageExecutionError("an assessment-derived response fails the variance rule")
        truth_derived = derived_response(truth, assessment_qe, assessment_qu)
        derived_names = (
            "overflow_COD", "overflow_TN", "overflow_TP", "overflow_TSS",
            "underflow_COD", "underflow_TN", "underflow_TP", "underflow_TSS",
            "normalized_clarifier_inventory",
        )
        derived_metrics = pd.concat(
            [
                self._coordinate_metrics(
                    truth_derived,
                    derived_response(prediction, assessment_qe, assessment_qu),
                    derived_scale,
                    form,
                    derived_names,
                )
                for form, prediction in (("raw", raw), ("affine", affine), ("deployed", deployed))
            ],
            ignore_index=True,
        )
        atomic_csv(
            self.run_root / "metrics" / "assessment_derived_metrics.csv",
            derived_metrics,
        )
        atomic_parquet(
            self.run_root / "metrics" / "assessment_qp_diagnostics.parquet",
            pd.DataFrame(qp_diagnostics),
        )
        summary = {
            "passed": passed,
            "development_rows": self.development_count,
            "assessment_rows": self.assessment_count,
            "fit_seconds": fit_seconds,
            "state_standardized_rmse": state_error,
            "correction_to_raw_error_ratio": correction_ratio,
            "correction_rms_p95": correction_p95,
            "nonexpansive_pass": nonexpansive_pass,
            "nonexpansive_failures": int(np.count_nonzero(deployed_norm_squared > nonexpansive_limit)),
            "fit_diagnostics": model.diagnostics.as_dict(),
            "model_sha256": sha256_file(model_path),
            "computational_feasibility_projection_passed": feasibility[
                "projection_passed"
            ],
            "article_reporting_eligible": feasibility["article_reporting_eligible"],
        }
        atomic_json(self.run_root / "metrics" / "assessment_summary.json", summary)
        if not passed:
            raise StageExecutionError(
                "the predeclared one-time assessment gate failed; production fitting is prohibited"
            )
        return summary

    def _stage_production(self) -> dict[str, Any]:
        decisions, influents, targets = self._load_dataset()
        started = perf_counter()
        model, row_scales = self._fit_response(decisions, influents, targets)
        fit_seconds = perf_counter() - started
        deployed, _, qp_diagnostics = self._deploy_rows(
            model, row_scales, decisions, influents, include_affine=False
        )
        q_effluent = 1.0 - decisions[:, 4]
        overflow = targets[:, self._layout.overflow_flow_slice] / q_effluent[:, None]
        overflow_composites = overflow @ mechanism.COMPOSITE_MATRIX.T
        quality_scale = np.std(overflow_composites, axis=0, ddof=0)
        reference = np.maximum(1.0, np.max(np.abs(overflow_composites), axis=0))
        if np.any(quality_scale <= 1.0e-12 * reference):
            raise StageExecutionError("a production effluent composite fails the variance rule")
        design_matrix = model.feature_map.transform(decisions, influents)
        q_matrix, qr_upper = np.linalg.qr(design_matrix, mode="reduced")
        leverages = np.sum(np.square(q_matrix), axis=1)
        leverage_max = float(np.max(leverages))
        if np.any(np.abs(np.diag(qr_upper)) <= np.finfo(np.float64).eps):
            raise StageExecutionError("the production QR factor is singular for leverage evaluation")

        model_path = self.run_root / "models" / "production_surrogate.npz"
        save_surrogate_bundle(
            model_path,
            model,
            row_scales,
            quality_scale=quality_scale,
            leverage_max=np.asarray([leverage_max]),
            feature_qr_upper=qr_upper,
        )
        atomic_parquet(
            self.run_root / "metrics" / "production_qp_diagnostics.parquet",
            pd.DataFrame(qp_diagnostics),
        )
        replay_error = float(
            np.sqrt(np.mean(np.square((deployed - targets) / model.response_scale)))
        )
        summary = {
            "rows": self.sample_count,
            "fit_seconds": fit_seconds,
            "all_qps_accepted": True,
            "training_replay_standardized_rmse": replay_error,
            "quality_scales": quality_scale.tolist(),
            "maximum_training_leverage": leverage_max,
            "leverage_factorization": "production thin-QR upper triangular factor",
            "fit_diagnostics": model.diagnostics.as_dict(),
            "model_sha256": sha256_file(model_path),
        }
        atomic_json(self.run_root / "metrics" / "production_summary.json", summary)
        return summary

    def _physical_objective_and_engineering_constraints(
        self,
        decisions: np.ndarray,
        state: np.ndarray,
        quality_scale: np.ndarray,
    ) -> tuple[float, np.ndarray, dict[str, float]]:
        hrt, aeration, internal, returned, waste = (float(value) for value in decisions)
        q_effluent = 1.0 - waste
        q_underflow = returned + waste
        q_clarifier = 1.0 + returned
        final_reactor = state[self._layout.reactor_slice(self._layout.stage_count - 1)]
        overflow_flow = state[self._layout.overflow_flow_slice]
        underflow_flow = state[self._layout.underflow_flow_slice]
        layers = state[self._layout.layer_slice]
        effluent = overflow_flow / q_effluent
        underflow = underflow_flow / q_underflow
        composites = mechanism.COMPOSITE_MATRIX @ effluent
        process = self.config["process"]
        decision_bounds = process["decision_bounds"]
        objective_settings = self.config["objective"]
        quality_weights = np.asarray(
            objective_settings["quality_composite_weights"], dtype=np.float64
        )
        if quality_weights.shape != (4,) or np.any(quality_weights < 0.0) or not np.isclose(
            np.sum(quality_weights), 1.0
        ):
            raise StageExecutionError("quality composite weights must be four nonnegative values summing to one")
        quality = float(quality_weights @ (composites / quality_scale))
        tss_underflow = float(mechanism.TSS_VECTOR @ underflow)
        objective = (
            float(objective_settings["quality_weight"]) * quality
            + float(objective_settings["H_weight"])
            * (hrt - float(decision_bounds["H"][0]))
            / (float(decision_bounds["H"][1]) - float(decision_bounds["H"][0]))
            + float(objective_settings["a_weight"]) * aeration
            + float(objective_settings["r_I_weight"])
            * (internal - float(decision_bounds["r_I"][0]))
            / (float(decision_bounds["r_I"][1]) - float(decision_bounds["r_I"][0]))
            + float(objective_settings["r_R_weight"])
            * (returned - float(decision_bounds["r_R"][0]))
            / (float(decision_bounds["r_R"][1]) - float(decision_bounds["r_R"][0]))
            + float(objective_settings["wasted_solids_weight"])
            * waste
            * tss_underflow
            / (
                float(decision_bounds["w"][1])
                * float(objective_settings["underflow_tss_reference_g_m3"])
            )
        )
        fresh_flow = float(process["fresh_flow_m3_per_d"])
        stage_volume = fresh_flow * hrt / (24.0 * self._layout.stage_count)
        reactor_inventory = 0.0
        for stage in range(self._layout.stage_count):
            reactor_inventory += stage_volume * float(
                mechanism.TSS_VECTOR @ state[self._layout.reactor_slice(stage)]
            )
        clarifier_inventory = float(self.config["clarifier"]["layer_volume_m3"]) * float(
            np.sum(layers)
        )
        boundary_solids = fresh_flow * (
            q_effluent * float(mechanism.TSS_VECTOR @ effluent) + waste * tss_underflow
        )
        if boundary_solids <= 1.0e-10 * fresh_flow * max(
            1.0, q_clarifier * float(mechanism.TSS_VECTOR @ final_reactor)
        ):
            raise FloatingPointError("whole-plant SRT denominator is numerically zero")
        srt = (reactor_inventory + clarifier_inventory) / boundary_solids
        area = float(self.config["clarifier"]["surface_area_m2"])
        surface_overflow_rate = fresh_flow * q_effluent / area
        solids_loading_rate = (
            fresh_flow
            * q_clarifier
            * float(mechanism.TSS_VECTOR @ final_reactor)
            / (1000.0 * area)
        )
        upper = self.config["upper_constraints"]
        engineering = np.asarray(
            [
                (float(upper["srt_d"][0]) - srt) / float(upper["srt_d"][0]),
                (srt - float(upper["srt_d"][1])) / float(upper["srt_d"][1]),
                (surface_overflow_rate - float(upper["surface_overflow_rate_max_m_d"]))
                / float(upper["surface_overflow_rate_max_m_d"]),
                (solids_loading_rate - float(upper["solids_loading_rate_max_kg_m2_d"]))
                / float(upper["solids_loading_rate_max_kg_m2_d"]),
                (tss_underflow - float(upper["underflow_tss_max_g_m3"]))
                / float(upper["underflow_tss_max_g_m3"]),
            ],
            dtype=np.float64,
        )
        return float(objective), engineering, {
            "quality": quality,
            "srt_d": srt,
            "surface_overflow_rate_m_d": surface_overflow_rate,
            "solids_loading_rate_kg_m2_d": solids_loading_rate,
            "underflow_tss_g_m3": tss_underflow,
        }

    def _state_process_diagnostics(
        self, decisions: np.ndarray, state: np.ndarray
    ) -> dict[str, Any]:
        operating = mechanism.OperatingPoint(*[float(value) for value in decisions])
        reactors = np.vstack(
            [state[self._layout.reactor_slice(stage)] for stage in range(self._layout.stage_count)]
        )
        final_reactor = reactors[-1]
        overflow_flow = state[self._layout.overflow_flow_slice]
        underflow_flow = state[self._layout.underflow_flow_slice]
        layers = state[self._layout.layer_slice]
        effluent = overflow_flow / operating.q_effluent
        underflow = underflow_flow / operating.q_underflow
        feed_mass = operating.q_clarifier * final_reactor
        recoveries = np.full(self._layout.component_count, np.nan, dtype=np.float64)
        available = np.abs(feed_mass) > 1.0e-12
        recoveries[available] = underflow_flow[available] / feed_mass[available]
        feed_tss = float(mechanism.TSS_VECTOR @ final_reactor)
        return {
            "effluent_concentration": effluent,
            "underflow_concentration": underflow,
            "effluent_composites": mechanism.COMPOSITE_MATRIX @ effluent,
            "underflow_composites": mechanism.COMPOSITE_MATRIX @ underflow,
            "component_recoveries": recoveries,
            "particulate_recoveries": recoveries[list(self._layout.particulate_indices)],
            "underflow_densification": (
                float(mechanism.TSS_VECTOR @ underflow) / feed_tss
                if feed_tss > 1.0e-12
                else np.nan
            ),
            "normalized_clarifier_inventory": (
                mechanism.CLARIFIER.layer_volume
                * float(np.sum(layers))
                / mechanism.CLARIFIER.fresh_flow
            ),
            "reactor_process_rates": np.vstack(
                [mechanism.process_rates(reactor) for reactor in reactors]
            ),
            "reactor_oxygen_transfer_rates": np.asarray(
                [
                    mechanism.oxygen_transfer(reactor, stage, operating.aeration)
                    for stage, reactor in enumerate(reactors)
                ],
                dtype=np.float64,
            ),
            "clarifier_interface_fluxes": mechanism.clarifier_fluxes(
                layers, feed_tss, operating
            ),
            "clarifier_rate_residual": mechanism.clarifier_rhs(
                layers, feed_tss, operating
            ),
        }

    def _surrogate_candidate(
        self,
        decisions: np.ndarray,
        influent: np.ndarray,
        model: QuadraticSurrogate,
        row_scales: NetworkRowScales,
        extras: Mapping[str, np.ndarray],
        projector: PhysicalProjector,
    ) -> dict[str, Any]:
        raw = model.predict(decisions, influent)
        operators = build_network_operators(
            influent,
            internal_recycle=float(decisions[2]),
            return_recycle=float(decisions[3]),
            waste_fraction=float(decisions[4]),
            invariant_operator=mechanism.INVARIANT_MATRIX,
            tss_weights=mechanism.TSS_VECTOR,
            layout=self._layout,
        )
        try:
            projection = projector.project(
                raw,
                operators.equality_matrix,
                operators.equality_rhs,
                operators.inequality_matrix,
                warm_start=None,
            )
        except Exception as exc:
            return {
                "accepted": False,
                "objective": None,
                "maximum_violation": None,
                "merit": 2.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        state = projection.state
        try:
            objective, engineering, quantities = self._physical_objective_and_engineering_constraints(
                decisions, state, extras["quality_scale"]
            )
            feature = model.feature_map.transform(decisions, influent)
            leverage = _qr_leverage(extras["feature_qr_upper"], feature)
            leverage_max = float(extras["leverage_max"][0])
            correction_rms = float(np.linalg.norm(projection.displacement) / math.sqrt(state.size))
            final_reactor = state[self._layout.reactor_slice(self._layout.stage_count - 1)]
            underflow_flow = state[self._layout.underflow_flow_slice]
            q_clarifier = 1.0 + float(decisions[3])
            recoveries: list[float] = []
            for component in self._layout.particulate_indices:
                reference = max(
                    1.0,
                    abs(float(model.response_center[self._layout.reactor_slice(4).start + component]))
                    + float(model.response_scale[self._layout.reactor_slice(4).start + component]),
                )
                feed_mass = q_clarifier * float(final_reactor[component])
                if feed_mass > 1.0e-10 * q_clarifier * reference:
                    recoveries.append(float(underflow_flow[component]) / feed_mass)
            recovery_spread = max(recoveries) - min(recoveries) if len(recoveries) >= 2 else 0.0
            layers = state[self._layout.layer_slice]
            operating = mechanism.OperatingPoint(*[float(value) for value in decisions])
            feed_tss = float(mechanism.TSS_VECTOR @ final_reactor)
            layer_residual = mechanism.CLARIFIER.layer_volume * mechanism.clarifier_rhs(
                layers, feed_tss, operating
            )
            flux_residual = float(np.max(np.abs(layer_residual))) / (
                mechanism.CLARIFIER.fresh_flow * max(q_clarifier * feed_tss, 1.0)
            )
            upper = self.config["upper_constraints"]
            trust = np.asarray(
                [
                    (correction_rms - float(upper["correction_rms_max"]))
                    / float(upper["correction_rms_max"]),
                    (leverage - leverage_max) / max(leverage_max, 1.0e-12),
                    (recovery_spread - float(upper["particulate_recovery_spread_max"]))
                    / float(upper["particulate_recovery_spread_max"]),
                    (flux_residual - float(upper["clarifier_flux_residual_max"]))
                    / float(upper["clarifier_flux_residual_max"]),
                ],
                dtype=np.float64,
            )
            constraints = np.concatenate((engineering, trust))
            maximum_violation = float(np.max(np.maximum(constraints, 0.0)))
            merit = feasibility_first_merit(
                objective, maximum_violation, accepted=True,
                feasibility_tolerance=float(upper["normalized_feasibility_tolerance"]),
            )
            return {
                "accepted": True,
                "objective": objective,
                "maximum_violation": maximum_violation,
                "merit": merit,
                "correction_rms": correction_rms,
                "leverage": leverage,
                "recovery_spread": recovery_spread,
                "clarifier_flux_residual": flux_residual,
                "constraints": constraints,
                "state": state,
                "qp_diagnostics": projection.diagnostics.as_dict(),
                "qp_warm_start_used": False,
                **quantities,
            }
        except Exception as exc:
            return {
                "accepted": False,
                "objective": None,
                "maximum_violation": None,
                "merit": 2.0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _search_settings(self, budget: int, *, mechanistic: bool, robustness: bool) -> SearchSettings:
        optimization = self.config["optimization"]
        if self.profile_name == "full":
            if not mechanistic:
                block = optimization["surrogate_full"]
                full_cap = int(block["full_box_direct_cap"])
                face_cap = int(block["new_points_per_face"])
                seeds = int(block["maximum_local_seeds"])
            else:
                name = "mechanistic_robustness_full" if robustness else "mechanistic_nominal_full"
                block = optimization[name]
                full_cap = int(block["full_box_direct_cap"])
                face_cap = int(block["new_points_per_face"])
                seeds = int(block["maximum_local_seeds"])
        else:
            full_cap = max(1, int(0.55 * budget))
            face_cap = max(1, int(0.015 * budget))
            seeds = 1 if robustness else min(4, max(1, budget // 100))
        return SearchSettings(
            total_budget=budget,
            full_direct_budget=full_cap,
            face_direct_budget=face_cap,
            direct_epsilon=float(optimization["direct_epsilon"]),
            direct_resolution=float(optimization["direct_maximum_side"]),
            local_seed_count=seeds,
            initial_seed_separation=float(optimization["basin_separation_linf"]),
            initial_mesh=float(optimization["pattern_initial_mesh"]),
            terminal_mesh=float(optimization["pattern_terminal_mesh"]),
        )

    def _surrogate_search_case(
        self,
        case_id: str,
        influent: np.ndarray,
        model: QuadraticSurrogate,
        row_scales: NetworkRowScales,
        extras: Mapping[str, np.ndarray],
    ) -> tuple[dict[str, Any], SearchResult]:
        details: dict[bytes, dict[str, Any]] = {}
        projector = self._projector(model, row_scales)

        def evaluate(decisions: np.ndarray) -> float:
            key = np.ascontiguousarray(decisions, dtype="<f8").tobytes()
            if key not in details:
                details[key] = self._surrogate_candidate(
                    decisions, influent, model, row_scales, extras, projector
                )
            return float(details[key]["merit"])

        bounds = tuple(
            tuple(float(value) for value in pair)
            for pair in self.config["process"]["decision_bounds"].values()
        )
        budget = int(self.profile["surrogate_search_budget"])
        search = deterministic_bounded_search(
            evaluate, bounds, settings=self._search_settings(budget, mechanistic=False, robustness=False)
        )
        archive_records: list[dict[str, Any]] = []
        for record in search.records:
            key = np.ascontiguousarray(record.physical_point, dtype="<f8").tobytes()
            detail = details[key]
            archive_records.append(
                {
                    "evaluation": record.evaluation,
                    "phase": record.phase,
                    "merit": record.value,
                    **{name: value for name, value in zip(self.config["process"]["decision_bounds"], record.physical_point)},
                    "accepted": detail["accepted"],
                    "objective": detail.get("objective"),
                    "maximum_violation": detail.get("maximum_violation"),
                    "correction_rms": detail.get("correction_rms"),
                    "leverage": detail.get("leverage"),
                    "recovery_spread": detail.get("recovery_spread"),
                    "clarifier_flux_residual": detail.get("clarifier_flux_residual"),
                    "qp_retried_cold": (
                        detail.get("qp_diagnostics", {}).get("retried_cold")
                        if detail.get("qp_diagnostics")
                        else None
                    ),
                    "error": detail.get("error"),
                }
            )
        case_root = self.run_root / "optimization" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        atomic_parquet(case_root / "surrogate_search.parquet", pd.DataFrame(archive_records))
        selected_key = np.ascontiguousarray(search.x, dtype="<f8").tobytes()
        selected_search = details[selected_key]
        # The search engine does not expose rectangle-parent or poll-center
        # lineage to its objective callback.  Every trial is therefore solved
        # cold: for this strictly convex QP that is an exact deterministic
        # equivalent of the declared warm-start route.  The incumbent is then
        # replayed in a separate cold solve as required by the article.
        selected = self._surrogate_candidate(
            search.x,
            influent,
            model,
            row_scales,
            extras,
            self._projector(model, row_scales),
        )
        tolerance = float(self.config["upper_constraints"]["normalized_feasibility_tolerance"])
        feasible = bool(
            selected["accepted"]
            and selected.get("maximum_violation") is not None
            and float(selected["maximum_violation"]) <= tolerance
        )
        summary = {
            "case_id": case_id,
            "surrogate_feasible": feasible,
            "surrogate_evaluations": search.evaluations,
            "surrogate_distinct_attempts": search.evaluations,
            "surrogate_qp_logical_evaluations": search.evaluations + 1,
            "surrogate_qp_search_cold_solves": search.evaluations,
            "surrogate_qp_final_cold_replays": 1,
            "qp_trial_start_policy": "all-cold exact deterministic equivalent",
            "final_cold_replay_performed": True,
            "final_cold_replay_accepted": bool(selected["accepted"]),
            "final_cold_qp_status": selected.get("qp_diagnostics", {}).get("status"),
            "final_cold_qp_iterations": selected.get("qp_diagnostics", {}).get("iterations"),
            "final_cold_qp_retried": selected.get("qp_diagnostics", {}).get("retried_cold"),
            "selected_decisions": search.x.tolist() if feasible else None,
            "surrogate_objective": selected.get("objective") if feasible else None,
            "surrogate_merit": selected.get("merit"),
            "search_cached_incumbent_merit": selected_search.get("merit"),
            "minimum_violation": min(
                (float(item["maximum_violation"]) for item in details.values() if item.get("maximum_violation") is not None),
                default=None,
            ),
        }
        if feasible:
            atomic_npz(case_root / "selected_surrogate_state.npz", state=selected["state"])
        return summary, search

    def _mechanistic_candidate(
        self,
        decisions: np.ndarray,
        influent: np.ndarray,
        quality_scale: np.ndarray,
    ) -> dict[str, Any]:
        if self.mechanistic_solver is not None:
            row = self.mechanistic_solver(-1, decisions.copy(), influent.copy())
        else:
            row = _solve_payload((-1, decisions.copy(), influent.copy(), self.config["mechanistic_solver"]))
        if not row.accepted:
            return {
                "accepted": False,
                "objective": None,
                "maximum_violation": None,
                "merit": 2.0,
                "error": row.error,
                "elapsed_seconds": row.elapsed_seconds,
                "mechanistic_solver_diagnostics": row.diagnostics,
            }
        try:
            objective, engineering, quantities = self._physical_objective_and_engineering_constraints(
                decisions, row.target, quality_scale
            )
            process_diagnostics = self._state_process_diagnostics(decisions, row.target)
            violation = float(np.max(np.maximum(engineering, 0.0)))
            merit = feasibility_first_merit(
                objective,
                violation,
                accepted=True,
                feasibility_tolerance=float(
                    self.config["upper_constraints"]["normalized_feasibility_tolerance"]
                ),
            )
            return {
                "accepted": True,
                "objective": objective,
                "maximum_violation": violation,
                "merit": merit,
                "state": row.target,
                "elapsed_seconds": row.elapsed_seconds,
                "mechanistic_solver_diagnostics": row.diagnostics,
                "process_diagnostics": process_diagnostics,
                **quantities,
            }
        except Exception as exc:
            return {
                "accepted": False,
                "objective": None,
                "maximum_violation": None,
                "merit": 2.0,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": row.elapsed_seconds,
            }

    def _mechanistic_reference(
        self,
        case_id: str,
        influent: np.ndarray,
        selected_decisions: np.ndarray,
        quality_scale: np.ndarray,
        *,
        robustness: bool,
    ) -> dict[str, Any]:
        budget_key = (
            "robustness_mechanistic_search_budget" if robustness else "nominal_mechanistic_search_budget"
        )
        budget = int(self.profile[budget_key])
        details: dict[bytes, dict[str, Any]] = {}
        phases: dict[bytes, str] = {}
        attempt_order: list[bytes] = []
        duplicate_cache_hits = 0

        def evaluate(decisions: np.ndarray, *, phase: str = "reference_search") -> float:
            nonlocal duplicate_cache_hits
            key = np.ascontiguousarray(decisions, dtype="<f8").tobytes()
            if key not in details:
                details[key] = self._mechanistic_candidate(decisions, influent, quality_scale)
                phases[key] = phase
                attempt_order.append(key)
            else:
                duplicate_cache_hits += 1
            return float(details[key]["merit"])

        # Insert the selected point and its complete clipped 50-direction stencil first.
        process_bounds = np.asarray(list(self.config["process"]["decision_bounds"].values()), dtype=float)
        span = process_bounds[:, 1] - process_bounds[:, 0]
        selected_z = (selected_decisions - process_bounds[:, 0]) / span
        pre_points = [selected_z]
        directions: list[np.ndarray] = []
        for coordinate in range(5):
            for sign in (-1.0, 1.0):
                direction = np.zeros(5)
                direction[coordinate] = sign
                directions.append(direction)
        for first in range(5):
            for second in range(first + 1, 5):
                for signs in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)):
                    direction = np.zeros(5)
                    direction[first], direction[second] = signs
                    directions.append(direction)
        pre_points.extend(np.clip(selected_z + direction / 128.0, 0.0, 1.0) for direction in directions)
        for pre_index, point in enumerate(pre_points):
            evaluate(
                selected_decisions
                if pre_index == 0
                else process_bounds[:, 0] + span * point,
                phase="selected_point" if pre_index == 0 else "selected_stencil",
            )
            if len(details) >= budget:
                break
        selected_key = np.ascontiguousarray(selected_decisions, dtype="<f8").tobytes()
        selected_detail = details[selected_key]
        stencil_distinct_attempts = len(details)
        remaining = budget - len(details)
        search: SearchResult | None = None
        if remaining >= 33:
            bounds = tuple(tuple(float(value) for value in pair) for pair in process_bounds)
            search = deterministic_bounded_search(
                evaluate,
                bounds,
                settings=self._search_settings(
                    remaining, mechanistic=True, robustness=robustness
                ),
            )
            for record in search.records:
                key = np.ascontiguousarray(record.physical_point, dtype="<f8").tobytes()
                if key in phases and phases[key] == "reference_search":
                    phases[key] = record.phase

        # A search partition can revisit a point already inserted by the
        # selected-point stencil.  Such a cache hit is free under the article's
        # distinct-attempt contract, so deterministically fill any released
        # units with fresh design points and finish at the exact case budget.
        fill_round = 0
        seed_base = int.from_bytes(sha256(case_id.encode("utf-8")).digest()[:8], "little")
        while len(details) < budget:
            required = budget - len(details)
            unit_fill = _splitmix64_latin_hypercube(
                max(32, required), 5, seed_base + fill_round
            )
            fill_round += 1
            before = len(details)
            for point in unit_fill:
                evaluate(process_bounds[:, 0] + span * point, phase="distinct_budget_fill")
                if len(details) == budget:
                    break
            if len(details) == before:
                raise StageExecutionError("could not construct distinct mechanistic budget-fill points")

        def is_feasible(detail: Mapping[str, Any]) -> bool:
            violation = detail.get("maximum_violation")
            return bool(
                detail.get("accepted", False)
                and violation is not None
                and float(violation) <= float(
                    self.config["upper_constraints"]["normalized_feasibility_tolerance"]
                )
            )

        best_key, best = min(
            details.items(), key=lambda item: (float(item[1]["merit"]), item[0])
        )
        best_decisions = np.frombuffer(best_key, dtype="<f8").astype(np.float64, copy=True)
        feasible = is_feasible(best)
        selected_feasible = is_feasible(selected_detail)
        archive = []
        for evaluation, key in enumerate(attempt_order, start=1):
            detail = details[key]
            point = np.frombuffer(key, dtype="<f8")
            archive.append(
                {
                    "evaluation": evaluation,
                    "phase": phases[key],
                    **{name: value for name, value in zip(self.config["process"]["decision_bounds"], point)},
                    "accepted": detail["accepted"],
                    "objective": detail.get("objective"),
                    "maximum_violation": detail.get("maximum_violation"),
                    "merit": detail["merit"],
                    "elapsed_seconds": detail.get("elapsed_seconds"),
                    "error": detail.get("error"),
                }
            )
        case_root = self.run_root / "optimization" / case_id
        atomic_parquet(
            case_root / "mechanistic_search.parquet",
            pd.DataFrame(archive),
        )
        selected_record = {
            key: value
            for key, value in selected_detail.items()
            if key not in {"state"}
        }
        selected_record.update(
            {
                "decisions": selected_decisions,
                "influent": influent,
                "engineering_feasible": selected_feasible,
            }
        )
        atomic_json(
            case_root / "selected_mechanistic_diagnostics.json",
            _json_scalar(selected_record),
        )
        if selected_detail.get("state") is not None:
            atomic_npz(
                case_root / "selected_mechanistic_state.npz",
                decisions=selected_decisions,
                influent=influent,
                state=np.asarray(selected_detail["state"], dtype=np.float64),
            )
        if feasible and best.get("state") is not None:
            atomic_npz(
                case_root / "mechanistic_best_state.npz",
                decisions=best_decisions,
                influent=influent,
                state=np.asarray(best["state"], dtype=np.float64),
            )
        return {
            "mechanistic_reference_feasible": feasible,
            "mechanistic_evaluations": len(details),
            "mechanistic_declared_budget": budget,
            "mechanistic_budget_exhausted_exactly": len(details) == budget,
            "mechanistic_selected_stencil_distinct_attempts": stencil_distinct_attempts,
            "mechanistic_duplicate_cache_hits": duplicate_cache_hits,
            "mechanistic_best_decisions": best_decisions.tolist() if feasible else None,
            "mechanistic_best_objective": best.get("objective") if feasible else None,
            "mechanistic_search_internal_evaluations": 0 if search is None else search.evaluations,
            "selected_mechanistic_accepted": bool(selected_detail.get("accepted", False)),
            "selected_mechanistic_engineering_feasible": selected_feasible,
            "selected_mechanistic_objective": selected_detail.get("objective"),
            "selected_mechanistic_maximum_violation": selected_detail.get("maximum_violation"),
            "selected_mechanistic_error": selected_detail.get("error"),
        }

    def _robustness_influents(self) -> tuple[np.ndarray, dict[str, Any]]:
        count = int(self.profile["robustness_cases"])
        seed = int(self.config["random_design"]["robustness_seed"])
        try:
            design_module = importlib.import_module("closed_loop.design")
            generated = design_module.generate_robustness_design(count, seed=seed)
            values = np.asarray(generated.physical, dtype=np.float64)
            metadata = {
                "seed": int(generated.seed),
                "final_state": int(generated.final_state),
                "draw_count": int(generated.draw_count),
                "columns": list(generated.columns),
            }
        except (ImportError, AttributeError):
            unit = generate_latin_hypercube(count, 20, seed)
            bounds = np.asarray(list(self.config["process"]["influent_bounds"].values()), dtype=float)
            values = bounds[:, 0] + unit * np.diff(bounds, axis=1).ravel()
            metadata = {"seed": seed, "generator": "SplitMix64 fallback"}
        if values.shape != (count, 20):
            raise StageExecutionError("robustness design has an inconsistent shape")
        return values, metadata

    def _stage_optimization(self) -> dict[str, Any]:
        model, row_scales, extras = load_surrogate_bundle(
            self.run_root / "models" / "production_surrogate.npz"
        )
        nominal = np.asarray(self.config["process"]["nominal_influent"], dtype=np.float64)
        robustness, robustness_metadata = self._robustness_influents()
        atomic_npz(self.run_root / "optimization" / "case_influents.npz", nominal=nominal, robustness=robustness)
        atomic_json(
            self.run_root / "optimization" / "robustness_generator.json",
            robustness_metadata,
        )
        cases = [("nominal", nominal, False)] + [
            (f"robustness_{index + 1:03d}", influent, True)
            for index, influent in enumerate(robustness)
        ]
        summaries: list[dict[str, Any]] = []
        for case_id, influent, is_robustness in cases:
            summary, _ = self._surrogate_search_case(
                case_id, influent, model, row_scales, extras
            )
            if summary["surrogate_feasible"]:
                selected = np.asarray(summary["selected_decisions"], dtype=np.float64)
                reference = self._mechanistic_reference(
                    case_id,
                    influent,
                    selected,
                    extras["quality_scale"],
                    robustness=is_robustness,
                )
                summary.update(reference)
                if not reference["selected_mechanistic_accepted"]:
                    summary["outcome_class"] = "mechanistic-evaluation failure"
                    summary["mechanistic_incumbent_gap"] = None
                elif not reference["selected_mechanistic_engineering_feasible"]:
                    summary["outcome_class"] = "decision-feasibility failure"
                    summary["mechanistic_incumbent_gap"] = None
                elif reference["mechanistic_reference_feasible"]:
                    regret = float(reference["selected_mechanistic_objective"]) - float(
                        reference["mechanistic_best_objective"]
                    )
                    if regret < -1.0e-8:
                        raise StageExecutionError("negative mechanistic incumbent gap exposes an archive inconsistency")
                    summary["mechanistic_incumbent_gap"] = max(0.0, regret)
                    summary["outcome_class"] = "verified finite-budget incumbent"
                else:
                    summary["outcome_class"] = "mechanistic-reference failure"
                    summary["mechanistic_incumbent_gap"] = None
            else:
                summary.update(
                    {
                        "mechanistic_reference_feasible": False,
                        "mechanistic_evaluations": 0,
                        "mechanistic_best_decisions": None,
                        "mechanistic_best_objective": None,
                        "mechanistic_incumbent_gap": None,
                        "selected_mechanistic_accepted": None,
                        "selected_mechanistic_engineering_feasible": None,
                        "selected_mechanistic_objective": None,
                        "selected_mechanistic_maximum_violation": None,
                        "outcome_class": "surrogate-optimization failure",
                    }
                )
            atomic_json(self.run_root / "optimization" / case_id / "summary.json", summary)
            summaries.append(summary)
        frame = pd.DataFrame(summaries)
        atomic_parquet(self.run_root / "optimization" / "case_summary.parquet", frame)
        atomic_csv(self.run_root / "tables" / "optimization_summary.csv", frame)
        result = {
            "case_count": len(summaries),
            "surrogate_feasible_cases": int(sum(bool(item["surrogate_feasible"]) for item in summaries)),
            "mechanistic_reference_feasible_cases": int(
                sum(bool(item["mechanistic_reference_feasible"]) for item in summaries)
            ),
        }
        atomic_json(self.run_root / "optimization" / "summary.json", result)
        return result

    def _stage_report(self) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        assessment = _load_json(self.run_root / "metrics" / "assessment_summary.json")
        production = _load_json(self.run_root / "metrics" / "production_summary.json")
        optimization = _load_json(self.run_root / "optimization" / "summary.json")
        dataset = _load_json(self.run_root / "checks" / "dataset_validation.json")

        dataset_table = pd.DataFrame(
            [
                {"quantity": "mechanistic rows", "value": dataset["rows"]},
                {"quantity": "development rows", "value": self.development_count},
                {"quantity": "assessment rows", "value": self.assessment_count},
                {"quantity": "target coordinates", "value": dataset["target_columns"]},
                {"quantity": "total generation seconds", "value": dataset["elapsed_seconds"]},
            ]
        )
        atomic_csv(self.run_root / "tables" / "dataset_summary.csv", dataset_table)
        assessment_table = pd.DataFrame(
            [
                {
                    "state_standardized_rmse": assessment["state_standardized_rmse"],
                    "correction_to_raw_error_ratio": assessment[
                        "correction_to_raw_error_ratio"
                    ],
                    "correction_rms_p95": assessment["correction_rms_p95"],
                    "nonexpansive_failures": assessment["nonexpansive_failures"],
                    "passed": assessment["passed"],
                }
            ]
        )
        atomic_csv(self.run_root / "tables" / "assessment_summary.csv", assessment_table)

        with np.load(
            self.run_root / "predictions" / "assessment_predictions.npz", allow_pickle=False
        ) as payload:
            truth = payload["truth"]
            raw = payload["raw"]
            deployed = payload["deployed"]
        flattened_truth = truth.ravel()
        flattened_raw = raw.ravel()
        flattened_deployed = deployed.ravel()
        if flattened_truth.size > 20_000:
            indices = np.linspace(0, flattened_truth.size - 1, 20_000, dtype=np.int64)
            flattened_truth = flattened_truth[indices]
            flattened_raw = flattened_raw[indices]
            flattened_deployed = flattened_deployed[indices]
        figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
        limits = (
            float(min(np.min(flattened_truth), np.min(flattened_raw), np.min(flattened_deployed))),
            float(max(np.max(flattened_truth), np.max(flattened_raw), np.max(flattened_deployed))),
        )
        for axis, prediction, title in zip(
            axes, (flattened_raw, flattened_deployed), ("Raw response", "Physically corrected")
        ):
            axis.scatter(flattened_truth, prediction, s=3, alpha=0.2, rasterized=True)
            axis.plot(limits, limits, color="black", linewidth=1)
            axis.set(xlabel="Mechanistic state", ylabel="Predicted state", title=title)
        figure.savefig(self.run_root / "figures" / "assessment_parity.png", dpi=220)
        figure.savefig(self.run_root / "figures" / "assessment_parity.pdf")
        plt.close(figure)

        optimization_frame = pd.read_parquet(
            self.run_root / "optimization" / "case_summary.parquet"
        )
        figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
        positions = np.arange(len(optimization_frame))
        values = pd.to_numeric(
            optimization_frame["surrogate_objective"], errors="coerce"
        ).to_numpy()
        axis.bar(positions, np.nan_to_num(values, nan=0.0), color="#4472C4")
        axis.set(
            xlabel="Case",
            ylabel="Selected surrogate objective",
            xticks=positions,
            xticklabels=optimization_frame["case_id"],
        )
        axis.tick_params(axis="x", rotation=90)
        figure.savefig(self.run_root / "figures" / "optimization_objectives.png", dpi=220)
        figure.savefig(self.run_root / "figures" / "optimization_objectives.pdf")
        plt.close(figure)

        report = {
            "run_id": self.run_id,
            "profile": self.profile_name,
            "article_eligible": bool(self.profile["article_eligible"]),
            "dataset": dataset,
            "assessment": assessment,
            "production": production,
            "optimization": optimization,
            "generated_utc": utc_now(),
        }
        atomic_json(self.run_root / "report" / "summary.json", report)
        return {
            "tables": 3,
            "figures": 4,
            "report_sha256": sha256_file(self.run_root / "report" / "summary.json"),
        }

    def _artifact_inventory(self) -> pd.DataFrame:
        excluded = {
            "COMPLETED.json",
            "manifest.json",
            "manifest.sha256",
            "artifact_inventory.csv",
        }
        records = []
        for path in sorted(self.run_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.run_root).as_posix()
            if relative in excluded:
                continue
            records.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        return pd.DataFrame(records, columns=("path", "bytes", "sha256"))

    def _terminal_replay(self) -> dict[str, Any]:
        """Reload the essential scientific record before the immutable seal."""

        decisions, influents, targets = self._load_dataset()
        dataset_path = self.run_root / "datasets" / "mechanistic_dataset.npz"
        dataset_check = _load_json(self.run_root / "checks" / "dataset_validation.json")
        if dataset_check.get("dataset_sha256") != sha256_file(dataset_path):
            raise StageExecutionError("terminal dataset replay failed its recorded digest")

        prediction_path = self.run_root / "predictions" / "assessment_predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as payload:
            assessment_rows = payload["row"]
            truth = payload["truth"]
            raw = payload["raw"]
            affine = payload["affine"]
            deployed = payload["deployed"]
        if not np.array_equal(
            assessment_rows,
            np.arange(self.development_count, self.sample_count, dtype=np.int64),
        ):
            raise StageExecutionError("terminal assessment replay found changed row membership")
        if any(
            matrix.shape != (self.assessment_count, mechanism.TARGET_SIZE)
            or not np.all(np.isfinite(matrix))
            for matrix in (truth, raw, affine, deployed)
        ):
            raise StageExecutionError("terminal assessment replay found an invalid prediction array")
        assessment = _load_json(self.run_root / "metrics" / "assessment_summary.json")
        if not bool(assessment.get("passed")):
            raise StageExecutionError("terminal replay found an unpassed assessment")

        production_path = self.run_root / "models" / "production_surrogate.npz"
        _, _, production_extras = load_surrogate_bundle(production_path)
        if "feature_qr_upper" not in production_extras:
            raise StageExecutionError("terminal production replay has no QR leverage factor")
        production = _load_json(self.run_root / "metrics" / "production_summary.json")
        production_qp = pd.read_parquet(
            self.run_root / "metrics" / "production_qp_diagnostics.parquet"
        )
        if not bool(production.get("all_qps_accepted")) or len(production_qp) != self.sample_count:
            raise StageExecutionError("terminal production replay found incomplete QP acceptance")

        optimization = _load_json(self.run_root / "optimization" / "summary.json")
        expected_cases = 1 + int(self.profile["robustness_cases"])
        if int(optimization.get("case_count", -1)) != expected_cases:
            raise StageExecutionError("terminal optimization replay found an incomplete case set")
        case_summaries: list[dict[str, Any]] = []
        for case_id in ["nominal", *[f"robustness_{index + 1:03d}" for index in range(expected_cases - 1)]]:
            summary = _load_json(self.run_root / "optimization" / case_id / "summary.json")
            if not bool(summary.get("final_cold_replay_performed")):
                raise StageExecutionError(f"terminal replay found no final cold QP for {case_id}")
            if bool(summary.get("surrogate_feasible")) and not bool(
                summary.get("mechanistic_budget_exhausted_exactly")
            ):
                raise StageExecutionError(
                    f"terminal replay found an incomplete mechanistic budget for {case_id}"
                )
            case_summaries.append(summary)

        report = _load_json(self.run_root / "report" / "summary.json")
        if report.get("run_id") != self.run_id or report.get("profile") != self.profile_name:
            raise StageExecutionError("terminal report replay refers to another run")
        result = {
            "passed": True,
            "dataset_rows_replayed": int(decisions.shape[0]),
            "assessment_predictions_replayed": int(truth.shape[0]),
            "production_qp_diagnostics_replayed": int(len(production_qp)),
            "optimization_cases_replayed": len(case_summaries),
            "all_values_finite": bool(
                np.all(np.isfinite(decisions))
                and np.all(np.isfinite(influents))
                and np.all(np.isfinite(targets))
            ),
            "essential_artifact_sha256": self._artifact_hashes(
                (
                    dataset_path,
                    prediction_path,
                    production_path,
                    self.run_root / "metrics" / "assessment_summary.json",
                    self.run_root / "metrics" / "production_summary.json",
                    self.run_root / "optimization" / "case_summary.parquet",
                    self.run_root / "report" / "summary.json",
                )
            ),
            "completed_utc": utc_now(),
        }
        atomic_json(self.run_root / "checks" / "terminal_replay.json", result)
        return result

    def _stage_complete(self) -> dict[str, Any]:
        required = STAGES[:-1]
        missing = [stage for stage in required if not self._is_stage_complete(stage)]
        if missing:
            raise StageExecutionError(f"cannot seal run; incomplete stages: {missing}")
        replay = self._terminal_replay()
        return {
            "terminal_replay_passed": True,
            "terminal_replay_sha256": sha256_file(
                self.run_root / "checks" / "terminal_replay.json"
            ),
            "terminal_replay": replay,
        }

    def _finalize_seal(self) -> None:
        # The complete-stage marker must exist before inventory construction.
        # Mutable seal metadata are deliberately excluded from the inventory,
        # so the recorded hashes remain true after the manifest is finalized.
        inventory = self._artifact_inventory()
        atomic_csv(self.run_root / "artifact_inventory.csv", inventory)
        inventory_digest = sha256_file(self.run_root / "artifact_inventory.csv")
        for record in inventory.to_dict(orient="records"):
            path = self.run_root / str(record["path"])
            if sha256_file(path) != record["sha256"]:
                raise StageExecutionError(f"artifact changed during final seal: {record['path']}")
        self._manifest["stages"]["complete"].update(
            {
                "artifact_count": len(inventory),
                "inventory_sha256": inventory_digest,
            }
        )
        self._manifest["status"] = "complete"
        self._manifest["completed_utc"] = utc_now()
        self._write_manifest()
        manifest_digest = sha256_file(self.manifest_path)
        _atomic_bytes(self.run_root / "manifest.sha256", f"{manifest_digest}  manifest.json\n".encode())
        completion = {
            "run_id": self.run_id,
            "profile": self.profile_name,
            "status": "complete",
            "article_eligible": bool(self.profile["article_eligible"]),
            "completed_utc": self._manifest["completed_utc"],
            "contract_sha256": self._contract["contract_sha256"],
            "manifest_sha256": manifest_digest,
            "inventory_sha256": inventory_digest,
            "artifact_count": len(inventory),
        }
        atomic_json(self.completion_path, completion)


__all__ = [
    "ContractMismatchError",
    "ClosedLoopWorkflow",
    "ImmutableRunError",
    "MechanisticRow",
    "STAGES",
    "StageExecutionError",
    "WorkflowError",
    "generate_latin_hypercube",
    "load_surrogate_bundle",
    "save_surrogate_bundle",
]
