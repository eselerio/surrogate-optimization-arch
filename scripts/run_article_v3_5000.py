"""Strict, resumable driver for the 5,000-input article-v3 calculation.

The article workload is exactly 4,000 development inputs and 1,000 untouched
test inputs. This driver creates a source-bound result tree and never reads a
preflight artifact. Expensive stages finish with atomic manifests.

Optimization enters through :func:`run_optimization_stage`, which evaluates
one deterministic local attempt for each route in each of eleven article
cases and publishes independent replay, equivalence, physical-audit, and
reporting checkpoints.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from time import perf_counter, perf_counter_ns, sleep
from typing import Any, Mapping

import casadi as ca
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from closed_loop.manuscript_v3 import (
    ARTICLE_FULL,
    DECISION_LOWER,
    DECISION_NAMES,
    DECISION_UPPER,
    RIDGE_GRID,
    AssessmentResult,
    StudyProfile,
    assess_raw_projected_mechanistic,
    clarifier_for,
    create_design,
    cross_validate_ridge,
    violation_record,
)
from closed_loop.model import (
    COMPONENTS,
    NOMINAL_INFLUENT,
    ArticleOperatingPoint,
    INVARIANT_MATRIX,
    INFLUENT_LOWER,
    INFLUENT_UPPER,
    TSS_VECTOR,
    assemble_target,
    branch_classification,
    generation_scale,
    solve_steady_state,
    unpack_state,
)
from closed_loop.projection import (
    LeastSquaresDiagnostics,
    NetworkLayout,
    PhysicalProjector,
    QuadraticFeatureMap,
    QuadraticSurrogate,
    build_network_operators,
)
from closed_loop.v3_reporting import write_reporting_tables
from closed_loop.v3_derivative_audit import audit_casadi_nlp_derivatives
from closed_loop.v3_replacement_generation import (
    MechanisticBlockResult,
    generate_mechanistic_block_with_replacements,
)
from closed_loop.v3_smooth import (
    CONTINUATION_SCHEDULE,
    DirectCase,
    DirectMultistartResult,
    DirectStartResult,
    SolverSettings,
    build_direct_nlp,
    branches_match as smooth_branches_match,
    compare_smooth_reference,
    fit_direct_assets,
    ordered_normalized_starts as direct_normalized_starts,
    solve_direct_multistart,
    solve_fixed_input_two_start,
)
from closed_loop.v3_surrogate_nlp import (
    EXACT_QP_CENTER_START,
    EXACT_QP_SINGLE_START_PROTOCOL,
    GAP_CONTINUATION,
    SurrogateCase,
    SurrogateMultistartResult,
    SurrogateSolverSettings,
    SurrogateStartResult,
    build_surrogate_nlp,
    build_surrogate_assets,
    cold_reproject,
    solve_surrogate_exact_qp_local,
)
from closed_loop.v3_trust import calibrate_trust_diagnostics


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "article_v3"
DEFAULT_RUN_ID = "article_full_5000_001"
RUNNER_SCHEMA = 5
ASSESSMENT_GATE_EXECUTION_POLICY = "advisory_continue"
DIRECT_SINGLE_CENTER_PROTOCOL = "smooth_direct_single_center_v1"
OPTIMIZATION_PROTOCOL = "single_center_local_exact_qp_v1"
RUN_ID_PATTERN = re.compile(r"^article_full_5000_[A-Za-z0-9][A-Za-z0-9_-]*$")

SOURCE_FILES = (
    "scripts/run_article_v3_5000.py",
    "scripts/build_main_closed_loop_v3.py",
    "main_closed_loop.ipynb",
    "closed_loop/model.py",
    "closed_loop/manuscript_v3.py",
    "closed_loop/projection.py",
    "closed_loop/v3_smooth.py",
    "closed_loop/v3_surrogate_nlp.py",
    "closed_loop/v3_active_set.py",
    "closed_loop/v3_derivative_audit.py",
    "closed_loop/v3_trust.py",
    "closed_loop/v3_reporting.py",
    "closed_loop/v3_replacement_generation.py",
    "config/params_manuscript_v3.json",
    "article/wip_v3/manuscript.tex",
    "article/wip_v3/supplementary_material.tex",
    "pyproject.toml",
    "uv.lock",
)
DESIGN_ARRAYS = (
    "development_decisions",
    "development_influents",
    "test_decisions",
    "test_influents",
    "robustness_influents",
)
INFERENCE_TIMING_WARMUPS = 5
INFERENCE_TIMING_BATCHES = 30


@dataclass(frozen=True)
class SourceContractMigrationAuthorization:
    """Narrow authorization for one already-started run's source migration."""

    migration_id: str
    run_id: str
    authorized_date: str
    reason: str
    predecessor_runner_schema: int
    successor_runner_schema: int
    predecessor_source_digest: str
    predecessor_contract_file_digest: str
    allowed_changed_source_files: frozenset[str]
    required_changed_source_files: frozenset[str]
    required_artifact_digests: Mapping[str, str]
    expected_accepted_rows: int
    expected_rejected_rows: int


@dataclass(frozen=True)
class AssessmentRecoveryMigrationAuthorization:
    """Pinned authorization to repair assessment without recomputing prior stages."""

    migration_id: str
    run_id: str
    authorized_date: str
    reason: str
    predecessor_runner_schema: int
    successor_runner_schema: int
    predecessor_source_digest: str
    predecessor_contract_file_digest: str
    required_prior_migration_ids: tuple[str, ...]
    predecessor_source_snapshot: str
    allowed_changed_source_files: frozenset[str]
    required_changed_source_files: frozenset[str]
    required_artifact_digests: Mapping[str, str]
    expected_effective_design_digest: str
    expected_ridge_input_digest: str


@dataclass(frozen=True)
class OptimizationProtocolMigrationAuthorization:
    """Pinned authorization to replace only the unfinished optimization stage."""

    migration_id: str
    run_id: str
    authorized_date: str
    reason: str
    predecessor_runner_schema: int
    successor_runner_schema: int
    predecessor_source_digest: str
    predecessor_contract_file_digest: str
    required_prior_migration_ids: tuple[str, ...]
    predecessor_source_snapshot: str
    allowed_changed_source_files: frozenset[str]
    required_changed_source_files: frozenset[str]
    required_artifact_digests: Mapping[str, str]
    expected_assessment_input_digest: str


GENERATION_REPLACEMENT_MIGRATION = SourceContractMigrationAuthorization(
    migration_id="article-v3-generation-replacement-v1",
    run_id=DEFAULT_RUN_ID,
    authorized_date="2026-08-23",
    reason=(
        "User-authorized replacement of rejected mechanistic generation attempts "
        "while retaining every independently verified accepted checkpoint."
    ),
    predecessor_runner_schema=2,
    successor_runner_schema=3,
    predecessor_source_digest=(
        "b1ea4a3dedc018acf461e0c9d6a4e9bf2e3124c41e1e7ea9f0dee16dc9765c87"
    ),
    predecessor_contract_file_digest=(
        "1435125e0538627d72149773cc2ac5c055dd6a1bd2da79d72836829225dbe8e5"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_replacement_generation.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_replacement_generation.py",
    }),
    required_artifact_digests={
        "inputs/generator_records.json": (
            "d8ac45c0f2cbba3cddd64ee171fbb9a0b922b246e95dce046ffb5a303355a669"
        ),
        "datasets/design.npz": (
            "e278a3bf56c4a8d1099eb88db2c3e86960027952b39547b7ac8f7f1b0ba95cb0"
        ),
        "datasets/development/mechanistic_rows_v3.npz": (
            "ac30ee5682884ea4aea21429c3016742b0fc825d5eaf6c09bdf7d26c875b454c"
        ),
        "datasets/development/mechanistic_diagnostics.csv": (
            "16d846f6f66154281ef977c925ae671a7ac1c80d423d0aac4436145daaba188b"
        ),
    },
    expected_accepted_rows=3_680,
    expected_rejected_rows=320,
)


ASSESSMENT_RECOVERY_MIGRATION = AssessmentRecoveryMigrationAuthorization(
    migration_id="article-v3-projection-audit-v1",
    run_id=DEFAULT_RUN_ID,
    authorized_date="2026-08-23",
    reason=(
        "User-authorized repair of the projection audit and continuation after "
        "an advisory assessment gate, without recomputing completed generation "
        "or ridge fitting."
    ),
    predecessor_runner_schema=3,
    successor_runner_schema=4,
    predecessor_source_digest=(
        "4e599e0fd18ff3ba17c7e9af1c047e53067ff610b9f7bb787f1effc230f3d063"
    ),
    predecessor_contract_file_digest=(
        "118c06a8d4c9f418e50477138e5eca7c8c376ecbd4653de7cdc74bf83a941f87"
    ),
    required_prior_migration_ids=("article-v3-generation-replacement-v1",),
    predecessor_source_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-projection-audit-v1-predecessor-source"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/projection.py",
        "closed_loop/v3_trust.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/projection.py",
        "closed_loop/v3_trust.py",
    }),
    required_artifact_digests={
        "inputs/generator_records.json": (
            "d8ac45c0f2cbba3cddd64ee171fbb9a0b922b246e95dce046ffb5a303355a669"
        ),
        "datasets/design.npz": (
            "e278a3bf56c4a8d1099eb88db2c3e86960027952b39547b7ac8f7f1b0ba95cb0"
        ),
        "datasets/development/block_complete.json": (
            "3409b03537613d0d98166cee151ff6e6451ef16ebd26b1cf058d25d827455c06"
        ),
        "datasets/test/block_complete.json": (
            "5387fdac043b36ab4518e35a26b59ea35b77f8f7ef7db2ddb9c3ec2c8ecbe95f"
        ),
        "datasets/effective_design.npz": (
            "7716745352e918d67f8c88e65be730f335b9b079a28cf1852de19d5d7194ca96"
        ),
        "datasets/effective_design_manifest.json": (
            "33a7d7172b178f7b220c065e0003edb8bc10cc74dfc27e89970d71a20d02c255"
        ),
        "metrics/mechanistic_generation_summary.csv": (
            "138755780cfc643259ec740b291d820268021c84ea46b076e841d0069726ae53"
        ),
        "models/ridge_complete.json": (
            "8fb5374758492d8f159b8218f6621cca112fe1d243e4f66f8bde52ada8a91a5d"
        ),
        "models/ridge_surrogate.npz": (
            "9817ba3461d8d95c9e5098ea809f9be6cf0b65724dc9fec12f883ce30eabc632"
        ),
        "metrics/ridge_cross_validation.csv": (
            "29153e71ea86ec2892cab4ff9f58d264b6feb67a80627611bc965e83142c81a5"
        ),
        "metrics/ridge_fold_membership.csv": (
            "9155b59b038cf5dd702ea8053977addd742779fcb904d6dbd5fb13aff4686cd2"
        ),
    },
    expected_effective_design_digest=(
        "2f71ebc39dcbe9d70b2c927ae273bf4fb3700ee89f6dd85ccb18bed942207d9b"
    ),
    expected_ridge_input_digest=(
        "6466850f5d4ac2e651e641e2e19b00d9fe9c9e5412621fbe7aefb6d121d5028d"
    ),
)


SINGLE_START_EXACT_QP_MIGRATION = OptimizationProtocolMigrationAuthorization(
    migration_id="article-v3-direct-active-set-v1",
    run_id=DEFAULT_RUN_ID,
    authorized_date="2026-08-23",
    reason=(
        "User-authorized replacement of the unfinished nine-start embedded-KKT "
        "optimization stage by one deterministic local attempt per route and "
        "case, with the surrogate solved by the seven-variable exact-QP route."
    ),
    predecessor_runner_schema=4,
    successor_runner_schema=5,
    predecessor_source_digest=(
        "5ca76a132a467fab76717d47c00e62e4901b51b0bbeefa6e144ed3264e843fc3"
    ),
    predecessor_contract_file_digest=(
        "c198eed5861079da4cc6b741264eee77bad035dc6ab4eaa39dfeebc018075501"
    ),
    required_prior_migration_ids=(
        "article-v3-generation-replacement-v1",
        "article-v3-projection-audit-v1",
    ),
    predecessor_source_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-direct-active-set-v1-predecessor-source"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_surrogate_nlp.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_surrogate_nlp.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_artifact_digests={
        "metrics/assessment_complete.json": (
            "2c0c9e37f539868e52b9ecd9f0f387f81f5a6e0d6d90ae9c46016bb74fc3fd3a"
        ),
        "metrics/admission_gate.json": (
            "3631f3fda28191dca854ed1cfe79fcf1114073217acbea0702d3a2db78c5f6eb"
        ),
        "metrics/untouched_prediction_metrics.csv": (
            "f446f0207b39fe4802951e94a75630620eae4ae390434679f812c12553e1aaaa"
        ),
        "metrics/physical_violations_assessment.csv": (
            "b9c5ce45431c80c99c003db89d163dba4c7a76098c1d57756ffa5ef6ce0fdc5b"
        ),
        "metrics/projection_qp_diagnostics.csv": (
            "dbb7e42a8a6c90d2bdc52883418316d3158d362b8a7e56a5a26c305dc034d9d1"
        ),
        "metrics/projection_feasibility_bound.csv": (
            "0781a621b42d69ca0c5fd983a156afc2afedd1a2f8d03a6a462d2ba77601ea6c"
        ),
        "predictions/untouched_test.npz": (
            "dbc11728e6d434d1d8b37b6623b9cd12f69deadfac2fe2fb82bdef7e05650b61"
        ),
        "metrics/trust_development_oof.csv": (
            "6658b47cd0c032e8434a82238d1ecae17afcb2acd17dd062b5bdfc297268a5dd"
        ),
        "metrics/trust_limits.json": (
            "73088c1cde7dddf6964256218b541f9fa035273bcac252fa025453930df56d4f"
        ),
        "models/trust_calibration.npz": (
            "f334c3d179f35d1ab2a748942854436cd88cc64e4e6190881bf15cd1d3409918"
        ),
    },
    expected_assessment_input_digest=(
        "49bc1639cfd85e0864dc07d16f33959e5b11c132ef7f1a95b9d89908f8d13323"
    ),
)


@dataclass(frozen=True)
class AnalysisBundle:
    passed: bool
    model: QuadraticSurrogate
    direct_assets: Any
    surrogate_assets: Any
    assessment: AssessmentResult | None
    gate: dict[str, Any]


@dataclass(frozen=True)
class GenerationResult:
    """Accepted development/test targets and their effective input design."""

    design: dict[str, object]
    development_targets: np.ndarray
    test_targets: np.ndarray

    def __iter__(self):
        """Preserve the historical two-target unpacking interface."""

        yield self.development_targets
        yield self.test_targets

    def __getitem__(self, index: int) -> np.ndarray:
        return (self.development_targets, self.test_targets)[index]


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Publish atomically despite transient Windows scanner/file-lock races."""

    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            transient = os.name == "nt" and getattr(exc, "winerror", None) in {5, 32}
            if not transient or attempt == 6:
                raise
            sleep(0.01 * (2**attempt))


def _json_ready(value: Any, *, nonfinite_to_none: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item, nonfinite_to_none=nonfinite_to_none)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, nonfinite_to_none=nonfinite_to_none) for item in value
        ]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist(), nonfinite_to_none=nonfinite_to_none)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            if nonfinite_to_none:
                return None
            raise ValueError("non-finite numbers are not permitted in JSON contracts")
        return result
    return value


def atomic_json(path: Path, value: Any, *, nonfinite_to_none: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_ready(value, nonfinite_to_none=nonfinite_to_none),
                stream, indent=2, sort_keys=True, allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file_digests() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"required article source is missing: {path}")
        result[relative] = file_digest(path)
    return result


def source_digest(files: Mapping[str, str] | None = None) -> str:
    manifest = dict(files or source_file_digests())
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def array_digest(**arrays: np.ndarray) -> str:
    digest = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _artifact_hashes(run: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    return {path.relative_to(run).as_posix(): file_digest(path) for path in paths}


def _artifacts_match(run: Path, expected: Mapping[str, str]) -> bool:
    if not expected:
        return False
    return all(
        (run / relative).is_file()
        and file_digest(run / relative) == expected_digest
        for relative, expected_digest in expected.items()
    )


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "pandas", "casadi", "osqp"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def resolve_run_directory(run_id: str, results_root: Path = RESULTS_ROOT) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise ValueError(
            "full-run id must match article_full_5000_<identifier>; "
            "preflight names and path components are forbidden"
        )
    root = results_root.resolve()
    run = (root / run_id).resolve()
    if run.parent != root:
        raise ValueError("the full-run directory must be a direct child of its result root")
    return run


def validate_authorized_profile(profile: StudyProfile) -> None:
    expected = {
        "name": "article_full",
        "development_count": 4_000,
        "test_count": 1_000,
        "robustness_count": 10,
        "layer_count": 10,
        "development_seed": 100_042,
        "test_seed": 100_043,
        "robustness_seed": 314_159,
        "article_eligible": True,
        "enforce_admission_gate": True,
    }
    actual = asdict(profile)
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items() if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"ARTICLE_FULL violates the authorized contract: {mismatches}")


def _expected_design_shapes(profile: StudyProfile) -> dict[str, tuple[int, int]]:
    return {
        "development_decisions": (profile.development_count, 7),
        "development_influents": (profile.development_count, 20),
        "test_decisions": (profile.test_count, 7),
        "test_influents": (profile.test_count, 20),
        "robustness_influents": (profile.robustness_count, 20),
    }


def validate_design(design: Mapping[str, object], profile: StudyProfile) -> None:
    for name, shape in _expected_design_shapes(profile).items():
        if name not in design:
            raise RuntimeError(f"fixed design is missing {name}")
        value = np.asarray(design[name], dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(f"fixed design {name} has invalid shape or values")
        lower, upper = (
            (DECISION_LOWER, DECISION_UPPER)
            if name.endswith("decisions") else (INFLUENT_LOWER, INFLUENT_UPPER)
        )
        if np.any(value < lower) or np.any(value > upper):
            raise RuntimeError(f"fixed design {name} lies outside its declared box")
    generators = design.get("generators")
    if not isinstance(generators, Mapping):
        raise RuntimeError("fixed design is missing generator records")
    expected_seeds = {
        "development": profile.development_seed,
        "test": profile.test_seed,
        "robustness": profile.robustness_seed,
    }
    for block, seed in expected_seeds.items():
        record = generators.get(block)
        if not isinstance(record, Mapping) or int(record.get("seed", -1)) != seed:
            raise RuntimeError(f"fixed design has an invalid {block} generator record")


def _design_digest(design: Mapping[str, object]) -> str:
    return array_digest(**{
        name: np.asarray(design[name], dtype="<f8") for name in DESIGN_ARRAYS
    })


def load_or_create_design(run: Path, profile: StudyProfile) -> dict[str, object]:
    expected = create_design(profile)
    validate_design(expected, profile)
    path = run / "datasets" / "design.npz"
    records_path = run / "inputs" / "generator_records.json"
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(DESIGN_ARRAYS):
                raise RuntimeError("existing design checkpoint has unexpected arrays")
            for name in DESIGN_ARRAYS:
                if not np.array_equal(stored[name], np.asarray(expected[name])):
                    raise RuntimeError("existing design checkpoint differs from fixed design")
    else:
        atomic_npz(path, **{name: np.asarray(expected[name]) for name in DESIGN_ARRAYS})
    expected_records = _json_ready(expected["generators"])
    if records_path.is_file():
        existing_records = json.loads(records_path.read_text(encoding="utf-8"))
        if existing_records != expected_records:
            raise RuntimeError("existing generator record differs from fixed design")
    else:
        atomic_json(records_path, expected_records)
    return expected


def _build_contract(
    run_id: str, profile: StudyProfile, source_files: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "runner_schema": RUNNER_SCHEMA,
        "run_id": run_id,
        "profile": asdict(profile),
        "fixed_dataset_total": 5_000,
        "development_test_split": [4_000, 1_000],
        "source_digest": source_digest(source_files),
        "source_files": dict(source_files),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_versions": _runtime_versions(),
        "assessment_gate_execution_policy": ASSESSMENT_GATE_EXECUTION_POLICY,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "preflight_artifacts_permitted": False,
        "full_run_admission_gate_bypass_permitted": False,
    }


def _canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def _validate_pinned_artifacts(
    run: Path, expected: Mapping[str, str],
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, required_digest in sorted(expected.items()):
        path = (run / relative).resolve()
        if run.resolve() not in path.parents or not path.is_file():
            raise RuntimeError(f"migration artifact is missing or outside the run: {relative}")
        observed_digest = file_digest(path)
        if observed_digest != required_digest:
            raise RuntimeError(
                f"migration refused because pinned artifact bytes changed: {relative}"
            )
        observed[relative] = observed_digest
    return observed


def _validate_retained_generation_checkpoints(
    run: Path, authorization: SourceContractMigrationAuthorization,
) -> dict[str, Any]:
    """Verify each retained accepted row against pinned aggregate artifacts."""

    pinned = _validate_pinned_artifacts(
        run, authorization.required_artifact_digests,
    )
    design_path = run / "datasets" / "design.npz"
    aggregate_path = run / "datasets" / "development" / "mechanistic_rows_v3.npz"
    diagnostics_path = (
        run / "datasets" / "development" / "mechanistic_diagnostics.csv"
    )
    try:
        with np.load(design_path, allow_pickle=False) as stored:
            decisions = np.asarray(stored["development_decisions"])
            influents = np.asarray(stored["development_influents"])
        with np.load(aggregate_path, allow_pickle=False) as stored:
            targets = np.asarray(stored["targets"])
            states_start_1 = np.asarray(stored["states_start_1"])
            states_start_2 = np.asarray(stored["states_start_2"])
        diagnostics = pd.read_csv(diagnostics_path)
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError("cannot inspect pinned generation artifacts") from exc
    row_count = len(decisions)
    if (
        len(influents) != row_count
        or len(targets) != row_count
        or len(states_start_1) != row_count
        or len(states_start_2) != row_count
        or len(diagnostics) != row_count
    ):
        raise RuntimeError("pinned generation artifacts disagree on row count")
    if not {"row", "accepted"}.issubset(diagnostics.columns):
        raise RuntimeError("pinned generation diagnostics omit row or accepted")
    rows = pd.to_numeric(diagnostics["row"], errors="coerce").to_numpy()
    if not np.array_equal(rows, np.arange(row_count)):
        raise RuntimeError("pinned generation diagnostics are not in fixed row order")
    accepted_values = diagnostics["accepted"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if accepted_values.isna().any():
        raise RuntimeError("pinned generation diagnostics have invalid acceptance flags")
    accepted_rows = np.flatnonzero(accepted_values.to_numpy(dtype=bool))
    rejected_rows = np.flatnonzero(~accepted_values.to_numpy(dtype=bool))
    if (
        len(accepted_rows) != authorization.expected_accepted_rows
        or len(rejected_rows) != authorization.expected_rejected_rows
    ):
        raise RuntimeError(
            "migration refused because accepted/rejected generation counts changed"
        )
    checkpoints: list[dict[str, Any]] = []
    row_contract_hashes: set[str] = set()
    for index in accepted_rows:
        relative = f"datasets/development/rows/row_{int(index):06d}.npz"
        path = run / relative
        if not path.is_file():
            raise RuntimeError(f"accepted generation checkpoint is missing: {relative}")
        try:
            with np.load(path, allow_pickle=False) as stored:
                record = json.loads(str(stored["record_json"].item()))
                valid = bool(
                    np.array_equal(stored["decision"], decisions[index])
                    and np.array_equal(stored["influent"], influents[index])
                    and np.array_equal(stored["target"], targets[index])
                    and np.array_equal(stored["state_start_1"], states_start_1[index])
                    and np.array_equal(stored["state_start_2"], states_start_2[index])
                    and int(record.get("row", -1)) == int(index)
                    and record.get("accepted") is True
                )
                row_contract_hash = str(stored["contract_hash"].item())
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot validate accepted generation checkpoint: {relative}"
            ) from exc
        if not valid or not row_contract_hash:
            raise RuntimeError(
                f"accepted generation checkpoint differs from pinned aggregates: {relative}"
            )
        row_contract_hashes.add(row_contract_hash)
        checkpoints.append({
            "original_candidate_index": int(index),
            "accepted_slot": int(index),
            "path": relative,
            "sha256": file_digest(path),
        })
    checkpoint_set_digest = _canonical_json_digest(checkpoints)
    return {
        "schema": 1,
        "block": "development",
        "original_row_count": row_count,
        "retained_accepted_count": len(accepted_rows),
        "excluded_rejected_count": len(rejected_rows),
        "excluded_original_candidate_indices": rejected_rows.astype(int).tolist(),
        "pinned_artifacts": pinned,
        "row_contract_hashes": sorted(row_contract_hashes),
        "checkpoint_set_digest": checkpoint_set_digest,
        "checkpoints": checkpoints,
    }


def _source_snapshot_manifest(
    run: Path, relative_directory: str, expected_files: Mapping[str, str],
) -> dict[str, Any]:
    """Validate and describe an exact, path-contained predecessor source snapshot."""

    root = (run / relative_directory).resolve()
    migrations_root = (run / "inputs" / "contract_migrations").resolve()
    if migrations_root not in root.parents or not root.is_dir():
        raise RuntimeError("predecessor source snapshot is missing or outside migrations")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    expected_names = set(map(str, expected_files))
    if actual != expected_names:
        raise RuntimeError(
            "predecessor source snapshot file set differs from its source manifest"
        )
    observed: dict[str, str] = {}
    for relative, expected_digest in sorted(expected_files.items()):
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(f"predecessor source snapshot is missing {relative}")
        digest = file_digest(path)
        if digest != expected_digest:
            raise RuntimeError(f"predecessor source snapshot changed: {relative}")
        observed[str(relative)] = digest
    return {
        "directory": relative_directory,
        "source_digest": source_digest(observed),
        "source_files": observed,
    }


def _validate_recorded_source_snapshot(
    run: Path, snapshot: Any,
) -> None:
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("recorded predecessor source snapshot is invalid")
    files = snapshot.get("source_files")
    if not isinstance(files, Mapping):
        raise RuntimeError("recorded predecessor source snapshot omits its manifest")
    observed = _source_snapshot_manifest(
        run, str(snapshot.get("directory", "")), files,
    )
    if observed != dict(snapshot):
        raise RuntimeError("recorded predecessor source snapshot is inconsistent")


def _validate_migration_history(run: Path, contract: Mapping[str, Any]) -> None:
    history = contract.get("contract_migrations")
    if not isinstance(history, list) or not history:
        raise RuntimeError("migrated contract has no migration history")
    seen: set[str] = set()
    prior_successor_digest: str | None = None
    prior_successor_schema: int | None = None
    for position, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            raise RuntimeError("contract migration history contains a non-object entry")
        migration_id = str(entry.get("migration_id", ""))
        relative = str(entry.get("record", ""))
        expected_digest = str(entry.get("record_digest", ""))
        if not migration_id or migration_id in seen:
            raise RuntimeError("contract migration history has duplicate/empty identifiers")
        seen.add(migration_id)
        path = (run / relative).resolve()
        migrations_root = (run / "inputs" / "contract_migrations").resolve()
        if migrations_root not in path.parents or not path.is_file():
            raise RuntimeError("contract migration record is missing or outside its directory")
        if file_digest(path) != expected_digest:
            raise RuntimeError(f"contract migration record changed: {migration_id}")
        predecessor_relative = str(entry.get("predecessor_contract", ""))
        predecessor_digest = str(entry.get("predecessor_contract_digest", ""))
        predecessor_path = (run / predecessor_relative).resolve()
        if (
            migrations_root not in predecessor_path.parents
            or not predecessor_path.is_file()
            or file_digest(predecessor_path) != predecessor_digest
        ):
            raise RuntimeError(
                f"archived predecessor contract changed: {migration_id}"
            )
        record = _load_json_object(path, description="contract migration record")
        predecessor = record.get("predecessor")
        successor = record.get("successor")
        if (
            record.get("migration_id") != migration_id
            or not isinstance(predecessor, Mapping)
            or not isinstance(successor, Mapping)
            or predecessor.get("archived_contract") != predecessor_relative
            or predecessor.get("archived_contract_digest") != predecessor_digest
            or predecessor.get("contract_file_digest") != predecessor_digest
            or entry.get("predecessor_source_digest")
            != predecessor.get("source_digest")
            or entry.get("successor_source_digest")
            != successor.get("source_digest")
        ):
            raise RuntimeError(f"contract migration record is inconsistent: {migration_id}")
        archived = _load_json_object(
            predecessor_path, description="archived predecessor contract",
        )
        archived_history = archived.get("contract_migrations", [])
        predecessor_files = predecessor.get("source_files")
        if (
            not isinstance(predecessor_files, Mapping)
            or source_digest(predecessor_files) != predecessor.get("source_digest")
            or archived.get("source_digest") != predecessor.get("source_digest")
            or archived.get("runner_schema") != predecessor.get("runner_schema")
            or archived.get("source_files") != predecessor.get("source_files")
            or archived_history != history[:position]
        ):
            raise RuntimeError(
                f"archived predecessor contract breaks migration chain: {migration_id}"
            )
        if position and (
            predecessor.get("source_digest") != prior_successor_digest
            or predecessor.get("runner_schema") != prior_successor_schema
        ):
            raise RuntimeError(f"contract migration chain is discontinuous: {migration_id}")
        prior_successor_digest = str(successor.get("source_digest", ""))
        successor_files = successor.get("source_files")
        if (
            not isinstance(successor_files, Mapping)
            or source_digest(successor_files) != prior_successor_digest
        ):
            raise RuntimeError(
                f"contract migration successor manifest is inconsistent: {migration_id}"
            )
        try:
            prior_successor_schema = int(successor.get("runner_schema", -1))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"contract migration record has invalid schema: {migration_id}"
            ) from exc
        snapshot = record.get("predecessor_source_snapshot")
        if snapshot is not None:
            _validate_recorded_source_snapshot(run, snapshot)
        for manifest_key in ("retained_checkpoint_manifest", "retained_stage_manifest"):
            manifest_reference = record.get(manifest_key)
            if manifest_reference is None:
                continue
            if not isinstance(manifest_reference, Mapping):
                raise RuntimeError(
                    f"contract migration has invalid {manifest_key}: {migration_id}"
                )
            manifest_path = (run / str(manifest_reference.get("path", ""))).resolve()
            if (
                migrations_root not in manifest_path.parents
                or not manifest_path.is_file()
                or file_digest(manifest_path) != manifest_reference.get("sha256")
            ):
                raise RuntimeError(
                    f"contract migration retained manifest changed: {migration_id}"
                )
    if (
        prior_successor_digest != contract.get("source_digest")
        or prior_successor_schema != int(contract.get("runner_schema", -1))
        or dict(successor_files) != contract.get("source_files")
    ):
        raise RuntimeError("contract migration history does not reach current contract")


def _migrate_source_contract(
    run: Path,
    previous: Mapping[str, Any],
    successor: Mapping[str, Any],
    authorization: SourceContractMigrationAuthorization,
) -> None:
    """Apply one explicit, allow-listed migration and leave immutable evidence."""

    contract_path = run / "inputs" / "contract.json"
    if str(previous.get("run_id")) != authorization.run_id:
        raise RuntimeError("source-contract migration is not authorized for this run id")
    if previous.get("contract_migrations"):
        raise RuntimeError("source-contract migration is one-time and was already used")
    if (
        int(previous.get("runner_schema", -1))
        != authorization.predecessor_runner_schema
        or int(successor.get("runner_schema", -1))
        != authorization.successor_runner_schema
        or previous.get("source_digest") != authorization.predecessor_source_digest
        or file_digest(contract_path) != authorization.predecessor_contract_file_digest
    ):
        raise RuntimeError("existing contract is not the pinned migration predecessor")
    old_files = previous.get("source_files")
    new_files = successor.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("source-contract migration requires source-file manifests")
    if source_digest(old_files) != previous.get("source_digest"):
        raise RuntimeError("predecessor source manifest is internally inconsistent")
    if source_digest(new_files) != successor.get("source_digest"):
        raise RuntimeError("successor source manifest is internally inconsistent")
    changed = {
        name for name in set(old_files) | set(new_files)
        if old_files.get(name) != new_files.get(name)
    }
    unauthorized = changed - authorization.allowed_changed_source_files
    missing_required = authorization.required_changed_source_files - changed
    if unauthorized or missing_required:
        raise RuntimeError(
            "source-contract migration refused arbitrary source drift; "
            f"unauthorized={sorted(unauthorized)}, missing_required={sorted(missing_required)}"
        )
    ignored = {"runner_schema", "source_digest", "source_files", "contract_migrations"}
    old_invariants = {key: value for key, value in previous.items() if key not in ignored}
    new_invariants = {key: value for key, value in successor.items() if key not in ignored}
    if old_invariants != new_invariants:
        raise RuntimeError("source-contract migration cannot change the run/profile contract")

    retention = _validate_retained_generation_checkpoints(run, authorization)
    migrations_directory = run / "inputs" / "contract_migrations"
    predecessor_archive = (
        migrations_directory / f"{authorization.migration_id}-predecessor-contract.json"
    )
    retention_path = migrations_directory / f"{authorization.migration_id}-retained.json"
    record_path = migrations_directory / f"{authorization.migration_id}.json"
    predecessor_bytes = contract_path.read_bytes()
    if sha256(predecessor_bytes).hexdigest() != authorization.predecessor_contract_file_digest:
        raise RuntimeError("predecessor contract bytes changed during migration validation")
    if predecessor_archive.is_file():
        if predecessor_archive.read_bytes() != predecessor_bytes:
            raise RuntimeError("archived predecessor contract bytes are inconsistent")
    else:
        atomic_bytes(predecessor_archive, predecessor_bytes)
    normalized_retention = _json_ready(retention)
    if retention_path.is_file():
        if _load_json_object(
            retention_path, description="retained-checkpoint manifest",
        ) != normalized_retention:
            raise RuntimeError("existing retained-checkpoint manifest is inconsistent")
    else:
        atomic_json(retention_path, normalized_retention)
    record_core = {
        "schema": 1,
        "migration_id": authorization.migration_id,
        "authorized_date": authorization.authorized_date,
        "reason": authorization.reason,
        "predecessor": {
            "runner_schema": previous["runner_schema"],
            "source_digest": previous["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": authorization.predecessor_contract_file_digest,
            "archived_contract": predecessor_archive.relative_to(run).as_posix(),
            "archived_contract_digest": file_digest(predecessor_archive),
        },
        "successor": {
            "runner_schema": successor["runner_schema"],
            "source_digest": successor["source_digest"],
            "source_files": dict(new_files),
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(changed)
        },
        "retained_checkpoint_manifest": {
            "path": retention_path.relative_to(run).as_posix(),
            "sha256": file_digest(retention_path),
            "checkpoint_set_digest": retention["checkpoint_set_digest"],
            "retained_accepted_count": retention["retained_accepted_count"],
            "excluded_rejected_count": retention["excluded_rejected_count"],
        },
    }
    if record_path.is_file():
        record = _load_json_object(record_path, description="contract migration record")
        without_time = dict(record)
        without_time.pop("applied_at_utc", None)
        if without_time != record_core or "applied_at_utc" not in record:
            raise RuntimeError("existing contract migration journal is inconsistent")
    else:
        record = {
            **record_core,
            "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(record_path, record)
    history_entry = {
        "migration_id": authorization.migration_id,
        "record": record_path.relative_to(run).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_archive.relative_to(run).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_archive),
        "predecessor_source_digest": previous["source_digest"],
        "successor_source_digest": successor["source_digest"],
    }
    migrated = dict(successor)
    migrated["contract_migrations"] = [history_entry]
    atomic_json(contract_path, migrated)
    _validate_migration_history(run, migrated)


def _validate_retained_assessment_checkpoints(
    run: Path, authorization: AssessmentRecoveryMigrationAuthorization,
) -> dict[str, Any]:
    """Prove generation and ridge checkpoints are complete before carrying them forward."""

    pinned = _validate_pinned_artifacts(run, authorization.required_artifact_digests)
    with np.load(run / "datasets" / "design.npz", allow_pickle=False) as stored:
        design = {name: np.asarray(stored[name]) for name in stored.files}
    design["generators"] = _load_json_object(
        run / "inputs" / "generator_records.json",
        description="generator records",
    )
    validate_design(design, ARTICLE_FULL)
    design_id = _design_digest(design)

    blocks: dict[str, MechanisticBlockResult] = {}
    for block, count in (
        ("development", ARTICLE_FULL.development_count),
        ("test", ARTICLE_FULL.test_count),
    ):
        loaded = _load_generation_checkpoint(
            run, block=block, count=count, profile=ARTICLE_FULL,
            source_id=authorization.predecessor_source_digest,
            design_id=design_id,
        )
        if loaded is None:
            raise RuntimeError(
                f"pinned {block} generation checkpoint cannot be independently loaded"
            )
        blocks[block] = loaded[0]

    effective_path = run / "datasets" / "effective_design.npz"
    with np.load(effective_path, allow_pickle=False) as stored:
        if set(stored.files) != set(DESIGN_ARRAYS):
            raise RuntimeError("pinned effective design has unexpected arrays")
        effective = {name: np.asarray(stored[name]) for name in DESIGN_ARRAYS}
    expected_effective = {
        **{name: np.asarray(design[name]) for name in DESIGN_ARRAYS},
        "development_decisions": blocks["development"].decisions,
        "development_influents": blocks["development"].influents,
        "test_decisions": blocks["test"].decisions,
        "test_influents": blocks["test"].influents,
    }
    if any(
        not np.array_equal(effective[name], expected_effective[name])
        for name in DESIGN_ARRAYS
    ):
        raise RuntimeError("pinned effective design differs from accepted inputs")
    effective_id = _design_digest(effective)
    if effective_id != authorization.expected_effective_design_digest:
        raise RuntimeError("pinned effective-design digest is not authorized")
    effective_manifest_path = run / "datasets" / "effective_design_manifest.json"
    effective_manifest = _load_json_object(
        effective_manifest_path, description="effective design manifest",
    )
    if (
        effective_manifest.get("source_digest")
        != authorization.predecessor_source_digest
        or effective_manifest.get("base_design_digest") != design_id
        or effective_manifest.get("effective_design_digest") != effective_id
        or effective_manifest.get("artifact")
        != {effective_path.relative_to(run).as_posix(): file_digest(effective_path)}
    ):
        raise RuntimeError("pinned effective design manifest is inconsistent")

    ridge_input_id = _ridge_input_digest(
        effective["development_decisions"],
        effective["development_influents"],
        blocks["development"].targets,
    )
    if ridge_input_id != authorization.expected_ridge_input_digest:
        raise RuntimeError("pinned ridge input digest is not authorized")
    ridge = _load_ridge(
        run,
        decisions=effective["development_decisions"],
        influents=effective["development_influents"],
        targets=blocks["development"].targets,
        input_id=ridge_input_id,
        source_id=authorization.predecessor_source_digest,
    )
    if ridge is None:
        raise RuntimeError("pinned ridge checkpoint cannot be independently loaded")

    stage_paths = {
        "generation/development": run / "datasets/development/block_complete.json",
        "generation/test": run / "datasets/test/block_complete.json",
        "generation/effective_design": effective_manifest_path,
        "ridge": run / "models/ridge_complete.json",
    }
    stages: dict[str, Any] = {}
    for stage, path in stage_paths.items():
        payload = _load_json_object(path, description=f"{stage} checkpoint")
        stages[stage] = {
            "checkpoint": path.relative_to(run).as_posix(),
            "checkpoint_sha256": file_digest(path),
            "artifact_source_digest": authorization.predecessor_source_digest,
            "artifacts": dict(payload.get("artifacts", {})),
        }
    stages["generation/effective_design"]["artifacts"] = dict(
        effective_manifest["artifact"]
    )
    generation_summary_path = run / "metrics" / "mechanistic_generation_summary.csv"
    generation_summary = pd.read_csv(generation_summary_path)
    if (
        list(generation_summary.get("block", [])) != ["development", "test"]
        or list(generation_summary.get("accepted_rows", [])) != [4_000, 1_000]
    ):
        raise RuntimeError("pinned generation summary is inconsistent")
    stages["generation/summary"] = {
        "checkpoint": generation_summary_path.relative_to(run).as_posix(),
        "checkpoint_sha256": file_digest(generation_summary_path),
        "artifact_source_digest": authorization.predecessor_source_digest,
        "artifacts": {},
    }
    return {
        "schema": 1,
        "source_digest": authorization.predecessor_source_digest,
        "effective_design_digest": effective_id,
        "ridge_input_digest": ridge_input_id,
        "pinned_artifacts": pinned,
        "stages": stages,
    }


def _migrate_assessment_recovery_contract(
    run: Path,
    previous: Mapping[str, Any],
    successor: Mapping[str, Any],
    authorization: AssessmentRecoveryMigrationAuthorization,
) -> None:
    """Append the audited projection/gate-policy migration to an existing run."""

    contract_path = run / "inputs" / "contract.json"
    history = previous.get("contract_migrations")
    if str(previous.get("run_id")) != authorization.run_id:
        raise RuntimeError("assessment-recovery migration is not authorized for this run id")
    if not isinstance(history, list):
        raise RuntimeError("assessment-recovery migration requires prior migration history")
    history_ids = tuple(str(entry.get("migration_id", "")) for entry in history)
    if history_ids != authorization.required_prior_migration_ids:
        raise RuntimeError("assessment-recovery migration has unexpected prior history")
    _validate_migration_history(run, previous)
    if (
        int(previous.get("runner_schema", -1))
        != authorization.predecessor_runner_schema
        or int(successor.get("runner_schema", -1))
        != authorization.successor_runner_schema
        or previous.get("source_digest") != authorization.predecessor_source_digest
        or file_digest(contract_path) != authorization.predecessor_contract_file_digest
    ):
        raise RuntimeError("existing contract is not the pinned assessment predecessor")
    old_files = previous.get("source_files")
    new_files = successor.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("assessment-recovery migration requires source manifests")
    if source_digest(old_files) != previous.get("source_digest"):
        raise RuntimeError("predecessor source manifest is internally inconsistent")
    if source_digest(new_files) != successor.get("source_digest"):
        raise RuntimeError("successor source manifest is internally inconsistent")
    changed = {
        name for name in set(old_files) | set(new_files)
        if old_files.get(name) != new_files.get(name)
    }
    unauthorized = changed - authorization.allowed_changed_source_files
    missing_required = authorization.required_changed_source_files - changed
    if unauthorized or missing_required:
        raise RuntimeError(
            "assessment-recovery migration refused arbitrary source drift; "
            f"unauthorized={sorted(unauthorized)}, missing_required={sorted(missing_required)}"
        )
    ignored = {
        "runner_schema", "source_digest", "source_files", "contract_migrations",
        "assessment_gate_execution_policy",
    }
    old_invariants = {key: value for key, value in previous.items() if key not in ignored}
    new_invariants = {key: value for key, value in successor.items() if key not in ignored}
    if old_invariants != new_invariants:
        raise RuntimeError("assessment-recovery migration cannot change run/profile data")
    if (
        "assessment_gate_execution_policy" in previous
        or successor.get("assessment_gate_execution_policy")
        != ASSESSMENT_GATE_EXECUTION_POLICY
    ):
        raise RuntimeError("assessment-recovery gate-policy transition is invalid")

    source_snapshot = _source_snapshot_manifest(
        run, authorization.predecessor_source_snapshot, old_files,
    )
    if source_snapshot["source_digest"] != authorization.predecessor_source_digest:
        raise RuntimeError("predecessor source snapshot has the wrong aggregate digest")
    retention = _validate_retained_assessment_checkpoints(run, authorization)
    migrations_directory = run / "inputs" / "contract_migrations"
    predecessor_archive = (
        migrations_directory / f"{authorization.migration_id}-predecessor-contract.json"
    )
    retention_path = migrations_directory / f"{authorization.migration_id}-retained.json"
    record_path = migrations_directory / f"{authorization.migration_id}.json"
    predecessor_bytes = contract_path.read_bytes()
    if sha256(predecessor_bytes).hexdigest() != authorization.predecessor_contract_file_digest:
        raise RuntimeError("predecessor contract bytes changed during migration validation")
    if predecessor_archive.is_file():
        if predecessor_archive.read_bytes() != predecessor_bytes:
            raise RuntimeError("archived predecessor contract bytes are inconsistent")
    else:
        atomic_bytes(predecessor_archive, predecessor_bytes)
    normalized_retention = _json_ready(retention)
    if retention_path.is_file():
        if _load_json_object(
            retention_path, description="retained-stage manifest",
        ) != normalized_retention:
            raise RuntimeError("existing retained-stage manifest is inconsistent")
    else:
        atomic_json(retention_path, normalized_retention)
    record_core = {
        "schema": 1,
        "migration_id": authorization.migration_id,
        "authorized_date": authorization.authorized_date,
        "reason": authorization.reason,
        "predecessor": {
            "runner_schema": previous["runner_schema"],
            "source_digest": previous["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": authorization.predecessor_contract_file_digest,
            "archived_contract": predecessor_archive.relative_to(run).as_posix(),
            "archived_contract_digest": file_digest(predecessor_archive),
        },
        "predecessor_source_snapshot": source_snapshot,
        "successor": {
            "runner_schema": successor["runner_schema"],
            "source_digest": successor["source_digest"],
            "source_files": dict(new_files),
            "assessment_gate_execution_policy": ASSESSMENT_GATE_EXECUTION_POLICY,
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(changed)
        },
        "retained_stage_manifest": {
            "path": retention_path.relative_to(run).as_posix(),
            "sha256": file_digest(retention_path),
            "source_digest": retention["source_digest"],
            "effective_design_digest": retention["effective_design_digest"],
            "ridge_input_digest": retention["ridge_input_digest"],
        },
    }
    if record_path.is_file():
        record = _load_json_object(record_path, description="contract migration record")
        without_time = dict(record)
        without_time.pop("applied_at_utc", None)
        if without_time != record_core or "applied_at_utc" not in record:
            raise RuntimeError("existing assessment migration journal is inconsistent")
    else:
        record = {**record_core, "applied_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(record_path, record)
    history_entry = {
        "migration_id": authorization.migration_id,
        "record": record_path.relative_to(run).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_archive.relative_to(run).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_archive),
        "predecessor_source_digest": previous["source_digest"],
        "successor_source_digest": successor["source_digest"],
    }
    migrated = dict(successor)
    migrated["contract_migrations"] = [*history, history_entry]
    atomic_json(contract_path, migrated)
    _validate_migration_history(run, migrated)


def _validate_retained_optimization_protocol_checkpoints(
    run: Path,
    previous: Mapping[str, Any],
    authorization: OptimizationProtocolMigrationAuthorization,
) -> dict[str, Any]:
    """Prove all completed pre-optimization stages before carrying them forward."""

    pinned = _validate_pinned_artifacts(run, authorization.required_artifact_digests)
    history = previous.get("contract_migrations")
    if not isinstance(history, list) or not history:
        raise RuntimeError("optimization migration requires prior migration history")
    latest = history[-1]
    latest_record = _load_json_object(
        run / str(latest.get("record", "")),
        description="latest predecessor migration record",
    )
    retained_reference = latest_record.get("retained_stage_manifest")
    if not isinstance(retained_reference, Mapping):
        raise RuntimeError("predecessor migration omits its retained-stage manifest")
    retained_path = (run / str(retained_reference.get("path", ""))).resolve()
    migrations_root = (run / "inputs" / "contract_migrations").resolve()
    if (
        migrations_root not in retained_path.parents
        or not retained_path.is_file()
        or file_digest(retained_path) != retained_reference.get("sha256")
    ):
        raise RuntimeError("predecessor retained-stage manifest changed")
    prior_retention = _load_json_object(
        retained_path, description="predecessor retained-stage manifest",
    )
    prior_stages = prior_retention.get("stages")
    if not isinstance(prior_stages, Mapping):
        raise RuntimeError("predecessor retained-stage manifest omits stages")
    required_prior_stages = (
        "generation/development",
        "generation/test",
        "generation/effective_design",
        "generation/summary",
        "ridge",
    )
    stages: dict[str, Any] = {}
    run_root = run.resolve()
    for stage in required_prior_stages:
        stage_record = prior_stages.get(stage)
        if not isinstance(stage_record, Mapping):
            raise RuntimeError(f"predecessor retained manifest omits {stage}")
        checkpoint = (run / str(stage_record.get("checkpoint", ""))).resolve()
        if (
            run_root not in checkpoint.parents
            or not checkpoint.is_file()
            or file_digest(checkpoint) != stage_record.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"retained {stage} checkpoint changed")
        artifacts = stage_record.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(f"retained {stage} artifact manifest is invalid")
        if artifacts and not _artifacts_match(run, artifacts):
            raise RuntimeError(f"retained {stage} artifacts changed")
        artifact_source = str(stage_record.get("artifact_source_digest", ""))
        if not artifact_source:
            raise RuntimeError(f"retained {stage} omits its source binding")
        stages[stage] = dict(stage_record)

    marker_path = run / "metrics" / "assessment_complete.json"
    marker = _load_json_object(marker_path, description="assessment checkpoint")
    marker_artifacts = marker.get("artifacts")
    if (
        marker.get("source_digest") != authorization.predecessor_source_digest
        or marker.get("input_digest")
        != authorization.expected_assessment_input_digest
        or not isinstance(marker.get("passed"), bool)
        or not isinstance(marker_artifacts, Mapping)
        or not _artifacts_match(run, marker_artifacts)
    ):
        raise RuntimeError("completed assessment checkpoint is not safely reusable")
    gate = _load_json_object(
        run / "metrics" / "admission_gate.json", description="assessment gate",
    )
    if (
        not isinstance(gate.get("passed"), bool)
        or gate.get("passed") != marker.get("passed")
        or gate.get("execution_policy") != ASSESSMENT_GATE_EXECUTION_POLICY
        or gate.get("optimization_permitted")
        != assessment_gate_allows_optimization(bool(gate.get("passed")))
    ):
        raise RuntimeError("completed assessment gate is inconsistent")
    stages["assessment"] = {
        "checkpoint": marker_path.relative_to(run).as_posix(),
        "checkpoint_sha256": file_digest(marker_path),
        "artifact_source_digest": authorization.predecessor_source_digest,
        "artifacts": dict(marker_artifacts),
    }
    return {
        "schema": 2,
        "predecessor_source_digest": authorization.predecessor_source_digest,
        "effective_design_digest": prior_retention.get("effective_design_digest"),
        "ridge_input_digest": prior_retention.get("ridge_input_digest"),
        "assessment_input_digest": authorization.expected_assessment_input_digest,
        "pinned_artifacts": pinned,
        "stages": stages,
    }


def _migrate_optimization_protocol_contract(
    run: Path,
    previous: Mapping[str, Any],
    successor: Mapping[str, Any],
    authorization: OptimizationProtocolMigrationAuthorization,
) -> None:
    """Append the single-start exact-QP protocol migration in place."""

    contract_path = run / "inputs" / "contract.json"
    history = previous.get("contract_migrations")
    if str(previous.get("run_id")) != authorization.run_id:
        raise RuntimeError("optimization migration is not authorized for this run id")
    if not isinstance(history, list):
        raise RuntimeError("optimization migration requires prior migration history")
    history_ids = tuple(str(entry.get("migration_id", "")) for entry in history)
    if history_ids != authorization.required_prior_migration_ids:
        raise RuntimeError("optimization migration has unexpected prior history")
    _validate_migration_history(run, previous)
    if (
        int(previous.get("runner_schema", -1))
        != authorization.predecessor_runner_schema
        or int(successor.get("runner_schema", -1))
        != authorization.successor_runner_schema
        or previous.get("source_digest") != authorization.predecessor_source_digest
        or file_digest(contract_path) != authorization.predecessor_contract_file_digest
    ):
        raise RuntimeError("existing contract is not the pinned optimization predecessor")
    old_files = previous.get("source_files")
    new_files = successor.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("optimization migration requires source manifests")
    if source_digest(old_files) != previous.get("source_digest"):
        raise RuntimeError("predecessor source manifest is internally inconsistent")
    if source_digest(new_files) != successor.get("source_digest"):
        raise RuntimeError("successor source manifest is internally inconsistent")
    changed = {
        name for name in set(old_files) | set(new_files)
        if old_files.get(name) != new_files.get(name)
    }
    unauthorized = changed - authorization.allowed_changed_source_files
    missing_required = authorization.required_changed_source_files - changed
    if unauthorized or missing_required:
        raise RuntimeError(
            "optimization migration refused arbitrary source drift; "
            f"unauthorized={sorted(unauthorized)}, missing_required={sorted(missing_required)}"
        )
    ignored = {
        "runner_schema", "source_digest", "source_files", "contract_migrations",
        "optimization_protocol",
    }
    old_invariants = {key: value for key, value in previous.items() if key not in ignored}
    new_invariants = {key: value for key, value in successor.items() if key not in ignored}
    if old_invariants != new_invariants:
        raise RuntimeError("optimization migration cannot change run/profile data")
    if (
        "optimization_protocol" in previous
        or successor.get("optimization_protocol") != OPTIMIZATION_PROTOCOL
    ):
        raise RuntimeError("optimization-protocol transition is invalid")

    source_snapshot = _source_snapshot_manifest(
        run, authorization.predecessor_source_snapshot, old_files,
    )
    if source_snapshot["source_digest"] != authorization.predecessor_source_digest:
        raise RuntimeError("predecessor source snapshot has the wrong aggregate digest")
    retention = _validate_retained_optimization_protocol_checkpoints(
        run, previous, authorization,
    )
    migrations_directory = run / "inputs" / "contract_migrations"
    predecessor_archive = (
        migrations_directory / f"{authorization.migration_id}-predecessor-contract.json"
    )
    retention_path = migrations_directory / f"{authorization.migration_id}-retained.json"
    record_path = migrations_directory / f"{authorization.migration_id}.json"
    predecessor_bytes = contract_path.read_bytes()
    if sha256(predecessor_bytes).hexdigest() != authorization.predecessor_contract_file_digest:
        raise RuntimeError("predecessor contract bytes changed during migration validation")
    if predecessor_archive.is_file():
        if predecessor_archive.read_bytes() != predecessor_bytes:
            raise RuntimeError("archived predecessor contract bytes are inconsistent")
    else:
        atomic_bytes(predecessor_archive, predecessor_bytes)
    normalized_retention = _json_ready(retention)
    if retention_path.is_file():
        if _load_json_object(
            retention_path, description="retained-stage manifest",
        ) != normalized_retention:
            raise RuntimeError("existing optimization retention manifest is inconsistent")
    else:
        atomic_json(retention_path, normalized_retention)
    record_core = {
        "schema": 1,
        "migration_id": authorization.migration_id,
        "authorized_date": authorization.authorized_date,
        "reason": authorization.reason,
        "predecessor": {
            "runner_schema": previous["runner_schema"],
            "source_digest": previous["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": authorization.predecessor_contract_file_digest,
            "archived_contract": predecessor_archive.relative_to(run).as_posix(),
            "archived_contract_digest": file_digest(predecessor_archive),
        },
        "predecessor_source_snapshot": source_snapshot,
        "successor": {
            "runner_schema": successor["runner_schema"],
            "source_digest": successor["source_digest"],
            "source_files": dict(new_files),
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(changed)
        },
        "retained_stage_manifest": {
            "path": retention_path.relative_to(run).as_posix(),
            "sha256": file_digest(retention_path),
            "predecessor_source_digest": retention["predecessor_source_digest"],
            "assessment_input_digest": retention["assessment_input_digest"],
        },
    }
    if record_path.is_file():
        record = _load_json_object(record_path, description="contract migration record")
        without_time = dict(record)
        without_time.pop("applied_at_utc", None)
        if without_time != record_core or "applied_at_utc" not in record:
            raise RuntimeError("existing optimization migration journal is inconsistent")
    else:
        record = {**record_core, "applied_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(record_path, record)
    history_entry = {
        "migration_id": authorization.migration_id,
        "record": record_path.relative_to(run).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_archive.relative_to(run).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_archive),
        "predecessor_source_digest": previous["source_digest"],
        "successor_source_digest": successor["source_digest"],
    }
    migrated = dict(successor)
    migrated["contract_migrations"] = [*history, history_entry]
    atomic_json(contract_path, migrated)
    _validate_migration_history(run, migrated)


def establish_contract(
    run: Path,
    contract: Mapping[str, Any],
    *,
    authorize_generation_replacement_migration: bool = False,
    authorize_assessment_recovery_migration: bool = False,
    authorize_single_start_exact_qp_migration: bool = False,
) -> None:
    if sum(map(bool, (
        authorize_generation_replacement_migration,
        authorize_assessment_recovery_migration,
        authorize_single_start_exact_qp_migration,
    ))) > 1:
        raise ValueError("authorize exactly one source-contract migration at a time")
    path = run / "inputs" / "contract.json"
    normalized = _json_ready(contract)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous == normalized:
            return
        previous_without_history = dict(previous)
        previous_without_history.pop("contract_migrations", None)
        if previous_without_history == normalized:
            _validate_migration_history(run, previous)
            return
        if authorize_generation_replacement_migration:
            _migrate_source_contract(
                run, previous, normalized, GENERATION_REPLACEMENT_MIGRATION,
            )
            return
        if authorize_assessment_recovery_migration:
            _migrate_assessment_recovery_contract(
                run, previous, normalized, ASSESSMENT_RECOVERY_MIGRATION,
            )
            return
        if authorize_single_start_exact_qp_migration:
            _migrate_optimization_protocol_contract(
                run, previous, normalized, SINGLE_START_EXACT_QP_MIGRATION,
            )
            return
        raise RuntimeError(
            "existing full-run contract differs; choose a new run id or provide the "
            "explicit authorized migration flag"
        )
    else:
        atomic_json(path, normalized)


def _checkpoint_source_is_authorized(
    run: Path,
    *,
    stage: str,
    checkpoint: Path,
    observed_source_id: str,
    current_source_id: str,
) -> bool:
    """Allow a carried-forward checkpoint only through its exact migration pin."""

    if observed_source_id == current_source_id:
        return True
    try:
        contract = _load_json_object(
            run / "inputs" / "contract.json", description="run contract",
        )
        if contract.get("source_digest") != current_source_id:
            return False
        _validate_migration_history(run, contract)
        history = contract["contract_migrations"]
        latest = history[-1]
        record = _load_json_object(
            run / str(latest["record"]), description="latest migration record",
        )
        if record.get("successor", {}).get("source_digest") != current_source_id:
            return False
        retained_reference = record.get("retained_stage_manifest")
        if not isinstance(retained_reference, Mapping):
            return False
        retained_path = (run / str(retained_reference.get("path", ""))).resolve()
        migrations_root = (run / "inputs" / "contract_migrations").resolve()
        if (
            migrations_root not in retained_path.parents
            or not retained_path.is_file()
            or file_digest(retained_path) != retained_reference.get("sha256")
        ):
            return False
        retained = _load_json_object(
            retained_path, description="retained-stage manifest",
        )
        stage_record = retained.get("stages", {}).get(stage)
        if not isinstance(stage_record, Mapping):
            return False
        relative = checkpoint.resolve().relative_to(run.resolve()).as_posix()
        return bool(
            stage_record.get("artifact_source_digest") == observed_source_id
            and stage_record.get("checkpoint") == relative
            and stage_record.get("checkpoint_sha256") == file_digest(checkpoint)
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


def assert_source_unchanged(expected: Mapping[str, str]) -> None:
    current = source_file_digests()
    if current != dict(expected):
        changed = sorted(set(current) | set(expected))
        changed = [name for name in changed if current.get(name) != expected.get(name)]
        raise RuntimeError(
            "article source changed during the run; do not mix artifacts. "
            f"Changed files: {', '.join(changed)}"
        )


def _validate_generation_block(
    targets: np.ndarray,
    diagnostics: pd.DataFrame,
    *,
    block: str,
    count: int,
    profile: StudyProfile,
) -> None:
    if targets.shape != (count, profile.response_count) or not np.all(np.isfinite(targets)):
        raise RuntimeError(f"{block} target block is incomplete or non-finite")
    required = {
        "row", "accepted", "root_difference_inf", "branch_agreement",
        "mass_residual_start_1", "mass_residual_start_2",
        "state_negativity_start_1", "state_negativity_start_2",
        "rate_negativity_start_1", "rate_negativity_start_2",
        "largest_real_eigenvalue_start_1", "largest_real_eigenvalue_start_2",
        "stability_agreement_start_1", "stability_agreement_start_2",
        "feed_tss_start_1", "feed_tss_start_2",
        "external_solids_loss_start_1", "external_solids_loss_start_2",
    }
    missing = required - set(diagnostics.columns)
    if missing:
        raise RuntimeError(f"{block} diagnostics omit columns: {sorted(missing)}")
    rows = np.asarray(diagnostics["row"], dtype=int)
    if len(diagnostics) != count or not np.array_equal(np.sort(rows), np.arange(count)):
        raise RuntimeError(f"{block} diagnostics do not cover every fixed row exactly once")
    accepted = diagnostics["accepted"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    branches = diagnostics["branch_agreement"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if accepted.isna().any() or not bool(accepted.all()):
        raise RuntimeError(f"{block} contains an unaccepted fixed mechanistic row")
    if branches.isna().any() or not bool(branches.all()):
        raise RuntimeError(f"{block} contains a two-start branch disagreement")
    numeric_columns = list(required - {"row", "accepted", "branch_agreement"})
    numeric = diagnostics[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.all(np.isfinite(numeric.to_numpy())):
        raise RuntimeError(f"{block} mechanistic audits contain non-finite values")
    checks = (
        ("root_difference_inf", np.less_equal, 1.0e-6),
        ("mass_residual_start_1", np.less_equal, 1.0e-8),
        ("mass_residual_start_2", np.less_equal, 1.0e-8),
        ("state_negativity_start_1", np.less_equal, 1.0e-10),
        ("state_negativity_start_2", np.less_equal, 1.0e-10),
        ("rate_negativity_start_1", np.less_equal, 1.0e-12),
        ("rate_negativity_start_2", np.less_equal, 1.0e-12),
        ("largest_real_eigenvalue_start_1", np.less_equal, -1.0e-8),
        ("largest_real_eigenvalue_start_2", np.less_equal, -1.0e-8),
        ("stability_agreement_start_1", np.less_equal, 1.0e-6),
        ("stability_agreement_start_2", np.less_equal, 1.0e-6),
        ("feed_tss_start_1", np.greater_equal, 1.0),
        ("feed_tss_start_2", np.greater_equal, 1.0),
        ("external_solids_loss_start_1", np.greater_equal, 1.0),
        ("external_solids_loss_start_2", np.greater_equal, 1.0),
    )
    failures = [name for name, comparison, limit in checks if not bool(
        np.all(comparison(numeric[name].to_numpy(), limit))
    )]
    if failures:
        raise RuntimeError(f"{block} failed mechanistic generation gates: {failures}")


def _load_generation_checkpoint(
    run: Path,
    *,
    block: str,
    count: int,
    profile: StudyProfile,
    source_id: str,
    design_id: str,
) -> tuple[MechanisticBlockResult, float] | None:
    marker_path = run / "datasets" / block / "block_complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            not _checkpoint_source_is_authorized(
                run, stage=f"generation/{block}", checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("design_digest") != design_id
            or marker.get("block") != block
            or int(marker.get("row_count", -1)) != count
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return None
        output = run / "datasets" / block
        target_path = output / "mechanistic_accepted_v3.npz"
        input_path = output / "accepted_inputs.npz"
        with np.load(target_path, allow_pickle=False) as stored:
            targets = np.asarray(stored["targets"], dtype=float)
        with np.load(input_path, allow_pickle=False) as stored:
            decisions = np.asarray(stored["decisions"], dtype=float)
            influents = np.asarray(stored["influents"], dtype=float)
            source_candidate_id = np.asarray(stored["source_candidate_id"], dtype=str)
        diagnostics = pd.read_csv(output / "accepted_diagnostics.csv")
        attempts = pd.read_csv(output / "all_attempts.csv")
        provenance = pd.read_csv(output / "accepted_provenance.csv")
        _validate_generation_block(
            targets, diagnostics, block=block, count=count, profile=profile,
        )
        if (
            decisions.shape != (count, 7)
            or influents.shape != (count, 20)
            or not np.all(np.isfinite(decisions))
            or not np.all(np.isfinite(influents))
            or np.any(decisions < DECISION_LOWER)
            or np.any(decisions > DECISION_UPPER)
            or np.any(influents < INFLUENT_LOWER)
            or np.any(influents > INFLUENT_UPPER)
            or len(provenance) != count
            or len(source_candidate_id) != count
            or not np.array_equal(
                source_candidate_id,
                provenance["source_candidate_id"].to_numpy(dtype=str),
            )
            or marker.get("effective_input_digest")
            != array_digest(
                decisions=np.asarray(decisions, dtype="<f8"),
                influents=np.asarray(influents, dtype="<f8"),
            )
        ):
            return None
        _validate_attempt_checkpoint_hashes(output, attempts)
        return MechanisticBlockResult(
            decisions=decisions, influents=influents, targets=targets,
            diagnostics=diagnostics, attempts=attempts, provenance=provenance,
        ), float(marker["elapsed_seconds"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _validate_attempt_checkpoint_hashes(
    output: Path, attempts: pd.DataFrame,
) -> None:
    required = {"checkpoint_path", "checkpoint_sha256", "candidate_id", "accepted"}
    if required - set(attempts.columns) or attempts.empty:
        raise RuntimeError("generation attempt ledger is incomplete")
    if attempts["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("generation attempt ledger contains duplicate candidates")
    root = output.resolve()
    for row in attempts.itertuples(index=False):
        path = (output / str(row.checkpoint_path)).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError("generation attempt checkpoint is missing or outside its block")
        if file_digest(path) != str(row.checkpoint_sha256):
            raise RuntimeError(f"generation attempt checkpoint changed: {path.name}")


def _boolean_series(values: pd.Series, *, description: str) -> pd.Series:
    converted = values.astype(str).str.lower().map({"true": True, "false": False})
    if converted.isna().any():
        raise RuntimeError(f"{description} contains invalid Boolean values")
    return converted.astype(bool)


def _write_generation_audits(
    output: Path, result: MechanisticBlockResult,
) -> None:
    physical = np.column_stack((result.decisions, result.influents))
    lower = np.concatenate((DECISION_LOWER, INFLUENT_LOWER))
    upper = np.concatenate((DECISION_UPPER, INFLUENT_UPPER))
    names = tuple(DECISION_NAMES) + tuple(COMPONENTS)
    coverage = pd.DataFrame([
        {
            "coordinate_index": index,
            "coordinate_group": "decision" if index < 7 else "influent",
            "coordinate": names[index],
            "declared_lower": lower[index],
            "declared_upper": upper[index],
            "accepted_minimum": float(np.min(physical[:, index])),
            "accepted_maximum": float(np.max(physical[:, index])),
            "accepted_span_fraction": float(
                np.ptp(physical[:, index]) / (upper[index] - lower[index])
            ),
            "outside_declared_box_count": int(np.count_nonzero(
                (physical[:, index] < lower[index])
                | (physical[:, index] > upper[index])
            )),
        }
        for index in range(physical.shape[1])
    ])
    attempts = result.attempts.copy()
    if "rejection_reason" not in attempts.columns:
        raise RuntimeError("generation attempt ledger omits rejection_reason")
    accepted = _boolean_series(attempts["accepted"], description="attempt ledger")
    reasons = attempts["rejection_reason"].astype(str)
    if bool(((accepted & reasons.ne("accepted")) | (~accepted & reasons.eq("accepted"))).any()):
        raise RuntimeError("generation rejection reasons disagree with acceptance flags")
    rejected = attempts.loc[attempts["rejection_reason"] != "accepted"]
    if rejected.empty:
        rejection_summary = pd.DataFrame(columns=(
            "rejection_reason", "attempt_count", "fraction_of_all_attempts",
        ))
    else:
        rejection_summary = (
            rejected.groupby("rejection_reason", dropna=False).size()
            .rename("attempt_count").reset_index()
            .sort_values("rejection_reason", kind="stable").reset_index(drop=True)
        )
        rejection_summary["fraction_of_all_attempts"] = (
            rejection_summary["attempt_count"] / len(attempts)
        )
    atomic_dataframe(output / "accepted_coordinate_coverage.csv", coverage)
    atomic_dataframe(output / "rejection_reason_summary.csv", rejection_summary)


def _generation_publication_paths(output: Path) -> tuple[Path, ...]:
    names = (
        "mechanistic_accepted_v3.npz", "accepted_inputs.npz",
        "accepted_diagnostics.csv", "all_attempts.csv",
        "accepted_provenance.csv", "base_checkpoint_migration.csv",
        "replacement_summary.json", "accepted_coordinate_coverage.csv",
        "rejection_reason_summary.csv",
    )
    fixed = tuple(output / name for name in names)
    summary_path = output / "replacement_summary.json"
    summary = _load_json_object(summary_path, description="replacement summary")
    round_count = int(summary.get("supplemental_round_count", -1))
    if round_count < 0:
        raise RuntimeError("replacement summary has an invalid round count")
    expected_manifests = tuple(
        output / "attempts" / "replacement" / f"round_{index:06d}" / "manifest.json"
        for index in range(1, round_count + 1)
    )
    actual_manifests = tuple(sorted(
        (output / "attempts" / "replacement").glob("round_*/manifest.json")
    ))
    if actual_manifests != expected_manifests:
        raise RuntimeError("supplemental round-manifest sequence is incomplete or unexpected")
    return fixed + expected_manifests


def _run_generation_block(
    run: Path,
    design: Mapping[str, object],
    *,
    block: str,
    profile: StudyProfile,
    source_files: Mapping[str, str],
    design_id: str,
) -> tuple[MechanisticBlockResult, float, bool]:
    count = profile.development_count if block == "development" else profile.test_count
    source_id = source_digest(source_files)
    checkpoint = _load_generation_checkpoint(
        run, block=block, count=count, profile=profile,
        source_id=source_id, design_id=design_id,
    )
    if checkpoint is not None:
        return (*checkpoint, True)
    started = perf_counter()
    result = generate_mechanistic_block_with_replacements(
        np.asarray(design[f"{block}_decisions"]),
        np.asarray(design[f"{block}_influents"]),
        profile,
        run / "datasets" / block,
        block=block,
    )
    elapsed = perf_counter() - started
    _validate_generation_block(
        result.targets, result.diagnostics,
        block=block, count=count, profile=profile,
    )
    _validate_attempt_checkpoint_hashes(run / "datasets" / block, result.attempts)
    _write_generation_audits(run / "datasets" / block, result)
    assert_source_unchanged(source_files)
    paths = _generation_publication_paths(run / "datasets" / block)
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"{block} generator did not publish its required artifacts")
    atomic_json(run / "datasets" / block / "block_complete.json", {
        "stage": "mechanistic_generation",
        "block": block,
        "source_digest": source_id,
        "design_digest": design_id,
        "row_count": count,
        "accepted_count": count,
        "target_shape": list(result.targets.shape),
        "attempt_count": len(result.attempts),
        "rejected_attempt_count": int(
            (~_boolean_series(
                result.attempts["accepted"], description="attempt ledger",
            )).sum()
        ),
        "replacement_slot_count": int(
            _boolean_series(
                result.provenance["replaced_base_candidate"],
                description="provenance ledger",
            ).sum()
        ),
        "effective_input_digest": array_digest(
            decisions=np.asarray(result.decisions, dtype="<f8"),
            influents=np.asarray(result.influents, dtype="<f8"),
        ),
        "elapsed_seconds": elapsed,
        "artifacts": _artifact_hashes(run, paths),
    })
    return result, elapsed, False


def run_generation(
    run: Path,
    design: Mapping[str, object],
    *,
    profile: StudyProfile,
    source_files: Mapping[str, str],
) -> GenerationResult:
    design_id = _design_digest(design)
    blocks: dict[str, tuple[MechanisticBlockResult, float, bool]] = {}
    for block in ("development", "test"):
        blocks[block] = _run_generation_block(
            run, design, block=block, profile=profile,
            source_files=source_files, design_id=design_id,
        )
    summary = pd.DataFrame([
        {
            "block": block,
            "fixed_candidate_rows": len(result[0].diagnostics),
            "required_accepted_rows": len(result[0].diagnostics),
            "accepted_rows": int(result[0].diagnostics["accepted"].astype(
                str
            ).str.lower().eq("true").sum()),
            "total_attempts": len(result[0].attempts),
            "excluded_rejected_attempts": int(
                (~_boolean_series(
                    result[0].attempts["accepted"], description="attempt ledger",
                )).sum()
            ),
            "replacement_slots": int(
                _boolean_series(
                    result[0].provenance["replaced_base_candidate"],
                    description="provenance ledger",
                ).sum()
            ),
            "elapsed_seconds": result[1],
            "reused_complete_checkpoint": result[2],
        }
        for block, result in blocks.items()
    ])
    summary_path = run / "metrics" / "mechanistic_generation_summary.csv"
    source_id = source_digest(source_files)
    marker_source_ids = {
        str(_load_json_object(
            run / "datasets" / block / "block_complete.json",
            description=f"{block} generation marker",
        ).get("source_digest", ""))
        for block in blocks
    }
    carried_forward = bool(
        all(result[2] for result in blocks.values())
        and marker_source_ids != {source_id}
    )
    if carried_forward:
        if len(marker_source_ids) != 1 or not _checkpoint_source_is_authorized(
            run, stage="generation/summary", checkpoint=summary_path,
            observed_source_id=next(iter(marker_source_ids)),
            current_source_id=source_id,
        ):
            raise RuntimeError("generation summary is not authorized for carry-forward")
    else:
        atomic_dataframe(summary_path, summary)
    effective_design = dict(design)
    for block, result in blocks.items():
        effective_design[f"{block}_decisions"] = result[0].decisions
        effective_design[f"{block}_influents"] = result[0].influents
    effective_path = run / "datasets" / "effective_design.npz"
    effective_arrays = {
        name: np.asarray(effective_design[name]) for name in DESIGN_ARRAYS
    }
    if effective_path.is_file():
        with np.load(effective_path, allow_pickle=False) as stored:
            if set(stored.files) != set(DESIGN_ARRAYS) or any(
                not np.array_equal(stored[name], effective_arrays[name])
                for name in DESIGN_ARRAYS
            ):
                raise RuntimeError("existing effective design differs from accepted inputs")
    else:
        atomic_npz(effective_path, **effective_arrays)
    effective_id = _design_digest(effective_design)
    effective_manifest = {
        "schema": 1,
        "base_design_digest": design_id,
        "effective_design_digest": effective_id,
        "source_digest": source_id,
        "artifact": {
            effective_path.relative_to(run).as_posix(): file_digest(effective_path),
        },
        "blocks": {
            block: {
                "accepted_input_artifact": (
                    run / "datasets" / block / "accepted_inputs.npz"
                ).relative_to(run).as_posix(),
                "accepted_input_artifact_sha256": file_digest(
                    run / "datasets" / block / "accepted_inputs.npz"
                ),
                "replacement_slots": int(
                    _boolean_series(
                        result[0].provenance["replaced_base_candidate"],
                        description="provenance ledger",
                    ).sum()
                ),
            }
            for block, result in blocks.items()
        },
    }
    effective_manifest_path = run / "datasets" / "effective_design_manifest.json"
    if effective_manifest_path.is_file():
        existing_manifest = _load_json_object(
            effective_manifest_path, description="effective design manifest",
        )
        expected_manifest = _json_ready(effective_manifest)
        existing_source_id = str(existing_manifest.get("source_digest", ""))
        existing_without_source = dict(existing_manifest)
        expected_without_source = dict(expected_manifest)
        existing_without_source.pop("source_digest", None)
        expected_without_source.pop("source_digest", None)
        if (
            existing_without_source != expected_without_source
            or not _checkpoint_source_is_authorized(
                run, stage="generation/effective_design",
                checkpoint=effective_manifest_path,
                observed_source_id=existing_source_id,
                current_source_id=source_id,
            )
        ):
            raise RuntimeError("existing effective design manifest differs")
    else:
        atomic_json(effective_manifest_path, effective_manifest)
    assert_source_unchanged(source_files)
    return GenerationResult(
        design=effective_design,
        development_targets=blocks["development"][0].targets,
        test_targets=blocks["test"][0].targets,
    )


def _ridge_input_digest(
    decisions: np.ndarray, influents: np.ndarray, targets: np.ndarray,
) -> str:
    return array_digest(
        development_decisions=np.asarray(decisions, dtype="<f8"),
        development_influents=np.asarray(influents, dtype="<f8"),
        development_targets=np.asarray(targets, dtype="<f8"),
    )


def _validate_ridge_scores(
    scores: pd.DataFrame, fold_membership: np.ndarray, row_count: int,
) -> None:
    required = {"fold", "gamma", "raw_nrmse", "selected"}
    if required - set(scores.columns):
        raise RuntimeError("ridge score checkpoint omits required columns")
    if len(scores) != 5 * len(RIDGE_GRID):
        raise RuntimeError("ridge score checkpoint does not contain the full 5-fold grid")
    if set(np.asarray(scores["fold"], dtype=int)) != {1, 2, 3, 4, 5}:
        raise RuntimeError("ridge score checkpoint has invalid fold identifiers")
    gamma = np.asarray(scores["gamma"], dtype=float)
    nrmse = np.asarray(scores["raw_nrmse"], dtype=float)
    if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(nrmse)):
        raise RuntimeError("ridge score checkpoint contains non-finite values")
    for candidate in RIDGE_GRID:
        if np.count_nonzero(np.isclose(gamma, candidate, rtol=1e-12, atol=0.0)) != 5:
            raise RuntimeError("ridge score checkpoint does not cover each penalty five times")
    selected = scores["selected"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if selected.isna().any() or int(selected.sum()) != 5:
        raise RuntimeError("ridge score checkpoint has an invalid selected penalty")
    chosen = gamma[selected.to_numpy()]
    if not np.allclose(chosen, chosen[0], rtol=0.0, atol=0.0):
        raise RuntimeError("ridge score checkpoint selects more than one penalty")
    membership = np.asarray(fold_membership, dtype=int)
    if membership.shape != (row_count,) or set(membership) != {1, 2, 3, 4, 5}:
        raise RuntimeError("ridge fold membership checkpoint is invalid")
    counts = np.bincount(membership, minlength=6)[1:]
    if int(counts.max() - counts.min()) > 1:
        raise RuntimeError("ridge folds do not form the declared balanced partition")


def save_ridge(
    run: Path, result: Any, *, input_id: str, source_id: str,
) -> None:
    model = result.model
    scores_path = run / "metrics" / "ridge_cross_validation.csv"
    fold_path = run / "metrics" / "ridge_fold_membership.csv"
    bundle_path = run / "models" / "ridge_surrogate.npz"
    atomic_dataframe(scores_path, result.scores)
    atomic_dataframe(fold_path, pd.DataFrame({
        "row": np.arange(len(result.fold_membership)),
        "fold": result.fold_membership,
    }))
    atomic_npz(
        bundle_path,
        input_digest=np.asarray(input_id),
        source_digest=np.asarray(source_id),
        decision_center=model.feature_map.decision_center,
        decision_scale=model.feature_map.decision_scale,
        influent_center=model.feature_map.influent_center,
        influent_scale=model.feature_map.influent_scale,
        term_center=model.feature_map.term_center,
        term_scale=model.feature_map.term_scale,
        variance_relative_tolerance=np.asarray(
            model.feature_map.variance_relative_tolerance
        ),
        response_center=model.response_center,
        response_scale=model.response_scale,
        coefficients=model.coefficients,
        ridge_penalty=np.asarray(model.ridge_penalty),
        diagnostics_json=np.asarray(json.dumps(asdict(model.diagnostics), sort_keys=True)),
        fold_membership=np.asarray(result.fold_membership, dtype=int),
        out_of_fold_raw=result.out_of_fold_raw,
        elapsed_seconds=np.asarray(result.elapsed_seconds),
    )
    paths = (scores_path, fold_path, bundle_path)
    atomic_json(run / "models" / "ridge_complete.json", {
        "stage": "ridge_cross_validation",
        "source_digest": source_id,
        "input_digest": input_id,
        "selected_penalty": model.ridge_penalty,
        "artifacts": _artifact_hashes(run, paths),
    })


def _load_ridge(
    run: Path,
    *,
    decisions: np.ndarray,
    influents: np.ndarray,
    targets: np.ndarray,
    input_id: str,
    source_id: str,
) -> tuple[QuadraticSurrogate, np.ndarray] | None:
    marker_path = run / "models" / "ridge_complete.json"
    bundle_path = run / "models" / "ridge_surrogate.npz"
    scores_path = run / "metrics" / "ridge_cross_validation.csv"
    fold_path = run / "metrics" / "ridge_fold_membership.csv"
    if not all(path.is_file() for path in (marker_path, bundle_path, scores_path, fold_path)):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            not _checkpoint_source_is_authorized(
                run, stage="ridge", checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("input_digest") != input_id
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return None
        with np.load(bundle_path, allow_pickle=False) as stored:
            if (
                str(stored["input_digest"].item()) != input_id
                or str(stored["source_digest"].item()) != marker_source_id
            ):
                return None
            feature_map = QuadraticFeatureMap(
                decision_center=np.asarray(stored["decision_center"], dtype=float),
                decision_scale=np.asarray(stored["decision_scale"], dtype=float),
                influent_center=np.asarray(stored["influent_center"], dtype=float),
                influent_scale=np.asarray(stored["influent_scale"], dtype=float),
                term_center=np.asarray(stored["term_center"], dtype=float),
                term_scale=np.asarray(stored["term_scale"], dtype=float),
                variance_relative_tolerance=float(stored["variance_relative_tolerance"]),
            )
            diagnostics = LeastSquaresDiagnostics(**json.loads(
                str(stored["diagnostics_json"].item())
            ))
            model = QuadraticSurrogate(
                feature_map=feature_map,
                response_center=np.asarray(stored["response_center"], dtype=float),
                response_scale=np.asarray(stored["response_scale"], dtype=float),
                coefficients=np.asarray(stored["coefficients"], dtype=float),
                diagnostics=diagnostics,
                ridge_penalty=float(stored["ridge_penalty"]),
            )
            oof = np.asarray(stored["out_of_fold_raw"], dtype=float)
            membership = np.asarray(stored["fold_membership"], dtype=int)
        expected_features = QuadraticFeatureMap.expected_feature_count(
            decisions.shape[1], influents.shape[1]
        )
        if (
            feature_map.decision_count != decisions.shape[1]
            or feature_map.influent_count != influents.shape[1]
            or feature_map.feature_count != expected_features
            or model.response_center.shape != (targets.shape[1],)
            or model.response_scale.shape != (targets.shape[1],)
            or model.coefficients.shape != (targets.shape[1], expected_features)
            or oof.shape != targets.shape
            or not np.all(np.isfinite(oof))
            or not all(np.all(np.isfinite(value)) for value in (
                feature_map.decision_center,
                feature_map.decision_scale,
                feature_map.influent_center,
                feature_map.influent_scale,
                feature_map.term_center,
                feature_map.term_scale,
                model.response_center,
                model.response_scale,
                model.coefficients,
            ))
            or not np.all(model.response_scale > 0.0)
            or not np.any(np.isclose(
                model.ridge_penalty, RIDGE_GRID, rtol=1e-12, atol=0.0,
            ))
        ):
            return None
        scores = pd.read_csv(scores_path)
        fold_frame = pd.read_csv(fold_path)
        if not np.array_equal(fold_frame["row"].to_numpy(), np.arange(len(decisions))):
            return None
        if not np.array_equal(fold_frame["fold"].to_numpy(dtype=int), membership):
            return None
        _validate_ridge_scores(scores, membership, len(decisions))
        return model, oof
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def fit_or_resume_ridge(
    run: Path,
    decisions: np.ndarray,
    influents: np.ndarray,
    targets: np.ndarray,
    *,
    source_files: Mapping[str, str],
) -> tuple[QuadraticSurrogate, np.ndarray, str]:
    input_id = _ridge_input_digest(decisions, influents, targets)
    source_id = source_digest(source_files)
    resumed = _load_ridge(
        run, decisions=decisions, influents=influents, targets=targets,
        input_id=input_id, source_id=source_id,
    )
    if resumed is not None:
        return resumed[0], resumed[1], input_id
    result = cross_validate_ridge(decisions, influents, targets)
    _validate_ridge_scores(result.scores, result.fold_membership, len(decisions))
    assert_source_unchanged(source_files)
    save_ridge(run, result, input_id=input_id, source_id=source_id)
    return result.model, result.out_of_fold_raw, input_id


def _validate_assessment(
    assessment: AssessmentResult, *, test_count: int, response_count: int,
) -> None:
    arrays = (assessment.raw, assessment.projected, assessment.projected_targets)
    if any(
        np.asarray(value).shape != (test_count, response_count)
        or not np.all(np.isfinite(value))
        for value in arrays
    ):
        raise RuntimeError("untouched-test predictions are incomplete or non-finite")
    qp = assessment.qp_diagnostics
    if len(qp) != 2 * test_count or set(qp["projection_input"]) != {
        "raw_prediction", "mechanistic_target",
    }:
        raise RuntimeError("projection audit does not cover both inputs for every test row")
    for kind in ("raw_prediction", "mechanistic_target"):
        subset = qp.loc[qp["projection_input"].eq(kind)]
        if len(subset) != test_count or set(np.asarray(subset["row"], dtype=int)) != set(
            range(test_count)
        ):
            raise RuntimeError(f"projection audit coverage is invalid for {kind}")
    violations = assessment.violations
    if len(violations) != 3 * test_count:
        raise RuntimeError("physical audit does not cover raw/projected/mechanistic results")
    for method in ("raw", "projected", "mechanistic"):
        if int(violations["method"].eq(method).sum()) != test_count:
            raise RuntimeError(f"physical audit coverage is invalid for {method}")
    if len(assessment.feasibility) != test_count:
        raise RuntimeError("finite-distance projection bound does not cover every test row")


def assessment_gate_allows_optimization(passed: bool) -> bool:
    """Return whether the recorded gate outcome permits downstream execution."""

    return bool(passed or ASSESSMENT_GATE_EXECUTION_POLICY == "advisory_continue")


def evaluate_admission_gate(
    assessment: AssessmentResult,
    *,
    correction_limit: float,
    trust_limits: Mapping[str, float],
    development_oof_projection_accepted: np.ndarray,
    test_count: int,
) -> dict[str, Any]:
    _validate_assessment(
        assessment, test_count=test_count, response_count=assessment.raw.shape[1],
    )
    complete = assessment.metrics.loc[
        assessment.metrics["method"].eq("raw")
        & assessment.metrics["block"].eq("complete_response")
        & assessment.metrics["coordinate"].eq("ALL")
    ]
    if len(complete) != 1:
        raise RuntimeError("complete-response raw assessment row is missing or duplicated")
    raw_nrmse = float(complete.iloc[0]["nrmse"])
    if not np.isfinite(raw_nrmse):
        raise RuntimeError("complete-response raw nRMSE is non-finite")
    qp_accepted = assessment.qp_diagnostics["accepted"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    feasibility = assessment.feasibility["bound_passed"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if qp_accepted.isna().any() or feasibility.isna().any():
        raise RuntimeError("projection acceptance columns are not Boolean")
    violations = assessment.violations
    physical: dict[str, dict[str, float | bool]] = {}
    for method in ("raw", "projected", "mechanistic"):
        rows = violations.loc[violations["method"].eq(method)]
        mass = float(rows["mass_conservation_violation_max"].max())
        negative = float(rows["nonnegativity_violation_max"].max())
        if not np.isfinite(mass) or not np.isfinite(negative):
            raise RuntimeError(f"{method} physical audit contains non-finite gate values")
        physical[method] = {
            "mass_conservation_violation_max": mass,
            "nonnegativity_violation_max": negative,
            "passed": bool(mass <= 1.0e-8 and negative <= 1.0e-10),
        }
    limits = {name: float(value) for name, value in trust_limits.items()}
    if set(limits) != {
        "correction", "regularized_leverage", "particulate_split",
        "reactor_residual", "clarifier_flux",
    } or not all(np.isfinite(value) and value >= 0.0 for value in limits.values()):
        raise RuntimeError("five finite, nonnegative trust limits were not frozen")
    if not np.isclose(
        limits["correction"], correction_limit, rtol=0.0, atol=0.0,
    ):
        raise RuntimeError("the correction gate and frozen trust limit disagree")
    oof_projection_accepted = np.asarray(development_oof_projection_accepted)
    if (
        oof_projection_accepted.ndim != 1
        or oof_projection_accepted.size < 1
        or oof_projection_accepted.dtype.kind != "b"
    ):
        raise RuntimeError(
            "development OOF projection acceptance must be a nonempty Boolean vector"
        )
    checks = {
        "raw_nrmse_below_one": raw_nrmse < 1.0,
        "all_development_oof_projection_qp_audits_passed": bool(
            oof_projection_accepted.all()
        ),
        "all_projection_qp_audits_passed": bool(qp_accepted.all()),
        "all_finite_distance_bounds_passed": bool(feasibility.all()),
        "projected_physical_audits_passed": bool(physical["projected"]["passed"]),
        "mechanistic_physical_audits_passed": bool(physical["mechanistic"]["passed"]),
        "all_five_trust_limits_frozen": True,
        "correction_limit_at_most_0_50": bool(correction_limit <= 0.50),
    }
    passed = bool(all(checks.values()))
    optimization_permitted = assessment_gate_allows_optimization(passed)
    return {
        "passed": passed,
        "execution_policy": ASSESSMENT_GATE_EXECUTION_POLICY,
        "optimization_permitted": optimization_permitted,
        "raw_complete_response_nrmse": raw_nrmse,
        **checks,
        "trust_limits": limits,
        "physical_audit_maxima": physical,
        "failure_action": (
            None if passed else
            "record advisory failure and continue without refitting"
        ),
    }


def _assessment_binding(
    design: Mapping[str, object],
    development_targets: np.ndarray,
    test_targets: np.ndarray,
) -> str:
    return array_digest(
        development_decisions=np.asarray(design["development_decisions"], dtype="<f8"),
        development_influents=np.asarray(design["development_influents"], dtype="<f8"),
        development_targets=np.asarray(development_targets, dtype="<f8"),
        test_decisions=np.asarray(design["test_decisions"], dtype="<f8"),
        test_influents=np.asarray(design["test_influents"], dtype="<f8"),
        test_targets=np.asarray(test_targets, dtype="<f8"),
    )


def load_assessment_checkpoint(
    run: Path, *, source_id: str, input_id: str,
) -> dict[str, Any] | None:
    marker_path = run / "metrics" / "assessment_complete.json"
    gate_path = run / "metrics" / "admission_gate.json"
    if not marker_path.is_file() or not gate_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            not _checkpoint_source_is_authorized(
                run,
                stage="assessment",
                checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("input_digest") != input_id
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return None
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            not isinstance(gate.get("passed"), bool)
            or gate.get("execution_policy") != ASSESSMENT_GATE_EXECUTION_POLICY
            or gate.get("optimization_permitted")
            != assessment_gate_allows_optimization(gate["passed"])
        ):
            return None
        return gate
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def run_assessment(
    run: Path,
    design: Mapping[str, object],
    development_targets: np.ndarray,
    test_targets: np.ndarray,
    *,
    profile: StudyProfile,
    source_files: Mapping[str, str],
) -> AnalysisBundle:
    development_decisions = np.asarray(design["development_decisions"])
    development_influents = np.asarray(design["development_influents"])
    input_id = _assessment_binding(design, development_targets, test_targets)
    source_id = source_digest(source_files)
    existing_gate = load_assessment_checkpoint(
        run, source_id=source_id, input_id=input_id,
    )
    model, oof_raw, _ = fit_or_resume_ridge(
        run, development_decisions, development_influents, development_targets,
        source_files=source_files,
    )
    layout = NetworkLayout(layer_count=profile.layer_count)
    direct_assets = fit_direct_assets(
        development_decisions, development_influents, development_targets,
        clarifier=clarifier_for(profile),
    )
    trust = calibrate_trust_diagnostics(
        model, development_decisions, development_influents, development_targets,
        oof_raw, direct_assets, layout=layout,
    )
    surrogate_assets = build_surrogate_assets(
        model, development_decisions, development_influents, development_targets,
        layout=layout,
        correction_rms_threshold=trust.correction_limit,
        trust_callbacks=trust.callbacks,
        split_rms_threshold=trust.split_limit,
        reactor_rms_threshold=trust.reactor_limit,
        flux_rms_threshold=trust.flux_limit,
    )
    features = model.feature_map.transform(development_decisions, development_influents)
    leverage = np.einsum(
        "ij,jk,ik->i", features, surrogate_assets.leverage_precision, features,
    )
    trust_values = np.column_stack((
        trust.development_values[:, 0], leverage, trust.development_values[:, 1:],
    ))
    if trust_values.shape != (profile.development_count, 5) or not np.all(
        np.isfinite(trust_values)
    ):
        raise RuntimeError("five development trust diagnostics were not evaluated")
    limits = {
        "correction": float(trust.correction_limit),
        "regularized_leverage": float(
            surrogate_assets.trust_thresholds.regularized_leverage
        ),
        "particulate_split": float(trust.split_limit),
        "reactor_residual": float(trust.reactor_limit),
        "clarifier_flux": float(trust.flux_limit),
    }
    if existing_gate is not None:
        return AnalysisBundle(
            passed=bool(existing_gate["passed"]), model=model,
            direct_assets=direct_assets, surrogate_assets=surrogate_assets,
            assessment=None, gate=existing_gate,
        )
    trust_frame = pd.DataFrame(trust_values, columns=[
        "correction", "regularized_leverage", "particulate_split",
        "reactor_residual", "clarifier_flux",
    ])
    trust_frame.insert(0, "row", np.arange(profile.development_count))
    trust_frame.insert(
        1, "projection_qp_accepted", trust.out_of_fold_projection_accepted,
    )
    atomic_dataframe(run / "metrics" / "trust_development_oof.csv", trust_frame)
    atomic_json(run / "metrics" / "trust_limits.json", limits)
    atomic_npz(
        run / "models" / "trust_calibration.npz",
        development_values=trust_values,
        out_of_fold_projected=trust.out_of_fold_projected,
        out_of_fold_projection_accepted=trust.out_of_fold_projection_accepted,
        split_scale=trust.split_scale,
    )
    assessment = assess_raw_projected_mechanistic(
        model,
        development_decisions,
        development_influents,
        development_targets,
        np.asarray(design["test_decisions"]),
        np.asarray(design["test_influents"]),
        test_targets,
        profile,
    )
    _validate_assessment(
        assessment, test_count=profile.test_count,
        response_count=profile.response_count,
    )
    paths = (
        run / "metrics" / "untouched_prediction_metrics.csv",
        run / "metrics" / "physical_violations_assessment.csv",
        run / "metrics" / "projection_qp_diagnostics.csv",
        run / "metrics" / "projection_feasibility_bound.csv",
        run / "predictions" / "untouched_test.npz",
        run / "metrics" / "trust_development_oof.csv",
        run / "metrics" / "trust_limits.json",
        run / "models" / "trust_calibration.npz",
        run / "metrics" / "admission_gate.json",
        run / "models" / "ridge_surrogate.npz",
        run / "models" / "ridge_complete.json",
        run / "metrics" / "ridge_cross_validation.csv",
        run / "metrics" / "ridge_fold_membership.csv",
    )
    atomic_dataframe(paths[0], assessment.metrics)
    atomic_dataframe(paths[1], assessment.violations)
    atomic_dataframe(paths[2], assessment.qp_diagnostics)
    atomic_dataframe(paths[3], assessment.feasibility)
    atomic_npz(
        paths[4], raw=assessment.raw, projected=assessment.projected,
        projected_targets=assessment.projected_targets, mechanistic=test_targets,
    )
    gate = evaluate_admission_gate(
        assessment, correction_limit=trust.correction_limit,
        trust_limits=limits,
        development_oof_projection_accepted=(
            trust.out_of_fold_projection_accepted
        ),
        test_count=profile.test_count,
    )
    atomic_json(paths[8], gate)
    assert_source_unchanged(source_files)
    atomic_json(run / "metrics" / "assessment_complete.json", {
        "stage": "untouched_test_assessment",
        "source_digest": source_id,
        "input_digest": input_id,
        "passed": gate["passed"],
        "artifacts": _artifact_hashes(run, paths),
    })
    return AnalysisBundle(
        passed=bool(gate["passed"]), model=model,
        direct_assets=direct_assets, surrogate_assets=surrogate_assets,
        assessment=assessment, gate=gate,
    )


def _raw_inference_batch(
    model: QuadraticSurrogate,
    decisions: np.ndarray,
    influents: np.ndarray,
) -> np.ndarray:
    return np.asarray(model.predict(decisions, influents), dtype=float)


def _projection_inference_batch(
    cached_raw: np.ndarray,
    decisions: np.ndarray,
    influents: np.ndarray,
    projector: PhysicalProjector,
    layout: NetworkLayout,
) -> int:
    """Project one cached-raw batch without evaluating the surrogate."""

    accepted = 0
    for row in range(len(cached_raw)):
        theta = decisions[row]
        operators = build_network_operators(
            influents[row],
            internal_recycle=float(theta[4]),
            return_recycle=float(theta[5]),
            waste_fraction=float(theta[6]),
            invariant_operator=INVARIANT_MATRIX,
            tss_weights=TSS_VECTOR,
            layout=layout,
        )
        result = projector.project(
            cached_raw[row],
            operators.equality_matrix,
            operators.equality_rhs,
            operators.inequality_matrix,
            warm_start=None,
            raise_on_failure=False,
        )
        state = np.asarray(result.state, dtype=float)
        if state.shape != cached_raw[row].shape or not np.all(np.isfinite(state)):
            raise RuntimeError(
                f"timed cached-raw projection returned an invalid state at test row {row}"
            )
        accepted += int(bool(result.accepted))
    return accepted


def _timing_contract_id(
    *,
    source_id: str,
    analysis_id: str,
    decisions: np.ndarray,
    influents: np.ndarray,
    cached_raw: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(b"article-v3-inference-timing-v1\0")
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(str(INFERENCE_TIMING_WARMUPS).encode())
    digest.update(str(INFERENCE_TIMING_BATCHES).encode())
    digest.update(array_digest(
        test_decisions=np.asarray(decisions, dtype="<f8"),
        test_influents=np.asarray(influents, dtype="<f8"),
        cached_raw=np.asarray(cached_raw, dtype="<f8"),
    ).encode())
    return digest.hexdigest()


def _load_cached_assessment_raw(
    run: Path,
    analysis: AnalysisBundle,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    path = run / "predictions" / "untouched_test.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            value = np.asarray(stored["raw"], dtype=float)
    elif analysis.assessment is not None:
        value = np.asarray(analysis.assessment.raw, dtype=float)
    else:
        raise RuntimeError("cached untouched-test raw predictions are unavailable")
    if value.shape != expected_shape or not np.all(np.isfinite(value)):
        raise RuntimeError("cached untouched-test raw predictions are invalid")
    return value


def _run_inference_timing_benchmark(
    run: Path,
    design: Mapping[str, object],
    analysis: AnalysisBundle,
    *,
    source_files: Mapping[str, str],
    analysis_id: str,
) -> pd.DataFrame:
    """Run the declared 5-warmup/30-batch inference benchmark resumably."""

    decisions = np.asarray(design["test_decisions"], dtype=float)
    influents = np.asarray(design["test_influents"], dtype=float)
    if decisions.ndim != 2 or decisions.shape[1] != 7:
        raise RuntimeError("timing test decisions must have seven columns")
    if influents.ndim != 2 or len(influents) != len(decisions):
        raise RuntimeError("timing test influents do not match the decision rows")
    row_count = len(decisions)
    if row_count < 1:
        raise RuntimeError("inference timing requires at least one test row")
    cached_raw = _load_cached_assessment_raw(
        run, analysis, (row_count, analysis.model.response_count),
    )
    source_id = source_digest(source_files)
    contract = _timing_contract_id(
        source_id=source_id,
        analysis_id=analysis_id,
        decisions=decisions,
        influents=influents,
        cached_raw=cached_raw,
    )
    marker_path = run / "metrics" / "inference_timing_complete.json"
    batches_path = run / "metrics" / "inference_timing_batches.csv"
    events_path = run / "metrics" / "timing_events.csv"
    summary_path = run / "metrics" / "inference_timing_summary.json"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("timing_contract") == contract
                and marker.get("source_digest") == source_id
                and marker.get("input_digest") == analysis_id
                and int(marker.get("row_count", -1)) == row_count
                and _artifacts_match(run, marker.get("artifacts", {}))
            ):
                frame = pd.read_csv(batches_path)
                counts = frame.groupby("category").size().to_dict()
                if counts == {
                    "qp_deployment": INFERENCE_TIMING_BATCHES,
                    "raw_inference": INFERENCE_TIMING_BATCHES,
                }:
                    return frame
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    checkpoints = run / "timing" / "inference_batches"
    verification_path = checkpoints / "cached_raw_verification.json"
    verification: dict[str, Any] | None = None
    if verification_path.is_file():
        try:
            candidate = json.loads(verification_path.read_text(encoding="utf-8"))
            if (
                candidate.get("timing_contract") == contract
                and candidate.get("passed") is True
                and float(candidate.get("maximum_scaled_difference", np.inf))
                <= 1.0e-12
            ):
                verification = candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for category in ("raw_inference", "qp_deployment"):
        for batch in range(INFERENCE_TIMING_BATCHES):
            path = checkpoints / f"{category}_{batch:02d}.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("timing_contract") == contract
                    and payload.get("category") == category
                    and int(payload.get("batch", -1)) == batch
                    and int(payload.get("response_count", -1)) == row_count
                    and int(payload.get("warmup_count", -1))
                    == INFERENCE_TIMING_WARMUPS
                    and float(payload.get("elapsed_seconds", -1.0)) >= 0.0
                    and np.isclose(
                        float(payload.get("per_response_latency_seconds", -1.0)),
                        float(payload["elapsed_seconds"]) / row_count,
                        rtol=1.0e-12,
                        atol=0.0,
                    )
                    and (
                        category != "qp_deployment"
                        or (
                            0 <= int(payload.get("projection_accepted_count", -1))
                            <= row_count
                            and np.isclose(
                                float(payload.get(
                                    "projection_accepted_fraction", np.nan,
                                )),
                                int(payload["projection_accepted_count"]) / row_count,
                                rtol=0.0,
                                atol=1.0e-15,
                            )
                        )
                    )
                ):
                    records[(category, batch)] = payload
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass

    projector = PhysicalProjector(
        analysis.model.response_scale,
        analysis.surrogate_assets.row_scales.equality,
        analysis.surrogate_assets.row_scales.inequality,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
        maximum_iterations=100_000,
        polish=True,
    )

    def publish(
        category: str, batch: int, elapsed_ns: int,
        *, projection_accepted_count: int | None = None,
    ) -> None:
        elapsed = float(elapsed_ns) / 1.0e9
        payload = {
            "timing_contract": contract,
            "category": category,
            "batch": batch,
            "elapsed_ns": int(elapsed_ns),
            "elapsed_seconds": elapsed,
            "per_response_latency_seconds": elapsed / row_count,
            "response_count": row_count,
            "warmup_count": INFERENCE_TIMING_WARMUPS,
            "raw_inference_included": category == "raw_inference",
            "fixed_row_order": True,
        }
        if category == "qp_deployment":
            if (
                projection_accepted_count is None
                or not 0 <= projection_accepted_count <= row_count
            ):
                raise RuntimeError("projection timing acceptance count is invalid")
            payload.update({
                "projection_accepted_count": int(projection_accepted_count),
                "projection_accepted_fraction": (
                    float(projection_accepted_count) / row_count
                ),
            })
        atomic_json(
            checkpoints / f"{category}_{batch:02d}.json", payload,
        )
        records[(category, batch)] = payload

    with threadpool_limits(limits=1):
        missing_raw = [
            batch for batch in range(INFERENCE_TIMING_BATCHES)
            if ("raw_inference", batch) not in records
        ]
        verification_was_warmup = False
        if verification is None:
            reproduced = _raw_inference_batch(
                analysis.model, decisions, influents,
            )
            if reproduced.shape != cached_raw.shape or not np.all(
                np.isfinite(reproduced)
            ):
                raise RuntimeError("cached-raw verification returned invalid output")
            scaled_difference = float(np.max(
                np.abs(reproduced - cached_raw)
                / analysis.model.response_scale[None, :]
            ))
            if scaled_difference > 1.0e-12:
                raise RuntimeError(
                    "cached assessment raw predictions do not reproduce under "
                    f"the timed model (scaled inf={scaled_difference:.3e})"
                )
            verification = {
                "timing_contract": contract,
                "passed": True,
                "maximum_scaled_difference": scaled_difference,
            }
            atomic_json(verification_path, verification)
            verification_was_warmup = bool(missing_raw)
        if missing_raw:
            remaining_warmups = (
                INFERENCE_TIMING_WARMUPS - 1
                if verification_was_warmup else INFERENCE_TIMING_WARMUPS
            )
            for _ in range(remaining_warmups):
                warm = _raw_inference_batch(analysis.model, decisions, influents)
                if warm.shape != cached_raw.shape or not np.all(np.isfinite(warm)):
                    raise RuntimeError("raw inference warmup returned invalid output")
            for batch in missing_raw:
                started = perf_counter_ns()
                raw = _raw_inference_batch(analysis.model, decisions, influents)
                elapsed_ns = perf_counter_ns() - started
                if raw.shape != cached_raw.shape or not np.all(np.isfinite(raw)):
                    raise RuntimeError("timed raw inference returned invalid output")
                publish("raw_inference", batch, elapsed_ns)

        missing_projection = [
            batch for batch in range(INFERENCE_TIMING_BATCHES)
            if ("qp_deployment", batch) not in records
        ]
        if missing_projection:
            for _ in range(INFERENCE_TIMING_WARMUPS):
                _projection_inference_batch(
                    cached_raw, decisions, influents, projector,
                    analysis.surrogate_assets.layout,
                )
            for batch in missing_projection:
                started = perf_counter_ns()
                accepted = _projection_inference_batch(
                    cached_raw, decisions, influents, projector,
                    analysis.surrogate_assets.layout,
                )
                elapsed_ns = perf_counter_ns() - started
                publish(
                    "qp_deployment", batch, elapsed_ns,
                    projection_accepted_count=accepted,
                )

    ordered = [
        records[(category, batch)]
        for category in ("raw_inference", "qp_deployment")
        for batch in range(INFERENCE_TIMING_BATCHES)
    ]
    frame = pd.DataFrame(ordered)
    atomic_dataframe(batches_path, frame)
    # Reporting uses this generic event stream for one-time and route timings.
    atomic_dataframe(events_path, frame)
    summary: dict[str, Any] = {
        "timing_contract": contract,
        "warmup_count": INFERENCE_TIMING_WARMUPS,
        "timed_batch_count_per_route": INFERENCE_TIMING_BATCHES,
        "response_count_per_batch": row_count,
        "raw_inference_excluded_from_qp_timing": True,
        "cached_raw_reproduction_scaled_inf": verification[
            "maximum_scaled_difference"
        ],
        "categories": {},
    }
    for category, group in frame.groupby("category", sort=True):
        values = np.asarray(group["per_response_latency_seconds"], dtype=float)
        summary["categories"][str(category)] = {
            "count": len(values),
            "total": float(np.sum(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }
        if str(category) == "qp_deployment":
            accepted_counts = np.asarray(
                group["projection_accepted_count"], dtype=int,
            )
            accepted_total = int(np.sum(accepted_counts))
            attempted_total = int(row_count * len(group))
            summary["categories"][str(category)].update({
                "projection_accepted_count": accepted_total,
                "projection_attempt_count": attempted_total,
                "projection_accepted_fraction": (
                    float(accepted_total) / attempted_total
                ),
                "projection_accepted_count_per_batch_minimum": int(
                    np.min(accepted_counts)
                ),
                "projection_accepted_count_per_batch_maximum": int(
                    np.max(accepted_counts)
                ),
            })
    atomic_json(summary_path, summary)
    assert_source_unchanged(source_files)
    batch_checkpoints = tuple(
        checkpoints / f"{category}_{batch:02d}.json"
        for category in ("raw_inference", "qp_deployment")
        for batch in range(INFERENCE_TIMING_BATCHES)
    )
    paths = (
        batches_path, events_path, summary_path, verification_path,
        *batch_checkpoints,
    )
    atomic_json(marker_path, {
        "stage": "inference_timing_benchmark",
        "timing_contract": contract,
        "source_digest": source_id,
        "input_digest": analysis_id,
        "row_count": row_count,
        "warmup_count": INFERENCE_TIMING_WARMUPS,
        "timed_batch_count_per_route": INFERENCE_TIMING_BATCHES,
        "artifacts": _artifact_hashes(run, paths),
    })
    return frame


def _route_contract_id(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    influent: np.ndarray,
    route: str,
    protocol: str,
    settings: Any,
    starts: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(case_id.encode())
    digest.update(route.encode())
    digest.update(protocol.encode())
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode())
    digest.update(np.ascontiguousarray(influent, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(starts, dtype="<f8").tobytes())
    return digest.hexdigest()


def _start_controls(result: Any, route: str) -> np.ndarray:
    del route
    return np.asarray(result.initial_normalized_controls, dtype=float)


def _validate_route_result_integrity(result: Any, route: str) -> None:
    """Reject non-finite computational results while permitting clean failures."""

    controls = _start_controls(result, route)
    if controls.shape != (7,) or not np.all(np.isfinite(controls)):
        raise RuntimeError(f"{route} attempt returned invalid initial controls")
    if route == "surrogate":
        final = result.final
        if final is None:
            return
        arrays = (
            final.normalized_controls, final.theta, final.raw, final.projected,
            final.displacement, final.objective_components,
            final.engineering_rows, final.engineering_quantities,
            final.trust_rows, final.trust_values,
        )
        scalars = (final.objective,)
    elif route == "direct":
        if not bool(result.feasible):
            return
        arrays = (
            result.normalized_controls, result.theta, result.state,
            result.response, result.engineering, result.objective_components,
        )
        scalars = (result.objective, result.feed_tss)
    else:
        raise ValueError(f"unknown optimization route {route!r}")
    if any(not np.all(np.isfinite(np.asarray(value, dtype=float))) for value in arrays):
        raise RuntimeError(f"{route} attempt returned a non-finite candidate")
    if not np.all(np.isfinite(np.asarray(scalars, dtype=float))):
        raise RuntimeError(f"{route} attempt returned a non-finite objective/state")


def _read_completed_starts(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    starts: np.ndarray,
) -> tuple[dict[int, Any], float]:
    result_type = SurrogateStartResult if route == "surrogate" else DirectStartResult
    completed: dict[int, Any] = {}
    elapsed = 0.0
    for index in range(len(starts)):
        path = case_directory / "checkpoints" / f"{route}_start_{index:02d}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("route_contract") != route_contract:
                continue
            if (
                payload.get("protocol") != route_protocol
                or int(payload.get("start_index", -1)) != index
                or not np.array_equal(
                    np.asarray(payload.get("normalized_start"), dtype=float), starts[index]
                )
            ):
                raise RuntimeError(f"current {route} attempt checkpoint is inconsistent")
            result = result_type.from_dict(payload["result"])
            if result.start_index != index or not np.array_equal(
                _start_controls(result, route), starts[index]
            ):
                raise RuntimeError(f"current {route} attempt result is inconsistent")
            _validate_route_result_integrity(result, route)
            completed[index] = result
            elapsed = max(elapsed, float(payload.get("cumulative_route_wall_seconds", 0.0)))
        except RuntimeError:
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"current {route} attempt checkpoint is corrupt: {path.name}"
            ) from exc
    return completed, elapsed


def _publish_partial_starts(
    case_directory: Path, route: str, completed: Mapping[int, Any],
) -> None:
    atomic_json(
        case_directory / f"{route}_starts.partial.json",
        [completed[index].as_dict() for index in sorted(completed)],
        nonfinite_to_none=True,
    )


def _write_start_checkpoint(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    starts: np.ndarray,
    result: Any,
    cumulative_elapsed: float,
) -> None:
    index = int(result.start_index)
    if not 0 <= index < len(starts) or not np.array_equal(
        _start_controls(result, route), starts[index]
    ):
        raise RuntimeError(f"{route} solver returned a mismatched start index/control")
    _validate_route_result_integrity(result, route)
    atomic_json(
        case_directory / "checkpoints" / f"{route}_start_{index:02d}.json",
        {
            "schema": 1,
            "route": route,
            "route_contract": route_contract,
            "protocol": route_protocol,
            "start_index": index,
            "normalized_start": starts[index],
            "cumulative_route_wall_seconds": cumulative_elapsed,
            "result": result.as_dict(),
        },
        nonfinite_to_none=True,
    )


def _load_complete_route(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    starts: np.ndarray,
) -> Any | None:
    marker_path = case_directory / f"{route}_complete.json"
    payload_path = case_directory / f"{route}.json"
    if not marker_path.is_file() or not payload_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("route_contract") != route_contract:
            return None
        if marker.get("protocol") != route_protocol:
            raise RuntimeError(f"current {route} completion marker has wrong protocol")
        if not _artifacts_match(case_directory, marker.get("artifacts", {})):
            raise RuntimeError(f"current {route} completed artifacts changed")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if (
            payload.get("route_contract") != route_contract
            or payload.get("protocol") != route_protocol
        ):
            raise RuntimeError(f"current {route} result payload is inconsistent")
        result_type = SurrogateStartResult if route == "surrogate" else DirectStartResult
        restored = tuple(result_type.from_dict(item) for item in payload["starts"])
        if (
            len(restored) != len(starts)
            or [item.start_index for item in restored] != list(range(len(starts)))
        ):
            raise RuntimeError(f"current {route} result has wrong attempt count")
        if any(not np.array_equal(_start_controls(item, route), starts[index])
               for index, item in enumerate(restored)):
            raise RuntimeError(f"current {route} result has wrong initial controls")
        for item in restored:
            _validate_route_result_integrity(item, route)
        selected_index = payload.get("selected_start")
        selected = None if selected_index is None else restored[int(selected_index)]
        result_class = SurrogateMultistartResult if route == "surrogate" else DirectMultistartResult
        if route == "surrogate":
            result = result_class(
                restored, selected, str(payload["status"]), protocol=route_protocol,
            )
        else:
            result = result_class(restored, selected, str(payload["status"]))
        return result, payload
    except RuntimeError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"current {route} completion checkpoint is corrupt") from exc


def _publish_complete_route(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    result: Any,
    elapsed_seconds: float,
) -> dict[str, Any]:
    start_count = len(result.starts)
    if start_count != 1:
        raise RuntimeError(f"{route} protocol requires exactly one local attempt")
    for item in result.starts:
        _validate_route_result_integrity(item, route)
    payload = result.as_dict()
    payload.update({
        "route": route,
        "route_contract": route_contract,
        "protocol": route_protocol,
        "optimization_attempt_count": start_count,
        "article_start_count": start_count,
        "maximum_wall_time": None,
        "elapsed_seconds": elapsed_seconds,
    })
    payload_path = case_directory / f"{route}.json"
    atomic_json(payload_path, payload, nonfinite_to_none=True)
    start_paths = tuple(
        case_directory / "checkpoints" / f"{route}_start_{index:02d}.json"
        for index in range(start_count)
    )
    if not all(path.is_file() for path in start_paths):
        raise RuntimeError(f"{route} is missing its atomic attempt checkpoint")
    paths = (payload_path, *start_paths)
    atomic_json(case_directory / f"{route}_complete.json", {
        "stage": f"{route}_single_local_attempt",
        "route_contract": route_contract,
        "protocol": route_protocol,
        "start_count": start_count,
        "optimization_attempt_count": start_count,
        "selected_start": payload["selected_start"],
        "status": payload["status"],
        "artifacts": _artifact_hashes(case_directory, paths),
    })
    return payload


def _run_surrogate_route(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    assets: Any,
    source_id: str,
    analysis_id: str,
    problem: Any | None = None,
) -> tuple[SurrogateMultistartResult, dict[str, Any]]:
    settings = SurrogateSolverSettings(maximum_wall_time=None)
    starts = np.asarray(EXACT_QP_CENTER_START, dtype=float).reshape(1, 7)
    contract = _route_contract_id(
        source_id=source_id, analysis_id=analysis_id, case_id=case_id,
        influent=influent, route="surrogate",
        protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        settings=settings, starts=starts,
    )
    restored = _load_complete_route(
        case_directory, route="surrogate", route_contract=contract,
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL, starts=starts,
    )
    if restored is not None:
        return restored
    completed, prior_elapsed = _read_completed_starts(
        case_directory, route="surrogate", route_contract=contract,
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL, starts=starts,
    )
    _publish_partial_starts(case_directory, "surrogate", completed)
    started = perf_counter()

    def checkpoint(result: SurrogateStartResult) -> None:
        completed[result.start_index] = result
        _write_start_checkpoint(
            case_directory, route="surrogate", route_contract=contract,
            route_protocol=EXACT_QP_SINGLE_START_PROTOCOL,
            starts=starts, result=result,
            cumulative_elapsed=prior_elapsed + perf_counter() - started,
        )
        _publish_partial_starts(case_directory, "surrogate", completed)

    result = solve_surrogate_exact_qp_local(
        assets,
        SurrogateCase(influent=influent, case_id=case_id),
        settings=settings,
        problem=problem,
        name="article_surrogate_exact_qp",
        progress_callback=checkpoint,
        completed_result=completed.get(0),
    )
    if (
        len(result.starts) != 1
        or result.protocol != EXACT_QP_SINGLE_START_PROTOCOL
        or result.starts[0].stages
    ):
        raise RuntimeError("surrogate route violated the single exact-QP attempt contract")
    if 0 not in completed:
        # A valid fresh solve always invokes the callback exactly once. This is
        # also a guard against a solver result that was never atomically saved.
        raise RuntimeError("surrogate exact-QP attempt was not checkpointed")
    elapsed = prior_elapsed + perf_counter() - started
    payload = _publish_complete_route(
        case_directory, route="surrogate", route_contract=contract,
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        result=result, elapsed_seconds=elapsed,
    )
    return result, payload


def _run_direct_route(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    assets: Any,
    development_decisions: np.ndarray,
    development_influents: np.ndarray,
    development_targets: np.ndarray,
    source_id: str,
    analysis_id: str,
) -> tuple[DirectMultistartResult, dict[str, Any]]:
    settings = SolverSettings(maximum_wall_time=None)
    starts = np.asarray(direct_normalized_starts()[0], dtype=float).reshape(1, 7)
    if not np.array_equal(starts[0], np.full(7, 0.5)):
        raise RuntimeError("direct route center start is not deterministic")
    contract = _route_contract_id(
        source_id=source_id, analysis_id=analysis_id, case_id=case_id,
        influent=influent, route="direct", protocol=DIRECT_SINGLE_CENTER_PROTOCOL,
        settings=settings, starts=starts,
    )
    restored = _load_complete_route(
        case_directory, route="direct", route_contract=contract,
        route_protocol=DIRECT_SINGLE_CENTER_PROTOCOL, starts=starts,
    )
    if restored is not None:
        return restored
    completed, prior_elapsed = _read_completed_starts(
        case_directory, route="direct", route_contract=contract,
        route_protocol=DIRECT_SINGLE_CENTER_PROTOCOL, starts=starts,
    )
    _publish_partial_starts(case_directory, "direct", completed)
    started = perf_counter()

    def checkpoint(result: DirectStartResult) -> None:
        completed[result.start_index] = result
        _write_start_checkpoint(
            case_directory, route="direct", route_contract=contract,
            route_protocol=DIRECT_SINGLE_CENTER_PROTOCOL,
            starts=starts, result=result,
            cumulative_elapsed=prior_elapsed + perf_counter() - started,
        )
        _publish_partial_starts(case_directory, "direct", completed)

    result = solve_direct_multistart(
        assets,
        DirectCase(influent=influent, case_id=case_id),
        development_decisions,
        development_influents,
        development_targets,
        settings=settings,
        starts=starts,
        allow_reduced_starts=True,
        completed_starts=completed,
        progress_callback=checkpoint,
    )
    if len(result.starts) != 1:
        raise RuntimeError("direct route violated the single-attempt contract")
    elapsed = prior_elapsed + perf_counter() - started
    payload = _publish_complete_route(
        case_directory, route="direct", route_contract=contract,
        route_protocol=DIRECT_SINGLE_CENTER_PROTOCOL,
        result=result, elapsed_seconds=elapsed,
    )
    return result, payload


def _unavailable_violation(method: str, case: str, reason: str) -> dict[str, Any]:
    return {
        "case": case,
        "method": method,
        "audit_available": False,
        "audit_unavailable_reason": reason,
        "mass_conservation_violation_max": np.nan,
        "mass_conservation_violation_count": 0,
        "nonnegativity_violation_max": np.nan,
        "nonnegativity_violation_count": 0,
    }


def _physical_record(
    method: str,
    case: str,
    response: np.ndarray,
    theta: np.ndarray,
    influent: np.ndarray,
    analysis: AnalysisBundle,
) -> dict[str, Any]:
    if response.shape != (analysis.surrogate_assets.layout.state_size,) or not np.all(
        np.isfinite(response)
    ):
        return _unavailable_violation(method, case, "response unavailable or non-finite")
    record = violation_record(
        method,
        case,
        response,
        theta,
        influent,
        analysis.surrogate_assets.layout,
        analysis.surrogate_assets.row_scales.equality,
        analysis.surrogate_assets.row_scales.inequality,
        analysis.model.response_scale,
    )
    record["audit_available"] = True
    record["audit_unavailable_reason"] = None
    return record


def _equivalence_error_payload(exc: Exception, elapsed: float) -> dict[str, Any]:
    return {
        "smooth_accepted": False,
        "reference_accepted": False,
        "accepted": False,
        "state_rms": None,
        "state_inf": None,
        "own_smooth_residual": None,
        "own_reference_residual": None,
        "cross_residual": None,
        "relative_objective_difference": None,
        "engineering_difference": None,
        "reference_root_difference_generation": None,
        "reference_root_difference_state_scale": None,
        "branch_agreement": False,
        "feasibility_agreement": False,
        "elapsed_seconds": elapsed,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _strict_equivalence(
    theta: np.ndarray,
    influent: np.ndarray,
    analysis: AnalysisBundle,
) -> tuple[Any | None, dict[str, Any]]:
    return _strict_equivalence_assets(
        theta, influent, analysis.direct_assets,
    )


def _strict_equivalence_assets(
    theta: np.ndarray,
    influent: np.ndarray,
    direct_assets: Any,
) -> tuple[Any | None, dict[str, Any]]:
    settings = SolverSettings(maximum_wall_time=None)
    started = perf_counter()
    try:
        smooth = solve_fixed_input_two_start(
            theta, influent, direct_assets, settings=settings,
        )
        diagnostics = compare_smooth_reference(
            theta, influent, direct_assets,
            smooth=smooth, settings=settings,
        )
        payload = asdict(diagnostics)
        payload["elapsed_seconds"] = perf_counter() - started
        payload["maximum_wall_time"] = None
        return smooth, payload
    except Exception as exc:
        return None, _equivalence_error_payload(exc, perf_counter() - started)


_EQUIVALENCE_WORKER_ASSETS: Any | None = None


def _initialize_equivalence_worker(direct_assets: Any) -> None:
    global _EQUIVALENCE_WORKER_ASSETS
    _EQUIVALENCE_WORKER_ASSETS = direct_assets


def _test_equivalence_worker(
    payload: tuple[int, np.ndarray, np.ndarray],
) -> tuple[int, np.ndarray, dict[str, Any]]:
    row, theta, influent = payload
    if _EQUIVALENCE_WORKER_ASSETS is None:
        raise RuntimeError("fixed-input equivalence worker was not initialized")
    with threadpool_limits(limits=1):
        smooth, equivalence = _strict_equivalence_assets(
            theta, influent, _EQUIVALENCE_WORKER_ASSETS,
        )
    response_count = int(_EQUIVALENCE_WORKER_ASSETS.response_count)
    smooth_response = (
        np.asarray(smooth.routes[0].response, dtype=float)
        if smooth is not None and smooth.accepted
        else np.full(response_count, np.nan)
    )
    return row, smooth_response, equivalence


def _reference_two_start(
    theta: np.ndarray,
    influent: np.ndarray,
    analysis: AnalysisBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    clarifier = analysis.direct_assets.clarifier
    state_size = analysis.direct_assets.state_count
    response_size = analysis.direct_assets.response_count
    started = perf_counter()
    try:
        operating = ArticleOperatingPoint(*map(float, theta))
        first = solve_steady_state(
            operating, influent, starts=(1,), clarifier=clarifier,
            logarithmic_only=True, strict_v3=True,
        )
        second = solve_steady_state(
            operating, influent, starts=(2,), clarifier=clarifier,
            logarithmic_only=True, strict_v3=True,
        )
        reactors, _ = unpack_state(first.state, clarifier)
        scale = generation_scale(influent, reactors[-1], clarifier)
        difference = float(np.max(np.abs(first.state - second.state) / scale))
        branches_match = branch_classification(
            first.state, clarifier,
        ) == branch_classification(second.state, clarifier)
        accepted = bool(
            first.accepted and second.accepted
            and difference <= 1.0e-6 and branches_match
        )
        response = assemble_target(first.state, operating, influent, clarifier)
        return (
            np.asarray(response, dtype=float),
            np.asarray(first.state, dtype=float),
            np.asarray(second.state, dtype=float),
            {
                "accepted": accepted,
                "scaled_root_difference": difference,
                "branch_agreement": branches_match,
                "elapsed_seconds": perf_counter() - started,
                "start_1_accepted": bool(first.accepted),
                "start_2_accepted": bool(second.accepted),
            },
        )
    except Exception as exc:
        return (
            np.full(response_size, np.nan),
            np.full(state_size, np.nan),
            np.full(state_size, np.nan),
            {
                "accepted": False,
                "scaled_root_difference": None,
                "branch_agreement": False,
                "elapsed_seconds": perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


def _test_equivalence_contract(
    source_id: str,
    analysis_id: str,
    row: int,
    theta: np.ndarray,
    influent: np.ndarray,
    reference: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(str(row).encode())
    for value in (theta, influent, reference):
        digest.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    return digest.hexdigest()


def _run_untouched_test_equivalence(
    run: Path,
    design: Mapping[str, object],
    test_targets: np.ndarray,
    analysis: AnalysisBundle,
    *,
    source_files: Mapping[str, str],
    analysis_id: str,
    parallel_workers: int,
) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    source_id = source_digest(source_files)
    marker_path = run / "metrics" / "smooth_reference_test_complete.json"
    diagnostics_path = run / "metrics" / "smooth_reference_equivalence_test.csv"
    violations_path = run / "metrics" / "physical_violations_equivalence_test.csv"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("source_digest") == source_id
                and marker.get("input_digest") == analysis_id
                and int(marker.get("row_count", -1)) == len(test_targets)
                and _artifacts_match(run, marker.get("artifacts", {}))
            ):
                diagnostics = pd.read_csv(diagnostics_path)
                violations = pd.read_csv(violations_path)
                if len(diagnostics) == len(test_targets):
                    return bool(marker["all_accepted"]), diagnostics, violations
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    decisions = np.asarray(design["test_decisions"], dtype=float)
    influents = np.asarray(design["test_influents"], dtype=float)
    rows_directory = run / "validation" / "untouched_test_equivalence" / "rows"
    payloads: dict[int, dict[str, Any]] = {}
    contracts: dict[int, str] = {}
    missing: list[int] = []
    for row in range(len(test_targets)):
        contract = _test_equivalence_contract(
            source_id, analysis_id, row, decisions[row], influents[row], test_targets[row],
        )
        contracts[row] = contract
        path = rows_directory / f"row_{row:06d}.json"
        if path.is_file():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if (
                    candidate.get("row_contract") == contract
                    and int(candidate["row"]) == row
                    and isinstance(candidate.get("equivalence"), Mapping)
                    and len(candidate.get("violations", [])) == 2
                ):
                    payloads[row] = candidate
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
        if row not in payloads:
            missing.append(row)

    def publish(row: int, smooth_response: np.ndarray, equivalence: Mapping[str, Any]) -> None:
        violations = [
            _physical_record(
                "smooth", f"test_{row:04d}:fixed_input", smooth_response,
                decisions[row], influents[row], analysis,
            ),
            _physical_record(
                "reference", f"test_{row:04d}:fixed_input", test_targets[row],
                decisions[row], influents[row], analysis,
            ),
        ]
        payload = {
                "row": row,
                "row_contract": contracts[row],
                "equivalence": dict(equivalence),
                "smooth_response": smooth_response,
                "reference_response": test_targets[row],
                "violations": violations,
        }
        atomic_json(
            rows_directory / f"row_{row:06d}.json",
            payload,
            nonfinite_to_none=True,
        )
        payloads[row] = _json_ready(payload, nonfinite_to_none=True)

    if missing:
        worker_count = max(1, min(int(parallel_workers), len(missing)))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_equivalence_worker,
            initargs=(analysis.direct_assets,),
        ) as pool:
            futures = {
                pool.submit(
                    _test_equivalence_worker,
                    (row, decisions[row], influents[row]),
                ): row
                for row in missing
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                row, smooth_response, equivalence = future.result()
                publish(row, smooth_response, equivalence)
                if completed_count % max(1, len(missing) // 100) == 0:
                    _write_state(
                        run, "untouched_test_equivalence", "running",
                        completed_rows=len(payloads), total_rows=len(test_targets),
                        parallel_workers=worker_count,
                    )
    diagnostic_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    for row in range(len(test_targets)):
        payload = payloads[row]
        equivalence_row = dict(payload["equivalence"])
        equivalence_row["row"] = row
        diagnostic_rows.append(equivalence_row)
        physical_rows.extend(payload["violations"])
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values("row").reset_index(drop=True)
    violations = pd.DataFrame(physical_rows)
    atomic_dataframe(diagnostics_path, diagnostics)
    atomic_dataframe(violations_path, violations)
    accepted = diagnostics["accepted"].astype(str).str.lower().eq("true")
    all_accepted = bool(len(diagnostics) == len(test_targets) and accepted.all())
    summary_path = run / "metrics" / "smooth_reference_equivalence_test_summary.json"
    atomic_json(summary_path, {
        "row_count": len(diagnostics),
        "accepted_count": int(accepted.sum()),
        "all_accepted": all_accepted,
        "required_row_count": len(test_targets),
        "actual_selected_decision_denominator_reported_separately": True,
        "parallel_workers": max(1, min(int(parallel_workers), len(test_targets))),
    })
    assert_source_unchanged(source_files)
    paths = (diagnostics_path, violations_path, summary_path)
    atomic_json(marker_path, {
        "stage": "untouched_test_smooth_reference_equivalence",
        "source_digest": source_id,
        "input_digest": analysis_id,
        "row_count": len(test_targets),
        "all_accepted": all_accepted,
        "artifacts": _artifact_hashes(run, paths),
    })
    return all_accepted, diagnostics, violations


def _selection_contract_id(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    route: str,
    route_payload: Mapping[str, Any],
    influent: np.ndarray,
    route_artifact_digest: str,
) -> str:
    digest = sha256()
    for value in (
        source_id, analysis_id, case_id, route,
        str(route_payload.get("route_contract", "")),
        str(route_payload.get("selected_start", "none")),
        route_artifact_digest,
    ):
        digest.update(value.encode())
    digest.update(np.ascontiguousarray(influent, dtype="<f8").tobytes())
    return digest.hexdigest()


def _load_selection_checkpoint(
    case_directory: Path,
    *,
    route: str,
    selection_contract: str,
) -> tuple[bool, bool, pd.DataFrame] | None:
    marker_path = case_directory / f"{route}_selection_complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("selection_contract") != selection_contract
            or not _artifacts_match(case_directory, marker.get("artifacts", {}))
        ):
            return None
        selected = bool(marker["selected"])
        accepted = bool(marker["accepted"])
        if not selected:
            return False, accepted, pd.DataFrame()
        violations = pd.read_csv(case_directory / f"{route}_physical_violations.csv")
        if set(violations["method"]) != {"raw", "projected", "smooth", "reference"}:
            return None
        return True, accepted, violations
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _combined_constraint_function(problem: Any, *, name: str) -> ca.Function:
    """Return equality and inequality rows in IPOPT's original order."""

    variable_count = int(problem.objective_function.numel_in(0))
    parameter_count = int(problem.objective_function.numel_in(1))
    variable = ca.MX.sym(f"{name}_x", variable_count)
    parameter = ca.MX.sym(f"{name}_p", parameter_count)
    return ca.Function(
        f"{name}_constraints",
        [variable, parameter],
        [ca.vertcat(
            problem.equality_function(variable, parameter),
            problem.inequality_function(variable, parameter),
        )],
    )


def _derivative_audit_contract(
    selection_contract: str,
    point: np.ndarray,
    multipliers: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(selection_contract.encode())
    digest.update(np.ascontiguousarray(point, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(multipliers, dtype="<f8").tobytes())
    return digest.hexdigest()


def _run_selected_derivative_audit(
    case_directory: Path,
    *,
    route: str,
    case_id: str,
    influent: np.ndarray,
    selected: Any,
    analysis: AnalysisBundle,
    selection_contract: str,
) -> dict[str, Any]:
    """Persist the route-appropriate first-order derivative audit."""

    path = case_directory / f"{route}_derivative_audit.json"
    marker_path = case_directory / f"{route}_derivative_audit_complete.json"
    exact_qp_route = bool(
        route == "surrogate"
        and getattr(selected, "protocol", None) == EXACT_QP_SINGLE_START_PROTOCOL
    )
    if exact_qp_route:
        final = selected.final
        if final is None:
            raise RuntimeError("selected exact-QP attempt has no final candidate")
        evidence = {
            "stationarity": final.stationarity.as_dict(),
            "lower_active_set": final.lower_active_set,
            "upper_kkt": final.upper_kkt,
            "projection_reproduction_passed": (
                final.feasibility.projection_reproduction_passed
            ),
        }
        evidence = _json_ready(evidence, nonfinite_to_none=True)
        audit_contract = sha256(
            selection_contract.encode()
            + _canonical_json_digest(evidence).encode()
        ).hexdigest()
        if marker_path.is_file() and path.is_file():
            marker = _load_json_object(
                marker_path, description="exact-QP derivative audit marker",
            )
            payload = _load_json_object(path, description="exact-QP derivative audit")
            if marker.get("audit_contract") != audit_contract:
                # A different selection is stale, not corrupt; overwrite it.
                pass
            elif (
                payload.get("audit_contract") != audit_contract
                or not isinstance(payload.get("passed"), bool)
                or not _artifacts_match(case_directory, marker.get("artifacts", {}))
            ):
                raise RuntimeError("current exact-QP derivative audit is corrupt")
            else:
                return payload
        lower = final.lower_active_set
        upper = final.upper_kkt
        passed = bool(
            final.stationarity.resolved
            and final.stationarity.stationary
            and final.stationarity.lower_qp_kkt_passed
            and isinstance(lower, Mapping)
            and lower.get("stable") is True
            and isinstance(upper, Mapping)
            and upper.get("feasible") is True
            and upper.get("stationary") is True
            and final.feasibility.projection_reproduction_passed is True
        )
        payload = {
            "route": route,
            "case": case_id,
            "selected_start": int(selected.start_index),
            "protocol": EXACT_QP_SINGLE_START_PROTOCOL,
            "audit_contract": audit_contract,
            "required": True,
            "audited_point": "exact_qp_active_set_endpoint",
            "passed": passed,
            "status": "passed" if passed else "stationarity_unresolved",
            "result": evidence,
        }
        atomic_json(path, payload, nonfinite_to_none=True)
        atomic_json(marker_path, {
            "stage": "selected_exact_qp_active_set_audit",
            "audit_contract": audit_contract,
            "route": route,
            "case": case_id,
            "passed": passed,
            "artifacts": _artifact_hashes(case_directory, (path,)),
        })
        return payload

    stages = tuple(selected.stages)
    point = (
        np.asarray(stages[-1].primal, dtype=float)
        if stages else np.empty(0, dtype=float)
    )
    multipliers_value = (
        None if not stages else stages[-1].constraint_multipliers
    )
    multipliers = (
        np.asarray(multipliers_value, dtype=float)
        if multipliers_value is not None else np.empty(0, dtype=float)
    )
    audit_contract = _derivative_audit_contract(
        selection_contract, point, multipliers,
    )
    if marker_path.is_file() and path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                marker.get("audit_contract") == audit_contract
                and payload.get("audit_contract") == audit_contract
                and isinstance(payload.get("passed"), bool)
                and _artifacts_match(case_directory, marker.get("artifacts", {}))
            ):
                return payload
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    payload: dict[str, Any] = {
        "route": route,
        "case": case_id,
        "selected_start": int(selected.start_index),
        "audit_contract": audit_contract,
        "required": True,
        "audited_point": "final_continuation_stage_primal",
        "passed": False,
        "status": "audit_exception",
    }
    try:
        if not stages:
            raise RuntimeError("selected start has no continuation stages")
        if multipliers_value is None:
            raise RuntimeError(
                "final continuation stage does not retain IPOPT constraint multipliers"
            )
        if route == "surrogate":
            expected_stage = float(GAP_CONTINUATION[-1])
            if not np.isclose(stages[-1].tau, expected_stage, rtol=0.0, atol=0.0):
                raise RuntimeError("selected surrogate start lacks the final gap stage")
            settings = SurrogateSolverSettings(maximum_wall_time=None)
            case = SurrogateCase(influent=influent, case_id=case_id)
            problem = build_surrogate_nlp(
                analysis.surrogate_assets,
                expected_stage,
                settings=settings,
                name=f"audit_surrogate_{case_id}",
                compile_solver=False,
            )
            parameters = case.parameter_vector(analysis.surrogate_assets)
        elif route == "direct":
            epsilon, receiver_half_width = CONTINUATION_SCHEDULE[-1]
            if (
                not np.isclose(stages[-1].epsilon, epsilon, rtol=0.0, atol=0.0)
                or not np.isclose(
                    stages[-1].receiver_half_width,
                    receiver_half_width,
                    rtol=0.0,
                    atol=0.0,
                )
            ):
                raise RuntimeError("selected direct start lacks the final smoothing stage")
            settings = SolverSettings(maximum_wall_time=None)
            case = DirectCase(influent=influent, case_id=case_id)
            problem = build_direct_nlp(
                analysis.direct_assets,
                epsilon=epsilon,
                receiver_half_width=receiver_half_width,
                settings=settings,
                name=f"audit_direct_{case_id}",
                compile_solver=False,
            )
            parameters = case.parameter_vector()
        else:
            raise ValueError(f"unknown optimization route {route!r}")
        constraints = _combined_constraint_function(
            problem, name=f"audit_{route}_{case_id}",
        )
        result = audit_casadi_nlp_derivatives(
            problem.objective_function,
            constraints,
            point,
            parameters,
            problem.lower_bounds,
            problem.upper_bounds,
            multipliers,
            name=f"article_{route}_{case_id}",
            raise_on_failure=False,
        )
        payload.update({
            "passed": bool(result.passed),
            "status": "passed" if result.passed else "failed_tolerance",
            "result": result.as_dict(),
        })
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(path, payload, nonfinite_to_none=True)
    atomic_json(marker_path, {
        "stage": "selected_continuation_derivative_audit",
        "audit_contract": audit_contract,
        "route": route,
        "case": case_id,
        "passed": bool(payload["passed"]),
        "artifacts": _artifact_hashes(case_directory, (path,)),
    })
    return payload


def _evaluate_selected_route(
    case_directory: Path,
    *,
    case_id: str,
    route: str,
    influent: np.ndarray,
    result: Any,
    route_payload: Mapping[str, Any],
    analysis: AnalysisBundle,
    source_id: str,
    analysis_id: str,
) -> tuple[bool, bool, pd.DataFrame]:
    selection_contract = _selection_contract_id(
        source_id=source_id, analysis_id=analysis_id, case_id=case_id,
        route=route, route_payload=route_payload, influent=influent,
        route_artifact_digest=file_digest(case_directory / f"{route}.json"),
    )
    restored = _load_selection_checkpoint(
        case_directory, route=route, selection_contract=selection_contract,
    )
    if restored is not None:
        return restored
    selected = result.selected
    if selected is None:
        route_path = case_directory / f"{route}.json"
        atomic_json(case_directory / f"{route}_selection_complete.json", {
            "stage": "selected_decision_cross_evaluation",
            "selection_contract": selection_contract,
            "selected": False,
            "accepted": False,
            "artifacts": _artifact_hashes(case_directory, (route_path,)),
        })
        return False, False, pd.DataFrame()
    final = selected.final if route == "surrogate" else selected
    if final is None:
        raise RuntimeError(f"{route} selected start has no final candidate")
    _validate_route_result_integrity(selected, route)
    derivative_audit = _run_selected_derivative_audit(
        case_directory,
        route=route,
        case_id=case_id,
        influent=influent,
        selected=selected,
        analysis=analysis,
        selection_contract=selection_contract,
    )
    theta = np.asarray(final.theta, dtype=float)
    if theta.shape != (7,) or not np.all(np.isfinite(theta)):
        raise RuntimeError(f"{route} selected candidate has invalid controls")
    normalized = (theta - DECISION_LOWER) / (DECISION_UPPER - DECISION_LOWER)
    case = SurrogateCase(influent=influent, case_id=case_id)
    raw = np.asarray(analysis.model.predict(theta, influent), dtype=float)
    projection = cold_reproject(
        analysis.surrogate_assets, case, normalized, raise_on_failure=False,
    )
    projected = np.asarray(projection.state, dtype=float)
    smooth, equivalence = _strict_equivalence(theta, influent, analysis)
    smooth_response = (
        np.asarray(smooth.routes[0].response, dtype=float)
        if smooth is not None and smooth.accepted
        else np.full(analysis.direct_assets.response_count, np.nan)
    )
    reference, reference_state_1, reference_state_2, replay = _reference_two_start(
        theta, influent, analysis,
    )
    optimizer_root = {
        "applicable": route == "direct",
        "state_scaled_inf": None,
        "feed_tss_scaled_absolute": None,
        "maximum_scaled_difference": None,
        "branch_agreement": None,
        "accepted": True,
    }
    if route == "direct":
        if smooth is not None and smooth.accepted and smooth.routes[0].branch is not None:
            state_difference = float(np.max(np.abs(
                (np.asarray(smooth.routes[0].state) - np.asarray(final.state))
                / analysis.direct_assets.state_scale
            )))
            feed_difference = float(abs(
                float(smooth.routes[0].feed_tss) - float(final.feed_tss)
            ) / analysis.direct_assets.feed_scale)
            branch_agreement = bool(
                final.branch is not None
                and smooth_branches_match(smooth.routes[0].branch, final.branch)
            )
            optimizer_root = {
                "applicable": True,
                "state_scaled_inf": state_difference,
                "feed_tss_scaled_absolute": feed_difference,
                "maximum_scaled_difference": max(state_difference, feed_difference),
                "branch_agreement": branch_agreement,
                "accepted": bool(
                    max(state_difference, feed_difference) <= 1.0e-6
                    and branch_agreement
                ),
            }
        else:
            optimizer_root["accepted"] = False
    accepted = bool(
        projection.accepted
        and equivalence.get("accepted") is True
        and replay.get("accepted") is True
        and optimizer_root["accepted"] is True
        and derivative_audit.get("passed") is True
        and np.all(np.isfinite(raw))
        and np.all(np.isfinite(projected))
        and np.all(np.isfinite(smooth_response))
        and np.all(np.isfinite(reference))
    )
    equivalence = dict(equivalence)
    equivalence.update({
        "accepted": accepted,
        "smooth_reference_comparison_accepted": bool(
            equivalence.get("accepted") is True
        ),
        "independent_reference_replay_accepted": bool(replay.get("accepted") is True),
        "selected_route": route,
        "selected_start": int(selected.start_index),
        "cold_projection_accepted": bool(projection.accepted),
        "reference_replay": replay,
        "optimizer_root_reproduction": optimizer_root,
        "derivative_audit": derivative_audit,
    })
    selected_path = case_directory / f"{route}_selected.npz"
    selected_arrays: dict[str, np.ndarray] = {
        "start_index": np.asarray(selected.start_index),
        "theta": theta,
        "raw": raw,
        "projected": projected,
        "smooth": smooth_response,
        "reference": reference,
    }
    if route == "direct":
        selected_arrays.update({
            "response": np.asarray(final.response, dtype=float),
            "state": np.asarray(final.state, dtype=float),
            "optimized_response": np.asarray(final.response, dtype=float),
        })
    else:
        selected_arrays["continuation_projected"] = np.asarray(final.projected, dtype=float)
    atomic_npz(selected_path, **selected_arrays)
    reference_path = case_directory / f"{route}_reference.npz"
    atomic_npz(
        reference_path, theta=theta, response=reference,
        state=reference_state_1, state_start_2=reference_state_2,
    )
    equivalence_path = case_directory / f"{route}_equivalence.json"
    atomic_json(equivalence_path, equivalence, nonfinite_to_none=True)
    case_label = f"{case_id}:{route}"
    violations = pd.DataFrame([
        _physical_record(method, case_label, response, theta, influent, analysis)
        for method, response in (
            ("raw", raw), ("projected", projected),
            ("smooth", smooth_response), ("reference", reference),
        )
    ])
    violation_path = case_directory / f"{route}_physical_violations.csv"
    atomic_dataframe(violation_path, violations)
    audit_path = case_directory / f"{route}_cross_evaluation.json"
    atomic_json(audit_path, {
        "selection_contract": selection_contract,
        "accepted": accepted,
        "cold_projection_accepted": bool(projection.accepted),
        "cold_projection_diagnostics": projection.diagnostics.as_dict(),
        "smooth_fixed_input_accepted": bool(smooth is not None and smooth.accepted),
        "strict_equivalence_accepted": bool(
            equivalence["smooth_reference_comparison_accepted"]
        ),
        "reference_replay": replay,
        "optimizer_root_reproduction": optimizer_root,
        "derivative_audit_passed": bool(derivative_audit.get("passed") is True),
        "derivative_audit_status": derivative_audit.get("status"),
    }, nonfinite_to_none=True)
    paths = (
        case_directory / f"{route}.json", selected_path, reference_path,
        equivalence_path, violation_path, audit_path,
        case_directory / f"{route}_derivative_audit.json",
        case_directory / f"{route}_derivative_audit_complete.json",
    )
    atomic_json(case_directory / f"{route}_selection_complete.json", {
        "stage": "selected_decision_cross_evaluation",
        "selection_contract": selection_contract,
        "selected": True,
        "accepted": accepted,
        "artifacts": _artifact_hashes(case_directory, paths),
    })
    return True, accepted, violations


def _case_contract_id(
    source_id: str, analysis_id: str, case_id: str, influent: np.ndarray,
) -> str:
    return sha256(
        source_id.encode() + analysis_id.encode() + OPTIMIZATION_PROTOCOL.encode()
        + case_id.encode()
        + np.ascontiguousarray(influent, dtype="<f8").tobytes()
    ).hexdigest()


def _load_case_checkpoint(
    run: Path,
    case_directory: Path,
    *,
    case_contract: str,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    marker_path = case_directory / "case_complete.json"
    violations_path = case_directory / "physical_violations.csv"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("case_contract") != case_contract
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return None
        violations = (
            pd.read_csv(violations_path)
            if bool(marker.get("selected_decision_count"))
            else pd.DataFrame()
        )
        return marker, violations
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def run_optimization_stage(
    *,
    run: Path,
    profile: StudyProfile,
    design: Mapping[str, object],
    development_targets: np.ndarray,
    test_targets: np.ndarray,
    analysis: AnalysisBundle,
    source_files: Mapping[str, str],
) -> bool:
    """Run all article cases, strict equivalence checks, replays, and reports.

    Each route uses one deterministic center start. The surrogate route calls
    the seven-variable exact-QP active-set optimizer directly, with no IPOPT
    continuation stages; the direct mechanistic route retains its smoothing
    continuation inside that one local attempt. Per-attempt and per-case
    checkpoints permit exact resume, and scientific validation failures are
    recorded without preventing later cases from running.
    """

    if profile != ARTICLE_FULL or profile.robustness_count != 10:
        raise RuntimeError("optimization is restricted to the article 4,000/1,000 profile")
    if not assessment_gate_allows_optimization(analysis.passed):
        raise RuntimeError("optimization cannot bypass the enforced admission gate")
    source_id = source_digest(source_files)
    analysis_id = _assessment_binding(design, development_targets, test_targets)
    case_inputs = [
        ("nominal", np.asarray(NOMINAL_INFLUENT, dtype=float)),
        *(
            (f"robustness_{index + 1:02d}", np.asarray(row, dtype=float))
            for index, row in enumerate(np.asarray(design["robustness_influents"]))
        ),
    ]
    if len(case_inputs) != 11:
        raise RuntimeError("the full article requires nominal plus ten robustness cases")
    development_decisions = np.asarray(design["development_decisions"], dtype=float)
    development_influents = np.asarray(design["development_influents"], dtype=float)
    selected_physical_frames: list[pd.DataFrame] = []
    selected_count = 0
    selected_equivalence_passed = True
    route_statuses: list[dict[str, Any]] = []
    shared_surrogate_problem: Any | None = None
    for case_number, (case_id, influent) in enumerate(case_inputs, start=1):
        _write_state(
            run, "optimization", "running", case=case_id,
            completed_cases=case_number - 1, total_cases=len(case_inputs),
        )
        case_directory = run / "optimization" / case_id
        case_directory.mkdir(parents=True, exist_ok=True)
        case_contract = _case_contract_id(source_id, analysis_id, case_id, influent)
        restored = _load_case_checkpoint(
            run, case_directory, case_contract=case_contract,
        )
        if restored is not None:
            marker, violations = restored
            if not violations.empty:
                selected_physical_frames.append(violations)
            selected_count += int(marker.get("selected_decision_count", 0))
            selected_equivalence_passed = bool(
                selected_equivalence_passed
                and marker.get("all_available_selected_equivalence_accepted", False)
            )
            route_statuses.extend(marker.get("routes", []))
            continue
        if shared_surrogate_problem is None:
            shared_surrogate_problem = build_surrogate_nlp(
                analysis.surrogate_assets,
                GAP_CONTINUATION[-1],
                settings=SurrogateSolverSettings(maximum_wall_time=None),
                name="article_surrogate_exact_qp_expressions",
                compile_solver=False,
            )
        surrogate, surrogate_payload = _run_surrogate_route(
            case_directory, case_id=case_id, influent=influent,
            assets=analysis.surrogate_assets, source_id=source_id,
            analysis_id=analysis_id, problem=shared_surrogate_problem,
        )
        direct, direct_payload = _run_direct_route(
            case_directory, case_id=case_id, influent=influent,
            assets=analysis.direct_assets,
            development_decisions=development_decisions,
            development_influents=development_influents,
            development_targets=development_targets,
            source_id=source_id, analysis_id=analysis_id,
        )
        route_rows: list[dict[str, Any]] = []
        case_frames: list[pd.DataFrame] = []
        case_selected_count = 0
        case_equivalence_passed = True
        for route, result, payload in (
            ("surrogate", surrogate, surrogate_payload),
            ("direct", direct, direct_payload),
        ):
            selected, accepted, violations = _evaluate_selected_route(
                case_directory, case_id=case_id, route=route,
                influent=influent, result=result, route_payload=payload,
                analysis=analysis, source_id=source_id, analysis_id=analysis_id,
            )
            if selected:
                case_selected_count += 1
                case_equivalence_passed = case_equivalence_passed and accepted
                case_frames.append(violations)
            route_rows.append({
                "case": case_id,
                "route": route,
                "status": result.status,
                "starts_attempted": len(result.starts),
                "optimization_attempts": len(result.starts),
                "protocol": payload.get("protocol"),
                "selected": selected,
                "selected_cross_evaluation_accepted": accepted if selected else None,
            })
        case_violations = (
            pd.concat(case_frames, ignore_index=True, sort=False)
            if case_frames else pd.DataFrame()
        )
        violation_path = case_directory / "physical_violations.csv"
        if not case_violations.empty:
            atomic_dataframe(violation_path, case_violations)
            selected_physical_frames.append(case_violations)
        selected_count += case_selected_count
        selected_equivalence_passed = (
            selected_equivalence_passed and case_equivalence_passed
        )
        route_statuses.extend(route_rows)
        assert_source_unchanged(source_files)
        artifact_paths = tuple(
            path for path in sorted(case_directory.rglob("*"))
            if path.is_file() and path.name != "case_complete.json"
        )
        atomic_json(case_directory / "case_complete.json", {
            "stage": "optimization_case",
            "case": case_id,
            "case_contract": case_contract,
            "routes": route_rows,
            "selected_decision_count": case_selected_count,
            "all_available_selected_equivalence_accepted": case_equivalence_passed,
            "artifacts": _artifact_hashes(run, artifact_paths),
        })
    _write_state(
        run, "untouched_test_equivalence", "running",
        completed_cases=len(case_inputs), total_rows=len(test_targets),
    )
    test_equivalence_passed, test_equivalence, test_physical = (
        _run_untouched_test_equivalence(
            run, design, test_targets, analysis,
            source_files=source_files, analysis_id=analysis_id,
            parallel_workers=profile.parallel_workers,
        )
    )
    del test_equivalence
    _write_state(
        run, "inference_timing", "running",
        warmups=INFERENCE_TIMING_WARMUPS,
        timed_batches=INFERENCE_TIMING_BATCHES,
        response_count=len(test_targets),
    )
    _run_inference_timing_benchmark(
        run,
        design,
        analysis,
        source_files=source_files,
        analysis_id=analysis_id,
    )
    selected_physical = (
        pd.concat(selected_physical_frames, ignore_index=True, sort=False)
        if selected_physical_frames else pd.DataFrame(columns=("case", "method"))
    )
    atomic_dataframe(
        run / "metrics" / "physical_violations_selected_cases.csv",
        selected_physical,
    )
    # A combined ledger gives the manuscript one explicit location containing
    # raw/projected/mechanistic assessment rows and smooth/reference validation.
    assessment_physical = pd.read_csv(
        run / "metrics" / "physical_violations_assessment.csv"
    )
    combined_frames = []
    for scope, frame in (
        ("untouched_test", assessment_physical),
        ("untouched_test_equivalence", test_physical),
        ("selected_decisions", selected_physical),
    ):
        item = frame.copy()
        item.insert(0, "analysis_scope", scope)
        combined_frames.append(item)
    atomic_dataframe(
        run / "metrics" / "physical_violations_all_analysis.csv",
        pd.concat(combined_frames, ignore_index=True, sort=False),
    )
    expected_cases = tuple(case_id for case_id, _ in case_inputs)
    report = write_reporting_tables(
        run,
        output_directory=run / "report" / "tables",
        expected_cases=expected_cases,
    )
    report_manifest = run / "report" / "tables" / "report_manifest.json"
    scientific_passed = bool(
        test_equivalence_passed
        and selected_equivalence_passed
        and selected_count == 2 * len(case_inputs)
    )
    final_status_path = run / "optimization" / "final_status.json"
    atomic_json(final_status_path, {
        "case_count": len(case_inputs),
        "route_count": len(route_statuses),
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "required_attempts_per_route": 1,
        "required_starts_per_route": 1,
        "surrogate_ipopt_continuation_stage_count": 0,
        "direct_smoothing_continuation_stage_count": len(CONTINUATION_SCHEDULE),
        "wall_time_ceiling": None,
        "untouched_test_equivalence_count": len(test_targets),
        "untouched_test_equivalence_all_accepted": test_equivalence_passed,
        "selected_decision_count": selected_count,
        "all_available_selected_equivalence_accepted": selected_equivalence_passed,
        "scientific_validation_passed": scientific_passed,
        "report_warning_count": len(report.warnings),
        "routes": route_statuses,
    }, nonfinite_to_none=True)
    assert_source_unchanged(source_files)
    final_paths = (
        run / "metrics" / "physical_violations_selected_cases.csv",
        run / "metrics" / "physical_violations_all_analysis.csv",
        run / "metrics" / "smooth_reference_equivalence_test.csv",
        run / "metrics" / "physical_violations_equivalence_test.csv",
        final_status_path,
        report_manifest,
        run / "metrics" / "smooth_reference_test_complete.json",
        run / "metrics" / "inference_timing_complete.json",
        run / "metrics" / "inference_timing_batches.csv",
        run / "metrics" / "timing_events.csv",
        run / "metrics" / "inference_timing_summary.json",
        *(run / "optimization" / case_id / "case_complete.json"
          for case_id, _ in case_inputs),
        *(path for path in sorted((run / "report" / "tables").glob("*.csv"))),
    )
    atomic_json(run / "optimization" / "optimization_complete.json", {
        "stage": "article_optimization_replay_reporting",
        "source_digest": source_id,
        "input_digest": analysis_id,
        "case_count": 11,
        "route_count": 22,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "required_attempts_per_route": 1,
        "selected_decision_count": selected_count,
        "scientific_validation_passed": scientific_passed,
        "artifacts": _artifact_hashes(run, final_paths),
    })
    return scientific_passed


def _prepare_run_directories(run: Path) -> None:
    for relative in (
        "inputs", "datasets", "models", "predictions", "metrics",
        "optimization", "report/tables", "report/figures",
    ):
        (run / relative).mkdir(parents=True, exist_ok=True)


def _write_state(run: Path, stage: str, status: str, **details: Any) -> None:
    atomic_json(run / "run_state.json", {
        "stage": stage, "status": status, "pid": os.getpid(), **details,
    })


def main(
    run_id: str,
    through: str,
    *,
    authorize_generation_replacement_migration: bool = False,
    authorize_assessment_recovery_migration: bool = False,
    authorize_single_start_exact_qp_migration: bool = False,
) -> None:
    if through not in {"generation", "assessment", "complete"}:
        raise ValueError("through must be generation, assessment, or complete")
    validate_authorized_profile(ARTICLE_FULL)
    run = resolve_run_directory(run_id)
    _prepare_run_directories(run)
    source_files = source_file_digests()
    contract = _build_contract(run_id, ARTICLE_FULL, source_files)
    establish_contract(
        run,
        contract,
        authorize_generation_replacement_migration=(
            authorize_generation_replacement_migration
        ),
        authorize_assessment_recovery_migration=(
            authorize_assessment_recovery_migration
        ),
        authorize_single_start_exact_qp_migration=(
            authorize_single_start_exact_qp_migration
        ),
    )
    design = load_or_create_design(run, ARTICLE_FULL)
    assert_source_unchanged(source_files)
    try:
        _write_state(run, "generation", "running")
        generation = run_generation(
            run, design, profile=ARTICLE_FULL, source_files=source_files,
        )
        design = generation.design
        development_targets = generation.development_targets
        test_targets = generation.test_targets
        if through == "generation":
            _write_state(run, "generation", "complete")
            return
        assessment_id = _assessment_binding(design, development_targets, test_targets)
        existing_gate = load_assessment_checkpoint(
            run, source_id=source_digest(source_files), input_id=assessment_id,
        )
        if (
            existing_gate is not None
            and not assessment_gate_allows_optimization(existing_gate["passed"])
        ):
            _write_state(run, "assessment", "admission_gate_failed")
            raise RuntimeError(
                f"full article admission gate failed; see "
                f"{run / 'metrics/admission_gate.json'}"
            )
        if through == "assessment" and existing_gate is not None:
            gate_passed = bool(existing_gate["passed"])
            _write_state(
                run, "assessment",
                "complete" if gate_passed else "complete_with_advisory_failures",
                admission_gate_passed=gate_passed,
                assessment_gate_execution_policy=ASSESSMENT_GATE_EXECUTION_POLICY,
            )
            return
        _write_state(run, "assessment", "running")
        analysis = run_assessment(
            run, design, development_targets, test_targets,
            profile=ARTICLE_FULL, source_files=source_files,
        )
        if not assessment_gate_allows_optimization(analysis.passed):
            _write_state(run, "assessment", "admission_gate_failed")
            raise RuntimeError(
                f"full article admission gate failed; see "
                f"{run / 'metrics/admission_gate.json'}"
            )
        if through == "assessment":
            _write_state(
                run, "assessment",
                "complete" if analysis.passed else "complete_with_advisory_failures",
                admission_gate_passed=bool(analysis.passed),
                assessment_gate_execution_policy=ASSESSMENT_GATE_EXECUTION_POLICY,
            )
            return
        _write_state(run, "optimization", "running")
        scientific_passed = run_optimization_stage(
            run=run, profile=ARTICLE_FULL, design=design,
            development_targets=development_targets, test_targets=test_targets,
            analysis=analysis, source_files=source_files,
        )
        assert_source_unchanged(source_files)
        _write_state(
            run, "complete",
            (
                "complete" if scientific_passed and analysis.passed
                else "complete_with_validation_failures"
            ),
            admission_gate_passed=bool(analysis.passed),
            assessment_gate_execution_policy=ASSESSMENT_GATE_EXECUTION_POLICY,
            optimization_validation_passed=scientific_passed,
            scientific_validation_passed=bool(
                analysis.passed and scientific_passed
            ),
        )
    except Exception as exc:
        state_path = run / "run_state.json"
        current = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file() else {}
        )
        if current.get("status") != "admission_gate_failed":
            _write_state(
                run, str(current.get("stage", "unknown")), "failed",
                error_type=type(exc).__name__, message=str(exc),
            )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", default=os.environ.get("ARTICLE_V3_RUN_ID", DEFAULT_RUN_ID),
    )
    parser.add_argument(
        "--through", choices=("generation", "assessment", "complete"),
        default="complete",
    )
    parser.add_argument(
        "--authorize-generation-replacement-migration",
        action="store_true",
        help=(
            "apply the one-time, pinned article-v3 generation-replacement source "
            "contract migration to the existing default run"
        ),
    )
    parser.add_argument(
        "--authorize-assessment-recovery-migration",
        action="store_true",
        help=(
            "apply the pinned article-v3 projection/audit recovery migration "
            "to the existing default run"
        ),
    )
    parser.add_argument(
        "--authorize-single-start-exact-qp-migration",
        action="store_true",
        help=(
            "apply the pinned article-v3 single-center exact-QP optimization "
            "migration to the existing default run"
        ),
    )
    arguments = parser.parse_args()
    main(
        arguments.run_id,
        arguments.through,
        authorize_generation_replacement_migration=(
            arguments.authorize_generation_replacement_migration
        ),
        authorize_assessment_recovery_migration=(
            arguments.authorize_assessment_recovery_migration
        ),
        authorize_single_start_exact_qp_migration=(
            arguments.authorize_single_start_exact_qp_migration
        ),
    )
