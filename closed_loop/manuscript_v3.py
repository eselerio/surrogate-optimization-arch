"""Executable core for the article/wip_v3 notebook.

This module keeps the notebook readable while implementing the manuscript's
dimension-parametric design, two-start mechanistic generation, five-fold ridge
selection, physical QP deployment, and explicit physical-violation accounting.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter, perf_counter_ns
import json
import math
import os
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from .design import SplitMix64, affine_map, unit_latin_hypercube
from .model import (
    ArticleOperatingPoint,
    ClarifierParameters,
    COMPONENTS,
    COMPOSITE_MATRIX,
    INFLUENT_LOWER,
    INFLUENT_UPPER,
    INVARIANT_MATRIX,
    N_COMPONENTS,
    N_STAGES,
    TSS_VECTOR,
    assemble_target,
    branch_classification,
    generation_scale,
    solve_steady_state,
    target_size,
    unpack_state,
)
from .projection import (
    LogOverflowTSSClosure,
    NetworkLayout,
    PhysicalProjector,
    QuadraticFeatureMap,
    QuadraticSurrogate,
    build_network_operators,
    fit_network_row_scales,
)
from .v3_parallel import BatchProgress, run_resumable_batches


DECISION_NAMES = ("H", "a_3", "a_4", "a_5", "r_I", "r_R", "w")
DECISION_LOWER = np.asarray([6.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.001])
DECISION_UPPER = np.asarray([36.0, 1.0, 1.0, 1.0, 4.0, 1.25, 0.05])
RIDGE_GRID = np.logspace(-8, 2, 11)
OVERFLOW_TSS_LOW_QUANTILE = 0.25
OVERFLOW_TSS_HIGH_QUANTILE = 0.90


@dataclass(frozen=True)
class StudyProfile:
    name: str
    development_count: int
    test_count: int
    robustness_count: int
    layer_count: int
    development_seed: int
    test_seed: int
    robustness_seed: int
    parallel_workers: int
    article_eligible: bool
    enforce_admission_gate: bool = True

    @property
    def response_count(self) -> int:
        """Full layer-resolved mechanistic response width (checkpoint format)."""

        return self.mechanistic_response_count

    @property
    def mechanistic_response_count(self) -> int:
        return (N_STAGES + 3) * N_COMPONENTS + self.layer_count

    @property
    def surrogate_response_count(self) -> int:
        """Reduced operational-response width, independent of layer count."""

        return (N_STAGES + 3) * N_COMPONENTS + 1


@dataclass(frozen=True)
class AssessmentResult:
    metrics: pd.DataFrame
    violations: pd.DataFrame
    qp_diagnostics: pd.DataFrame
    feasibility: pd.DataFrame
    raw: np.ndarray
    projected: np.ndarray
    projected_targets: np.ndarray
    overflow_tss_closure: np.ndarray | None = None


@dataclass(frozen=True)
class RidgeSelectionResult:
    model: QuadraticSurrogate
    scores: pd.DataFrame
    fold_membership: np.ndarray
    out_of_fold_raw: np.ndarray
    elapsed_seconds: float


@dataclass(frozen=True)
class LogOverflowClosureSelectionResult:
    closure: LogOverflowTSSClosure
    scores: pd.DataFrame
    fold_membership: np.ndarray
    out_of_fold_log: np.ndarray
    out_of_fold_tss: np.ndarray
    exact_overflow_tss: np.ndarray
    elapsed_seconds: float


def reduce_mechanistic_responses(
    responses: np.ndarray,
    layer_count: int,
    *,
    layer_volumes_m3: np.ndarray | None = None,
) -> np.ndarray:
    """Map full mechanistic responses to ``(m,c_1,...,c_N,g_E,g_U,M_cl)``.

    The full response remains the immutable generation/checkpoint format.  The
    returned array is the statistical response and replaces all layer-wise TSS
    coordinates by their volume-weighted clarifier solids inventory.
    """

    count = int(layer_count)
    if count < 1 or count != layer_count:
        raise ValueError("layer_count must be a positive integer")
    values = np.asarray(responses, dtype=np.float64)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    shared_count = (N_STAGES + 3) * N_COMPONENTS
    expected = shared_count + count
    if values.ndim != 2 or values.shape[1] != expected or not np.all(np.isfinite(values)):
        raise ValueError(
            f"mechanistic responses must be finite with {expected} coordinates"
    )
    if layer_volumes_m3 is None:
        volumes = np.full(count, 6_000.0 / count)
    else:
        volumes = np.asarray(layer_volumes_m3, dtype=np.float64)
    if (
        volumes.shape != (count,)
        or not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0.0)
    ):
        raise ValueError("layer_volumes_m3 must contain one positive finite volume per layer")
    if not np.all(volumes == volumes[0]):
        raise ValueError(
            "layer_volumes_m3 must be equal because the reduced projection "
            "uses an equal-volume layer envelope"
        )
    inventory = values[:, shared_count:] @ volumes
    reduced = np.concatenate((values[:, :shared_count], inventory[:, None]), axis=1)
    return reduced[0] if single else reduced


TEST_500 = StudyProfile(
    name="test_500_l5", development_count=400, test_count=100,
    robustness_count=5, layer_count=5, development_seed=500_042,
    test_seed=500_043, robustness_seed=500_314_159,
    parallel_workers=max(1, min(12, (os.cpu_count() or 2) - 1)),
    article_eligible=False, enforce_admission_gate=False,
)
ARTICLE_FULL = StudyProfile(
    name="article_full", development_count=4_000, test_count=1_000,
    robustness_count=10, layer_count=10, development_seed=100_042,
    test_seed=100_043, robustness_seed=314_159,
    parallel_workers=max(1, min(12, (os.cpu_count() or 2) - 1)),
    article_eligible=True, enforce_admission_gate=True,
)


def clarifier_for(profile: StudyProfile) -> ClarifierParameters:
    """Preserve the 4 m / 6000 m3 Clarifier while changing numerical layers."""

    return clarifier_for_layers(profile.layer_count)


def _operating(theta: np.ndarray) -> ArticleOperatingPoint:
    return ArticleOperatingPoint(*map(float, theta))


def _v3_unit_latin_hypercube(
    count: int, dimensions: int, seed: int,
) -> tuple[np.ndarray, int, int]:
    """Generate the exact v3 midpoint-jittered, dimension-major LHS."""

    if count < 1 or dimensions < 1:
        raise ValueError("Latin-hypercube dimensions must be positive.")
    stream = SplitMix64(seed)
    coordinates = np.empty((count, dimensions), dtype=float)
    denominator = float(1 << 53)
    upper_open = np.nextafter(1.0, 0.0)
    for dimension in range(dimensions):
        permutation = list(range(count))
        for index in range(count - 1, 0, -1):
            swap = stream.randbelow(index + 1)
            permutation[index], permutation[swap] = permutation[swap], permutation[index]
        for row in range(count):
            jitter = (float(stream.next_uint64() >> 11) + 0.5) / denominator
            coordinates[row, dimension] = min(
                upper_open, (permutation[row] + jitter) / count,
            )
    return coordinates, stream.state, stream.draw_count


def _design_block(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    unit, final_state, draws = _v3_unit_latin_hypercube(count, 27, seed)
    lower = np.concatenate((DECISION_LOWER, INFLUENT_LOWER))
    upper = np.concatenate((DECISION_UPPER, INFLUENT_UPPER))
    physical = lower + unit * (upper - lower)
    return physical[:, :7], physical[:, 7:], {
        "seed": seed, "final_state": final_state, "draw_count": draws,
    }


def create_design(profile: StudyProfile) -> dict[str, object]:
    development = _design_block(profile.development_count, profile.development_seed)
    test = _design_block(profile.test_count, profile.test_seed)
    unit, final_state, draws = _v3_unit_latin_hypercube(
        profile.robustness_count, N_COMPONENTS, profile.robustness_seed
    )
    robust = INFLUENT_LOWER + unit * (INFLUENT_UPPER - INFLUENT_LOWER)
    return {
        "development_decisions": development[0],
        "development_influents": development[1],
        "test_decisions": test[0],
        "test_influents": test[1],
        "robustness_influents": robust,
        "generators": {
            "development": development[2], "test": test[2],
            "robustness": {"seed": profile.robustness_seed,
                           "final_state": final_state, "draw_count": draws},
        },
    }


def _solve_design_row(payload: tuple[int, np.ndarray, np.ndarray, int]) -> dict[str, object]:
    index, theta, influent, layer_count = payload
    clarifier = ClarifierParameters(
        layer_count=layer_count, feed_layer=(layer_count - 1) // 2,
        layer_volume=6_000.0 / layer_count,
    )
    operating = _operating(theta)
    started = perf_counter()
    with threadpool_limits(limits=1):
        first = solve_steady_state(
            operating, influent, starts=(1,), clarifier=clarifier,
            logarithmic_only=True, strict_v3=True,
        )
        second = solve_steady_state(
            operating, influent, starts=(2,), clarifier=clarifier,
            logarithmic_only=True, strict_v3=True,
        )
    first_reactors, _ = unpack_state(first.state, clarifier)
    scale = generation_scale(influent, first_reactors[-1], clarifier)
    root_difference = float(np.max(np.abs(first.state - second.state) / scale))
    first_branches = branch_classification(first.state, clarifier)
    second_branches = branch_classification(second.state, clarifier)
    branch_agreement = first_branches == second_branches
    accepted = bool(
        first.accepted and second.accepted and root_difference <= 1.0e-6
        and branch_agreement
    )
    return {
        "index": index, "accepted": accepted,
        "target": assemble_target(first.state, operating, influent, clarifier),
        "state": first.state, "state_start_2": second.state,
        "elapsed_seconds": perf_counter() - started,
        "root_difference_inf": root_difference,
        "start_1": first.diagnostics, "start_2": second.diagnostics,
        "branch_agreement": branch_agreement,
        "branch_classification": first_branches,
        "routes": [first.route, second.route],
    }


def generate_mechanistic_block(
    decisions: np.ndarray,
    influents: np.ndarray,
    profile: StudyProfile,
    output: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate all fixed rows; any unresolved row stops the fixed design."""

    output.mkdir(parents=True, exist_ok=True)
    rows_directory = output / "rows"
    rows_directory.mkdir(parents=True, exist_ok=True)
    cache = output / "mechanistic_rows_v3.npz"
    diagnostics_path = output / "mechanistic_diagnostics.csv"
    targets = np.full((len(decisions), profile.mechanistic_response_count), np.nan)
    states_start_1 = np.full((len(decisions), N_STAGES * N_COMPONENTS + profile.layer_count), np.nan)
    states_start_2 = np.full_like(states_start_1, np.nan)
    contract_payload = (
        json.dumps(asdict(profile), sort_keys=True).encode("utf-8")
        + np.ascontiguousarray(decisions, dtype="<f8").tobytes()
        + np.ascontiguousarray(influents, dtype="<f8").tobytes()
        + Path(__file__).read_bytes()
        + (Path(__file__).with_name("model.py")).read_bytes()
    )
    contract_hash = sha256(contract_payload).hexdigest()

    def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
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

    def record_from_result(result: dict[str, object]) -> dict[str, object]:
        first = result["start_1"]
        second = result["start_2"]
        assert isinstance(first, dict) and isinstance(second, dict)
        return {
            "row": int(result["index"]),
            "accepted": bool(result["accepted"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "root_difference_inf": float(result["root_difference_inf"]),
            "branch_agreement": bool(result["branch_agreement"]),
            "branch_classification": json.dumps(result["branch_classification"], sort_keys=True),
            "minimum_state_start_1": first["minimum_state"],
            "minimum_state_start_2": second["minimum_state"],
            "state_negativity_start_1": first["v3_state_negativity_max"],
            "state_negativity_start_2": second["v3_state_negativity_max"],
            "rate_negativity_start_1": first["v3_rate_negativity_max"],
            "rate_negativity_start_2": second["v3_rate_negativity_max"],
            "mass_residual_start_1": first["v3_balance_residual"],
            "mass_residual_start_2": second["v3_balance_residual"],
            "largest_real_eigenvalue_start_1": first["largest_real_eigenvalue"],
            "largest_real_eigenvalue_start_2": second["largest_real_eigenvalue"],
            "stability_agreement_start_1": first["stability_eigenvalue_agreement"],
            "stability_agreement_start_2": second["stability_eigenvalue_agreement"],
            "feed_tss_start_1": first["feed_tss_g_m3"],
            "feed_tss_start_2": second["feed_tss_g_m3"],
            "external_solids_loss_start_1": first["external_solids_loss_g_m3"],
            "external_solids_loss_start_2": second["external_solids_loss_g_m3"],
            "route_start_1": result["routes"][0],
            "route_start_2": result["routes"][1],
        }

    records_by_row: dict[int, dict[str, object]] = {}
    completed_rows: set[int] = set()
    for index in range(len(decisions)):
        path = rows_directory / f"row_{index:06d}.npz"
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as stored:
                valid = bool(
                    str(stored["contract_hash"].item()) == contract_hash
                    and np.array_equal(stored["decision"], decisions[index])
                    and np.array_equal(stored["influent"], influents[index])
                    and stored["target"].shape == (profile.mechanistic_response_count,)
                    and stored["state_start_1"].shape == (states_start_1.shape[1],)
                    and stored["state_start_2"].shape == (states_start_2.shape[1],)
                )
                if not valid:
                    continue
                targets[index] = stored["target"]
                states_start_1[index] = stored["state_start_1"]
                states_start_2[index] = stored["state_start_2"]
                record = json.loads(str(stored["record_json"].item()))
            if bool(record["accepted"]) and np.isfinite(targets[index]).all():
                records_by_row[index] = record
                completed_rows.add(index)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue

    payloads = [
        (i, decisions[i], influents[i], profile.layer_count)
        for i in range(len(decisions)) if i not in completed_rows
    ]
    if completed_rows:
        print(
            f"[{output.name}] reusing {len(completed_rows)}/{len(decisions)} "
            "accepted row checkpoints",
            flush=True,
        )
    progress_interval = max(1, len(decisions) // 100)
    newly_completed = 0
    with ProcessPoolExecutor(max_workers=profile.parallel_workers) as pool:
        futures = {pool.submit(_solve_design_row, item): item[0] for item in payloads}
        for future in as_completed(futures):
            result = future.result()
            i = int(result["index"])
            targets[i] = np.asarray(result["target"], dtype=float)
            states_start_1[i] = np.asarray(result["state"], dtype=float)
            states_start_2[i] = np.asarray(result["state_start_2"], dtype=float)
            record = record_from_result(result)
            records_by_row[i] = record
            atomic_npz(
                rows_directory / f"row_{i:06d}.npz",
                contract_hash=np.asarray(contract_hash),
                decision=np.asarray(decisions[i], dtype=float),
                influent=np.asarray(influents[i], dtype=float),
                target=targets[i], state_start_1=states_start_1[i],
                state_start_2=states_start_2[i],
                record_json=np.asarray(json.dumps(record, sort_keys=True)),
            )
            newly_completed += 1
            total_completed = len(completed_rows) + newly_completed
            if total_completed % progress_interval == 0 or total_completed == len(decisions):
                print(
                    f"[{output.name}] mechanistic checkpoints "
                    f"{total_completed}/{len(decisions)}",
                    flush=True,
                )
    diagnostics = pd.DataFrame(
        [records_by_row[index] for index in sorted(records_by_row)]
    ).reset_index(drop=True)
    atomic_npz(
        cache, contract_hash=np.asarray(contract_hash), targets=targets,
        states_start_1=states_start_1, states_start_2=states_start_2,
    )
    temporary_csv = diagnostics_path.with_suffix(".csv.tmp")
    diagnostics.to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, diagnostics_path)
    failures = diagnostics.loc[~diagnostics["accepted"]]
    if len(diagnostics) != len(decisions) or len(failures):
        raise RuntimeError(
            f"{len(failures)} of {len(decisions)} fixed mechanistic design rows failed; "
            "no row was replaced. "
            f"See {diagnostics_path}."
        )
    return targets, diagnostics


def _fold_permutation(count: int, seed: int = 271_828) -> np.ndarray:
    stream = SplitMix64(seed)
    values = list(range(count))
    for i in range(count - 1, 0, -1):
        j = stream.randbelow(i + 1)
        values[i], values[j] = values[j], values[i]
    return np.asarray(values, dtype=int)


def cross_validate_ridge(
    decisions: np.ndarray, influents: np.ndarray, targets: np.ndarray,
) -> RidgeSelectionResult:
    """Five-fold raw-response CV and the manuscript one-standard-error rule."""

    order = _fold_permutation(len(decisions))
    folds = np.array_split(order, 5)
    rows: list[dict[str, float | int]] = []
    predictions: dict[float, np.ndarray] = {
        float(gamma): np.full_like(targets, np.nan, dtype=float) for gamma in RIDGE_GRID
    }
    membership = np.empty(len(decisions), dtype=int)
    started = perf_counter()
    for fold_index, validation in enumerate(folds):
        membership[validation] = fold_index + 1
        fitting = np.setdiff1d(order, validation, assume_unique=True)
        for gamma in RIDGE_GRID:
            model = QuadraticSurrogate.fit_ridge(
                decisions[fitting], influents[fitting], targets[fitting],
                ridge_penalty=float(gamma),
            )
            prediction = model.predict(decisions[validation], influents[validation])
            predictions[float(gamma)][validation] = prediction
            score = float(np.sqrt(np.mean(np.square(
                (prediction - targets[validation]) / model.response_scale
            ))))
            rows.append({"fold": fold_index + 1, "gamma": gamma, "raw_nrmse": score})
    scores = pd.DataFrame(rows)
    summary = scores.groupby("gamma")["raw_nrmse"].agg(["mean", "std"]).reset_index()
    summary["standard_error"] = summary["std"] / math.sqrt(5.0)
    minimum = summary.loc[summary["mean"].idxmin()]
    eligible = summary.loc[summary["mean"] <= minimum["mean"] + minimum["standard_error"]]
    selected = float(eligible["gamma"].max())
    scores = scores.merge(summary, on="gamma", how="left")
    scores["selected"] = scores["gamma"].eq(selected)
    final = QuadraticSurrogate.fit_ridge(
        decisions, influents, targets, ridge_penalty=selected,
    )
    return RidgeSelectionResult(
        model=final, scores=scores, fold_membership=membership,
        out_of_fold_raw=predictions[selected], elapsed_seconds=perf_counter() - started,
    )


def overflow_tss_from_response(
    responses: npt.ArrayLike,
    decisions: npt.ArrayLike,
    layout: NetworkLayout,
) -> np.ndarray:
    states = np.asarray(responses, dtype=np.float64)
    theta = np.asarray(decisions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != layout.state_size:
        raise ValueError("responses have inconsistent dimensions for overflow TSS")
    if theta.shape != (states.shape[0], 7):
        raise ValueError("decisions have inconsistent dimensions for overflow TSS")
    q_effluent = 1.0 - theta[:, 6]
    if np.any(q_effluent <= 0.0):
        raise ValueError("overflow flow must be strictly positive")
    values = states[:, layout.overflow_flow_slice] @ TSS_VECTOR / q_effluent
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("mechanistic overflow TSS must be finite and strictly positive")
    return values


def cross_validate_log_overflow_closure(
    decisions: np.ndarray,
    influents: np.ndarray,
    targets: np.ndarray,
    *,
    layout: NetworkLayout | None = None,
    reference_concentration: float = 1.0,
) -> LogOverflowClosureSelectionResult:
    """Select and refit the development-only quadratic log-overflow closure."""

    layout = layout or NetworkLayout()
    exact = overflow_tss_from_response(targets, decisions, layout)
    log_exact = np.log(exact / float(reference_concentration))
    order = _fold_permutation(len(decisions))
    folds = np.array_split(order, 5)
    rows: list[dict[str, float | int]] = []
    log_predictions = {
        float(gamma): np.full(len(decisions), np.nan, dtype=np.float64)
        for gamma in RIDGE_GRID
    }
    membership = np.empty(len(decisions), dtype=int)
    started = perf_counter()
    for fold_index, validation in enumerate(folds):
        membership[validation] = fold_index + 1
        fitting = np.setdiff1d(order, validation, assume_unique=True)
        for gamma in RIDGE_GRID:
            closure = LogOverflowTSSClosure.fit_ridge(
                decisions[fitting],
                influents[fitting],
                exact[fitting],
                ridge_penalty=float(gamma),
                reference_concentration=reference_concentration,
            )
            prediction = np.asarray(
                closure.predict_log(decisions[validation], influents[validation]),
                dtype=np.float64,
            )
            log_predictions[float(gamma)][validation] = prediction
            score = float(np.sqrt(np.mean(np.square(prediction - log_exact[validation]))))
            rows.append({"fold": fold_index + 1, "gamma": gamma, "log_rmse": score})
    scores = pd.DataFrame(rows)
    summary = scores.groupby("gamma")["log_rmse"].agg(["mean", "std"]).reset_index()
    summary["standard_error"] = summary["std"] / math.sqrt(5.0)
    minimum = summary.loc[summary["mean"].idxmin()]
    eligible = summary.loc[summary["mean"] <= minimum["mean"] + minimum["standard_error"]]
    selected = float(eligible["gamma"].max())
    scores = scores.merge(summary, on="gamma", how="left")
    scores["selected"] = scores["gamma"].eq(selected)
    final = LogOverflowTSSClosure.fit_ridge(
        decisions,
        influents,
        exact,
        ridge_penalty=selected,
        reference_concentration=reference_concentration,
    )
    selected_log = log_predictions[selected]
    selected_tss = reference_concentration * np.exp(selected_log)
    if not np.all(np.isfinite(selected_tss)) or np.any(selected_tss <= 0.0):
        raise RuntimeError("out-of-fold log-overflow predictions are invalid")
    return LogOverflowClosureSelectionResult(
        closure=final,
        scores=scores,
        fold_membership=membership,
        out_of_fold_log=selected_log,
        out_of_fold_tss=selected_tss,
        exact_overflow_tss=exact,
        elapsed_seconds=perf_counter() - started,
    )


def select_ridge(
    decisions: np.ndarray, influents: np.ndarray, targets: np.ndarray,
) -> tuple[QuadraticSurrogate, pd.DataFrame]:
    """Compatibility wrapper around :func:`cross_validate_ridge`."""

    result = cross_validate_ridge(decisions, influents, targets)
    return result.model, result.scores


def _operators(
    theta: np.ndarray,
    influent: np.ndarray,
    layout: NetworkLayout,
    overflow_tss_closure: float | None = None,
):
    return build_network_operators(
        influent, internal_recycle=float(theta[4]),
        return_recycle=float(theta[5]), waste_fraction=float(theta[6]),
        invariant_operator=INVARIANT_MATRIX, tss_weights=TSS_VECTOR,
        layout=layout, overflow_tss_closure=overflow_tss_closure,
    )


def response_coordinate_names(layout: NetworkLayout) -> tuple[str, ...]:
    names = [f"mixer:{name}" for name in COMPONENTS]
    for stage in range(layout.stage_count):
        names.extend(f"reactor_{stage + 1}:{name}" for name in COMPONENTS)
    names.extend(f"overflow_flow:{name}" for name in COMPONENTS)
    names.extend(f"underflow_flow:{name}" for name in COMPONENTS)
    names.append("clarifier_inventory:TSS_mass")
    return tuple(names)


def violation_record(
    method: str, case: str, state: np.ndarray, theta: np.ndarray,
    influent: np.ndarray, layout: NetworkLayout, equality_scale: np.ndarray,
    inequality_scale: np.ndarray, state_scale: np.ndarray,
    overflow_tss_closure: float | None = None,
) -> dict[str, object]:
    operators = _operators(
        theta, influent, layout, overflow_tss_closure=overflow_tss_closure,
    )
    equality_physical = operators.equality_matrix @ state - operators.equality_rhs
    equality = equality_physical / equality_scale
    inequality_physical = operators.inequality_matrix @ state
    inequality = inequality_physical / inequality_scale
    particulate_inequality = inequality[: len(layout.particulate_indices)]
    inventory_inequality = inequality[len(layout.particulate_indices) :]
    negative = np.maximum(-state / state_scale, 0.0)
    invariant_count = INVARIANT_MATRIX.shape[0]
    family_slices = {
        "mixer_component": slice(0, N_COMPONENTS),
        "reactor_invariant": slice(
            N_COMPONENTS, N_COMPONENTS + layout.stage_count * invariant_count,
        ),
        "clarifier_component": slice(
            N_COMPONENTS + layout.stage_count * invariant_count,
            2 * N_COMPONENTS + layout.stage_count * invariant_count,
        ),
        "soluble_passthrough": slice(
            2 * N_COMPONENTS + layout.stage_count * invariant_count,
            2 * N_COMPONENTS + layout.stage_count * invariant_count + len(layout.soluble_indices),
        ),
    }
    if overflow_tss_closure is not None:
        family_slices["overflow_tss_closure"] = slice(
            equality_physical.size - 1, equality_physical.size,
        )
    family_maxima = {
        name: float(np.max(np.abs(equality[index])))
        for name, index in family_slices.items()
    }
    final = state[layout.reactor_slice(layout.stage_count - 1)]
    overflow = state[layout.overflow_flow_slice]
    underflow = state[layout.underflow_flow_slice]
    q_u = float(theta[5] + theta[6])
    external_terms = np.vstack((
        INVARIANT_MATRIX @ influent,
        -(INVARIANT_MATRIX @ overflow),
        -(theta[6] / q_u) * (INVARIANT_MATRIX @ underflow),
    ))
    external_scaled = np.abs(np.sum(external_terms, axis=0)) / np.maximum(
        1.0, np.max(np.abs(external_terms), axis=0),
    )
    family_maxima["external_invariant"] = float(np.max(external_scaled))
    base_equality_count = equality.size - int(overflow_tss_closure is not None)
    combined_mass = np.concatenate((np.abs(equality[:base_equality_count]), external_scaled))
    names = response_coordinate_names(layout)
    negative_indices = np.flatnonzero(negative > 1.0e-10)
    # A reduced response contains no internal layer profile.  Nonlinear layer
    # balances and settling-flux audits are therefore performed only on the
    # retained full mechanistic state, never reconstructed from this vector.
    nonlinear_status = "not_applicable_to_reduced_response"
    nonlinear_max = math.nan
    rate_negativity = math.nan
    return {
        "case": case, "method": method,
        "mass_conservation_violation_max": float(np.max(combined_mass)),
        "mass_conservation_violation_count": int(np.count_nonzero(combined_mass > 1e-8)),
        **{f"mass_{name}_max": value for name, value in family_maxima.items()},
        "mass_physical_residual_max": float(np.max(np.abs(equality_physical))),
        "network_inequality_violation_max": float(np.max(np.maximum(inequality, 0.0))),
        "network_inequality_violation_count": int(np.count_nonzero(inequality > 1e-8)),
        "particulate_densification_violation_max": float(
            np.max(np.maximum(particulate_inequality, 0.0))
        ),
        "clarifier_inventory_bound_violation_max": float(
            np.max(np.maximum(inventory_inequality, 0.0))
        ),
        "overflow_tss_closure_mg_L": (
            math.nan if overflow_tss_closure is None else float(overflow_tss_closure)
        ),
        "overflow_tss_closure_residual_g_m3": (
            math.nan
            if overflow_tss_closure is None
            else float(
                (TSS_VECTOR @ overflow) / (1.0 - float(theta[6]))
                - float(overflow_tss_closure)
            )
        ),
        "overflow_tss_closure_scaled_residual": (
            math.nan if overflow_tss_closure is None else float(equality[-1])
        ),
        "nonnegativity_violation_max": float(np.max(negative)),
        "nonnegativity_violation_count": int(np.count_nonzero(negative > 1e-10)),
        "negative_coordinates": ";".join(names[index] for index in negative_indices),
        "minimum_coordinate": float(np.min(state)),
        "nonlinear_balance_residual_max": nonlinear_max,
        "rate_nonnegativity_violation_max": rate_negativity,
        "nonlinear_audit_status": nonlinear_status,
    }


def clarifier_for_layers(layer_count: int) -> ClarifierParameters:
    """Return the fixed 4 m/6000 m3 vessel discretized into ``layer_count`` cells."""

    return ClarifierParameters(
        layer_count=layer_count,
        feed_layer=(layer_count - 1) // 2,
        layer_volume=6_000.0 / layer_count,
    )


def _response_blocks(layout: NetworkLayout) -> dict[str, np.ndarray]:
    blocks: dict[str, np.ndarray] = {
        "mixer": np.arange(layout.mixer_slice.start, layout.mixer_slice.stop),
    }
    for stage in range(layout.stage_count):
        block = layout.reactor_slice(stage)
        blocks[f"reactor_{stage + 1}"] = np.arange(block.start, block.stop)
    blocks["clarifier_overflow"] = np.arange(
        layout.overflow_flow_slice.start, layout.overflow_flow_slice.stop,
    )
    blocks["clarifier_underflow"] = np.arange(
        layout.underflow_flow_slice.start, layout.underflow_flow_slice.stop,
    )
    blocks["clarifier_inventory"] = np.asarray([layout.inventory_index])
    blocks["clarifier_complete"] = np.arange(
        layout.overflow_flow_slice.start, layout.inventory_slice.stop,
    )
    blocks["complete_response"] = np.arange(layout.state_size)
    return blocks


def _prediction_metric_rows(
    method: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
    layout: NetworkLayout,
) -> list[dict[str, object]]:
    names = response_coordinate_names(layout)
    rows: list[dict[str, object]] = []
    error = prediction - reference
    standardized = error / scale
    for block_name, indices in _response_blocks(layout).items():
        block_error = error[:, indices]
        block_standardized = standardized[:, indices]
        target = reference[:, indices]
        target_centered = target - np.mean(target, axis=0)
        ss_res = np.sum(np.square(block_error), axis=0)
        ss_tot = np.sum(np.square(target_centered), axis=0)
        coordinate_r2 = np.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, np.nan)
        rows.append({
            "method": method, "block": block_name, "coordinate": "ALL",
            "sample_count": int(reference.shape[0]),
            "coordinate_count": int(indices.size),
            "rmse": float(np.sqrt(np.mean(np.square(block_error)))),
            "mae": float(np.mean(np.abs(block_error))),
            "bias": float(np.mean(block_error)),
            "nrmse": float(np.sqrt(np.mean(np.square(block_standardized)))),
            "nmae": float(np.mean(np.abs(block_standardized))),
            "r2_mean": float(np.nanmean(coordinate_r2)),
        })
        for local, coordinate in enumerate(indices):
            coordinate_error = block_error[:, local]
            coordinate_standardized = block_standardized[:, local]
            rows.append({
                "method": method, "block": block_name,
                "coordinate": names[int(coordinate)],
                "sample_count": int(reference.shape[0]), "coordinate_count": 1,
                "rmse": float(np.sqrt(np.mean(np.square(coordinate_error)))),
                "mae": float(np.mean(np.abs(coordinate_error))),
                "bias": float(np.mean(coordinate_error)),
                "nrmse": float(np.sqrt(np.mean(np.square(coordinate_standardized)))),
                "nmae": float(np.mean(np.abs(coordinate_standardized))),
                "r2_mean": float(coordinate_r2[local]),
            })
    return rows


def _overflow_closure_metric_rows(
    prediction: np.ndarray,
    reference: np.ndarray,
    development_reference: np.ndarray,
) -> list[dict[str, object]]:
    """Return concentration/log metrics with development-frozen tail strata."""

    predicted = np.asarray(prediction, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    development = np.asarray(development_reference, dtype=np.float64)
    if (
        predicted.shape != exact.shape
        or predicted.ndim != 1
        or development.ndim != 1
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(exact))
        or not np.all(np.isfinite(development))
        or np.any(predicted <= 0.0)
        or np.any(exact <= 0.0)
        or np.any(development <= 0.0)
    ):
        raise ValueError("overflow closure metric inputs must be finite positive vectors")
    low = float(np.quantile(development, OVERFLOW_TSS_LOW_QUANTILE))
    high = float(np.quantile(development, OVERFLOW_TSS_HIGH_QUANTILE))
    strata = (
        ("all", np.ones(len(exact), dtype=bool), np.ones(len(development), dtype=bool)),
        ("low_q25", exact <= low, development <= low),
        ("upper_q90", exact >= high, development >= high),
    )
    rows: list[dict[str, object]] = []
    for stratum, mask, development_mask in strata:
        for scale_name, values, targets, development_values in (
            ("concentration", predicted, exact, development),
            ("log", np.log(predicted), np.log(exact), np.log(development)),
        ):
            count = int(np.count_nonzero(mask))
            if count:
                error = values[mask] - targets[mask]
                centered = targets[mask] - np.mean(targets[mask])
                ss_tot = float(np.sum(np.square(centered)))
                frozen_scale = max(
                    1.0e-12,
                    float(np.std(development_values[development_mask], ddof=0)),
                )
                rmse = float(np.sqrt(np.mean(np.square(error))))
                mae = float(np.mean(np.abs(error)))
                bias = float(np.mean(error))
                r2 = (
                    math.nan
                    if ss_tot <= 0.0
                    else float(1.0 - np.sum(np.square(error)) / ss_tot)
                )
            else:
                rmse = mae = bias = r2 = frozen_scale = math.nan
            rows.append({
                "method": "log_overflow_closure",
                "block": (
                    "clarifier_overflow_tss"
                    if scale_name == "concentration"
                    else "clarifier_overflow_tss_log"
                ),
                "coordinate": (
                    "TSS" if scale_name == "concentration" else "log(TSS/1mgL)"
                ),
                "stratum": stratum,
                "sample_count": count,
                "coordinate_count": 1,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "nrmse": rmse / frozen_scale if count else math.nan,
                "nmae": mae / frozen_scale if count else math.nan,
                "r2_mean": r2,
                "development_low_q25_mg_L": low,
                "development_upper_q90_mg_L": high,
            })
    return rows


_HOLDOUT_PROJECTION_CONTEXT: tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    NetworkLayout,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    PhysicalProjector,
] | None = None


def _initialize_holdout_projection_worker(
    raw: np.ndarray,
    targets: np.ndarray,
    decisions: np.ndarray,
    influents: np.ndarray,
    layout: NetworkLayout,
    state_scale: np.ndarray,
    equality_scale: np.ndarray,
    inequality_scale: np.ndarray,
    overflow_tss_closure: np.ndarray | None,
) -> None:
    global _HOLDOUT_PROJECTION_CONTEXT
    _HOLDOUT_PROJECTION_CONTEXT = (
        np.asarray(raw, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(decisions, dtype=np.float64),
        np.asarray(influents, dtype=np.float64),
        layout,
        np.asarray(state_scale, dtype=np.float64),
        np.asarray(equality_scale, dtype=np.float64),
        np.asarray(inequality_scale, dtype=np.float64),
        (
            None
            if overflow_tss_closure is None
            else np.asarray(overflow_tss_closure, dtype=np.float64)
        ),
        PhysicalProjector(
            state_scale,
            equality_scale,
            inequality_scale,
            absolute_tolerance=1.0e-12,
            relative_tolerance=1.0e-12,
        ),
    )


def _holdout_projection_batch(bounds: tuple[int, int]) -> Mapping[str, np.ndarray]:
    if _HOLDOUT_PROJECTION_CONTEXT is None:
        raise RuntimeError("holdout-projection worker was not initialized")
    (
        raw,
        targets,
        decisions,
        influents,
        layout,
        state_scale,
        equality_scale,
        inequality_scale,
        overflow_tss_closure,
        projector,
    ) = _HOLDOUT_PROJECTION_CONTEXT
    start, stop = bounds
    count = stop - start
    projected = np.empty((count, layout.state_size), dtype=np.float64)
    projected_targets = np.empty_like(projected)
    qp_rows: list[str] = []
    feasibility_rows: list[str] = []
    violation_rows: list[str] = []
    for local, row in enumerate(range(start, stop)):
        closure_value = (
            None
            if overflow_tss_closure is None
            else float(overflow_tss_closure[row])
        )
        operators = _operators(
            decisions[row], influents[row], layout,
            overflow_tss_closure=closure_value,
        )
        started = perf_counter_ns()
        projection = projector.project(
            raw[row],
            operators.equality_matrix,
            operators.equality_rhs,
            operators.inequality_matrix,
            raise_on_failure=False,
        )
        qp_elapsed = perf_counter_ns() - started
        projected[local] = projection.state
        target_projection = projector.project(
            targets[row],
            operators.equality_matrix,
            operators.equality_rhs,
            operators.inequality_matrix,
            raise_on_failure=False,
        )
        projected_targets[local] = target_projection.state
        for kind, result, elapsed in (
            ("raw_prediction", projection, qp_elapsed),
            ("mechanistic_target", target_projection, math.nan),
        ):
            qp_rows.append(json.dumps({
                "row": row,
                "projection_input": kind,
                "accepted": bool(result.accepted),
                "elapsed_ns": elapsed,
                **result.diagnostics.as_dict(),
            }))
        raw_distance = float(np.linalg.norm(
            (raw[row] - targets[row]) / state_scale
        ))
        projected_distance = float(np.linalg.norm(
            (projected[local] - targets[row]) / state_scale
        ))
        target_feasibility = float(np.linalg.norm(
            (projected_targets[local] - targets[row]) / state_scale
        ))
        upper_bound = raw_distance + target_feasibility
        feasibility_rows.append(json.dumps({
            "row": row,
            "target_feasibility_distance": target_feasibility,
            "raw_distance": raw_distance,
            "projected_distance": projected_distance,
            "finite_feasibility_bound": upper_bound,
            "bound_slack": upper_bound - projected_distance,
            "bound_passed": bool(projected_distance <= upper_bound + 1.0e-10),
            "raw_projection_qp_passed": bool(projection.accepted),
            "target_projection_qp_passed": bool(target_projection.accepted),
        }))
        for method, state in (
            ("raw", raw[row]),
            ("projected", projected[local]),
            ("mechanistic", targets[row]),
        ):
            violation_rows.append(json.dumps(violation_record(
                method,
                f"test_{row:04d}",
                state,
                decisions[row],
                influents[row],
                layout,
                equality_scale,
                inequality_scale,
                state_scale,
                closure_value,
            )))
    return {
        "projected": projected,
        "projected_targets": projected_targets,
        "qp_json": np.asarray(qp_rows),
        "feasibility_json": np.asarray(feasibility_rows),
        "violations_json": np.asarray(violation_rows),
    }


def _validate_holdout_projection_batch(
    start: int, stop: int, payload: Mapping[str, np.ndarray],
) -> None:
    count = stop - start
    projected = np.asarray(payload["projected"])
    projected_targets = np.asarray(payload["projected_targets"])
    if (
        projected.ndim != 2
        or projected.shape[0] != count
        or projected_targets.shape != projected.shape
        or np.asarray(payload["qp_json"]).shape != (2 * count,)
        or np.asarray(payload["feasibility_json"]).shape != (count,)
        or np.asarray(payload["violations_json"]).shape != (3 * count,)
    ):
        raise ValueError("holdout-projection batch payload has invalid dimensions")
    qp = [json.loads(str(value)) for value in np.asarray(payload["qp_json"])]
    feasibility = [
        json.loads(str(value)) for value in np.asarray(payload["feasibility_json"])
    ]
    violations = [
        json.loads(str(value)) for value in np.asarray(payload["violations_json"])
    ]
    expected_rows = list(range(start, stop))
    if [int(item.get("row", -1)) for item in qp] != [
        row for row in expected_rows for _ in range(2)
    ] or [item.get("projection_input") for item in qp] != [
        kind
        for _ in expected_rows
        for kind in ("raw_prediction", "mechanistic_target")
    ]:
        raise ValueError("holdout-projection QP row ordering is invalid")
    if [int(item.get("row", -1)) for item in feasibility] != expected_rows:
        raise ValueError("holdout-projection feasibility row ordering is invalid")
    if [item.get("case") for item in violations] != [
        f"test_{row:04d}" for row in expected_rows for _ in range(3)
    ] or [item.get("method") for item in violations] != [
        method
        for _ in expected_rows
        for method in ("raw", "projected", "mechanistic")
    ]:
        raise ValueError("holdout-projection violation row ordering is invalid")


def assess_raw_projected_mechanistic(
    model: QuadraticSurrogate,
    development_decisions: np.ndarray,
    development_influents: np.ndarray,
    development_targets: np.ndarray,
    test_decisions: np.ndarray,
    test_influents: np.ndarray,
    test_targets: np.ndarray,
    profile: StudyProfile,
    *,
    overflow_closure: LogOverflowTSSClosure | None = None,
    development_overflow_tss_closure: np.ndarray | None = None,
    parallel_workers: int = 1,
    batch_size: int = 64,
    checkpoint_directory: Path | None = None,
    checkpoint_contract: str | None = None,
    progress: Callable[[BatchProgress], None] | None = None,
) -> AssessmentResult:
    layout = NetworkLayout(layer_count=profile.layer_count)
    state_scale = model.response_scale
    holdout_closure = (
        None
        if overflow_closure is None
        else np.asarray(
            overflow_closure.predict(test_decisions, test_influents),
            dtype=np.float64,
        )
    )
    if overflow_closure is not None:
        development_closure = np.asarray(
            development_overflow_tss_closure
            if development_overflow_tss_closure is not None
            else overflow_closure.predict(development_decisions, development_influents),
            dtype=np.float64,
        )
        if development_closure.shape != (len(development_decisions),):
            raise ValueError("development overflow-TSS closure predictions are invalid")
    else:
        development_closure = None
    row_scales = fit_network_row_scales(
        development_targets, development_influents,
        internal_recycle=development_decisions[:, 4],
        return_recycle=development_decisions[:, 5],
        waste_fraction=development_decisions[:, 6],
        invariant_operator=INVARIANT_MATRIX, tss_weights=TSS_VECTOR,
        layout=layout, minimum_scale=1.0,
        overflow_tss_closure=development_closure,
    )
    raw = model.predict(test_decisions, test_influents)
    if checkpoint_directory is not None and not checkpoint_contract:
        raise ValueError(
            "checkpoint_contract is required when holdout checkpoints are enabled"
        )
    batches = run_resumable_batches(
        stage="whole_system_holdout_projection_audit",
        row_count=len(raw),
        batch_size=batch_size,
        parallel_workers=parallel_workers,
        checkpoint_directory=checkpoint_directory,
        contract_digest=checkpoint_contract or "unpersisted",
        payload_names=(
            "projected",
            "projected_targets",
            "qp_json",
            "feasibility_json",
            "violations_json",
        ),
        worker=_holdout_projection_batch,
        validate=_validate_holdout_projection_batch,
        initializer=_initialize_holdout_projection_worker,
        initargs=(
            raw,
            test_targets,
            test_decisions,
            test_influents,
            layout,
            state_scale,
            row_scales.equality,
            row_scales.inequality,
            holdout_closure,
        ),
        progress=progress,
    )
    projected = np.vstack([batch["projected"] for batch in batches])
    projected_targets = np.vstack(
        [batch["projected_targets"] for batch in batches]
    )
    qp_rows = [
        json.loads(str(record))
        for batch in batches for record in batch["qp_json"]
    ]
    feasibility_rows = [
        json.loads(str(record))
        for batch in batches for record in batch["feasibility_json"]
    ]
    violations = [
        json.loads(str(record))
        for batch in batches for record in batch["violations_json"]
    ]
    metrics: list[dict[str, object]] = []
    for method, values in (("raw", raw), ("projected", projected)):
        metrics.extend(_prediction_metric_rows(
            method, values, test_targets, model.response_scale, layout,
        ))
    if holdout_closure is not None:
        exact_overflow = overflow_tss_from_response(test_targets, test_decisions, layout)
        development_exact_overflow = overflow_tss_from_response(
            development_targets, development_decisions, layout,
        )
        metrics.extend(_overflow_closure_metric_rows(
            holdout_closure,
            exact_overflow,
            development_exact_overflow,
        ))
    correction = (projected - raw) / model.response_scale
    metrics.append({
        "method": "projection_correction", "block": "complete_response",
        "coordinate": "ALL", "sample_count": len(raw),
        "coordinate_count": layout.state_size, "rmse": math.nan, "mae": math.nan,
        "bias": math.nan,
        "nrmse": float(np.sqrt(np.mean(np.square(correction)))),
        "nmae": float(np.mean(np.abs(correction))), "r2_mean": math.nan,
    })
    return AssessmentResult(
        metrics=pd.DataFrame(metrics), violations=pd.DataFrame(violations),
        qp_diagnostics=pd.DataFrame(qp_rows), feasibility=pd.DataFrame(feasibility_rows),
        raw=raw, projected=projected, projected_targets=projected_targets,
        overflow_tss_closure=holdout_closure,
    )


def engineering_quantities(
    theta: np.ndarray, state: np.ndarray, layout: NetworkLayout,
    profile: StudyProfile,
) -> dict[str, float]:
    values = np.asarray(state, dtype=float)
    if values.shape != (layout.state_size,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"state must be a finite reduced response with {layout.state_size} coordinates"
        )
    operating = _operating(theta)
    reactors = np.vstack([values[layout.reactor_slice(i)] for i in range(N_STAGES)])
    g_e = values[layout.overflow_flow_slice]
    g_u = values[layout.underflow_flow_slice]
    clarifier_inventory = float(values[layout.inventory_index])
    feed_tss = float(TSS_VECTOR @ reactors[-1])
    underflow_tss = float(TSS_VECTOR @ g_u / operating.q_underflow)
    external_loss = float(TSS_VECTOR @ g_e + theta[6] * TSS_VECTOR @ g_u / operating.q_underflow)
    reactor_volume = 10_000.0 * theta[0] / (24.0 * N_STAGES)
    inventory = reactor_volume * float(np.sum(reactors @ TSS_VECTOR))
    inventory += clarifier_inventory
    loss_rate = 10_000.0 * external_loss
    return {
        "srt_d": inventory / loss_rate if loss_rate > 0.0 else np.inf,
        "sor_m_d": 10_000.0 * operating.q_effluent / 1_500.0,
        "slr_kg_m2_d": 1.0e-3 * 10_000.0 * operating.q_clarifier * feed_tss / 1_500.0,
        "underflow_tss_g_m3": underflow_tss,
        "feed_tss_g_m3": feed_tss,
        "external_solids_loss_g_m3": external_loss,
    }


def objective_value(
    theta: np.ndarray, state: np.ndarray, layout: NetworkLayout,
    quality_scale: np.ndarray,
) -> float:
    operating = _operating(theta)
    c_e = state[layout.overflow_flow_slice] / operating.q_effluent
    quality = float(np.mean((COMPOSITE_MATRIX @ c_e) / quality_scale))
    hrt = (theta[0] - 6.0) / 30.0
    aeration = theta[0] * float(np.sum(theta[1:4])) / (36.0 * 3.0)
    internal = theta[4] / 4.0
    returned = (theta[5] - 0.25) / 1.0
    underflow_tss = float(TSS_VECTOR @ state[layout.underflow_flow_slice] / operating.q_underflow)
    wasted = theta[6] * underflow_tss / (0.05 * 15_000.0)
    return float(np.dot(
        np.asarray([0.50, 0.15, 0.20, 0.05, 0.05, 0.05]),
        np.asarray([quality, hrt, aeration, internal, returned, wasted]),
    ))


def optimize_surrogate_case(
    model: QuadraticSurrogate,
    influent: np.ndarray,
    development_decisions: np.ndarray,
    development_influents: np.ndarray,
    development_targets: np.ndarray,
    profile: StudyProfile,
    *,
    overflow_closure: LogOverflowTSSClosure | None = None,
    development_overflow_tss_closure: npt.ArrayLike | None = None,
) -> dict[str, object]:
    """Nine-start projected-response optimization with independent final QP replay.

    This is the outer-refinement portion of the article protocol.  Endpoint
    records are deliberately labeled stationarity-unresolved until the full
    primal--dual-gap continuation/KKT audit is executed for the article run.
    """

    if (overflow_closure is None) != (development_overflow_tss_closure is None):
        raise ValueError(
            "overflow_closure and its development predictions must be supplied together"
        )
    layout = NetworkLayout(layer_count=profile.layer_count)
    row_scales = fit_network_row_scales(
        development_targets, development_influents,
        internal_recycle=development_decisions[:, 4],
        return_recycle=development_decisions[:, 5],
        waste_fraction=development_decisions[:, 6],
        invariant_operator=INVARIANT_MATRIX, tss_weights=TSS_VECTOR,
        layout=layout, minimum_scale=1.0,
        overflow_tss_closure=development_overflow_tss_closure,
    )
    projector = PhysicalProjector(
        model.response_scale, row_scales.equality, row_scales.inequality,
        absolute_tolerance=1.0e-12, relative_tolerance=1.0e-12,
    )
    q_e_dev = 1.0 - development_decisions[:, 6]
    c_e_dev = development_targets[:, layout.overflow_flow_slice] / q_e_dev[:, None]
    quality_scale = np.maximum(1.0, np.std(c_e_dev @ COMPOSITE_MATRIX.T, axis=0, ddof=0))
    cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray, float, dict[str, float]]] = {}

    def evaluate(unit_theta: np.ndarray):
        key = tuple(np.round(np.clip(unit_theta, 0.0, 1.0), 12))
        if key not in cache:
            theta = DECISION_LOWER + np.asarray(key) * (DECISION_UPPER - DECISION_LOWER)
            raw = model.predict(theta, influent)
            closure_value = (
                None if overflow_closure is None
                else float(overflow_closure.predict(theta, influent))
            )
            operators = _operators(
                theta, influent, layout,
                overflow_tss_closure=closure_value,
            )
            projection = projector.project(
                raw, operators.equality_matrix, operators.equality_rhs,
                operators.inequality_matrix,
            )
            engineering = engineering_quantities(theta, projection.state, layout, profile)
            objective = objective_value(theta, projection.state, layout, quality_scale)
            cache[key] = theta, projection.state, objective, engineering
        return cache[key]

    def constraints(unit_theta: np.ndarray) -> np.ndarray:
        theta, state, _, eng = evaluate(unit_theta)
        raw = model.predict(theta, influent)
        correction = float(np.linalg.norm((state - raw) / model.response_scale) / math.sqrt(layout.state_size))
        return np.asarray([
            30.0 - eng["srt_d"],
            100.0 - eng["slr_kg_m2_d"],
            15_000.0 - eng["underflow_tss_g_m3"],
            eng["feed_tss_g_m3"] - 1.0,
            eng["external_solids_loss_g_m3"] - 1.0,
            0.50 - correction,
        ])

    unit, _, _ = unit_latin_hypercube(8, 7, seed=271_828)
    starts = np.vstack((np.full(7, 0.5), unit))
    candidates = []
    started = perf_counter()
    for start_index, start in enumerate(starts):
        result = minimize(
            lambda value: evaluate(value)[2], start, method="SLSQP",
            bounds=[(0.0, 1.0)] * 7,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 250, "ftol": 1.0e-10, "disp": False},
        )
        theta, state, objective, engineering = evaluate(result.x)
        feasible = bool(np.min(constraints(result.x)) >= -1.0e-6)
        candidates.append({
            "start": start_index, "success": bool(result.success),
            "feasible": feasible, "objective": objective,
            "theta": theta, "state": state, "engineering": engineering,
            "iterations": int(result.nit), "message": str(result.message),
        })
    feasible = [item for item in candidates if item["feasible"]]
    if not feasible:
        return {"status": "no validated feasible local incumbent", "candidates": candidates}
    selected = min(feasible, key=lambda item: (item["objective"], tuple(item["theta"])))
    selected = dict(selected)
    selected["status"] = "validated feasible local incumbent; stationarity unresolved"
    selected["elapsed_seconds"] = perf_counter() - started
    selected["qp_evaluations"] = len(cache)
    selected["candidates"] = candidates
    return selected


def replay_selected_case(
    case: str, selected: dict[str, object], influent: np.ndarray,
    model: QuadraticSurrogate, development_decisions: np.ndarray,
    development_influents: np.ndarray, development_targets: np.ndarray,
    profile: StudyProfile,
    *,
    overflow_closure: LogOverflowTSSClosure | None = None,
    development_overflow_tss_closure: npt.ArrayLike | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    if "theta" not in selected:
        return {"case": case, "status": selected["status"]}, pd.DataFrame()
    if (overflow_closure is None) != (development_overflow_tss_closure is None):
        raise ValueError(
            "overflow_closure and its development predictions must be supplied together"
        )
    theta = np.asarray(selected["theta"], dtype=float)
    clarifier = clarifier_for(profile)
    operating = _operating(theta)
    first = solve_steady_state(operating, influent, starts=(1,), clarifier=clarifier)
    second = solve_steady_state(operating, influent, starts=(2,), clarifier=clarifier)
    mechanistic = reduce_mechanistic_responses(first.target, profile.layer_count)
    layout = NetworkLayout(layer_count=profile.layer_count)
    row_scales = fit_network_row_scales(
        development_targets, development_influents,
        internal_recycle=development_decisions[:, 4],
        return_recycle=development_decisions[:, 5],
        waste_fraction=development_decisions[:, 6],
        invariant_operator=INVARIANT_MATRIX, tss_weights=TSS_VECTOR,
        layout=layout, minimum_scale=1.0,
        overflow_tss_closure=development_overflow_tss_closure,
    )
    raw = model.predict(theta, influent)
    closure_value = (
        None if overflow_closure is None
        else float(overflow_closure.predict(theta, influent))
    )
    operators = _operators(
        theta, influent, layout, overflow_tss_closure=closure_value,
    )
    projected = PhysicalProjector(
        model.response_scale, row_scales.equality, row_scales.inequality,
        absolute_tolerance=1.0e-12, relative_tolerance=1.0e-12,
    ).project(raw, operators.equality_matrix, operators.equality_rhs,
              operators.inequality_matrix).state
    violations = pd.DataFrame([
        violation_record(method, case, state, theta, influent, layout,
                         row_scales.equality, row_scales.inequality,
                         model.response_scale, closure_value)
        for method, state in (("raw", raw), ("projected", projected),
                              ("mechanistic", mechanistic))
    ])
    return {
        "case": case, "status": selected["status"],
        "theta": theta.tolist(), "surrogate_objective": selected["objective"],
        "two_start_reference_accepted": bool(first.accepted and second.accepted),
        "reference_root_difference_inf": float(np.max(np.abs(first.state - second.state))),
        "raw_to_reference_nrmse": float(np.sqrt(np.mean(((raw - mechanistic) / model.response_scale) ** 2))),
        "projected_to_reference_nrmse": float(np.sqrt(np.mean(((projected - mechanistic) / model.response_scale) ** 2))),
    }, violations


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ARTICLE_FULL", "TEST_500", "DECISION_NAMES", "RIDGE_GRID", "StudyProfile",
    "LogOverflowClosureSelectionResult", "cross_validate_log_overflow_closure",
    "overflow_tss_from_response",
    "assess_raw_projected_mechanistic", "clarifier_for", "create_design",
    "engineering_quantities", "generate_mechanistic_block", "objective_value",
    "optimize_surrogate_case", "replay_selected_case", "select_ridge",
    "reduce_mechanistic_responses", "violation_record", "write_json",
]
