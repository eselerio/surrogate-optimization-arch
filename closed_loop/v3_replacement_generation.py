"""Deterministic, resumable replacement generation for the v3 study.

The original candidate design remains immutable.  Candidates that fail the
two-start mechanistic acceptance contract are retained as audited attempts,
while deterministic supplemental candidates fill their vacated output slots.
Development and test blocks use independent SplitMix64 streams.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterator
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd

from .design import SplitMix64
from . import manuscript_v3 as core
from .model import INFLUENT_LOWER, INFLUENT_UPPER, N_COMPONENTS, N_STAGES


REPLACEMENT_SCHEMA = "article-v3-replacement-generation-v1"
_DIMENSION_COUNT = 27
_FLOAT53_DENOMINATOR = float(1 << 53)


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Publish atomically despite transient Windows file-sharing locks."""

    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            transient = (
                isinstance(error, PermissionError)
                or getattr(error, "winerror", None) in {5, 32, 33}
                or getattr(error, "errno", None) == 13
            )
            if not transient or attempt == 7:
                raise
            time.sleep(0.025 * (2 ** attempt))


@dataclass(frozen=True)
class MechanisticBlockResult:
    """An accepted block plus complete attempt and source provenance."""

    decisions: np.ndarray
    influents: np.ndarray
    targets: np.ndarray
    diagnostics: pd.DataFrame
    attempts: pd.DataFrame
    provenance: pd.DataFrame

    def __iter__(self) -> Iterator[object]:
        """Retain the historical ``targets, diagnostics = result`` interface."""

        yield self.targets
        yield self.diagnostics


@dataclass(frozen=True)
class _Candidate:
    block: str
    round_index: int
    candidate_index: int
    candidate_ordinal: int
    decision: np.ndarray
    influent: np.ndarray
    checkpoint: Path

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.block}:r{self.round_index:06d}:"
            f"c{self.candidate_index:06d}"
        )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
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


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
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


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_block(
    block: str | None,
    decisions: np.ndarray,
    profile: core.StudyProfile,
    output: Path,
) -> tuple[str, int, int]:
    name = block or output.name
    if name not in {"development", "test"}:
        matches = []
        if len(decisions) == profile.development_count:
            matches.append("development")
        if len(decisions) == profile.test_count:
            matches.append("test")
        if len(matches) != 1:
            raise ValueError(
                "block must be 'development' or 'test' when its identity "
                "cannot be inferred uniquely"
            )
        name = matches[0]
    count = (
        profile.development_count if name == "development" else profile.test_count
    )
    seed = profile.development_seed if name == "development" else profile.test_seed
    if len(decisions) != count:
        raise ValueError(f"{name} requires exactly {count} initial candidates")
    return name, count, seed


def _base_contract_hash(
    decisions: np.ndarray,
    influents: np.ndarray,
    profile: core.StudyProfile,
) -> str:
    """Replay the unchanged legacy row-checkpoint contract exactly."""

    payload = (
        json.dumps(asdict(profile), sort_keys=True).encode("utf-8")
        + np.ascontiguousarray(decisions, dtype="<f8").tobytes()
        + np.ascontiguousarray(influents, dtype="<f8").tobytes()
        + Path(core.__file__).read_bytes()
        + Path(core.__file__).with_name("model.py").read_bytes()
    )
    return sha256(payload).hexdigest()


def _replacement_contract_hash(profile: core.StudyProfile, block: str) -> str:
    payload = (
        REPLACEMENT_SCHEMA.encode("utf-8")
        + block.encode("utf-8")
        + json.dumps(asdict(profile), sort_keys=True).encode("utf-8")
        + Path(__file__).read_bytes()
        + Path(core.__file__).read_bytes()
        + Path(core.__file__).with_name("model.py").read_bytes()
    )
    return sha256(payload).hexdigest()


def _record_from_result(
    result: dict[str, object], candidate: _Candidate,
) -> dict[str, object]:
    first = result["start_1"]
    second = result["start_2"]
    assert isinstance(first, dict) and isinstance(second, dict)
    record = {
        "candidate_id": candidate.candidate_id,
        "candidate_round": candidate.round_index,
        "candidate_index": candidate.candidate_index,
        "candidate_ordinal": candidate.candidate_ordinal,
        "accepted": bool(result["accepted"]),
        "attempt_status": "accepted" if result["accepted"] else "rejected",
        "error_type": "",
        "error_message": "",
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "root_difference_inf": float(result["root_difference_inf"]),
        "branch_agreement": bool(result["branch_agreement"]),
        "branch_classification": json.dumps(
            result["branch_classification"], sort_keys=True,
        ),
        "accepted_start_1": bool(first["passed"]),
        "accepted_start_2": bool(second["passed"]),
        "scaled_residual_start_1": first["scaled_residual_inf"],
        "scaled_residual_start_2": second["scaled_residual_inf"],
        "clarifier_component_residual_start_1": first["clarifier_component_residual"],
        "clarifier_component_residual_start_2": second["clarifier_component_residual"],
        "plant_boundary_residual_start_1": first["plant_boundary_residual"],
        "plant_boundary_residual_start_2": second["plant_boundary_residual"],
        "clarifier_tss_residual_start_1": first["clarifier_tss_residual"],
        "clarifier_tss_residual_start_2": second["clarifier_tss_residual"],
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
        "layer_envelope_start_1": bool(first["layer_envelope"]),
        "layer_envelope_start_2": bool(second["layer_envelope"]),
        "finite_nonnegative_rates_start_1": bool(first["finite_nonnegative_rates"]),
        "finite_nonnegative_rates_start_2": bool(second["finite_nonnegative_rates"]),
        "locally_stable_start_1": bool(first["locally_stable"]),
        "locally_stable_start_2": bool(second["locally_stable"]),
        "soluble_passthrough_error_start_1": first["soluble_passthrough_error"],
        "soluble_passthrough_error_start_2": second["soluble_passthrough_error"],
        "route_start_1": result["routes"][0],
        "route_start_2": result["routes"][1],
    }
    return _annotate_rejection(record)


def _finite_exceeds(record: dict[str, object], names: tuple[str, ...], limit: float) -> bool:
    for name in names:
        try:
            value = float(record.get(name, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > limit:
            return True
    return False


def _finite_below(record: dict[str, object], names: tuple[str, ...], limit: float) -> bool:
    for name in names:
        try:
            value = float(record.get(name, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value < limit:
            return True
    return False


def _explicit_false(record: dict[str, object], names: tuple[str, ...]) -> bool:
    return any(name in record and record[name] is False for name in names)


def _annotate_rejection(record: dict[str, object]) -> dict[str, object]:
    """Attach overlapping flags and a deterministic primary rejection reason.

    Primary precedence is solver exception, mass/residual, stability,
    nonnegativity, domain, root distance, branch disagreement, then an
    otherwise-unclassified solver rejection.  ``rejection_reasons`` retains
    every applicable flag in that same order.
    """

    item = dict(record)
    if bool(item.get("accepted", False)):
        flags = {
            "solver_exception": False,
            "mass_or_residual": False,
            "stability": False,
            "nonnegativity": False,
            "domain": False,
            "root_distance": False,
            "branch_disagreement": False,
            "other_solver_rejection": False,
        }
        primary = "accepted"
        reasons = "accepted"
    else:
        solver_exception = str(item.get("attempt_status", "")) == "solver_exception"
        mass_or_residual = _finite_exceeds(item, (
            "mass_residual_start_1", "mass_residual_start_2",
            "scaled_residual_start_1", "scaled_residual_start_2",
            "clarifier_component_residual_start_1",
            "clarifier_component_residual_start_2",
            "plant_boundary_residual_start_1",
            "plant_boundary_residual_start_2",
            "clarifier_tss_residual_start_1", "clarifier_tss_residual_start_2",
        ), 1.0e-8)
        stability = (
            _finite_exceeds(item, (
                "largest_real_eigenvalue_start_1",
                "largest_real_eigenvalue_start_2",
            ), -1.0e-8)
            or _finite_exceeds(item, (
                "stability_agreement_start_1",
                "stability_agreement_start_2",
            ), 1.0e-6)
        )
        nonnegativity = (
            _finite_exceeds(item, (
                "state_negativity_start_1", "state_negativity_start_2",
            ), 1.0e-10)
            or _finite_exceeds(item, (
                "rate_negativity_start_1", "rate_negativity_start_2",
            ), 1.0e-12)
            or _explicit_false(item, (
                "finite_nonnegative_rates_start_1",
                "finite_nonnegative_rates_start_2",
            ))
        )
        domain = (
            _finite_below(item, (
                "feed_tss_start_1", "feed_tss_start_2",
                "external_solids_loss_start_1",
                "external_solids_loss_start_2",
            ), 1.0)
            or _explicit_false(item, (
                "layer_envelope_start_1", "layer_envelope_start_2",
            ))
            or _finite_exceeds(item, (
                "soluble_passthrough_error_start_1",
                "soluble_passthrough_error_start_2",
            ), 1.0e-10)
        )
        root_distance = _finite_exceeds(
            item, ("root_difference_inf",), 1.0e-6,
        )
        branch_disagreement = item.get("branch_agreement") is False
        flags = {
            "solver_exception": solver_exception,
            "mass_or_residual": mass_or_residual,
            "stability": stability,
            "nonnegativity": nonnegativity,
            "domain": domain,
            "root_distance": root_distance,
            "branch_disagreement": branch_disagreement,
        }
        flags["other_solver_rejection"] = not any(flags.values())
        ordered = [name for name, active in flags.items() if active]
        primary = ordered[0]
        reasons = ";".join(ordered)
    for name, active in flags.items():
        item[f"rejected_{name}"] = bool(active)
    item["rejection_reason"] = primary
    item["rejection_reasons"] = reasons
    return item


def _error_record(candidate: _Candidate, error: BaseException) -> dict[str, object]:
    record: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "candidate_round": candidate.round_index,
        "candidate_index": candidate.candidate_index,
        "candidate_ordinal": candidate.candidate_ordinal,
        "accepted": False,
        "attempt_status": "solver_exception",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "elapsed_seconds": np.nan,
        "root_difference_inf": np.nan,
        "branch_agreement": False,
        "branch_classification": "{}",
        "route_start_1": "",
        "route_start_2": "",
    }
    for name in (
        "minimum_state_start_1", "minimum_state_start_2",
        "state_negativity_start_1", "state_negativity_start_2",
        "rate_negativity_start_1", "rate_negativity_start_2",
        "mass_residual_start_1", "mass_residual_start_2",
        "largest_real_eigenvalue_start_1", "largest_real_eigenvalue_start_2",
        "stability_agreement_start_1", "stability_agreement_start_2",
        "feed_tss_start_1", "feed_tss_start_2",
        "external_solids_loss_start_1", "external_solids_loss_start_2",
    ):
        record[name] = np.nan
    return _annotate_rejection(record)


def _supplemental_coordinates(
    count: int, starting_state: int,
) -> tuple[np.ndarray, int, int]:
    """Draw a row-major open-unit block from a continued SplitMix64 state."""

    stream = SplitMix64(starting_state)
    unit = np.empty((count, _DIMENSION_COUNT), dtype=float)
    for row in range(count):
        for dimension in range(_DIMENSION_COUNT):
            unit[row, dimension] = (
                float(stream.next_uint64() >> 11) + 0.5
            ) / _FLOAT53_DENOMINATOR
    return unit, stream.state, stream.draw_count


def _physical_supplemental(
    count: int, starting_state: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    unit, final_state, draws = _supplemental_coordinates(count, starting_state)
    lower = np.concatenate((core.DECISION_LOWER, INFLUENT_LOWER))
    upper = np.concatenate((core.DECISION_UPPER, INFLUENT_UPPER))
    physical = lower + unit * (upper - lower)
    physical = np.minimum(physical, np.nextafter(upper, lower))
    return physical[:, :7], physical[:, 7:], final_state, draws


def _load_attempt(
    candidate: _Candidate,
    *,
    expected_contract_hash: str,
    state_size: int,
    response_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]] | None:
    path = candidate.checkpoint
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            valid = bool(
                str(stored["contract_hash"].item()) == expected_contract_hash
                and np.array_equal(stored["decision"], candidate.decision)
                and np.array_equal(stored["influent"], candidate.influent)
                and stored["target"].shape == (response_count,)
                and stored["state_start_1"].shape == (state_size,)
                and stored["state_start_2"].shape == (state_size,)
            )
            if not valid:
                raise RuntimeError(
                    f"immutable attempt checkpoint does not match its contract: {path}"
                )
            target = np.asarray(stored["target"], dtype=float)
            first = np.asarray(stored["state_start_1"], dtype=float)
            second = np.asarray(stored["state_start_2"], dtype=float)
            record = json.loads(str(stored["record_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot validate immutable attempt checkpoint: {path}") from error
    # Legacy records predate explicit candidate metadata.  Enrich them in
    # memory only; the original checkpoint bytes remain untouched.
    record = dict(record)
    record.update({
        "candidate_id": candidate.candidate_id,
        "candidate_round": candidate.round_index,
        "candidate_index": candidate.candidate_index,
        "candidate_ordinal": candidate.candidate_ordinal,
        "attempt_status": (
            "accepted" if bool(record.get("accepted", False)) else "rejected"
        ),
        "error_type": str(record.get("error_type", "")),
        "error_message": str(record.get("error_message", "")),
    })
    record = _annotate_rejection(record)
    accepted = bool(record.get("accepted", False))
    if accepted and not (
        np.all(np.isfinite(target))
        and np.all(np.isfinite(first))
        and np.all(np.isfinite(second))
    ):
        raise RuntimeError(f"accepted immutable attempt is non-finite: {path}")
    return target, first, second, record


def _write_attempt(
    candidate: _Candidate,
    *,
    contract_hash: str,
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    record: dict[str, object],
) -> None:
    if candidate.checkpoint.exists():
        raise RuntimeError(
            f"refusing to overwrite immutable attempt: {candidate.checkpoint}"
        )
    _atomic_npz(
        candidate.checkpoint,
        contract_hash=np.asarray(contract_hash),
        decision=np.asarray(candidate.decision, dtype=float),
        influent=np.asarray(candidate.influent, dtype=float),
        target=np.asarray(target, dtype=float),
        state_start_1=np.asarray(first, dtype=float),
        state_start_2=np.asarray(second, dtype=float),
        record_json=np.asarray(json.dumps(record, sort_keys=True)),
    )


def _solve_candidates(
    candidates: list[_Candidate],
    profile: core.StudyProfile,
    contract_hash: str,
) -> list[tuple[_Candidate, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]:
    state_size = N_STAGES * N_COMPONENTS + profile.layer_count
    loaded: dict[str, tuple[_Candidate, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = {}
    missing: list[_Candidate] = []
    for candidate in candidates:
        cached = _load_attempt(
            candidate, expected_contract_hash=contract_hash, state_size=state_size,
            response_count=profile.mechanistic_response_count,
        )
        if cached is None:
            missing.append(candidate)
        else:
            loaded[candidate.candidate_id] = (candidate, *cached)

    label = (
        f"{candidates[0].block} round {candidates[0].round_index}"
        if candidates else "empty candidate batch"
    )
    if loaded:
        print(
            f"[{label}] reusing {len(loaded)}/{len(candidates)} immutable attempts",
            flush=True,
        )

    if missing:
        progress_interval = max(1, len(candidates) // 100)
        completed_now = 0
        with ProcessPoolExecutor(max_workers=profile.parallel_workers) as pool:
            futures = {
                pool.submit(
                    core._solve_design_row,
                    (
                        candidate.candidate_index,
                        candidate.decision,
                        candidate.influent,
                        profile.layer_count,
                    ),
                ): candidate
                for candidate in missing
            }
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    result = future.result()
                    target = np.asarray(result["target"], dtype=float)
                    first = np.asarray(result["state"], dtype=float)
                    second = np.asarray(result["state_start_2"], dtype=float)
                    record = _record_from_result(result, candidate)
                except Exception as error:  # an individual candidate is auditable, not fatal
                    target = np.full(profile.mechanistic_response_count, np.nan)
                    first = np.full(state_size, np.nan)
                    second = np.full(state_size, np.nan)
                    record = _error_record(candidate, error)
                _write_attempt(
                    candidate, contract_hash=contract_hash, target=target,
                    first=first, second=second, record=record,
                )
                loaded[candidate.candidate_id] = (
                    candidate, target, first, second, record,
                )
                completed_now += 1
                completed_total = len(candidates) - len(missing) + completed_now
                if (
                    completed_total % progress_interval == 0
                    or completed_total == len(candidates)
                ):
                    print(
                        f"[{label}] immutable attempt checkpoints "
                        f"{completed_total}/{len(candidates)}",
                        flush=True,
                    )
    return [loaded[candidate.candidate_id] for candidate in candidates]


def _round_manifest(
    path: Path,
    *,
    block: str,
    round_index: int,
    count: int,
    starting_state: int,
    final_state: int,
    starting_draw_count: int,
    draw_count: int,
    decisions: np.ndarray,
    influents: np.ndarray,
) -> dict[str, object]:
    payload = {
        "schema": REPLACEMENT_SCHEMA,
        "block": block,
        "round": round_index,
        "candidate_count": count,
        "starting_state": int(starting_state),
        "final_state": int(final_state),
        "starting_draw_count": int(starting_draw_count),
        "draw_count": int(draw_count),
        "ending_draw_count": int(starting_draw_count + draw_count),
        "design_digest": sha256(
            np.ascontiguousarray(decisions, dtype="<f8").tobytes()
            + np.ascontiguousarray(influents, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"supplemental round manifest differs: {path}")
    else:
        _atomic_json(path, payload)
    return payload


def generate_mechanistic_block_with_replacements(
    decisions: np.ndarray,
    influents: np.ndarray,
    profile: core.StudyProfile,
    output: Path,
    *,
    block: str | None = None,
) -> MechanisticBlockResult:
    """Return exactly the requested number of accepted mechanistic rows.

    Accepted base rows retain their original slots.  Each rejected base row is
    immutable, and accepted supplemental candidates fill the rejected slots in
    ascending order.  Supplemental rounds contain exactly the remaining
    deficit and continue the block's SplitMix64 stream; they never draw from or
    promote candidates across the development/test boundary.
    """

    decisions = np.asarray(decisions, dtype=float)
    influents = np.asarray(influents, dtype=float)
    output = Path(output)
    block_name, required_count, seed = _resolve_block(
        block, decisions, profile, output,
    )
    if decisions.shape != (required_count, 7):
        raise ValueError(f"{block_name} decisions have an invalid shape")
    if influents.shape != (required_count, N_COMPONENTS):
        raise ValueError(f"{block_name} influents have an invalid shape")
    if not np.all(np.isfinite(decisions)) or not np.all(np.isfinite(influents)):
        raise ValueError(f"{block_name} initial candidates must be finite")

    # Replaying the initial generator both validates its identity and gives the
    # exact state at which the independent supplemental stream begins.
    expected_decisions, expected_influents, generator = core._design_block(
        required_count, seed,
    )
    if not (
        np.array_equal(decisions, expected_decisions)
        and np.array_equal(influents, expected_influents)
    ):
        raise ValueError(
            f"{block_name} initial candidates do not match the declared v3 design"
        )

    output.mkdir(parents=True, exist_ok=True)
    rows_directory = output / "rows"
    rows_directory.mkdir(parents=True, exist_ok=True)
    replacement_directory = output / "attempts" / "replacement"
    base_contract = _base_contract_hash(decisions, influents, profile)
    replacement_contract = _replacement_contract_hash(profile, block_name)
    state_size = N_STAGES * N_COMPONENTS + profile.layer_count

    base_candidates = [
        _Candidate(
            block=block_name, round_index=0, candidate_index=index,
            candidate_ordinal=index, decision=decisions[index],
            influent=influents[index],
            checkpoint=rows_directory / f"row_{index:06d}.npz",
        )
        for index in range(required_count)
    ]
    preexisting = {
        candidate.candidate_id: candidate.checkpoint.is_file()
        for candidate in base_candidates
    }
    base_results = _solve_candidates(base_candidates, profile, base_contract)

    slots: dict[int, tuple[_Candidate, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = {}
    attempts = list(base_results)
    for item in base_results:
        candidate, target, first, second, record = item
        if bool(record["accepted"]):
            slots[candidate.candidate_index] = item

    failed_slots = [index for index in range(required_count) if index not in slots]
    stream_state = int(generator["final_state"])
    stream_draw_count = int(generator["draw_count"])
    candidate_ordinal = required_count
    round_index = 1
    while failed_slots:
        deficit = len(failed_slots)
        round_dir = replacement_directory / f"round_{round_index:06d}"
        round_decisions, round_influents, final_state, draws = _physical_supplemental(
            deficit, stream_state,
        )
        _round_manifest(
            round_dir / "manifest.json", block=block_name,
            round_index=round_index, count=deficit,
            starting_state=stream_state, final_state=final_state,
            starting_draw_count=stream_draw_count, draw_count=draws,
            decisions=round_decisions, influents=round_influents,
        )
        candidates = [
            _Candidate(
                block=block_name, round_index=round_index,
                candidate_index=index,
                candidate_ordinal=candidate_ordinal + index,
                decision=round_decisions[index],
                influent=round_influents[index],
                checkpoint=round_dir / f"candidate_{index:06d}.npz",
            )
            for index in range(deficit)
        ]
        results = _solve_candidates(candidates, profile, replacement_contract)
        attempts.extend(results)
        accepted = [item for item in results if bool(item[4]["accepted"])]
        for slot, item in zip(failed_slots, accepted, strict=False):
            slots[slot] = item
        failed_slots = [index for index in failed_slots if index not in slots]
        stream_state = final_state
        stream_draw_count += draws
        candidate_ordinal += deficit
        print(
            f"[{block_name}] replacement round {round_index}: "
            f"{len(accepted)}/{deficit} accepted; {len(failed_slots)} slots remain",
            flush=True,
        )
        round_index += 1

    accepted_decisions = np.empty((required_count, 7), dtype=float)
    accepted_influents = np.empty((required_count, N_COMPONENTS), dtype=float)
    targets = np.empty((required_count, profile.mechanistic_response_count), dtype=float)
    states_start_1 = np.empty((required_count, state_size), dtype=float)
    states_start_2 = np.empty_like(states_start_1)
    diagnostic_records: list[dict[str, object]] = []
    provenance_records: list[dict[str, object]] = []
    for slot in range(required_count):
        candidate, target, first, second, attempt_record = slots[slot]
        accepted_decisions[slot] = candidate.decision
        accepted_influents[slot] = candidate.influent
        targets[slot] = target
        states_start_1[slot] = first
        states_start_2[slot] = second
        diagnostic = dict(attempt_record)
        diagnostic["row"] = slot
        diagnostic["accepted_slot"] = slot
        diagnostic["source_candidate_id"] = candidate.candidate_id
        diagnostic["source_candidate_round"] = candidate.round_index
        diagnostic["source_candidate_index"] = candidate.candidate_index
        diagnostic["source_candidate_ordinal"] = candidate.candidate_ordinal
        diagnostic_records.append(diagnostic)
        provenance_records.append({
            "accepted_slot": slot,
            "base_candidate_id": (
                f"{block_name}:r000000:c{slot:06d}"
            ),
            "source_candidate_id": candidate.candidate_id,
            "source_candidate_round": candidate.round_index,
            "source_candidate_index": candidate.candidate_index,
            "source_candidate_ordinal": candidate.candidate_ordinal,
            "replaced_base_candidate": bool(candidate.round_index > 0),
        })

    diagnostics = pd.DataFrame(diagnostic_records)
    accepted_slots_by_candidate = {
        str(record["source_candidate_id"]): int(record["accepted_slot"])
        for record in provenance_records
    }
    attempt_records = []
    for candidate, _target, _first, _second, record in attempts:
        item = dict(record)
        item["checkpoint_path"] = str(candidate.checkpoint.relative_to(output))
        item["checkpoint_sha256"] = _file_digest(candidate.checkpoint)
        item["selected_for_accepted_block"] = (
            candidate.candidate_id in accepted_slots_by_candidate
        )
        item["accepted_slot"] = accepted_slots_by_candidate.get(
            candidate.candidate_id, np.nan,
        )
        attempt_records.append(item)
    attempts_frame = pd.DataFrame(attempt_records).sort_values(
        ["candidate_ordinal"], kind="stable",
    ).reset_index(drop=True)
    provenance = pd.DataFrame(provenance_records)

    migration = pd.DataFrame([
        {
            "candidate_id": candidate.candidate_id,
            "candidate_index": candidate.candidate_index,
            "checkpoint_path": str(candidate.checkpoint.relative_to(output)),
            "checkpoint_sha256": _file_digest(candidate.checkpoint),
            "original_contract_hash": base_contract,
            "preexisting_checkpoint": preexisting[candidate.candidate_id],
            "accepted": bool(item[4]["accepted"]),
            "preserved_without_rewrite": preexisting[candidate.candidate_id],
        }
        for candidate, item in zip(base_candidates, base_results, strict=True)
    ])

    _atomic_npz(
        output / "accepted_inputs.npz",
        decisions=accepted_decisions,
        influents=accepted_influents,
        source_candidate_id=provenance["source_candidate_id"].to_numpy(str),
        source_candidate_round=provenance["source_candidate_round"].to_numpy(int),
        source_candidate_index=provenance["source_candidate_index"].to_numpy(int),
        source_candidate_ordinal=provenance["source_candidate_ordinal"].to_numpy(int),
    )
    _atomic_npz(
        output / "mechanistic_accepted_v3.npz",
        contract_hash=np.asarray(replacement_contract),
        targets=targets,
        states_start_1=states_start_1,
        states_start_2=states_start_2,
    )
    _atomic_dataframe(output / "accepted_diagnostics.csv", diagnostics)
    _atomic_dataframe(output / "all_attempts.csv", attempts_frame)
    _atomic_dataframe(output / "accepted_provenance.csv", provenance)
    _atomic_dataframe(output / "base_checkpoint_migration.csv", migration)
    _atomic_json(output / "replacement_summary.json", {
        "schema": REPLACEMENT_SCHEMA,
        "block": block_name,
        "requested_accepted_count": required_count,
        "accepted_count": required_count,
        "base_attempt_count": required_count,
        "base_accepted_count": int(sum(
            bool(item[4]["accepted"]) for item in base_results
        )),
        "supplemental_attempt_count": len(attempts) - required_count,
        "supplemental_accepted_count": int(sum(
            bool(item[4]["accepted"]) for item in attempts[required_count:]
        )),
        "supplemental_round_count": round_index - 1,
        "initial_seed": int(seed),
        "initial_final_state": int(generator["final_state"]),
        "initial_draw_count": int(generator["draw_count"]),
        "replacement_final_state": int(stream_state),
        "replacement_draw_count": int(stream_draw_count - generator["draw_count"]),
        "total_stream_draw_count": int(stream_draw_count),
    })
    return MechanisticBlockResult(
        decisions=accepted_decisions, influents=accepted_influents,
        targets=targets, diagnostics=diagnostics, attempts=attempts_frame,
        provenance=provenance,
    )


__all__ = [
    "MechanisticBlockResult",
    "REPLACEMENT_SCHEMA",
    "generate_mechanistic_block_with_replacements",
]
