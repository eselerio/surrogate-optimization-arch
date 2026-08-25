"""Deterministic, resumable process-batch execution for article-v3 assessments.

The helpers in this module deliberately know nothing about the numerical
method being evaluated.  A caller supplies a top-level batch worker and a
strict payload validator.  Completed batches are written atomically as
``.npz`` files, bound to a caller-provided scientific input digest, and loaded
in row order on restart.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from time import sleep
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits


BATCH_CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True)
class BatchProgress:
    """Progress published by :func:`run_resumable_batches`."""

    stage: str
    completed_rows: int
    total_rows: int
    reused_rows: int
    newly_completed_rows: int
    completed_batches: int
    total_batches: int
    parallel_workers: int
    batch_size: int


BatchPayload = dict[str, np.ndarray]
BatchWorker = Callable[[tuple[int, int]], Mapping[str, Any]]
BatchValidator = Callable[[int, int, Mapping[str, np.ndarray]], None]
ProgressCallback = Callable[[BatchProgress], None]


def _payload_digest(payload: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    for name in sorted(payload):
        value = np.asarray(payload[name])
        if value.dtype.hasobject:
            raise ValueError(f"batch payload {name!r} cannot use object dtype")
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


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
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                sleep(0.05 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalize_payload(
    payload: Mapping[str, Any], expected_names: Sequence[str],
) -> BatchPayload:
    expected = tuple(expected_names)
    if set(payload) != set(expected):
        raise ValueError(
            "batch worker returned unexpected fields: "
            f"expected={sorted(expected)}, observed={sorted(payload)}"
        )
    normalized: BatchPayload = {}
    for name in expected:
        value = np.asarray(payload[name])
        if value.dtype.hasobject:
            raise ValueError(f"batch payload {name!r} cannot use object dtype")
        normalized[name] = value.copy()
    return normalized


def _checkpoint_path(directory: Path, start: int, stop: int) -> Path:
    return directory / f"batch_{start:06d}_{stop:06d}.npz"


def _load_checkpoint(
    path: Path,
    *,
    stage: str,
    contract_digest: str,
    row_count: int,
    batch_size: int,
    start: int,
    stop: int,
    payload_names: Sequence[str],
    validate: BatchValidator,
) -> BatchPayload | None:
    if not path.is_file():
        return None
    metadata_names = {
        "checkpoint_schema",
        "stage",
        "contract_digest",
        "row_count",
        "batch_size",
        "start",
        "stop",
        "payload_digest",
    }
    expected_names = metadata_names | set(payload_names)
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != expected_names:
                return None
            valid_metadata = bool(
                int(stored["checkpoint_schema"].item()) == BATCH_CHECKPOINT_SCHEMA
                and str(stored["stage"].item()) == stage
                and str(stored["contract_digest"].item()) == contract_digest
                and int(stored["row_count"].item()) == row_count
                and int(stored["batch_size"].item()) == batch_size
                and int(stored["start"].item()) == start
                and int(stored["stop"].item()) == stop
            )
            if not valid_metadata:
                return None
            payload = {
                name: np.asarray(stored[name]).copy() for name in payload_names
            }
            if str(stored["payload_digest"].item()) != _payload_digest(payload):
                return None
        validate(start, stop, payload)
        return payload
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_checkpoint(
    path: Path,
    *,
    stage: str,
    contract_digest: str,
    row_count: int,
    batch_size: int,
    start: int,
    stop: int,
    payload: Mapping[str, np.ndarray],
) -> None:
    normalized = {name: np.asarray(value) for name, value in payload.items()}
    _atomic_npz(
        path,
        checkpoint_schema=np.asarray(BATCH_CHECKPOINT_SCHEMA, dtype=np.int64),
        stage=np.asarray(stage),
        contract_digest=np.asarray(contract_digest),
        row_count=np.asarray(row_count, dtype=np.int64),
        batch_size=np.asarray(batch_size, dtype=np.int64),
        start=np.asarray(start, dtype=np.int64),
        stop=np.asarray(stop, dtype=np.int64),
        payload_digest=np.asarray(_payload_digest(normalized)),
        **normalized,
    )


def _execute_worker(worker: BatchWorker, bounds: tuple[int, int]) -> Mapping[str, Any]:
    # Every process owns a single numerical thread.  This prevents a pool of
    # workers from each spawning a full BLAS thread pool.
    with threadpool_limits(limits=1):
        return worker(bounds)


def run_resumable_batches(
    *,
    stage: str,
    row_count: int,
    batch_size: int,
    parallel_workers: int,
    checkpoint_directory: Path | None,
    contract_digest: str,
    payload_names: Sequence[str],
    worker: BatchWorker,
    validate: BatchValidator,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    progress: ProgressCallback | None = None,
) -> list[BatchPayload]:
    """Execute fixed contiguous batches and return them in ascending row order.

    A checkpoint is reusable only when its stage, complete-input digest,
    row/batch geometry, payload fields, payload hash, and caller validation all
    agree.  Invalid or partial files are ignored and atomically replaced after
    recomputation.
    """

    if not stage:
        raise ValueError("batch stage must be nonempty")
    if row_count < 0:
        raise ValueError("row_count must be nonnegative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be positive")
    if not contract_digest:
        raise ValueError("contract_digest must be nonempty")
    if not payload_names or len(set(payload_names)) != len(tuple(payload_names)):
        raise ValueError("payload_names must be unique and nonempty")

    ranges = [
        (start, min(start + batch_size, row_count))
        for start in range(0, row_count, batch_size)
    ]
    if not ranges:
        return []
    directory = None if checkpoint_directory is None else Path(checkpoint_directory)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)

    results: dict[int, BatchPayload] = {}
    reused_rows = 0
    missing: list[tuple[int, int]] = []
    for start, stop in ranges:
        payload = None
        if directory is not None:
            payload = _load_checkpoint(
                _checkpoint_path(directory, start, stop),
                stage=stage,
                contract_digest=contract_digest,
                row_count=row_count,
                batch_size=batch_size,
                start=start,
                stop=stop,
                payload_names=payload_names,
                validate=validate,
            )
        if payload is None:
            missing.append((start, stop))
        else:
            results[start] = payload
            reused_rows += stop - start

    worker_count = max(1, min(int(parallel_workers), len(missing) or 1))
    completed_rows = reused_rows
    newly_completed_rows = 0

    def publish(bounds: tuple[int, int], raw_payload: Mapping[str, Any]) -> None:
        nonlocal completed_rows, newly_completed_rows
        start, stop = bounds
        payload = _normalize_payload(raw_payload, payload_names)
        validate(start, stop, payload)
        if directory is not None:
            _save_checkpoint(
                _checkpoint_path(directory, start, stop),
                stage=stage,
                contract_digest=contract_digest,
                row_count=row_count,
                batch_size=batch_size,
                start=start,
                stop=stop,
                payload=payload,
            )
        results[start] = payload
        newly_completed_rows += stop - start
        completed_rows += stop - start
        if progress is not None:
            progress(BatchProgress(
                stage=stage,
                completed_rows=completed_rows,
                total_rows=row_count,
                reused_rows=reused_rows,
                newly_completed_rows=newly_completed_rows,
                completed_batches=len(results),
                total_batches=len(ranges),
                parallel_workers=worker_count,
                batch_size=batch_size,
            ))

    if progress is not None and reused_rows:
        progress(BatchProgress(
            stage=stage,
            completed_rows=reused_rows,
            total_rows=row_count,
            reused_rows=reused_rows,
            newly_completed_rows=0,
            completed_batches=len(results),
            total_batches=len(ranges),
            parallel_workers=worker_count,
            batch_size=batch_size,
        ))

    if missing and worker_count == 1:
        if initializer is not None:
            initializer(*initargs)
        for bounds in missing:
            publish(bounds, _execute_worker(worker, bounds))
    elif missing:
        # ``spawn`` exercises the same explicit serialization contract on every
        # platform and avoids inheriting solver-library state from the parent.
        context = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=initializer,
            initargs=initargs,
        )
        pending: dict[Any, tuple[int, int]] = {}
        remaining = iter(missing)

        def fill_queue() -> None:
            while len(pending) < 2 * worker_count:
                try:
                    bounds = next(remaining)
                except StopIteration:
                    return
                pending[pool.submit(_execute_worker, worker, bounds)] = bounds

        try:
            fill_queue()
            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    bounds = pending.pop(future)
                    publish(bounds, future.result())
                fill_queue()
        except BaseException:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    if set(results) != {start for start, _ in ranges}:
        raise RuntimeError(f"{stage} batch execution did not cover every row")
    return [results[start] for start, _ in ranges]


__all__ = [
    "BATCH_CHECKPOINT_SCHEMA",
    "BatchProgress",
    "run_resumable_batches",
]
