from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from closed_loop.projection import NetworkLayout
from closed_loop.v3_parallel import BatchProgress, run_resumable_batches
from closed_loop.v3_trust import ParticulateSplitRows, SmoothReactorRows


_OFFSET = 0
_FAIL_START: int | None = None
_CALLS: list[tuple[int, int]] = []


def _initialize_worker(offset: int, fail_start: int | None = None) -> None:
    global _OFFSET, _FAIL_START
    _OFFSET = int(offset)
    _FAIL_START = fail_start


def _batch_worker(bounds: tuple[int, int]):
    start, stop = bounds
    _CALLS.append(bounds)
    if _FAIL_START == start:
        raise RuntimeError("intentional batch failure")
    rows = np.arange(start, stop, dtype=np.int64)
    return {"row": rows, "value": rows**2 + _OFFSET}


def _validate_batch(start: int, stop: int, payload) -> None:
    expected = np.arange(start, stop, dtype=np.int64)
    np.testing.assert_array_equal(payload["row"], expected)
    if np.asarray(payload["value"]).shape != expected.shape:
        raise ValueError("value shape mismatch")


class ResumableBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        _CALLS.clear()

    def _run(
        self,
        directory: Path | None,
        *,
        workers: int,
        contract: str = "contract-a",
        fail_start: int | None = None,
        progress=None,
    ):
        return run_resumable_batches(
            stage="unit-stage",
            row_count=7,
            batch_size=2,
            parallel_workers=workers,
            checkpoint_directory=directory,
            contract_digest=contract,
            payload_names=("row", "value"),
            worker=_batch_worker,
            validate=_validate_batch,
            initializer=_initialize_worker,
            initargs=(11, fail_start),
            progress=progress,
        )

    def test_serial_fallback_avoids_process_pool_and_keeps_row_order(self) -> None:
        with patch(
            "closed_loop.v3_parallel.ProcessPoolExecutor",
            side_effect=AssertionError("serial execution constructed a process pool"),
        ):
            batches = self._run(None, workers=1)
        rows = np.concatenate([batch["row"] for batch in batches])
        values = np.concatenate([batch["value"] for batch in batches])
        np.testing.assert_array_equal(rows, np.arange(7))
        np.testing.assert_array_equal(values, np.arange(7) ** 2 + 11)
        self.assertEqual(_CALLS, [(0, 2), (2, 4), (4, 6), (6, 7)])

    def test_spawn_parallel_matches_serial_and_aggregates_in_order(self) -> None:
        serial = self._run(None, workers=1)
        parallel = self._run(None, workers=2)
        for serial_batch, parallel_batch in zip(serial, parallel, strict=True):
            np.testing.assert_array_equal(serial_batch["row"], parallel_batch["row"])
            np.testing.assert_array_equal(
                serial_batch["value"], parallel_batch["value"]
            )

    def test_interruption_reuses_only_atomic_completed_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                self._run(root, workers=1, fail_start=2)
            self.assertTrue((root / "batch_000000_000002.npz").is_file())
            self.assertFalse((root / "batch_000002_000004.npz").exists())
            self.assertEqual(list(root.glob("*.tmp")), [])

            _CALLS.clear()
            updates: list[BatchProgress] = []
            batches = self._run(root, workers=1, progress=updates.append)
            self.assertEqual(_CALLS, [(2, 4), (4, 6), (6, 7)])
            self.assertEqual(updates[0].reused_rows, 2)
            self.assertEqual(updates[-1].completed_rows, 7)
            np.testing.assert_array_equal(
                np.concatenate([batch["row"] for batch in batches]), np.arange(7)
            )

    def test_corrupt_or_stale_checkpoint_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run(root, workers=1)
            (root / "batch_000002_000004.npz").write_bytes(b"truncated")
            _CALLS.clear()
            self._run(root, workers=1)
            self.assertEqual(_CALLS, [(2, 4)])

            _CALLS.clear()
            self._run(root, workers=1, contract="contract-b")
            self.assertEqual(
                _CALLS, [(0, 2), (2, 4), (4, 6), (6, 7)]
            )

    def test_worker_failure_never_publishes_failed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                self._run(root, workers=1, fail_start=4)
            self.assertTrue((root / "batch_000000_000002.npz").is_file())
            self.assertTrue((root / "batch_000002_000004.npz").is_file())
            self.assertFalse((root / "batch_000004_000006.npz").exists())
            self.assertEqual(list(root.glob("*.tmp")), [])


class SpawnSafeTrustCallbackTests(unittest.TestCase):
    def test_particulate_callback_pickle_round_trip_preserves_numeric_formula(self) -> None:
        layout = NetworkLayout()
        particulate = (1, 2, 3)
        scale = np.asarray([2.0, 3.0, 4.0])
        weights = np.arange(1.0, layout.component_count + 1.0)
        callback = ParticulateSplitRows(layout, particulate, scale, weights)
        restored = pickle.loads(pickle.dumps(callback))
        response = np.arange(1.0, layout.state_size + 1.0)
        final = response[layout.reactor_slice(layout.stage_count - 1)]
        underflow = response[layout.underflow_flow_slice]
        expected = (
            (weights @ final) * underflow[list(particulate)]
            - (weights @ underflow) * final[list(particulate)]
        ) / scale
        np.testing.assert_allclose(
            restored(np.zeros(7), response, response, np.zeros(20)), expected
        )

    def test_smooth_callback_is_pickleable(self) -> None:
        from types import SimpleNamespace

        callback = SmoothReactorRows(
            direct_assets=SimpleNamespace(
                balance_scale=np.ones(105),
                marker="pickleable",
            )
        )
        restored = pickle.loads(pickle.dumps(callback))
        self.assertEqual(restored.direct_assets.marker, "pickleable")
        self.assertEqual(restored.epsilon, 1.0e-8)
        residual = np.arange(1.0, 101.0)
        with patch(
            "closed_loop.v3_trust._smooth_reactor_residual",
            return_value=residual,
        ) as mocked:
            observed = restored(
                np.ones(7), np.ones(161), np.ones(161), np.ones(20)
            )
        np.testing.assert_array_equal(observed, residual)
        self.assertEqual(mocked.call_args.args[-1], 1.0e-8)


if __name__ == "__main__":
    unittest.main()
