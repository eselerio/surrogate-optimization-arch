import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from closed_loop import v3_trust
from closed_loop.model import N_COMPONENTS, N_STAGES
from closed_loop.projection import NetworkLayout


class _Projector:
    states: list[np.ndarray] = []
    accepted: list[bool] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self._row = 0

    def project(self, *args, **kwargs):
        del args, kwargs
        row = self._row
        self._row += 1
        return SimpleNamespace(
            state=self.states[row], accepted=self.accepted[row],
        )


class TrustCalibrationProjectionTests(unittest.TestCase):
    def _calibrate(self, states: list[np.ndarray], accepted: list[bool]):
        layout = NetworkLayout(layer_count=3)
        row_count = len(states)
        _Projector.states = states
        _Projector.accepted = accepted
        model = SimpleNamespace(response_scale=np.ones(layout.state_size))
        decisions = np.zeros((row_count, 7))
        influents = np.ones((row_count, N_COMPONENTS))
        targets = np.ones((row_count, layout.state_size))
        raw = np.ones_like(targets)
        row_scales = SimpleNamespace(equality=np.ones(1), inequality=np.ones(1))
        operators = SimpleNamespace(
            equality_matrix=np.zeros((1, layout.state_size)),
            equality_rhs=np.zeros(1),
            inequality_matrix=np.zeros((1, layout.state_size)),
        )
        direct_assets = SimpleNamespace(
            balance_scale=np.ones(N_STAGES * N_COMPONENTS + layout.layer_count),
            clarifier=SimpleNamespace(layer_volume=600.0, layer_count=layout.layer_count),
        )
        balance = np.zeros(N_STAGES * N_COMPONENTS)
        with patch.object(v3_trust, "PhysicalProjector", _Projector), patch.object(
            v3_trust, "fit_network_row_scales", return_value=row_scales,
        ), patch.object(
            v3_trust, "build_network_operators", return_value=operators,
        ), patch.object(
            v3_trust, "_smooth_reactor_residual",
            return_value=balance,
        ):
            return v3_trust.calibrate_trust_diagnostics(
                model, decisions, influents, targets, raw, direct_assets,
                layout=layout,
            )

    def test_finite_rejected_projection_is_retained_and_exposed(self) -> None:
        layout = NetworkLayout(layer_count=3)
        states = [
            np.ones(layout.state_size),
            np.full(layout.state_size, 2.0),
        ]
        result = self._calibrate(states, [False, True])
        self.assertEqual(result.development_values.shape, (2, 3))
        self.assertFalse(hasattr(result, "flux_limit"))
        self.assertFalse(hasattr(result.callbacks, "flux_rows"))
        np.testing.assert_array_equal(
            result.out_of_fold_projection_accepted,
            np.array([False, True]),
        )
        np.testing.assert_allclose(result.out_of_fold_projected, states)

    def test_nonfinite_or_wrong_shape_projection_remains_fatal(self) -> None:
        layout = NetworkLayout(layer_count=3)
        invalid_states = (
            np.full(layout.state_size, np.nan),
            np.ones(layout.state_size - 1),
        )
        for state in invalid_states:
            with self.subTest(shape=state.shape), self.assertRaisesRegex(
                RuntimeError, "invalid state",
            ):
                self._calibrate([state], [False])


if __name__ == "__main__":
    unittest.main()
