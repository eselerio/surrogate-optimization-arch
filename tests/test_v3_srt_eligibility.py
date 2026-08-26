from __future__ import annotations

import unittest

import numpy as np

from closed_loop.model import (
    ClarifierParameters,
    INFLUENT_LOWER,
    INFLUENT_UPPER,
    N_COMPONENTS,
    N_STAGES,
    TSS_VECTOR,
)
from closed_loop.v3_smooth import (
    DECISION_LOWER,
    DECISION_UPPER,
    DirectAssets,
    SmoothScales,
    engineering_feasible,
    engineering_quantities,
)
from scripts.run_article_v3_5000 import _casewise_comparison_row


LAYER_COUNT = 5
PARTICULATE_INDEX = int(np.flatnonzero(TSS_VECTOR > 0.0)[0])
PARTICULATE_TSS_WEIGHT = float(TSS_VECTOR[PARTICULATE_INDEX])
CONTROLS = np.asarray([6.0, 0.2, 0.3, 0.4, 2.0, 0.75, 0.02])


def _assets() -> DirectAssets:
    state_count = N_STAGES * N_COMPONENTS + LAYER_COUNT
    return DirectAssets(
        clarifier=ClarifierParameters(
            layer_count=LAYER_COUNT,
            feed_layer=(LAYER_COUNT - 1) // 2,
            layer_volume=6_000.0 / LAYER_COUNT,
        ),
        smoothing=SmoothScales(
            10.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 250.0, 10_000.0,
        ),
        state_center=np.ones(state_count),
        state_scale=np.ones(state_count),
        feed_scale=100.0,
        balance_scale=np.ones(state_count),
        quality_scale=np.ones(4),
        envelope_scale=np.ones(2 * (LAYER_COUNT - 2)),
        engineering_scale=np.ones(4),
        decision_center=(DECISION_LOWER + DECISION_UPPER) / 2.0,
        decision_scale=(DECISION_UPPER - DECISION_LOWER) / np.sqrt(12.0),
        influent_center=(INFLUENT_LOWER + INFLUENT_UPPER) / 2.0,
        influent_scale=(INFLUENT_UPPER - INFLUENT_LOWER) / np.sqrt(12.0),
    )


def _response(
    *,
    reactor_tss: float = 100.0,
    effluent_tss: float = 5.0,
    underflow_tss: float = 1_000.0,
    layers: tuple[float, ...] = (5.0, 20.0, 50.0, 100.0, 1_000.0),
) -> np.ndarray:
    response = np.zeros((N_STAGES + 3) * N_COMPONENTS + LAYER_COUNT)
    for stage in range(N_STAGES):
        start = (stage + 1) * N_COMPONENTS
        response[start + PARTICULATE_INDEX] = (
            reactor_tss / PARTICULATE_TSS_WEIGHT
        )
    q_effluent = 1.0 - CONTROLS[6]
    q_underflow = CONTROLS[5] + CONTROLS[6]
    response[(N_STAGES + 1) * N_COMPONENTS + PARTICULATE_INDEX] = (
        q_effluent * effluent_tss / PARTICULATE_TSS_WEIGHT
    )
    response[(N_STAGES + 2) * N_COMPONENTS + PARTICULATE_INDEX] = (
        q_underflow * underflow_tss / PARTICULATE_TSS_WEIGHT
    )
    response[(N_STAGES + 3) * N_COMPONENTS :] = np.asarray(layers)
    return response


def _route_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_available": True,
        "native_feasible": True,
        "exact_replay_valid": True,
        "comparison_valid": True,
        "exact_reference_objective": 1.0,
        "exact_reference_objective_components": [1.0] * 6,
        "normalized_controls": [0.5] * 7,
        "local_convergence_certified": True,
        "first_order_stationarity_certified": True,
    }
    payload.update(overrides)
    return payload


class MinimumSrtEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assets = _assets()
        self.baseline = _response()

    def test_below_eight_day_srt_is_reported_but_remains_eligible(self) -> None:
        quantities = engineering_quantities(CONTROLS, self.baseline, self.assets)
        self.assertGreater(quantities["srt_d"], 0.0)
        self.assertLess(quantities["srt_d"], 8.0)
        self.assertTrue(engineering_feasible(CONTROLS, self.baseline, self.assets))

        row = _casewise_comparison_row(
            "below_eight_days", _route_payload(), _route_payload(),
        )
        self.assertTrue(row["comparison_eligible"])
        self.assertIsNone(row["ineligibility_reasons"])
        self.assertTrue(row["minimum_srt_is_descriptive_not_eligibility_gate"])

    def test_each_retained_engineering_violation_remains_ineligible(self) -> None:
        cases = {
            "maximum_srt": _response(
                layers=(5.0, 15_000.0, 15_000.0, 15_000.0, 15_000.0),
            ),
            "solids_loading_rate": _response(
                reactor_tss=10_000.0,
                effluent_tss=1_000.0,
                underflow_tss=1_000.0,
                layers=(1_000.0,) * LAYER_COUNT,
            ),
            "underflow_tss": _response(
                underflow_tss=16_000.0,
                layers=(5.0, 20.0, 50.0, 100.0, 16_000.0),
            ),
            "external_solids_loss": _response(
                reactor_tss=1.0,
                effluent_tss=0.1,
                underflow_tss=20.0,
                layers=(0.1, 0.2, 0.4, 0.8, 20.0),
            ),
            "feed_tss": _response(reactor_tss=0.5),
        }
        negative = self.baseline.copy()
        negative[0] = -1.0
        cases["nonnegativity"] = negative
        upper_envelope = self.baseline.copy()
        upper_envelope[-4] = 4.0
        cases["upper_layer_envelope"] = upper_envelope
        lower_envelope = self.baseline.copy()
        lower_envelope[-2] = 1_100.0
        cases["lower_layer_envelope"] = lower_envelope

        for name, response in cases.items():
            with self.subTest(name=name):
                self.assertFalse(
                    engineering_feasible(CONTROLS, response, self.assets)
                )
                row = _casewise_comparison_row(
                    name,
                    _route_payload(comparison_valid=False),
                    _route_payload(),
                )
                self.assertFalse(row["comparison_eligible"])
                self.assertEqual(
                    row["ineligibility_reasons"],
                    "surrogate_exact_engineering_infeasible",
                )

    def test_other_pairwise_exclusion_conditions_are_unchanged(self) -> None:
        exclusions = {
            "surrogate_no_candidate": (
                _route_payload(candidate_available=False), _route_payload(),
            ),
            "direct_no_candidate": (
                _route_payload(), _route_payload(candidate_available=False),
            ),
            "surrogate_reference_invalid": (
                _route_payload(exact_replay_valid=False), _route_payload(),
            ),
            "direct_reference_invalid": (
                _route_payload(), _route_payload(exact_replay_valid=False),
            ),
            "surrogate_native_infeasible": (
                _route_payload(native_feasible=False), _route_payload(),
            ),
            "direct_native_infeasible": (
                _route_payload(), _route_payload(native_feasible=False),
            ),
        }
        for expected, (surrogate, direct) in exclusions.items():
            with self.subTest(expected=expected):
                row = _casewise_comparison_row("case", surrogate, direct)
                self.assertFalse(row["comparison_eligible"])
                self.assertIn(expected, row["ineligibility_reasons"].split(";"))


if __name__ == "__main__":
    unittest.main()
