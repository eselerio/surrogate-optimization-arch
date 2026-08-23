import unittest

import numpy as np

from closed_loop.manuscript_v3 import TEST_500, clarifier_for, create_design
from closed_loop.model import (
    ArticleOperatingPoint, NOMINAL_INFLUENT, solve_steady_state,
)
from closed_loop.projection import NetworkLayout, QuadraticFeatureMap


class ManuscriptV3ContractTests(unittest.TestCase):
    def test_requested_verification_dimensions_and_independent_blocks(self) -> None:
        design = create_design(TEST_500)
        self.assertEqual(design["development_decisions"].shape, (400, 7))
        self.assertEqual(design["development_influents"].shape, (400, 20))
        self.assertEqual(design["test_decisions"].shape, (100, 7))
        self.assertEqual(design["robustness_influents"].shape, (5, 20))
        self.assertFalse(np.array_equal(
            design["development_influents"][:100], design["test_influents"]
        ))
        feature_map = QuadraticFeatureMap.fit(
            design["development_decisions"], design["development_influents"]
        )
        self.assertEqual(feature_map.feature_count, 406)
        self.assertEqual(NetworkLayout(layer_count=5).state_size, 165)

    def test_five_layer_independent_aeration_mechanism_closes(self) -> None:
        clarifier = clarifier_for(TEST_500)
        operating = ArticleOperatingPoint(18.0, 0.2, 0.3, 0.4, 2.0, 0.75, 0.02)
        result = solve_steady_state(
            operating, NOMINAL_INFLUENT, starts=(1,), clarifier=clarifier
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.state.shape, (105,))
        self.assertEqual(result.target.shape, (165,))
        self.assertLessEqual(result.diagnostics["plant_boundary_residual"], 1e-8)
        self.assertGreaterEqual(result.diagnostics["minimum_state"], -1e-10)


if __name__ == "__main__":
    unittest.main()
