import json
from pathlib import Path
import unittest

import nbformat

from scripts import build_main_closed_loop_v3 as builder


ROOT = Path(__file__).resolve().parents[1]


class MainClosedLoopNotebookTests(unittest.TestCase):
    def test_checked_in_notebook_matches_deterministic_builder(self) -> None:
        expected = builder.build_notebook()
        actual = nbformat.read(ROOT / "main_closed_loop.ipynb", as_version=4)

        self.assertEqual(actual, expected)
        self.assertEqual(
            actual.metadata.surrogate_optimization_arch.response_schema,
            "clarifier_inventory_v1",
        )
        metadata = actual.metadata.surrogate_optimization_arch
        self.assertEqual(metadata.schema, 7)
        self.assertEqual(metadata.runner_schema, 11)
        self.assertEqual(
            list(metadata.routes), ["surrogate", "shared_unit", "direct"],
        )
        self.assertEqual(
            dict(metadata.route_symbols),
            {"surrogate": "S", "shared_unit": "U", "direct": "M"},
        )
        source = "\n".join(cell.source for cell in actual.cells)
        self.assertIn("article_full_50000_three_route_001", source)
        self.assertIn("shared_unit_fold_models.npz", source)
        self.assertIn("shared_unit_fold_membership.csv", source)
        self.assertIn("shared_unit_post_selection_root_diagnostics.csv", source)
        self.assertIn("shared_unit_trust_post_selection_holdout.csv", source)
        self.assertIn("casewise_exact_common_reference_v4", source)
        self.assertIn("robustness_casewise_three_route_v2", source)
        self.assertEqual(len({cell.id for cell in actual.cells}), len(actual.cells))
        for cell in actual.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])

    def test_three_route_configuration_contract(self) -> None:
        config = json.loads(
            (ROOT / "config" / "params_manuscript_v3.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["schema_version"], 5)
        self.assertEqual(config["execution"]["runner_schema"], 11)
        optimization = config["optimization"]
        self.assertEqual(
            optimization["route_ids"], ["surrogate", "shared_unit", "direct"]
        )
        self.assertEqual(
            optimization["runner_protocol"], "three_route_single_center_v2"
        )
        self.assertEqual(
            optimization["shared_unit_protocol"],
            "shared_unit_value_only_single_center_v1",
        )
        unit = config["shared_unit_surrogate"]
        self.assertEqual(unit["reactor_coefficient_shape"], [20, 276])
        self.assertEqual(unit["clarifier_coefficient_shape"], [41, 276])
        self.assertEqual(unit["development_reactor_transition_count"], 66_855)
        self.assertTrue(unit["recycle_closure"]["both_starts_must_succeed"])
        self.assertEqual(
            config["reporting"]["validation_protocol"],
            "casewise_exact_common_reference_v4",
        )
        self.assertEqual(
            config["reporting"]["timing_protocol"],
            "robustness_casewise_three_route_v2",
        )
        self.assertEqual(
            config["reporting"]["pairwise_comparisons"],
            ["S-U", "S-M", "U-M"],
        )


if __name__ == "__main__":
    unittest.main()
