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
        self.assertEqual(len({cell.id for cell in actual.cells}), len(actual.cells))
        for cell in actual.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])


if __name__ == "__main__":
    unittest.main()
