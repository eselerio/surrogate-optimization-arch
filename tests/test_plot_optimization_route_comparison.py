from pathlib import Path
import tempfile
import unittest

from scripts import plot_optimization_route_comparison as plot_routes


class OptimizationRoutePlotArtifactTests(unittest.TestCase):
    def test_casewise_reference_is_preferred_with_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "nominal"
            case.mkdir()
            legacy = case / "surrogate_selected.npz"
            legacy.write_bytes(b"legacy")

            self.assertEqual(
                plot_routes._route_reference_artifact(case, "surrogate"), legacy,
            )

            current = case / "surrogate_casewise_reference.npz"
            current.write_bytes(b"current")
            self.assertEqual(
                plot_routes._route_reference_artifact(case, "surrogate"), current,
            )

    def test_casewise_shared_unit_artifact_enables_three_route_plotting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "nominal"
            case.mkdir()
            (case / "shared_unit_casewise_reference.npz").write_bytes(b"current")

            self.assertEqual(plot_routes._routes(root), plot_routes.ROUTE_ORDER)


if __name__ == "__main__":
    unittest.main()
