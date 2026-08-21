"""Tests for the manuscript-defined deterministic Latin-hypercube designs."""

from __future__ import annotations

import unittest

import numpy as np

from closed_loop.design import (
    DECISION_BOUNDS,
    DECISION_COLUMNS,
    INFLUENT_BOUNDS,
    INFLUENT_COLUMNS,
    ROBUSTNESS_COLUMNS,
    TRAINING_COLUMNS,
    SplitMix64,
    affine_map,
    generate_design,
    generate_robustness_design,
    generate_training_design,
    unit_latin_hypercube,
    validate_design,
    validate_unit_latin_hypercube,
)


class SplitMix64Tests(unittest.TestCase):
    def test_seed_42_matches_published_splitmix64_words(self) -> None:
        stream = SplitMix64(42)
        expected = (
            0xBDD732262FEB6E95,
            0x28EFE333B266F103,
            0x47526757130F9F52,
            0x581CE1FF0E4AE394,
            0x09BC585A244823F2,
        )
        self.assertEqual(tuple(stream.next_uint64() for _ in expected), expected)
        self.assertEqual(stream.draw_count, len(expected))
        self.assertEqual(
            stream.state,
            (42 + len(expected) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1),
        )

    def test_stream_restarts_exactly_and_float_is_half_open(self) -> None:
        left, right = SplitMix64(314159), SplitMix64(314159)
        left_values = [left.random_float53() for _ in range(100)]
        right_values = [right.random_float53() for _ in range(100)]
        self.assertEqual(left_values, right_values)
        self.assertTrue(all(0.0 <= value < 1.0 for value in left_values))

    def test_invalid_unsigned_seed_or_bound_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SplitMix64(-1)
        with self.assertRaises(ValueError):
            SplitMix64(1 << 64)
        stream = SplitMix64(0)
        with self.assertRaises(ValueError):
            stream.randbelow(0)

    def test_randbelow_rejects_the_modulo_bias_tail(self) -> None:
        class ScriptedStream(SplitMix64):
            def __init__(self) -> None:
                self.words = iter(((1 << 64) - 1, 5))

            def next_uint64(self) -> int:
                return next(self.words)

        # For upper=3, only UINT64_MAX lies outside L_b=2^64-1.  It must be
        # consumed and rejected before the second word maps to residue 2.
        self.assertEqual(ScriptedStream().randbelow(3), 2)


class UnitLatinHypercubeTests(unittest.TestCase):
    def test_every_dimension_has_one_point_per_stratum(self) -> None:
        rows, dimensions = 257, 25
        unit, final_state, draws = unit_latin_hypercube(rows, dimensions, seed=42)
        self.assertEqual(unit.shape, (rows, dimensions))
        self.assertTrue(np.all(unit >= 0.0))
        self.assertTrue(np.all(unit < 1.0))
        validate_unit_latin_hypercube(unit)

        expected = np.arange(rows)
        strata = np.floor(unit * rows).astype(int)
        for dimension in range(dimensions):
            np.testing.assert_array_equal(np.sort(strata[:, dimension]), expected)
        self.assertGreaterEqual(draws, dimensions * (2 * rows - 1))
        self.assertEqual(
            final_state,
            (42 + draws * 0x9E3779B97F4A7C15) & ((1 << 64) - 1),
        )

    def test_design_is_bitwise_reproducible(self) -> None:
        first = unit_latin_hypercube(71, 20, seed=314159)
        second = unit_latin_hypercube(71, 20, seed=314159)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])

        changed = unit_latin_hypercube(71, 20, seed=314160)
        self.assertFalse(np.array_equal(first[0], changed[0]))

    def test_one_point_design_consumes_one_jitter_per_dimension(self) -> None:
        unit, _, draws = unit_latin_hypercube(1, 4, seed=7)
        self.assertEqual(draws, 4)
        self.assertEqual(unit.shape, (1, 4))
        validate_unit_latin_hypercube(unit)

    def test_validation_detects_repeated_stratum_and_bad_domain(self) -> None:
        invalid_strata = np.asarray([[0.10], [0.20], [0.80]])
        with self.assertRaisesRegex(ValueError, "one point per stratum"):
            validate_unit_latin_hypercube(invalid_strata)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            validate_unit_latin_hypercube(np.asarray([[1.0]]))


class PhysicalDesignTests(unittest.TestCase):
    def test_training_design_has_exact_25_dimension_order_and_bounds(self) -> None:
        design = generate_training_design(200, seed=42)
        self.assertEqual(design.columns, TRAINING_COLUMNS)
        self.assertEqual(design.columns[:5], DECISION_COLUMNS)
        self.assertEqual(design.columns[5:], INFLUENT_COLUMNS)
        self.assertEqual(design.unit.shape, (200, 25))
        self.assertEqual(design.physical.shape, (200, 25))
        validate_design(design)

        for index, name in enumerate(design.columns):
            bounds = DECISION_BOUNDS if name in DECISION_BOUNDS else INFLUENT_BOUNDS
            lower, upper = bounds[name]
            self.assertTrue(np.all(design.physical[:, index] >= lower), name)
            self.assertTrue(np.all(design.physical[:, index] < upper), name)
        self.assertEqual(INFLUENT_BOUNDS["S_ALK"], (1.6, 5.2))

    def test_affine_map_replays_named_bounds(self) -> None:
        unit = np.asarray([[0.0, 0.25], [0.5, np.nextafter(1.0, 0.0)]])
        physical, lower, upper = affine_map(
            unit,
            ("first", "second"),
            {"first": (-2.0, 2.0), "second": (10.0, 30.0)},
        )
        np.testing.assert_array_equal(lower, [-2.0, 10.0])
        np.testing.assert_array_equal(upper, [2.0, 30.0])
        np.testing.assert_allclose(physical[0], [-2.0, 15.0], rtol=0.0, atol=0.0)
        self.assertEqual(physical[1, 0], 0.0)
        self.assertLess(physical[1, 1], 30.0)

    def test_robustness_design_restarts_in_influent_order(self) -> None:
        first = generate_robustness_design(100, seed=314159)
        second = generate_robustness_design(100, seed=314159)
        self.assertEqual(first.columns, ROBUSTNESS_COLUMNS)
        self.assertEqual(first.columns, INFLUENT_COLUMNS)
        np.testing.assert_array_equal(first.unit, second.unit)
        np.testing.assert_array_equal(first.physical, second.physical)
        self.assertEqual(first.final_state, second.final_state)
        self.assertEqual(first.draw_count, second.draw_count)

    def test_generic_mapping_is_column_order_driven(self) -> None:
        forward = generate_design(
            20,
            ("x", "y"),
            {"x": (0.0, 1.0), "y": (100.0, 200.0)},
            seed=9,
        )
        reversed_columns = generate_design(
            20,
            ("y", "x"),
            {"x": (0.0, 1.0), "y": (100.0, 200.0)},
            seed=9,
        )
        np.testing.assert_array_equal(forward.unit, reversed_columns.unit)
        np.testing.assert_allclose(
            forward.physical[:, 0],
            (reversed_columns.physical[:, 0] - 100.0) / 100.0,
            rtol=0.0,
            atol=5.0e-16,
        )
        np.testing.assert_array_equal(
            forward.physical[:, 1], 100.0 + 100.0 * reversed_columns.physical[:, 1]
        )

    def test_mismatched_bounds_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Bounds do not match columns"):
            generate_design(10, ("x", "y"), {"x": (0.0, 1.0)}, seed=1)


if __name__ == "__main__":
    unittest.main()
