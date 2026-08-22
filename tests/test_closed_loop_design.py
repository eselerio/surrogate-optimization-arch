"""Tests for the manuscript-defined deterministic Latin-hypercube designs."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
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
    generate_iid_design,
    generate_robustness_design,
    generate_study_design_blocks,
    generate_training_design,
    unit_iid_uniform,
    unit_latin_hypercube,
    validate_design,
    validate_unit_box,
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

    def test_open_float52_excludes_both_faces_and_replays(self) -> None:
        left, right = SplitMix64(200043), SplitMix64(200043)
        left_values = [left.random_open_float52() for _ in range(100)]
        right_values = [right.random_open_float52() for _ in range(100)]
        self.assertEqual(left_values, right_values)
        self.assertTrue(all(0.0 < value < 1.0 for value in left_values))

        class EndpointStream(SplitMix64):
            def __init__(self, word: int) -> None:
                self.word = word

            def next_uint64(self) -> int:
                return self.word

        self.assertEqual(EndpointStream(0).random_open_float52(), 2.0**-53)
        self.assertEqual(
            EndpointStream((1 << 64) - 1).random_open_float52(), 1.0 - 2.0**-53
        )

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

    def test_affine_map_keeps_largest_open_midpoint_below_upper_bound(self) -> None:
        physical, _, upper = affine_map(
            np.asarray([[1.0 - 2.0**-53]]),
            ("H",),
            {"H": (6.0, 36.0)},
        )
        self.assertLess(physical[0, 0], upper[0])
        self.assertEqual(physical[0, 0], np.nextafter(36.0, 6.0))

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

    def test_iid_block_is_row_major_open_and_not_an_lhs(self) -> None:
        unit, final_state, draws = unit_iid_uniform(13, 4, seed=200043)
        self.assertEqual(draws, 52)
        self.assertEqual(unit.shape, (13, 4))
        validate_unit_box(unit, open_interval=True)
        replay, replay_state, replay_draws = unit_iid_uniform(13, 4, seed=200043)
        np.testing.assert_array_equal(unit, replay)
        self.assertEqual((final_state, draws), (replay_state, replay_draws))

        design = generate_iid_design(
            13,
            ("x", "y"),
            {"x": (0.0, 1.0), "y": (-2.0, 2.0)},
            seed=200043,
        )
        validate_design(
            design, require_latin_hypercube=False, require_open_unit=True
        )

    def test_study_blocks_have_independent_profile_streams_and_fixed_order(self) -> None:
        blocks = generate_study_design_blocks(
            14,
            2,
            4,
            development_seed=200042,
            calibration_seed=200043,
            assessment_seed=200044,
        )
        self.assertEqual(blocks.counts, (14, 2, 4))
        self.assertEqual(blocks.columns, TRAINING_COLUMNS)
        self.assertEqual(blocks.physical.shape, (20, 25))
        np.testing.assert_array_equal(
            blocks.physical[:14], blocks.development.physical
        )
        np.testing.assert_array_equal(
            blocks.physical[14:16], blocks.calibration.physical
        )
        np.testing.assert_array_equal(
            blocks.physical[16:], blocks.assessment.physical
        )
        full = generate_study_design_blocks(
            14,
            2,
            4,
            development_seed=42,
            calibration_seed=43,
            assessment_seed=44,
        )
        self.assertFalse(np.array_equal(blocks.calibration.unit, full.calibration.unit))
        self.assertFalse(np.array_equal(blocks.assessment.unit, full.assessment.unit))

    def test_test_2000_block_streams_have_frozen_fingerprints(self) -> None:
        blocks = generate_study_design_blocks(
            1400,
            200,
            400,
            development_seed=200042,
            calibration_seed=200043,
            assessment_seed=200044,
        )
        expected = (
            (
                "ce6ed13bac6c766c0c3fb6e5bead91a26b818324c9d65d4556fb0d5fbdf78a3e",
                17125270497545002381,
                69975,
            ),
            (
                "ea2c1f9b02e297b3f5f5e1cee2ddfd30205a8d5603a41701becc14cb90ee985e",
                3134908853478131603,
                5000,
            ),
            (
                "290f4b331984016d4dfe13834a02a28e07f53b24b8d0ff9c4dcaf4409186060c",
                6269817706956063164,
                10000,
            ),
        )
        for block, (digest, final_state, draws) in zip(
            (blocks.development, blocks.calibration, blocks.assessment),
            expected,
            strict=True,
        ):
            canonical = block.unit.astype("<f8", copy=False).tobytes(order="C")
            self.assertEqual(sha256(canonical).hexdigest(), digest)
            self.assertEqual((block.final_state, block.draw_count), (final_state, draws))

    def test_study_blocks_reject_reused_stream_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            generate_study_design_blocks(
                14,
                2,
                4,
                development_seed=17,
                calibration_seed=18,
                assessment_seed=18,
            )

    def test_mismatched_bounds_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Bounds do not match columns"):
            generate_design(10, ("x", "y"), {"x": (0.0, 1.0)}, seed=1)


class SchemaThreeProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "config" / "params_closed_loop.json"
        cls.config = json.loads(path.read_text(encoding="utf-8"))

    def test_profiles_freeze_independent_blocks_and_distinct_streams(self) -> None:
        self.assertEqual(self.config["schema_version"], 3)
        self.assertEqual(
            self.config["article"]["title"],
            "Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate",
        )
        expected = {
            "full": ((14000, 42), (2000, 43), (4000, 44), (100, 314159)),
            "test_2000": (
                (1400, 200042), (200, 200043), (400, 200044), (10, 2000314159)
            ),
            "unit": ((420, 60042), (60, 60043), (120, 60044), (1, 600314159)),
        }
        profiles = self.config["design"]["profiles"]
        all_seeds: list[int] = []
        for profile_name, values in expected.items():
            profile = profiles[profile_name]
            for block_name, (count, seed) in zip(
                ("development", "calibration", "assessment"), values[:3], strict=True
            ):
                block = profile["blocks"][block_name]
                self.assertEqual((block["count"], block["seed"]), (count, seed))
                all_seeds.append(seed)
            self.assertEqual(
                (profile["robustness"]["count"], profile["robustness"]["seed"]),
                values[3],
            )
            all_seeds.append(values[3][1])
        self.assertEqual(len(all_seeds), len(set(all_seeds)))

    def test_workloads_and_nlp_settings_match_the_frozen_contract(self) -> None:
        workloads = self.config["workloads"]
        self.assertEqual(
            (
                workloads["full"]["bdf_routes_max"],
                workloads["full"]["combined_nlp_starts"],
            ),
            (20113, 1017),
        )
        self.assertEqual(
            (
                workloads["test_2000"]["bdf_routes_max"],
                workloads["test_2000"]["combined_nlp_starts"],
            ),
            (2023, 207),
        )
        self.assertEqual(
            (
                workloads["unit"]["bdf_routes_max"],
                workloads["unit"]["combined_nlp_starts"],
            ),
            (614, 126),
        )
        for workload in workloads.values():
            self.assertEqual(workload["qp_evaluations"], 0)
            self.assertEqual(workload["direct_evaluations"], 0)

        feasibility = self.config["execution"]["computational_feasibility"]
        self.assertEqual(feasibility["maximum_projected_resident_memory_gib"], 25.0)
        nlp = self.config["optimization"]["nlp"]
        self.assertEqual(nlp["solver"], "IPOPT")
        self.assertEqual(nlp["linear_solver"], "MUMPS")
        self.assertEqual(nlp["maximum_iterations"], 2500)
        self.assertEqual(nlp["bound_relax_factor"], 0.0)
        self.assertEqual(
            nlp["combined_dimensions"],
            {"variables": 115, "equalities": 110, "general_inequalities": 9},
        )
        self.assertEqual(self.config["optimization"]["multistart"]["count"], 9)
        self.assertEqual(self.config["optimization"]["smoothing"]["epsilon"], 1e-8)
        self.assertEqual(
            self.config["optimization"]["combined_nlp"]["trust_families"],
            ["conformal fidelity", "development leverage"],
        )

    def test_obsolete_projection_and_direct_configuration_is_not_active(self) -> None:
        self.assertNotIn("deployment_qp", self.config["surrogate"])
        self.assertNotIn("surrogate_full", self.config["optimization"])
        self.assertNotIn("direct_epsilon", self.config["optimization"])
        self.assertNotIn("surrogate_nlp", self.config["optimization"])
        self.assertNotIn("mechanistic_nlp", self.config["optimization"])


if __name__ == "__main__":
    unittest.main()
