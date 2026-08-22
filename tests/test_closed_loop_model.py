"""Staged unit tests for the coupled closed-loop mechanistic model."""

from __future__ import annotations

import unittest

import numpy as np

from closed_loop.model import (
    CLARIFIER,
    COMPONENT_INDEX,
    INVARIANT_MATRIX,
    NOMINAL_INFLUENT,
    N_COMPONENTS,
    N_LAYERS,
    N_PROCESSES,
    STATE_SIZE,
    STOICHIOMETRIC_MATRIX,
    TARGET_SIZE,
    TSS_VECTOR,
    OperatingPoint,
    assemble_target,
    audit_mechanistic_matrices,
    clarifier_fluxes,
    clarifier_rhs,
    coupled_rhs,
    initial_state,
    mixer_state,
    process_rates,
    reconstruct_clarifier,
    settling_velocity,
    solve_steady_state,
    zero_state_solution,
)


class MatrixAndKineticsTests(unittest.TestCase):
    def test_dimensions_and_invariants(self) -> None:
        self.assertEqual(STOICHIOMETRIC_MATRIX.shape, (N_PROCESSES, N_COMPONENTS))
        self.assertEqual(INVARIANT_MATRIX.shape, (5, N_COMPONENTS))
        audit = audit_mechanistic_matrices()
        self.assertTrue(audit["passed"], audit)
        self.assertLessEqual(float(np.max(np.abs(INVARIANT_MATRIX @ STOICHIOMETRIC_MATRIX.T))), 1e-12)

    def test_rates_are_complete_finite_and_nonnegative(self) -> None:
        rates = process_rates(NOMINAL_INFLUENT)
        self.assertEqual(rates.shape, (N_PROCESSES,))
        self.assertTrue(np.all(np.isfinite(rates)))
        self.assertTrue(np.all(rates >= 0.0))
        # The corrected fermentation rate is q_fe itself, not q_fe * mu_H.
        ix = COMPONENT_INDEX
        c = NOMINAL_INFLUENT
        p11_expected = (
            3.0 * (.2 / (.2 + c[ix["S_O"]]))
            * (.5 / (.5 + c[ix["S_NO2"]] + c[ix["S_NO3"]]))
            * c[ix["S_F"]] / (4.0 + c[ix["S_F"]])
            * c[ix["S_ALK"]] / (.1 + c[ix["S_ALK"]]) * c[ix["X_H"]]
        )
        self.assertAlmostEqual(rates[10], p11_expected, places=12)


class ClarifierAndRecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.operating = OperatingPoint(18.0, .6, 2.0, .75, .02)
        self.feed = NOMINAL_INFLUENT.copy()
        self.feed_tss = float(TSS_VECTOR @ self.feed)
        self.layers = self.feed_tss * np.asarray([.01, .02, .05, .1, .2, .5, 1., 2., 3., 4.])

    def test_settling_bounds_and_flux_dimensions(self) -> None:
        velocity = settling_velocity(self.layers, self.feed_tss)
        self.assertTrue(np.all(velocity >= 0.0))
        self.assertTrue(np.all(velocity <= CLARIFIER.maximum_settling_velocity))
        flux = clarifier_fluxes(self.layers, self.feed_tss, self.operating)
        self.assertEqual(flux.shape, (N_LAYERS + 1,))
        self.assertAlmostEqual(
            flux[0],
            -CLARIFIER.fresh_flow * self.operating.q_effluent / CLARIFIER.area * self.layers[0],
        )

    def test_layer_balance_telescopes_to_external_tss_balance(self) -> None:
        rhs = clarifier_rhs(self.layers, self.feed_tss, self.operating)
        accumulated = CLARIFIER.layer_volume * rhs.sum()
        external = CLARIFIER.fresh_flow * (
            self.operating.q_clarifier * self.feed_tss
            - self.operating.q_effluent * self.layers[0]
            - self.operating.q_underflow * self.layers[-1]
        )
        self.assertAlmostEqual(accumulated, external, places=6)

    def test_component_reconstruction_and_mixer_close(self) -> None:
        ce, cu = reconstruct_clarifier(self.feed, self.layers)
        np.testing.assert_allclose(ce[:10], self.feed[:10])
        np.testing.assert_allclose(cu[:10], self.feed[:10])
        self.assertAlmostEqual(float(TSS_VECTOR @ ce), self.layers[0])
        self.assertAlmostEqual(float(TSS_VECTOR @ cu), self.layers[-1])
        mixer = mixer_state(NOMINAL_INFLUENT, self.feed, cu, self.operating)
        balance = (
            self.operating.q_process * mixer - NOMINAL_INFLUENT
            - self.operating.internal_recycle * self.feed
            - self.operating.return_sludge * cu
        )
        np.testing.assert_allclose(balance, 0.0, atol=1e-12)

    def test_coupled_state_and_target_dimensions(self) -> None:
        state = initial_state(NOMINAL_INFLUENT, start=2)
        self.assertEqual(state.shape, (STATE_SIZE,))
        derivative = coupled_rhs(state, self.operating, NOMINAL_INFLUENT)
        self.assertEqual(derivative.shape, (STATE_SIZE,))
        self.assertTrue(np.all(np.isfinite(derivative)))
        target = assemble_target(state, self.operating, NOMINAL_INFLUENT)
        self.assertEqual(target.shape, (TARGET_SIZE,))


class SteadySolverSmokeTests(unittest.TestCase):
    def test_exact_zero_fixture_exercises_full_solve_route(self) -> None:
        result = zero_state_solution()
        self.assertTrue(result.accepted, result.diagnostics)
        # scipy keeps bound-active coordinates infinitesimally inside the
        # orthant, so replay against the manuscript's 1e-8 contract.
        self.assertLessEqual(float(result.diagnostics["scaled_residual_inf"]), 1e-8)
        self.assertEqual(result.target.shape, (TARGET_SIZE,))

    def test_loaded_nominal_state_converges_and_is_stable(self) -> None:
        result = solve_steady_state(
            OperatingPoint(21.0, .5, 2.0, .75, .02),
            NOMINAL_INFLUENT,
            starts=(1,),
        )
        self.assertTrue(result.accepted, result.diagnostics)
        self.assertEqual(result.route, "scaled-bdf")
        self.assertLessEqual(float(result.diagnostics["scaled_residual_inf"]), 1e-8)
        self.assertLessEqual(float(result.diagnostics["largest_real_eigenvalue"]), 1e-8)

    def test_high_throughflow_box_corner_converges(self) -> None:
        result = solve_steady_state(
            OperatingPoint(6.0, 1.0, 4.0, 1.25, .05),
            NOMINAL_INFLUENT,
            starts=(1,),
        )
        self.assertTrue(result.accepted, result.diagnostics)

    def test_zero_aeration_low_waste_corner_uses_positive_fallback(self) -> None:
        result = solve_steady_state(
            OperatingPoint(36.0, 0.0, 0.0, .25, .001),
            NOMINAL_INFLUENT,
            starts=(1,),
        )
        self.assertTrue(result.accepted, result.diagnostics)
        self.assertEqual(result.route, "log-bdf")
        self.assertGreaterEqual(float(result.diagnostics["minimum_state"]), -1e-10)

    def test_slow_solids_mode_closes_the_plant_boundary(self) -> None:
        # This prescribed design exercises a slow Clarifier inventory mode for
        # which the local derivative can be small before the accumulated plant
        # boundary balance reaches the independent 1e-8 acceptance threshold.
        decisions = np.array([
            7.601629632672628,
            0.531404810145,
            0.06715255853995941,
            0.6228666363318475,
            0.01572435395674802,
        ])
        influent = np.array([
            0.26290134172788404, 151.58546177298450, 37.248727952966853,
            24.010258042743573, 2.7188025593036338, 1.0455364709367772,
            0.080097917739729033, 11.835171908208736, 69.369743646875321,
            2.4450814032412236, 74.678176147930898, 247.41435225227497,
            42.011655432467435, 54.241480515968988, 7.1726225681351545,
            29.302221935565996, 5.8838010483177943, 1.8092176754970950,
            7.0793671585974449, 2.8251250461849899,
        ])
        result = solve_steady_state(OperatingPoint(*decisions), influent)
        self.assertTrue(result.accepted, result.diagnostics)
        self.assertLessEqual(float(result.diagnostics["plant_boundary_residual"]), 1e-8)


if __name__ == "__main__":
    unittest.main()
