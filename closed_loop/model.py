"""Coupled ASM2d-TSN mixer--reactor--clarifier model.

The implementation follows the complete mechanistic definition in
``article/wip_v3/supplementary_material.tex``.  It deliberately keeps the recycle mixer
and clarifier outlets algebraic: the dynamic/steady unknown is therefore the
five 20-component reactor states followed by ten clarifier-layer TSS states.
No concentration or reaction rate is clipped during a solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

COMPONENTS: tuple[str, ...] = (
    "S_O",
    "S_F",
    "S_A",
    "S_NH4",
    "S_NO2",
    "S_NO3",
    "S_N2",
    "S_PO4",
    "S_I",
    "S_ALK",
    "X_I",
    "X_S",
    "X_H",
    "X_PAO",
    "X_PP",
    "X_PHA",
    "X_AOB",
    "X_NOB",
    "X_MeP",
    "X_MeOH",
)
COMPONENT_INDEX: Mapping[str, int] = {name: i for i, name in enumerate(COMPONENTS)}
N_COMPONENTS = len(COMPONENTS)
N_PROCESSES = 28
N_STAGES = 5
N_LAYERS = 10
STATE_SIZE = N_STAGES * N_COMPONENTS + N_LAYERS
TARGET_SIZE = N_COMPONENTS * (1 + N_STAGES + 2) + N_LAYERS
SOLUBLE = np.arange(10, dtype=int)
PARTICULATE = np.arange(10, 20, dtype=int)

INFLUENT_LOWER = np.asarray(
    [0.0, 20.0, 5.0, 12.0, 0.0, 0.0, 0.0, 2.0, 10.0, 1.6,
     20.0, 60.0, 15.0, 5.0, 2.0, 1.0, 0.5, 0.5, 0.0, 0.0],
    dtype=float,
)
INFLUENT_UPPER = np.asarray(
    [0.5, 180.0, 80.0, 55.0, 3.0, 8.0, 2.0, 18.0, 90.0, 5.2,
     120.0, 280.0, 100.0, 60.0, 20.0, 30.0, 8.0, 8.0, 12.0, 12.0],
    dtype=float,
)
NOMINAL_INFLUENT = (INFLUENT_LOWER + INFLUENT_UPPER) / 2.0

# Rows are COD, reported TN, TP, and TSS in the fixed component order.
I_PMEP = 1.0 / 4.87
COMPOSITE_MATRIX = np.asarray(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, .01, 0, .02, 0, .07, .07, 0, 0, .07, .07, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, .01, 0, .02, .02, 1, 0, .02, .02, I_PMEP, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .75, .75, .90, .90, 3.23, .60, .90, .90, 1, 1],
    ],
    dtype=float,
)
TSS_VECTOR = COMPOSITE_MATRIX[3].copy()


@dataclass(frozen=True)
class OperatingPoint:
    """Five case-study design and operating decisions."""

    hrt_hours: float
    aeration: float
    internal_recycle: float
    return_sludge: float
    waste_sludge: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.hrt_hours, self.aeration, self.internal_recycle,
             self.return_sludge, self.waste_sludge], dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Operating decisions must be finite.")
        if self.hrt_hours <= 0.0:
            raise ValueError("hrt_hours must be positive.")
        if not 0.0 <= self.aeration <= 1.0:
            raise ValueError("aeration must lie in [0, 1].")
        if self.internal_recycle < 0.0 or self.return_sludge <= 0.0:
            raise ValueError("Recycle ratios must satisfy r_I >= 0 and r_R > 0.")
        if not 0.0 <= self.waste_sludge < 1.0:
            raise ValueError("waste_sludge must lie in [0, 1).")

    @property
    def q_process(self) -> float:
        return 1.0 + self.internal_recycle + self.return_sludge

    @property
    def q_clarifier(self) -> float:
        return 1.0 + self.return_sludge

    @property
    def q_underflow(self) -> float:
        return self.return_sludge + self.waste_sludge

    @property
    def q_effluent(self) -> float:
        return 1.0 - self.waste_sludge

    @property
    def stage_dilution_rate(self) -> float:
        """Actual stage throughflow divided by one stage volume, d^-1."""

        return 120.0 * self.q_process / self.hrt_hours

    def aeration_for_stage(self, stage: int) -> float:
        """Return the stage-specific aeration setting (legacy tied-aeration case)."""

        return 0.0 if stage < 2 else self.aeration


@dataclass(frozen=True)
class ArticleOperatingPoint:
    """Seven-control operating point specified by the wip_v3 manuscript."""

    hrt_hours: float
    aeration_3: float
    aeration_4: float
    aeration_5: float
    internal_recycle: float
    return_sludge: float
    waste_sludge: float

    def __post_init__(self) -> None:
        values = np.asarray(self.as_array(), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Operating decisions must be finite.")
        if self.hrt_hours <= 0.0:
            raise ValueError("hrt_hours must be positive.")
        if np.any((values[1:4] < 0.0) | (values[1:4] > 1.0)):
            raise ValueError("Stage aeration settings must lie in [0, 1].")
        if self.internal_recycle < 0.0 or self.return_sludge <= 0.0:
            raise ValueError("Recycle ratios must satisfy r_I >= 0 and r_R > 0.")
        if not 0.0 <= self.waste_sludge < 1.0:
            raise ValueError("waste_sludge must lie in [0, 1).")

    def as_array(self) -> FloatArray:
        return np.asarray([
            self.hrt_hours, self.aeration_3, self.aeration_4,
            self.aeration_5, self.internal_recycle, self.return_sludge,
            self.waste_sludge,
        ], dtype=float)

    def aeration_for_stage(self, stage: int) -> float:
        if stage not in range(N_STAGES):
            raise ValueError("stage must be in range(5).")
        return 0.0 if stage < 2 else (self.aeration_3, self.aeration_4, self.aeration_5)[stage - 2]

    @property
    def q_process(self) -> float:
        return 1.0 + self.internal_recycle + self.return_sludge

    @property
    def q_clarifier(self) -> float:
        return 1.0 + self.return_sludge

    @property
    def q_underflow(self) -> float:
        return self.return_sludge + self.waste_sludge

    @property
    def q_effluent(self) -> float:
        return 1.0 - self.waste_sludge

    @property
    def stage_dilution_rate(self) -> float:
        return 120.0 * self.q_process / self.hrt_hours


@dataclass(frozen=True)
class ClarifierParameters:
    fresh_flow: float = 10_000.0
    area: float = 1_500.0
    layer_volume: float = 600.0
    maximum_settling_velocity: float = 250.0
    theoretical_settling_velocity: float = 474.0
    hindered_coefficient: float = 0.000576
    low_concentration_coefficient: float = 0.00286
    nonsettleable_fraction: float = 0.00228
    flux_threshold: float = 3_000.0
    layer_count: int = 10
    feed_layer: int = 4  # zero-based: fifth layer from the top

    def __post_init__(self) -> None:
        if self.layer_count < 3:
            raise ValueError("A Clarifier requires at least three layers.")
        if self.feed_layer not in range(self.layer_count):
            raise ValueError("feed_layer must identify an existing Clarifier layer.")
        positive = (
            self.fresh_flow, self.area, self.layer_volume,
            self.maximum_settling_velocity, self.theoretical_settling_velocity,
            self.hindered_coefficient, self.low_concentration_coefficient,
            self.flux_threshold,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Clarifier geometry and settling parameters must be positive and finite.")
        if not 0.0 <= self.nonsettleable_fraction <= 1.0:
            raise ValueError("nonsettleable_fraction must lie in [0, 1].")


CLARIFIER = ClarifierParameters()

# All fixed ASM2d-TSN kinetic and stoichiometric values in the manuscript.
PARAMETERS: Mapping[str, float] = {
    "Y_H": .625, "Y_PAO": .625, "Y_PHA": .20, "Y_PO4": .40,
    "Y_AOB": .18, "Y_NOB": .08, "f_SI": 0.0, "f_XI": .10,
    "i_NSI": .01, "i_NSF": 0.0, "i_NXI": .02, "i_NXS": 0.0,
    "i_NBM": .07, "i_PSI": 0.0, "i_PSF": 0.0, "i_PXI": .01,
    "i_PXS": 0.0, "i_PBM": .02, "i_PMeP": I_PMEP,
    "K_H": 3.0, "eta_hyd_NO3": .6, "eta_hyd_NO2": .6,
    "eta_hyd_fe": .4, "K_O_hyd": .20, "K_NO3_hyd": .5,
    "K_NO2_hyd": .5, "K_NOx_hyd": .5, "K_X": .10,
    "mu_H": 6.0, "q_fe": 3.0, "b_H": .40,
    "eta_H_NO3": .9, "eta_H_NO2": .9, "K_O_H": .20,
    "K_F": 4.0, "K_fe": 4.0, "K_A": 4.0, "K_NO3_H": .5,
    "K_NO2_H": .5, "K_NOx_H": .5, "K_NH4_H": .05,
    "K_PO4_H": .01, "K_ALK_H": .10,
    "q_PHA": 5.0, "q_PP": .60, "mu_PAO": .56,
    "eta_PAO_NO3": .07, "eta_PAO_NO2": .90,
    "b_PAO": .20, "b_PP": .20, "b_PHA": .20, "K_O_PAO": .20,
    "K_NO3_PAO": .5, "K_NO2_PAO": .5, "K_NOx_PAO": .5,
    "K_NH4_PAO": .05, "K_PS": .20, "K_PO4_PAO": .01,
    "K_ALK_PAO": .10, "K_PP": .01, "K_max": .34,
    "K_IPP": .02, "K_PHA": .01,
    "mu_AOB": 1.81, "mu_NOB": 1.52, "b_AOB": .20,
    "b_NOB": .17, "K_O_AOB": .74, "K_O_NOB": 1.75,
    "K_NH4_AOB": .5, "K_NO2_NOB": .5, "K_NH4_NOB": .05, "K_ALK_nit": .5,
    "K_PO4_nit": .01, "k_PRE": 1.0, "k_RED": .60,
    "gamma_OH": 3.45, "K_ALK_PRE": .50, "K_ALK_chem": .50,
}


def _e(name: str) -> FloatArray:
    vector = np.zeros(N_COMPONENTS, dtype=float)
    vector[COMPONENT_INDEX[name]] = 1.0
    return vector


def build_stoichiometric_matrix(parameters: Mapping[str, float] = PARAMETERS) -> FloatArray:
    """Assemble the complete 28-by-20 Petersen matrix from continuity rules."""

    p = parameters
    nu = np.zeros((N_PROCESSES, N_COMPONENTS), dtype=float)
    e = {name: _e(name) for name in COMPONENTS}
    yh, ypao, ypha = p["Y_H"], p["Y_PAO"], p["Y_PHA"]
    beta_h = (1.0 - yh) / ((8.0 / 7.0) * yh)
    gamma_h = (1.0 - yh) / (1.72 * yh)
    beta_s, gamma_s = ypha / (8.0 / 7.0), ypha / 1.72
    beta_p = (1.0 - ypao) / ((8.0 / 7.0) * ypao)
    gamma_p = (1.0 - ypao) / (1.72 * ypao)

    hydrolysis = (1.0 - p["f_SI"]) * e["S_F"] + p["f_SI"] * e["S_I"] - e["X_S"]
    nu[0:4] = hydrolysis
    nu[4] = -(1.0 / yh) * e["S_F"] + (1.0 - 1.0 / yh) * e["S_O"] + e["X_H"]
    nu[5] = -(1.0 / yh) * e["S_A"] + (1.0 - 1.0 / yh) * e["S_O"] + e["X_H"]
    nu[6] = -(1.0 / yh) * e["S_F"] + beta_h * (e["S_NO2"] - e["S_NO3"]) + e["X_H"]
    nu[7] = -(1.0 / yh) * e["S_F"] + gamma_h * (e["S_N2"] - e["S_NO2"]) + e["X_H"]
    nu[8] = -(1.0 / yh) * e["S_A"] + beta_h * (e["S_NO2"] - e["S_NO3"]) + e["X_H"]
    nu[9] = -(1.0 / yh) * e["S_A"] + gamma_h * (e["S_N2"] - e["S_NO2"]) + e["X_H"]
    nu[10] = e["S_A"] - e["S_F"]
    decay_h = p["f_XI"] * e["X_I"] + (1.0 - p["f_XI"]) * e["X_S"]
    nu[11] = decay_h - e["X_H"]
    nu[12] = -e["S_A"] - p["Y_PO4"] * e["X_PP"] + e["X_PHA"]
    nu[13] = -ypha * e["S_O"] + e["X_PP"] - ypha * e["X_PHA"]
    nu[14] = beta_s * (e["S_NO2"] - e["S_NO3"]) + e["X_PP"] - ypha * e["X_PHA"]
    nu[15] = gamma_s * (e["S_N2"] - e["S_NO2"]) + e["X_PP"] - ypha * e["X_PHA"]
    nu[16] = (1.0 - 1.0 / ypao) * e["S_O"] + e["X_PAO"] - (1.0 / ypao) * e["X_PHA"]
    nu[17] = beta_p * (e["S_NO2"] - e["S_NO3"]) + e["X_PAO"] - (1.0 / ypao) * e["X_PHA"]
    nu[18] = gamma_p * (e["S_N2"] - e["S_NO2"]) + e["X_PAO"] - (1.0 / ypao) * e["X_PHA"]
    nu[19] = decay_h - e["X_PAO"]
    nu[20] = -e["X_PP"]
    nu[21] = e["S_A"] - e["X_PHA"]
    nu[22] = (-(3.43 - p["Y_AOB"]) / p["Y_AOB"] * e["S_O"]
              + (1.0 / p["Y_AOB"]) * e["S_NO2"] + e["X_AOB"])
    nu[23] = (-(1.14 - p["Y_NOB"]) / p["Y_NOB"] * e["S_O"]
              - (1.0 / p["Y_NOB"]) * e["S_NO2"]
              + (1.0 / p["Y_NOB"]) * e["S_NO3"] + e["X_NOB"])
    nu[24] = decay_h - e["X_AOB"]
    nu[25] = decay_h - e["X_NOB"]
    nu[26] = -e["S_PO4"] - p["gamma_OH"] * e["X_MeOH"] + (1.0 / p["i_PMeP"]) * e["X_MeP"]
    nu[27] = e["S_PO4"] + p["gamma_OH"] * e["X_MeOH"] - (1.0 / p["i_PMeP"]) * e["X_MeP"]

    n_weights = {
        "S_F": p["i_NSF"], "S_I": p["i_NSI"], "S_N2": 1.0,
        "S_NO2": 1.0, "S_NO3": 1.0, "X_I": p["i_NXI"],
        "X_S": p["i_NXS"], "X_H": p["i_NBM"], "X_PAO": p["i_NBM"],
        "X_AOB": p["i_NBM"], "X_NOB": p["i_NBM"],
    }
    nu[:, COMPONENT_INDEX["S_NH4"]] = -sum(
        weight * nu[:, COMPONENT_INDEX[name]] for name, weight in n_weights.items()
    )
    p_weights = {
        "S_F": p["i_PSF"], "S_I": p["i_PSI"], "X_I": p["i_PXI"],
        "X_S": p["i_PXS"], "X_H": p["i_PBM"], "X_PAO": p["i_PBM"],
        "X_AOB": p["i_PBM"], "X_NOB": p["i_PBM"], "X_PP": 1.0,
        "X_MeP": p["i_PMeP"],
    }
    phosphorus = -sum(
        weight * nu[:, COMPONENT_INDEX[name]] for name, weight in p_weights.items()
    )
    nu[:26, COMPONENT_INDEX["S_PO4"]] = phosphorus[:26]
    nu[:, COMPONENT_INDEX["S_ALK"]] = (
        nu[:, COMPONENT_INDEX["S_NH4"]] / 14.0
        - nu[:, COMPONENT_INDEX["S_NO2"]] / 14.0
        - nu[:, COMPONENT_INDEX["S_NO3"]] / 14.0
        + nu[:, COMPONENT_INDEX["S_PO4"]] / 31.0
    )
    return nu


STOICHIOMETRIC_MATRIX = build_stoichiometric_matrix()


def build_invariant_matrix(
    parameters: Mapping[str, float] = PARAMETERS, *, normalized: bool = True,
) -> FloatArray:
    """Return the five named reaction- and aeration-invariant inventory rows."""

    p = parameters
    e = {name: _e(name) for name in COMPONENTS}
    b_si = e["S_I"]
    b_n = (
        e["S_NH4"] + e["S_NO2"] + e["S_NO3"] + e["S_N2"]
        + p["i_NSI"] * e["S_I"] + p["i_NSF"] * e["S_F"]
        + p["i_NXI"] * e["X_I"] + p["i_NXS"] * e["X_S"]
        + p["i_NBM"] * (e["X_H"] + e["X_PAO"] + e["X_AOB"] + e["X_NOB"])
    )
    b_p = (
        e["S_PO4"] + p["i_PSI"] * e["S_I"] + p["i_PSF"] * e["S_F"]
        + p["i_PXI"] * e["X_I"] + p["i_PXS"] * e["X_S"] + e["X_PP"]
        + p["i_PBM"] * (e["X_H"] + e["X_PAO"] + e["X_AOB"] + e["X_NOB"])
        + p["i_PMeP"] * e["X_MeP"]
    )
    b_alk = e["S_ALK"] - e["S_NH4"] / 14.0 + e["S_NO2"] / 14.0 + e["S_NO3"] / 14.0 - e["S_PO4"] / 31.0
    b_metal = p["gamma_OH"] * e["X_MeP"] + (1.0 / p["i_PMeP"]) * e["X_MeOH"]
    matrix = np.vstack((b_si, b_n, b_p, b_alk, b_metal))
    if normalized:
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


INVARIANT_MATRIX = build_invariant_matrix()


def audit_mechanistic_matrices(tolerance: float = 1e-10) -> dict[str, float | int | bool]:
    """Evaluate the manuscript's rank, invariant, and normalization contract."""

    nu, invariant = STOICHIOMETRIC_MATRIX, INVARIANT_MATRIX
    row_norm_error = float(np.max(np.abs(np.linalg.norm(invariant, axis=1) - 1.0)))
    invariant_error = float(np.max(np.abs(invariant @ nu.T)))
    aeration_error = float(np.max(np.abs(invariant @ _e("S_O"))))
    result: dict[str, float | int | bool] = {
        "stoichiometric_rank": int(np.linalg.matrix_rank(nu)),
        "invariant_rank": int(np.linalg.matrix_rank(invariant)),
        "invariant_error": invariant_error,
        "aeration_invariant_error": aeration_error,
        "row_norm_error": row_norm_error,
    }
    result["passed"] = bool(
        result["stoichiometric_rank"] == 15
        and result["invariant_rank"] == 5
        and invariant_error <= tolerance
        and aeration_error <= tolerance
        and row_norm_error <= tolerance
    )
    return result


def _state(state: ArrayLike, expected_size: int = N_COMPONENTS) -> FloatArray:
    result = np.asarray(state, dtype=float)
    if result.shape != (expected_size,):
        raise ValueError(f"Expected shape ({expected_size},), received {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError("State values must be finite.")
    return result


def _monod(value: float, half_saturation: float) -> float:
    return value / (half_saturation + value)


def _inhibition(value: float, half_saturation: float) -> float:
    return half_saturation / (half_saturation + value)


def _share(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def process_rates(state: ArrayLike, parameters: Mapping[str, float] = PARAMETERS) -> FloatArray:
    """Return all 28 non-negative process rates at one component state."""

    c = _state(state)
    if np.min(c) < 0.0:
        raise ValueError("ASM2d-TSN rates are defined only for non-negative states.")
    p, ix = parameters, COMPONENT_INDEX
    so, sf, sa = c[ix["S_O"]], c[ix["S_F"]], c[ix["S_A"]]
    snh4, sno2, sno3 = c[ix["S_NH4"]], c[ix["S_NO2"]], c[ix["S_NO3"]]
    spo4, salk = c[ix["S_PO4"]], c[ix["S_ALK"]]
    xs, xh, xpao = c[ix["X_S"]], c[ix["X_H"]], c[ix["X_PAO"]]
    xpp, xpha = c[ix["X_PP"]], c[ix["X_PHA"]]
    xaob, xnob = c[ix["X_AOB"]], c[ix["X_NOB"]]
    xmep, xmeoh = c[ix["X_MeP"]], c[ix["X_MeOH"]]
    snox = sno2 + sno3
    alpha2, alpha3 = _share(sno2, snox), _share(sno3, snox)
    carbon = sf + sa
    alpha_f, alpha_a = _share(sf, carbon), _share(sa, carbon)
    theta_x = _share(xs, p["K_X"] * xh + xs)
    r_pp = _share(xpp, xpao) if xpao > 0.0 else 0.0
    r_pha = _share(xpha, xpao) if xpao > 0.0 else 0.0
    pi_pp, pi_pha = _monod(r_pp, p["K_PP"]), _monod(r_pha, p["K_PHA"])
    capacity = max(p["K_max"] - r_pp, 0.0)
    c_pp = capacity / (p["K_IPP"] + capacity)
    lh = _monod(snh4, p["K_NH4_H"]) * _monod(spo4, p["K_PO4_H"]) * _monod(salk, p["K_ALK_H"])
    lp = _monod(snh4, p["K_NH4_PAO"]) * _monod(spo4, p["K_PO4_PAO"]) * _monod(salk, p["K_ALK_PAO"])
    ln = _monod(spo4, p["K_PO4_nit"]) * _monod(salk, p["K_ALK_nit"])
    mo_hyd, io_hyd = _monod(so, p["K_O_hyd"]), _inhibition(so, p["K_O_hyd"])
    mo_h, io_h = _monod(so, p["K_O_H"]), _inhibition(so, p["K_O_H"])
    mo_p, io_p = _monod(so, p["K_O_PAO"]), _inhibition(so, p["K_O_PAO"])
    alk_p = _monod(salk, p["K_ALK_PAO"])
    common_pp = p["q_PP"] * _monod(spo4, p["K_PS"]) * alk_p * pi_pha * c_pp

    rates = np.asarray(
        [
            p["K_H"] * mo_hyd * theta_x * xh,
            p["eta_hyd_NO2"] * p["K_H"] * io_hyd * _monod(sno2, p["K_NO2_hyd"]) * alpha2 * theta_x * xh,
            p["eta_hyd_NO3"] * p["K_H"] * io_hyd * _monod(sno3, p["K_NO3_hyd"]) * alpha3 * theta_x * xh,
            p["eta_hyd_fe"] * p["K_H"] * io_hyd * _inhibition(snox, p["K_NOx_hyd"]) * theta_x * xh,
            p["mu_H"] * mo_h * _monod(sf, p["K_F"]) * alpha_f * lh * xh,
            p["mu_H"] * mo_h * _monod(sa, p["K_A"]) * alpha_a * lh * xh,
            p["mu_H"] * io_h * _monod(sf, p["K_F"]) * alpha_f * lh * p["eta_H_NO3"] * _monod(sno3, p["K_NO3_H"]) * alpha3 * xh,
            p["mu_H"] * io_h * _monod(sf, p["K_F"]) * alpha_f * lh * p["eta_H_NO2"] * _monod(sno2, p["K_NO2_H"]) * alpha2 * xh,
            p["mu_H"] * io_h * _monod(sa, p["K_A"]) * alpha_a * lh * p["eta_H_NO3"] * _monod(sno3, p["K_NO3_H"]) * alpha3 * xh,
            p["mu_H"] * io_h * _monod(sa, p["K_A"]) * alpha_a * lh * p["eta_H_NO2"] * _monod(sno2, p["K_NO2_H"]) * alpha2 * xh,
            p["q_fe"] * io_h * _inhibition(snox, p["K_NOx_H"]) * _monod(sf, p["K_fe"]) * _monod(salk, p["K_ALK_H"]) * xh,
            p["b_H"] * xh,
            p["q_PHA"] * _monod(sa, p["K_A"]) * io_p * _inhibition(snox, p["K_NOx_PAO"]) * alk_p * pi_pp * xpao,
            common_pp * mo_p * xpao,
            common_pp * io_p * p["eta_PAO_NO3"] * _monod(sno3, p["K_NO3_PAO"]) * alpha3 * xpao,
            common_pp * io_p * p["eta_PAO_NO2"] * _monod(sno2, p["K_NO2_PAO"]) * alpha2 * xpao,
            p["mu_PAO"] * mo_p * lp * pi_pha * xpao,
            p["mu_PAO"] * io_p * lp * pi_pha * p["eta_PAO_NO3"] * _monod(sno3, p["K_NO3_PAO"]) * alpha3 * xpao,
            p["mu_PAO"] * io_p * lp * pi_pha * p["eta_PAO_NO2"] * _monod(sno2, p["K_NO2_PAO"]) * alpha2 * xpao,
            p["b_PAO"] * alk_p * xpao,
            p["b_PP"] * alk_p * xpp,
            p["b_PHA"] * alk_p * xpha,
            p["mu_AOB"] * _monod(so, p["K_O_AOB"]) * _monod(snh4, p["K_NH4_AOB"]) * ln * xaob,
            p["mu_NOB"] * _monod(so, p["K_O_NOB"]) * _monod(sno2, p["K_NO2_NOB"]) * ln * _monod(snh4, p["K_NH4_NOB"]) * xnob,
            p["b_AOB"] * xaob,
            p["b_NOB"] * xnob,
            p["k_PRE"] * spo4 * xmeoh * _monod(salk, p["K_ALK_PRE"]),
            p["k_RED"] * p["i_PMeP"] * _monod(salk, p["K_ALK_chem"]) * xmep,
        ],
        dtype=float,
    )
    if rates.shape != (N_PROCESSES,) or not np.all(np.isfinite(rates)) or np.min(rates) < 0.0:
        raise FloatingPointError("Every ASM2d-TSN process rate must be finite and non-negative.")
    return rates


def reaction_source(state: ArrayLike) -> FloatArray:
    """Component production rates, nu.T @ rho, in concentration per day."""

    return STOICHIOMETRIC_MATRIX.T @ process_rates(state)


def oxygen_transfer(state: ArrayLike, stage: int, aeration: float) -> float:
    """Unclipped oxygen-transfer source for one zero-based reactor stage."""

    if stage not in range(N_STAGES):
        raise ValueError("stage must be in range(5).")
    return 0.0 if stage < 2 else 47.0 * aeration * (8.5 - _state(state)[0])


def _stage_aeration(operating: OperatingPoint | ArticleOperatingPoint, stage: int) -> float:
    return float(operating.aeration_for_stage(stage))


def settling_velocity(
    solids: ArrayLike | float,
    feed_tss: float,
    parameters: ClarifierParameters = CLARIFIER,
) -> FloatArray | float:
    """Takacs double-exponential velocity with its physical outer bounds."""

    values = np.asarray(solids, dtype=float)
    if np.any(values < 0.0) or feed_tss < 0.0:
        raise ValueError("TSS concentrations must be non-negative.")
    delta = values - parameters.nonsettleable_fraction * feed_tss
    with np.errstate(over="ignore", invalid="ignore"):
        raw = parameters.theoretical_settling_velocity * (
            np.exp(-parameters.hindered_coefficient * delta)
            - np.exp(-parameters.low_concentration_coefficient * delta)
        )
    velocity = np.maximum(0.0, np.minimum(parameters.maximum_settling_velocity, raw))
    if not np.all(np.isfinite(velocity)):
        raise FloatingPointError("Settling velocity is non-finite.")
    return float(velocity) if values.ndim == 0 else velocity


def clarifier_fluxes(
    layers: ArrayLike,
    feed_tss: float,
    operating: OperatingPoint,
    parameters: ClarifierParameters = CLARIFIER,
) -> FloatArray:
    """Return 11 downward-positive boundary/interface TSS fluxes."""

    layer_count = parameters.layer_count
    s = _state(layers, layer_count)
    if np.min(s) < 0.0:
        raise ValueError("Clarifier layer TSS must be non-negative.")
    velocity = np.asarray(settling_velocity(s, feed_tss, parameters), dtype=float)
    gravity = velocity * s
    flux = np.empty(layer_count + 1, dtype=float)
    v_e = parameters.fresh_flow * operating.q_effluent / parameters.area
    v_u = parameters.fresh_flow * operating.q_underflow / parameters.area
    flux[0] = -v_e * s[0]
    for left in range(layer_count - 1):
        settling = gravity[left]
        if s[left + 1] > parameters.flux_threshold:
            settling = min(settling, gravity[left + 1])
        flux[left + 1] = (
            -v_e * s[left + 1] + settling
            if left < parameters.feed_layer
            else v_u * s[left] + settling
        )
    flux[-1] = v_u * s[-1]
    if not np.all(np.isfinite(flux)):
        raise FloatingPointError("Clarifier interface flux is non-finite.")
    return flux


def clarifier_rhs(
    layers: ArrayLike,
    feed_tss: float,
    operating: OperatingPoint,
    parameters: ClarifierParameters = CLARIFIER,
) -> FloatArray:
    """Finite-volume derivatives for the ten clarifier TSS layers."""

    flux = clarifier_fluxes(layers, feed_tss, operating, parameters)
    derivative = parameters.area * (flux[:-1] - flux[1:]) / parameters.layer_volume
    derivative[parameters.feed_layer] += (
        parameters.fresh_flow * operating.q_clarifier * feed_tss / parameters.layer_volume
    )
    return derivative


def reconstruct_clarifier(
    reactor_outlet: ArrayLike, layers: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Reconstruct overflow and underflow component concentrations."""

    feed, s = _state(reactor_outlet), np.asarray(layers, dtype=float)
    if s.shape != (s.size,) or s.size < 3:
        raise ValueError("Clarifier layers must be a one-dimensional vector with at least three entries.")
    if np.min(feed) < 0.0 or np.min(s) < 0.0:
        raise ValueError("Clarifier reconstruction requires non-negative states.")
    effluent, underflow = feed.copy(), feed.copy()
    feed_tss = float(TSS_VECTOR @ feed)
    if feed_tss == 0.0:
        effluent[PARTICULATE] = 0.0
        underflow[PARTICULATE] = 0.0
    else:
        fractions = feed[PARTICULATE] / feed_tss
        effluent[PARTICULATE] = fractions * s[0]
        underflow[PARTICULATE] = fractions * s[-1]
    return effluent, underflow


def mixer_state(
    influent: ArrayLike,
    reactor_outlet: ArrayLike,
    clarifier_underflow: ArrayLike,
    operating: OperatingPoint,
) -> FloatArray:
    """Close the headworks mixer around fresh feed, MLR, and RAS."""

    x, c5, cu = _state(influent), _state(reactor_outlet), _state(clarifier_underflow)
    return (
        x + operating.internal_recycle * c5 + operating.return_sludge * cu
    ) / operating.q_process


def state_size(clarifier: ClarifierParameters = CLARIFIER) -> int:
    return N_STAGES * N_COMPONENTS + clarifier.layer_count


def target_size(clarifier: ClarifierParameters = CLARIFIER) -> int:
    return N_COMPONENTS * (1 + N_STAGES + 2) + clarifier.layer_count


def unpack_state(
    state: ArrayLike, clarifier: ClarifierParameters = CLARIFIER,
) -> tuple[FloatArray, FloatArray]:
    values = _state(state, state_size(clarifier))
    return (
        values[: N_STAGES * N_COMPONENTS].reshape(N_STAGES, N_COMPONENTS),
        values[-clarifier.layer_count :],
    )


def coupled_rhs(
    state: ArrayLike,
    operating: OperatingPoint,
    influent: ArrayLike,
    clarifier: ClarifierParameters = CLARIFIER,
) -> FloatArray:
    """Coupled 110-state derivative after eliminating mixer/outlet algebraics."""

    reactors, layers = unpack_state(state, clarifier)
    x = _state(influent)
    if np.min(reactors) < 0.0 or np.min(layers) < 0.0 or np.min(x) < 0.0:
        raise ValueError("The coupled model is defined on the non-negative orthant.")
    _, underflow = reconstruct_clarifier(reactors[-1], layers)
    mixer = mixer_state(x, reactors[-1], underflow, operating)
    derivative = np.empty_like(reactors)
    upstream = mixer
    for stage in range(N_STAGES):
        source = reaction_source(reactors[stage])
        source[COMPONENT_INDEX["S_O"]] += oxygen_transfer(
            reactors[stage], stage, _stage_aeration(operating, stage)
        )
        derivative[stage] = operating.stage_dilution_rate * (upstream - reactors[stage]) + source
        upstream = reactors[stage]
    feed_tss = float(TSS_VECTOR @ reactors[-1])
    layer_derivative = clarifier_rhs(layers, feed_tss, operating, clarifier)
    result = np.concatenate((derivative.ravel(), layer_derivative))
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("The coupled derivative must remain finite.")
    return result


def residual_scales(
    influent: ArrayLike,
    reactor_outlet: ArrayLike,
    clarifier: ClarifierParameters = CLARIFIER,
) -> FloatArray:
    """Fixed component references and current clarifier feed-TSS reference."""

    _state(influent)
    outlet = _state(reactor_outlet)
    component = np.tile(np.maximum(1.0, INFLUENT_UPPER), N_STAGES)
    layer = np.full(clarifier.layer_count, max(1.0, float(TSS_VECTOR @ outlet)))
    return np.concatenate((component, layer))


def scaled_residual(
    state: ArrayLike,
    operating: OperatingPoint | ArticleOperatingPoint,
    influent: ArrayLike,
    clarifier: ClarifierParameters = CLARIFIER,
) -> FloatArray:
    reactors, _ = unpack_state(state, clarifier)
    return coupled_rhs(state, operating, influent, clarifier) / residual_scales(
        influent, reactors[-1], clarifier
    )


def initial_state(
    influent: ArrayLike,
    start: int = 1,
    clarifier: ClarifierParameters = CLARIFIER,
) -> FloatArray:
    """Construct either of the two fully specified manuscript initial states."""

    x = _state(influent)
    if np.min(x) < 0.0:
        raise ValueError("Influent must be non-negative.")
    if start == 1:
        reactors = np.tile(x, (N_STAGES, 1))
        layers = np.full(clarifier.layer_count, float(TSS_VECTOR @ x))
    elif start == 2:
        factors = np.asarray([1.5, 2.0, 2.5, 3.0, 3.5])
        reactors = np.tile(x, (N_STAGES, 1))
        reactors[:, PARTICULATE] *= factors[:, None]
        feed_tss = float(TSS_VECTOR @ reactors[-1])
        declared = np.asarray([.002, .005, .01, .03, .10, .50, 1.25, 2.25, 3.25, 4.00])
        if clarifier.layer_count == declared.size:
            profile = declared
        else:
            # Preserve the declared top and bottom factors while sampling the
            # same depthwise log-profile for a dimension-parametric test case.
            depth = np.linspace(0.0, 1.0, clarifier.layer_count)
            profile = np.exp(np.interp(depth, np.linspace(0.0, 1.0, declared.size), np.log(declared)))
        layers = feed_tss * profile
    else:
        raise ValueError("start must equal 1 or 2.")
    return np.concatenate((reactors.ravel(), layers))


def jacobian_sparsity(clarifier: ClarifierParameters = CLARIFIER):
    """Conservative structural sparsity for finite-difference steady solves."""

    from scipy.sparse import lil_matrix

    size = state_size(clarifier)
    layer_count = clarifier.layer_count
    pattern = lil_matrix((size, size), dtype=int)
    # CSTR 1 sees itself and the algebraically recycled CSTR-5/bottom-layer state.
    pattern[0:20, 0:20] = 1
    pattern[0:20, 80:100] = 1
    pattern[0:20, size - 1] = 1
    # Each later CSTR sees its immediate upstream state and its own kinetic state.
    for stage in range(1, N_STAGES):
        rows = slice(stage * 20, (stage + 1) * 20)
        pattern[rows, (stage - 1) * 20 : (stage + 1) * 20] = 1
    # All layer rates share CSTR-5 feed TSS; each flux is otherwise local.
    for layer in range(layer_count):
        row = 100 + layer
        pattern[row, 80:100] = 1
        for neighbor in range(max(0, layer - 1), min(layer_count, layer + 2)):
            pattern[row, 100 + neighbor] = 1
    return pattern.tocsr()


def _scaled_algebraic_residual(terms: Iterable[FloatArray]) -> float:
    arrays = [np.asarray(term, dtype=float) for term in terms]
    numerator = np.abs(np.sum(arrays, axis=0))
    denominator = np.maximum(1.0, np.sum(np.abs(arrays), axis=0))
    return float(np.max(numerator / denominator))


def _termwise_balance_residual(terms: Iterable[ArrayLike]) -> FloatArray:
    """Scale a heterogeneous balance by its largest separately evaluated term.

    This is the acceptance convention in the v3 supplement.  It intentionally
    differs from a residual norm divided by a state scale and from scaling by
    the sum of absolute terms.
    """

    arrays = [np.asarray(term, dtype=float) for term in terms]
    if not arrays:
        return np.empty(0, dtype=float)
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("Every term in one physical balance must have the same shape.")
    numerator = np.abs(np.sum(arrays, axis=0))
    denominator = np.maximum(1.0, np.maximum.reduce([np.abs(array) for array in arrays]))
    return np.asarray(numerator / denominator, dtype=float)


def generation_scale(
    influent: ArrayLike,
    reactor_outlet: ArrayLike,
    clarifier: ClarifierParameters = CLARIFIER,
) -> FloatArray:
    """Return the manuscript ``D_y,gen`` diagonal for one fixed input."""

    _state(influent)
    outlet = _state(reactor_outlet)
    feed_tss = max(1.0, float(TSS_VECTOR @ outlet))
    return np.concatenate((
        np.tile(np.maximum(1.0, INFLUENT_UPPER), N_STAGES),
        np.full(clarifier.layer_count, feed_tss, dtype=float),
    ))


def mechanistic_balance_audit(
    state: ArrayLike,
    operating: OperatingPoint | ArticleOperatingPoint,
    influent: ArrayLike,
    clarifier: ClarifierParameters = CLARIFIER,
    *,
    balance_tolerance: float = 1.0e-8,
    state_tolerance: float = 1.0e-10,
    rate_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Audit every v3 mechanistic balance, domain guard, and sign condition."""

    reactors, layers = unpack_state(state, clarifier)
    x = _state(influent)
    c_e, c_u = reconstruct_clarifier(reactors[-1], layers)
    mixer = mixer_state(x, reactors[-1], c_u, operating)
    q_p = operating.q_process
    q_c = operating.q_clarifier
    q_e = operating.q_effluent
    q_u = operating.q_underflow

    mixer_rows = _termwise_balance_residual((
        q_p * mixer, -x, -operating.internal_recycle * reactors[-1],
        -operating.return_sludge * c_u,
    ))

    reactor_rows: list[float] = []
    rates_by_stage: list[FloatArray] = []
    upstream = mixer
    for stage in range(N_STAGES):
        rates = process_rates(reactors[stage])
        rates_by_stage.append(rates)
        process_terms = rates[:, None] * STOICHIOMETRIC_MATRIX
        oxygen = np.zeros(N_COMPONENTS, dtype=float)
        oxygen[COMPONENT_INDEX["S_O"]] = oxygen_transfer(
            reactors[stage], stage, _stage_aeration(operating, stage)
        )
        for component in range(N_COMPONENTS):
            terms = [
                np.asarray(operating.stage_dilution_rate * upstream[component]),
                np.asarray(-operating.stage_dilution_rate * reactors[stage, component]),
                *(np.asarray(value) for value in process_terms[:, component]),
                np.asarray(oxygen[component]),
            ]
            reactor_rows.append(float(_termwise_balance_residual(terms)))
        upstream = reactors[stage]

    feed_tss = float(TSS_VECTOR @ reactors[-1])
    flux = clarifier_fluxes(layers, feed_tss, operating, clarifier)
    incoming = clarifier.area * flux[:-1] / clarifier.layer_volume
    outgoing = -clarifier.area * flux[1:] / clarifier.layer_volume
    feed = np.zeros(clarifier.layer_count, dtype=float)
    feed[clarifier.feed_layer] = (
        clarifier.fresh_flow * q_c * feed_tss / clarifier.layer_volume
    )
    layer_rows = _termwise_balance_residual((incoming, outgoing, feed))

    clarifier_rows = _termwise_balance_residual((
        q_c * reactors[-1], -q_e * c_e, -q_u * c_u,
    ))
    soluble_rows = np.concatenate((
        _termwise_balance_residual((c_e[SOLUBLE], -reactors[-1, SOLUBLE])),
        _termwise_balance_residual((c_u[SOLUBLE], -reactors[-1, SOLUBLE])),
    ))
    endpoint_rows = np.asarray([
        float(_termwise_balance_residual((
            np.asarray(q_e * layers[0]), np.asarray(-q_e * (TSS_VECTOR @ c_e)),
        ))),
        float(_termwise_balance_residual((
            np.asarray(q_u * layers[-1]), np.asarray(-q_u * (TSS_VECTOR @ c_u)),
        ))),
    ])
    external_invariant_rows = _termwise_balance_residual((
        INVARIANT_MATRIX @ x,
        -q_e * (INVARIANT_MATRIX @ c_e),
        -operating.waste_sludge * (INVARIANT_MATRIX @ c_u),
    ))

    families: dict[str, FloatArray] = {
        "mixer_component": mixer_rows,
        "reactor_component": np.asarray(reactor_rows, dtype=float),
        "clarifier_component": clarifier_rows,
        "soluble_outlet_identity": soluble_rows,
        "tss_endpoint_identity": endpoint_rows,
        "external_invariant": external_invariant_rows,
        "clarifier_layer": layer_rows,
    }
    all_balances = np.concatenate(tuple(families.values()))
    family_maxima = {
        name: (0.0 if values.size == 0 else float(np.max(values)))
        for name, values in families.items()
    }
    family_counts = {
        name: int(np.count_nonzero(values > balance_tolerance))
        for name, values in families.items()
    }
    rates_flat = np.concatenate(rates_by_stage)
    scaled_state_negativity = np.maximum(-np.asarray(state, dtype=float), 0.0)
    scaled_rate_negativity = np.maximum(-rates_flat, 0.0)
    layer_envelope_values = np.concatenate((
        layers[0] - layers[1:-1], layers[1:-1] - layers[-1]
    )) if clarifier.layer_count > 2 else np.empty(0)
    external_solids_loss = float(
        q_e * (TSS_VECTOR @ c_e) + operating.waste_sludge * (TSS_VECTOR @ c_u)
    )
    maximum_balance = float(np.max(all_balances)) if all_balances.size else 0.0
    maximum_state_negativity = float(np.max(scaled_state_negativity))
    maximum_rate_negativity = float(np.max(scaled_rate_negativity))
    maximum_envelope = (
        float(np.max(np.maximum(layer_envelope_values, 0.0)))
        if layer_envelope_values.size else 0.0
    )
    passed = bool(
        maximum_balance <= balance_tolerance
        and maximum_state_negativity <= state_tolerance
        and maximum_rate_negativity <= rate_tolerance
        and maximum_envelope <= state_tolerance
        and feed_tss >= 1.0
        and external_solids_loss >= 1.0
    )
    return {
        "passed": passed,
        "maximum_balance_residual": maximum_balance,
        "balance_violation_count": int(np.count_nonzero(all_balances > balance_tolerance)),
        "balance_family_maxima": family_maxima,
        "balance_family_violation_counts": family_counts,
        "state_negativity_max": maximum_state_negativity,
        "state_negativity_count": int(np.count_nonzero(scaled_state_negativity > state_tolerance)),
        "rate_negativity_max": maximum_rate_negativity,
        "rate_negativity_count": int(np.count_nonzero(scaled_rate_negativity > rate_tolerance)),
        "layer_envelope_violation_max": maximum_envelope,
        "feed_tss_g_m3": feed_tss,
        "external_solids_loss_g_m3": external_solids_loss,
    }


def stability_audit(
    state: ArrayLike,
    operating: OperatingPoint | ArticleOperatingPoint,
    influent: ArrayLike,
    clarifier: ClarifierParameters = CLARIFIER,
    *,
    stability_margin: float = 1.0e-8,
    agreement_tolerance: float = 1.0e-6,
) -> dict[str, float | bool]:
    """Run the v3 two-step, domain-respecting scaled Jacobian audit."""

    reactors, _ = unpack_state(state, clarifier)
    y = _state(state, state_size(clarifier))
    scale = generation_scale(influent, reactors[-1], clarifier)
    z = y / scale

    def scaled_rhs(value: FloatArray) -> FloatArray:
        return coupled_rhs(scale * value, operating, influent, clarifier) / scale

    base = scaled_rhs(z)
    rightmost: list[float] = []
    base_step = float(np.finfo(float).eps ** (1.0 / 3.0))
    for step in (base_step, 2.0 * base_step):
        jacobian = np.empty((y.size, y.size), dtype=float)
        for column in range(y.size):
            direction = np.zeros(y.size, dtype=float)
            direction[column] = step
            if z[column] > 2.0 * step:
                jacobian[:, column] = (
                    scaled_rhs(z + direction) - scaled_rhs(z - direction)
                ) / (2.0 * step)
            else:
                jacobian[:, column] = (
                    -3.0 * base + 4.0 * scaled_rhs(z + direction)
                    - scaled_rhs(z + 2.0 * direction)
                ) / (2.0 * step)
        eigenvalues = np.linalg.eigvals(jacobian)
        rightmost.append(float(np.max(np.real(eigenvalues))))
    agreement = abs(rightmost[0] - rightmost[1])
    largest = max(rightmost)
    return {
        "passed": bool(
            np.isfinite(largest)
            and agreement <= agreement_tolerance
            and largest <= -stability_margin
        ),
        "rightmost_eigenvalue_step_1": rightmost[0],
        "rightmost_eigenvalue_step_2": rightmost[1],
        "rightmost_eigenvalue_agreement": float(agreement),
        "largest_real_eigenvalue": float(largest),
    }


def branch_classification(
    state: ArrayLike,
    clarifier: ClarifierParameters = CLARIFIER,
) -> dict[str, tuple[bool, ...]]:
    """Return every nonsmooth branch used by the v3 equivalence contract."""

    reactors, layers = unpack_state(state, clarifier)
    feed_tss = float(TSS_VECTOR @ reactors[-1])
    delta = layers - clarifier.nonsettleable_fraction * feed_tss
    with np.errstate(over="ignore", invalid="ignore"):
        raw_velocity = clarifier.theoretical_settling_velocity * (
            np.exp(-clarifier.hindered_coefficient * delta)
            - np.exp(-clarifier.low_concentration_coefficient * delta)
        )
    velocity = np.maximum(0.0, np.minimum(clarifier.maximum_settling_velocity, raw_velocity))
    gravity = layers * velocity
    return {
        "receiver_limited": tuple(bool(value > clarifier.flux_threshold) for value in layers[1:]),
        "settling_floor": tuple(bool(value <= 0.0) for value in raw_velocity),
        "settling_cap": tuple(bool(value >= clarifier.maximum_settling_velocity) for value in raw_velocity),
        "flux_minimum_receiver": tuple(
            bool(layers[index + 1] > clarifier.flux_threshold and gravity[index + 1] < gravity[index])
            for index in range(clarifier.layer_count - 1)
        ),
        "storage_capacity_positive": tuple(
            bool(PARAMETERS["K_max"] * reactor[COMPONENT_INDEX["X_PAO"]]
                 - reactor[COMPONENT_INDEX["X_PP"]] > 0.0)
            for reactor in reactors
        ),
    }


def diagnostics(
    state: ArrayLike,
    operating: OperatingPoint | ArticleOperatingPoint,
    influent: ArrayLike,
    *,
    residual_tolerance: float = 1e-8,
    check_stability: bool = True,
    clarifier: ClarifierParameters = CLARIFIER,
    strict_v3: bool = False,
) -> dict[str, float | bool]:
    """Replay the local and external physical acceptance checks."""

    reactors, layers = unpack_state(state, clarifier)
    x = _state(influent)
    ce, cu = reconstruct_clarifier(reactors[-1], layers)
    rhs = coupled_rhs(state, operating, x, clarifier)
    scaled_inf = float(np.linalg.norm(
        rhs / residual_scales(x, reactors[-1], clarifier), ord=np.inf
    ))
    clarifier_error = _scaled_algebraic_residual(
        (operating.q_clarifier * reactors[-1], -operating.q_effluent * ce, -operating.q_underflow * cu)
    )
    stage_volume_over_q0_days = operating.hrt_hours / (24.0 * N_STAGES)
    total_source = np.zeros(N_COMPONENTS)
    finite_rates = True
    for stage in range(N_STAGES):
        rates = process_rates(reactors[stage])
        source = STOICHIOMETRIC_MATRIX.T @ rates
        source[0] += oxygen_transfer(
            reactors[stage], stage, _stage_aeration(operating, stage)
        )
        total_source += stage_volume_over_q0_days * source
        finite_rates &= bool(np.all(np.isfinite(rates)) and np.min(rates) >= -1e-12)
    boundary_error = _scaled_algebraic_residual(
        (x, total_source, -operating.q_effluent * ce, -operating.waste_sludge * cu)
    )
    feed_tss = float(TSS_VECTOR @ reactors[-1])
    tss_closure = abs(
        operating.q_clarifier * feed_tss
        - operating.q_effluent * layers[0]
        - operating.q_underflow * layers[-1]
    ) / max(
        1.0,
        operating.q_clarifier * feed_tss
        + operating.q_effluent * layers[0]
        + operating.q_underflow * layers[-1],
    )
    lower_recovery = operating.q_underflow / operating.q_clarifier
    # Ratios are not numerically identifiable at a vanishing solids feed.  The
    # tolerance is in the same scaled-concentration units as the acceptance
    # residual and prevents bound-interior perturbations of an exact zero from
    # being misclassified as a physical recovery failure.
    if feed_tss > residual_tolerance:
        eta = operating.q_underflow * layers[-1] / (operating.q_clarifier * feed_tss)
    else:
        eta = np.nan
    soluble_pass = max(float(np.max(np.abs(ce[SOLUBLE] - reactors[-1, SOLUBLE]))),
                        float(np.max(np.abs(cu[SOLUBLE] - reactors[-1, SOLUBLE]))))
    layer_envelope = bool(np.all(layers[1:-1] >= layers[0] - 1e-10)
                          and np.all(layers[1:-1] <= layers[-1] + 1e-10))
    recovery_ok = bool(np.isnan(eta) or lower_recovery - 1e-10 <= eta <= 1.0 + 1e-10)
    v3_audit = mechanistic_balance_audit(
        state, operating, x, clarifier, balance_tolerance=residual_tolerance,
    )
    physical_pass = bool(
        np.min(state) >= -1e-10
        and scaled_inf <= residual_tolerance
        and clarifier_error <= residual_tolerance
        and boundary_error <= residual_tolerance
        and tss_closure <= residual_tolerance
        and soluble_pass <= 1e-10
        and layer_envelope
        and recovery_ok
        and finite_rates
        and (not strict_v3 or bool(v3_audit["passed"]))
    )
    if check_stability and physical_pass:
        if strict_v3:
            stability = stability_audit(state, operating, x, clarifier)
            stable = bool(stability["passed"])
            largest_real_eigenvalue = float(stability["largest_real_eigenvalue"])
        else:
            stable, largest_real_eigenvalue = stability_screen(
                state, operating, x, clarifier=clarifier
            )
            stability = {
                "rightmost_eigenvalue_step_1": float(largest_real_eigenvalue),
                "rightmost_eigenvalue_step_2": float("nan"),
                "rightmost_eigenvalue_agreement": float("nan"),
            }
    elif check_stability:
        stable, largest_real_eigenvalue = False, np.nan
        stability = {
            "rightmost_eigenvalue_step_1": float("nan"),
            "rightmost_eigenvalue_step_2": float("nan"),
            "rightmost_eigenvalue_agreement": float("nan"),
        }
    else:
        stable, largest_real_eigenvalue = True, np.nan
        stability = {
            "rightmost_eigenvalue_step_1": float("nan"),
            "rightmost_eigenvalue_step_2": float("nan"),
            "rightmost_eigenvalue_agreement": float("nan"),
        }
    passed = bool(physical_pass and stable)
    return {
        "passed": passed,
        "scaled_residual_inf": scaled_inf,
        "clarifier_component_residual": clarifier_error,
        "plant_boundary_residual": boundary_error,
        "clarifier_tss_residual": float(tss_closure),
        "minimum_state": float(np.min(state)),
        "underflow_recovery": float(eta),
        "minimum_recovery": float(lower_recovery),
        "soluble_passthrough_error": soluble_pass,
        "layer_envelope": layer_envelope,
        "finite_nonnegative_rates": finite_rates,
        "locally_stable": stable,
        "largest_real_eigenvalue": float(largest_real_eigenvalue),
        "stability_eigenvalue_step_1": float(stability["rightmost_eigenvalue_step_1"]),
        "stability_eigenvalue_step_2": float(stability["rightmost_eigenvalue_step_2"]),
        "stability_eigenvalue_agreement": float(stability["rightmost_eigenvalue_agreement"]),
        "v3_balance_residual": float(v3_audit["maximum_balance_residual"]),
        "v3_balance_violation_count": int(v3_audit["balance_violation_count"]),
        "v3_state_negativity_max": float(v3_audit["state_negativity_max"]),
        "v3_state_negativity_count": int(v3_audit["state_negativity_count"]),
        "v3_rate_negativity_max": float(v3_audit["rate_negativity_max"]),
        "v3_rate_negativity_count": int(v3_audit["rate_negativity_count"]),
        "feed_tss_g_m3": float(v3_audit["feed_tss_g_m3"]),
        "external_solids_loss_g_m3": float(v3_audit["external_solids_loss_g_m3"]),
    }


@dataclass
class SteadyStateResult:
    state: FloatArray
    operating: OperatingPoint
    influent: FloatArray
    accepted: bool
    start: int
    nfev: int
    cost: float
    message: str
    diagnostics: dict[str, float | bool] = field(default_factory=dict)
    route: str = "unknown"
    integration_time_days: float = 0.0
    integration_steps: int = 0
    clarifier: ClarifierParameters = CLARIFIER

    @property
    def reactors(self) -> FloatArray:
        return unpack_state(self.state, self.clarifier)[0]

    @property
    def layers(self) -> FloatArray:
        return unpack_state(self.state, self.clarifier)[1]

    @property
    def effluent(self) -> FloatArray:
        return reconstruct_clarifier(self.reactors[-1], self.layers)[0]

    @property
    def underflow(self) -> FloatArray:
        return reconstruct_clarifier(self.reactors[-1], self.layers)[1]

    @property
    def mixer(self) -> FloatArray:
        return mixer_state(self.influent, self.reactors[-1], self.underflow, self.operating)

    @property
    def target(self) -> FloatArray:
        return assemble_target(self.state, self.operating, self.influent, self.clarifier)


def reduced_jacobian(
    state: ArrayLike,
    operating: OperatingPoint,
    influent: ArrayLike,
    clarifier: ClarifierParameters = CLARIFIER,
) -> FloatArray:
    """Finite-difference Jacobian after all algebraic loop variables are eliminated."""

    size = state_size(clarifier)
    y = _state(state, size)
    x = _state(influent)
    base = coupled_rhs(y, operating, x, clarifier)
    jacobian = np.empty((size, size), dtype=float)
    relative_step = np.sqrt(np.finfo(float).eps)
    for column in range(size):
        step = relative_step * max(1.0, abs(y[column]))
        forward = y.copy()
        forward[column] += step
        if y[column] - step >= 0.0:
            backward = y.copy()
            backward[column] -= step
            jacobian[:, column] = (
                coupled_rhs(forward, operating, x, clarifier)
                - coupled_rhs(backward, operating, x, clarifier)
            ) / (2.0 * step)
        else:
            jacobian[:, column] = (
                coupled_rhs(forward, operating, x, clarifier) - base
            ) / step
    return jacobian


def stability_screen(
    state: ArrayLike,
    operating: OperatingPoint,
    influent: ArrayLike,
    *,
    tolerance: float = 1e-8,
    clarifier: ClarifierParameters = CLARIFIER,
) -> tuple[bool, float]:
    """Return local stability and the largest real reduced-Jacobian eigenvalue."""

    eigenvalues = np.linalg.eigvals(
        reduced_jacobian(state, operating, influent, clarifier)
    )
    largest_real = float(np.max(np.real(eigenvalues)))
    return bool(np.isfinite(largest_real) and largest_real <= tolerance), largest_real


def _integrate_to_steady_state(
    operating: OperatingPoint,
    influent: FloatArray,
    y0: FloatArray,
    *,
    horizon_days: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    logarithmic: bool,
    steady_tolerance: float,
    clarifier: ClarifierParameters = CLARIFIER,
):
    """Scaled BDF relaxation; the log form guarantees positive trial states."""

    from scipy.integrate import solve_ivp

    reactors, _ = unpack_state(y0, clarifier)
    scale = residual_scales(influent, reactors[-1], clarifier)
    sparsity = jacobian_sparsity(clarifier)
    maximum_step = horizon_days / 100.0
    if logarithmic:
        if np.any(y0 <= 0.0):
            raise ValueError("Logarithmic relaxation requires a strictly positive initial state.")
        transformed0 = np.log(y0 / scale)

        def transformed_rhs(_time: float, transformed: FloatArray) -> FloatArray:
            scaled_state = np.exp(transformed)
            physical_state = scale * scaled_state
            return coupled_rhs(physical_state, operating, influent, clarifier) / (scale * scaled_state)

        def steady_event(_time: float, transformed: FloatArray) -> float:
            physical_state = scale * np.exp(transformed)
            return float(np.linalg.norm(
                scaled_residual(physical_state, operating, influent, clarifier), ord=np.inf
            ) - steady_tolerance)

        steady_event.terminal = True
        steady_event.direction = -1

        integration = solve_ivp(
            transformed_rhs,
            (0.0, horizon_days),
            transformed0,
            method="BDF",
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=maximum_step,
            jac_sparsity=sparsity,
            events=steady_event,
        )
        endpoint = scale * np.exp(integration.y[:, -1])
    else:
        def steady_event(_time: float, scaled: FloatArray) -> float:
            return float(np.linalg.norm(
                scaled_residual(scale * scaled, operating, influent, clarifier), ord=np.inf
            ) - steady_tolerance)

        steady_event.terminal = True
        steady_event.direction = -1
        integration = solve_ivp(
            lambda _time, scaled: coupled_rhs(
                scale * scaled, operating, influent, clarifier
            ) / scale,
            (0.0, horizon_days),
            y0 / scale,
            method="BDF",
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=maximum_step,
            jac_sparsity=sparsity,
            events=steady_event,
        )
        # A successful status alone is insufficient: every stored BDF state is
        # audited so the positivity guarantee covers the numerical trajectory,
        # not only the final point.  Any boundary touch is retried in log space.
        if np.any(integration.y <= 0.0):
            raise ValueError("Scaled BDF stored a non-positive trial state.")
        endpoint = scale * integration.y[:, -1]
    if not integration.success:
        raise RuntimeError(str(integration.message))
    return endpoint, integration


def solve_steady_state(
    operating: OperatingPoint | ArticleOperatingPoint,
    influent: ArrayLike,
    *,
    max_nfev: int = 5_000,
    tolerance: float = 1e-9,
    acceptance_tolerance: float = 1e-8,
    starts: tuple[int, ...] = (1, 2),
    minimum_relaxation_days: float = 400.0,
    solids_turnovers: float = 50.0,
    integration_rtol: float = 1e-7,
    integration_atol: float = 1e-9,
    clarifier: ClarifierParameters = CLARIFIER,
    logarithmic_only: bool = False,
    strict_v3: bool = False,
) -> SteadyStateResult:
    """Relax the positive dynamics first, then polish only if acceptance requires it.

    The horizon follows the slow external solids-removal time scale rather than a
    fixed calendar duration.  A direct scaled-state BDF attempt is fastest; if an
    internal Newton iterate crosses the non-negative boundary, a log-state BDF
    retry guarantees that every evaluated physical state is positive.
    """

    from scipy.optimize import least_squares

    x = _state(influent)
    if np.min(x) < 0.0:
        raise ValueError("Influent must be non-negative.")
    candidates: list[SteadyStateResult] = []
    horizon = max(
        float(minimum_relaxation_days),
        float(solids_turnovers) / max(operating.waste_sludge, 1e-3),
    )
    for start in starts:
        y0 = initial_state(x, start, clarifier)
        initial_replay = diagnostics(
            y0, operating, x, residual_tolerance=acceptance_tolerance,
            clarifier=clarifier, strict_v3=strict_v3,
        )
        if bool(initial_replay["passed"]):
            return SteadyStateResult(
                state=y0.copy(), operating=operating, influent=x.copy(),
                accepted=True, start=start, nfev=1, cost=0.0,
                message="The prescribed initial state already satisfies the acceptance contract.",
                diagnostics=initial_replay, route="initial-state", clarifier=clarifier,
            )
        endpoint = None
        integration = None
        route = "scaled-bdf"
        integration_errors: list[str] = []
        logarithmic_routes = (True,) if logarithmic_only else (False, True)
        for logarithmic in logarithmic_routes:
            try:
                integration_start = np.maximum(y0, 1.0e-8) if logarithmic else y0
                endpoint, integration = _integrate_to_steady_state(
                    operating,
                    x,
                    integration_start,
                    horizon_days=horizon,
                    relative_tolerance=integration_rtol,
                    absolute_tolerance=integration_atol,
                    logarithmic=logarithmic,
                    steady_tolerance=acceptance_tolerance / 10.0,
                    clarifier=clarifier,
                )
                route = "log-bdf" if logarithmic else "scaled-bdf"
                break
            except (FloatingPointError, RuntimeError, ValueError) as error:
                integration_errors.append(f"{'log' if logarithmic else 'scaled'} BDF: {error}")
        if endpoint is None or integration is None:
            # Preserve a deterministic last resort for an exact-boundary start
            # that cannot be transformed to logarithmic coordinates.
            endpoint = y0
            route = "bounded-least-squares"
        else:
            replay = diagnostics(
                endpoint, operating, x,
                residual_tolerance=acceptance_tolerance,
                clarifier=clarifier, strict_v3=strict_v3,
            )
            if bool(replay["passed"]):
                return SteadyStateResult(
                    state=endpoint.copy(), operating=operating, influent=x.copy(),
                    accepted=True, start=start, nfev=int(integration.nfev), cost=0.0,
                    message=str(integration.message), diagnostics=replay, route=route,
                    integration_time_days=float(integration.t[-1]),
                    integration_steps=int(integration.t.size),
                    clarifier=clarifier,
                )

        # The bounded calculation is now a local polish/fallback, never the
        # expensive first route from a hydraulically inconsistent raw start.
        result = least_squares(
            lambda y: scaled_residual(y, operating, x, clarifier), endpoint,
            bounds=(np.zeros(state_size(clarifier)), np.full(state_size(clarifier), np.inf)),
            jac_sparsity=jacobian_sparsity(clarifier), x_scale=np.maximum(1.0, endpoint),
            xtol=tolerance, ftol=tolerance, gtol=tolerance,
            max_nfev=max_nfev, tr_solver="lsmr",
        )
        replay = diagnostics(
            result.x, operating, x,
            residual_tolerance=acceptance_tolerance,
            clarifier=clarifier, strict_v3=strict_v3,
        )
        candidate = SteadyStateResult(
            state=result.x.copy(), operating=operating, influent=x.copy(),
            accepted=bool(replay["passed"]), start=start, nfev=int(result.nfev),
            cost=float(result.cost), message="; ".join(integration_errors + [str(result.message)]),
            diagnostics=replay, route=f"{route}+bounded-polish",
            integration_time_days=(0.0 if integration is None else float(integration.t[-1])),
            integration_steps=(0 if integration is None else int(integration.t.size)),
            clarifier=clarifier,
        )
        candidates.append(candidate)
        if candidate.accepted:
            return candidate
    return min(candidates, key=lambda item: float(item.diagnostics["scaled_residual_inf"]))


def assemble_target(
    state: ArrayLike,
    operating: OperatingPoint | ArticleOperatingPoint,
    influent: ArrayLike,
    clarifier: ClarifierParameters = CLARIFIER,
) -> FloatArray:
    """Return ``(m,c1,...,c5,g_E,g_U,s1,...,sL)``."""

    reactors, layers = unpack_state(state, clarifier)
    ce, cu = reconstruct_clarifier(reactors[-1], layers)
    mixer = mixer_state(influent, reactors[-1], cu, operating)
    ge, gu = operating.q_effluent * ce, operating.q_underflow * cu
    target = np.concatenate((mixer, reactors.ravel(), ge, gu, layers))
    if target.shape != (target_size(clarifier),):
        raise AssertionError("Mechanistic target dimension changed unexpectedly.")
    return target


def zero_state_solution(operating: OperatingPoint | None = None) -> SteadyStateResult:
    """Cheap exact fixture used to exercise the full coupled steady-solver route."""

    # With no influent and no aeration the origin is the exact physical steady
    # state; this isolates the nonlinear-solver plumbing from biological load.
    point = operating or OperatingPoint(18.0, 0.0, 2.0, .75, .02)
    return solve_steady_state(point, np.zeros(N_COMPONENTS), max_nfev=20, starts=(1,))


__all__ = [
    "CLARIFIER", "COMPONENTS", "COMPONENT_INDEX", "COMPOSITE_MATRIX",
    "INFLUENT_LOWER", "INFLUENT_UPPER", "INVARIANT_MATRIX", "NOMINAL_INFLUENT",
    "N_COMPONENTS", "N_LAYERS", "N_PROCESSES", "N_STAGES", "PARAMETERS",
    "PARTICULATE", "SOLUBLE", "STATE_SIZE", "STOICHIOMETRIC_MATRIX",
    "TARGET_SIZE", "TSS_VECTOR", "ArticleOperatingPoint", "ClarifierParameters", "OperatingPoint",
    "SteadyStateResult", "assemble_target", "audit_mechanistic_matrices",
    "branch_classification", "build_invariant_matrix", "build_stoichiometric_matrix", "clarifier_fluxes",
    "clarifier_rhs", "coupled_rhs", "diagnostics", "generation_scale", "initial_state",
    "jacobian_sparsity", "mixer_state", "oxygen_transfer", "process_rates",
    "reaction_source", "reconstruct_clarifier", "reduced_jacobian", "residual_scales",
    "mechanistic_balance_audit", "scaled_residual", "settling_velocity", "solve_steady_state", "state_size",
    "target_size", "unpack_state",
    "stability_audit", "stability_screen", "zero_state_solution",
]
