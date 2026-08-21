"""Deterministic Latin-hypercube designs for the closed-loop study.

The generator in this module is a direct implementation of the random-design
contract stated in ``article/wip_v2/manuscript.tex``.  It intentionally does
not use NumPy's or SciPy's random-number APIs: the SplitMix64 state transition,
unbiased Fisher--Yates shuffles, jitter conversion, dimension order, and draw
consumption are all explicit and therefore reproducible across platforms.

For the mechanistic design, dimensions are generated in the fixed order
``(H, a, r_I, r_R, w, x_1, ..., x_20)``.  The robustness design restarts an
independent SplitMix64 stream and contains the 20 influent dimensions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .model import COMPONENTS


FloatArray = NDArray[np.float64]

UINT64_MODULUS = 1 << 64
UINT64_MASK = UINT64_MODULUS - 1
SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15
SPLITMIX64_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
SPLITMIX64_MULTIPLIER_2 = 0x94D049BB133111EB
FLOAT53_DENOMINATOR = 1 << 53

DECISION_COLUMNS: tuple[str, ...] = ("H", "a", "r_I", "r_R", "w")
INFLUENT_COLUMNS: tuple[str, ...] = tuple(COMPONENTS)
TRAINING_COLUMNS: tuple[str, ...] = DECISION_COLUMNS + INFLUENT_COLUMNS
ROBUSTNESS_COLUMNS: tuple[str, ...] = INFLUENT_COLUMNS

DECISION_BOUNDS: dict[str, tuple[float, float]] = {
    "H": (6.0, 36.0),
    "a": (0.0, 1.0),
    "r_I": (0.0, 4.0),
    "r_R": (0.25, 1.25),
    "w": (0.001, 0.05),
}

INFLUENT_BOUNDS: dict[str, tuple[float, float]] = {
    "S_O": (0.0, 0.5),
    "S_F": (20.0, 180.0),
    "S_A": (5.0, 80.0),
    "S_NH4": (12.0, 55.0),
    "S_NO2": (0.0, 3.0),
    "S_NO3": (0.0, 8.0),
    "S_N2": (0.0, 2.0),
    "S_PO4": (2.0, 18.0),
    "S_I": (10.0, 90.0),
    "S_ALK": (1.6, 5.2),
    "X_I": (20.0, 120.0),
    "X_S": (60.0, 280.0),
    "X_H": (15.0, 100.0),
    "X_PAO": (5.0, 60.0),
    "X_PP": (2.0, 20.0),
    "X_PHA": (1.0, 30.0),
    "X_AOB": (0.5, 8.0),
    "X_NOB": (0.5, 8.0),
    "X_MeP": (0.0, 12.0),
    "X_MeOH": (0.0, 12.0),
}

TRAINING_BOUNDS: dict[str, tuple[float, float]] = {
    **DECISION_BOUNDS,
    **INFLUENT_BOUNDS,
}


def _unsigned_64(value: int, *, label: str) -> int:
    """Validate rather than silently wrap a declared unsigned 64-bit value."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    converted = int(value)
    if converted < 0 or converted > UINT64_MASK:
        raise ValueError(f"{label} must lie in [0, 2^64 - 1].")
    return converted


class SplitMix64:
    """Minimal SplitMix64 stream with observable state and draw count."""

    def __init__(self, seed: int) -> None:
        self._state = _unsigned_64(seed, label="seed")
        self._draw_count = 0

    @property
    def state(self) -> int:
        """Current unsigned 64-bit state, after all consumed words."""

        return self._state

    @property
    def draw_count(self) -> int:
        """Number of generated words, including rejection-sampling draws."""

        return self._draw_count

    def next_uint64(self) -> int:
        """Return the next word using arithmetic modulo ``2^64``."""

        self._state = (self._state + SPLITMIX64_INCREMENT) & UINT64_MASK
        value = self._state
        value = (
            (value ^ (value >> 30)) * SPLITMIX64_MULTIPLIER_1
        ) & UINT64_MASK
        value = (
            (value ^ (value >> 27)) * SPLITMIX64_MULTIPLIER_2
        ) & UINT64_MASK
        value ^= value >> 31
        self._draw_count += 1
        return value & UINT64_MASK

    def random_float53(self) -> float:
        """Return ``(U >> 11) / 2^53``, which lies in ``[0, 1)``."""

        return float(self.next_uint64() >> 11) / float(FLOAT53_DENOMINATOR)

    def randbelow(self, upper: int) -> int:
        """Draw uniformly from ``range(upper)`` without modulo bias."""

        if isinstance(upper, bool) or not isinstance(upper, (int, np.integer)):
            raise TypeError("upper must be an integer.")
        bound = int(upper)
        if bound <= 0 or bound > UINT64_MODULUS:
            raise ValueError("upper must lie in [1, 2^64].")
        limit = UINT64_MODULUS - (UINT64_MODULUS % bound)
        while True:
            word = self.next_uint64()
            if word < limit:
                return word % bound


@dataclass(frozen=True)
class LatinHypercubeDesign:
    """A labeled unit design and its affine physical realization."""

    columns: tuple[str, ...]
    unit: FloatArray
    physical: FloatArray
    lower: FloatArray
    upper: FloatArray
    seed: int
    final_state: int
    draw_count: int

    @property
    def n_points(self) -> int:
        return int(self.unit.shape[0])

    @property
    def n_dimensions(self) -> int:
        return int(self.unit.shape[1])

    def as_dict(self) -> dict[str, FloatArray]:
        """Return physical columns as independent one-dimensional arrays."""

        return {
            name: self.physical[:, index].copy()
            for index, name in enumerate(self.columns)
        }

    def to_frame(self):  # type annotation intentionally avoids a hard pandas import
        """Return a physical-design DataFrame when pandas is available."""

        import pandas as pd

        return pd.DataFrame(self.physical.copy(), columns=self.columns)


def _validate_size(n_points: int, n_dimensions: int) -> tuple[int, int]:
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise TypeError("n_points must be an integer.")
    if isinstance(n_dimensions, bool) or not isinstance(n_dimensions, (int, np.integer)):
        raise TypeError("n_dimensions must be an integer.")
    rows, dimensions = int(n_points), int(n_dimensions)
    if rows <= 0 or rows > FLOAT53_DENOMINATOR:
        raise ValueError("n_points must lie in [1, 2^53].")
    if dimensions <= 0:
        raise ValueError("n_dimensions must be positive.")
    return rows, dimensions


def unit_latin_hypercube(
    n_points: int,
    n_dimensions: int,
    *,
    seed: int,
) -> tuple[FloatArray, int, int]:
    """Generate the exact manuscript unit-box design.

    Returns ``(coordinates, final_state, draw_count)``.  A dimension's entire
    Fisher--Yates permutation is generated before its row jitters, and all
    dimensions are completed in increasing index order.
    """

    rows, dimensions = _validate_size(n_points, n_dimensions)
    stream = SplitMix64(seed)
    coordinates = np.empty((rows, dimensions), dtype=np.float64)

    for dimension in range(dimensions):
        permutation = list(range(rows))
        for index in range(rows - 1, 0, -1):
            swap_index = stream.randbelow(index + 1)
            permutation[index], permutation[swap_index] = (
                permutation[swap_index],
                permutation[index],
            )
        for row in range(rows):
            jitter = stream.random_float53()
            coordinates[row, dimension] = (permutation[row] + jitter) / rows

    validate_unit_latin_hypercube(coordinates)
    return coordinates, stream.state, stream.draw_count


def _ordered_bound_arrays(
    columns: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
) -> tuple[tuple[str, ...], FloatArray, FloatArray]:
    names = tuple(str(name) for name in columns)
    if not names or len(set(names)) != len(names):
        raise ValueError("columns must be nonempty and uniquely named.")
    if set(bounds) != set(names):
        missing = sorted(set(names).difference(bounds))
        extra = sorted(set(bounds).difference(names))
        raise ValueError(f"Bounds do not match columns; missing={missing}, extra={extra}.")

    lower = np.empty(len(names), dtype=np.float64)
    upper = np.empty(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        pair = np.asarray(bounds[name], dtype=np.float64)
        if pair.shape != (2,) or not np.all(np.isfinite(pair)):
            raise ValueError(f"Bounds for {name} must be two finite values.")
        if pair[1] <= pair[0]:
            raise ValueError(f"Upper bound for {name} must exceed its lower bound.")
        lower[index], upper[index] = pair
    return names, lower, upper


def affine_map(
    unit: FloatArray,
    columns: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Map labeled unit coordinates to their physical intervals."""

    names, lower, upper = _ordered_bound_arrays(columns, bounds)
    coordinates = np.asarray(unit, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != len(names):
        raise ValueError(f"unit must have shape (n, {len(names)}).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("unit contains non-finite coordinates.")
    if np.any(coordinates < 0.0) or np.any(coordinates >= 1.0):
        raise ValueError("unit coordinates must lie in [0, 1).")
    physical = lower + coordinates * (upper - lower)
    return physical, lower, upper


def generate_design(
    n_points: int,
    columns: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
    *,
    seed: int,
) -> LatinHypercubeDesign:
    """Generate, map, and independently validate a labeled LHS."""

    names, _, _ = _ordered_bound_arrays(columns, bounds)
    unit, final_state, draw_count = unit_latin_hypercube(
        n_points, len(names), seed=seed,
    )
    physical, lower, upper = affine_map(unit, names, bounds)
    design = LatinHypercubeDesign(
        columns=names,
        unit=unit,
        physical=physical,
        lower=lower,
        upper=upper,
        seed=_unsigned_64(seed, label="seed"),
        final_state=final_state,
        draw_count=draw_count,
    )
    validate_design(design)
    return design


def generate_training_design(
    n_points: int,
    *,
    seed: int = 42,
    decision_bounds: Mapping[str, Sequence[float]] = DECISION_BOUNDS,
    influent_bounds: Mapping[str, Sequence[float]] = INFLUENT_BOUNDS,
) -> LatinHypercubeDesign:
    """Generate the 25-dimensional mechanistic training design."""

    if set(decision_bounds) != set(DECISION_COLUMNS):
        raise ValueError("decision_bounds must define exactly H, a, r_I, r_R, and w.")
    if set(influent_bounds) != set(INFLUENT_COLUMNS):
        raise ValueError("influent_bounds must define exactly the 20 ASM components.")
    ordered_bounds = {
        **{name: decision_bounds[name] for name in DECISION_COLUMNS},
        **{name: influent_bounds[name] for name in INFLUENT_COLUMNS},
    }
    return generate_design(
        n_points, TRAINING_COLUMNS, ordered_bounds, seed=seed,
    )


def generate_robustness_design(
    n_points: int,
    *,
    seed: int = 314159,
    influent_bounds: Mapping[str, Sequence[float]] = INFLUENT_BOUNDS,
) -> LatinHypercubeDesign:
    """Restart the stream and generate a 20-dimensional influent design."""

    if set(influent_bounds) != set(INFLUENT_COLUMNS):
        raise ValueError("influent_bounds must define exactly the 20 ASM components.")
    ordered_bounds = {name: influent_bounds[name] for name in INFLUENT_COLUMNS}
    return generate_design(
        n_points, ROBUSTNESS_COLUMNS, ordered_bounds, seed=seed,
    )


def validate_unit_latin_hypercube(unit: FloatArray) -> None:
    """Require finite unit coordinates and one point in every stratum."""

    coordinates = np.asarray(unit, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] <= 0 or coordinates.shape[1] <= 0:
        raise ValueError("A unit LHS must be a nonempty two-dimensional array.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("A unit LHS must contain only finite values.")
    if np.any(coordinates < 0.0) or np.any(coordinates >= 1.0):
        raise ValueError("Unit LHS coordinates must lie in [0, 1).")

    rows = coordinates.shape[0]
    expected = np.arange(rows, dtype=np.int64)
    strata = np.floor(coordinates * rows).astype(np.int64)
    for dimension in range(coordinates.shape[1]):
        if not np.array_equal(np.sort(strata[:, dimension]), expected):
            raise ValueError(
                f"Dimension {dimension} does not contain exactly one point per stratum."
            )


def validate_design(design: LatinHypercubeDesign) -> None:
    """Replay the complete unit, labeling, affine-map, and bound contract."""

    validate_unit_latin_hypercube(design.unit)
    if design.physical.shape != design.unit.shape:
        raise ValueError("Physical and unit design shapes differ.")
    if design.unit.shape[1] != len(design.columns):
        raise ValueError("The number of labels does not match the design width.")
    if design.lower.shape != (len(design.columns),) or design.upper.shape != design.lower.shape:
        raise ValueError("Physical bound arrays do not match the design width.")
    if not np.all(np.isfinite(design.physical)):
        raise ValueError("The physical design contains non-finite values.")
    if np.any(design.physical < design.lower) or np.any(design.physical >= design.upper):
        raise ValueError("The physical design lies outside its half-open bounds.")
    replay = design.lower + design.unit * (design.upper - design.lower)
    if not np.array_equal(replay, design.physical):
        raise ValueError("The physical design does not replay from its unit coordinates.")
    expected_state = (
        design.seed + design.draw_count * SPLITMIX64_INCREMENT
    ) & UINT64_MASK
    if design.final_state != expected_state:
        raise ValueError("The recorded SplitMix64 final state is inconsistent with its draw count.")


__all__ = [
    "DECISION_BOUNDS",
    "DECISION_COLUMNS",
    "INFLUENT_BOUNDS",
    "INFLUENT_COLUMNS",
    "LatinHypercubeDesign",
    "ROBUSTNESS_COLUMNS",
    "SplitMix64",
    "TRAINING_BOUNDS",
    "TRAINING_COLUMNS",
    "affine_map",
    "generate_design",
    "generate_robustness_design",
    "generate_training_design",
    "unit_latin_hypercube",
    "validate_design",
    "validate_unit_latin_hypercube",
]
