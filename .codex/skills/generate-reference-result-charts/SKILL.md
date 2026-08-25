---
name: generate-reference-result-charts
description: Reproduce the article-v3 Extended-ICSOR-versus-smooth-NLP chart package for a target results run by using another results folder as the visual and analytical reference. Use when asked to generate, regenerate, or transfer the established result charts from a reference run, especially when surrogate accuracy must be evaluated as COD, TN, TP, and TSS composites.
---

# Generate Reference Result Charts

Reproduce the reference chart package from the target run's own audited artifacts. Match the reference file set and visual intent; never copy the reference images or substitute its numerical data.

## Required inputs and defaults

Identify from the request or repository context:

- the target article-v3 result directory;
- the reference chart directory or reference result directory;
- the desired target subfolder.

If only the two result runs are given, find the reference package below `report/figures/` and create a target subfolder with the same basename. For the established two-route package, the basename is `system_surrogate_vs_smooth_nlp`.

Treat route ID `surrogate` as **Extended ICSOR** and route ID `direct` as **Smooth NLP**. Do not reintroduce, mention, or depend on the retired shared-unit route.

## Protect the scientific run

Read the target's `run_state.json` before acting.

- Chart generation is read-only with respect to scientific inputs, models, optimization checkpoints, metrics, and contracts.
- Write only the requested figures, chart metadata, and a reusable chart generator when needed.
- If the scientific run is active, do not edit any file listed in its `inputs/contract.json` `source_files` map. Put new chart logic in a standalone script that is not source-bound.
- Do not wait for `report/tables/` when the completed audited NPZ and metrics artifacts already contain everything required. Fall back to those artifacts as described below.
- If casewise result files are genuinely incomplete, monitor the run or report the exact missing inputs; do not fabricate cases or reuse values from the reference run.

## Start by reverse-engineering the reference

Inventory the reference directory recursively. Read `chart_index.csv` and `chart_summary.csv` when present. Record exact filenames, formats, chart ordering, route scope, case labels, panel layouts, scales, eligibility shading, and annotations.

Search the repository for the generator and chart stems before writing new code. Reuse a compatible current generator when it follows this skill's composite-performance rules. Otherwise create or adapt a reusable command-line generator that accepts target run and output paths rather than hardcoding one run.

The established package contains these 19 PNG/SVG pairs:

1. `q01_holdout_composite_accuracy`
2. `q02_holdout_accuracy_by_response_block`
3. `q03_holdout_component_accuracy`
4. `q04_holdout_component_accuracy_by_stage`
5. `q05_surrogate_effluent_prediction_vs_mechanistic`
6. `q06_smooth_nlp_effluent_prediction_vs_mechanistic`
7. `q07_exact_optimal_objective`
8. `q08_exact_water_quality_component`
9. `q09_exact_effluent_composites`
10. `q10_exact_economic_component`
11. `q11_optimal_operating_values`
12. `q12_primary_optimization_time`
13. `q13_exact_objective_value_comparison`
14. `q14_cod_main_treatment_train_profiles`
15. `q15_tn_main_treatment_train_profiles`
16. `q16_tp_main_treatment_train_profiles`
17. `q17_tss_main_treatment_train_profiles`
18. `q05_surrogate_percent_removal_vs_mechanistic` (`5R`)
19. `q06_smooth_nlp_percent_removal_vs_mechanistic` (`6R`)

Preserve these stems even though Q2--Q4 now describe composite rather than individual ASM-component performance. This keeps references and article links stable.

## Prefer these target artifacts

Use target-run data only. The usual sources are:

- `datasets/effective_design.npz` for development/test decisions and robustness influents;
- `predictions/post_selection_holdout.npz` for `mechanistic`, `raw`, and `projected` reduced responses;
- `datasets/development/mechanistic_accepted_v3.npz` for the development mechanistic targets used to derive quality scales;
- `optimization/<case>/surrogate_casewise_reference.npz`;
- `optimization/<case>/direct_casewise_reference.npz`;
- `metrics/case_common_reference_comparison.csv` for pair eligibility;
- `metrics/robustness_case_timing.csv` for primary optimization time;
- completed `report/tables/selected_quality.csv`, `scenario_controls.csv`, and `scenario_comparison.csv` when available.

Infer robustness cases from the target design and available optimization directories. Preserve the reference label convention: nominal is `N`, then robustness cases are `R1`, `R2`, and so forth.

If reporting tables are absent, derive them as follows:

- selected quality: load each route's casewise-reference NPZ, use `projected` for Extended ICSOR, `optimizer_native` for Smooth NLP, and `exact_reference` for mechanistic replay;
- controls: read `theta` from each casewise-reference NPZ;
- eligibility: use `comparison_eligible` in `metrics/case_common_reference_comparison.csv`;
- exact profiles: use `exact_reference_full` and the corresponding `theta`.

Validate array shapes and finiteness before plotting. Treat Boolean CSV fields robustly when encoded as strings.

## Composite definitions are mandatory for prediction performance

Q1--Q6 and 5R/6R must assess COD, TN, TP, and TSS composites, not the 20 individual state components and not a complete-response aggregate in raw coordinate units.

Import the repository's authoritative `COMPOSITE_MATRIX`, `NOMINAL_INFLUENT`, and `TSS_VECTOR`; do not duplicate their coefficients.

For a reduced response with layout

`(mixer, reactor_1, ..., reactor_5, overflow_component_flows, underflow_component_flows, clarifier_inventory)`, use:

- mixer: columns `0:20`;
- reactors 1--5: successive 20-column blocks `20:120`;
- overflow concentration: columns `120:140 / (1 - w)`;
- underflow concentration: columns `140:160 / (r_R + w)`.

Apply `COMPOSITE_MATRIX` to concentration vectors. Never compare overflow or underflow component flows directly with concentration composites.

For coordinate-normalized composite errors, calculate each location/composite truth range on the target holdout, normalize errors by that range, then aggregate. Do not normalize COD, TN, TP, and TSS with one shared scale. Permit negative R² values when supported by the data; do not clip or hide them.

Use these performance-chart meanings:

- Q1: raw versus projected aggregate nRMSE, nMAE, and mean R² across COD/TN/TP/TSS and all supported locations;
- Q2: raw versus projected composite nRMSE by system location;
- Q3: raw versus projected nRMSE, nMAE, and R² by COD/TN/TP/TSS;
- Q4: raw, projected, and percentage-change heatmaps by location and COD/TN/TP/TSS;
- Q5: Extended-ICSOR projected effluent composites versus exact mechanistic replay at its selected decisions;
- Q6: smooth-NLP native effluent composites versus exact mechanistic replay at its selected decisions.

## Removal charts

Build influent composites from `NOMINAL_INFLUENT` for the nominal case and `robustness_influents` for robustness cases. For each composite, calculate

`removal_percent = 100 * (influent - effluent) / influent`.

Compare the optimizer-native removal with exact-replay removal in percentage points. Label the metric as removal error in **percentage points**, not percent relative error.

## Exact optimization charts

For each case and route, load `theta` and `exact_reference` from the route's casewise-reference NPZ. Derive:

- effluent concentration from `exact_reference[120:140] / (1 - w)`;
- underflow concentration from `exact_reference[140:160] / (r_R + w)`;
- COD/TN/TP/TSS via `COMPOSITE_MATRIX`;
- underflow TSS via `TSS_VECTOR`.

Derive the four effluent quality scales from the target development data:

`std((development_targets[:, 120:140] / (1 - development_w)) @ COMPOSITE_MATRIX.T, ddof=0)`.

Use the established objective components and weights:

- quality: mean of the four effluent composites divided by their development scales;
- HRT: `(H - 6) / 30`;
- aeration: `H * (a3 + a4 + a5) / 108`;
- internal recycle: `r_I / 4`;
- return sludge: `(r_R - 0.25) / 1`;
- wasting: `w * underflow_TSS / 750`;
- weights: `(0.50, 0.15, 0.20, 0.05, 0.05, 0.05)`.

Q7 and Q13 show exact total objectives, Q8 the exact quality component, Q9 the exact effluent composites, Q10 the weighted economic/resource decomposition, Q11 the selected controls, and Q12 primary optimization time only. Exclude certification, recovery, and exact-replay time from Q12.

Shade cases where `comparison_eligible` is false, but retain their plotted values and explain the shading. Calculate paired win counts only over eligible cases.

## Treatment-train profiles

Q14--Q17 show exact mechanistic replay at each route's selected decision. The serial path is:

`Influent -> Mixer -> R1 -> R2 -> R3 -> R4 -> R5 -> Clarifier effluent`.

Use `exact_reference_full`:

- mixer `0:20`;
- reactors as successive blocks through `120`;
- clarifier effluent `120:140 / (1 - w)`.

Prepend the case's influent composite. Do not place clarifier underflow on this serial liquid-treatment path; it is a side stream. Use log y-axes when all values are positive. Show nominal and every robustness case, with line style or annotation distinguishing ineligible cases. Use separate Extended-ICSOR and Smooth-NLP panels.

## Output contract

Produce:

- every reference chart in both PNG and SVG;
- `chart_index.csv` with `question`, `png`, and `svg`, including 5R and 6R;
- `chart_summary.csv` with the principal numerical comparisons;
- `holdout_composite_metrics.csv` containing raw/projected nRMSE, nMAE, and mean R².

Use clear route labels (`Extended ICSOR`, `Smooth NLP`), case annotations, units, lower/higher-is-better cues, and eligibility legends. Keep filenames identical to the reference package unless the user explicitly requests new names.

## Verification before handoff

1. Compare target and reference basenames. The only expected additional file is `holdout_composite_metrics.csv` unless the user requested otherwise.
2. Confirm `chart_index.csv` has 19 rows and every indexed PNG/SVG exists and is nonempty.
3. Confirm all numerical inputs came from the target run.
4. Visually inspect at least Q1, Q4, Q5, and one of Q14--Q17 for clipping, unreadable legends, incorrect units, bad axes, and missing cases.
5. Report the output folder and concise composite metrics. Mention any ineligible cases or intentionally unavailable charts.

Do not claim success after merely creating a generator. Execute it, validate the artifacts, and leave the complete chart package in the target result folder.
