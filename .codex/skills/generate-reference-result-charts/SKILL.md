---
name: generate-reference-result-charts
description: Generate or regenerate the established eight-figure Extended-ICSOR-versus-Smooth-NLP result package with consecutive PNG filenames, a presentation-plan README, and numerical metadata, using the target run's own data.
---

# Generate Reference Result Charts

Generate exactly the eight untitled PNG figures listed below. Their order groups
holdout validation, selected-decision performance, process behavior, objective
trade-offs, and optimization time. Use the current package at
`results/article_v3/article_full_10000_001/report/figures` as the visual reference
when none is specified. Derive every number from the target run.

## Inputs and generation

Identify the target result directory. Default the output to its `report/figures`
folder unless the user specifies another location. A separate reference run is
optional. Reference images guide layout, never numerical values.

Reuse `scripts/generate_composite_two_route_charts.py`. From the repository root:

```powershell
uv run python -u scripts/generate_composite_two_route_charts.py <RUN_DIR> --output <RUN_DIR>/report/figures
```

The normal invocation must generate all eight figures and the README.
`--existing-only` preserves an explicitly reduced selection. Do not use it to
create a complete package when files are missing.

Read the target's `run_state.json`. Keep scientific inputs, fitted models,
optimization records, metrics, and contracts unchanged. When a run is active,
do not edit files in its `inputs/contract.json` `source_files` map. New chart
logic must remain outside that source-bound set. Write only requested figure
outputs, chart metadata, and necessary standalone chart-generation code.

## Exact figure set and order

Historical question IDs identify analytical sources, not the new sequence.

| Sequence | Required PNG | Historical sources | Contents and layout |
| --- | --- | --- | --- |
| 1 | `q01_holdout_accuracy_overview.png` | Q1/Q2/Q3 | A 3-by-3 grid. Rows show aggregate, location, and composite accuracy. Columns show nRMSE, nMAE, and mean location R². Compare raw and projected predictions throughout. |
| 2 | `q02_holdout_accuracy_by_location.png` | Q4 | Three heatmaps in one horizontal row showing raw nRMSE, projected nRMSE, and percentage change by location and composite. Raw and projected maps share a color scale. Negative change means improvement. |
| 3 | `q03_holdout_parity_all_locations.png` | Q18 | A 2-by-2 composite parity grid pooling all holdout observations across eight locations. Use log axes, a one-to-one line, shared logarithmic hexagon density, a labelled color bar, and markers explicitly named `Median — <location>`. Annotate nRMSE and mean location R². |
| 4 | `q04_effluent_and_removal_parity.png` | Q5/Q5R | A 2-by-4 Extended-ICSOR parity grid. Top row shows effluent concentration with teal dots; bottom row shows removal with orange dots. Columns are COD, TN, TP, TSS. Label x axes `Mechanistic (mg/L)` and `Mechanistic removal (%)`. |
| 5 | `q05_effluent_and_operating_values.png` | Q9/Q11 | Three rows and four columns. Top row shows mechanistic effluent COD, TN, TP, TSS for both routes. Lower rows show HRT, a3, a4, a5, internal recycle, return sludge, and wasting; turn the unused control panel off. Share one route legend. |
| 6 | `q06_treatment_train_profiles.png` | Q14/Q15/Q16/Q17 | Four rows for COD, TN, TP, TSS and two route columns. Each composite has identical logarithmic y limits for Extended ICSOR and Smooth NLP. Include nominal and all ten scenario profiles with one scenario legend. |
| 7 | `q07_objective_quality_economic.png` | Q7/Q8/Q10 | Three panels in one horizontal row showing total mechanistic objective, normalized water quality, and stacked weighted resource contributions. Resource bars are solid for Extended ICSOR and hatched for Smooth NLP. |
| 8 | `q08_optimization_time.png` | Q12 | Paired bars for both routes and every scenario with log seconds labelled `Optimization time (s; log scale)`. |

Do not emit separate constituent charts, retired comparisons, or SVGs.
Regeneration must not restore charts merged into this set. Replace obsolete
generated filenames only within the requested package after the new PNGs are
verified. Preserve unrelated user files.

## Shared presentation conventions

- Display `surrogate` as **Extended ICSOR** and `direct` as **Smooth NLP**.
- Use **Influent scenario** on scenario axes. Nominal is **N**, and scenarios
  1–10 are **S1–S10**. Internal artifact identifiers may retain `robustness_01`.
- Include the nominal result whenever scenarios are shown, including parity
  point labels. R1–R5 on location axes remain biological reactor labels.
- Omit figure and panel titles. Use axes, units, legends, metric annotations,
  the README, and manuscript captions to explain the panels.
- Keep concentrations in mg/L, removals in %, and removal differences in
  **percentage points**. Keep nRMSE, nMAE, and mean location R² consistent.
- Use **projection** and **optimization time**. Do not use reconciliation or
  qualify the displayed time as primary. Explain timing scope in the README.
- Avoid tiny-range scientific-offset control ticks. Use plain 0–1 limits for
  identically zero or effectively constant series within that range,
  particularly a3 and return sludge.
- In the established target run, include the supplied S1 Smooth NLP result as
  normal and valid in figures and paired summaries. Prefer the scientific
  casewise record, then the existing
  `report/chart_overrides/robustness_01_direct_casewise_reference.npz`.
  Do not add ineligibility shading or change scientific status files.
- For other target runs, require actual available data for every panel and
  scenario. Identify missing inputs rather than fabricating values or silently
  omitting scenarios. Do not claim a complete package when required data are
  unavailable.

## Target data and calculations

Prefer completed `report/tables/selected_quality.csv` and
`scenario_controls.csv`. Add nominal rows and any available result omitted by
these tables from the target's audited casewise records.

Usual sources are:

- `datasets/effective_design.npz` for development/test decisions and influents;
- `predictions/post_selection_holdout.npz` for `mechanistic`, `raw`, `projected`;
- `datasets/development/mechanistic_accepted_v3.npz` for development targets;
- `optimization/<case>/{surrogate,direct}_casewise_reference.npz` for `theta`,
  `projected`, `optimizer_native`, `exact_reference`, and full profiles;
- `metrics/robustness_case_timing.csv` for scenario optimization time;
- `report/tables/selected_candidate_reference_evaluation.csv` for nominal time.

Validate array shapes, finiteness, scenario coverage, and Boolean CSV fields.
Do not wait for tables when completed audited arrays supply the required data.

Import authoritative `COMPOSITE_MATRIX`, `NOMINAL_INFLUENT`, and `TSS_VECTOR`.
For the five-reactor, 20-component reduced response, mixer concentrations are
`0:20`, reactor concentrations are successive blocks in `20:120`, overflow
concentrations are `120:140 / (1-w)`, and underflow concentrations are
`140:160 / (r_R+w)`. Apply the composite matrix after converting outlet flows
to concentrations. Validate the target layout before applying these slices.

Normalize each holdout location/composite error by its own mechanistic holdout
range. Pool normalized errors for nRMSE and nMAE. Calculate R² separately at
each location/composite and average over the coordinates indicated by the
panel. Never calculate one pooled R² across locations or clip negative R².

Calculate removal as `100*(influent-effluent)/influent` with the nominal or
scenario's fresh influent. Figure 4 compares projected Extended ICSOR with the
mechanistic response at the same controls.

Figures 5–7 use `exact_reference` at each route's selected controls. Derive
quality scales as `std(development_effluent @ COMPOSITE_MATRIX.T, ddof=0)`.
The established objective components are quality (mean of composites divided
by their development standard deviations), HRT `(H-6)/30`, aeration
`H*(a3+a4+a5)/108`, internal recycle `r_I/4`, return sludge `(r_R-0.25)/1`, and
wasting `w*underflow_TSS/750`. Weights are
`(0.50,0.15,0.20,0.05,0.05,0.05)`. Figure 7's middle panel displays unweighted
quality; its objective contribution is 0.50 times that value. Resource stacks
are already weighted. If a target declares other bounds or priorities, use
its contract rather than silently applying these case-study constants.

Figure 6 follows `Influent → Mixer → R1 → R2 → R3 → R4 → R5 → Clarifier effluent`.
Prepend fresh-influent composites. Do not place underflow in this liquid path.

Optimization time includes the recorded search, surrogate fallback, and direct
smoothing sequence. It excludes generation, fitting, convergence certification,
additional recovery, and independent evaluation with the original mechanistic
model. State whether time summaries include nominal or cover only S1–S10.

## README and numerical metadata

Always create `README.md` in the output folder. Include:

- the presentation plan grouping items 1–3 as holdout validation, 4–6 as
  selected-decision and process assessment, and 7–8 as objective trade-offs and
  optimization time;
- all eight filenames in order, each figure's question, panel interpretation,
  data sources, and metric definitions;
- route names, nominal/scenario/reactor conventions, units, density hexagons
  versus location medians, shared profile scales, and timing exclusions.

q01–q08 denote Results presentation order. LaTeX may number them Figures 2–9
when an earlier plant flowsheet is Figure 1.

Write `chart_index.csv` with exactly eight rows and fields `results_sequence`,
`source_questions`, `presentation_role`, and `png`. Also write
`chart_summary.csv` with principal numerical comparisons and
`holdout_composite_metrics.csv` with raw/projected nRMSE, nMAE, and mean R².
Legacy summary question IDs are acceptable when the index explains the mapping.
Use calculated data for numerical interpretation rather than estimates read
from axes.

## Verification

Execute the generator. Confirm exactly eight PNGs with the listed names, zero
SVGs, eight index entries resolving to nonempty files, and a matching README.
Inspect representative holdout, selected-decision, control, and profile figures
for titles, readability, correct axes, nominal coverage, scenario labels, and
shared scales. Report the output folder and relevant calculated metrics.

When revising this skill, validate its instructions and verify generation in a
temporary output directory. Do not overwrite the published package solely to
test the skill.
