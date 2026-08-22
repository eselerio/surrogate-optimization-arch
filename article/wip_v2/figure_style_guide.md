# Figure guide: combined Mixer-Reactor-Clarifier optimization

This guide accompanies **Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate**. Every figure must use “Clarifier” for the unit and must distinguish statistical fidelity, engineering feasibility, local NLP convergence, and exact mechanistic validation.

## Visual language

- Fresh influent: blue (`#2F6BFF`)
- Mixed liquor and biological stages: green (`#2E8B57`)
- RAS: dark orange (`#C76D00`)
- WAS: brown (`#7A4A21`)
- Clarified effluent: cyan (`#258EA6`)
- Raw statistical response: gray (`#777777`)
- Combined-NLP state: violet (`#6F4CC3`)
- Exact BDF validation: red outline (`#C43C35`)

Use white backgrounds, vector line art, direct units, colorblind-safe contrasts, and restrained grids. Never encode acceptance using color alone; add symbols or labels. Concentrations use ordinary state symbols, while Clarifier outlet component mass flows use `g_E` and `g_U`.

## Required process schematic

Show, from left to right:

1. fresh influent and the inlet mixer;
2. five equal-volume CSTRs;
3. MLR from reactor 5 to the mixer;
4. the ten-layer Clarifier with the feed at layer 5 from the top;
5. clarified effluent, RAS, and WAS; and
6. the external plant boundary.

Use arrows and labels that make clear that MLR and RAS are internal recycles and WAS is an external solids withdrawal. A Clarifier inset should depict upward overflow, downward underflow, settling flux, the receiver-capacity threshold, and the derived solids inventory.

## Required method diagram

Use four vertically separated information phases:

```text
independent mechanistic blocks
  -> development-only 351-feature, 170-response OLS
  -> calibration fidelity radius
  -> untouched assessment gate
  -> nine-start combined physics-constrained statistical NLP
  -> one selected local candidate
  -> one independent exact BDF validation
```

Annotate the data blocks as 70% LHS development, 10% iid calibration, and 20% iid assessment. Show that no refit occurs after development. Mark the combined NLP as 115 variables, 110 smooth steady-state equality rows, nine explicit inequalities, and variable bounds. Identify the nine inequalities as two domain guards, five engineering rows, fidelity, and leverage. State that the engineering score `J` is the sole objective.

## Prediction and calibration figures

Prediction panels should compare raw surrogate values with mechanistic targets on the untouched assessment block. Report coordinate/block standardized RMSE, bias, and R-squared where defined. A calibration panel should show the development-independent score distribution, the fixed 95% order statistic `delta`, and assessment coverage. State that conformal coverage is marginal under the iid sampling model and does not certify optimizer-selected inputs.

For plant-state parity plots, group coordinates by mixer, reactors 1--5, overflow mass flow, underflow mass flow, and Clarifier layers. Avoid plotting all 170 labels on one unreadable axis.

## Optimization figures

Show all nine start outcomes rather than only the winner. Distinguish accepted KKT points, solver failures, exact integration failure, residual/stability failure, domain/engineering failure, statistical-fidelity failure, and branch disagreement. The exact-fidelity reference is `d_BDF/delta - 1 <= 1e-8`, using the same normalized feasibility tolerance as the NLP. Objective traces may show IPOPT iterations but must not imply global convergence.

For each reported decision, display the selected combined-NLP controls and smooth physical state beside its one exact BDF replay. Show `J_NLP`, `J_exact`, `d/delta`, `d_BDF/delta`, leverage, and scaled branch agreement. Plot component and composite outlet differences as signed `NLP - BDF` values and identify zero explicitly. Do not draw a second optimization route or a comparison between two candidates.

## Clarifier and reactor diagnostics

Recommended Clarifier panels include the ten-layer TSS profile, exact versus smooth interface flux, overflow/underflow TSS, particulate recovery consistency, inventory contribution to SRT, and capacity activity. Highlight the fixed transition band around `X_t`; do not imply that the smooth branch equals the discontinuous exact branch there.

Recommended reactor panels show axial dissolved oxygen, ammonium/nitrite/nitrate, phosphate, readily biodegradable carbon, active biomass, and total solids. Plot exact BDF states separately from smooth combined-NLP states when branch agreement is under discussion.

## Robustness and sensitivity

Use separate outcome-count and continuous-summary figures so failures are not silently omitted. Continuous plots must state their eligible denominator. Show nominal plus 100 full-study robustness cases, three underflow-TSS cases, and ten objective-weight cases as distinct groups. Verification-profile figures must be labeled as execution tests and must not be mixed with full-study results.

## Timing and reproducibility

Timing plots separate development, calibration, and assessment generation; exact BDF replay; combined-NLP starts; fitting; calibration scoring; assessment; and reporting. Show count, median, interquartile range, 95th percentile, maximum, total CPU time, and wall time. Memory panels use the resident-set high-water mark of each complete stage. State the worker count, one-thread-per-NLP rule, 1.5 projection safety factor, 25-GiB memory gate, planned logical workload, actual attempts, interrupted attempts, and verified cache reuses.

Every final figure should be traceable to a sealed artifact and include units, case/profile identity, eligible sample count, and whether values come from the raw statistical response, smooth combined-NLP state, or exact BDF replay. Figures produced before terminal replay are provisional and unsealed; only a passed terminal replay and immutable seal can release a figure as validated.
