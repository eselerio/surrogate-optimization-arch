# Figure Style Guide

**Article title:** Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate

This internal guide governs every reader-facing diagram and numerical figure for the five-CSTR Mixer-Reactor-Clarifier activated-sludge study. Figures must explain the recycle topology, distinguish concentration from component mass flow, and show where the mechanistic model, statistical surrogate, physical projection, and bounded searches enter the calculation.

Numerical figures may use only one complete scientific run that passes the article's numerical and physical acceptance checks. Do not combine values from different runs, omit failed cases, interpolate missing scenarios, or promote a preliminary calculation into a reported result.

## Fixed Scientific and Visual Contract

Every figure must be consistent with the following study definition:

- five equal-volume CSTRs in series;
- CSTRs 1--2 unaerated and CSTRs 3--5 operated at one shared aerobic setting;
- mixed-liquor recycle (MLR) withdrawn from CSTR 5 before clarification and returned to the inlet mixer;
- a non-reactive, ten-layer flux-limited secondary Clarifier;
- Clarifier underflow divided hydraulically into return activated sludge (RAS) and waste activated sludge (WAS), which share one composition;
- five optimized controls, \((H,a,r_I,r_R,w)\);
- 20 ASM2d-TSN components and the four derived reporting quantities COD, TN, TP, and TSS;
- a 170-coordinate system target \(\chi=(m,c_1,\ldots,c_5,g_E,g_U,s_1,\ldots,s_{10})\);
- one fixed 351-column unique quadratic feature map fitted by standardized multiresponse ordinary least squares (OLS);
- a candidate-dependent equality diagnostic and one unique scaled-\(L_2\) physical-correction quadratic program (QP); and
- validation against the complete coupled ASM2d-TSN--Clarifier model, with prediction fidelity kept separate from decision regret.

Use the normalized flow identities consistently:

\[
q_P=1+r_I+r_R,\qquad
q_C=1+r_R,\qquad
q_U=r_R+w,\qquad
q_E=1-w.
\]

The Clarifier outlet coordinates represented by the surrogate are component mass flows normalized by fresh flow,

\[
g_E=q_Ec_E,\qquad g_U=q_Uc_U,
\]

not unconstrained outlet concentrations. If both forms appear, identify them explicitly in the axis, legend, or annotation. The ten layer-TSS coordinates \(s_1,\ldots,s_{10}\) are also direct targets; Clarifier solids inventory is derived from them and is not a separate regression target.

## Visual Language

Use a restrained journal-scale style with white backgrounds, vector line art, direct units, and no decorative gradients or three-dimensional effects. Use faint grids only for quantitative comparison and avoid dense legends over data.

Recommended plotting defaults are:

```python
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
```

Use consistent colors by stream or state:

| Meaning | Color | Hex | Secondary cue |
|---|---|---:|---|
| Fresh influent | steel blue | `#457B9D` | solid arrow |
| MLR | muted amber | `#D99033` | long-dashed arrow |
| RAS | muted plum | `#76527A` | dash-dot arrow |
| WAS | slate gray | `#6C757D` | dotted arrow |
| Treated effluent | mineral teal | `#2A9D8F` | solid arrow |
| Unaerated CSTR | pale blue-gray | `#A8C1D1` | `UNAERATED` label |
| Aerobic CSTR | warm sand | `#E9C46A` | `AEROBIC` label |
| Raw surrogate | coral rust | `#E76F51` | open marker |
| Affine diagnostic | muted amber | `#D99033` | half-filled marker |
| Deployed state | mineral teal | `#2A9D8F` | filled marker |
| Mechanistic response | deep teal | `#264653` | black-edged marker |
| Mechanistic reference | near black | `#222222` | star or cross |

Color must never be the only identity channel. Preserve the stated line styles, marker fills, labels, or hatching so each figure remains interpretable in grayscale and for common forms of color-vision deficiency.

## Required Process-Topology Figure

The principal Mixer-Reactor-Clarifier schematic must be readable from left to right and show:

1. fresh influent \(Q_0,x\) entering a three-stream mixer;
2. MLR \(r_IQ_0,c_5\) and RAS \(r_RQ_0,c_U\) returning to that mixer;
3. mixed flow \(q_PQ_0,m\) entering CSTR 1;
4. CSTRs 1--2 labeled unaerated and CSTRs 3--5 labeled aerobic;
5. the MLR takeoff from CSTR 5 before the Clarifier;
6. Clarifier feed \(q_CQ_0,c_5\);
7. ten visible Clarifier layers, ordered from the top water surface to the bottom hopper, with the feed entering layer 5 from the top;
8. overflow \(q_EQ_0,c_E\) leaving as treated effluent;
9. underflow \(q_UQ_0,c_U\) splitting into RAS and WAS; and
10. no connection suggesting that MLR passes through the Clarifier.

Place the four flow identities beneath or beside the schematic. Use a small Clarifier annotation to state

\[
g_E+g_U=q_Cc_5.
\]

Do not depict the Clarifier as a reacting vessel or imply that densification creates particulate mass.

## Required Modeling-Workflow Figure

The modeling workflow should separate four information stages:

1. **Mechanistic generation:** a coupled Mixer-Reactor-Clarifier calculation produces 20,000 accepted 170-coordinate states.
2. **Fixed fitting and assessment:** rows 1--16,000 fit one standardized 351-feature, 170-response OLS model by column-pivoted QR; an independent SVD checks rank, condition number, and coefficient agreement; rows 16,001--20,000 provide the untouched pass--fail assessment. After that assessment is sealed, the same estimator is refitted on all 20,000 rows.
3. **Physical deployment and optimization:** \((\vartheta,x)\) produces \(\chi_{\mathrm{raw}}\); one scaled strictly convex QP returns the unique deployed state \(\widehat\chi\); the five-control upper problem then evaluates the effluent objective, engineering constraints, and surrogate-trust screens.
4. **Independent validation:** the coupled mechanistic model is evaluated at the selected controls and searched independently to obtain a resolution-qualified reference incumbent and regret.

The surrogate and deployment portion should read:

```text
(five controls, 20-component influent)
   -> standardized 351-feature unique quadratic map
   -> fixed 170-response OLS coefficient matrix
   -> raw Mixer-Reactor-Clarifier state
   -> scaled-L2 physical-correction QP
   -> verified deployed state and upper-level evaluation
```

Show the equality-only affine projection as a side branch used to diagnose closure error during assessment. It is not a required precursor to every deployment QP. Annotate the QP with 77 equalities (mixer closure, five reactor-invariant blocks, Clarifier component balance, soluble pass-through, and two endpoint-TSS balances), 170 non-negativity bounds, and 26 Clarifier direction/layer-envelope inequalities. Upper SRT, hydraulic-loading, solids-loading, pumpability, leverage, correction, recovery-spread, and nonlinear-flux screens remain outside the lower QP.

The estimator is one fixed standardized OLS calculation. Show only its feature construction, QR solution, independent SVD audit, assessment gate, and unchanged production refit.

## Mechanistic Solver-Route Figure

If the numerical solution route is visualized, show its prescribed BDF-first order:

```text
prescribed Start 1
   -> scaled-coordinate BDF relaxation
      horizon <= max(400, 40/w) d; maximum step = horizon/100
      early stop at scaled derivative <= 1e-9 d^-1
   -> positive log-coordinate BDF retry only if the direct route loses positivity
   -> bounded least-squares polish only if the relaxed endpoint misses acceptance
   -> full residual, positivity, stability, and physical replay
   -> prescribed Start 2 only if the complete Start-1 route fails
```

The \(\max(400,40/w)\)-day value is an upper relaxation horizon, not a claim that every trajectory runs to that time. Do not draw least squares as the primary steady-state solver, imply cross-row warm starts, or depict clipping as a positivity treatment.

## Clarifier-Densification Inset

A compact Clarifier inset should make RAS densification intuitive. Show a particulate feed mass divided between overflow and underflow, with the underflow occupying the smaller liquid flow. State

\[
\eta_j=\frac{g_{U,j}}{q_Cc_{5,j}},\qquad
\delta_j=\frac{c_{U,j}}{c_{5,j}}
=\eta_j\frac{q_C}{q_U}.
\]

Use a separate soluble symbol or lane to show that solubles follow the water:

\[
E_Sg_U=q_UE_Sc_5.
\]

Do not imply that every particulate coordinate has an identifiable componentwise recovery when its feed denominator is below the declared tolerance.

## Main-Text Numerical Figure Recipes

### Nominal Axial and Clarifier Response

Use aligned panels for the nominal selected operation:

- axial profiles through \(m,c_1,\ldots,c_5\) for dissolved oxygen, nitrogen species, phosphorus species, and solids;
- overflow and underflow COD, TN, TP, and TSS, with deployed-surrogate and mechanistic values paired;
- the ten-layer Clarifier TSS profile; and
- selected \((H,a,r_I,r_R,w)\), derived Clarifier inventory, whole-plant SRT, hydraulic and solids loadings, total TSS recovery, and underflow densification in a compact linked table or callout.

Do not connect unlike ASM components as though they were one continuous scalar. Axial lines may connect the same component across process locations. Include native units on every concentration axis.

### Predictive Fidelity

Prediction figures must distinguish process location and prediction state. Suitable displays are:

- paired deployed-versus-mechanistic marks for overflow and underflow composites;
- component-error heat maps with the eight 20-component blocks in the fixed order \(m,c_1,\ldots,c_5,g_E,g_U\);
- a separate ten-coordinate panel for the Clarifier layer-TSS profile and its derived inventory; and
- raw, affine-diagnostic, and deployed errors or correction displacements in aligned panels rather than pooled.

Mass-flow, concentration, and layer-TSS blocks must not share an unqualified physical-unit axis. Use target-standardized metrics for cross-block comparisons and native units only in block-specific panels.

### Decision Quality, Search Coverage, and Robustness

Keep prediction error and operational regret visually distinct. Recommended panels are:

- normalized selected controls for all 100 robustness cases, with bounds at 0 and 1;
- mechanistic objective at the surrogate-selected decision paired with the independently searched mechanistic reference;
- regret with zero shown as a reference, never as a truncation rule;
- normalized operating displacement and control-bound activity; and
- search-attempt composition or final spatial resolution when it materially helps interpret the finite-budget incumbent.

If search budgets are displayed, use the exact contracts:

- surrogate search: 25,000 distinct attempts, with at most 18,000 full-box DIRECT keys, all 32 corners, at most 300 keys on each of ten coordinate faces, and at least 3,968 attempts plus unused DIRECT quota available for multi-basin pattern refinement;
- nominal mechanistic reference: 10,000 attempts, including the selected point and 50-direction stencil, corners, at most 7,000 full-box DIRECT keys, at most 100 keys per face, and at least 1,917 attempts for local refinement; and
- each robustness mechanistic reference: 2,500 attempts, including the selected point and stencil, corners, at most 1,700 full-box DIRECT keys, at most 25 keys per face, and at least 467 attempts for local refinement.

Robustness summaries are descriptive finite-sample summaries. Label medians, interquartile ranges, 95th percentiles, maxima, and failure counts explicitly. Do not label them as confidence intervals or statistical-significance evidence.

### Physical-Feasibility Diagnostics

Use separate aligned panels for quantities with different dimensions or tolerances:

- mixer component-balance residual;
- maximum reactor-invariant residual across the five tanks;
- Clarifier component mass-balance and soluble-pass-through residuals;
- Clarifier endpoint and layer-envelope residuals;
- plant-boundary invariant residual;
- minimum deployed coordinate;
- particulate densification-direction slack; and
- QP equality, inequality, stationarity, dual-feasibility, and complementarity residuals.

Logarithmic axes are appropriate for positive residual magnitudes, but zero values require an explicitly documented plotting floor. Draw the applicable acceptance tolerance as a labeled dashed line. Never replace a below-tolerance residual by zero in the source data.

## Detailed Diagnostic Figure Recipes

When space and the article's reporting plan permit, useful diagnostic figures include:

- mechanistic acceptance counts and residual distributions;
- representative BDF relaxation, positivity-retry, and optional-polish diagnostics;
- singular-value spectra and condition numbers for the 16,000-by-351 and 20,000-by-351 designs;
- QR optimality and QR--SVD coefficient-agreement checks;
- predictive errors for all 170 targets;
- raw-to-affine and raw-to-deployed correction distributions;
- component, total-particulate, and TSS recovery and densification distributions;
- low-, median-, and high-loading ten-layer Clarifier profiles;
- search-incumbent histories, attempt allocation, and achieved resolution;
- operating-weight and underflow-TSS-cap sensitivity results; and
- timing distributions for raw inference, matrix construction, QP solution, end-to-end deployment, surrogate search, and mechanistic search.

These diagnostics must describe the fixed estimator and its numerical checks directly.

## Metric and Axis Labels

Use the following terminology consistently:

- `target nRMSE` and `target nMAE` for errors standardized by fitting-only target scales;
- `RMSE` or `MAE` only with a named component, layer TSS, or composite and its physical unit;
- `Clarifier mass-balance residual` for \(\|g_E+g_U-q_Cc_5\|\);
- `plant-boundary invariant residual` for the external fresh-feed/effluent/WAS check;
- `TSS recovery` for a mass-flow fraction and `underflow densification` for a concentration ratio;
- `decision regret` for the matched mechanistic objective gap in the manuscript definition; and
- `best verified incumbent` rather than `global optimum` unless a genuine certificate exists.

COD, TN, TP, and TSS must be computed after obtaining the relevant 20-component state. Identify whether each composite belongs to CSTR 5, Clarifier overflow \(c_E\), or Clarifier underflow \(c_U\).

## Source-Data and Population Rules

Every final numerical figure must resolve to complete, traceable rows from the same accepted scientific run. Required pairing rules are:

- raw, affine-diagnostic, deployed, and mechanistic states use the same sample identifier where compared;
- surrogate-selected and selected-point mechanistic states use the same influent and controls;
- regret pairs use the same influent, objective, bounds, upper constraints, and mechanistic search contract;
- Clarifier recovery uses matched \(c_5,g_E,g_U\) and flow ratios; and
- robustness plots contain all 100 prescribed scenarios or clearly account for every failure class.

Never interpolate missing scenarios, silently discard failed cases, combine preliminary and complete runs, or infer a displayed value from a rounded manuscript number.

## Captions

Each caption must define:

- every abbreviation used in the figure;
- the process location and whether coordinates are concentrations, normalized mass flows, or layer TSS;
- the number and source of cases;
- whether values are raw, affine-diagnostic, deployed, or mechanistic;
- the summary statistic and dispersion measure;
- the physical or numerical tolerance when shown; and
- the finite-budget meaning of a reference or search incumbent.

Captions may interpret agreement with the coupled mechanistic model but must not claim validation against a full-scale plant.

## Accessibility and Export

- Use at least 7-point text at final journal size.
- Keep labels horizontal where practical and rotations no steeper than 45 degrees.
- Use marker shape, fill, line style, or labels in addition to color.
- Avoid red/green as the only contrast.
- Keep legends outside dense process schematics and data regions.
- Use vector PDF for line art and plots; rasterize only genuinely raster content.
- Embed fonts and inspect final-size output for clipped arrows, subscripts, and error bars.
- Use stable, descriptive filenames and exclude local paths, run identifiers, cell numbers, and development annotations from reader-facing material.

## Final Figure Checklist

Before a figure enters the submission package, verify that:

1. it comes from one complete accepted scientific run;
2. the topology has five CSTRs, the correct MLR takeoff, and the RAS/WAS split;
3. every flow label satisfies the declared identities;
4. concentration, normalized mass-flow, and layer-TSS coordinates are not conflated;
5. the Clarifier is non-reactive and mass conserving;
6. the fixed 351-feature OLS and 170-target response are represented accurately;
7. raw, affine-diagnostic, deployed, and mechanistic states use fixed visual semantics;
8. all expected scenarios, components, process blocks, and failure classes are present;
9. physical residuals are compared with their declared tolerances;
10. decision regret is not confused with prediction error;
11. finite-budget incumbents are not labeled globally optimal; and
12. no preliminary, cross-run, internal-path, or external-methods assumption remains.
