# Figure Style Guide for Closed-Loop Reactor--Settler Optimization

This internal guide governs every reader-facing diagram and numerical figure for the study of bilevel optimization of a five-CSTR activated-sludge train with mixed-liquor recycle (MLR), a ten-layer secondary settler, return activated sludge (RAS), waste activated sludge (WAS), and a system-level ICSOR surrogate.

Figures must explain the process topology, distinguish concentration from component mass flow, and show where the physical guarantees enter the calculation. Numerical figures may be populated only from one completed, article-eligible full-profile run whose terminal checks, inventory, and manifest hashes have passed. Smoke runs, partial robustness cohorts, provisional checkpoints, and values assembled across runs are prohibited.

## Fixed Scientific and Visual Contract

Every figure must be consistent with the following study definition:

- five equal-volume CSTRs in series;
- CSTRs 1--2 unaerated and CSTRs 3--5 operated at one shared aerobic setting;
- MLR withdrawn from CSTR 5 before clarification and returned to the inlet mixer;
- a non-reactive, ten-layer flux-limited secondary settler;
- separator underflow divided hydraulically into RAS and WAS, which share one composition;
- five optimized controls, \((H,a,r_I,r_R,w)\);
- 20 ASM2d-TSN components and the four derived reporting quantities COD, TN, TP, and TSS;
- a 161-coordinate system target \((m,c_1,\ldots,c_5,g_E,g_U,\overline M_{\mathrm{cl}})\);
- a 551-column second-order system-level ICSOR feature map;
- a candidate-dependent affine equality projection followed by one unique scaled-\(L_2\) physical-correction QP; and
- validation against the complete coupled ASM2d-TSN--settler network, with prediction fidelity kept separate from decision regret.

Use the normalized flow identities consistently:

\[
q_P=1+r_I+r_R,\qquad
q_C=1+r_R,\qquad
q_U=r_R+w,\qquad
q_E=1-w.
\]

The separator coordinates are component mass flows normalized by fresh flow,

\[
g_E=q_Ec_E,\qquad g_U=q_Uc_U,
\]

not unconstrained outlet concentrations. If both forms appear, the distinction must be explicit in the axis, legend, or annotation.

## Visual Language

Use a restrained, journal-scale style with white backgrounds, vector line art, direct units, and no decorative gradients or three-dimensional effects. Use faint grids only for quantitative comparison. Avoid dense legends over data.

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
| Anoxic CSTR | pale blue-gray | `#A8C1D1` | `ANX` label |
| Aerobic CSTR | warm sand | `#E9C46A` | `AER` label |
| Raw surrogate | coral rust | `#E76F51` | open marker |
| Affine state | muted amber | `#D99033` | half-filled marker |
| Deployed state | mineral teal | `#2A9D8F` | filled marker |
| Mechanistic response | deep teal | `#264653` | black-edged marker |
| Mechanistic reference | near black | `#222222` | star or cross |

Color must never be the only identity channel. Preserve the stated line styles, marker fills, labels, or hatching so the figure remains interpretable in grayscale and for common forms of color-vision deficiency.

## Required Process-Topology Figure

The principal process schematic must be readable from left to right and must show:

1. fresh influent \(Q_0,x\) entering a three-stream mixer;
2. MLR \(r_IQ_0,c_5\) and RAS \(r_RQ_0,c_U\) returning to that mixer;
3. mixed flow \(q_PQ_0,m\) entering CSTR 1;
4. CSTRs 1--2 labeled unaerated/anoxic and CSTRs 3--5 labeled aerobic;
5. an MLR takeoff from CSTR 5 before the settler;
6. clarifier feed \(q_CQ_0,c_5\);
7. a ten-layer settler drawn with ten visible horizontal bands and the feed layer identified;
8. overflow \(q_EQ_0,c_E\) leaving as treated effluent;
9. underflow \(q_UQ_0,c_U\) splitting into RAS and WAS; and
10. no direct connection suggesting that MLR passes through the separator.

Place the four flow identities beneath or beside the schematic. Use a small separator annotation to state

\[
g_E+g_U=q_Cc_5.
\]

Do not depict the separator as a reacting vessel or imply that densification creates particulate mass.

## Required Modeling-Workflow Figure

The modeling workflow should separate four information stages:

1. **Mechanistic generation:** the closed-loop model produces accepted 161-coordinate states, including normalized clarifier solids inventory.
2. **Training and freezing:** 16,000 estimation rows select and fit the network ICSOR; 4,000 untouched rows assess it; a production model is then refit on all 20,000 rows and frozen.
3. **Bilevel evaluation:** \((\vartheta,x)\) produces \(\chi_{\mathrm{raw}}\), then \(\chi_{\mathrm{aff}}\), then the unique deployed state \(\widehat\chi\), and finally the overflow objective and upper constraints.
4. **Independent validation:** the coupled mechanistic network is evaluated at the selected controls and separately searched to obtain a resolution-qualified reference and regret.

The lower-level portion must show the responsibilities of its two transformations:

```text
(theta, x)
   -> system-level ICSOR raw state
   -> candidate-specific equality projection
   -> scaled-L2 QP with non-negativity and TSS densification
   -> verified deployed network state
```

Annotate the equality block with mixer closure, five reactor invariant blocks, separator component balance, and soluble pass-through. Annotate the QP block with non-negativity, the underflow-TSS inequality, and independent KKT checks. Do not suggest that upper discharge, SRT, hydraulic-loading, or pump-capacity limits are absorbed into the lower correction; they are upper-level acceptance constraints.

## Separator-Densification Inset

A compact clarifier inset should make RAS densification intuitive. Show a particulate feed mass divided between overflow and underflow, with the underflow occupying the smaller liquid flow. State

\[
\eta_j=\frac{g_{U,j}}{q_Cc_{5,j}},\qquad
\delta_j=\frac{c_{U,j}}{c_{5,j}}
=\eta_j\frac{q_C}{q_U}.
\]

Use a separate soluble symbol or lane to show that solubles follow the water:

\[
E_Sg_U=q_UE_Sc_5.
\]

The inset must not imply that all particulate coordinates have a defined componentwise recovery when their feed denominator is effectively zero. Such coordinates are reported as not numerically identifiable under the declared denominator tolerance.

## Main-Text Numerical Figure Recipes

### Nominal Axial and Separator Response

Use aligned panels for the nominal selected operation:

- axial profiles through \(m,c_1,\ldots,c_5\) for dissolved oxygen, nitrogen species, phosphorus species, and solids;
- overflow and underflow COD, TN, TP, and TSS, with surrogate and mechanistic values paired;
- a ten-layer solids profile through the settler; and
- the selected \((H,a,r_I,r_R,w)\), clarifier solids inventory, whole-plant SRT, hydraulic loading, solids loading, total TSS recovery, and underflow densification in a compact linked table or callout.

Do not connect unlike ASM components as though they were one continuous scalar. Axial lines may connect the same component across process locations. Include native units on every concentration axis.

### Predictive Fidelity

Prediction figures must distinguish process location and prediction state. Suitable displays are:

- paired deployed-versus-mechanistic marks for overflow and underflow composites;
- component-error heat maps with the eight 20-component blocks in the fixed order \(m,c_1,\ldots,c_5,g_E,g_U\); and
- a paired normalized clarifier-inventory prediction panel; and
- raw, affine, and deployed errors or correction displacements shown in aligned panels rather than pooled.

Mass-flow blocks and concentration blocks must not share an unqualified physical-unit axis. Use component-standardized metrics for cross-block comparisons and native units only in block-specific panels.

### Decision Quality and Robustness

Keep prediction error and operational regret visually distinct. Recommended panels are:

- normalized selected controls for all 100 robustness cases, with bounds at 0 and 1;
- mechanistic objective at the surrogate decision paired with the independently searched mechanistic reference;
- a regret distribution with zero shown as a reference, not as a truncation point;
- normalized operating displacement; and
- bound-activity frequencies for \(H,a,r_I,r_R,w\).

Robustness summaries are descriptive finite-sample summaries. Label medians, interquartile ranges, 95th percentiles, maxima, and failure counts explicitly. Do not label them as confidence intervals or statistical significance evidence.

### Physical-Feasibility Diagnostics

Use separate aligned panels for quantities with different dimensions or tolerances:

- mixer component-balance residual;
- maximum reactor-invariant residual across the five tanks;
- separator component mass-balance residual;
- soluble-pass-through residual;
- plant-boundary invariant residual;
- minimum deployed coordinate;
- underflow-TSS densification slack; and
- stationarity, dual-feasibility, and complementarity residuals.

Logarithmic axes are appropriate for positive residual magnitudes, but zero values require an explicitly documented plotting floor. Draw the applicable acceptance tolerance as a labeled dashed line. Never replace a below-tolerance residual by zero in the source data.

## Supplementary Figure Recipes

The supplement should retain enough detail to diagnose both the mechanistic generator and the surrogate:

- attempted-versus-accepted mechanistic-state counts and residual distributions;
- representative steady-solve and dynamic-relaxation diagnostics;
- training and validation objective histories for the selected network ICSOR;
- predictive errors for every one of the 161 output coordinates;
- blockwise coupling summaries for the eight 20-by-20 diagonal blocks of \(\widehat\Gamma\);
- raw-to-affine and affine-to-deployed correction distributions;
- component, total-particulate, and TSS recovery and densification distributions;
- ten-layer settler solids profiles at low, median, and high loading;
- search-incumbent histories and evaluation counts;
- nominal penalty-weight sensitivity results;
- underflow-TSS-cap sensitivity at 12, 15, and 20 g L$^{-1}$; and
- timing distributions for raw inference, equality-operator construction, affine projection, QP solution, end-to-end deployment, surrogate search, and mechanistic search.

Do not render a dense 161-by-161 coupling matrix as the only coupling diagnostic. Respect the declared block-diagonal mask, label the eight process-location blocks, and show the uncoupled clarifier-inventory coordinate separately.

## Metric and Axis Labels

Use the following terminology consistently:

- `component nRMSE` and `component nMAE` for errors standardized by training-only coordinate scales;
- `RMSE` or `MAE` only with a named component or composite and its physical unit;
- `separator mass-balance residual` for \(\|g_E+g_U-q_Cc_5\|\);
- `plant-boundary invariant residual` for the external fresh-feed/effluent/WAS check;
- `TSS recovery` for a mass-flow fraction and `underflow densification` for a concentration ratio;
- `decision regret` for the mechanistic objective gap in the manuscript definition; and
- `best verified incumbent` rather than `global optimum` unless a genuine certificate exists.

COD, TN, TP, and TSS must always be computed after obtaining the relevant 20-component state. State whether the reported composite is for CSTR 5, overflow \(c_E\), or underflow \(c_U\).

## Source-Data and Population Rules

Every final numerical figure must resolve to manifest-tracked rows from the same accepted full run. The rendering code must fail closed when an expected scenario, process block, component, control, prediction state, or mechanistic pair is absent.

Required pairing rules are:

- raw, affine, and deployed states use the same sample identifier;
- surrogate-selected and selected-point mechanistic states use the same influent and controls;
- regret pairs use the same influent, objective, bounds, upper constraints, and mechanistic search contract;
- separator recovery uses matched \(c_5,g_E,g_U\) and flow ratios; and
- robustness plots contain all 100 accepted validation scenarios or are explicitly marked diagnostic and excluded from submission.

Never interpolate missing scenarios, silently discard failed cases, pool smoke and full runs, or infer a displayed value from a rounded manuscript number.

## Captions

Each caption must be self-contained and define:

- all abbreviations used in the figure;
- the process location and whether coordinates are concentrations or normalized mass flows;
- the number and source of cases;
- whether values are raw, affine, deployed, or mechanistic;
- the summary statistic and dispersion measure;
- the physical or numerical tolerance when shown; and
- the finite-budget meaning of a mechanistic reference or search incumbent.

Captions should interpret the visual comparison without claiming plant validation. The study is validated against the coupled mechanistic model.

## Accessibility and Export

- Use at least 7-point text at final journal size.
- Keep labels horizontal where practical and rotations no steeper than 45 degrees.
- Use marker shape, fill, line style, or labels in addition to color.
- Avoid red/green as the only contrast.
- Keep legends outside dense process schematics and data regions.
- Use vector PDF for line art and plots; rasterize only genuinely raster content.
- Embed fonts and inspect the final rendered size for clipped arrows, subscripts, and error bars.
- Use stable, flat filenames and exclude local paths, run identifiers, cell numbers, and development annotations.

## Final Figure Checklist

Before a figure enters the submission package, verify that:

1. it comes from the single sealed, article-eligible full run;
2. the depicted topology has five CSTRs, the correct MLR takeoff, and the RAS/WAS split;
3. all flow labels satisfy the declared identities;
4. concentration and normalized mass-flow coordinates are not conflated;
5. the separator is shown as non-reactive and mass conserving;
6. raw, affine, deployed, and mechanistic states use fixed visual semantics;
7. all expected scenarios, components, and process blocks are present;
8. physical residuals are compared with their declared tolerances;
9. decision regret is not confused with prediction error;
10. finite-budget incumbents are not labeled globally optimal;
11. all numerical source rows and hashes pass the article finalizer; and
12. no provisional, smoke-profile, cross-run, or internal-path information remains.
