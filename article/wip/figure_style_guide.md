# Figure Style Guide for the Physical-Enforcement Benchmark

This internal authoring guide applies to the current study of post-prediction mass-conservation and non-negativity enforcement across 13 CPU-based statistical regressors.

The manuscript uses an inline graphical abstract, an evaluation-workflow diagram, an in-distribution data-scarcity figure, a projection-effect figure, and an OOD comparison figure. The supplement uses all-model learning curves, a per-component error atlas, physical-admissibility diagnostics, and OOD residual distributions.

Numerical graphics are populated only from one completed, article-eligible 13-model bundle whose terminal assertions and manifest hash have been verified. The external article finalizer consumes the manifest-tracked source tables and renders the submission figures according to this guide. Notebook-generated images are diagnostic exports; where their geometry differs from this guide, the external finalizer is authoritative. Estimated, partial, and cross-run values are prohibited.

## Scientific Contract

Every final results figure must use the same benchmark contract as the tables:

- 22 inputs and 20 ASM2d-TSN effluent-component targets;
- eleven nested totals from 500 to 10,000;
- persistent seed-42 five-fold assignments shared by every model;
- one raw and one projected prediction for each scored row;
- primary aggregate metrics labeled exactly `nMSE`, `nRMSE`, and `nMAE`;
- COD, TN, TP, and TSS shown only as secondary quantities derived from the component vector;
- mild and severe OOD strata kept separate; and
- every displayed value resolves to the single accepted run and its manifest-tracked source rows.

## Descriptive Fold Summaries

Curves, bars, and table-linked callouts summarize the five outer-fold values as the arithmetic mean plus or minus the sample standard deviation. Compute the standard deviation with denominator four (`ddof=1`).

Use these phrases in captions and legends:

- `five-fold mean` for the central curve or marker;
- `sample SD` for an error bar or band; and
- `descriptive fold-to-fold variation` when interpretation is needed.

Do not label sample-SD bars as confidence intervals, error of the mean, or statistical evidence. Do not add p-values, stars, or pairwise-ranking annotations to this descriptive benchmark.

Minimal calculation pattern:

```python
fold_mean = fold_values.mean(axis=0)
fold_sample_sd = fold_values.std(axis=0, ddof=1)
```

## Metric Labels

The component-standardized aggregate labels are not interchangeable with unqualified physical-unit metrics.

- Use `nMSE` for mean squared standardized component error.
- Use `nRMSE` for the square root of nMSE.
- Use `nMAE` for mean absolute standardized component error.
- Use `ΔnRMSE (projected − raw)` for the paired projection effect.
- Add a component subscript, such as `nRMSE_j`, for component-specific standardized panels.
- Use unqualified `RMSE` or `MAE` only when the axis also states the physical component or composite and its unit.

Never label the primary component-standardized axis as `aggregate RMSE`, `effective RMSE`, or `test RMSE`.

## Visual Language

The visual style should remain restrained, readable in a single-column layout, and stable across main-text and supplementary figures:

- white figure and axes backgrounds;
- faint dotted grids only where they aid quantitative reading;
- vector line art and text whenever practical;
- direct axis labels with metric and unit;
- no decorative gradients, shadows, or three-dimensional effects;
- marker shapes as a second identity channel in addition to color; and
- concise titles, with scientific interpretation left to the caption and text.

Recommended plotting defaults:

```python
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
```

## Palette and Semantics

Use the manuscript's muted earth-and-mineral palette:

```python
COLORS = {
    "deep_teal": "#264653",
    "mineral_green": "#2A9D8F",
    "muted_amber": "#E9C46A",
    "warm_sand": "#F4A261",
    "coral_rust": "#E76F51",
    "muted_plum": "#6D597A",
    "steel_blue": "#577590",
    "brick_red": "#BC4749",
    "slate_gray": "#8D99AE",
    "dark_gray": "#5C6770",
    "olive_gray": "#6B705C",
    "blue_green": "#4D908E",
    "umber": "#9C6644",
}
```

For paired prediction-state figures, semantics take priority over model identity:

- raw: coral rust;
- projected: mineral green;
- zero or tolerance reference: dark gray, dashed; and
- mechanistic target, if shown: deep teal or black.

For multi-model displays, keep the exact notebook color and marker assignment throughout the manuscript and supplement. The fixed roster order and mapping are:

| Model | Notebook key | Hex color | Matplotlib marker | Shape |
|---|---|---:|:---:|---|
| XGBoost | `xgboost_regressor` | `#577590` | `o` | circle |
| LightGBM | `lightgbm_regressor` | `#264653` | `s` | square |
| CatBoost | `catboost_regressor` | `#E76F51` | `^` | triangle up |
| AdaBoost | `adaboost_regressor` | `#F4A261` | `v` | triangle down |
| Random Forest | `random_forest_regressor` | `#8D99AE` | `D` | diamond |
| Extra Trees | `extra_trees_regressor` | `#6B705C` | `d` | thin diamond |
| SVR | `svr_regressor` | `#BC4749` | `P` | filled plus |
| k-NN | `knn_regressor` | `#E9C46A` | `>` | triangle right |
| PLS | `pls_regressor` | `#5C6770` | `<` | triangle left |
| Multi-task Elastic Net | `multitask_elastic_net_regressor` | `#2A9D8F` | `h` | hexagon |
| Multi-task Lasso | `multitask_lasso_regressor` | `#4D908E` | `+` | plus |
| MLP | `ann_deep_regressor` | `#6D597A` | `p` | pentagon |
| TabNet | `tabnet_regressor` | `#9C6644` | `*` | star |

No model should receive a privileged visual highlight. The scientific intervention is the shared projection, represented through raw/projected semantics.

## Main-Text Figure Recipes

### Graphical Abstract and Evaluation Workflow

The workflow should read left to right:

1. accepted mechanistic states and the 22 inputs;
2. 13 CPU-based learners;
3. raw 20-component prediction;
4. Kircher--Votsmeier projection; and
5. paired accuracy, physical, OOD, and timing evaluation.

Use rounded boxes, one arrow direction, and no model-specific architecture details. The diagram should make clear that COD, TN, TP, and TSS are derived only after component-space prediction and correction. Do not show artificial clipping of mechanistic rates; the simulator evaluates the rate vector directly.

### In-Distribution Data-Scarcity Figure

The external finalizer renders one aligned two-panel figure from the accepted ID fold summaries and timing summaries. The left panel shows raw-prediction `nRMSE` against total dataset size. The right panel shows model setup time against the same x-axis.

- Plot five-fold means as lines with markers.
- Plot sample SD as symmetric bars or a light band.
- State `five-fold mean with sample SD` in the caption.
- Use all 13 models and all eleven nested sizes.
- Label the right panel `Model setup time (s)` and use a logarithmic y-axis after confirming all plotted values are positive.
- Preserve the fixed roster order, colors, and markers in both panels.
- Keep inference latency out of the setup-time panel; report it separately.
- Do not substitute the notebook's single-panel learning-curve image for this required two-panel composition.

### Projection-Effect Figure

Render a model-by-sample-size heat map of

```text
ΔnRMSE = nRMSE_projected − nRMSE_raw
```

using the accepted paired fold results. Rows follow the fixed 13-model roster and columns follow the eleven nested totals in ascending order. Each cell is the arithmetic mean of the five within-fold projected-minus-raw differences. Use a zero-centered diverging color scale with symmetric limits; negative values denote lower nRMSE after projection and positive values denote higher nRMSE. Report paired fold dispersion in the linked table or supplement rather than encoding it as a second heat-map variable. Do not render projection-effect curves. The caption must state that feasibility and predictive error are distinct outcomes.

### OOD Raw/Projected Figure

The main-text severe-OOD figure uses horizontal paired marks: one roster-ordered row per model, raw and projected nRMSE points connected by a thin neutral line. Put nRMSE on the horizontal axis and the fixed 13-model roster on the vertical axis. Map raw to coral rust and projected to mineral green, with distinct marker or fill treatment for grayscale accessibility. Do not use grouped bars, do not reorder models by observed performance, and do not pool mild and severe results. Mild OOD remains in the linked table and supplementary displays. The caption must state that projection restores the declared physical contract but need not reduce extrapolation error.

The finalizer uses the severe-OOD source-data export rather than the notebook's diagnostic bar rendering.

## Supplementary Figure Recipes

### All-Model Learning Curves

Show all 13 models over all eleven data sizes. If one panel is too dense, split the learners into aligned panels with identical axes. Do not rank models by eye through line thickness; use equal widths and the fixed color/marker mapping.

### Per-Component Error Atlas

Use the fixed 20-component order on the x-axis and models on the y-axis. Separate raw nRMSE$_j$, projected nRMSE$_j$, and projection displacement into aligned panels. A shared color scale is appropriate only when the metric and range are identical. Physical-unit component panels need their own units or separate scales.

For annotated heatmaps, change text color according to cell luminance and omit cell numbers when the matrix becomes illegible at journal size.

### Physical-Admissibility Diagnostics

Keep mass-conservation residual, non-negativity frequency/magnitude, backtracking activation, and projection displacement in separate aligned panels. Do not merge differently scaled quantities on dual y-axes. A log scale may be used for positive residual magnitudes, but zero values must be handled and described explicitly.

### OOD Distributions

Show sample-level residual or displacement distributions separately for mild and severe OOD inputs. Paired raw/projected points are preferable when row identity matters. Use identical limits across severity panels when direct comparison is intended.

## Final Population and Rendering

The external finalizer must fail closed unless the run is marked complete and article-eligible, the manifest digest verifies, all 13 model keys and all expected row cardinalities are present, and raw/projected rows pair exactly. It must then:

- calculate displayed summaries from the accepted source rows using the definitions in this guide;
- preserve the manuscript's retained figure roles, labels, and model order;
- render the required two-panel scaling figure, projection heat map, and horizontal severe-OOD paired marks even when notebook diagnostic images use a different layout;
- leave no missing field, fold, model, or size silently blank and never interpolate it; and
- emit captions and graphics without run-status notices, local paths, or development annotations.

## Accessibility and Layout Checks

- Verify that every series remains identifiable in grayscale through marker or line style.
- Avoid red/green as the only raw/projected distinction; use fill, marker, or hatching as well.
- Keep model labels horizontal when possible and rotate them no more than 45 degrees.
- Use at least 7-point text at final rendered size.
- Define every abbreviation in the caption or main text.
- Keep legends outside dense plotting regions.
- Use consistent decimal precision within a panel.

## Export Rules

- Prefer vector PDF for line art and plots.
- Embed fonts and inspect the final rendered size.
- Use tight bounding boxes without clipping labels or sample-SD bars.
- Keep final figure filenames flat and stable for submission packaging.
- Do not display local filesystem paths, repository locations, user names, run-history notes, or code-cell identifiers in reader-facing figures.
- Preserve a raster version only when the submission system explicitly requires it.

## Final Figure Checklist

Before any result figure is treated as final, confirm that:

1. the figure comes from the one verified, article-eligible completed bundle;
2. all five folds use the persistent shared assignment;
3. sample SD uses denominator four;
4. metric labels are exactly nMSE, nRMSE, nMAE, or a clearly unit-bearing physical metric;
5. raw and projected outputs refer to identical rows;
6. every model series uses its frozen accepted configuration;
7. OOD severity and in-distribution results are not pooled;
8. captions describe fold summaries as five-fold mean with sample SD;
9. every displayed value and model label has been cross-checked against its source row; and
10. no reader-facing internal path or development record remains.
