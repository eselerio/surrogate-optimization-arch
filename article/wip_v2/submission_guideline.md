# Submission Guideline

This internal checklist governs the manuscript, Supplementary Material, figures, tables, and source-data package for the five-CSTR closed-loop activated-sludge optimization study.

## Article-Eligibility Gate

Populate submission assets only from one completed full-profile run that has passed every terminal assertion and whose manifest, registered file inventory, and cryptographic hashes verify. The accepted run must contain the complete mechanistic dataset, assessment and production ICSOR artifacts, nominal optimization, all robustness cases, selected-point mechanistic evaluations, mechanistic reference searches, physical diagnostics, and timings.

The article finalizer must fail closed when any expected row, component, process location, scenario, coefficient block, solver record, or paired mechanistic result is absent. Never:

- promote smoke-profile output;
- report a partial robustness cohort;
- combine coefficients, scales, simulations, or summaries from different runs;
- infer a missing value from rounded manuscript text;
- silently remove a failed scenario; or
- describe an incomplete finite-budget search as a final optimum.

## Fixed Process Contract

The submitted article concerns one prescribed topology:

- fresh flow \(Q_0=10{,}000\) m\(^3\) d\(^{-1}\);
- five equal-volume CSTRs in series;
- CSTRs 1--2 unaerated and CSTRs 3--5 sharing one aerobic intensity;
- MLR withdrawn from CSTR 5 before clarification and returned to the inlet mixer;
- a non-reactive, ten-layer one-dimensional flux-limited secondary settler;
- settler underflow split hydraulically into RAS and WAS with common state \(c_U\); and
- treated overflow state \(c_E\) used for the effluent-quality objective.

The optimized control vector and primary domain are:

| Control | Domain |
|---|---:|
| Total fresh-flow HRT \(H\) | \([6,36]\) h |
| Shared aerobic setting \(a\) | \([0,1]\) |
| MLR ratio \(r_I=Q_I/Q_0\) | \([0,4]\) |
| RAS ratio \(r_R=Q_R/Q_0\) | \([0.25,1.25]\) |
| WAS ratio \(w=Q_W/Q_0\) | \([0.001,0.05]\) |

The total HRT is divided equally among tanks. With

\[
q_P=1+r_I+r_R,
\]

the actual per-pass HRT in each CSTR is \(H/(5q_P)\). CSTRs 3--5 use \(k_La=47a\) d\(^{-1}\); CSTRs 1--2 use \(k_La=0\). Dissolved-oxygen saturation is 8.5 g O\(_2\) m\(^{-3}\).

The settler contract is: area 1500 m\(^2\), total depth 4.0 m, ten equal 0.4-m/600-m\(^3\) layers, feed at the sixth layer from the bottom (fifth from the top; zero-based top-to-bottom array index 4), \(v_0^{\max}=250\) m d\(^{-1}\), \(v_0=474\) m d\(^{-1}\), \(r_h=0.000576\) m\(^3\) g\(^{-1}\), \(r_p=0.00286\) m\(^3\) g\(^{-1}\), \(f_{ns}=0.00228\), and both flux-limiting and blanket-diagnostic thresholds equal to 3000 g TSS m\(^{-3}\). The case screens are surface overflow at most 20 m d\(^{-1}\), feed solids loading at most 100 kg TSS m\(^{-2}\) d\(^{-1}\), and underflow TSS at most 15,000 g TSS m\(^{-3}\). Treat the first as a nonbinding safety screen at fixed throughput and repeat the underflow-TSS analysis at 12,000 and 20,000 g TSS m\(^{-3}\).

The normalized network flows must satisfy

\[
q_P=1+r_I+r_R,\quad
q_C=1+r_R,\quad
q_U=r_R+w,\quad
q_E=1-w,\quad
q_C=q_E+q_U,
\]

with every divisor strictly positive. Any change to the topology, layer count, return location, zone aeration, flow definition, or operating domain creates a different study contract and requires a newly generated full run.

## Mechanistic Generator Contract

ASM2d with two-step nitrification supplies 28 process rates and 20 component states in the fixed manuscript order. It is coupled to the versioned ten-layer Takacs-type clarification--thickening model. Settler area, depth, feed layer, layer volumes, settling and compression parameters, flux-limitation convention, underflow-solids capacity, and all numerical settings must be stored at full precision.

Construct the invariant basis from the unrounded 28-by-20 stoichiometric matrix using the declared SVD rank threshold. Require numerical rank 15, hence \(K=5\) and \(A\in\mathbb R^{5\times20}\), and verify \(A\nu^\top\), \(AA^\top-I_5\), and \(Ae_{S_O}\) against the \(10^{-10}\) limits before generating or scoring any state.

The complete mixer, five reactors, ten settler layers, MLR loop, RAS loop, and WAS withdrawal are solved as one closed-loop steady problem. The settler is non-reactive for this study. Consequently, separator component flow is conserved exactly:

\[
g_E+g_U=q_Cc_5,
\qquad
g_E=q_Ec_E,
\qquad
g_U=q_Uc_U.
\]

Soluble coordinates follow the water phase. Particulate coordinates follow the stored hindered- and compression-settling flux model. RAS densification is a consequence of particulate recovery into the smaller underflow; it is never implemented as particulate generation.

Every retained mechanistic state must pass, using unrounded arrays:

- mixer component balance;
- all five CSTR component-balance residual limits;
- all ten settler-layer balance limits;
- MLR, RAS, WAS, overflow, and total hydraulic closure;
- separator component mass balance;
- soluble transport consistency;
- underflow TSS no smaller than separator-feed TSS;
- plant-boundary invariant balance;
- non-negativity and declared state bounds;
- finite process rates and settling fluxes; and
- steady-solve acceptance under the versioned tolerance contract.

Use the two starts, renewed solve, and at-most-100-d BDF route in the Supplementary Material. The numerical contract is steady-solve \(\texttt{xtol}=\texttt{ftol}=\texttt{gtol}=10^{-9}\), 5000 maximum residual evaluations, BDF \(\texttt{rtol}=10^{-8}\), \(\texttt{atol}=10^{-10}\), scaled balance and terminal-derivative maxima of \(10^{-8}\), minimum coordinate \(-10^{-10}\), and maximum real reduced-Jacobian eigenvalue \(10^{-8}\) d\(^{-1}\). Record which start and route succeeded. State or initialization bounds are not reaction-rate or settling-flux clipping. Do not silently cap a computed process rate or flux after evaluation.

## Dataset and Information Boundaries

One scrambled strength-1 Latin hypercube with root seed 42 jointly samples 20 fresh-influent components and the five controls. Sampling continues up to 500,000 candidates until exactly 20,000 accepted closed-loop steady states are retained; reaching the cap first invalidates the full run.

Each accepted row must store the 161-coordinate target

\[
\chi_{\mathrm{ASM}}=(m,c_1,c_2,c_3,c_4,c_5,g_E,g_U,\overline M_{\mathrm{cl}}),
\]

as well as:

- all controls and derived flow ratios;
- actual per-pass HRT;
- \(c_E\) and \(c_U\);
- all ten settler-layer states;
- COD, TN, TP, and TSS calculated from component states;
- component, total-particulate, and TSS recoveries and densification;
- hydraulic loading, solids loading, underflow TSS, and SRT;
- reactor oxygen-transfer rates;
- all mechanistic residuals and non-negativity diagnostics;
- candidate and attempt identifiers; and
- solver route, status, iterations, and wall time.

Rejected attempts remain archived. Candidate identifiers, attempt records, and retained rows must link one-to-one. Parallel completion order must not change the deterministic accepted design.

The accepted rows are reproducibly permuted and split into 16,000 coefficient-estimation rows and 4,000 untouched predictive-test rows. No nominal-case or robustness response may enter:

- feature or target scaling;
- physical-residual scaling;
- hyperparameter selection;
- stopping-rule selection;
- objective or penalty selection;
- operating-domain adjustment; or
- solver-tolerance tuning.

After the 4,000-row assessment has been completed without revision, one production artifact is refit on all 20,000 accepted rows. It is frozen before the nominal or robustness mechanistic responses are generated.

## System-Level ICSOR Contract

The implemented input contains the five controls and 20 influent components. Its ordered second-order feature map has 551 columns. The system response has eight 20-component blocks plus the normalized clarifier solids inventory and therefore 161 coordinates.

The coupling matrix is block diagonal:

\[
\Gamma=\operatorname{blkdiag}
(\Gamma_m,\Gamma_1,\ldots,\Gamma_5,\Gamma_E,\Gamma_U,0_{1\times1}),
\]

where every block is 20-by-20 with a zero diagonal and selected off-diagonal bound. Cross-unit physical connection is supplied by the network operator, not an undeclared dense coupling block.

The frozen dimensions are \(B\in\mathbb R^{161\times551}\) and \(\Gamma,D_\chi\in\mathbb R^{161\times161}\). The 75-by-75 balance scale \(D_b\) must use training-only characteristic balance-term magnitudes with positive predeclared floors; accepted near-zero residuals must not be used as scales. After each box-constrained convex \(\Gamma\) update, apply the tested \(2^{-k}\), \(k=0,\ldots,24\), contraction sequence; if it is exhausted, retain the source behavior \(\Gamma=0\) and record the conditioning fallback.

The recursive estimator inherits the tested \(B\rightarrow\Gamma\rightarrow\widehat X_\chi\) update structure but is a new network fit. It must not reuse the single-CSTR coefficients or hyperparameters. The fitted-state update uses the sample-dependent equality operator \(\mathcal H(\vartheta_i)\), its right-hand side \(b(x_i)\), the flow-dependent underflow-TSS row, and non-negativity.

Reuse the training routines from `icsor-model/src/models/ml/icsor_coupled_qp.py`; record the exact source commit and file hash. Retain its feature order `[vartheta, x, 1, vartheta_otimes_vartheta, x_otimes_x, vartheta_otimes_x]`, `solved`/`solved inaccurate` status acceptance, zero-Gamma conditioning fallback, non-negative-target fitted-state fallback, and final-completed-sweep freeze. The network adaptation changes only the documented dimensions, block mask, physical operator/inequality, mixed-coordinate scaling, and L2 deployment. Before accepting a network artifact, run the adapted routines on the original 20-output fixture and verify compatible objective histories and returned blocks within the declared tolerances. Archive every training fallback count; the deployment QP remains fail-closed.

Hyperparameters are selected only within five fixed seeded folds of the 16,000 estimation rows. Complete 100 seeded tree-structured Parzen estimator trials without pruning, test-response access, optimization-validation response access, or a result-dependent timeout. Use the fixed \(0.70E_{\mathrm{blk}}+0.10E_M+0.20E_E\) score and the search domains and numerical settings in the Supplementary Material.

The selection score and rejection checks must use the frozen contract. Retain:

- all five objective terms by sweep;
- running-best and final objective histories;
- solver statuses, residuals, iterations, warm starts, and fallbacks;
- selected hyperparameters;
- symmetry checks for ordered quadratic blocks;
- coupling bounds, masks, and condition numbers;
- fitted-state physical diagnostics; and
- complete source-code and environment provenance.

The 4,000-row predictive test is evaluated once. A failed test or artifact check cannot be repaired by selecting another trial after seeing the test responses.

## Unique Deployment Contract

For each fixed \((\vartheta,x)\), calculate the raw state by solving

\[
(I_{161}-\widehat\Gamma)\chi_{\mathrm{raw}}
=\widehat B\phi(\vartheta,x).
\]

Build the candidate-specific equality operator from:

- exact mixer component closure;
- five reactor invariant blocks;
- separator component mass balance; and
- ten soluble-pass-through rows.

Use the fixed 75-row order declared in the Supplementary Material and verify full row rank with an unrounded rank-revealing factorization at every candidate. Do not delete or reorder rows candidate by candidate, and do not replace the network operator with the single-CSTR invariant matrix.

The affine state is the equality projection in the same training-scaled metric used by the final QP. The final deployed response is the unique scaled-\(L_2\) solution satisfying the same equalities, componentwise non-negativity, and

\[
t_X^\top g_U\geq q_Ut_X^\top c_5.
\]

The positive coordinate scales must come only from the applicable fitting targets. Raw, affine, and deployed arrays remain distinct in every artifact and table.

Deployment uses the manuscript's strict numerical contract: \(10^{-8}\) absolute and relative QP tolerances, polishing, one warm attempt, and at most one newly constructed cold retry. Accept a lower response only when the optimal solver status and independent checks pass for:

- \(\|\mathcal H\widehat\chi-b\|_\infty\leq10^{-8}\);
- minimum coordinate at least \(-10^{-10}\);
- stationarity residual at most \(10^{-8}\);
- dual-feasibility residual within tolerance;
- complementarity residual at most \(10^{-8}\); and
- underflow-TSS slack at least \(-10^{-8}\).

Also recompute separator component balance and the plant-boundary identity

\[
Ax=A g_E+\frac{w}{q_U}A g_U
=q_EAc_E+wAc_U.
\]

Failure after the cold retry invalidates the upper candidate. Never score the influent, a clipped raw point, an affine point, or a preceding solution as a fallback.

## Upper Optimization Contract

The nominal objective uses overflow composites only:

\[
\begin{aligned}
J={}&0.50\left(\tfrac14\mathbf 1_4^\top D_T^{-1}I_{\mathrm{comp}}c_E\right)
+0.15\,H_{\mathrm{norm}}+0.20a\\
&+0.05(r_I)_{\mathrm{norm}}
+0.05(r_R)_{\mathrm{norm}}
+0.05w_{\mathrm{norm}}.
\end{aligned}
\]

The production overflow scales in \(D_T\) and all penalty coefficients are frozen before optimization responses are inspected. The quality coefficient and five operating coefficients sum to one. Sensitivity studies vary the five operating coefficients while preserving the same production surrogate.

Every accepted upper candidate must satisfy the declared:

- SRT interval \([8,30]\) d;
- clarifier surface-overflow limit;
- clarifier solids-loading limit;
- maximum pumpable underflow TSS;
- discharge limits, when activated; and
- training-domain and recycle-adjusted hydraulic checks.

These are upper-level operating or capacity constraints. Do not move them into the physical-correction QP merely to make a candidate acceptable.

Compute whole-plant SRT with clarifier inventory included:

\[
\mathrm{SRT}=
\frac{\sum_{i=1}^{5}V_i t_X^\top c_i+Q_0\overline M_{\mathrm{cl}}}
{Q_0(q_Et_X^\top c_E+wt_X^\top c_U)}.
\]

Archive the ten-layer mechanistic inventory used to verify \(\overline M_{\mathrm{cl}}\). Do not substitute a reactor-only SRT without changing the declared study contract.

Normalize the five controls to \([0,1]^5\). Each influent receives at most 25,000 distinct attempted surrogate candidates: 18,000 full-box DIRECT evaluations, all 32 corners, 300 DIRECT evaluations on each of the ten coordinate faces, and up to 3968 local pattern evaluations. Failed and infeasible candidates consume budget. Halve the local step from \(1/16\) through \(1/512\); early termination also requires relative incumbent change below \(10^{-8}\) over 500 attempted evaluations. Store achieved spatial resolution, failures, and termination reasons. Recompute the final incumbent with a cold lower solve.

Use `best verified incumbent at the reported resolution and budget`, not `global optimum`, unless a mathematical global certificate is obtained.

## Mechanistic Validation and Regret

For the nominal case and every robustness influent, solve the complete coupled mechanistic network at the frozen surrogate-selected controls. Do not refit, recalibrate, or rerun the surrogate optimization in response to this result.

Prediction validation compares matched deployed and mechanistic states at the identical influent and controls. Decision validation performs an independent mechanistic search with the same five-dimensional box, objective, upper constraints, flow definitions, and acceptance rules.

Regret is

\[
\mathcal R_s=
J[\widehat\vartheta_s,\chi_{\mathrm{ASM}}(\widehat\vartheta_s;x_s)]
-J[\vartheta_{\mathrm{ASM},s}^*,
\chi_{\mathrm{ASM}}(\vartheta_{\mathrm{ASM},s}^*;x_s)].
\]

Insert the mechanistic evaluation at the surrogate-selected controls into the reference archive first and count it toward the budget. The nominal reference receives at most 10,000 attempted candidates; each robustness reference receives 2500, including a deduplicated \(3^5\) neighborhood, and failures consume budget. If the selected point violates mechanistic capacity constraints, classify a decision-feasibility failure and do not report ordinary finite regret. Otherwise, regret below \(-10^{-6}\) triggers one prespecified additional block of at most 2500 attempts and investigation; it is never clipped to zero. The mechanistic reference remains the best resolution-qualified finite-budget incumbent, not a certified global optimum.

After all model, objective, and solver choices are frozen, generate exactly 100 robustness influents with an independent scrambled Latin hypercube and root seed 314159. No robustness response may feed back into model selection or scaling.

## Metrics and Reporting

Report predictive behavior separately for the eight process blocks \(m,c_1,\ldots,c_5,g_E,g_U\). Use training-only positive scales for component-standardized errors. Do not pool concentration and normalized mass-flow blocks under an unqualified physical-unit metric.

Required predictive summaries include:

- component nRMSE and nMAE by block and overall;
- per-component physical-unit errors within each block;
- overflow and underflow COD, TN, TP, and TSS errors;
- raw, affine, and deployed prediction errors;
- affine and QP correction displacement;
- component, total-particulate, and TSS recovery errors;
- underflow densification errors;
- normalized clarifier-inventory error; and
- predicted-versus-mechanistic objective error at the selected point.

Required decision summaries include:

- selected \(H,a,r_I,r_R,w\);
- objective decomposition;
- normalized operating displacement;
- regret;
- surrogate and mechanistic evaluation counts;
- search termination classification;
- control-bound activity; and
- upper-constraint activity or rejection frequency.

Required physical diagnostics include distributions and maxima of:

- mixer closure residual;
- each reactor invariant residual;
- separator component mass-balance residual;
- soluble-pass-through residual;
- plant-boundary invariant residual;
- minimum raw, affine, deployed, and mechanistic coordinate;
- underflow-TSS slack;
- lower-QP stationarity, dual-feasibility, and complementarity residuals; and
- mechanistic CSTR, layer, hydraulic, and recycle-closure residuals.

Across the 100 robustness scenarios report mean, median, interquartile range, 95th percentile, maximum, failure count, and relevant bound-activity fractions. These are descriptive finite-sample summaries, not confidence intervals or significance evidence.

COD, TN, TP, and TSS must be derived only after the relevant 20-component state is available. Always identify whether the composite belongs to \(c_5\), \(c_E\), or \(c_U\).

## Timing Contract

Use a monotonic high-resolution counter under fixed thread settings. Keep the following durations separate:

- mechanistic dataset generation;
- hyperparameter selection;
- 16,000-row assessment fit;
- 20,000-row production fit;
- raw coupled-system inference;
- equality-operator construction and factorization;
- affine projection;
- lower-QP setup and solution;
- end-to-end deployed inference;
- surrogate optimization;
- mechanistic selected-point validation; and
- independent mechanistic reference search.

Use one numerical-library thread. Single-state inference comprises 100 untimed warmups followed by 1000 timed evaluations of a fixed seeded sequence. Batch inference comprises two untimed batches followed by 20 timed batches of 1000 candidates. Report cold and warm lower-QP setup/solve times separately, plus medians, interquartile ranges, and per-state batch throughput. Retain processor, logical core count, installed RAM, operating system, Python and package versions, thread limits, and solver versions. Do not relabel installed RAM as peak process memory.

## Required Main-Article Assets

Static methods tables must match the accepted configuration:

- topology, flows, controls, and operating bounds;
- fresh-influent domain and component order;
- ASM2d-TSN and ten-layer-settler numerical settings;
- mechanistic acceptance criteria;
- system-level feature and 161-coordinate output contracts;
- ICSOR model-selection domains and selected settings;
- lower-QP settings and acceptance tolerances; and
- outer and mechanistic search budgets.

Result tables must be regenerated together from the accepted run:

- mechanistic dataset acceptance and residual validation;
- 4,000-row predictive assessment by process block;
- selected hyperparameters, fit convergence, and production-artifact checks;
- nominal selected controls and objective decomposition;
- nominal predicted and mechanistic reactor, overflow, and underflow summaries;
- nominal recovery, densification, loadings, SRT, and physical diagnostics;
- robustness prediction and decision summaries;
- regret, displacement, failures, and bound activity; and
- timing results.

Required figure roles are governed by `figure_style_guide.md` and include:

- the five-CSTR/MLR/RAS/WAS/ten-layer-settler topology;
- the system-level ICSOR and unique-QP workflow;
- a separator densification explanation;
- nominal axial and separator responses;
- predictive fidelity and physical residuals; and
- robustness decisions and mechanistic regret.

All figures and tables must have machine-readable source data registered in the manifest.

## Supplement Requirements

The Supplementary Material and its source data must contain:

1. the complete 20-component order, units, and fresh-influent ranges;
2. all ASM2d-TSN parameters and the full-precision stoichiometric matrix;
3. the ten-layer settler geometry, parameters, flux equations, and numerical conventions;
4. mixer, reactor, separator, flow, RAS/WAS, and plant-boundary equations;
5. the mechanistic design, acceptance, and retry protocol;
6. the exact 551-column feature contract and 161-coordinate output contract;
7. the generalized network ICSOR objective and all three block updates;
8. hyperparameter trials, selected values, convergence histories, and solver diagnostics;
9. blockwise and per-coordinate predictive metrics for raw, affine, and deployed states;
10. the candidate-specific equality operator, rank tests, lower QP, and KKT checks;
11. recovery, densification, SRT, hydraulic-loading, and solids-loading definitions;
12. nominal and 100-case row-level optimization and validation outputs;
13. candidate archives and search-termination details;
14. complete timing repetitions and summaries; and
15. scientific provenance, environment fields, manifest, and source-data hashes.

Large coefficient, candidate-history, and row-level files should be supplied as machine-readable supplementary data rather than rounded or truncated in the PDF.

## Generation and Finalization Order

1. Freeze the five-CSTR topology, ten-layer settler, component order, parameters, domains, seeds, and tolerances.
2. Verify stoichiometric invariants, flow identities, separator equations, and the candidate-specific equality-rank contract.
3. Generate or resume attempts until 20,000 accepted mechanistic states are sealed.
4. Freeze the deterministic 16,000/4,000 split and all training-only scales.
5. Complete all 100 inner hyperparameter trials and freeze the selected settings.
6. Fit the 16,000-row assessment artifact and evaluate the 4,000 untouched rows once.
7. Refit and seal the production artifact on all 20,000 rows.
8. Freeze the nominal objective, penalty weights, upper constraints, and search settings.
9. Solve and mechanistically validate the nominal case.
10. Generate the 100 seed-314159 robustness influents.
11. Complete all surrogate optimizations, selected-point mechanistic solves, and mechanistic reference searches.
12. Recompute physical, predictive, decision, and timing summaries from row-level sources.
13. Verify terminal assertions, expected cardinalities, manifest inventory, and hashes.
14. Run the external finalizer, which must consume only the sealed article-eligible bundle.
15. Build both PDFs from a clean staged copy and visually inspect every page and figure.

## Final Scientific Checks

- The topology and every stream direction match the manuscript.
- The four normalized flow identities close at every row and candidate.
- MLR bypasses the settler and RAS/WAS share the underflow composition.
- The non-reactive separator conserves every component.
- Solubles follow the water and particulate densification comes from mass recovery.
- All mechanistic CSTR, layer, hydraulic, recycle, and boundary residuals pass.
- The 161 target coordinates and 551 feature columns have fixed labels and order.
- No predictive-test, nominal, or robustness response leaked into selection or scaling.
- The production coefficients, masks, scales, and equality replay cases are immutable.
- Every accepted deployed state passes equality, non-negativity, densification, and KKT checks.
- Upper capacity and discharge constraints were not hidden inside the lower projection.
- Every selected point was independently simulated by the complete mechanistic network.
- Every regret pair uses identical objectives, bounds, and upper constraints.
- Finite-budget results are called verified incumbents, not certified global optima.
- Every displayed value resolves to a registered row from the one accepted full run.
- No smoke, partial, cross-run, provisional, local-path, or development-status text remains.

## Submission Package

Prepare a clean, flat submission package containing only:

- manuscript and Supplementary Material sources;
- required class and style files;
- bibliography files actually used;
- final referenced figures;
- declaration, highlights, and cover letter;
- machine-readable source tables cited by the article or supplement; and
- any explicitly submitted coefficient or row-level supplementary archives.

Exclude internal guides, notebooks, checkpoints, unreferenced diagnostics, raw build directories, caches, temporary files, and local run logs. Compile from the staged package, verify cross-references and fonts, inspect the topology and recycle arrows visually, and confirm that submission metadata use the final title and sole-author information consistently.
