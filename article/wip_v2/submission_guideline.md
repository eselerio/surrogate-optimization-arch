# Submission Guideline

**Article title:** Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate

This internal checklist governs the manuscript, figures, tables, declarations, and source-data package for the five-CSTR Mixer-Reactor-Clarifier activated-sludge optimization study. The article must be self-contained: every equation, parameter, numerical convention, fitting rule, acceptance test, search budget, and validation definition needed to reproduce the study must appear in the main text or its integral appendices.

## Article-Eligibility Gate

Populate submission assets only from one complete scientific run that passes every terminal numerical and physical assertion. The accepted record must contain:

- all 20,000 prescribed mechanistic design rows and their diagnostics;
- the 16,000-row development fit and 4,000-row assessment;
- the production refit on all 20,000 rows;
- nominal optimization and its cold incumbent replay;
- all 100 robustness cases;
- mechanistic evaluations at every reported surrogate-selected decision;
- nominal and robustness mechanistic reference searches;
- physical, predictive, decision, and timing diagnostics; and
- a complete source-data inventory with stable identifiers and checksums.

Reject the reporting record when an expected row, process location, component, Clarifier layer, scenario, coefficient, solver record, or paired mechanistic result is absent. Never combine scales, coefficients, simulations, or summaries from different runs; infer missing values from rounded prose; omit a failed scenario; or describe an incomplete finite-budget search as a final optimum.

## Fixed Mixer-Reactor-Clarifier Contract

The submitted article concerns one prescribed topology:

- fresh flow \(Q_0=10{,}000\) m\(^3\) d\(^{-1}\);
- five equal-volume CSTRs in series;
- CSTRs 1--2 unaerated and CSTRs 3--5 sharing one aerobic intensity;
- MLR withdrawn from CSTR 5 before clarification and returned to the inlet mixer;
- a non-reactive, ten-layer one-dimensional flux-limited secondary Clarifier;
- Clarifier underflow split hydraulically into RAS and WAS with common state \(c_U\); and
- Clarifier overflow state \(c_E\) used for the effluent-quality objective.

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

The Clarifier has area 1500 m\(^2\), total depth 4.0 m, and ten equal 0.4-m, 600-m\(^3\) layers. Feed enters layer 5 from the top (layer 6 from the bottom; zero-based top-to-bottom index 4). The fixed parameters are \(v_0^{\max}=250\) m d\(^{-1}\), \(v_0=474\) m d\(^{-1}\), \(r_h=0.000576\) m\(^3\) g\(^{-1}\), \(r_p=0.00286\) m\(^3\) g\(^{-1}\), \(f_{ns}=0.00228\), and \(X_t=3000\) g TSS m\(^{-3}\). The blanket diagnostic threshold is also 3000 g TSS m\(^{-3}\).

The primary upper-level engineering limits are SRT in \([8,30]\) d, surface overflow no greater than 20 m d\(^{-1}\), feed solids loading no greater than 100 kg TSS m\(^{-2}\) d\(^{-1}\), and underflow TSS no greater than 15,000 g TSS m\(^{-3}\). Underflow-cap sensitivity uses 12,000 and 20,000 g TSS m\(^{-3}\) without altering the production surrogate.

Every row and candidate must satisfy

\[
q_P=1+r_I+r_R,\quad
q_C=1+r_R,\quad
q_U=r_R+w,\quad
q_E=1-w,\quad
q_C=q_E+q_U,
\]

with every divisor strictly positive. A change to the topology, layer count, return location, aeration zoning, flow definitions, or operating domain defines a different study and requires a newly generated scientific run.

## Mechanistic Generator Contract

ASM2d with two-step nitrification supplies 28 process rates and 20 component states in the fixed order stated in the article. Couple those reactions to the ten-layer Takacs-type Clarifier exactly as defined in the article.

Construct the invariant operator from the unrounded 28-by-20 stoichiometric matrix using the stated SVD threshold. Require numerical rank 15, hence five invariant rows, and verify

\[
\|A\nu^\top\|_\infty\leq10^{-10},\qquad
\|AA^\top-I_5\|_\infty\leq10^{-10},\qquad
\|Ae_{S_O}\|_\infty\leq10^{-10}
\]

before generating or scoring any state.

The dynamic state is

\[
y=(c_1,\ldots,c_5,s_1,\ldots,s_{10})\in\mathbb R_+^{110}.
\]

At every residual evaluation, reconstruct particulate overflow and underflow from the Clarifier-feed proportions and endpoint TSS, pass soluble components with the water, and reconstruct the inlet mixer from the current MLR and RAS states. The mixer, five reactors, Clarifier, MLR, RAS, and WAS withdrawal must therefore be solved as one closed loop.

Because the Clarifier is non-reactive,

\[
g_E+g_U=q_Cc_5,\qquad
g_E=q_Ec_E,\qquad
g_U=q_Uc_U.
\]

Solubles follow the liquid phase; particulate components share the stored TSS settling flux while retaining the Clarifier-feed particulate proportions. Densification must emerge from recovery into the smaller underflow and must never be imposed by creating particulate mass.

### BDF-First Steady-State Route

Use the two prescribed initial states and execute the full route for Start 1 before trying Start 2. For each start:

1. Scale every reactor coordinate by \(\max(1,x_j^U)\) and every layer coordinate by the start-specific \(\max[1,t_X^\top c_5(y_0)]\).
2. Integrate the scaled coupled dynamics with variable-order BDF.
3. Set the relaxation horizon to

   \[
   T_{\mathrm{rel}}=\max(400,40/w)\ \mathrm d
   \]

   and the adaptive maximum step to \(T_{\mathrm{rel}}/100\). This horizon is an upper limit; terminate early when the scaled derivative infinity norm reaches \(10^{-9}\) d\(^{-1}\).
4. Use BDF relative tolerance \(10^{-7}\) and scaled-coordinate absolute tolerance \(10^{-9}\).
5. Check every stored physical state. If the direct scaled-coordinate route encounters a non-positive state or internal failure, retry that start in log-scaled coordinates so positivity is structural rather than clipped.
6. Replay the relaxed endpoint against the complete acceptance contract. Only an endpoint that misses acceptance may enter a non-negative bounded trust-region-reflective least-squares polish with \(\texttt{xtol}=\texttt{ftol}=\texttt{gtol}=10^{-9}\) and at most 5000 residual evaluations.
7. Accept the branch only after the complete residual, non-negativity, physical-closure, and local-stability replay passes.

Start 1 fixes the represented branch whenever its complete route succeeds. Start 2 is attempted only if that complete route fails. Do not initialize a design row from a preceding row, clip a negative state into acceptance, or cap a reaction rate, settling speed, or flux after evaluation.

Every accepted mechanistic state must pass:

- scaled CSTR and Clarifier-layer derivative maxima no greater than \(10^{-8}\) d\(^{-1}\);
- minimum state coordinate no smaller than \(-10^{-10}\);
- largest real reduced-Jacobian eigenvalue no greater than \(10^{-8}\) d\(^{-1}\);
- component Clarifier and plant-boundary residuals no greater than \(10^{-8}\) under the article's termwise scaling;
- hydraulic split and RAS/WAS common-composition checks;
- soluble pass-through, particulate-recovery, Clarifier-layer-envelope, and finite-rate checks; and
- the exact flow and external-boundary identities.

Archive the initialization route, BDF status, positivity retry, optional polish, solver effort, residuals, stability result, and wall time for every attempted state.

## Mechanistic Design and Stored Targets

Generate one strength-1 Latin hypercube containing exactly 20,000 points in the 25-dimensional box ordered as

\[
(H,a,r_I,r_R,w,x_1,\ldots,x_{20}).
\]

Use the article's fully specified SplitMix64 stream, unbiased Fisher--Yates permutations, root state 42, dimension order, row order, and affine range maps. Every one of the 20,000 points must return an accepted state under the fixed solver route. A failed point is not replaced; an unresolved point invalidates the scientific run.

For each accepted row, retain the 170-coordinate target

\[
\chi_{\mathrm{ASM}}=(m,c_1,\ldots,c_5,g_E,g_U,s_1,\ldots,s_{10}).
\]

Also retain controls, influent, flow ratios, recovered \(c_E\) and \(c_U\), full component states of all Clarifier layers, composite quantities, reaction and oxygen-transfer rates, recoveries, densification, loadings, whole-plant SRT, acceptance residuals, stability diagnostics, route status, and solve time. Clarifier solids inventory is derived from the ten layer-TSS targets.

Rows 1--16,000 are the development set. Rows 16,001--20,000 are the untouched predictive-assessment set. Membership is fixed before inspecting responses. No assessment, nominal, or robustness response may alter centers, scales, feature definitions, estimator rules, acceptance limits, objective weights, operating bounds, or numerical tolerances.

## Fixed Statistical Surrogate Contract

The surrogate input contains five controls and 20 influent components. Standardize these 25 inputs from the applicable fitting rows, form all unique second-order terms, and standardize the 350 nonconstant terms again. With one leading constant, the ordered feature vector contains

\[
1+5+20+\frac{5(6)}2+\frac{20(21)}2+5(20)=351
\]

columns. Serialize within-vector quadratic pairs lexicographically with \(j\leq k\), and serialize control--influent products with the influent index varying fastest.

The response contains 170 standardized targets: eight 20-coordinate process blocks \((m,c_1,\ldots,c_5,g_E,g_U)\) and ten Clarifier layer-TSS coordinates. The fitted matrix therefore has dimensions

\[
\widehat C\in\mathbb R^{170\times351}.
\]

For every scalar control, influent, nonconstant feature, and target coordinate in a fitting set \(\mathcal I\), require

\[
s_{\mathcal I}(v)>10^{-12}\max\left(1,\max_{i\in\mathcal I}|v_i|\right).
\]

Fit all 170 outputs simultaneously by the single standardized OLS problem

\[
\widehat C_{\mathcal I}
=\arg\min_C\frac{1}{n(170)}
\left\|\widetilde Y_{\mathcal I}-\Phi_{\mathcal I}C^\top\right\|_F^2.
\]

Use column-pivoted Householder QR for the coefficient calculation and one iterative-refinement replay in the same factorization. Independently calculate the thin SVD of the standardized design. Require full column rank under

\[
\tau_{\mathrm{rank}}=\max(n,351)\varepsilon_{\mathrm{mach}}\sigma_1,
\qquad \sigma_{351}>\tau_{\mathrm{rank}},
\]

and require \(\kappa_2(\Phi)\leq10^8\). Check both the scaled normal-equation residual and QR--SVD coefficient agreement against

\[
100\,\kappa_2(\Phi)\varepsilon_{\mathrm{mach}}.
\]

All centers, scales, balance scales, coefficients, and diagnostics used for the development response must come only from rows 1--16,000. Evaluate rows 16,001--20,000 once against the fixed development response and the predeclared predictive, numerical, and correction-reliance gates. A failed gate invalidates the study; assessment responses do not modify the estimator.

After the assessment record is sealed, refit the unchanged equations on all 20,000 rows. Recompute every center, scale, balance scale, QR factorization, SVD check, and coefficient. Freeze this production response before generating any nominal or robustness mechanistic response.

## Unique Physical-Projection Contract

For each fixed \((\vartheta,x)\), calculate

\[
\widetilde\chi_{\mathrm{raw}}=\widehat C\phi(\vartheta,x),
\qquad
\chi_{\mathrm{raw}}=\mu_\chi+D_\chi\widetilde\chi_{\mathrm{raw}}.
\]

Build the candidate-specific 77-row equality operator in this fixed order:

- 20 mixer component balances;
- 25 reactor-invariant balances, five for each CSTR;
- 20 Clarifier component mass balances;
- 10 soluble-pass-through balances; and
- 2 Clarifier endpoint-TSS balances.

Build the fixed-order 26-row inequality operator from ten particulate underflow-densification directions and sixteen top-to-bottom Clarifier layer-envelope directions. Add componentwise non-negativity for all 170 state coordinates.

The equality-only affine projection is an assessment diagnostic. Deployment solves the direct scaled QP

\[
\begin{aligned}
\widehat u={}&\arg\min_u\ \tfrac12\|u\|_2^2\\
\text{subject to }&
D_b^{-1}\{\mathcal H(\vartheta)[\chi_{\mathrm{raw}}+D_\chi u]-b(x)\}=0,\\
&\chi_{\mathrm{raw}}+D_\chi u\geq0,\\
&D_g^{-1}\mathcal G(\vartheta)[\chi_{\mathrm{raw}}+D_\chi u]\leq0,
\end{aligned}
\]

and returns \(\widehat\chi=\chi_{\mathrm{raw}}+D_\chi\widehat u\). The identity Hessian and the explicit no-conversion feasible point make the deployed state unique for each candidate.

Compute \(D_b\) and \(D_g\) from the RMS magnitudes of the named signed physical terms over the applicable fitting rows, with the declared positive floors. Do not scale a balance with nearly zero accepted residuals.

Use double precision, absolute and relative QP tolerances \(10^{-8}\), polishing, and at most 100,000 iterations. Verify full row rank of the scaled equality matrix at every candidate. Independently replay the KKT conditions and require

\[
r_E,r_G,r_{\mathrm{stat}},r_{\mathrm{dual}},r_{\mathrm{comp}}\leq10^{-8},
\qquad r_+\leq10^{-10}.
\]

Every DIRECT, face, pattern, assessment, production, timing, and incumbent-replay QP starts with zero primal and dual arrays. One newly constructed cold retry is permitted after a failed acceptance replay. Failure after that retry invalidates the candidate. Never score a clipped raw state, affine diagnostic, influent state, or preceding solution as a fallback.

Recompute the plant-boundary identity, Clarifier balances, layer inventory, and recovered concentrations from the unrounded deployed state. The projection guarantees the stated linear network, sign, densification-direction, and layer-envelope conditions; it does not guarantee the 28 nonlinear reaction equations or the nonlinear Clarifier flux law.

## Upper Optimization Contract

Use Clarifier overflow composites only in the quality term. With equal COD, TN, TP, and TSS weights, the primary objective is

\[
\begin{aligned}
J={}&0.50\,\alpha_Q^\top D_T^{-1}I_{\mathrm{comp}}c_E
+0.15\frac{H-H^L}{H^U-H^L}
+0.20a\\
&+0.05\frac{r_I-r_I^L}{r_I^U-r_I^L}
+0.05\frac{r_R-r_R^L}{r_R^U-r_R^L}
+0.05\frac{w\,t_X^\top c_U}{w^UX_U^{\max}},
\end{aligned}
\]

where \(\alpha_Q=\tfrac14\mathbf1_4\). The final term represents normalized wasted-solids mass, not the WAS ratio alone. Obtain \(D_T\) from the four production-fit overflow-composite standard deviations and freeze all coefficients and normalizations before optimization.

Every accepted upper candidate must pass the five scaled engineering rows in the fixed process contract--two SRT bounds, surface overflow, feed solids loading, and underflow TSS--and these production-response trust screens:

- scaled correction magnitude \(\kappa_{\mathrm{corr}}\leq0.50\);
- production-design leverage no greater than its maximum design-row value;
- particulate recovery spread \(\Delta_\eta\leq0.05\); and
- normalized nonlinear Clarifier flux residual \(r_{\mathrm{flux}}\leq0.05\).

These are upper-level acceptance conditions. Do not move them into the physical-correction QP to manufacture a feasible state.

Compute whole-plant SRT from all five reactor inventories and all ten Clarifier-layer inventories divided by solids leaving through overflow and WAS:

\[
\mathrm{SRT}=
\frac{\sum_{i=1}^{5}V_it_X^\top c_i+
\sum_{\ell=1}^{10}V_{\mathrm{cl},\ell}s_\ell}
{Q_0(q_Et_X^\top c_E+wt_X^\top c_U)}.
\]

Reject an invalid near-zero denominator; do not assign an artificial infinite SRT.

### Surrogate Search Budget

Normalize all five controls to \([0,1]^5\). Give each influent at most 25,000 distinct attempted surrogate candidates:

- at most 18,000 globally new keys from the full-box DIRECT partition, including its center;
- all 32 corners as explicit boundary probes;
- at most 300 globally new keys on each of the ten four-dimensional coordinate faces; and
- at least 3,968 attempts, plus unused DIRECT quota, for up to four separated pattern-search basins.

Use \(\epsilon_{\mathrm{DIR}}=10^{-4}\), the article's deterministic tie handling, and maximum-side resolution \(1/1024\). Pattern searches begin with mesh \(1/16\), poll axial directions before lexicographically ordered pairwise directions, and halve the mesh through \(1/512\). Failed and infeasible distinct candidates consume budget; cached duplicates do not. Cold-recompute the final incumbent.

Report the best verified incumbent at the achieved resolution and budget. If no feasible candidate exists, report `no feasible incumbent` and the smallest violation, but do not define selected controls or finite regret.

## Mechanistic Validation and Regret

For the nominal influent and every robustness influent, solve the complete coupled mechanistic model at the frozen surrogate-selected controls without changing the surrogate, objective, constraints, or tolerances.

Distinguish:

- surrogate optimization failure;
- mechanistic evaluation failure;
- mechanistic decision-feasibility failure; and
- accepted, engineering-feasible selected decision.

Only the fourth class has ordinary finite regret. Insert the accepted selected-point mechanistic evaluation into the reference archive first and count it against the budget.

The nominal mechanistic reference receives 10,000 distinct attempts. Before DIRECT, evaluate the selected point and its clipped, deduplicated 50-direction stencil at step \(2^{-7}\), followed by uncached corners. Allocate at most 7,000 globally new keys to full-box DIRECT, at most 100 to each coordinate face, and at least 1,917 to local multi-basin refinement.

Each robustness reference receives 2,500 distinct attempts. Use the same selected-point/stencil/corner order, at most 1,700 full-box DIRECT keys, at most 25 keys per face, one local seed, and at least 467 local-refinement attempts.

All failed mechanistic attempts consume budget. With \(\widehat\vartheta_s\) the surrogate-selected decision and \(\vartheta_{\mathrm{ASM},s}^*\) the best verified mechanistic incumbent,

\[
\mathcal R_s=
J[\widehat\vartheta_s,\chi_{\mathrm{ASM}}(\widehat\vartheta_s;x_s)]
-J[\vartheta_{\mathrm{ASM},s}^*,
\chi_{\mathrm{ASM}}(\vartheta_{\mathrm{ASM},s}^*;x_s)].
\]

Because the selected point is inserted first, regret must be non-negative up to roundoff. A value below \(-10^{-8}\) indicates inconsistent objectives, constraints, or archive handling and invalidates the case. Never clip regret to zero. Describe it as a finite-budget incumbent gap, not a global-optimality certificate.

Generate exactly 100 robustness influents only after the nominal problem, production response, objective, scales, and numerical rules are frozen. Use the same stated Latin-hypercube construction restricted to the 20 influent dimensions, restarting the 64-bit state at 314159.

## Metrics and Reporting

Report predictive behavior separately for the eight 20-coordinate blocks \(m,c_1,\ldots,c_5,g_E,g_U\) and the ten-coordinate Clarifier layer profile. Use fitting-only positive target scales for standardized errors. Do not pool concentration, normalized mass-flow, and layer-TSS coordinates under an unqualified physical-unit metric.

Required predictive summaries include:

- target nRMSE, nMAE, bias, and \(R^2\) by block and overall;
- per-coordinate physical-unit errors within each block;
- Clarifier overflow and underflow COD, TN, TP, and TSS errors;
- raw, affine-diagnostic, and deployed errors;
- physical-correction displacement and reliance;
- component, total-particulate, and TSS recovery and densification errors;
- Clarifier layer-profile and derived-inventory errors; and
- predicted-versus-mechanistic objective error at the selected decision.

Required decision summaries include:

- selected \(H,a,r_I,r_R,w\);
- objective decomposition;
- normalized operating displacement;
- finite-budget regret when defined;
- surrogate and mechanistic attempt counts;
- achieved resolution and termination classification;
- control-bound activity; and
- engineering and trust-screen activity or rejection frequency.

Required physical diagnostics include distributions and maxima of:

- mixer closure;
- each reactor invariant;
- Clarifier component balance and soluble pass-through;
- Clarifier endpoint and layer-envelope relations;
- plant-boundary invariants;
- minimum raw, affine-diagnostic, deployed, and mechanistic coordinates;
- particulate densification-direction slack;
- QP equality, inequality, stationarity, dual-feasibility, and complementarity residuals; and
- mechanistic CSTR, layer, hydraulic, recycle-closure, stability, and finite-rate checks.

Across the 100 robustness scenarios report mean, median, interquartile range, 95th percentile, maximum, failure count, and relevant bound-activity fractions. These are descriptive finite-sample summaries, not confidence intervals or significance evidence.

Derive COD, TN, TP, and TSS only after the relevant 20-component state is available. Always identify whether a composite belongs to CSTR 5, Clarifier overflow \(c_E\), or Clarifier underflow \(c_U\).

## Timing Contract

Use a monotonic high-resolution counter with one numerical-library thread. Keep these durations separate:

- mechanistic design generation;
- 16,000-row development fit;
- 4,000-row assessment deployment;
- 20,000-row production fit and replay;
- raw 170-response inference;
- equality/inequality matrix construction;
- physical-correction QP setup and solve;
- end-to-end deployed inference;
- surrogate optimization;
- selected-point mechanistic validation; and
- independent mechanistic reference search.

Single-state inference uses 100 untimed warmups followed by 1000 timed evaluations of a fixed seeded sequence. Batch inference uses two untimed batches followed by 20 timed batches of 1000 candidates. Report cold and warm QP setup and solve separately, together with medians, interquartile ranges, and per-state batch throughput. Retain processor, logical-core count, installed RAM, operating system, language and numerical-package versions, thread limits, and solver versions. Do not relabel installed RAM as peak process memory.

The article's computational feasibility gate uses the first 256 immutable design points for the mechanistic timing preflight and 1000 evenly spaced development rows for cold QP timing. The projected scientific workloads are

\[
N_{\mathrm{QP}}^{\mathrm{sci}}=2{,}549{,}101,
\qquad
N_{\mathrm{mech}}^{\mathrm{sci}}=280{,}000.
\]

Proceed only if the declared 30-core-day and 51.2-GiB projected limits and every preflight acceptance check pass. Timing observations must not change scientific sample counts, tolerances, fitting rules, or search budgets.

## Required Article Assets

Static methods tables must match the accepted contract:

- topology, flows, controls, operating domains, and upper screens;
- fresh-influent domain and 20-component order;
- ASM2d-TSN parameters and the complete stoichiometric construction;
- ten-layer Clarifier geometry, flux law, and numerical conventions;
- BDF-first mechanistic route and acceptance criteria;
- 351-feature order and 170-target order;
- OLS, QR, SVD, rank, conditioning, and coefficient checks;
- 77 equality rows, 26 physical-direction rows, and QP acceptance rules; and
- surrogate and mechanistic search budgets.

Result tables must be regenerated together from the accepted run:

- mechanistic design acceptance, solver routes, and residual validation;
- 4,000-row assessment by process block and prediction form;
- QR/SVD diagnostics and production-response checks;
- nominal selected controls and objective decomposition;
- nominal deployed and mechanistic reactor, overflow, underflow, and layer summaries;
- nominal recovery, densification, loadings, SRT, trust screens, and physical diagnostics;
- robustness prediction and decision summaries;
- regret, displacement, failures, search effort, and bound activity; and
- timing results.

Required figure roles include:

- the five-CSTR Mixer-Reactor-Clarifier topology with MLR, RAS, and WAS;
- the BDF-first mechanistic generation route;
- the fixed OLS and unique physical-projection workflow;
- an intuitive Clarifier densification explanation;
- nominal axial and Clarifier responses;
- predictive fidelity and physical residuals; and
- robustness decisions, search effort, and mechanistic regret.

All figures and tables require complete machine-readable source data. The manuscript and its integral appendices must contain the full methodological specification; an external document may provide additional row-level data but must not be necessary to understand or reproduce the method.

## Generation and Finalization Order

1. Freeze the five-CSTR topology, ten-layer Clarifier, component order, parameters, domains, seeds, and tolerances.
2. Verify stoichiometric invariants, flow identities, Clarifier equations, feature/target order, and physical-projection rank contracts.
3. Run the 256-point mechanistic preflight and evaluate the declared feasibility gate.
4. Generate all 20,000 immutable mechanistic design points with the BDF-first solver route.
5. Freeze rows 1--16,000 as development and rows 16,001--20,000 as assessment.
6. Fit the development OLS by pivoted QR and complete the independent SVD checks.
7. Evaluate the 4,000 assessment rows once and seal the pass--fail record.
8. Refit and freeze the unchanged production response on all 20,000 rows.
9. Freeze the nominal objective, upper constraints, trust screens, and search settings.
10. Solve and mechanistically validate the nominal case.
11. Generate the 100 seed-314159 robustness influents.
12. Complete every surrogate optimization, selected-point mechanistic solve, and mechanistic reference search.
13. Recompute physical, predictive, decision, and timing summaries from row-level records.
14. Verify expected cardinalities, pairings, solver routes, acceptance assertions, inventory, and checksums.
15. Build the submission from a clean staged copy and visually inspect every page, equation, table, and figure.

## Final Scientific Checks

- The title is exactly **Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate**.
- The topology and every stream direction match the Mixer-Reactor-Clarifier formulation.
- All normalized flow identities close at every row and candidate.
- MLR bypasses the Clarifier; RAS and WAS share the underflow composition.
- The non-reactive Clarifier conserves every component.
- Solubles follow the water; particulate densification comes from mass recovery.
- Every mechanistic state follows the prescribed BDF-first route and passes the complete replay.
- The target has 170 fixed coordinates and the feature vector has 351 fixed columns.
- The 16,000/4,000 assessment separation is immutable.
- Both the development and production designs pass the QR/SVD numerical contract.
- Every accepted deployed state passes equality, non-negativity, physical-direction, layer-envelope, and KKT checks.
- Upper engineering and trust conditions remain outside the physical projection.
- Every selected decision is independently simulated by the complete mechanistic model.
- Every regret pair uses identical objectives, bounds, and engineering constraints.
- Finite-budget results are called verified incumbents, not certified global optima.
- Every displayed value resolves to a complete source row from the accepted scientific run.
- Reader-facing text contains no local paths, run-control labels, cell numbers, or dependence on an external methods narrative.

## Submission Package

Prepare a clean, flat submission package containing only:

- manuscript source and its integral appendices;
- required class and style files;
- bibliography files actually used;
- final referenced figures;
- declaration, highlights, and cover letter; and
- machine-readable source tables and any row-level data archives explicitly submitted with the article.

Exclude internal guides, exploratory notebooks, checkpoints, unreferenced diagnostics, caches, temporary files, local run logs, and machine-specific paths. Compile from the staged package, verify cross-references and embedded fonts, inspect the Mixer-Reactor-Clarifier topology and recycle arrows visually, and confirm that all submission metadata use the exact final title and author information.
