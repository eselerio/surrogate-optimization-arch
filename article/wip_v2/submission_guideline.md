# Submission and reproducibility guide

## Article identity

Use the title **Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate** exactly. Use “Clarifier” for the process unit; “settling” and “separation” may describe physical actions but are not equipment names.

The case study fixes five completely mixed biological reactors and a ten-layer Clarifier. The five optimized controls are nominal HRT, aeration, MLR ratio, RAS ratio, and WAS ratio. Do not imply that the number of stages is optimized.

## Mechanistic data

Generate three independent 25-dimensional blocks in the fixed order `(H,a,r_I,r_R,w,x_1,...,x_20)`:

- full: 14,000-point development LHS with seed 42, 2,000-point iid calibration block with seed 43, and 4,000-point iid assessment block with seed 44;
- `test_2000`: 1,400/200/400 rows with seeds 200042/200043/200044; and
- unit: 420/60/120 rows with seeds 60042/60043/60044.

The iid blocks use row-major open-unit SplitMix64 draws. Profile streams are disjoint; verification rows never enter the full-study record. Persist generator state and draw count for every block.

Solve each row by the fixed exact ASM2d-TSN--Clarifier BDF route. Check positivity, scaled steady residual, external and internal balances, hydraulic identities, Clarifier closure, finite rates, and reduced-Jacobian stability. Never replace a failed row or initialize it from a preceding row. Persist checkpoints after 4, 16, 64, and 256 rows before continuing.

## Fixed statistical surrogate

Fit the single 351-feature, 170-response standardized quadratic OLS model using development rows only. Population means and standard deviations, feature scales, state scales, QR factors, coefficients, objective scales, mechanistic row scales, and smoothing scales are development-only quantities. Require full column rank, the declared condition-number limit, finite coefficients, and independent pivoted-QR/SVD agreement. There is no feature selection, regularization search, candidate-model comparison, or production refit.

Calculate calibration scores

\[
d_i=\frac1{170}\|D_\chi^{-1}(\chi_i-\widehat\chi_i)\|_2^2
\]

and use the one-based index `min(n_C, ceil((n_C+1)*0.95))`. Require a finite radius `0 < delta <= 1`. The split-conformal statement is marginal under the iid sampling model before streams are fixed; it is not conditional on the realized radius, assessment passage, or optimization.

Evaluate the frozen model once on assessment. Require finite predictions, complete-state standardized RMSE below one, and empirical `d <= delta` coverage of at least 0.90. Other coordinate, block, and composite metrics are descriptive. Assessment must not alter the estimator or any downstream setting.

## Combined physics-constrained statistical NLP

Optimize five normalized controls and 110 internally scaled reactor-and-Clarifier states. Reconstruct the complete 170-coordinate response from these physical states, the controls, and the case influent. Minimize the engineering objective `J` alone; fidelity is a hard constraint and is never an objective penalty.

Impose 100 smooth reactor and ten smooth Clarifier steady-state equalities, true nonnegative state bounds, bounded controls, two positive-solids guards, five engineering inequalities, `d/delta <= 1`, and development leverage no larger than the maximum development leverage. The NLP therefore has 115 variables, 110 equalities, and nine explicit inequalities plus variable bounds. Use the manuscript’s smooth positive parts, safe biochemical divisions, exact-feasible-domain Clarifier reciprocal, smooth min/max functions, and compact C2 receiver-threshold transition. Preserve the original nonsmoothed kinetics and Clarifier in data generation and exact validation.

Use true IPOPT variable bounds with `bound_relax_factor=0`. Solve nine independent starts: the center followed by the eight rows of the exact five-dimensional LHS generated with seed 271828. At each control seed, initialize `y` from the nearest accepted development state in frozen standardized 25-dimensional input distance. Apply the declared initialization-only coordinate floor and smallest-index tie rule. Do not warm-start across starts or cases.

CasADi supplies exact derivatives. IPOPT uses MUMPS, adaptive barrier updates, the manuscript tolerances, and at most 2,500 iterations per start. This cap bounds an unproductive tail without reducing the nine-start design; reaching it fails that start and does not relax acceptance. The mechanistic least-squares polish retains its separate limit of 5,000 residual evaluations. Accept only `Solve_Succeeded` or `Solved_To_Acceptable_Level` together with the independent primal, stationarity, dual, complementarity, nonnegativity, and bound replay. Select the smallest accepted engineering objective with the declared numerical and lexicographic tie rules. Report one multistart local candidate, never a global optimum.

Commit each completed NLP start atomically under its profile, case, ordered-start, and scientific-contract identity. A resume may reuse it only after verifying identity, contract hash, payload hash, dimensions, and finite values. Apply the same atomic, verified-cache rule to the selected candidate's exact replay. Maintain an invocation ledger with logical identity, attempt number, start/completion status, and reuse status so planned logical work, realized logical results, physical attempts, interrupted attempts, and verified reuses remain distinguishable.

## Exact validation

Run the exact fixed-start BDF route once at the selected combined-NLP controls. Do not initialize BDF from the NLP state and do not substitute a second-best start after selection.

Validation requires exact residual and stability acceptance, both domain guards, all five engineering inequalities, `d_BDF/delta - 1 <= 1e-8`, and scaled infinity-norm branch agreement no larger than `1e-3` between the exact and selected smooth state. Evaluate and retain every downstream check for every finite exact state even when an earlier check fails. Report the first failure in the declared order as the one mutually exclusive classification. The combined-NLP point and this validation outcome are the single decision route.

## Cases, workload, and resources

The full study contains the nominal case, 100 robustness influents, two additional underflow-TSS cases, and ten objective-weight cases: 113 cases total. It attempts 1,017 combined-NLP starts and at most 20,113 exact BDF routes. The `test_2000` profile has 23 cases, 207 starts, and at most 2,023 BDF routes. The unit profile has 14 cases, 126 starts, and at most 614 BDF routes. These are planned logical maxima for uninterrupted completion. An interrupted numerical call whose result was not atomically committed may be attempted again, so report actual physical attempts from the invocation ledger as well as these logical counts.

Use the first 256 design rows as the empirical BDF timing preflight. The combined-NLP panel contains the nominal and all sensitivity cases plus up to ten robustness profiles selected by the specified span-normalized greedy maximin rule: 23 cases for `test_2000` and full, and 14 for unit. Time all nine combined starts and reuse only verified atomic results. The projection always uses the declared full-study workload. Apply the 1.5 timing safety factor, label projections as empirical, and enforce the 30-core-day and 25-GiB gates in unit, `test_2000`, and full; a verification-profile pass does not waive the independently recomputed full-profile gate.

## Reporting and completion

Retain every start, KKT replay, failure status, selected state, exact validation, objective component, inequality value, duration, stage resident-memory high-water mark, checkpoint-verification result, invocation-ledger entry, and source/config/dependency hash. Record exposed numerical-library versions and hash a loaded binary when its version is unavailable. Time development, calibration, and assessment generation separately. Report robustness failures separately from continuous summaries. Include the single outcome table, TSS and objective-weight sensitivity tables, control/constraint activity, reactor profiles, Clarifier layers and streams, fidelity diagnostics, and branch agreement. Define every component and composite outlet difference as signed `NLP - BDF`; positive values mean the smooth NLP state is larger. Define `scaled_residual_inf` as the largest exact scaled dynamic-balance residual across all reactor-component and Clarifier-layer rows, not as a kinetic-only residual.

All reports generated before terminal replay are provisional and unsealed. A run is complete only after terminal replay reconstructs predictions and the calibration order statistic; replays constraints and KKT diagnostics; reloads every finite stored exact state; independently recomputes all residual, stability, domain, engineering, fidelity, branch, objective, and reporting checks even when an earlier check fails; reconstructs table inputs; verifies checkpoint and artifact hashes; and writes the immutable completion seal. Only that passed replay and seal release validated results. Never describe an interrupted or verification-profile run as the full article result.
