# Manuscript Citation Needs

- Manuscript: `article/wip_v3/manuscript.tex`
- Scope: Introduction through Methodology
- Status: full audit complete; 17 items resolved; no open items

## CN01

> Both optimization routes impose $8\le\mathrm{SRT}\le30$ d, $\mathrm{SLR}\le100$ kg TSS m$^{-2}$ d$^{-1}$, $X_U\le15{,}000$ g TSS m$^{-3}$, $X_N\ge1$ g TSS m$^{-3}$, and $\dot M_X/Q_0\ge1$ g TSS m$^{-3}$.

| Field | Detail |
|---|---|
| Location | Methodology, Case-study system and operating domain |
| Support type | Qualification |
| Why support is needed | The exact numerical limits may appear universal or literature-derived, but the manuscript does not identify their basis. |
| Required evidence | Either sources supporting the exact limits for this plant class or wording that identifies them as case-specific screening choices and explains their roles. |
| Effective searches | `("activated sludge" AND (SRT OR "solids loading rate" OR "return sludge concentration") AND (design OR operation))` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Henze et al. (2008), *Biological Wastewater Treatment* | Existing manuscript reference | Provides process context for solids age, recycle, and clarifier operation. | Does not establish this exact combined set of numerical limits as universal criteria. | Rejected for exact-value support |
| Leu et al. (2012) | Existing manuscript reference | Shows that solids retention time affects energy use, effluent quality, and process stability. | Does not justify the exact 8--30 d interval or the clarifier thresholds. | Supporting context only |

### Resolution

- Status: Resolved
- Final wording: “Both optimization routes use case-specific screening limits of ... These values define the comparison and are not proposed as universal design criteria.”
- Citation keys: None
- Manuscript action: Reframed the values as declared screening limits and explained the role of each group of limits. Removed marker `CN01`.

## CN02

> The holdout is a partition of the same accepted simulation stream as the model-development set. It is not independent confirmation. The final response definition, overflow closure, numerical procedures, and engineering eligibility rules were not specified independently of all inspected data. The holdout and scenario results therefore provide descriptive post-selection evidence.

| Field | Detail |
|---|---|
| Location | Methodology, Mechanistic generation and analysis sample |
| Support type | External citation and qualification |
| Why support is needed | The study-specific facts are internal, but the inference about post-selection evidence follows a broader reproducibility and data-leakage principle. |
| Required evidence | A methodological source explaining why analysis choices made after data inspection limit confirmatory interpretation. |
| Effective searches | `("data leakage" OR "post-selection inference") AND (model evaluation OR reproducibility) AND science` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Kaufman et al. (2012), “Leakage in data mining” | https://doi.org/10.1145/2382577.2382579 | Defines leakage through unavailable information and recommends separation of learning from prediction assessment. | Addresses predictive data analysis broadly rather than this activated sludge application. | Recommended |

### Resolution

- Status: Resolved
- Final wording: “Separating model development from prediction assessment reduces leakage of information into reported predictive performance.”
- Citation keys: `Kaufman2012`
- Selected reference: Kaufman, S., Rosset, S., Perlich, C., Stitelman, O., 2012. Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data* 6, 1--21.
- Manuscript action: Added the citation, retained the study-specific qualification, and removed marker `CN02`.

## CN03

> Out-of-fold predictions are used for projection assessment and trust calibration. Final fits are used only for the holdout, optimization, and selected-decision evaluations.

| Field | Detail |
|---|---|
| Location | Methodology, Model fitting, projection assessment, and trust calibration |
| Support type | External citation |
| Why support is needed | The passage adopts out-of-fold assessment to separate calibration from in-sample fitted values without naming the statistical principle. |
| Required evidence | An authoritative source on cross-validation, out-of-sample prediction, and model selection without training-set reuse. |
| Effective searches | `("out-of-fold predictions" OR "cross-validation predictions") AND (model selection OR calibration OR assessment)` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Hastie et al. (2009), *The Elements of Statistical Learning* | Existing manuscript reference | Gives the standard cross-validation framework for model selection and out-of-sample error assessment. | Does not prescribe the four trust diagnostics used in this study. | Recommended |

### Resolution

- Status: Resolved
- Final wording: “Out-of-fold predictions are used for projection assessment and trust calibration so that these calculations do not reuse fitted values from the same rows.”
- Citation keys: `Hastie2009`
- Manuscript action: Reused the existing cross-validation reference and removed marker `CN03`.

## CN04

> Response derivatives are obtained from the active-set Karush--Kuhn--Tucker (KKT) system when rank, conditioning, complementarity, and derivative checks pass. Sequential least-squares programming (SLSQP) then performs the outer search.

| Field | Detail |
|---|---|
| Location | Methodology, Operating searches and convergence assessment |
| Support type | External citation and citation-fit check |
| Why support is needed | The active-set sensitivity step and named SLSQP algorithm are established methods and should be tied to their methodological sources. |
| Required evidence | A source for parametric quadratic-program sensitivity through a fixed active set and the original or authoritative SLSQP description. |
| Effective searches | `("parametric quadratic programming" AND sensitivity AND "active set")`; `("sequential least squares programming" OR SLSQP) AND Kraft` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Tøndel et al. (2003) | Existing manuscript reference | Establishes the active-set and piecewise structure of multiparametric quadratic-program solutions. | The manuscript adds its own conditioning, complementarity, and derivative audits. | Recommended for active-set sensitivity |
| Kraft (1988), *A software package for sequential quadratic programming* | https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html | The SciPy SLSQP documentation identifies Kraft's report as the algorithm source. | Technical report rather than a journal article. | Recommended for SLSQP |

### Resolution

- Status: Resolved
- Final wording: “Response derivatives are obtained from the active-set Karush--Kuhn--Tucker system ... Sequential least-squares programming then performs the outer search.”
- Citation keys: `Tondel2003`, `Kraft1988`
- Selected reference: Kraft, D., 1988. *A Software Package for Sequential Quadratic Programming*. Technical Report DFVLR-FB 88-28, German Aerospace Center, Cologne.
- Manuscript action: Reused the active-set reference, added the SLSQP source, and removed marker `CN04`.

## CN05

> If the derivative chain cannot be validated, Constrained Optimization BY Quadratic Approximations (COBYQA) continues with function values from the same audited projection.

| Field | Detail |
|---|---|
| Location | Methodology, Operating searches and convergence assessment |
| Support type | External citation |
| Why support is needed | COBYQA is a named derivative-free trust-region method, but no algorithm reference is attached. |
| Required evidence | A primary or authoritative source describing COBYQA and its model-based derivative-free use. |
| Effective searches | `COBYQA derivative-free optimization Ragonneau Zhang` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Ragonneau (2022), *Model-Based Derivative-Free Optimization Methods and Software* | https://theses.lib.polyu.edu.hk/handle/200/12294 | Official COBYQA documentation recommends the thesis and describes COBYQA as a derivative-free trust-region SQP method using quadratic interpolation models. | Doctoral thesis rather than a journal article. | Recommended |
| Ragonneau and Zhang (2024), PDFO | https://doi.org/10.1007/s12532-024-00257-9 | Reviews Powell's derivative-free trust-region solvers and model-based optimization software. | PDFO does not itself document COBYQA in detail. | Rejected as indirect |

### Resolution

- Status: Resolved
- Final wording: “If the derivative chain cannot be validated, Constrained Optimization BY Quadratic Approximations continues with function values from the same audited projection.”
- Citation keys: `Ragonneau2022`
- Selected reference: Ragonneau, T.M., 2022. *Model-Based Derivative-Free Optimization Methods and Software*. Ph.D. thesis, The Hong Kong Polytechnic University.
- Manuscript action: Added the official recommended algorithm source and removed marker `CN05`.

## CN06

> These finite polls provide evidence only for the tested directions and resolutions. They do not establish stationarity or local optimality.

| Field | Detail |
|---|---|
| Location | Methodology, Operating searches and convergence assessment |
| Support type | External citation and qualification |
| Why support is needed | This is a general claim about what finite direct-search polls can certify. |
| Required evidence | Convergence theory showing the additional asymptotic or direction-density conditions required for stationarity results in directional direct search. |
| Effective searches | `("mesh adaptive direct search" OR "directional direct search") AND convergence AND stationarity` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Audet and Dennis (2006), “Mesh Adaptive Direct Search Algorithms for Constrained Optimization” | https://doi.org/10.1137/040603371 | Shows that stationarity results rely on limiting behavior and asymptotically dense refining directions, while finitely many polling directions give weaker conclusions. | The study's custom polls are not an implementation of the full MADS algorithm. | Recommended |

### Resolution

- Status: Resolved
- Final wording: “They do not establish stationarity or local optimality because direct-search stationarity results require limiting behavior and suitable refining directions.”
- Citation keys: `AudetDennis2006`
- Selected reference: Audet, C., Dennis Jr., J.E., 2006. Mesh adaptive direct search algorithms for constrained optimization. *SIAM Journal on Optimization* 17, 188--217.
- Manuscript action: Narrowed the claim to the relevant convergence conditions, added the citation, and removed marker `CN06`.

## CN07

> The direct route uses the Interior Point OPTimizer (IPOPT) with three smoothing stages and a full-state endpoint audit.

| Field | Detail |
|---|---|
| Location | Methodology, Operating searches and convergence assessment |
| Support type | External citation |
| Why support is needed | IPOPT is a named large-scale nonlinear programming method and requires its standard algorithm citation. |
| Required evidence | The primary IPOPT algorithm paper and a source supporting smooth nonlinear-program continuation if claimed as general practice. |
| Effective searches | `IPOPT interior point filter line search Wachter Biegler 2006` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Wächter and Biegler (2006), “On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming” | https://doi.org/10.1007/s10107-004-0559-y | Primary algorithm paper for the interior-point filter line-search method implemented in IPOPT. | Does not prescribe this study's three smoothing stages. | Recommended for IPOPT |
| Buzzi-Ferraris and Manenti (2013) | Existing manuscript reference | Supports derivative-based nonlinear programming and smooth formulations. | Does not define the study's continuation schedule. | Supporting context |

### Resolution

- Status: Resolved
- Final wording: “The direct route uses the Interior Point OPTimizer. The study applies three smoothing stages and a full-state endpoint audit.”
- Citation keys: `WachterBiegler2006`
- Selected reference: Wächter, A., Biegler, L.T., 2006. On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. *Mathematical Programming* 106, 25--57.
- Manuscript action: Attached the source only to IPOPT, separated the study-specific continuation choice, and removed marker `CN07`.

## CN08

> Native objectives cannot by themselves separate decision quality from response error or route-specific normalization. Each available selected candidate $k\in\{S,M\}$ is therefore replayed from two initial states on the nonsmooth ten-layer mechanistic model.

| Field | Detail |
|---|---|
| Location | Methodology, Exact replay and paired decision comparison |
| Support type | External citation and internal derivation |
| Why support is needed | The objective decomposition follows from this study's definitions, while the need to verify surrogate-selected decisions on a higher-fidelity model is a broader surrogate-optimization practice. |
| Required evidence | An internal pointer to the objective definitions and a review or primary source recommending high-fidelity verification of surrogate optimization decisions. |
| Effective searches | `("surrogate-based optimization" AND (validation OR verification) AND "high-fidelity model")` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Bhosekar and Ierapetritou (2018), “Advances in surrogate based modeling, feasibility analysis, and optimization” | https://doi.org/10.1016/j.compchemeng.2017.09.017 | Reviews surrogate validation, feasibility analysis, and model management in optimization. | Does not specify the two-start replay or the exact objective decomposition used here. | Recommended with internal equation pointer |

### Resolution

- Status: Resolved
- Final wording: “Native objective values mix decision quality with response error and route-specific normalization, as shown by Equation (comparisons) ... This higher-fidelity check follows the validation role used in surrogate-based optimization.”
- Citation keys: `Bhosekar2018`
- Manuscript action: Added the internal equation pointer, reused the existing review, and removed marker `CN08`.

## CN09

> A solver success flag is insufficient for acceptance. The applicable numerical, physical, engineering, stability, and branch checks must also pass.

| Field | Detail |
|---|---|
| Location | Methodology, Influent scenarios, timing, and failure accounting |
| Support type | Qualification |
| Why support is needed | The first sentence reads as a universal statement, while the listed checks are the study's declared acceptance policy. |
| Required evidence | Rewrite as a direct study rule and connect it to the previously defined audits; no external citation is required unless a broader claim is retained. |
| Effective searches | `("nonlinear optimization" AND termination AND feasibility AND verification)` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Buzzi-Ferraris and Manenti (2013) | Existing manuscript reference | Discusses nonlinear optimization stopping and feasibility conditions. | Cannot support the study-specific physical, stability, and branch checks. | Rejected for the complete claim |

### Resolution

- Status: Resolved
- Final wording: “This study does not accept a candidate from the solver success flag alone. The candidate must also pass the applicable numerical, physical, engineering, stability, and branch checks.”
- Citation keys: None
- Manuscript action: Recast the passage as the study's acceptance rule and removed marker `CN09`.

## CN10

> Scientific admission requires finite and audited out-of-fold projections for all model-development states. It also requires raw normalized RMSE below one for the complete response and the clarifier-inventory coordinate.

| Field | Detail |
|---|---|
| Location | Methodology, Model fitting, projection assessment, and trust calibration |
| Support type | Qualification |
| Why support is needed | The normalized-RMSE cutoff is a declared admission rule, but the passage does not explain its scale or whether it is universal. |
| Required evidence | Explain that one represents one model-development response-scale unit and state that the cutoff is study-specific. |
| Effective searches | `("normalized RMSE" AND threshold AND "surrogate model")` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| None required | -- | The threshold follows the manuscript's normalization definition. | No external source can make the chosen cutoff universal. | Resolve internally |

### Resolution

- Status: Resolved
- Final wording: “Because each error is divided by its model-development response scale, this cutoff requires aggregate error below one response-scale unit. It is a study-specific admission rule rather than a universal accuracy threshold.”
- Citation keys: None
- Manuscript action: Explained the normalized scale, qualified the threshold, and removed marker `CN10`.

## CN11

> Correction, recovery, and reactor limits are the out-of-fold 95th percentiles from the model-development set. The leverage limit is the maximum model-development leverage under the final feature map.

| Field | Detail |
|---|---|
| Location | Methodology, Model fitting, projection assessment, and trust calibration |
| Support type | Qualification |
| Why support is needed | The 95th percentile and maximum leverage are empirical guardrails, but the passage could be mistaken for statistical confidence limits. |
| Required evidence | State the coverage purpose and clarify that the limits are screening rules rather than confidence bounds. |
| Effective searches | `("applicability domain" AND percentile AND leverage AND surrogate)` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Sahigara et al. (2012) | Existing manuscript reference | Supports feature leverage as an applicability-domain diagnostic. | Does not prescribe a 95th-percentile threshold for the other diagnostics. | Supporting context only |

### Resolution

- Status: Resolved
- Final wording: “These quantiles retain the range observed for most development rows while screening the upper tail ... These limits are empirical guardrails rather than probabilistic confidence bounds.”
- Citation keys: None
- Manuscript action: Explained the empirical coverage purpose, distinguished the limits from confidence bounds, and removed marker `CN11`.

## CN12

> For papermaking wastewater treatment, \citet{He2023} used a Kriging surrogate to support optimization of treatment performance and greenhouse gas emissions at lower computational cost than mechanistic simulation.

| Field | Detail |
|---|---|
| Location | Introduction, surrogate modeling examples |
| Support type | Citation-fit check and qualification |
| Why support is needed | The accessible record confirms a Kriging surrogate for low-carbon process optimization but does not expose evidence for the specific lower-cost comparison with mechanistic simulation. |
| Required evidence | Direct evidence of the computational comparison or narrower wording limited to the verified application. |
| Effective searches | `("papermaking wastewater" AND Kriging AND optimization AND computational cost)` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| He et al. (2023) | https://doi.org/10.1016/j.jclepro.2023.139039 | Verified title, authors, venue, and use of a Kriging surrogate for papermaking wastewater treatment optimization. | Accessible metadata did not verify the claimed direct computational-cost comparison. | Retain with narrower wording |

### Resolution

- Status: Resolved
- Final wording: “For papermaking wastewater treatment, \citet{He2023} used a Kriging surrogate to optimize treatment performance and greenhouse gas emissions.”
- Citation keys: `He2023`
- Manuscript action: Removed the unverified direct computational-cost comparison, retained the verified application, and removed marker `CN12`.

## CN13

> This distinction matters during optimization because the search for favorable surrogate predictions can exploit model errors \citep{Bhosekar2018}.

| Field | Detail |
|---|---|
| Location | Introduction, low-concentration prediction risk |
| Support type | External citation and citation-fit check |
| Why support is needed | The statement describes optimizer behavior more specifically than the attached broad review citation. |
| Required evidence | A direct model-management source showing why approximation quality must be controlled during optimization. |
| Effective searches | `("approximation models" AND optimization AND "trust region" AND management)` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Alexandrov et al. (1998), “A trust-region framework for managing the use of approximation models in optimization” | https://doi.org/10.1007/BF01197433 | Develops explicit model management for optimization with approximation models. | Structural optimization context rather than wastewater treatment. | Recommended |
| Bhosekar and Ierapetritou (2018) | https://doi.org/10.1016/j.compchemeng.2017.09.017 | Reviews surrogate modeling, feasibility analysis, and optimization in chemical engineering. | Broader than the specific optimizer-error mechanism. | Supporting |

### Resolution

- Status: Resolved
- Final wording: “Approximation quality must also be managed during the optimization search and checked against the reference model.”
- Citation keys: `Alexandrov1998`, `Bhosekar2018`
- Selected reference: Alexandrov, N.M., Dennis Jr., J.E., Lewis, R.M., Torczon, V., 1998. A trust-region framework for managing the use of approximation models in optimization. *Structural Optimization* 15, 16--23.
- Manuscript action: Replaced the loose exploitation claim with a model-management statement, added the direct source, and removed marker `CN13`.

## CN14

> A penalty on equation residuals encourages physical agreement but does not enforce exact conservation by construction \citep{Beucler2021}.

| Field | Detail |
|---|---|
| Location | Introduction, hard and soft physical constraints |
| Support type | Citation-fit check and internal logic |
| Why support is needed | The source presents analytic hard constraints, while the sentence contrasts them with residual penalties. The distinction should be stated directly and without implying that the source evaluated every penalty method. |
| Required evidence | A hard-constraint source plus wording limited to the mathematical difference between penalizing and imposing an equality. |
| Effective searches | `("analytic constraints" AND neural networks AND conservation AND hard constraint)` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Beucler et al. (2021) | https://doi.org/10.1103/PhysRevLett.126.098302 | Presents analytic constraints embedded in neural networks that emulate physical systems. | Does not establish a universal empirical failure of all penalty methods. | Recommended with narrower wording |

### Resolution

- Status: Resolved
- Final wording: “A residual penalty can reduce violations, but exact conservation at a feasible prediction requires conservation to be imposed as an equality. Analytic output constraints provide one way to impose such relations by construction.”
- Citation keys: `Beucler2021`
- Manuscript action: Limited the claim to the mathematical distinction between a penalty and an imposed equality and removed marker `CN14`.

## CN15

> The cited studies address complementary parts of the plant-wide problem. The Invariant-Constrained Second-Order Regression (ICSOR) formulation of \citet{Selerio2026} applies to the response of a single continuous stirred-tank reactor (CSTR). It does not represent the mixer, reactor train, secondary clarifier, and recycle streams as one coupled response. In contrast, \citet{He2023} demonstrates surrogate-assisted optimization of a wastewater treatment process, but does not establish hard output-space reconciliation of all connected component responses. The gap addressed here is the integration of these two capabilities.

| Field | Detail |
|---|---|
| Location | Introduction, literature gap |
| Support type | Qualification and citation-fit check |
| Why support is needed | The gap statement can be read as an exhaustive field-wide novelty claim although it is supported by a defined set of cited studies. |
| Required evidence | Limit the claim to the reviewed literature and state the two missing capabilities concretely. |
| Effective searches | `("plant-wide" AND surrogate AND wastewater AND conservation AND optimization)` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Selerio (2026) | Existing manuscript reference | Defines ICSOR for a single reactor response. | Does not cover the plant-wide extension. | Recommended for first capability |
| He et al. (2023) | https://doi.org/10.1016/j.jclepro.2023.139039 | Demonstrates surrogate-assisted wastewater treatment optimization. | Does not provide the hard system-wide output projection formulated here. | Recommended for second capability |

### Resolution

- Status: Resolved
- Final wording: “Within the studies reviewed here, two relevant capabilities remain separate ... This study combines plant-wide output reconciliation with surrogate-based operating optimization.”
- Citation keys: `Selerio2026`, `He2023`
- Manuscript action: Limited the novelty statement to the reviewed literature, named the two capabilities, and removed marker `CN15`.

## CN16

> The assessment remains descriptive because the final analysis protocol was not specified independently of all inspected data.

| Field | Detail |
|---|---|
| Location | Introduction, study objective and evidence scope |
| Support type | External citation and qualification |
| Why support is needed | The study-specific disclosure is internal, but the restriction on confirmatory interpretation follows the same development-assessment separation discussed later in Methodology. |
| Required evidence | Reuse the verified leakage reference and align the Introduction with the Methodology wording. |
| Effective searches | `("learn-predict separation" AND leakage AND model assessment)` |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| Kaufman et al. (2012) | https://doi.org/10.1145/2382577.2382579 | Defines information leakage and recommends separation of learning from prediction assessment. | General predictive-modeling source rather than an activated sludge study. | Recommended |

### Resolution

- Status: Resolved
- Final wording: “The assessment is described as post-selection evidence because the final analysis protocol was not specified independently of all inspected data.”
- Citation keys: `Kaufman2012`
- Manuscript action: Aligned the Introduction with the Methodology evidence-scope statement and removed marker `CN16`.

## CN17

> For the ordered endpoints $X_E\le X_U$, the interval is necessary and sufficient for the existence of at least one profile satisfying the endpoint envelope and having inventory $M_{\rm cl}$. Every intermediate inventory can be obtained by assigning a common admissible value to the internal layers.

| Field | Detail |
|---|---|
| Location | Theory and calculation, Clarifier outlet balances and aggregate inventory |
| Support type | Internal derivation |
| Why support is needed | The claim is a mathematical result of the stated bounds, but its necessity and sufficiency are asserted without the short constructive proof. |
| Required evidence | Add the endpoint constructions for the lower and upper inventories and the continuous interpolation argument for intermediate inventories. |
| Effective searches | Not applicable |
| Status | Resolved |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|
| None required | -- | The result follows directly from Equations~(layer endpoints)--(inventory envelope). | An external citation would not replace the manuscript's own proof. | Resolve internally |

### Resolution

- Status: Resolved
- Final wording: “The lower bound is attained when every internal layer is assigned $X_E$, and the upper bound is attained when every internal layer is assigned $X_U$. Assigning all internal layers a common value between these endpoints changes the inventory continuously between the two bounds.”
- Citation keys: None
- Manuscript action: Added the constructive necessity-and-sufficiency proof and removed marker `CN17`.
