# Results and Discussion Citation Needs

The IDs below identify the original blue `RCxx` tags audited in `article/wip_v3/manuscript.tex`. Each quoted passage preserves the exact originally selected wording. The citation insertion pass is complete: citations support 22 selected passages, and three passages were resolved as direct consequences of the manuscript's own results or arithmetic. All resolved blue tags and the unused marking macro have been removed. The resolution table below records the source and disposition of every ID; adjacent claims sometimes share a citation.

The Results/Discussion references were subsequently refreshed on 6 October 2026. The resolution table reflects the current manuscript. Fifteen citation placements now use recent evidence, with six new papers: four dated 2026 and two dated 2025. Original discovery records below are retained as history; the newer source records and replacement rationale are in `article/wip_v3/citation_audit/results_reference_refresh.md`.

The audit covers all eight results subsections, from aggregate holdout accuracy through optimization time, and stops before Conclusions. Entries identify either a need for external methodological or process support, or an opportunity for an example study to strengthen an interpretation already supported by this manuscript. “Strengthening” entries are optional contextual support, not missing evidence for the reported numerical results. Citation discovery and insertion were completed on 6 October 2026 using publisher pages, author/institutional repositories, the authors' papers, and EPA technical documentation. The original Google Scholar queries are retained for future searches.

Internal numerical results, declared procedures, algebraic consequences of the manuscript's equations, and descriptive captions were reviewed and left unmarked. No claims about model families absent from this section were added. Related entries are kept separate where they require different evidence; one well-matched source may support several entries. Mechanistic evaluation here establishes agreement with the reference model, not experimental validation of actual plant performance.

## RC01

> Lower prediction error at every coordinate is therefore not guaranteed.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Methodological support: projection enforces feasibility but need not improve each output coordinate. Aggregate improvement should not be generalized to componentwise improvement. | A constrained-output projection analysis or example showing componentwise error increases despite feasibility or lower aggregate error. Distinguish distance in the projection norm from coordinatewise accuracy; account for a constraint set that may not contain the reference response exactly. | `("output projection" OR "constrained regression") AND ("prediction error" OR accuracy) AND ("physical constraints" OR conservation)` |

## RC02

> The underflow supplies both returned biomass and wasted solids, so an error in this stream can affect recycle mixing and the solids-related objective terms.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the flow connections are defined internally, but a process study would explain why underflow prediction errors matter operationally. | Activated sludge or clarifier modeling that links underflow concentration to RAS biomass return, WAS solids discharge, and plant solids balances. The present objective terms remain supported by this manuscript's definitions. | `("activated sludge" OR "secondary clarifier") AND ("underflow concentration" OR "return activated sludge") AND ("solids balance" OR wasting)` |

## RC03

> Favorable average accuracy across the plant therefore does not establish equally strong prediction at the discharge point.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the location-specific results establish this case; literature can place the limitation of pooled metrics in a broader evaluation context. | Multioutput or wastewater prediction evaluation showing that aggregated accuracy masks errors in critical outputs or treatment locations, and recommending output-specific assessment. | `("multioutput regression" OR "wastewater prediction") AND ("aggregate error" OR "prediction accuracy") AND ("effluent quality" OR "output-specific evaluation")` |

## RC04

> Agreement near the central part of a concentration range can coexist with errors at its ends.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Methodological support: agreement of central summaries is insufficient evidence for accuracy at extremes. The sentence offers a general explanation of parity plots. | Residual analysis or surrogate evaluation across response ranges demonstrating tail errors concealed by medians or pooled fit statistics. Do not infer extrapolation solely from being at a range endpoint. | `("regression diagnostics" OR "surrogate validation") AND ("prediction error" OR residuals) AND (extremes OR tails)` |

## RC05

> A positive overflow-TSS prediction therefore does not ensure a conservative estimate of the residual concentration.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the underprediction count supports the specific finding; an external source can clarify why non-negativity and conservative prediction are distinct requirements. | A physically constrained prediction or safety-oriented surrogate study distinguishing admissibility from one-sided error control. A non-negativity guarantee alone must not be presented as an upper prediction bound. | `("physically constrained" OR "non-negative regression") AND ("surrogate model" OR prediction) AND ("conservative prediction" OR "prediction bounds")` |

## RC06

> Satisfaction of the imposed physical relations and accuracy against the mechanistic model must therefore be assessed separately.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Methodological support: this recommends separate evaluation criteria for physically constrained surrogates. | Physics-constrained surrogate research evaluating conservation or feasibility residuals separately from predictive errors, ideally documenting that constraint satisfaction does not establish kinetic or reference-model fidelity. | `("physics-constrained" OR "physics-consistent") AND (surrogate OR "machine learning") AND ("physical consistency" OR conservation) AND (accuracy OR validation)` |

## RC07

> Such errors would affect assessment against a nutrient limit near these concentrations.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the hypothetical threshold crossing follows from the reported values; an application study would substantiate its operational relevance. | Wastewater nutrient prediction or control research showing how prediction errors near effluent concentration constraints change feasibility or compliance assessment. No particular legal limit or jurisdiction is asserted here. | `("wastewater treatment" OR "activated sludge") AND ("nutrient limits" OR "effluent constraints") AND ("prediction error" OR uncertainty)` |

## RC08

> Reporting concentration and removal together exposes this difference in scale.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the calculation supports the example; comparable reporting practices would support the recommendation to use both measures. | Treatment-performance evaluation showing that high percentage removal can conceal meaningful differences in residual concentrations, with influent concentrations or the denominator made explicit. | `("wastewater treatment" OR "effluent quality") AND ("removal efficiency" OR "percentage removal") AND ("residual concentration" OR "effluent concentration")` |

## RC09

> The mechanistic evaluation is needed to establish the treatment performance used in the route comparison. This is particularly relevant when the predicted effluent approaches a prescribed treatment limit.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Methodological support: the argument extends the selected-decision discrepancies into a recommendation to check surrogate-selected controls using a higher-fidelity model, especially near constraints. | Surrogate optimization that reevaluates selected decisions against the original model and checks objective and constraint discrepancies. Wastewater examples with effluent limits would strengthen the application. Model agreement must not be equated with measured plant performance. | `("surrogate optimization" OR "surrogate-assisted optimization") AND ("high-fidelity verification" OR "constraint satisfaction") AND (wastewater OR "process optimization")` |

## RC10

> An assessment of an existing plant would need to fix its installed volumes.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Practical support: this distinguishes capacity selection from operational optimization at an installed facility. | An existing-plant optimization study that treats reactor volumes as fixed infrastructure while varying aeration, recycle, wasting, or influent flow. Clarify that capacity expansion or retrofit studies have a different scope. | `("activated sludge" OR "wastewater treatment plant") AND ("operational optimization" OR "optimal operation") AND ("fixed volume" OR "existing plant")` |

## RC11

> Identical aeration settings can still give different oxygen-transfer capacity because reactor volume depends on $H$.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the manuscript's proxy establishes the volume scaling; oxygen-transfer literature would give the interpretation a physical basis. | Oxygen-transfer formulations with volumetric transfer coefficient and reactor volume, distinguishing total transfer capacity from transfer per unit volume. Support the meaning of the setting used here rather than assume all aeration controls map identically to capacity. | `("activated sludge" OR aeration) AND ("oxygen transfer capacity" OR "volumetric mass transfer coefficient") AND ("reactor volume" OR scaling)` |

## RC12

> The waste ratio alone does not determine wasted-solids production.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the product in the manuscript establishes the calculation; a solids-management study would clarify its operational interpretation. | WAS solids mass discharge expressed as waste flow times waste-stream solids concentration. Distinguish discharged solids mass from biological solids generation or sludge yield. | `("waste activated sludge" OR "sludge wasting") AND ("solids mass flow" OR "sludge production") AND ("waste flow" OR concentration)` |

## RC13

> Whole-plant SRT additionally depends on reactor and clarifier inventories.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Domain support: inclusion of clarifier solids in whole-plant sludge age is a process definition that benefits from an authoritative source. | SRT defined using retained solids inventory divided by external solids losses, explicitly including reactor and clarifier inventories and identifying the effluent and wasting terms. | `("solids retention time" OR "sludge age") AND ("clarifier inventory" OR "clarifier solids") AND ("mass balance" OR "activated sludge")` |

## RC14

> Returned sludge and mixed liquor reintroduce material circulating within the plant.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the mixer equation supports this study; literature can contextualize the increase in mixed-stream concentrations without treating it as new external loading. | Activated sludge process balances describing RAS and mixed-liquor recycle transport and the distinction between internal circulating mass and external influent load. | `("activated sludge" OR "biological nutrient removal") AND ("return activated sludge" OR "mixed liquor recycle") AND ("mass balance" OR mixing)` |

## RC15

> Their changes across R1--R5 cannot by themselves identify nitrification, denitrification, or phosphorus uptake.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Domain and interpretation support: changes in composite totals are being distinguished from evidence of particular biological pathways. | Activated sludge modeling or process diagnostics linking nitrification to nitrogen species conversion, denitrification to nitrogen removal, and phosphorus uptake to soluble/particulate partitioning. Explain why TN/TP totals alone are insufficient to identify the listed processes. | `("activated sludge" OR "nutrient removal") AND (nitrification OR denitrification OR "phosphorus uptake") AND (speciation OR "process diagnosis")` |

## RC16

> A nearly constant total can conceal redistribution among component forms.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Domain support: the sentence interprets stable composite concentrations as potentially masking biochemical redistribution. | A nutrient or COD fractionation study showing transformation among soluble/particulate, organic/inorganic, or oxidized/reduced forms with relatively unchanged composite totals. | `("activated sludge" OR wastewater) AND (fractionation OR speciation) AND ("total nitrogen" OR "total phosphorus" OR "chemical oxygen demand")` |

## RC17

> Higher mixed-liquor TP also need not imply higher effluent TP when phosphorus leaves the clarifier with retained solids.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Domain support: this explains how phosphorus partitioning and solids separation can decouple mixed-liquor and effluent TP. | Biological phosphorus-removal or clarifier studies relating biomass-bound phosphorus, solids capture, and effluent soluble/particulate TP. Clarifier retention is an internal separation; net plant phosphorus removal requires external solids discharge. | `("biological phosphorus removal" OR "activated sludge") AND ("particulate phosphorus" OR "biomass phosphorus") AND (clarifier OR "effluent phosphorus")` |

## RC18

> The biological stages determine the soluble and particulate composition entering it.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the model structure supplies the immediate explanation, while a process study can explain the coupling between biological conversion and clarification. | Activated sludge/clarifier coupling in which reactor processes change soluble and particulate component fractions transported to the clarifier. A suitable source should support this modeled pathway without implying that every real clarifier is nonreactive. | `("activated sludge model" OR "biological treatment") AND ("soluble and particulate" OR fractionation) AND ("secondary clarifier" OR "reactor clarifier coupling")` |

## RC19

> A process diagnosis would additionally require soluble nutrient species, DO, outlet solids flows, and whole-plant SRT.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Domain support: the passage recommends a specific set of diagnostic variables beyond composite profiles. | Process monitoring or modeling that explains the roles of nutrient speciation, DO, solids discharge, and sludge age in interpreting nutrient conversion and solids retention. Several sources may be needed to support the complete list; it is not necessarily a universal minimum monitoring set. | `("activated sludge" OR "biological nutrient removal") AND (diagnostics OR monitoring) AND ("dissolved oxygen" OR "nutrient speciation") AND ("sludge age" OR "solids balance")` |

## RC20

> Solving the complete layer model supplies the nonlinear clarifier checks that the aggregate surrogate inventory cannot provide.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Methodological and domain support: the limitation of aggregate inventory is being contrasted with spatially resolved settling equations. | Layered clarifier modeling with nonlinear settling fluxes and vertical solids distributions, showing which relationships are lost in aggregate inventory descriptions. The full solution checks the selected layer model, not every possible real clarifier behavior. | `("secondary clarifier" OR "layered settling model") AND ("settling flux" OR "solids profile") AND ("aggregate model" OR "mass balance")` |

## RC21

> Actual oxygen transfer also depends on the DO deficit.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Domain support: this invokes the oxygen-transfer driving force to distinguish capacity from realized transfer. | The oxygen-transfer relation involving transfer coefficient, liquid volume, and the difference between oxygen saturation and actual DO. If actual plant transfer is discussed, identify applicable correction factors without claiming the present proxy measures them. | `("oxygen transfer rate" OR "oxygen mass transfer") AND ("dissolved oxygen deficit" OR "saturation concentration") AND ("activated sludge" OR aeration)` |

## RC22

> Response approximation, surrogate trust restrictions, and local search behavior can still affect the selected controls,

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Methodological support: the passage proposes three plausible mechanisms for route differences, rather than experimentally isolating them. | Surrogate optimization sources showing approximation-induced decision shifts, restrictions from trust/applicability domains, and sensitivity of local nonlinear optimization to starts or local solutions. Sources support plausibility only, not attribution of this study's gaps to a particular mechanism. | `("surrogate optimization" OR "approximation models") AND ("trust region" OR "applicability domain") AND ("local minima" OR "decision quality")` |

## RC23

> Its reduction does not directly quantify monetary savings or measured energy consumption.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: this limitation follows from the declared objective; an engineering assessment study can clarify the extra modeling needed to infer money or energy from normalized proxies. | Wastewater optimization distinguishing dimensionless weighted resource indicators from calibrated energy consumption and operating costs, ideally explaining equipment efficiencies and cost conversion. | `("wastewater optimization" OR "activated sludge optimization") AND ("energy consumption" OR "operating cost") AND (proxy OR "normalized objective")` |

## RC24

> Individual effluent concentrations still require assessment when prescribed treatment targets govern operation.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Practical and methodological support: a weighted objective can trade among outcomes and does not itself guarantee that every treatment target is met. | Wastewater optimization that imposes individual effluent quality constraints separately from weighted quality/resource objectives. A study should demonstrate the distinction between optimizing a score and satisfying each prescribed limit. | `("wastewater treatment" OR "activated sludge") AND ("effluent constraints" OR "effluent limits") AND ("weighted objective" OR "multiobjective optimization")` |

## RC25

> They do not establish equal decision quality or a reduction in the time of the complete calculation.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Strengthening: the reported timing scope and objectives establish the study-specific limitation; a benchmarking source would support the broader evaluation principle. | Surrogate optimization comparisons that report both reference-evaluated decision quality and total computational cost, including data generation, fitting, optimization, and verification. If repeated use is considered, account explicitly for amortization of development costs. | `("surrogate-assisted optimization" OR "surrogate optimization") AND ("computational cost" OR "offline cost") AND (benchmarking OR "solution quality")` |

## Citation insertion resolutions

All 25 entries are addressed. There are 19 `\citep` placements supporting 22 selected passages. The current Results/Discussion section uses 13 distinct reference keys after the recency refresh. Prose and numerical results were preserved. Citations provide process or methodological support; they do not attribute this manuscript's numerical results to external sources. The support descriptions below distinguish source content from inferences made using the present equations or results.

| ID | Disposition and inserted keys | Evidence location and fit |
|---|---|---|
| RC01 | Cited: `Selerio2026,SturmSilva2025` | Selerio reports an accuracy tradeoff with physical guarantees; Sturm–Silva Section 3.2 shows species-level degradation after unweighted correction. |
| RC02 | Cited: `Zhang2026` | Secondary-treatment methods define RAS/WAS connections and component balances. Sludge-disposal costing uses solids mass flow. Error propagation into this objective is an inference here. |
| RC03 | Resolved internally; no external citation inserted | The immediately reported overflow error and the location-specific figure establish the contrast with pooled accuracy. A general reference would not strengthen this numerical inference. |
| RC04 | Cited: `Shahbazi2026` | Introduction and benchmark experiments distinguish frequent target ranges from poorly predicted rare/extreme ranges. Supports the possibility, not a diagnosis of imbalance here. |
| RC05 | Resolved internally; no external citation inserted | The 1,008 positive but underpredicted overflow-TSS values directly refute a conservative upper-estimate guarantee. No separate example is needed. |
| RC06 | Cited: `Selerio2026,Valente2025` | Both report predictive accuracy and physical compliance separately; Selerio also reports a tradeoff. |
| RC07 | Cited for application context: `Ren2026,Zhang2026` | Ren calibrates nutrient-limit safeguards; Zhang evaluates TN/TP and species concentrations against design targets. The present threshold crossing is an internal inference. |
| RC08 | Resolved internally; no external citation inserted | The stated removal equation and S1 concentration/removal calculations demonstrate the scale difference. A source reporting both metrics would add no evidence to this arithmetic. |
| RC09 | Cited: `Hameed2026,Ren2026` | Hameed checks original black-box feasibility during surrogate search; Ren evaluates nutrient-safe scheduling in BSM2 under disturbances. Neither experimentally validates the present decisions. |
| RC10 | Cited: `Zhang2026` | Secondary-treatment methods specify constant user-defined reactor volumes. Existing-plant interpretation is an inference, not a prohibition on retrofit studies. |
| RC11 | Cited: `Miederer2025` | Equation 2 includes reactor volume in aeration demand. Dependence on the present HRT and proxy follows internally. |
| RC12 | Cited jointly with RC13: `Alex2008` | Section 7 defines disposal solids using stream concentration times waste flow. |
| RC13 | Cited jointly with RC12: `Alex2008` | Section 2.4 includes reactor and settler inventories in sludge age. |
| RC14 | Cited: `Zhang2026` | Secondary-treatment methods include RAS/internal recirculation and component inlet balances. |
| RC15 | Cited jointly with RC16–RC17: `USEPA2009,Zhang2026` | EPA distinguishes pathways; Zhang's mASM2d retains individual nutrient forms. Insufficiency of totals alone is an inference. |
| RC16 | Cited jointly with RC15 and RC17: `USEPA2009,Zhang2026` | Nutrient forms and component transformations support the redistribution interpretation. |
| RC17 | Cited jointly with RC15–RC16: `USEPA2009,Zhang2026` | Phosphorus storage and nonreactive solids separation support the interpretation. Retention requires external solids discharge for net removal. |
| RC18 | Cited: `Zhang2026` | Equations 9–11 describe reactor component conversion before nonreactive clarification. |
| RC19 | Cited: `USEPA2009,Alex2008` | EPA Chapters 3, 5, and 6 cover species, DO, and SRT; BSM1 Sections 2.4 and 7 supply inventories and solids losses. The diagnostic list is contextual, not a universal monitoring minimum. |
| RC20 | Cited: `Takacs1991,Zhang2026` | Original settling formulation plus recent implementation and layerwise benchmark checks. Omitted equations of the present aggregate model are established internally. |
| RC21 | Cited: `Zhang2026` | Equation 15 explicitly includes the saturation-minus-DO driving force. |
| RC22 | Cited: `Pedrozo2025,Hameed2026` | Surrogate-family differences, trust-region restrictions, and local search behavior support possible mechanisms. No attribution of this study's gaps is established. |
| RC23 | Cited for comparison context: `Miederer2025,Zhang2026` | Miederer separates dimensionless cost indices from energy; Zhang explicitly models equipment power and prices. The present limitation follows internally. |
| RC24 | Cited: `Ren2026,Zhang2026` | Nutrient-safe scheduling and output-specific design-target assessment. Neither paper supplies limits for the present case. |
| RC25 | Cited: `Bliek2023` | Section 5.3 compares achieved objective values under time budgets including evaluation, training, and acquisition. It supports fuller cost accounting, not a total-cost estimate for this study. |

## Original discovery source records

The linked records below preserve evidence from the original insertion pass. They are historical: the current resolution table and recency-refresh report take precedence for current citation assignments. Branco2017 is no longer cited and its bibliography entry was removed. For new journal/proceedings references, author names, title, year, and volume/pages or article number were checked against the publication or its institutional record. DOI fields were used where verified; no DOI was invented for the BSM1 report, EPA report, or PMLR paper. This was not a Crossref audit of the entire bibliography.

| Key | Source and verification | Bibliography action |
|---|---|---|
| `Alex2008` | Alex et al. (2008), [Benchmark Simulation Model no. 1 (BSM1), original technical report](https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf), TEIE-7229, Lund University, 62 pages. The original title page supplies the full 12-author list; the [Lund catalog](https://portal.research.lu.se/en/publications/benchmark-simulation-model-no-1-bsm1/) lists only nine authors. The reference follows the report itself. | Added with report number and institutional URL. |
| `Alexandrov1998` | [Author-hosted paper](https://www.cs.wm.edu/~va/research/adlt.pdf) and [author publication record](https://www.cs.wm.edu/~vjtorc/research/index.html); Structural Optimization 15, 16–23; DOI [10.1007/BF01197433](https://doi.org/10.1007/BF01197433). | Reused; added verified DOI. |
| `Bliek2023` | Bliek, Guijt, Karlsson, Verwer, and de Weerdt (2023), [published paper in TU Delft's repository](https://pure.tudelft.nl/ws/portalfiles/portal/159400892/1_s2.0_S1568494623007627_main.pdf), Applied Soft Computing 147, 110744; [institutional metadata](https://research.tue.nl/en/publications/benchmarking-surrogate-based-optimisation-algorithms-on-expensive/); DOI [10.1016/j.asoc.2023.110744](https://doi.org/10.1016/j.asoc.2023.110744). The final published title differs from the EXPObench preprint title. | Added the journal version with DOI. |
| `Boyd2004` | Boyd and Vandenberghe (2004), [author-hosted Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), Cambridge University Press. Sections 1.4.1 and 4.1 distinguish local/global solutions; Section 8.1 defines projection. | Reused. |
| `Branco2017` | Branco, Torgo, and Ribeiro (2017), [PMLR publication record](https://proceedings.mlr.press/v74/branco17a.html) and [paper](https://proceedings.mlr.press/v74/branco17a/branco17a.pdf), SMOGN: A Pre-processing Approach for Imbalanced Regression, PMLR 74, 36–50. | Added with proceedings information and stable URL. |
| `JeppssonDiehl1996` | [Authors' institutional record and abstract](https://portal.research.lu.se/en/publications/on-the-modelling-of-the-dynamic-propagation-of-biological-compone/), Water Science and Technology 34(5), 85–92; DOI [10.1016/0273-1223(96)00632-4](https://doi.org/10.1016/0273-1223(96)00632-4). Fit was assessed from the authors' abstract, not an unavailable full text. | Reused; added verified DOI. |
| `Sahigara2012` | [Authors' paper indexed in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6268288/) and [the authors' applicability-domain toolbox page](https://michem.unimib.it/download/matlab-toolboxes/applicability-domain-toolbox-for-matlab/), Molecules 17, 4791–4810; DOI [10.3390/molecules17054791](https://doi.org/10.3390/molecules17054791). Search-indexed methods/abstract and the author page were available; full-text opening encountered access checks. Used for domain restrictions only. | Reused; added verified DOI. |
| `Selerio2026` | [Publisher record and highlights](https://www.sciencedirect.com/science/article/pii/S2772508126000426), Digital Chemical Engineering 20, 100329; DOI [10.1016/j.dche.2026.100329](https://doi.org/10.1016/j.dche.2026.100329). The publisher explicitly identifies a physical-guarantee/accuracy tradeoff. Direct full-text opening failed; no uninspected numerical detail was cited. | Reused; added verified DOI. |
| `Takacs1991` | [Publisher article record](https://www.sciencedirect.com/science/article/pii/004313549190066Y), Water Research 25(10), 1263–1271; DOI [10.1016/0043-1354(91)90066-Y](https://doi.org/10.1016/0043-1354(91)90066-Y). BSM1 Section 2.3.3 independently provides the double-exponential settling formulation used for the model context. | Reused; added verified DOI. |
| `USEPA2009` | U.S. Environmental Protection Agency (2009), [Nutrient Control Design Manual: State of Technology Review Report](https://www.epa.gov/sites/default/files/2019-02/documents/nutrient-control-design-manual-state-tech.pdf), EPA/600/R-09/012, January 2009. Title page and [EPA technical-resource listing](https://www.epa.gov/nutrientpollution/technical-resources-nutrient-pollution) distinguish it from the August 2010 manual. The report was prepared by the Cadmus Group for EPA; institutional authorship follows EPA's bibliographic usage. | Added with report number and official URL. |
| `Valente2025` | [Publisher paper](https://www.nature.com/articles/s42005-025-02329-1), Communications Physics 8, 433; DOI [10.1038/s42005-025-02329-1](https://doi.org/10.1038/s42005-025-02329-1). Predictive errors and physical-law residuals are evaluated separately. | Reused; added verified DOI. |

## Candidate decisions and limits

- The Bhosekar–Ierapetritou review and Forrester–Keane review were searched for optimization guidance. The primary Alexandrov paper and published Bliek benchmark offered more direct support for the selected methodological statements, so no redundant review citation was added here.
- The existing Henze books were considered for nutrient-process context. The accessible EPA report supplied inspectable species and removal descriptions, so it was preferred for this pass. An ASM2d search result with an incomplete institutional author list was not used to create a new reference.
- Valente's projection study supports separate accuracy and physics assessments; its observed accuracy improvements were not treated as proof that every coordinate must improve. Boyd supplies projection definitions, and the componentwise limitation is a mathematical inference.
- Branco supplies an example of difficulties at rare response values. The citation does not establish that the present dataset is imbalanced, nor that its endpoint errors arise from the same cause.
- Bliek supplies a benchmark of solution quality and fuller runtime accounting. It does not include or estimate every development and verification cost of the present workflow.
- RC03, RC05, and RC08 were resolved using the manuscript's own evidence. They are not unresolved citation gaps.

## Verification record

- Every ID RC01–RC25 has one resolution-table row in original order.
- All 25 original passage quotes and their search/support worksheets were preserved.
- All 25 blue wrappers and the unused `\rccite` definition were removed; no prose was rewritten.
- All 13 current Results/Discussion source keys have bibliography entries. The six refresh additions are `Hameed2026`, `Miederer2025`, `Pedrozo2025`, `Ren2026`, `Shahbazi2026`, and `Zhang2026`; `Branco2017` was superseded and removed.
- No task-specific scripts or executable resources were created or run.
- PDFs were not rebuilt; verification covers the edited source and reference mappings.
