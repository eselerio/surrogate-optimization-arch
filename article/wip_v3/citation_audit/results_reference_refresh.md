# Results and Discussion Reference Refresh

Manuscript: `article/wip_v3/manuscript.tex`. Completed 6 October 2026. Scope: Results and Discussion citations and their bibliography entries.

Fifteen of the 19 citation placements were refreshed, using six newly added papers and the existing Sturm–Silva (2025) reference. Four new papers are dated 2026 and two are dated 2025. Discussion prose, numeric results, and Theory and calculation citations were preserved. The current section uses 13 distinct sources, ten from 2023–2026 and three older sources retained for specific support. Branco (2017) was replaced and its now-uncited bibliography entry removed.

## New references and verification

| Key | Verified publication | Support and evidence inspected |
|---|---|---|
| `Hameed2026` | Hameed, G., Chen, T., del Rio Chanona, A., Biegler, L.T., Short, M. (2026). Trust-region filter algorithms utilizing Hessian information for gray-box optimization. AIChE Journal 72, e70236. DOI [10.1002/aic.70236](https://doi.org/10.1002/aic.70236). | [Publisher full text](https://aiche.onlinelibrary.wiley.com/doi/full/10.1002/aic.70236), especially Sections 2.2–2.4 and 4–6: original black-box feasibility checks, trust-region restrictions, surrogate choice, and local-search sensitivity. Uses benchmark and engineering problems, not the present wastewater system. [Author institutional record](https://openresearch.surrey.ac.uk/esploro/outputs/journalArticle/Trustregion-filter-algorithms-utilizing-Hessian-information/991094728102346) confirms the five-author list. |
| `Miederer2025` | Miederer, J., Meier, L., Elhaus, N., Markthaler, S., Karl, J. (2025). Energy Management Model for Wastewater Treatment Plants. Energy Reports 13, 6349–6361. DOI [10.1016/j.egyr.2025.05.045](https://doi.org/10.1016/j.egyr.2025.05.045). | [Publisher record](https://www.sciencedirect.com/science/article/pii/S2352484725003245) and [authors' institutional metadata](https://cris.fau.de/publications/344978885/). Search-indexed author-uploaded article text exposed Section 2.3, Equations 1–6: dimensionless operating-cost index, separate energy terms, and volume-dependent aeration. Direct publisher full-text opening failed. |
| `Pedrozo2025` | Pedrozo, H.A., Zamarripa, M.A., Uribe-Rodríguez, A., Panagakos, G., Diaz, M.S., Biegler, L.T. (2025). Surrogate model optimization: A comparison case study with pooling problems of CO2 point sources. Computers & Chemical Engineering 200, 109199. DOI [10.1016/j.compchemeng.2025.109199](https://doi.org/10.1016/j.compchemeng.2025.109199). | [Publisher record, abstract, and author list](https://www.sciencedirect.com/science/article/pii/S0098135425002030). Compares five surrogate families and one-shot/TRF optimization; predictive accuracy does not determine optimization performance. Abstract-level fit was sufficient for the general statement; no uninspected results were added. |
| `Ren2026` | Ren, H., Cao, J., Zhao, W., Jin, J., Chen, B., Wu, X. (2026). XGBoost-Assisted Effluent Ammonium Buffer-First Aeration Scheduling for Wastewater Treatment Under Influent Disturbances: A BSM2 Study. Water Environment Research 98, e70520. DOI [10.1002/wer.70520](https://doi.org/10.1002/wer.70520). | [Publisher record](https://onlinelibrary.wiley.com/doi/10.1002/wer.70520), first published 5 August 2026, and [search-indexed authors' full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13439487/), Sections 3.2–3.3. Predictive nutrient-limit screening is calibrated and tested under disturbances and mismatch. Its ammonium evidence is application context; it does not supply this study's TN/TP values or guarantee real-plant compliance. |
| `Shahbazi2026` | Shahbazi, S., Mohammadi, H., Afsharchi, M. (2026). Hybrid imbalanced regression through unified data-level and algorithm-level balancing. Expert Systems with Applications 322, 131908. DOI [10.1016/j.eswa.2026.131908](https://doi.org/10.1016/j.eswa.2026.131908). | [Publisher article metadata and substantive text](https://www.sciencedirect.com/science/article/pii/S0957417426008213) and [authors' manuscript](https://arxiv.org/html/2606.01221v1). Discusses errors in sparse/extreme target ranges and tests balancing methods across benchmark datasets. Does not establish imbalance as the cause of errors in the present data. The published journal version is cited. |
| `Zhang2026` | Zhang, X., Rai, S., Wang, Z., Li, Y., Guest, J.S. (2026). An agile benchmarking framework for wastewater resource recovery technologies. npj Clean Water 9, 4. DOI [10.1038/s41545-025-00537-4](https://doi.org/10.1038/s41545-025-00537-4). | [Publisher article](https://www.nature.com/articles/s41545-025-00537-4), Methods: Secondary treatment, Aeration, Pumping and mechanical mixing, and Sludge disposal. Equations 9–11 use fixed reactor volumes and component balances; Equation 15 uses the DO deficit; the final clarifier uses nonreactive Takács settling with RAS/WAS streams. Figure 2 distinguishes total nutrients from species and individual targets. Publisher version-of-record date is 7 January 2026, volume 9/article 4; December 2025 early publication and DOI year were not used as the final reference year. |

## Replacement decisions

| Original audit IDs | Previous support | Current support | Reason |
|---|---|---|---|
| RC01 | Boyd2004, Selerio2026 | Selerio2026, SturmSilva2025 | Recent conservation-correction studies directly demonstrate that accuracy improvements are not universal. Boyd remains in Theory for the mathematical projection properties. |
| RC02 | JeppssonDiehl1996, Alex2008 | Zhang2026 | Recent connected plant model explicitly includes RAS/WAS and solids-disposal mass flows. |
| RC04 | Branco2017 | Shahbazi2026 | Recent peer-reviewed benchmark examines poor prediction of rare/extreme response values. |
| RC07 | USEPA2009 | Ren2026, Zhang2026 | Recent nutrient safeguards plus TN/TP target-based plant-model evaluation; the present threshold-crossing example remains internal. |
| RC09 | Alexandrov1998, USEPA2009 | Hameed2026, Ren2026 | Recent original-model feasibility checks and wastewater nutrient-limit validation. |
| RC10 | Alex2008 | Zhang2026 | Explicit constant user-defined reactor volume during simulation. |
| RC11 | Alex2008 | Miederer2025 | Equation 2 explicitly multiplies reactor volume by aeration coefficient. |
| RC14 | Alex2008 | Zhang2026 | Recent recirculation and component-balance implementation. |
| RC15–RC17 | USEPA2009 | USEPA2009, Zhang2026 | Recent component-resolved nutrient modeling supplements comprehensive species/process guidance. One shared citation placement. |
| RC18 | JeppssonDiehl1996 | Zhang2026 | Component-resolved biological conversion followed by nonreactive clarification. |
| RC20 | Takacs1991, JeppssonDiehl1996 | Takacs1991, Zhang2026 | Retains the original implemented model and adds a current implementation with layerwise checks. |
| RC21 | Alex2008 | Zhang2026 | Equation 15 directly states the oxygen-transfer driving force. |
| RC22 | Alexandrov1998, Sahigara2012, Boyd2004 | Pedrozo2025, Hameed2026 | Recent empirical evidence on surrogate choice and local/trust-region search behavior. The present causal attribution remains unresolved. |
| RC23 | Alex2008 | Miederer2025, Zhang2026 | Recent separation of dimensionless cost indices, modeled equipment energy, and price-based operating costs. |
| RC24 | Alex2008, USEPA2009 | Ren2026, Zhang2026 | Recent work separately checks nutrient safety and output-specific treatment targets. |

RC06 and RC25 already use recent references (2025/2026 and 2023), so they were retained. RC03, RC05, and RC08 remain internally supported without external citations.

## Older sources retained

- **Alex et al. (2008):** retained for RC12–RC13 and RC19 because its sludge-age definition explicitly includes reactor and settler inventories and external solids losses. Papers about sludge control were not treated as replacements without that exact definition.
- **USEPA (2009):** retained for nutrient forms, conversion, phosphorus storage, and the broad diagnostic list (RC15–RC17 and RC19). Recent application studies strengthen this context but do not replace the complete guidance.
- **Takács et al. (1991):** retained at RC20 because this is the original settling formulation implemented in the model. A recent paper using that formulation supports current relevance but does not replace its provenance.

Other older references remain elsewhere in the manuscript where they support methods or theory. Their removal would be a different audit.

## Alternatives assessed

The 2024–2025 trust-region and constrained Bayesian-optimization candidates were superseded by the directly relevant 2026 Hameed study. Recent secondary-clarifier digital-twin and sludge-blanket studies were not used as substitutes for component-transport or inventory definitions they did not explicitly establish. Broad nutrient-removal reviews and unrelated energy/recovery papers were not selected solely for a recent date. The 2026 ASM review was considered, but Zhang's original component-resolved benchmarking study offered inspectable equations and a closer model match.

## Verification

All six added keys have bibliography entries with verified DOI links, author lists, years, and volume/page or article identifiers. The pre-edit snapshot was compared against the edited source; changes to the Results/Discussion are citation substitutions/additions only. Theory citations were preserved. The matching citation-needs worksheet was updated to the current mappings. No PDFs were rebuilt.
