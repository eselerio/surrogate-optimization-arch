# Introduction Citation Resolution

All eight citation needs have been addressed in [manuscript.tex](manuscript.tex). The blue tags and their macro have been removed. The IDs below are retained only as an editorial record. The original excerpts and search guidance later in this file are historical and no longer describe highlighted manuscript text.

## Resolution Summary

| Original ID | Resolution | References used |
|---|---|---|
| RC01 | Replaced the broad list of operating effects with specific statements about reactor-Clarifier solids coupling and the effects of sludge age on oxygen transfer and effluent quality. | `HartelPopel1992`, `Leu2012` |
| RC02 | Replaced the claim about common practice with the specific use of dynamic relaxation to assist a steady-state activated sludge calculation. | `Selerio2026` |
| RC03 | Added the papermaking wastewater optimization example without importing the supplied numerical speedup claims. | `He2023` |
| RC04 | Cited negative component predictions and stoichiometric conservation violations. Removed the unsupported extension to inventories between multiple reactors. | `Selerio2026` |
| RC05 | Replaced "many components close to zero" with named effluent constituents and the reported effects of longer sludge age. Retained the simple relative-error example as arithmetic illustration. | `Leu2012` |
| RC06 | Linked favorable aggregate errors and physical violations to the activated sludge comparison. Kept accuracy and physical consistency as separate assessment requirements. | `Selerio2026` |
| RC07 | Distinguished residual penalties from exact conservation and explained why decision bounds alone do not constrain output balances. Removed the adjacent unqualified priority claim about the first projection method. | `Beucler2021`, `Selerio2026` |
| RC08 | Replaced the universal absence claim with a comparison of constrained single-reactor prediction and surrogate-assisted treatment optimization. Stated their integration as this study's objective rather than asserting an exhaustive literature result. | `Selerio2026`, `He2023` |

## Reference Checks

- The new H\"artel-P\"opel, Leu, and He entries use the manuscript's existing bibliography format. Their author, title, year, journal, volume, and page or article-number fields were checked against Crossref records.
- H\"artel and P\"opel's correct DOI is [10.2166/wst.1992.0128](https://doi.org/10.2166/wst.1992.0128). The supplied DOI ending in `0125` identifies a different paper. The retrieved abstract supports coupling among Clarifier dynamics, reactor solids, and effluent quality.
- Leu, Chan, and Stenstrom's record is [10.2175/106143011X12989211841052](https://doi.org/10.2175/106143011X12989211841052). The retrieved abstract supports the sludge-age, oxygen-transfer, particle-removal, and biodegradable-organic-carbon statements.
- The registered author list for [He et al. (2023)](https://doi.org/10.1016/j.jclepro.2023.139039) is He, Hong, Zheng, Wang, Xiong, and Man, rather than the list in the supplied summary. The synopsis supplied by the author supports the qualitative optimization example. Full text was not retrieved, and no numerical timing claim was imported.
- The existing Selerio and Beucler bibliography entries were retained. The Selerio numerical-method and benchmark statements draw on the supplied source synopsis. Conflicting replacement entries for Beucler and the Sturm papers were not adopted.
- The other supplied papers were not needed for this focused revision. In particular, no Al et al. entry was added because the supplied DOI did not resolve to a confirmed record during this check. This does not establish that the paper does not exist.
- These checks establish bibliographic consistency for the selected new entries, not an exhaustive literature search or independent verification of every result in the supplied summaries.

## Original Audit

## RC01

> Changes in recycle, aeration, or wasting affect reactor conditions, power demand, solids age, treated-water quality, and the solids returned to the biology.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This describes broad, linked effects of manipulated variables in activated sludge treatment. | Process modeling or operating studies showing how recycle, aeration, and wasting influence reactor behavior, energy or oxygen demand, solids retention, effluent quality, and solids return. Multiple sources may be needed to cover the whole list. | `("activated sludge" OR "wastewater treatment") AND ("internal recycle" OR aeration OR wasting) AND ("solids retention time" OR "effluent quality" OR energy)` |

## RC02

> Their predicted steady state is commonly obtained by integrating these equations forward in time until the transient changes become negligible.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| The sentence describes a common numerical practice, not a consequence of the cited model definitions alone. | Activated sludge and Clarifier simulation work that obtains steady operating states by time integration or dynamic simulation to convergence. | `("activated sludge model" OR "secondary clarifier") AND ("steady state" OR "steady-state") AND ("dynamic simulation" OR "time integration")` |

## RC03

> Statistical surrogates can be evaluated much faster than mechanistic models.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| The computational-speed comparison is general and depends on the models being compared. | Process optimization or wastewater studies comparing prediction evaluation time for fitted statistical surrogates and mechanistic simulations, with a clearly stated scope. | `("surrogate model" OR "metamodel") AND ("mechanistic model" OR "process simulation") AND ("evaluation time" OR "computational cost")` |

## RC04

> An unconstrained regression may predict a negative concentration or fail to preserve a conserved inventory between reactors.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| The passage attributes two specific physical failures to unconstrained statistical prediction. | Examples or methodological analyses of unconstrained process surrogates violating non-negativity or mass conservation in biological or chemical systems. | `("unconstrained surrogate" OR "machine learning") AND ("negative concentrations" OR nonnegativity) AND ("mass conservation" OR "mass balance")` |

## RC05

> This concern is particularly important in wastewater treatment because the removal processes bring the treated-water concentrations of many components close to zero.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| The assertion about low treated-water concentrations is a domain claim and motivates the sensitivity to prediction error near zero. | Wastewater effluent measurements or activated sludge model results showing low concentrations of removed soluble or particulate components relative to influent concentrations. The word "many" should match the scope of the evidence. | `("activated sludge" OR "wastewater effluent") AND ("low concentration" OR "near zero") AND ("component removal" OR nutrients OR solids)` |

## RC06

> A regression fitted directly to concentration data may therefore appear accurate over the overall concentration range while producing poor or physically implausible treated-water predictions.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| The numerical example preceding this sentence explains relative error, but the claim extends to how regression performance can hide errors at the effluent end. | Evidence or analysis of concentration-scale fitting or aggregate error metrics masking low-concentration effluent error or physically implausible predictions. | `("wastewater" OR "water quality") AND ("surrogate model" OR regression) AND ("low concentration" OR effluent) AND ("relative error" OR "prediction error")` |

## RC07

> These formulations retain process knowledge or impose operating requirements during decision making, but soft equation residuals and optimization-level constraints do not guarantee that every unconstrained statistical prediction satisfies exact balances and bounds.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This draws a methodological distinction between penalties or constraints on decisions and hard feasibility of each model prediction. | Technical comparisons of soft physics penalties or optimization-level constraints with hard-constrained outputs or post-prediction projection, including the conditions under which exact feasibility is guaranteed. The already cited example papers can be discussed for their actual constraint scope. | `("physics-informed" OR "soft constraints") AND ("hard constraints" OR "output projection") AND ("mass conservation" OR feasibility)` |

## RC08

> No study was identified that combines hard output-space physical reconciliation of a statistical activated sludge surrogate with plant-wide operational optimization.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This is a novelty claim about the literature and cannot be proved by one isolated citation or a search that finds no result. | A documented comparison of relevant prior work, especially single-unit output projections, hybrid activated sludge models, and plant-wide surrogate optimization. Assess whether any earlier paper combines both requirements before retaining this wording. | `("activated sludge" OR "wastewater treatment plant") AND ("output projection" OR "physical reconciliation" OR "hard constraints") AND ("surrogate optimization" OR "plant-wide optimization")` |