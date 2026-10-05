# Theory and Calculation Reference Transfer

- Source reference pool: `surrogate-projection-arch/article/ESM_v0/manuscript.tex`.
- Edited manuscript: `surrogate-optimization-arch/article/wip_v3/manuscript.tex`.
- Scope: Section 2, Theory and calculation; bibliography changes needed for the transferred citations.
- Completed: 6 October 2026.
- Result: 10 citation placements, two new bibliography entries, and reuse of six existing references. Section prose, equations, headings, and order are preserved.

## Projection attribution

The source manuscript's Section 2.3 implements the Kircher–Votsmeier conservative null-space correction followed, when necessary, by positivity backtracking. The destination's Section 2.5 instead solves a strictly convex quadratic program with equalities, non-negativity, densification, inventory inequalities, and a fitted overflow closure. These are different algorithms.

Kircher and Votsmeier is cited for simultaneous conservation/positivity enforcement after prediction and the related conservative-subspace principle. It is not presented as the source of the destination's full plant-wide QP. Selerio (2026) supplies ICSOR provenance, Valente et al. supplies related output-projection methodology, and the existing Boyd references supply the convex-set, uniqueness, and projection-distance results. The plant-specific balance and inventory derivations retain their own equations as support.

## Insertions and evidence

| Placement | Citation keys | Fit and limits |
|---|---|---|
| 2.2, reactor component-balance introduction | `Henze2006` | Activated sludge stoichiometric/kinetic component-balance framework. Already in both bibliographies. |
| 2.2, conserved component combinations before the invariant operator | `SturmWexler2022,KircherVotsmeier2025` | Stoichiometric conservation and conservative subspaces. The specific choice of rows satisfying both `A nu^T = 0` and `A e_SO = 0` remains the destination's declared construction; it is not universal conservation of externally supplied oxygen. |
| 2.4, first presentation of extended ICSOR | `Selerio2026` | Explicitly credits the original ICSOR method as requested. The connected plant response is this manuscript's extension, not an assertion that the original study modeled the entire plant. |
| 2.4, unconstrained raw regression lacks sign/balance enforcement | `KircherVotsmeier2025` | The cited work motivates explicit simultaneous enforcement of conservation and positivity. It does not analyze this exact ridge fit. |
| 2.5, introduction of post-prediction joint projection | `KircherVotsmeier2025,Valente2025` | Related conservative/positive correction and output projection onto physical constraints; not attribution of the full plant constraints to either source. |
| 2.5, scaling to avoid domination by large response coordinates | `SturmSilva2025` | Section 2.2 uses species weights for magnitude and uncertainty. Supports scale-aware correction, not this exact inventory/nutrient normalization. |
| 2.5, introduction of projected displacement and response | `Selerio2026,Valente2025` | ICSOR projection provenance and weighted output-projection formulation. The equations here add the connected plant response and its constraints. |
| 2.5, constraint enforcement does not itself establish improved accuracy | `SturmSilva2025` | Section 3.2 shows that an unweighted conservation correction can degrade predictive accuracy. The following convex projection guarantee retains its existing Boyd citation and the requirement that the reference lie in the set. |
| 2.6, treatment/resource requirements in the objective | `Araujo2013,PadronPaez2020` | Activated sludge operating costs with treatment constraints, and wastewater multiobjective trade-offs. Neither source supplies the present dimensionless objective or priority values. |
| 2.6, need for operating constraints beyond an objective | `Araujo2013` | Sections 3–4 optimize operating costs subject to environmental and operability constraints. Does not justify the exact case-specific bounds here. |

## Source records

- **KircherVotsmeier2025 — added:** Kircher, T., and Votsmeier, M. (2025). Machine Learning Surrogate Models for Mechanistic Kinetics: Embedding Atom Balance and Positivity. The Journal of Physical Chemistry Letters 16, 4715–4723. [ACS publisher record and abstract](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5c00602), DOI [10.1021/acs.jpclett.5c00602](https://doi.org/10.1021/acs.jpclett.5c00602). Publisher abstract confirms projection and linear-interpolation backtracking; source-manuscript equations supplied the local implementation comparison. Direct publisher full-text opening was unavailable.
- **Araujo2013 — added:** Araujo, A. C. B. de, Gallani, S., Mulas, M., and Skogestad, S. (2013). Sensitivity Analysis of Optimal Operation of an Activated Sludge Process Model for Economic Controlled Variable Selection. Industrial & Engineering Chemistry Research 52, 9908–9921. [Authors' institutional record](https://research.aalto.fi/en/publications/sensitivity-analysis-of-optimal-operation-of-an-activated-sludge-/), DOI [10.1021/ie4006673](https://doi.org/10.1021/ie4006673). Search-indexed article text exposed operating-objective and constraint descriptions; direct ACS opening failed.
- **Henze2006 — reused:** Original activated sludge model compilation, already cited in the destination and in the source's component-balance discussion. No bibliography change.
- **SturmWexler2022 — reused:** [Publisher full text](https://gmd.copernicus.org/articles/15/3417/2022/gmd-15-3417-2022.html), DOI [10.5194/gmd-15-3417-2022](https://doi.org/10.5194/gmd-15-3417-2022). Added DOI to the existing entry. The published paper uses P. O. Sturm; the source manuscript's P. M. initials were not copied.
- **SturmSilva2025 — reused:** [Publisher paper, especially Sections 2.2 and 3.2](https://pubs.acs.org/doi/abs/10.1021/acsestair.4c00220), DOI [10.1021/acsestair.4c00220](https://doi.org/10.1021/acsestair.4c00220). Added DOI. Retained the destination's 2025 volume-2 issue year; the source lists the 2024 online-publication year.
- **Valente2025 — reused:** [Publisher paper](https://www.nature.com/articles/s42005-025-02329-1) and [authors' institutional record](https://researchportal.ulisboa.pt/en/publications/physics-consistent-machine-learning-with-output-projection-onto-p/), DOI [10.1038/s42005-025-02329-1](https://doi.org/10.1038/s42005-025-02329-1). Retained the correct article number 433 in the destination; the source's 180 was not transferred.
- **PadronPaez2020 — reused:** [Publisher record and abstract](https://www.sciencedirect.com/science/article/abs/pii/S009813541931004X), DOI [10.1016/j.compchemeng.2020.106850](https://doi.org/10.1016/j.compchemeng.2020.106850). Added DOI and corrected the second author's name to De-León Almaraz in the bibliography and extended natbib label.
- **Selerio2026 — reused, explicitly requested:** [Publisher record](https://www.sciencedirect.com/science/article/pii/S2772508126000426), DOI [10.1016/j.dche.2026.100329](https://doi.org/10.1016/j.dche.2026.100329). Already in the destination bibliography; added citations in Sections 2.4 and 2.5.

## Other candidates assessed

- Beucler, Hansen, HardNet, Raissi, Karniadakis, Kashinath, and Utkarsh describe architectural, training, neural-network, or probabilistic constraint mechanisms. The existing section does not explain those mechanisms, so transferring them would require additional prose or imply that this deterministic regression/QP implements them. No citation was added for that reason.
- Sturm and Wexler (2020) overlaps the selected 2022 conservation reference; their 2023 superspecies transport study is not the scalar clarifier-inventory reduction used here.
- The source's learner-family references for trees, kernels, neural networks, PLS, Lasso, and Elastic Net do not support the destination's second-order ridge fit specifically. They were not transferred as generic regression citations.
- Guerrero and Duan support particular nutrient mechanisms and carbon-source effects. Those interpretations are not developed in Section 2; no new mechanistic explanation was added.
- Durkin, Aboagye, and Bhosekar already support the broader optimization context elsewhere in the destination. Their findings do not supply the exact projection/trust diagnostics or engineering bounds here.
- Environmental-impact, tabular-learning, uncertainty-ensemble, and effluent-prediction papers were not transferred to the theory simply because they occur in the source bibliography.
- No external paper was attached to the original hydraulic identities, endpoint-envelope interval proof, fitted overflow closure, case-specific scaling, or smoothing protocol as though it supplied those derivations.

## Validation

The edited Theory and calculation section was compared with a pre-edit snapshot: changes are citation insertions only. The eight referenced keys have destination bibliography entries, with two new keys added in alphabetical order. Section 2.4 now cites Selerio (2026). No change was made to the source projection manuscript. PDFs were not rebuilt.
