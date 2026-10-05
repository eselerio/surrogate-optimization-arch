# Main manuscript reference audit

Audited on 6 October 2026. Main source is `article/wip_v3/manuscript.tex`.

## Outcome

- All 50 bibliography entries are cited in the manuscript body.
- Every body citation has a matching bibliography key. No duplicate keys were found.
- Forty-six cited works have reliable Crossref identity matches. Four other sources were verified separately and retained without invented DOIs.
- Twenty-eight missing DOI links were added. Existing valid DOI links were retained.
- The Tøndel et al. (2003) title was restored to **An algorithm for multi-parametric quadratic programming and explicit MPC solutions**. The published Automatica article DOI was used, rather than a similarly titled conference paper.
- An access link to the author's archived 2022 thesis was added for Ragonneau (2022).
- No references or body citations were removed.
- Twelve shared supplementary entries were synchronized with the audited main bibliography. This includes correcting van Loosdrecht's initials to M.C.M. in the supplement and restoring the same Tøndel title.

The reference format is manual `thebibliography` with natbib `\bibitem` labels, rather than the plain-text APA list supported by the skill's bundled PowerShell parser. `audit_natbib_references.py` implements the equivalent report-only workflow for this format. It preserves the existing Elsevier author-year layout and records Crossref metadata and rejected candidates in `manuscript.reference-audit.json`. Bibliography edits were made only after reviewing those results.

## Sources without reliable Crossref matches

| Citation | Verification and decision |
|---|---|
| Alex et al. (2008) | [Lund University's technical-report index](https://www2.iea.lth.se/publications/pubtech.html) confirms the title, year, and report identifier TEIE-7229. The report's original access link was retained. The PDF endpoint could not be retrieved during this audit because of timeout/certificate issues. The index has a shortened author list, so it was not used to replace the manuscript's full list. No reliable Crossref DOI was found. |
| Kraft (1988) | The [original report scan](https://degenerateconic.com/uploads/2018/03/DFVLR_FB_88_28.pdf) identifies Dieter Kraft, the title, July 1988, and report DFVLR-FB 88-28. The report was retained. No reliable Crossref DOI was found. |
| Ragonneau (2022) | The [author's archived thesis](https://arxiv.org/abs/2210.12018) and [thesis title page](https://optimization-online.org/wp-content/uploads/2023/02/thesis.pdf) confirm the title, author, university, and August 2022 date. [PolyU's later institutional record](https://theses.lib.polyu.edu.hk/handle/200/12294) lists 2023, so the added access link identifies the archived 2022 work actually cited. No reliable Crossref DOI was found. |
| U.S. Environmental Protection Agency (2009) | The [agency's original report](https://www.epa.gov/sites/default/files/2019-02/documents/nutrient-control-design-manual-state-tech.pdf) confirms the title, January 2009, and EPA/600/R-09/012 identifier. The reference and original access link were retained. No reliable Crossref DOI was found. |

Low-confidence Crossref search results for these four sources refer to different works. Their candidate DOIs were rejected, not added to the manuscript.

## Publication dates and metadata differences

**Unresolved edition year — Henze et al. (2006).** Crossref identifies the same titled work and leading authors at DOI [10.2166/9781780402369](https://doi.org/10.2166/9781780402369), but records publication on 30 December 2015. The book also has an original print publication in 2000, as identified in [IWA Publishing's own book description for ADM1](https://iwaponline.com/ebooks/book/152/Anaerobic-Digestion-Model-No-1-ADM1). These dates do not establish the year of the edition used in this manuscript. The 2006 year and citation key were retained consistently in the manuscript and supplement, and the DOI was added as a verified identifier for the work. The specific edition year still needs confirmation; this audit does not certify 2006 as correct.

**Resolved publication-date difference — Zhang et al. (2026).** Crossref records early publication in December 2025. The [publisher's citation and article history](https://doi.org/10.1038/s41545-025-00537-4) assign volume 9, article 4, and the version of record to 7 January 2026. The manuscript's 2026 citation was retained.

**Other metadata limitations.** Author/editor fields, accented surnames, surname particles, HTML markup, and separate subtitle fields were reviewed rather than treated as evidence of a different study. The Buzzi-Ferraris and Manenti book is confirmed by [Wiley-VCH's own catalog](https://www.wiley-vch.de/en/areas-interest/engineering/nonlinear-systems-and-optimization-for-the-chemical-engineer-978-3-527-33274-8), although Crossref puts its names in editor fields. For Jeppsson and Diehl, the existing Elsevier DOI [10.1016/0273-1223(96)00632-4](https://doi.org/10.1016/0273-1223(96)00632-4) identifies the correct titled work; the more complete IWA Crossref record was used to check authors and pages. The existing DOI was retained.

## Formatting and scope

All entries retain the manuscript's existing author-year punctuation, author initials, venue and volume fields, page ranges or article identifiers, and terminal periods. DOI links use the established `\url{https://doi.org/...}` form. Multi-author natbib labels use et al.; two-author labels name both authors. Exact published titles retain their original acronyms and wording.

The audit checks citation-reference consistency and bibliographic identity. It does not re-assess whether each source supports every scientific claim. DOI identity verification does not independently establish the edition year noted above.

## Validation

Final citation-key, duplicate-key, author-year-label, DOI, and shared-entry checks and the archived PDF build are recorded in `manuscript.reference-audit.json`.
