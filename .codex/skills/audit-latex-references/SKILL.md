---
name: audit-latex-references
description: Audit and revise manual APA-style References sections in LaTeX manuscripts. Use when Codex needs to compare article-body author-year citations against a References list, remove uncited or Crossref-unverified references, verify study existence through Crossref metadata, add DOI fields when available, check APA-like formatting, revise the .tex file, and rebuild the manuscript PDF.
---

# Audit LaTeX References

## Overview

Use the bundled PowerShell script to run a repeatable citation/reference audit for LaTeX manuscripts that keep references manually under `\section*{References}`. The script extracts author-year citations from the article body, compares them with `\noindent` reference entries, queries Crossref, adds DOI fields for reliable matches, writes an audit report, and can call the local `build-latex-pdf` skill script to regenerate a PDF.

## Workflow

1. Inspect the manuscript briefly so you know whether the References section is manual APA-style text or BibTeX/BibLaTeX.
2. For manual APA-style references, run the script in report mode first.
3. Review the report for missing body citations, uncited references, Crossref-unverified references, APA warnings, and low-confidence DOI candidates.
4. Rerun with editing enabled after deciding how to handle flagged items.
5. Rebuild the PDF through the script or through `build-latex-pdf`.
6. Report the revised `.tex` path, audit report path, PDF path, removed references, added DOIs, and any unresolved references.

## Commands

Report only:

```powershell
powershell -ExecutionPolicy Bypass -File ".codex\skills\audit-latex-references\scripts\audit_latex_references.ps1" `
  -TexFile "artifacts\wip\manuscript.tex" `
  -NoEdit
```

Revise the file, remove uncited references, add reliable DOI matches, and rebuild the PDF:

```powershell
powershell -ExecutionPolicy Bypass -File ".codex\skills\audit-latex-references\scripts\audit_latex_references.ps1" `
  -TexFile "artifacts\wip\manuscript.tex" `
  -BuildPdf
```

Strict Crossref cleanup after review:

```powershell
powershell -ExecutionPolicy Bypass -File ".codex\skills\audit-latex-references\scripts\audit_latex_references.ps1" `
  -TexFile "artifacts\wip\manuscript.tex" `
  -RemoveUnverified `
  -BuildPdf
```

Use an explicit source root when the `.tex` file depends on nearby assets:

```powershell
powershell -ExecutionPolicy Bypass -File ".codex\skills\audit-latex-references\scripts\audit_latex_references.ps1" `
  -SourceRoot "artifacts\wip" `
  -TexFile "artifacts\wip\manuscript.tex" `
  -BuildPdf
```

## Script Behavior

- Parses body citations before `\section*{References}` using common author-year patterns such as `(Author et al., 2024)`, `(Author & Coauthor, 2024, 2025)`, and `Author et al. (2024)`.
- Parses manual reference entries that begin with `\noindent`.
- Treats a reference as cited when its first-author surname and year appear in the extracted body citations.
- Removes uncited reference entries when editing is enabled.
- Queries Crossref with the reference title, container, first author, and year.
- Adds or updates `doi: ...` only for Crossref matches above the confidence threshold.
- Leaves cited but Crossref-unverified references in place by default and flags them for agent review.
- Removes Crossref-unverified references only when `-RemoveUnverified` is passed.
- Writes a JSON audit report next to the `.tex` file using the suffix `.reference-audit.json`.
- Calls `.codex\skills\build-latex-pdf\scripts\build_latex_pdf.ps1` when `-BuildPdf` is passed.

## Review Rules

- Do not silently remove a cited reference just because Crossref did not return a reliable match. Check whether the reference is malformed, a preprint without Crossref metadata, a non-study source, or a citation that should be removed from the body.
- If body citations are missing from the References list, add the correct reference entry or revise the body citation before considering the manuscript clean.
- If the report lists APA warnings, revise those entries manually. The script checks basic APA shape but does not fully rewrite author names, capitalization, journal styling, or pagination.
- If Crossref returns several plausible matches, prefer the published journal or proceedings DOI over preprint, supplement, SSRN, ChemRxiv, or repository DOIs when the title and venue identify the same work.
- Rerun the script after manual fixes until missing citations and unresolved reference issues are either cleared or explicitly reported.

## Parameters

- `-TexFile`: Required path to the LaTeX manuscript.
- `-SourceRoot`: Optional LaTeX source folder for PDF staging. Defaults to the `.tex` file directory.
- `-ProjectRoot`: Optional repository root. Defaults to `git rev-parse --show-toplevel`, then the current directory.
- `-NoEdit`: Report only; do not modify the `.tex` file.
- `-RemoveUnverified`: Remove references that cannot be verified in Crossref with a reliable match.
- `-BuildPdf`: Generate a PDF after the reference audit.
- `-MinCrossrefScore`: Minimum score for a DOI/reference match. Default is `130`.
- `-Rows`: Number of Crossref candidates to inspect per reference. Default is `8`.

## Reporting

After running the workflow, report:

- The `.tex` file revised.
- The audit JSON report path.
- The generated PDF path when built.
- References removed because they were uncited.
- References removed because they were Crossref-unverified, if strict cleanup was requested.
- DOI additions or updates.
- Any body citations still missing from References.
- Any cited references still lacking reliable Crossref verification or DOI.
