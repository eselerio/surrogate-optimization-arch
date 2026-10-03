---
name: manage-manuscript-citations
description: 'Audit LaTeX manuscripts for statements that need proof, support, or citations; highlight exact passages with linked IDs; create or update a citation-needs Markdown worksheet; search for and verify scholarly references; add effective citations and bibliography entries; remove resolved highlights; and validate the final manuscript. Use for citation audits, citation discovery, reference searches, unsupported-claim reviews, or end-to-end manuscript citation cleanup in any section.'
argument-hint: 'Specify the manuscript, section scope, and mode: audit, search, integrate, or full.'
user-invocable: true
disable-model-invocation: false
---

# Manage Manuscript Citations

## Purpose

Use this skill to maintain an auditable connection among manuscript claims, evidence needs, scholarly searches, and final citations. It supports four modes.

- **Audit** identifies passages that need external citation or another explicit form of support. It highlights those passages and creates a linked Markdown worksheet.
- **Search** finds and verifies references for existing worksheet items without changing the manuscript unless the user requests it.
- **Integrate** adds verified citations and bibliography entries, revises wording when needed for accurate source fit, removes resolved highlights, and updates the worksheet.
- **Full** performs audit, search, and integration in one workflow.

The skill is self-contained. Do not load or invoke another skill to perform any stage.

## Inputs And Defaults

Determine these inputs from the request and workspace before editing.

1. **Manuscript**: Use the requested `.tex` file. If none is named, use the active LaTeX manuscript or ask for the path when more than one plausible manuscript exists.
2. **Section scope**: Honor named sections exactly. If the user requests the whole article, audit the main body from Introduction through Conclusion. Exclude the abstract, highlights, nomenclature, acknowledgments, data statements, appendices, and references unless explicitly included.
3. **Mode**: Use the mode requested by the user. Infer `audit` from requests to identify or mark unsupported statements, `search` from requests to find sources, `integrate` from requests to add already selected references, and `full` when all stages are requested.
4. **Worksheet path**: Use the requested path. Otherwise create `<manuscript-directory>/citation_audit/<manuscript-name>_citation_needs.md`, where `<manuscript-name>` is the filename without `.tex`. For example, `manuscript.tex` becomes `manuscript_citation_needs.md`.
5. **Marker sequence**: Continue from the highest marker ID in the current manuscript. If the manuscript has no markers but its worksheet contains earlier IDs, ask whether to continue that sequence or restart at `CN01`. Otherwise use `CN01`, `CN02`, and so on. Use three digits only when more than 99 items are needed.

Ask a question only when the manuscript, scope, or requested mode cannot be inferred safely.

Before starting a mode, check its prerequisites.

- Audit can start from a manuscript alone.
- Search requires an existing worksheet with open items. If none exists, ask for its path or offer to run Audit first.
- Integrate requires an existing worksheet and at least one verified candidate or user-supplied reference. If these are absent, run Search when requested or explain what is missing.
- Full starts from a manuscript and creates or reuses the worksheet automatically.

## Core Principles

1. Read the full requested span before deciding what to mark.
2. Treat citation proximity and citation adequacy as different questions. A nearby citation must support the exact claim.
3. Mark the smallest complete passage that captures one support need.
4. Separate external claims from facts established by the manuscript's equations, methods, tables, figures, or reported results.
5. Do not add a citation merely to make a sentence appear supported. Verify source-to-claim fit.
6. Prefer a narrower accurate statement over a sweeping statement supported only indirectly.
7. Preserve a traceable record in the worksheet from initial wording through final resolution.
8. Do not create or execute task-specific scripts for this workflow. Use direct reading, search, editing, and standard validation commands.

## Support Classification

Classify every flagged passage before deciding that it needs a literature citation.

| Support type | Use when | Normal resolution |
|---|---|---|
| External citation | The passage states general scientific knowledge, prior findings, accepted methodology, algorithm behavior, or practical implications beyond this study. | Add one or more verified scholarly citations. |
| Internal derivation | A mathematical or logical claim should follow from equations or assumptions in the manuscript, but the derivation is absent or unclear. | Add a concise derivation, proof, or equation reference. |
| Internal evidence | A claim concerns this study's data or results but lacks a table, figure, appendix, or methods pointer. | Add the relevant internal reference or qualify the claim. |
| Qualification | The claim is broader or more causal than the available evidence supports. | Narrow, split, or rewrite the claim. Add a citation only for the remaining external part. |
| Citation-fit check | A citation is present, but its relevance, scope, or placement is doubtful. | Verify the source, replace it, move it, or revise the sentence. |

Only external-citation and citation-fit items require literature searching by default. Record the other types because the user asked for statements needing proof or support, not only missing references.

## What Usually Needs External Support

Mark concise passages that contain one or more of the following.

- General scientific, engineering, clinical, economic, or domain facts.
- Claims about causal mechanisms or why an observed behavior occurs.
- Statements about established models, algorithms, software, numerical methods, or statistical properties.
- Claims about advantages, limitations, robustness, generalization, extrapolation, computational cost, or practical risk.
- Literature-state statements, novelty claims, and research-gap claims.
- Comparisons with prior work or claims that a method is common, standard, preferred, or widely used.
- Broad implications that extend beyond the manuscript's own cases or data.
- Specialized mathematical propositions not derived locally and not elementary for the intended audience.

## What Usually Does Not Need External Support

Do not mark these unless they include a broader unsupported claim.

- Definitions, notation, and declared study-specific assumptions.
- Direct descriptions of the authors' data, code, fitting procedure, or experimental design.
- Numeric results already tied clearly to a table, figure, or reported calculation.
- Transparent consequences of adjacent equations or definitions.
- Simple transitions, section previews, or statements of purpose.
- Cautious limitations that merely state what the present model does not include.
- Widely understood facts that are elementary for the journal's audience.

Do not over-tag. An exhaustive audit means every sentence is considered, not that every sentence is highlighted.

## Audit Mode

### 1. Establish Boundaries And Existing Infrastructure

1. Locate the requested section headings and the next out-of-scope heading.
2. Read the preamble and check for `xcolor` and existing citation-need macros.
3. Search the manuscript for existing marker IDs and continue their sequence.
4. Read the complete scoped prose, including nearby captions when they contain interpretation.

### 2. Perform Two Passes

**Pass A: sentence inventory**

For each sentence, decide whether it is externally sourced, internally established, study-specific, or merely connective. Note every plausible support gap.

**Pass B: discriminating review**

For each plausible gap, ask:

1. Does an attached or nearby citation support the complete claim?
2. Does the claim follow directly from an equation, definition, table, figure, or stated assumption?
3. Is the claim appropriate as common knowledge for the target journal audience?
4. Would a citation resolve the issue, or does the wording need proof, an internal pointer, or qualification?
5. Is the passage minimal, or can unrelated wording be removed without losing the support need?

Mark only items that remain after this review.

### 3. Add LaTeX Markers

If no compatible marker exists, load `xcolor` once and add:

```latex
\newcommand{\citationneed}[2]{\textcolor{blue}{[#1] #2}}
```

If the manuscript already uses another review-marker convention, adapt to it and document the syntax in the worksheet. If `xcolor` cannot be added without a package conflict, use a consistent bold or underline marker and report that limitation.

Wrap exact passages as follows:

```latex
\citationneed{CN01}{Exact passage that needs support.}
```

Rules:

- Preserve the original wording inside the marker during audit-only mode.
- Keep sentence punctuation inside the marker when it belongs to the selected passage.
- Do not wrap section headings, paragraph breaks, floats, list boundaries, or multiple unrelated claims in one marker.
- Split passages when different clauses need different evidence.
- Preserve all LaTeX commands and mathematics exactly.
- When a passage contains a fragile command such as `\footnote` or `\marginpar`, keep that command outside the marker while preserving the visible sentence.
- Do not mark bibliography entries, table data, source code, or generated files.

### 4. Create Or Update The Worksheet

Use this structure:

```markdown
# Manuscript Citation Needs

- Manuscript: `<path>`
- Scope: `<sections>`
- Status: audit complete; references not yet integrated

## CN01

> Exact passage inside the LaTeX marker.

| Field | Detail |
|---|---|
| Location | Section and nearby subsection or paragraph anchor |
| Support type | External citation, internal derivation, internal evidence, qualification, or citation-fit check |
| Why support is needed | Specific explanation of the unsupported inference or generalization |
| Required evidence | What a suitable source, derivation, or internal result must establish |
| Effective searches | `("exact concept" OR synonym) AND (method OR domain)` |
| Status | Open |

### Candidate references

| Candidate | DOI or stable URL | Evidence of fit | Limitations | Decision |
|---|---|---|---|---|

### Resolution

Pending.
```

Worksheet rules:

- The block quote must match the marked text word for word, excluding the macro wrapper.
- Keep the rationale specific to the claim.
- Use practical Scholar-style Boolean searches with two to four concept groups. For example, `("support vector machine" OR SVM) AND (surrogate model) AND uncertainty` uses quoted phrases for exact concepts, `OR` for alternatives, and `AND` to narrow the search.
- Preserve LaTeX notation in quotes.
- Keep IDs in manuscript order.
- Do not invent candidates during audit mode.

## Search Mode

### 1. Recover The Exact Evidence Need

Read each open worksheet item in its manuscript context. Do not search from keywords alone. Split an item when one sentence contains claims that require different source types.

### 2. Search In Layers

Search in this order when practical.

1. Existing manuscript bibliography for a source already cited elsewhere.
2. Exact concept and domain terms from the worksheet.
3. Primary studies for specific empirical or methodological claims.
4. Authoritative reviews, standards, or textbooks for broad established principles.
5. Original algorithm or theorem papers for mathematical and numerical claims.

Use scholarly indexes, publisher pages, DOI registries, institutional repositories, and accessible full text. Google Scholar queries may guide discovery, but do not treat search-result snippets as verification.

### 3. Verify Every Candidate

For each candidate:

1. Confirm exact title, authors, year, venue, volume, pages or article number, and DOI or stable URL from reliable metadata.
2. Read the abstract and relevant full-text passage when available.
3. Record the section, theorem, equation, or close paraphrase that supports the claim.
4. Check population, process, model class, operating regime, and assumptions against the manuscript statement.
5. Note limitations. A source from another domain may support a mathematical principle but not a domain-specific conclusion.
6. Prefer the most direct source. Use multiple citations only when a claim has distinct components.
7. Reject title-only matches and sources that merely cite the needed result secondhand when the primary source is available.

Never fabricate metadata, DOI values, quotations, page numbers, or source conclusions. Mark uncertain details as unverified.

### 4. Update Candidate Tables

Record both accepted and important rejected candidates. Use `Recommended`, `Supporting`, `Rejected`, or `Unverified` in the Decision column. Explain why the recommended source fits and what it does not support.

In search-only mode, do not alter manuscript wording, citation commands, highlights, or bibliography entries unless explicitly requested.

## Full Mode

Run the stages in this order.

1. Complete Audit Mode and its validation.
2. If no items are found, update the worksheet with a zero-item result and stop.
3. Run Search Mode for every open external-citation or citation-fit item. Process large audits in manuscript order and keep each item's candidate table current before moving on.
4. Run Integration Mode only for items with verified support. Leave unsupported or ambiguous items open and highlighted.
5. Run integration validation, cross-manuscript consistency checks, and the PDF build when available or requested.

Do not remove all markers as a batch before each item has a recorded resolution. A failure in one item must not prevent verified items from being resolved, but the completion report must list every remaining open item.

## Integration Mode

### 1. Choose The Resolution

For each open item, select the smallest defensible resolution.

- Add a citation when a verified source supports the existing wording.
- Split a sentence when separate clauses need separate sources.
- Narrow or qualify wording when the source supports only part of the claim.
- Add an equation, figure, table, appendix, or methods reference for internally established claims.
- Leave the item open when no adequate support is found. Do not force a weak citation.
- Do not integrate candidates with unverified metadata or uncertain source-to-claim fit unless the user explicitly accepts that risk. Keep those items open.

### 2. Add Effective Citations

Follow the manuscript's existing citation package and style.

If the manuscript has no active citation command or bibliography system, do not assume `natbib`. Ask the user before adding a package or creating a bibliography environment, then use the journal template's documented convention.

- Use textual citations such as `\citet{Key}` when the authors are part of the sentence.
- Use parenthetical citations such as `\citep{Key}` when the source supports the preceding statement.
- Place a citation immediately after the supported clause or sentence.
- Avoid placing one citation after a paragraph containing several unrelated claims.
- Reuse an existing bibliography key when the source is already present.
- Avoid citing reviews for precise claims when an accessible primary source is more direct.

### 3. Add Bibliography Entries

Match the manuscript's current bibliography system exactly.

- For an inline `thebibliography`, copy its punctuation, author–year label style, capitalization, ordering, journal naming, page ranges, and article-number conventions.
- For BibTeX or BibLaTeX, add a syntactically valid entry to the active database and preserve its field conventions.
- Define every cited key exactly once.
- Use verified metadata. Do not silently alter published titles to satisfy manuscript prose preferences.
- Keep entries in the manuscript's established order, normally alphabetical when that pattern is present.

### 4. Remove Resolved Markers

Replace the marker with the final prose and citation:

```latex
\citationneed{CN01}{Original passage.}
```

becomes, for example:

```latex
Revised supported passage \citep{VerifiedKey}.
```

Remove only resolved markers. Keep unresolved items highlighted. If no marker uses remain anywhere in the manuscript, remove the unused `\citationneed` macro. Remove `xcolor` only when it is otherwise unused.

### 5. Update The Worksheet

Preserve the original audited quote. Update the item as follows:

- Set Status to `Resolved`, `Partially resolved`, or `Open`.
- Keep the candidate table and decisions.
- In Resolution, record the final manuscript wording, citation keys, full selected references, support route, and any wording change made for source fit.
- Update the document-level status to summarize open and resolved counts.
- State that the highlight was removed only when it was actually removed.

Example:

```markdown
### Resolution

- Status: Resolved
- Final wording: "Revised supported passage."
- Citation keys: `VerifiedKey`
- Manuscript action: Added the citation and removed marker `CN01`.
- Reason for revision: Narrowed the claim to the operating regime tested by the source.
```

## Cross-Manuscript Consistency

During Audit, note material contradictions or inconsistent restatements that affect the evidence need. After changing prose in Integration or Full mode, inspect the nearest related statements in the abstract, introduction, methods, results, conclusion, captions, and supplementary references when applicable.

- Keep terminology, capitalization, hyphenation, and acronym expansion consistent.
- Spell out each acronym at its first main-text use and use the acronym consistently afterward.
- Do not let a qualified theory claim remain overstated in the abstract or conclusion.
- Keep detailed formulation choices in theory or methods rather than adding them to the introduction.
- Use direct formal sentences, simple vocabulary, and one main idea per sentence.
- Avoid changing unrelated text during a citation task.

## Validation

### Audit Validation

1. Count marker IDs in the manuscript and item headings in the worksheet.
2. Confirm the ID sets, order, and zero padding match exactly. New audit IDs must be consecutive. After integration, gaps are acceptable when removed IDs are retained in the worksheet as resolved.
3. Compare every worksheet quote with its marked LaTeX passage word for word.
4. Confirm every marker lies within the requested scope.
5. Confirm no task-specific scripts or executable artifacts were added.

### Search Validation

1. Confirm every recommended reference has verified metadata.
2. Confirm every recommendation includes direct evidence of source-to-claim fit.
3. Confirm limitations and rejected near-matches are recorded.
4. Confirm search-only mode did not alter the manuscript unless requested.

### Integration Validation

1. Confirm every resolved ID has no remaining manuscript marker.
2. Confirm every unresolved ID remains marked and documented.
3. Confirm each new citation key is defined exactly once and each new bibliography entry is cited.
4. Search for stale marker macros, orphan IDs, duplicate references, and prohibited wording specified by the user.
5. Run editor diagnostics and a whitespace or diff check.
6. Build the PDF when requested or when a local LaTeX engine is available.

For PDF validation, use an existing repository build command when documented. Otherwise run from the manuscript project directory using the first available option:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
tectonic manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

Run `pdflatex` twice when it is the selected engine. The commands above are alternatives, not one sequence. Inspect the log for undefined citations, undefined references, multiply defined labels, and fatal errors. Report nonfatal layout warnings separately. Do not claim a successful build unless the PDF exists.

## Completion Report

Report only what was actually completed.

- Mode and section scope.
- Manuscript and worksheet paths.
- Number of items marked, searched, resolved, partially resolved, and still open.
- References added or reused.
- Whether markers and unused macros were removed.
- PDF path and build result when built.
- Remaining uncertainties, unresolved evidence needs, or validation warnings.