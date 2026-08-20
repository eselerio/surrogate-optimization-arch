---
name: mark-results-citation-needs
description: Identify statements, arguments, interpretations, and explanations in the Results and Discussion section of a LaTeX manuscript that need citations; mark those exact passages in blue with sequential IDs in the .tex source; and create a matching markdown index with the exact selected text, rationale, required citation content, and Google Scholar Boolean keyword searches. Use when Codex is asked to audit citation needs in Results/Discussion prose, highlight claim fragments in LaTeX, or generate a citation-search worksheet for a manuscript.
---

# Mark Results Citation Needs

## Purpose

Use this skill to perform an editorial citation-needs pass on a LaTeX manuscript. The output must be two linked artifacts:

1. The manuscript `.tex` file with selected Results/Discussion passages wrapped in blue ID tags.
2. A markdown file listing each ID, the exact selected passage, why it needs a citation, what a suitable citation must support, and effective Google Scholar keywords.

Do not create, include, or execute task-specific scripts for this skill. Use direct reading, search, and careful manual editing.

## Scope

Apply the audit only to the Results and Discussion content requested by the user.

- Prefer `\section{Results and Discussion}` when present.
- If Results and Discussion are separate sections, audit both sections.
- Stop before Conclusion, References, Appendix, Supplementary Material, or the next unrelated major section.
- Audit body prose by default. Do not tag table rows, CSV inputs, figure files, or bibliography entries.
- Avoid tagging captions unless the user explicitly asks for captions to be included or the caption contains an interpretive claim that clearly requires external support.

## What To Mark

Mark concise passages that make claims needing support beyond the manuscript's own data, tables, figures, or methods. Favor the smallest complete phrase or sentence that captures the support need.

Mark passages such as:

- General scientific or domain claims, e.g. what activated-sludge states require physically.
- Methodological interpretations, e.g. why accuracy metrics are not enough to establish physical feasibility.
- Explanations of model-family behavior, e.g. why boosted trees, neural networks, TabNet, SVR, or neighbor/tree methods behave a certain way.
- Claims about unconstrained regression, non-negativity, extrapolation, physical admissibility, or downstream optimization/control risk.
- Broad practical implications that extend beyond the observed numeric results.

Usually do not mark:

- Purely internal numeric results already backed by a table or figure.
- Direct descriptions of the study design, dataset, software, or evaluation protocol already established in Methods.
- Simple transitions, summaries, or signposting.
- Conclusions that are narrowly and transparently derived from the immediately cited table or figure, unless the sentence also makes a broader external claim.

Do not over-tag. It is better to mark fewer high-value citation needs than to turn the Results and Discussion into a wall of blue.

## LaTeX Marking Procedure

1. Read the manuscript preamble and the full Results/Discussion span before editing.
2. Add blue highlighting support if absent:

```latex
\usepackage{xcolor}
\newcommand{\rccite}[2]{\textcolor{blue}{[#1] #2}}
```

3. If `xcolor` is already loaded, do not add it again. If `\rccite` already exists, continue its existing ID sequence unless the user asks to restart.
4. Assign IDs sequentially as `RC01`, `RC02`, `RC03`, etc. Use two digits unless more than 99 passages are marked.
5. Wrap the exact selected passage:

```latex
\rccite{RC01}{This is the exact passage that needs citation.}
```

6. Preserve the original wording inside the macro. Do not rewrite the manuscript prose unless the user explicitly asks for revisions.
7. Do not wrap across section headings, floats, tables, figures, item boundaries, or paragraph boundaries. Split into multiple IDs if needed.
8. Keep punctuation inside the macro when the selected sentence includes the punctuation.
9. Preserve LaTeX math and commands inside the selected text when they are part of the exact passage.

## Markdown Output

Create the markdown file in the location requested by the user. If the user does not specify a location, use a nearby citation-output directory such as `article/results_citation/results_discussion_citation_needs.md` for article manuscripts, creating the directory if needed.

Use this exact structure:

```markdown
# Results and Discussion Citation Needs

The IDs below match the blue `RCxx` tags inserted in `<manuscript path>`. Each quoted passage is the exact highlighted portion from the Results and Discussion section.

## RC01

> Exact selected passage from the LaTeX source.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| Explain why the passage needs external support. | Describe the evidence, concept, domain fact, or methodological point a suitable citation must support. | `("term one" OR "term two") AND ("term three" OR concept) AND keyword` |
```

Repeat one section per ID in the same order as the manuscript.

Markdown requirements:

- The block quote must match the highlighted LaTeX passage word for word, excluding only the wrapping `\rccite{ID}{...}` macro.
- Preserve LaTeX math and symbols as written in the selected passage.
- Keep each table to one row unless the citation need truly has distinct support requirements.
- Make the rationale specific. Do not use generic text such as "needs citation because it is a claim."
- Make keywords practical for Google Scholar: use quoted phrases for exact concepts, uppercase Boolean operators, and 2-4 concept groups.
- Do not invent actual references or insert citations unless the user explicitly asks for citation discovery.

## Verification

Before finishing:

1. Count `\rccite{RCxx}` tags in the manuscript.
2. Count `## RCxx` sections in the markdown file.
3. Confirm the ID sets match exactly and are in ascending order.
4. Spot-check that each markdown block quote exactly matches the corresponding highlighted passage.
5. Confirm no scripts or executable resources were added to the skill's workflow or output.

If the user also asks to regenerate PDFs, use the repository's LaTeX build workflow or another appropriate build skill after this skill's edits are complete.

## Final Response

Report:

- The manuscript file edited.
- The markdown file created.
- The number of citation-needing passages marked.
- Whether PDFs were rebuilt, if applicable.
