---
name: revise-manuscript-summary-sections
description: Revise a LaTeX manuscript's Highlights, Abstract, and Conclusion so they synthesize the most important results with a readable narrative for a broad audience. Use when Codex is asked to identify the main findings of a study and rewrite front-matter or end-matter summary sections, especially requests to make results clearer, more concise, more general-public-readable, more narrative, or better aligned with the Results and Discussion.
---

# Revise Manuscript Summary Sections

## Purpose

Use this skill to turn a manuscript's strongest results into a coherent story across three high-visibility sections:

1. Highlights
2. Abstract
3. Conclusion

This is an editorial synthesis workflow, not a script-driven task. Do not add bundled scripts or generated resources for this skill. Read the manuscript, reason from its evidence, and edit the LaTeX source directly.

## Evidence Gathering

Before editing, inspect enough of the manuscript to know what the study actually shows.

Read, at minimum:

- the current Abstract and Highlights;
- the Introduction or final gap/objective paragraph;
- the Methods overview or evaluation-design passages needed to understand the approach;
- the full Results and Discussion section;
- the current Conclusion;
- captions or tables that contain the key metrics cited in Results and Discussion.

Use the manuscript's own Results and Discussion as the evidence source. Do not invent metrics, claims, sample sizes, or implications. Do not add outside references unless the user separately asks for citation discovery or reference editing.

## Finding The Most Important Results

Rank findings by their role in the paper's argument, not by numerical size alone.

Prefer results that:

- answer the central research question;
- change how the reader should interpret the problem;
- hold across the broadest experimental scope;
- distinguish the work from prior or ordinary practice;
- have practical implications for interpretation, design, optimization, control, policy, or future use;
- clarify a tradeoff or boundary condition, such as when a method improves one criterion but not another.

Treat statistical metrics as anchors for the story. Include exact values when they make the claim credible or concrete, but do not let the section become a list of numbers.

Good pattern:

```text
Projection made every evaluated prediction physically admissible, with maximum residuals near numerical precision.
```

Then add metrics:

```text
All 766,350 projected predictions conserved mass and remained non-negative.
```

Avoid the reverse, where the paragraph is driven by a pile of metrics before the reader knows why they matter.

## Writing Principles

Write for an educated non-specialist whenever the user asks for general readability.

- Lead with the problem in plain language.
- Explain why the result matters before naming technical details.
- Use active, direct sentences.
- Define or soften unavoidable jargon on first use.
- Prefer "physical rules" or "physically admissible state" over unexplained technical shorthand, unless the manuscript's audience requires the exact term.
- Keep acronyms only when they are already standard in the manuscript and useful.
- Avoid overclaiming: distinguish demonstrated results from implications, scope, and future possibilities.
- Preserve the manuscript's scientific precision even while simplifying the surface prose.

## Highlights

Revise the `highlights` environment or equivalent journal highlight list.

Requirements:

- Use at most five bullets.
- Keep each bullet brief and self-contained.
- Make each bullet a result or implication, not a task description.
- Avoid starting every bullet with the same grammatical shape.
- Prefer narrative findings over methods inventory.
- Include a metric only if it strengthens the bullet without making it feel crowded.

For Elsevier-style LaTeX, preserve the structure:

```latex
\begin{highlights}
\item ...
\item ...
\end{highlights}
```

## Abstract

The abstract should be concise and complete. It should contain:

1. why the study was conducted;
2. what approach was taken;
3. the most important results;
4. the main implication or takeaway.

Make the abstract readable as a short story:

- problem;
- intervention or comparison;
- decisive result;
- nuanced result, if important;
- practical meaning.

Do not overfill the abstract with every model, table, or secondary metric. Use only the metrics needed to substantiate the central findings.

## Conclusion

Write the conclusion as an elaborate synthesis, not a repeat of the abstract.

A strong journal-article conclusion should:

1. restate the problem the study addressed;
2. explain why the problem matters;
3. summarize the approach at a high level;
4. identify the central findings in order of importance;
5. interpret what the findings mean;
6. acknowledge important nuance, tradeoffs, or limits;
7. state the practical or scientific implication without exaggeration.

Use paragraphs rather than bullets unless the journal style or user request calls for bullets. The conclusion may be longer than the abstract, but it should still move cleanly from problem to evidence to implication.

Avoid:

- introducing new results not present in Results and Discussion;
- claiming field or real-world validity from simulated-only evidence;
- implying that better feasibility automatically means better predictive accuracy;
- ending with a vague generic sentence that could fit any paper.

## LaTeX Editing Procedure

1. Locate the target sections with search, for example `\begin{abstract}`, `\begin{highlights}`, and `\section{Conclusion}`.
2. Draft the revised prose from the evidence hierarchy before editing.
3. Edit only the requested sections unless a neighboring sentence must change for consistency.
4. Preserve LaTeX syntax, math notation, escaped characters, labels, and journal environments.
5. Keep section order and front-matter structure intact.
6. Do not remove existing citation-needed markup, comments, or macros outside the edited sections unless the user asks.
7. If the manuscript contains citation-needed highlights inside a section being rewritten, either preserve them where still applicable or remove them only if the revised sentence no longer contains that claim.

## Verification

Before finishing:

- Confirm the Highlights contain no more than five `\item` entries.
- Confirm the Abstract names the reason, approach, main results, and implication.
- Confirm the Conclusion includes synthesis, nuance, scope, and implication.
- Check that all numeric claims in the revised text appear in the manuscript's Results and Discussion or its cited tables/figures.
- Run a LaTeX build when the user asks for a PDF or when the repository has an established build workflow and the edit may affect compilability.
- If a build is run, report the output PDF and any nonfatal warnings that matter.

## Final Response

Report:

- the manuscript file edited;
- the sections revised;
- the main narrative emphasis introduced;
- whether the PDF was rebuilt, if applicable.
