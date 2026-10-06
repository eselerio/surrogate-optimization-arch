# Writer Agent Guidelines for the Dissertation Manuscript

These guidelines define the preferred writing style, scope discipline, and manuscript hygiene for `article/manuscript.tex`.

## Core Writing Posture

- Write as a dissertation manuscript for an external reader who does not know the codebase, local project folders, draft history, reference manuscripts, or the writer.
- Keep the prose focused on the research topic, not on how the manuscript is being written.
- Remove self-referential scaffolding. Avoid statements about what a section, chapter, writer, article, or dissertation "must" do unless the sentence is part of the formal research problem or scope.
- Avoid roadmap prose inside substantive chapters. A chapter should read as a self-contained study, not as a bridge to another chapter.
- Prefer topic-centered phrasing such as "the study," "the analysis," "the research," "the model," "the surrogate," or the substantive noun being discussed.

## Self-Contained Manuscript Rule

- Do not mention codebase artifacts anywhere in the visible manuscript.
- Do not refer to local paths, folders, archives, manifests, stored prediction files, operator files, scripts, notebooks, logs, or repository structure.
- Do not cite or refer to local manuscripts as evidence. The dissertation may be based on those studies, but it must cite only published literature.
- When describing data or methods, write as if the data, procedure, and analysis belong to the study itself, not as if they were recovered from a project folder.

## Citation and Reference Discipline

- Cite only published or otherwise verifiable scholarly works.
- Verify new references through Crossref or another authoritative bibliographic source before adding them to `references.bib`.
- Do not cite the local journal articles or local draft manuscripts.
- Treat unsupported broad claims as defects. A writer agent should actively look for sweeping statements about wastewater treatment, activated sludge modeling, surrogate modeling, scientific machine learning, physical admissibility, interpretability, preprocessing, validation, or generalization and reinforce them with citations.
- Prefer citations already used in the published-source bibliographies of the reference manuscripts under `references/icsor-manuscript` and `references/surrogates-manuscript`. These local manuscripts are not cited directly; their bibliographies are used as source pools for published references.
- Maximize appropriate citation support without turning paragraphs into citation dumps. Most substantive claims, especially claims about trends, gaps, model capabilities, physical constraints, evaluation practice, or methodological choices, should either be cited or revised into a narrower claim that does not need outside support.
- When adding a citation, integrate it into the discussion by explaining why the cited work matters. The preferred pattern is to name what the reference contributes, then connect it to the dissertation argument.
- Use citations intentionally. Do not attach long strings of citations to broad claims without explaining the contribution of the cited works.
- Prefer `\citet{...}` when authors are part of the sentence and `\citep{...}` when the citation supports the sentence parenthetically.
- Foundational works should be cited when they anchor important concepts, methods, models, or evaluation choices.

## Citation Reinforcement Workflow

- Before finalizing a section, scan for sentences that make generalized claims such as "studies show," "the literature indicates," "machine learning can," "surrogates are useful," "accuracy does not guarantee," "physical constraints are needed," or "evaluation should."
- For each broad claim, first search the two reference manuscripts for relevant published works and their DOI-bearing reference entries.
- Add the published reference to `references.bib` only after checking that it exists and that the metadata are accurate.
- Revise the sentence or surrounding paragraph so the citation is not ornamental. The discussion should make clear whether the citation supports a literature trend, a methodological choice, a scientific limitation, or a foundational concept.
- Use multiple citations when a claim spans multiple ideas, but avoid unsupported bundling. If one sentence makes two claims, either cite both claims properly or split the sentence.
- Keep citation support current with manuscript revisions. If a claim is broadened during rewriting, reassess whether it now needs additional citation support.

## Prose Style

- Use a narrative dissertation style. The prose should develop ideas in connected paragraphs rather than as notes or compressed report fragments.
- Refrain from the information presentation pattern where a general statement is followed by a colon and an enumerated set of ideas.
- Avoid placeholder or instruction-like text, including phrases such as "write this section," "will be explained later," "the review must," or "the article should."
- Avoid equations and mathematical symbols in the Review of Related Literature. Keep the RRL conceptual and narrative.
- Use equations only in method sections when they are necessary for clarity and when prose alone would be insufficient.
- Avoid excessive abbreviation density. Define important abbreviations before use and do not rely on abbreviations when a readable phrase is better.
- Abbreviations must be introduced chapter by chapter. Do not assume that an abbreviation introduced in the Introduction or in an earlier chapter remains available in a later chapter.
- The first use of an abbreviation within each chapter must give the expanded form followed by the abbreviation in parentheses. After that point, the abbreviation may be used within the same chapter.
- If an abbreviation appears only once or twice in a chapter, prefer the expanded form and avoid introducing the abbreviation.
- Keep chapter rationales focused on the problem, gap, and motivation. Do not place detailed methodological decisions in the rationale.

## Required Terminology

- Always use "wastewater treatment," not "wastewater-treatment."
- Always use "activated sludge," not "activated-sludge."
- Always use "machine learning," not "machine-learning."
- Use "nonnegativity" consistently when referring to the physical requirement that component concentrations remain nonnegative.
- Use "post prediction" consistently unless a hyphenated form is required by a cited title or direct quotation.

## Review of Related Literature

- The RRL must be a comprehensive, integrated discussion of the literature, not a list of article summaries.
- Present references intently by explaining what each cited work contributes to the argument.
- Minimize unsupported synthesis in the RRL. Claims about literature trends, research gaps, model behavior, or disciplinary needs should be anchored in published references whenever possible.
- Build the flow from broad wastewater treatment concerns to activated sludge modeling, surrogate modeling, physical admissibility, constraint enforcement, interpretability, and the dissertation gap.
- Do not include equations, symbolic derivations, or implementation details in the RRL.
- Do not describe how the RRL is written or how it should be read.

## Statement of the Problem, Significance, Framework, and Scope

- State the research problem in terms of substantive wastewater treatment and surrogate modeling needs.
- Avoid saying that "the dissertation must" do something. Prefer "the research examines," "a rigorous assessment requires," or a direct statement of the problem.
- The significance section should explain why the work matters scientifically, methodologically, and practically.
- The research framework may use Theory of Change, but it should be written as a research logic, not as a chapter-by-chapter table of contents.
- Scope and limitations should define what the study does and does not claim without apologetic or draft-like language.

## Chapter-Level Writing

- Each substantive chapter should be readable as a self-contained journal-style study.
- Do not assume the reader has read another chapter to understand the purpose, method, or interpretation of the current chapter.
- Because each chapter must be self-contained, abbreviations, model names, metric names, and specialist shorthand must be expanded at first use within that chapter.
- Do not write "the following chapter," "later chapters," or similar cross-chapter scaffolding inside a chapter body.
- Rationale sections should explain the research problem and gap. They should not specify detailed sampling designs, model dimensions, hyperparameter protocols, or file provenance.
- Method sections should provide the procedural detail needed to reproduce the analysis, but without mentioning codebase artifacts.
- Results sections should report findings directly and distinguish empirical results from interpretation.
- Synthesis sections should integrate the findings with the chapter's research question and the relevant literature.

## Chapter 1 Specific Direction

- Chapter 1 focuses on machine learning surrogates and their raw predictions.
- The chapter should establish whether modern tabular machine learning surrogates can learn the activated sludge simulator response from paired inputs and outputs.
- Raw predictions should be discussed as unadjusted model outputs.
- Do not evaluate mass conservation, nonnegativity, projection, or physical correction as Chapter 1 endpoints.
- Composite water quality quantities may be discussed as derived summaries of component predictions, not as replacements for component-level prediction.

## Method Writing

- Describe the data-generation, modeling, validation, and evaluation protocols as study methods.
- State sampling, solver, preprocessing, tuning, and evaluation rules clearly when they belong in the method.
- Do not mention that data came from a folder, archive, manifest, script, or prior code execution.
- Keep leakage control explicit. Transformations, hyperparameter selection, and scoring references should be described in relation to training and validation partitions.
- Distinguish in-distribution evaluation from beyond-domain evaluation without treating either as a code artifact.

## Final Manuscript Hygiene Checks

- Search for and remove banned forms: `wastewater-treatment`, `activated-sludge`, and `machine-learning`.
- Search for and remove codebase artifact terms in visible prose, including `archive`, `manifest`, `repository`, `codebase`, `workspace`, `path`, `file`, and local manuscript references.
- Search for placeholder terms such as `TODO`, `Obligatory`, dummy keywords, and section-writing instructions.
- Search for self-referential roadmap language such as `this chapter`, `following chapter`, `the dissertation must`, and `will be explained later`.
- Search for sweeping unsupported claims and reinforce them with citations from published works, prioritizing the published references already present in the two reference manuscripts.
- Check that all citation keys in `manuscript.tex` exist in `references.bib`.
- Build the PDF after substantive edits and report the generated PDF path.
