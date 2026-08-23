# Physically constrained surrogate optimization

This repository is the executable companion to the manuscript in
`article/wip_v3`. The canonical study entry point is `main_closed_loop.ipynb`;
the v3 contract is recorded separately in `config/params_manuscript_v3.json`.
The earlier five-control/351-feature workflow is retained only as legacy code
and is not an article-v3 result source.

The v3 calculation uses seven controls, twenty influent coordinates, five
reactors, and a dimension-parametric layered Clarifier. A 406-feature ridge
surrogate predicts the complete mixer/reactor/outlet/layer response. Each raw
prediction is reconciled by an independently audited convex physical
projection. Operational results compare the projected-surrogate route with a
separate smooth mechanistic NLP and then replay available decisions with the
nonsmooth reference model.

Every analysis retains mass-conservation and non-negativity diagnostics for
raw, projected, smooth-mechanistic, and reference/mechanistic responses.
Rejected mechanistic generations remain in the candidate-attempt denominator
and audit ledger but are excluded from fitted/test datasets and replaced.
Optimization failures and missing endpoints remain in their case denominator.

## Environment

From the repository root:

```powershell
uv sync --frozen
uv run python -m unittest discover -s tests -v
```

CasADi supplies IPOPT/MUMPS for the direct smooth-mechanistic comparator.
OSQP resolves the physical projection QPs used by assessment and by the
seven-variable surrogate optimizer. Scientific artifacts are written below
`results/article_v3/<run-id>`.

## Mandatory preflight

The implementation preflight is deliberately separate from the article
design:

- 400 development plus 100 untouched test inputs;
- five fresh influent robustness scenarios plus the nominal case;
- five Clarifier layers, with feed layer 3 and 1,200 m³ per layer;
- independent seeds 500042, 500043, and 500314159;
- one deterministic box-center optimization start per route and case;
- a 600-second wall-time ceiling for each IPOPT stage used by the historical
  preflight protocol.

Resume completed generation, fitting, and assessment artifacts with:

```powershell
$env:PYTHONPATH = "."
uv run python scripts\resume_v3_preflight.py
uv run python scripts\run_v3_optimization_phase.py --case-limit 6
```

The optimization driver checkpoints every case and completed route. Reporting
can be refreshed safely while a run is incomplete:

```powershell
$env:PYTHONPATH = "."
uv run python -c "from pathlib import Path; from closed_loop.v3_reporting import write_reporting_tables; write_reporting_tables(Path('results/article_v3/test_500_l5_revision_001'))"
```

Preflight values are implementation evidence only. A failed accuracy, QP, or
trust gate is retained and may be followed by diagnostic optimization to test
the code paths, but it can never authorize or populate article results.

## Full article calculation

The article profile uses exactly 5,000 **accepted** mechanistic inputs: 4,000
development and 1,000 untouched test rows, ten fresh influent
scenarios plus nominal, and ten Clarifier layers. Its seeds are 100042,
100043, and 314159. Both optimization routes now use one deterministic
box-center start per case and accept the resulting validated local solution;
they make no global-optimality claim. The surrogate route directly solves the
seven-variable exact-QP active-set problem. It cold-solves and audits the
original projection QP at each distinct trial control and does not run the
former approximately 450-variable embedded-KKT IPOPT problem or any of its
seven gap-continuation stages. The direct smooth-mechanistic route remains an
IPOPT NLP and retains its separate three-stage smoothing continuation. The
nominal case and all ten robustness cases are still attempted for both routes.
An optimization failure is recorded casewise and does not suppress the
remaining cases, replay, physical audits, timing, or reporting.

The earlier nine-start surrogate gap-continuation protocol is retired for
production. This explicit revision reuses the verified 5,000-row generation,
fit, and assessment artifacts in the existing run; it does not restart data
generation or refit the surrogate.

This 5,000-input workload is an explicit user-authorized revision dated
2026-08-23. It supersedes the earlier draft's 800/200 article workload without
changing its seeds or allowing any of the 500 preflight rows into the article
fit, test, or result tables.

Candidate round 0 retains the original independent development/test Latin
hypercubes. If a two-start mechanistic candidate fails any generation check,
the workflow keeps its complete audit, excludes it from the accepted dataset,
and continues that block's stored SplitMix64 stream. Supplemental rounds are
sized to the current deficit and consume 27 midpoint-open coordinates per
candidate in row-major order. Accepted replacements fill failed slots in
ascending order until the two blocks contain exactly 4,000 and 1,000 accepted
rows. Development candidates never cross into the test block.

The completed accepted set is conditioned on mechanistic acceptance and is
not one global strength-one Latin hypercube. Reports therefore use attempted
candidate counts for generation acceptance/failure rates, accepted counts for
fitting and test metrics, and retain the candidate-to-final-slot mapping. Here
"untouched test" means untouched by fitting and tuning; it does not mean an
unconditional sample of the whole input box.

This replacement policy was explicitly authorized after the first development
candidate round had been inspected. Resuming the existing run does not start
from scratch: the runner verifies the old contract and hashes, preserves all
accepted row checkpoints and failed attempts, records the migration, and only
generates the missing replacements. A failed migration verification is a
run-integrity error and is never hidden by recomputation.

The full calculation begins only after the reduced workflow has completed its
implementation checks. An ordinary mechanistic-candidate failure does not stop
generation. For the current model-function exercise, later scientific
admission gates remain unchanged and determine article eligibility, but they
are advisory for execution: a failure is recorded and propagated while the
remaining optimization, replay, timing, and reporting paths are attempted
without refitting. A failed gate is never relabeled as a pass. Non-finite or
incomplete numerical objects and run-integrity failures still stop execution.

To execute the full article notebook from a resumable named run directory:

```powershell
$env:ARTICLE_V3_PROFILE = "article_full"
$env:ARTICLE_V3_RUN_ID = "article_full_5000_001"
uv run jupyter nbconvert --to notebook --execute main_closed_loop.ipynb `
  --output "main_closed_loop.$($env:ARTICLE_V3_RUN_ID).executed.ipynb" `
  --output-dir results\executed_notebooks `
  --ExecutePreprocessor.kernel_name=python3 `
  --ExecutePreprocessor.timeout=-1
```

To resume the already-started default run directly, authorizing the pinned
single-start exact-QP migration while reusing its verified generation, fit,
and assessment artifacts, use:

```powershell
$env:PYTHONPATH = "."
uv run python -u scripts\run_article_v3_5000.py `
  --run-id article_full_5000_001 `
  --through complete `
  --authorize-single-start-exact-qp-migration
```

Generation publishes `all_attempts.csv`, `accepted_provenance.csv`,
`accepted_inputs.npz`, `mechanistic_accepted_v3.npz`,
`accepted_diagnostics.csv`, `base_checkpoint_migration.csv`, and
`replacement_summary.json` separately for the development and test blocks.
The original row checkpoints and their prior assembled artifacts are retained
unchanged.

Set `ARTICLE_V3_PROFILE=test_500_l5` only when intentionally rebuilding the
article-ineligible preflight; it is not the default article workload.

An article result is releasable only when its artifact manifest verifies the
complete attempt ledger, exactly 4,000/1,000 accepted rows and their provenance,
all mechanistic and QP audits, trust gates, both optimization
routes for every case, reference/equivalence checks, physical-violation
ledger, and required reporting tables.
