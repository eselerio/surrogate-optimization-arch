# Physically constrained surrogate optimization

This repository is the executable companion to the manuscript in
`article/wip_v3`. The canonical study entry point is `main_closed_loop.ipynb`;
the v3 contract is recorded separately in `config/params_manuscript_v3.json`.
The earlier five-control/351-feature workflow is retained only as legacy code
and is not an article-v3 result source.

The v3 calculation uses seven controls, twenty influent coordinates, five
reactors, and a dimension-parametric layered Clarifier. The mechanistic model
retains every Clarifier layer, but the 406-feature ridge surrogate predicts a
161-coordinate operational response: mixer and reactor states, overflow and
underflow component flows, and one total Clarifier-solids inventory. It does
not predict or reconstruct the layer profile. Each raw prediction is
reconciled by an independently audited convex physical projection.
Operational results compare the projected-surrogate route with a separate
smooth mechanistic NLP. Each selected decision is then replayed on the same
exact nonsmooth layered model so that both routes share one physical and
objective reference.

Every analysis retains mass-conservation and non-negativity diagnostics for
raw, projected, optimizer-native, and exact-mechanistic responses. The exact
ledger retains both replay starts when a second start is required.
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

The current article notebook freezes the 16,714 accepted states available from
the interrupted 50,000-target generation: 13,371 development rows and 3,343
post-selection holdout rows. No new mechanistic state is generated. The
mechanistic generator, direct optimizer, and exact replay retain ten Clarifier
layers; only the statistical response is reduced. The runner also preserves
the earlier 5,000- and 50,000-target profiles for explicit historical or fresh
runs. Ten influent scenarios plus nominal use seed 314159. Both optimization
routes now use one deterministic
box-center start per case and search only that local basin; they make no
global-optimality claim. The surrogate route primarily uses analytical
active-set sensitivities in the seven controls. If those derivatives are
unavailable, deterministic value-only COBYQA continues from the same center,
and every distinct fallback trial cold-solves the unchanged exact projection
QP. No finite-difference derivative replaces a failed active-set audit.

The fallback retains the best validated feasible point it visited and cold
replays that point independently. Surrogate convergence has two explicitly
different tiers: a strong first-order certificate from the independent
endpoint active-set and upper-KKT audits, or a finite-resolution certificate
from complete feasible no-descent polls at normalized radii \(10^{-3}\) and
\(10^{-4}\). Protocol v3 uses 14 signed coordinate directions at the coarse
radius and 106 directions at the fine radius: the signed axes, 84 signed
pairwise diagonals, and eight Helmert-simplex directions. After a winning poll
step, exact-QP-feasible trials expand deterministically along that same coupled
ray by factors 2, 4, and so on, for at most 16 probes. These acceleration
trials move the search but never certify it: a fresh complete two-radius poll
at the final endpoint remains mandatory. The shared safety budget is 10,000
exact-QP evaluations. Every completed poll must contain at least one
feasible nonzero displacement; feasible-direction rank is reported only as a
diagnostic because an active constrained feasible cone need not span
\(\mathbb R^7\). Any accepted fine-scale move invalidates the earlier coarse
check and triggers coarse revalidation. This tier is finite-direction,
finite-resolution evidence only, not a KKT, Clarke-stationarity, local-, or
global-optimality proof. An incomplete poll or failed audit leaves a feasible
incumbent with unresolved stationarity.

The surrogate route does not run the former approximately 450-variable
embedded-KKT IPOPT problem or any of its seven gap-continuation stages.

Preliminary v1 poll outputs are archived and excluded from article analysis:
all eleven cases were unresolved, comprising three evaluation-budget outcomes
and eight rank-inconclusive outcomes. Those observations motivated the v2
direction, budget, and feasible-cone correction. A subsequent v2 run showed a
fixed-step fine-poll crawl in robustness case 05 (36 accepted moves before the
2,500-evaluation safety limit), which motivated v3 ray acceleration. Both
predecessor result sets remain archived and excluded rather than reclassified.

The direct smooth-mechanistic route remains an IPOPT NLP and retains its
separate three-stage smoothing continuation. It receives at most one recovery
attempt, only after its primary solve fails and only when that attempt can be
initialized from the same case's certified surrogate decision; a successful
primary direct solve is never given an extra basin search. The nominal case
and all ten robustness cases are still attempted for both routes. An
optimization failure or unresolved audit is recorded casewise and does not
suppress the remaining cases, replay, physical audits, timing aggregation, or
reporting. Runtime comparisons use only the durations recorded for robustness
cases 01--10; no repeated inference or projection benchmark is run over the
frozen post-selection holdout block.

Whole-holdout smooth/reference equivalence is retired and is not an article
admission gate. The 3,343-row post-selection holdout block is used only for
descriptive surrogate prediction assessment. Instead, every available selected
surrogate and direct candidate is replayed casewise from both declared starts
on the exact nonsmooth mechanistic model. A branch-boundary ambiguity is retained as a
qualification and is not by itself a rejection. Comparison eligibility still
requires a valid exact replay, physical and stability audits, and engineering
feasibility. A valid replay's exact objective remains reportable when its
engineering check fails, but that candidate cannot enter a paired comparison.

The earlier nine-start surrogate gap-continuation protocol is retired for
production. The reduced-response revision reuses only verified mechanistic
generation artifacts and their frozen partition. The historical layer-output
fit, assessment, trust calibration, surrogate optimization, replay, timing,
and reports are deliberately not reused; the 161-output surrogate and every
downstream artifact are rebuilt under a new schema.

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
generation. For the current model-function exercise, the revised scientific
admission gates determine article eligibility, but they are advisory for
execution: a failure is recorded and propagated while the
remaining optimization, replay, timing, and reporting paths are attempted
without refitting. A failed gate is never relabeled as a pass. Non-finite or
incomplete numerical objects and run-integrity failures still stop execution.

To execute the full article notebook from a resumable named run directory:

```powershell
$env:ARTICLE_V3_DATASET_COUNT = "50000"
$env:ARTICLE_V3_USE_FROZEN_ACCEPTED_CHECKPOINTS = "1"
$env:ARTICLE_V3_RUN_ID = "article_full_50000_reduced_001"
$env:ARTICLE_V3_REUSE_FROM_RUN_ID = "article_full_50000_003"
uv run jupyter nbconvert --to notebook --execute main_closed_loop.ipynb `
  --output "main_closed_loop.$($env:ARTICLE_V3_RUN_ID).executed.ipynb" `
  --output-dir results\executed_notebooks `
  --ExecutePreprocessor.kernel_name=python3 `
  --ExecutePreprocessor.timeout=-1
```

Every rerun is created in a new self-contained directory. For the reduced
response, the runner byte-copies and hash-verifies only the accepted
mechanistic datasets and generation provenance from the declared source run.
It derives the 161-coordinate responses in the target and refits all
downstream artifacts. For the current revised run, use:

```powershell
$env:PYTHONPATH = "."
uv run python -u scripts\run_article_v3_5000.py `
  --run-id article_full_50000_reduced_001 `
  --reuse-from-run-id article_full_50000_003 `
  --use-frozen-accepted-checkpoints `
  --through complete
```

Generation publishes `all_attempts.csv`, `accepted_provenance.csv`,
`accepted_inputs.npz`, `mechanistic_accepted_v3.npz`,
`accepted_diagnostics.csv`, `base_checkpoint_migration.csv`, and
`replacement_summary.json` separately for the development and test blocks.
The original row checkpoints and their prior assembled artifacts are retained
unchanged.

The historical preflight uses separate scripts and remains article-ineligible;
it is not the default article workload.

An article result is releasable only when its artifact manifest verifies the
complete attempt ledger, exactly 13,371/3,343 frozen rows and their provenance,
all mechanistic and QP audits, trust gates, both optimization
routes for every case, casewise common-reference checks, physical-violation
ledger, and required reporting tables.
