# Recycling mixer–reactor–clarifier activated sludge optimization

This repository contains the executable study for **“Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate.”** The canonical entry points are [main_closed_loop.ipynb](main_closed_loop.ipynb) and [scripts/run_closed_loop.py](scripts/run_closed_loop.py). Both use [config/params_closed_loop.json](config/params_closed_loop.json) and the same staged workflow.

The calculation couples five ASM2d-TSN CSTRs, mixed-liquor and return-sludge recycles, sludge wasting, and a ten-layer secondary Clarifier. It generates 170-coordinate mechanistic targets, fits a fixed 351-feature standardized quadratic response, deploys each prediction through a 77-equality/26-inequality physical QP, and performs nominal and influent-robustness optimization with independent mechanistic reference searches.

## Environment

From the repository root, create the locked Python 3.12 environment:

```powershell
uv sync --frozen
```

The workflow writes only beneath `results/closed_loop/<run-id>`. A run ID binds the configuration and scientific module hashes. Completed runs are immutable.

## Staged 2,000-point verification

The `test_2000` profile exercises every scientific stage with exactly 2,000 mechanistic design points, an ordered 1,600/400 development-assessment split, five robustness cases, and reduced search budgets. It is intentionally marked article-ineligible: a 2,000-point Latin hypercube is not a prefix of the independent 20,000-point article design.

Choose a new run ID, then execute the notebook:

```powershell
$env:CLOSED_LOOP_PROFILE = "test_2000"
$env:CLOSED_LOOP_RUN_ID = "verify_2000_001"

uv run --frozen jupyter nbconvert `
  --to notebook `
  --execute main_closed_loop.ipynb `
  --output "main_closed_loop.$($env:CLOSED_LOOP_RUN_ID).executed.ipynb" `
  --output-dir results\executed_notebooks `
  --ExecutePreprocessor.kernel_name=python3 `
  --ExecutePreprocessor.timeout=-1
```

The equivalent command-line execution is:

```powershell
uv run --frozen python scripts\run_closed_loop.py `
  --profile test_2000 `
  --run-id verify_2000_001 `
  --through complete
```

For explicit checks between long stages, advance the same unsealed run in order:

```powershell
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_2000_001 --through static
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_2000_001 --through pilot
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_2000_001 --through dataset
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_2000_001 --through assessment
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_2000_001 --through complete
```

The pilot stage seals checkpoints at 4, 16, 64, and 256 rows. Dataset generation then resumes without recomputing accepted chunks. Any rejected row stops the workflow and is not resampled.

## Full article calculation

The `full` profile uses the manuscript contract: 20,000 mechanistic design rows, an ordered 16,000/4,000 assessment, 100 independent robustness influents, 25,000 surrogate evaluations per case, a 10,000-evaluation nominal mechanistic reference, and 2,500 mechanistic evaluations per robustness case.

```powershell
$env:CLOSED_LOOP_PROFILE = "full"
$env:CLOSED_LOOP_RUN_ID = "article_closed_loop_001"
uv run --frozen python scripts\run_closed_loop.py --profile full --run-id article_closed_loop_001 --through complete
```

This is a large CPU workload. Run the `test_2000` profile successfully before committing resources to it.

## Tests

The tests are deliberately layered around the same boundaries as the workflow:

```powershell
uv run --frozen python -m unittest discover -s tests -v
```

They cover stoichiometric invariants and kinetics, Clarifier flux and mass closure, loaded and boundary steady states, the exact random design, feature uniqueness and OLS audits, QP/KKT acceptance, deterministic boundary search, artifact round trips, checkpoints, and resume behavior.

## Scientific record

A completed run includes:

- immutable input and generator records;
- checkpointed mechanistic chunks, row diagnostics, and the consolidated dataset;
- ordered split metadata, development and production surrogate bundles;
- raw, affine, and deployed assessment predictions and QP diagnostics;
- nominal and robustness search archives and mechanistic incumbent comparisons;
- generated tables, figures, timing summaries, hashes, and `COMPLETED.json`.

A directory without a valid completion seal is an interrupted or failed run, not a finished result set.
