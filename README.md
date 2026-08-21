# Bilevel optimization of activated sludge operation

This repository contains the reproducible computation for the article **"Bilevel Optimization of Activated Sludge Operation Using a Physically Constrained Machine Learning Surrogate."** The analysis is implemented in [main.ipynb](main.ipynb): it runs ASM2d-TSN, generates the mechanistic data, trains and freezes ICSOR, solves the nominal and robustness optimization cases, validates the selected operations against ASM2d-TSN, and writes the article tables, figures, diagnostics, timings, and row-level data.

The notebook is self-contained with respect to study code; it does not import repository-specific Python helper modules. It does require the versioned configuration, workbook, and dependency lock in this repository.

## Prerequisites

- Windows PowerShell, with the repository root as the working directory.
- [`uv`](https://docs.astral.sh/uv/) available on `PATH`.
- Python 3.12. The notebook deliberately rejects other Python minor versions.
- The tracked ASM2d-TSN workbook at `data/asm2d-tsn/asm2d_tsn_workbook.xlsx`.
- Sufficient local CPU, memory, and disk space. The full study is substantially heavier than the smoke profile.

Create the locked environment once:

```powershell
Set-Location C:\path\to\surrogate-optimization-arch
uv sync --frozen
New-Item -ItemType Directory -Force -Path results\executed_notebooks | Out-Null
```

All commands below must be launched from the repository root because the notebook resolves its inputs relative to that directory.

## Smoke test

The smoke profile executes every stage with reduced sample counts and search budgets. Choose a new run ID each time:

```powershell
$env:SURROGATE_OPT_PROFILE = "smoke"
$env:SURROGATE_OPT_RUN_ID = "smoke_local_001"

uv run --frozen jupyter nbconvert `
  --to notebook `
  --execute main.ipynb `
  --output "main.$($env:SURROGATE_OPT_RUN_ID).executed.ipynb" `
  --output-dir results\executed_notebooks `
  --ExecutePreprocessor.kernel_name=python3 `
  --ExecutePreprocessor.timeout=-1
```

A successful run ends with `status: complete` in `results/<run-id>/manifest.json` and creates `results/<run-id>/COMPLETED.json`. Smoke outputs are always marked `article_eligible: false` and must not be used as article results.

The current notebook and locked environment passed an end-to-end smoke test on 21 August 2026 as `dev_smoke_005`: 160 of 160 mechanistic candidates were accepted, the nominal case and three robustness cases completed, and 109 artifacts were sealed in the inventory. The run took approximately 1 minute 45 seconds on the development workstation; this is a functional check, not a full-profile runtime estimate.

## Full article run

Use a fresh run ID and select the full profile:

```powershell
$env:SURROGATE_OPT_PROFILE = "full"
$env:SURROGATE_OPT_RUN_ID = "article_full_001"

uv run --frozen jupyter nbconvert `
  --to notebook `
  --execute main.ipynb `
  --output "main.$($env:SURROGATE_OPT_RUN_ID).executed.ipynb" `
  --output-dir results\executed_notebooks `
  --ExecutePreprocessor.kernel_name=python3 `
  --ExecutePreprocessor.timeout=-1
```

The full contract includes 10,000 accepted ASM2d-TSN training states, an 8,000/2,000 assessment split, a production refit, one nominal case, and 100 independently generated robustness influents. For every nominal or robustness case, it performs both the surrogate search and an independent mechanistic reference search. The configured ceilings are 60 ICSOR outer sweeps per fit, 10,000 verified surrogate candidates per case, and 2,000 verified mechanistic candidates per case. Dataset generation permits up to 25 attempted candidates per required accepted state. The default full profile uses 12 solver chains/workers.

This workload is intentionally CPU-intensive and may require many hours or longer depending on the workstation and solver behavior. No full-profile runtime should be inferred from the smoke test. Only a successfully sealed `full` run is marked `article_eligible: true`.

## Run IDs, resume, and immutability

Each run writes to `results/<run-id>`. The run contract binds the resolved configuration, workbook, notebook, dependency lock, path configuration, and numerical environment.

- An interrupted run can be resumed by rerunning the same profile and run ID under the identical contract.
- A changed notebook, input, dependency lock, profile, or numerical environment requires a new run ID.
- A completed run is immutable. Reusing its run ID fails deliberately, even if all files are intact.
- Never delete only a marker or individual artifact to force a rerun. Preserve the sealed run and choose a new ID.

Run IDs may contain letters, numbers, dots, underscores, and hyphens, and must begin with a letter or number.

If a run fails after sealing its mechanistic dataset but its notebook or configuration must be corrected,
a new run can import that dataset without repeating simulation. Set
`SURROGATE_OPT_DATASET_SOURCE_RUN_ID` to the failed source run and choose a different active run ID.
The import proceeds only after the dataset and attempt-ledger hashes, sample linkage, workbook hash,
target count, and complete simulation configuration pass validation.

ICSOR coefficient estimation is a self-contained port of the tested recursive-QP implementation in
`icsor-model/src/models/ml/icsor_coupled_qp.py`; its source commit and SHA-256 are frozen in
`config/params.json` and written to the model metadata. The sole intentional algorithmic replacement is
the strictly convex scaled-L2 deployment projection used to obtain a unique constrained response.

## Outputs

The executed notebook copy is written to `results/executed_notebooks/`. The scientific record is the sealed run directory:

```text
results/<run-id>/
|-- COMPLETED.json
|-- manifest.json
|-- manifest.sha256
|-- artifact_inventory.csv
|-- inputs/          # resolved configuration and immutable input snapshots
|-- datasets/        # accepted states, all attempts, completion and validation records
|-- matrices/        # physical operators and matrix validation
|-- splits/          # frozen assessment partition
|-- models/          # B, Gamma, arrays, scales, histories, and QP diagnostics
|-- predictions/     # row-level held-out predictions
|-- metrics/         # accuracy and physical-feasibility diagnostics
|-- timing/          # raw timing repetitions
|-- optimization/    # nominal and per-robustness search archives and solver summaries
|-- validation/      # nominal summary and detailed robustness inputs/results
|-- tables/          # CSV and LaTeX article tables
`-- figures/         # PNG/PDF figures and plotted source data
```

Key files include:

- `models/production_icsor_arrays.npz`, `production_B_long.csv`, and `production_Gamma_long.csv` for the frozen production surrogate;
- `metrics/assessment_accuracy.csv` and `assessment_physical_diagnostics.csv` for predictive and physical assessment;
- `optimization/<case-id>/surrogate_search.parquet` and `mechanistic_search.parquet` for every evaluated candidate;
- `validation/nominal_summary.json` and `robustness_results.parquet` for prediction and decision validation;
- `timing/all_timing_repetitions.csv` and `tables/timing_summary.csv` for training, inference, and reporting timings;
- `artifact_inventory.csv` for the byte size and SHA-256 digest of every sealed scientific artifact.

Treat `COMPLETED.json`, `manifest.json`, `manifest.sha256`, and `artifact_inventory.csv` as a single completion seal. A directory without a valid completion seal is not a finished article run.
