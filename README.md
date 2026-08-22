# Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate

This repository contains the executable study for **“Optimization of a Recycling Mixer-Reactor-Clarifier Activated Sludge System Using a Physically-Constrained Statistical Surrogate.”** The canonical entry points are `main_closed_loop.ipynb` and `scripts/run_closed_loop.py`; both read `config/params_closed_loop.json` and execute the same staged workflow.

The process model couples an inlet mixer, five ASM2d-TSN CSTRs, mixed-liquor and return-sludge recycles, sludge wasting, and a ten-layer secondary Clarifier. A fixed 351-feature quadratic surrogate predicts 170 plant-state coordinates. One sparse nonlinear program jointly optimizes five controls and 110 reactor-and-Clarifier states, reconstructs the complete response, imposes the smooth mechanistic balances, and requires both conformal fidelity and development support. One exact nonsmoothed BDF replay validates the selected decision.

## Environment

Create the locked Python 3.12 environment from the repository root:

```powershell
uv sync --frozen
```

CasADi 3.7.2 supplies automatic derivatives and its bundled IPOPT/MUMPS solver. Scientific outputs are written only below `results/closed_loop/<run-id>`. A run ID binds the configuration, source hashes, dependencies, and random-generator records; a sealed run is immutable.

An unsealed run may be resumed. Before reuse, every completed NLP-start and exact-replay artifact is checked against its case, selected candidate, dimensions, and stored digest. Verified artifacts are reused without another scientific solver call, and the persistent invocation record—not an inferred case multiple—supplies realized NLP and BDF counts. Exact fidelity is judged on the same normalized constraint scale as the NLP: `d_BDF / delta - 1` must not exceed the configured normalized feasibility tolerance (currently `1e-8`). Physical projection QPs and DIRECT searches have no executable route and therefore always have zero evaluations.

## Staged 2,000-row verification

The `test_2000` profile uses independent blocks of 1,400 development, 200 calibration, and 400 untouched assessment rows. It runs the nominal case, ten fresh-influent robustness cases, and twelve sensitivity cases. Each case uses nine deterministic starts of the combined NLP and permits one exact validation replay, for a maximum of 207 NLP starts and 2,023 BDF routes. Verification data use streams distinct from the full study and are never reused as article results.

The lightweight `unit` profile uses 420/60/120 rows and one robustness case. Its 14 cases require 126 combined-NLP starts and at most 614 BDF routes.

```powershell
$env:CLOSED_LOOP_PROFILE = "test_2000"
$env:CLOSED_LOOP_RUN_ID = "verify_combined_2000_001"

uv run --frozen jupyter nbconvert `
  --to notebook `
  --execute main_closed_loop.ipynb `
  --output "main_closed_loop.$($env:CLOSED_LOOP_RUN_ID).executed.ipynb" `
  --output-dir results\executed_notebooks `
  --ExecutePreprocessor.kernel_name=python3 `
  --ExecutePreprocessor.timeout=-1
```

The equivalent command-line run is:

```powershell
uv run --frozen python scripts\run_closed_loop.py `
  --profile test_2000 `
  --run-id verify_combined_2000_001 `
  --through complete
```

For inspection between expensive stages, advance the same unsealed run in order:

```powershell
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through static
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through pilot
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through dataset
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through fit
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through calibration
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through assessment
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through nlp_preflight
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through optimization
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through report
uv run --frozen python scripts\run_closed_loop.py --profile test_2000 --run-id verify_combined_2000_001 --through complete
```

The pilot persists checkpoints after 4, 16, 64, and 256 accepted design rows. A failed mechanistic row is not replaced. Later stages likewise stop at their declared numerical or scientific gate rather than silently changing the method. The `report` stage writes an immutable summary with `release_status="provisional_pending_terminal_replay"` and `release_authority="COMPLETED.json is created only after terminal replay and sealing"`. Its tables and figures become final scientific outputs only after terminal algebraic, KKT, and exact-state replay succeeds and the `complete` stage writes the immutable seal and `COMPLETED.json`, where `report_release_status="terminally_sealed"` records release authorization.

## Full article calculation

The `full` profile uses 14,000 development, 2,000 independent calibration, and 4,000 untouched assessment rows. It contains 113 optimization cases: nominal, 100 robustness, two additional Clarifier underflow-TSS limits, and ten objective-weight sensitivities. The planned maximum is 1,017 combined-NLP starts and 20,113 BDF routes, with no physical QPs or DIRECT evaluations.

```powershell
uv run --frozen python scripts\run_closed_loop.py `
  --profile full `
  --run-id article_combined_001 `
  --through complete
```

Run and review `test_2000` before allocating resources to the full calculation.

## Tests

```powershell
uv run --frozen python -m unittest discover -s tests -v
```

The tests cover the kinetic and stoichiometric model, Clarifier closure, independent random-design blocks, feature serialization and OLS audits, conformal calibration, smooth symbolic parity, combined-NLP IPOPT/KKT acceptance, multistart selection, checkpoints, exact replay classification, and immutable resume behavior.

## Scientific record

A completed run contains generator states and draw counts; checkpointed mechanistic data; the development-only surrogate; calibration and assessment records; all combined-NLP starts; independent KKT replays; exact BDF validation states; robustness and sensitivity summaries; timings, hashes, figures, and a completion seal. A directory without a valid seal is an interrupted or failed run, not a finished result set.
