# Extended ICSOR optimization

This repository is the executable companion to the manuscript in
`article/wip_v3`. The canonical study entry point is `main_closed_loop.ipynb`;
the v3 contract is recorded in `config/params_manuscript_v3.json`.

The study compares two methods for the same recycling activated-sludge plant:

- extended ICSOR (route `S`), and
- a smooth mechanistic nonlinear program (route `M`).

Extended ICSOR retains the second-order, 406-feature whole-system ridge
regression of the original surrogate. It extends the invariant-constraint
projection described by Selerio Jr. (2026), *An interpretable statistical
surrogate of activated sludge systems that preserves mass conservation and
component non-negativity*, from an isolated response to the complete closed
recycle system. The 161-coordinate response contains mixer and reactor states,
Clarifier overflow and underflow component flows, and aggregate Clarifier-solids
inventory. It does not predict internal Clarifier layer profiles.

The system-wide projection is cold-solved and audited at every distinct
extended-ICSOR optimization trial. It enforces recycle mixing,
stoichiometric-invariant transport, Clarifier component conservation, soluble
pass-through, particulate densification, aggregate-inventory bounds, and
non-negativity. It does not replace mechanistic kinetic or settling-flux
checks; every selected route-S and route-M decision is replayed on the same
exact nonsmooth layered model.

## Environment

From the repository root:

```powershell
uv sync --frozen
uv run python -m unittest discover -s tests -v
```

CasADi supplies IPOPT/MUMPS for the smooth mechanistic comparator. OSQP
resolves the extended-ICSOR projection QPs. Result artifacts are written below
`results/article_v3/<run-id>`.

## Article calculation

The article notebook freezes 16,714 accepted states from an interrupted
50,000-target generation: 13,371 development rows and 3,343 post-selection
holdout rows. Ten influent scenarios plus the nominal case use the deterministic
robustness design seeded with 314159. The holdout and scenarios are descriptive
post-selection evidence, not confirmatory validation data.

Both methods use the same seven controls, operating bounds, objective,
engineering requirements, and exact-reference comparison. The surrogate uses
active-set projection sensitivities where those audits pass; otherwise it uses
deterministic value-only COBYQA and two-scale feasible no-descent polls.
The smooth NLP retains its three-stage continuation and may use one conditional
recovery from a certified route-S decision after a failed primary solve.

To execute a named resumable run:

```powershell
$env:PYTHONPATH = "."
uv run python -u scripts\run_article_v3_5000.py `
  --run-id article_full_50000_reduced_001 `
  --use-frozen-accepted-checkpoints `
  --authorize-parallel-assessment-migration `
  --assessment-workers 12 `
  --assessment-batch-size 64 `
  --through complete
```

Assessment rows run in deterministic process batches with one numerical thread
per worker. Each completed batch is atomically checkpointed and reused after an
interruption. Worker count may change on restart; changing batch size starts a
new checkpoint geometry.

An article result is releasable only when its artifact manifest verifies the
frozen accepted-set provenance, mechanistic and projection audits, the two
optimization routes in every case, exact-reference replay, physical-audit
ledger, and required reporting tables.
