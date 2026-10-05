# Extended ICSOR optimization

This repository is the executable companion to the manuscript in
`article/wip_v3`. The canonical study entry point is `main_closed_loop.ipynb`;
the study contract is recorded in `config/parameters.json`.

The study compares two methods for the same recycling activated sludge plant:

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

The article notebook attempts 10,000 fixed Latin-hypercube candidates directly.
Independent seeded streams contain 8,000 model-development candidates and
2,000 descriptive holdout candidates. Rejected candidates remain audited and
are excluded without replacement. The accepted subsets continue through the
analysis. Ten influent scenarios plus the
nominal case use the deterministic robustness design seeded with 314159. The
holdout and scenarios provide descriptive post-selection evidence.

Both methods use the same seven controls, operating bounds, objective,
engineering requirements, and exact-reference comparison. The surrogate uses
active-set projection sensitivities where those audits pass; otherwise it uses
deterministic value-only COBYQA and two-scale feasible no-descent polls.
The smooth NLP retains its three-stage continuation and may use one conditional
recovery from a certified route-S decision after a failed primary solve.
SRT, SOR, and SLR remain descriptive quantities. The retained engineering
safeguards cover underflow TSS, feed TSS, and positive external solids loss.
The physical mass-conservation threshold is `1e-6`.

Only the primary route search is measured. The metric is labeled `Time` and is
reported in seconds. Certification, recovery, exact replay, fitting, and
generation are excluded from Time.

To execute a named resumable run:

This launches the complete production workload and can run for many hours or
days. Use a new run ID for an independent calculation. Reusing the same ID
resumes only artifacts that pass the source, parameter, input, and checkpoint
integrity checks.

```powershell
uv run python -u -m scripts.run_article_v3_5000 `
  --run-id article_full_10000_001 `
  --dataset-count 10000 `
  --through complete
```

Mechanistic candidate attempts are atomically checkpointed by row and reused
after an interruption. Completed downstream stages are also reused only after
their contract and artifact hashes pass validation.

An article result is releasable only when its artifact manifest verifies the
accepted-set provenance, mechanistic and projection audits, the two
optimization routes in every case, exact-reference replay, physical-audit
ledger, required reporting tables, and publication figures.
