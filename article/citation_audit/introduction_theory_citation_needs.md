# Introduction and Theory Citation Needs

Status: resolved in `article/wip_v3/manuscript.tex`. The blue `RCxx` tags were removed after adding `KelleyKeyes1998`, `Tofallis2015`, `Boyd2004`, `Sahigara2012`, `Tondel2003`, and `Colson2007` at the relevant statements.

The quoted passages below preserve the wording that was originally audited in the Introduction and Theory and calculation sections.

## RC01

> Time integration can provide an initial state for a steady-state calculation.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This states a general numerical strategy for initializing steady-state process calculations, rather than a result established by this manuscript. The following activated-sludge example is cited, but the broader methodological claim is not. | A numerical process-simulation source showing that dynamic integration or relaxation is used to approach a steady state or initialize a steady-state nonlinear solve. | `("dynamic relaxation" OR "time integration") AND ("steady-state initialization" OR "steady-state calculation") AND (process simulation OR activated sludge)` |

## RC02

> Small absolute errors can be important when the treated-water concentration is low, yet contribute little to a fit over the full plant response. A separate logarithmic model gives this endpoint a relative-error scale without changing the raw multiresponse fit.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| These sentences generalize about the behavior of absolute-error fitting across response scales and use that behavior to justify a separate log-response model. The equations show how the chosen model works, but they do not establish the broader statistical rationale. | A statistical or environmental-modeling source explaining log-transformed positive responses, multiplicative or relative-error behavior, and the risk that low-magnitude responses are underweighted by absolute-error objectives. | `("log transformation" OR "log response model") AND ("relative error" OR "multiplicative error") AND ("water quality" OR concentration OR multiresponse)` |

## RC03

> These scales are chosen to improve numerical conditioning.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| The sentence invokes a general numerical-optimization benefit of scaling constraint rows. The formulation defines the scales but does not itself demonstrate improved conditioning. | A numerical optimization reference explaining how objective, variable, or constraint scaling affects conditioning and solver reliability without changing the feasible set. | `("constraint scaling" OR "row scaling") AND (conditioning OR "numerical stability") AND ("quadratic programming" OR optimization)` |

## RC04

> Regularized feature leverage measures support from the fitted input features.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This assigns a statistical interpretation to regularized leverage as a measure of feature-space support. That interpretation is not derived in the manuscript and is more specialized than the nearby study-specific diagnostic definitions. | A regression or surrogate-model reference defining leverage, including a regularized or ridge form where applicable, and explaining its use for detecting weak support, extrapolation, or influential feature vectors. | `("ridge leverage score" OR "regularized leverage") AND (extrapolation OR "data support" OR "applicability domain") AND (regression OR surrogate)` |

## RC05

> the upper problem remains generally nonconvex and piecewise smooth.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This characterizes the general geometry and differentiability of an optimization problem containing a parametric projection QP. Uniqueness of the lower projection is supported nearby, but nonconvexity and piecewise smoothness of the resulting upper problem are not established there. | A parametric quadratic-programming, differentiable optimization, or bilevel-optimization source describing active-set-dependent piecewise smooth solution mappings and the possible nonconvexity of an upper problem composed with such a mapping. | `("parametric quadratic programming" OR "differentiable optimization") AND ("piecewise smooth" OR "solution mapping") AND (nonconvex OR bilevel OR "active set")` |