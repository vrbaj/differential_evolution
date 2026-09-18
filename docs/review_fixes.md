# Review resolution

This records the disposition of `review/REVIEW.md`. The review reports bugs,
scientific documentation gaps, release decisions, and proposed future features.
The original reports and reproduction scripts are preserved unchanged.

## Correctness and API changes

| Findings | Resolution |
|---|---|
| C1 | Tent uses independently seeded coordinate chains, refreshed before binary precision collapses them. |
| C2 | SHADE adaptation ignores non-finite improvements, rescales weights to avoid overflow, and guards invalid sampling locations. |
| C3 | All 13 fixed-dimensional objectives reject inputs other than two coordinates. Published optimum regressions cover all 15 benchmarks. |
| C4, D6, D7 | Duplicate diversity names are rejected; iterable inputs are not compared to strings; measures must be callable. |
| H1 | Negative infinity is preserved; NaN becomes positive infinity. |
| H2 | CurrentTo operators reuse the resolved F unless a separate difference scale is specified. |
| H3 | Messages and metadata name the exhausted limit; no usable solution gives success=False. Success means a usable objective value was found, not mathematical convergence. |
| H4, M9 | Every objective call obeys the budget, including initialization. Ranked initializers reuse survivor fitness: 2*NP evaluations, not 3*NP. An insufficient initialization budget raises before overspending. |
| H5 | Exceptions propagate by default. Explicit objective_errors='penalize' converts ValueError/ArithmeticError to positive infinity and counts failed calls. Programming errors are not swallowed. |
| H6 | DirectedMutation is explicitly a project variant. Normalized finite fitness gaps replace unstable dimensional coefficients; fallback counts appear in result metadata. The previous paper attribution is withdrawn. |
| H7 | Sobol uses a seed-dependent digital shift. scramble=False retains the known unshifted sequence. Limits remain 40 dimensions and 2**30 points. |
| H8 | Mutation/crossover and their controllers are deep-copied together per optimizer. Inspect adapted state through the optimizer, not the original component bundle. Custom components must support deepcopy. |
| H9 | Plain DE defaults to clipping. SHADE/L-SHADE default to midpoint repair. Explicit NoBoundaryHandler still permits unconstrained trials. |
| H10 | JSON exports represent non-finite numbers as strings ('inf', '-inf', 'nan'); exports prohibit nonstandard numeric JSON constants. |
| M1, M2 | Corrected p-best/r1/r2 exclusions, ties, archive insertion limits, SHADE arithmetic CR versus L-SHADE Lehmer CR/terminal memory, and fixed L-SHADE p=0.11. r2 reads the generation snapshot during a step. |
| M3, M4 | Best-family random difference indices exclude only the target. Built-in minimum populations and schedule minima are validated before objective evaluation. |
| M5 | Default tanh interval is [-3, 0], matching slow early/fast late reduction. |
| M6 | Removed redundant quasi-opposition branch; distribution remains uniform between midpoint and opposite. |
| M7 | Documented that published trigonometric mutation depends on objective offsets; its formula is intentionally retained. |
| M8 | run() is single-use. population_size remains configuration; current_population_size reports current state. |
| M10 | L-SHADE warns if no evaluation budget is supplied; generation-based reduction remains an explicit compatibility variant. |
| D1, D2, D4, D5 | Structural interfaces now annotate optimizer components, controllers, RNGs, and optional lifecycle hooks. Added py.typed, construction-time crossover-controller signature validation, and standalone adaptive initialization. |
| D3 | Optional repair(trial, target, bounds, rng) hook supports parent-aware handlers without breaking existing three-argument callables. Added midpoint and reflection handlers. |

Non-finite improvements are excluded from adaptation rather than assigned an
arbitrary finite penalty. Tied SHADE trials survive and enter the archive but do
not update memories. Plain DE retains strict selection.

## Performance, validation, and packaging

| Findings | Resolution |
|---|---|
| P1 | Hoisted exclusion-set construction out of the sampling comprehension. Sampling still constructs an eligible-index list; no unverified speedup claim is made. |
| P2 | Cached dataclass field discovery by component class. Controller values are still read from the instance to allow explicit replacement. |
| P3 | Pairwise distances are shared across the three built-in pairwise statistics per generation. Diversity remains opt-in and in coordinate units, not box-normalized. |
| P4 | SHADE r2 uses index rejection sampling instead of materializing tagged population/archive candidates. |
| P5 | snapshot_interval controls thinning. Full snapshots still cost O(NP*D) each; streaming is a future feature. |
| P6 | Result serialization performs one recursive dataclass conversion. |
| T1–T3 | Added initializer, bounds, history/budget, reproducibility, memory-formula, non-finite objective, controller isolation, and benchmark-optimum regressions. |
| K2–K4 | Added the already-declared MIT license, packaged all compatibility shims, and replaced local absolute documentation links. |
| K5–K6 | Added an OS/Python CI matrix, lint/type checks, test dependencies, project URLs/keywords/classifiers, and a typing marker. Kept the declared Python >=3.11 support range. |
| K7 | Removed pickle loading from GPD evaluation. A data-bound objective factory loads observations once; the compatibility path accepts cached plain-text data, not pickle. Parameter-domain violations return infinity. |
| K8 | Generated API stubs are ignored and generated during documentation builds; removed the missing template path and ignored build/checker outputs. Historical audit/migration documents remain at their existing linked paths. |

## Decisions and future work

- **K1:** Distribution naming remains a release-owner decision. The review reports
  that `differential-evolution` is occupied; this change does not claim ownership
  or availability and does not publish anything. Choose and verify a distribution
  name before release; the Python import name can remain unchanged.
- **D8:** Batch/parallel objectives, stopping policies, callbacks, constraints,
  mixed variables, strategy pools, restarts, islands, and JADE are feature
  proposals, not implemented fixes. The package remains a serial, box-bounded,
  continuous minimizer with generation/evaluation limits.
- **T4:** A publication-quality CEC/BBOB benchmark campaign is still separate work.
  Known-optimum and algorithm-formula regressions do not establish comparative
  performance or paper-level replication.
- **H6 bibliography:** No claim is made that the normalized project variant matches
  the inaccessible directed-mutation paper. Verifying that historical source is
  unnecessary for the explicitly project-specific formula now documented.
- **K5/K6 editorial recommendations:** Citation identifiers, author contact/ORCID,
  a conduct policy, and publication metadata need actual project-owner data.
  No such identity information has been invented.

## Verification

Run `python -m unittest discover -s tests -v`, `ruff check differential_evolution`,
and `mypy differential_evolution`. Build documentation with Sphinx `-W` and build
and smoke-test an installed wheel outside the checkout.

The review scripts are diagnostic artifacts, not a reliable regression suite:
C4's correct rejection stops repro.py, M6 recomputes old source-independent
formulas, M7 demands a property the published operator does not have, P6 treats
identical serialization results as proof of duplicate computation, and M10
ignores the permitted warning-based resolution. Regression tests assert the
intended behavior directly instead of altering those reports to turn them green.

Validation completed for this change: **87 tests passed**, including the same
suite against the installed wheel from outside the checkout; Ruff and mypy
passed; Sphinx built with warnings treated as errors. The wheel includes the
compatibility modules, MIT license, and py.typed marker. Local validation used
Python 3.14; CI covers the declared 3.11–3.13 matrix on Linux, macOS, and Windows.
