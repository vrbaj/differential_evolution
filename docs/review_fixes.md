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
arbitrary finite penalty. Tied SHADE trials survive but do not enter the archive
or update memories. The earlier tie-archiving policy was corrected in round 5
below. Plain DE retains strict selection.

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

## Matous round 3 (2026-10-03)

The 87-test validation above describes the earlier change, not the current suite.

| Finding | Status and root cause |
|---|---|
| 1: evaluation budget cut short | Fixed. The inherited default of 100 generations stopped budget-driven runs early. `max_generations` now defaults to `None`; at least one limit is required, and explicit dual limits still stop at the first exhausted limit. README examples omit the premature generation cap. |
| 2: non-finite NSDE/tanh parameters | Fixed. Ordered comparisons alone did not reject NaN, and the NSDE mean had no validation. All NSDE float parameters and both tanh endpoints must be finite. Saturated tanh endpoints with identical mapped values are also rejected to prevent division by zero. The shared trial-repair path rejects non-finite coordinates before objective evaluation. |
| 3: missing round-2 regressions | Already resolved in the working tree: the pre-existing `tests/test_message_regressions.py` covers round-2 fixes 1, 2, 3, 4, 6, 7, and 14, plus exception cleanup and the intentional TypeError policy. Verified without changing that file. Added six round-3 regression tests separately. |
| 4: standalone adaptive crossover | Fixed. Lazy initialization sized state only for the first requested index. Later indices now extend state with the initial rate, preserving pending and committed values. |
| 5: legacy one-factor current-to mutation | Fixed. The adapter unconditionally indexed the second factor. All four current-to strategies now use the modern shared-factor default when only one factor is supplied. |

The round-2 TypeError claim is not a bug: penalty mode intentionally catches
ValueError/ArithmeticError, while programming errors propagate (H5 above).
Partial-generation counting remains documented behavior. Round-2 performance,
API-export, lint-scope, packaging, and cosmetic suggestions are unchanged;
these are not required for the confirmed round-3 correctness fixes.

Validation for round 3: `python3 -m unittest discover -s tests` passed **101 tests**;
`python3 -m doctest docs/*.rst README.md MIGRATION.md` passed;
`git diff --check` passed. Ruff and mypy could not run because those modules are
not installed in the current environment. No wheel or Sphinx build was run for
this change. Existing staged review text and untracked paper assets were preserved.

## Matous round-2 finding 11: checker scope

Fixed the package-only CI scope: Ruff now runs `python -m ruff check .`,
and `python -m mypy` reads the expanded configured targets (package, three
distributed compatibility modules, examples, tests, and Sphinx configuration).
Historical `review_*` diagnostic artifacts and the unrelated `paper_template`
draft are excluded from Ruff. No lint rules were disabled.

Removed the unused example import, sorted compatibility exports, annotated
legacy initializer helpers and empty Sphinx settings, and retained all test
assertions while resolving comprehension, adjacent-pair, and closure findings.
Mypy still uses its default policy for unannotated function bodies; the legacy
initializer helpers are now annotated so their bodies are checked.

Validation: Ruff 0.16.10 passed; mypy 2.4.0 passed across 29 source files;
103 unittest tests, documentation doctests, five legacy initializer smoke checks,
and `git diff --check` passed on Python 3.12. The CI OS/Python matrix was not
executed locally. Tools were installed in `/tmp`, without changing dependencies.

## Direct implementation review fixes

All seven findings were reproduced against the current code and addressed:

| Finding | Resolution |
|---|---|
| 1: infinite initial coordinates | Added overflow-safe interpolation, uniform sampling, midpoints, and opposites for all built-in initializers. Initial populations must contain finite coordinates inside the bounds; non-finite inputs are rejected before objective calls, including calls made by ranked initializers. |
| 2: repair introduces infinity | Midpoint arithmetic avoids overflowing sums. Reflection uses exact rational arithmetic only when its floating-point intermediate values overflow. Repaired vectors are validated for dimension and finiteness before evaluation. |
| 3: GPD numerical overflow | Divide each observation before summing in the exponential limit; divide log1p terms by shape rather than forming its reciprocal. Exceptional overflowing/underflowing products and ratios use a high-precision decimal fallback. |
| 4: non-finite archive rates | SHADE and L-SHADE reject non-finite archive rates during construction, before evaluations. |
| 5: quadratic index-list construction | Sparse mutation samples use compact ranks and exclusion mapping; small/dense requests retain the existing list path. SHADE/L-SHADE r1 sampling maps a rank past the target. Seeded sampling matches the original candidate-list implementation. |
| 6: legacy coherence always zero | A private legacy optimizer retains the preceding population, allowing the wrapper to calculate coherence on demand without evaluating diversity during every run. |
| 7: regression files absent from Git | Added the three previously untracked regression files to the Git index, together with the new direct-review regression suite. The unrelated paper draft remains untouched. |

Validation on Python 3.12: 116 tests passed; Ruff and mypy passed (31 source
files); documentation doctests, both maintained examples, and diff checks passed.
The indexed test files reproduce the full 116-test suite. The deterministic SHADE
fixture now specifies the compact r1 rank that maps to the same original index;
its archive/memory assertions are unchanged.

At N=2000, five generations, two-dimensional sphere, profiled DE sampling time
fell from 0.531 s to 0.075 s and SHADE r1 sampling from 0.406 s to 0.042 s.
These local measurements are diagnostic, not cross-platform benchmark guarantees.
No wheel/Sphinx build or CI OS/Python matrix was executed for these fixes.

## Code review round 4 (2026-10-05)

All nine findings in `review_04_honza/codex_fix_tasks.md` were confirmed against
the current code. None was already resolved. F8 identifies a missing warning
about a deliberate scheduling rule; the rule itself remains unchanged.

| Finding | Resolution |
|---|---|
| F1: missing custom mutation example | Fixed the literalinclude start marker to match the example import. A regression checks every top-level RST literalinclude target and its start/end markers; the built HTML contains `ShrinkMutation`. |
| F2: generic installed root modules | Removed the setuptools `py-modules` declaration. Root shims remain available from a checkout; README and MIGRATION describe installed imports through `differential_evolution.compat` and `differential_evolution.benchmarks`. A clean wheel excludes all three root modules and passes isolated modern/legacy import and execution checks. This supersedes the earlier K2–K4 decision to distribute the root shims. |
| F3: examples without stopping limits | Added `max_generations=100` to both component-composition snippets. An AST-based regression checks optimizer calls with `objective=` in RST Python blocks, doctests, README, and MIGRATION for a stopping limit. |
| F4: diversity classes accepted as instances | Reject classes with an actionable TypeError during construction. Tests cover direct and list specifications with zero objective calls, and retain support for instances and aliases. |
| F5: standalone adaptive scale growth | Reject negative target indices and extend per-target state without resetting pending or committed proposals. Tests cover growth from four to eight members, acceptance/rejection, repeated proposals, and unchanged RNG state when reusing proposals or rejecting an index. |
| F6: L-SHADE warning location | Adjusted the stack level so the generation-only warning points to the caller; regression checks its filename. |
| F7: empty legacy mutation factors | Raise a descriptive ValueError before indexing an empty factor list. Regression covers all eight legacy strategies while expecting their deprecation warning. |
| F8: generation limit dominates reduction budget | Added the requested dual-limit warning and documented its one-population initialization assumption. Warning locations are correct for DE, SHADE, and L-SHADE. Tests verify the reported example still ends at 50 members after 200 generations, plus cases that should not warn. No progress, stopping, or optimization formulas changed. |
| F9: documentation drift | Documented snapshot intervals and objective exception policy in dataclass field order, custom mutation minimum-population validation, and all five boundary handlers with their defaults. |

Validation on Python 3.14.7:

- `python -m unittest discover -s tests -v`: **125 tests passed** (116 existing
  tests plus nine new tests in `tests/test_round4_regressions.py`). Running the
  new tests against a temporary copy of the original code reproduced the defects.
- `python -m doctest docs/*.rst README.md MIGRATION.md`: passed.
- `python -m ruff check .` and `python -m mypy`: passed (34 source files).
  Ruff/mypy and build tools were installed in `/tmp`; no project dependencies
  were changed. These checks used `PYTHONPATH=/tmp/round4-tools`.
- All four requested `PYTHONPATH=. python examples/<name>.py` commands passed:
  `basic_usage`, `custom_components`, `diversity_measures`, and
  `population_initialization`.
- `PYTHONPATH=.docdeps:. python -m sphinx -W -b html docs docs/_build/html`:
  passed using existing local documentation dependencies.
- `python -m pip wheel --no-deps --no-build-isolation` succeeded using the
  build tools in `/tmp`. The first build exposed stale root modules in the
  existing ignored `build/` directory. A clean source copy in `/tmp` produced
  the verified wheel, preserving existing checkout build artifacts. Release
  wheels must be built from a clean build directory. Isolated installed-wheel
  smoke tests passed with checkout imports disabled.
- The requested seeded DE/L-SHADE check was identical before and after:
  `4.247056040185271e-10 1.3534418030758388e-10`.
- Final diff inspection and `git diff --check` passed. Historical review
  artifacts and unrelated untracked files were preserved.

No requested local check was unavailable. The CI Python/OS matrix was not run
locally. The F8 warning is a configuration heuristic: ranked or custom
initializers may consume more evaluations than its initialization estimate.

## Code review round 5 (2026-10-07)

Rechecked every item in `review_05_yuwen/REVIEW04.md` against the current code,
the 2014 paper, and the released L-SHADE 1.0.1 source. The review's baseline had
116 tracked tests; the nine round-4 tests existed only in the local workspace.
The earlier 125-test result was a local result, not a clean-checkout count.

| Finding | Resolution |
|---|---|
| R1: regressions absent from Git | Fixed by adding the existing nine round-4 tests and the new round-5 regression file to the index. Added release guidance to check untracked tests and quote test counts from a clean release checkout. No commit or push was created. |
| R2: ties archive parents | Fixed. Ties still replace parents, but only strict fitness improvements archive them, including improvements to negative infinity. Non-finite improvements remain excluded from adaptation. Constant-objective and infinite-fitness regressions cover both solvers. This supersedes the earlier tie-archiving policy. |
| R3: archive visible within generation | Fixed. Improved parents are buffered until all processed trials are generated. Real r2-sampler tests cover complete/partial generations and next-generation visibility; failed generations do not publish buffered parents. Archive eviction policy is otherwise unchanged. |
| R4: ranked-initializer budget preflight | Not a bug. Retained the deliberate guarded-evaluation policy and its existing regression: an insufficient budget is consumed, then the next attempted call raises without overspending. Documented the candidate-pool budget requirement; no new initializer interface was added. |
| R5: mutation overflow | Fixed for built-in donor arithmetic with a fallback to exact finite-input arithmetic on overflow. Truly unrepresentable donors become signed infinities and can be repaired by clip/midpoint/random-reset handlers. NaNs remain invalid; reflection explicitly rejects non-finite input and all repaired vectors must be finite. Tests cover every built-in mutation, 20 seeds and 20 generations, plus both SHADE variants. The earlier pre-repair infinity rejection test was updated to assert the chosen repair policy and unchanged objective-call safety. |
| R6: historical Sobol link | Fixed. README now links the maintained algorithm reference and describes digital shifting, scramble=False, and sequence limits. Historical audit text is unchanged. |
| R7: halfway rounding | Confirmed difference, retained as an explicit compatibility policy rather than a bug. Added halfway regressions for p-best size, population reduction and archive capacity. Documented nearest-even rounding versus released C++ rounding/truncation; no claim of trajectory equivalence. |
| R8: diversity overflow | Fixed. Stable Euclidean norms, overflow-safe means, per-dimension scaled variance, coherence and histogram arithmetic handle finite extreme coordinates. Shared optimizer diagnostics agree with standalone measures. Tests cover initialization/full runs, representable means of unrepresentable distances, mixed coordinate scales and genuinely infinite variance. |
| N1: default provenance | Not a bug. Documented the L-SHADE paper's tuned defaults versus released code; population size remains caller-supplied. |
| N2: terminal CR persistence | Not a bug. Retained the paper's absorbing state and documented the C++ discrepancy. Added a memory-wraparound regression for successful CR sets [0], [.8], [.8]. |
| N3: reduction order/ties | Not a bug. Retained stable fitness ordering and earlier tied survivors; regression tests pin both reported cases. Documented released C++'s different ordering/tie-removal policy. |
| N4: SHADE versions | Not a bug. Explicitly distinguished the library's 2013 SHADE scheme from SHADE 1.1.1 in the reference documentation. Existing memory-formula tests remain intact. |
| N5: RNG interface | Not a bug. Documented required Python-random-style methods and the need for a NumPy Generator adapter; added no dependency or implied native NumPy support. |
| N6: ranking/checker scope | Already resolved. The shared ranking and existing generation/partial-generation tests remain; full-repository Ruff and configured mypy checks passed. Developer instructions now show the full CI commands. |

Also documented all remaining reference differences from section 2: archive
append/delete versus overwrite, rounded/random archive reduction versus
truncation, survivor ordering, strict actual-call budgets versus an uncounted
tail, and competition-only optimum/epsilon truncation. These are explicit
policies, not performance claims. SHADE/L-SHADE seeded trajectories change due
to R2/R3. Ordinary DE sampling and coordinate operation order remain unchanged.

The distribution-name collision remains a release-owner decision; the existing
[PyPI project](https://pypi.org/project/differential-evolution/) is unrelated.
Citation metadata requires verified owner details.
No distribution was renamed/published, no identity metadata was invented, and
the proposed API extensions and CEC/BBOB campaign remain separate future work.

Validation: 141 unittest tests passed on Python 3.12.3 and 3.14.7 (116 originally
tracked, nine round-4 tests, 16 round-5 tests), plus full-scope Ruff/mypy,
documentation doctests, all four
examples, Sphinx with warnings treated as errors, and final diff checks.
New coverage includes 300 extreme-bound full runs. A comparison against the
original package of 70 ordinary seeded DE runs (14 mutation configurations,
five seeds) matched populations, fitness, best histories and RNG state exactly.
Tooling was reused from `/tmp/round4-tools` and `.docdeps`; no dependencies were
added. The clean staged-file snapshot was also tested, so the count includes
only files present in the proposed Git changes.
A clean wheel built from that snapshot excludes the legacy root modules and
passes isolated installed-package smoke tests with extreme coordinates.
The review's exact Rand1 reproduction failed on seeds 2 and 15 before the fix
and on none of seeds 0–19 afterward; its initial diversity example now returns
a finite diameter instead of raising OverflowError.

Limitations: no remote CI OS/Python matrix, compiled C++ trajectory comparison,
or research-performance campaign was run. Signed overflow still requires a
capable boundary handler; custom components must handle their own arithmetic.
Distances or variances truly beyond float range are reported as positive
infinity. These limits are documented in `docs/discrepancies.rst`.

## Bug review round 6 (2026-10-07)

Verified all five findings in `BUG_REPORT.md` using the `.bug_review` diagnostic
scripts and the current code. All were confirmed; the original report and
scripts are preserved unchanged.

| Finding | Resolution |
|---|---|
| BUG-1: reflection escapes bounds | Fixed. Feasible coordinates are returned unchanged and the floating-point reflected result is clamped to the interval. The exact-arithmetic branch and finite-input requirement remain. Tests cover endpoints, mirror images, fixed bounds, randomized intervals and warm-start runs with a box-limited objective. The supplied reproducer now reports zero out-of-bounds objective calls instead of 16. |
| BUG-2: impossible schedule minimum | Fixed. Construction rejects a schedule minimum above the initial population size for DE, SHADE and L-SHADE, including L-SHADE's automatic schedule. Equality remains valid and is tested through complete runs. |
| BUG-3: fractional counts | Fixed. Population size, stopping counts and snapshot interval require integers, excluding booleans. Memory size and built-in/custom schedule minima are validated consistently. Optional stopping limits still accept None; zero generations remain valid. Invalid construction performs no objective calls. |
| BUG-4: invalid mutation scales | Fixed. Only difference_scale may be None. Scale controllers must accept propose(context), checked without invoking them. Regressions cover every scale-bearing mutation, wrong controller signatures, valid built-in controllers and simple custom controllers. Existing finite numeric scale policy is unchanged. |
| BUG-5: documentation drift | Fixed all five items: MIT license file, all five boundary handlers and their API pages, independently seeded Tent chains refreshed every 32 steps, best-family population minima, and seeded Sobol digital shifting. An API-reference regression checks every exported boundary handler. |

Validation: **150 tests passed** on Python 3.12.3 and 3.14.7, including nine new
tests in `tests/test_round6_regressions.py`. The new tests reproduce the defects
against a temporary copy of the original code. Full-scope Ruff and mypy
(36 source files), documentation doctests, Sphinx HTML/doctest builds with
warnings as errors, all four examples with `-W error`, and diff checks passed.
Existing tools in `/tmp/round4-tools` and `.docdeps` were reused; no dependencies
were added.

The supplied component sweep reported no failures: 200,000 boundary cases per
handler, 4,000 initialization cases per configuration, Sobol projection checks,
100,000 index-sampling cases, and 50,000 cases per crossover. The run sweep
completed 5,952 cases and skipped 48 invalid population-size configurations.
Its sole flag, case 3609, is the report's already identified false positive:
an initialization-only L-SHADE run cannot reduce after a generation because
none is processed. The same flag occurs on the original code. Added explicit
coverage of that legitimate behavior; the supplied checker was not altered.

The fixes and regression file are staged for inclusion in the next commit;
no commit or push was created. The remote CI OS/Python matrix was not run.
