# Changelog

## 0.2.0 — 16 October 2026

### Changed

- Generic `DifferentialEvolution` defaults to `BinomialCrossover(0.9)`.
  To retain 0.1.0 donor-only behavior, explicitly use `IdentityCrossover()`.
- `ackley_function` accepts any nonempty dimension. Other fixed-dimensional
  benchmark contracts remain unchanged.

### Added

- Ordered `batch_objective(candidates)` evaluation for DE, SHADE and L-SHADE,
  supporting vectorized adapters and external worker pools. Submitted candidates
  count against the budget even on batch failure; batch exceptions propagate.
- Result provenance: initial constructor configuration, seed, package/Python
  versions and initial `random.Random` state. Unsupported custom objects are
  identified rather than silently claimed to be fully serialized.
- `BoundNormalizedDiversity(measure)` for diagnostics in bound coordinates.
- Batch and penalty-constraint examples, migration guidance and regression tests.
- Scientific validation using NIST calibration data and BBOB experiments;
  companion manuscript and experiment artifacts are maintained outside Git.
- Source-distribution checks exclude local manuscripts and review files.
- Version-tag pushes trigger the checked PyPI release workflow; manual TestPyPI
  and PyPI dispatches remain available.

### Validation

162 tests pass, with 94.7% measured statement coverage before release preparation.
Twelve explicitly configured scalar runs match 0.1.0 populations, histories and
snapshots. Ruff, mypy, and isolated wheel/source-distribution checks pass.
Saved scientific experiments record development version 0.2.0.dev0 and their
original source fingerprints; those records are not relabeled as release runs.
Reference-C++/PyADE equivalence and distributed speedup remain unestablished.

## 0.1.0 — 9 October 2026

Initial DEFoundry release: composable DE, dedicated SHADE/L-SHADE, population
initializers, boundary repair, parameter controllers, diversity and run histories.
