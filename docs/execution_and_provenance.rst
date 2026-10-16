Execution and provenance (0.2.0)
===============================================

``DifferentialEvolution`` now defaults to ``BinomialCrossover(0.9)``.
Pass ``IdentityCrossover()`` to retain donor-only behavior from 0.1.0.
``SHADE`` and ``LSHADE`` retain fixed current-to-pbest/1/bin trial construction;
their inherited mutation/crossover placeholders are not extension points.

Ordered batch evaluation
------------------------

Pass ``batch_objective(candidates)`` returning one scalar per candidate in input
order. An external ordered ``executor.map`` or a vectorized adapter can implement
this interface; the library does not create pools. See
``examples/batch_evaluation.py``. Threads are suitable for I/O or native calls
that release the GIL; CPU-bound Python objectives generally need processes.

Standard initialization and each budget-limited prefix of trials can be batched.
Ranked initializers use the required scalar objective to rank their candidate
pool. All trial parents and adaptive archives remain generation snapshots.
Deterministic scalar and batch evaluations produce identical seeded trajectories
with built-in controllers in the regression tests. Custom controllers must not
rely on inter-target commit side effects: batch commits occur after evaluation.
Objectives must not share the optimizer RNG if trajectory equality is required.

Every submitted candidate is charged to the evaluation budget, including when a
batch raises or returns the wrong number of values. Batch errors propagate,
even with ``objective_errors="penalize"``; implement individual penalties inside
the evaluator. NaNs become positive infinity; negative infinity is preserved.
Inputs are copied before invoking the batch evaluator.

Provenance and diagnostics
--------------------------

Result metadata records package/Python versions, seed, initial constructor
configuration and, for ``random.Random``, its pre-initialization state.
Custom callable code, closures, datasets and arbitrary objects are not fully
serialized: descriptors mark their configuration incomplete. Preserve them
alongside results. No cross-version trajectory or checkpoint-restart guarantee
is implied.

``BoundNormalizedDiversity(AveragePairwiseDistance())`` applies a diagnostic in
bound coordinates. Fixed coordinates become zero. Out-of-bound coordinates are
not clipped. Movement diagnostics transform both generations consistently.
Ackley now accepts any positive dimension; other fixed-dimensional benchmarks
retain their dimension checks.

Problem extensions
------------------

``examples/penalty_constraint.py`` demonstrates scalar penalty optimization and
separate constraint-violation reporting. Finite penalties need not produce a
feasible constrained optimum. Integer repair, replicated noisy evaluations and
scalarization are possible user adaptations, not dedicated mixed-integer,
noise-aware or multi-objective solvers. Inner simulator calls in a replicated
objective must be budgeted separately by the user.
