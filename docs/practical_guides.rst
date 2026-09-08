Practical guides
================

Choosing a mutation strategy
----------------------------

``Rand1`` and ``Rand2``
   Good starting points for broad exploration.

``Best1`` and ``Best2``
   More exploitative because the current generation best vector is used as the
   base.

``CurrentToBest1`` and ``CurrentToBest2``
   Useful when you want explicit attraction toward the best vector while still
   adding differential perturbations.

``SHADE`` and ``LSHADE``
   Prefer these when you want an adaptive algorithm from the DE literature
   rather than a manually assembled configuration.

Choosing population size
------------------------

This repository does not impose a universal default rule for ``population_size``.
The minimum valid population size depends on the mutation strategy:

* 4 for ``Rand1`` and most three-index strategies;
* 6 for two-difference variants such as ``Rand2``;
* 3 or 5 for some best-based operators when the current target already is the
  generation best, otherwise 4 or 6 as documented in the algorithm reference;
* at least 4 for ``SHADE`` and ``LSHADE``.

Choosing ``F`` and ``CR``
-------------------------

This repository does not force one canonical parameter rule, but the built-in
components suggest a few conservative defaults.

* ``Rand1`` with ``F`` around ``0.5`` to ``0.9`` and binomial ``CR`` around
  ``0.7`` to ``0.9`` is a practical baseline.
* Exploitative strategies such as ``Best1`` or ``CurrentToBest1`` often need a
  smaller ``F`` than broad-exploration ``Rand1``.
* If manual tuning is undesirable, use ``jde_rand_1_bin()`` or the dedicated
  ``SHADE`` / ``LSHADE`` solvers.

Handling bounded problems
-------------------------

Choose a boundary policy explicitly:

* ``ClipBoundaryHandler`` if projection is acceptable;
* ``RandomResetBoundaryHandler`` if you prefer to re-randomize only violating
  coordinates;
* ``NoBoundaryHandler`` only when your objective can safely handle out-of-range
  points or your mutation/crossover setup already guarantees feasibility.

Reproducible experiments
------------------------

For reproducible experiments:

* pass ``seed=...``;
* record the optimizer result;
* for full trajectory analysis, set ``record_snapshots=True`` and persist the
  result to JSON or pickle.

Monitoring convergence
----------------------

The package currently exposes:

* ``best_fitness_history`` for best-so-far quality;
* ``diversity_history`` for population spread;
* ``population_size_history`` for shrinking schedules;
* SHADE-specific snapshot fields for archive and memory state.

The package does not currently include callback hooks or live progress
reporters, so monitoring is post hoc through the returned result object.

Common mistakes
---------------

* Using a mutation strategy with a population smaller than its index-sampling
  requirements.
* Forgetting that all optimizers minimize.
* Passing a custom component that mutates the shared population in place.
* Assuming that all named DE variants are decomposable into independent
  mutation and crossover components. SHADE and L-SHADE are implemented as
  dedicated classes precisely because their archive and memory mechanisms are
  integral to the algorithm.
