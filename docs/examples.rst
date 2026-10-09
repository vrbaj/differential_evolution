Examples
========

Runnable example scripts
------------------------

The repository contains small runnable examples:

.. literalinclude:: ../examples/basic_usage.py
   :language: python
   :caption: ``examples/basic_usage.py``

.. literalinclude:: ../examples/custom_components.py
   :language: python
   :caption: ``examples/custom_components.py``

Run them from the repository root with:

.. code-block:: bash

   python3 -m examples.basic_usage
   python3 -m examples.custom_components
   python3 -m examples.diversity_measures
   python3 -m examples.population_initialization

They use only the public API imported from :mod:`differential_evolution`.

Choosing population initialization
----------------------------------

.. literalinclude:: ../examples/population_initialization.py
   :language: python
   :caption: ``examples/population_initialization.py``

Pass a component through ``initializer=`` to select the starting population.
The same option works with ``DifferentialEvolution``, ``SHADE``, and ``LSHADE``.
Omitting it selects ``RandomInitializer()``.

* ``RandomInitializer`` draws independent uniform coordinates inside the bounds.
* ``TentInitializer`` transforms seeded coordinate chains with a tent map.
* ``OppositionInitializer`` evaluates random points and their opposites, then
  keeps the best ``population_size`` points.
* ``QuasiOppositionInitializer`` evaluates random points and samples between
  each opposite and the coordinate midpoint, then keeps the best points.
* ``SobolInitializer`` uses a low-discrepancy sequence with a seeded digital
  shift by default. Set ``scramble=False`` for the unshifted sequence. The
  implementation supports 1–40 dimensions; a power-of-two population size
  retains the sequence's balanced blocks.

Random, tent, and Sobol initialization use ``population_size`` objective
calls. Opposition and quasi-opposition evaluate twice that many candidates;
retained fitness values are reused. Set an evaluation budget large enough for
initialization. This example gives every run the same total budget, so the two
ranked initializers leave fewer evaluations for subsequent generations.

Call ``initialize()`` to inspect the starting population and fitness, then
``run()`` to continue optimization. Calling ``run()`` directly also initializes
the optimizer. Use a fresh optimizer for another run. A fixed seed makes each
configuration reproducible; the example illustrates usage rather than ranking
initializers by a single run's final fitness.

Tracking diversity measures
---------------------------

.. literalinclude:: ../examples/diversity_measures.py
   :language: python
   :caption: ``examples/diversity_measures.py``

Set ``diversity_measures`` to component objects, string aliases, or a mixture.
Leave it as ``None`` to disable tracking. Histories are keyed by canonical names:
``"radius"`` becomes ``"population_radius"`` and ``"coherence"`` becomes
``"population_coherence"``. Avoid requesting the same measure twice through
aliases and objects.

Each history includes the initial population at index zero and one entry per
processed generation, including a final partial generation. Snapshot thinning
changes only which snapshots are saved; diversity histories still record every
generation.

Diameter is the largest pairwise distance, radius measures spread around the
population center, and average pairwise distance summarizes typical separation.
Distances use the original coordinate units. Coherence compares center movement
with average individual movement and needs consecutive populations; its initial
value is zero. Diversity statistics describe the population rather than prove
convergence, and pairwise measures add quadratic work in population size.

Recording a run for later analysis
----------------------------------

.. code-block:: python

   from differential_evolution import (
       AveragePairwiseDistance,
       LSHADE,
       PopulationDiameter,
       sphere_function,
   )

   optimizer = LSHADE(
       objective=sphere_function,
       bounds=[(-5.0, 5.0), (-5.0, 5.0)],
       population_size=18,
       min_population_size=4,
       max_generations=30,
       max_evaluations=400,
       diversity_measures=[
           PopulationDiameter(),
           AveragePairwiseDistance(),
       ],
       record_snapshots=True,
       seed=123,
   )
   result = optimizer.run()
   result.save_json("run.json")

This saves per-generation best values, optional populations, diversity
statistics, population sizes, and the SHADE/L-SHADE archive and memory state.
