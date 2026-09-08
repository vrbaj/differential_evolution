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

They use only the public API imported from :mod:`differential_evolution`.

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
