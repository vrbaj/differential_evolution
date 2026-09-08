Customization and extension
===========================

Combining built-in components
-----------------------------

Any mathematically compatible mutation and crossover components can be combined
explicitly with the general optimizer.

.. code-block:: python

   from differential_evolution import (
       Best1,
       DifferentialEvolution,
       ExponentialCrossover,
   )

   optimizer = DifferentialEvolution(
       objective=my_objective,
       bounds=my_bounds,
       population_size=30,
       mutation=Best1(scale=0.6),
       crossover=ExponentialCrossover(crossover_rate=0.8),
   )

Custom mutation strategy
------------------------

A custom mutation strategy is any callable receiving a
:class:`~differential_evolution.MutationContext` and returning a donor vector.

.. literalinclude:: ../examples/custom_components.py
   :language: python
   :start-after: from differential_evolution import BinomialCrossover, DifferentialEvolution, sphere_function
   :end-before: class ReplaceFirstCoordinate

The contract is:

* do not mutate ``context.population`` in place;
* return a list of length equal to the target-vector dimension;
* use ``context.rng`` for any stochastic behavior if reproducibility matters;
* respect any index-distinctness constraints your operator requires.
* document any minimum population-size requirement, because the generic
  optimizer does not infer it automatically.

Custom crossover operator
-------------------------

A crossover operator is any callable with the shape
``(target_vector, donor_vector, rng) -> trial_vector``.

.. literalinclude:: ../examples/custom_components.py
   :language: python
   :start-after: class ReplaceFirstCoordinate:
   :end-before: def main

Custom diversity measure
------------------------

Custom diversity measures can be passed through ``diversity_measures`` when the
object exposes a ``name`` attribute and is callable with
``(population, bounds, previous_population=None)``.

Preserving reproducibility
--------------------------

To preserve reproducibility:

* use the supplied ``rng`` rather than ``random`` module globals;
* avoid in-place edits of stored populations;
* document any additional random draws introduced by a new component;
* keep acceptance-dependent state updates explicit, as done by adaptive
  parameter controllers.

Array-shape conventions
-----------------------

All built-in components operate on Python ``list[float]`` vectors and
``list[list[float]]`` populations.

* target, donor, and trial vectors all have shape ``(D,)``;
* populations have shape ``(NP, D)``;
* bounds have shape ``(D, 2)`` as ``(lower, upper)`` pairs.

Custom components should return new lists rather than references into the
population snapshot.
