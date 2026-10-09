Quick start
===========

Minimal reproducible run
------------------------

The example below minimizes the 2-D sphere function,

.. math::

   f(\mathbf{x}) = \sum_{j=1}^{D} x_j^2,

on the square :math:`[-5, 5]^2`.

.. doctest::

   >>> from differential_evolution import (
   ...     BinomialCrossover,
   ...     DifferentialEvolution,
   ...     Rand1,
   ...     sphere_function,
   ... )
   >>> optimizer = DifferentialEvolution(
   ...     objective=sphere_function,
   ...     bounds=[(-5.0, 5.0), (-5.0, 5.0)],
   ...     population_size=20,
   ...     mutation=Rand1(scale=0.8),
   ...     crossover=BinomialCrossover(crossover_rate=0.9),
   ...     max_generations=20,
   ...     seed=123,
   ... )
   >>> result = optimizer.run()
   >>> len(result.x)
   2
   >>> result.fun >= 0.0
   True

Interpreting the result
-----------------------

The returned :class:`~differential_evolution.OptimizeResult` contains:

``x``
   Best vector found at the end of the run.

``fun``
   Objective value of ``x``.

``nit``
   Number of completed generations.

``nfev``
   Number of objective evaluations.

``best_history``
   Best vector after each recorded generation state, including the initialized
   population.

``diversity_history``
   Generation-by-generation diversity metrics when enabled.

``population_size_history``
   Population size after initialization and after each generation.

Using SHADE or L-SHADE
----------------------

When you need the full success-history adaptive algorithms instead of a
hand-composed solver, use the dedicated classes.

.. doctest::

   >>> from differential_evolution import LSHADE, sphere_function
   >>> optimizer = LSHADE(
   ...     objective=sphere_function,
   ...     bounds=[(-5.0, 5.0), (-5.0, 5.0)],
   ...     population_size=10,
   ...     min_population_size=4,
   ...     max_generations=3,
   ...     max_evaluations=40,
   ...     seed=7,
   ... )
   >>> result = optimizer.run()
   >>> result.population_size_history[0]
   10
