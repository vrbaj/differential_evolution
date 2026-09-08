Introduction
============

Differential Evolution (DE) is a population-based optimization method for
continuous variables. A DE solver keeps a population of candidate vectors and
improves them by repeatedly generating a donor vector through mutation,
combining donor and target vectors through crossover, and accepting better
trial vectors through one-to-one greedy selection :cite:`storn_price_1997`.

This repository has two complementary goals:

* provide a small composable DE implementation in which mutation, crossover,
  initialization, boundary handling, diversity monitoring, and population-size
  schedules can be combined explicitly; and
* provide dedicated implementations of success-history adaptive variants such
  as SHADE and L-SHADE, whose archive and memory mechanisms are integral to the
  algorithm rather than plug-in details.

Design principles
-----------------

The current code base is intentionally explicit.

* Built-in mutation strategies are small callable objects.
* Crossover operators are separate from mutation.
* Random-number generation is explicit through ``seed`` or ``rng``.
* Mathematical formulas are documented against implementation and tests.
* Population diversity and run trajectories can be recorded for later
  inspection.

What is implemented
-------------------

General composable DE
   The :class:`~differential_evolution.DifferentialEvolution` class supports
   a wide range of mutation strategies, crossover operators, boundary
   handlers, diversity measures, scale-factor controllers, and
   population-reduction schedules.

Success-history adaptive DE
   :class:`~differential_evolution.SHADE` implements success-history based
   parameter adaptation :cite:`tanabe_fukunaga_2013`.

L-SHADE
   :class:`~differential_evolution.LSHADE` extends SHADE with linear
   population size reduction :cite:`tanabe_fukunaga_2014`.

Scientific scope
----------------

The package is aimed at users who need both readability and control:

* researchers who want transparent formulas and reproducibility;
* engineers who want a lightweight optimization tool without a large framework;
* users exploring how different DE components interact.

The documentation describes current behavior, including deliberate
project-specific choices such as the ``AggregatedDistribution`` diversity
measure and the tanh-based population schedule.

Current maturity
----------------

The repository is usable and has deterministic tests for formulas, crossover
invariants, reproducibility, diversity tracking, population schedules, and the
SHADE/L-SHADE state updates that are exposed publicly. The package version is
still ``0.1.0``, so users should treat the API as stable enough for
experimentation but not yet frozen for long-term downstream commitments.
