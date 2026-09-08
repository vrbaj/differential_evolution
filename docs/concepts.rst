Core concepts
=============

Notation
--------

The documentation uses the following notation.

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\mathbf{x}_{i,g}`
     - target vector for individual :math:`i` in generation :math:`g`
   * - :math:`\mathbf{v}_{i,g}`
     - donor vector produced by mutation
   * - :math:`\mathbf{u}_{i,g}`
     - trial vector after crossover
   * - :math:`\mathbf{x}_{best,g}`
     - best vector in generation :math:`g`
   * - :math:`F`
     - differential weight or scaling factor
   * - :math:`CR`
     - crossover rate
   * - :math:`D`
     - problem dimension
   * - :math:`NP`
     - population size

Execution cycle
---------------

For the composable optimizer the execution cycle is:

1. initialize a population inside the given bounds;
2. evaluate the whole population;
3. for each target vector, compute a donor vector by mutation;
4. combine target and donor vectors through crossover;
5. repair boundary violations if a boundary handler is configured;
6. evaluate the trial vector;
7. accept the trial if it improves the target;
8. optionally update adaptive parameter controllers, diversity histories,
   snapshots, and population schedules.

The generic optimizer performs mutation and crossover against a generation
snapshot, not against partially updated individuals from the same generation.
This is verified in the test suite and matches the usual synchronous DE model.

Minimization
------------

All optimizers in this repository are minimizers. Selection always uses
strict improvement:

.. math::

   \mathbf{x}_{i,g+1} =
   \begin{cases}
     \mathbf{u}_{i,g}, & \text{if } f(\mathbf{u}_{i,g}) < f(\mathbf{x}_{i,g}), \\
     \mathbf{x}_{i,g}, & \text{otherwise.}
   \end{cases}

Randomness and reproducibility
------------------------------

The package does not rely on module-level global random state.

* ``seed=...`` creates an internal ``random.Random`` instance.
* ``rng=...`` allows you to pass your own generator-like object.

Runs are reproducible when the same objective function, bounds, algorithm
configuration, and random source are used.

The repository does not currently expose a NumPy RNG interface, callback
protocol, or built-in parallel objective evaluation API. Random draws therefore
come from the supplied ``random.Random``-like object and are serialized through
the optimization loop.

Boundary handling
-----------------

The general optimizer supports three boundary handlers:

``NoBoundaryHandler``
   Leaves the trial vector unchanged.

``ClipBoundaryHandler``
   Clips each coordinate to its interval.

``RandomResetBoundaryHandler``
   Replaces each violating coordinate with a fresh uniform sample from its
   allowed interval.

Stopping
--------

Two stopping limits are currently implemented:

``max_generations``
   The maximum number of completed generations.

``max_evaluations``
   An optional objective-evaluation budget. If the budget is exhausted in the
   middle of a generation, the processed subset of targets is retained and the
   generation is closed with the resulting state.

There is no separate tolerance-based stopping criterion at present.

Component composition
---------------------

The composable optimizer allows users to build configurations such as
``DE/rand/1/bin`` or ``DE/best/1/exp`` explicitly:

.. code-block:: python

   DifferentialEvolution(
       objective=my_objective,
       bounds=my_bounds,
       population_size=30,
       mutation=Rand1(scale=0.8),
       crossover=BinomialCrossover(crossover_rate=0.9),
   )

Not every named algorithm decomposes cleanly into independent mutation and
crossover components. The dedicated ``SHADE`` and ``LSHADE`` classes exist for
that reason.
