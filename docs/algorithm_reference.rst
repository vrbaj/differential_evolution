Algorithm reference
===================

Overview
--------

.. list-table::
   :header-rows: 1

   * - Category
     - Implemented objects
     - Literature basis
   * - Mutation
     - ``Rand1``, ``Rand2``, ``Best1``, ``Best2``,
       ``CurrentToBest1``, ``CurrentToBest2``,
       ``CurrentToRand1``, ``CurrentToRand2``,
       ``TrigonometricMutation``, ``DirectedMutation``,
       ``NeighborhoodSearchMutation``
     - :cite:`storn_price_1997,price_storn_lampinen_2005,fan_lampinen_2003`
   * - Crossover
     - ``BinomialCrossover``, ``ExponentialCrossover``,
       ``IdentityCrossover``
     - :cite:`storn_price_1997,price_storn_lampinen_2005`
   * - Initialization
     - ``RandomInitializer``, ``TentInitializer``,
       ``OppositionInitializer``, ``QuasiOppositionInitializer``,
       ``SobolInitializer``
     - :cite:`price_storn_lampinen_2005,sobol_1967,bratley_fox_1988`
   * - Boundary handling
     - ``NoBoundaryHandler``, ``ClipBoundaryHandler``,
       ``RandomResetBoundaryHandler``
     - implementation choices documented here
   * - Adaptive parameters
     - ``RandomizedScaleFactor``, ``AdaptiveScaleFactor``,
       ``AdaptiveCrossoverRate``, ``jde_rand_1_bin``
     - :cite:`brest_etal_2006`
   * - Population reduction
     - ``LinearPopulationReduction``,
       ``HyperbolicTangentPopulationReduction``
     - :cite:`tanabe_fukunaga_2014`
   * - Dedicated solvers
     - ``SHADE``, ``LSHADE``
     - :cite:`tanabe_fukunaga_2013,tanabe_fukunaga_2014`

Notation and scope
------------------

Unless stated otherwise:

* :math:`\mathbf{x}_{i,g}` is target vector :math:`i` in generation :math:`g`;
* :math:`\mathbf{v}_{i,g}` is the donor vector produced by mutation;
* :math:`\mathbf{u}_{i,g}` is the trial vector after crossover;
* :math:`\mathbf{x}_{best,g}` is the best vector in the current generation;
* :math:`f` is the minimized objective function;
* :math:`F` is a differential weight and :math:`CR` a crossover rate;
* all sampled indices are mutually distinct unless a section says otherwise.

This page documents the algorithms as they are implemented in this repository,
including project-specific variants and edge-case behavior verified in tests.

Mutation strategies
-------------------

Classical DE family
~~~~~~~~~~~~~~~~~~~

``Rand1`` (DE/rand/1)
   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{r_1,g}
      + F \left(\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g}\right).

   Requirements:
   :math:`r_1`, :math:`r_2`, and :math:`r_3` exclude the target index
   :math:`i`. Minimum valid population size: 4.

``Rand2`` (DE/rand/2)
   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{r_1,g}
      + F \left(\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g}\right)
      + F \left(\mathbf{x}_{r_4,g} - \mathbf{x}_{r_5,g}\right).

   Minimum valid population size: 6.

``Best1`` (DE/best/1)
   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{best,g}
      + F \left(\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g}\right).

   Implementation detail:
   the sampled indices exclude both ``target_index`` and ``best_index``.
   Minimum valid population size: 3 when the target already is the current
   best vector, otherwise 4.

``Best2`` (DE/best/2)
   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{best,g}
      + F \left(\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g}\right)
      + F \left(\mathbf{x}_{r_3,g} - \mathbf{x}_{r_4,g}\right).

   Minimum valid population size: 5 when the target already is the current
   best vector, otherwise 6.

``CurrentToBest1`` (DE/current-to-best/1)
   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{i,g}
      + F_b \left(\mathbf{x}_{best,g} - \mathbf{x}_{i,g}\right)
      + F_d \left(\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g}\right).

   When ``difference_scale=None``, the implementation uses
   :math:`F_d = F_b`, which recovers the conventional single-:math:`F` form.
   The sampled difference indices exclude both ``target_index`` and
   ``best_index``. Minimum valid population size: 3 when the target already is
   the current best vector, otherwise 4.

``CurrentToBest2`` (DE/current-to-best/2)
   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{i,g}
      + F_b \left(\mathbf{x}_{best,g} - \mathbf{x}_{i,g}\right)
      + F_d \left(\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g}\right)
      + F_d \left(\mathbf{x}_{r_3,g} - \mathbf{x}_{r_4,g}\right).

   Minimum valid population size: 5 when the target already is the current
   best vector, otherwise 6.

``CurrentToRand1`` (DE/current-to-rand/1)
   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{i,g}
      + K \left(\mathbf{x}_{r_1,g} - \mathbf{x}_{i,g}\right)
      + F \left(\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g}\right).

   Literature often presents current-to-rand mutation without an additional
   crossover stage. In this package the closest direct representation of that
   form is ``CurrentToRand1(...)+IdentityCrossover()``. Minimum valid
   population size: 4.

``CurrentToRand2`` (project-specific extension)
   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{i,g}
      + K \left(\mathbf{x}_{r_1,g} - \mathbf{x}_{i,g}\right)
      + F \left(\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g}\right)
      + F \left(\mathbf{x}_{r_4,g} - \mathbf{x}_{r_5,g}\right).

   This is documented as a repository extension rather than a standard
   historical DE name. Minimum valid population size: 6.

Extended strategies
~~~~~~~~~~~~~~~~~~~

``TrigonometricMutation``
   With probability :math:`p_m`, this applies the Fan-Lampinen
   trigonometric operator :cite:`fan_lampinen_2003`:

   .. math::

      \mathbf{v}_{i,g} =
      \frac{\mathbf{x}_{r_1,g} + \mathbf{x}_{r_2,g} + \mathbf{x}_{r_3,g}}{3}
      + (p_2 - p_1)(\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g})
      + (p_3 - p_2)(\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g})
      + (p_1 - p_3)(\mathbf{x}_{r_3,g} - \mathbf{x}_{r_1,g}),

   where

   .. math::

      p_k = \frac{|f(\mathbf{x}_{r_k,g})|}
      {|f(\mathbf{x}_{r_1,g})| + |f(\mathbf{x}_{r_2,g})| + |f(\mathbf{x}_{r_3,g})|}.

   Otherwise it falls back to ``DE/rand/1`` using the same sampled indices.
   When the denominator is zero, the implementation uses equal weights
   :math:`1/3`. When the selected fitness values are non-finite, it falls back
   to ``DE/rand/1``. Minimum valid population size: 4.

``DirectedMutation``
   Orders three sampled vectors by objective value and extrapolates from the
   best sampled vector:

   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{b,g}
      + \frac{1-f_b}{f_{w_1}}(\mathbf{x}_{b,g}-\mathbf{x}_{w_1,g})
      + \frac{1-f_b}{f_{w_2}}(\mathbf{x}_{b,g}-\mathbf{x}_{w_2,g}).

   If any of the three selected objective values is non-positive or
   non-finite, the implementation falls back to ``DE/rand/1`` with the
   original sampled order. The exact paper mapping should be treated as
   requiring manual review; the formula above matches the implementation.

``NeighborhoodSearchMutation``
   Implements the repository's NSDE-style Gaussian/Cauchy mutation:

   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{r_1,g}
      + \delta \left(\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g}\right),

   where :math:`\delta` is drawn from :math:`\mathcal{N}(\mu, \sigma)` with
   probability ``gaussian_probability`` and from a zero-centered Cauchy law
   with scale ``cauchy_scale`` otherwise. The exact primary-source citation is
   not yet stored in the bibliography, so the mathematical description here is
   intentionally implementation-first.

Crossover operators
-------------------

``BinomialCrossover`` (``/bin``)
   For each coordinate :math:`j \in \{1, \dots, D\}`:

   .. math::

      u_{i,g}^{(j)} =
      \begin{cases}
        v_{i,g}^{(j)}, & \text{if } \rho_j < CR \text{ or } j = j_{rand}, \\
        x_{i,g}^{(j)}, & \text{otherwise.}
      \end{cases}

   The forced coordinate :math:`j_{rand}` is sampled uniformly from
   :math:`\{1, \dots, D\}`. This guarantees that at least one donor coordinate
   is inherited even when :math:`CR = 0`.

``ExponentialCrossover`` (``/exp``)
   One start index :math:`j_{start}` is sampled uniformly. A contiguous donor
   block of length at least one is then copied with cyclic wrap-around while
   fresh uniform samples remain below :math:`CR`:

   .. math::

      L = 1 + \max \left\{ \ell \geq 0 :
      \rho_1 < CR, \dots, \rho_{\ell} < CR \right\},

   truncated at :math:`D`. Coordinates
   :math:`j_{start}, j_{start}+1, \dots, j_{start}+L-1` are interpreted modulo
   :math:`D`.

``IdentityCrossover``
   Returns the donor vector unchanged. This is useful for variants whose
   conventional form is specified purely through mutation.

Selection
---------

All optimizers in this repository are minimizers and use strict one-to-one
selection:

.. math::

   \mathbf{x}_{i,g+1} =
   \begin{cases}
     \mathbf{u}_{i,g}, & \text{if } f(\mathbf{u}_{i,g}) < f(\mathbf{x}_{i,g}), \\
     \mathbf{x}_{i,g}, & \text{otherwise.}
   \end{cases}

Equal-fitness trials do not replace the target.

Initialization
--------------

``RandomInitializer``
   Independent uniform sampling within each bound interval:

   .. math::

      x_{i,0}^{(j)} \sim U(l_j, u_j).

``TentInitializer``
   Uses one evolving tent-map state

   .. math::

      z_{t+1} =
      \begin{cases}
        2 z_t, & z_t < 0.5, \\
        2 (1-z_t), & z_t \ge 0.5,
      \end{cases}

   and rescales each emitted value into the active bound interval. This is a
   project-provided initializer; a canonical reference is not yet stored in the
   bibliography.

``OppositionInitializer``
   Samples a point and its opposite point, evaluates both, and retains the best
   half of the doubled pool. For coordinate :math:`j`:

   .. math::

      \tilde{x}^{(j)} = l_j + u_j - x^{(j)}.

   Implementation detail:
   this initializer ranks candidates by calling the supplied objective during
   initialization, so it consumes additional evaluations before the optimizer
   evaluates the retained population again.

``QuasiOppositionInitializer``
   Uses quasi-opposite points relative to the coordinate-wise midpoint
   :math:`m_j = (l_j + u_j)/2`. The quasi-opposite coordinate is sampled on the
   line segment between the midpoint and the exact opposite point. As with
   ``OppositionInitializer``, the doubled candidate set is ranked by objective
   value during initialization and therefore consumes additional evaluations.

``SobolInitializer``
   Generates a Sobol low-discrepancy sequence using a Bratley-Fox style
   direction-number recurrence for up to 40 dimensions
   :cite:`sobol_1967,bratley_fox_1988`.

Boundary handling
-----------------

The boundary handlers are explicit implementation choices rather than named DE
algorithms.

``NoBoundaryHandler``
   Leaves the trial vector unchanged.

``ClipBoundaryHandler``
   Coordinate-wise projection to interval bounds:

   .. math::

      u_{i,g}^{(j)} \leftarrow \min(\max(u_{i,g}^{(j)}, l_j), u_j).

``RandomResetBoundaryHandler``
   Replaces each violating coordinate with a fresh uniform draw:

   .. math::

      u_{i,g}^{(j)} \leftarrow
      \begin{cases}
        u_{i,g}^{(j)}, & l_j \le u_{i,g}^{(j)} \le u_j, \\
        U(l_j, u_j), & \text{otherwise.}
      \end{cases}

Adaptive control parameters
---------------------------

``RandomizedScaleFactor``
   Samples a fresh :math:`F` from :math:`U(F_{min}, F_{max})` for each target.

``AdaptiveScaleFactor`` and ``AdaptiveCrossoverRate``
   Implement the jDE-style per-individual adaptation of :math:`F_i` and
   :math:`CR_i` from :cite:`brest_etal_2006`:

   .. math::

      F_i' =
      \begin{cases}
        U(F_l, F_u), & \rho_i < \tau_F, \\
        F_i, & \text{otherwise,}
      \end{cases}
      \qquad
      CR_i' =
      \begin{cases}
        U(CR_l, CR_u), & \rho_i' < \tau_{CR}, \\
        CR_i, & \text{otherwise.}
      \end{cases}

   The proposed value is committed only if the associated trial vector is
   accepted by selection. Otherwise the previous per-individual value is
   retained.

``jde_rand_1_bin()``
   Returns the mutation and crossover objects corresponding to the jDE
   ``DE/rand/1/bin`` setup.

SHADE and L-SHADE
-----------------

``SHADE``
   Implements success-history based parameter adaptation
   :cite:`tanabe_fukunaga_2013`. The solver uses the
   ``current-to-pbest/1/bin`` core, an external archive, memory vectors
   :math:`M_F` and :math:`M_{CR}`, Cauchy sampling for :math:`F`, Gaussian
   sampling for :math:`CR`, and weighted memory updates from successful trial
   vectors.

   The donor formula implemented here is

   .. math::

      \mathbf{v}_{i,g} = \mathbf{x}_{i,g}
      + F_i \left(\mathbf{x}_{pbest,g} - \mathbf{x}_{i,g}\right)
      + F_i \left(\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g}\right),

   where :math:`\mathbf{x}_{r_2,g}` can come either from the current
   population or from the archive. The implementation samples a temporary
   elite proportion :math:`p` uniformly from
   :math:`[2/NP, p_{best}]`, converts it to a top-set size with ``ceil``, and
   then samples ``pbest`` from that truncated elite set.

   Parameter sampling is implemented as

   .. math::

      F_i \sim \min\left(1, \operatorname{Cauchy}(M_{F,k}, 0.1)\right)
      \text{ resampled until } F_i > 0,

   and

   .. math::

      CR_i \sim \operatorname{clip}\left(\mathcal{N}(M_{CR,k}, 0.1), 0, 1\right).

   After successful replacements, the memory update implemented here is

   .. math::

      M_{F,k} \leftarrow \frac{\sum_s w_s F_s^2}{\sum_s w_s F_s},
      \qquad
      M_{CR,k} \leftarrow \sum_s w_s CR_s,

   with weights :math:`w_s = \Delta f_s / \sum_m \Delta f_m`.

``LSHADE``
   Extends SHADE with linear population size reduction
   :cite:`tanabe_fukunaga_2014`. If no explicit ``population_schedule`` is
   supplied, the class installs ``LinearPopulationReduction`` automatically.

Population schedules
--------------------

``LinearPopulationReduction``
   Uses a linear schedule between the initial and minimum population size:

   .. math::

      NP_g = \operatorname{round}
      \left(NP_0 + (NP_{min} - NP_0) q_g\right),

   where :math:`q_g = nfev / nfev_{max}` when an evaluation budget is
   available, otherwise :math:`q_g = g / G_{max}`.

``HyperbolicTangentPopulationReduction``
   Uses a project-specific normalized hyperbolic tangent curve:

   .. math::

      z_g = a + (b-a) q_g,
      \qquad
      s_g = \frac{\tanh(z_g) - \tanh(a)}{\tanh(b) - \tanh(a)},

   followed by the same rounded interpolation between :math:`NP_0` and
   :math:`NP_{min}`.

Diversity measures
------------------

The package can record none, one, or several diversity measures during a run.
These diagnostics do not change mutation, crossover, selection, or adaptation.

* ``PopulationDiameter``: maximum pairwise Euclidean distance.
* ``PopulationRadius``: maximum Euclidean distance from the population center.
* ``AverageDistanceAroundPopulationCenter``: mean distance from the center.
* ``AverageDistanceAroundAllIndividuals``: mean distance averaged over all
  ordered center/other pairs, including zero self-distances.
* ``AveragePairwiseDistance``: mean distance over unordered pairs.
* ``PopulationCoherence``: center displacement divided by mean per-individual
  displacement relative to the previous generation.
* ``DimensionalVariance``: mean per-coordinate variance around the center.
* ``AggregatedDistribution``: project-specific variance-to-mean occupancy
  statistic over normalized one-dimensional histograms.
