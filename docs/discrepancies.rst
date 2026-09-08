Implementation notes and suspected discrepancies
================================================

This page separates confirmed implementation choices from possible areas that
deserve further review.

Confirmed project-specific choices
----------------------------------

``AggregatedDistribution``
   This diversity measure is not presented as a canonical DE metric from the
   primary literature. The implementation uses a documented marginal
   variance-to-mean occupancy statistic.

``HyperbolicTangentPopulationReduction``
   This is a project-specific nonlinear schedule inspired by tanh-based
   reduction papers rather than a reproduction of one fixed published formula.

``CurrentToRand2``
   Treated as a project-specific extension rather than a standard classical DE
   naming convention.

Potentially nonstandard or worth manual review
----------------------------------------------

``OppositionInitializer`` and ``QuasiOppositionInitializer``
   These initializers evaluate a doubled candidate pool during initialization
   and the optimizer then evaluates the retained population again. This is
   current behavior, not a documentation mistake, but users should be aware
   that initialization consumes extra objective evaluations.

``DirectedMutation``
   The implementation follows the formula documented in this repository, but
   the directed-mutation family is less standardized in the DE literature than
   the classical ``rand`` and ``best`` families. If a future paper source is
   preferred, the exact naming and equation should be checked again.

``NeighborhoodSearchMutation``
   The implementation is a clear Gaussian/Cauchy differential mutation, but a
   fully verified primary-source citation is not yet stored in the bibliography.
   The documentation therefore describes the implemented mathematics directly
   and leaves the exact paper mapping for manual review.

``CurrentToRand1`` and ``CurrentToRand2``
   These mutation operators can be paired with any crossover object in the
   generic optimizer. In much of the DE literature the current-to-rand family
   is presented without an additional crossover stage, so
   ``IdentityCrossover`` is the closest canonical composition.

``SHADE`` and ``LSHADE`` defaults
   The current implementation follows the success-history memory mechanism,
   archive usage, and linear population reduction structure from the Tanabe and
   Fukunaga papers. Experimental parameter defaults in published benchmarks may
   still vary, so production research runs should set them explicitly.
