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
   and reuse the retained fitness. Their cost is twice the population size,
   compared with one population for uniform, Tent, and Sobol initialization.

``DirectedMutation``
   A project-specific variant using normalized fitness gaps. The previous raw
   fitness coefficient could diverge near zero. Its unverified attribution to
   Fan-Lampinen has been removed; the trigonometric paper is not its source.

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

Review corrections
------------------

SHADE uses an arithmetic CR memory and no terminal CR state. L-SHADE uses
Lehmer means for both memories, terminal CR slots, and fixed p=0.11. Both
allow the target in the p-best set, exclude only target and r1 from r2's
population candidates, accept ties, and cap the archive on insertion.
Only strictly improving finite differences contribute to adaptation.
Both default to parent-aware midpoint repair. Explicit handlers override it.

L-SHADE without an evaluation budget retains a generation-based schedule for
compatibility and emits a warning. Pass max_evaluations for evaluation-based
linear population reduction. Trigonometric mutation intentionally depends on
absolute fitness values and therefore on objective offsets.
