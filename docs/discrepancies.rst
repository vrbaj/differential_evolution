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
population candidates and accept ties. Only strictly improved parents enter
the archive, after the processed generation's trials have been generated.
The archive is capped after each buffered insertion. Ties never add parents.
Only strictly improving finite differences contribute to adaptation.
Both default to parent-aware midpoint repair. Explicit handlers override it.

L-SHADE without an evaluation budget retains a generation-based schedule for
compatibility and emits a warning. Pass max_evaluations for evaluation-based
linear population reduction. Trigonometric mutation intentionally depends on
absolute fitness values and therefore on objective offsets.

Evaluation budgets and partial generations
------------------------------------------

When both stopping limits are supplied, built-in population schedules follow
the evaluation budget. A ``UserWarning`` is emitted when
``population_size * (max_generations + 1) < max_evaluations``: with standard
initialization, the generation limit ends the run before the evaluation budget,
so population reduction is only partial and may not shrink the population.
This check assumes one initial population's worth of evaluations; ranked or
custom initializers may consume more. The progress formula is unchanged.

When an evaluation budget runs out partway through a generation, evaluated
trials retain their selection results and unprocessed targets remain unchanged.
The partial generation counts once toward ``nit`` and all generation histories;
population reduction uses the consumed evaluations (or this generation count
when no evaluation budget is provided). Thus ``nit`` counts started generations
with at least one evaluated trial, rather than only full population sweeps.

Ranked initializers may require twice the population size in evaluations.
Insufficient budgets still raise at the next attempted evaluation after using
the available budget. This deliberate policy is unchanged; callers should
budget for the full initialization pool.

Numerical range
---------------

Built-in mutations retain ordinary floating-point arithmetic for finite
results, with exact-arithmetic fallback when an intermediate overflows.
Representable donors are recovered without changing random draws. A donor
outside float range becomes signed infinity. Clip, midpoint and random-reset
handlers can repair it; reflection and no-repair configurations require finite
trial coordinates. NaNs are rejected before repair and every repaired vector
must be finite before the objective is called. Custom mutations and objectives
remain responsible for their own arithmetic. Successful initialization alone
does not guarantee a usable unbounded or custom-component run.

Diversity uses stable Euclidean norms and means. Variance is computed on scaled
coordinates. A genuinely unrepresentable distance or variance is reported as
positive infinity, rather than raising an overflow exception; representable
means can still be recovered when individual distances exceed float range.
Histogram normalization also supports intervals wider than float range.

Versioned reference policies
----------------------------

These distinctions refer to the `2014 paper
<https://ryojitanabe.github.io/pdf/tf-cec2014.pdf>`_ and the released
`L-SHADE 1.0.1 C++ code
<https://ryojitanabe.github.io/code/LSHADE1.0.1_CEC2014.zip>`_. They do not assert
end-to-end trajectory equivalence or measured performance effects.

* L-SHADE's archive rate 2.6, memory size 6, and p=0.11 follow the paper's
  Table II tuned settings. Released C++ uses 1.4, 5, and 0.11, respectively.
  The paper/code use an initial size of 18 times dimension; this library
  requires callers to supply the population size.
* This library's ``SHADE`` targets the 2013 scheme with random p, arithmetic
  CR memory and no terminal CR state. Released SHADE 1.1.1 is a different
  version with fixed p, Lehmer CR memory and a terminal mechanism; its defaults
  must not be substituted without identifying that version.
* L-SHADE terminal CR slots remain terminal after memory-index wraparound,
  following the paper. The released C++ resets a destination slot before
  testing its terminal status, allowing positive CR to restore it.
* Python's nearest-even rounding is retained explicitly for compatibility.
  At NP=150 and p=0.11, the elite set has 16 members; C++ rounds it to 17.
  A linear population target of 8.5 becomes 8 here and 9 in C++. Archive
  capacity also uses nearest-even rounding, both initially and after reduction;
  released C++ rounds initially but truncates after reduction.
* Population reduction retains the best members in stable fitness order,
  favoring earlier indices on ties. Released C++ repeatedly erases the first
  worst member, retaining the original relative order and later tied members.
* At capacity, each buffered archive insertion appends then randomly deletes
  from the enlarged archive. The new parent can be removed. Released C++
  overwrites an old slot, so the new parent survives that insertion. Reduction
  randomly prunes this library's archive, while released C++ truncates storage.
* This library enforces the actual objective-call budget and selects only the
  evaluated prefix of a final generation. Released C++ evaluates a whole
  generation before stopping its evaluation-count loop, and can evaluate an
  uncounted tail. Its competition harness also truncates near-optimal fitness
  using known optima. This general-purpose library does not apply that rule.

Random sources and optional extensions
--------------------------------------

Built-in components require the Python-random-style methods ``random``,
``uniform``, ``randrange``, ``sample`` and ``gauss``. Use ``seed=123`` or
``rng=random.Random(123)``. A NumPy ``Generator`` requires an adapter implementing
this interface; direct NumPy support is not currently promised. Penalty mode
catches exception types, not intent: it cannot distinguish domain errors from
programming mistakes that also raise ``ValueError``.
