> Historical refactor audit. The current behavior supersedes the formulas and
> exclusion rules below where noted in [review fixes](docs/review_fixes.md).
> In particular, the old DirectedMutation attribution/formula is withdrawn;
> Best-family exclusions, initialization, and SHADE/L-SHADE rules changed.

# Algorithm Audit

## Scope

This audit covers the Differential Evolution variants and crossover behavior implemented in the original repository:

- `DE/rand/1`
- `DE/rand/2`
- `DE/best/1`
- `DE/best/2`
- `DE/current-to-best/1`
- `DE/current-to-best/2`
- `DE/current-to-rand/1`
- `DE/current-to-rand/2`
- binomial crossover

It also covers the newly added `TrigonometricMutation` component because its mathematical meaning needs explicit documentation.

The original README also claimed a Sobol initializer. The original implementation was incomplete and mathematically invalid; it has now been replaced with a correct Bratley-Fox-style Sobol implementation for the first 40 dimensions.

## Canonical references

- R. Storn and K. Price, "Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces," *Journal of Global Optimization*, 1997. DOI: https://doi.org/10.1023/A:1008202821328
- K. Price, R. Storn, and J. Lampinen, *Differential Evolution: A Practical Approach to Global Optimization*, Springer, 2005. DOI: https://doi.org/10.1007/3-540-31306-0
- S. Das and P. N. Suganthan, "Differential Evolution: A Survey of the State-of-the-Art," *IEEE Transactions on Evolutionary Computation*, 2011. DOI: https://doi.org/10.1109/TEVC.2010.2059031

Notation below follows the standard DE literature:

- `x_i,g`: target vector for index `i` in generation `g`
- `v_i,g`: donor vector
- `u_i,g`: trial vector
- `x_best,g`: best vector in the current generation
- `F`: differential weight
- `CR`: crossover rate
- `r1`, `r2`, ...: sampled population indices

## Summary of confirmed defects in the original implementation

1. Binomial crossover was implemented incorrectly.
   - Original condition: keep the target coordinate when `cr > crossover or random_dimension != dimension`.
   - Because `random_dimension != dimension` is true for all but one coordinate, the donor vector could only contribute in at most one coordinate.
   - Canonical binomial crossover must independently consider every coordinate and force at least one donor coordinate.

2. Candidate index sampling was incorrect.
   - The original code used `sample(range(self.population_size - 1), 6)`.
   - This excluded the last population member from ever being sampled.
   - It also failed to express the actual exclusion constraints directly.

3. Mutation and crossover were entangled.
   - Mutation was computed per-coordinate inside the crossover loop.
   - This prevented independent composition of mutation and crossover components.

4. The implementation relied on implicit global RNG state.
   - The original code used `numpy.random` module functions directly.
   - Reproducibility depended on global state rather than explicit user-controlled RNG input.

5. Objective evaluations were duplicated and uncounted.
   - Individuals were re-evaluated repeatedly inside `evolve()` and `get_best()`.
   - Evaluation counting was not tracked.

6. Boundary handling was not actually implemented.
   - A comment claimed boundary repair, but no repair logic existed.

7. The generation update loop did not make the execution model explicit.
   - The original `get_best()` was called before the loop, which was correct for generation-level best use.
   - However, mutation/crossover logic was interleaved tightly enough that invariants were hard to verify.

8. The old Sobol initializer was not a valid Sobol sequence implementation.
   - It used bitwise XOR operators where exponentiation-like logic was intended.
   - It did not match standard Sobol direction-number construction.
   - It reused one scalar across every coordinate of a point.

## Variant-by-variant audit

### `DE/rand/1`

- Current repository name: `DE/rand/1`
- Canonical donor formula: `v_i,g = x_r1,g + F * (x_r2,g - x_r3,g)`
- Intended base vector: random
- Sampled indices:
  - `r1`, `r2`, `r3` must be mutually distinct
  - `r1`, `r2`, `r3` must exclude `i`
- Minimum valid population size: 4
- Target exclusion: yes
- Original implementation correctness:
  - Mutation formula was correct.
  - Index sampling had the shared defect described above.
- Change made:
  - Preserved the formula.
  - Replaced sampling with centralized distinct-index sampling.

### `DE/rand/2`

- Current repository name: `DE/rand/2`
- Canonical donor formula: `v_i,g = x_r1,g + F * (x_r2,g - x_r3,g) + F * (x_r4,g - x_r5,g)`
- Intended base vector: random
- Sampled indices:
  - `r1` through `r5` mutually distinct
  - all exclude `i`
- Minimum valid population size: 6
- Target exclusion: yes
- Original implementation correctness:
  - Mutation formula was correct.
  - Index sampling had the shared defect described above.
- Change made:
  - Preserved the formula.
  - Replaced sampling with centralized distinct-index sampling.

### `DE/best/1`

- Current repository name: `DE/best/1`
- Canonical donor formula: `v_i,g = x_best,g + F * (x_r1,g - x_r2,g)`
- Intended base vector: best vector from the current generation snapshot
- Sampled indices:
  - `r1`, `r2` distinct
  - `r1`, `r2` exclude `i`
  - in this refactor, `r1`, `r2` also exclude `best_index` so the sampled vectors are distinct from the base vector when possible
- Minimum valid population size:
  - 3 when `i` is the best index
  - 4 otherwise
- Target exclusion: yes for sampled indices; the base vector may equal the target when the target is best
- Original implementation correctness:
  - Mutation formula was correct.
  - Index sampling had the shared defect described above.
- Change made:
  - Preserved the formula.
  - Made best-vector timing explicit and used centralized sampling.

### `DE/best/2`

- Current repository name: `DE/best/2`
- Canonical donor formula: `v_i,g = x_best,g + F * (x_r1,g - x_r2,g) + F * (x_r3,g - x_r4,g)`
- Intended base vector: best vector from the current generation snapshot
- Sampled indices:
  - `r1` through `r4` mutually distinct
  - sampled indices exclude `i`
  - sampled indices exclude `best_index` in the refactor
- Minimum valid population size:
  - 5 when `i` is the best index
  - 6 otherwise
- Target exclusion: yes for sampled indices; base may equal target if target is best
- Original implementation correctness:
  - Mutation formula was correct.
  - Index sampling had the shared defect described above.
- Change made:
  - Preserved the formula.
  - Made best-vector timing explicit and used centralized sampling.

### `DE/current-to-best/1`

- Current repository name: `DE/current-to-best/1`
- Canonical donor formula: `v_i,g = x_i,g + F * (x_best,g - x_i,g) + F * (x_r1,g - x_r2,g)`
- Original repository formula: `x_i,g + F_best * (x_best,g - x_i,g) + F_diff * (x_r1,g - x_r2,g)`
- Intended base vector: current target vector
- Sampled indices:
  - `r1`, `r2` distinct
  - sampled indices exclude `i`
  - sampled indices exclude `best_index` in the refactor
- Minimum valid population size:
  - 3 when `i` is the best index
  - 4 otherwise
- Target exclusion: yes for sampled difference indices
- Original implementation correctness:
  - The structure was correct.
  - The use of two scale factors under the canonical name is a project-specific extension or ambiguity, not a mathematical bug.
  - Index sampling had the shared defect described above.
- Change made:
  - Preserved the two-factor behavior through `CurrentToBest1(scale, difference_scale=...)`.
  - The default new API treats `difference_scale=None` as the canonical single-`F` form.

### `DE/current-to-best/2`

- Current repository name: `DE/current-to-best/2`
- Canonical donor formula: `v_i,g = x_i,g + F * (x_best,g - x_i,g) + F * (x_r1,g - x_r2,g) + F * (x_r3,g - x_r4,g)`
- Original repository formula: `x_i,g + F_best * (x_best,g - x_i,g) + F_diff * (x_r1,g - x_r2,g) + F_diff * (x_r3,g - x_r4,g)`
- Intended base vector: current target vector
- Sampled indices:
  - `r1` through `r4` mutually distinct
  - sampled indices exclude `i`
  - sampled indices exclude `best_index` in the refactor
- Minimum valid population size:
  - 5 when `i` is the best index
  - 6 otherwise
- Target exclusion: yes for sampled difference indices
- Original implementation correctness:
  - The structure was correct.
  - The two-factor choice is a documented extension or ambiguity rather than a sign error.
  - Index sampling had the shared defect described above.
- Change made:
  - Preserved the old two-factor behavior in the compatibility layer.
  - Exposed the canonical single-`F` form in the new API when `difference_scale` is omitted.

### `DE/current-to-rand/1`

- Current repository name: `DE/current-to-rand/1`
- Common donor formula in the DE literature: `v_i,g = x_i,g + K * (x_r1,g - x_i,g) + F * (x_r2,g - x_r3,g)`
- Intended base vector: current target vector attracted toward a random vector
- Sampled indices:
  - `r1`, `r2`, `r3` mutually distinct
  - all exclude `i`
- Minimum valid population size: 4
- Target exclusion: yes for sampled indices
- Original implementation correctness:
  - The donor formula matched a recognized current-to-rand structure.
  - The implementation then applied binomial crossover, which is mathematically possible but not the canonical presentation in much of the literature.
  - Index sampling had the shared defect described above.
- Change made:
  - Preserved the donor formula as an independent mutation component.
  - Documented that pairing with `IdentityCrossover` is the closest built-in representation of the standard no-extra-crossover form.
  - Pairing with binomial or exponential crossover remains available as a noncanonical but explicit composition.

### `DE/current-to-rand/2`

- Current repository name: `DE/current-to-rand/2`
- Project donor formula: `v_i,g = x_i,g + K * (x_r1,g - x_i,g) + F * (x_r2,g - x_r3,g) + F * (x_r4,g - x_r5,g)`
- Intended base vector: current target vector attracted toward a random vector
- Sampled indices:
  - `r1` through `r5` mutually distinct
  - all exclude `i`
- Minimum valid population size: 6
- Target exclusion: yes for sampled indices
- Original implementation correctness:
  - The formula is a defensible project-specific extension of the current-to-rand family.
  - It is not one of the most standard named variants in the original DE papers.
  - Index sampling had the shared defect described above.
- Change made:
  - Preserved the formula as a built-in strategy.
  - Documented it as a project-specific extension rather than silently rebranding it as canonical.

### `TrigonometricMutation`

- Current package name: `TrigonometricMutation`
- Canonical source:
  - H.-Y. Fan and J. Lampinen, "A trigonometric mutation operation to differential evolution," *Journal of Global Optimization*, 2003. DOI: https://doi.org/10.1023/A:1024653025686
- Canonical hybrid structure:
  - sample mutually distinct `r1`, `r2`, `r3`, excluding `i`
  - with probability `p_m`, use the trigonometric donor
  - otherwise use `DE/rand/1`
- Trigonometric donor formula:
  - `v_i,g = (x_r1,g + x_r2,g + x_r3,g)/3`
  - `+ (p2 - p1) * (x_r1,g - x_r2,g)`
  - `+ (p3 - p2) * (x_r2,g - x_r3,g)`
  - `+ (p1 - p3) * (x_r3,g - x_r1,g)`
  - where `p1 = |f_r1| / (|f_r1| + |f_r2| + |f_r3|)`, and likewise for `p2`, `p3`
- Intended base vectors: three randomly sampled vectors, weighted by objective values
- Sampled indices:
  - `r1`, `r2`, `r3` must be mutually distinct
  - all exclude `i`
- Minimum valid population size: 4
- Target exclusion: yes
- Original repository correctness:
  - not implemented
- Change made:
  - added the canonical Fan-Lampinen hybrid form as a composable mutation component
  - when the selected absolute fitness sum is zero, equal weights `1/3` are used to avoid division by zero
  - when the selected fitness data are non-finite, the implementation falls back to `DE/rand/1`

### `DirectedMutation`

- Current package name: `DirectedMutation`
- Canonical source:
  - H.-Y. Fan and J. Lampinen, "A trigonometric mutation operation to differential evolution," *Journal of Global Optimization*, 2003. DOI: https://doi.org/10.1023/A:1024653025686
- Directed donor formula:
  - sample mutually distinct `r1`, `r2`, `r3`, excluding `i`
  - reorder them so `f(x_b) <= f(x_w1) <= f(x_w2)`
  - `v_i,g = x_b + ((1 - f_b) / f_w1) * (x_b - x_w1) + ((1 - f_b) / f_w2) * (x_b - x_w2)`
- Intended base vector: the best vector among the three sampled vectors
- Sampled indices:
  - `r1`, `r2`, `r3` must be mutually distinct
  - all exclude `i`
- Minimum valid population size: 4
- Target exclusion: yes
- Original repository correctness:
  - not implemented
- Change made:
  - added the directed mutation as a composable component
  - because the canonical coefficients are undefined for non-positive or non-finite worse objective values, the implementation falls back to `DE/rand/1` in those cases

### `NeighborhoodSearchMutation`

- Current package name: `NeighborhoodSearchMutation`
- Canonical source:
  - Z. Yang, J. He, and X. Yao, "Making a Difference to Differential Evolution," in *Advances in Metaheuristics for Hard Optimization*, Springer, 2008.
  - The same NSDE definition is summarized in Yang, Tang, and Yao, "Differential evolution for high-dimensional function optimization," and the Information Sciences 2008 DECC-G paper.
- Canonical donor formula:
  - `v_i,g = x_r1,g + d_i * N(0.5, 0.5)` with probability `0.5`
  - `v_i,g = x_r1,g + d_i * delta` otherwise
  - where `d_i = x_r2,g - x_r3,g`
  - `delta` is a Cauchy random variable with scale parameter `1`
- Intended base vector: random
- Sampled indices:
  - `r1`, `r2`, `r3` mutually distinct
  - all exclude `i`
- Minimum valid population size: 4
- Target exclusion: yes
- Original repository correctness:
  - not implemented
- Change made:
  - added NSDE as a dedicated mutation component with configurable Gaussian probability, Gaussian parameters, and Cauchy scale
  - crossover remains an independent component, so `NeighborhoodSearchMutation` can be paired with binomial, exponential, or identity crossover

## Crossover audit

### Binomial crossover

- Current repository name: `binomial`
- Canonical trial formula:
  - Choose `j_rand` uniformly from `{0, ..., D - 1}`
  - For each coordinate `j`, use donor coordinate when `rand_j < CR` or `j == j_rand`
  - Otherwise keep the target coordinate
- Target vector exclusion: not applicable
- Minimum valid dimensionality: 1
- Required invariant: at least one coordinate must come from the donor
- Original implementation correctness:
  - Incorrect.
  - The condition was reversed and tied to `random_dimension != dimension`, which forced almost every coordinate to remain from the target.
  - `CR = 1` did not produce an all-donor trial vector.
  - `CR = 0` only worked accidentally for one coordinate, but for the wrong reason.
- Change made:
  - Reimplemented canonical binomial crossover.
  - Added deterministic tests for `CR = 0`, `CR = 1`, and donor-coordinate guarantees.

### Exponential crossover

- Current repository status: implemented in the refactor
- Canonical trial formula:
  - Choose a starting coordinate `j_start`
  - Copy a contiguous cyclic block from donor to target
  - Always copy at least one donor coordinate
  - Continue while fresh uniform samples are less than `CR`, until `D` coordinates have been copied
- Original implementation correctness:
  - Not applicable; no implementation existed.
- Change made:
  - Added a correct cyclic exponential crossover operator with deterministic tests.

## Execution-model audit

### Best-vector timing

- Canonical requirement:
  - The generation best used in `best/*` and `current-to-best/*` must come from the unmodified generation snapshot.
- Original implementation:
  - `get_best()` was called once before the per-target loop, which was correct.
  - The code structure made this easy to miss during maintenance.
- Change made:
  - The refactor snapshots both population and fitness at the start of each generation and passes `best_index` and `best_vector` through `MutationContext`.

### Immediate versus generational updates

- Canonical requirement:
  - Trial-vector generation for all targets in generation `g` should use the same original generation snapshot.
- Original implementation:
  - Replacement wrote into `self.population[idx]` during the loop.
  - The mutation formula happened to rely only on the sampled vectors, and sampled indices were drawn before replacement, but the code did not make the generation snapshot explicit.
- Change made:
  - The optimizer now creates `population_snapshot` and `fitness_snapshot` and writes winners into a separate `next_population`.

### RNG handling

- Original implementation:
  - Used module-level NumPy RNG functions directly.
- Change made:
  - The refactor uses an explicit RNG object, defaulting to `random.Random(seed)`.
  - A run is reproducible when created with the same seed.

## Sobol initialization audit

### Original implementation status

- File: the former `sobol_initialization` in [population_initialization.py](population_initialization.py)
- Result: incorrect
- Reasons:
  - It did not implement the standard direction-number recurrence.
  - It used Python's `^` operator, which is bitwise XOR, in places where the apparent intent was arithmetic recurrence.
  - It constructed one scalar quasirandom number and copied it into all coordinates, which is not a valid multi-dimensional Sobol sequence.

### Replacement implementation

- New component: `SobolInitializer` in [initializers.py](differential_evolution/initializers.py)
- Construction:
  - Uses the Bratley-Fox recurrence with primitive polynomials and initial direction numbers.
  - Generates the first `n` points including the zero vector.
  - Supports dimensions 1 through 40.
- Sampling properties:
  - deterministic
  - independent of the optimizer RNG
  - scaled coordinate-wise into the user-provided bounds

### Verified behavior

- The first eight points in 2D match the standard sequence:
  - `(0, 0)`
  - `(1/2, 1/2)`
  - `(3/4, 1/4)`
  - `(1/4, 3/4)`
  - `(3/8, 3/8)`
  - `(7/8, 7/8)`
  - `(5/8, 1/8)`
  - `(1/8, 5/8)`

## Additional non-DE corrections

- `ackley_function` in the original repository omitted the `0.5` factor inside the square root for the 2-D form documented in its own docstring.
- `schaffer_n2_function` used `x^2 + y^2` inside the sine term, while the documented function is based on `x^2 - y^2`.
- `testing_functions.himmelblaus_function` was renamed internally to the conventional `himmelblau_function`, with a compatibility alias preserved.

## Remaining limitations

- Sobol initialization currently supports only the first 40 dimensions, matching the embedded direction-number table.
- The optimizer currently supports minimization only.
- Boundary handling is explicit but limited to `none`, `clip`, and `random reset`.
- No separate stopping-criterion protocol has been added yet beyond `max_generations`; the control flow is separated cleanly enough that this can be added later without changing mutation or crossover APIs.

## Population diversity measures

The package now supports optional generation-by-generation recording of several diversity measures. These are diagnostics only; they do not alter selection or parameter updates.

- `PopulationDiameter`:
  - `max_{i,j} ||x_i - x_j||`
- `PopulationRadius`:
  - `max_i ||x_i - c||`, where `c` is the population center
- `AverageDistanceAroundPopulationCenter`:
  - `(1 / N) * sum_i ||x_i - c||`
- `AverageDistanceAroundAllIndividuals`:
  - `(1 / N) * sum_i [(1 / N) * sum_j ||x_i - x_j||]`
- `AveragePairwiseDistance`:
  - `(2 / (N * (N - 1))) * sum_{i < j} ||x_i - x_j||`
- `PopulationCoherence`:
  - `||c_g - c_{g-1}|| / [(1 / N) * sum_i ||x_i,g - x_i,g-1||]`
  - defined as `0` for the first recorded generation
- `DimensionalVariance`:
  - the mean per-dimension variance around the population center
- `AggregatedDistribution`:
  - project-specific definition
  - each coordinate is normalized into `[0, 1]`, partitioned into `ceil(sqrt(N))` bins, and summarized by the variance-to-mean ratio of the marginal occupancy counts

Two of these measures are intentionally both present even though they are closely related:

- `AverageDistanceAroundAllIndividuals` includes self-zero terms and counts ordered pairs
- `AveragePairwiseDistance` averages over unordered distinct pairs only

## Population reduction schedules

### `LinearPopulationReduction`

- Current package name: `LinearPopulationReduction`
- Canonical inspiration:
  - R. Tanabe and A. Fukunaga, "Improving the Search Performance of SHADE Using Linear Population Size Reduction," *2014 IEEE Congress on Evolutionary Computation*. DOI: https://doi.org/10.1109/CEC.2014.6900380
- Implemented rule:
  - let `progress` be `NFE / MAX_NFE` when `max_evaluations` is provided
  - otherwise let `progress` be `generation / max_generations`
  - `NP(progress) = round(NP_init + (NP_min - NP_init) * progress)`
- Survivor selection:
  - after each generation, remove the worst individuals by current fitness until the target size is reached
- Compatibility constraint:
  - `NP_min` must stay large enough for the chosen mutation strategy

### `HyperbolicTangentPopulationReduction`

- Current package name: `HyperbolicTangentPopulationReduction`
- Canonical status:
  - this is a documented project-specific nonlinear schedule inspired by tanh-based population reduction papers
- Implemented rule:
  - map normalized progress into `[start, end]`
  - convert it with a normalized `tanh` curve
  - interpolate population size between `NP_init` and `NP_min`
- Survivor selection:
  - worst individuals by current fitness are removed, as in the linear schedule

### Execution-model note

- Population reduction is applied after each completed generation step.
- If `max_evaluations` cuts a generation short, the processed subset is accepted, then the reduction schedule is applied to the resulting population state.

## SHADE and L-SHADE

### `SHADE`

- Current package name: `SHADE`
- Canonical sources:
  - R. Tanabe and A. Fukunaga, "Success-History Based Parameter Adaptation for Differential Evolution," *2013 IEEE Congress on Evolutionary Computation*. DOI: https://doi.org/10.1109/CEC.2013.6557555
- Implemented mutation/crossover core:
  - `current-to-pbest/1/bin`
  - `v_i,g = x_i,g + F_i * (x_pbest,g - x_i,g) + F_i * (x_r1,g - x_r2,g)`
- Parameter generation:
  - choose a memory slot `r_i`
  - sample `F_i` from a Cauchy distribution centered at `M_F[r_i]` with scale `0.1`, resampling while `F_i <= 0` and clipping at `1`
  - sample `CR_i` from a normal distribution centered at `M_CR[r_i]` with standard deviation `0.1`, clipping into `[0, 1]`
- `pbest` selection:
  - sample `p_i` from `U(2 / NP, p_max)` with `p_max = 0.2` by default
  - choose `x_pbest` uniformly from the top `ceil(p_i * NP)` individuals, excluding the target when possible
- Archive:
  - stores replaced parent vectors
  - contributes candidate `r2`
  - is randomly trimmed to the configured maximum size
- Memory update:
  - uses weighted Lehmer mean for successful `F`
  - uses weighted arithmetic mean for successful `CR`
  - weights are proportional to objective improvement

### `LSHADE`

- Current package name: `LSHADE`
- Canonical source:
  - R. Tanabe and A. Fukunaga, "Improving the Search Performance of SHADE Using Linear Population Size Reduction," *2014 IEEE Congress on Evolutionary Computation*. DOI: https://doi.org/10.1109/CEC.2014.6900380
- Implemented extension:
  - full `SHADE` machinery
  - linear population size reduction schedule
- Project decision:
  - `LSHADE` is exposed as a dedicated optimizer class rather than as a preset on the generic DE optimizer, because the success-history memory and archive are integral algorithmic components, not optional add-ons

## Scale-factor extensions

### `RandomizedScaleFactor`

- Current package name: `RandomizedScaleFactor`
- Interpretation used:
  - uniform dither of the differential weight
  - a fresh `F ~ Uniform(lower, upper)` is sampled for each target mutation
- Canonical status:
  - a standard DE control-parameter randomization technique
  - the exact per-generation versus per-vector sampling choice varies in the literature
- Decision made:
  - this package uses per-target sampling because it composes cleanly with the mutation-component API and is easy to test

### `AdaptiveScaleFactor`

- Current package name: `AdaptiveScaleFactor`
- Canonical source for the interpretation:
  - J. Brest, S. Greiner, B. Boskovic, M. Mernik, and V. Zumer, "Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical Benchmark Problems," *IEEE Transactions on Evolutionary Computation*, 2006. DOI: https://doi.org/10.1109/TEVC.2006.872133
- Interpretation used:
  - jDE-style self-adaptation of the scale factor `F`
  - each individual carries its own current `F_i`
  - before mutating target `i`, a new candidate `F_i'` is proposed with probability `tau`
  - if the trial vector wins selection, `F_i'` replaces `F_i`; otherwise the previous `F_i` is retained
- Decision made:
  - this package now also implements the matching `CR_i` adaptation on the crossover side

### `AdaptiveCrossoverRate`

- Current package name: `AdaptiveCrossoverRate`
- Canonical source for the interpretation:
  - J. Brest, S. Greiner, B. Boskovic, M. Mernik, and V. Zumer, "Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical Benchmark Problems," *IEEE Transactions on Evolutionary Computation*, 2006. DOI: https://doi.org/10.1109/TEVC.2006.872133
- Interpretation used:
  - jDE-style self-adaptation of the crossover rate `CR`
  - each individual carries its own `CR_i`
  - before crossover for target `i`, a new candidate `CR_i'` is proposed with probability `tau`
  - if the trial vector wins selection, `CR_i'` replaces `CR_i`; otherwise the previous `CR_i` is retained
- Decision made:
  - the controller is implemented independently of the crossover operator so it can drive binomial or exponential crossover
  - the `jde_rand_1_bin()` helper returns the mutation/crossover pairing closest to Brest et al.'s published jDE setup
