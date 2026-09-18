from __future__ import annotations

import math
import random
import tempfile
import unittest
import warnings

from differential_evolution import (
    AggregatedDistribution,
    AverageDistanceAroundAllIndividuals,
    AverageDistanceAroundPopulationCenter,
    AveragePairwiseDistance,
    Best1,
    Best2,
    BinomialCrossover,
    ClipBoundaryHandler,
    CurrentToBest1,
    CurrentToBest2,
    CurrentToRand1,
    CurrentToRand2,
    DifferentialEvolution,
    DimensionalVariance,
    DirectedMutation,
    AdaptiveCrossoverRate,
    ExponentialCrossover,
    IdentityCrossover,
    MutationContext,
    NeighborhoodSearchMutation,
    NoBoundaryHandler,
    PopulationCoherence,
    PopulationDiameter,
    PopulationRadius,
    LinearPopulationReduction,
    HyperbolicTangentPopulationReduction,
    Rand1,
    Rand2,
    RandomResetBoundaryHandler,
    RandomizedScaleFactor,
    SobolInitializer,
    TrigonometricMutation,
    AdaptiveScaleFactor,
    SHADE,
    LSHADE,
    jde_rand_1_bin,
    ackley_function,
    schaffer_n2_function,
    sphere_function,
)
from differential_evolution.compat import DifferentialEvolution as LegacyDifferentialEvolution
from differential_evolution.mutation import sample_distinct_indices
from differential_evolution.population_schedules import PopulationScheduleState


class DeterministicRNG:
    def __init__(
        self,
        *,
        samples: list[list[int]] | None = None,
        gauss_values: list[float] | None = None,
        random_values: list[float] | None = None,
        randrange_values: list[int] | None = None,
        uniform_values: list[float] | None = None,
    ) -> None:
        self.samples = list(samples or [])
        self.gauss_values = list(gauss_values or [])
        self.random_values = list(random_values or [])
        self.randrange_values = list(randrange_values or [])
        self.uniform_values = list(uniform_values or [])

    def sample(self, population, k):
        sample = self.samples.pop(0)
        if len(sample) != k:
            raise AssertionError(f"Expected sample length {k}, received {sample!r}")
        return list(sample)

    def random(self):
        return self.random_values.pop(0)

    def randrange(self, stop):
        value = self.randrange_values.pop(0)
        if not 0 <= value < stop:
            raise AssertionError(f"randrange value {value} is invalid for stop={stop}")
        return value

    def gauss(self, mean, stddev):
        return self.gauss_values.pop(0)

    def uniform(self, lower, upper):
        if self.uniform_values:
            return self.uniform_values.pop(0)
        return lower + (upper - lower) * self.random()


def make_context(rng: DeterministicRNG) -> MutationContext:
    population = [
        [10.0, 20.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
    ]
    fitness = [500.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    return MutationContext(
        population=population,
        fitness=fitness,
        target_index=0,
        best_index=1,
        best_vector=list(population[1]),
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        rng=rng,
    )


def make_diversity_population() -> list[list[float]]:
    return [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]


class MutationFormulaTests(unittest.TestCase):
    def test_rand1_formula(self):
        context = make_context(DeterministicRNG(samples=[[1, 2, 3]]))
        donor = Rand1(scale=0.5)(context)
        self.assertEqual(donor, [0.0, 1.0])

    def test_rand2_formula(self):
        context = make_context(DeterministicRNG(samples=[[1, 2, 3, 4, 5]]))
        donor = Rand2(scale=0.5)(context)
        self.assertEqual(donor, [-1.0, 0.0])

    def test_best1_formula(self):
        context = make_context(DeterministicRNG(samples=[[2, 3]]))
        donor = Best1(scale=0.5)(context)
        self.assertEqual(donor, [0.0, 1.0])

    def test_best2_formula(self):
        context = make_context(DeterministicRNG(samples=[[2, 3, 4, 5]]))
        donor = Best2(scale=0.5)(context)
        self.assertEqual(donor, [-1.0, 0.0])

    def test_current_to_best1_formula(self):
        context = make_context(DeterministicRNG(samples=[[2, 3]]))
        donor = CurrentToBest1(scale=0.25, difference_scale=0.5)(context)
        self.assertEqual(donor, [6.75, 14.5])

    def test_current_to_best2_formula(self):
        context = make_context(DeterministicRNG(samples=[[2, 3, 4, 5]]))
        donor = CurrentToBest2(scale=0.25, difference_scale=0.5)(context)
        self.assertEqual(donor, [5.75, 13.5])

    def test_current_to_rand1_formula(self):
        context = make_context(DeterministicRNG(samples=[[2, 3, 4]]))
        donor = CurrentToRand1(scale=0.25, difference_scale=0.5)(context)
        self.assertEqual(donor, [7.25, 15.0])

    def test_current_to_rand2_formula(self):
        context = make_context(DeterministicRNG(samples=[[2, 3, 4, 1, 5]]))
        donor = CurrentToRand2(scale=0.25, difference_scale=0.5)(context)
        self.assertEqual(donor, [3.25, 11.0])

    def test_population_size_validation_for_sampling(self):
        with self.assertRaisesRegex(ValueError, "Population too small"):
            sample_distinct_indices(4, 3, random.Random(1), excluded=(0, 1))

    def test_randomized_scale_factor_changes_mutation_weight(self):
        context = make_context(DeterministicRNG(samples=[[1, 2, 3]], uniform_values=[0.25]))
        donor = Rand1(scale=RandomizedScaleFactor(lower=0.1, upper=0.9))(context)
        self.assertEqual(donor, [0.5, 1.5])

    def test_trigonometric_mutation_formula(self):
        context = make_context(DeterministicRNG(samples=[[1, 2, 3]], random_values=[0.2]))
        donor = TrigonometricMutation(probability=1.0, scale=0.5)(context)
        self.assertAlmostEqual(donor[0], 1.0)
        self.assertAlmostEqual(donor[1], 2.0)

    def test_trigonometric_mutation_falls_back_to_rand1(self):
        context = make_context(DeterministicRNG(samples=[[1, 2, 3]], random_values=[0.9]))
        donor = TrigonometricMutation(probability=0.5, scale=0.5)(context)
        self.assertEqual(donor, [0.0, 1.0])

    def test_trigonometric_mutation_uses_equal_weights_when_selected_fitness_sum_is_zero(self):
        context = MutationContext(
            population=make_context(DeterministicRNG()).population,
            fitness=[10.0, 0.0, 0.0, 0.0, 4.0, 5.0],
            target_index=0,
            best_index=1,
            best_vector=[1.0, 2.0],
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            rng=DeterministicRNG(samples=[[1, 2, 3]], random_values=[0.1]),
        )
        donor = TrigonometricMutation(probability=1.0, scale=0.5)(context)
        self.assertEqual(donor, [3.0, 4.0])

    def test_neighborhood_search_mutation_gaussian_branch(self):
        rng = DeterministicRNG(samples=[[1, 2, 3]], random_values=[0.2], gauss_values=[0.25])
        donor = NeighborhoodSearchMutation(
            gaussian_probability=0.5,
            gaussian_mean=0.5,
            gaussian_stddev=0.5,
            cauchy_scale=1.0,
        )(make_context(rng))
        self.assertEqual(donor, [0.5, 1.5])

    def test_neighborhood_search_mutation_cauchy_branch(self):
        rng = DeterministicRNG(samples=[[1, 2, 3]], random_values=[0.8, 0.75])
        donor = NeighborhoodSearchMutation(
            gaussian_probability=0.5,
            gaussian_mean=0.5,
            gaussian_stddev=0.5,
            cauchy_scale=1.0,
        )(make_context(rng))
        expected_factor = math.tan(math.pi * 0.25)
        self.assertAlmostEqual(donor[0], 1.0 + expected_factor * (3.0 - 5.0))
        self.assertAlmostEqual(donor[1], 2.0 + expected_factor * (4.0 - 6.0))

    def test_directed_mutation_formula(self):
        context = make_context(DeterministicRNG(samples=[[2, 3, 4]]))
        donor = DirectedMutation()(context)
        self.assertAlmostEqual(donor[0], -2.0)
        self.assertAlmostEqual(donor[1], -1.0)

    def test_directed_mutation_falls_back_when_coefficients_are_undefined(self):
        context = MutationContext(
            population=make_context(DeterministicRNG()).population,
            fitness=[10.0, 1.0, math.inf, 3.0, 4.0, 5.0],
            target_index=0,
            best_index=1,
            best_vector=[1.0, 2.0],
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            rng=DeterministicRNG(samples=[[2, 3, 4]]),
        )
        donor = DirectedMutation(fallback_scale=0.5)(context)
        self.assertEqual(donor, [2.0, 3.0])


class DiversityMeasureTests(unittest.TestCase):
    def test_population_diameter(self):
        value = PopulationDiameter()(make_diversity_population(), [(0.0, 2.0), (0.0, 2.0)])
        self.assertAlmostEqual(value, math.sqrt(8.0))

    def test_population_radius(self):
        value = PopulationRadius()(make_diversity_population(), [(0.0, 2.0), (0.0, 2.0)])
        self.assertAlmostEqual(value, math.sqrt(20.0) / 3.0)

    def test_average_distance_around_population_center(self):
        value = AverageDistanceAroundPopulationCenter()(
            make_diversity_population(),
            [(0.0, 2.0), (0.0, 2.0)],
        )
        self.assertAlmostEqual(value, (math.sqrt(8.0) + 2.0 * math.sqrt(20.0)) / 9.0)

    def test_average_distance_around_all_individuals(self):
        value = AverageDistanceAroundAllIndividuals()(
            make_diversity_population(),
            [(0.0, 2.0), (0.0, 2.0)],
        )
        self.assertAlmostEqual(value, (8.0 + 2.0 * math.sqrt(8.0)) / 9.0)

    def test_average_pairwise_distance(self):
        value = AveragePairwiseDistance()(make_diversity_population(), [(0.0, 2.0), (0.0, 2.0)])
        self.assertAlmostEqual(value, (4.0 + math.sqrt(8.0)) / 3.0)

    def test_population_coherence(self):
        value = PopulationCoherence()(
            make_diversity_population(),
            [(0.0, 2.0), (0.0, 2.0)],
            previous_population=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        self.assertAlmostEqual(value, math.sqrt(0.5))

    def test_dimensional_variance(self):
        value = DimensionalVariance()(make_diversity_population(), [(0.0, 2.0), (0.0, 2.0)])
        self.assertAlmostEqual(value, 8.0 / 9.0)

    def test_aggregated_distribution(self):
        value = AggregatedDistribution()(make_diversity_population(), [(0.0, 2.0), (0.0, 2.0)])
        self.assertAlmostEqual(value, 1.0 / 6.0)


class PopulationScheduleTests(unittest.TestCase):
    def test_linear_population_reduction_matches_lshade_style_progress(self):
        schedule = LinearPopulationReduction(min_population_size=4)
        state = PopulationScheduleState(
            generation=5,
            max_generations=10,
            evaluations=50,
            max_evaluations=100,
            initial_population_size=20,
            current_population_size=20,
            min_population_size=4,
        )
        self.assertEqual(schedule.target_size(state), 12)

    def test_hyperbolic_tangent_population_reduction_is_monotone(self):
        schedule = HyperbolicTangentPopulationReduction(min_population_size=4)
        sizes = []
        for evaluations in (0, 25, 50, 75, 100):
            sizes.append(
                schedule.target_size(
                    PopulationScheduleState(
                        generation=0,
                        max_generations=10,
                        evaluations=evaluations,
                        max_evaluations=100,
                        initial_population_size=20,
                        current_population_size=20,
                        min_population_size=4,
                    )
                )
            )
        self.assertEqual(sizes[0], 20)
        self.assertEqual(sizes[-1], 4)
        self.assertTrue(all(left >= right for left, right in zip(sizes, sizes[1:])))


class CrossoverTests(unittest.TestCase):
    def test_binomial_guarantees_one_donor_coordinate_at_cr_zero(self):
        crossover = BinomialCrossover(crossover_rate=0.0)
        trial = crossover(
            [1.0, 1.0, 1.0],
            [9.0, 8.0, 7.0],
            DeterministicRNG(randrange_values=[1], random_values=[0.5, 0.5]),
        )
        self.assertEqual(trial, [1.0, 8.0, 1.0])

    def test_binomial_uses_all_donor_coordinates_at_cr_one(self):
        crossover = BinomialCrossover(crossover_rate=1.0)
        trial = crossover(
            [1.0, 1.0],
            [9.0, 8.0],
            DeterministicRNG(randrange_values=[0], random_values=[0.9]),
        )
        self.assertEqual(trial, [9.0, 8.0])

    def test_exponential_wraps_around(self):
        crossover = ExponentialCrossover(crossover_rate=0.6)
        trial = crossover(
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
            DeterministicRNG(randrange_values=[3], random_values=[0.2, 0.3, 0.9]),
        )
        self.assertEqual(trial, [1.0, 2.0, 0.0, 4.0])

    def test_identity_crossover_returns_donor(self):
        donor = [3.0, 4.0]
        trial = IdentityCrossover()([0.0, 0.0], donor, random.Random(1))
        self.assertEqual(trial, donor)

    def test_adaptive_crossover_rate_changes_binomial_behavior(self):
        crossover = BinomialCrossover(
            crossover_rate=AdaptiveCrossoverRate(initial=0.25, tau=1.0, lower=0.25, upper=0.25)
        )
        crossover.initialize(population_size=2, rng=random.Random(1))
        crossover.set_target_index(1)
        trial = crossover(
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            DeterministicRNG(randrange_values=[2], random_values=[0.0, 0.0, 0.5], uniform_values=[0.25]),
        )
        self.assertEqual(trial, [1.0, 0.0, 3.0])
        crossover.commit(1, accepted=True)
        self.assertEqual(crossover.crossover_rate.values(), [0.25, 0.25])


class OptimizerTests(unittest.TestCase):
    def test_components_are_composable(self):
        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            population_size=8,
            mutation=Best1(scale=0.5),
            crossover=ExponentialCrossover(crossover_rate=0.8),
            max_generations=3,
            seed=7,
        )
        result = optimizer.run()
        self.assertEqual(result.nit, 3)
        self.assertEqual(len(result.x), 2)

    def test_custom_components_can_be_injected(self):
        class FixedMutation:
            def __call__(self, context):
                return [42.0 for _ in context.population[context.target_index]]

        class FirstCoordinateCrossover:
            def __call__(self, target_vector, donor_vector, rng):
                trial = list(target_vector)
                trial[0] = donor_vector[0]
                return trial

        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            population_size=5,
            mutation=FixedMutation(),
            crossover=FirstCoordinateCrossover(),
            max_generations=1,
            seed=1,
        )
        result = optimizer.run()
        self.assertEqual(result.nit, 1)
        self.assertEqual(result.nfev, 10)

    def test_reproducibility_with_fixed_seed(self):
        kwargs = dict(
            objective=sphere_function,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            population_size=12,
            mutation=Rand1(scale=0.8),
            crossover=BinomialCrossover(crossover_rate=0.7),
            max_generations=5,
            seed=11,
        )
        result_one = DifferentialEvolution(**kwargs).run()
        result_two = DifferentialEvolution(**kwargs).run()
        self.assertEqual(result_one.x, result_two.x)
        self.assertEqual(result_one.best_fitness_history, result_two.best_fitness_history)

    def test_generational_update_uses_snapshot_population(self):
        class CountingMutation:
            def __init__(self):
                self.first_generation_targets = []

            def __call__(self, context):
                if len(self.first_generation_targets) < 4:
                    self.first_generation_targets.append(context.population[context.target_index][0])
                target = context.population[context.target_index][0]
                return [target - 1.0]

        mutation = CountingMutation()
        optimizer = DifferentialEvolution(
            objective=lambda x: x[0] ** 2,
            bounds=[(-10.0, 10.0)],
            population_size=4,
            mutation=mutation,
            crossover=IdentityCrossover(),
            max_generations=1,
            seed=3,
        )
        optimizer.population = [[4.0], [3.0], [2.0], [1.0]]
        optimizer.fitness = [16.0, 9.0, 4.0, 1.0]
        optimizer._step()
        self.assertEqual(optimizer.mutation.first_generation_targets, [4.0, 3.0, 2.0, 1.0])
        self.assertEqual(optimizer.population, [[3.0], [2.0], [1.0], [0.0]])

    def test_trial_vectors_do_not_modify_snapshot_population_in_place(self):
        class MirrorMutation:
            def __call__(self, context):
                return [-value for value in context.population[context.target_index]]

        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            population_size=4,
            mutation=MirrorMutation(),
            crossover=IdentityCrossover(),
            max_generations=1,
            seed=2,
        )
        optimizer.population = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        optimizer.fitness = [5.0, 13.0, 25.0, 41.0]
        snapshot = [list(individual) for individual in optimizer.population]
        optimizer._step()
        self.assertEqual(snapshot, [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])

    def test_clip_boundary_handler(self):
        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-1.0, 1.0)],
            population_size=4,
            mutation=lambda context: [5.0],
            crossover=IdentityCrossover(),
            boundary_handler=ClipBoundaryHandler(),
            max_generations=1,
            seed=1,
        )
        optimizer.population = [[0.5], [0.25], [0.1], [0.0]]
        optimizer.fitness = [0.25, 0.0625, 0.01, 0.0]
        optimizer._step()
        self.assertEqual(optimizer.population[-1], [0.0])

    def test_random_reset_boundary_handler(self):
        handler = RandomResetBoundaryHandler()
        bounded = handler([3.0, 0.5], [(-1.0, 1.0), (0.0, 1.0)], DeterministicRNG(uniform_values=[-0.25]))
        self.assertEqual(bounded, [-0.25, 0.5])

    def test_non_finite_objective_values_are_treated_as_infinite(self):
        optimizer = DifferentialEvolution(
            objective=lambda x: math.nan if x[0] > 0 else x[0] ** 2,
            bounds=[(-1.0, 1.0)],
            population_size=4,
            mutation=lambda context: [1.0],
            crossover=IdentityCrossover(),
            max_generations=1,
            seed=4,
        )
        optimizer.population = [[-0.5], [-0.25], [0.0], [-0.75]]
        optimizer.fitness = [0.25, 0.0625, 0.0, 0.5625]
        optimizer._step()
        self.assertEqual(optimizer.fitness, [0.25, 0.0625, 0.0, 0.5625])

    def test_evaluation_count_matches_initialization_plus_trials(self):
        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            population_size=6,
            mutation=Rand1(scale=0.5),
            crossover=BinomialCrossover(crossover_rate=0.5),
            max_generations=4,
            seed=3,
        )
        result = optimizer.run()
        self.assertEqual(result.nfev, 6 + 6 * 4)

    def test_invalid_dimension_raises_informative_error(self):
        optimizer = DifferentialEvolution(
            objective=lambda x: 0.0,
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            population_size=4,
            mutation=lambda context: [0.0, 0.0],
            crossover=IdentityCrossover(),
            max_generations=0,
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "dimension"):
            optimizer._evaluate_vector([1.0])

    def test_adaptive_scale_factor_updates_only_after_accepted_trial(self):
        scale_factor = AdaptiveScaleFactor(initial=0.5, tau=1.0, lower=0.1, upper=0.9)
        optimizer = DifferentialEvolution(
            objective=lambda x: x[0],
            bounds=[(-10.0, 10.0)],
            population_size=4,
            mutation=Rand1(scale=scale_factor),
            crossover=IdentityCrossover(),
            max_generations=1,
            rng=DeterministicRNG(
                samples=[[1, 2, 3], [0, 2, 3], [1, 0, 3], [0, 1, 2]],
                random_values=[0.0, 0.0, 0.0, 0.0],
                uniform_values=[0.2, 0.3, 0.4, 0.7],
            ),
        )
        optimizer.population = [[4.0], [1.0], [3.0], [2.0]]
        optimizer.fitness = [4.0, 1.0, 3.0, 2.0]
        optimizer._step()
        self.assertEqual(optimizer.mutation.scale.values(), [0.2, 0.5, 0.4, 0.5])

    def test_adaptive_scale_factor_reuses_same_value_within_one_target(self):
        scale_factor = AdaptiveScaleFactor(initial=0.5, tau=1.0, lower=0.1, upper=0.9)
        rng = DeterministicRNG(random_values=[0.0], uniform_values=[0.3])
        scale_factor.initialize(population_size=2, rng=rng)
        context = MutationContext(
            population=[[1.0], [2.0]],
            fitness=[1.0, 2.0],
            target_index=0,
            best_index=0,
            best_vector=[1.0],
            bounds=[(-1.0, 1.0)],
            rng=rng,
        )
        self.assertEqual(scale_factor.propose(context), 0.3)
        self.assertEqual(scale_factor.propose(context), 0.3)
        scale_factor.commit(0, accepted=False)
        self.assertEqual(scale_factor.values(), [0.5, 0.5])

    def test_adaptive_crossover_rate_updates_only_after_accepted_trial(self):
        crossover_rate = AdaptiveCrossoverRate(initial=0.9, tau=1.0, lower=0.1, upper=0.9)
        optimizer = DifferentialEvolution(
            objective=lambda x: x[0],
            bounds=[(-10.0, 10.0)],
            population_size=4,
            mutation=lambda context: [0.0],
            crossover=BinomialCrossover(crossover_rate=crossover_rate),
            max_generations=1,
            rng=DeterministicRNG(
                randrange_values=[0, 0, 0, 0],
                random_values=[0.0, 0.0, 0.0, 0.0],
                uniform_values=[0.2, 0.3, 0.4, 0.7],
            ),
        )
        optimizer.population = [[4.0], [1.0], [3.0], [2.0]]
        optimizer.fitness = [4.0, 1.0, 3.0, 2.0]
        optimizer._step()
        self.assertEqual(optimizer.crossover.crossover_rate.values(), [0.2, 0.3, 0.4, 0.7])

    def test_jde_convenience_helper_builds_paper_components(self):
        components = jde_rand_1_bin()
        self.assertIsInstance(components.mutation, Rand1)
        self.assertIsInstance(components.crossover, BinomialCrossover)
        self.assertIsInstance(components.mutation.scale, AdaptiveScaleFactor)
        self.assertIsInstance(components.crossover.crossover_rate, AdaptiveCrossoverRate)

    def test_diversity_history_tracks_multiple_measures(self):
        class FixedInitializer:
            def __call__(self, population_size, bounds, objective, rng):
                return make_diversity_population()

        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(0.0, 2.0), (0.0, 2.0)],
            population_size=3,
            mutation=lambda context: list(context.population[context.target_index]),
            crossover=IdentityCrossover(),
            initializer=FixedInitializer(),
            diversity_measures=["population_diameter", "population_coherence", "average_pairwise_distance"],
            max_generations=1,
            seed=1,
        )
        result = optimizer.run()
        self.assertEqual(sorted(result.diversity_history), [
            "average_pairwise_distance",
            "population_coherence",
            "population_diameter",
        ])
        self.assertEqual(len(result.diversity_history["population_diameter"]), 2)
        self.assertEqual(result.diversity_history["population_coherence"], [0.0, 0.0])
        self.assertAlmostEqual(result.diversity_history["population_diameter"][0], math.sqrt(8.0))

    def test_no_diversity_measures_produces_empty_history(self):
        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-1.0, 1.0)],
            population_size=4,
            mutation=lambda context: [0.0],
            crossover=IdentityCrossover(),
            max_generations=0,
            seed=1,
        )
        result = optimizer.run()
        self.assertEqual(result.diversity_history, {})
        self.assertEqual(result.population_size_history, [4])

    def test_linear_population_schedule_removes_worst_individuals(self):
        optimizer = DifferentialEvolution(
            objective=lambda x: x[0],
            bounds=[(-10.0, 10.0)],
            population_size=4,
            mutation=lambda context: [context.population[context.target_index][0]],
            crossover=IdentityCrossover(),
            population_schedule=LinearPopulationReduction(min_population_size=2),
            max_generations=1,
            max_evaluations=4,
            seed=1,
        )
        optimizer.population = [[4.0], [1.0], [3.0], [2.0]]
        optimizer.fitness = [4.0, 1.0, 3.0, 2.0]
        optimizer._step()
        self.assertEqual(optimizer.population, [[1.0], [2.0]])
        self.assertEqual(optimizer.fitness, [1.0, 2.0])
        self.assertEqual(optimizer.population_size_history[-1], 2)

    def test_population_schedule_resizes_adaptive_controllers(self):
        optimizer = DifferentialEvolution(
            objective=sphere_function, bounds=[(-10., 10.)], population_size=6,
            mutation=Rand1(scale=AdaptiveScaleFactor()),
            crossover=BinomialCrossover(AdaptiveCrossoverRate()),
            population_schedule=LinearPopulationReduction(min_population_size=4),
            max_generations=1, seed=1,
        )
        optimizer.run()
        self.assertEqual(len(optimizer.mutation.scale.values()), 4)
        self.assertEqual(len(optimizer.crossover.crossover_rate.values()), 4)

    def test_partial_generation_respects_max_evaluations(self):
        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-1.0, 1.0)],
            population_size=4,
            mutation=lambda context: [0.0],
            crossover=IdentityCrossover(),
            max_generations=5,
            max_evaluations=6,
            seed=1,
        )
        result = optimizer.run()
        self.assertEqual(result.nfev, 6)
        self.assertEqual(result.nit, 1)

    def test_result_can_save_snapshots(self):
        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-1.0, 1.0)],
            population_size=4,
            mutation=lambda context: [0.0],
            crossover=IdentityCrossover(),
            max_generations=1,
            record_snapshots=True,
            seed=1,
        )
        result = optimizer.run()
        with tempfile.TemporaryDirectory() as directory:
            json_path = f"{directory}/result.json"
            pickle_path = f"{directory}/result.pkl"
            result.save_json(json_path)
            result.save_pickle(pickle_path)
            with open(json_path, "r", encoding="utf-8") as handle:
                contents = handle.read()
            self.assertIn("\"snapshots\"", contents)
            self.assertTrue(len(result.snapshots) >= 1)


class SHADETests(unittest.TestCase):
    def test_shade_updates_archive_and_memory_after_success(self):
        optimizer = SHADE(
            objective=lambda x: x[0],
            bounds=[(-10.0, 10.0)],
            population_size=4,
            max_generations=1,
            max_evaluations=5,
            memory_size=2,
            rng=DeterministicRNG(
                randrange_values=[0, 3, 0],
                random_values=[0.5],
                gauss_values=[1.0],
                uniform_values=[0.5],
                samples=[[1], [2]],
            ),
        )
        optimizer.population = [[4.0], [1.0], [3.0], [2.0]]
        optimizer.fitness = [4.0, 1.0, 3.0, 2.0]
        optimizer.nfev = 4
        optimizer._step()
        self.assertEqual(optimizer.archive, [[4.0]])
        self.assertEqual(optimizer.memory_f[0], 0.5)
        self.assertEqual(optimizer.memory_cr[0], 1.0)
        self.assertEqual(optimizer.memory_index, 1)

    def test_shade_records_archive_and_memory_in_snapshots(self):
        optimizer = SHADE(
            objective=sphere_function,
            bounds=[(-1.0, 1.0)],
            population_size=6,
            max_generations=1,
            record_snapshots=True,
            seed=1,
        )
        result = optimizer.run()
        self.assertTrue(result.snapshots)
        self.assertIn("archive", result.snapshots[0].extra)
        self.assertIn("memory_f", result.snapshots[0].extra)

    def test_lshade_reduces_population_size(self):
        optimizer = LSHADE(
            objective=sphere_function,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            population_size=10,
            min_population_size=4,
            max_generations=5,
            max_evaluations=40,
            seed=7,
        )
        result = optimizer.run()
        self.assertLess(result.population_size_history[-1], result.population_size_history[0])
        self.assertEqual(result.metadata["algorithm"], "LSHADE")


class CompatibilityTests(unittest.TestCase):
    def test_legacy_wrapper_runs_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer = LegacyDifferentialEvolution(
                sphere_function,
                bounds=[[-5.0, 5.0], [-5.0, 5.0]],
                max_iterations=2,
                population_size=8,
                mutation=[0.5, 0.25],
                crossover=0.7,
                strategy="DE/current-to-best/1",
                population_initialization_algorithm="random",
                seed=9,
            )
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        optimizer.initialize()
        optimizer.evolve()
        best = optimizer.get_best()
        self.assertEqual(len(best), 2)
        self.assertEqual(optimizer.generation, 2)

    def test_legacy_measure_diversity_supports_new_population_metrics(self):
        optimizer = LegacyDifferentialEvolution(
            sphere_function,
            bounds=[[0.0, 2.0], [0.0, 2.0]],
            max_iterations=0,
            population_size=4,
            mutation=[0.5],
            crossover=0.5,
            strategy="DE/rand/1",
            population_initialization_algorithm="random",
            seed=1,
        )
        optimizer.population = make_diversity_population()
        self.assertAlmostEqual(optimizer.measure_diversity("population_diameter"), math.sqrt(8.0))

    def test_legacy_sobol_initializer_runs(self):
        optimizer = LegacyDifferentialEvolution(
            sphere_function,
            bounds=[[-5.0, 5.0], [-5.0, 5.0]],
            max_iterations=1,
            population_size=4,
            mutation=[0.5],
            crossover=0.5,
            strategy="DE/rand/1",
            population_initialization_algorithm="sobol",
        )
        optimizer.initialize()
        self.assertEqual(len(optimizer.population), 4)
        self.assertTrue(all(-5 <= value <= 5 for point in optimizer.population for value in point))


class SobolInitializerTests(unittest.TestCase):
    def test_first_points_in_two_dimensions_match_known_sequence(self):
        points = SobolInitializer(scramble=False)(
            8,
            [(0.0, 1.0), (0.0, 1.0)],
            lambda _: 0.0,
            random.Random(123),
        )
        self.assertEqual(
            points,
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.375, 0.375],
                [0.875, 0.875],
                [0.625, 0.125],
                [0.125, 0.625],
            ],
        )

    def test_scaling_to_bounds(self):
        points = SobolInitializer(scramble=False)(
            4,
            [(-1.0, 1.0), (10.0, 20.0)],
            lambda _: 0.0,
            random.Random(1),
        )
        self.assertEqual(
            points,
            [[-1.0, 10.0], [0.0, 15.0], [0.5, 12.5], [-0.5, 17.5]],
        )

    def test_dimension_limit_is_reported(self):
        with self.assertRaisesRegex(ValueError, "supports dimensions 1 through 40"):
            SobolInitializer()(2, [(0.0, 1.0)] * 41, lambda _: 0.0, random.Random(1))


class BenchmarkRegressionTests(unittest.TestCase):
    def test_ackley_global_minimum(self):
        self.assertAlmostEqual(ackley_function([0.0, 0.0]), 0.0, places=12)

    def test_schaffer_n2_global_minimum(self):
        self.assertAlmostEqual(schaffer_n2_function([0.0, 0.0]), 0.0, places=12)

    def test_smoke_optimization_on_sphere(self):
        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            population_size=20,
            mutation=Rand1(scale=0.8),
            crossover=BinomialCrossover(crossover_rate=0.9),
            boundary_handler=NoBoundaryHandler(),
            max_generations=25,
            seed=21,
        )
        result = optimizer.run()
        self.assertLess(result.fun, 0.1)


if __name__ == "__main__":
    unittest.main()
