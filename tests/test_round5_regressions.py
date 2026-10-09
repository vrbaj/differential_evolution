"""Behaviour regressions for review_05_yuwen/REVIEW04.md."""

import math
import random
import unittest
from unittest.mock import Mock, patch

import differential_evolution as de
from differential_evolution._numeric import sum_differences
from differential_evolution.diversity import _population_center
from differential_evolution.population_schedules import PopulationScheduleState


class ArchiveRegressions(unittest.TestCase):
    def test_ties_replace_targets_without_archiving_or_adapting(self):
        for cls in (de.SHADE, de.LSHADE):
            with self.subTest(cls=cls):
                options = {'min_population_size': 10} if cls is de.LSHADE else {}
                opt = cls(lambda x: 1, [(-5, 5)] * 2, 10, max_evaluations=40, seed=1, **options)
                opt.initialize()
                before = [vector[:] for vector in opt.population]
                result = opt.run()
                self.assertEqual(result.nit, 3)
                self.assertNotEqual(before, opt.population)
                self.assertEqual(opt.archive, [])
                self.assertEqual(opt.memory_f, [0.5] * opt.memory_size)
                self.assertEqual(opt.memory_cr, [0.5] * opt.memory_size)
                self.assertEqual(opt.memory_index, 0)

    def test_strict_infinite_improvements_archive_but_do_not_adapt(self):
        for cls in (de.SHADE, de.LSHADE):
            with self.subTest(cls=cls):
                opt = cls(lambda x: 1, [(-5, 5)], 4, max_evaluations=8, seed=1)
                opt.initialize()
                before = [vector[:] for vector in opt.population]
                opt.fitness = [1, math.inf, -math.inf, 0]
                opt.objective = lambda x: -math.inf
                opt.run()
                self.assertEqual(opt.archive, [before[i] for i in (0, 1, 3)])
                self.assertEqual(opt.memory_f, [0.5] * opt.memory_size)
                self.assertEqual(opt.memory_cr, [0.5] * opt.memory_size)
                self.assertEqual(opt.memory_index, 0)

    def test_actual_r2_draws_only_use_the_generation_start_archive(self):
        for cls in (de.SHADE, de.LSHADE):
            for budget in (6, 8, 10):
                with self.subTest(cls=cls, budget=budget):
                    opt = cls(lambda x: 100, [(-5, 5)], 4, max_evaluations=budget, seed=2,
                              archive_rate=4)
                    opt.initialize()
                    opt.archive = [[-4.5]]
                    opt.objective = lambda x, opt=opt: -opt.nfev
                    sample = opt._sample_r2_vector
                    while opt.nfev < budget:
                        start_archive = [vector[:] for vector in opt.archive]
                        parents = [vector[:] for vector in opt.population]
                        drawn = []

                        def draw(target, p_best, r1, opt=opt, sample=sample, drawn=drawn):
                            # Exercise the real sampler, always requesting its
                            # final archive slot. Newly inserted parents must
                            # not expand/change that pool until the next step.
                            with patch.object(opt._rng, 'randrange', side_effect=lambda stop: stop - 1):
                                vector = sample(target, p_best, r1)
                            drawn.append(vector)
                            return vector

                        with patch.object(opt, '_sample_r2_vector', side_effect=draw):
                            processed = opt._step()
                        self.assertEqual(drawn, [start_archive[-1]] * processed)
                        self.assertEqual(opt.archive, start_archive + parents[:processed])
                    self.assertEqual(opt.nfev, budget)

    def test_failed_generation_does_not_publish_buffered_parents(self):
        opt = de.SHADE(lambda x: 100, [(-5, 5)], 4, max_evaluations=8, seed=1)
        opt.initialize()
        before = [vector[:] for vector in opt.population]
        opt.objective = Mock(side_effect=[0, ValueError('objective failed')])
        with self.assertRaisesRegex(ValueError, 'objective failed'):
            opt.run()
        self.assertEqual(opt.archive, [])
        self.assertEqual(opt.population, before)
        self.assertFalse(hasattr(opt, '_generation_population'))


class ReferencePolicyRegressions(unittest.TestCase):
    def test_lshade_pbest_halfway_sizes_use_ties_to_even(self):
        for size, expected in ((150, 16), (250, 28)):
            opt = de.LSHADE(de.sphere_function, [(-5, 5)], size, max_evaluations=size, seed=1)
            with patch.object(opt._rng, 'sample', return_value=[0]) as sample:
                opt._sample_p_best_index(0, list(range(size)))
            self.assertEqual(list(sample.call_args.args[0]), list(range(expected)))

    def test_population_reduction_halfway_sizes_use_ties_to_even(self):
        schedule = de.LinearPopulationReduction(4)
        for evaluations, budget, expected in ((20, 80, 8), (50, 120, 8)):
            state = PopulationScheduleState(1, None, evaluations, budget, 10, 10, 4)
            self.assertEqual(schedule.target_size(state), expected)

    def test_archive_capacity_halfway_sizes_use_ties_to_even(self):
        for size, expected in ((5, 2), (7, 4)):
            opt = de.SHADE(de.sphere_function, [(-5, 5)], size,
                           max_evaluations=size, archive_rate=0.5, seed=1)
            opt.initialize()
            opt.archive = [[float(i)] for i in range(10)]
            opt._resize_archive()
            self.assertEqual(len(opt.archive), expected)

    def test_terminal_cr_is_absorbing_after_memory_slot_wraparound(self):
        opt = de.LSHADE(de.sphere_function, [(-5, 5)], 4, max_evaluations=8, memory_size=2)
        for cr in (0, 0.8, 0.8):
            opt._update_memory([0.5], [cr], [1])
        self.assertIsNone(opt.memory_cr[0])
        self.assertAlmostEqual(opt.memory_cr[1], 0.8)
        self.assertEqual(opt.memory_index, 1)
        self.assertEqual(opt._sample_crossover_rate(0), 0)

    def test_reduction_preserves_stable_fitness_order_and_earlier_ties(self):
        for fitness, expected in (([5, 1, 4, 2, 3], [1, 3, 4, 2]), ([1] * 5, [0, 1, 2, 3])):
            opt = de.DifferentialEvolution(de.sphere_function, [(0, 5)], 5, de.Rand1(0.5),
                                           max_generations=1,
                                           population_schedule=de.LinearPopulationReduction(4))
            opt.population = [[float(i)] for i in range(5)]
            opt.fitness = fitness
            opt.nit = 1
            self.assertEqual(opt._apply_population_schedule(), expected)
            self.assertEqual(opt.population, [[float(i)] for i in expected])


class ExtremeCoordinateRegressions(unittest.TestCase):
    def test_donor_intermediate_overflow_and_cancellation(self):
        self.assertEqual(sum_differences(1e308, (0.5, -1e308, 1e308)), 0)
        self.assertEqual(sum_differences(1e308, (0, -1e308, 1e308)), 1e308)
        self.assertEqual(sum_differences(1e308, (1, 1e308, -1e308), (1, -1e308, 1e308)), 1e308)
        self.assertEqual(sum_differences(1e308, (1, 1e308, -1e308)), math.inf)
        self.assertEqual(sum_differences(-1e308, (1, -1e308, 1e308)), -math.inf)

    def test_trigonometric_finite_arithmetic_order_is_preserved(self):
        context = de.MutationContext([[0], [1e16], [1], [-1e16]], [0] * 4,
                                     0, 0, [0], [(-1e16, 1e16)], random.Random(1))
        donor = de.TrigonometricMutation(probability=1, scale=0.5)._trigonometric_donor(context, 1, 2, 3)
        self.assertEqual(donor, [0])  # (1e16 + 1 - 1e16) / 3, in the original order.

    def test_signed_overflow_can_be_repaired_but_nan_cannot(self):
        for handler in (de.ClipBoundaryHandler(), de.MidpointBoundaryHandler(),
                        de.RandomResetBoundaryHandler()):
            objective = Mock(return_value=0)
            opt = de.DifferentialEvolution(objective, [(-1, 1)], 4, de.Rand1(0.5),
                                           max_generations=1, boundary_handler=handler, seed=1)
            for value in (math.inf, -math.inf):
                repaired = opt._repair_trial([value], [0])
                self.assertTrue(-1 <= repaired[0] <= 1)
            with self.assertRaisesRegex(ValueError, 'NaN'):
                opt._repair_trial([math.nan], [0])
            objective.assert_not_called()
        for handler in (de.NoBoundaryHandler(), de.ReflectionBoundaryHandler()):
            opt.boundary_handler = handler
            for value in (math.inf, -math.inf):
                with self.assertRaisesRegex(ValueError, 'finite'):
                    opt._repair_trial([value], [0])
        with self.assertRaisesRegex(ValueError, 'finite'):
            de.ReflectionBoundaryHandler()([math.inf], [(-1, 1)], random.Random(1))

    def test_all_builtin_mutations_complete_extreme_bound_runs(self):
        mutations = [de.Rand1(0.5), de.Rand2(0.5), de.Best1(0.5), de.Best2(0.5),
                     de.CurrentToBest1(0.5), de.CurrentToBest2(0.5),
                     de.CurrentToRand1(0.5), de.CurrentToRand2(0.5),
                     de.TrigonometricMutation(probability=0, scale=0.5),
                     de.TrigonometricMutation(probability=1, scale=0.5),
                     de.DirectedMutation(), de.NeighborhoodSearchMutation(gaussian_probability=0),
                     de.NeighborhoodSearchMutation(gaussian_probability=1)]
        for mutation in mutations:
            for seed in range(20):
                with self.subTest(mutation=mutation, seed=seed):
                    handler = (de.ClipBoundaryHandler(), de.MidpointBoundaryHandler(),
                               de.RandomResetBoundaryHandler())[seed % 3]
                    result = de.DifferentialEvolution(
                        lambda x: sum((value / 1e308) ** 2 for value in x),
                        [(-1e308, 1e308)] * 2, 8, mutation, max_generations=20,
                        boundary_handler=handler, seed=seed,
                    ).run()
                    self.assertEqual(result.nit, 20)
                    self.assertEqual(result.nfev, 168)
                    self.assertTrue(all(math.isfinite(value) and abs(value) <= 1e308
                                        for vector in result.population for value in vector))

    def test_shade_family_completes_extreme_bound_runs(self):
        for cls in (de.SHADE, de.LSHADE):
            for seed in range(20):
                with self.subTest(cls=cls, seed=seed):
                    result = cls(lambda x: sum((value / 1e308) ** 2 for value in x),
                                 [(-1e308, 1e308)] * 2, 8, max_generations=20,
                                 max_evaluations=168, seed=seed).run()
                    self.assertEqual(result.nit, 20)
                    self.assertTrue(all(math.isfinite(value) and abs(value) <= 1e308
                                        for vector in result.population for value in vector))


class DiversityRegressions(unittest.TestCase):
    def test_large_diversity_initialization_and_full_runs(self):
        measures = [de.PopulationDiameter(), de.PopulationRadius(), de.AveragePairwiseDistance(),
                    de.AverageDistanceAroundAllIndividuals(), de.AverageDistanceAroundPopulationCenter(),
                    de.PopulationCoherence(), de.DimensionalVariance(), de.AggregatedDistribution()]
        for scale in (1e155, 1e308):
            for generations in (0, 20):
                for seed in range(4):
                    with self.subTest(scale=scale, generations=generations, seed=seed):
                        bounds = [(-scale, scale)]
                        result = de.DifferentialEvolution(
                            lambda x: 0, bounds, 4, de.Rand1(0.5), max_generations=generations,
                            diversity_measures=measures, seed=seed,
                        ).run()
                        for measure in measures:
                            history = result.diversity_history[measure.name]
                            self.assertEqual(len(history), generations + 1)
                            self.assertTrue(all(not math.isnan(value) and value >= 0 for value in history))
                            self.assertEqual(history[-1], measure(result.population, bounds))
                        if scale == 1e155:
                            self.assertTrue(math.isfinite(result.diversity_history['population_diameter'][0]))

    def test_representable_distances_centers_means_and_variances(self):
        self.assertEqual(de.PopulationDiameter()([[0], [1e155]], [(0, 1e155)]), 1e155)
        self.assertEqual(_population_center([[1e308], [1e308]]), [1e308])
        self.assertEqual(de.PopulationRadius()([[1e308], [1e308]], [(0, 1e308)]), 0)
        population = [[-1e308], [0], [1e308]]
        bounds = [(-1e308, 1e308)]
        self.assertEqual(de.PopulationDiameter()(population, bounds), math.inf)
        self.assertAlmostEqual(de.AveragePairwiseDistance()(population, bounds) / 1e308, 4 / 3)
        self.assertAlmostEqual(de.AverageDistanceAroundAllIndividuals()(population, bounds) / 1e308, 8 / 9)
        self.assertAlmostEqual(de.AverageDistanceAroundPopulationCenter()(population, bounds) / 1e308, 2 / 3)
        self.assertEqual(de.PopulationCoherence()([[-1e308]] * 2, bounds, [[1e308]] * 2), 1)
        variance = de.DimensionalVariance()([[1e154, 0], [-1e154, 0]], [(-1e154, 1e154)] * 2)
        self.assertAlmostEqual(variance / 1e308, 0.5)
        self.assertEqual(de.DimensionalVariance()(population, bounds), math.inf)
        self.assertEqual(de.DimensionalVariance()([[1e308]] * 2, bounds), 0)
        mixed = de.DimensionalVariance()([[1e308, 1], [1e308, -1]], bounds * 2)
        self.assertAlmostEqual(mixed, 0.5)
