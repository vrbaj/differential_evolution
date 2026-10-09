"""Regressions for the direct implementation review."""

import math
import random
import unittest
from unittest.mock import Mock

import differential_evolution as de
from differential_evolution.benchmarks import gpd_negative_log_likelihood
from differential_evolution.compat import DifferentialEvolution as LegacyDE


class ProjectReviewTests(unittest.TestCase):
    def test_all_initializers_support_large_finite_bounds(self):
        for initializer in (de.RandomInitializer(), de.TentInitializer(),
                            de.SobolInitializer(), de.OppositionInitializer(),
                            de.QuasiOppositionInitializer()):
            for bounds in ([(-1e308, 1e308)], [(1e308, 1.1e308)],
                           [(-1.1e308, -1e308)]):
                with self.subTest(initializer=initializer, bounds=bounds):
                    seen = []

                    def objective(x, bounds=bounds, seen=seen):
                        self.assertTrue(math.isfinite(x[0]))
                        self.assertTrue(bounds[0][0] <= x[0] <= bounds[0][1])
                        seen.append(x)
                        return 0

                    result = de.DifferentialEvolution(
                        objective, bounds, 4, de.Rand1(.5), initializer=initializer,
                        max_generations=0, seed=1).run()
                    self.assertTrue(result.success)
                    self.assertTrue(all(math.isfinite(x[0]) for x in result.population))
                    expected = 8 if isinstance(initializer, (de.OppositionInitializer,
                                                              de.QuasiOppositionInitializer)) else 4
                    self.assertEqual(len(seen), expected)
                    self.assertGreater(len({x[0] for x in seen}), 1)

    def test_invalid_initializer_vectors_rejected_before_evaluation(self):
        for value in (math.nan, math.inf, -math.inf, 2.0):
            objective = Mock(return_value=0)
            opt = de.DifferentialEvolution(
                objective, [(0, 1)], 4, de.Rand1(.5), max_generations=0,
                initializer=lambda n, b, o, r, value=value: [[value] for _ in range(n)])
            with self.assertRaisesRegex(ValueError, 'finite and within bounds'):
                opt.run()
            objective.assert_not_called()

    def test_ranked_initializer_cannot_evaluate_nonfinite_vectors(self):
        def initializer(n, b, objective, rng):
            objective([math.inf])
            return [[0.5] for _ in range(n)]

        objective = Mock(return_value=0)
        opt = de.DifferentialEvolution(objective, [(0, 1)], 4, de.Rand1(.5),
                                       initializer=initializer, max_generations=0)
        with self.assertRaisesRegex(ValueError, 'must be finite'):
            opt.run()
        objective.assert_not_called()
        self.assertEqual(opt.nfev, 0)

    def test_large_midpoint_and_reflection_repairs_are_finite_and_correct(self):
        midpoint = de.MidpointBoundaryHandler().repair(
            [0], [1e308], [(1e308, 1.1e308)], None)
        self.assertEqual(midpoint, [1e308])
        reflection = de.ReflectionBoundaryHandler()
        self.assertEqual(reflection([-1], [(0, 1e308)], None), [1])
        self.assertEqual(reflection([1e308], [(-1e308, 1e308)], None), [1e308])
        self.assertEqual(reflection([-1.5e308], [(0, 1e308)], None), [5e307])
        # Here only the subtraction overflows, rather than the period.
        self.assertTrue(math.isfinite(reflection([-1e308], [(1e308, 1.1e308)], None)[0]))

    def test_midpoint_repair_cannot_create_a_spurious_infinite_optimum(self):
        result = de.DifferentialEvolution(
            lambda x: -abs(x[0]), [(1e308, 1.1e308)], 4,
            mutation=lambda context: [0.0],
            initializer=lambda n, b, o, r: [[1e308] for _ in range(n)],
            boundary_handler=de.MidpointBoundaryHandler(), max_generations=1).run()
        self.assertTrue(result.success)
        self.assertEqual(result.x, [1e308])
        self.assertEqual(result.fun, -1e308)
        self.assertEqual(result.nfev, 8)

    def test_repaired_vector_is_validated_before_objective_call(self):
        for repaired in ([math.nan], [math.inf], [-math.inf], [], [0, 0]):
            objective = Mock(return_value=0)
            opt = de.DifferentialEvolution(objective, [(0, 1)], 4, de.Rand1(.5),
                                           max_generations=1,
                                           boundary_handler=lambda t, b, r, repaired=repaired: repaired)
            opt.initialize()
            with self.assertRaises(ValueError):
                opt.run()
            self.assertEqual(objective.call_count, 4)
            self.assertEqual(opt.nfev, 4)

    def test_random_reset_supports_wide_bounds(self):
        repaired = de.RandomResetBoundaryHandler()([math.inf], [(-1e308, 1e308)],
                                                   random.Random(1))
        self.assertTrue(math.isfinite(repaired[0]))
        self.assertTrue(-1e308 <= repaired[0] <= 1e308)

    def test_gpd_near_zero_shape_and_large_values(self):
        for shape in (0, 1e-320, -1e-320, 1e-12, -1e-12):
            self.assertAlmostEqual(gpd_negative_log_likelihood([1])([1, shape]), 1)
            self.assertEqual(gpd_negative_log_likelihood([0])([1, shape]), 0)
        expected = 2 * math.log(1e308)
        self.assertAlmostEqual(gpd_negative_log_likelihood([1e308])([1e308, 1e308]), expected)
        self.assertAlmostEqual(gpd_negative_log_likelihood([1e308, 1e308])([1e308, 0]),
                               expected + 2)
        # Ratio and product overflow although the final logarithmic value is finite.
        value = gpd_negative_log_likelihood([1e100])([1e-210, 1e308])
        self.assertAlmostEqual(value, math.log(1e100) + math.log(1e308))
        # Product underflows despite a representable final contribution.
        self.assertEqual(gpd_negative_log_likelihood([1e-200])([1, 1e-200]), 1e-200)
        self.assertEqual(gpd_negative_log_likelihood([2])([1, -.5]), math.inf)

    def test_archive_rates_rejected_before_evaluations(self):
        for cls in (de.SHADE, de.LSHADE):
            for rate in (math.nan, math.inf, -math.inf, -1):
                objective = Mock(return_value=0)
                with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
                    cls(objective, [(0, 1)], 4, max_evaluations=8, archive_rate=rate)
                objective.assert_not_called()

    def test_compact_sampling_mapping_and_distribution(self):
        rng = Mock()
        rng.sample.return_value = [0, 1, 2, 3]
        self.assertEqual(de.sample_distinct_indices(1000, 4, rng, excluded=(0, 2, 4)),
                         [1, 3, 5, 6])
        self.assertIsInstance(rng.sample.call_args.args[0], range)
        rng = random.Random(7)
        counts = [0] * 40
        for _ in range(10000):
            sample = de.sample_distinct_indices(40, 3, rng, excluded=(0, 2, 39))
            self.assertEqual(len(set(sample)), 3)
            self.assertFalse({0, 2, 39}.intersection(sample))
            for index in sample:
                counts[index] += 1
        self.assertEqual(counts[0], 0)
        self.assertTrue(all(650 < counts[i] < 1000 for i in range(40) if i not in (0, 2, 39)))
        # Dense requests retain the list fallback.
        self.assertEqual(set(de.sample_distinct_indices(40, 38, rng, excluded=(0, 39))),
                         set(range(1, 39)))
        with self.assertRaises(ValueError):
            de.sample_distinct_indices(40, 39, rng, excluded=(0, 39))

    def test_compact_sampling_preserves_reference_rng_sequence(self):
        for size in (40, 200, 2000):
            for excluded in ((0, 2, size - 1), (10,), (0, 0, -1, size)):
                rng = random.Random(11)
                reference = random.Random(11)
                candidates = [i for i in range(size) if i not in excluded]
                for _ in range(30):
                    self.assertEqual(de.sample_distinct_indices(size, 3, rng, excluded),
                                     reference.sample(candidates, 3))
                self.assertEqual(rng.getstate(), reference.getstate())

    def test_shade_r1_compact_mapping(self):
        opt = de.SHADE(de.sphere_function, [(0, 1)], 40, max_generations=1)
        for target in (0, 20, 39):
            for rank in range(39):
                rng = Mock()
                rng.sample.return_value = [rank]
                opt._rng = rng
                self.assertEqual(opt._sample_r1_index(target, 0), rank + (rank >= target))
                self.assertIsInstance(rng.sample.call_args.args[0], range)

    def test_legacy_coherence_matches_consecutive_populations(self):
        with self.assertWarns(DeprecationWarning):
            opt = LegacyDE(de.sphere_function, [(-5, 5)], 3, 4, [.5], .9,
                           'DE/rand/1', 'random', seed=1)
        opt._optimizer.record_snapshots = True
        opt.evolve()
        previous, current = opt._optimizer.snapshots[-2:]
        expected = de.PopulationCoherence()(current.population, [(-5, 5)], previous.population)
        self.assertGreater(expected, 0)
        self.assertEqual(opt.measure_diversity('coherence'), expected)
        self.assertEqual(opt.measure_diversity('population_coherence'), expected)
