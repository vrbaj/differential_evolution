"""Regression tests for reusing the generation's p-best ranking."""

import builtins
import unittest
from unittest.mock import patch

from differential_evolution import LSHADE, SHADE, sphere_function


class RepeatedSortSHADE(SHADE):
    """Reference path that sorts independently for every target."""

    def _sample_p_best_index(self, target_index, fitness, ranked_indices=None):
        return super()._sample_p_best_index(target_index, fitness)


class RepeatedSortLSHADE(LSHADE):
    """Reference path retaining L-SHADE's fixed p-best fraction."""

    def _sample_p_best_index(self, target_index, fitness, ranked_indices=None):
        return super()._sample_p_best_index(target_index, fitness)


class GenerationRankingTests(unittest.TestCase):
    def test_one_ranking_sort_per_generation_including_partial_generations(self):
        for cls in (SHADE, LSHADE):
            with self.subTest(cls=cls):
                opt = cls(sphere_function, [(-5, 5)] * 3, 20,
                          max_evaluations=57, seed=2)
                with patch('differential_evolution.shade.sorted',
                           wraps=builtins.sorted, create=True) as sort:
                    result = opt.run()
                self.assertEqual(result.nfev, 57)
                self.assertEqual(sort.call_count, result.nit)
                self.assertGreater(result.nit, 1)
                self.assertEqual([len(call.args[0]) for call in sort.call_args_list],
                                 result.population_size_history[:-1])

    def test_cached_ranking_preserves_seeded_runs_with_ties_and_reduction(self):
        for cls, reference in ((SHADE, RepeatedSortSHADE),
                               (LSHADE, RepeatedSortLSHADE)):
            for objective in (sphere_function, lambda x: 1.0):
                with self.subTest(cls=cls, objective=objective):
                    options = dict(objective=objective, bounds=[(-5, 5)] * 3,
                                   population_size=20, max_evaluations=257,
                                   record_snapshots=True, seed=2)
                    opt = cls(**options)
                    old = reference(**options)
                    result = opt.run()
                    expected = old.run()
                    self.assertEqual(result.population, expected.population)
                    self.assertEqual(result.fitness, expected.fitness)
                    self.assertEqual(result.best_history, expected.best_history)
                    self.assertEqual(result.snapshots, expected.snapshots)
                    self.assertEqual(result.nfev, expected.nfev)
                    self.assertEqual(opt.memory_f, old.memory_f)
                    self.assertEqual(opt.memory_cr, old.memory_cr)
                    self.assertEqual(opt.archive, old.archive)
                    self.assertEqual(opt._rng.getstate(), old._rng.getstate())
