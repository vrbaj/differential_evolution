"""Regressions for confirmed findings in message.txt."""

import math
import random
import unittest
from unittest.mock import Mock

import differential_evolution as de
from differential_evolution.diversity import resolve_diversity_measures


class MessageRegressions(unittest.TestCase):
    def optimizer(self, cls=de.DifferentialEvolution, **kwargs):
        options = dict(objective=de.sphere_function, bounds=[(-5, 5)],
                       population_size=10, max_generations=10, seed=2)
        if cls is de.DifferentialEvolution:
            options['mutation'] = de.Rand1(0.5)
        options.update(kwargs)
        return cls(**options)

    def test_lshade_schedule_minimum_must_match(self):
        with self.assertRaisesRegex(ValueError, 'must match'):
            self.optimizer(de.LSHADE, max_evaluations=100,
                           min_population_size=6,
                           population_schedule=de.LinearPopulationReduction(4))
        result = self.optimizer(de.LSHADE, max_evaluations=100,
                                min_population_size=6,
                                population_schedule=de.LinearPopulationReduction(6)).run()
        self.assertEqual(result.metadata['min_population_size'], 6)
        self.assertEqual(len(result.population), 6)

    def test_trigonometric_probability_endpoints(self):
        rng = Mock()
        rng.sample.return_value = [1, 2, 3]
        rng.random.return_value = 0.0
        context = de.MutationContext([[0], [1], [3], [7]], [0, 1, 9, 49],
                                     0, 0, [0], [(-10, 10)], rng)
        self.assertEqual(de.TrigonometricMutation(0, 0.5)(context),
                         de.Rand1(0.5)(context))
        mutation = de.TrigonometricMutation(1, 0.5)
        self.assertEqual(mutation(context), mutation._trigonometric_donor(context, 1, 2, 3))

    def test_tent_uses_initial_seeds(self):
        rng = Mock()
        rng.random.side_effect = [0.1, 0.3]
        population = de.TentInitializer()(3, [(0, 1)] * 2, lambda x: 0, rng)
        for actual, expected in zip(population, [[0.2, 0.6], [0.4, 0.8], [0.8, 0.4]], strict=True):
            for value, reference in zip(actual, expected, strict=True):
                self.assertAlmostEqual(value, reference)
        self.assertEqual(rng.random.call_count, 2)

    def test_final_snapshot_and_partial_generation(self):
        for cls in (de.DifferentialEvolution, de.SHADE, de.LSHADE):
            for budget, generations in ((110, [0, 3, 6, 9, 10]),
                                         (25, [0, 2]), (10, [0])):
                with self.subTest(cls=cls, budget=budget):
                    # Keep population size fixed to exercise exact budget boundaries.
                    options = {'min_population_size': 10} if cls is de.LSHADE else {}
                    result = self.optimizer(cls, max_evaluations=budget,
                                            record_snapshots=True, snapshot_interval=3,
                                            diversity_measures='diameter', **options).run()
                    self.assertEqual([s.generation for s in result.snapshots], generations)
                    self.assertEqual(result.snapshots[-1].population, result.population)
                    self.assertEqual(result.snapshots[-1].fitness, result.fitness)
                    self.assertEqual(result.snapshots[-1].evaluations, result.nfev)
                    self.assertEqual(len(result.population_size_history), result.nit + 1)
                    self.assertEqual(len(result.diversity_history['population_diameter']), result.nit + 1)

    def test_nonfinite_scales_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            for factory in (de.Rand1, de.ConstantScaleFactor,
                            lambda v: de.CurrentToBest1(0.5, v),
                            lambda v: de.DirectedMutation(v),
                            lambda v: de.TrigonometricMutation(1, v),
                            lambda v: de.RandomizedScaleFactor(0.1, v),
                            lambda v: de.AdaptiveScaleFactor(upper=v)):
                with self.subTest(value=value, factory=factory), self.assertRaises(ValueError):
                    factory(value)
        controller = Mock()
        controller.propose.return_value = math.nan
        context = de.MutationContext([[0]] * 4, [0] * 4, 0, 0, [0], [(0, 1)], random.Random(1))
        with self.assertRaises(ValueError):
            de.Rand1(controller)(context)
        self.assertEqual(de.ConstantScaleFactor(-3).value, -3)

    def test_none_diversity_in_iterables(self):
        self.assertEqual(resolve_diversity_measures(['none']), [])
        self.assertEqual(resolve_diversity_measures(iter(['none'])), [])
        self.assertEqual([m.name for m in resolve_diversity_measures(['none', 'diameter'])],
                         ['population_diameter'])

    def test_type_errors_propagate_in_penalty_mode(self):
        with self.assertRaises(TypeError):
            self.optimizer(objective=lambda x: None, objective_errors='penalize').run()

    def test_shade_metadata_and_exception_cleanup(self):
        for cls in (de.SHADE, de.LSHADE):
            opt = self.optimizer(cls, max_evaluations=100)
            self.assertNotIn('mutation_fallbacks', opt.run().metadata)
            opt = self.optimizer(cls, max_evaluations=100)
            opt.initialize()
            opt.objective = Mock(side_effect=ValueError('failed trial'))
            with self.assertRaises(ValueError):
                opt.run()
            self.assertFalse(hasattr(opt, '_generation_population'))
