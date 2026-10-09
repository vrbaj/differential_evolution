"""Regressions for BUG-1 through BUG-5 in the .bug_review reports."""

import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import differential_evolution as de


class RoundSixRegressions(unittest.TestCase):
    def test_reflection_keeps_feasible_values_and_bounds_reflected_values(self):
        reflect = de.ReflectionBoundaryHandler()
        rng = random.Random(0)
        intervals = [(-1.0, 0.3), (-5.0, 0.9), (-5.0, 0.2), (-0.9, -0.2), (0.3, 0.3)]
        for _ in range(2000):
            lower = round(rng.uniform(-10, 10), 1)
            intervals.append((lower, round(lower + rng.uniform(0.1, 10), 1)))
        for lower, upper in intervals:
            width = upper - lower
            inside = rng.uniform(lower, upper)
            with self.subTest(lower=lower, upper=upper):
                for value in (lower, upper, inside):
                    self.assertEqual(reflect([value], [(lower, upper)], rng), [value])
                for value in (upper + 2 * width, lower - width, rng.uniform(lower - 30, upper + 30)):
                    out = reflect([value], [(lower, upper)], rng)[0]
                    self.assertTrue(lower <= out <= upper, (value, out))

    def test_reflection_warm_start_never_evaluates_outside_the_box(self):
        bounds = [(-5.0, 0.9), (-1.0, 0.3)]

        def initialize(size, bounds, objective, rng):
            population = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(size)]
            population[0] = [hi for _, hi in bounds]
            return population

        def objective(vector):
            self.assertTrue(all(lo <= value <= hi for value, (lo, hi) in zip(vector, bounds, strict=True)))
            return math.sqrt(0.9 - vector[0]) + math.sqrt(0.3 - vector[1])

        for seed in range(4):
            with self.subTest(seed=seed):
                result = de.DifferentialEvolution(
                    objective, bounds, 10, de.Rand1(0.8), de.BinomialCrossover(0.5),
                    initializer=initialize, boundary_handler=de.ReflectionBoundaryHandler(),
                    max_generations=50, seed=seed,
                ).run()
                self.assertEqual(result.nfev, 510)
                self.assertEqual(result.fun, 0)
                for vector in result.population:
                    self.assertTrue(all(lo <= value <= hi for value, (lo, hi) in zip(vector, bounds, strict=True)))

    def test_schedule_minimum_cannot_exceed_initial_population(self):
        for cls in (de.DifferentialEvolution, de.SHADE, de.LSHADE):
            for schedule in (de.LinearPopulationReduction, de.HyperbolicTangentPopulationReduction):
                objective = Mock(return_value=0)
                options = {'mutation': de.Rand1(0.5)} if cls is de.DifferentialEvolution else {}
                if cls is de.LSHADE:
                    options['min_population_size'] = 50
                with self.subTest(cls=cls, schedule=schedule):
                    with self.assertRaisesRegex(ValueError, 'must not exceed population_size'):
                        cls(objective, [(-5, 5)], 10, max_evaluations=100,
                            population_schedule=schedule(50), **options)
                    objective.assert_not_called()
                    if cls is de.LSHADE:
                        options['min_population_size'] = 10
                    result = cls(objective, [(-5, 5)], 10, max_evaluations=30,
                                 population_schedule=schedule(10), seed=1, **options).run()
                    self.assertEqual(result.population_size_history, [10, 10, 10])
        objective = Mock(return_value=0)
        with self.assertRaisesRegex(ValueError, 'must not exceed population_size'):
            de.LSHADE(objective, [(-5, 5)], 10, min_population_size=50, max_evaluations=100)
        objective.assert_not_called()

    def test_optimizer_counts_reject_nonintegers_before_objective_calls(self):
        for cls in (de.DifferentialEvolution, de.SHADE, de.LSHADE):
            for name in ('population_size', 'max_generations', 'max_evaluations', 'snapshot_interval'):
                invalid = (10.0, 2.5, True, False, math.nan, math.inf, '10')
                if name in ('population_size', 'snapshot_interval'):
                    invalid += (None,)
                for value in invalid:
                    objective = Mock(return_value=0)
                    options = dict(objective=objective, bounds=[(-5, 5)], population_size=10,
                                   max_generations=2, max_evaluations=30)
                    if cls is de.DifferentialEvolution:
                        options['mutation'] = de.Rand1(0.5)
                    options[name] = value
                    with self.subTest(cls=cls, name=name, value=value):
                        with self.assertRaisesRegex(TypeError, name + ' must be an integer'):
                            cls(**options)
                        objective.assert_not_called()

    def test_memory_and_schedule_counts_are_validated(self):
        for value in (4.0, 4.5, True, None, '4'):
            for cls in (de.SHADE, de.LSHADE):
                with self.subTest(cls=cls, value=value), self.assertRaisesRegex(TypeError, 'memory_size'):
                    cls(de.sphere_function, [(-5, 5)], 10, memory_size=value, max_evaluations=30)
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, 'min_population_size'):
                de.LSHADE(de.sphere_function, [(-5, 5)], 10, min_population_size=value, max_evaluations=30)
            for cls in (de.LinearPopulationReduction, de.HyperbolicTangentPopulationReduction):
                with self.subTest(cls=cls, value=value), self.assertRaisesRegex(TypeError, 'min_population_size'):
                    cls(value)
        for cls in (de.DifferentialEvolution, de.SHADE, de.LSHADE):
            options = {'mutation': de.Rand1(0.5)} if cls is de.DifferentialEvolution else {}
            with self.subTest(cls=cls), self.assertRaisesRegex(TypeError, 'min_population_size'):
                cls(de.sphere_function, [(-5, 5)], 10, max_evaluations=30,
                    population_schedule=SimpleNamespace(min_population_size=4.0), **options)

    def test_optional_limits_and_zero_generations_remain_valid(self):
        for cls in (de.DifferentialEvolution, de.SHADE, de.LSHADE):
            options = {'mutation': de.Rand1(0.5)} if cls is de.DifferentialEvolution else {}
            initial = cls(de.sphere_function, [(-5, 5)], 4, max_generations=0,
                          max_evaluations=4, **options).run()
            self.assertEqual((initial.nit, initial.nfev), (0, 4))
            # Reduction runs after a generation, not after initialization alone.
            initialized_only = cls(de.sphere_function, [(-5, 5)], 6,
                                   max_evaluations=6, **options).run()
            self.assertEqual((initialized_only.nit, initialized_only.nfev), (0, 6))
            self.assertEqual(initialized_only.population_size_history, [6])
            budget_only = cls(de.sphere_function, [(-5, 5)], 4, max_evaluations=9, **options).run()
            self.assertEqual(budget_only.nfev, 9)
        generation_only = de.DifferentialEvolution(de.sphere_function, [(-5, 5)], 4,
                                                  de.Rand1(0.5), max_generations=2).run()
        self.assertEqual((generation_only.nit, generation_only.nfev), (2, 12))

    def test_mutation_scale_errors_are_reported_at_construction(self):
        factories = (de.Rand1, de.Rand2, de.Best1, de.Best2, de.CurrentToBest1,
                     de.CurrentToBest2, de.CurrentToRand1, de.CurrentToRand2,
                     lambda scale: de.TrigonometricMutation(1, scale), de.DirectedMutation)
        for factory in factories:
            with self.subTest(factory=factory):
                with self.assertRaisesRegex(TypeError, 'scale'):
                    factory(None)
                for invalid in (de.AdaptiveCrossoverRate(), SimpleNamespace(propose=0)):
                    with self.assertRaisesRegex(TypeError, r'propose\(context\)'):
                        factory(invalid)
        for cls in (de.CurrentToBest1, de.CurrentToBest2, de.CurrentToRand1, de.CurrentToRand2):
            with self.subTest(cls=cls), self.assertRaisesRegex(TypeError, r'propose\(context\)'):
                cls(0.5, difference_scale=de.AdaptiveCrossoverRate())

    def test_valid_scale_controllers_and_optional_difference_scales_still_run(self):
        for controller in (de.ConstantScaleFactor(0.5), de.RandomizedScaleFactor(0.4, 0.8),
                           de.AdaptiveScaleFactor(), SimpleNamespace(propose=lambda context: 0.5)):
            for factory in (de.Rand1, de.CurrentToBest1, de.CurrentToBest2, de.CurrentToRand1, de.CurrentToRand2):
                with self.subTest(controller=controller, factory=factory):
                    result = de.DifferentialEvolution(de.sphere_function, [(-5, 5)], 6,
                                                      factory(controller), max_generations=2, seed=1).run()
                    self.assertEqual((result.nit, result.nfev), (2, 18))

    def test_api_reference_lists_all_exported_boundary_handlers(self):
        root = Path(__file__).resolve().parents[1]
        reference = (root / 'docs/api/crossover.rst').read_text(encoding='utf-8')
        for name in de.__all__:
            if name.endswith('BoundaryHandler'):
                with self.subTest(name=name):
                    self.assertIn(name, reference)
