"""Regression coverage for the third Matous review."""

import math
import random
import unittest

import differential_evolution as de
from differential_evolution.compat import DifferentialEvolution as LegacyDE


class RoundThreeRegressions(unittest.TestCase):
    def test_budget_only_runs_beyond_100_generations(self):
        for cls in (de.DifferentialEvolution, de.SHADE, de.LSHADE):
            with self.subTest(cls=cls):
                options = {'mutation': de.Rand1(0.5)} if cls is de.DifferentialEvolution else {}
                if cls is de.DifferentialEvolution:
                    options['population_schedule'] = de.LinearPopulationReduction(4)
                result = cls(de.sphere_function, [(-5, 5)], 10,
                             max_evaluations=2000, seed=1, **options).run()
                self.assertEqual(result.nfev, 2000)
                self.assertGreater(result.nit, 100)
                self.assertEqual(result.metadata['termination_reason'], 'max_evaluations')
                if cls is not de.SHADE:
                    self.assertEqual(len(result.population), 4)

    def test_limits_are_required_and_explicit_generation_cap_is_respected(self):
        with self.assertRaisesRegex(ValueError, 'Provide'):
            de.DifferentialEvolution(de.sphere_function, [(-5, 5)], 4, de.Rand1(0.5))
        for cap, evaluations in ((0, 4), (2, 12)):
            result = de.DifferentialEvolution(
                de.sphere_function, [(-5, 5)], 4, de.Rand1(0.5),
                max_generations=cap, max_evaluations=100, seed=1).run()
            self.assertEqual(result.nit, cap)
            self.assertEqual(result.nfev, evaluations)
            self.assertEqual(result.metadata['termination_reason'], 'max_generations')

    def test_nonfinite_nsde_and_tanh_parameters_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            for parameter in ('gaussian_probability', 'gaussian_mean',
                              'gaussian_stddev', 'cauchy_scale'):
                with (self.subTest(parameter=parameter, value=value),
                      self.assertRaisesRegex(ValueError, 'finite')):
                    de.NeighborhoodSearchMutation(**{parameter: value})
            for parameter in ('start', 'end'):
                with (self.subTest(parameter=parameter, value=value),
                      self.assertRaisesRegex(ValueError, 'finite')):
                    de.HyperbolicTangentPopulationReduction(4, **{parameter: value})
        with self.assertRaisesRegex(ValueError, 'distinct tanh'):
            de.HyperbolicTangentPopulationReduction(4, start=100, end=101)
        de.NeighborhoodSearchMutation(gaussian_mean=-1, gaussian_stddev=0)

    def test_nan_trials_rejected_and_infinities_repaired_before_evaluation(self):
        for cls in (de.DifferentialEvolution, de.SHADE):
            for value in (math.nan, math.inf, -math.inf):
                calls = []
                options = {'mutation': de.Rand1(0.5)} if cls is de.DifferentialEvolution else {}
                opt = cls(lambda x, calls=calls: calls.append(x) or 0, [(0, 1)], 4,
                          max_generations=1, **options)
                opt.initialize()
                if math.isnan(value):
                    with self.assertRaisesRegex(ValueError, 'NaN'):
                        opt._repair_trial([value], opt.population[0])
                else:
                    # Signed overflow can be repaired by the default clip/midpoint
                    # handlers; non-finite repaired coordinates still cannot run.
                    repaired = opt._repair_trial([value], opt.population[0])
                    self.assertTrue(math.isfinite(repaired[0]))
                    self.assertTrue(0 <= repaired[0] <= 1)
                self.assertEqual(len(calls), 4)

    def test_standalone_crossover_growth_preserves_pending_and_committed_values(self):
        rng = random.Random(1)
        controller = de.AdaptiveCrossoverRate(tau=1)
        first = controller.propose(2, rng)
        controller.propose(5, rng)
        self.assertEqual(controller.propose(2, rng), first)
        controller.commit(2, True)
        controller.commit(5, False)
        self.assertEqual(controller.values()[2], first)
        self.assertEqual(controller.values()[5], controller.initial)
        controller.resize([5, 2])
        self.assertEqual(controller.values(), [controller.initial, first])
        with self.assertRaises(ValueError):
            controller.propose(-1, rng)

    def test_legacy_current_to_strategies_accept_one_or_two_factors(self):
        for suffix in ('best/1', 'best/2', 'rand/1', 'rand/2'):
            for factors in ([0.5], [0.5, 0.8]):
                with self.subTest(suffix=suffix, factors=factors):
                    with self.assertWarns(DeprecationWarning):
                        opt = LegacyDE(de.sphere_function, [(-5, 5)], 2, 8,
                                       factors, 0.9, 'DE/current-to-' + suffix, 'random', seed=1)
                    self.assertEqual(opt._optimizer.mutation.difference_scale,
                                     factors[1] if len(factors) > 1 else None)
                    opt.evolve()
                    self.assertEqual(opt.generation, 2)
