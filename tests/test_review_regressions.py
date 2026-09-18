"""Behavioral regressions for review/REVIEW.md; no source-text assertions."""

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import differential_evolution as de
from differential_evolution.benchmarks import gpd_negative_log_likelihood
from differential_evolution.diversity import resolve_diversity_measures
from differential_evolution.population_schedules import PopulationScheduleState


class ReviewRegressions(unittest.TestCase):
    def optimizer(self, **kwargs):
        options = dict(
            objective=de.sphere_function,
            bounds=[(-5.0, 5.0)] * 3,
            population_size=8,
            mutation=de.Rand1(0.5),
            seed=3,
            max_generations=4,
        )
        options.update(kwargs)
        return de.DifferentialEvolution(**options)

    def test_tent_population_is_diverse_reproducible_and_bounded(self):
        for seed in range(5):
            args = (50, [(-5.0, 5.0)] * 30, lambda x: 0)
            pop = de.TentInitializer()(*args, random.Random(seed))
            self.assertEqual(pop, de.TentInitializer()(*args, random.Random(seed)))
            self.assertEqual(len(set(map(tuple, pop))), 50)
            self.assertTrue(all(-5 < value < 5 for point in pop for value in point))

    def test_sobol_seed_and_dyadic_balance(self):
        args = (32, [(0.0, 1.0)] * 40, lambda x: 0)
        first = de.SobolInitializer()(*args, random.Random(1))
        self.assertEqual(first, de.SobolInitializer()(*args, random.Random(1)))
        self.assertNotEqual(first, de.SobolInitializer()(*args, random.Random(2)))
        for dim in range(40):
            self.assertEqual({int(point[dim] * 32) for point in first}, set(range(32)))

    def test_fixed_dimension_benchmarks_reject_invalid_lengths(self):
        for name in de.__all__:
            if name.endswith("_function") and name not in (
                "sphere_function",
                "rastrigin_function",
            ):
                for dimension in (0, 1, 3, 10):
                    with (
                        self.subTest(name=name, dimension=dimension),
                        self.assertRaises(ValueError),
                    ):
                        getattr(de, name)([0.0] * dimension)

    def test_known_benchmark_optima(self):
        cases = {
            "beale": ([3, 0.5], 0),
            "booth": ([1, 3], 0),
            "matyas": ([0, 0], 0),
            "himmelblau": ([3, 2], 0),
            "bukin": ([-10, 1], 0),
            "mccormick": ([-0.54719, -1.54719], -1.9133),
            "three_hump_camel": ([0, 0], 0),
            "ackley": ([0, 0], 0),
            "goldstein_price": ([0, -1], 3),
            "levi": ([1, 1], 0),
            "easom": ([math.pi, math.pi], -1),
            "eggholder": ([512, 404.2319], -959.6407),
            "schaffer_n2": ([0, 0], 0),
            "sphere": ([0, 0], 0),
            "rastrigin": ([0, 0], 0),
        }
        for name, (point, expected) in cases.items():
            self.assertAlmostEqual(
                getattr(de, name + "_function")(point), expected, delta=0.001
            )

    def test_diversity_names_and_iterables(self):
        with self.assertRaises(ValueError):
            resolve_diversity_measures(["radius", "population_radius"])

        class IterableNames:
            def __eq__(self, other):
                raise AssertionError("Do not compare iterables to strings")

            def __iter__(self):
                return iter(["radius", "diameter"])

        self.assertEqual(len(resolve_diversity_measures(IterableNames())), 2)

        class Fake:
            name = "fake"

            def __init__(self):
                self.__call__ = "not callable"

        with self.assertRaises(TypeError):
            resolve_diversity_measures([Fake()])

    def test_memory_remains_finite_with_infeasible_parents(self):
        for cls in (de.SHADE, de.LSHADE):
            opt = cls(
                objective=lambda x: math.inf if x[0] < 0 else sum(v * v for v in x),
                bounds=[(-5.0, 5.0)] * 5,
                population_size=20,
                max_generations=30,
                max_evaluations=620,
                seed=3,
            )
            opt.run()
            self.assertTrue(all(math.isfinite(v) for v in opt.memory_f))
            self.assertTrue(all(v is None or math.isfinite(v) for v in opt.memory_cr))
            opt._update_memory([0.5, 0.7], [0.2, 0.8], [1e308, 1e308])
            self.assertTrue(all(math.isfinite(v) for v in opt.memory_f))
            opt.memory_f[0] = math.nan
            opt.memory_cr[0] = math.nan
            self.assertTrue(math.isfinite(opt._sample_scale_factor(0)))
            self.assertTrue(math.isfinite(opt._sample_crossover_rate(0)))

    def test_shade_and_lshade_memory_formulas(self):
        for cls, expected in ((de.SHADE, 0.5), (de.LSHADE, 0.68)):
            opt = cls(
                objective=de.sphere_function,
                bounds=[(-1.0, 1.0)],
                population_size=8,
                max_evaluations=80,
            )
            opt._update_memory([0.2, 0.8], [0.2, 0.8], [1.0, 1.0])
            self.assertAlmostEqual(opt.memory_cr[0], expected)
            self.assertAlmostEqual(opt.memory_f[0], 0.68)
            opt._update_memory([0.5], [0.0], [1.0])
            self.assertEqual(opt.memory_cr[1], 0.0 if cls is de.SHADE else None)

    def test_shade_sampling_plateaus_and_archive_limit(self):
        opt = de.SHADE(
            objective=lambda x: 1.0,
            bounds=[(-1.0, 1.0)] * 2,
            population_size=8,
            max_generations=1,
            seed=2,
        )
        opt.initialize()
        self.assertIn(
            0, {opt._sample_p_best_index(0, list(range(8))) for _ in range(100)}
        )
        self.assertIn(1, {opt._sample_r1_index(0, 1) for _ in range(100)})
        before = [v[:] for v in opt.population]
        opt._step()
        self.assertNotEqual(before, opt.population)
        self.assertEqual(opt.memory_f, [0.5] * 6)
        self.assertLessEqual(len(opt.archive), opt.population_size)

    def test_negative_infinity_nan_and_termination(self):
        result = self.optimizer(objective=lambda x: -math.inf).run()
        self.assertEqual(result.fun, -math.inf)
        result = self.optimizer(objective=lambda x: math.nan).run()
        self.assertFalse(result.success)
        opt = self.optimizer(max_evaluations=10)
        result = opt.run()
        self.assertEqual(result.nfev, 10)
        self.assertIn("max_evaluations", result.message)
        with self.assertRaises(RuntimeError):
            opt.run()

    def test_initialization_budget_and_no_survivor_reevaluation(self):
        with self.assertRaises(ValueError):
            self.optimizer(max_evaluations=7)
        for initializer in (
            de.OppositionInitializer(),
            de.QuasiOppositionInitializer(),
        ):
            calls = []
            opt = self.optimizer(
                initializer=initializer,
                max_generations=0,
                max_evaluations=16,
                objective=lambda x: calls.append(x) or sum(v * v for v in x),
            )
            self.assertEqual(opt.run().nfev, 16)
            self.assertEqual(len(calls), 16)
            opt = self.optimizer(initializer=initializer, max_evaluations=9)
            with self.assertRaises(ValueError):
                opt.run()
            self.assertEqual(opt.nfev, 9)

    def test_exception_policy(self):
        def invalid(x):
            raise ValueError("simulation failed")

        with self.assertRaises(ValueError):
            self.optimizer(objective=invalid).run()
        result = self.optimizer(objective=invalid, objective_errors="penalize").run()
        self.assertFalse(result.success)
        self.assertEqual(result.nfev, 40)

    def test_component_ownership(self):
        components = de.jde_rand_1_bin(tau_f=1.0, tau_cr=1.0)
        opt = self.optimizer(
            mutation=components.mutation, crossover=components.crossover
        )
        opt.run()
        before = opt.mutation.scale.values()
        second = self.optimizer(
            mutation=components.mutation, crossover=components.crossover
        )
        second.run()
        self.assertEqual(before, opt.mutation.scale.values())
        self.assertIsNot(opt.mutation.scale, second.mutation.scale)

    def test_current_to_mutations_draw_default_scale_once(self):
        class Counter:
            calls = 0

            def propose(self, context):
                self.calls += 1
                return 0.1 * self.calls

        for cls in (
            de.CurrentToBest1,
            de.CurrentToBest2,
            de.CurrentToRand1,
            de.CurrentToRand2,
        ):
            scale = Counter()
            context = de.MutationContext(
                [[float(i)] for i in range(8)],
                list(range(8)),
                0,
                1,
                [1.0],
                [(-10.0, 10.0)],
                random.Random(3),
            )
            cls(scale)(context)
            self.assertEqual(scale.calls, 1)

    def test_directed_mutation_is_affine_fitness_invariant(self):
        donors = []
        for factor, offset in ((1.0, 0.0), (0.001, 0.0), (1.0, -100.0)):
            context = de.MutationContext(
                [[float(i)] for i in range(8)],
                [factor * i + offset for i in range(8)],
                0,
                1,
                [1.0],
                [(-10.0, 10.0)],
                random.Random(3),
            )
            donors.append(de.DirectedMutation()(context))
        for donor in donors[1:]:
            self.assertAlmostEqual(donor[0], donors[0][0])

    def test_boundaries_and_population_validation(self):
        result = self.optimizer(objective=lambda x: sum((v - 20) ** 2 for v in x)).run()
        self.assertTrue(all(-5 <= v <= 5 for point in result.population for v in point))
        with self.assertRaises(ValueError):
            self.optimizer(population_size=4, mutation=de.Rand2(0.5))
        with self.assertRaises(ValueError):
            de.ClipBoundaryHandler()([1, 2], [(0, 1)], random.Random())
        self.assertEqual(
            de.MidpointBoundaryHandler().repair([-5, 8], [2, 3], [(0, 4)] * 2, None),
            [1, 3.5],
        )
        self.assertEqual(
            de.ReflectionBoundaryHandler()([-9, 10], [(0, 4)] * 2, None), [1, 2]
        )

    def test_standalone_crossover_and_wrong_controller(self):
        with self.assertRaises(TypeError):
            de.BinomialCrossover(de.AdaptiveScaleFactor())
        output = de.BinomialCrossover(de.AdaptiveCrossoverRate())(
            [1, 2], [3, 4], random.Random(0)
        )
        self.assertEqual(len(output), 2)

    def test_json_nonfinite_and_snapshot_thinning(self):
        result = self.optimizer(
            objective=lambda x: math.inf, record_snapshots=True, snapshot_interval=2
        ).run()
        self.assertEqual([s.generation for s in result.snapshots], [0, 2, 4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result.save_json(path)
            data = json.loads(
                path.read_text(),
                parse_constant=lambda x: self.fail("Invalid JSON constant"),
            )
        self.assertEqual(data["fun"], "inf")
        self.assertEqual(len(data["snapshots"]), 3)

    def test_tanh_schedule_and_lshade_warning(self):
        state = PopulationScheduleState(50, 100, 0, None, 100, 100, 10)
        self.assertGreater(
            de.HyperbolicTangentPopulationReduction(10).target_size(state), 55
        )
        with self.assertWarns(UserWarning):
            de.LSHADE(objective=de.sphere_function, bounds=[(-1, 1)], population_size=4)

    def test_population_configuration_is_preserved(self):
        opt = de.LSHADE(
            objective=de.sphere_function,
            bounds=[(-5, 5)],
            population_size=10,
            max_evaluations=60,
            max_generations=100,
            seed=1,
        )
        opt.run()
        self.assertEqual(opt.population_size, 10)
        self.assertEqual(opt.current_population_size, 4)

    def test_initializer_population_size_is_validated(self):
        with self.assertRaises(ValueError):
            self.optimizer(initializer=lambda *args: [[0.0, 0.0, 0.0]]).run()

    def test_shared_diversity_statistics_match_standalone(self):
        measures = [
            de.PopulationDiameter(),
            de.AveragePairwiseDistance(),
            de.AverageDistanceAroundAllIndividuals(),
        ]
        opt = self.optimizer(diversity_measures=measures)
        result = opt.run()
        for measure in measures:
            self.assertAlmostEqual(
                result.diversity_history[measure.name][-1],
                measure(result.population, opt.bounds),
            )

    def test_directed_fallback_is_reported(self):
        opt = self.optimizer(objective=lambda x: 1.0, mutation=de.DirectedMutation())
        result = opt.run()
        self.assertEqual(result.metadata["mutation_fallbacks"], 32)

    def test_gpd_objective(self):
        objective = gpd_negative_log_likelihood([0, 1, 2])
        self.assertEqual(objective([1, 0]), 3.0)
        self.assertEqual(objective([0, 1]), math.inf)
        self.assertEqual(objective([1, -1]), math.inf)
        with self.assertRaises(ValueError):
            gpd_negative_log_likelihood([-1])
