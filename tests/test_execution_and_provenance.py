"""Contracts for ordered evaluation, provenance and dimensionless diagnostics."""
import json
import math
import random
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import differential_evolution as de


class ExecutionAndProvenanceTests(unittest.TestCase):
    def options(self):
        return dict(objective=de.sphere_function, bounds=[(-5., 5.)] * 3,
                    population_size=12, max_evaluations=101, seed=7,
                    record_snapshots=True)

    def test_ordered_batches_preserve_builtin_trajectories_and_partial_budget(self):
        jde = de.jde_rand_1_bin()
        for cls, components in [
            (de.DifferentialEvolution, dict(mutation=de.Rand1(0.8))),
            (de.DifferentialEvolution, dict(mutation=jde.mutation, crossover=jde.crossover)),
            (de.SHADE, {}), (de.LSHADE, {}),
        ]:
            with self.subTest(cls=cls, components=components):
                sizes = []
                def evaluate(vectors, sizes=sizes):
                    sizes.append(len(vectors))
                    return [de.sphere_function(vector) for vector in vectors]
                scalar = cls(**self.options(), **components).run()
                batch = cls(**self.options(), **components, batch_objective=evaluate).run()
                self.assertEqual(scalar.population, batch.population)
                self.assertEqual(scalar.snapshots, batch.snapshots)
                self.assertEqual(scalar.best_history, batch.best_history)
                self.assertEqual(batch.nfev, 101)
                self.assertEqual(sum(sizes), 101)
                self.assertEqual(sizes[0], 12)
                self.assertEqual(sizes[-1], 101 - sum(sizes[:-1]))

    def test_external_ordered_executor(self):
        with ThreadPoolExecutor(2) as pool:
            result = de.LSHADE(**self.options(), batch_objective=lambda xs: pool.map(de.sphere_function, xs)).run()
        self.assertEqual(result.population, de.LSHADE(**self.options()).run().population)

    def test_batch_failure_charges_submitted_candidates_and_cleans_archive_context(self):
        optimizer = de.SHADE(**self.options(), objective_errors="penalize")
        optimizer.initialize()
        def fail(vectors):
            raise ValueError("worker failed")
        optimizer.batch_objective = fail
        with self.assertRaisesRegex(ValueError, "worker failed"):
            optimizer.run()
        self.assertEqual(optimizer.nfev, 24)
        self.assertFalse(hasattr(optimizer, "_generation_population"))

    def test_invalid_result_length_and_nan_policy(self):
        optimizer = de.DifferentialEvolution(**self.options(), mutation=de.Rand1(0.5),
                                             batch_objective=lambda xs: [0.0])
        with self.assertRaisesRegex(ValueError, "one value per candidate"):
            optimizer.run()
        self.assertEqual(optimizer.nfev, 12)
        optimizer = de.DifferentialEvolution(**self.options(), mutation=de.Rand1(0.5),
                                             batch_objective=lambda xs: [math.nan] * len(xs))
        result = optimizer.run()
        self.assertFalse(result.success)
        self.assertTrue(all(value == math.inf for value in result.fitness))

    def test_negative_infinity_and_input_isolation(self):
        def evaluate(vectors):
            for vector in vectors:
                vector[:] = [999.] * len(vector)
            return [-math.inf] * len(vectors)
        result = de.SHADE(**self.options(), batch_objective=evaluate).run()
        self.assertEqual(result.fun, -math.inf)
        self.assertTrue(all(-5 <= x <= 5 for vector in result.population for x in vector))

    def test_ranked_initializers_keep_scalar_budget_accounting(self):
        for initializer in [de.OppositionInitializer(), de.QuasiOppositionInitializer()]:
            options = dict(self.options(), initializer=initializer)
            scalar = de.LSHADE(**options).run()
            batch = de.LSHADE(**options, batch_objective=lambda xs: map(de.sphere_function, xs)).run()
            self.assertEqual(scalar.population, batch.population)
            self.assertEqual(batch.nfev, 101)

    def test_provenance_captures_initial_configuration_and_is_json_safe(self):
        components = de.jde_rand_1_bin(initial_f=0.7)
        optimizer = de.DifferentialEvolution(**self.options(), mutation=components.mutation,
                                             crossover=components.crossover)
        result = optimizer.run()
        self.assertEqual(result.metadata["seed"], 7)
        self.assertEqual(result.metadata["package_version"], de.__version__)
        self.assertEqual(result.metadata["configuration"]["mutation"]["parameters"]["scale"]["parameters"]["initial"], 0.7)
        self.assertIn("python_version", result.metadata)
        self.assertNotIn("0x", json.dumps(result.metadata))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run.json"
            result.save_json(target)
            self.assertEqual(json.loads(target.read_text())["metadata"]["seed"], 7)
        result.metadata["configuration"]["bounds"][0][0] = -999
        self.assertEqual(optimizer._configuration["bounds"][0][0], -5)

    def test_random_state_is_before_component_initialization(self):
        options = self.options()
        options.pop("seed")
        rng = random.Random(22)
        initial = rng.getstate()
        result = de.SHADE(**options, rng=rng).run()
        self.assertEqual(result.metadata["initial_random_state"], initial)
        self.assertIsNone(result.metadata["seed"])

    def test_ackley_dimension_and_independent_formula(self):
        for dimension in [1, 2, 10, 30]:
            self.assertAlmostEqual(de.ackley_function([0.] * dimension), 0., places=12)
            x = [0.3 + i / 10 for i in range(dimension)]
            expected = (-20 * math.exp(-.2 * math.sqrt(sum(v*v for v in x) / dimension))
                        -math.exp(sum(math.cos(2*math.pi*v) for v in x) / dimension) + math.e + 20)
            self.assertAlmostEqual(de.ackley_function(x), expected, places=12)
        with self.assertRaises(ValueError):
            de.ackley_function([])

    def test_default_crossover_is_binomial(self):
        optimizer = de.DifferentialEvolution(**self.options(), mutation=de.Rand1(.8))
        self.assertIsInstance(optimizer.crossover, de.BinomialCrossover)
        self.assertEqual(optimizer.crossover.crossover_rate, .9)

    def test_normalized_measure_is_invariant_to_coordinate_units(self):
        measure = de.BoundNormalizedDiversity(de.AveragePairwiseDistance())
        value = measure([[0., 0.], [1., 1.]], [(0., 1.), (0., 1.)])
        transformed = measure([[10., -100.], [20., 100.]], [(10., 20.), (-100., 100.)])
        self.assertAlmostEqual(value, transformed)
        self.assertAlmostEqual(value, math.sqrt(2))
        self.assertEqual(measure([[2., 1.], [2., 1.]], [(2., 2.), (0., 2.)]), 0)
        self.assertAlmostEqual(measure([[-1e308], [1e308]], [(-1e308, 1e308)]), 1)

    def test_normalized_movement_and_invalid_dimensions(self):
        measure = de.BoundNormalizedDiversity(de.PopulationCoherence())
        self.assertAlmostEqual(measure([[2.], [4.]], [(0., 10.)], [[1.], [3.]]), 1.)
        with self.assertRaises(ValueError):
            measure([[1., 2.]], [(0., 1.)])
        with self.assertRaises(ValueError):
            measure([[1.]], [(2., 1.)])
        result = de.SHADE(**self.options(), diversity_measures=[measure]).run()
        self.assertIn("normalized_population_coherence", result.diversity_history)
