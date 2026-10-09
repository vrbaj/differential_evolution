"""Regression coverage for the fourth code review."""

import ast
import doctest
from pathlib import Path
import random
import re
import textwrap
import unittest
import warnings

import differential_evolution as de
from differential_evolution.compat import DifferentialEvolution as LegacyDE


ROOT = Path(__file__).resolve().parents[1]


class DocumentationRegressions(unittest.TestCase):
    def test_literalinclude_targets_and_markers_exist(self):
        count = 0
        for path in (ROOT / 'docs').glob('*.rst'):
            for match in re.finditer(
                r'^\.\. literalinclude:: (.+)\n((?:[ \t]+[^\n]*\n)*)',
                path.read_text(encoding='utf-8'), re.MULTILINE,
            ):
                count += 1
                with self.subTest(path=path.name, target=match[1]):
                    target = path.parent / match[1].strip()
                    self.assertTrue(target.is_file(), str(target))
                    source = target.read_text(encoding='utf-8')
                    for option, marker in re.findall(
                        r':(start-after|end-before): (.+)', match[2],
                    ):
                        self.assertIn(marker, source, f'{path.name}: {option}')
        self.assertGreater(count, 0)

    def test_documented_optimizer_calls_have_stopping_limits(self):
        count = 0
        paths = [*(ROOT / 'docs').glob('*.rst'), ROOT / 'README.md', ROOT / 'MIGRATION.md']
        for path in paths:
            source = path.read_text(encoding='utf-8')
            if path.suffix == '.md':
                snippets = re.findall(r'^```python\n(.*?)^```', source, re.MULTILINE | re.DOTALL)
            else:
                snippets = [textwrap.dedent(block) for block in re.findall(
                    r'^\.\. code-block:: python\n\n((?:[ \t]+[^\n]*\n|\n)+)',
                    source, re.MULTILINE,
                )]
                snippets.extend(example.source for example in doctest.DocTestParser().get_examples(source))
            for snippet in snippets:
                for node in ast.walk(ast.parse(snippet)):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                        continue
                    if node.func.id not in ('DifferentialEvolution', 'SHADE', 'LSHADE'):
                        continue
                    keywords = {keyword.arg for keyword in node.keywords}
                    if 'objective' not in keywords:
                        continue  # Migration fragments intentionally omit required arguments.
                    count += 1
                    with self.subTest(path=path.name, snippet=snippet):
                        self.assertTrue(keywords & {'max_generations', 'max_evaluations'})
        self.assertGreater(count, 0)


class RoundFourRegressions(unittest.TestCase):
    def test_diversity_classes_are_rejected_before_objective_calls(self):
        for specification in (de.PopulationDiameter, [de.PopulationDiameter]):
            calls = []
            with self.subTest(specification=specification):
                with self.assertRaisesRegex(TypeError, 'instance.*not the class'):
                    de.DifferentialEvolution(
                        lambda x, calls=calls: calls.append(x) or 0, [(-5, 5)], 4, de.Rand1(0.5),
                        max_generations=1, diversity_measures=specification,
                    )
                self.assertEqual(calls, [])

    def test_diversity_instances_and_aliases_remain_supported(self):
        for specification in (de.PopulationDiameter(), [de.PopulationDiameter()], 'diameter'):
            with self.subTest(specification=specification):
                result = de.DifferentialEvolution(
                    de.sphere_function, [(-5, 5)], 4, de.Rand1(0.5),
                    max_generations=1, diversity_measures=specification, seed=7,
                ).run()
                self.assertEqual(len(result.diversity_history['population_diameter']), 2)

    def test_standalone_scale_growth_preserves_pending_and_committed_values(self):
        rng = random.Random(1)
        controller = de.AdaptiveScaleFactor(tau=1)

        def context(size, index):
            return de.MutationContext(
                population=[[0.0] for _ in range(size)], fitness=[0.0] * size,
                target_index=index, best_index=0, best_vector=[0.0],
                bounds=[(-5, 5)], rng=rng,
            )

        first = controller.propose(context(4, 0))
        controller.commit(0, True)
        self.assertEqual(len(controller.values()), 4)
        pending = controller.propose(context(4, 1))
        last = controller.propose(context(8, 7))
        self.assertGreaterEqual(last, controller.lower)
        self.assertLessEqual(last, controller.upper)
        self.assertEqual(controller.values(), [first] + [controller.initial] * 7)
        rng_state = rng.getstate()
        self.assertEqual(controller.propose(context(8, 1)), pending)
        self.assertEqual(controller.propose(context(8, 7)), last)
        self.assertEqual(rng.getstate(), rng_state)
        controller.commit(1, True)
        controller.commit(7, False)
        self.assertEqual(controller.values()[1], pending)
        self.assertEqual(controller.values()[7], controller.initial)
        before = controller.values()
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            controller.propose(context(8, -1))
        self.assertEqual(controller.values(), before)
        self.assertEqual(rng.getstate(), rng_state)

    def test_lshade_generation_warning_points_to_caller(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            de.LSHADE(de.sphere_function, [(-5, 5)], 4, max_generations=1)
        self.assertEqual(len(caught), 1)
        self.assertEqual(caught[0].category, UserWarning)
        self.assertIn('without max_evaluations', str(caught[0].message))
        self.assertEqual(Path(caught[0].filename).resolve(), Path(__file__).resolve())

    def test_empty_legacy_mutation_factors_raise_value_error(self):
        for strategy in (
            'DE/rand/1', 'DE/rand/2', 'DE/best/1', 'DE/best/2',
            'DE/current-to-best/1', 'DE/current-to-best/2',
            'DE/current-to-rand/1', 'DE/current-to-rand/2',
        ):
            with (
                self.subTest(strategy=strategy),
                self.assertWarns(DeprecationWarning),
                self.assertRaisesRegex(ValueError, 'requires at least one mutation factor'),
            ):
                LegacyDE(de.sphere_function, [(-5, 5)], 1, 8, [], 0.9, strategy, 'random')

    def test_dual_limit_schedule_warning_and_unchanged_progress(self):
        for cls in (de.DifferentialEvolution, de.SHADE, de.LSHADE):
            options = {'mutation': de.Rand1(0.5)} if cls is de.DifferentialEvolution else {}
            with self.subTest(cls=cls):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    optimizer = cls(
                        de.sphere_function, [(-5, 5)], 50, max_generations=200,
                        max_evaluations=10**6, population_schedule=de.LinearPopulationReduction(4),
                        seed=7, **options,
                    )
                self.assertEqual(len(caught), 1)
                self.assertEqual(caught[0].category, UserWarning)
                self.assertIn('evaluation budget', str(caught[0].message))
                self.assertIn('max_generations', str(caught[0].message))
                self.assertEqual(Path(caught[0].filename).resolve(), Path(__file__).resolve())
                result = optimizer.run()
                self.assertEqual(result.nit, 200)
                self.assertEqual(result.nfev, 50 * 201)
                self.assertEqual(len(result.population), 50)

    def test_schedule_warning_is_limited_to_generation_dominated_dual_limits(self):
        for generations, evaluations, schedule in (
            (None, 1000, de.LinearPopulationReduction(4)),
            (200, None, de.LinearPopulationReduction(4)),
            (200, 50 * 201, de.LinearPopulationReduction(4)),
            (200, 1000, de.LinearPopulationReduction(4)),
            (200, 10**6, None),
        ):
            with self.subTest(generations=generations, evaluations=evaluations, schedule=schedule):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    de.DifferentialEvolution(
                        de.sphere_function, [(-5, 5)], 50, de.Rand1(0.5),
                        max_generations=generations, max_evaluations=evaluations,
                        population_schedule=schedule,
                    )
                self.assertEqual(caught, [])
