"""Repository, packaging, performance and API-surface audit — findings that are not
runtime bugs and so cannot be expressed as a behavioural assertion.

Covers K1-K8, T1-T4, D1, D3, D4, D8, P1, P2, P3, P5, and the lint summary.
Run from the root of the repository checkout:

    python audit.py [--offline]

Optional tools (ruff, mypy, coverage, build) are used when present and skipped otherwise.
"""
from __future__ import annotations

import inspect
import json
import math
import shutil
import subprocess
import sys
import time
import timeit
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "differential_evolution").is_dir():
    ROOT = Path.cwd()
OFFLINE = "--offline" in sys.argv

FAILURES: list[str] = []


def check(tag: str, ok: bool, observed: str, expected: str) -> None:
    print(f"  observed: {observed}")
    print(f"  expected: {expected}")
    print(f"  -> {'PASS' if ok else 'FAIL'}  [{tag}]\n")
    if not ok:
        FAILURES.append(tag)


def report(tag: str, observed: str) -> None:
    """A measurement with no pass/fail threshold."""
    print(f"  {observed}\n  -> MEASURED  [{tag}]\n")


def header(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def run(*command: str) -> tuple[int, str]:
    proc = subprocess.run(command, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def have(tool: str) -> bool:
    return shutil.which(tool) is not None or _module_available(tool)


def _module_available(name: str) -> bool:
    code, _ = run(sys.executable, "-c", f"import {name}")
    return code == 0


# ============================================================ packaging / repository
header("K1  Is the PyPI distribution name available?")
if OFFLINE:
    print("  skipped (--offline)\n")
else:
    results = {}
    for name in ("differential-evolution", "differential_evolution"):
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=15) as fh:
                info = json.load(fh)["info"]
            results[name] = f"TAKEN v{info['version']} — {(info.get('summary') or '')[:60]}"
        except Exception as exc:  # noqa: BLE001 - a 404 means available
            results[name] = f"available ({getattr(exc, 'code', type(exc).__name__)})"
    taken = any("TAKEN" in v for v in results.values())
    check("K1", not taken, "; ".join(f"{k}: {v}" for k, v in results.items()),
          "the declared distribution name to be free on PyPI")

header("K2  Is there a LICENSE file, and is it declared the modern way?")
license_files = [p for p in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING")
                 if (ROOT / p).exists()]
pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
declared = pyproject["project"].get("license")
legacy_table = isinstance(declared, dict)
check("K2", bool(license_files) and not legacy_table,
      f"license files on disk: {license_files or 'NONE'}; pyproject declares {declared!r}"
      + (" (PEP 639 deprecates the table form)" if legacy_table else ""),
      "a LICENSE file present, plus license = \"MIT\" and license-files = [\"LICENSE\"]")

header("K3  Do the root compatibility shims survive `pip install`?")
shims = ["main.py", "population_initialization.py", "testing_functions.py"]
present_on_disk = [s for s in shims if (ROOT / s).exists()]
find_cfg = pyproject.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
py_modules = pyproject.get("tool", {}).get("setuptools", {}).get("py-modules", [])
packaged = bool(py_modules)
readme = (ROOT / "README.md").read_text(encoding="utf-8")
claims_shims = "compatibility shims" in readme
check("K3", not (claims_shims and present_on_disk and not packaged),
      f"README claims shims: {claims_shims}; on disk: {present_on_disk}; "
      f"packages.find include={find_cfg.get('include')}; py-modules={py_modules or 'none'} "
      "-> the shims are NOT in the wheel",
      "either py-modules declares them, or README/MIGRATION stop promising `from main import ...`")

header("K4  Absolute local paths leaking into the published README")
offenders = {}
for path in ROOT.rglob("*"):
    if path.suffix in {".md", ".rst", ".py"} and ".git" not in path.parts:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = text.count("/home/") + text.count("C:\\Users\\")
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits
check("K4", not offenders,
      f"absolute local paths found in {offenders or 'no files'}",
      "repo-relative links only (README becomes the PyPI long description)")

header("K5  Standard project files and CI")
wanted = [".github/workflows", "CITATION.cff", "CHANGELOG.md", "CONTRIBUTING.md",
          "differential_evolution/py.typed", ".pre-commit-config.yaml"]
missing = [w for w in wanted if not (ROOT / w).exists()]
check("K5", not missing, f"missing: {missing}",
      "at minimum a CI workflow, CITATION.cff and py.typed before a paper submission")

header("K6  pyproject.toml completeness")
project = pyproject["project"]
absent = [key for key in ("classifiers", "keywords", "urls") if key not in project]
authors = project.get("authors", [])
no_email = all("email" not in a for a in authors)
extras = list(project.get("optional-dependencies", {}))
tool_sections = [k for k in ("pytest", "ruff", "mypy") if k in pyproject.get("tool", {})]
check("K6", not absent and not no_email and "test" in extras,
      f"missing keys: {absent}; author email declared: {not no_email}; "
      f"optional-dependencies: {extras}; tool sections: {tool_sections or 'none'}; "
      f"requires-python: {project.get('requires-python')!r}",
      "classifiers, keywords, urls, author email, a `test` extra, and tool config")

header("K7  gpd_ll_function: per-evaluation unpickling of a missing file")
from differential_evolution import benchmarks  # noqa: E402

source = inspect.getsource(benchmarks.gpd_ll_function)
default_path = inspect.signature(benchmarks.gpd_ll_function).parameters["sample_path"].default
import differential_evolution as de  # noqa: E402

exported = "gpd_ll_function" in de.__all__
check("K7", "pickle.load" not in source,
      f"pickle.load inside the objective body: {'pickle.load' in source}; "
      f"default sample file {default_path!r} exists: {(ROOT / str(default_path)).exists()}; "
      f"exported from the package: {exported}",
      "no per-evaluation file I/O, no pickle, and a data file that actually ships")

header("K8  Repository junk")
gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
junk_lines = [ln for ln in gitignore if ln.strip().startswith("/=") or ln.strip() == "=8,"]
requirements = (ROOT / "requirements.txt")
req_real = [ln for ln in requirements.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")] if requirements.exists() else []
generated = list((ROOT / "docs" / "api" / "generated").glob("*.rst"))
templates = (ROOT / "docs" / "_templates").exists()
conf = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
declares_templates = "templates_path" in conf
check("K8", not junk_lines and not (requirements.exists() and not req_real)
      and not generated and not (declares_templates and not templates),
      f".gitignore junk lines: {junk_lines}; requirements.txt has "
      f"{len(req_real)} real entries but exists: {requirements.exists()}; "
      f"checked-in autosummary files: {len(generated)}; "
      f"conf.py declares templates_path but docs/_templates exists: {templates}",
      "no shell-accident lines, no empty requirements.txt, generated rst gitignored")

# ============================================================ API surface
header("D1  Are the declared Protocols actually used in annotations?")
from differential_evolution.optimizer import DifferentialEvolution  # noqa: E402

annotations = DifferentialEvolution.__annotations__
object_typed = [k for k, v in annotations.items() if v in ("object", "object | None")]
check("D1", not object_typed,
      f"fields annotated as bare `object`: {object_typed}",
      "annotations referencing the Protocols declared in scales.py / crossovers.py / "
      "boundaries.py / initializers.py / diversity.py")

header("D3  Can the BoundaryHandler protocol express midpoint / bounce-back repair?")
from differential_evolution.boundaries import BoundaryHandler  # noqa: E402

params = list(inspect.signature(BoundaryHandler.__call__).parameters)
handlers = [n for n in de.__all__ if n.endswith("BoundaryHandler")]
check("D3", "target_vector" in params or "parent" in params,
      f"protocol signature is {params}; shipped handlers: {handlers}",
      "the target/parent vector to be passed, so midpoint repair "
      "(used by the reference L-SHADE) is implementable")

header("D4  Are the optional lifecycle hooks discoverable?")
optimizer_source = inspect.getsource(sys.modules["differential_evolution.optimizer"])
shade_source = inspect.getsource(sys.modules["differential_evolution.shade"])
import ast  # noqa: E402

hooks = sorted({line.split('"')[1] for line in (optimizer_source + shade_source).splitlines()
                if 'hasattr(' in line and '"' in line})
protocols: dict[str, list[str]] = {}
for module in ("scales", "crossovers", "boundaries", "initializers", "diversity",
               "population_schedules", "mutation"):
    tree = ast.parse((ROOT / "differential_evolution" / f"{module}.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "Protocol" for base in node.bases):
            protocols[f"{module}.{node.name}"] = [
                item.name for item in node.body if isinstance(item, ast.FunctionDef)]
declared_anywhere = {name for methods in protocols.values() for name in methods}
undeclared = [h for h in hooks if h not in declared_anywhere]
has_mutation_protocol = any(k.startswith("mutation.") for k in protocols)
check("D4", not undeclared and has_mutation_protocol,
      f"hooks the optimizer probes with hasattr(): {hooks}; "
      f"declared in no Protocol at all: {undeclared or 'none'}; "
      f"Protocols and their methods: {protocols}; "
      f"a Protocol exists for the mutation operator: {has_mutation_protocol}",
      "a Protocol for every extension point (mutation has none) declaring the full "
      "lifecycle, not only __call__")

header("D8  Capability inventory against the 'more functionality' claim")
api = set(de.__all__)
wanted = {
    "JADE-family mutation": {"CurrentToPBest1"},
    "stopping criteria": {"ToleranceStop", "StagnationStop", "TargetValueStop"},
    "callbacks / iteration": {"Callback"},
    "parallel evaluation": set(),
    "constraint handling": {"FeasibilityRules", "PenaltyConstraint"},
    "maximisation": set(),
    "restart strategies": {"Restart"},
    "reflection / midpoint repair": {"ReflectBoundaryHandler", "MidpointBoundaryHandler"},
}
missing_features = [k for k, names in wanted.items() if not (names & api)]
run_params = set(inspect.signature(DifferentialEvolution.__init__).parameters)
check("D8", not missing_features,
      f"absent from __all__: {missing_features}; optimizer accepts {sorted(run_params - {'self'})}",
      "the capability set the paper intends to claim over the two existing DE libraries")

# ============================================================ testing
header("T1  Coverage of the initializer module")
if have("coverage"):
    run(sys.executable, "-m", "coverage", "run", "-m", "pytest", "tests", "-q")
    _, out = run(sys.executable, "-m", "coverage", "report", "-m",
                 "--include=differential_evolution/*")
    lines = [ln for ln in out.splitlines() if "initializers" in ln or "benchmarks" in ln
             or ln.startswith("TOTAL")]
    uncovered_initializers = any("initializers" in ln and "100%" not in ln for ln in lines)
    check("T1", not uncovered_initializers, "\n            ".join(lines),
          "100% on initializers.py — the Tent/OBL/QOBL bodies are where C1 lives")
else:
    print("  skipped (coverage not installed)\n")

header("T2/T3/T4  Test suite shape")
test_files = list((ROOT / "tests").glob("test_*.py"))
text = "\n".join(p.read_text(encoding="utf-8") for p in test_files)
start = time.perf_counter()
code, out = run(sys.executable, "-m", "pytest", "tests", "-q")
duration = time.perf_counter() - start
n_tests = text.count("    def test")
markers = {
    "property-based (hypothesis)": "hypothesis" in text,
    "parametrised component sweep": "parametrize" in text or "subTest" in text,
    "reference-value regression": "cec" in text.lower() or "reference" in text.lower(),
    "bounds-respected invariant": "bounds" in text and "assert" in text and "outside" in text,
    "nfev accounting invariant": "nfev" in text,
}
check("T2", all(markers.values()),
      f"{n_tests} tests run in {duration:.2f}s; invariants present: {markers}",
      "invariant/property tests and reference-value regressions, not only "
      "hand-computed formula checks")

header("T4/C3  Benchmark inventory and dimension handling")
benchmark_names = [n for n in de.__all__ if n.endswith("_function")]
silently_truncating = []
for name in benchmark_names:
    fn = getattr(de, name)
    try:
        two_d = fn([0.1, 0.2])
        five_d = fn([0.1, 0.2, 7.0, 7.0, 7.0])
    except Exception:  # noqa: BLE001 - genuinely n-D functions differ, that is fine
        continue
    if two_d == five_d:
        silently_truncating.append(name)
check("T4", len(benchmark_names) > 30 and not silently_truncating,
      f"{len(benchmark_names)} benchmark functions exported; "
      f"{len(silently_truncating)} of them return the same value for 2-D and 5-D input, "
      f"i.e. silently ignore the extra coordinates: {silently_truncating}",
      "a standard suite (CEC2014/2017/2022 or BBOB/COCO), and a dimension check on every "
      "fixed-dimension function")

# ============================================================ performance
header("P1  sample_distinct_indices — current implementation vs rejection sampling")
current = """
def f(population_size, count, rng, excluded=()):
    available = [i for i in range(population_size) if i not in set(excluded)]
    if count > len(available): raise ValueError
    return list(rng.sample(available, count))
"""
proposed = """
def f(population_size, count, rng, excluded=()):
    ex = set(excluded)
    if count > population_size - len(ex): raise ValueError
    out = []
    while len(out) < count:
        i = rng.randrange(population_size)
        if i in ex or i in out: continue
        out.append(i)
    return out
"""
setup = "import random; rng=random.Random(0); N=100"
timings = {}
for label, src in (("current", current), ("rejection sampling", proposed)):
    timings[label] = timeit.timeit("f(N,3,rng,(7,))", setup=setup + src, number=200_000)
report("P1", f"200k calls at NP=100: current {timings['current']:.3f}s "
             f"({1e6 * timings['current'] / 2e5:.2f} us/call) vs rejection "
             f"{timings['rejection sampling']:.3f}s "
             f"({1e6 * timings['rejection sampling'] / 2e5:.2f} us/call) — "
             f"{timings['current'] / timings['rejection sampling']:.1f}x")

header("P2  dataclasses.fields() calls per run")
import dataclasses  # noqa: E402

original_fields = dataclasses.fields
calls = {"n": 0}


def counting_fields(obj):
    calls["n"] += 1
    return original_fields(obj)


dataclasses.fields = counting_fields
import differential_evolution.mutation as mutation_module  # noqa: E402
import differential_evolution.crossovers as crossovers_module  # noqa: E402

mutation_module.fields = counting_fields
crossovers_module.fields = counting_fields
from differential_evolution import BinomialCrossover, ClipBoundaryHandler, Rand1  # noqa: E402
from differential_evolution import sphere_function as sphere  # noqa: E402

res = DifferentialEvolution(objective=sphere, bounds=[(-5.0, 5.0)] * 10, population_size=50,
                            mutation=Rand1(scale=0.5),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            boundary_handler=ClipBoundaryHandler(),
                            max_generations=100, seed=0).run()
dataclasses.fields = original_fields
mutation_module.fields = original_fields
crossovers_module.fields = original_fields
report("P2", f"{calls['n']:,} dataclasses.fields() calls for {res.nfev:,} evaluations "
             f"({calls['n'] / res.nfev:.1f} per trial vector) — the controller list is "
             "recomputed on every commit, even for a plain float scale")

header("P3  Cost of the diversity machinery")


def rastrigin(x):
    return 10.0 * len(x) + sum(v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in x)


timings = {}
for measures, label in ((None, "none"),
                        ("population_diameter", "one measure"),
                        (["population_diameter", "average_pairwise_distance",
                          "population_radius", "dimensional_variance",
                          "population_coherence",
                          "average_distance_around_all_individuals"], "six measures")):
    start = time.perf_counter()
    DifferentialEvolution(objective=rastrigin, bounds=[(-5.12, 5.12)] * 20,
                          population_size=100, mutation=Rand1(scale=0.5),
                          crossover=BinomialCrossover(crossover_rate=0.9),
                          boundary_handler=ClipBoundaryHandler(), max_generations=300,
                          diversity_measures=measures, seed=0).run()
    timings[label] = time.perf_counter() - start
report("P3", "30 000-evaluation run: " + ", ".join(
    f"{k} {v:.2f}s ({v / timings['none']:.1f}x)" for k, v in timings.items()))

header("P5  record_snapshots memory cost")
res = DifferentialEvolution(objective=sphere, bounds=[(-5.0, 5.0)] * 30, population_size=100,
                            mutation=Rand1(scale=0.7),
                            crossover=BinomialCrossover(crossover_rate=0.9),
                            boundary_handler=ClipBoundaryHandler(),
                            max_generations=200, record_snapshots=True, seed=1).run()
floats = sum(len(s.population) * len(s.population[0]) for s in res.snapshots)
knobs = [p for p in inspect.signature(DifferentialEvolution.__init__).parameters
         if "snapshot" in p or "every" in p or "thin" in p]
report("P5", f"{len(res.snapshots)} snapshots holding {floats:,} floats "
             f"(~{floats * 64 / 1e6:.0f} MB as Python objects); thinning knobs available: {knobs}")

header("Framework overhead per objective evaluation")
for label, factory in (
    ("DE/rand/1/bin", lambda: DifferentialEvolution(
        objective=lambda x: 0.0, bounds=[(-5.12, 5.12)] * 20, population_size=100,
        mutation=Rand1(scale=0.5), crossover=BinomialCrossover(crossover_rate=0.9),
        boundary_handler=ClipBoundaryHandler(), max_generations=300, seed=0)),
):
    start = time.perf_counter()
    res = factory().run()
    elapsed = time.perf_counter() - start
    report("overhead", f"{label}: {elapsed:.2f}s for {res.nfev:,} evaluations with a constant "
                       f"objective -> {1e6 * elapsed / res.nfev:.1f} us/evaluation of pure overhead")

# ============================================================ lint
header("Lint / type-check summary")
if have("ruff"):
    _, out = run(sys.executable, "-m", "ruff", "check", "--select", "E,W,F,B,SIM,PERF,RUF",
                 "--line-length", "120", "--statistics", "differential_evolution")
    print("  ruff:\n    " + "\n    ".join(out.strip().splitlines()[-12:]) + "\n")
else:
    print("  ruff not installed\n")
if have("mypy"):
    _, out = run(sys.executable, "-m", "mypy", "--ignore-missing-imports",
                 "differential_evolution")
    summary = [ln for ln in out.splitlines() if ln.startswith("Found") or ln.startswith("Success")]
    print("  mypy: " + (summary[-1] if summary else "no summary line") + "\n")
else:
    print("  mypy not installed\n")

header("SUMMARY")
if FAILURES:
    print(f"  {len(FAILURES)} checks reproduce a reported finding: {', '.join(FAILURES)}")
else:
    print("  all pass/fail checks pass")
print("  performance and lint entries above are measurements, not pass/fail assertions.")
