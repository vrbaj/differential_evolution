"""Check archive contents, an isolated wheel installation, and the sdist tests."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import tomllib
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile


def check_distributions(directory: Path) -> None:
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("Expected exactly one wheel and one sdist in the distribution directory")
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    with ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        assert "differential_evolution/py.typed" in names, "Missing typing marker"
        assert all(
            name.startswith("differential_evolution/") or ".dist-info/" in name
            for name in names
        ), "Unexpected top-level modules in wheel (check for stale build files)"
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(wheel.read(metadata_name))
        assert metadata["Name"] == project["name"]
        assert metadata["Version"] == project["version"]
        assert metadata["Requires-Python"] == project["requires-python"]
        assert metadata["License-Expression"] == "MIT"
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)

    with tempfile.TemporaryDirectory(prefix="de-distribution-") as temporary:
        root = Path(temporary)
        installed = root / "installed"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
             "--target", str(installed), str(wheels[0].resolve())],
            check=True,
        )
        # Ignore environment, user site, editable installs, and the checkout.
        subprocess.run(
            [sys.executable, "-I", "-S", "-c", """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import differential_evolution as de
import differential_evolution.compat
assert Path(de.__file__).is_relative_to(Path(sys.argv[1]))
for name in de.__all__:
    assert getattr(de, name) is not None
for optimizer_type in (de.DifferentialEvolution, de.SHADE, de.LSHADE):
    options = dict(objective=de.sphere_function, bounds=[(-5., 5.)] * 2,
                   population_size=12, max_evaluations=36, seed=123)
    if optimizer_type is de.DifferentialEvolution:
        options.update(mutation=de.Rand1(scale=0.8),
                       crossover=de.BinomialCrossover(crossover_rate=0.9))
    result = optimizer_type(**options).run()
    assert result.success
    assert result.nfev == 36
print('Installed wheel: public imports and DE/SHADE/LSHADE runs passed')
""", str(installed)],
            cwd=root,
            check=True,
        )
        with tarfile.open(sdists[0]) as sdist:
            sdist.extractall(root / "source", filter="data")
        source, = (root / "source").iterdir()
        for required in ("LICENSE", "README.md", "pyproject.toml", "tests", "docs/conf.py",
                         "main.py", "population_initialization.py", "testing_functions.py"):
            assert (source / required).exists(), f"Missing sdist file: {required}"
        subprocess.run(
            [sys.executable, "-E", "-m", "unittest", "discover", "-s", "tests"],
            cwd=source,
            check=True,
        )
    print("Wheel and source distribution checks passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/check_distribution.py DIST_DIRECTORY")
    check_distributions(Path(sys.argv[1]))
