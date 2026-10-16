# Publishing DEFoundry

The PyPI distribution is `defoundry`; the import package remains
`differential_evolution`. Version `0.1.0` was the first release; `0.2.0` adds ordered batch evaluation
and run provenance (see `CHANGELOG.md`).
The unrelated PyPI project `differential-evolution` must not be used.

## One-time account setup

1. Confirm that `defoundry` can be registered on both PyPI and TestPyPI.
   A missing project page is not a reservation or a guarantee of availability.
2. Create GitHub environments named `testpypi` and `pypi` in
   `vrbaj/differential_evolution`. Restrict release deployments to version tags;
   configure required reviewers if desired.
3. On each index, configure a GitHub Trusted Publisher (a **pending publisher**
   for the first release) with these exact values:

   | Field | Value |
   | --- | --- |
   | PyPI project | `defoundry` |
   | Owner | `vrbaj` |
   | Repository | `differential_evolution` |
   | Workflow filename | `release.yml` |
   | Environment | `testpypi` on TestPyPI, `pypi` on PyPI |

   See [PyPI's pending publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
   The accounts and publisher registrations are separate on the two indexes.
   No API token is needed by this workflow.

## Validate locally

Use a clean checkout so old `build/`, `dist/`, or egg-info files cannot pollute
the release. Do not copy an existing build directory into it.

```bash
python3 -m venv .venv
# POSIX; on Windows use .venv\Scripts\activate
. .venv/bin/activate
python -m pip install -e '.[test,docs,release]'
python -m unittest discover -s tests -v
python -m ruff check .
python -m mypy
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check --strict dist/*
python scripts/check_distribution.py dist
```

`python -m build` creates the source distribution, then builds the wheel from
that archive. The distribution check verifies metadata, license and typing data,
rejects stray wheel modules, installs the wheel without dependencies into a
temporary directory, runs DE/SHADE/L-SHADE without access to the checkout or
site-packages, and runs the test suite from the extracted source archive.
CI runs this check along with the Python 3.11–3.14 test matrix on Linux, Windows,
and macOS. Only `differential_evolution` is installed by the wheel; the legacy
top-level shims are available in the source archive. Installed users should use
`differential_evolution.compat`.

## Release a version

1. Set `[project].version` in `pyproject.toml` and `__version__` in
   `differential_evolution/_version.py`; Sphinx reads the project metadata.
   Update `CHANGELOG.md`, installation instructions and migration guidance.
2. Run the local validation above. Commit the release files and push the commit
   plus a matching annotated tag, such as `v0.2.0`.
3. Pushing a version tag automatically starts **release**. It checks that the tag
   matches package metadata, runs the complete test matrix and distribution
   checks, and publishes the verified artifacts through the `pypi` environment's
   trusted publisher. A protected environment may require its configured approval.
4. Confirm the workflow succeeds and verify a fresh installation:

   ```bash
   python -m pip install defoundry==0.2.0
   python -c "from differential_evolution import DifferentialEvolution, SHADE, LSHADE"
   ```

For a manual run, select **release** in GitHub Actions, choose a version **tag**
as the ref, and select `testpypi` or `pypi`. Branch refs are rejected. TestPyPI
and PyPI have separate publisher registrations; a manual TestPyPI run does not
publish to production. A tag push starts production publication automatically,
so TestPyPI is an optional separate check in this workflow.

Each run uploads exactly the artifacts it verified. Publishing a GitHub release
alone does not upload to PyPI. Published PyPI versions cannot be overwritten;
do not move their tags, and use a new version for corrections. The workflow does
not register project names or configure accounts/publisher registrations itself.
