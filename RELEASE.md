# Publishing DEFoundry

The PyPI distribution is `defoundry`; the import package remains
`differential_evolution`. Version `0.1.0` is the first release.
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

1. Set `[project].version` in `pyproject.toml`; Sphinx reads the same value.
   Confirm the README's installation text matches the release status and write
   release notes for user-visible changes.
2. Commit all release files, push the commit and a matching tag such as `v0.1.0`.
   Do not move a tag after publishing it.
3. In GitHub Actions, manually run **release**, selecting that tag as the ref
   and `testpypi` as the index. The workflow refuses branch refs or tags that
   do not match the package version. All tests and distribution checks must
   pass before the upload job can run.
4. Check the TestPyPI project page and install the exact version in a new venv:

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ --no-deps defoundry==0.1.0
   python -c "from differential_evolution import DifferentialEvolution, SHADE, LSHADE"
   ```

5. Run **release** again for the same tag, choosing `pypi`. This run builds and
   checks its own artifacts before uploading. Afterwards verify a fresh install
   with `python -m pip install defoundry==0.1.0` and inspect the PyPI README.

Update the example version for subsequent releases. Each workflow run uploads
exactly the artifacts it verified. Publishing a GitHub release or pushing a tag
alone does not upload anything. PyPI versions cannot be overwritten; make a new
version for corrections. This repository setup does not register the name,
configure accounts, or upload a release by itself.
