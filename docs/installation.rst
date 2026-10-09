Installation
============

Supported Python versions
-------------------------

The project metadata currently declares support for Python 3.11 and later.

PyPI status
-----------

The distribution name is ``defoundry``; Python imports continue to use
``differential_evolution``. Install from PyPI:

.. code-block:: bash

   python3 -m pip install defoundry

The repository includes distribution checks and a manual publishing workflow.
See the `release guide <https://github.com/vrbaj/differential_evolution/blob/master/RELEASE.md>`_
for maintainer setup and release instructions.

Installation from source
------------------------

From the repository root:

.. code-block:: bash

   python3 -m pip install -e .

This installs the :mod:`differential_evolution` package in editable mode.

Documentation dependencies
--------------------------

The documentation uses Sphinx and ``sphinxcontrib-bibtex``.

.. code-block:: bash

   python3 -m pip install -e .[docs]

If your shell treats brackets specially, quote the extra:

.. code-block:: bash

   python3 -m pip install -e '.[docs]'

Development installation
------------------------

For development and testing:

.. code-block:: bash

   python3 -m pip install -e '.[test,docs,release]'
   python3 -m unittest discover -s tests -v

Read the Docs
-------------

The repository includes a ``.readthedocs.yaml`` configuration so the standard
Read the Docs build can install the documentation requirements and build the
HTML output from ``docs/``.

License
-------

The project is distributed under the MIT license; see the top-level ``LICENSE``
file, which is also included in the distribution.
