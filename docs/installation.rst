Installation
============

Supported Python versions
-------------------------

The project metadata currently declares support for Python 3.11 and later.

PyPI status
-----------

There is currently no verified PyPI release workflow or published package page
in this repository. The reliable installation path is therefore from source.

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

   python3 -m pip install -e .[docs]
   python3 -m unittest discover -s tests -v

Read the Docs
-------------

The repository includes a ``.readthedocs.yaml`` configuration so the standard
Read the Docs build can install the documentation requirements and build the
HTML output from ``docs/``.

License
-------

The project metadata declares the license as MIT. At the time of writing there
is no separate top-level ``LICENSE`` file in the repository, so that should be
added if the project is intended for redistribution.
