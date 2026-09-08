Developer guide
===============

Development workflow
--------------------

Run the test suite from the repository root:

.. code-block:: bash

   python3 -m unittest discover -s tests -v

Build the documentation:

.. code-block:: bash

   python3 -m sphinx -W --keep-going -b html docs docs/_build/html

For Read the Docs style builds, the repository configuration is:

.. code-block:: text

   .readthedocs.yaml
   docs/conf.py
   docs/requirements.txt

Adding a new mutation strategy
------------------------------

For the composable optimizer:

* implement a callable object that accepts
  :class:`~differential_evolution.MutationContext`;
* document its mathematical formula and sampling constraints;
* add deterministic unit tests for the formula and edge cases;
* add it to the public API only if it is intended for users.

Adding a new crossover operator
-------------------------------

* implement a callable taking ``target_vector``, ``donor_vector``, and ``rng``;
* specify whether at least one donor coordinate is guaranteed;
* test ``CR = 0`` and ``CR = 1`` behavior when applicable.

Adding a new dedicated solver
-----------------------------

If an algorithm has tightly coupled state, such as archives or parameter
memories, prefer a dedicated optimizer class rather than forcing it through the
generic mutation/crossover interfaces.

Documentation expectations
--------------------------

Before submitting a documentation-related change:

* run the unit tests;
* run the Sphinx HTML build with warnings treated as errors;
* execute or doctest any new examples;
* update the bibliography when adding literature-backed algorithms.
* when a source cannot be verified, mark it clearly for manual review instead
  of inventing a citation.
