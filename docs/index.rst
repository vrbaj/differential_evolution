Differential Evolution documentation
====================================

This manual documents the :mod:`differential_evolution` package as it is
implemented in this repository. It covers the composable general-purpose
optimizer, the dedicated :class:`~differential_evolution.SHADE` and
:class:`~differential_evolution.LSHADE` solvers, the mathematical definitions
of the implemented strategies, and the practical steps needed to reproduce
results and inspect runs after they finish.

.. note::

   The project is still young. The implementation is functional and tested,
   but the public API should be treated as pre-1.0 and therefore potentially
   subject to change.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   introduction
   installation
   quickstart
   concepts
   algorithm_reference
   customization
   practical_guides
   examples
   api/index
   developer_guide
   discrepancies
   references
   citation_and_license
