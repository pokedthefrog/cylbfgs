=======
cyLBFGS
=======

**cyLBFGS** is a Cython wrapper around `a slightly modified version`_ of
Naoaki Okazaki (chokkan)'s liblbfgs_ optimisation library. It implements both
the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) and Orthant-Wise
Limited-memory Quasi-Newton (OWL-QN) methods.

This package aims to provide Python users with a cleaner, more comprehensive
interface to the L-BFGS algorithm than is currently available in SciPy_,
including access to the OWL-QN algorithm for solving L1-regularised problems.

Installation
============

Directly from GitHub:

- **with poetry**::

    poetry add git+https://github.com/pokedthefrog/cylbfgs.git

- **or with pip**::

    pip install git+https://github.com/pokedthefrog/cylbfgs.git

Authors
=======
This package is based on the `dedupe.io fork of PyLBFGS`_ by Forest Gregg. The
original code was written by Lars Buitinck.


.. _a slightly modified version: https://github.com/pokedthefrog/liblbfgs
.. _liblbfgs: https://www.chokkan.org/software/liblbfgs/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
.. _dedupe.io fork of PyLBFGS: https://github.com/dedupeio/pylbfgs
