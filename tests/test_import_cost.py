"""
What ``import leman`` is allowed to drag in.

scikit-learn is needed by exactly one function, :func:`fitting.fit_sparse_lifetime`,
and it is a heavy import. Importing it at module level made every user of the
package — including someone who only wanted to plot — pay for it, and broke every
existing environment until scikit-learn was installed (**E22**).

Checked in a fresh interpreter, because the test session has almost certainly
imported scikit-learn already for something else, and ``sys.modules`` in this
process would then say nothing about what the package itself pulls in.
"""

import subprocess
import sys

import numpy as np

from leman import fitting


def test_importing_the_package_does_not_import_sklearn():
    probe = (
        "import sys; import leman; "
        "print(any(m == 'sklearn' or m.startswith('sklearn.') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "False", f"import leman pulled in scikit-learn ({out})"


def test_the_function_that_needs_sklearn_can_still_reach_it():
    """
    The other half: a lazy import that is never exercised is a NameError waiting
    to happen, and nothing else in the suite calls this function.

    A smoke test only — that the call completes and returns the result type. It
    makes no claim about the lifetimes, which are fitted against a model that is
    misaligned with the data (**A24**).
    """
    t = np.linspace(-0.2, 2.0, 200)
    y = np.where(t >= 0, np.exp(-t / 0.5), 0.0)
    kernel = np.array([0.25, 0.5, 0.25])  # a stand-in IRF, not a measured one

    result = fitting.fit_sparse_lifetime(t, y, kernel, n_tau=30)

    assert isinstance(result, fitting.SparseLifetimeResult)
