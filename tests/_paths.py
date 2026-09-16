"""Where the committed example data lives, anchored on this file.

`DATA` is derived from `__file__`, not from the working directory, so a test that
opens a committed export resolves it the same way whether pytest was started from
the repository root, from `tests/`, or from an editor's run button.

Paths built from `DATA` are handed to the loaders as `str`, which is what they
take today — `ACSpectralSweep`, `RamanSpectrum` and the rest do
`self.path = str(path)` or pass the argument straight to `np.loadtxt`.
"""

from pathlib import Path

# tests/_paths.py -> tests/ -> the repository root.
DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
