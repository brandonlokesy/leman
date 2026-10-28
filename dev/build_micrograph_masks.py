"""
Build the text masks that ``leman.loaders.Micrograph`` matches an image's info bar
against, and write them to ``src/leman/data/micrograph_masks.npz``.

Run from the repository root after changing any ``_MICROGRAPH_*`` box or threshold
in ``loaders.py``, or after adding a magnification:

    conda run --no-capture-output -n viz-sci-plot python dev/build_micrograph_masks.py

Each mask is cut from one committed example, ``<M>x_example_1.jpg``, and the
``X:`` mask from ``100x_example_1.jpg``. Every other committed example is then an
image the masks were not built from, which is what lets the tests show the match
generalises. ``tests/test_micrograph.py`` rebuilds the masks the same way and
fails if the shipped file differs.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from leman.loaders import (
    _MICROGRAPH_LABEL_BOX,
    _MICROGRAPH_SCALE_UM,
    _MICROGRAPH_X_BOX,
    _micrograph_text_mask,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "examples" / "data" / "optical-microscope"
TARGET = ROOT / "src" / "leman" / "data" / "micrograph_masks.npz"


def build_masks(source: Path = SOURCE) -> dict:
    """Return ``{"5": mask, …, "100": mask, "x": mask}``, bool arrays."""
    def frame(name):
        return np.asarray(Image.open(source / name).convert("RGB"))

    # One label mask per magnification, keyed by the magnification as a string
    # because .npz keys must be strings.
    masks = {
        str(m): _micrograph_text_mask(frame(f"{m}x_example_1.jpg")[_MICROGRAPH_LABEL_BOX])
        for m in _MICROGRAPH_SCALE_UM
    }
    masks["x"] = _micrograph_text_mask(frame("100x_example_1.jpg")[_MICROGRAPH_X_BOX])
    return masks


if __name__ == "__main__":
    TARGET.parent.mkdir(exist_ok=True)
    np.savez_compressed(TARGET, **build_masks())
    print(f"wrote {TARGET.relative_to(ROOT)}")
