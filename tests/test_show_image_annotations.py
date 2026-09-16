"""
What ``_ACImg.show_image`` draws on top of the image, and what it labels.

The legend is built by ``show_image`` from the artists it actually drew. Neither
drawer can build it: ``ax.legend(handles=[...])`` takes an explicit list, so a
legend raised inside one drawer silently omits the other's artist. That is what
used to happen — the background-region box carried ``label="bg region"`` and no
legend ever contained it, because the only legend in this path was the one the
laser-circle helper built from ``handles=[circle]``.

The box and the circle are read back off the axes by patch type, since
``show_image`` returns only ``(fig, ax)``.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings

import numpy as np
import pytest

from leman.loaders import ACImg

SHAPE = (8, 8)
SIGNAL_FILL = 100.0
BG_REGION = (slice(0, 2), slice(0, 2))


class _FakeLaserRef:
    """The published duck-type: centre and 1/e² radius, nothing else."""
    center_x, center_y, radius = 4.0, 4.0, 2.0


def _image(tmp_path, name="frame.csv", **kwargs):
    img = np.full(SHAPE, SIGNAL_FILL)
    img[BG_REGION] = 1.0
    path = tmp_path / name
    np.savetxt(path, img, delimiter=",")
    return ACImg(path, **kwargs)


def _drawn(ax):
    """(n boxes, n circles, legend labels) — labels is None when no legend."""
    kinds = [type(p).__name__ for p in ax.patches]
    legend = ax.get_legend()
    return (kinds.count("Rectangle"), kinds.count("Circle"),
            [t.get_text() for t in legend.get_texts()] if legend else None)


def test_the_box_reaches_the_legend(tmp_path):
    # The regression: this labelled the box and then never showed the label.
    image = _image(tmp_path, bg_region=BG_REGION)
    fig, ax = image.show_image(show_bg_region=True, legend=True)

    assert _drawn(ax) == (1, 0, ["bg region"])
    plt.close(fig)


def test_the_box_and_the_circle_are_labelled_together(tmp_path):
    # Two labelled artists, one legend. The old code listed only the circle,
    # because the circle helper built the legend from handles=[circle].
    image = _image(tmp_path, bg_region=BG_REGION, laser_ref=_FakeLaserRef())
    fig, ax = image.show_image(show_bg_region=True, laser_annotation=True,
                              legend=True)

    boxes, circles, labels = _drawn(ax)
    assert (boxes, circles) == (1, 1)
    assert labels == ["bg region", "$1/e^2$ Radius (2.0 px)"]
    plt.close(fig)


def test_legend_false_draws_no_legend_even_with_annotations(tmp_path):
    # legend= decides. Asking for an annotation no longer switches it on:
    # show_bg_region used to force legend=True, which then labelled the circle
    # rather than the box it was forced on behalf of.
    image = _image(tmp_path, bg_region=BG_REGION, laser_ref=_FakeLaserRef())
    fig, ax = image.show_image(show_bg_region=True, laser_annotation=True,
                              legend=False)

    assert _drawn(ax) == (1, 1, None)
    plt.close(fig)


def test_legend_true_alone_draws_nothing_to_label(tmp_path):
    # No annotation asked for, so there is no handle and no empty legend box.
    image = _image(tmp_path, bg_region=BG_REGION)
    fig, ax = image.show_image(legend=True)

    assert _drawn(ax) == (0, 0, None)
    plt.close(fig)


def test_asking_for_an_absent_region_warns_and_draws_nothing(tmp_path):
    # Silent before: the caller asked to see the region and got an unannotated
    # image back. Always the case on a class that takes no bg_region at all.
    image = _image(tmp_path)

    with pytest.warns(UserWarning, match="without a bg_region") as caught:
        fig, ax = image.show_image(show_bg_region=True, legend=True)

    assert _drawn(ax) == (0, 0, None)
    # Measured, not assumed: the warning must blame the caller's line.
    assert caught[0].filename == __file__
    plt.close(fig)


def test_a_present_region_does_not_warn(tmp_path):
    image = _image(tmp_path, bg_region=BG_REGION)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, ax = image.show_image(show_bg_region=True)

    assert _drawn(ax)[0] == 1
    plt.close(fig)
