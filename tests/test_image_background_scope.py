"""
Which single-image classes accept a background region, and which refuse one.

``_ACImg`` builds ``img`` and so is where ``bg_region`` / ``bg_stat``
live, but that is a fact about where the array is assembled, not a claim that
every kind of image wants a pedestal removed.  The scope is deliberate:

* A real-space PL frame has a genuinely dark corner, and numbers are computed
  from the frame — ``analyse_diffusion_cloud`` thresholds it and takes areas
  and second moments — so a pedestal biases a result.
* A white-light frame's corner is substrate, which reflects, and nothing
  computes from the frame.  The correction such an image wants is a ratio
  against a reference frame, which is not what a scalar subtraction is.

Pinned as a table because the refusal is expressed by a *narrower*
``__init__`` signature.  Deleting that override would silently inherit the
base's parameters and hand back the knob, and no other test would notice.
"""

import numpy as np
import pytest

from leman.loaders import (
    ACImg,
    ACSampleImg,
    SingleImage,
)

SHAPE = (6, 6)
SIGNAL_FILL = 100.0
# A corner whose values are [1, 1, 1, 21]: median 1, mean 6, so a test can say
# which statistic ran rather than only that something was subtracted.
BG_REGION = (slice(0, 2), slice(0, 2))
BG_MEDIAN = 1.0


def _image_csv(tmp_path, name="frame.csv"):
    img = np.full(SHAPE, SIGNAL_FILL)
    img[BG_REGION] = [[1.0, 1.0], [1.0, 21.0]]
    path = tmp_path / name
    np.savetxt(path, img, delimiter=",")
    return path


def test_a_pl_frame_takes_a_background_region(tmp_path):
    image = ACImg(_image_csv(tmp_path), bg_region=BG_REGION)

    assert image.bg_region == BG_REGION
    assert np.allclose(image.img.max(), SIGNAL_FILL - BG_MEDIAN)
    # The file's own counts stay reachable, per raw-arrays-are-never-mutated.
    assert np.allclose(image.img_raw.max(), SIGNAL_FILL)


@pytest.mark.parametrize("cls", [ACSampleImg, SingleImage])
def test_the_classes_that_refuse_one_refuse_it_at_the_signature(cls, tmp_path):
    with pytest.raises(TypeError, match="bg_region"):
        cls(_image_csv(tmp_path), bg_region=BG_REGION)


@pytest.mark.parametrize("cls", [ACSampleImg, SingleImage])
def test_a_refusing_class_still_carries_the_attribute_as_none(cls, tmp_path):
    """
    The base assigns ``bg_region``, so the attribute exists either way and is
    always ``None`` here.  Anything reading it off a duck-typed image — the
    viewer's ``show_bg_region=`` does — therefore finds no region rather than
    an AttributeError.
    """
    image = cls(_image_csv(tmp_path))

    assert image.bg_region is None
    assert image.bg_stat == "median"
    assert np.allclose(image.img, image.img_raw)
