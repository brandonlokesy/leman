"""
Tests for the diffusion-cloud analysis/plotting double-subtraction fix (A5).

``analyse_diffusion_cloud`` and ``plot_diffusion_cloud`` accept both a bare
2-D array and an ``_ACImg``-family object (which may already be
background-subtracted, from a ``bg_region`` passed at *construction* time).
The bug: handing the plotting function's already-corrected ``.img`` array
to the analysis function, plus a *second* ``bg_region``, subtracted twice.
The fix relies on passing the image *object* through so ``_load_image`` can
apply its own "use img_raw for an _ACImg" rule, and on
``DiffusionResult.image`` carrying the array that was actually analysed
(what a caller should display), rather than every caller re-deriving it.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np

from leman import plotting
from leman.diffusion import (
    analyse_diffusion_cloud,
    analyse_diffusion_sequence,
)
from leman.loaders import ACImg

BG_LEVEL  = 50.0
BG_REGION = (slice(0, 3), slice(0, 3))  # flat corner patch, away from the cloud


def _make_image(tmp_path):
    img = np.full((10, 10), BG_LEVEL)
    img[5:8, 5:8] += 500.0  # bright "cloud", well above background + threshold
    path = tmp_path / "frame.csv"
    np.savetxt(path, img, delimiter=",")
    return path


def test_loader_bg_region_is_a_single_subtraction(tmp_path):
    # Sanity check on the loader itself, independent of the diffusion code:
    # bg_region at construction leaves that patch at ~0 exactly once.
    image = ACImg(_make_image(tmp_path), bg_region=BG_REGION)
    assert np.allclose(image.img[BG_REGION], 0.0, atol=1e-9)
    assert np.allclose(image.img_raw[BG_REGION], BG_LEVEL, atol=1e-9)


def test_analyse_diffusion_cloud_does_not_double_subtract(tmp_path):
    image = ACImg(_make_image(tmp_path), bg_region=BG_REGION)

    result = analyse_diffusion_cloud(image, bg_region=BG_REGION, smooth_sigma=0.0)

    # A second subtraction of the same already-corrected region would drive
    # it to -BG_LEVEL instead of leaving it at ~0.
    assert np.allclose(result.image[BG_REGION], 0.0, atol=1e-9)
    assert not np.allclose(result.image[BG_REGION], -BG_LEVEL, atol=1e-3)
    # The cloud itself is unaffected either way.
    assert result.area_px2 > 0


def test_analyse_diffusion_cloud_on_a_bare_array_still_subtracts_once(tmp_path):
    # A caller who was never handed an _ACImg (a raw array) must
    # still get exactly one subtraction, not zero.
    img = np.full((10, 10), BG_LEVEL)
    img[5:8, 5:8] += 500.0

    result = analyse_diffusion_cloud(img, bg_region=BG_REGION, smooth_sigma=0.0)
    assert np.allclose(result.image[BG_REGION], 0.0, atol=1e-9)


def test_plot_diffusion_cloud_displays_the_same_array_it_analysed(tmp_path):
    image = ACImg(_make_image(tmp_path), bg_region=BG_REGION)

    fig, ax, result = plotting.plot_diffusion_cloud(
        image, bg_region=BG_REGION, smooth_sigma=0.0, laser_annotation=False,
    )
    assert np.allclose(result.image[BG_REGION], 0.0, atol=1e-9)

    displayed = np.asarray(ax.images[0].get_array())
    assert np.array_equal(displayed, result.image)
    matplotlib.pyplot.close(fig)


def test_plot_diffusion_cloud_with_precomputed_result_uses_its_image(tmp_path):
    image = ACImg(_make_image(tmp_path), bg_region=BG_REGION)
    result = analyse_diffusion_cloud(image, bg_region=BG_REGION, smooth_sigma=0.0)

    fig, ax, result_out = plotting.plot_diffusion_cloud(
        image, result=result, laser_annotation=False,
    )
    assert result_out is result
    displayed = np.asarray(ax.images[0].get_array())
    assert np.array_equal(displayed, result.image)
    matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# scale_units — the unit pixel_scale was calibrated in (B2)
# ---------------------------------------------------------------------------
#
# pixel_scale converts pixels to a real-space length, and the scale is typically
# calibrated per session against a known laser-spot size, so the unit is the
# caller's choice.  It therefore has to travel with the result: nothing
# downstream can infer it, and before B2 it was accepted and dropped.


def _plain_cloud():
    """A bright square on a dark field — no loader, no background involved."""
    img = np.zeros((20, 20))
    img[8:12, 8:12] = 100.0
    return img


def test_the_real_space_lines_carry_the_unit():
    text = repr(analyse_diffusion_cloud(_plain_cloud(),
                                        pixel_scale=65.0, scale_units="nm"))

    assert "Centroid (nm)" in text
    assert "Area (nm²)" in text          # an area is in the unit squared
    assert "(real)" not in text          # the placeholder this replaced


def test_the_default_unit_is_micrometres():
    text = repr(analyse_diffusion_cloud(_plain_cloud(), pixel_scale=0.065))

    assert "Centroid (µm)" in text
    assert "Area (µm²)" in text


def test_no_scale_means_no_unit_is_printed():
    # The unit must never appear beside a number that is still in pixels.
    text = repr(analyse_diffusion_cloud(_plain_cloud()))

    assert "Centroid (px)" in text
    assert "µm" not in text
    assert "Centroid (" in text and text.count("Centroid") == 1


def test_the_unit_reaches_every_frame_of_a_sequence():
    frames = [_plain_cloud() for _ in range(3)]

    seq = analyse_diffusion_sequence(frames, pixel_scale=65.0, scale_units="nm",
                                     smooth_sigma=0.0)

    assert seq.scale_units == "nm"
    assert all(f.scale_units == "nm" for f in seq.frames)


def test_the_trajectory_plot_labels_its_axis_with_the_unit():
    # The assertion that fails before B2: the label read "Centroid (real)",
    # the word "real" standing where a unit belongs.
    frames = [_plain_cloud() for _ in range(3)]
    seq = analyse_diffusion_sequence(frames, pixel_scale=65.0, scale_units="nm",
                                     smooth_sigma=0.0)

    fig, ax = plotting.plot_centroid_trajectory(seq, use_real=True)

    assert ax.get_ylabel() == "Centroid (nm)"
    assert [line.get_label() for line in ax.lines] == ["x (nm)", "y (nm)"]
    matplotlib.pyplot.close(fig)


def test_the_trajectory_plot_still_says_px_without_a_scale():
    frames = [_plain_cloud() for _ in range(3)]
    seq = analyse_diffusion_sequence(frames, smooth_sigma=0.0)

    fig, ax = plotting.plot_centroid_trajectory(seq, use_real=True)

    # use_real=True but nothing was converted, so the pixel branch holds.
    assert ax.get_ylabel() == "Centroid (px)"
    matplotlib.pyplot.close(fig)
