"""
Tests for Micrograph — the main-lab Olympus microscope's JPEG export.

Run against the committed examples (examples/data/optical-microscope/): 24 images
with an info bar at 5×–100×, four BW_/aperture_ exports with none, and one
transfer-stage image that must be refused. The label masks are cut from the
``*_example_1`` files, so every other labelled example is an image the masks were
not built from; those are the cases that show the match generalises.

An overlong sample name is simulated by painting white over the bar and saving as
PNG, which is lossless, so the painted pixels reach the loader unchanged.
"""

import importlib.util
import re

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image

from leman.loaders import Micrograph, _micrograph_masks
from leman.plotting import MicrographPlot, plot_micrograph

from _paths import DATA

ROOT = DATA / "optical-microscope"
LABELLED = sorted(p for p in ROOT.glob("*.jpg") if re.match(r"\d+x_example_\d+", p.name))
NO_BAR = sorted(p for p in ROOT.glob("*.jpg") if p.name.startswith(("BW_", "aperture_")))
FIFTY = str(ROOT / "50x_example_2.jpg")
BW = str(ROOT / "BW_example_50x.jpg")


def magnification_in_name(path):
    return int(re.match(r"(\d+)x_", path.name).group(1))


def test_the_committed_examples_are_all_found():
    assert len(LABELLED) == 24
    assert len(NO_BAR) == 4


# ---------------------------------------------------------------------------
# Reading the bar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", LABELLED, ids=lambda p: p.name)
def test_every_label_reads_the_magnification_in_its_file_name(path):
    img = Micrograph(str(path))
    assert img.has_info_bar
    assert img.label_magnification == magnification_in_name(path)
    assert img.magnification == magnification_in_name(path)


@pytest.mark.parametrize("path", NO_BAR, ids=lambda p: p.name)
def test_an_export_without_a_bar_loads_with_nothing_known(path):
    img = Micrograph(str(path))
    assert not img.has_info_bar
    assert img.label_magnification is None
    assert img.magnification is None
    assert img.pixel_size is None


def test_an_image_from_another_instrument_is_refused():
    with pytest.raises(ValueError, match="2848 × 2848 px"):
        Micrograph(str(ROOT / "example_reject.png"))


def test_a_wrong_size_is_refused_before_the_pixels_are_decoded(monkeypatch):
    def no_decoding(*args, **kwargs):
        raise AssertionError("pixels were decoded")
    monkeypatch.setattr(Image.Image, "convert", no_decoding)
    with pytest.raises(ValueError, match="2848 × 2848 px"):
        Micrograph(str(ROOT / "example_reject.png"))


def test_nominal_pixel_size_is_the_scale_bar_over_363_px():
    assert Micrograph(FIFTY).pixel_size == pytest.approx(50 / 363)


def test_raw_image_is_the_full_rgb_frame():
    img = Micrograph(FIFTY)
    assert img.img_raw.shape == (2048, 2448, 3)
    assert img.img_raw.dtype == np.uint8
    assert img.img is img.img_raw


# ---------------------------------------------------------------------------
# A sample name long enough to run into the scale bar or the label
# ---------------------------------------------------------------------------


def painted(tmp_path, last_col):
    """A real 50× frame with white text-like paint from column 600 to *last_col*."""
    frame = np.asarray(Image.open(FIFTY).convert("RGB")).copy()
    frame[2010:2047, 600:last_col + 1] = 255
    out = tmp_path / "painted.png"
    Image.fromarray(frame).save(out)
    return str(out)


def test_a_name_across_the_whole_scale_bar_still_reads(tmp_path):
    img = Micrograph(painted(tmp_path, last_col=2185))
    assert img.has_info_bar
    assert img.magnification == 50


def test_a_name_covering_the_label_is_unreadable_not_guessed(tmp_path):
    path = painted(tmp_path, last_col=2300)
    with pytest.raises(ValueError, match="cannot be read"):
        Micrograph(path)
    assert Micrograph(path, magnification=50).label_magnification is None


# ---------------------------------------------------------------------------
# Declared magnification, label_reads=, pixel_size=
# ---------------------------------------------------------------------------


def test_a_matching_magnification_is_accepted():
    assert Micrograph(FIFTY, magnification=50).magnification == 50


def test_a_disagreeing_magnification_is_refused_and_the_fix_is_named():
    with pytest.raises(ValueError, match="magnification=100, label_reads=50"):
        Micrograph(FIFTY, magnification=100)


def test_label_reads_overrides_a_wrong_label():
    img = Micrograph(FIFTY, magnification=100, label_reads=50)
    assert img.magnification == 100
    assert img.label_magnification == 50
    assert img.pixel_size == pytest.approx(25 / 363)
    assert "label reads 50×" in repr(img)


def test_label_reads_must_match_the_label():
    with pytest.raises(ValueError, match="reads 50×"):
        Micrograph(FIFTY, magnification=100, label_reads=20)


def test_label_reads_needs_magnification():
    with pytest.raises(ValueError, match="needs magnification="):
        Micrograph(FIFTY, label_reads=50)


def test_label_reads_equal_to_magnification_is_refused():
    with pytest.raises(ValueError, match="nothing to correct"):
        Micrograph(FIFTY, magnification=50, label_reads=50)


def test_label_reads_on_an_image_without_a_bar_is_refused():
    with pytest.raises(ValueError, match="no info bar"):
        Micrograph(BW, magnification=100, label_reads=50)


def test_an_uncalibrated_magnification_is_refused():
    with pytest.raises(ValueError, match="pixel_size="):
        Micrograph(BW, magnification=40)


def test_magnification_on_an_image_without_a_bar():
    img = Micrograph(BW, magnification=50)
    assert img.magnification == 50
    assert img.pixel_size == pytest.approx(50 / 363)


def test_pixel_size_replaces_the_nominal_value_and_keeps_the_label():
    img = Micrograph(FIFTY, pixel_size=0.14)
    assert img.pixel_size == 0.14
    assert img.label_magnification == 50
    # The label may be the wrong one, so no magnification is left in use.
    assert img.magnification is None
    assert "given" in repr(img)


def test_a_given_pixel_size_prints_its_equivalent_magnification():
    # 80 µm over 587 px: 2500 µm / 363 px divided by 80/587 µm/px ≈ 50.5×.
    text = repr(Micrograph(FIFTY, pixel_size=80 / 587))
    assert "≈ 50.5× at the nominal calibration" in text
    assert "label reads 50×" in text


@pytest.mark.parametrize("kwargs", [
    {"magnification": 50.0},
    {"magnification": True},
    {"magnification": "50"},
    {"magnification": 100, "label_reads": 50.0},
    {"pixel_size": "0.14"},
    {"pixel_size": True},
])
def test_an_argument_of_the_wrong_type_is_refused_with_a_clear_message(kwargs):
    with pytest.raises(ValueError, match="must be a"):
        Micrograph(FIFTY, **kwargs)


def test_numpy_numbers_are_accepted():
    assert Micrograph(FIFTY, magnification=np.int64(50)).magnification == 50
    assert Micrograph(BW, pixel_size=np.float32(0.14)).pixel_size == pytest.approx(0.14)


def test_pixel_size_with_magnification_is_refused():
    with pytest.raises(ValueError, match="not both"):
        Micrograph(BW, magnification=50, pixel_size=0.14)


@pytest.mark.parametrize("bad", [0, -0.1, float("nan")])
def test_a_non_positive_pixel_size_is_refused(bad):
    with pytest.raises(ValueError, match="positive"):
        Micrograph(BW, pixel_size=bad)


# ---------------------------------------------------------------------------
# remove_info_bar=
# ---------------------------------------------------------------------------


def test_removing_the_bar_drops_46_rows_and_leaves_the_raw_frame():
    img = Micrograph(FIFTY, remove_info_bar=True)
    assert img.img.shape == (2002, 2448, 3)
    assert img.img_raw.shape == (2048, 2448, 3)
    np.testing.assert_array_equal(img.img, img.img_raw[:2002])


def test_removing_a_bar_that_is_not_there_is_refused():
    with pytest.raises(ValueError, match="no info bar"):
        Micrograph(BW, remove_info_bar=True)


# ---------------------------------------------------------------------------
# plot_micrograph
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_the_return_is_named_in_order_and_still_a_tuple():
    # Written out, so renaming or reordering a field has to be done here too.
    assert MicrographPlot._fields == ("fig", "ax", "im", "scalebar")
    res = plot_micrograph(Micrograph(FIFTY))
    assert type(res) is MicrographPlot
    fig, ax, im, scalebar = res
    assert (fig, ax, im, scalebar) == (res.fig, res.ax, res.im, res.scalebar)


def test_the_members_hold_what_their_names_say():
    res = plot_micrograph(Micrograph(FIFTY))
    assert isinstance(res.fig, Figure)
    assert isinstance(res.ax, Axes)
    assert isinstance(res.im, AxesImage)
    assert isinstance(res.scalebar, ScaleBar)
    assert res.scalebar in res.ax.artists


def test_the_scale_bar_uses_the_micrographs_pixel_size_in_um():
    res = plot_micrograph(Micrograph(FIFTY, magnification=100, label_reads=50))
    assert res.scalebar.dx == pytest.approx(25 / 363)
    assert res.scalebar.units == "um"


def test_scalebar_keywords_reach_the_scale_bar():
    res = plot_micrograph(Micrograph(FIFTY), location="lower right", fixed_value=10,
                          font_properties={"size": 24})
    # Stored as matplotlib's legend code: 4 is "lower right".
    assert res.scalebar.location == 4
    assert res.scalebar.fixed_value == 10


def test_the_image_drawn_is_img_so_a_removed_bar_stays_removed():
    res = plot_micrograph(Micrograph(FIFTY, remove_info_bar=True))
    assert res.im.get_array().shape == (2002, 2448, 3)


def test_the_axes_are_hidden():
    assert not plot_micrograph(Micrograph(FIFTY)).ax.axison


def test_no_scale_bar_gives_a_none_member():
    assert plot_micrograph(Micrograph(FIFTY), scalebar=False).scalebar is None


def test_scale_bar_keywords_without_a_scale_bar_are_refused():
    with pytest.raises(ValueError, match="no effect"):
        plot_micrograph(Micrograph(FIFTY), scalebar=False, location="lower right")


def test_an_unknown_pixel_size_refuses_a_scale_bar_but_still_plots_without():
    img = Micrograph(BW)
    with pytest.raises(ValueError, match="magnification= or pixel_size="):
        plot_micrograph(img)
    assert plot_micrograph(img, scalebar=False).scalebar is None


def test_drawing_into_given_axes():
    fig, ax = plt.subplots()
    res = plot_micrograph(Micrograph(FIFTY), ax=ax)
    assert res.ax is ax and res.fig is fig


# ---------------------------------------------------------------------------
# The shipped masks
# ---------------------------------------------------------------------------


def test_the_derived_lengths_are_the_measured_ones():
    """46 rows and 363 px are measured facts; the constants are derived from positions."""
    from leman.loaders import _MICROGRAPH_BAR_ROWS, _MICROGRAPH_SCALE_PX
    assert _MICROGRAPH_BAR_ROWS == 46
    assert _MICROGRAPH_SCALE_PX == 363


def test_the_cached_masks_cannot_be_changed():
    x_mask, label_masks = _micrograph_masks()
    with pytest.raises(ValueError):
        x_mask[0, 0] = True
    with pytest.raises(ValueError):
        label_masks[50][0, 0] = True
    with pytest.raises(TypeError):
        label_masks[40] = label_masks[50]


def test_shipped_masks_match_a_rebuild_from_the_committed_examples():
    """A stale .npz, or a box or threshold changed without re-running the script."""
    script = DATA.parent.parent / "dev" / "build_micrograph_masks.py"
    spec = importlib.util.spec_from_file_location("build_micrograph_masks", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuilt = module.build_masks()

    x_mask, label_masks = _micrograph_masks()
    np.testing.assert_array_equal(x_mask, rebuilt["x"])
    assert set(label_masks) == {int(k) for k in rebuilt if k != "x"}
    for m, mask in label_masks.items():
        np.testing.assert_array_equal(mask, rebuilt[str(m)])
