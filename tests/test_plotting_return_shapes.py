"""
The four long plotting returns are named, and still tuples.

Nine plotting functions return ``fig, ax`` or ``fig, ax, artist``, which needs no
names. Four return more than that, and their members are reached by name through a
``NamedTuple`` rather than by counting positions.

Two properties have to hold together, and each is a way the change can go wrong:

* **Still a tuple.** Every existing caller unpacks positionally — the suite alone
  has around forty such lines, plus the README and the example notebooks. If a
  return stopped being a tuple, or changed length, all of them break at once.
* **Field order matches the return statement.** ``_fields`` being right does not
  prove the values are in the matching order: writing
  ``SpectralSeriesPlot(fig, ax, lines, cb, ax_twin)`` would keep the names correct
  and hand every caller the wrong two objects, in silence. So each member is
  checked by *what kind of thing* it holds, which is what pins the pairing.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from leman import plotting
from leman.loaders import ACSpectralSweep

from test_loaders import GATES, make_spectral_csv


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "sweep.csv"
    make_spectral_csv(path)
    return ACSpectralSweep(str(path), spectra_type="PL")


@pytest.fixture
def gated_scan(tmp_path):
    """A scan with the wiring declared, so ``plot_current`` has currents to draw."""
    path = tmp_path / "gated.csv"
    make_spectral_csv(path)
    return ACSpectralSweep(str(path), spectra_type="PL", gates=GATES)


@pytest.fixture
def image():
    return np.arange(16.0).reshape(4, 4)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# The names
# ---------------------------------------------------------------------------

# Written out rather than derived from the classes, so that renaming or reordering a
# field has to be done here too — which is the point of pinning it.
EXPECTED_FIELDS = {
    plotting.SpectrumPlot:       ("fig", "ax", "line", "ax_twin"),
    plotting.CurrentPlot:        ("fig", "ax_left", "ax_right", "lines"),
    plotting.ImagePlot:          ("fig", "ax", "im", "circle", "cb"),
    plotting.SpectralSeriesPlot: ("fig", "ax", "cb", "lines", "ax_twin"),
}


@pytest.mark.parametrize("cls", list(EXPECTED_FIELDS), ids=lambda c: c.__name__)
def test_every_long_return_names_its_members_in_order(cls):
    assert cls._fields == EXPECTED_FIELDS[cls]


@pytest.mark.parametrize("cls", list(EXPECTED_FIELDS), ids=lambda c: c.__name__)
def test_the_figure_comes_first(cls):
    """Every caller in the package unpacks ``fig`` first; nothing may displace it."""
    assert cls._fields[0] == "fig"


# ---------------------------------------------------------------------------
# Still a tuple, and the named class rather than a bare one
# ---------------------------------------------------------------------------


def test_a_spectrum_return_unpacks_positionally(scan):
    res = plotting.plot_spectrum(scan, index=0, twin_axis=True)
    assert isinstance(res, tuple)
    assert type(res) is plotting.SpectrumPlot
    fig, ax, line, ax_twin = res
    assert (fig, ax, line, ax_twin) == (res.fig, res.ax, res.line, res.ax_twin)


def test_a_series_return_unpacks_positionally(scan):
    res = plotting.plot_spectral_series(scan, twin_axis=True)
    assert isinstance(res, tuple)
    assert type(res) is plotting.SpectralSeriesPlot
    fig, ax, cb, lines, ax_twin = res
    assert (fig, ax, cb, lines, ax_twin) == (res.fig, res.ax, res.cb,
                                             res.lines, res.ax_twin)


def test_an_image_return_unpacks_positionally(image):
    res = plotting.plot_image(image)
    assert isinstance(res, tuple)
    assert type(res) is plotting.ImagePlot
    fig, ax, im, circle, cb = res
    assert (fig, ax, im, circle, cb) == (res.fig, res.ax, res.im,
                                         res.circle, res.cb)


def test_a_current_return_unpacks_positionally(gated_scan):
    res = plotting.plot_current(gated_scan)
    assert isinstance(res, tuple)
    assert type(res) is plotting.CurrentPlot
    fig, ax_left, ax_right, lines = res
    assert (fig, ax_left, ax_right, lines) == (res.fig, res.ax_left,
                                               res.ax_right, res.lines)


# ---------------------------------------------------------------------------
# Names land on the artist they promise
# ---------------------------------------------------------------------------


def test_the_series_members_hold_what_their_names_say(scan):
    res = plotting.plot_spectral_series(scan, twin_axis=True, colorbar=True)
    assert isinstance(res.fig, Figure)
    assert isinstance(res.ax, Axes)
    assert isinstance(res.cb, Colorbar)
    assert res.lines and all(isinstance(line, Line2D) for line in res.lines)
    assert res.ax_twin is not None and res.ax_twin.get_xlabel()


def test_the_spectrum_members_hold_what_their_names_say(scan):
    res = plotting.plot_spectrum(scan, index=0, twin_axis=True)
    assert isinstance(res.fig, Figure)
    assert isinstance(res.ax, Axes)
    assert isinstance(res.line, Line2D)
    assert res.ax_twin is not None and res.ax_twin.get_xlabel()


def test_the_image_members_hold_what_their_names_say(image):
    res = plotting.plot_image(image)
    assert isinstance(res.fig, Figure)
    assert isinstance(res.ax, Axes)
    assert isinstance(res.im, AxesImage)
    assert isinstance(res.cb, Colorbar)
    # Matplotlib's own back-reference must agree with the member: one colorbar,
    # reachable either way.
    assert res.im.colorbar is res.cb


def test_the_current_members_hold_what_their_names_say(gated_scan):
    res = plotting.plot_current(gated_scan)
    assert isinstance(res.fig, Figure)
    assert isinstance(res.ax_left, Axes) and isinstance(res.ax_right, Axes)
    assert res.ax_right is not res.ax_left
    assert res.lines and all(isinstance(line, Line2D) for line in res.lines)
    # lines holds the current traces only: the power trace was drawn on ax_right.
    assert all(line.axes is res.ax_left for line in res.lines)


# ---------------------------------------------------------------------------
# The optional members
# ---------------------------------------------------------------------------


def test_the_conjugate_axis_is_none_when_it_was_not_drawn(scan):
    assert plotting.plot_spectral_series(scan).ax_twin is None
    assert plotting.plot_spectrum(scan, index=0).ax_twin is None


def test_the_colorbar_is_none_when_it_was_not_drawn(scan, image):
    assert plotting.plot_spectral_series(scan, colorbar=False).cb is None
    assert plotting.plot_image(image, colorbar=False).cb is None


def test_the_laser_circle_is_none_when_it_was_not_drawn(image):
    assert plotting.plot_image(image).circle is None
    # A bare array carries no laser reference, so asking for the overlay draws
    # nothing rather than raising.
    assert plotting.plot_image(image, laser_annotation=True).circle is None


def test_the_shape_does_not_change_with_the_optional_members(scan):
    """The complaint that started this: one caller must not meet two shapes."""
    with_twin    = plotting.plot_spectral_series(scan, twin_axis=True)
    without_twin = plotting.plot_spectral_series(scan, twin_axis=False)
    assert len(with_twin) == len(without_twin) == 5
    assert type(with_twin) is type(without_twin) is plotting.SpectralSeriesPlot
