"""
Tests for the refusal of ``clim`` together with ``rescale_img``.

One rule across two functions, so the tests are parametrised over both rather
than split by file.  ``clim`` is read in the data's own units and the rescale
remaps the values to [0, 1] before the colour scale is applied, so limits in
those units fall outside the rescaled range and every cell draws as one colour.
The figure is the failure and not the argument: a flat panel reads as a
measurement that went wrong, so it is refused rather than warned about.

The guard is also checked to run before the figure is made, and each function
with each argument alone, so it cannot be satisfied by refusing more than it
was meant to.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from leman import plotting
from leman.loaders import ACSpectralSweep

from test_loaders import make_spectral_csv

IMAGE = np.arange(16.0).reshape(4, 4)
CLIM = (500.0, 1500.0)


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path)
    return ACSpectralSweep(str(path), spectra_type="PL")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# Both adapters take the scan so one parametrisation covers both functions;
# plot_image draws a plain array and ignores it.
def _map_call(scan, **kwargs):
    return plotting.plot_spectral_map(scan, **kwargs)[2]        # the mesh


def _image_call(scan, **kwargs):
    return plotting.plot_image(IMAGE, **kwargs).im             # the image


CALLS = [
    pytest.param(_map_call,   "plot_spectral_map()", id="plot_spectral_map"),
    pytest.param(_image_call, "plot_image()",        id="plot_image"),
]


@pytest.mark.parametrize("call, what", CALLS)
def test_the_pair_is_refused(scan, call, what):
    with pytest.raises(ValueError) as excinfo:
        call(scan, clim=CLIM, rescale_img=True)

    message = str(excinfo.value)
    assert what in message
    for keyword in ("clim=", "rescale_img="):
        assert keyword in message


@pytest.mark.parametrize("call, what", CALLS)
def test_a_refusal_leaves_no_figure_behind(scan, call, what):
    """The guard runs before the figure is made, so nothing needs closing."""
    before = len(plt.get_fignums())

    with pytest.raises(ValueError):
        call(scan, clim=CLIM, rescale_img=True)

    assert len(plt.get_fignums()) == before


@pytest.mark.parametrize("call, what", CALLS)
def test_clim_alone_still_sets_the_limits(scan, call, what):
    assert call(scan, clim=CLIM).get_clim() == CLIM


@pytest.mark.parametrize("call, what", CALLS)
def test_rescale_alone_still_works_and_marks_the_colour_bar(scan, call, what):
    artist = call(scan, rescale_img=True)

    assert artist.get_clim() == (0.0, 1.0)
    assert "norm." in artist.colorbar.ax.get_ylabel()
