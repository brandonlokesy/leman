"""
Tests for ``plotting.plot_spectral_series``' nest pinning.

The series and the 2-D map resolve their spectra through one shared helper, so a
nest pinned for one is pinned the same way for the other.  These two tests pin
the series' half of that contract: an unpinned nest is refused by name, and a
held axis colours the lines by the axis left free rather than by the flat index.

Only the nest path is covered here.  The labels, the return shape and the
spectral-axis vocabulary have their own suites.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from leman import plotting
from leman.loaders import ACSpectralSweep

from test_loaders import make_spectral_csv
from test_loaders_nesting import N_FAST, RASTER, SLOW_VALUES


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def nested(tmp_path):
    """A raster declared on raw rows, N_FAST inside N_SLOW."""
    path = tmp_path / "raster.csv"
    make_spectral_csv(path, params=RASTER)
    return ACSpectralSweep(str(path), spectra_type="PL",
                                 fast_sweep="Scanner X",
                                 slow_sweep="Scanner Y")


def test_an_unpinned_nest_is_refused(nested):
    """Every line would be drawn, at settings no single coordinate names."""
    with pytest.raises(ValueError) as excinfo:
        plotting.plot_spectral_series(nested)

    message = str(excinfo.value)
    assert "plot_spectral_series()" in message
    for keyword in ("fast=", "index_fast=", "slow=", "index_slow="):
        assert keyword in message


def test_holding_the_slow_axis_colours_by_the_fast_one(nested):
    res = plotting.plot_spectral_series(nested, slow=SLOW_VALUES[1])

    assert len(res.lines) == N_FAST
    assert res.cb.ax.get_ylabel() == nested.nesting.fast_axis_label
    # The colour limits span the free axis, so the drawn lines use the whole
    # colormap — the flat index would run 0 to n_sweeps instead.
    assert np.allclose(res.cb.mappable.get_clim(),
                       (nested.nesting.fast_axis.min(),
                        nested.nesting.fast_axis.max()))
