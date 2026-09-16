"""
Tests for ``pixel_slice`` — a spectral window as a position, not as data.

Covers the two axes and the index reversal between them, the inclusive
order-insensitive bounds, that the result views rather than copies, the clipping
warning and where it points, and the four refusals.

The fixture's axis is the suite's own 10-pixel 800–809 nm ramp: small, evenly
spaced and ascending, so every expected slice is countable by hand.
"""

import warnings

import numpy as np
import pytest

from leman.constants import HC_EV_NM
from leman.loaders import ACSpectralSweep

from test_loaders import make_spectral_csv, WAVELENGTH, N_PIXELS


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "sweep.csv"
    make_spectral_csv(path)
    return ACSpectralSweep(str(path), spectra_type="PL")


@pytest.fixture
def unordered(tmp_path):
    """A scan whose wavelength axis is out of order, which no export writes."""
    wl = WAVELENGTH.copy()
    wl[2], wl[9] = wl[9], wl[2]          # 800, 801, 809, 803, ..., 808, 802
    path = tmp_path / "unordered.csv"
    make_spectral_csv(path, wavelength=wl)
    return ACSpectralSweep(str(path), spectra_type="PL")


# ---------------------------------------------------------------------------
# What it selects
# ---------------------------------------------------------------------------


def test_bounds_are_inclusive(scan):
    # 802, 803, 804, 805 nm are pixels 2 to 5 of an ascending 800-809 ramp.
    assert scan.pixel_slice((802.0, 805.0), x_axis="wavelength") == slice(2, 6)


def test_a_reversed_pair_is_the_same_window(scan):
    assert (scan.pixel_slice((805.0, 802.0), x_axis="wavelength")
            == scan.pixel_slice((802.0, 805.0), x_axis="wavelength"))


def test_the_whole_axis_comes_back_whole(scan):
    assert (scan.pixel_slice((scan.wavelength.min(), scan.wavelength.max()),
                             x_axis="wavelength")
            == slice(0, N_PIXELS))


def test_the_selected_points_are_exactly_those_inside_the_window(scan):
    window = (1.535, 1.545)
    px     = scan.pixel_slice(window)
    inside = (scan.energy >= window[0]) & (scan.energy <= window[1])
    # The slice and the mask select the same pixels — the slice being the form
    # that views rather than copies.
    assert np.array_equal(np.flatnonzero(inside), np.arange(N_PIXELS)[px])


def test_the_two_axes_select_the_same_pixels_at_reversed_positions(scan):
    # Energy is stored ascending and wavelength ascending, so pixel i of one is
    # pixel n-1-i of the other: the same physical window is a different slice.
    window_nm = (802.0, 805.0)
    wl_px = scan.pixel_slice(window_nm, x_axis="wavelength")
    assert wl_px != slice(0, N_PIXELS)                  # a genuine sub-window

    # The same window in eV.  Converted with the package's own hc so that the two
    # bounds land on the boundary pixels exactly rather than a rounding away.
    en_px = scan.pixel_slice(tuple(sorted(HC_EV_NM / np.array(window_nm))))
    assert en_px == slice(N_PIXELS - wl_px.stop, N_PIXELS - wl_px.start)
    assert np.allclose(np.sort(scan.wavelength[wl_px]),
                       np.sort(HC_EV_NM / scan.energy[en_px]))


# ---------------------------------------------------------------------------
# What it returns
# ---------------------------------------------------------------------------


def test_slicing_with_it_views_both_the_axis_and_the_spectra(scan):
    px = scan.pixel_slice((1.535, 1.545))
    x  = scan.energy[px]
    y  = scan.get_spectrum_at(value=1.0)[px]
    assert np.shares_memory(x, scan.energy)
    assert np.shares_memory(y, scan.best_energy_spectra)
    assert x.shape == y.shape                # one slice cuts both to one length


def test_it_indexes_the_pixel_axis_of_the_full_spectra_array(scan):
    px = scan.pixel_slice((802.0, 805.0), x_axis="wavelength")
    assert scan.spectra[px].shape == (4, scan.n_sweeps)
    assert np.array_equal(scan.spectra[px, 0], scan.spectra[2:6, 0])


# ---------------------------------------------------------------------------
# The clipping warning
# ---------------------------------------------------------------------------


def test_a_bound_beyond_the_axis_warns_and_clips(scan):
    with pytest.warns(UserWarning, match="lower bound 700 nm is below"):
        px = scan.pixel_slice((700.0, 805.0), x_axis="wavelength")
    assert px == slice(0, 6)


def test_both_bounds_beyond_the_axis_are_named_in_one_warning(scan):
    with pytest.warns(UserWarning) as caught:
        px = scan.pixel_slice((700.0, 900.0), x_axis="wavelength")
    assert px == slice(0, N_PIXELS)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "lower bound 700 nm" in message and "upper bound 900 nm" in message


def test_a_bound_inside_half_a_pixel_of_the_end_is_silent(scan):
    # Half of the 1 nm step: 799.6 is a rounded-off end point, not a window that
    # was cut down.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (scan.pixel_slice((799.6, 802.0), x_axis="wavelength")
                == slice(0, 3))


def test_the_warning_points_at_the_caller(scan):
    # stacklevel is measured, not read off the def lines: the frame count is
    # what decides whether a researcher sees their own line or a library one.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan.pixel_slice((700.0, 805.0), x_axis="wavelength")
    assert len(caught) == 1
    assert caught[0].filename == __file__


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_empty_window_raises_and_gives_the_span(scan):
    with pytest.raises(ValueError, match=r"no point of the wavelength axis"):
        scan.pixel_slice((700.0, 750.0), x_axis="wavelength")
    with pytest.raises(ValueError, match=r"spans 800–809 nm in 10 points"):
        scan.pixel_slice((700.0, 750.0), x_axis="wavelength")


def test_an_unordered_axis_raises_rather_than_widening_the_window(unordered):
    # Pixels 0, 1, 3 and 4 hold 800, 801, 803 and 804 nm, with 809 between them,
    # so the window is not one run of pixels and is not a slice.
    with pytest.raises(ValueError, match="not ordered"):
        unordered.pixel_slice((800.0, 804.0), x_axis="wavelength")


def test_an_unordered_axis_still_serves_a_contiguous_window(unordered):
    assert unordered.pixel_slice((803.0, 805.0), x_axis="wavelength") == slice(3, 6)


@pytest.mark.parametrize("x_range", [1.6, (1.6,), (1.6, 1.7, 1.8), "1.6-1.7"])
def test_x_range_must_be_a_pair(scan, x_range):
    with pytest.raises(TypeError, match=r"x_range must be a \(lo, hi\) pair"):
        scan.pixel_slice(x_range)


@pytest.mark.parametrize("x_range", [(np.nan, 1.7), (1.6, np.inf)])
def test_a_non_finite_bound_raises(scan, x_range):
    with pytest.raises(ValueError, match="must both be finite"):
        scan.pixel_slice(x_range)


def test_an_unknown_x_axis_raises(scan):
    with pytest.raises(ValueError, match="must be 'energy' or 'wavelength'"):
        scan.pixel_slice((1.6, 1.7), x_axis="eV")
