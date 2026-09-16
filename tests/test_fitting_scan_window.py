"""
``fit_scan_peak``'s spectral window — what it selects, and what it refuses.

The window is resolved by the same helper ``pixel_slice`` uses, so a window that
holds no pixel is refused before any fit runs and a bound past the end of the axis
warns rather than being clipped in silence. Two consequences are pinned here
because they are contracts rather than incidentals: the fitted ``x_fit`` is a
**view** into the scan's axis, and a reversed pair of bounds is the same window.

The fixture carries a synthetic Lorentzian rather than the suite's default counts.
The default ``roi1`` is a monotone ramp, so ``argmax`` lands on the last pixel and
nothing converges; every assertion here would then be about a failed fit. A little
deterministic noise is added on purpose — an exact fit gives a vanishing covariance,
and those σ are the WLS weights inside ``extract_dipole_length``.
"""

import warnings

import numpy as np
import pytest

from leman import fitting
from leman.loaders import (
    ACSpectralSweep,
    DeviceGeometry,
    StackLayer,
)

from test_loaders import make_spectral_csv

# 40 pixels at 1 nm steps: wide enough that a sub-window still holds ~10 points,
# and the arithmetic stays countable by hand.  780-819 nm is ~1.514-1.590 eV.
N_PX          = 40
WAVELENGTH_NM = 780.0 + np.arange(N_PX)
CENTRES_NM    = np.array([800.0, 800.5, 801.0])      # one small shift per sweep
AMPLITUDES    = np.array([1000.0, 1100.0, 1200.0])
FWHM_NM       = 5.0
PEDESTAL      = 50.0

# A window around the peak, in eV — 1.54-1.56 eV is ~795-805 nm.
WINDOW_EV = (1.54, 1.56)
GATES     = {"top": "V_A", "bottom": "V_B"}


def _lorentzian_sweeps() -> np.ndarray:
    """``(n_pixels, n_sweeps)`` counts: one Lorentzian per sweep point."""
    # (n_pixels, 1) detector axis broadcast against (1, n_sweeps) centres and
    # amplitudes: every sweep sits on the same axis with its own peak.
    half   = 0.5 * FWHM_NM
    prof   = 1.0 / (1.0 + ((WAVELENGTH_NM[:, None] - CENTRES_NM[None, :]) / half) ** 2)
    counts = PEDESTAL + AMPLITUDES[None, :] * prof
    return counts + np.random.default_rng(0).normal(0.0, 5.0, counts.shape)


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "peak.csv"
    make_spectral_csv(path, roi1=_lorentzian_sweeps(), wavelength=WAVELENGTH_NM)
    return ACSpectralSweep(str(path), spectra_type="PL")


@pytest.fixture
def field_scan(tmp_path):
    """The same spectra with a field axis, for the extract_dipole_length chain."""
    path = tmp_path / "peak_field.csv"
    make_spectral_csv(path, roi1=_lorentzian_sweeps(), wavelength=WAVELENGTH_NM)
    geometry = DeviceGeometry(
        tmdc_stack=[StackLayer("MoSe2"), StackLayer("WSe2")],
        d_hbn_top=50.0, d_hbn_bottom=50.0,
    )
    return ACSpectralSweep(str(path), spectra_type="PL",
                                 geometry=geometry, gates=GATES,
                                 sweep="electric_field")


def _clipping_warnings(records) -> list:
    """The window warnings among *records* — convergence ones can share the list."""
    return [r for r in records if "clipped" in str(r.message)]


# ---------------------------------------------------------------------------
# What the window selects
# ---------------------------------------------------------------------------


def test_a_window_fits_exactly_the_pixels_inside_it(scan):
    results = fitting.fit_scan_peak(scan, x_range=WINDOW_EV)
    inside  = (scan.energy >= WINDOW_EV[0]) & (scan.energy <= WINDOW_EV[1])

    for i, result in enumerate(results):
        assert np.array_equal(result.x_fit, scan.energy[inside])
        # The invariant the mask spelling used to provide: the same numbers as
        # fitting the hand-cut arrays.
        expected = fitting.fit_lorentzian(
            scan.energy[inside],
            scan.best_energy_spectra[inside, i].astype(float),
        )
        assert result.params == pytest.approx(expected.params, rel=1e-12)


def test_no_window_fits_the_whole_axis(scan):
    whole = fitting.fit_scan_peak(scan)
    assert whole[0].x_fit.shape == (scan.n_pixels,)

    spanned = fitting.fit_scan_peak(
        scan, x_range=(scan.energy.min(), scan.energy.max()))
    assert np.array_equal(whole[0].x_fit, spanned[0].x_fit)
    assert whole[0].params == pytest.approx(spanned[0].params, rel=1e-12)


def test_the_window_views_the_axis_rather_than_copying_it(scan):
    # The test that catches a regression to boolean masking, which copies.
    results = fitting.fit_scan_peak(scan, x_range=WINDOW_EV)
    assert np.shares_memory(results[0].x_fit, scan.energy)


def test_a_reversed_pair_is_the_same_window(scan):
    forward  = fitting.fit_scan_peak(scan, x_range=WINDOW_EV)
    backward = fitting.fit_scan_peak(scan, x_range=WINDOW_EV[::-1])
    assert np.array_equal(forward[0].x_fit, backward[0].x_fit)
    assert forward[0].params == pytest.approx(backward[0].params, rel=1e-12)


def test_a_window_is_read_on_the_axis_it_was_given_for(scan):
    window_nm = (795.0, 805.0)
    fitted    = fitting.fit_scan_peak(scan, x_axis="wavelength",
                                      x_range=window_nm)
    inside    = ((scan.wavelength >= window_nm[0])
                 & (scan.wavelength <= window_nm[1]))
    assert np.array_equal(fitted[0].x_fit, scan.wavelength[inside])

    # The same numbers as an energy window are nowhere near the energy axis, so
    # the units are not silently reinterpreted.
    with pytest.raises(ValueError, match="no point of the energy axis"):
        fitting.fit_scan_peak(scan, x_axis="energy", x_range=window_nm)


def test_placeholders_carry_the_window(scan):
    mask    = np.array([True, False, True])
    results = fitting.fit_scan_peak(scan, x_range=WINDOW_EV, sweep_mask=mask)

    assert len(results) == scan.n_sweeps
    skipped = results[1]
    assert not skipped.converged
    n_window = results[0].x_fit.size
    assert skipped.x_fit.size == n_window
    assert skipped.y_fit.size == n_window
    assert skipped.residuals.size == n_window


# ---------------------------------------------------------------------------
# Refusals and the warning
# ---------------------------------------------------------------------------


def test_an_empty_window_is_refused_and_names_the_axis(scan):
    # Previously this reached numpy as `y.max()` on a zero-size array, naming
    # neither x_range nor the axis, on the first sweep of the loop.
    with pytest.raises(ValueError) as excinfo:
        fitting.fit_scan_peak(scan, x_range=(1.0, 1.1))
    message = str(excinfo.value)
    assert "fit_scan_peak()" in message
    assert "no point of the energy axis" in message
    assert "spans" in message


def test_a_window_narrower_than_the_model_is_refused(scan):
    # Non-empty, so the window helper is satisfied; three points cannot carry the
    # four free parameters of a Lorentzian with a constant baseline. curve_fit
    # reports this as a TypeError, which the RuntimeError handler misses.
    with pytest.raises(ValueError) as excinfo:
        fitting.fit_scan_peak(scan, x_range=(scan.energy[0], scan.energy[2]))
    message = str(excinfo.value)
    assert "covers 3 points" in message
    assert "4 free parameters" in message


def test_the_minimum_follows_the_model_not_the_axis(scan):
    # The same three points are enough without the baseline term, which is what
    # makes this the model's minimum rather than a constant.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = fitting.fit_scan_peak(
            scan, x_range=(scan.energy[0], scan.energy[2]), baseline="none")
    assert results[0].x_fit.size == 3


def test_a_bound_beyond_the_axis_warns_and_still_fits(scan):
    with pytest.warns(UserWarning, match=r"fit_scan_peak\(\): lower bound"):
        results = fitting.fit_scan_peak(scan, x_range=(1.0, 1.56))
    # Clipped to the axis, not refused: the fit ran on what was there.
    assert results[0].x_fit[0] == pytest.approx(scan.energy.min())
    assert results[0].converged


def test_the_window_warning_points_at_the_caller(scan):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitting.fit_scan_peak(scan, x_range=(1.0, 1.56))
    clipping = _clipping_warnings(caught)
    assert len(clipping) == 1
    assert clipping[0].filename == __file__


def test_the_window_warning_points_at_the_caller_through_the_dipole_chain(field_scan):
    # Two frames deeper than the call above, which is why the depth is an argument
    # of the implementation rather than a constant.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitting.extract_dipole_length(field_scan, x_range=(1.0, 1.56))
    clipping = _clipping_warnings(caught)
    assert len(clipping) == 1
    assert clipping[0].filename == __file__
