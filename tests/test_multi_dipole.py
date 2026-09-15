# tests/test_multi_dipole.py
"""Tests for multi-dipole peak tracking and extraction."""

from dataclasses import dataclass

import numpy as np
import pytest

from tmdc_optics_tools.fitting import (
    DipoleResult,
    MultiDipoleResult,
    PeakTrack,
    _dipole_bootstrap,
    _dipole_wls,
    _resolve_sweep_ranges,
    extract_dipole_lengths,
    track_peak_energies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lorentzian(x, amp, center, fwhm):
    gamma = fwhm / 2
    return amp * gamma**2 / ((x - center)**2 + gamma**2)


@dataclass
class _MockScan:
    """Lightweight scan stand-in for testing peak tracking."""
    energy              : np.ndarray
    best_energy_spectra : np.ndarray
    ef                  : np.ndarray

    @property
    def n_sweeps(self):
        return self.best_energy_spectra.shape[1]

    @property
    def sweep_axis(self):
        if self.ef is not None:
            return self.ef
        return np.arange(self.n_sweeps, dtype=float)

    @property
    def sweep_label(self):
        return r"$E_F$" if self.ef is not None else "Sweep index"

    @property
    def sweep_unit(self):
        return "mV/nm" if self.ef is not None else ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_piecewise_scan(
    n_pixels=200,
    n_sweeps=60,
    energy_range=(1.40, 1.60),
    ef_range=(-30.0, 30.0),
    amp=100.0,
    fwhm=0.01,
):
    """
    Build a mock scan with 3 piecewise-linear Stark-shift regimes.

    Segment A : ef in [-30, -5)  slope = -3.0e-4 eV/(mV/nm)  d = 0.30 nm
    Segment B : ef in [-5, +5]   slope = -6.5e-4 eV/(mV/nm)  d = 0.65 nm
    Segment C : ef in (+5, +30]  slope = -3.0e-4 eV/(mV/nm)  d = 0.30 nm

    Intercept chosen so the peak is centred around 1.50 eV.
    """
    energy = np.linspace(*energy_range, n_pixels)
    ef     = np.linspace(*ef_range, n_sweeps)

    # Piecewise peak centres: E(F) = slope * F + intercept.
    slope_a, slope_c = -3.0e-4, -3.0e-4
    slope_b          = -6.5e-4

    # Continuity at the boundaries: match at ef = -5 and ef = +5.
    # Choose intercept_b so peak is ~1.50 eV at ef=0.
    intercept_b = 1.50
    intercept_a = intercept_b + (slope_b - slope_a) * (-5.0)
    intercept_c = intercept_b + (slope_b - slope_c) * 5.0

    centres = np.empty(n_sweeps)
    for k, f in enumerate(ef):
        if f <= -5.0:
            centres[k] = slope_a * f + intercept_a
        elif f <= 5.0:
            centres[k] = slope_b * f + intercept_b
        else:
            centres[k] = slope_c * f + intercept_c

    # Build spectra: one Lorentzian per sweep column.
    spectra = np.empty((n_pixels, n_sweeps))
    for k in range(n_sweeps):
        spectra[:, k] = _lorentzian(energy, amp, centres[k], fwhm)

    return _MockScan(energy=energy, best_energy_spectra=spectra, ef=ef), centres


@pytest.fixture
def piecewise_scan():
    scan, centres = _make_piecewise_scan()
    return scan, centres


# ---------------------------------------------------------------------------
# track_peak_energies
# ---------------------------------------------------------------------------

class TestTrackPeakEnergies:

    def test_argmax_recovers_peaks(self, piecewise_scan):
        scan, centres = piecewise_scan
        track = track_peak_energies(scan)

        pixel_step = float(np.median(np.diff(scan.energy)))
        assert track.peak_energies == pytest.approx(centres, abs=pixel_step)
        assert track.method == "argmax"
        assert track.converged.all()
        assert track.sweep_values is not None
        assert track.sweep_label == r"$E_F$"
        assert track.sweep_unit == "mV/nm"

    def test_argmax_with_x_range(self, piecewise_scan):
        scan, centres = piecewise_scan
        track = track_peak_energies(scan, x_range=(1.45, 1.55))

        pixel_step = float(np.median(np.diff(scan.energy)))
        assert track.peak_energies == pytest.approx(centres, abs=pixel_step)

    def test_x_range_empty_raises(self, piecewise_scan):
        scan, _ = piecewise_scan
        with pytest.raises(ValueError, match="selects no pixel"):
            track_peak_energies(scan, x_range=(2.0, 2.5))

    def test_unknown_method_raises(self, piecewise_scan):
        scan, _ = piecewise_scan
        with pytest.raises(ValueError, match="not recognised"):
            track_peak_energies(scan, method="magic")

    def test_fit_method_not_implemented(self, piecewise_scan):
        scan, _ = piecewise_scan
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            track_peak_energies(scan, method="fit")

    def test_scan_without_ef(self):
        energy = np.linspace(1.4, 1.6, 50)
        spectra = np.random.default_rng(0).random((50, 10))
        scan = _MockScan(energy=energy, best_energy_spectra=spectra, ef=None)
        track = track_peak_energies(scan)
        assert len(track.peak_energies) == 10
        assert len(track.sweep_values) == 10
        assert track.sweep_unit == ""

    def test_missing_attribute_raises(self):
        with pytest.raises(ValueError, match="no .* attribute"):
            track_peak_energies("not a scan")

    def test_peak_amplitudes_match_spectra(self, piecewise_scan):
        scan, _ = piecewise_scan
        track = track_peak_energies(scan)
        spectra = scan.best_energy_spectra
        idx = np.argmax(spectra, axis=0)
        expected = spectra[idx, np.arange(spectra.shape[1])]
        np.testing.assert_array_equal(track.peak_amplitudes, expected)


# ---------------------------------------------------------------------------
# _resolve_sweep_ranges
# ---------------------------------------------------------------------------

class TestResolveSweepRanges:

    def test_coord_ranges(self):
        sv = np.linspace(-30, 30, 60)
        masks = _resolve_sweep_ranges(
            sv, coord_ranges=[(-30, -5), (5, 30)],
            index_ranges=None, n_sweeps=60,
        )
        assert len(masks) == 2
        assert all(m.dtype == bool for m in masks)
        for m in masks:
            assert m.sum() >= 2

    def test_index_ranges(self):
        sv = np.linspace(-30, 30, 60)
        masks = _resolve_sweep_ranges(
            sv, coord_ranges=None,
            index_ranges=[(0, 20), (40, 59)], n_sweeps=60,
        )
        assert len(masks) == 2
        assert masks[0].sum() == 21
        assert masks[1].sum() == 20

    def test_empty_range_raises(self):
        sv = np.linspace(-30, 30, 60)
        with pytest.raises(ValueError, match="selects no sweep point"):
            _resolve_sweep_ranges(
                sv, coord_ranges=[(100, 200)],
                index_ranges=None, n_sweeps=60,
            )

    def test_too_few_points_raises(self):
        sv = np.array([0.0, 10.0, 20.0])
        with pytest.raises(ValueError, match="only 1 point"):
            _resolve_sweep_ranges(
                sv, coord_ranges=[(9.5, 10.5)],
                index_ranges=None, n_sweeps=3,
            )

    def test_mutual_exclusion_both(self):
        sv = np.linspace(-10, 10, 20)
        with pytest.raises(ValueError, match="exactly one"):
            _resolve_sweep_ranges(
                sv, coord_ranges=[(-5, 5)],
                index_ranges=[(0, 10)], n_sweeps=20,
            )

    def test_mutual_exclusion_neither(self):
        sv = np.linspace(-10, 10, 20)
        with pytest.raises(ValueError, match="exactly one"):
            _resolve_sweep_ranges(
                sv, coord_ranges=None,
                index_ranges=None, n_sweeps=20,
            )

    def test_overlap_warns(self):
        sv = np.linspace(-30, 30, 60)
        with pytest.warns(UserWarning, match="overlap"):
            _resolve_sweep_ranges(
                sv, coord_ranges=[(-10, 5), (0, 10)],
                index_ranges=None, n_sweeps=60,
            )

    def test_boundary_snap_warns(self):
        # Steps of 10, travel 100.  Threshold = max(0.5 * 10, 0.001 * 100) = 5.
        # Asking for 50 as a lower bound when the nearest inside is 90 → 40 away.
        sv = np.array([0.0, 10.0, 90.0, 100.0])
        with pytest.warns(UserWarning, match="away"):
            _resolve_sweep_ranges(
                sv, coord_ranges=[(50.0, 100.0)],
                index_ranges=None, n_sweeps=4,
            )

    def test_compound_index_range(self):
        sv = np.linspace(-30, 30, 60)
        # Second element is a compound range: two sub-ranges OR'd together.
        masks = _resolve_sweep_ranges(
            sv, coord_ranges=None,
            index_ranges=[(0, 10), [(20, 25), (30, 35)], (50, 59)],
            n_sweeps=60,
        )
        assert len(masks) == 3
        assert masks[0].sum() == 11
        # Compound: 6 + 6 = 12 points, with index 26-29 excluded.
        assert masks[1].sum() == 12
        assert not masks[1][26]
        assert masks[1][20] and masks[1][25] and masks[1][30] and masks[1][35]

    def test_compound_coord_range(self):
        sv = np.linspace(-30, 30, 61)  # step = 1
        masks = _resolve_sweep_ranges(
            sv, coord_ranges=[(-30, -20), [(-10, -5), (5, 10)]],
            index_ranges=None, n_sweeps=61,
        )
        assert len(masks) == 2
        # Compound: points at -10...-5 and 5...10, skipping -4...4.
        assert not masks[1][30]  # sv=0 is excluded

    def test_index_out_of_bounds_raises(self):
        sv = np.linspace(-10, 10, 20)
        with pytest.raises(ValueError, match="out of bounds"):
            _resolve_sweep_ranges(
                sv, coord_ranges=None,
                index_ranges=[(0, 25)], n_sweeps=20,
            )

    def test_unit_appears_in_snap_warning(self):
        sv = np.array([0.0, 10.0, 90.0, 100.0])
        with pytest.warns(UserWarning, match="mV/nm"):
            _resolve_sweep_ranges(
                sv, coord_ranges=[(50.0, 100.0)],
                index_ranges=None, n_sweeps=4,
                unit="mV/nm", param_name="ef_ranges",
            )

    def test_param_name_appears_in_error(self):
        sv = np.linspace(-30, 30, 60)
        with pytest.raises(ValueError, match="ef_ranges"):
            _resolve_sweep_ranges(
                sv, coord_ranges=[(100, 200)],
                index_ranges=None, n_sweeps=60,
                param_name="ef_ranges",
            )


# ---------------------------------------------------------------------------
# All-NaN sigma handling in fitters
# ---------------------------------------------------------------------------

class TestAllNanSigma:

    def test_wls_all_nan_gives_finite_errors(self):
        rng = np.random.default_rng(42)
        ef  = np.linspace(-20, 20, 30)
        E   = -3e-4 * ef + 1.5 + rng.normal(0, 1e-4, 30)
        sig = np.full(30, np.nan)

        slope, slope_err, intercept, intercept_err = _dipole_wls(ef, E, sig)
        assert np.isfinite(slope)
        assert np.isfinite(slope_err)
        assert slope_err > 0
        assert np.isfinite(intercept)
        assert np.isfinite(intercept_err)

    def test_bootstrap_all_nan_gives_finite_errors(self):
        rng_data = np.random.default_rng(42)
        ef  = np.linspace(-20, 20, 30)
        E   = -3e-4 * ef + 1.5 + rng_data.normal(0, 1e-4, 30)
        sig = np.full(30, np.nan)

        slope, slope_err, intercept, intercept_err = _dipole_bootstrap(
            ef, E, sig, n_bootstrap=500, rng=np.random.default_rng(0),
        )
        assert np.isfinite(slope)
        assert np.isfinite(slope_err)
        assert slope_err > 0


# ---------------------------------------------------------------------------
# extract_dipole_lengths
# ---------------------------------------------------------------------------

class TestExtractDipoleLengths:

    def test_recovers_known_slopes(self, piecewise_scan):
        scan, _ = piecewise_scan
        result = extract_dipole_lengths(
            scan,
            ef_ranges=[(-30, -5), (-5, 5), (5, 30)],
        )

        assert isinstance(result, MultiDipoleResult)
        assert len(result.segments) == 3

        # Segment A and C: d ≈ 0.30 nm.
        assert result.segments[0].dipole_length == pytest.approx(0.30, rel=0.1)
        assert result.segments[2].dipole_length == pytest.approx(0.30, rel=0.1)
        # Segment B: d ≈ 0.65 nm.
        assert result.segments[1].dipole_length == pytest.approx(0.65, rel=0.1)

    def test_from_peak_track(self, piecewise_scan):
        scan, _ = piecewise_scan
        track = track_peak_energies(scan)
        result = extract_dipole_lengths(
            track,
            ef_ranges=[(-30, -5), (-5, 5), (5, 30)],
        )
        assert isinstance(result, MultiDipoleResult)
        assert result.track is track

    def test_index_ranges_work(self, piecewise_scan):
        scan, _ = piecewise_scan
        result = extract_dipole_lengths(
            scan,
            index_ranges=[(0, 24), (25, 34), (35, 59)],
        )
        assert len(result.segments) == 3
        assert result.index_ranges is not None
        assert result.ef_ranges is None

    def test_non_ef_sweep_raises(self):
        track = PeakTrack(
            sweep_values=np.linspace(0, 100, 10),
            sweep_label="Power",
            sweep_unit="µW",
            peak_energies=np.linspace(1.5, 1.6, 10),
            peak_amplitudes=np.ones(10),
            peak_errors=np.full(10, np.nan),
            converged=np.ones(10, dtype=bool),
            method="argmax",
        )
        with pytest.raises(ValueError, match="electric-field sweep"):
            extract_dipole_lengths(track, ef_ranges=[(0, 50)])

    def test_partial_failure_warns(self):
        # NaN peak energies in the first segment cause curve_fit to fail.
        track = PeakTrack(
            sweep_values=np.array([1.0, 2.0, 5.0, 10.0, 15.0]),
            sweep_label=r"$E_F$",
            sweep_unit="mV/nm",
            peak_energies=np.array([np.nan, np.nan, 1.51, 1.52, 1.53]),
            peak_amplitudes=np.array([np.nan, np.nan, 100.0, 110.0, 120.0]),
            peak_errors=np.full(5, np.nan),
            converged=np.ones(5, dtype=bool),
            method="argmax",
        )
        with pytest.warns(UserWarning, match="failed"):
            result = extract_dipole_lengths(
                track,
                index_ranges=[(0, 1), (2, 4)],
            )
        assert np.isnan(result.segments[0].dipole_length)
        assert np.isfinite(result.segments[1].dipole_length)

    def test_bad_fit_method_raises(self, piecewise_scan):
        scan, _ = piecewise_scan
        with pytest.raises(ValueError, match="not recognised"):
            extract_dipole_lengths(
                scan, ef_ranges=[(-10, 10)], fit_method="magic",
            )

    def test_each_segment_is_dipole_result(self, piecewise_scan):
        scan, _ = piecewise_scan
        result = extract_dipole_lengths(
            scan, ef_ranges=[(-30, -5), (5, 30)],
        )
        for seg in result.segments:
            assert isinstance(seg, DipoleResult)
            assert len(seg.ef) == scan.n_sweeps
            assert len(seg.converged_mask) == scan.n_sweeps


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:

    def test_peak_track_repr(self, piecewise_scan):
        scan, _ = piecewise_scan
        track = track_peak_energies(scan)
        r = repr(track)
        assert "argmax" in r
        assert "60" in r  # sweep count
        assert "eV" in r
        assert "mV/nm" in r

    def test_multi_dipole_result_repr(self, piecewise_scan):
        scan, _ = piecewise_scan
        result = extract_dipole_lengths(
            scan, ef_ranges=[(-30, -5), (5, 30)],
        )
        r = repr(result)
        assert "2 segments" in r
        assert "nm" in r
