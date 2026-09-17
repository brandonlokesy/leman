"""
Tests for the pseudo-Voigt peak fitting added to fitting.py: the
``voigt_approx``/``multi_voigt`` line-shape functions and the
``fit_voigt``/``fit_multi_voigt`` entry points.

No existing test file covers the peak-fitting family in this module
(``fit_lorentzian``, ``fit_gaussian``, ``fit_multi_lorentzian`` have none
either), so these are synthetic, from first principles: build a known
line shape, add a baseline, fit it back, and check the recovered
parameters -- not just that the call does not crash.
"""

import numpy as np
import pytest

from leman.fitting import (
    FitResult,
    classify_raman_layer,
    extract_fit_param_map,
    fit_multi_voigt,
    fit_raman_modes,
    fit_voigt,
    locate_residual_peak,
    lorentzian,
    multi_voigt,
    voigt_approx,
)

from _paths import DATA

RAMAN_DIR = str(DATA / "Raman")

X = np.linspace(0.0, 100.0, 1000)


# ---------------------------------------------------------------------------
# Line-shape functions
# ---------------------------------------------------------------------------


def test_voigt_approx_peaks_at_center_and_decays_away():
    y = voigt_approx(X, amplitude=10.0, center=50.0, fwhm_g=3.0, fwhm_l=3.0)
    assert X[np.argmax(y)] == pytest.approx(50.0, abs=0.2)
    # Exactly at the center (not just the nearest grid point), both the
    # Gaussian and Lorentzian components evaluate to 1.0, so the weighted
    # sum is exactly `amplitude` regardless of eta.
    assert voigt_approx(np.array([50.0]), 10.0, 50.0, 3.0, 3.0)[0] == pytest.approx(10.0)
    assert y[0] < 0.01 * y.max()
    assert y[-1] < 0.01 * y.max()


def test_voigt_approx_reduces_to_lorentzian_when_fwhm_g_is_negligible():
    # fwhm_g -> 0 should leave (almost) pure Lorentzian character.
    y_voigt = voigt_approx(X, amplitude=5.0, center=40.0, fwhm_g=1e-6, fwhm_l=6.0)
    y_lorentz = lorentzian(X, amplitude=5.0, center=40.0, fwhm=6.0)
    assert np.allclose(y_voigt, y_lorentz, atol=1e-2)


def test_multi_voigt_is_the_sum_of_its_components():
    params = (10.0, 30.0, 2.0, 2.0, 6.0, 70.0, 3.0, 1.0)
    y_multi = multi_voigt(X, *params)
    y_sum = (voigt_approx(X, *params[:4]) + voigt_approx(X, *params[4:]))
    assert np.allclose(y_multi, y_sum)


def test_multi_voigt_rejects_wrong_parameter_count():
    with pytest.raises(ValueError, match="4 parameters per peak"):
        multi_voigt(X, 1.0, 2.0, 3.0)  # 3 params -- not a multiple of 4


# ---------------------------------------------------------------------------
# fit_voigt -- single peak
# ---------------------------------------------------------------------------


def test_fit_voigt_recovers_known_peak_with_constant_baseline():
    true = dict(amplitude=8.0, center=52.0, fwhm_g=4.0, fwhm_l=2.5)
    offset = 3.0
    y = voigt_approx(X, **true) + offset

    result = fit_voigt(X, y, baseline="constant")

    assert result.converged
    assert result.params["center"] == pytest.approx(true["center"], abs=0.1)
    assert result.params["amplitude"] == pytest.approx(true["amplitude"], rel=0.05)
    assert result.params["offset"] == pytest.approx(offset, abs=0.2)
    assert result.r_squared > 0.999


def test_fit_voigt_without_baseline_matches_baseline_none():
    y = voigt_approx(X, amplitude=5.0, center=45.0, fwhm_g=2.0, fwhm_l=2.0)
    result = fit_voigt(X, y, baseline="none")
    assert result.converged
    assert "offset" not in result.params
    assert result.params["center"] == pytest.approx(45.0, abs=0.1)


def test_fit_voigt_p0_length_validation():
    y = voigt_approx(X, amplitude=5.0, center=45.0, fwhm_g=2.0, fwhm_l=2.0)
    with pytest.raises(ValueError, match="expected 4"):
        fit_voigt(X, y, p0=(1.0, 2.0, 3.0))  # missing fwhm_l


# ---------------------------------------------------------------------------
# fit_multi_voigt -- several peaks
# ---------------------------------------------------------------------------


def test_fit_multi_voigt_recovers_two_well_separated_peaks():
    true = [(12.0, 25.0, 2.0, 2.0), (7.0, 75.0, 3.0, 1.5)]
    y = multi_voigt(X, *[v for peak in true for v in peak])

    result = fit_multi_voigt(
        X, y, p0=[(10.0, 20.0, 2.0, 2.0), (6.0, 70.0, 2.0, 2.0)],
    )

    assert result.converged
    assert result.params["center_0"] == pytest.approx(25.0, abs=0.1)
    assert result.params["center_1"] == pytest.approx(75.0, abs=0.1)
    assert result.r_squared > 0.999


def test_fit_multi_voigt_with_constant_baseline():
    true = [(10.0, 30.0, 2.0, 2.0), (10.0, 60.0, 2.0, 2.0)]
    offset = 4.0
    y = multi_voigt(X, *[v for peak in true for v in peak]) + offset

    result = fit_multi_voigt(
        X, y, p0=[(8.0, 28.0, 2.0, 2.0), (8.0, 58.0, 2.0, 2.0)],
    )

    assert result.converged
    assert result.params["offset"] == pytest.approx(offset, abs=0.3)
    assert result.params["center_0"] == pytest.approx(30.0, abs=0.2)
    assert result.params["center_1"] == pytest.approx(60.0, abs=0.2)


def test_fit_multi_voigt_param_names_scale_with_peak_count():
    true = [(10.0, 20.0, 2.0, 2.0), (8.0, 50.0, 2.0, 2.0), (5.0, 80.0, 2.0, 2.0)]
    y = multi_voigt(X, *[v for peak in true for v in peak])

    result = fit_multi_voigt(X, y, p0=true)

    expected_keys = {
        "amp_0", "center_0", "fwhm_g_0", "fwhm_l_0",
        "amp_1", "center_1", "fwhm_g_1", "fwhm_l_1",
        "amp_2", "center_2", "fwhm_g_2", "fwhm_l_2",
        "offset",
    }
    assert set(result.params) == expected_keys
    assert result.params["center_2"] == pytest.approx(80.0, abs=0.2)


def test_fit_multi_voigt_auto_detects_peaks_without_p0():
    true = [(10.0, 25.0, 2.0, 2.0), (8.0, 75.0, 2.0, 2.0)]
    y = multi_voigt(X, *[v for peak in true for v in peak])

    result = fit_multi_voigt(X, y, baseline="none")

    centers = sorted(v for k, v in result.params.items() if k.startswith("center"))
    assert centers[0] == pytest.approx(25.0, abs=0.5)
    assert centers[1] == pytest.approx(75.0, abs=0.5)


# ---------------------------------------------------------------------------
# locate_residual_peak -- find an unmodelled peak's position from a fit's
# own leftover residual, rather than guessing (e.g. from literature)
# ---------------------------------------------------------------------------


def test_locate_residual_peak_finds_an_unmodelled_shoulder():
    # A dominant peak at 50 plus a much smaller, unmodelled one at 65 --
    # fitting only the dominant peak should leave a residual bump at 65.
    y = (voigt_approx(X, amplitude=50.0, center=50.0, fwhm_g=2.0, fwhm_l=2.0)
         + voigt_approx(X, amplitude=5.0, center=65.0, fwhm_g=3.0, fwhm_l=3.0))

    main_only = fit_voigt(X, y, p0=(50.0, 50.0, 2.0, 2.0), baseline="none")
    position, height = locate_residual_peak(main_only, search_range=(55.0, 90.0))

    assert position == pytest.approx(65.0, abs=1.0)
    assert height > 0


def test_locate_residual_peak_raises_outside_the_fit_range():
    y = voigt_approx(X, amplitude=10.0, center=50.0, fwhm_g=2.0, fwhm_l=2.0)
    result = fit_voigt(X, y, p0=(10.0, 50.0, 2.0, 2.0), baseline="none")
    with pytest.raises(ValueError, match="search_range"):
        locate_residual_peak(result, search_range=(500.0, 600.0))


def test_locate_residual_peak_ignores_negative_residuals():
    # Deliberately over-estimate the peak (too much amplitude) so the
    # residual is negative everywhere in range -- the "peak" found should
    # still be the least-negative point, not mistaken for a real feature,
    # and callers are expected to check `height > 0` themselves.
    true_amp = 10.0
    y = voigt_approx(X, amplitude=true_amp, center=50.0, fwhm_g=2.0, fwhm_l=2.0)
    over_fit_curve = voigt_approx(X, amplitude=true_amp * 2, center=50.0,
                                   fwhm_g=2.0, fwhm_l=2.0)

    from leman.fitting import FitResult
    fake_result = FitResult(
        params={}, errors={}, x_fit=X, y_fit=over_fit_curve,
        residuals=y - over_fit_curve, r_squared=0.0, model="fake",
    )
    position, height = locate_residual_peak(fake_result, search_range=(40.0, 60.0))
    assert height <= 0


# ---------------------------------------------------------------------------
# fit_raman_modes -- constants.RAMAN_MODES-driven fit, exercised here against
# WSe2 bilayer (E2g/A1g + 2LA(M) + B2g) and monolayer (E2g/A1g + 2LA(M), no
# B2g) -- packaged as a reusable fit rather than one-off notebook code
# ---------------------------------------------------------------------------

RAMAN_X = np.linspace(220.0, 340.0, 3000)


def test_fit_raman_modes_recovers_known_synthetic_bilayer_peaks():
    # Realistic amplitude ratios: 2LA(M) and B2g are both roughly 10x weaker
    # than the dominant E2g/A1g peak, matching the reference spectra.
    true_centers = {"e2g_a1g": 250.6, "shoulder": 258.7, "b2g": 309.2}
    y = (
        voigt_approx(RAMAN_X, 60000.0, true_centers["e2g_a1g"], 1.2, 1.3)
        + voigt_approx(RAMAN_X, 6000.0, true_centers["shoulder"], 3.5, 3.0)
        + voigt_approx(RAMAN_X, 2500.0, true_centers["b2g"], 1.0, 0.5)
        + 900.0  # constant baseline, as in the real exports
    )

    result = fit_raman_modes(RAMAN_X, y, material="WSe2", n_layers=2)

    assert result.converged
    assert result.r_squared > 0.99
    assert result.params["center_0"] == pytest.approx(true_centers["e2g_a1g"], abs=0.3)
    assert result.params["center_1"] == pytest.approx(true_centers["shoulder"], abs=0.5)
    assert result.params["center_2"] == pytest.approx(true_centers["b2g"], abs=0.3)


def test_fit_raman_modes_seed_override_follows_a_shifted_bilayer_e2g_a1g():
    # A strained sample whose E2g/A1g and B2g modes have shifted well outside
    # the default seeds' tolerance -- overriding the seeds should still find
    # the shoulder correctly via its own residual-based search.
    true_centers = {"e2g_a1g": 246.0, "shoulder": 254.5, "b2g": 305.0}
    y = (
        voigt_approx(RAMAN_X, 60000.0, true_centers["e2g_a1g"], 1.2, 1.3)
        + voigt_approx(RAMAN_X, 6000.0, true_centers["shoulder"], 3.5, 3.0)
        + voigt_approx(RAMAN_X, 2500.0, true_centers["b2g"], 1.0, 0.5)
        + 900.0
    )

    result = fit_raman_modes(
        RAMAN_X, y, material="WSe2", n_layers=2,
        seeds={"E2g/A1g": 246.0, "B2g": 305.0},
        shoulder_range=(248.0, 280.0),
    )

    assert result.converged
    assert result.params["center_0"] == pytest.approx(true_centers["e2g_a1g"], abs=0.3)
    assert result.params["center_1"] == pytest.approx(true_centers["shoulder"], abs=0.5)
    assert result.params["center_2"] == pytest.approx(true_centers["b2g"], abs=0.3)


@pytest.mark.parametrize("fname,expected", [
    ("unstrained_bilayer.txt",  (250.6, 258.5, 309.2)),
    ("strained_bilayer1.txt",   (250.6, 258.7, 309.1)),
    ("strained_bilayer2.txt",   (250.6, 258.8, 309.1)),
])
def test_fit_raman_modes_bilayer_on_the_real_reference_spectra(fname, expected):
    from leman.loaders import RamanSpectrum

    s = RamanSpectrum(f"{RAMAN_DIR}/{fname}")
    result = fit_raman_modes(s.shift, s.counts, material="WSe2", n_layers=2)

    assert result.converged
    assert result.r_squared > 0.95
    e2g_a1g, shoulder, b2g = expected
    assert result.params["center_0"] == pytest.approx(e2g_a1g, abs=0.5)
    assert result.params["center_1"] == pytest.approx(shoulder, abs=1.0)
    assert result.params["center_2"] == pytest.approx(b2g, abs=0.5)


def test_fit_raman_modes_recovers_known_synthetic_monolayer_peaks():
    true_centers = {"e2g_a1g": 250.1, "shoulder": 260.4}
    y = (
        voigt_approx(RAMAN_X, 27000.0, true_centers["e2g_a1g"], 1.2, 1.3)
        + voigt_approx(RAMAN_X, 1500.0, true_centers["shoulder"], 3.5, 3.0)
        + 500.0
    )

    result = fit_raman_modes(RAMAN_X, y, material="WSe2", n_layers=1)

    assert result.converged
    assert result.r_squared > 0.99
    assert result.params["center_0"] == pytest.approx(true_centers["e2g_a1g"], abs=0.3)
    assert result.params["center_1"] == pytest.approx(true_centers["shoulder"], abs=0.5)
    assert "amp_2" not in result.params  # no B2g component


@pytest.mark.parametrize("fname,expected", [
    ("unstrained_monolayer.txt", (250.1, 260.3)),
    ("strained_monolayer1.txt",  (250.1, 260.5)),
    ("strained_monolayer2.txt",  (250.1, 260.5)),
])
def test_fit_raman_modes_monolayer_on_the_real_reference_spectra(fname, expected):
    from leman.loaders import RamanSpectrum

    s = RamanSpectrum(f"{RAMAN_DIR}/{fname}")
    result = fit_raman_modes(s.shift, s.counts, material="WSe2", n_layers=1)

    assert result.converged
    assert result.r_squared > 0.95
    e2g_a1g, shoulder = expected
    assert result.params["center_0"] == pytest.approx(e2g_a1g, abs=0.5)
    assert result.params["center_1"] == pytest.approx(shoulder, abs=1.0)
    assert "amp_2" not in result.params


def test_fit_raman_modes_unknown_material_or_layer_count_raises():
    y = voigt_approx(RAMAN_X, 1000.0, 250.0, 2.0, 2.0)
    with pytest.raises(KeyError):
        fit_raman_modes(RAMAN_X, y, material="MoS2", n_layers=1)
    with pytest.raises(KeyError):
        fit_raman_modes(RAMAN_X, y, material="WSe2", n_layers=3)


def test_fit_raman_modes_survives_a_negative_shoulder_seed():
    # When the discovery fit over-predicts across the shoulder range,
    # locate_residual_peak returns a negative height.  Before the fix,
    # that negative seed violated _bounds' amplitude >= 0 and curve_fit
    # raised ValueError, which fit_multi_voigt did not catch.
    #
    # The optimizer is too flexible to produce this naturally on clean
    # synthetic data, so we patch locate_residual_peak to return a
    # negative height — the scenario the defect entry (A26) describes
    # happening on weak map pixels.
    from unittest.mock import patch

    y = (
        voigt_approx(RAMAN_X, 60000.0, 250.6, 1.2, 1.3)
        + voigt_approx(RAMAN_X, 6000.0, 258.7, 3.5, 3.0)
        + voigt_approx(RAMAN_X, 2500.0, 309.2, 1.0, 0.5)
        + 900.0
    )

    with patch("leman.fitting.locate_residual_peak", return_value=(265.0, -150.0)):
        with pytest.warns(UserWarning, match="non-positive height") as caught:
            result = fit_raman_modes(RAMAN_X, y, material="WSe2", n_layers=2)

    assert caught[0].filename == __file__
    assert isinstance(result, FitResult)


# ---------------------------------------------------------------------------
# classify_raman_layer
# ---------------------------------------------------------------------------


def test_classify_raman_layer_synthetic_bilayer_and_monolayer():
    baseline = 400.0
    bilayer_y = (
        voigt_approx(RAMAN_X, 60000.0, 250.6, 1.2, 1.3)
        + voigt_approx(RAMAN_X, 6000.0, 258.7, 3.5, 3.0)
        + voigt_approx(RAMAN_X, 2500.0, 309.2, 1.0, 0.5)
        + baseline
    )
    monolayer_y = (
        voigt_approx(RAMAN_X, 27000.0, 250.1, 1.2, 1.3)
        + voigt_approx(RAMAN_X, 1400.0, 260.4, 3.5, 3.0)
        + baseline
    )
    assert classify_raman_layer(RAMAN_X, bilayer_y, material="WSe2") == 2
    assert classify_raman_layer(RAMAN_X, monolayer_y, material="WSe2") == 1


def test_classify_raman_layer_uses_local_baseline_not_zero():
    # A high baseline alone (no real B2g peak) must not read as bilayer.
    y = voigt_approx(RAMAN_X, 27000.0, 250.1, 1.2, 1.3) + 5000.0
    assert classify_raman_layer(RAMAN_X, y, material="WSe2") == 1


@pytest.mark.parametrize("fname,expected", [
    ("unstrained_bilayer.txt", 2),
    ("strained_bilayer1.txt", 2),
    ("strained_bilayer2.txt", 2),
    ("unstrained_monolayer.txt", 1),
    ("strained_monolayer1.txt", 1),
    ("strained_monolayer2.txt", 1),
])
def test_classify_raman_layer_on_the_real_reference_spectra(fname, expected):
    from leman.loaders import RamanSpectrum

    s = RamanSpectrum(f"{RAMAN_DIR}/{fname}")
    assert classify_raman_layer(s.shift, s.counts, material="WSe2") == expected


def test_classify_raman_layer_on_every_pixel_of_the_real_map():
    from leman.loaders import RamanMap

    m = RamanMap(f"{RAMAN_DIR}/map2.txt")
    labels = {
        classify_raman_layer(m.shift, m.spectrum_at(ix, iy), material="WSe2")
        for ix in range(m.n_x) for iy in range(m.n_y)
    }
    # Every pixel classifies as one of the two -- no third label leaks through,
    # and both are actually present in this map (confirmed by inspection).
    assert labels == {1, 2}


def test_classify_raman_layer_unknown_material_raises():
    y = voigt_approx(RAMAN_X, 1000.0, 250.0, 2.0, 2.0)
    with pytest.raises(KeyError):
        classify_raman_layer(RAMAN_X, y, material="MoS2")


# ---------------------------------------------------------------------------
# extract_fit_param_map -- pull one fitted parameter into a 2-D array from a
# grid of FitResult, as used for RamanMap-style per-pixel fits
# ---------------------------------------------------------------------------


def _fake_result(params):
    return FitResult(params=params, errors={}, x_fit=np.array([]),
                      y_fit=np.array([]), residuals=np.array([]),
                      r_squared=1.0, model="fake")


def _results_grid(entries):
    """entries: 2-D nested list of params dicts -> object array of FitResult."""
    grid = np.empty((len(entries), len(entries[0])), dtype=object)
    for iy, row in enumerate(entries):
        for ix, params in enumerate(row):
            grid[iy, ix] = _fake_result(params)
    return grid


def test_extract_fit_param_map_reads_the_requested_key():
    grid = _results_grid([
        [{"center_0": 1.0}, {"center_0": 2.0}],
        [{"center_0": 3.0}, {"center_0": 4.0}],
    ])
    arr = extract_fit_param_map(grid, "center", 0)
    assert np.array_equal(arr, [[1.0, 2.0], [3.0, 4.0]])


def test_extract_fit_param_map_missing_key_is_nan():
    # A monolayer pixel's fit has no center_2 (B2g) at all.
    grid = _results_grid([
        [{"center_0": 1.0, "center_2": 9.0}, {"center_0": 2.0}],
    ])
    arr = extract_fit_param_map(grid, "center", 2)
    assert arr[0, 0] == pytest.approx(9.0)
    assert np.isnan(arr[0, 1])


def test_extract_fit_param_map_only_label_masks_regardless_of_key_presence():
    # Both pixels' fits happen to carry center_2, but only_label=2 should
    # still blank the pixel labelled 1 -- the mask is about what the mode
    # was fit *for*, not merely whether the key exists.
    grid = _results_grid([[{"center_2": 9.0}, {"center_2": 8.0}]])
    label_grid = np.array([[1, 2]])
    arr = extract_fit_param_map(grid, "center", 2, label_grid=label_grid, only_label=2)
    assert np.isnan(arr[0, 0])
    assert arr[0, 1] == pytest.approx(8.0)
