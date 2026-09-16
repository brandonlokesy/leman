"""
Tests for ``processing.remove_cosmic_rays``.

Synthetic PL spectra with known spikes planted at known pixels, so detection can
be checked exactly rather than approximately.  Three things are pinned here:

* a 3-pixel spike is found *in full*.  Its flat top has a near-zero Laplacian, so
  the interior pixel is only reachable after the edges have been replaced and the
  Laplacian recomputed — this is the regression test for the iteration.
* 2-D input behaves exactly like calling the function column by column, so
  results never depend on how spectra happened to be batched.
* ``cross_sweep_veto`` only ever removes detections, and the default warns rather
  than silently replacing a feature that recurs in every sweep.
"""

import warnings

import numpy as np
import pytest

from leman.processing import remove_cosmic_rays

N_PIX     = 400
N_SWEEPS  = 24
PEAK_CTR  = 200.0
PEAK_SIG  = 25.0            # broad PL peak: many pixels wide, must not be flagged
PEDESTAL  = 100.0
NOISE     = 3.0

CR_1PX    = 80              # pixel index of the single-pixel spike
CR_3PX    = slice(300, 303) # pixel indices of the three-pixel spike
CR_SWEEP1 = 5               # sweep carrying the 1-px spike, in 2-D cases
CR_SWEEP3 = 17              # sweep carrying the 3-px spike
HOT_PIXEL = 320             # narrow feature present in *every* sweep


def _spectrum(seed=0, amp=1500.0):
    """One PL spectrum: broad Gaussian peak on a dark-count pedestal."""
    x = np.arange(N_PIX)
    rng = np.random.default_rng(seed)
    peak = amp * np.exp(-((x - PEAK_CTR) / PEAK_SIG) ** 2)
    return peak + PEDESTAL + rng.normal(0, NOISE, N_PIX)


def _sweep(seed=0, persistent=False):
    """
    A gate sweep: the peak drifts and its intensity varies 10x across sweeps,
    which is what forces the noise estimate to be made per exposure.
    """
    x = np.arange(N_PIX)
    rng = np.random.default_rng(seed)
    out = np.empty((N_PIX, N_SWEEPS))
    for j in range(N_SWEEPS):
        amp = 200.0 + 1800.0 * j / (N_SWEEPS - 1)
        ctr = PEAK_CTR + 8.0 * j / (N_SWEEPS - 1)
        out[:, j] = amp * np.exp(-((x - ctr) / PEAK_SIG) ** 2) + PEDESTAL
    out += rng.normal(0, NOISE, out.shape)
    if persistent:
        # Indistinguishable from a hot pixel by construction — that is the point.
        out[HOT_PIXEL, :] += 900.0
    return out


def _plant_spikes(arr):
    """Add a 1-px and a 3-px cosmic ray. Returns ``(data, truth_mask)``."""
    data  = arr.copy()
    truth = np.zeros(arr.shape, dtype=bool)
    if arr.ndim == 1:
        data[CR_1PX] += 4000.0
        data[CR_3PX] += 3000.0
        truth[CR_1PX] = True
        truth[CR_3PX] = True
    else:
        data[CR_1PX, CR_SWEEP1] += 4000.0
        data[CR_3PX, CR_SWEEP3] += 3000.0
        truth[CR_1PX, CR_SWEEP1] = True
        truth[CR_3PX, CR_SWEEP3] = True
    return data, truth


# ---------------------------------------------------------------------------
# Detection on a single spectrum
# ---------------------------------------------------------------------------


def test_finds_exactly_the_planted_spikes():
    clean = _spectrum()
    data, truth = _plant_spikes(clean)
    cleaned, mask = remove_cosmic_rays(data)
    assert np.array_equal(mask, truth)                  # no misses, no false positives
    assert np.abs(cleaned - clean).max() < 5 * NOISE    # replaced with sane values


def test_three_pixel_spike_needs_iteration():
    """
    A flat-topped spike hides its interior from the Laplacian until the edges
    have been replaced.  One pass must therefore find strictly less than three.
    """
    data, _ = _plant_spikes(_spectrum())
    one_pass  = remove_cosmic_rays(data, max_iter=1)[1][CR_3PX]
    converged = remove_cosmic_rays(data, max_iter=3)[1][CR_3PX]
    assert converged.all()
    assert one_pass.sum() < converged.sum()


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6])
def test_spike_of_any_width_up_to_the_window_is_removed(width):
    """
    Every pixel of the spike is flagged and put back on the local baseline, for any
    width the default window can still find an unflagged neighbour for.

    The replacement median ignores the flagged pixels.  Were it taken over the whole
    window, it would be drawn from the spike itself as soon as half the window was
    flagged -- from *width* 4 at the default ``median_window=7`` -- and the plateau
    it produced would then survive every later pass.
    """
    clean = _spectrum()
    data  = clean.copy()
    spike = slice(300, 300 + width)
    data[spike] += 3000.0

    cleaned, mask = remove_cosmic_rays(data, max_iter=10)

    assert mask[spike].all()                                    # no interior left behind
    assert not mask[:295].any() and not mask[320:].any()        # and nothing else flagged
    assert np.abs(cleaned[spike] - clean[spike]).max() < 5 * NOISE


def _curved_spike(centre=304, width=9, amp=3000.0):
    """
    A cosmic ray *width* pixels wide with curvature at every pixel, so all of them
    are individually detectable.  A flat-topped spike hides its interior from the
    Laplacian instead, which is a different limit (see the module docstring).
    """
    x = np.arange(N_PIX)
    return amp * np.exp(-((x - centre) / (width / 4.5)) ** 2)


def test_run_wider_than_the_window_warns_and_keeps_its_raw_values():
    """
    A flagged run at least ``median_window`` wide leaves its centre with no
    unflagged neighbour, so no median exists for it.  Those pixels keep their raw
    values and the caller is told which, rather than the repair half-working with
    nothing said.
    """
    clean  = _spectrum()
    data   = clean + _curved_spike(width=9)      # 9 px, wider than median_window=7
    centre = slice(303, 306)

    with pytest.warns(UserWarning, match="median_window"):
        cleaned, mask = remove_cosmic_rays(data, median_window=7, max_iter=10)

    assert mask[300:309].all()                            # all nine detected
    assert np.array_equal(cleaned[centre], data[centre])  # centre could not be filled
    # The pixels that did have a clean neighbour were still repaired.
    assert np.abs(cleaned[300:303] - clean[300:303]).max() < 5 * NOISE


def test_a_wide_enough_window_repairs_the_same_run(recwarn):
    """The warning names a larger median_window as the remedy, so it has to work."""
    clean = _spectrum()
    data  = clean + _curved_spike(width=9)
    spike = slice(300, 309)

    cleaned, mask = remove_cosmic_rays(data, median_window=15, max_iter=10)

    assert mask[spike].all()
    assert np.abs(cleaned[spike] - clean[spike]).max() < 5 * NOISE
    assert not [w for w in recwarn if "median_window" in str(w.message)]


def test_broad_peak_and_noise_are_left_alone():
    """A real PL peak is many pixels wide and must survive untouched."""
    clean = _spectrum()
    cleaned, mask = remove_cosmic_rays(clean)
    assert not mask.any()
    assert np.array_equal(cleaned, clean)


def test_input_array_is_never_mutated():
    data, _ = _plant_spikes(_spectrum())
    frozen  = data.copy()
    remove_cosmic_rays(data)
    assert np.array_equal(data, frozen)


def test_integer_counts_are_accepted():
    data, truth = _plant_spikes(_spectrum())
    cleaned, mask = remove_cosmic_rays(data.astype(int))
    assert np.array_equal(mask, truth)
    assert cleaned.dtype == float


def test_even_median_window_is_forced_odd():
    data, _ = _plant_spikes(_spectrum())
    even = remove_cosmic_rays(data, median_window=6)[0]
    odd  = remove_cosmic_rays(data, median_window=7)[0]
    assert np.array_equal(even, odd)


# ---------------------------------------------------------------------------
# 2-D input: columns must stay independent
# ---------------------------------------------------------------------------


def test_2d_matches_column_by_column_1d():
    """Results must not depend on whether spectra are passed singly or batched."""
    data, _ = _plant_spikes(_sweep())
    cleaned, mask = remove_cosmic_rays(data)
    for j in range(N_SWEEPS):
        col_cleaned, col_mask = remove_cosmic_rays(data[:, j])
        assert np.array_equal(col_cleaned, cleaned[:, j])
        assert np.array_equal(col_mask, mask[:, j])


def test_2d_finds_spikes_in_the_right_sweeps():
    data, truth = _plant_spikes(_sweep())
    cleaned, mask = remove_cosmic_rays(data)
    assert np.array_equal(mask, truth)
    assert mask.shape == data.shape
    assert mask.dtype == bool


def test_axis_1_is_the_transpose_of_axis_0():
    data, _ = _plant_spikes(_sweep())
    cleaned_0, mask_0 = remove_cosmic_rays(data)
    cleaned_1, mask_1 = remove_cosmic_rays(data.T, axis=1)
    assert np.array_equal(cleaned_1, cleaned_0.T)
    assert np.array_equal(mask_1, mask_0.T)


def test_rejects_3d_input_and_bad_axis():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        remove_cosmic_rays(np.zeros((4, 4, 4)))
    with pytest.raises(ValueError, match="axis"):
        remove_cosmic_rays(_sweep(), axis=2)


# ---------------------------------------------------------------------------
# Cross-sweep veto
# ---------------------------------------------------------------------------


def test_veto_spares_a_feature_that_recurs_every_sweep():
    """
    A cosmic ray cannot recur at one pixel; a Raman line or a hot pixel does.
    The veto must keep the former and stop replacing the latter.
    """
    data, truth = _plant_spikes(_sweep(persistent=True))
    cleaned, mask = remove_cosmic_rays(data, cross_sweep_veto=True)
    assert np.array_equal(mask, truth)                        # only the real CRs
    assert not mask[HOT_PIXEL].any()
    assert np.array_equal(cleaned[HOT_PIXEL], data[HOT_PIXEL])  # left in the data


def test_default_replaces_the_recurring_feature_in_every_sweep():
    """The behaviour the veto exists to avoid — pinned so the contrast is explicit."""
    data, _ = _plant_spikes(_sweep(persistent=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned, mask = remove_cosmic_rays(data)
    assert mask[HOT_PIXEL].all()
    assert cleaned[HOT_PIXEL].max() < data[HOT_PIXEL].min()


def test_veto_only_ever_removes_detections():
    data, _ = _plant_spikes(_sweep(persistent=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = remove_cosmic_rays(data)[1]
    vetoed = remove_cosmic_rays(data, cross_sweep_veto=True)[1]
    assert not (vetoed & ~default).any()


def test_veto_needs_a_sweep_axis():
    with pytest.raises(ValueError, match="2-D"):
        remove_cosmic_rays(_spectrum(), cross_sweep_veto=True)
    with pytest.raises(ValueError, match="at least 3 sweeps"):
        remove_cosmic_rays(_sweep()[:, :2], cross_sweep_veto=True)


# ---------------------------------------------------------------------------
# The default must not damage data silently
# ---------------------------------------------------------------------------


def test_persistent_detections_warn_under_the_default():
    data, _ = _plant_spikes(_sweep(persistent=True))
    with pytest.warns(UserWarning, match=f"{HOT_PIXEL}"):
        remove_cosmic_rays(data)


def test_no_warning_when_nothing_recurs():
    data, _ = _plant_spikes(_sweep())
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # any warning fails the test
        remove_cosmic_rays(data)


def test_no_warning_when_the_veto_handles_it():
    data, _ = _plant_spikes(_sweep(persistent=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        remove_cosmic_rays(data, cross_sweep_veto=True)


def test_single_spectrum_does_not_warn():
    """A 1-D call has no sweep axis to judge recurrence against."""
    data, _ = _plant_spikes(_spectrum())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        remove_cosmic_rays(data)
