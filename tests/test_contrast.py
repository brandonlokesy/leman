"""
Tests for reference-spectrum contrast: the pure maths in `processing` and the
`bg_spectrum=` / `reference=` wiring on ACSpectralSweep.

Synthetic throughout, except one end-to-end check against the committed
reflectance pair.  The real sample sweep is 314 MB and takes ~20 s to parse, so it
is deliberately kept out of the fast path.
"""

import warnings

import numpy as np
import pytest

from leman import processing
from leman.loaders import ACSpectralSweep, SingleSpectrum

from test_loaders import N_PIXELS, N_SWEEPS, WAVELENGTH, make_spectral_csv, _roi1
from _paths import DATA


def _write_reference(path, counts) -> None:
    """A 2-row reference CSV in SingleSpectrum format, on the fixture's axis."""
    rows = [",".join(f"{w}" for w in WAVELENGTH),
            ",".join(f"{c}" for c in counts)]
    path.write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# processing.subtract_spectrum
# ---------------------------------------------------------------------------


def test_subtract_spectrum_broadcasts_over_sweeps():
    spectra    = np.arange(12, dtype=float).reshape(4, 3)
    background = np.array([1.0, 2.0, 3.0, 4.0])
    out = processing.subtract_spectrum(spectra, background)
    # One measured background, removed from every sweep.
    assert out.shape == spectra.shape
    for j in range(3):
        assert np.allclose(out[:, j], spectra[:, j] - background)


def test_subtract_spectrum_accepts_matching_2d():
    spectra = np.ones((4, 3))
    out = processing.subtract_spectrum(spectra, np.full((4, 3), 0.25))
    assert np.allclose(out, 0.75)


def test_subtract_spectrum_rejects_wrong_length():
    with pytest.raises(ValueError, match="share an"):
        processing.subtract_spectrum(np.ones((4, 3)), np.ones(5))


def test_subtract_spectrum_does_not_mutate_input():
    spectra = np.ones((4, 3))
    processing.subtract_spectrum(spectra, np.ones(4))
    assert np.allclose(spectra, 1.0)


# ---------------------------------------------------------------------------
# processing.spectral_contrast
# ---------------------------------------------------------------------------


def test_contrast_formula():
    sample    = np.array([[2.0], [3.0], [4.0]])
    reference = np.array([1.0, 2.0, 2.0])
    out, guarded = processing.spectral_contrast(sample, reference)
    assert np.allclose(out[:, 0], [1.0, 0.5, 1.0])     # (S - R) / R
    assert not guarded.any()


def test_ratio_is_contrast_plus_one():
    rng = np.random.default_rng(0)
    sample    = rng.uniform(1, 10, size=(6, 4))
    reference = rng.uniform(1, 10, size=6)
    rc, _    = processing.spectral_contrast(sample, reference, mode="contrast")
    ratio, _ = processing.spectral_contrast(sample, reference, mode="ratio")
    assert np.allclose(ratio, rc + 1.0)


def test_jacobian_cancels_in_a_ratio():
    # (S*J)/(R*J) = S/R exactly, which is why energy_contrast is built with the
    # Jacobian off regardless of apply_jacobian.
    rng = np.random.default_rng(1)
    wl        = np.linspace(650.0, 780.0, 40)
    sample    = rng.uniform(1, 10, size=(40, 3))
    reference = rng.uniform(1, 10, size=40)

    plain, _ = processing.spectral_contrast(sample, reference)
    scaled, _ = processing.spectral_contrast(
        processing.jacobian_correction_wvl2E(sample, wl, axis=0),
        processing.jacobian_correction_wvl2E(reference, wl, axis=0),
    )
    assert np.allclose(plain, scaled)


def test_zero_reference_pixels_are_nan_and_reported():
    sample    = np.ones((4, 2))
    reference = np.array([1.0, 0.0, 2.0, -1.0])
    with pytest.warns(UserWarning, match="at or below 0"):
        out, guarded = processing.spectral_contrast(sample, reference)
    assert list(guarded) == [False, True, False, True]
    assert np.isnan(out[1]).all() and np.isnan(out[3]).all()
    # A zero reference must not become inf, which would swamp any colour scale.
    assert np.isfinite(out[[0, 2]]).all()


def test_min_reference_raises_the_floor():
    sample    = np.ones((3, 1))
    reference = np.array([10.0, 0.5, 20.0])
    with pytest.warns(UserWarning):
        out, guarded = processing.spectral_contrast(
            sample, reference, min_reference=1.0)
    assert list(guarded) == [False, True, False]
    assert np.isnan(out[1, 0])


def test_unknown_contrast_mode_lists_the_options():
    with pytest.raises(ValueError, match="ratio"):
        processing.spectral_contrast(np.ones((3, 1)), np.ones(3), mode="absorbance")


def test_contrast_rejects_mismatched_reference_length():
    with pytest.raises(ValueError, match="share an"):
        processing.spectral_contrast(np.ones((4, 2)), np.ones(5))


# ---------------------------------------------------------------------------
# Loader wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_and_ref(tmp_path):
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    ref = tmp_path / "reference.csv"
    # A reference a little below the sample, so contrast is positive and finite.
    _write_reference(ref, [_roi1(r, 0) - 5.0 for r in range(N_PIXELS)])
    return csv, ref


def test_no_reference_means_no_contrast(csv_and_ref):
    csv, _ = csv_and_ref
    s = ACSpectralSweep(str(csv), spectra_type="R")
    assert s.contrast is None
    assert s.energy_contrast is None
    assert s.reference is None


def test_reference_from_path_builds_contrast(csv_and_ref):
    csv, ref = csv_and_ref
    s = ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref))
    expected = (s.spectra - s.reference[:, None]) / s.reference[:, None]
    assert np.allclose(s.contrast, expected)
    assert s.energy_contrast.shape == s.contrast.shape


def test_reference_accepts_a_single_spectrum_object(csv_and_ref):
    csv, ref = csv_and_ref
    s = ACSpectralSweep(str(csv), spectra_type="R",
                              reference=SingleSpectrum(str(ref)))
    assert s.contrast is not None


def test_reference_accepts_a_bare_array(csv_and_ref):
    # The escape hatch for a caller who has aligned the axes themselves.
    csv, _ = csv_and_ref
    values = np.full(N_PIXELS, 50.0)
    s = ACSpectralSweep(str(csv), spectra_type="R", reference=values)
    assert np.allclose(s.reference, values)


def test_bare_array_of_wrong_length_rejected(csv_and_ref):
    csv, _ = csv_and_ref
    with pytest.raises(ValueError, match="bare array"):
        ACSpectralSweep(str(csv), spectra_type="R",
                              reference=np.ones(N_PIXELS + 1))


def test_reference_on_a_different_axis_raises_rather_than_resampling(tmp_path):
    # Interpolating would change the numbers and smooth the data, so it cannot be
    # a default; the message must say what to do instead.
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    ref = tmp_path / "shifted.csv"
    rows = [",".join(f"{w + 50.0}" for w in WAVELENGTH),
            ",".join("100.0" for _ in WAVELENGTH)]
    ref.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="not resampled automatically"):
        ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref))


def test_reference_scale_biases_the_contrast_as_documented(csv_and_ref):
    csv, ref = csv_and_ref
    plain  = ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref))
    scaled = ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref),
                                   reference_scale=2.0)
    # (S - kR)/(kR) is not a rescaling of (S - R)/R -- that is the whole point of
    # requiring a matched exposure.
    assert not np.allclose(scaled.contrast, plain.contrast / 2.0)
    assert np.allclose(scaled.reference, plain.reference * 2.0)


def test_bg_spectrum_subtracted_before_contrast(csv_and_ref):
    csv, ref = csv_and_ref
    bg = np.full(N_PIXELS, 10.0)
    s = ACSpectralSweep(str(csv), spectra_type="R",
                              bg_spectrum=bg, reference=str(ref))
    corrected = s.spectra - bg[:, None]
    assert np.allclose(s.contrast,
                       (corrected - s.reference[:, None]) / s.reference[:, None])


def test_bg_spectrum_alone_populates_energy_spectra_bg(csv_and_ref):
    csv, _ = csv_and_ref
    bg = np.full(N_PIXELS, 3.0)
    s = ACSpectralSweep(str(csv), spectra_type="PL", bg_spectrum=bg)
    assert s.energy_spectra_bg is not None
    assert s.best_energy_spectra is s.energy_spectra_bg
    # Raw arrays are never mutated after load.
    assert np.allclose(s.spectra[:, 0], [_roi1(r, 0) for r in range(N_PIXELS)])


@pytest.mark.filterwarnings("ignore:apply_jacobian=True with no background")
def test_jacobian_never_applied_to_contrast(csv_and_ref):
    # No background here on purpose: this checks the ratio, and the missing-
    # background warning is asserted on in test_jacobian_background.
    csv, ref = csv_and_ref
    off = ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref),
                                apply_jacobian=False)
    on  = ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref),
                                apply_jacobian=True)
    assert np.allclose(on.energy_contrast, off.energy_contrast)
    # ...while the ordinary energy spectra do respond to it.
    assert not np.allclose(on.energy_spectra, off.energy_spectra)


def test_best_energy_spectra_never_returns_the_contrast(csv_and_ref):
    csv, ref = csv_and_ref
    s = ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref))
    assert s.best_energy_spectra is s.energy_spectra
    assert not np.allclose(s.best_energy_spectra, s.energy_contrast)


def test_spectra_type_not_mutated_by_supplying_a_reference(csv_and_ref):
    csv, ref = csv_and_ref
    s = ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref))
    assert s.spectra_type == "R"
    # The subject here is that spectra_type survives a reference=, not the
    # wording: a CCD records counts whichever beam produced them, so the raw
    # array is an intensity and only the derived contrast is a ratio.
    assert s.signal_label == "Reflected intensity (counts)"
    assert s.contrast_label == r"$\Delta R/R_0$"


def test_contrast_label_follows_the_mode(csv_and_ref):
    csv, ref = csv_and_ref
    s = ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref),
                              contrast="ratio")
    assert s.contrast_label == r"$R/R_0$"
    assert np.allclose(s.contrast,
                       s.spectra / s.reference[:, None])


def test_contrast_reachable_through_the_plotting_registry(csv_and_ref):
    from leman import plotting
    csv, ref = csv_and_ref
    s = ACSpectralSweep(str(csv), spectra_type="R", reference=str(ref))
    assert np.allclose(plotting._resolve_spectra(s, "contrast", "energy"),
                       s.energy_contrast)
    # And is unavailable, with a pointed message, when no reference was given.
    plain = ACSpectralSweep(str(csv), spectra_type="R")
    with pytest.raises(ValueError, match="reference="):
        plotting._resolve_spectra(plain, "contrast", "energy")


# ---------------------------------------------------------------------------
# One end-to-end check against the committed reflectance pair
# ---------------------------------------------------------------------------


def test_real_substrate_reference_shares_the_sample_axis():
    # The committed pair is same-session, so the axes match element-wise and no
    # resampling question arises.  Reading the 22 KB reference is cheap; the
    # 314 MB sample sweep is deliberately not loaded here.
    ref = SingleSpectrum(
        str(DATA / "reflectance-contrast" / "substrate_26_07_24_18_20_15.csv"))
    assert ref.n_pixels == 1340
    assert ref.wavelength[0] == pytest.approx(656.906, abs=1e-3)
    assert ref.wavelength[-1] == pytest.approx(781.759, abs=1e-3)
    # Every pixel is usable, so the contrast has no guarded gaps.
    assert (ref.best_spectra > 0).all()
