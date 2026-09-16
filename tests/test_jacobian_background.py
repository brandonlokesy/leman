"""
Tests for how the Jacobian and background subtraction interact.

Both halves come from one fact: the λ² factor scales a constant dark pedestal
into a curved baseline rather than leaving it a flat offset.  So the order of the
two operations changes the numbers, and requesting the Jacobian with nothing
subtracted produces an energy-space array whose baseline rises towards the red
for purely instrumental reasons.

* **Ordering** — background comes off in wavelength space *first*, for both
  background mechanisms.
* **The warning** — both classes exposing ``apply_jacobian``
  (ACSpectralSweep, SingleSpectrum) say so at load time when no background
  was supplied.

Synthetic fixtures throughout, reusing the spectral CSV builder from
``test_loaders``.
"""

import warnings

import numpy as np
import pytest

from leman import processing
from leman.constants import HC_EV_NM
from leman.loaders import ACSpectralSweep, SingleSpectrum

from test_loaders import N_PIXELS, WAVELENGTH, make_spectral_csv

# Matches the opening clause of both warnings; narrow enough that an unrelated
# UserWarning from the same load is not mistaken for this one.
MATCH = "apply_jacobian=True with no background"


def _write_two_row_csv(path, counts) -> None:
    """A 2-row [wavelength; counts] CSV on the spectral fixture's axis."""
    rows = [",".join(f"{w}" for w in WAVELENGTH),
            ",".join(f"{c}" for c in counts)]
    path.write_text("\n".join(rows) + "\n")


def _jacobian_warnings(recorded) -> list:
    """The subset of *recorded* warnings that are this one."""
    return [w for w in recorded if MATCH in str(w.message)]


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path)
    return path


@pytest.fixture
def bg_path(tmp_path):
    path = tmp_path / "dark.csv"
    _write_two_row_csv(path, np.full(WAVELENGTH.size, 5.0))
    return path


# ---------------------------------------------------------------------------
# Order of operations: background in wavelength space, then the Jacobian
# ---------------------------------------------------------------------------


def _ascending_energy(spectra, wavelength) -> np.ndarray:
    """Reorder a wavelength-space array onto the ascending energy axis."""
    return spectra[np.argsort(HC_EV_NM / wavelength)]


def test_bg_spectrum_is_subtracted_before_the_jacobian(csv_path):
    # A flat pedestal B: subtracting first gives J·(S − B), subtracting after
    # gives J·S − B, and the two differ by B·(λ²/hc − 1) — a curve, not an offset.
    pedestal = 7.0
    scan = ACSpectralSweep(
        str(csv_path), spectra_type="PL", apply_jacobian=True,
        bg_spectrum=np.full(N_PIXELS, pedestal),
    )
    right = _ascending_energy(
        processing.jacobian_correction_wvl2E(
            scan.spectra - pedestal, scan.wavelength, axis=0),
        scan.wavelength,
    )
    wrong = _ascending_energy(
        processing.jacobian_correction_wvl2E(
            scan.spectra, scan.wavelength, axis=0),
        scan.wavelength,
    ) - pedestal

    assert np.allclose(scan.energy_spectra_bg, right)
    # Guards the test itself: if these agreed, the assertion above would be
    # satisfied by either order and would pin nothing.
    assert not np.allclose(right, wrong)


def test_bg_region_is_subtracted_before_the_jacobian(csv_path):
    # Same ordering for the window-mean mechanism.  subtract_background is reused
    # rather than reimplemented, so this pins the order and not its estimator.
    window = (806.0, 809.0)
    scan = ACSpectralSweep(
        str(csv_path), spectra_type="PL", apply_jacobian=True,
        bg_region_nm=window,
    )
    corrected = processing.subtract_background(
        scan.spectra, bg_region=window, x=scan.wavelength, axis=0)
    right = _ascending_energy(
        processing.jacobian_correction_wvl2E(
            corrected, scan.wavelength, axis=0),
        scan.wavelength,
    )
    assert np.allclose(scan.energy_spectra_bg, right)

    # And the reversed order is genuinely a different array.
    wrong = processing.subtract_background(
        _ascending_energy(
            processing.jacobian_correction_wvl2E(
                scan.spectra, scan.wavelength, axis=0),
            scan.wavelength,
        ),
        bg_region=window, x=scan.wavelength, axis=0,
    )
    assert not np.allclose(right, wrong)


def test_pre_jacobian_array_carries_no_background(csv_path):
    # energy_spectra_pre_jacobian is the uncorrected array the warning points at,
    # so it must track spectra rather than the background-subtracted version.
    scan = ACSpectralSweep(
        str(csv_path), spectra_type="PL", apply_jacobian=True,
        bg_region_nm=(806.0, 809.0),
    )
    assert np.allclose(
        scan.energy_spectra_pre_jacobian,
        _ascending_energy(scan.spectra, scan.wavelength),
    )


# ---------------------------------------------------------------------------
# ACSpectralSweep — the missing-background warning
# ---------------------------------------------------------------------------


def test_jacobian_without_background_warns(csv_path):
    with pytest.warns(UserWarning, match=MATCH):
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              apply_jacobian=True)


def test_warning_names_the_arguments_that_satisfy_it(csv_path):
    # A warning a researcher cannot act on gets ignored, so the fix must be in it.
    with pytest.warns(UserWarning, match=MATCH) as record:
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              apply_jacobian=True)
    message = str(_jacobian_warnings(record)[0].message)
    assert "bg_region_nm" in message
    assert "bg_region_eV" in message
    assert "bg_spectrum" in message


def test_bg_region_nm_silences_it(csv_path):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              apply_jacobian=True, bg_region_nm=(806.0, 809.0))
    assert _jacobian_warnings(record) == []


def test_bg_region_eV_silences_it(csv_path):
    # Resolved to nm before the check, so the eV entry point is covered too.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              apply_jacobian=True, bg_region_eV=(1.53, 1.54))
    assert _jacobian_warnings(record) == []


def test_bg_spectrum_silences_it(csv_path, bg_path):
    # A measured dark frame removes the pedestal just as a window mean does.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              apply_jacobian=True, bg_spectrum=str(bg_path))
    assert _jacobian_warnings(record) == []


def test_no_warning_without_the_jacobian(csv_path):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ACSpectralSweep(str(csv_path), spectra_type="PL")
    assert _jacobian_warnings(record) == []


def test_reference_alone_does_not_silence_it(csv_path, tmp_path):
    # The Jacobian never reaches energy_contrast, but energy_spectra still gets
    # it, so a reference is not a substitute for subtracting the background.
    reference = tmp_path / "bare.csv"
    _write_two_row_csv(reference, np.full(WAVELENGTH.size, 50.0))
    with pytest.warns(UserWarning, match=MATCH):
        ACSpectralSweep(str(csv_path), spectra_type="RC",
                              apply_jacobian=True, reference=str(reference))


def test_warning_points_at_the_uncorrected_array(csv_path):
    with pytest.warns(UserWarning, match=MATCH) as record:
        scan = ACSpectralSweep(str(csv_path), spectra_type="PL",
                                     apply_jacobian=True)
    assert "energy_spectra_pre_jacobian" in str(_jacobian_warnings(record)[0].message)
    # And it exists, uncorrected, as the message claims.
    assert not np.allclose(scan.energy_spectra_pre_jacobian, scan.energy_spectra)


# ---------------------------------------------------------------------------
# SingleSpectrum
# ---------------------------------------------------------------------------


def test_single_spectrum_jacobian_without_background_warns(tmp_path):
    path = tmp_path / "one.csv"
    _write_two_row_csv(path, np.arange(WAVELENGTH.size, dtype=float) + 10.0)
    with pytest.warns(UserWarning, match=MATCH):
        SingleSpectrum(str(path), apply_jacobian=True)


def test_single_spectrum_bg_region_silences_it(tmp_path):
    path = tmp_path / "one.csv"
    _write_two_row_csv(path, np.arange(WAVELENGTH.size, dtype=float) + 10.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        SingleSpectrum(str(path), apply_jacobian=True,
                       bg_region_nm=(806.0, 809.0))
    assert _jacobian_warnings(record) == []


def test_single_spectrum_no_warning_without_the_jacobian(tmp_path):
    path = tmp_path / "one.csv"
    _write_two_row_csv(path, np.arange(WAVELENGTH.size, dtype=float) + 10.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        SingleSpectrum(str(path))
    assert _jacobian_warnings(record) == []


def test_aux_spectrum_load_does_not_warn(csv_path, bg_path):
    # bg_spectrum= is resolved by constructing a SingleSpectrum internally; that
    # construction must not emit the warning on the caller's behalf.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              bg_spectrum=str(bg_path))
    assert _jacobian_warnings(record) == []
