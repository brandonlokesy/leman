"""
Tests for SyntheticSpectralSweep — a spectral sweep built from arrays.

Its value is that it is the *same* object the file loaders produce, so the tests
check two things: that the arrays go in unchanged and in the stated order, and
that the shared machinery (correction ladder, sweep axis, nest, fitting) then
works on it as it does on measured data.
"""

import numpy as np
import pytest

from leman import SyntheticSpectralSweep
from leman.constants import HC_EV_NM
from leman.fitting import fit_scan_peak, lorentzian

ENERGY = np.linspace(1.65, 1.75, 401)               # eV, ascending
POWER  = np.array([1.0, 2.0, 4.0, 8.0])             # µW
CENTRE = 1.70                                       # eV
FWHM   = 5e-3                                       # eV


def _peaks(amplitudes=POWER) -> np.ndarray:
    """(n_pixels, n_sweeps): one Lorentzian on ENERGY per amplitude."""
    return np.stack([lorentzian(ENERGY, a, CENTRE, FWHM) for a in amplitudes],
                    axis=1)


def _scan(spectra=None, **kwargs) -> SyntheticSpectralSweep:
    kwargs.setdefault("energy", ENERGY)
    kwargs.setdefault("parameters", {"Excitation Power": POWER})
    kwargs.setdefault("spectra_type", "PL")
    return SyntheticSpectralSweep(_peaks() if spectra is None else spectra,
                                  **kwargs)


# ---------------------------------------------------------------------------
# The arrays go in unchanged
# ---------------------------------------------------------------------------


def test_energy_input_comes_back_on_the_energy_axis():
    scan = _scan()
    assert np.allclose(scan.energy, ENERGY)
    assert np.allclose(scan.energy_spectra, _peaks())
    assert np.allclose(scan.best_energy_spectra, _peaks())


def test_energy_input_is_stored_on_ascending_wavelength():
    # An ascending energy grid is a descending wavelength grid, so the rows are
    # reversed — values reordered, not converted.
    scan = _scan()
    assert np.all(np.diff(scan.wavelength) > 0)
    assert np.allclose(scan.wavelength, HC_EV_NM / ENERGY[::-1])
    assert np.allclose(scan.spectra, _peaks()[::-1])


def test_wavelength_input_is_stored_as_given():
    wl      = np.linspace(700.0, 760.0, 61)
    spectra = np.random.default_rng(0).random((61, 3))
    scan    = SyntheticSpectralSweep(spectra, wavelength=wl, spectra_type="PL")
    assert np.array_equal(scan.wavelength, wl)
    assert np.array_equal(scan.spectra, spectra)
    assert np.array_equal(scan.best_spectra, spectra)


def test_descending_wavelength_is_reordered():
    wl      = np.linspace(760.0, 700.0, 61)
    spectra = np.random.default_rng(1).random((61, 2))
    scan    = SyntheticSpectralSweep(spectra, wavelength=wl, spectra_type="PL")
    assert np.array_equal(scan.wavelength, wl[::-1])
    assert np.array_equal(scan.spectra, spectra[::-1])


def test_every_rung_exists_and_the_corrections_are_off():
    # The ladder is built as for a file, so absent corrections are None rather
    # than missing attributes — _resolve_spectra tells the two apart.
    scan = _scan()
    for rung in ("spectra_cr", "spectra_bg", "contrast",
                 "energy_spectra_cr", "energy_spectra_bg", "energy_contrast"):
        assert getattr(scan, rung) is None
    assert scan.energy_spectra_pre_jacobian is scan.energy_spectra


def test_input_arrays_are_copied():
    spectra = _peaks()
    rows    = {"Excitation Power": POWER.copy()}
    scan    = _scan(spectra, parameters=rows)
    spectra[:] = 0.0
    rows["Excitation Power"][:] = 0.0
    assert scan.spectra.any()
    assert np.array_equal(scan.power, POWER)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_neither_axis_refuses():
    with pytest.raises(ValueError, match="neither"):
        SyntheticSpectralSweep(_peaks(), spectra_type="PL")


def test_both_axes_refuse():
    with pytest.raises(ValueError, match="both"):
        _scan(wavelength=HC_EV_NM / ENERGY)


def test_non_monotonic_axis_refuses():
    axis = ENERGY.copy()
    axis[10], axis[11] = axis[11], axis[10]
    with pytest.raises(ValueError, match="strictly"):
        _scan(energy=axis)


def test_non_positive_energy_refuses():
    with pytest.raises(ValueError, match="positive"):
        _scan(energy=ENERGY - 2.0)


def test_transposed_spectra_name_the_fix():
    # One spectrum per row is how a loop filling spectra[i, :] lays them out.
    with pytest.raises(ValueError, match=r"spectra\.T"):
        _scan(_peaks().T)


def test_short_parameter_row_refuses():
    with pytest.raises(ValueError, match="Excitation Power"):
        _scan(parameters={"Excitation Power": POWER[:-1]})


def test_spectra_type_is_required():
    with pytest.raises(ValueError, match="spectra_type"):
        SyntheticSpectralSweep(_peaks(), energy=ENERGY)


def test_to_hdf5_refuses(tmp_path):
    with pytest.raises(NotImplementedError, match="synthetic"):
        _scan().to_hdf5(tmp_path / "scan.h5")
    assert not (tmp_path / "scan.h5").exists()


# ---------------------------------------------------------------------------
# The shared machinery works on it
# ---------------------------------------------------------------------------


def test_power_sweep_reads_the_row_in_microwatts():
    scan = _scan(sweep="power")
    assert np.array_equal(scan.sweep_axis, POWER)
    assert scan.sweep_unit == "µW"


def test_row_label_sweep():
    temperature = np.array([4.0, 10.0, 20.0, 40.0])
    scan = _scan(parameters={"T": temperature},
                 sweep="T", sweep_label="Temperature", sweep_unit="K")
    assert np.array_equal(scan.sweep_axis, temperature)
    assert scan.sweep_unit == "K"


def test_repr_names_the_class_and_no_file():
    text = repr(_scan(sweep="power"))
    assert text.startswith("SyntheticSpectralSweep")
    assert "<synthetic>" in text


def test_bg_region_removes_a_known_pedestal():
    # The intended way to test a correction: add the artefact, request the
    # correction, get the clean curve back.  A Lorentzian's tail is not zero
    # inside the window, so the window mean also removes that much of the line;
    # the tolerance is that residue at the largest amplitude.
    pedestal = 3.0
    window   = (1.650, 1.655)
    scan     = _scan(_peaks() + pedestal, bg_region_eV=window)
    residue  = lorentzian(window[1], POWER.max(), CENTRE, FWHM)
    assert np.allclose(scan.best_energy_spectra, _peaks(), atol=residue)
    assert not np.allclose(scan.best_energy_spectra, _peaks() + pedestal,
                           atol=pedestal / 2)


def test_declared_nest_resolves():
    # 3 gate settings (fast) × 4 powers (slow), flattened in acquisition order:
    # the gate changes at every point, the power once per pass of the gate.
    gate    = np.tile([-1.0, 0.0, 1.0], 4)
    power   = np.repeat(POWER, 3)
    spectra = np.stack([lorentzian(ENERGY, p, CENTRE + 1e-3 * v, FWHM)
                        for v, p in zip(gate, power)], axis=1)
    scan = SyntheticSpectralSweep(
        spectra, energy=ENERGY,
        parameters={"V_g": gate, "Excitation Power": power},
        spectra_type="PL",
        fast_sweep="V_g", slow_sweep="power",
    )
    assert scan.is_nested
    assert scan.nesting.shape == (4, 3)
    assert np.array_equal(scan.nesting.fast_axis, [-1.0, 0.0, 1.0])
    assert np.array_equal(scan.nesting.slow_axis, POWER)


def test_fit_recovers_the_line_it_was_built_with():
    for r in fit_scan_peak(_scan()):
        assert r.params["center"] == pytest.approx(CENTRE, abs=1e-6)
        assert r.params["fwhm"]   == pytest.approx(FWHM, rel=1e-4)
