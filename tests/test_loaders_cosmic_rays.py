"""
Tests for the ``cosmic_rays=`` declaration on ACSpectralSweep.

Detection itself is covered by ``test_processing_cosmic_rays``; what is pinned
here is the loader's side of it:

* **opt-in** — nothing happens without the argument, and ``spectra`` stays the
  file's own counts whether or not a repair was asked for.
* **position in the chain** — the repair reaches every derived array, so a spike
  planted inside the ``bg_region`` window does not inflate the pedestal estimate
  and does not survive into a contrast.
* **the declaration is checked before the read**, and against the signature it is
  forwarded to.

Spectra with a Gaussian peak on a noisy pedestal, written through the export-layout
builder in ``test_loaders``: the MAD noise estimate needs real scatter to work
against, which the builder's default index ramp does not have.
"""

import numpy as np
import pytest

from leman.constants import HC_EV_NM
from leman.loaders import ACSpectralSweep

from test_loaders import N_SWEEPS, make_spectral_csv

N_PIX     = 400
WL        = 700.0 + 0.25 * np.arange(N_PIX)      # nm, ascending, uniform pixels
CR_PIXEL  = 80                                   # planted spike, in the continuum
CR_SWEEP  = 1
BG_PIXEL  = 20                                   # planted spike, inside BG_REGION
BG_SWEEP  = 2
BG_REGION = (WL[10], WL[40])                      # continuum window, peak-free


def _sweep() -> np.ndarray:
    """A (N_PIX, N_SWEEPS) sweep: a Gaussian peak on a noisy dark pedestal."""
    x   = np.arange(N_PIX)
    rng = np.random.default_rng(0)
    amp = np.array([400.0, 1200.0, 2000.0])       # intensity grows across sweeps
    # (N_PIX, 1) peak shape broadcast against (1, N_SWEEPS) amplitudes: one peak
    # per sweep point, same centre and width, differing only in height.
    peak = np.exp(-((x - 250.0) / 25.0) ** 2)[:, None] * amp[None, :]
    return peak + 100.0 + rng.normal(0.0, 3.0, (N_PIX, N_SWEEPS))


def _with_spikes(clean: np.ndarray) -> np.ndarray:
    """Plant one spike in the continuum and one inside the background window."""
    data = clean.copy()
    data[CR_PIXEL, CR_SWEEP] += 4000.0
    data[BG_PIXEL, BG_SWEEP] += 4000.0
    return data


@pytest.fixture(scope="module")
def spiked() -> np.ndarray:
    return _with_spikes(_sweep())


@pytest.fixture
def csv_path(tmp_path, spiked):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path, roi1=spiked, wavelength=WL)
    return path


def _load(csv_path, **kwargs) -> ACSpectralSweep:
    return ACSpectralSweep(csv_path, spectra_type="PL", **kwargs)


def _ascending_energy(spectra) -> np.ndarray:
    """Reorder a wavelength-space array onto the ascending energy axis."""
    return spectra[np.argsort(HC_EV_NM / WL)]


# ---------------------------------------------------------------------------
# Opt-in
# ---------------------------------------------------------------------------


def test_absent_by_default(csv_path, spiked):
    """No declaration, no repair — and the spike is still there to prove it."""
    scan = _load(csv_path)

    assert scan.cosmic_rays is None
    assert scan.spectra_cr is None
    assert scan.cosmic_ray_mask is None
    assert scan.spectra[CR_PIXEL, CR_SWEEP] == pytest.approx(
        spiked[CR_PIXEL, CR_SWEEP])


def test_empty_dict_opts_in(csv_path):
    """``{}`` is a declaration, and accepts every default of the function."""
    scan = _load(csv_path, cosmic_rays={})

    assert scan.cosmic_rays == {}
    assert scan.cosmic_ray_mask[CR_PIXEL, CR_SWEEP]
    assert scan.spectra_cr[CR_PIXEL, CR_SWEEP] < scan.spectra[CR_PIXEL, CR_SWEEP]


def test_spectra_is_never_repaired_in_place(csv_path, spiked):
    """The repair adds an array; it does not replace the one the file gave."""
    scan = _load(csv_path, cosmic_rays={"sigma_threshold": 4.0})

    np.testing.assert_allclose(scan.spectra, spiked)
    assert scan.spectra_cr is not scan.spectra


def test_broad_peak_is_left_alone(csv_path):
    """A peak tens of pixels wide is signal, whatever its height."""
    scan = _load(csv_path, cosmic_rays={})

    assert not scan.cosmic_ray_mask[240:260, :].any()


# ---------------------------------------------------------------------------
# Position in the correction chain
# ---------------------------------------------------------------------------


def test_the_repair_is_its_own_rung_on_both_axes(csv_path):
    """``spectra_cr`` has an energy-axis mirror; the first rung stays the file's."""
    scan = _load(csv_path, cosmic_rays={})

    np.testing.assert_allclose(scan.energy_spectra_cr,
                               _ascending_energy(scan.spectra_cr))
    np.testing.assert_allclose(scan.energy_spectra,
                               _ascending_energy(scan.spectra))
    np.testing.assert_allclose(scan.energy_spectra_pre_jacobian,
                               _ascending_energy(scan.spectra))
    # The guard: without a flagged spike the two rungs would be equal and every
    # assertion above would hold for the wrong reason.
    assert not np.allclose(scan.energy_spectra, scan.energy_spectra_cr)


def test_repair_alone_leaves_no_background_array(csv_path):
    """A repair is not a background subtraction, so the bg rungs stay None."""
    scan = _load(csv_path, cosmic_rays={})

    assert scan.spectra_bg is None
    assert scan.energy_spectra_bg is None
    # Both accessors fall back to the repair rung, and to the same one.
    assert scan.best_energy_spectra is scan.energy_spectra_cr
    assert scan.best_spectra is scan.spectra_cr


def test_spike_in_the_window_does_not_bias_the_pedestal(csv_path, spiked):
    """
    The whole reason the repair runs first: BG_PIXEL sits inside BG_REGION, so an
    unrepaired spike there is averaged into the pedestal and over-subtracted from
    every pixel of that sweep.
    """
    without = _load(csv_path, bg_region_nm=BG_REGION)
    with_cr = _load(csv_path, bg_region_nm=BG_REGION, cosmic_rays={})

    assert with_cr.cosmic_ray_mask[BG_PIXEL, BG_SWEEP]

    # The window holds 31 pixels, so a 4000-count spike shifts its mean by ~129.
    n_window   = int(((WL >= BG_REGION[0]) & (WL <= BG_REGION[1])).sum())
    over_sub   = 4000.0 / n_window
    difference = (with_cr.energy_spectra_bg[:, BG_SWEEP]
                  - without.energy_spectra_bg[:, BG_SWEEP])
    # Compare well away from the repaired pixel itself, where the two arrays also
    # differ by the replacement.
    np.testing.assert_allclose(difference[200:300], over_sub, rtol=0.05)

    # The sweeps carrying no spike in the window are untouched by either load.
    np.testing.assert_allclose(with_cr.energy_spectra_bg[200:300, 0],
                               without.energy_spectra_bg[200:300, 0])


def test_contrast_is_formed_from_the_repair(tmp_path, spiked):
    """A spike in the sample arm must not reach the ratio."""
    scan_path = tmp_path / "scan.csv"
    make_spectral_csv(scan_path, roi1=spiked, wavelength=WL)

    ref_path = tmp_path / "bare.csv"
    ref_path.write_text(
        ",".join(f"{w}" for w in WL) + "\n"
        + ",".join(f"{v}" for v in np.full(N_PIX, 500.0)) + "\n"
    )

    scan = ACSpectralSweep(
        scan_path, spectra_type="R", reference=ref_path, cosmic_rays={},
    )

    # (S − R)/R at the repaired pixel, formed from the repaired numerator.
    expected = (scan.spectra_cr[CR_PIXEL, CR_SWEEP] - 500.0) / 500.0
    assert scan.contrast[CR_PIXEL, CR_SWEEP] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The declaration is checked, and checked early
# ---------------------------------------------------------------------------


def test_unknown_key_raises(csv_path):
    with pytest.raises(ValueError, match="unknown key"):
        _load(csv_path, cosmic_rays={"sigma": 4.0})


@pytest.mark.parametrize("key", ["spectra", "axis"])
def test_loader_owned_arguments_are_rejected(csv_path, key):
    """Which array, and which way round it is stored, are not the caller's."""
    with pytest.raises(ValueError, match="unknown key"):
        _load(csv_path, cosmic_rays={key: 1})


def test_declaration_is_checked_before_the_file_is_read(tmp_path):
    """A typo is reported without paying for the decode of a large export."""
    missing = tmp_path / "not-written.csv"
    with pytest.raises(ValueError, match="unknown key"):
        ACSpectralSweep(missing, spectra_type="PL",
                              cosmic_rays={"sigma": 4.0})


def test_repr_reports_the_repair(csv_path):
    scan = _load(csv_path, cosmic_rays={})
    assert "Cosmic rays" in repr(scan)
    assert "replaced" in repr(scan)


def test_hdf5_records_the_declaration_without_replaying_it(tmp_path, csv_path):
    """
    Provenance, like apply_jacobian: the archive says a repair was made, and the
    stored spectra are the raw ones, so reading does not silently redo it.
    """
    scan = _load(csv_path, cosmic_rays={"sigma_threshold": 4.0})
    h5 = tmp_path / "scan.h5"
    scan.to_hdf5(h5)

    reloaded = ACSpectralSweep(h5)

    assert reloaded.source_metadata["cosmic_rays"] == {"sigma_threshold": 4.0}
    assert reloaded.cosmic_rays is None
    assert reloaded.spectra_cr is None
    np.testing.assert_allclose(reloaded.spectra, scan.spectra)
