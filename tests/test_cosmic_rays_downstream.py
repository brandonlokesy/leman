"""
A declared cosmic-ray repair reaches the wavelength axis, not just the energy one.

Every energy-space array is built from the repaired counts inside the loader, so the
energy axis was never at risk. The wavelength axis was: consumers chose their array
with ``best_energy_spectra if x_axis == "energy" else spectra``, naming a property on
one side and the file's own counts on the other, so a repair was dropped on sight of
``x_axis="wavelength"``. ``best_spectra`` is the missing counterpart, and every
consumer now asks the scan rather than picking an array itself.

The first test is the guard the rest depend on: if the planted spike were not actually
flagged, ``spectra_cr`` would equal ``spectra`` and every assertion below would pass
while testing nothing. The spike recipe is imported from the loader-level cosmic-ray
tests rather than rebuilt, so both files describe the same measurement.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from leman import fitting, plotting
from leman.loaders import (
    ACSpectralSweep,
    SingleSpectrum,
    _resolve_spectra,
)

from test_loaders import N_SWEEPS, make_spectral_csv
from test_loaders_cosmic_rays import (
    BG_REGION,
    CR_PIXEL,
    CR_SWEEP,
    N_PIX,
    WL,
    _sweep,
    _with_spikes,
)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path, roi1=_with_spikes(_sweep()), wavelength=WL)
    return path


@pytest.fixture
def repaired(csv_path):
    return ACSpectralSweep(str(csv_path), spectra_type="PL", cosmic_rays={})


@pytest.fixture
def untouched(csv_path):
    return ACSpectralSweep(str(csv_path), spectra_type="PL")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _spike_column(scan, attr: str) -> np.ndarray:
    return np.asarray(getattr(scan, attr))[:, CR_SWEEP].astype(float)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_the_planted_spike_is_really_repaired(repaired):
    """Without this, every assertion below could pass on identical arrays."""
    assert repaired.cosmic_ray_mask[CR_PIXEL, CR_SWEEP]
    assert (repaired.spectra_cr[CR_PIXEL, CR_SWEEP]
            < repaired.spectra[CR_PIXEL, CR_SWEEP])
    assert not np.array_equal(_spike_column(repaired, "spectra_cr"),
                              _spike_column(repaired, "spectra"))


# ---------------------------------------------------------------------------
# The accessor
# ---------------------------------------------------------------------------


def test_best_spectra_is_the_repair_when_one_was_declared(repaired):
    assert repaired.best_spectra is repaired.spectra_cr


def test_best_spectra_is_the_file_when_none_was(untouched):
    assert untouched.spectra_cr is None
    assert untouched.best_spectra is untouched.spectra


def test_the_resolver_serves_it_for_the_wavelength_axis(repaired):
    assert np.array_equal(_resolve_spectra(repaired, "best", "wavelength"),
                          repaired.best_spectra)
    # An explicitly named source is still exactly what it names.
    assert np.array_equal(_resolve_spectra(repaired, "raw", "wavelength"),
                          repaired.spectra)


def test_a_single_spectrum_gets_its_own_notion_of_best(tmp_path):
    # SingleSpectrum has no cosmic-ray repair but does hold a background-corrected
    # wavelength array, which the resolver used to pass over.
    path = tmp_path / "one.csv"
    path.write_text(
        ",".join(f"{w}" for w in WL) + "\n"
        + ",".join(f"{v}" for v in _sweep()[:, 0]) + "\n"
    )
    spectrum = SingleSpectrum(str(path), bg_region_nm=BG_REGION)

    assert spectrum.spectra_bg is not None
    served = _resolve_spectra(spectrum, "best", "wavelength")
    assert np.array_equal(served, spectrum.best_spectra)
    assert not np.array_equal(served, spectrum.spectra)


# ---------------------------------------------------------------------------
# The consumers
# ---------------------------------------------------------------------------


def _map_column(scan) -> np.ndarray:
    # median_kernel=1 turns the filter off: this function's default of 3 smooths
    # across sweeps, and the question here is which array reached the mesh.
    _, _, mesh = plotting.plot_spectral_map(scan, x_axis="wavelength",
                                            median_kernel=1)
    # get_array() is (n_sweeps, n_pixels): pcolormesh reads C as (rows=y,
    # cols=x) and the map's y is the sweep axis. Indexing the row rather than
    # reshaping matters — a reshape to (n_pixels, n_sweeps) would scramble.
    data = np.asarray(mesh.get_array())
    return data[CR_SWEEP, :].astype(float)


def _line_column(scan) -> np.ndarray:
    _, _, line, _ = plotting.plot_spectrum(scan, index=CR_SWEEP, x_axis="wavelength")
    return np.asarray(line.get_ydata(), dtype=float)


def _panel_column(scan) -> np.ndarray:
    panel = plotting.SpectrumLinePanel(scan, x_axis="wavelength")
    _, ax = plt.subplots()
    panel.init_artists(ax, range(N_SWEEPS))
    return panel._y[:, CR_SWEEP].astype(float)


@pytest.mark.parametrize("served", [
    pytest.param(_map_column,   id="plot_spectral_map"),
    pytest.param(_line_column,  id="plot_spectrum"),
    pytest.param(_panel_column, id="SpectrumLinePanel"),
])
def test_the_wavelength_axis_shows_the_repair(repaired, served):
    values = served(repaired)
    assert np.array_equal(values, _spike_column(repaired, "spectra_cr"))
    assert not np.array_equal(values, _spike_column(repaired, "spectra"))


@pytest.mark.parametrize("served", [
    pytest.param(_map_column,   id="plot_spectral_map"),
    pytest.param(_line_column,  id="plot_spectrum"),
    pytest.param(_panel_column, id="SpectrumLinePanel"),
])
def test_without_a_declaration_the_file_is_shown_unchanged(untouched, served):
    assert np.array_equal(served(untouched), _spike_column(untouched, "spectra"))


def test_the_colour_scale_is_built_from_the_repair(repaired):
    """
    ``SpectrumLinePanel(cmap=…)`` colours each trace by its peak, and a cosmic ray is
    the largest value in its column — so a scale built from the raw array would be
    stretched by the artefact, and every real frame squashed into the bottom of the
    colour map. A second consumer of the same array, so the same guard applies.
    """
    panel = plotting.SpectrumLinePanel(repaired, x_axis="wavelength", cmap="viridis")
    _, ax = plt.subplots()
    panel.init_artists(ax, range(N_SWEEPS))

    repaired_peaks = np.asarray(repaired.spectra_cr, float).max(axis=0)
    raw_peaks      = np.asarray(repaired.spectra,    float).max(axis=0)

    assert panel.mappable.norm.vmax == repaired_peaks.max()
    assert panel.mappable.norm.vmax != raw_peaks.max()


def test_the_wavelength_axis_fits_the_repair(repaired):
    # Gaussian, because the fixture's peak is one — a converged fit on both sides
    # is what makes the comparison meaningful.
    results = fitting.fit_scan_peak(repaired, x_axis="wavelength", model="gaussian")
    assert results[CR_SWEEP].converged

    expected = fitting.fit_gaussian(
        repaired.wavelength, _spike_column(repaired, "spectra_cr"))
    assert results[CR_SWEEP].params == pytest.approx(expected.params, rel=1e-12)

    # And the spikes really do move the answer, so this is not a null test.
    on_spikes = fitting.fit_gaussian(
        repaired.wavelength, _spike_column(repaired, "spectra"))
    assert results[CR_SWEEP].params["center"] != pytest.approx(
        on_spikes.params["center"], rel=1e-9)


# ---------------------------------------------------------------------------
# The energy axis, which was already right
# ---------------------------------------------------------------------------


def test_the_energy_axis_is_untouched_by_the_change(repaired):
    _, _, line, _ = plotting.plot_spectrum(repaired, index=CR_SWEEP, x_axis="energy")
    assert np.array_equal(np.asarray(line.get_ydata(), dtype=float),
                          _spike_column(repaired, "best_energy_spectra"))
