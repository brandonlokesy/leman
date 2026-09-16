"""
Tests for signal labelling in `plotting`.

Two rules are under test throughout:

* ``None`` derives the label from the scan's measurement type, so a reflectance
  or contrast sweep is not labelled as PL;
* a supplied string is used **verbatim** — nothing is ever appended to it.

Synthetic data only; these assert strings, not numbers.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from leman import plotting
from leman.constants import SIGNAL_LABELS
from leman.loaders import ACSpectralSweep, SingleSpectrum

from test_loaders import WAVELENGTH, make_spectral_csv
from test_contrast import _write_reference


CUSTOM = "Counts / s"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path)
    return path


def _scan(csv_path, spectra_type="PL"):
    return ACSpectralSweep(str(csv_path), spectra_type=spectra_type)


@pytest.fixture
def contrast_scan(tmp_path, csv_path):
    """A reflectance scan with a reference, so the contrast sources resolve."""
    ref = tmp_path / "reference.csv"
    _write_reference(ref, np.full(len(WAVELENGTH), 50.0))
    return ACSpectralSweep(str(csv_path), spectra_type="R",
                                 reference=str(ref))


@pytest.fixture
def single_spectrum(tmp_path):
    path = tmp_path / "single.csv"
    _write_reference(path, np.linspace(10.0, 20.0, len(WAVELENGTH)))
    return SingleSpectrum(str(path))


def _colorbar_text(mesh) -> str:
    """The label matplotlib actually rendered, read back off the colour bar."""
    return mesh.colorbar.ax.get_ylabel()


# ---------------------------------------------------------------------------
# The vocabulary itself
# ---------------------------------------------------------------------------

def test_counts_are_named_as_intensities():
    """
    A CCD integrates photons, so everything it records directly is an intensity
    in counts, whichever beam produced it; the ratio names belong to the derived
    arrays.  Re-adding ``("Reflectance", "counts")`` fails this whatever key it
    goes under.
    """
    for key, (name, unit) in SIGNAL_LABELS.items():
        assert ("intensity" in name.lower()) == (unit == "counts"), (
            f"{key!r} maps to {(name, unit)!r}: a counts-valued quantity must be "
            f"named as an intensity, and a ratio must carry no unit."
        )


@pytest.mark.parametrize("spectra_type", sorted(SIGNAL_LABELS))
def test_signal_unit_agrees_with_signal_label(csv_path, spectra_type):
    scan = _scan(csv_path, spectra_type)
    if scan.signal_unit:
        assert scan.signal_label == f"{scan.signal_name} ({scan.signal_unit})"
    else:
        assert scan.signal_label == scan.signal_name


# ---------------------------------------------------------------------------
# Derived labels follow the measurement type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spectra_type", sorted(SIGNAL_LABELS))
def test_colorbar_follows_spectra_type(csv_path, spectra_type):
    scan = _scan(csv_path, spectra_type)
    _, _, mesh = plotting.plot_spectral_map(scan)
    assert _colorbar_text(mesh) == scan.signal_label


@pytest.mark.parametrize("spectra_type", sorted(SIGNAL_LABELS))
def test_ylabel_follows_spectra_type(csv_path, spectra_type):
    scan = _scan(csv_path, spectra_type)
    _, ax, _, _ = plotting.plot_spectrum(scan, index=0)
    assert ax.get_ylabel() == scan.signal_label


def test_dimensionless_type_gets_no_unit(csv_path):
    """What a blanket f"{name} (counts)" gets wrong."""
    scan = _scan(csv_path, "RC")
    _, _, mesh = plotting.plot_spectral_map(scan)
    assert _colorbar_text(mesh) == r"$\Delta R/R_0$"
    assert "counts" not in _colorbar_text(mesh)


def test_normalised_dimensionless_is_not_marked_again(csv_path):
    """A ratio already reads as normalised; "(norm.)" would say it twice."""
    scan = _scan(csv_path, "RC")
    _, _, mesh = plotting.plot_spectral_map(scan, rescale_img=True)
    assert _colorbar_text(mesh) == r"$\Delta R/R_0$"
    assert "norm." not in _colorbar_text(mesh)


def test_rescaling_replaces_the_unit(csv_path):
    scan = _scan(csv_path, "PL")
    _, _, mesh = plotting.plot_spectral_map(scan, rescale_img=True)
    assert _colorbar_text(mesh) == "PL intensity (norm.)"


def test_no_unit_is_appended_twice(csv_path):
    """The concrete defect the compose-then-append design invited."""
    scan = _scan(csv_path, "PL")
    _, _, mesh = plotting.plot_spectral_map(scan, rescale_img=True)
    assert _colorbar_text(mesh).count("(norm.)") == 1
    assert "(counts)" not in _colorbar_text(mesh)


# ---------------------------------------------------------------------------
# The override is verbatim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rescale_img", [False, True])
def test_spectral_map_override_is_verbatim(csv_path, rescale_img):
    scan = _scan(csv_path, "PL")
    _, _, mesh = plotting.plot_spectral_map(
        scan, colorbar_label=CUSTOM, rescale_img=rescale_img)
    assert _colorbar_text(mesh) == CUSTOM


@pytest.mark.parametrize("normalize", [False, True])
def test_spectrum_override_is_verbatim(csv_path, normalize):
    scan = _scan(csv_path, "PL")
    _, ax, _, _ = plotting.plot_spectrum(scan, index=0, ylabel=CUSTOM,
                                         normalize=normalize)
    assert ax.get_ylabel() == CUSTOM


@pytest.mark.parametrize("normalize", [False, True])
def test_single_spectrum_override_is_verbatim(single_spectrum, normalize):
    _, ax, _ = plotting.plot_single_spectrum(
        single_spectrum, ylabel=CUSTOM, normalize=normalize)
    assert ax.get_ylabel() == CUSTOM


@pytest.mark.parametrize("spectrum_offset", [0.0, 5.0])
def test_spectral_series_override_is_verbatim(csv_path, spectrum_offset):
    scan = _scan(csv_path, "PL")
    ax = plotting.plot_spectral_series(
        scan, ylabel=CUSTOM, spectrum_offset=spectrum_offset).ax
    assert ax.get_ylabel() == CUSTOM


@pytest.mark.parametrize("rescale_img", [False, True])
def test_plot_image_honours_label_when_rescaled(rescale_img):
    """plot_image used to discard the argument whenever rescale_img was set."""
    im = plotting.plot_image(
        np.arange(16.0).reshape(4, 4), colorbar_label=CUSTOM,
        rescale_img=rescale_img).im
    assert im.colorbar.ax.get_ylabel() == CUSTOM


def test_plot_image_default_still_marks_rescaling():
    im = plotting.plot_image(np.arange(16.0).reshape(4, 4),
                             rescale_img=True).im
    assert im.colorbar.ax.get_ylabel() == "Intensity (norm.)"


# ---------------------------------------------------------------------------
# Contrast is a different quantity from the raw signal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x_axis", ["energy", "wavelength"])
def test_contrast_source_uses_contrast_label(contrast_scan, x_axis):
    # One source name, either axis: the label follows the quantity, not the axis.
    ax = plotting.plot_spectral_series(
        contrast_scan, spectra_source="contrast", x_axis=x_axis).ax
    assert ax.get_ylabel() == contrast_scan.contrast_label
    assert "counts" not in ax.get_ylabel()


def test_raw_source_keeps_the_signal_label(contrast_scan):
    ax = plotting.plot_spectral_series(contrast_scan,
                                       spectra_source="raw").ax
    assert ax.get_ylabel() == contrast_scan.signal_label
    assert ax.get_ylabel() == "Reflected intensity (counts)"


def test_contrast_source_offset_drops_no_unit(contrast_scan):
    """A dimensionless signal has no unit to drop, so it is only marked shifted."""
    ax = plotting.plot_spectral_series(
        contrast_scan, spectra_source="contrast", spectrum_offset=0.01).ax
    assert ax.get_ylabel() == f"{contrast_scan.contrast_label} (offset)"


def test_intensity_offset_marks_arbitrary_units(csv_path):
    scan = _scan(csv_path, "PL")
    ax = plotting.plot_spectral_series(scan, spectrum_offset=5.0).ax
    assert ax.get_ylabel() == "PL intensity (a.u., offset)"


# ---------------------------------------------------------------------------
# Objects that declare no measurement type
# ---------------------------------------------------------------------------

def test_single_spectrum_falls_back_to_neutral(single_spectrum):
    """A 2-row CSV is as likely a reflectance reference as PL — so not "PL"."""
    _, ax, _ = plotting.plot_single_spectrum(single_spectrum)
    assert ax.get_ylabel() == "Intensity (counts)"
    assert "PL" not in ax.get_ylabel()


def test_single_spectrum_normalised(single_spectrum):
    _, ax, _ = plotting.plot_single_spectrum(single_spectrum, normalize=True)
    assert ax.get_ylabel() == "Intensity (norm.)"


# ---------------------------------------------------------------------------
# SpectrumLinePanel
# ---------------------------------------------------------------------------

def test_panel_ylabel_defaults_to_the_scan(csv_path):
    scan = _scan(csv_path, "R")
    panel = plotting.SpectrumLinePanel(scan, sweep_attr="scanner_y")
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))
    assert ax.get_ylabel() == scan.signal_label


def test_panel_ylabel_override_is_verbatim(csv_path):
    scan = _scan(csv_path, "R")
    panel = plotting.SpectrumLinePanel(scan, sweep_attr="scanner_y",
                                       ylabel=CUSTOM)
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))
    assert ax.get_ylabel() == CUSTOM


def test_panel_colorbar_label_says_it_is_a_peak(csv_path):
    """
    The bar spans the range of per-frame *peaks* while the y-axis spans the full
    data range, so labelling both the same would put identical words on two scales
    that disagree.
    """
    scan = _scan(csv_path, "R")
    panel = plotting.SpectrumLinePanel(scan, sweep_attr="scanner_y", cmap="viridis")
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))

    assert panel.colorbar.ax.get_ylabel() == f"Peak {scan.signal_label}"
    assert panel.colorbar.ax.get_ylabel() != ax.get_ylabel()


def test_panel_colorbar_label_override_is_verbatim(csv_path):
    """No "Peak " prefix on a caller's string — nothing is ever appended to one."""
    scan = _scan(csv_path, "R")
    panel = plotting.SpectrumLinePanel(scan, sweep_attr="scanner_y",
                                       cmap="viridis", colorbar_label=CUSTOM)
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))

    assert panel.colorbar.ax.get_ylabel() == CUSTOM
