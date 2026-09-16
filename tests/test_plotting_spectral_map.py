"""
Tests for ``plotting.plot_spectral_map`` and its deprecated alias.

Seven things are pinned. First, the y-axis comes from ``sweep_axis`` rather than
the ``gate_axis`` alias, which is what lets the alias be deleted: a test that
only checked the values would pass either way, so the declared sweep is varied
(index, power, piezo) and the axis compared against the property.

Second, a declared nest is pinned on one axis and mapped along the other, and an
unpinned one is refused — on a nest ``sweep_axis`` is the flat index, so the map
would put consecutive rows at unrelated settings.  The block drawn is compared
against ``as_grid`` rather than only counted, since a wrong row has the right
shape.

Third, ``y_axis`` and ``spectra_source`` choose what is plotted and what it is
called, including that a contrast source is not labelled as PL.

Fourth, ``plot_pl_map_Vab_scan`` warns and then produces a figure identical to
the new name's — a shim that warns but has drifted is worse than no shim.

Fifth, ``median_kernel`` runs no filter unless a kernel is named, and still runs
one when it is.  The other array comparisons here pass ``median_kernel=1``, so
they cannot see the default; one test names no kernel so that it can.

Sixth, the mesh holds one row per sweep point.  ``pcolormesh`` reads ``C`` as
(rows=y, cols=x) and the map's y is the sweep axis, so the ``(n_pixels, n)``
block is handed over transposed; a test that only checked values would not
notice a flip, and every coordinate assertion here depends on the orientation.

Seventh, one non-finite cell does not blank a rescaled panel.  The rescale takes
its limits from the array's own min and max, which a single ``NaN`` makes
``NaN``, so the assertions are on the values and not only on the mask — every
cell being masked is exactly the failure being pinned against.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from leman import plotting
from leman.loaders import ACSpectralSweep

from test_loaders import (
    PARAMS, POWER_SCALE, WAVELENGTH, make_spectral_csv,
)
from test_loaders_nesting import FAST_VALUES, N_FAST, N_SLOW, RASTER, SLOW_VALUES
from test_contrast import _write_reference


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path)
    return path


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# The y-axis follows the declared sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sweep, expected", [
    (None,      np.arange(len(PARAMS["V_A"]))),
    ("power",   PARAMS["Excitation Power"] * POWER_SCALE),
    ("piezo_y", PARAMS["Scanner Y"]),
])
def test_y_axis_is_the_declared_sweep(csv_path, sweep, expected):
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL", sweep=sweep)
    fig, ax, mesh = plotting.plot_spectral_map(scan)

    # shading="auto" resolves to "nearest" for equal-shaped X/Y/C, so
    # _coordinates holds cell *edges* — (n_sweeps+1, n_pixels+1, 2), with the
    # outer edges extrapolated half a cell beyond the data. The sweep axis runs
    # down the rows, so averaging adjacent edges of column 0 recovers the
    # sweep-axis centres that were passed in.
    y_edges = mesh._coordinates[:, 0, 1]
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    assert np.allclose(y_centres, expected)
    assert np.allclose(scan.sweep_axis, expected)
    assert ax.get_ylabel() == scan.sweep_axis_label


def test_x_axis_switches_between_energy_and_wavelength(csv_path):
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL")

    _, ax_e, _ = plotting.plot_spectral_map(scan, x_axis="energy")
    _, ax_w, _ = plotting.plot_spectral_map(scan, x_axis="wavelength")

    assert ax_e.get_xlabel() == "Energy (eV)"
    assert ax_w.get_xlabel() == "Wavelength (nm)"


# ---------------------------------------------------------------------------
# The mesh orientation
# ---------------------------------------------------------------------------

def test_the_mesh_holds_one_row_per_sweep_point(csv_path):
    """
    ``pcolormesh`` reads ``C`` as (rows=y, cols=x), and y is the sweep axis.

    Pinned by name because the block is handed over transposed: going back to a
    2-D coordinate pair would flip it, and the other tests here would then fail
    inside ``_y_centres`` or on a shape, neither of which says what changed.
    """
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL")
    _, _, mesh = plotting.plot_spectral_map(scan, median_kernel=1)

    assert mesh.get_array().shape == (scan.n_sweeps, scan.n_pixels)

    # Column 0 of the drawn array is detector pixel 0 across every sweep point,
    # which is row 0 of the block the scan serves.
    drawn = np.asarray(mesh.get_array())
    assert np.allclose(drawn[:, 0], scan.best_energy_spectra[0, :])


# ---------------------------------------------------------------------------
# The deprecated alias
# ---------------------------------------------------------------------------

def test_old_name_warns_and_matches_new_name(csv_path):
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL", sweep="power")

    _, _, new_mesh = plotting.plot_spectral_map(scan)
    with pytest.warns(FutureWarning, match="plot_spectral_map"):
        _, _, old_mesh = plotting.plot_pl_map_Vab_scan(scan)

    assert np.allclose(new_mesh.get_array(), old_mesh.get_array())
    assert np.allclose(new_mesh._coordinates, old_mesh._coordinates)


def test_old_name_forwards_keyword_arguments(csv_path):
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL")

    with pytest.warns(FutureWarning):
        _, ax, mesh = plotting.plot_pl_map_Vab_scan(
            scan, x_axis="wavelength", median_kernel=1, clim=(0.0, 0.5),
        )

    assert ax.get_xlabel() == "Wavelength (nm)"
    assert mesh.get_clim() == (0.0, 0.5)


# ---------------------------------------------------------------------------
# Fixtures for the nest and the contrast source
# ---------------------------------------------------------------------------

@pytest.fixture
def nested(tmp_path):
    """A raster declared on raw rows, N_FAST inside N_SLOW."""
    path = tmp_path / "raster.csv"
    make_spectral_csv(path, params=RASTER)
    return ACSpectralSweep(str(path), spectra_type="PL",
                                 fast_sweep="Scanner X",
                                 slow_sweep="Scanner Y")


@pytest.fixture
def contrast_scan(tmp_path, csv_path):
    """A reflectance scan with a reference, so the contrast source resolves."""
    ref = tmp_path / "reference.csv"
    _write_reference(ref, np.full(len(WAVELENGTH), 50.0))
    return ACSpectralSweep(str(csv_path), spectra_type="R",
                                 reference=str(ref))


def _y_centres(mesh) -> np.ndarray:
    """
    The sweep-axis centres a mesh was built from.

    ``shading="auto"`` resolves to ``"nearest"`` for equal-shaped X/Y/C, so
    ``_coordinates`` holds cell *edges* with the outer ones extrapolated half a
    cell beyond the data.  The sweep axis runs down the rows, so averaging
    adjacent edges of column 0 recovers the centres.
    """
    edges = mesh._coordinates[:, 0, 1]
    return 0.5 * (edges[:-1] + edges[1:])


# ---------------------------------------------------------------------------
# A declared nest is pinned on one axis
# ---------------------------------------------------------------------------

def test_an_unpinned_nest_is_refused(nested):
    """Every row of the flat map would jump between fast and slow settings."""
    with pytest.raises(ValueError) as excinfo:
        plotting.plot_spectral_map(nested)

    message = str(excinfo.value)
    assert "plot_spectral_map()" in message
    for keyword in ("fast=", "index_fast=", "slow=", "index_slow="):
        assert keyword in message


def test_holding_the_slow_axis_maps_the_fast_one(nested):
    _, ax, mesh = plotting.plot_spectral_map(nested, slow=SLOW_VALUES[1])

    assert np.allclose(_y_centres(mesh), nested.nesting.fast_axis)
    assert ax.get_ylabel() == nested.nesting.fast_axis_label
    assert mesh.get_array().shape == (N_FAST, nested.n_pixels)


def test_holding_the_fast_axis_maps_the_slow_one(nested):
    _, ax, mesh = plotting.plot_spectral_map(nested, fast=FAST_VALUES[1])

    assert np.allclose(_y_centres(mesh), nested.nesting.slow_axis)
    assert ax.get_ylabel() == nested.nesting.slow_axis_label
    assert mesh.get_array().shape == (N_SLOW, nested.n_pixels)


def test_a_position_pins_the_same_axis_as_a_coordinate(nested):
    _, _, by_value = plotting.plot_spectral_map(nested, slow=SLOW_VALUES[1])
    _, _, by_index = plotting.plot_spectral_map(nested, index_slow=1)

    assert np.allclose(by_value.get_array(), by_index.get_array())
    assert np.allclose(by_value._coordinates, by_index._coordinates)


def test_the_mapped_block_is_the_scans_own_grid_row(nested):
    """Not merely the right shape — the right N_FAST columns of the grid."""
    _, _, mesh = plotting.plot_spectral_map(
        nested, index_slow=1, median_kernel=1)

    grid = nested.as_grid(nested.best_energy_spectra)   # (n_pixels, n_slow, n_fast)
    assert np.allclose(mesh.get_array(), grid[:, 1, :].T)


def test_the_nest_keywords_need_a_declared_nest(csv_path):
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL")

    with pytest.raises(ValueError) as excinfo:
        plotting.plot_spectral_map(scan, slow=1.0)

    message = str(excinfo.value)
    assert "plot_spectral_map()" in message
    assert "fast_sweep=" in message and "slow_sweep=" in message


# ---------------------------------------------------------------------------
# y_axis reads a flat sweep against another quantity
# ---------------------------------------------------------------------------

def test_y_axis_reads_the_sweep_against_another_quantity(csv_path):
    """Declared on the index, plotted against Scanner Y."""
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL")
    _, ax, mesh = plotting.plot_spectral_map(scan, y_axis="piezo_y")

    assert np.allclose(_y_centres(mesh), PARAMS["Scanner Y"])
    assert ax.get_ylabel() != scan.sweep_axis_label


def test_y_axis_names_itself_when_it_names_no_quantity(csv_path):
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL")

    with pytest.raises(ValueError, match="y_axis"):
        plotting.plot_spectral_map(scan, y_axis="not_a_quantity")


def test_y_axis_does_not_apply_to_a_nest(nested):
    """The free axis carries its own coordinate, so nothing chooses another."""
    with pytest.raises(ValueError) as excinfo:
        plotting.plot_spectral_map(nested, slow=SLOW_VALUES[1], y_axis="piezo_y")

    assert "y_axis='piezo_y'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# spectra_source picks the correction state, and labels it
# ---------------------------------------------------------------------------

def test_the_raw_source_plots_the_files_own_counts(csv_path):
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL",
                                 apply_jacobian=False)
    _, _, mesh = plotting.plot_spectral_map(
        scan, spectra_source="raw", x_axis="wavelength", median_kernel=1)

    assert np.allclose(mesh.get_array(), scan.spectra.T)


def test_the_contrast_source_takes_the_contrast_label(contrast_scan):
    fig, _, _ = plotting.plot_spectral_map(contrast_scan,
                                           spectra_source="contrast")

    assert fig.axes[-1].get_ylabel() == contrast_scan.contrast_label


# ---------------------------------------------------------------------------
# The median filter is off unless asked for, and reachable when it is
# ---------------------------------------------------------------------------

def test_no_median_filter_runs_unless_a_kernel_is_named(csv_path):
    """
    The default draws the scan's own array, unfiltered.

    Every other array comparison in this file passes ``median_kernel=1``
    explicitly, so all of them would keep passing whatever the default were.
    This one names no kernel, which is what pins the default itself.
    """
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL",
                                 apply_jacobian=False)
    _, _, mesh = plotting.plot_spectral_map(
        scan, spectra_source="raw", x_axis="wavelength")

    assert np.allclose(mesh.get_array(), scan.spectra.T)


def test_a_kernel_above_one_still_filters(csv_path):
    """The square filter stays reachable, so opting in must change the array."""
    scan = ACSpectralSweep(str(csv_path), spectra_type="PL",
                                 apply_jacobian=False)
    _, _, mesh = plotting.plot_spectral_map(
        scan, spectra_source="raw", x_axis="wavelength", median_kernel=3)

    assert not np.allclose(mesh.get_array(), scan.spectra.T)


# ---------------------------------------------------------------------------
# A non-finite cell does not blank the rescaled panel
# ---------------------------------------------------------------------------

def test_rescale_img_survives_a_guarded_contrast_pixel(tmp_path, csv_path):
    """
    One guarded reference pixel must not empty the whole map.

    ``rescale_intensity(in_range="image")`` reads its limits off the array's own
    min and max, so a single ``NaN`` makes both ``NaN`` and every cell comes
    back ``NaN``.  The route in is a real one: ``spectral_contrast`` guards a
    zero-count reference pixel to ``NaN``, which is a non-finite *column* of the
    map rather than one stray cell.
    """
    ref = tmp_path / "reference.csv"
    counts = np.full(len(WAVELENGTH), 50.0)
    counts[2] = 0.0                                  # guarded: cannot divide by it
    _write_reference(ref, counts)

    with pytest.warns(UserWarning, match="reference pixel"):
        scan = ACSpectralSweep(str(csv_path), spectra_type="R",
                                     reference=str(ref))

    # The wavelength axis keeps the file's pixel order, so the guarded pixel is
    # column 2 of the map; the energy axis would argsort it elsewhere.
    _, _, mesh = plotting.plot_spectral_map(
        scan, spectra_source="contrast", x_axis="wavelength", rescale_img=True)

    # (n_sweeps, n_pixels), so the guarded pixel is a masked column.
    arr = mesh.get_array()
    assert arr.mask[:, 2].all()
    assert not arr.mask[:, 0].any()

    finite = arr.compressed()
    assert finite.size == arr.size - scan.n_sweeps    # only that column is gone
    assert np.isfinite(finite).all()
    assert (finite.min(), finite.max()) == (0.0, 1.0)  # genuinely rescaled
