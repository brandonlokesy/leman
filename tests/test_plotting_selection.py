"""
Tests for choosing which spectrum ``plotting.plot_spectrum`` draws.

A point is named either by coordinate (``value``, ``fast``, ``slow``) or by
integer position (``index``, ``index_fast``, ``index_slow``).  What is under
test is that the two spellings select the same data, that the selection is the
scan's own — so an ambiguous coordinate is refused and a distant one warns
exactly as it does for ``get_spectrum_at`` — and that the legend names the
coordinate the caller addressed.

Synthetic data throughout, reusing the sweep builders the loader tests use.
"""

import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from leman import plotting
from leman.loaders import ACSpectralSweep

from test_loaders import PARAMS, make_spectral_csv
from test_loaders_nesting import (
    FAST_VALUES, N_FAST, N_SLOW, RASTER, SLOW_VALUES,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def flat(tmp_path):
    """Sweep index 0, 1, 2 against Scanner Y = 7.0, 7.5, 8.0 V."""
    path = tmp_path / "flat.csv"
    make_spectral_csv(path)
    return ACSpectralSweep(str(path), spectra_type="PL",
                                 sweep="piezo_y")


@pytest.fixture
def undeclared(tmp_path):
    """The same file with no sweep declared, so the axis is the flat index."""
    path = tmp_path / "undeclared.csv"
    make_spectral_csv(path)
    return ACSpectralSweep(str(path), spectra_type="PL")


@pytest.fixture
def nested(tmp_path):
    """A nest declared on raw rows, which carry no unit of their own."""
    path = tmp_path / "raster.csv"
    make_spectral_csv(path, params=RASTER)
    return ACSpectralSweep(str(path), spectra_type="PL",
                                 fast_sweep="Scanner X",
                                 slow_sweep="Scanner Y")


@pytest.fixture
def nested_curated(tmp_path):
    """The same raster declared through the registry, so both axes know volts."""
    path = tmp_path / "raster_curated.csv"
    make_spectral_csv(path, params=RASTER)
    return ACSpectralSweep(str(path), spectra_type="PL",
                                 fast_sweep="piezo_x",
                                 slow_sweep="piezo_y")


def _ydata(line) -> np.ndarray:
    return np.asarray(line.get_ydata(), float)


# ---------------------------------------------------------------------------
# A coordinate and a position select the same spectrum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("i", range(len(PARAMS["Scanner Y"])))
def test_a_coordinate_selects_the_point_holding_it(flat, i):
    """Scanner Y = 7.5 must draw the same line as index 1."""
    _, _, by_value, _ = plotting.plot_spectrum(flat, value=PARAMS["Scanner Y"][i])
    _, _, by_index, _ = plotting.plot_spectrum(flat, index=i)
    assert np.array_equal(_ydata(by_value), _ydata(by_index))


def test_the_selected_spectrum_is_the_scans_own_column(flat):
    _, _, line, _ = plotting.plot_spectrum(flat, value=8.0)
    assert np.array_equal(_ydata(line), flat.best_energy_spectra[:, 2])


def test_a_negative_index_counts_from_the_end(flat):
    _, _, last, _ = plotting.plot_spectrum(flat, index=-1)
    assert np.array_equal(_ydata(last), flat.best_energy_spectra[:, -1])


def test_wavelength_axis_selects_the_same_column(flat):
    _, _, line, _ = plotting.plot_spectrum(flat, value=8.0, x_axis="wavelength")
    assert np.array_equal(_ydata(line), flat.spectra[:, 2].astype(float))


def test_a_coordinate_can_be_read_against_another_quantity(flat):
    """The sweep is declared in piezo_y; the point is asked for in V_A."""
    _, _, line, _ = plotting.plot_spectrum(flat, value=1.0, axis="V_A")
    assert np.array_equal(_ydata(line), flat.best_energy_spectra[:, 2])


# ---------------------------------------------------------------------------
# Nests are addressed on both axes
# ---------------------------------------------------------------------------

def test_both_nest_coordinates_pin_one_spectrum(nested):
    i_fast, i_slow = 1, 2
    _, _, line, _ = plotting.plot_spectrum(
        nested, fast=FAST_VALUES[i_fast], slow=SLOW_VALUES[i_slow])
    expected = nested.best_energy_spectra[:, i_slow * N_FAST + i_fast]
    assert np.array_equal(_ydata(line), expected)


def test_nest_positions_select_the_same_point_as_nest_coordinates(nested):
    i_fast, i_slow = 1, 2
    _, _, by_value, _ = plotting.plot_spectrum(
        nested, fast=FAST_VALUES[i_fast], slow=SLOW_VALUES[i_slow])
    _, _, by_index, _ = plotting.plot_spectrum(
        nested, index_fast=i_fast, index_slow=i_slow)
    assert np.array_equal(_ydata(by_value), _ydata(by_index))


def test_a_free_nest_axis_is_refused(nested):
    """One free axis is a line per point, which this function cannot return."""
    with pytest.raises(ValueError, match="more than one spectrum"):
        plotting.plot_spectrum(nested, fast=FAST_VALUES[0])
    with pytest.raises(ValueError, match="more than one spectrum"):
        plotting.plot_spectrum(nested, index_slow=0)


def test_a_flat_coordinate_on_a_nest_is_refused(nested):
    with pytest.raises(ValueError, match="does not locate a point"):
        plotting.plot_spectrum(nested, value=2.0)


def test_the_nest_axes_need_a_declared_nest(flat):
    with pytest.raises(ValueError, match="need a declared nest"):
        plotting.plot_spectrum(flat, fast=2.0, slow=5.0)


# ---------------------------------------------------------------------------
# The two spellings are exclusive, and one is required
# ---------------------------------------------------------------------------

def test_naming_a_point_both_ways_is_refused(flat):
    with pytest.raises(ValueError, match="not both"):
        plotting.plot_spectrum(flat, value=7.5, index=0)


def test_naming_no_point_is_refused(flat):
    with pytest.raises(ValueError, match="needs a point"):
        plotting.plot_spectrum(flat)


def test_axis_does_not_apply_to_a_position(flat):
    """axis= says what a *coordinate* is read against; an index has no axis."""
    with pytest.raises(ValueError, match="does not apply"):
        plotting.plot_spectrum(flat, index=0, axis="V_A")


# ---------------------------------------------------------------------------
# axis=None is the default, not a quantity
# ---------------------------------------------------------------------------

def test_axis_none_searches_the_declared_sweep_axis(flat):
    """
    Left to reach the scan, an undeclared axis reads as the flat index, so the
    coordinate would be searched against 0, 1, 2, ... instead of the sweep.
    """
    _, _, explicit, _ = plotting.plot_spectrum(flat, value=8.0, axis="sweep")
    _, _, defaulted, _ = plotting.plot_spectrum(flat, value=8.0, axis=None)
    assert np.array_equal(_ydata(defaulted), _ydata(explicit))
    assert np.array_equal(_ydata(defaulted), flat.best_energy_spectra[:, 2])


def test_axis_none_is_accepted_alongside_a_position(flat):
    """The same word cannot be the default in one spelling and refused in the other."""
    _, _, line, _ = plotting.plot_spectrum(flat, index=0, axis=None)
    assert np.array_equal(_ydata(line), flat.best_energy_spectra[:, 0])


def test_axis_none_labels_the_axis_that_was_searched(flat):
    """
    The legend is built from the axis name too, so an unresolved None names the
    point after an axis that was never searched even when the data is right.
    """
    _, _, line, _ = plotting.plot_spectrum(flat, value=8.0, axis=None)
    assert line.get_label() == r"Piezo $y$ (V) = 8"
    assert "Sweep index" not in line.get_label()


# ---------------------------------------------------------------------------
# A point is never given positionally
# ---------------------------------------------------------------------------

def test_a_bare_number_is_refused(flat):
    """
    Ambiguous by construction: on a sweep whose coordinates span its positions,
    neither the result nor a warning would say which was taken.
    """
    with pytest.raises(TypeError, match="positional"):
        plotting.plot_spectrum(flat, 7.5)


def test_both_spellings_state_which_they_mean(flat):
    """value= and index= are symmetric — neither gets the shorter call."""
    _, _, by_value, _ = plotting.plot_spectrum(flat, value=7.5)
    _, _, by_index, _ = plotting.plot_spectrum(flat, index=1)
    assert np.array_equal(_ydata(by_value), _ydata(by_index))


# ---------------------------------------------------------------------------
# A selector under a keyword that is not one
# ---------------------------------------------------------------------------

def test_the_no_point_error_names_what_arrived(flat):
    """
    Every selector is keyword-only, so a renamed or misspelt one is absorbed by
    the style passthrough. Reporting "no point" without naming it leaves the
    caller looking at the point they thought they had given.
    """
    with pytest.raises(ValueError) as exc:
        plotting.plot_spectrum(flat, sweep_index=1)
    msg = str(exc.value)
    assert "sweep_index=1" in msg
    assert "names no point" in msg
    assert "value=" in msg and "index=" in msg


def test_an_unrecognised_selector_of_any_name_is_named(flat):
    """Not keyed to one renamed parameter — any stray keyword is reported."""
    with pytest.raises(ValueError, match=r"indx=3"):
        plotting.plot_spectrum(flat, indx=3)


def test_style_kwargs_are_not_reported_when_a_point_was_given(flat):
    """The passthrough is only named when it is the likely cause."""
    _, _, line, _ = plotting.plot_spectrum(flat, value=7.5, color="k", lw=2)
    assert line.get_color() == "k"


def test_a_stray_keyword_alongside_a_point_still_reaches_matplotlib(flat):
    """With a point named, an unknown property is matplotlib's to reject."""
    with pytest.raises(AttributeError, match="sweep_index"):
        plotting.plot_spectrum(flat, value=7.5, sweep_index=1)


# ---------------------------------------------------------------------------
# A position must be a whole number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"index": 1.9},
    {"index": 1.0},                            # integral, but still not an int
    {"index_fast": 1.9, "index_slow": 0},
    {"index_fast": 1, "index_slow": 0.5},
])
def test_a_fractional_position_is_refused_not_truncated(nested, flat, kwargs):
    scan = nested if any("fast" in k or "slow" in k for k in kwargs) else flat
    with pytest.raises(TypeError, match="needs an integer"):
        plotting.plot_spectrum(scan, **kwargs)


def test_the_fractional_position_message_offers_the_coordinate(nested):
    """1.9 on a position keyword is far more likely a coordinate misfiled."""
    with pytest.raises(TypeError, match=r"pass it as fast="):
        plotting.plot_spectrum(nested, index_fast=1.9, index_slow=0)


def test_numpy_integers_are_still_positions(nested):
    """An index out of argmin/argmax is np.int64, not int."""
    _, _, line, _ = plotting.plot_spectrum(nested, index_fast=np.int64(1),
                                           index_slow=np.int64(2))
    assert np.array_equal(_ydata(line),
                          nested.best_energy_spectra[:, 2 * N_FAST + 1])


# ---------------------------------------------------------------------------
# A wrong position spelling is refused in this function's own vocabulary
# ---------------------------------------------------------------------------

def test_a_flat_position_on_a_nest_names_the_position_keywords(nested):
    """
    The scan's own message says "fast= and/or slow=", which here are *coordinates*
    — following it selects a different point and draws it without complaint.
    """
    with pytest.raises(ValueError) as exc:
        plotting.plot_spectrum(nested, index=3)
    msg = str(exc.value)
    assert "index_fast= and index_slow=" in msg
    assert "Name the axes: fast= and/or slow=." not in msg


def test_nest_positions_on_a_flat_sweep_name_index(flat):
    """The scan offers "the index positionally"; here the positional slot is a value."""
    with pytest.raises(ValueError) as exc:
        plotting.plot_spectrum(flat, index_fast=1, index_slow=0)
    msg = str(exc.value)
    assert "Use index=" in msg
    assert "positionally" not in msg


def test_following_the_advice_selects_what_was_asked_for(nested):
    """The point of the re-wording: the remedy it names must be the right one."""
    with pytest.raises(ValueError) as exc:
        plotting.plot_spectrum(nested, index=3)
    assert "index_fast=" in str(exc.value)

    # 3 = slow 0, fast 3 in a 4-fast nest; the coordinate spelling of the same
    # request would be fast=6.0, and fast=3 would land somewhere else entirely.
    _, _, line, _ = plotting.plot_spectrum(nested, index_fast=3, index_slow=0)
    assert np.array_equal(_ydata(line), nested.best_energy_spectra[:, 3])


# ---------------------------------------------------------------------------
# The lookup is the scan's own, so its policies reach the plot
# ---------------------------------------------------------------------------

def test_a_coordinate_far_from_any_point_warns(flat):
    with pytest.warns(UserWarning, match="found no point there"):
        plotting.plot_spectrum(flat, value=500.0)


def test_an_ambiguous_coordinate_is_refused(tmp_path):
    """A hysteresis loop passes the same gate voltage twice."""
    loop   = np.concatenate([np.linspace(0.0, 6.0, 4), np.linspace(4.0, 0.0, 3)])
    params = {"V_A": loop, "V_B": 30.0 - loop,
              "Excitation Power": np.full(loop.size, 1e-6)}
    path = tmp_path / "loop.csv"
    make_spectral_csv(path, params=params)
    scan = ACSpectralSweep(str(path), spectra_type="PL", sweep="V_A")

    with pytest.raises(ValueError, match="4"):
        plotting.plot_spectrum(scan, value=4.0)


def test_an_exact_coordinate_does_not_warn(flat):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plotting.plot_spectrum(flat, value=7.5)


# ---------------------------------------------------------------------------
# The legend names the coordinate the caller addressed
# ---------------------------------------------------------------------------

def test_the_legend_names_the_declared_sweep_axis(flat):
    _, _, line, _ = plotting.plot_spectrum(flat, value=7.5)
    assert line.get_label() == r"Piezo $y$ (V) = 7.5"


def test_the_legend_names_the_axis_that_was_searched(flat):
    """Addressed in V_A, so labelling it with piezo_y would misreport it."""
    _, _, line, _ = plotting.plot_spectrum(flat, value=1.0, axis="V_A")
    assert line.get_label().startswith("V_A = 1")


def test_the_legend_names_both_nest_coordinates(nested):
    _, _, line, _ = plotting.plot_spectrum(nested, fast=FAST_VALUES[1],
                                           slow=SLOW_VALUES[2])
    assert line.get_label() == "Scanner X = 2, Scanner Y = 10"


def test_a_curated_nest_axis_carries_its_unit(nested_curated):
    """A raw row states no unit; a registry key does, and the legend shows it."""
    _, _, line, _ = plotting.plot_spectrum(nested_curated, fast=FAST_VALUES[1],
                                           slow=SLOW_VALUES[2])
    assert line.get_label() == r"Piezo $x$ (V) = 2, Piezo $y$ (V) = 10"


def test_an_undeclared_sweep_is_labelled_by_index(undeclared):
    _, _, line, _ = plotting.plot_spectrum(undeclared, index=1)
    assert line.get_label() == "Sweep index = 1"


def test_a_supplied_label_is_used_verbatim(flat):
    _, _, line, _ = plotting.plot_spectrum(flat, value=7.5, label="my spectrum")
    assert line.get_label() == "my spectrum"
