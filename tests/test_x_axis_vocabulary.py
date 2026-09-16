"""
One spectral-axis vocabulary, one refusal, shared by every ``x_axis=``.

``constants.X_AXES`` names the two orderings a spectrum can be served on, and
``_x_axis_name_unit`` is the only thing that decides whether a value is one of
them. Each entry point still picks its own arrays — what is shared is the
refusal, and the reason to share it is that a value which is neither key is
otherwise read as the wavelength one by the branch below it, and returns or fits
the wrong array in silence.

The five parametrised entry points are the whole public surface of that
vocabulary: two accessors, the spectral-window helper, the peak fitter and a
plotting function. One test pins what the refusal deliberately does *not*
cover — asking for a wavelength-space source on the energy axis still only
warns, because an explicitly named source is served on its own axis.

The final section covers the one place both rows are shown at once: a plot's
conjugate top axis, which reads the *other* row of the same table.
"""

import warnings

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from leman import fitting, plotting, processing
from leman.constants import X_AXES, _x_axis_name_unit
from leman.loaders import ACSpectralSweep

from test_loaders import make_spectral_csv


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "sweep.csv"
    make_spectral_csv(path)
    return ACSpectralSweep(str(path), spectra_type="PL")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x_axis", sorted(X_AXES))
def test_every_row_unpacks_to_a_name_and_a_unit(x_axis):
    """A row is what an axis label is composed from, so both fields must be there."""
    name, unit = X_AXES[x_axis]
    assert isinstance(name, str) and name
    assert isinstance(unit, str) and unit


def test_the_composed_labels_are_the_ones_the_plots_carry(scan):
    labels = {axis: plotting._resolve_x_axis(scan, axis)[1] for axis in X_AXES}
    assert labels == {"energy": "Energy (eV)", "wavelength": "Wavelength (nm)"}


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


def test_the_refusal_names_every_key():
    # Derived from the table rather than written out, so a key added to X_AXES
    # cannot go unmentioned in the message.
    with pytest.raises(ValueError) as excinfo:
        _x_axis_name_unit("eV")
    message = str(excinfo.value)
    assert all(repr(key) in message for key in X_AXES)
    assert "'eV'" in message


def test_the_refusal_names_the_calling_function():
    with pytest.raises(ValueError, match=r"^pixel_slice\(\): x_axis must be"):
        _x_axis_name_unit("eV", what="pixel_slice()")


def test_an_unhashable_axis_is_a_value_error_not_a_type_error():
    with pytest.raises(ValueError, match="must be 'energy' or 'wavelength'"):
        _x_axis_name_unit(["energy"])


@pytest.mark.parametrize("call", [
    pytest.param(lambda s: s.pixel_slice((1.6, 1.7), x_axis="eV"),
                 id="pixel_slice"),
    pytest.param(lambda s: s.get_spectrum_at(value=0.0, x_axis="eV"),
                 id="get_spectrum_at"),
    pytest.param(lambda s: s.get_spectrum_by_index(0, x_axis="eV"),
                 id="get_spectrum_by_index"),
    pytest.param(lambda s: fitting.fit_scan_peak(s, x_axis="eV"),
                 id="fit_scan_peak"),
    pytest.param(lambda s: plotting.plot_spectral_map(s, x_axis="eV"),
                 id="plot_spectral_map"),
])
def test_every_entry_point_refuses_an_unknown_axis(scan, call):
    # Each of these used to serve the wavelength array instead. The accessors
    # resolve the sweep point first, so their coordinates are ones the fixture
    # really holds — the axis is what is being refused, not the point.
    with pytest.raises(ValueError, match="must be 'energy' or 'wavelength'"):
        call(scan)


def test_a_valid_axis_still_reaches_the_data(scan):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert scan.get_spectrum_at(value=0.0, x_axis="wavelength").shape == (
            scan.n_pixels,)
        assert np.array_equal(
            scan.get_spectrum_at(value=0.0, x_axis="wavelength"),
            scan.spectra[:, 0],
        )


# ---------------------------------------------------------------------------
# The axis and the source cannot disagree
# ---------------------------------------------------------------------------


def test_a_named_source_is_served_on_the_axis_asked_for(scan):
    # A source names a correction state and the axis names the representation, so
    # every state exists on both and no combination can be a mismatch.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        on_wavelength = scan.get_spectrum_at(value=0.0, source="raw",
                                             x_axis="wavelength")
        on_energy     = scan.get_spectrum_at(value=0.0, source="raw",
                                             x_axis="energy")
    assert np.array_equal(on_wavelength, scan.spectra[:, 0])
    assert np.array_equal(on_energy, scan.energy_spectra[:, 0])


# ---------------------------------------------------------------------------
# The conjugate top axis reads the other row of the same table
# ---------------------------------------------------------------------------


@pytest.fixture
def wide_scan(tmp_path):
    """
    A sweep spanning 500-800 nm, which is what a real PL spectrum covers.

    The shared fixture spans 9 nm, too narrow for matplotlib to place a whole
    number of nanometres between ticks — and round ticks are the property that
    tells the two possible mechanisms apart.
    """
    path = tmp_path / "wide.csv"
    make_spectral_csv(path, wavelength=np.linspace(500.0, 800.0, 31))
    return ACSpectralSweep(str(path), spectra_type="PL")


def _drawn_with_twin(scan, x_axis):
    """Draw once, since ticks and limits are only decided at draw time."""
    fig, ax, _, ax_twin = plotting.plot_spectrum(scan, index=0, x_axis=x_axis,
                                                 twin_axis=True)
    fig.canvas.draw()
    return fig, ax, ax_twin


def test_no_conjugate_axis_by_default(scan):
    fig, ax, _, ax_twin = plotting.plot_spectrum(scan, index=0)

    assert ax_twin is None
    assert ax.child_axes == []


@pytest.mark.parametrize("x_axis, conjugate", [
    ("energy",     "wavelength"),
    ("wavelength", "energy"),
])
def test_the_conjugate_axis_is_labelled_from_the_other_row(scan, x_axis,
                                                           conjugate):
    """
    Both directions, because a wavelength plot wants eV on top just as much.

    Composed from the table rather than compared against a literal: the label is
    the same string the primary axis would carry if the two were swapped.
    """
    _, _, ax_twin = _drawn_with_twin(scan, x_axis)
    name, unit = _x_axis_name_unit(conjugate)

    assert ax_twin is not None
    assert ax_twin.get_xlabel() == f"{name} ({unit})"


def test_the_conversion_is_the_packages_own_constant(scan):
    """
    Guards against a hardcoded 1239.84 drifting from constants.HC_EV_NM, which is
    what every other energy/wavelength conversion in the package goes through.

    Read off the limits rather than the transform callables, which are a
    matplotlib private. Sorted because the reciprocal reverses the ordering: the
    low-energy edge is the long-wavelength one.
    """
    _, ax, ax_twin = _drawn_with_twin(scan, "energy")

    expected = processing.energy_to_wavelength(np.asarray(ax.get_xlim()))
    assert sorted(ax_twin.get_xlim()) == pytest.approx(sorted(expected))


def test_the_conjugate_axis_follows_a_later_change_of_limits(scan):
    """
    A caller zooms after plotting, so the top axis has to be a live transform of
    the bottom one rather than a set of tick labels fixed when it was built.
    """
    fig, ax, ax_twin = _drawn_with_twin(scan, "energy")
    zoom = (1.535, 1.545)
    ax.set_xlim(*zoom)
    fig.canvas.draw()

    expected = processing.energy_to_wavelength(np.asarray(zoom))
    assert sorted(ax_twin.get_xlim()) == pytest.approx(sorted(expected))


def test_the_conjugate_axis_ticks_are_round_in_its_own_unit(wide_scan):
    """
    Matplotlib picks the ticks in the unit being displayed, so the nm labels land
    on round wavelengths.  Relabelling the eV ticks instead would put them at
    495.9, 550.4, … — the same axis, unreadable.
    """
    _, _, ax_twin = _drawn_with_twin(wide_scan, "energy")

    lo, hi = sorted(ax_twin.get_xlim())
    shown = [t for t in ax_twin.get_xticks() if lo <= t <= hi]
    assert shown, "the conjugate axis placed no ticks in range"
    assert all(float(t).is_integer() for t in shown)


def test_the_conjugate_axis_emits_no_divide_warning(scan):
    """
    Matplotlib evaluates the transform across the axis, including 0, where the
    reciprocal is undefined.  Without the errstate guard every draw warns.
    """
    fig, ax, _, _ = plotting.plot_spectrum(scan, index=0, twin_axis=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig.canvas.draw()


# ---------------------------------------------------------------------------
# The same axis on plot_spectral_series
# ---------------------------------------------------------------------------
#
# One helper, so one battery. These are the shorter version of the block above:
# what is worth re-pinning per call site is that the helper is reached at all, that
# the axis comes back, and that it is built late enough to survive the colorbar
# taking its space out of the host axes.


def _series_with_twin(scan, x_axis="energy", **kwargs):
    fig, ax, cb, lines, ax_twin = plotting.plot_spectral_series(
        scan, x_axis=x_axis, twin_axis=True, **kwargs)
    fig.canvas.draw()
    return fig, ax, ax_twin


def test_the_series_has_no_conjugate_axis_by_default(scan):
    fig, ax, cb, lines, ax_twin = plotting.plot_spectral_series(scan)

    assert ax_twin is None
    assert ax.child_axes == []


@pytest.mark.parametrize("x_axis, conjugate", [
    ("energy",     "wavelength"),
    ("wavelength", "energy"),
])
def test_the_series_conjugate_axis_is_labelled_from_the_other_row(scan, x_axis,
                                                                 conjugate):
    _, _, ax_twin = _series_with_twin(scan, x_axis)
    name, unit = _x_axis_name_unit(conjugate)

    assert ax_twin is not None
    assert ax_twin.get_xlabel() == f"{name} ({unit})"


def test_the_series_conjugate_axis_follows_a_later_change_of_limits(scan):
    """
    The reason this matters here specifically: the committed example notebook for
    this function zooms with set_xlim on the line after the call.
    """
    fig, ax, ax_twin = _series_with_twin(scan)
    zoom = (1.535, 1.545)
    ax.set_xlim(*zoom)
    fig.canvas.draw()

    expected = processing.energy_to_wavelength(np.asarray(zoom))
    assert sorted(ax_twin.get_xlim()) == pytest.approx(sorted(expected))


def test_the_series_conjugate_axis_spans_the_host_axes(scan):
    """
    This function is the one that draws a colorbar under the conjugate axis, and a
    colorbar takes its space out of the host axes.  The top axis has to end up over
    the shrunk host rather than over where the host used to be.

    A secondary axis is a zero-height strip pinned to the parent's top edge, so the
    comparison is on the horizontal extent and on sitting at that edge — not on the
    whole box.
    """
    fig, ax, ax_twin = _series_with_twin(scan, colorbar=True)

    host, top = ax.get_position(), ax_twin.get_position()
    assert (top.x0, top.x1) == pytest.approx((host.x0, host.x1))
    assert top.y0 == pytest.approx(host.y1)


def test_the_series_conjugate_axis_ticks_are_round_in_its_own_unit(wide_scan):
    _, _, ax_twin = _series_with_twin(wide_scan)

    lo, hi = sorted(ax_twin.get_xlim())
    shown = [t for t in ax_twin.get_xticks() if lo <= t <= hi]
    assert shown, "the conjugate axis placed no ticks in range"
    assert all(float(t).is_integer() for t in shown)


def test_the_series_conjugate_axis_emits_no_divide_warning(scan):
    fig, ax, cb, lines, ax_twin = plotting.plot_spectral_series(scan, twin_axis=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig.canvas.draw()
