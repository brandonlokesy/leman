"""
Tests for what counts as *the same instrument setting* on a sweep axis.

``_axis_atol`` answers one question — how far apart may two readings be and still be
one setting — and two things depend on the answer: the warning that a declared sweep
axis does not label its points individually, and the refusal of a coordinate that does
not identify one spectrum (decision ``0004`` §5).

Judging that on a fraction of the axis's own span is circular when the axis never
moved, because the span is then the instrument's own scatter. The rows below cover both
cases a span alone cannot separate — one setting read many times, and a genuinely fine
sweep — and each is pinned to the count it should produce.

Every row is drawn from its own seeded generator, so no test depends on the order the
others ran in. The scatter on the held rows is kept inside the range the detection
reaches; ``test_the_reach_of_the_detection_is_deliberate`` pins where that range ends.
"""

import warnings

import numpy as np
import pytest

from leman.loaders import (
    ACSpectralSweep,
    _AXIS_MIN_MOVES,
    _AXIS_RTOL,
    _axis_atol,
    _axis_driven,
    _count_distinct,
)

from test_loaders import make_spectral_csv


def _scatter(seed, size):
    """Read-back scatter from a generator of its own, so run order cannot move it."""
    return np.random.default_rng(seed).normal(0.0, 1.0, size)


# label, readings in acquisition order, settings the axis really holds.
AXES = [
    ("a setting read 12 times by a source-meter",
     500 + 0.01 * _scatter(1, 12), 1),
    ("a setting read in kelvin, where the offset dwarfs the scatter",
     300 + 0.02 * _scatter(2, 20), 1),
    ("the same, negative, so |mean| would not measure its size",
     -500 + 0.01 * _scatter(3, 12), 1),
    ("a row that is identically zero",
     np.zeros(12), 1),
    ("a narrow sweep on a large offset: 300.0 to 300.2 K",
     np.linspace(300.0, 300.2, 20) + 2e-3 * _scatter(4, 20), 20),
    ("a wide sweep: 300 to 320 K",
     np.linspace(300, 320, 20) + 2e-3 * _scatter(5, 20), 20),
    ("a flattened nest, whose axis restarts every row",
     np.tile(np.arange(-5, 6.0), 6) + 6e-6 * _scatter(6, 66), 11),
    ("a fine sweep on a large offset: 201 steps over 20 mV at 5 V",
     np.linspace(4.99, 5.01, 201) + 6e-6 * _scatter(7, 201), 201),
    ("a coarse sweep with no repeats: 6 log-spaced powers",
     2.0 ** np.arange(6), 6),
    ("the fewest points a sweep can have",
     np.array([1.0, 2.0, 3.0]), 3),
    # Both of these are narrow for their offset *and* coarse, so the first two signs
    # fail together and only the direction of travel is left to see them by.
    ("a narrow sweep coarse enough that only its direction shows: 6 points over 0.2 K",
     np.linspace(300.0, 300.2, 6) + 2e-3 * _scatter(15, 6), 6),
    ("a five-point gate step of 1 mV at 5 V",
     np.linspace(5.000, 5.004, 5) + 2e-6 * _scatter(16, 5), 5),
]


@pytest.mark.parametrize("readings, n_settings",
                         [(a, n) for _, a, n in AXES],
                         ids=[name for name, _, _ in AXES])
def test_an_axis_resolves_to_the_settings_it_holds(readings, n_settings):
    assert _count_distinct(readings, _axis_atol(readings)) == n_settings
    # A row holding one setting is by definition not driven, and vice versa.
    assert _axis_driven(readings) == (n_settings > 1)


@pytest.mark.parametrize("readings", [a for _, a, n in AXES if n == 1 and np.ptp(a) > 0],
                         ids=[name for name, a, n in AXES
                              if n == 1 and np.ptp(a) > 0])
def test_the_held_rows_would_defeat_a_span_only_tolerance(readings):
    """
    Pin that the fixtures reach the case rather than passing for free.

    Judging on 0.1% of the span alone splits these rows into nearly one setting per
    reading, which is what made the two safeguards below fall silent.
    """
    span_only = _count_distinct(readings, 1e-3 * float(np.ptp(readings)))
    assert span_only >= readings.size - 4      # nearly every reading its own setting


def test_the_reach_of_the_detection_is_deliberate():
    """
    A held setting is only recognised while its read-back is stable for its own size.

    All three signs of a driven axis have to fail before one is called held, and the
    first of them — span against RMS magnitude — fires once the scatter exceeds *rtol*
    of the reading. So a source-meter holding a gate is recognised and a power meter
    holding a power is not. Pinned rather than left implicit, because it is the boundary
    a reader would otherwise have to rediscover.
    """
    held = lambda rel, n, seed: not _axis_driven(500 * (1 + rel * _scatter(seed, n)))

    assert held(2e-6, 20, 11)      # source-meter on a gate: 20 uV on 5 V
    assert held(2e-5, 20, 12)      # any clean commanded read-back
    assert held(6.7e-5, 20, 13)    # 0.02 K on 300 K
    # Above the threshold the row reads as driven, so it keeps every reading as its own
    # setting. That is the pre-existing behaviour, not a regression, and it is why a
    # noisy power meter holding one setting is still not collapsed.
    assert not held(4e-3, 20, 14)  # power meter: 3 uW on 800 uW


def test_direction_is_only_evidence_once_a_row_has_moved_enough():
    """
    Where the third sign starts and stops.

    It reads the direction of travel and nothing else, so it sees a sweep the other two
    miss — too coarse for the step test, too narrow for the span test — at the price of
    being a coin toss on a row with barely any moves. ``_AXIS_MIN_MOVES`` is where that
    price is paid: below it the sign abstains, and the row falls back to the other two.
    """
    # A sweep neither of the first two signs can see, at the shortest length the third
    # one will speak for and one reading shorter.
    narrow = lambda n: np.linspace(300.0, 300.2, n)
    assert np.ptp(narrow(6)) < _AXIS_RTOL * np.sqrt(np.mean(narrow(6) ** 2))
    assert _axis_driven(narrow(_AXIS_MIN_MOVES + 1))
    assert not _axis_driven(narrow(_AXIS_MIN_MOVES))

    # Direction is read in acquisition order, so a sweep run downwards is as visible as
    # one run up, and a row that turns round is not a sweep at all.
    assert _axis_driven(narrow(6)[::-1])
    assert not _axis_driven(np.array([300.0, 300.1, 300.2, 300.1, 300.0, 300.1]))

    # Standing still is not turning round: a slow axis stepping between plateaus keeps
    # one direction across the repeats, which is what dropping the zero steps preserves.
    assert _axis_driven(np.repeat(np.linspace(300.0, 300.2, 5), 4))

    # Directionless scatter at the same size stays held, which is the case the sign must
    # not cost us.
    assert not _axis_driven(300 + 0.02 * _scatter(17, 20))


def _repeats_csv(path, n=12, seed=101):
    """A file of *n* spectra taken at one power setting, read back with scatter."""
    params = {
        "V_A":              np.full(n, 1.0),
        "V_B":              np.full(n, -1.0),
        "Excitation Power": (500 + 0.01 * _scatter(seed, n)) / 0.303e6,
        "Scanner X":        np.full(n, 5.0),
        "Scanner Y":        np.full(n, 7.0),
    }
    make_spectral_csv(path, params=params)
    return path


def test_repeat_measurements_warn_at_load(tmp_path):
    """
    The warning exists to catch exactly this, and used to stay silent.

    With a span-only tolerance every reading was its own setting, so the count matched
    the sweep length and the check returned early — concluding that an axis holding one
    value labelled all twelve points individually.
    """
    path = _repeats_csv(tmp_path / "repeats.csv")
    with pytest.warns(UserWarning) as rec:
        ACSpectralSweep(str(path), spectra_type="PL", sweep="power")

    msg = "\n".join(str(w.message) for w in rec)
    assert "takes only 1 value" in msg          # singular, not "1 different values"
    assert "held at one setting" in msg
    # A row that never moved is not a nest axis, so the nest advice must not appear.
    assert "fast_sweep=" not in msg
    assert "varying_parameters()" in msg


def test_a_coordinate_on_an_undriven_axis_is_refused(tmp_path):
    """Decision 0004 §5: an ambiguous coordinate is refused by the accessors."""
    path = _repeats_csv(tmp_path / "repeats.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan = ACSpectralSweep(str(path), spectra_type="PL", sweep="power")

    with pytest.raises(ValueError) as exc:
        scan.get_spectrum_at(500.0)
    msg = str(exc.value)
    assert "does not identify one spectrum" in msg
    assert "held at one setting" in msg
    assert "get_spectrum_by_index()" in msg     # the escape it offers
    # Selecting by position is unaffected: there is nothing ambiguous about an index.
    assert scan.get_spectrum_by_index(3).shape == (scan.spectra.shape[0],)


def test_an_undriven_axis_does_not_also_warn_that_the_value_is_absent(tmp_path):
    """
    ``nearest_index`` warns once, about ambiguity — not about distance.

    The distance warning fires when a request misses by more than half a typical gap
    between readings. On an undriven axis those gaps are scatter-sized, so it would
    report 500.0 as absent from an axis that holds nothing else.
    """
    path = _repeats_csv(tmp_path / "repeats.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan = ACSpectralSweep(str(path), spectra_type="PL", sweep="power")

    with pytest.warns(UserWarning) as rec:
        scan.nearest_index(500.0)
    messages = [str(w.message) for w in rec]
    assert len(messages) == 1
    assert "does not identify one" in messages[0]
    assert "found no point there" not in messages[0]


def test_a_fine_sweep_on_a_large_offset_keeps_all_its_settings(tmp_path):
    """
    The regression the rejected fix would have caused.

    Flooring the tolerance at a fraction of the row's *magnitude* absorbs the whole of a
    sweep whose steps are small compared with its offset, because the offset has nothing
    to do with the step size.
    """
    n = 21
    volts = np.linspace(4.99, 5.01, n) + 6e-6 * _scatter(102, n)
    params = {
        "V_A":              volts,
        "V_B":              np.full(n, -1.0),
        "Excitation Power": np.full(n, 2e-6),
        "Scanner X":        np.full(n, 5.0),
        "Scanner Y":        np.full(n, 7.0),
    }
    path = tmp_path / "fine.csv"
    make_spectral_csv(path, params=params)

    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no repeat warning: every point differs
        scan = ACSpectralSweep(str(path), spectra_type="PL", sweep="V_A")
    assert _count_distinct(scan.sweep_axis, _axis_atol(scan.sweep_axis)) == n
    # And a coordinate on it still selects one spectrum.
    assert scan.get_spectrum_at(float(volts[7])).shape == (scan.spectra.shape[0],)


def test_varying_parameters_and_the_tolerance_agree(tmp_path):
    """
    One helper decides both, so the report and the grouping cannot contradict.

    A row held at one setting must be absent from the report *and* collapse to one
    setting; a row that moved must appear *and* keep its settings.
    """
    path = _repeats_csv(tmp_path / "repeats.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan = ACSpectralSweep(str(path), spectra_type="PL", sweep="power")

    varying = scan.varying_parameters()
    for label, arr in scan.parameters.items():
        if arr[np.isfinite(arr)].size < 2:
            continue
        collapsed = _count_distinct(arr, _axis_atol(arr)) == 1
        assert (label in varying) is not collapsed, label

    assert "Excitation Power" not in varying    # scatter on a held setting


def test_varying_parameters_reports_a_narrow_sweep_on_a_large_offset(tmp_path):
    """
    The second of the two signs earns its place here too.

    Span against magnitude alone calls a 300.0-300.2 K sweep noise, so the row the
    experiment was actually about would be missing from the report.
    """
    n = 20
    temperature = np.linspace(300.0, 300.2, n) + 2e-3 * _scatter(103, n)
    params = {
        "V_A":              np.full(n, 1.0),
        "V_B":              np.full(n, -1.0),
        "Excitation Power": np.full(n, 2e-6),
        "Scanner X":        np.full(n, 5.0),
        "T":                temperature,
    }
    path = tmp_path / "temperature.csv"
    make_spectral_csv(path, params=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan = ACSpectralSweep(str(path), spectra_type="PL")

    assert "T" in scan.varying_parameters()
    # Pin that it is the second sign doing the work: the first one fails here.
    assert np.ptp(temperature) < 1e-3 * np.sqrt(np.mean(temperature ** 2))
    assert _axis_driven(temperature)
