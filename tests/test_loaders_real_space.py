"""
Tests for ACImgSweep frame ordering.

Frame order is checked synthetically, because every committed export is
zero-padded and so cannot exhibit the lexicographic failure.  Each synthetic
frame is filled with its own iteration number, so a test can assert *which* file
landed at an index rather than merely how many there are.  The committed
sequences then pin that the padded case still loads silently.
"""

import shutil
import warnings

import numpy as np
import pytest

from leman.loaders import (
    ACImgSweep,
    _classify_csv,
    _resolve_frame,
)

from _paths import DATA

SHAPE = (4, 5)                      # (ny, nx) — small; no frame content is analysed

# A committed 11-frame sequence, zero-padded iter_0000 … iter_0010, alongside a
# timestamped spectral file whose name also ends "_iter_0".
REAL_DIR     = str(DATA / "stark-shift")
REAL_PREFIX  = "PL-dual-gate-sweep_"
REAL_FRAMES  = 11
REAL_SPECTRAL = "PL-dual-gate-sweep_26_05_15_14_03_18_iter_0.csv"

# The only committed two-row file, used as a real-width negative case.
REAL_SPECTRUM = str(DATA / "single-spectra"
                         / "ref_single_spectrum_26_07_01_14_42_47.csv")


def _frame(tmp_path, name, fill) -> None:
    """Write one numeric-grid frame, filled with *fill* so it is identifiable."""
    np.savetxt(tmp_path / name, np.full(SHAPE, float(fill)), delimiter=",")


def _spectrum(tmp_path, name) -> None:
    """Write a two-row single spectrum: row 0 a wavelength axis, row 1 counts."""
    np.savetxt(tmp_path / name, np.vstack([np.linspace(650, 700, 8),
                                           np.arange(8.0)]), delimiter=",")


def _export(tmp_path, name, roles) -> None:
    """Write a one-block AttoCube export header, *roles* naming its columns."""
    header = ",".join(["Parameters Labels"] + list(roles))
    (tmp_path / name).write_text(f"{header}\nTemperature,4.2,0.0,0.0\n")


SPECTRAL_ROLES = ("Par_0", "Wavelength0", "ExpROI1_0", "ExpROI2_0")
TEMPORAL_ROLES = ("Par_0", "Wavelength0", "Exp_0")

# A signal-free corner whose four values are [1, 1, 1, 21]: median 1, mean 6.  The
# two statistics therefore give distinguishable pedestals, which a constant-filled
# frame could not show.
BG_REGION   = (slice(0, 2), slice(0, 2))
BG_MEDIAN   = 1.0
BG_MEAN     = 6.0
SIGNAL_FILL = 100.0


def _pedestal_frame(tmp_path, name) -> None:
    """Write a frame sitting on a pedestal, with a signal-free corner to measure it."""
    img = np.full(SHAPE, SIGNAL_FILL)
    img[BG_REGION] = [[1.0, 1.0], [1.0, 21.0]]
    np.savetxt(tmp_path / name, img, delimiter=",")


def _pedestal_scan(tmp_path, **kwargs):
    """
    A two-frame scan of pedestal frames, built with *kwargs* on the loader.

    Creates *tmp_path*, so a test wanting two independently-loaded scans can pass
    two subdirectories of its own ``tmp_path``.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in (0, 1):
        _pedestal_frame(tmp_path, f"pl_iter_{i}.csv")
    return ACImgSweep(tmp_path, prefix="pl_", **kwargs)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_iter_10_sorts_after_iter_2(tmp_path):
    # Lexicographic order would give 0, 10, 2 and pair every frame with the wrong
    # index. 3..9 are absent, so the gap warning is expected alongside the order.
    for i in (0, 2, 10):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    with pytest.warns(UserWarning, match="missing iteration"):
        scan = ACImgSweep(tmp_path, prefix="pl_")

    assert [f.stem for f in scan.files] == ["pl_iter_0", "pl_iter_2", "pl_iter_10"]
    # The assertion that actually matters: the frame at index 2 is iter_10's data.
    assert np.allclose(scan.load_frame(2), 10.0)


def test_mixed_padding_widths_order_numerically(tmp_path):
    # Padding width varies between exports, so it cannot be relied on: 4-digit
    # "0010" sorts before 6-digit "000002" as text, and after it as a number.
    _frame(tmp_path, "pl_iter_000002.csv", 2)
    _frame(tmp_path, "pl_iter_0010.csv", 10)
    with pytest.warns(UserWarning, match="missing iteration"):
        scan = ACImgSweep(tmp_path, prefix="pl_")

    assert np.allclose(scan.load_frame(0), 2.0)
    assert np.allclose(scan.load_frame(1), 10.0)


def test_consecutive_frames_do_not_warn(tmp_path):
    for i in (0, 1, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan = ACImgSweep(tmp_path, prefix="pl_")

    assert [str(c.message) for c in caught] == []
    assert scan.n_frames == 3


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_gap_in_frames_warns_and_does_not_close_up(tmp_path):
    for i in (0, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    with pytest.warns(UserWarning, match=r"missing iteration\(s\) \[1\]"):
        scan = ACImgSweep(tmp_path, prefix="pl_")

    # Nothing is dropped and nothing is invented; index 1 is iteration 2.
    assert scan.n_frames == 2
    assert np.allclose(scan.load_frame(1), 2.0)


def test_frames_without_iter_suffix_warn(tmp_path):
    for name in ("a", "b"):
        _frame(tmp_path, f"pl_{name}.csv", 0)
    with pytest.warns(UserWarning, match="no '_iter_N' suffix"):
        scan = ACImgSweep(tmp_path, prefix="pl_")

    assert [f.stem for f in scan.files] == ["pl_a", "pl_b"]


def test_two_runs_in_one_directory_warn_on_collision(tmp_path):
    # Two acquisitions dropped into one folder. Every file is a legitimate numeric
    # grid, so nothing upstream rejects them and the indices collide pairwise.
    for i in (0, 1, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
        _frame(tmp_path, f"pl_run2_iter_{i}.csv", 100 + i)
    with pytest.warns(UserWarning, match="claimed by more than one file") as caught:
        scan = ACImgSweep(tmp_path, prefix="pl_")

    # Names the files, not just the count — that is what says two runs were merged.
    assert "pl_run2_iter_0.csv" in str(caught[0].message)
    # Nothing is dropped and no winner is picked, so the frame count exceeds the
    # three distinct points and index i is not iteration i.
    assert scan.n_frames == 6


def test_collision_and_gap_both_warn(tmp_path):
    # Independent conditions: the gap check compares against the set of indices,
    # which a repeat leaves unchanged, so neither implies the other.
    _frame(tmp_path, "pl_iter_0.csv", 0)
    _frame(tmp_path, "pl_run2_iter_0.csv", 100)
    _frame(tmp_path, "pl_iter_2.csv", 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan = ACImgSweep(tmp_path, prefix="pl_")

    messages = " | ".join(str(c.message) for c in caught)
    assert "claimed by more than one file" in messages
    assert "missing iteration(s) [1]" in messages
    assert scan.n_frames == 3


def test_warning_points_at_the_caller(tmp_path):
    # stacklevel=3 at the call site. Without it the warning blames a line inside
    # loaders.py, which tells a researcher nothing about which scan complained.
    for i in (0, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    with pytest.warns(UserWarning) as caught:
        ACImgSweep(tmp_path, prefix="pl_")

    assert caught[0].filename == __file__


# ---------------------------------------------------------------------------
# The committed zero-padded sequences
# ---------------------------------------------------------------------------


def test_committed_sequence_loads_without_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan = ACImgSweep(REAL_DIR, prefix=REAL_PREFIX)

    assert [str(c.message) for c in caught] == []
    assert scan.n_frames == REAL_FRAMES
    assert scan.files[0].stem.endswith("iter_0000")
    assert scan.files[-1].stem.endswith("iter_0010")


def test_committed_single_spectrum_is_not_an_image(tmp_path):
    # The real committed two-row file, at its real width. Its first row is a
    # wavelength axis, so a first-line float test takes it for a frame.
    shutil.copy(REAL_SPECTRUM, tmp_path / "pl_iter_0.csv")

    assert _classify_csv(tmp_path / "pl_iter_0.csv") == "spectrum"


# ---------------------------------------------------------------------------
# Classification by row count and header
# ---------------------------------------------------------------------------


def test_three_rows_is_the_image_boundary(tmp_path):
    # A grid is an image from three rows up; two rows is a spectrum and one is
    # neither. All three are numeric on their first line, which is why the row
    # count has to do the work.
    for n_rows, expected in ((3, "image"), (2, "spectrum"), (1, "too_short")):
        name = f"grid_{n_rows}.csv"
        np.savetxt(tmp_path / name, np.ones((n_rows, 5)), delimiter=",")
        assert _classify_csv(tmp_path / name) == expected


def test_exports_are_named_by_their_block_layout(tmp_path):
    # Both are headed files that are not frames, but they are not the same file:
    # naming a TRPL export "spectral" would fork the layout vocabulary.
    _export(tmp_path, "pl_spectral.csv", SPECTRAL_ROLES)
    _export(tmp_path, "pl_temporal.csv", TEMPORAL_ROLES)
    _export(tmp_path, "pl_unknown.csv", ("Par_0", "Wavelength0", "Foo_0"))

    classify = _classify_csv
    assert classify(tmp_path / "pl_spectral.csv") == "spectral"
    assert classify(tmp_path / "pl_temporal.csv") == "temporal"
    assert classify(tmp_path / "pl_unknown.csv") == "unrecognised"


def test_directory_of_single_spectra_raises(tmp_path):
    # The A9 failure: every file here used to load as a 2xN "frame" and reach the
    # diffusion routines as one.
    for i in (0, 1, 2):
        _spectrum(tmp_path, f"pl_iter_{i}.csv")
    with pytest.raises(ValueError) as excinfo:
        ACImgSweep(tmp_path, prefix="pl_")

    message = str(excinfo.value)
    assert "SingleSpectrum" in message          # says where to take them instead
    assert "pl_iter_1.csv" in message           # and which files it means


def test_temporal_export_alone_names_the_trpl_loader(tmp_path):
    _export(tmp_path, "pl_iter_0.csv", TEMPORAL_ROLES)
    with pytest.raises(ValueError, match="ACTRPLSweep"):
        ACImgSweep(tmp_path, prefix="pl_")


# ---------------------------------------------------------------------------
# What is said about the files that were skipped
# ---------------------------------------------------------------------------


def test_stray_spectrum_among_frames_is_excluded_and_named(tmp_path):
    for i in (0, 1, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    _spectrum(tmp_path, "pl_ref.csv")
    with pytest.warns(UserWarning, match="not real-space images") as caught:
        scan = ACImgSweep(tmp_path, prefix="pl_")

    assert "pl_ref.csv" in str(caught[0].message)
    assert [f.stem for f in scan.files] == ["pl_iter_0", "pl_iter_1", "pl_iter_2"]


def test_export_among_frames_is_excluded_in_silence(tmp_path):
    # An acquisition writes its parameter export beside the frames every time, so
    # warning here would fire on every legitimate load.
    for i in (0, 1, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    _export(tmp_path, "pl_params.csv", SPECTRAL_ROLES)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan = ACImgSweep(tmp_path, prefix="pl_")

    assert [str(c.message) for c in caught] == []
    assert scan.n_frames == 3


def test_skip_warning_points_at_the_caller(tmp_path):
    # stacklevel=2, measured rather than read off the call stack: the warning is
    # useless if it blames a line inside loaders.py.
    for i in (0, 1):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    _spectrum(tmp_path, "pl_ref.csv")
    with pytest.warns(UserWarning) as caught:
        ACImgSweep(tmp_path, prefix="pl_")

    assert caught[0].filename == __file__


# ---------------------------------------------------------------------------
# Background subtraction
# ---------------------------------------------------------------------------


def test_load_frame_is_raw_even_when_a_bg_region_was_given(tmp_path):
    # The rule this class shares with the spectral loaders: the file's own counts
    # stay reachable, and the correction is a separate array.
    scan = _pedestal_scan(tmp_path, bg_region=BG_REGION)

    assert scan.load_frame(0)[BG_REGION].tolist() == [[1.0, 1.0], [1.0, 21.0]]
    assert np.allclose(scan.load_frame(0).max(), SIGNAL_FILL)


def test_load_frame_bg_subtracts_the_region_statistic(tmp_path):
    # median and mean differ by construction, so this pins which one ran rather
    # than merely that something was subtracted.
    med  = _pedestal_scan(tmp_path / "a", bg_region=BG_REGION)
    mean = _pedestal_scan(tmp_path / "b", bg_region=BG_REGION, bg_stat="mean")

    assert np.allclose(med.load_frame_bg(0).max(),  SIGNAL_FILL - BG_MEDIAN)
    assert np.allclose(mean.load_frame_bg(0).max(), SIGNAL_FILL - BG_MEAN)


def test_load_frame_bg_raises_without_a_bg_region(tmp_path):
    # Returning the raw frame would answer a different question in silence.
    scan = _pedestal_scan(tmp_path)
    with pytest.raises(ValueError, match="without a bg_region"):
        scan.load_frame_bg(0)


def test_unrecognised_bg_stat_is_rejected_at_construction(tmp_path):
    # _apply_bg_region falls through to the mean for anything but "median", so an
    # unchecked typo would silently change the estimator.
    with pytest.raises(ValueError, match="bg_stat 'mediam' is not recognised"):
        _pedestal_scan(tmp_path, bg_region=BG_REGION, bg_stat="mediam")


# ---------------------------------------------------------------------------
# _resolve_frame
# ---------------------------------------------------------------------------


def test_best_follows_whether_a_bg_region_was_set(tmp_path):
    with_bg    = _pedestal_scan(tmp_path / "a", bg_region=BG_REGION)
    without_bg = _pedestal_scan(tmp_path / "b")

    assert np.allclose(_resolve_frame(with_bg, 0, "best").max(),
                       SIGNAL_FILL - BG_MEDIAN)
    assert np.allclose(_resolve_frame(without_bg, 0, "best").max(), SIGNAL_FILL)


def test_raw_ignores_a_bg_region_and_bg_requires_one(tmp_path):
    with_bg    = _pedestal_scan(tmp_path / "a", bg_region=BG_REGION)
    without_bg = _pedestal_scan(tmp_path / "b")

    assert np.allclose(_resolve_frame(with_bg, 0, "raw").max(), SIGNAL_FILL)
    with pytest.raises(ValueError, match="not available on this scan"):
        _resolve_frame(without_bg, 0, "bg")


def test_unrecognised_frame_source_names_the_choices(tmp_path):
    scan = _pedestal_scan(tmp_path)
    with pytest.raises(ValueError, match=r"is not recognised.*'best'"):
        _resolve_frame(scan, 0, "subtracted")


def test_an_object_with_only_load_frame_degrades_to_raw():
    # The published duck-type is load_frame(idx) + n_frames, so "best" must not
    # require an accessor a stand-in has no reason to implement.
    class _Minimal:
        n_frames = 1
        def load_frame(self, idx):
            return np.full(SHAPE, SIGNAL_FILL)

    assert np.allclose(_resolve_frame(_Minimal(), 0, "best").max(), SIGNAL_FILL)


def test_spectral_companion_excluded_despite_iter_0_collision():
    # The timestamped spectral export in this directory also ends "_iter_0", so it
    # would collide on index 0 with iter_0000 if it ever passed the numeric-grid
    # check. It is excluded by content, which is what keeps the indices distinct.
    scan = ACImgSweep(REAL_DIR, prefix=REAL_PREFIX)

    assert REAL_SPECTRAL not in [f.name for f in scan.files]
    assert scan.n_frames == REAL_FRAMES
