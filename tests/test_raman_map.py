"""
Tests for RamanMap.

The real map (examples/data/Raman/map2.txt) is a 10 x 8 grid, Y fast/inner
and X slow/outer -- confirmed by reading its rows, not assumed. Synthetic
files here cover shuffled row order (RamanMap indexes by each row's own
(X, Y), not file position), an incomplete/aborted grid, and a repeated
position, which the row count alone cannot detect.
"""

import numpy as np
import pytest

from tmdc_optics_tools.loaders import RamanMap

from _paths import DATA

MAP_PATH = str(DATA / "Raman" / "map2.txt")


def _write_map(path, xs, ys, shift, counts_fn, shuffle=False, drop_last=False,
               duplicate=False):
    """
    counts_fn(x, y) -> 1-D array of length len(shift), the spectrum at (x, y).
    """
    rows = [(x, y) for x in xs for y in ys]  # X outer, Y inner, as in the real export
    if duplicate:
        # Repeat the first position in place of the last. The row count and both
        # unique-axis counts are unchanged, so the complete-grid guard still
        # passes -- while one cell gets two rows and one gets none.
        rows[-1] = rows[0]
    if shuffle:
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in rng.permutation(len(rows))]
    if drop_last:
        rows = rows[:-1]

    lines = ["#Acq. time (s)=\t3"]
    shift_row = ["", ""] + [f"{v}" for v in shift]
    lines.append("\t".join(shift_row))
    for x, y in rows:
        counts = counts_fn(x, y)
        lines.append("\t".join([f"{x}", f"{y}"] + [f"{c}" for c in counts]))
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("latin-1"))


SHIFT = np.linspace(100.0, 200.0, 5)
XS = [0.0, 1.0, 2.0]
YS = [0.0, 1.0]


def _counts_fn(x, y):
    # Each (x, y) gets a distinct, recoverable spectrum: baseline = 100*x + 10*y.
    return SHIFT * 0.0 + (100.0 * x + 10.0 * y)


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


def test_grid_shape_and_axes(tmp_path):
    path = tmp_path / "map.txt"
    _write_map(path, XS, YS, SHIFT, _counts_fn)

    m = RamanMap(path)
    assert m.n_x == len(XS) and m.n_y == len(YS)
    assert np.array_equal(m.x, XS)
    assert np.array_equal(m.y, YS)
    assert m.shift.size == SHIFT.size
    assert m.counts.shape == (len(YS), len(XS), SHIFT.size)


def test_spectrum_at_matches_the_encoded_position(tmp_path):
    path = tmp_path / "map.txt"
    _write_map(path, XS, YS, SHIFT, _counts_fn)
    m = RamanMap(path)

    for ix, x in enumerate(XS):
        for iy, y in enumerate(YS):
            expected = 100.0 * x + 10.0 * y
            assert np.allclose(m.spectrum_at(ix, iy), expected)


def test_shuffled_row_order_still_indexes_correctly(tmp_path):
    # RamanMap must place each row by its own (X, Y), not by file position.
    path = tmp_path / "map.txt"
    _write_map(path, XS, YS, SHIFT, _counts_fn, shuffle=True)
    m = RamanMap(path)

    for ix, x in enumerate(XS):
        for iy, y in enumerate(YS):
            expected = 100.0 * x + 10.0 * y
            assert np.allclose(m.spectrum_at(ix, iy), expected)


def test_incomplete_grid_raises(tmp_path):
    path = tmp_path / "map.txt"
    _write_map(path, XS, YS, SHIFT, _counts_fn, drop_last=True)
    with pytest.raises(ValueError, match="not a complete grid"):
        RamanMap(path)


def test_duplicate_scan_position_is_refused_and_named(tmp_path):
    # XS x YS = 6 cells and the file still has 6 rows, so the complete-grid
    # guard passes: (0.0, 0.0) is written twice and (2.0, 1.0) never. Before
    # the position check, that cell held uninitialised memory and plotted as
    # data.
    path = tmp_path / "map.txt"
    _write_map(path, XS, YS, SHIFT, _counts_fn, duplicate=True)
    with pytest.raises(ValueError, match=r"repeats scan position.*\(0, 0\)"):
        RamanMap(path)


def test_a_complete_grid_leaves_no_cell_unwritten(tmp_path):
    # counts is NaN-filled before the rows are scattered in, so a cell nothing
    # wrote reads as absent. On a complete grid nothing may be NaN -- shuffled,
    # because that is the path where the scatter indices do the real work.
    path = tmp_path / "map.txt"
    _write_map(path, XS, YS, SHIFT, _counts_fn, shuffle=True)
    m = RamanMap(path)
    assert np.isfinite(m.counts).all()


def test_no_data_rows_raises(tmp_path):
    path = tmp_path / "map.txt"
    path.write_text("#Acq. time (s)=\t3\n")
    with pytest.raises(ValueError, match="no data rows"):
        RamanMap(path)


# ---------------------------------------------------------------------------
# Real file
# ---------------------------------------------------------------------------


def test_real_map_shape_and_units():
    m = RamanMap(MAP_PATH)
    assert m.n_x == 10 and m.n_y == 8
    assert m.x.min() == pytest.approx(-10.6649, abs=1e-3)
    assert m.x.max() == pytest.approx(0.112262, abs=1e-3)
    assert m.y.min() == pytest.approx(0.08981, abs=1e-3)
    assert m.y.max() == pytest.approx(14.7288, abs=1e-3)
    assert m.shift.size == 1024


def test_real_map_leaves_no_cell_unwritten():
    m = RamanMap(MAP_PATH)
    assert np.isfinite(m.counts).all()


def test_real_map_spot_checks_against_hand_parsed_rows():
    m = RamanMap(MAP_PATH)
    main_region = (m.shift >= 245) & (m.shift <= 255)

    ix = np.searchsorted(m.x, 0.112262)
    iy = np.searchsorted(m.y, 2.1811)
    assert m.spectrum_at(ix, iy)[main_region].max() == pytest.approx(14037.0)

    ix2 = np.searchsorted(m.x, -10.6649)
    iy2 = np.searchsorted(m.y, 14.7288)
    assert m.spectrum_at(ix2, iy2)[main_region].max() == pytest.approx(60809.0)


# ---------------------------------------------------------------------------
# nearest_index
# ---------------------------------------------------------------------------


def test_nearest_index_exact_match(tmp_path):
    path = tmp_path / "map.txt"
    _write_map(path, XS, YS, SHIFT, _counts_fn)
    m = RamanMap(path)
    for ix, x in enumerate(XS):
        for iy, y in enumerate(YS):
            assert m.nearest_index(x, y) == (ix, iy)


def test_nearest_index_snaps_to_the_closest_grid_point(tmp_path):
    path = tmp_path / "map.txt"
    _write_map(path, XS, YS, SHIFT, _counts_fn)
    m = RamanMap(path)
    # XS = [0.0, 1.0, 2.0], YS = [0.0, 1.0] -- pick points that are not exact.
    assert m.nearest_index(0.4, 0.1) == (0, 0)
    assert m.nearest_index(1.6, 0.9) == (2, 1)


def test_nearest_index_on_the_real_map():
    m = RamanMap(MAP_PATH)
    assert m.nearest_index(0.11, 2.18) == (9, 1)
    assert m.nearest_index(-10.7, 14.8) == (0, 7)
