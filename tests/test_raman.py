"""
Tests for RamanSpectrum.

Real LabRAM exports (examples/data/Raman/) have a `#`-prefixed header block
whose length varies file to file, and are written in latin-1 (the degree
sign in "#Detector temperature (°C)=" is not valid UTF-8) -- both are
exercised directly against those files. Synthetic files cover both routes to
the map-export refusal: the field count read before the parse, and the array
shape checked after it.
"""

import numpy as np
import pytest

from leman.loaders import RamanSpectrum

from _paths import DATA

RAMAN_DIR = str(DATA / "Raman")


def _write_spectrum(path, n_header_lines=3, degree_sign=False):
    lines = [f"#Setting {i}=\tvalue" for i in range(n_header_lines)]
    if degree_sign:
        lines.append("#Detector temperature (\xb0C)=\t-60.1")
    shift  = np.linspace(100.0, 200.0, 10)
    counts = np.arange(10, dtype=float)
    for s, c in zip(shift, counts):
        lines.append(f"{s}\t{c}")
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("latin-1"))
    return shift, counts


# ---------------------------------------------------------------------------
# Real files -- varying header length, latin-1 encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fname", [
    "unstrained_bilayer.txt", "strained_bilayer1.txt", "strained_bilayer2.txt",
])
def test_real_files_load_with_the_same_shift_axis(fname):
    s = RamanSpectrum(f"{RAMAN_DIR}/{fname}")
    assert s.shift.size == s.counts.size == 1024
    assert s.shift[0] == pytest.approx(36.5878)
    assert s.shift[-1] == pytest.approx(554.669)
    assert np.all(np.diff(s.shift) > 0)


def test_header_length_difference_does_not_shift_the_data():
    # strained_bilayer1.txt has 2 extra "#Peaks:Edit=" lines vs. the other
    # two -- a fixed-row-count skip would misalign it relative to them.
    a = RamanSpectrum(f"{RAMAN_DIR}/unstrained_bilayer.txt")
    b = RamanSpectrum(f"{RAMAN_DIR}/strained_bilayer1.txt")
    assert np.array_equal(a.shift, b.shift)


def test_map_export_is_rejected_not_misread():
    # Matched on the class's own wording. A bare pytest.raises(ValueError) --
    # which this was -- is also satisfied by numpy's "number of columns changed
    # from 1024 to 1026 at row 2", which is what this file used to raise, so it
    # passed while the written message was unreachable.
    with pytest.raises(ValueError, match="use RamanMap"):
        RamanSpectrum(f"{RAMAN_DIR}/map2.txt")


# ---------------------------------------------------------------------------
# Synthetic files
# ---------------------------------------------------------------------------


def test_variable_header_length_is_skipped_by_content(tmp_path):
    shift, counts = _write_spectrum(tmp_path / "short_header.txt", n_header_lines=1)
    _write_spectrum(tmp_path / "long_header.txt", n_header_lines=20)

    s_short = RamanSpectrum(tmp_path / "short_header.txt")
    s_long  = RamanSpectrum(tmp_path / "long_header.txt")

    assert np.allclose(s_short.shift, shift)
    assert np.allclose(s_short.counts, counts)
    assert np.array_equal(s_short.shift, s_long.shift)


def test_degree_sign_in_header_does_not_break_loading(tmp_path):
    path = tmp_path / "with_degree.txt"
    shift, counts = _write_spectrum(path, degree_sign=True)
    s = RamanSpectrum(path)
    assert np.allclose(s.shift, shift)


def test_no_header_at_all_still_loads(tmp_path):
    path = tmp_path / "no_header.txt"
    shift = np.linspace(50.0, 60.0, 5)
    counts = np.arange(5, dtype=float)
    path.write_text("\n".join(f"{s}\t{c}" for s, c in zip(shift, counts)) + "\n")
    s = RamanSpectrum(path)
    assert np.allclose(s.shift, shift)
    assert np.allclose(s.counts, counts)


def test_extra_columns_raise_a_clear_error(tmp_path):
    # A spatial-map-shaped row: (X, Y, counts_0, counts_1, ...) -- not a
    # single spectrum, must not be silently truncated to the first 2 columns.
    path = tmp_path / "map_like.txt"
    path.write_text("0.0\t0.0\t1\t2\t3\n0.0\t1.0\t4\t5\t6\n")
    with pytest.raises(ValueError, match="spatial-map"):
        RamanSpectrum(path)


def test_map_shaped_first_row_is_refused_before_the_parse(tmp_path):
    # A map's first row is two *empty* tab fields then the shift axis. The
    # default whitespace delimiter collapses them, so np.loadtxt sees fewer
    # columns on row 0 than on row 1 and raises its own error before the shape
    # check can name RamanMap -- which is why the field count is read first.
    path = tmp_path / "ragged_first_row.txt"
    path.write_text(
        "#Acq. time (s)=\t3\n"
        "\t\t100.0\t200.0\t300.0\n"
        "0.0\t0.0\t1\t2\t3\n"
        "0.0\t1.0\t4\t5\t6\n"
    )
    with pytest.raises(ValueError, match="fields on its first data line"):
        RamanSpectrum(path)


def test_space_separated_wide_body_is_refused_after_the_parse(tmp_path):
    # Tab-splitting the first line gives 1 field here, so the pre-parse count
    # cannot see this one and the post-parse shape check is what refuses it.
    # Both guards are load-bearing, and this is the case that keeps the second
    # reachable.
    path = tmp_path / "spaced.txt"
    path.write_text("0.0 0.0 1 2 3\n0.0 1.0 4 5 6\n")
    with pytest.raises(ValueError, match="Expected 2 columns"):
        RamanSpectrum(path)
