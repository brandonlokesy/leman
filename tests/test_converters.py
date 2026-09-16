"""
Tests for leman.converters.

Frames are written synthetically into ``tmp_path``, so nothing here needs the
lab share.  Two cases carry the weight and neither exists on the branch this
module was ported from:

* a folder holding frames *and* a two-row spectrum must convert only the frames
  (**A9** — a first-line float test takes the spectrum for an image);
* an **unpadded** ``_iter_N`` sequence must stack in numeric order (**A7** —
  every committed export is zero-padded, which hides the lexicographic failure,
  and so was the branch's own fixture).

Each frame is filled with its own iteration number, so a test can assert *which*
file landed on a page rather than merely how many pages there are.
"""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from leman import converters
from leman.loaders import (
    ACSpectralSweep,
    ACTRPLSweep,
    DeviceGeometry,
    StackLayer,
)

from test_loaders import GATES, PARAMS, make_spectral_csv
from test_loaders_trpl import _synth_decay
from _paths import DATA

TRPL_DIR = str(DATA / "TRPL")

SHAPE = (4, 5)                      # (ny, nx) — small; no frame content is analysed


def _frame(directory, name, fill) -> np.ndarray:
    """Write one numeric-grid frame filled with *fill*, and return it."""
    img = np.full(SHAPE, float(fill))
    np.savetxt(Path(directory) / name, img, delimiter=",")
    return img


def _spectrum(directory, name) -> None:
    """Write a two-row single spectrum: row 0 a wavelength axis, row 1 counts."""
    np.savetxt(Path(directory) / name,
               np.vstack([np.linspace(650, 700, 8), np.arange(8.0)]),
               delimiter=",")


def _export(directory, name) -> None:
    """Write a one-block AttoCube spectral export header."""
    header = "Parameters Labels,Par_0,Wavelength0,ExpROI1_0,ExpROI2_0"
    (Path(directory) / name).write_text(f"{header}\nTemperature,4.2,0.0,0.0\n")


# ---------------------------------------------------------------------------
# A9 — a spectrum is not a frame
# ---------------------------------------------------------------------------

def test_spectrum_beside_frames_is_not_converted(tmp_path):
    # The ported branch selected files by parsing the first field of line 1 as a
    # float, which a wavelength axis passes.  The spectrum would have become a
    # 2 x 8 "image".
    for i in (0, 1, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    _spectrum(tmp_path, "pl_iter_3.csv")

    report = converters.convert_path(tmp_path)

    assert len(report.outputs) == 3
    assert [p.stem for p in report.outputs] == ["pl_iter_0", "pl_iter_1", "pl_iter_2"]
    assert {f.name: kind for f, kind in report.skipped.items()} == {
        "pl_iter_3.csv": "spectrum"
    }


def test_converting_a_spectrum_directly_is_refused(tmp_path):
    _spectrum(tmp_path, "spec.csv")

    with pytest.raises(ValueError) as excinfo:
        converters.convert_image_csv_to_tiff(tmp_path / "spec.csv")

    # The refusal says what the file is and where it should go instead, in the
    # same words the loaders use.
    assert "two-row single spectrum" in str(excinfo.value)
    assert "SingleSpectrum" in str(excinfo.value)


def test_an_export_beside_frames_is_skipped_not_stacked(tmp_path):
    for i in (0, 1):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    _export(tmp_path, "pl_parameters.csv")

    stack_path = converters.convert_image_dir_to_tiff_stack(tmp_path, out=tmp_path)

    assert tifffile.imread(stack_path).shape == (2, *SHAPE)


# ---------------------------------------------------------------------------
# A7 — acquisition order, not filename order
# ---------------------------------------------------------------------------

def test_unpadded_stack_is_in_acquisition_order(tmp_path):
    # iter_2 … iter_10 unpadded: lexicographically "10" sorts before "2", so a
    # filename-ordered stack would open with the last frame.
    fills = list(range(2, 11))
    for i in fills:
        _frame(tmp_path, f"pl_iter_{i}.csv", i)

    stack_path = converters.convert_image_dir_to_tiff_stack(
        tmp_path, prefix="pl_iter_", out=tmp_path)
    stack = tifffile.imread(stack_path)

    assert stack.shape == (len(fills), *SHAPE)
    # Every page identifies its own frame, so this pins the order and not just
    # the count.
    assert [int(page[0, 0]) for page in stack] == fills


def test_a_three_frame_stack_is_three_pages(tmp_path):
    # Left to guess, tifffile stores a (3, ny, nx) array as one RGB image with
    # separate colour planes, so a three-point sweep opens as a single colour
    # frame.  Three is the only count that triggers it, hence three here.
    for i in (0, 1, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)

    stack_path = converters.convert_image_dir_to_tiff_stack(tmp_path, out=tmp_path)

    with tifffile.TiffFile(stack_path) as tif:
        assert len(tif.pages) == 3
        assert tif.pages[0].photometric == tifffile.PHOTOMETRIC.MINISBLACK
    assert [int(page[0, 0]) for page in tifffile.imread(stack_path)] == [0, 1, 2]


def test_stack_is_named_after_the_prefix(tmp_path):
    for i in (0, 1):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)

    stack_path = converters.convert_image_dir_to_tiff_stack(
        tmp_path, prefix="pl_iter_", out=tmp_path)

    assert stack_path.name == "pl_iter_stack.tif"


def test_a_gap_warns_at_the_callers_line(tmp_path):
    # Measured, not read off a def line: the warning must be attributed to this
    # test file rather than to anything inside the package.
    for i in (0, 1, 3):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)

    with pytest.warns(UserWarning, match="missing iteration") as caught:
        converters.convert_image_dir_to_tiff_stack(tmp_path, out=tmp_path)

    assert Path(caught[0].filename).name == "test_converters.py"


def test_empty_directory_names_what_it_found(tmp_path):
    _spectrum(tmp_path, "spec.csv")

    with pytest.raises(ValueError) as excinfo:
        converters.convert_image_dir_to_tiff_stack(tmp_path)

    assert "No real-space image CSV" in str(excinfo.value)
    assert "spec.csv" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Pixel type
# ---------------------------------------------------------------------------

def test_integer_counts_become_uint16(tmp_path):
    img = _frame(tmp_path, "img.csv", 300)
    tif = converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    back = tifffile.imread(tif)
    assert back.dtype == np.uint16
    assert np.array_equal(back.astype(float), img)


def test_counts_above_uint16_become_uint32(tmp_path):
    # float32 cannot hold every integer past 2**24, so the fallback the branch
    # used here was not lossless.
    img = _frame(tmp_path, "img.csv", 70000)
    tif = converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    back = tifffile.imread(tif)
    assert back.dtype == np.uint32
    assert np.array_equal(back.astype(float), img)


def test_non_integer_counts_become_float32(tmp_path):
    _frame(tmp_path, "img.csv", 1.5)
    tif = converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    assert tifffile.imread(tif).dtype == np.float32


def test_dtype_can_be_forced(tmp_path):
    _frame(tmp_path, "img.csv", 300)
    tif = converters.convert_image_csv_to_tiff(
        tmp_path / "img.csv", out=tmp_path, dtype="float32")

    assert tifffile.imread(tif).dtype == np.float32


def test_unknown_dtype_is_refused(tmp_path):
    _frame(tmp_path, "img.csv", 1)

    with pytest.raises(ValueError, match="Unsupported image dtype"):
        converters.convert_image_csv_to_tiff(
            tmp_path / "img.csv", out=tmp_path, dtype="int8")


# ---------------------------------------------------------------------------
# Where output lands
# ---------------------------------------------------------------------------

def test_a_raw_folder_writes_to_its_sibling_converted(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _frame(raw, "img.csv", 1)

    tif = converters.convert_image_csv_to_tiff(raw / "img.csv")

    assert tif == tmp_path / "converted" / "img.tif"
    assert tif.is_file()


def test_any_other_folder_gets_a_converted_subfolder(tmp_path):
    _frame(tmp_path, "img.csv", 1)

    tif = converters.convert_image_csv_to_tiff(tmp_path / "img.csv")

    assert tif == tmp_path / "converted" / "img.tif"


def test_out_names_a_file_verbatim(tmp_path):
    _frame(tmp_path, "img.csv", 1)

    tif = converters.convert_image_csv_to_tiff(
        tmp_path / "img.csv", out=tmp_path / "elsewhere" / "named.tif")

    assert tif == tmp_path / "elsewhere" / "named.tif"
    assert tif.is_file()


def test_a_stack_lands_beside_the_frames_own_output(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in (0, 1):
        _frame(raw, f"pl_iter_{i}.csv", i)

    stack_path = converters.convert_image_dir_to_tiff_stack(raw)

    assert stack_path == tmp_path / "converted" / "raw_stack.tif"


# ---------------------------------------------------------------------------
# Overwriting
# ---------------------------------------------------------------------------

def test_an_existing_output_is_refused(tmp_path):
    _frame(tmp_path, "img.csv", 1)
    converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)


def test_overwrite_replaces_the_file(tmp_path):
    _frame(tmp_path, "img.csv", 1)
    converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    _frame(tmp_path, "img.csv", 7)
    tif = converters.convert_image_csv_to_tiff(
        tmp_path / "img.csv", out=tmp_path, overwrite=True)

    assert tifffile.imread(tif)[0, 0] == 7


def test_a_refused_conversion_creates_no_folder(tmp_path):
    _spectrum(tmp_path, "spec.csv")

    with pytest.raises(ValueError):
        converters.convert_image_csv_to_tiff(tmp_path / "spec.csv")

    assert not (tmp_path / "converted").exists()


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def test_recursive_run_keeps_folders_apart(tmp_path):
    # The same frame name in two sweeps.  Resolved once for the whole run, the
    # second would overwrite the first with nothing said.
    for name, fill in (("scan_a", 1), ("scan_b", 2)):
        raw = tmp_path / name / "raw"
        raw.mkdir(parents=True)
        _frame(raw, "pl_iter_0.csv", fill)

    report = converters.convert_path(tmp_path, recursive=True)

    assert len(report.outputs) == 2
    assert not report.errors
    assert tifffile.imread(tmp_path / "scan_a" / "converted" / "pl_iter_0.tif")[0, 0] == 1
    assert tifffile.imread(tmp_path / "scan_b" / "converted" / "pl_iter_0.tif")[0, 0] == 2


def test_recursive_stack_is_one_per_folder(tmp_path):
    for name in ("scan_a", "scan_b"):
        raw = tmp_path / name / "raw"
        raw.mkdir(parents=True)
        for i in (0, 1):
            _frame(raw, f"pl_iter_{i}.csv", i)

    report = converters.convert_path(tmp_path, recursive=True, stack=True)

    assert sorted(p.name for p in report.outputs) == ["raw_stack.tif", "raw_stack.tif"]
    assert {p.parent.parent.name for p in report.outputs} == {"scan_a", "scan_b"}


def test_a_batch_continues_past_a_failure(tmp_path):
    for i in (0, 1):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    # Pre-place one output so its conversion is refused while the other succeeds.
    (tmp_path / "converted").mkdir()
    (tmp_path / "converted" / "pl_iter_0.tif").write_bytes(b"")

    report = converters.convert_path(tmp_path)

    assert [p.stem for p in report.outputs] == ["pl_iter_1"]
    assert len(report.errors) == 1
    assert report.errors[0][0].name == "pl_iter_0.csv"
    assert "already exists" in report.errors[0][1]


def test_a_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        converters.convert_path(tmp_path / "nope")


def test_report_unpacks_as_a_tuple(tmp_path):
    _frame(tmp_path, "pl_iter_0.csv", 1)

    outputs, skipped, errors = converters.convert_path(tmp_path)

    assert len(outputs) == 1 and not skipped and not errors


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_writes_frames_and_returns_zero(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in (0, 1):
        _frame(raw, f"pl_iter_{i}.csv", i)

    assert converters.main([str(raw)]) == 0
    assert sorted(p.name for p in (tmp_path / "converted").iterdir()) == [
        "pl_iter_0.tif", "pl_iter_1.tif"
    ]


def test_cli_stack_writes_one_file(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in (0, 1):
        _frame(raw, f"pl_iter_{i}.csv", i)

    assert converters.main([str(raw), "--stack"]) == 0
    assert [p.name for p in (tmp_path / "converted").iterdir()] == ["raw_stack.tif"]


def test_cli_returns_one_on_failure(tmp_path):
    _frame(tmp_path, "pl_iter_0.csv", 1)
    converters.main([str(tmp_path)])

    # Second run has nothing to overwrite with, so every conversion is refused.
    assert converters.main([str(tmp_path)]) == 1


def test_cli_returns_one_on_a_missing_path(tmp_path):
    assert converters.main([str(tmp_path / "nope")]) == 1


# ---------------------------------------------------------------------------
# Spectral export -> HDF5
# ---------------------------------------------------------------------------

def test_spectral_csv_round_trips_through_the_archive(tmp_path):
    # The point of routing through the loader and write_sweep rather than a
    # private layout: the archive is the package's one format, so the loader
    # reopens it.  The dev/hdf5 branch's version wrote an .h5 that could not be.
    csv = tmp_path / "sweep.csv"
    make_spectral_csv(csv)

    h5 = converters.convert_spectral_csv_to_hdf5(csv, "PL", out=tmp_path)
    assert h5.suffix == ".h5"

    original = ACSpectralSweep(csv, spectra_type="PL")
    reopened = ACSpectralSweep(h5, spectra_type="PL")

    assert np.array_equal(reopened.wavelength, original.wavelength)
    assert np.array_equal(reopened.spectra_roi1, original.spectra_roi1)
    assert np.array_equal(reopened.spectra_roi2, original.spectra_roi2)
    assert reopened.parameter_labels == original.parameter_labels
    for label in original.parameter_labels:
        assert np.array_equal(reopened[label], original[label])


def test_the_archive_takes_declarations_it_was_not_converted_with(tmp_path):
    # Only spectra_type is declared at conversion time.  This is what makes that
    # safe: the loader takes sweep, gates and geometry from its arguments and only
    # falls back to stored metadata, so the physics is declared at read time
    # exactly as it is for the CSV.
    csv = tmp_path / "sweep.csv"
    make_spectral_csv(csv)
    h5 = converters.convert_spectral_csv_to_hdf5(csv, "PL", out=tmp_path)

    bare = ACSpectralSweep(h5, spectra_type="PL")
    assert bare.sweep_type == "index"            # nothing was claimed for it

    geom = DeviceGeometry(
        tmdc_stack   = [StackLayer("MoSe2"), StackLayer("WSe2", n_layers=2)],
        d_hbn_top    = 53.0,
        d_hbn_bottom = 46.0,
    )
    declared = ACSpectralSweep(
        h5, spectra_type="PL", sweep="V_A", gates=GATES, geometry=geom)

    assert declared.sweep_type == "V_A"
    assert np.array_equal(declared.v_top, PARAMS["V_A"])
    assert declared.ef is not None               # needs both gates and a geometry


def test_spectra_type_is_required_and_says_so(tmp_path):
    csv = tmp_path / "sweep.csv"
    make_spectral_csv(csv)

    with pytest.raises(ValueError) as excinfo:
        converters.convert_spectral_csv_to_hdf5(csv, None, out=tmp_path)

    # The loader's own message, reached through its own resolver, so there is no
    # second copy of the wording to drift.
    assert "spectra_type is required" in str(excinfo.value)


def test_an_unknown_spectra_type_is_refused_before_the_decode(tmp_path):
    csv = tmp_path / "sweep.csv"
    make_spectral_csv(csv)

    with pytest.raises(ValueError, match="not a recognised measurement type"):
        converters.convert_spectral_csv_to_hdf5(csv, "PLL", out=tmp_path)

    assert not (tmp_path / "converted").exists()


def test_converting_a_frame_as_spectral_is_refused(tmp_path):
    _frame(tmp_path, "img.csv", 1)

    with pytest.raises(ValueError) as excinfo:
        converters.convert_spectral_csv_to_hdf5(tmp_path / "img.csv", "PL")

    assert "not a spectral export CSV" in str(excinfo.value)


def test_an_archive_lands_in_converted(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_spectral_csv(raw / "sweep.csv")

    h5 = converters.convert_spectral_csv_to_hdf5(raw / "sweep.csv", "PL")

    assert h5 == tmp_path / "converted" / "sweep.h5"


def test_an_existing_archive_is_refused(tmp_path):
    csv = tmp_path / "sweep.csv"
    make_spectral_csv(csv)
    converters.convert_spectral_csv_to_hdf5(csv, "PL", out=tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        converters.convert_spectral_csv_to_hdf5(csv, "PL", out=tmp_path)


# ---------------------------------------------------------------------------
# A mixed folder: frames beside a spectral export, which is what is exported
# ---------------------------------------------------------------------------

def test_a_mixed_folder_writes_both_kinds(tmp_path):
    for i in (0, 1):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    make_spectral_csv(tmp_path / "sweep.csv")

    report = converters.convert_path(tmp_path, spectra_type="PL")

    assert sorted(p.name for p in report.outputs) == [
        "pl_iter_0.tif", "pl_iter_1.tif", "sweep.h5"
    ]
    assert not report.errors


def test_without_spectra_type_the_frames_still_convert(tmp_path):
    for i in (0, 1):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    make_spectral_csv(tmp_path / "sweep.csv")

    report = converters.convert_path(tmp_path)

    assert sorted(p.name for p in report.outputs) == ["pl_iter_0.tif", "pl_iter_1.tif"]
    assert len(report.errors) == 1
    assert report.errors[0][0].name == "sweep.csv"
    assert "spectra_type is required" in report.errors[0][1]


# ---------------------------------------------------------------------------
# TRPL: a directory is one sweep
# ---------------------------------------------------------------------------

def _trpl_dir(directory, prefix="TRPL_", n=3):
    """Write *n* synthetic decays sharing one time axis into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _synth_decay(directory / f"{prefix}iter_{i}.csv",
                     params={"V_A": float(i), "Excitation Power": 1e-4})
    return directory


def test_trpl_directory_becomes_one_archive(tmp_path):
    folder = _trpl_dir(tmp_path / "raw")

    h5 = converters.convert_trpl_dir_to_hdf5(folder)

    assert h5 == tmp_path / "converted" / "raw.h5"
    original = ACTRPLSweep(folder)
    reopened = ACTRPLSweep(h5)
    assert np.array_equal(reopened.time, original.time)
    assert np.array_equal(reopened.decays, original.decays)
    assert reopened.n_sweeps == 3


def test_a_named_trpl_directory_converts(tmp_path):
    folder = _trpl_dir(tmp_path / "sweep")

    report = converters.convert_path(folder)

    assert [p.name for p in report.outputs] == ["sweep.h5"]
    assert not report.errors


def test_a_discovered_trpl_directory_defers(tmp_path):
    # Which files form one sweep is a declaration, not something to infer, so a
    # folder reached by recursion is reported rather than guessed at.
    folder = _trpl_dir(tmp_path / "sweep")

    report = converters.convert_path(tmp_path, recursive=True)

    assert not report.outputs
    assert report.skipped == {folder: "trpl_directory"}


def test_prefix_picks_one_of_two_measurements_sharing_a_folder(tmp_path):
    # The right_spots case: two measurements in one folder claim the same
    # _iter_N indices, so without a prefix they merge into a single wrong sweep.
    folder = tmp_path / "raw"
    _trpl_dir(folder, prefix="right1_")
    _trpl_dir(folder, prefix="right2_")

    h5 = converters.convert_trpl_dir_to_hdf5(folder, prefix="right1_")

    assert h5.name == "right1.h5"
    assert ACTRPLSweep(h5).n_sweeps == 3        # not 6


def test_a_committed_trpl_folder_writes_one_archive_not_two(tmp_path):
    # The real folder holds an IRF, a laser-spot image and the sweep's
    # parameter-table companion.  The companion has a spectral header, and must be
    # read as part of the sweep rather than converted as a sweep of its own.
    report = converters.convert_path(TRPL_DIR, out=tmp_path, spectra_type="PL")

    archives = [p for p in report.outputs if p.suffix == ".h5"]
    assert len(archives) == 1
    assert ACTRPLSweep(archives[0]).n_sweeps == 3


# ---------------------------------------------------------------------------
# out= is an output root: the tree is mirrored beneath it
# ---------------------------------------------------------------------------

def _spot_tree(root):
    """A raw/spot<N>/<measurement>/ tree, as a benchmarking run produces."""
    for spot in ("spot01", "spot02"):
        sweep_dir = root / "raw" / spot / "01-PL-Vbot"
        ref_dir   = root / "raw" / spot / "ref"
        sweep_dir.mkdir(parents=True)
        ref_dir.mkdir(parents=True)
        make_spectral_csv(sweep_dir / "sweep.csv")
        # The same reference filename under two spots — the case that collides
        # when a tree is flattened into one folder.
        _frame(ref_dir, "laser_ref.csv", 1)
    return root / "raw"


def test_out_mirrors_the_source_tree(tmp_path):
    raw = _spot_tree(tmp_path)

    report = converters.convert_path(
        raw, out=tmp_path / "converted", recursive=True, spectra_type="PL")

    assert not report.errors
    assert sorted(p.relative_to(tmp_path / "converted").as_posix()
                  for p in report.outputs) == [
        "spot01/01-PL-Vbot/sweep.h5",
        "spot01/ref/laser_ref.tif",
        "spot02/01-PL-Vbot/sweep.h5",
        "spot02/ref/laser_ref.tif",
    ]


def test_a_repeated_filename_survives_the_mirror(tmp_path):
    # Flattened into one folder these two overwrite, or refuse.  Mirrored, each
    # keeps its own place, which is the whole reason out= is a root.
    raw = _spot_tree(tmp_path)

    report = converters.convert_path(
        raw, out=tmp_path / "converted", recursive=True, spectra_type="PL")

    refs = sorted(p for p in report.outputs if p.name == "laser_ref.tif")
    assert len(refs) == 2
    assert {p.parent.parent.name for p in refs} == {"spot01", "spot02"}


def test_a_non_recursive_run_addresses_out_directly(tmp_path):
    # One folder, so there is no tree to mirror and out= is the folder itself.
    (tmp_path / "raw").mkdir()
    for i in (0, 1):
        _frame(tmp_path / "raw", f"pl_iter_{i}.csv", i)

    report = converters.convert_path(tmp_path / "raw", out=tmp_path / "converted")

    assert sorted(p.parent for p in report.outputs) == [tmp_path / "converted"] * 2


def test_pointing_at_raw_mirrors_into_its_sibling(tmp_path):
    # No out= at all.  The folder named in the call is "raw", so its sibling
    # converted/ is the root and the tree goes under it — which is what keeps a
    # nested experiment from scattering converted/ folders inside raw/.
    raw = _spot_tree(tmp_path)

    report = converters.convert_path(raw, recursive=True, spectra_type="PL")

    assert not report.errors
    assert sorted(p.relative_to(tmp_path / "converted").as_posix()
                  for p in report.outputs) == [
        "spot01/01-PL-Vbot/sweep.h5",
        "spot01/ref/laser_ref.tif",
        "spot02/01-PL-Vbot/sweep.h5",
        "spot02/ref/laser_ref.tif",
    ]


def test_an_uppercase_raw_counts(tmp_path):
    (tmp_path / "RAW").mkdir()
    make_spectral_csv(tmp_path / "RAW" / "sweep.csv")

    report = converters.convert_path(tmp_path / "RAW", spectra_type="PL")

    assert report.outputs == [tmp_path / "converted" / "sweep.h5"]


def test_pointing_inside_raw_does_not_search_upward(tmp_path):
    # The deliberate limit of the default: only the folder actually named is
    # examined, so the destination is readable off the call.  from_raw= is the
    # opt-in that looks further, and the next test is its half of the pair.
    raw = _spot_tree(tmp_path)

    report = converters.convert_path(
        raw / "spot01" / "01-PL-Vbot", spectra_type="PL")

    assert report.outputs == [
        raw / "spot01" / "01-PL-Vbot" / "converted" / "sweep.h5"
    ]


def test_from_raw_searches_upward_from_a_measurement_folder(tmp_path):
    raw = _spot_tree(tmp_path)

    report = converters.convert_path(
        raw / "spot01" / "01-PL-Vbot", spectra_type="PL", from_raw=True)

    assert report.outputs == [
        tmp_path / "converted" / "spot01" / "01-PL-Vbot" / "sweep.h5"
    ]


def test_from_raw_places_a_single_file(tmp_path):
    # The case the default cannot help with at all: nothing was named but the
    # file, so there is no folder to read.
    raw = _spot_tree(tmp_path)

    report = converters.convert_path(
        raw / "spot01" / "01-PL-Vbot" / "sweep.csv",
        spectra_type="PL", from_raw=True)

    assert report.outputs == [
        tmp_path / "converted" / "spot01" / "01-PL-Vbot" / "sweep.h5"
    ]


def test_from_raw_warns_and_falls_back_when_there_is_no_raw(tmp_path):
    folder = tmp_path / "mydata"
    folder.mkdir()
    make_spectral_csv(folder / "sweep.csv")

    with pytest.warns(UserWarning, match="no folder called 'raw'"):
        report = converters.convert_path(
            folder, spectra_type="PL", from_raw=True)

    assert report.outputs == [folder / "converted" / "sweep.h5"]


def test_beside_writes_next_to_the_source(tmp_path):
    # For a targeted conversion: no converted/ level at all.  The committed
    # examples/data archives sit beside their CSVs this way.
    raw = _spot_tree(tmp_path)
    src = raw / "spot01" / "01-PL-Vbot" / "sweep.csv"

    report = converters.convert_path(src, spectra_type="PL", beside=True)

    assert report.outputs == [src.with_suffix(".h5")]
    assert not (src.parent / "converted").exists()


def test_beside_over_a_tree_leaves_every_output_with_its_source(tmp_path):
    raw = _spot_tree(tmp_path)

    report = converters.convert_path(
        raw, recursive=True, spectra_type="PL", beside=True)

    assert not report.errors
    assert sorted(p.relative_to(raw).as_posix() for p in report.outputs) == [
        "spot01/01-PL-Vbot/sweep.h5",
        "spot01/ref/laser_ref.tif",
        "spot02/01-PL-Vbot/sweep.h5",
        "spot02/ref/laser_ref.tif",
    ]
    assert not list(raw.rglob("converted"))


def test_beside_is_the_same_answer_as_out_naming_that_folder(tmp_path):
    # It is shorthand, not a second placement rule.  If these ever diverge, one
    # of them has grown a meaning of its own.
    raw = _spot_tree(tmp_path)
    src = raw / "spot01" / "01-PL-Vbot" / "sweep.csv"

    by_beside = converters.convert_path(src, spectra_type="PL", beside=True)
    by_out = converters.convert_path(
        src, spectra_type="PL", out=src.parent, overwrite=True)

    assert by_beside.outputs == by_out.outputs


def test_beside_and_out_together_are_refused(tmp_path):
    raw = _spot_tree(tmp_path)

    with pytest.raises(ValueError, match="at most one of out=, from_raw= and beside="):
        converters.convert_path(
            raw, out=tmp_path / "elsewhere", beside=True, spectra_type="PL")


def test_beside_and_from_raw_together_are_refused(tmp_path):
    raw = _spot_tree(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        converters.convert_path(raw, from_raw=True, beside=True, spectra_type="PL")

    # The message names which two were given, not just that there was a clash.
    assert "from_raw=, beside=" in str(excinfo.value)


def test_out_and_from_raw_together_are_refused(tmp_path):
    raw = _spot_tree(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        converters.convert_path(
            raw, out=tmp_path / "elsewhere", from_raw=True, spectra_type="PL")

    assert "out=, from_raw=" in str(excinfo.value)


def test_the_cli_reports_the_refusal_rather_than_raising(tmp_path):
    raw = _spot_tree(tmp_path)

    assert converters.main(
        [str(raw), "--out", str(tmp_path / "elsewhere"), "--from-raw"]) == 1


def test_out_naming_a_file_is_used_verbatim(tmp_path):
    # Meaningful with a stack, which is one output for the whole folder.
    (tmp_path / "raw").mkdir()
    for i in (0, 1):
        _frame(tmp_path / "raw", f"pl_iter_{i}.csv", i)

    report = converters.convert_path(
        tmp_path / "raw", out=tmp_path / "mystack.tif", stack=True)

    assert report.outputs == [tmp_path / "mystack.tif"]


def test_the_cli_reports_a_deferred_directory(tmp_path, capsys):
    _trpl_dir(tmp_path / "sweep")

    assert converters.main([str(tmp_path), "--recursive"]) == 0
    out = capsys.readouterr().out
    assert "deferred" in out
    assert "name it on its own" in out
