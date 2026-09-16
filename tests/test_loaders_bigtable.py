"""`BTSpectralSweep` — the headerless export, and what it refuses.

Two kinds of fixture, with a division of labour worth stating.

**The two committed exports** are what the loader was established from, and they
are the only thing that can catch a misreading shared by the parser and a
synthetic writer built from the same assumption. They pin the positional
parameter map against rows read by hand, and they pin the two independent facts
the filenames assert — that `PL_Doping` sweeps its two gates together and
`PL_Ez` sweeps them oppositely — which the loader works out from the numbers
without being told.

The strongest of these compares the loader against an **independent slicing** of
the same file, written from the MATLAB the format was described by rather than
from this package's parser. That is the one check a shared misunderstanding
cannot pass.

**Synthetic files** cover every refusal and every warning, because none of them
can be reached with a good file. Each is matched on the class's own wording: a
bare `pytest.raises(ValueError)` is also satisfied by pandas failing on a ragged
row, which is how an unreachable refusal once passed for months (A29).

`stacklevel` is measured here, not read off the `def` lines. The chain runs
caller → `__init__` → `_decode_and_describe` → `BTSpectralSweep._decode`
→ `_Sweep._decode` → `_decode_csv` → the warning, and the easy frame to miss is
this class's own `_decode` override.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from leman.loaders import (
    ACSpectralSweep,
    BTSpectralSweep,
    DeviceGeometry,
)

from _paths import DATA

BIG = DATA / "big-table-spectral"
DOPING = str(BIG / "PL_Doping_1LWSe2_p20uW_exp1sec_250508_190421.csv")
EZ = str(BIG / "PL_Ez_1LWSe2_p20uW_exp1sec_250508_190009.csv")

N_PIXELS = 1340
WL_FIRST, WL_LAST = 702.02690, 977.97310

# The wiring is per sample on this setup, so every test that touches an electrode
# declares it. Gate 2 as the top gate is a choice made for these tests, not a
# fact about the file.
GATES = {"top": "Gate Voltage 2", "bottom": "Gate Voltage 1", "channel": None}
GEOM = DeviceGeometry.from_single("WSe2", d_hbn_top=30, d_hbn_bottom=26)


@pytest.fixture
def ez():
    return BTSpectralSweep(EZ, spectra_type="PL", gates=GATES)


# ---------------------------------------------------------------------------
# Real files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path, n_sweeps", [(DOPING, 61), (EZ, 101)])
def test_shape_and_range(path, n_sweeps):
    scan = BTSpectralSweep(path, spectra_type="PL")
    assert scan.spectra.shape == (N_PIXELS, n_sweeps)
    assert scan.n_pixels == N_PIXELS
    assert scan.n_sweeps == n_sweeps
    assert scan.wavelength[0] == pytest.approx(WL_FIRST)
    assert scan.wavelength[-1] == pytest.approx(WL_LAST)
    # Ascending, because every downstream selector assumes it.
    assert np.all(np.diff(scan.wavelength) > 0)
    assert np.all(np.diff(scan.energy) > 0)


def test_the_positional_row_map_against_rows_read_by_hand():
    # Read out of the file with awk, not with this parser. 1-based row numbers,
    # as the acquisition program and dev/instruments/big-table.md count them.
    scan = BTSpectralSweep(EZ, spectra_type="PL")
    p = scan.parameters
    assert p["X Position"][0] == pytest.approx(59.0)      # row 1
    assert p["Y Position"][0] == pytest.approx(108.0)     # row 2
    assert p["Z Position"][0] == pytest.approx(23.0)      # row 3
    assert p["Gate Voltage 1"][0] == pytest.approx(-4.99935)   # row 4
    assert p["Gate Voltage 1"][-1] == pytest.approx(4.96901)
    assert p["Gate Voltage 2"][0] == pytest.approx(5.60494)    # row 6
    assert p["Gate Voltage 2"][-1] == pytest.approx(-5.58247)
    assert p["Reflection"][0] == pytest.approx(-4.78949)       # row 8
    assert np.all(p["Gate Voltage 3"] == 0.0)                  # row 9
    assert p["Excitation Power"][0] == pytest.approx(21.34305)  # row 11


def test_unidentified_rows_keep_a_positional_name():
    scan = BTSpectralSweep(EZ, spectra_type="PL")
    # Exactly the block this loader exposes, no more and no less.
    assert len(scan.parameters) == 30
    # Rows 10, 12, 13, 17 and 30 are not claimed to be anything, but they are
    # still readable -- an unnamed row is the point, a missing one is not.
    for n in (10, 12, 13, 17, 30):
        assert f"Row {n}" in scan.parameters
    assert scan.parameters["Row 13"][0] == pytest.approx(-0.01871)
    assert scan.parameters["Row 17"][0] == pytest.approx(1.0)
    assert scan.parameters["Row 30"][0] == pytest.approx(100000.0)


def test_the_loader_agrees_with_an_independent_slicing():
    # Written from the MATLAB that described this format, not from the parser:
    #   wl = data(:,2);  pl = data(:,4:4:end);  param = data(:,1:4:end);
    # This is the one check a misunderstanding shared with a synthetic writer
    # cannot pass.
    raw = np.loadtxt(EZ, delimiter=",")
    scan = BTSpectralSweep(EZ, spectra_type="PL")
    assert np.array_equal(raw[:, 1], scan.wavelength)
    assert np.array_equal(raw[:, 3::4], scan.spectra)
    par = raw[:, 0::4]                      # (n_rows, n_sweeps)
    assert np.array_equal(par[3], scan.parameters["Gate Voltage 1"])
    assert np.array_equal(par[5], scan.parameters["Gate Voltage 2"])
    assert np.array_equal(par[7], scan.parameters["Reflection"])
    assert np.array_equal(par[10], scan.parameters["Excitation Power"])


@pytest.mark.parametrize("path, expected", [
    (DOPING, "correlated (doping-like)"),
    (EZ, "anti-correlated (field-like)"),
])
def test_gate_mode_recovers_what_the_filename_says(path, expected):
    # The loader is told nothing about which kind of scan this is; it reports the
    # correlation between the two declared gate rows. That it agrees with the
    # filename is evidence the row map is right.
    scan = BTSpectralSweep(path, spectra_type="PL", gates=GATES)
    assert expected in scan.gate_mode


@pytest.mark.parametrize("path", [DOPING, EZ])
def test_the_gate_rows_outrank_everything_that_moved(path):
    # Evidence, not a detector: varying_parameters ranks by span against a row's
    # own RMS, so a leakage current can outrank a gate. Both gates and both
    # currents should be the top four either way.
    scan = BTSpectralSweep(path, spectra_type="PL")
    assert set(list(scan.varying_parameters())[:4]) == {
        "Gate Voltage 1", "Gate Voltage 2", "Gate Current 1", "Gate Current 2",
    }


@pytest.mark.parametrize("path", [DOPING, EZ])
def test_power_needs_no_scale_factor(path):
    # Both filenames say p20uW. The AttoCube's 0.303e6 would put this at 6e6.
    scan = BTSpectralSweep(path, spectra_type="PL")
    assert scan.curated_parameters["power"][1] == 1.0
    assert scan.curated_parameters["power"][2] == "µW"
    assert 15.0 < scan.power.mean() < 30.0


def test_positions_are_piezo_units_not_volts():
    # A distance needs a per-stage calibration no file carries, so the unit
    # states what the number is rather than converting it.
    scan = BTSpectralSweep(EZ, spectra_type="PL")
    assert scan.curated_parameters["scanner_x"][2] == "piezo units"
    assert scan.scanner_x[0] == pytest.approx(59.0)


def test_the_field_scan_resolves_a_field_axis():
    scan = BTSpectralSweep(EZ, spectra_type="PL", gates=GATES,
                                 geometry=GEOM, sweep="electric_field")
    assert scan.sweep_type == "electric_field"
    assert scan.sweep_unit == "mV/nm"
    # Anti-correlated gates, so the field sweeps through zero symmetrically.
    assert scan.sweep_axis[0] < 0 < scan.sweep_axis[-1]
    assert abs(scan.sweep_axis[0]) == pytest.approx(abs(scan.sweep_axis[-1]),
                                                    rel=0.05)


def test_the_doping_scan_resolves_a_density_axis():
    scan = BTSpectralSweep(DOPING, spectra_type="PL", gates=GATES,
                                 geometry=GEOM, sweep="carrier_density")
    assert scan.sweep_type == "carrier_density"
    assert scan.sweep_axis[0] < 0 < scan.sweep_axis[-1]


def test_a_row_label_works_as_a_sweep_axis():
    # Every parameter row is a declarable axis, including an unnamed one -- which
    # is what makes leaving a row unnamed cost nothing. Row 12 rather than Row 17
    # because 17 is held at one value all file, and declaring a held row as the
    # axis warns; that is correct, and a different test's business.
    scan = BTSpectralSweep(EZ, spectra_type="PL", sweep="Row 12",
                                 sweep_label="Unknown channel", sweep_unit="A?")
    assert scan.sweep_axis_label == "Unknown channel (A?)"
    assert scan.sweep_axis.size == 101


def test_declaring_a_held_row_as_the_axis_warns():
    # Row 17 reads 1.0 at every sweep point. Plotted against it every point lands
    # in the same place and only one is drawn, so it says so rather than drawing.
    with pytest.warns(UserWarning, match="takes only 1 value"):
        BTSpectralSweep(EZ, spectra_type="PL", sweep="Row 17")


def test_the_correction_ladder_runs(ez):
    scan = BTSpectralSweep(
        EZ, spectra_type="PL", gates=GATES, cosmic_rays={},
        bg_region_nm=(705, 715), apply_jacobian=True,
    )
    # Raw arrays are never mutated: each correction is a new rung.
    assert scan.spectra_cr is not None and scan.spectra_cr is not scan.spectra
    assert scan.spectra_bg is not None
    assert scan.best_spectra is scan.spectra_bg
    assert scan.energy_spectra_pre_jacobian is not scan.energy_spectra
    assert np.array_equal(scan.spectra, np.loadtxt(EZ, delimiter=",")[:, 3::4])


def test_it_is_a_drop_in_for_plotting_and_fitting(ez):
    # The whole point of the class. Imported here rather than at module scope so
    # a matplotlib backend problem cannot fail collection for the other tests.
    import matplotlib
    matplotlib.use("Agg")
    from leman import fitting, plotting

    fig, ax, mesh = plotting.plot_spectral_map(ez)
    # One mesh row per sweep point, per 0031.
    assert mesh.get_array().shape == (ez.n_sweeps, ez.n_pixels)

    # Reachability only, and deliberately not the fitted values: this sample has
    # four overlapping features, so a single-peak fit is the wrong model for it
    # and six of these points converge to a Lorentzian tens of eV wide (A31).
    # What is asserted is that the scan satisfies what fitting reads off it.
    fits = fitting.fit_scan_peak(ez, x_range=(1.60, 1.70))
    assert len(fits) == ez.n_sweeps
    import matplotlib.pyplot as plt
    plt.close("all")


# --- What it refuses, against real files -----------------------------------

@pytest.mark.parametrize("path, names", [
    (str(DATA / "stark-shift" / "PL-dual-gate-sweep_26_05_15_14_03_18_iter_0.csv"),
     "ACSpectralSweep"),
    (str(DATA / "TRPL" / "TRPL_26_07_30_16_29_39_iter_0.csv"),
     "ACTRPLSweep"),
])
def test_a_headed_export_is_refused_and_the_right_class_named(path, names):
    # The message asks the header which layout it is, rather than guessing at
    # the more common one.
    with pytest.raises(ValueError, match=f"Use {names} instead"):
        BTSpectralSweep(path, spectra_type="PL")


def test_a_two_row_file_is_sent_to_single_spectrum():
    path = str(DATA / "single-spectra" / "ref_single_spectrum_26_07_01_14_42_47.csv")
    with pytest.raises(ValueError, match="Load it with SingleSpectrum"):
        BTSpectralSweep(path, spectra_type="PL")


def test_an_image_whose_width_does_not_divide_is_refused():
    # 185 columns. The parameter companion written beside an image sequence is
    # headed, so this reaches the column-count test.
    path = str(DATA / "position-scan" / "PL" / "pl_26_06_26_10_19_26_iter_0.csv")
    with pytest.raises(ValueError, match="Use ACSpectralSweep instead"):
        BTSpectralSweep(path, spectra_type="PL")


def test_an_image_whose_width_does_divide_is_still_refused():
    # 512 columns, so the column count cannot tell this from a 128-block sweep.
    # The repeated-axis check is what separates them, and this is a real
    # committed frame rather than a synthetic one.
    path = str(DATA / "exciton-diffusion" / "power_dep_iter_0000.csv")
    with pytest.raises(ValueError, match="do not share one axis"):
        BTSpectralSweep(path, spectra_type="PL")


def test_the_attocube_loader_refuses_a_bigtable_file():
    # The mirror refusal. Its message names this class only when the column
    # count divides by four, which a BigTable export's does.
    with pytest.raises(ValueError, match="BTSpectralSweep"):
        ACSpectralSweep(EZ, spectra_type="PL")


def test_hdf5_is_refused_in_both_directions(tmp_path, ez):
    with pytest.raises(NotImplementedError, match="cannot be written as HDF5"):
        ez.to_hdf5(tmp_path / "x.h5")
    h5 = str(DATA / "stark-shift" / "PL-dual-gate-sweep_26_05_15_14_03_18_iter_0.h5")
    with pytest.raises(NotImplementedError, match="cannot be read as HDF5"):
        BTSpectralSweep(h5)


def test_the_wiring_is_never_assumed():
    # It varies between samples on this setup, so an undeclared scan refuses
    # rather than defaulting to the rows _CURATED names.
    scan = BTSpectralSweep(EZ, spectra_type="PL")
    with pytest.raises(ValueError, match="not declared"):
        scan.v_top
    # gate_mode is the exception: it describes an undeclared scan on purpose.
    assert scan.gate_mode is not None


def test_gate_three_has_a_voltage_and_no_current():
    # Row 9 is a gate voltage; whichever row carries its current is one of the
    # unidentified ones, so it is not paired. A voltage and no current is the
    # honest outcome, not an error.
    scan = BTSpectralSweep(
        EZ, spectra_type="PL",
        gates={"top": "Gate Voltage 3", "bottom": "Gate Voltage 1",
               "channel": None})
    assert np.all(scan.v_top == 0.0)
    with pytest.raises(ValueError, match="not a source-meter channel"):
        scan.i_top


# ---------------------------------------------------------------------------
# Synthetic — every refusal and warning a good file cannot reach
# ---------------------------------------------------------------------------

def _write(path, *, n_px=40, n_blocks=3, mismatch=False, long_block=False,
           split_axis=False, ragged_width=False, all_zero=False,
           axis_stops_early=False):
    """Write a minimal export: [par, wavelength, signal, signal], no header.

    One builder with flags, so every fixture below comes from the same statement
    of the layout and a change to it cannot fix one case while breaking another.

    The grid is exactly *n_px* rows, because that is what the format does: the
    parameter column is **overlaid on the pixel rows** rather than occupying rows
    of its own, so a real export has one row per spectrometer pixel and 1340 of
    them against a 30-row parameter block. `axis_stops_early` breaks that on
    purpose.
    """
    width = 3 if ragged_width else 4
    d = np.zeros((n_px, n_blocks * width))
    if not all_zero:
        for b in range(n_blocks):
            d[:, width * b + 1] = 700.0 + 0.2 * np.arange(n_px)
            d[:, width * b + 2] = 100.0 + b
            if width == 4:
                d[:, width * b + 3] = 100.0 + b
            d[3, width * b] = -1.0 + b          # a Gate Voltage 1 that varies
            if long_block:
                d[30, width * b] = 7.0          # row 31, past the exposed block
    if mismatch:
        d[2, 3] = 999.0                          # block 0's copy disagrees
    if split_axis:
        d[:, width + 1] += 5.0                   # block 1 on a different axis
    if axis_stops_early:
        # The axis runs out half way and the rest stays at zero, which is what
        # a truncated export or a stride read one column out looks like. Zero is
        # finite, so the pixel mask keeps it.
        d[n_px // 2:, 1::width] = 0.0
    np.savetxt(path, d, delimiter=",", fmt="%.5f")
    return str(path)


def test_disagreeing_signal_columns_warn_and_name_the_pixel(tmp_path):
    path = _write(tmp_path / "mismatch.csv", mismatch=True)
    with pytest.warns(UserWarning, match=r"at pixel 2, sweep point 0") as caught:
        scan = BTSpectralSweep(path, spectra_type="PL")
    # The third column is what is read, so the 999.0 does not reach `spectra`.
    assert scan.spectra[2, 0] == pytest.approx(100.0)
    # Measured: the warning must blame the line that built the scan.
    assert caught[0].filename == __file__


def test_a_parameter_row_past_the_block_warns(tmp_path):
    path = _write(tmp_path / "long.csv", long_block=True)
    with pytest.warns(UserWarning, match=r"row 31, past the 30 rows") as caught:
        BTSpectralSweep(path, spectra_type="PL")
    assert caught[0].filename == __file__


def test_a_good_synthetic_file_is_silent(tmp_path):
    # The counterpart the two above need: neither warning may fire on a file
    # that is simply correct, or they are useless.
    path = _write(tmp_path / "clean.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scan = BTSpectralSweep(path, spectra_type="PL")
    assert scan.n_sweeps == 3
    assert scan.n_pixels == 40


def test_an_axis_that_stops_part_way_is_refused(tmp_path):
    # There is no header stating a pixel count, so nothing else would notice: a
    # zero is finite and survives the pixel mask, then reaches hc/λ as a
    # division by zero two steps later. Refused on the axis instead.
    path = _write(tmp_path / "stops.csv", axis_stops_early=True)
    with pytest.raises(ValueError, match="does not increase"):
        BTSpectralSweep(path, spectra_type="PL")


def test_blocks_on_different_axes_are_refused(tmp_path):
    path = _write(tmp_path / "split.csv", split_axis=True)
    with pytest.raises(ValueError, match="do not share one axis"):
        BTSpectralSweep(path, spectra_type="PL")


def test_a_width_that_is_not_a_whole_block_is_refused(tmp_path):
    # 9 columns from three 3-wide blocks.
    path = _write(tmp_path / "ragged.csv", ragged_width=True)
    with pytest.raises(ValueError, match="not a whole number of 4-column"):
        BTSpectralSweep(path, spectra_type="PL")


def test_a_file_with_no_measurement_is_refused(tmp_path):
    path = _write(tmp_path / "empty.csv", all_zero=True)
    with pytest.raises(ValueError, match="contains no spectra"):
        BTSpectralSweep(path, spectra_type="PL")


def test_spectra_type_is_required(tmp_path):
    path = _write(tmp_path / "clean.csv")
    with pytest.raises(ValueError, match="spectra_type is required"):
        BTSpectralSweep(path)


def test_the_curated_registry_is_the_bigtable_one(tmp_path):
    path = _write(tmp_path / "clean.csv")
    scan = BTSpectralSweep(path, spectra_type="PL")
    labels = {k: v[0] for k, v in scan.curated_parameters.items()}
    assert labels["v_top"] == "Gate Voltage 1"
    assert labels["v_bot"] == "Gate Voltage 2"
    assert labels["power"] == "Excitation Power"
    assert labels["scanner_x"] == "X Position"
    # Not the AttoCube's, which is the failure this guards.
    assert "V_A" not in labels.values()
    assert "Scanner X" not in labels.values()


def test_there_is_no_roi_argument(tmp_path):
    # One signal, not two. A caller reaching for the AttoCube's argument should
    # be told, not silently ignored.
    path = _write(tmp_path / "clean.csv")
    with pytest.raises(TypeError, match="roi"):
        BTSpectralSweep(path, spectra_type="PL", roi=1)
