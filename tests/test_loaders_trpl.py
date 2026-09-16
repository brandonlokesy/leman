"""
Tests for ACTRPLSweep.

The real example files are small (3 × 142 KB decays plus an 11 MB companion), so
the directory case is covered against them directly.  Filename-ordering and
axis-mismatch behaviour is covered synthetically, since those cases cannot be
produced from the committed data.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from leman.loaders import (
    ACSpectralSweep,
    ACTRPLSweep,
    DeviceGeometry,
    StackLayer,
)

from _paths import DATA

TRPL_DIR   = str(DATA / "TRPL")
ONE_DECAY  = f"{TRPL_DIR}/TRPL_26_07_30_16_29_39_iter_0.csv"
COMPANION  = f"{TRPL_DIR}/TRPL_26_07_30_16_30_11_iter_0.csv"
SPECTRAL   = str(DATA / "stark-shift"
                    / "PL-dual-gate-sweep_26_05_15_14_03_18_iter_0.csv")

N_BINS   = 3205
N_SWEEPS = 3
# V_A steps +1.68 / ~0 / -1.68 V against V_B -1.0 / ~0 / +1.0 V: a field sweep.
V_A_EXPECTED = [1.679972, 2.325371e-06, -1.679972]

# The channel-to-gate wiring the loaders refuse to assume.
GATES = {"top": "V_A", "bottom": "V_B"}


def _synth_decay(path, n_bins=8, t_step=4.0e-3, params=None, counts=None) -> None:
    """Write a minimal temporal export: [Par_0, Wavelength0, Exp_0] + padding."""
    params = params or {"V_A": 1.0, "Excitation Power": 1e-4}
    labels = list(params)
    lines  = ["Parameters Labels,Par_0,Wavelength0,Exp_0,,,"]
    for r in range(n_bins):
        label = labels[r] if r < len(labels) else ""
        par   = params[label] if label else 0.0
        cnt   = counts[r] if counts is not None else r
        lines.append(f"{label},{par},{r * t_step},{cnt},,,")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# A single decay is just n_sweeps == 1
# ---------------------------------------------------------------------------


def test_single_file_is_one_sweep_point():
    d = ACTRPLSweep(ONE_DECAY)
    assert d.n_sweeps == 1
    assert d.n_bins == N_BINS
    assert d.decays.shape == (N_BINS, 1)
    assert d.spectra_type == "TRPL"          # defaults; the class name declares it
    assert d.files == [Path(ONE_DECAY)]


def test_time_axis_is_ns_ascending():
    d = ACTRPLSweep(ONE_DECAY)
    assert np.all(np.diff(d.time) > 0)
    assert d.time[0] == 0.0
    assert d.time.max() == pytest.approx(12.817, abs=1e-3)   # ~12.8 ns range
    assert (d.time[1] - d.time[0]) == pytest.approx(4.0e-3, rel=1e-3)  # 4 ps bins
    assert d.axis_label == "Time (ns)"


def test_energy_machinery_absent():
    # hc/t is meaningless and divides by zero at t=0, so none of it should exist.
    d = ACTRPLSweep(ONE_DECAY)
    for attr in ("energy", "energy_spectra", "energy_spectra_cr",
                 "energy_spectra_bg", "apply_jacobian", "bg_region_nm",
                 "wavelength"):
        assert not hasattr(d, attr), attr
    # Nor `spectra` and its rungs: a TRPL sweep handed to a spectral plot must
    # raise, not draw time as though it were wavelength.
    assert not hasattr(d, "spectra")
    assert not hasattr(d, "spectra_bg")


# ---------------------------------------------------------------------------
# The directory case
# ---------------------------------------------------------------------------


def test_directory_assembles_the_sweep():
    s = ACTRPLSweep(TRPL_DIR)
    assert s.n_sweeps == N_SWEEPS
    assert s.n_bins == N_BINS
    assert s.decays.shape == (N_BINS, N_SWEEPS)
    assert len(s.files) == N_SWEEPS


def test_files_ordered_by_iter_index_not_filename():
    s = ACTRPLSweep(TRPL_DIR, gates=GATES)
    assert [f.stem[-6:] for f in s.files] == ["iter_0", "iter_1", "iter_2"]
    # Parameters must follow that order, or every decay pairs with the wrong point.
    assert np.allclose(s.v_top, V_A_EXPECTED)


def test_companion_excluded_from_data_files_despite_iter_0_collision():
    # The companion is also named iter_0 and is written last, so only its content
    # distinguishes it.
    s = ACTRPLSweep(TRPL_DIR)
    assert s.metadata_file is not None
    assert s.metadata_file.name.endswith("16_30_11_iter_0.csv")
    assert s.metadata_file not in s.files
    assert s.n_declared_sweeps == N_SWEEPS


def test_declared_parameters_exposed_but_not_enforced():
    # Independent read-backs: the swept gate agrees closely, drifting channels do
    # not, and the loader does not adjudicate.
    s = ACTRPLSweep(TRPL_DIR)
    assert s.declared_parameters is not None
    assert np.allclose(s.declared_parameters["V_A"], s.parameters["V_A"],
                       rtol=0, atol=1e-3)
    assert "Fianium_Select_A6" in s.declared_parameters


def test_loading_a_directory_emits_no_warnings():
    # A warning that fires on every real sweep is a warning that gets ignored.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ACTRPLSweep(TRPL_DIR)
    assert [str(c.message) for c in caught] == []


def test_field_sweep_over_the_directory():
    geom = DeviceGeometry(tmdc_stack=[StackLayer("MoSe2"), StackLayer("WSe2")],
                          d_hbn_top=53, d_hbn_bottom=46)
    s = ACTRPLSweep(TRPL_DIR, sweep="electric_field", geometry=geom,
                          gates=GATES)
    assert s.sweep_axis_label == r"$E_F$ (mV/nm)"
    assert s.gate_mode == "dual-gate, anti-correlated (field-like)"
    # Symmetric about zero, and monotonic in sweep order.
    assert np.all(np.diff(s.sweep_axis) > 0)
    assert s.sweep_axis[1] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Pre-pulse background
# ---------------------------------------------------------------------------


def test_pre_pulse_window_subtracted():
    s = ACTRPLSweep(TRPL_DIR, bg_region_ns=(0.0, 1.0))
    assert s.bg_region_ns == (0.0, 1.0)
    assert s.decays_bg is not None
    assert s.best_decays is s.decays_bg
    # The window's own mean is removed, so it now averages ~0 there.
    pre = (s.time >= 0.0) & (s.time <= 1.0)
    assert np.allclose(s.decays_bg[pre].mean(axis=0), 0.0, atol=1e-9)
    # Raw counts are never mutated.
    assert s.decays.min() >= 0.0


def test_no_background_leaves_decays_untouched():
    s = ACTRPLSweep(TRPL_DIR)
    assert s.decays_bg is None
    assert s.best_decays is s.decays


# ---------------------------------------------------------------------------
# Cross-format rejection
# ---------------------------------------------------------------------------


def test_spectral_file_rejected_naming_the_right_class():
    with pytest.raises(ValueError, match="ACSpectralSweep"):
        ACTRPLSweep(SPECTRAL)


def test_temporal_file_rejected_by_the_spectral_class():
    with pytest.raises(ValueError, match="ACTRPLSweep"):
        ACSpectralSweep(ONE_DECAY, spectra_type="TRPL")


def test_companion_alone_names_the_trpl_class():
    with pytest.raises(ValueError, match="metadata companion"):
        ACSpectralSweep(COMPANION, spectra_type="PL")


def test_directory_rejected_by_the_spectral_class():
    with pytest.raises(ValueError, match="reads a single file"):
        ACSpectralSweep(TRPL_DIR, spectra_type="PL")


# ---------------------------------------------------------------------------
# Synthetic: ordering, gaps, and axis agreement
# ---------------------------------------------------------------------------


def test_iter_10_sorts_after_iter_2(tmp_path):
    # Lexicographic order would put iter_10 first and misalign the whole sweep.
    # 3..9 are absent here, so the gap warning is expected alongside the ordering.
    for i in (2, 10):
        _synth_decay(tmp_path / f"TRPL_iter_{i}.csv",
                     params={"V_A": float(i), "Excitation Power": 1e-4})
    with pytest.warns(UserWarning, match="missing iteration"):
        s = ACTRPLSweep(tmp_path, gates=GATES)
    assert [f.stem for f in s.files] == ["TRPL_iter_2", "TRPL_iter_10"]
    assert np.allclose(s.v_top, [2.0, 10.0])


def test_consecutive_iterations_do_not_warn(tmp_path):
    for i in (0, 1, 2):
        _synth_decay(tmp_path / f"TRPL_iter_{i}.csv",
                     params={"V_A": float(i), "Excitation Power": 1e-4})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = ACTRPLSweep(tmp_path, gates=GATES)
    assert [str(c.message) for c in caught] == []
    assert np.allclose(s.v_top, [0.0, 1.0, 2.0])


def test_gap_in_iteration_sequence_warns(tmp_path):
    for i in (0, 2):
        _synth_decay(tmp_path / f"TRPL_iter_{i}.csv",
                     params={"V_A": float(i), "Excitation Power": 1e-4})
    with pytest.warns(UserWarning, match=r"missing iteration\(s\) \[1\]"):
        s = ACTRPLSweep(tmp_path)
    assert s.n_sweeps == 2


def test_duplicate_iteration_index_warns(tmp_path):
    # Two decays claiming iter_0. Both carry a temporal header, so classification
    # keeps them and the collision reaches the sort — unlike the real companion,
    # which is a spectral file and is separated out before it.
    _synth_decay(tmp_path / "TRPL_iter_0.csv", params={"V_A": 0.0})
    _synth_decay(tmp_path / "TRPL_rerun_iter_0.csv", params={"V_A": 5.0})
    with pytest.warns(UserWarning, match="claimed by more than one file"):
        s = ACTRPLSweep(tmp_path)
    assert s.n_sweeps == 2


def test_files_without_iter_suffix_warn(tmp_path):
    for name in ("a", "b"):
        _synth_decay(tmp_path / f"TRPL_{name}.csv",
                     params={"V_A": 1.0, "Excitation Power": 1e-4})
    with pytest.warns(UserWarning, match="no '_iter_N' suffix"):
        ACTRPLSweep(tmp_path)


def test_mismatched_bin_count_raises(tmp_path):
    _synth_decay(tmp_path / "TRPL_iter_0.csv", n_bins=8)
    _synth_decay(tmp_path / "TRPL_iter_1.csv", n_bins=9)
    with pytest.raises(ValueError, match="time bins"):
        ACTRPLSweep(tmp_path)


def test_time_axis_disagreement_beyond_tolerance_raises(tmp_path):
    _synth_decay(tmp_path / "TRPL_iter_0.csv", t_step=4.0e-3)
    _synth_decay(tmp_path / "TRPL_iter_1.csv", t_step=5.0e-3)
    with pytest.raises(ValueError, match="time_rtol"):
        ACTRPLSweep(tmp_path)


def test_small_time_axis_jitter_is_tolerated(tmp_path):
    # Real files differ in the seventh figure of the bin width; equality would
    # reject every genuine sweep.
    _synth_decay(tmp_path / "TRPL_iter_0.csv", t_step=4.0003242e-3)
    _synth_decay(tmp_path / "TRPL_iter_1.csv", t_step=4.0003283e-3)
    s = ACTRPLSweep(tmp_path)
    assert s.n_sweeps == 2


def test_empty_directory_lists_what_it_looked_at(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing here\n")
    with pytest.raises(ValueError, match="No TRPL data files found"):
        ACTRPLSweep(tmp_path)


def test_prefix_filters_the_directory(tmp_path):
    _synth_decay(tmp_path / "TRPL_iter_0.csv", params={"V_A": 1.0})
    _synth_decay(tmp_path / "other_iter_0.csv", params={"V_A": 9.0})
    s = ACTRPLSweep(tmp_path, prefix="TRPL_", gates=GATES)
    assert s.n_sweeps == 1
    assert np.allclose(s.v_top, [1.0])


# ---------------------------------------------------------------------------
# IRF references are excluded from a directory sweep by name, not content
# ---------------------------------------------------------------------------
# An IRF file has the identical [Par, Wavelength, Exp] shape as a real decay,
# so nothing in its content distinguishes it -- without this, a directory
# scan with no prefix folds it in as a spurious extra sweep point.


def test_irf_named_file_excluded_even_without_a_prefix(tmp_path):
    _synth_decay(tmp_path / "TRPL_iter_0.csv", params={"V_A": 1.0})
    _synth_decay(tmp_path / "TRPL_iter_1.csv", params={"V_A": 2.0})
    _synth_decay(tmp_path / "IRF_iter_2.csv", params={"V_A": 999.0})

    s = ACTRPLSweep(tmp_path)

    assert s.n_sweeps == 2
    assert np.allclose(s["V_A"], [1.0, 2.0])  # channel term -- no gate mapping declared
    assert s.irf_files == ["IRF_iter_2.csv"]


def test_irf_match_is_case_insensitive_and_by_stem_not_just_prefix(tmp_path):
    _synth_decay(tmp_path / "TRPL_iter_0.csv", params={"V_A": 1.0})
    _synth_decay(tmp_path / "reference_irf_iter_1.csv", params={"V_A": 999.0})

    s = ACTRPLSweep(tmp_path)

    assert s.n_sweeps == 1
    assert s.irf_files == ["reference_irf_iter_1.csv"]


def test_single_file_load_has_no_irf_files(tmp_path):
    d = ACTRPLSweep(ONE_DECAY)
    assert d.irf_files == []


def test_all_candidates_irf_raises_with_a_clear_message(tmp_path):
    _synth_decay(tmp_path / "IRF_iter_0.csv", params={"V_A": 1.0})
    with pytest.raises(ValueError, match="IRF reference"):
        ACTRPLSweep(tmp_path)
