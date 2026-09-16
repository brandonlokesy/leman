"""
Tests for ACSpectralSweep parameter loading.

Builds a small synthetic spectral CSV in the AttoCube export layout so the suite
runs without the lab network share.  Focuses on the generalized parameter store
(every labeled row exposed via ``parameters`` / ``get_parameter`` / ``[]``), the
curated attributes, and the deprecated ``ACPLVabScan`` shim.

The fixture CSV is written from the same reading of the export layout as the
parser, so it cannot catch a misunderstanding shared by both — see E9 in
``dev/defects.md``.  It does pin the decoding *contract*.
"""

import warnings

import numpy as np
import pytest

from leman.constants import EPS_HBN
from leman.loaders import (
    ACPLVabScan,
    ACSpectralSweep,
    ACTRPLSweep,
    DeviceGeometry,
    StackLayer,
    _read_block_layout,
)

# 3 sweeps x 10 pixels; the first eight pixel rows also carry labeled scalars,
# leaving 2 unlabeled pixel rows to exercise the labeled-row filter.
N_SWEEPS = 3
N_PIXELS = 10
WAVELENGTH = 800.0 + np.arange(N_PIXELS)          # 800 .. 809 nm

PARAMS = {
    "V_A":              np.array([0.0, 0.5, 1.0]),
    "V_B":              np.array([0.0, -0.5, -1.0]),
    "Excitation Power": np.array([1e-6, 2e-6, 3e-6]),
    "I_A":              np.array([1e-9, 2e-9, 3e-9]),
    "I_B":              np.array([4e-9, 5e-9, 6e-9]),
    "Scanner X":        np.array([5.0, 5.0, 5.0]),
    "Scanner Y":        np.array([7.0, 7.5, 8.0]),
    "Galvo_X":          np.array([10.0, 11.0, 12.0]),
}
POWER_SCALE = 0.303e6  # ACPLVabScan default

# The channel-to-gate wiring the loaders refuse to assume.  Shared so that a test
# only spells it out when the wiring itself is the subject.
GATES = {"top": "V_A", "bottom": "V_B"}


def _roi1(r, i):
    return 100 + r * 10 + i


def _roi2(r, i):
    return 200 + r * 10 + i


def make_spectral_csv(
    path,
    params      : dict = None,
    zero_blocks : int  = 0,
    interleave  : bool = False,
    roi1        : np.ndarray = None,
    wavelength  : np.ndarray = None,
) -> None:
    """
    Write a spectral CSV with labeled parameter rows + 2 padding columns.

    Parameters
    ----------
    params : dict, optional
        Overrides the default :data:`PARAMS` row set, so a test can supply
        awkward labels (e.g. one containing ``/``) without a second builder.
        The sweep count is taken from the row length, so a longer row set writes
        a longer sweep — which is how a raster fixture is built.
    roi1 : np.ndarray, shape (n_pixels, N_SWEEPS), optional
        Overrides the ``ExpROI1`` counts, for a test that needs spectra with
        structure rather than the default index ramp.  More pixel rows than
        *params* has labels is fine — the surplus rows are unlabeled, which is
        what the export does.
    wavelength : np.ndarray, shape (n_pixels,), optional
        Wavelength axis to write, defaulting to :data:`WAVELENGTH`.  Give it
        whenever *roi1* has a different pixel count.
    zero_blocks : int
        Append this many extra *declared* blocks filled entirely with literal
        zeros — what the real exporter does when it over-allocates the header.
        These are numeric, not empty, so no NaN-based strip removes them; the
        314 MB reflectance raster ships 2091 of them.
    interleave : bool
        Put the zero-filled blocks *before* the real ones instead of after,
        to exercise the anomaly warning.  Ignored when ``zero_blocks == 0``.
    """
    params = PARAMS if params is None else params
    labels = list(params)                          # first len(params) rows
    wl     = WAVELENGTH if wavelength is None else np.asarray(wavelength, float)
    n_pixels = wl.size
    # The rows are one value per sweep point, so their length *is* the count.
    n_sweeps = len(next(iter(params.values())))

    n_declared = n_sweeps + zero_blocks
    header = ["Parameters Labels"]
    for i in range(n_declared):
        header += [f"Par_{i}", f"Wavelength{i}", f"ExpROI1_{i}", f"ExpROI2_{i}"]
    header += ["", ""]                             # padding columns

    zeros = ["0.0", "0.0", "0.0", "0.0"]
    lines = [",".join(header)]
    for r in range(n_pixels):
        label = labels[r] if r < len(labels) else ""
        par = params[label] if label else np.zeros(n_sweeps)
        real = []
        for i in range(n_sweeps):
            counts = _roi1(r, i) if roi1 is None else roi1[r, i]
            real += [f"{par[i]}", f"{wl[r]}",
                     f"{counts}", f"{_roi2(r, i)}"]
        pad = zeros * zero_blocks
        row = [label] + (pad + real if interleave else real + pad)
        row += ["", ""]                            # padding cells
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path)
    return path


@pytest.fixture
def scan(csv_path):
    return ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES)


@pytest.fixture
def unwired(csv_path):
    """The same scan with no declared wiring — every gate role then refuses."""
    return ACSpectralSweep(str(csv_path), spectra_type="PL")


# ---------------------------------------------------------------------------
# Generic parameter store
# ---------------------------------------------------------------------------


def test_all_labeled_parameters_exposed(scan):
    assert set(scan.parameter_labels) == set(PARAMS)
    for label, expected in PARAMS.items():
        assert np.allclose(scan.parameters[label], expected), label


def test_parameter_arrays_have_sweep_length(scan):
    assert scan.n_sweeps == N_SWEEPS
    assert scan.n_pixels == N_PIXELS
    for arr in scan.parameters.values():
        assert arr.shape == (N_SWEEPS,)


def test_unlabeled_pixel_rows_excluded(scan):
    # Only the labeled rows become parameters; the 2 trailing pixel rows do not.
    assert len(scan.parameters) == len(PARAMS)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def test_get_parameter_and_scale(scan):
    assert np.allclose(scan.get_parameter("Galvo_X"), PARAMS["Galvo_X"])
    assert np.allclose(scan.get_parameter("Galvo_X", scale=2.0),
                       PARAMS["Galvo_X"] * 2.0)


def test_getitem_sugar(scan):
    assert np.allclose(scan["Galvo_X"], PARAMS["Galvo_X"])


def test_missing_parameter_raises_with_available(scan):
    with pytest.raises(KeyError) as exc:
        scan.get_parameter("does_not_exist")
    assert "Galvo_X" in str(exc.value)  # error lists available labels


# ---------------------------------------------------------------------------
# Backward-compatible curated attributes
# ---------------------------------------------------------------------------


def test_curated_attributes_unchanged(scan):
    assert np.allclose(scan.v_top, PARAMS["V_A"])           # raw
    assert np.allclose(scan.v_bot, PARAMS["V_B"])           # raw
    assert np.allclose(scan.power, PARAMS["Excitation Power"] * POWER_SCALE)
    # GATES wires V_A to the top gate, so I_A is the top gate's current.
    assert np.allclose(scan.i_top, PARAMS["I_A"] * 1e9)     # -> nA
    assert np.allclose(scan.i_bot, PARAMS["I_B"] * 1e9)     # -> nA


def test_curated_attrs_are_raw_views_of_parameters(scan):
    # parameters holds raw units; curated power/current are scaled, not raw.
    assert np.allclose(scan.v_top, scan.parameters["V_A"])
    assert np.allclose(scan.power / POWER_SCALE, scan.parameters["Excitation Power"])


def test_ef_none_without_geometry(scan):
    assert scan.ef is None


def test_undeclared_sweep_gives_index_axis_not_a_guess(scan):
    # V_A, V_B, Excitation Power and Scanner Y all vary in the fixture, so any
    # auto-pick would be a coin toss.  The axis must fall back to the index.
    assert scan.sweep_type == "index"
    assert np.allclose(scan.sweep_axis, np.arange(N_SWEEPS))
    assert scan.sweep_axis_label == "Sweep index"


# ---------------------------------------------------------------------------
# Spectra / axes
# ---------------------------------------------------------------------------


def test_spectra_and_wavelength(scan):
    assert np.allclose(scan.wavelength, WAVELENGTH)
    expected = np.array([[_roi1(r, i) for i in range(N_SWEEPS)]
                         for r in range(N_PIXELS)], dtype=float)
    assert np.allclose(scan.spectra, expected)
    # energy axis is ascending
    assert np.all(np.diff(scan.energy) > 0)


def test_roi2_selection(csv_path):
    scan2 = ACSpectralSweep(str(csv_path), spectra_type="PL", roi=2)
    expected = np.array([[_roi2(r, i) for i in range(N_SWEEPS)]
                         for r in range(N_PIXELS)], dtype=float)
    assert np.allclose(scan2.spectra, expected)


def test_both_rois_always_loaded(scan):
    # roi= selects what `spectra` points at; neither ROI is discarded at load,
    # which is where a reference channel would live for a non-PL measurement.
    roi1 = np.array([[_roi1(r, i) for i in range(N_SWEEPS)]
                     for r in range(N_PIXELS)], dtype=float)
    roi2 = np.array([[_roi2(r, i) for i in range(N_SWEEPS)]
                     for r in range(N_PIXELS)], dtype=float)
    assert np.allclose(scan.spectra_roi1, roi1)
    assert np.allclose(scan.spectra_roi2, roi2)
    assert scan.spectra is scan.spectra_roi1


def test_padding_columns_stripped(scan):
    # 2 padding cols make raw n_cols = 14 (not divisible by 4); they must be
    # stripped to 12 -> 3 sweeps without raising.
    assert scan.n_sweeps == N_SWEEPS


# ---------------------------------------------------------------------------
# Curated registry (#2) and property backing (#1)
# ---------------------------------------------------------------------------


def test_curated_parameters_registry(scan):
    reg = scan.curated_parameters
    assert reg["v_top"] == ("V_A", 1.0, "V")
    assert reg["power"] == ("Excitation Power", 0.303e6, "µW")
    # The current's row is not a registry default — it is resolved from gates.
    assert reg["i_top"] == ("I_A", 1e9, "nA")
    # The scanners are piezos; the rows carry drive voltage, not a distance.
    assert reg["scanner_x"] == ("Scanner X", 1.0, "V")
    assert reg["scanner_y"] == ("Scanner Y", 1.0, "V")


def test_scanner_position_properties(scan):
    assert np.allclose(scan.scanner_x, PARAMS["Scanner X"])
    assert np.allclose(scan.scanner_y, PARAMS["Scanner Y"])
    # scale 1.0 -> curated view equals the raw parameter row
    assert np.allclose(scan.scanner_x, scan.parameters["Scanner X"])


def test_curated_properties_are_read_only(scan):
    # Properties backed by self.parameters -> no setter.
    with pytest.raises(AttributeError):
        scan.v_top = np.zeros(N_SWEEPS)
    with pytest.raises(AttributeError):
        scan.ef = np.zeros(N_SWEEPS)


def test_constructor_overrides_flow_through_registry(csv_path):
    # scale 1.0 means the raw row, which is not microwatts — so the unit is
    # restated too.  Leaving it at the registry's "µW" is exactly the fault this
    # argument exists to prevent, and the loader warns about it.
    s = ACSpectralSweep(
        str(csv_path), spectra_type="PL",
        gates={"top": "V_B", "bottom": "V_A"}, curated_labels={"power": "Galvo_X"},
        curated_scales={"power": 1.0}, curated_units={"power": "counts"},
    )
    # scale override: power now equals the raw row it was pointed at
    assert np.allclose(s.power, s.parameters["Galvo_X"])
    assert s.curated_parameters["power"] == ("Galvo_X", 1.0, "counts")
    # gates override: v_top now reads the V_B row
    assert np.allclose(s.v_top, s.parameters["V_B"])
    assert s.curated_parameters["v_top"][0] == "V_B"


def test_unknown_curated_name_rejected(csv_path):
    with pytest.raises(ValueError, match="not a curated parameter"):
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              curated_labels={"v_topp": "V_B"})


def test_ef_property_with_geometry(csv_path):
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="electric_field", geometry=geom, gates=GATES)
    assert s.ef is not None
    assert np.allclose(s.ef, geom.electric_field(s.v_top, s.v_bot))
    assert np.allclose(s.sweep_axis, s.ef)
    assert s.sweep_axis_label == r"$E_F$ (mV/nm)"


# ---------------------------------------------------------------------------
# Missing curated rows (E1): load, then raise only if accessed
# ---------------------------------------------------------------------------


def test_missing_curated_row_still_loads(csv_path):
    # A file from a different instrument configuration must not be unloadable
    # just because one curated row is absent.
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              curated_labels={"power": "NoSuchRow"})
    assert s.n_sweeps == N_SWEEPS
    with pytest.raises(KeyError, match="Galvo_X"):   # error lists what is available
        s.power


def test_declared_sweep_requires_its_own_row(csv_path):
    # The fail-fast that remains is the one the caller's declaration implies.
    with pytest.raises(KeyError, match="power"):
        ACSpectralSweep(str(csv_path), spectra_type="PL", sweep="power",
                              curated_labels={"power": "NoSuchRow"})


def test_electric_field_sweep_requires_geometry(csv_path):
    with pytest.raises(ValueError, match="DeviceGeometry"):
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="electric_field", gates=GATES)


# ---------------------------------------------------------------------------
# spectra_type and sweep resolution
# ---------------------------------------------------------------------------


def test_spectra_type_is_required(csv_path):
    with pytest.raises(ValueError, match="spectra_type is required"):
        ACSpectralSweep(str(csv_path))


def test_spectra_type_vocabulary_enforced(csv_path):
    with pytest.raises(ValueError, match="not a recognised measurement type"):
        ACSpectralSweep(str(csv_path), spectra_type="photoluminescence")


def test_signal_label_follows_spectra_type(csv_path):
    pl = ACSpectralSweep(str(csv_path), spectra_type="PL")
    rc = ACSpectralSweep(str(csv_path), spectra_type="RC")
    assert pl.signal_label == "PL intensity (counts)"
    assert rc.signal_label == r"$\Delta R/R_0$"     # dimensionless: no unit
    assert rc.spectroscopy == "Reflectance contrast"


def test_raw_row_as_sweep_axis(csv_path):
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="Galvo_X", sweep_unit="V")
    assert np.allclose(s.sweep_axis, PARAMS["Galvo_X"])
    assert s.sweep_axis_label == "Galvo_X (V)"


def test_unknown_sweep_lists_both_registries(csv_path):
    with pytest.raises(ValueError) as exc:
        ACSpectralSweep(str(csv_path), spectra_type="PL", sweep="nonsense")
    assert "electric_field" in str(exc.value)   # known sweep types
    assert "Galvo_X" in str(exc.value)          # rows in this file


def test_piezo_sweep_uses_scanner_row(csv_path):
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="piezo_y")
    assert np.allclose(s.sweep_axis, PARAMS["Scanner Y"])
    assert s.sweep_axis_label == r"Piezo $y$ (V)"


def test_old_position_sweep_key_is_refused_with_the_new_name(csv_path):
    # No shim for the rename: the unknown-key path lists the valid types, so the
    # error hands back the name that replaced it.
    with pytest.raises(ValueError) as exc:
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="position_y")
    assert "piezo_y" in str(exc.value)


# ---------------------------------------------------------------------------
# Diagnostics: what actually varied
# ---------------------------------------------------------------------------


def test_varying_parameters_excludes_static_rows(scan):
    varying = scan.varying_parameters()
    assert "Scanner X" not in varying          # constant at 5.0 in the fixture
    assert "V_A" in varying and "Scanner Y" in varying
    assert varying["V_A"] == (0.0, 1.0, 1.0)   # (min, max, span)


def test_varying_parameters_ranks_a_row_straddling_zero(tmp_path):
    """
    An anti-symmetric gate pair has a mean of zero and a magnitude that is not.

    Scaling the span by the mean divided by ~2.2e-308 and returned ``inf``, so the
    two gates tied with each other and outranked every row that really did move
    most.  Scaling by RMS keeps the rank finite and ordered.
    """
    n = 12
    v = np.tile(np.linspace(-3.0, 3.0, 4), 3)
    assert float(np.mean(v)) == 0.0            # the case, exactly

    params = {
        "V_A":              v,
        "V_B":              -v,
        "Excitation Power": np.full(n, 2e-6),
        "Scanner X":        np.tile(np.arange(4.0) * 2, 3),
        "Scanner Y":        np.repeat(np.arange(3.0) * 5, 4),
    }
    path = tmp_path / "antisymmetric.csv"
    make_spectral_csv(path, params=params)
    s = ACSpectralSweep(str(path), spectra_type="PL")

    with warnings.catch_warnings():
        warnings.simplefilter("error")         # a numpy overflow would raise here
        varying = s.varying_parameters()

    assert set(varying) == {"V_A", "V_B", "Scanner X", "Scanner Y"}
    assert varying["V_A"] == (-3.0, 3.0, 6.0)
    # Finite and ordered, rather than two rows tied at inf ahead of everything.
    assert list(varying).index("Scanner X") > list(varying).index("V_A")


def test_varying_parameters_still_separates_jitter_from_a_sweep(tmp_path):
    """The distinction the scale exists to make, unchanged by using RMS."""
    params = {
        "V_A":              np.array([10.000, 10.001, 10.000]),   # wobble on 10 V
        "V_B":              np.array([0.002, 0.003, 0.002]),      # same, on 2 mV
        "Excitation Power": np.full(3, 2e-6),
    }
    path = tmp_path / "jitter.csv"
    make_spectral_csv(path, params=params)
    varying = ACSpectralSweep(str(path), spectra_type="PL").varying_parameters()

    assert "V_A" not in varying                # 1e-4 of its magnitude: noise
    assert "V_B" in varying                    # 0.4 of its magnitude: a sweep


def test_gate_mode_detects_antisymmetric_sweep(scan):
    # V_A goes 0 -> +1 while V_B goes 0 -> -1: a field-like sweep.
    assert scan.gate_mode == "dual-gate, anti-correlated (field-like)"


def test_gate_mode_none_when_gate_rows_absent(csv_path):
    s = ACSpectralSweep(
        str(csv_path), spectra_type="PL",
        gates={"top": "NoSuchRow", "bottom": "AlsoMissing"},
    )
    assert s.gate_mode is None


def test_gate_mode_describes_the_one_gate_it_can_see(csv_path):
    # One declared row is missing from the file, so only the other can be
    # described.  Reporting on it beats returning None and saying nothing.
    s = ACSpectralSweep(
        str(csv_path), spectra_type="PL",
        gates={"top": "NoSuchRow", "bottom": "V_B"},
    )
    assert s.gate_mode == "bottom-gate only"


def test_gate_mode_needs_no_declared_wiring(unwired):
    # How many channels moved together is a property of the data, so the
    # correlation verdict is the same with or without a mapping.
    assert unwired.gate_mode == "dual-gate, anti-correlated (field-like)"


def test_gate_mode_names_the_channel_when_wiring_is_undeclared(tmp_path):
    # Only V_B moves.  Undeclared, calling that "bottom-gate only" would assert a
    # wiring nothing recorded -- the exact mislabelling this refuses to make.
    params = dict(PARAMS)
    params["V_A"] = np.zeros(N_SWEEPS)
    csv = tmp_path / "one_gate.csv"
    make_spectral_csv(csv, params=params)

    assert (ACSpectralSweep(str(csv), spectra_type="PL").gate_mode
            == "single gate driven ('V_B')")
    # Declared, the role is known and is what gets reported.
    assert (ACSpectralSweep(str(csv), spectra_type="PL",
                                  gates=GATES).gate_mode
            == "bottom-gate only")
    # ...and the opposite wiring names the other electrode for the same file.
    assert (ACSpectralSweep(str(csv), spectra_type="PL",
                                  gates={"top": "V_B", "bottom": "V_A"}).gate_mode
            == "top-gate only")


def test_repr_survives_missing_rows(csv_path):
    # __repr__ raised for every geometry once already (A2); it must not raise
    # here either, whatever the file happens to lack.
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_labels={"power": "NoSuchRow"})
    assert "ACSpectralSweep" in repr(s)
    assert "Photoluminescence" in repr(s)


# ---------------------------------------------------------------------------
# The channel-to-gate mapping: stated, never assumed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sweep", ["electric_field", "top_voltage",
                                   "bottom_voltage"])
def test_gate_sweep_refuses_an_undeclared_wiring(csv_path, sweep):
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    with pytest.raises(ValueError, match="not declared") as exc:
        ACSpectralSweep(str(csv_path), spectra_type="PL", sweep=sweep,
                              geometry=geom)
    # The message has to name the rows this file actually offers, or the reader
    # cannot act on it.
    assert "V_A" in str(exc.value) and "V_B" in str(exc.value)


def test_non_gate_sweeps_are_unaffected(csv_path):
    # An ungated measurement must not have to state a wiring it does not use.
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", sweep="power")
    assert s.gates is None
    assert np.allclose(s.sweep_axis, PARAMS["Excitation Power"] * POWER_SCALE)


@pytest.mark.parametrize("attr", ["v_top", "v_bot", "i_top", "i_bot", "i_channel"])
def test_gate_properties_refuse_an_undeclared_wiring(unwired, attr):
    with pytest.raises(ValueError, match="not declared"):
        getattr(unwired, attr)


def test_ef_refuses_an_undeclared_wiring_only_when_a_geometry_was_given(csv_path):
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    with pytest.raises(ValueError, match="not declared"):
        ACSpectralSweep(str(csv_path), spectra_type="PL", geometry=geom).ef
    # Saying that no field was computed needs no wiring, so this stays None
    # rather than raising.
    assert ACSpectralSweep(str(csv_path), spectra_type="PL").ef is None


def test_transposing_the_wiring_negates_the_field(csv_path):
    # The whole reason the mapping cannot be defaulted: it sets the sign, and so
    # the sign of any dipole extracted downstream.
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    kw = dict(spectra_type="PL", sweep="electric_field", geometry=geom)
    forward = ACSpectralSweep(str(csv_path), gates=GATES, **kw)
    swapped = ACSpectralSweep(
        str(csv_path), gates={"top": "V_B", "bottom": "V_A"}, **kw)
    assert np.allclose(forward.ef, -swapped.ef)


def test_gates_are_recorded_on_the_scan(scan, unwired):
    assert scan.gates == GATES
    assert unwired.gates is None
    # A copy, so the caller's dict cannot be edited into the scan afterwards.
    scan.gates["top"] = "V_B"
    assert scan.gates["top"] == "V_A"


@pytest.mark.parametrize("bad, match", [
    (("V_A", "V_B"),                  "must be a dict"),
    ({"top": "V_A", "bot": "V_B"},    "unknown role"),
    ({"channel": "V_B"},              "at least one gate electrode"),
    # A lone gate cannot be told from a two-gate device whose other gate was
    # forgotten, which is the ambiguity this argument exists to remove.
    ({"top": "V_A"},                  "ambiguous"),
    ({"bottom": "V_A"},               "ambiguous"),
])
def test_gates_rejects_an_ambiguous_declaration(csv_path, bad, match):
    with pytest.raises(ValueError, match=match):
        ACSpectralSweep(str(csv_path), spectra_type="PL", gates=bad)


def test_gate_rows_cannot_be_set_through_curated_labels(csv_path):
    # One fact, one spelling: two mechanisms that could disagree about the wiring
    # is the confusion this whole contract removes.
    for name in ("v_top", "v_bot", "i_top", "i_bot", "i_channel"):
        with pytest.raises(ValueError, match="cannot be set through curated_labels"):
            ACSpectralSweep(str(csv_path), spectra_type="PL",
                                  curated_labels={name: "V_B"})


# ---------------------------------------------------------------------------
# Electrode currents: the source-meter channel's other row
# ---------------------------------------------------------------------------


def test_currents_follow_the_declared_wiring(csv_path):
    # A source-meter channel's bias and current are one terminal, so transposing
    # the wiring must transpose the currents with the voltages — this is the whole
    # reason they are role-named rather than channel-named.
    kw = dict(spectra_type="PL")
    forward = ACSpectralSweep(str(csv_path), gates=GATES, **kw)
    swapped = ACSpectralSweep(
        str(csv_path), gates={"top": "V_B", "bottom": "V_A"}, **kw)
    assert np.allclose(forward.i_top, PARAMS["I_A"] * 1e9)
    assert np.allclose(swapped.i_top, PARAMS["I_B"] * 1e9)
    assert np.allclose(forward.i_top, swapped.i_bot)


def test_channel_current_is_the_contacts_own_row(csv_path):
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              gates={"bottom": "V_A", "channel": "V_B"})
    assert np.allclose(s.i_bot, PARAMS["I_A"] * 1e9)
    assert np.allclose(s.i_channel, PARAMS["I_B"] * 1e9)


def test_grounded_electrode_has_a_voltage_but_no_current(csv_path):
    # Grounding fixes the potential, which is why v_channel is zeros; it says
    # nothing about the current, which still flows and simply was not recorded.
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=BOTTOM_ONLY)
    assert np.allclose(s.v_channel, np.zeros(N_SWEEPS))
    with pytest.raises(ValueError, match="not its current"):
        s.i_channel


def test_gate_on_a_non_source_meter_row_has_no_current(csv_path):
    # The sibling row is looked up in a table of what the format's channels are,
    # not guessed from the spelling of the declared row.
    s = ACSpectralSweep(
        str(csv_path), spectra_type="PL",
        gates={"bottom": "Galvo_X", "channel": None})
    assert np.allclose(s.v_bot, PARAMS["Galvo_X"])
    with pytest.raises(ValueError, match="not a source-meter channel"):
        s.i_bot


def test_curated_scales_flips_a_current_with_its_voltage(csv_path):
    # Reversed leads invert both rows at that terminal; each is stated separately
    # because curated_scales is keyed by curated attribute.  The scale *replaces*
    # the registry's, so flipping a current keeps its A -> nA factor: -1.0 here
    # would silently return amps.
    #
    # The units are restated unchanged because a sign flip does not change them:
    # -1e9 still lands in nA.  That is also what silences the rescaled-without-a-
    # unit warning, which is the point of stating them rather than a workaround.
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_scales={"v_top": -1.0, "i_top": -1e9},
                              curated_units ={"v_top": "V",  "i_top": "nA"})
    assert np.allclose(s.v_top, -PARAMS["V_A"])
    assert np.allclose(s.i_top, -PARAMS["I_A"] * 1e9)
    # The raw rows are untouched by a curated scale.
    assert np.allclose(s["I_A"], PARAMS["I_A"])


# ---------------------------------------------------------------------------
# curated_units — the unit a rescaled row is now in (B6)
#
# The registry entry is (row, scale, unit) and all three are declarable.  The
# unit matters beyond the registry's own view: a curated-backed sweep axis reads
# its unit from here, so it reaches sweep_axis_label, the repr and every legend.
# ---------------------------------------------------------------------------


def test_a_unit_override_reaches_the_registry(csv_path):
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_scales={"scanner_x": 12.5},
                              curated_units ={"scanner_x": "µm"})
    assert s.curated_parameters["scanner_x"] == ("Scanner X", 12.5, "µm")
    # The scale still does what it always did.
    assert np.allclose(s.scanner_x, PARAMS["Scanner X"] * 12.5)


def test_a_unit_override_reaches_the_sweep_axis_label(csv_path):
    # This is the defect: converting a piezo row to µm used to leave every label
    # reading V, because the axis took its unit from _SWEEP_TYPES rather than
    # from the registry the override lands in.  Scanner Y is the varying row.
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              sweep="piezo_y",
                              curated_scales={"scanner_y": 12.5},
                              curated_units ={"scanner_y": "µm"})
    assert s.sweep_unit == "µm"
    assert s.sweep_axis_label == r"Piezo $y$ (µm)"
    assert np.allclose(s.sweep_axis, PARAMS["Scanner Y"] * 12.5)


def test_an_explicit_sweep_unit_still_wins_over_the_registry(csv_path):
    # sweep_unit= is the caller's last word on the axis label, unchanged by this.
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              sweep="piezo_y", sweep_unit="nm",
                              curated_scales={"scanner_y": 12.5},
                              curated_units ={"scanner_y": "µm"})
    assert s.sweep_unit == "nm"
    # The registry keeps what it was told; only the axis label was overridden.
    assert s.curated_parameters["scanner_y"][2] == "µm"


def test_a_unit_override_reaches_the_repr_power_line(csv_path):
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_scales={"power": 1.0},
                              curated_units ={"power": "counts"})
    # Startswith, not "Power" in: the Varying line lists "Excitation Power" too.
    power_line = [ln for ln in repr(s).splitlines()
                  if ln.strip().startswith("Power")]
    assert len(power_line) == 1
    assert power_line[0].endswith("counts")


def test_the_curated_backed_sweep_units_are_unchanged_by_the_collapse(csv_path):
    # Regression pin for moving these units out of _SWEEP_TYPES and into the
    # registry: with no override, every axis must read exactly as it always did.
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    expected = {
        "top_voltage"    : "V",
        "bottom_voltage" : "V",
        "power"          : "µW",
        "piezo_y"        : "V",
        "electric_field" : "mV/nm",
        None             : "",
    }
    for sweep, unit in expected.items():
        s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                                  sweep=sweep, geometry=geom)
        assert s.sweep_unit == unit, f"sweep={sweep!r}"


def test_rescaling_without_a_unit_warns(csv_path):
    with pytest.warns(UserWarning, match="curated_scales rescaled") as caught:
        s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                                  curated_scales={"scanner_x": 12.5})
    # The numbers are converted either way — it is the label that is now wrong.
    assert np.allclose(s.scanner_x, PARAMS["Scanner X"] * 12.5)
    assert s.curated_parameters["scanner_x"][2] == "V"
    # The message has to name the entry and the unit left standing.
    assert "'scanner_x'" in str(caught[0].message)
    assert "'V'" in str(caught[0].message)
    # Measured, not assumed: the warning must blame the caller's line.
    assert caught[0].filename == __file__


def test_rescaling_with_a_unit_is_silent(csv_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_scales={"scanner_x": 12.5},
                              curated_units ={"scanner_x": "µm"})
    assert not [w for w in caught if "rescaled" in str(w.message)]


def test_restating_the_scale_in_force_is_not_a_rescale(csv_path):
    # Nothing changed, so there is nothing to warn about: the registry default
    # for power is already this value.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_scales={"power": POWER_SCALE})
    assert not [w for w in caught if "rescaled" in str(w.message)]


def test_a_gate_name_is_accepted_by_curated_units(csv_path):
    # curated_labels refuses these, because a row is a wiring claim and gates=
    # is its one spelling.  A unit claims nothing about wiring, so it is accepted
    # here exactly as curated_scales accepts it.
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_units={"v_top": "mV"})
    assert s.curated_parameters["v_top"] == ("V_A", 1.0, "mV")
    with pytest.raises(ValueError, match="cannot be set through curated_labels"):
        ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_labels={"v_top": "Galvo_X"})


def test_unknown_curated_units_name_rejected(csv_path):
    with pytest.raises(ValueError, match="not a curated parameter"):
        ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_units={"nonsense": "µm"})


def test_a_none_unit_is_refused(csv_path):
    # str(None) would store the string "None" and label an axis with it.  The
    # dimensionless spelling is "", which is what the sweep index uses.
    with pytest.raises(ValueError, match='pass ""'):
        ACSpectralSweep(str(csv_path), spectra_type="PL", gates=GATES,
                              curated_units={"scanner_x": None})


# ---------------------------------------------------------------------------
# Single-gated devices: one gate plus a contact to the TMDC
# ---------------------------------------------------------------------------

# The user-facing case: one electrode drives the bottom gate, the other contacts
# the TMDC to ground it.  "channel": None says the contact is hard-grounded with no
# row recording it, which is what keeps the density reference fixed.
BOTTOM_ONLY = {"bottom": "V_A", "channel": None}


def test_single_gate_declaration_records_the_topology(csv_path):
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              gates=BOTTOM_ONLY, sweep="bottom_voltage")
    assert s.gates == BOTTOM_ONLY
    assert s.is_dual_gated is False
    assert np.allclose(s.v_bot, PARAMS["V_A"])
    assert np.allclose(s.v_channel, np.zeros(N_SWEEPS))   # hard-grounded
    assert np.allclose(s.sweep_axis, PARAMS["V_A"])


def test_channel_on_a_recorded_row_is_read_from_it(csv_path):
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              gates={"bottom": "V_A", "channel": "V_B"})
    assert np.allclose(s.v_channel, PARAMS["V_B"])
    # The channel is not a gate, so it does not enter gate_mode.
    assert s.gate_mode == "bottom-gate only"


def test_single_gate_device_has_no_top_gate_and_no_field(csv_path):
    geom = DeviceGeometry.from_single("WS2", d_hbn_bottom=46)
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              gates=BOTTOM_ONLY, geometry=geom)
    # One gate is one degree of freedom: there is no gate-to-gate difference, so
    # no displacement field, and the error has to say that rather than just refuse.
    for attr in ("v_top", "ef"):
        with pytest.raises(ValueError, match="single-gated device has no"):
            getattr(s, attr)


def test_electric_field_sweep_refused_on_a_single_gate_device(csv_path):
    geom = DeviceGeometry.from_single("WS2", d_hbn_bottom=46)
    with pytest.raises(ValueError, match="does not have"):
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="electric_field", geometry=geom,
                              gates=BOTTOM_ONLY)


def test_channel_requires_its_own_declaration(scan):
    # A dual-gate declaration says nothing about a contact to the TMDC.
    with pytest.raises(ValueError, match="'channel' electrode"):
        scan.v_channel


def test_grounded_electrode_declared_as_none_reads_as_zero(csv_path):
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    s = ACSpectralSweep(
        str(csv_path), spectra_type="PL", geometry=geom,
        gates={"top": None, "bottom": "V_A"},
    )
    assert s.is_dual_gated is True
    assert np.allclose(s.v_top, np.zeros(N_SWEEPS))
    assert np.allclose(s.ef, geom.electric_field(np.zeros(N_SWEEPS),
                                                 PARAMS["V_A"]))
    # "grounded" rather than a row name, so the repr does not invent one.
    assert "top ← grounded" in repr(s)


def test_sweeping_a_grounded_electrode_is_refused(csv_path):
    # Its voltage is zero at every point, so it is not an axis.
    with pytest.raises(ValueError, match="tied to ground"):
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="top_voltage",
                              gates={"top": None, "bottom": "V_A"})


# ---------------------------------------------------------------------------
# Carrier density: what a single gate actually controls
# ---------------------------------------------------------------------------

# eps_0 * 3.9 / 46 nm, the geometric gate capacitance of the fixture below.
C_BOTTOM_46NM = 8.8541878128e-12 * 3.9 / 46e-9      # F/m^2


def test_gate_capacitance_is_geometric_and_uses_only_that_gates_hbn():
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    assert geom.gate_capacitance("bottom") == pytest.approx(C_BOTTOM_46NM)
    # The TMDC is the counter-electrode, not a slab inside the capacitor, so its
    # thickness must not enter: a thicker TMDC changes nothing here.
    thick = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46,
                                       n_layers=5)
    assert thick.gate_capacitance("bottom") == pytest.approx(C_BOTTOM_46NM)
    # And the top gate's own hBN gives a different, smaller capacitance.
    assert geom.gate_capacitance("top") < geom.gate_capacitance("bottom")


def test_gate_capacitance_refuses_a_gate_with_no_dielectric():
    geom = DeviceGeometry.from_single("WS2", d_hbn_bottom=46)
    with pytest.raises(ValueError, match="d_hbn_top is None"):
        geom.gate_capacitance("top")
    with pytest.raises(ValueError, match="gate must be one of"):
        geom.gate_capacitance("side")


def test_carrier_density_per_volt_matches_the_capacitance():
    geom = DeviceGeometry.from_single("WS2", d_hbn_bottom=46)
    n = geom.carrier_density(v_bot=np.array([0.0, 1.0]))
    # dn/dV = C/e, converted to cm^-2: ~4.7e11 cm^-2 per volt for 46 nm hBN.
    per_volt = C_BOTTOM_46NM / 1.602176634e-19 * 1e-4
    assert n[0] == pytest.approx(0.0)
    assert n[1] == pytest.approx(per_volt, rel=1e-6)
    assert per_volt == pytest.approx(4.7e11, rel=0.02)


def test_carrier_density_sums_over_supplied_gates_and_shifts_with_v_ref():
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    v = np.array([0.0, 1.0])
    both = geom.carrier_density(v_top=v, v_bot=v)
    # Each gate injects charge through its own capacitance, so the two add.
    assert np.allclose(both, geom.carrier_density(v_bot=v)
                             + geom.carrier_density(v_top=v))
    # v_ref shifts the reference gate voltage, so it offsets the whole axis.
    shifted = geom.carrier_density(v_bot=v, v_ref=1.0)
    assert np.allclose(shifted, geom.carrier_density(v_bot=v - 1.0))
    with pytest.raises(ValueError, match="at least one gate voltage"):
        geom.carrier_density()


def test_carrier_density_axis_on_a_single_gate_device(csv_path):
    geom = DeviceGeometry.from_single("WSe2", d_hbn_bottom=46)
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="carrier_density", geometry=geom,
                              gates=BOTTOM_ONLY)
    assert np.allclose(s.carrier_density,
                       geom.carrier_density(v_bot=PARAMS["V_A"]))
    assert np.allclose(s.sweep_axis, s.carrier_density)
    assert s.sweep_axis_label == r"$\Delta n$ (cm$^{-2}$)"


def test_carrier_density_needs_a_declared_channel(csv_path):
    # Charge comes from the contact, so a density is defined against it.
    geom = DeviceGeometry.from_single("WSe2", d_hbn_top=53, d_hbn_bottom=46)
    s = ACSpectralSweep(str(csv_path), spectra_type="PL",
                              geometry=geom, gates=GATES)
    with pytest.raises(ValueError, match="'channel' electrode"):
        s.carrier_density
    with pytest.raises(ValueError, match="'channel' electrode"):
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="carrier_density", geometry=geom, gates=GATES)


def test_carrier_density_warns_when_the_channel_is_driven(csv_path):
    # The density is referenced to the contact; a contact that is itself swept
    # moves the reference under the axis, and the file cannot say whether that was
    # a source-drain bias or a wiring mistake.
    geom = DeviceGeometry.from_single("WSe2", d_hbn_bottom=46)
    s = ACSpectralSweep(str(csv_path), spectra_type="PL", geometry=geom,
                              gates={"bottom": "V_A", "channel": "V_B"})
    with pytest.warns(UserWarning, match="reference moves with the axis"):
        s.carrier_density
    # A hard-grounded channel is fixed by declaration, so it must stay silent.
    quiet = ACSpectralSweep(str(csv_path), spectra_type="PL",
                                  geometry=geom, gates=BOTTOM_ONLY)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet.carrier_density


def test_carrier_density_needs_a_geometry(csv_path):
    assert ACSpectralSweep(str(csv_path), spectra_type="PL",
                                 gates=BOTTOM_ONLY).carrier_density is None
    with pytest.raises(ValueError, match="needs a DeviceGeometry"):
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="carrier_density", gates=BOTTOM_ONLY)


def test_carrier_density_sweep_needs_the_gates_hbn_thickness(csv_path):
    # Declared a bottom gate, but the geometry gives no bottom hBN to gate through.
    geom = DeviceGeometry.from_single("WSe2", d_hbn_top=53)
    with pytest.raises(ValueError, match="d_hbn_bottom is None"):
        ACSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="carrier_density", geometry=geom,
                              gates=BOTTOM_ONLY)


def test_repr_says_when_the_wiring_is_undeclared(scan, unwired):
    assert "top ← 'V_A'" in repr(scan)
    assert "bottom ← 'V_B'" in repr(scan)
    # Must render rather than raise, and say plainly that nothing was declared.
    text = repr(unwired)
    assert "not declared" in text
    # No field line, whose sign would be undefined.  Matched on the line rather
    # than on the string, which also appears in the hint above it.
    assert not [ln for ln in text.splitlines() if ln.strip().startswith("E_F")]


# ---------------------------------------------------------------------------
# Block-layout detection and zero-filled (unwritten) blocks
# ---------------------------------------------------------------------------


def test_layout_read_from_header_alone(csv_path):
    layout = _read_block_layout(str(csv_path))
    assert layout["kind"] == "spectral"
    assert layout["block_width"] == 4
    assert layout["n_blocks"] == N_SWEEPS
    assert layout["roles"] == ("Par", "Wavelength", "ExpROI1", "ExpROI2")


def test_label_column_is_not_mistaken_for_a_block(csv_path):
    # The header's own first column is "Parameters Labels"; a prefix match on
    # "Par" would treat it as a block start and give block_width == 1.
    assert _read_block_layout(str(csv_path))["block_width"] == 4


def test_trailing_zero_blocks_dropped(tmp_path):
    # The exporter over-allocates and fills the surplus with literal zeros, not
    # blanks, so nothing NaN-based removes them.  Keeping them would fabricate
    # sweep points that were never measured.
    csv = tmp_path / "overallocated.csv"
    make_spectral_csv(csv, zero_blocks=N_SWEEPS)
    s = ACSpectralSweep(str(csv), spectra_type="PL")

    assert s.n_sweeps == N_SWEEPS
    assert s.n_declared_sweeps == 2 * N_SWEEPS
    assert not (s.spectra.sum(axis=0) == 0).any()
    assert "zero-filled and dropped" in repr(s)
    # Parameters are trimmed in step with the spectra, or every sweep point
    # would be misaligned against its own parameter values.
    assert np.allclose(s.parameters["V_A"], PARAMS["V_A"])
    for arr in s.parameters.values():
        assert arr.shape == (N_SWEEPS,)


def test_interleaved_zero_blocks_warn_and_keep_everything(tmp_path):
    # Interleaving would mean the format model is wrong, so guessing which
    # columns to discard could silently misalign the sweep.  Keep all, warn.
    csv = tmp_path / "interleaved.csv"
    make_spectral_csv(csv, zero_blocks=2, interleave=True)
    with pytest.warns(UserWarning, match="interleaved"):
        s = ACSpectralSweep(str(csv), spectra_type="PL")
    assert s.n_sweeps == N_SWEEPS + 2


def test_all_blocks_unwritten_names_the_metadata_companion(tmp_path):
    # A spectral-layout file whose every block is zero-filled is the metadata
    # companion written alongside a TRPL sweep.
    csv = tmp_path / "companion.csv"
    make_spectral_csv(csv, zero_blocks=0)
    text = csv.read_text().splitlines()
    # Blank out the wavelength and both ROI columns, keeping the Par values.
    rewritten = [text[0]]
    for line in text[1:]:
        cells = line.split(",")
        for i in range(N_SWEEPS):
            for off in (2, 3, 4):                  # Wavelength, ROI1, ROI2
                cells[1 + i * 4 + off - 1] = "0.0"
        rewritten.append(",".join(cells))
    csv.write_text("\n".join(rewritten) + "\n")

    with pytest.raises(ValueError, match="metadata companion"):
        ACSpectralSweep(str(csv), spectra_type="PL")


def test_headerless_csv_named_by_row_count(tmp_path):
    # A 2-row spectrum and a real-space image are both bare numeric grids with no
    # header, so only the row count separates them.  Each must be pointed at the
    # class that reads it, rather than both at whichever was guessed.
    spectrum = tmp_path / "one_spectrum.csv"
    spectrum.write_text("800.0,801.0,802.0\n10.0,11.0,12.0\n")
    with pytest.raises(ValueError, match="SingleSpectrum"):
        ACSpectralSweep(str(spectrum), spectra_type="PL")

    image = tmp_path / "frame.csv"
    image.write_text("1,2,3\n4,5,6\n7,8,9\n")
    with pytest.raises(ValueError, match="ACImgSweep"):
        ACSpectralSweep(str(image), spectra_type="PL")


def test_row_count_in_that_message_is_not_a_capped_read(tmp_path):
    # The check stops after three lines, so it must describe the shape as
    # "more than two rows" and never report the bounded count as the real one.
    image = tmp_path / "tall.csv"
    image.write_text("\n".join(",".join("1" for _ in range(4)) for _ in range(50)) + "\n")
    with pytest.raises(ValueError, match="more than two rows"):
        ACSpectralSweep(str(image), spectra_type="PL")


def test_single_spectrum_rejection_reaches_the_trpl_class_too(tmp_path):
    # Both sweep classes share the layout reader, so both give the same guidance.
    spectrum = tmp_path / "one_spectrum.csv"
    spectrum.write_text("800.0,801.0\n10.0,11.0\n")
    with pytest.raises(ValueError, match="SingleSpectrum"):
        ACTRPLSweep(str(spectrum))


def test_selected_roi_all_zero_warns(tmp_path):
    # ExpROI2 is blank except on two-spot galvo scans, and a flat zero array is
    # otherwise indistinguishable from a valid dark measurement.
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv, params={"V_A": PARAMS["V_A"]})
    text = csv.read_text().splitlines()
    rewritten = [text[0]]
    for line in text[1:]:
        cells = line.split(",")
        for i in range(N_SWEEPS):
            cells[1 + i * 4 + 3] = "0.0"           # ExpROI2
        rewritten.append(",".join(cells))
    csv.write_text("\n".join(rewritten) + "\n")

    with pytest.warns(UserWarning, match="ExpROI2"):
        ACSpectralSweep(str(csv), spectra_type="PL", roi=2)


# ---------------------------------------------------------------------------
# Deprecated ACPLVabScan shim
# ---------------------------------------------------------------------------


def test_shim_warns_and_reproduces_old_defaults(csv_path):
    with pytest.warns(FutureWarning, match="ACSpectralSweep"):
        s = ACPLVabScan(str(csv_path))
    assert s.spectra_type == "PL"
    # Old gate_axis was v_top when no geometry was supplied.
    assert s.sweep_type == "top_voltage"
    assert np.allclose(s.gate_axis, s.v_top)


def test_shim_with_geometry_uses_field_axis(csv_path):
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    with pytest.warns(FutureWarning):
        s = ACPLVabScan(str(csv_path), geometry=geom)
    assert np.allclose(s.gate_axis, s.ef)
    assert s.gate_axis_label == r"$E_F$ (mV/nm)"


def test_shim_translates_old_label_arguments(csv_path):
    # The shim has no unit argument of its own, so power_scale= necessarily
    # arrives without one and the rescaled-without-a-unit warning fires too.
    with pytest.warns(FutureWarning), pytest.warns(UserWarning, match="rescaled"):
        s = ACPLVabScan(str(csv_path), power_scale=1.0,
                              top_gate_label="V_B")
    assert s.curated_parameters["v_top"][0] == "V_B"
    assert np.allclose(s.power, s.parameters["Excitation Power"])


# ---------------------------------------------------------------------------
# DeviceGeometry.to_dict / from_dict
# ---------------------------------------------------------------------------


def test_to_dict_round_trips_a_single_material_stack():
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    geom2 = DeviceGeometry.from_dict(geom.to_dict())
    assert [(l.material, l.n_layers, l.d_monolayer, l.eps) for l in geom2.tmdc_stack] == \
           [(l.material, l.n_layers, l.d_monolayer, l.eps) for l in geom.tmdc_stack]
    assert geom2.d_hbn_top == geom.d_hbn_top
    assert geom2.d_hbn_bottom == geom.d_hbn_bottom
    assert geom2.eps_hbn == geom.eps_hbn
    assert geom2.eps_stack == pytest.approx(geom.eps_stack)


def test_to_dict_round_trips_a_heterostructure():
    # Two distinct layers: guards against a per-layer field (materially, the
    # d_monolayer that was once lost to a variable-name collision in to_dict)
    # being silently shared or overwritten across layers.
    geom = DeviceGeometry(
        tmdc_stack=[StackLayer("WS2", n_layers=2), StackLayer("MoS2", n_layers=1)],
        d_hbn_top=30,
        d_hbn_bottom=40,
    )
    geom2 = DeviceGeometry.from_dict(geom.to_dict())
    assert len(geom2.tmdc_stack) == 2
    assert [(l.material, l.n_layers, l.d_monolayer, l.eps) for l in geom2.tmdc_stack] == \
           [(l.material, l.n_layers, l.d_monolayer, l.eps) for l in geom.tmdc_stack]


def test_to_dict_output_is_actually_json_serializable():
    # Regression test: the original to_dict built each layer's "d_monolayer"
    # entry from the wrong name in scope, so it held a circular reference to
    # the dict being constructed rather than the layer's own thickness.
    import json

    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    d = geom.to_dict()
    assert d["tmdc_stack"][0]["d_monolayer"] == geom.tmdc_stack[0].d_monolayer
    json.dumps(d)  # must not raise on a circular reference


def test_to_dict_label_is_none_unless_explicitly_set():
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    assert geom.label is None
    d = geom.to_dict()
    assert d["label"] is None       # the raw label, not the derived stack_label
    assert geom.stack_label          # stack_label itself still derives a string
    geom2 = DeviceGeometry.from_dict(d)
    assert geom2.label is None
    assert geom2.stack_label == geom.stack_label


def test_to_dict_preserves_an_explicit_label():
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46,
                                       label="hBN/1L-WS2/hBN, back-gated")
    d = geom.to_dict()
    assert d["label"] == "hBN/1L-WS2/hBN, back-gated"
    geom2 = DeviceGeometry.from_dict(d)
    assert geom2.label == geom.label


def test_to_dict_records_a_missing_top_hbn_as_none():
    geom = DeviceGeometry.from_single("WSe2", d_hbn_bottom=46)
    d = geom.to_dict()
    assert d["d_hbn_top"] is None
    geom2 = DeviceGeometry.from_dict(d)
    assert geom2.d_hbn_top is None


def test_from_dict_requires_only_tmdc_stack():
    # d_hbn_top / d_hbn_bottom / eps_hbn / label are optional in from_dict,
    # matching DeviceGeometry.__init__'s own defaults.
    geom = DeviceGeometry.from_dict({"tmdc_stack": [{"material": "WS2"}]})
    assert geom.d_hbn_top is None
    assert geom.d_hbn_bottom is None
    assert geom.eps_hbn == EPS_HBN
    assert geom.label is None


def test_from_dict_builds_stack_layers_from_plain_dicts():
    geom = DeviceGeometry.from_dict({
        "tmdc_stack": [
            {"material": "WSe2", "n_layers": 1, "d_monolayer": 0.65, "eps": 7.0},
        ],
        "d_hbn_top": 12.0,
        "d_hbn_bottom": 18.0,
        "eps_hbn": 3.0,
    })
    layer = geom.tmdc_stack[0]
    assert (layer.material, layer.n_layers, layer.d_monolayer, layer.eps) == \
           ("WSe2", 1, 0.65, 7.0)
    assert geom.d_hbn_top == 12.0
    assert geom.d_hbn_bottom == 18.0
    assert geom.eps_hbn == 3.0
