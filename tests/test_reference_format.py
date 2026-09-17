# tests/test_reference_format.py
#
# Round-trip tests for the leman.reference HDF5 schema.
# No network access — all data is synthetic.

import numpy as np
import h5py
import pytest

from leman.reference.processors.processor import (
    Processor,
    REF_FORMAT_NAME,
    REF_FORMAT_VERSION,
)
from leman.reference.loader import load_reference, ReferenceDataset


# ---------------------------------------------------------------------------
# Minimal test processor that writes synthetic data via base-class helpers
# ---------------------------------------------------------------------------

class _TestProcessor(Processor):
    """A processor that writes synthetic data for testing."""

    def __init__(self, meta, out_dir):
        super().__init__(meta, out_dir)
        self.layout = None
        self.data = {}

    def run(self):
        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            if self.layout == "single":
                self._write_single_spectrum(hf, **self.data)

            elif self.layout == "sweep":
                self._write_1d_sweep(hf, **self.data)

            elif self.layout == "nested":
                self._write_nested_sweep(hf, **self.data)

            elif self.layout == "provenance":
                self._write_single_spectrum(
                    hf,
                    self.data["axis_values"],
                    "energy", "eV", "Energy",
                    self.data["intensity"],
                    "counts",
                )
                self._write_provenance(hf, **self.data["provenance"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_META = {
    "material":    "test_mat",
    "source":      "TestSource2026",
    "title":       "A test paper",
    "doi":         "10.0000/test",
    "dataset_doi": "10.0000/testdata",
    "spectroscopy": "PL",
}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the loader at a temporary directory instead of the real data."""
    monkeypatch.setattr(
        "leman.reference.loader.DATA_DIR", tmp_path,
    )
    return tmp_path


def _make_processor(data_dir):
    return _TestProcessor(_META, data_dir)


# ---------------------------------------------------------------------------
# Tests — root attributes
# ---------------------------------------------------------------------------

def test_root_attrs_complete(data_dir):
    proc = _make_processor(data_dir)
    proc.layout = "single"
    proc.data = dict(
        axis_values=np.linspace(1.5, 2.0, 50),
        axis_name="energy", axis_units="eV", axis_label="Energy",
        intensity=np.random.rand(50),
        intensity_units="counts",
    )
    proc.run()

    with h5py.File(proc.out_path, "r") as hf:
        assert hf.attrs["format"] == REF_FORMAT_NAME
        assert hf.attrs["format_version"] == REF_FORMAT_VERSION
        assert "created" in hf.attrs
        assert "toolkit_version" in hf.attrs
        assert hf.attrs["material"] == "test_mat"
        assert hf.attrs["source"] == "TestSource2026"
        assert hf.attrs["spectroscopy"] == "PL"


# ---------------------------------------------------------------------------
# Tests — single spectrum round-trip
# ---------------------------------------------------------------------------

def test_single_spectrum_roundtrip(data_dir):
    energy = np.linspace(1.5, 2.0, 100)
    intensity = np.sin(energy * 10)

    proc = _make_processor(data_dir)
    proc.layout = "single"
    proc.data = dict(
        axis_values=energy,
        axis_name="energy", axis_units="eV", axis_label="Energy",
        intensity=intensity,
        intensity_units="arb. u.",
    )
    proc.run()

    ref = load_reference("test_mat", "TestSource2026")
    assert isinstance(ref, ReferenceDataset)
    assert ref.material == "test_mat"
    assert ref.spectroscopy == "PL"

    sp = ref.spectrum
    np.testing.assert_array_equal(sp.energy, energy)
    np.testing.assert_array_equal(sp.intensity, intensity)
    assert sp.energy_unit == "eV"
    assert sp.intensity_unit == "arb. u."


def test_single_spectrum_sweep_access(data_dir):
    """Single-spectrum files are also accessible via ref['spectrum'].default()."""
    energy = np.linspace(1.5, 2.0, 50)
    intensity = np.ones(50)

    proc = _make_processor(data_dir)
    proc.layout = "single"
    proc.data = dict(
        axis_values=energy,
        axis_name="energy", axis_units="eV", axis_label="Energy",
        intensity=intensity,
        intensity_units="counts",
    )
    proc.run()

    ref = load_reference("test_mat", "TestSource2026")
    sp_via_property = ref.spectrum
    sp_via_sweep = ref["spectrum"].default()
    assert sp_via_property is sp_via_sweep


# ---------------------------------------------------------------------------
# Tests — 1-D sweep round-trip
# ---------------------------------------------------------------------------

def test_1d_sweep_roundtrip(data_dir):
    n_pixels, n_sweeps = 80, 5
    energy = np.linspace(1.4, 1.9, n_pixels)
    voltages = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    intensity_2d = np.random.rand(n_pixels, n_sweeps)

    proc = _make_processor(data_dir)
    proc.layout = "sweep"
    proc.data = dict(
        axis_values=energy,
        axis_name="energy", axis_units="eV", axis_label="Energy",
        sweep_values=voltages,
        sweep_name="gate_voltage", sweep_units="V", sweep_label="Gate voltage",
        default_index=2,
        intensity_2d=intensity_2d,
        intensity_units="dimensionless",
    )
    proc.run()

    ref = load_reference("test_mat", "TestSource2026")
    series = ref["gate_voltage"]
    assert len(series.spectra) == n_sweeps
    assert series.parameter_name == "gate_voltage"
    assert series.parameter_unit == "V"

    default_sp = series.default()
    assert default_sp.parameter_value == 0.0
    assert default_sp.is_default
    np.testing.assert_array_equal(default_sp.energy, energy)
    np.testing.assert_array_equal(default_sp.intensity, intensity_2d[:, 2])


def test_1d_sweep_spectrum_property_raises(data_dir):
    """A sweep dataset does not have a .spectrum property."""
    proc = _make_processor(data_dir)
    proc.layout = "sweep"
    proc.data = dict(
        axis_values=np.linspace(1.5, 2.0, 20),
        axis_name="energy", axis_units="eV", axis_label="Energy",
        sweep_values=np.array([0.0, 1.0]),
        sweep_name="x", sweep_units="", sweep_label="X",
        default_index=0,
        intensity_2d=np.ones((20, 2)),
        intensity_units="counts",
    )
    proc.run()

    ref = load_reference("test_mat", "TestSource2026")
    with pytest.raises(AttributeError, match="sweeps"):
        _ = ref.spectrum


# ---------------------------------------------------------------------------
# Tests — nested sweep round-trip
# ---------------------------------------------------------------------------

def test_nested_roundtrip(data_dir):
    energy_a = np.linspace(1.3, 1.8, 100)
    energy_b = np.linspace(1.4, 1.9, 120)
    power_a = np.array([10.0, 20.0, 40.0])
    power_b = np.array([5.0, 15.0, 25.0, 35.0])
    int_a = np.random.rand(100, 3)
    int_b = np.random.rand(120, 4)

    conditions = [
        dict(
            outer_value=300.0, is_default=False,
            axis_values=energy_a, axis_name="energy",
            axis_units="eV", axis_label="Energy",
            inner_values=power_a, inner_name="power",
            inner_units="uW", inner_label="Power",
            default_index=2,
            intensity_2d=int_a, intensity_units="counts",
        ),
        dict(
            outer_value=0.0, is_default=True,
            axis_values=energy_b, axis_name="energy",
            axis_units="eV", axis_label="Energy",
            inner_values=power_b, inner_name="power",
            inner_units="uW", inner_label="Power",
            default_index=1,
            intensity_2d=int_b, intensity_units="counts",
        ),
    ]
    sweep_meta = dict(
        outer_name="electric_field",
        outer_units="mV/nm",
        outer_label="Electric field",
        default_value=0.0,
    )

    proc = _make_processor(data_dir)
    proc.layout = "nested"
    proc.data = dict(conditions=conditions, sweep_meta=sweep_meta)
    proc.run()

    ref = load_reference("test_mat", "TestSource2026")
    outer = ref["electric_field"]
    assert len(outer.spectra) == 2

    cond_300 = outer[300.0]
    assert isinstance(cond_300, FieldCondition)
    np.testing.assert_array_equal(cond_300.energy, energy_a)
    assert len(cond_300["power"].spectra) == 3

    cond_0 = outer[0.0]
    np.testing.assert_array_equal(cond_0.energy, energy_b)
    assert cond_0.is_default
    assert len(cond_0["power"].spectra) == 4

    default_sp = cond_0.default_spectrum("power")
    np.testing.assert_array_equal(default_sp.intensity, int_b[:, 1])


def test_nested_different_axes(data_dir):
    """Conditions with different energy axis lengths load independently."""
    conditions = [
        dict(
            outer_value=1.0, is_default=True,
            axis_values=np.linspace(1.0, 2.0, 50),
            axis_name="energy", axis_units="eV", axis_label="Energy",
            inner_values=np.array([0.0]),
            inner_name="x", inner_units="", inner_label="X",
            default_index=0,
            intensity_2d=np.ones((50, 1)),
            intensity_units="counts",
        ),
        dict(
            outer_value=2.0, is_default=False,
            axis_values=np.linspace(1.2, 2.2, 70),
            axis_name="energy", axis_units="eV", axis_label="Energy",
            inner_values=np.array([0.0]),
            inner_name="x", inner_units="", inner_label="X",
            default_index=0,
            intensity_2d=np.ones((70, 1)),
            intensity_units="counts",
        ),
    ]

    proc = _make_processor(data_dir)
    proc.layout = "nested"
    proc.data = dict(
        conditions=conditions,
        sweep_meta=dict(
            outer_name="field", outer_units="V",
            outer_label="Field", default_value=1.0,
        ),
    )
    proc.run()

    ref = load_reference("test_mat", "TestSource2026")
    cond_1 = ref["field"][1.0]
    cond_2 = ref["field"][2.0]
    assert cond_1.energy.shape == (50,)
    assert cond_2.energy.shape == (70,)


# ---------------------------------------------------------------------------
# Tests — format gate
# ---------------------------------------------------------------------------

def test_old_format_refused(data_dir):
    """An HDF5 file without the format tag is refused."""
    path = data_dir / "test_mat__TestSource2026.h5"
    with h5py.File(path, "w") as hf:
        hf.attrs["material"] = "test_mat"
        hf.attrs["source"] = "TestSource2026"
        hf.create_dataset("energy", data=np.array([1.0, 2.0]))
        hf.create_dataset("spectra", data=np.array([0.5, 0.8]))

    with pytest.raises(ValueError, match="regenerate"):
        load_reference("test_mat", "TestSource2026")


def test_wrong_major_version_refused(data_dir):
    """A file with a future major version is refused."""
    path = data_dir / "test_mat__TestSource2026.h5"
    with h5py.File(path, "w") as hf:
        hf.attrs["format"] = REF_FORMAT_NAME
        hf.attrs["format_version"] = "2.0"
        hf.attrs["material"] = "test_mat"

    with pytest.raises(ValueError, match="format version"):
        load_reference("test_mat", "TestSource2026")


# ---------------------------------------------------------------------------
# Tests — provenance
# ---------------------------------------------------------------------------

def test_provenance_stored(data_dir):
    proc = _make_processor(data_dir)
    proc.layout = "provenance"
    proc.data = dict(
        axis_values=np.linspace(1.5, 2.0, 30),
        intensity=np.ones(30),
        provenance=dict(
            summary="test pipeline",
            smooth_kernel=3,
            remove_cosmics=False,
        ),
    )
    proc.run()

    with h5py.File(proc.out_path, "r") as hf:
        assert "provenance" in hf
        assert hf["provenance"].attrs["pipeline_summary"] == "test pipeline"
        assert hf["provenance"].attrs["smooth_kernel"] == 3
        assert hf["provenance"].attrs["remove_cosmics"] == False


# ---------------------------------------------------------------------------
# Tests — shape validation in writers
# ---------------------------------------------------------------------------

def test_single_spectrum_shape_mismatch_raises(data_dir):
    proc = _make_processor(data_dir)
    with h5py.File(proc.out_path, "w") as hf:
        proc._write_metadata(hf)
        with pytest.raises(ValueError, match="shapes differ"):
            proc._write_single_spectrum(
                hf,
                np.ones(10), "energy", "eV", "Energy",
                np.ones(12), "counts",
            )


def test_1d_sweep_shape_mismatch_raises(data_dir):
    proc = _make_processor(data_dir)
    with h5py.File(proc.out_path, "w") as hf:
        proc._write_metadata(hf)
        with pytest.raises(ValueError, match="does not match"):
            proc._write_1d_sweep(
                hf,
                np.ones(10), "energy", "eV", "Energy",
                np.ones(3), "x", "", "X",
                0,
                np.ones((10, 5)),  # n_sweeps mismatch: 5 != 3
                "counts",
            )


# Bring FieldCondition into scope for the nested test's isinstance check.
from leman.reference.loader import FieldCondition
