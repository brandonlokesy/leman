# src/leman/reference/loader.py

import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"

from ..constants import SPECTROSCOPY_TYPES
from .processors.processor import REF_FORMAT_NAME, REF_FORMAT_VERSION

_REF_FORMAT_MAJOR = REF_FORMAT_VERSION.split(".")[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_str(value):
    """Decode HDF5 bytes to str; pass through str and None unchanged."""
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """
    A single reference spectrum.

    Attributes
    ----------
    energy          : photon energy axis (eV)
    intensity       : signal values
    label           : human-readable label
    spectroscopy    : measurement type, one of SPECTROSCOPY_TYPES
    energy_unit     : unit of the energy axis
    intensity_unit  : unit of the intensity axis
    parameter_value : sweep parameter value this spectrum was taken at
    is_default      : whether this is the canonical spectrum for its sweep
    """
    energy:          np.ndarray
    intensity:       np.ndarray
    label:           str            = ""
    spectroscopy:    str            = "PL"
    energy_unit:     str            = "eV"
    intensity_unit:  str            = "counts"
    parameter_value: Optional[float] = None
    is_default:      bool           = False

    def normalised(self) -> np.ndarray:
        """Return intensity normalised to [0, 1]."""
        s = self.intensity
        return (s - s.min()) / (s.max() - s.min())


@dataclass
class SweepSeries:
    """
    A set of spectra taken while sweeping one parameter.

    Attributes
    ----------
    parameter_name : e.g. "electric_field", "power"
    parameter_unit : e.g. "mV/nm", "uW"
    spectra        : {parameter_value: Spectrum} sorted by parameter value
    default_value  : parameter_value of the canonical spectrum
    """
    parameter_name: str
    parameter_unit: str
    spectra:        dict   = field(default_factory=dict)
    default_value:  Optional[float] = None

    def default(self) -> "Spectrum | FieldCondition":
        """Return the canonical entry, matched by nearest key to avoid float drift."""
        if self.default_value is None:
            raise ValueError(f"No default set for sweep '{self.parameter_name}'")
        if self.default_value in self.spectra:
            return self.spectra[self.default_value]
        nearest = min(self.spectra.keys(), key=lambda k: abs(k - self.default_value))
        return self.spectra[nearest]

    def values(self) -> list:
        """Return parameter values in sorted order."""
        return sorted(self.spectra.keys())

    def all_spectra(self) -> list:
        """Return all Spectrum objects in sorted parameter order."""
        return [self.spectra[v] for v in self.values()]

    def __getitem__(self, param_value: float) -> Spectrum:
        return self.spectra[param_value]


@dataclass
class FieldCondition:
    """
    One condition of an outer sweep (e.g. one electric field value),
    containing an energy axis, metadata, and an inner SweepSeries.

    Attributes
    ----------
    parameter_value : value of the outer sweep parameter, e.g. 300.0
    parameter_unit  : unit of the outer sweep parameter, e.g. "mV/nm"
    energy          : photon energy axis (eV), may differ per condition
    energy_unit     : unit of energy axis
    is_default      : whether this is the default outer condition
    sweeps          : {sweep_name: SweepSeries} — inner sweeps, e.g. "power"
    """
    parameter_value: float
    parameter_unit:  str
    energy:          np.ndarray
    energy_unit:     str  = "eV"
    is_default:      bool = False
    sweeps:          dict = field(default_factory=dict)

    def default_spectrum(self, sweep_name: str) -> Spectrum:
        """Return the default spectrum for a named inner sweep."""
        return self.sweeps[sweep_name].default()

    def __getitem__(self, sweep_name: str) -> SweepSeries:
        return self.sweeps[sweep_name]


@dataclass
class ReferenceDataset:
    """
    All data from one paper for one material.

    Attributes
    ----------
    material      : e.g. "2L_WSe2"
    source        : e.g. "Tagarelli2023"
    doi           : publication DOI
    dataset_doi   : Zenodo record DOI
    title         : full paper title
    about         : description from the registry
    spectroscopy  : measurement type, one of SPECTROSCOPY_TYPES
    sweeps        : {sweep_name: SweepSeries}
    """
    material:     str
    source:       str
    doi:          str  = ""
    dataset_doi:  str  = ""
    title:        str  = ""
    about:        str  = ""
    spectroscopy: str  = "PL"
    sweeps:       dict = field(default_factory=dict)
    _single_spectrum: Optional[Spectrum] = field(
        default=None, repr=False, compare=False,
    )

    @property
    def spectrum(self) -> Spectrum:
        """The single spectrum, for non-sweep references.

        Raises
        ------
        AttributeError
            If this dataset contains sweeps rather than a single spectrum.
        """
        if self._single_spectrum is None:
            raise AttributeError(
                "This dataset contains sweeps, not a single spectrum. "
                "Use ref['sweep_name'] instead."
            )
        return self._single_spectrum

    def default_spectrum(self, sweep_name: str) -> Spectrum:
        """Return the default spectrum for a named sweep."""
        return self.sweeps[sweep_name].default()

    def __getitem__(self, sweep_name: str):
        return self.sweeps[sweep_name]

    def __repr__(self):
        if self._single_spectrum is not None:
            sweep_line = "single spectrum"
        else:
            sweep_summary = {k: len(v.spectra) for k, v in self.sweeps.items()}
            sweep_line = f"sweeps={sweep_summary}"
        return (
            f"ReferenceDataset("
            f"material={self.material!r}, "
            f"source={self.source!r}, "
            f"spectroscopy={self.spectroscopy!r}, "
            f"{sweep_line})\n\n"
            f"{'Title':<12}: {self.title}\n"
            f"{'Source':<12}: {self.source}\n"
            f"{'DOI':<12}: {self.doi}\n"
            f"{'Dataset':<12}: {self.dataset_doi}\n"
            f"{'Material':<12}: {self.material}\n"
            f"{'Measurement':<12}: {self.spectroscopy}"
        )


# ---------------------------------------------------------------------------
# Layout readers
# ---------------------------------------------------------------------------

def _read_axis(group: h5py.Group):
    """Read the single dataset inside an ``axes/`` group.

    Returns (values, name, unit).
    """
    names = list(group.keys())
    if len(names) != 1:
        raise ValueError(
            f"Expected exactly one axis dataset, got {names}"
        )
    name = names[0]
    ds = group[name]
    return ds[:].astype(np.float64), name, _as_str(ds.attrs.get("units", ""))


def _load_single(hf: h5py.File, dataset: ReferenceDataset) -> None:
    """Load layout A (single spectrum) into *dataset*."""
    x_values, axis_name, x_unit = _read_axis(hf["axes"])
    intensity = hf["spectra/intensity"][:].astype(np.float64)
    intensity_unit = _as_str(
        hf["spectra/intensity"].attrs.get("units", "counts")
    )

    sp = Spectrum(
        energy=x_values,
        intensity=intensity,
        spectroscopy=dataset.spectroscopy,
        energy_unit=x_unit,
        intensity_unit=intensity_unit,
    )

    dataset._single_spectrum = sp
    dataset.sweeps["spectrum"] = SweepSeries(
        parameter_name="none",
        parameter_unit="",
        spectra={0: sp},
        default_value=0,
    )


def _load_1d_sweep(hf: h5py.File, dataset: ReferenceDataset) -> None:
    """Load layout B (1-D sweep) into *dataset*."""
    x_values, axis_name, x_unit = _read_axis(hf["axes"])

    sweep_ds = hf["sweep/values"]
    sweep_vals = sweep_ds[:].astype(np.float64)
    sweep_name = _as_str(sweep_ds.attrs.get("name", "sweep"))
    sweep_unit = _as_str(sweep_ds.attrs.get("units", ""))
    default_idx = int(hf["sweep/default_index"][()])

    intensity_2d = hf["spectra/intensity"][:].astype(np.float64)
    intensity_unit = _as_str(
        hf["spectra/intensity"].attrs.get("units", "counts")
    )

    spectra = {}
    for i, sv in enumerate(sweep_vals):
        spectra[float(sv)] = Spectrum(
            energy=x_values,
            intensity=intensity_2d[:, i],
            label=f"{sv}",
            spectroscopy=dataset.spectroscopy,
            energy_unit=x_unit,
            intensity_unit=intensity_unit,
            parameter_value=float(sv),
            is_default=(i == default_idx),
        )

    dataset.sweeps[sweep_name] = SweepSeries(
        parameter_name=sweep_name,
        parameter_unit=sweep_unit,
        spectra=spectra,
        default_value=float(sweep_vals[default_idx]),
    )


def _load_nested(hf: h5py.File, dataset: ReferenceDataset) -> None:
    """Load layout C (nested sweep) into *dataset*."""
    meta = hf["sweep_meta"]
    outer_name = _as_str(meta.attrs["outer_name"])
    outer_unit = _as_str(meta.attrs["outer_units"])
    outer_default = float(meta.attrs["default_value"])

    cond_root = hf["conditions"]
    conditions = {}

    for idx_str in sorted(cond_root.keys(), key=int):
        grp = cond_root[idx_str]
        outer_value = float(grp.attrs["outer_value"])
        is_default = bool(grp.attrs["is_default"])

        x_values, axis_name, x_unit = _read_axis(grp["axes"])

        inner_ds = grp["sweep/values"]
        inner_vals = inner_ds[:].astype(np.float64)
        inner_name = _as_str(inner_ds.attrs.get("name", "sweep"))
        inner_unit = _as_str(inner_ds.attrs.get("units", ""))
        inner_default_idx = int(grp["sweep/default_index"][()])

        intensity_2d = grp["spectra/intensity"][:].astype(np.float64)
        intensity_unit = _as_str(
            grp["spectra/intensity"].attrs.get("units", "counts")
        )

        inner_spectra = {}
        for i, iv in enumerate(inner_vals):
            inner_spectra[float(iv)] = Spectrum(
                energy=x_values,
                intensity=intensity_2d[:, i],
                label=f"{iv}",
                spectroscopy=dataset.spectroscopy,
                energy_unit=x_unit,
                intensity_unit=intensity_unit,
                parameter_value=float(iv),
                is_default=(i == inner_default_idx),
            )

        inner_sweep = SweepSeries(
            parameter_name=inner_name,
            parameter_unit=inner_unit,
            spectra=inner_spectra,
            default_value=float(inner_vals[inner_default_idx]),
        )

        condition = FieldCondition(
            parameter_value=outer_value,
            parameter_unit=outer_unit,
            energy=x_values,
            energy_unit=x_unit,
            is_default=is_default,
            sweeps={inner_name: inner_sweep},
        )
        conditions[outer_value] = condition

    dataset.sweeps[outer_name] = SweepSeries(
        parameter_name=outer_name,
        parameter_unit=outer_unit,
        spectra=conditions,
        default_value=outer_default,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_reference(material: str, source: str) -> ReferenceDataset:
    """
    Load a processed reference dataset from its HDF5 file.

    Parameters
    ----------
    material : str
        Material identifier, e.g. ``"2L_WSe2"``.
    source : str
        First-author-year citation key, e.g. ``"Tagarelli2023"``.

    Returns
    -------
    ReferenceDataset

    Raises
    ------
    FileNotFoundError
        If the ``.h5`` file does not exist.
    ValueError
        If the file uses an older or unrecognised format.

    Examples
    --------
    >>> ref = load_reference("2L_WSe2", "Tagarelli2023")
    >>> sp  = ref.default_spectrum("electric_field")
    >>> ax.plot(sp.energy, sp.normalised())

    >>> for val in ref["electric_field"].values():
    ...     cond = ref["electric_field"][val]
    ...     sp   = cond.default_spectrum("power")
    ...     ax.plot(sp.energy, sp.normalised(), label=f"E={val} mV/nm")
    """
    path = DATA_DIR / f"{material}__{source}.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"No data found for {material!r} / {source!r}.\n"
            f"Expected: {path}\n"
            f"Run `python -m leman.reference.registry` to download and process."
        )

    with h5py.File(path, "r") as hf:
        fmt = _as_str(hf.attrs.get("format"))
        if fmt != REF_FORMAT_NAME:
            raise ValueError(
                f"'{path.name}' uses format {fmt!r}, not {REF_FORMAT_NAME!r}.\n"
                f"Run `python -m leman.reference.registry` to regenerate."
            )
        version = _as_str(hf.attrs.get("format_version", ""))
        if version.split(".")[0] != _REF_FORMAT_MAJOR:
            raise ValueError(
                f"'{path.name}' is format version {version!r}; "
                f"this reader requires {_REF_FORMAT_MAJOR}.x.\n"
                f"Run `python -m leman.reference.registry` to regenerate."
            )

        dataset = ReferenceDataset(
            material     = _as_str(hf.attrs.get("material", material)),
            source       = _as_str(hf.attrs.get("source", source)),
            doi          = _as_str(hf.attrs.get("doi", "")),
            dataset_doi  = _as_str(hf.attrs.get("dataset_doi", "")),
            title        = _as_str(hf.attrs.get("title", "")),
            about        = _as_str(hf.attrs.get("about", "")),
            spectroscopy = _as_str(hf.attrs.get("spectroscopy", "PL")),
        )

        if "conditions" in hf:
            _load_nested(hf, dataset)
        elif "sweep" in hf:
            _load_1d_sweep(hf, dataset)
        elif "axes" in hf and "spectra" in hf:
            _load_single(hf, dataset)
        else:
            raise ValueError(
                f"'{path.name}' has an unrecognised layout "
                f"(no 'conditions/', 'sweep/', or 'axes/' + 'spectra/' groups)."
            )

    return dataset
