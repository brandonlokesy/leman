# leman/hdf5.py
"""
Self-describing HDF5 storage for AttoCube sweeps.

A raw AttoCube CSV export is a wide text grid that records the numbers but not
what they mean: nothing in it says whether the spectra are PL or reflectance,
which parameter was scanned, which acquisition channel drove which gate, or what
stack the device was.  That context currently lives in whoever ran the
measurement.  This module writes it *next to the data*, so a scan can be re-read
months later — or by someone else in the group — and come back as the same
object.

One format serves both measured axes.  A TRPL sweep additionally arrives as a
*directory* of per-point CSVs plus a metadata companion, so writing it collapses
four files into one archive: the committed 11.57 MB example becomes 0.069 MB.

Layout
------
::

    /                              attrs: format, format_version, created,
                                          toolkit_version
    ├── metadata/                  attrs: spectra_type, axis_kind, sweep_type,
    │   │                                 sweep_label, sweep_unit, source_file,
    │   │                                 curated_labels, curated_scales,
    │   │                                 curated_units,
    │   │                                 gates, fast_sweep, slow_sweep,
    │   │                                 n_fast, n_slow (only if the shape was
    │   │                                 asserted), fast_group_by, slow_group_by
    │   │                                 (only where they differ from the axis),
    │   │                                 cosmic_rays (each only if declared),
    │   │                                 + provenance (see below)
    │   └── geometry/              attrs: d_hbn_top, d_hbn_bottom, eps_hbn, label
    │       └── tmdc_stack         structured dataset, one row per layer
    ├── auxiliary/                 written only if either spectrum was supplied
    │   ├── bg_spectrum            (n_points,)  attrs: units
    │   └── reference              (n_points,)  attrs: units, contrast_mode,
    │                                                 scale_applied
    ├── axes/
    │   └── wavelength | time      (n_points,)          float64, attrs: units
    ├── parameters/                one dataset per instrument row,
    │                              (n_sweeps,) float64, raw file units
    └── spectra/  or  decays/
        ├── roi1                   (n_points, n_sweeps) compressed   [spectral]
        ├── roi2                   (n_points, n_sweeps) compressed   [spectral]
        └── counts                 (n_points, n_sweeps) compressed   [temporal]

The axis dataset is **named for the physical quantity it holds**, so ``h5ls``
alone says what kind of measurement a file contains; ``metadata/axis_kind``
records it authoritatively as well.  Which loader may read a file follows from
it, exactly as the header decides for a CSV — :class:`ACSpectralSweep` and
:class:`ACTRPLSweep` each reject the other's archives by name.  There is no
factory: the caller names the class they expect, and is told when they are wrong.

Scalars live in group **attributes** rather than as 0-d datasets: that is the
idiomatic HDF5 home for them, and it keeps ``h5ls -v`` output readable.

What is deliberately *not* stored
---------------------------------
Everything derivable from what is:

* **The energy axis** — it is ``hc/λ``, and storing it invites the two copies to
  disagree.
* **The energy-space spectra**, with or without Jacobian and background.  These
  depend on ``apply_jacobian`` and ``bg_region_nm``, which are *loading choices*
  rather than properties of the measurement.  Writing them would freeze one
  session's choices into the archive, and a later reader could not tell the
  stored array from a raw one.
* **The cosmic-ray-repaired spectra** — the same argument, and the reason the
  stored ``spectra`` is the array the source file held.
* **The sweep axis** — a field axis is ``eps_stack``-weighted arithmetic on
  ``V_A``/``V_B`` and the geometry, all of which are stored.
* **The nest coordinates** — ``fast_sweep`` / ``slow_sweep`` record the two axis
  *names*, and the values come back off the parameter rows the same way they were
  read the first time.  A stored copy could disagree with the rows beside it.
* **The nest shape, when the readings established it.** Re-deriving it on read is
  what keeps the overlap checks doing their job.  ``n_fast`` / ``n_slow`` are stored
  only when the writing session *asserted* the shape, which is a declaration the
  readings cannot recover — the same reasoning as ``gates``.

A declaration is a different thing from a correction, and the two are stored the
same way but read back differently.  ``gates``, ``n_fast`` / ``n_slow``,
``fast_group_by`` / ``slow_group_by`` and ``fast_sweep`` / ``slow_sweep``
say what the measurement *was* — how it was wired, how it was nested — and are
**replayed**, because losing them would turn a stated fact back into an unknown
one.  So ``apply_jacobian``, ``bg_region_nm`` / ``bg_region_ns``, ``cosmic_rays``, and
the ``bg_spectrum`` / ``reference`` spectra *are* recorded, but as provenance of
the session that wrote the file, exposed on read as
:attr:`~leman.loaders.ACSpectralSweep.source_metadata`.  They
are **not** replayed: re-applying a correction because a file mentions one would
make loading a decision, which is the one thing loading must not be.  Raw arrays
in, corrections opt-in — the same rule as everywhere else in the package.

The auxiliary spectra — a measured background, and a bare-substrate reference —
have their own group rather than sitting under ``metadata/``: they are measured
arrays on the file's own axis, in the same units as the signal, and not
descriptions of the measurement.  The two scalars that qualify a reference,
``contrast_mode`` and ``scale_applied``, are attributes **of the reference
dataset**, so the factor a contrast depends on travels with the array it applies
to.  ``scale_applied`` records a scaling already present in the stored values.

Both are stored as **arrays rather than paths**, so the archive stands alone once
the substrate CSV has moved: a contrast can be rebuilt from the ``.h5`` by passing
``reference=scan.source_metadata["reference"]`` back in, which keeps the
not-replayed rule while losing nothing.  Do not pass ``reference_scale`` alongside
it — the stored values already carry that factor, and supplying it again applies
it twice.

Functions
---------
write_sweep
    An :class:`~leman.loaders.ACSpectralSweep` or
    :class:`~leman.loaders.ACTRPLSweep` -> ``.h5``.
read_sweep
    ``.h5`` -> the payload dict the loader builds from.  Called for you when a
    path with an HDF5 suffix is passed to the loader; use it directly only to
    inspect a file without constructing a scan.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import h5py
import numpy as np

from . import __version__

# Bump the minor when a field is added (readers stay compatible); bump the major
# only for a change that an older reader would mis-read.  2.0 moved the auxiliary
# spectra out of /metadata into /auxiliary, which is exactly such a change: a
# reader that looks in the old place finds nothing there and drops a recorded
# reference without erroring.  Hence the major gate on read — a silently missing
# reference is worse than a refused file.
FORMAT_NAME    = "leman.attocube_sweep"
FORMAT_VERSION = "2.3"
_FORMAT_MAJOR  = FORMAT_VERSION.split(".")[0]

# Files written under the old package name or the even older single-axis name.
_LEGACY_FORMAT_NAMES = (
    "tmdc_optics_tools.attocube_sweep",
    "tmdc_optics_tools.spectral_sweep",
)

# How each loader's layout is stored: the axis dataset's name carries the physical
# quantity, so `h5ls` alone says what kind of measurement a file holds.
_AXIS_KIND_FOR_LAYOUT = {
    "spectral": {"name": "wavelength", "units": "nm",
                 "label": "Wavelength (nm)", "group": "spectra"},
    "temporal": {"name": "time",       "units": "ns",
                 "label": "Time (ns)",  "group": "decays"},
}

# Which loader reads a stored axis kind back, for the error when they disagree.
_CLASS_FOR_AXIS_KIND = {
    "wavelength": "ACSpectralSweep",
    "time":       "ACTRPLSweep",
}

# Keys stored as JSON strings because HDF5 attributes have no mapping type.
_JSON_ATTRS = ("curated_labels", "curated_scales", "curated_units",
               "gates", "cosmic_rays")

# Structured dtype for the TMDC stack: one row per layer, so layer *order* — the
# physical stacking sequence — is carried by the dataset rather than by dataset
# names that would need sorting to recover it.
_STACK_DTYPE = np.dtype([
    ("material",    h5py.string_dtype()),
    ("n_layers",    "i4"),
    ("d_monolayer", "f8"),
    ("eps",         "f8"),
])


# ---------------------------------------------------------------------------
# Label <-> HDF5 name
# ---------------------------------------------------------------------------

def _hdf5_key(label: str) -> str:
    """
    Sanitise an instrument row label into an HDF5-safe dataset name.

    ``/`` is the HDF5 path separator, so a label containing one would silently
    create a subgroup.  The original label is kept as a ``label`` attribute on
    the dataset, and that — not the sanitised name — is what is read back.
    """
    return label.replace("/", "_").strip()


# ---------------------------------------------------------------------------
# Geometry <-> HDF5
# ---------------------------------------------------------------------------

def _write_geometry(parent: h5py.Group, geometry) -> None:
    """Store a :class:`~leman.loaders.DeviceGeometry` under *parent*."""
    grp = parent.create_group("geometry")
    for name in ("d_hbn_top", "d_hbn_bottom"):
        value = getattr(geometry, name)
        # None (no hBN on that side) has no HDF5 attribute type; NaN round-trips
        # through float attributes and is unambiguous for a thickness.
        grp.attrs[name] = float("nan") if value is None else float(value)
    grp.attrs["eps_hbn"] = float(geometry.eps_hbn)
    grp.attrs["label"]   = geometry.label or ""

    stack = np.empty(len(geometry.tmdc_stack), dtype=_STACK_DTYPE)
    for i, layer in enumerate(geometry.tmdc_stack):
        stack[i] = (layer.material, layer.n_layers,
                    layer.d_monolayer, layer.eps)
    grp.create_dataset("tmdc_stack", data=stack)


def _read_geometry(grp: h5py.Group):
    """Rebuild a :class:`~leman.loaders.DeviceGeometry` from *grp*."""
    from .loaders import DeviceGeometry, StackLayer

    def _thickness(name):
        value = float(grp.attrs[name])
        return None if np.isnan(value) else value

    stack = [
        StackLayer(
            material    = row["material"].decode() if isinstance(row["material"], bytes)
                          else str(row["material"]),
            n_layers    = int(row["n_layers"]),
            d_monolayer = float(row["d_monolayer"]),
            eps         = float(row["eps"]),
        )
        for row in grp["tmdc_stack"][()]
    ]
    return DeviceGeometry(
        tmdc_stack   = stack,
        d_hbn_top    = _thickness("d_hbn_top"),
        d_hbn_bottom = _thickness("d_hbn_bottom"),
        eps_hbn      = float(grp.attrs["eps_hbn"]),
        label        = grp.attrs["label"] or None,
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_sweep(
    scan,
    path,
    compression : str  = "gzip",
    overwrite   : bool = False,
):
    """
    Write an AttoCube sweep to HDF5.

    Every signal array and instrument parameter row is stored verbatim in file
    units, so the result is a lossless replacement for the source CSV — not a
    processed derivative of it.  For a spectral sweep that means **both** ROIs,
    since ``ExpROI2`` carries the remote spot of a two-spot galvo scan.  See the
    module docstring for the layout and for what is intentionally left out.

    Parameters
    ----------
    scan : ACSpectralSweep or ACTRPLSweep
        The sweep to store.  Its axis kind decides the layout written.
    path : str or Path
        Destination file.  A missing parent directory is created.
    compression : str or None
        Dataset compression passed to h5py, applied to the two spectra arrays
        (the largest datasets by far).  Default ``"gzip"``; ``None`` disables it.
    overwrite : bool
        Replace *path* if it already exists.  Default ``False``, which raises
        ``FileExistsError`` — writing over an existing archive is destructive and
        so is opt-in.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    FileExistsError
        If *path* exists and *overwrite* is ``False``.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"'{path}' already exists. Pass overwrite=True to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as hf:
        hf.attrs["format"]          = FORMAT_NAME
        hf.attrs["format_version"]  = FORMAT_VERSION
        hf.attrs["created"]         = _dt.datetime.now().astimezone().isoformat()
        hf.attrs["toolkit_version"] = __version__

        kind = _AXIS_KIND_FOR_LAYOUT[scan._LAYOUT_KIND]

        meta = hf.create_group("metadata")
        meta.attrs["spectra_type"]   = scan.spectra_type
        meta.attrs["axis_kind"]      = kind["name"]
        meta.attrs["sweep_type"]     = scan.sweep_type
        meta.attrs["sweep_label"]    = scan.sweep_label
        meta.attrs["sweep_unit"]     = scan.sweep_unit
        meta.attrs["source_file"]    = scan.path

        # The resolved curated map, so an unusual power scale is restored rather
        # than silently reverting to the class defaults.  All three elements go,
        # because a scale without its unit says nothing about what the numbers
        # are — the reader would put a µm/V-rescaled piezo row back under "V".
        # Unlike the labels, the units need no gate strip on read: a unit is not
        # a claim about which channel reached which electrode.
        curated = scan.curated_parameters
        meta.attrs["curated_labels"] = json.dumps(
            {name: cfg[0] for name, cfg in curated.items()})
        meta.attrs["curated_scales"] = json.dumps(
            {name: cfg[1] for name, cfg in curated.items()})
        meta.attrs["curated_units"]  = json.dumps(
            {name: cfg[2] for name, cfg in curated.items()})

        # The channel-to-gate mapping, written *only* when the writing session
        # declared it.  The curated dump above always carries a resolved label for
        # both gate rows and so cannot distinguish a declared mapping from a
        # defaulted one; keeping this separate is what stops a round trip turning
        # an unstated wiring into a stated one.
        if scan.gates is not None:
            meta.attrs["gates"] = json.dumps(scan.gates)

        # The 2-D nest, written only when the writing session declared one — a
        # declaration like `gates`, replayed on read, not provenance.  Only the two
        # axis *names* are stored: the coordinates are re-derived from the
        # parameter rows, and a second copy could disagree with them.
        if scan.nesting is not None:
            meta.attrs["fast_sweep"] = scan.nesting.fast_type
            meta.attrs["slow_sweep"] = scan.nesting.slow_type
            # The shape, written *only* when the writing session asserted it. A nest
            # whose shape the readings established re-establishes it on read, which
            # keeps the overlap checks doing their job; storing the counts either way
            # would turn every round trip into an assertion and downgrade those
            # checks to warnings.  Without this, a file that needed the assertion
            # would write successfully and then refuse to read back.
            if scan.nesting.asserted:
                meta.attrs["n_fast"] = int(scan.nesting.n_fast)
                meta.attrs["n_slow"] = int(scan.nesting.n_slow)
            # The grouping rows, written only where they differ from the axis they
            # group.  Which row drives an instrument is per-session configuration that
            # nothing in the file states, so losing it would turn a stated fact back
            # into an unknown one — and the shape would then be resolved from the
            # labelled row, which is exactly what could not resolve it.
            for side in ("fast", "slow"):
                group = getattr(scan.nesting, f"{side}_group")
                if group is not None and group != getattr(scan.nesting, f"{side}_type"):
                    meta.attrs[f"{side}_group_by"] = group

        # Provenance of the writing session's loading choices — recorded, and
        # deliberately not replayed on read.  See the module docstring.
        #
        # Tested by what the object *is*, not by which export layout it reads:
        # the correction ladder below belongs to every sweep of spectra, so a
        # second spectral instrument must reach it rather than fall through to
        # the temporal branch.
        from .loaders import _SpectralSweep   # local: avoids an import cycle
        if isinstance(scan, _SpectralSweep):
            # Not universal among spectral sweeps: a region-of-interest
            # selection exists only where the instrument writes more than one.
            roi = getattr(scan, "roi", None)
            if roi is not None:
                meta.attrs["roi"] = int(roi)
            meta.attrs["apply_jacobian"] = bool(scan.apply_jacobian)
            # The declaration, not the repaired array: `spectra` is written as the
            # source file had it, so a reader that wants the repair asks for it.
            if scan.cosmic_rays is not None:
                meta.attrs["cosmic_rays"] = json.dumps(scan.cosmic_rays)
            if scan.bg_region_nm is not None:
                meta.attrs["bg_region_nm"] = np.asarray(scan.bg_region_nm,
                                                        dtype=float)
        else:
            if scan.bg_region_ns is not None:
                meta.attrs["bg_region_ns"] = np.asarray(scan.bg_region_ns,
                                                        dtype=float)
            # An assembled sweep came from many files; record which, in order, so
            # the archive says what it replaced.
            meta.attrs["source_files"] = [f.name for f in scan.files]

        if scan.geometry is not None:
            _write_geometry(meta, scan.geometry)

        # Auxiliary spectra are stored as arrays, not paths: a path goes stale and
        # the archive should stand alone.  Like the other corrections they are
        # recorded, not replayed — but keeping the values means a contrast can be
        # rebuilt from the .h5 alone.
        aux_spectra = {name: getattr(scan, name, None)
                       for name in ("bg_spectrum", "reference")}
        if any(values is not None for values in aux_spectra.values()):
            aux = hf.create_group("auxiliary")
            for name, values in aux_spectra.items():
                if values is None:
                    continue
                dset = aux.create_dataset(name, data=np.asarray(values, float))
                dset.attrs["units"] = "counts"

            if aux_spectra["reference"] is not None:
                # Both scalars qualify this one array, so they hang off it rather
                # than off /metadata: an unmatched exposure biases a contrast, and
                # the factor that accounts for it is worthless separated from the
                # values it was applied to.  scan.reference already carries the
                # scaling (loaders applies it at construction), so the name says
                # applied — a reader must not multiply by it again.
                ref = aux["reference"]
                ref.attrs["contrast_mode"] = scan.contrast_mode
                if scan.reference_scale is not None:
                    ref.attrs["scale_applied"] = float(scan.reference_scale)

        # Axis dataset named for the physical quantity it holds, so `h5ls` says
        # what the file is; metadata/axis_kind records it authoritatively too.
        axes = hf.create_group("axes")
        axis = axes.create_dataset(
            kind["name"], data=np.asarray(getattr(scan, scan._AXIS_ATTR), float))
        axis.attrs["units"] = kind["units"]
        axis.attrs["label"] = kind["label"]

        params = hf.create_group("parameters")
        for label, values in scan.parameters.items():
            dset = params.create_dataset(_hdf5_key(label),
                                         data=np.asarray(values, dtype=float))
            dset.attrs["label"] = label      # authoritative; the name is sanitised
            dset.attrs["units"] = "raw"

        signals = hf.create_group(kind["group"])
        for name, arr in _signal_arrays(scan):
            dset = signals.create_dataset(
                name, data=np.asarray(arr, dtype=float),
                compression=compression,
            )
            dset.attrs["units"] = "counts"
            dset.attrs["axes"]  = f"{kind['name']}, sweep"

    return path


def _signal_arrays(scan) -> list:
    """
    The signal datasets to store, as ``(name, array)`` pairs.

    Read off the class's own ``_HDF5_SIGNALS`` rather than inferred from its
    export layout.  Two instruments can share a wavelength axis and still
    write a different number of signal arrays, so a layout test would quietly
    store the wrong datasets for the second one.
    """
    if not scan._HDF5_SIGNALS:
        raise NotImplementedError(
            f"{type(scan).__name__} declares no _HDF5_SIGNALS, so there is "
            f"no statement of which arrays an archive of it must carry. Set "
            f"the class attribute to a ((dataset name, attribute), ...) tuple."
        )
    return [(name, getattr(scan, attr)) for name, attr in scan._HDF5_SIGNALS]


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_sweep(path) -> dict:
    """
    Read an HDF5 file written by :func:`write_sweep`.

    Returns the same payload contract the CSV decoder produces, which is what
    lets one loader class serve both formats.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
        ``axis_kind`` : ``"wavelength"`` or ``"time"`` — which loader may read it.
        ``wavelength`` + ``roi1``, ``roi2`` for a spectral sweep, or
        ``time`` + ``counts`` for a temporal one, all float64.
        ``parameters`` : dict[str, (n_sweeps,) float64], keyed by the *original*
            instrument labels, in raw file units.
        ``metadata`` : dict of the recorded measurement metadata —
            ``spectra_type``, ``sweep``, ``sweep_label``, ``sweep_unit``, ``roi``,
            ``geometry`` (a rebuilt :class:`DeviceGeometry` or absent),
            ``curated_labels``, ``curated_scales``, ``gates`` (the declared
            channel-to-gate mapping, or ``None`` if the writer never declared
            one), plus the writing session's
            ``apply_jacobian``, ``bg_region_nm`` / ``bg_region_ns``,
            ``cosmic_rays``, ``bg_spectrum``, ``reference`` and ``source_files``
            as provenance.

    Raises
    ------
    ValueError
        If the file is not in this format, was written by a different major
        format version, or is missing a required group.
    """
    path = Path(path)
    with h5py.File(path, "r") as hf:
        fmt = _as_str(hf.attrs.get("format"))
        if fmt != FORMAT_NAME and fmt not in _LEGACY_FORMAT_NAMES:
            raise ValueError(
                f"'{path}' is not a {FORMAT_NAME} file (format={fmt!r}). "
                f"Reference datasets in reference/data/ use a different layout "
                f"and are read by leman.reference instead."
            )
        # Refuse a major-version mismatch rather than read around it: the groups
        # this reader looks in are not where an older writer put them, so a
        # tolerant read would return a scan missing its reference and say nothing.
        stored_version = _as_str(hf.attrs.get("format_version")) or ""
        if stored_version.split(".")[0] != _FORMAT_MAJOR:
            raise ValueError(
                f"'{path}' is format_version {stored_version!r}; this reader "
                f"requires {_FORMAT_MAJOR}.x. Version {_FORMAT_MAJOR}.0 moved the "
                f"auxiliary spectra (bg_spectrum, reference) from /metadata to "
                f"/auxiliary, so reading this file here would silently drop any it "
                f"records. Rewrite the archive from its source CSV with "
                f"scan.to_hdf5(path, overwrite=True)."
            )

        for group in ("metadata", "axes", "parameters"):
            if group not in hf:
                raise ValueError(
                    f"'{path}' is missing the required '{group}' group; the file "
                    f"may be truncated or written by an incompatible version "
                    f"(format_version={hf.attrs.get('format_version')!r})."
                )

        m = hf["metadata"].attrs
        metadata = {
            "spectra_type"   : _as_str(m.get("spectra_type")),
            # Named "sweep" here to match the loader's constructor argument.
            "sweep"          : _as_str(m.get("sweep_type")),
            "sweep_label"    : _as_str(m.get("sweep_label")),
            "sweep_unit"     : _as_str(m.get("sweep_unit")),
            # Already named for the constructor arguments they replay.
            "fast_sweep"     : _as_str(m.get("fast_sweep")),
            "slow_sweep"     : _as_str(m.get("slow_sweep")),
            "n_fast"         : int(m["n_fast"]) if "n_fast" in m else None,
            "n_slow"         : int(m["n_slow"]) if "n_slow" in m else None,
            "fast_group_by"  : _as_str(m.get("fast_group_by")),
            "slow_group_by"  : _as_str(m.get("slow_group_by")),
            "roi"            : int(m["roi"]) if "roi" in m else None,
            "source_file"    : _as_str(m.get("source_file")),
            "apply_jacobian" : bool(m["apply_jacobian"]) if "apply_jacobian" in m
                               else None,
            "bg_region_nm"   : tuple(np.asarray(m["bg_region_nm"], dtype=float))
                               if "bg_region_nm" in m else None,
            "bg_region_ns"   : tuple(np.asarray(m["bg_region_ns"], dtype=float))
                               if "bg_region_ns" in m else None,
            "source_files"   : ([_as_str(v) for v in m["source_files"]]
                                if "source_files" in m else None),
        }
        for key in _JSON_ATTRS:
            metadata[key] = json.loads(_as_str(m[key])) if key in m else None

        # The auxiliary spectra come back as arrays so a contrast can be rebuilt
        # from the archive alone.  Like every other correction they are provenance
        # here: the loader does not re-apply them.  They are reported inside
        # `metadata` because that is what the loader consumes as source_metadata —
        # the file groups them by what they are, the payload by what reads them.
        if "auxiliary" in hf:
            aux = hf["auxiliary"]
            for name in ("bg_spectrum", "reference"):
                if name in aux:
                    metadata[name] = aux[name][()]
            if "reference" in aux:
                ref = aux["reference"].attrs
                metadata["contrast_mode"] = _as_str(ref.get("contrast_mode"))
                # Already folded into the stored values; reported so the archive
                # says what was done, not so a reader can apply it.
                if "scale_applied" in ref:
                    metadata["reference_scale"] = float(ref["scale_applied"])

        # Drop keys the file did not record so the loader's "argument > file >
        # default" resolution sees an absence, not an explicit None.
        metadata = {k: v for k, v in metadata.items() if v is not None}

        if "geometry" in hf["metadata"]:
            metadata["geometry"] = _read_geometry(hf["metadata/geometry"])

        parameters = {
            _as_str(dset.attrs.get("label", name)): dset[()]
            for name, dset in hf["parameters"].items()
        }

        payload = {"parameters": parameters, "metadata": metadata}

        # Which axis the file holds decides which loader can read it, exactly as
        # the header does for a CSV.  Named datasets, so the check is a lookup.
        stored = [name for name in _CLASS_FOR_AXIS_KIND if name in hf["axes"]]
        if len(stored) != 1:
            raise ValueError(
                f"'{path}' has {len(stored)} recognised axis dataset(s) under "
                f"/axes ({stored}); expected exactly one of "
                f"{list(_CLASS_FOR_AXIS_KIND)}."
            )
        axis_kind = stored[0]

        if axis_kind == "wavelength":
            if "spectra" not in hf:
                raise ValueError(f"'{path}' has a wavelength axis but no /spectra.")
            payload["wavelength"] = hf["axes/wavelength"][()]
            payload["roi1"]       = hf["spectra/roi1"][()]
            payload["roi2"]       = hf["spectra/roi2"][()]
        else:
            if "decays" not in hf:
                raise ValueError(f"'{path}' has a time axis but no /decays.")
            payload["time"]   = hf["axes/time"][()]
            payload["counts"] = hf["decays/counts"][()]

        payload["axis_kind"] = axis_kind
        return payload


def _as_str(value):
    """Decode an HDF5 string attribute, which may come back as bytes."""
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)
