# leman/converters.py
"""
Convert AttoCube CSV exports to compact formats.

Three export shapes, three destinations: a real-space frame becomes TIFF, a
spectral sweep becomes HDF5, and a directory of TRPL decays becomes one HDF5
archive.  Nothing here interprets a measurement — no correction is applied and no
axis is derived; this module rewrites a file in a better format and leaves the
deciding to the loaders.

The AttoCube software writes a real-space frame as a bare ``H x W`` grid of
comma-separated numbers with no header — a text encoding of what is a binary
raster.  A gate or position sweep arrives as a directory of such frames, one per
acquisition point, indexed by an ``_iter_N`` filename suffix.  Text costs roughly
six bytes per count, so a sweep is an order of magnitude larger than the same
data as TIFF, and no image viewer will open it.

Converting produces either one ``.tif`` per frame, preserving the input
one-to-one, or a single multi-page ``.tif`` per directory.  A stack is the more
convenient object to hand to ImageJ, but combining frames asserts that they
belong together, so it is asked for rather than assumed.

Where output lands
------------------
Into a ``converted/`` folder, created if it does not exist.  ``converted/`` rather
than ``processed/`` because a conversion is not an analysis: the output holds the
same numbers the export held, so it can be deleted and regenerated, whereas what
an analysis extracts from the raw data cannot.  Both names are fixed and take no
parameter.

The single-file converters read the source's **immediate parent**: a file in a
folder called ``raw`` writes to that folder's sibling ``converted/``, and anything
else writes to a ``converted/`` inside its own folder.  *out* overrides, naming
either a directory or a file.

:func:`convert_path` resolves a **root** once for the whole run and gives every
output its source's position within it, so two folders holding an identically
named frame cannot overwrite one another:

| argument | root |
|---|---|
| ``beside=True`` | the source's own folder — no ``converted/`` level |
| ``out=`` | that path, with the tree mirrored beneath it |
| ``from_raw=True`` | nearest **ancestor** ``raw``'s sibling ``converted/`` |
| none, and the named folder is ``raw`` | that folder's sibling ``converted/`` |
| none, otherwise | no root; the per-file rule above applies |

The default reads only the folder the call names, never one it searches for, so a
destination is readable off the call; ``from_raw=`` is the opt-in that looks
upward.  The three are mutually exclusive.

Public functions
----------------
convert_image_csv_to_tiff
    Single image CSV -> ``.tif``.
convert_image_dir_to_tiff_stack
    A directory of image CSVs -> one multi-page ``.tif``, in acquisition order.
convert_spectral_csv_to_hdf5
    Single spectral export -> ``.h5``, through the loader and
    :func:`leman.hdf5.write_sweep`.
convert_trpl_dir_to_hdf5
    A directory of TRPL decays -> one ``.h5``.
convert_path
    Convert a file, directory, or tree in one call, continuing past failures.
main
    ``tmdc-convert`` command-line entry point.

Notes
-----
The HDF5 side loads with :class:`~leman.loaders.ACSpectralSweep`
or :class:`~leman.loaders.ACTRPLSweep` and writes with
:func:`leman.hdf5.write_sweep`, so there is one archive format in the
package and a converted sweep reopens by handing the ``.h5`` back to its loader.

Only *spectra_type* is declared at conversion time, because a raw spectral export
records no measurement type and none can be inferred.  What was swept, which
channel reached which gate, and the device stack are read-time arguments: the
loader takes each from its argument when given and from the file otherwise, so an
archive is declared against exactly as the CSV was.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tifffile

from .constants import SPECTROSCOPY_TYPES
from .loaders import (
    ACSpectralSweep,
    ACTRPLSweep,
    _CSV_KIND_REASON,
    _classify_csv,
    _order_by_iter,
)

# The folder a source has to sit in for its output to become a sibling, and the
# folder output always lands in.  Fixed names rather than parameters: a setting
# would be a second place the answer lives, and the first time it disagreed with
# the folder on screen it would cost more than it saved.
#
# "converted", not "processed": nothing here extracts, fits or derives anything.
# The numbers in the output are the numbers that were in the CSV, in a container
# that is smaller and that a viewer will open.  A "processed" folder is for what
# an analysis pulls *out* of the raw data, which is a different thing and belongs
# beside this rather than mixed into it.
_RAW_DIR = "raw"
_OUT_DIR = "converted"

_DTYPES = ("auto", "uint16", "uint32", "float32")

# A directory of temporal files is one sweep, not a file kind, so it needs a
# reason of its own alongside the CSV kinds the loaders name.
_TRPL_DIR_KIND = "trpl_directory"

_SKIP_REASON = {
    **_CSV_KIND_REASON,
    _TRPL_DIR_KIND: "a TRPL sweep directory — name it on its own to convert it, "
                    "with prefix= if it holds more than one measurement",
}

# How a refusal names the kind it wanted.  Only the kinds this module converts
# need an entry; anything else is a programming error, not a user's.
_KIND_ARTICLE = {
    "image"   : "a real-space image CSV",
    "spectral": "a spectral export CSV",
}

# Largest value uint16 holds.  Integer counts above it need uint32, because
# float32 cannot represent every integer past 2**24.
_U16_MAX = 65535


class ConversionReport(NamedTuple):
    """
    What a :func:`convert_path` run did.

    A tuple, so it unpacks as ``outputs, skipped, errors``.

    Attributes
    ----------
    outputs : list of Path
        Files written, in the order they were written.
    skipped : dict
        ``{Path: kind}`` for every CSV that was not a real-space image, named as
        the loaders name it.  Reported rather than warned about, because a
        directory of spectra is the likeliest way to get an empty run and
        "nothing written" is not a diagnosis.
    errors : list of tuple
        ``(source, message)`` for each conversion that raised.  A batch continues
        past a failure, so this can be non-empty alongside outputs.
    """

    outputs : list
    skipped : dict
    errors  : list


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def _default_output(src, new_suffix: str, out=None) -> Path:
    """
    Resolve where converting *src* to *new_suffix* should write.

    Resolves only — nothing is created and nothing is checked for existence, so
    this is safe to call before deciding whether to convert at all.

    Parameters
    ----------
    src : str or Path
        Source file.
    new_suffix : str
        Suffix of the output, leading dot included, e.g. ``".tif"``.
    out : str or Path, optional
        Explicit destination.  A path carrying a suffix is used verbatim; one
        without (or an existing directory) is treated as a directory and the
        output is named after *src*.

    Returns
    -------
    pathlib.Path
        With *out* omitted: ``<parent>/converted/<stem><suffix>``, where
        ``<parent>`` is the grandparent when *src* sits in a ``raw/`` folder and
        the source's own folder otherwise.
    """
    src  = Path(src)
    name = src.stem + new_suffix

    if out is not None:
        out = Path(out)
        return out / name if (out.suffix == "" or out.is_dir()) else out

    # A raw/ folder already names its counterpart, so the output belongs beside
    # it rather than nested inside it.
    root = src.parent.parent if src.parent.name.lower() == _RAW_DIR else src.parent
    return root / _OUT_DIR / name


def _dir_output(directory, stem: str, new_suffix: str, out=None) -> Path:
    """
    Resolve where converting a whole *directory* to one file should write.

    The same rule as :func:`_default_output`, applied to a folder rather than a
    file, so a directory-wide output lands where the folder's own per-file output
    would have.  ``scan/raw/`` gives ``scan/converted/``; ``scan/frames/`` gives
    ``scan/frames/converted/``.

    Parameters
    ----------
    directory : str or Path
        The folder being converted.
    stem : str
        Filename stem for the single output, e.g. ``"raw_stack"``.
    new_suffix : str
        Suffix of the output, leading dot included.
    out : str or Path, optional
        Explicit destination, read as in :func:`_default_output`.

    Returns
    -------
    pathlib.Path
        Nothing is created; see :func:`_claim_target`.
    """
    directory = Path(directory)
    name      = stem + new_suffix

    if out is not None:
        out = Path(out)
        return out / name if (out.suffix == "" or out.is_dir()) else out

    root = directory.parent if directory.name.lower() == _RAW_DIR else directory
    return root / _OUT_DIR / name


def _refuse_existing(target: Path, overwrite: bool) -> Path:
    """
    Refuse *target* if it already exists, creating nothing.

    Overwriting is opt-in for the same reason it is in
    :func:`leman.hdf5.write_sweep`: the file being replaced may be the
    only copy.  This is the check on its own, for writers that create their own
    parent directory — calling it before a decode is what lets a re-run refuse in
    milliseconds instead of parsing tens of MB and then refusing.

    Raises
    ------
    FileExistsError
        If *target* exists and *overwrite* is False.
    """
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"'{target}' already exists. Pass overwrite=True (or --overwrite) to "
            f"replace it, or give a different out= path."
        )
    return target


def _raw_ancestor(start: Path, search_upward: bool):
    """
    The ``raw/`` folder that governs *start*, or ``None``.

    Parameters
    ----------
    start : Path
        The directory a run was pointed at.
    search_upward : bool
        ``False`` looks at *start* alone — the folder the caller actually named,
        so the answer is visible in the command they typed.  ``True`` walks up
        the path to the nearest ancestor called ``raw``, which finds the folder a
        caller pointed *inside* but cannot be predicted from the command alone.

    Returns
    -------
    Path or None
        Matching is case-insensitive, so ``RAW/`` and ``Raw/`` both count.
    """
    for candidate in (start, *start.parents) if search_upward else (start,):
        if candidate.name.lower() == _RAW_DIR:
            return candidate
    return None


def _claim_target(target: Path, overwrite: bool) -> Path:
    """
    Refuse *target* if it already exists, then create its parent directory.

    For writers that do not create their own directory — ``tifffile.imwrite`` does
    not, ``hdf5.write_sweep`` does.  The refusal runs first, so a refused
    conversion leaves no empty ``converted/`` folder behind.

    Raises
    ------
    FileExistsError
        If *target* exists and *overwrite* is False.
    """
    _refuse_existing(target, overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

def _folder_kinds(directory, prefix=None) -> dict:
    """
    Name every CSV in *directory*, as ``{Path: kind}``.

    One classification pass per folder, so routing a mixed folder — frames beside
    a spectral export, which is what an acquisition writes — reads the opening
    lines of each file once rather than once per output kind.

    Parameters
    ----------
    directory : str or Path
    prefix : str, optional
        Filename prefix to select.  Omitted, every ``*.csv`` is considered.

    Returns
    -------
    dict
        ``{Path: kind}`` in filename order, kinds as
        :func:`leman.loaders._classify_csv` names them.
    """
    pattern = f"{prefix}*.csv" if prefix else "*.csv"
    return {f: _classify_csv(f) for f in sorted(Path(directory).glob(pattern))}


def _image_frames(directory, prefix=None, *, stacklevel: int) -> tuple:
    """
    Every real-space image CSV in *directory*, in acquisition order.

    Selection is by content, not by name: a parameter export written beside the
    frames, and a two-row single spectrum, are both excluded and returned as
    *skipped* instead.  Ordering is by the ``_iter_N`` suffix, so ``iter_10``
    follows ``iter_2``; export padding widths vary and cannot be relied on.

    Parameters
    ----------
    directory : str or Path
    prefix : str, optional
        Filename prefix to select, e.g. ``"PL-dual-gate-sweep_iter_"``.  Omitted,
        every ``*.csv`` in the folder is considered.
    stacklevel : int
        Passed straight to the ordering helper, whose warnings about a missing or
        repeated ``_iter_N`` should point at the caller's own call.

    Returns
    -------
    (frames, skipped) : (list of Path, dict)
        Ordered image paths, and ``{Path: kind}`` for everything excluded.
    """
    kinds   = _folder_kinds(directory, prefix)
    images  = [f for f, kind in kinds.items() if kind == "image"]
    skipped = {f: kind for f, kind in kinds.items() if kind != "image"}
    if not images:
        return [], skipped
    return _order_by_iter(images, directory, stacklevel=stacklevel), skipped


def _describe(skipped: dict) -> str:
    """One indented ``name — why`` line per entry of a ``{Path: kind}`` map."""
    return "\n".join(
        f"  {f.name} — {_SKIP_REASON.get(kind, kind)}"
        for f, kind in skipped.items()
    )


def _require_kind(path: Path, kind: str) -> None:
    """
    Raise unless *path* is a CSV of the given kind.

    Parameters
    ----------
    path : Path
    kind : str
        The kind :func:`leman.loaders._classify_csv` must return, e.g.
        ``"image"`` or ``"spectral"``.

    Raises
    ------
    ValueError
        Naming what the file is instead, in the same words the loaders use, so a
        spectrum is never silently written out as a two-pixel-tall image and a
        frame is never handed to a sweep loader.
    """
    found = _classify_csv(path)
    if found != kind:
        raise ValueError(
            f"'{path}' is not {_KIND_ARTICLE[kind]}: "
            f"{_SKIP_REASON.get(found, found)}."
        )


# ---------------------------------------------------------------------------
# Pixel type
# ---------------------------------------------------------------------------

def _as_image_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    """
    Cast an image array for TIFF output.

    ``dtype="auto"`` keeps integer counts exactly — ``uint16`` up to 65535 and
    ``uint32`` above it — and falls back to ``float32`` for anything else.  That
    fallback is a **narrowing**: the CSV is read as ``float64``, so a non-integer
    or negative image loses precision.  Force ``"float32"`` knowingly, or keep the
    CSV.

    Parameters
    ----------
    arr : np.ndarray
    dtype : {"auto", "uint16", "uint32", "float32"}

    Returns
    -------
    np.ndarray
        A new array of the chosen type.

    Raises
    ------
    ValueError
        If *dtype* is not one of the four.
    """
    if dtype == "auto":
        finite = arr[np.isfinite(arr)]
        # An all-NaN frame has no integers to preserve, so it takes the float
        # branch rather than casting NaN to an unsigned type.
        is_count = bool(finite.size) and bool(
            np.all(finite >= 0) and np.all(finite == np.round(finite))
        )
        if is_count:
            return arr.astype(np.uint16 if finite.max() <= _U16_MAX else np.uint32)
        return arr.astype(np.float32)
    if dtype in _DTYPES[1:]:
        return arr.astype(dtype)
    raise ValueError(
        f"Unsupported image dtype '{dtype}'. Choose from: {list(_DTYPES)}."
    )


# ---------------------------------------------------------------------------
# Image CSV -> TIFF
# ---------------------------------------------------------------------------

def convert_image_csv_to_tiff(
    path,
    out       = None,
    dtype     : str  = "auto",
    overwrite : bool = False,
) -> Path:
    """
    Convert a single real-space image CSV to a TIFF file.

    Parameters
    ----------
    path : str or Path
        Source image CSV.  A file that is not a numeric grid of at least three
        rows is refused rather than converted.
    out : str or Path, optional
        Destination file or directory.  Omitted, the file lands in a
        ``converted/`` folder — see the module docstring.
    dtype : {"auto", "uint16", "uint32", "float32"}
        Pixel type.  ``"auto"`` keeps integer counts exactly.
    overwrite : bool
        Replace an existing output file.  Default False, which raises.

    Returns
    -------
    pathlib.Path
        The ``.tif`` written.

    Raises
    ------
    ValueError
        If *path* is not a real-space image CSV, or *dtype* is unknown.
    FileExistsError
        If the output exists and *overwrite* is False.
    """
    path = Path(path)
    _require_kind(path, "image")
    target = _claim_target(_default_output(path, ".tif", out), overwrite)
    tifffile.imwrite(target, _as_image_dtype(np.loadtxt(path, delimiter=","), dtype))
    return target


def convert_image_dir_to_tiff_stack(
    directory,
    prefix    = None,
    out       = None,
    dtype     : str  = "auto",
    overwrite : bool = False,
) -> Path:
    """
    Combine a directory of real-space image CSVs into one multi-page TIFF.

    Frames are ordered by their ``_iter_N`` suffix, so page *i* of the stack is
    acquisition *i*.  A gap in that sequence is warned about and never closed up,
    so a stack built from an incomplete export has fewer pages than iterations.
    One pixel type is chosen for the whole stack.

    Parameters
    ----------
    directory : str or Path
        Folder holding the frames.
    prefix : str, optional
        Filename prefix to select, e.g. ``"PL-dual-gate-sweep_iter_"``.  Omitted,
        every image CSV in the folder is used.
    out : str or Path, optional
        Destination file or directory.  Omitted, the stack lands in the same
        ``converted/`` folder its frames would have, named after *prefix* with
        trailing separators stripped, or after the folder.
    dtype : {"auto", "uint16", "uint32", "float32"}
        Pixel type for every page.
    overwrite : bool
        Replace an existing output file.  Default False, which raises.

    Returns
    -------
    pathlib.Path
        The multi-page ``.tif`` written.

    Raises
    ------
    ValueError
        If the folder holds no image CSV matching *prefix*.  The message names
        what was found instead.
    FileExistsError
        If the output exists and *overwrite* is False.
    """
    directory = Path(directory)
    # 4 frames out to the caller's line: the ordering helper, _image_frames, this
    # function, the call.  A run arriving through convert_path lands one frame
    # short, on convert_path's own line.
    frames, skipped = _image_frames(directory, prefix, stacklevel=4)
    if not frames:
        pattern = f"{prefix}*.csv" if prefix else "*.csv"
        if skipped:
            raise ValueError(
                f"No real-space image CSV matching '{pattern}' in '{directory}'. "
                f"Found {len(skipped)} other file(s):\n{_describe(skipped)}"
            )
        raise ValueError(f"No CSV files matching '{pattern}' in '{directory}'.")

    stem = (prefix.rstrip("_- ") if prefix else directory.name) + "_stack"
    # One rule for every directory-wide output, so a stack sits beside the
    # per-frame output rather than in a second place.
    target = _claim_target(_dir_output(directory, stem, ".tif", out), overwrite)
    # (n_frames, ny, nx) — every frame read once, cast as one block so the whole
    # stack shares a pixel type.
    stack = np.stack([np.loadtxt(f, delimiter=",") for f in frames])
    # photometric is not decoration: left to guess, tifffile reads a 3-frame
    # stack as one RGB image with separate colour planes, and a 3- or 4-point
    # sweep would open as a single colour frame instead of its pages.
    tifffile.imwrite(target, _as_image_dtype(stack, dtype), photometric="minisblack")
    return target


# ---------------------------------------------------------------------------
# Spectral and temporal CSV -> HDF5
# ---------------------------------------------------------------------------

def convert_spectral_csv_to_hdf5(
    path,
    spectra_type : str,
    out          = None,
    overwrite    : bool = False,
    compression  : str  = "gzip",
) -> Path:
    """
    Convert a spectral export CSV to a self-describing HDF5 archive.

    Loads the file with :class:`~leman.loaders.ACSpectralSweep`
    and writes it with :func:`leman.hdf5.write_sweep`, so the result is
    the package's one archive format and reopens by handing the ``.h5`` back to the
    loader.  Every signal array and parameter row is stored in file units.

    What was swept, which channel reached which gate, and the device stack are
    **not** declared here.  They are read-time arguments: the loader takes each
    from its argument when given and from the file's metadata otherwise, so an
    archive written by this function is declared against exactly as the CSV was.

    Parameters
    ----------
    path : str or Path
        Source spectral CSV.  Anything else is refused rather than converted.
    spectra_type : str
        Required, and the one measurement fact that cannot be recovered later: a
        raw export records no type and none can be inferred from the data.  One of
        the keys of :data:`leman.constants.SPECTROSCOPY_TYPES`.
    out : str or Path, optional
        Destination file or directory.  Omitted, the archive lands in a
        ``converted/`` folder — see the module docstring.
    overwrite : bool
        Replace an existing archive.  Default False, which raises.
    compression : str or None
        Passed to :func:`~leman.hdf5.write_sweep`, which applies it to
        the two spectra arrays.  ``None`` disables it.

    Returns
    -------
    pathlib.Path
        The ``.h5`` written.

    Raises
    ------
    ValueError
        If *path* is not a spectral export, or *spectra_type* is not recognised.
    FileExistsError
        If the archive exists and *overwrite* is False.

    Examples
    --------
    >>> h5 = convert_spectral_csv_to_hdf5("scan/raw/sweep.csv", "PL")
    >>> scan = ACSpectralSweep(h5, spectra_type="PL", sweep="V_A")
    """
    path = Path(path)
    _require_kind(path, "spectral")
    # Validated through the loader's own resolver, against the empty metadata a raw
    # export has, so a missing or mistyped type raises the message the loader would
    # have raised — before tens of MB are parsed, and with no second copy of it.
    ACSpectralSweep._resolve_spectra_type(spectra_type, {})
    # Checked before the decode too, so a re-run refuses in milliseconds.
    target = _refuse_existing(_default_output(path, ".h5", out), overwrite)
    scan   = ACSpectralSweep(path, spectra_type=spectra_type)
    return scan.to_hdf5(target, overwrite=overwrite, compression=compression)


def convert_trpl_dir_to_hdf5(
    directory,
    prefix      = None,
    out         = None,
    overwrite   : bool = False,
    compression : str  = "gzip",
) -> Path:
    """
    Convert a directory of TRPL exports to one self-describing HDF5 archive.

    A TRPL sweep is a *directory* — one TCSPC decay per file — so this collapses
    many files into a single archive.  Which files count is
    :class:`~leman.loaders.ACTRPLSweep`'s decision, not this
    function's: it excludes an IRF reference by name, reads a spectral-header file
    in the folder as the parameter-table companion rather than a sweep, and orders
    the rest by their ``_iter_N`` suffix.

    No *spectra_type* argument, because the loader defaults it to ``"TRPL"`` — the
    class already declares the modality.

    Parameters
    ----------
    directory : str or Path
        Folder holding the per-point exports.
    prefix : str, optional
        Filename prefix to select, e.g. ``"TRPL_"``.  **Required in practice when
        the folder holds more than one measurement**: without it every temporal
        file is taken as a point of one sweep, and two measurements sharing a
        folder merge.  The loader warns when they claim the same iteration
        indices, but a warning is not a refusal.
    out : str or Path, optional
        Destination file or directory.  Omitted, the archive lands in a
        ``converted/`` folder, named after *prefix* with trailing separators
        stripped, or after the folder.
    overwrite : bool
        Replace an existing archive.  Default False, which raises.
    compression : str or None
        Passed to :func:`~leman.hdf5.write_sweep`.

    Returns
    -------
    pathlib.Path
        The ``.h5`` written.

    Raises
    ------
    ValueError
        If the folder holds no TRPL data file matching *prefix*.  The loader's
        message counts what it looked at and why each file was excluded.
    FileExistsError
        If the archive exists and *overwrite* is False.
    """
    directory = Path(directory)
    stem      = prefix.rstrip("_- ") if prefix else directory.name
    target    = _refuse_existing(_dir_output(directory, stem, ".h5", out), overwrite)
    scan      = ACTRPLSweep(directory, prefix=prefix)
    return scan.to_hdf5(target, overwrite=overwrite, compression=compression)


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def convert_path(
    path,
    out          = None,
    recursive    : bool = False,
    stack        : bool = False,
    dtype        : str  = "auto",
    overwrite    : bool = False,
    spectra_type : str  = None,
    prefix       : str  = None,
    compression  : str  = "gzip",
    from_raw     : bool = False,
    beside       : bool = False,
) -> ConversionReport:
    """
    Convert a file, a directory, or a tree, continuing past failures.

    Every failure is collected rather than raised, so one unreadable frame does not
    abandon a sweep.  Call a single-purpose converter directly to have the
    exception instead.

    Routing is by content, per folder:

    ==========================  ==================================================
    what the folder holds       what is written
    ==========================  ==================================================
    image CSVs                  one ``.tif`` each, or one stack with *stack*
    temporal CSVs               one ``.h5`` for the whole folder — but only if this
                                is the directory that was named; see below
    spectral CSVs               one ``.h5`` each, unless the folder is a TRPL
                                sweep, in which case a spectral file is that
                                sweep's parameter-table companion
    anything else               reported in ``skipped``
    ==========================  ==================================================

    **A TRPL directory converts only when you name it.** A folder reached by
    *recursive* is reported in ``skipped`` instead, because which files form one
    sweep is a declaration rather than something to infer: two measurements in one
    folder merge into a single wrong archive, and the loader warns about the
    colliding iteration indices without refusing.  Name that folder on its own,
    with *prefix*, to convert it.

    Parameters
    ----------
    path : str or Path
        A single CSV, or a directory of them.
    out : str or Path, optional
        Output **root**.  The tree under *path* is mirrored beneath it, so
        ``<path>/spot01/01-PL/sweep.csv`` becomes
        ``<out>/spot01/01-PL/sweep.h5``.  A path carrying a suffix is taken as one
        filename and used verbatim — meaningful with *stack*, and a
        ``FileExistsError`` on the second output otherwise.  Cannot be combined
        with *from_raw*.

        Omitted, the root is derived from the folder *named in the call*: if that
        folder is called ``raw``, the root is its sibling ``converted/`` and the
        tree is mirrored there; otherwise each source folder gets its own
        ``converted/``.  So pointing at ``EXP/raw`` places the whole experiment
        under ``EXP/converted`` with no argument at all, while pointing at one
        measurement folder inside ``raw/`` does not — nothing is searched for, so
        the destination is readable off the call.
    recursive : bool
        Descend into subdirectories.  Ignored when *path* is a file.
    stack : bool
        Write one multi-page TIFF per folder instead of one file per frame.
    dtype : {"auto", "uint16", "uint32", "float32"}
        Pixel type for image output.
    overwrite : bool
        Replace existing output files.
    spectra_type : str, optional
        Measurement type for spectral exports, which cannot record their own.
        Without it a spectral CSV becomes an entry in ``errors`` carrying the
        loader's message, and the images in the same folder still convert.
    prefix : str, optional
        Filename prefix to select, applied in every folder.  Narrows which files
        form a TRPL sweep, and names a TIFF stack.
    compression : str or None
        Passed to :func:`~leman.hdf5.write_sweep` for HDF5 output.
    from_raw : bool
        Look **upward** from *path* for the nearest folder called ``raw``, and
        mirror the tree from there into its sibling ``converted/``.  This places
        output correctly even when the call points inside ``raw/`` — at one
        measurement folder, or at a single file — which the default deliberately
        does not.

        Off by default because it reads folders the call does not name: a ``raw``
        high up an unrelated path anchors everything below it, and the result
        cannot be predicted from the call.  Turn it on having checked the path.
        Warns and falls back to the default when no ``raw`` is found.
    beside : bool
        Write each output into **the folder its source came from**, with no
        ``converted/`` level.  For targeted conversions — one file, or a folder
        whose outputs belong next to their inputs, as the committed
        ``examples/data`` archives do.

        Shorthand only: identical to passing ``out=`` naming that folder, and
        preferred over it for a one-off because a boolean cannot be mistyped into
        a valid-but-wrong path.  It will write inside ``raw/`` if that is where
        the source is, and does not warn — an explicit argument is the caller
        deciding.

    Returns
    -------
    ConversionReport
        ``(outputs, skipped, errors)``.

    Raises
    ------
    FileNotFoundError
        If *path* is neither a file nor a directory.
    ValueError
        If more than one of *out*, *from_raw* and *beside* is given.

    Warns
    -----
    UserWarning
        If *from_raw* is asked for and no ``raw`` folder is found.
    """
    path = Path(path)
    outputs, skipped, errors = [], {}, []

    given = [name for name, on in (("out=", out is not None),
                                   ("from_raw=", from_raw),
                                   ("beside=", beside)) if on]
    if len(given) > 1:
        raise ValueError(
            f"Give at most one of out=, from_raw= and beside=; got "
            f"{', '.join(given)}. All three answer the same question — where "
            f"output goes."
        )

    # The two halves of a destination: a root to write under, and the folder that
    # positions are measured from.  Resolved once, because a per-folder answer is
    # what let a tree flatten into one place before.
    base = path.parent if path.is_file() else path
    if beside:
        # Root and anchor the same folder, so every relative path is "." and each
        # output lands in the folder its source came from.  No new placement
        # concept — the shorthand for out= naming that folder.
        root, anchor = base, base
    elif out is not None:
        root, anchor = Path(out), base
    else:
        raw = _raw_ancestor(base, from_raw)
        if raw is not None:
            root, anchor = raw.parent / _OUT_DIR, raw
        else:
            root, anchor = None, base
            if from_raw:
                warnings.warn(
                    f"from_raw=True, but no folder called '{_RAW_DIR}' was found "
                    f"at or above '{base}'. Each source folder gets its own "
                    f"'{_OUT_DIR}/' instead. Pass out= to place the output tree.",
                    UserWarning, stacklevel=2,
                )

    def destination(folder: Path):
        """
        Where *folder*'s output goes: *root* with the folder's own position under
        *anchor* appended.  ``None`` leaves each converter on the ``converted/``
        rule.

        A folder sitting at *anchor* has a relative path of ``.``, which
        ``joinpath`` drops, so a run with nothing below it addresses *root*
        directly and the mirror only shows up where there is a tree to mirror.
        """
        if root is None:
            return None
        if root.suffix and not root.is_dir():
            return root                       # an explicit filename, not a root
        return root / folder.relative_to(anchor)

    def attempt(convert, source):
        """Run one conversion, recording its output or its failure."""
        try:
            outputs.append(convert())
        except Exception as exc:              # a batch reports and carries on
            errors.append((source, str(exc)))

    if path.is_file():
        into = destination(base)
        kind = _classify_csv(path)
        if kind == "spectral":
            attempt(lambda: convert_spectral_csv_to_hdf5(
                path, spectra_type, out=into, overwrite=overwrite,
                compression=compression), path)
        else:
            attempt(lambda: convert_image_csv_to_tiff(
                path, out=into, dtype=dtype, overwrite=overwrite), path)
        return ConversionReport(outputs, skipped, errors)

    if not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: '{path}'")

    # Every folder holding a CSV, deduplicated: output is resolved per folder, and
    # a stack and a TRPL sweep are both per folder.
    csvs = path.rglob("*.csv") if recursive else path.glob("*.csv")

    for folder in sorted({csv.parent for csv in csvs}):
        kinds = _folder_kinds(folder, prefix)
        # Never rebind `out`: destination() reads it, so a rebind would mirror the
        # second folder against the first folder's output.
        into  = destination(folder)
        # A temporal file makes the whole folder one sweep, which is what decides
        # whether its spectral files are sweeps or that sweep's companion.
        is_trpl = any(kind == "temporal" for kind in kinds.values())
        named   = folder == path

        # image, spectral and temporal are each either converted below or absorbed
        # into a sweep the loader reads for itself.  Everything else is reported.
        for f, kind in kinds.items():
            if kind not in ("image", "spectral", "temporal"):
                skipped[f] = kind

        if is_trpl:
            if named:
                attempt(lambda: convert_trpl_dir_to_hdf5(
                    folder, prefix=prefix, out=into, overwrite=overwrite,
                    compression=compression), folder)
            else:
                skipped[folder] = _TRPL_DIR_KIND
        else:
            for f, kind in kinds.items():
                if kind == "spectral":
                    attempt(lambda f=f: convert_spectral_csv_to_hdf5(
                        f, spectra_type, out=into, overwrite=overwrite,
                        compression=compression), f)

        images = [f for f, kind in kinds.items() if kind == "image"]
        if not images:
            continue
        if stack:
            attempt(lambda: convert_image_dir_to_tiff_stack(
                folder, prefix=prefix, out=into, dtype=dtype,
                overwrite=overwrite), folder)
        else:
            # Filename order, not acquisition order: each frame becomes its own
            # file named after itself, so nothing here depends on the sequence.
            # Ordering would only add _order_by_iter's warning about frames that
            # carry no _iter_N — which two standalone reference images beside a
            # sweep legitimately do not.
            for frame in images:
                attempt(lambda frame=frame: convert_image_csv_to_tiff(
                    frame, out=into, dtype=dtype, overwrite=overwrite), frame)

    return ConversionReport(outputs, skipped, errors)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """
    Entry point for the ``tmdc-convert`` command.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse.  ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit status: 1 if any conversion failed, 0 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="tmdc-convert",
        description=(
            "Convert AttoCube CSV exports to compact formats: real-space images to "
            "TIFF, spectral and TRPL sweeps to HDF5. Output lands in a converted/ "
            "folder: the sibling of raw/ when the source is in one, and a folder "
            "beside the source otherwise."
        ),
    )
    parser.add_argument("path", help="A CSV, or a directory of them.")
    parser.add_argument(
        "--out", default=None,
        help="Output root. The tree under PATH is mirrored beneath it, so "
             "raw/spot01/01-PL/ lands in OUT/spot01/01-PL/. Default: point at a "
             "raw/ folder and its sibling converted/ is used; point anywhere else "
             "and each source folder gets its own converted/.",
    )
    parser.add_argument(
        "--beside", action="store_true",
        help="Write each output into the folder its source came from, with no "
             "converted/ level. For targeted conversions. Cannot be combined "
             "with --out or --from-raw.",
    )
    parser.add_argument(
        "--from-raw", action="store_true",
        help="Search upward for the nearest raw/ folder and mirror from there, "
             "so output is placed correctly even when PATH is inside raw/. Reads "
             "folders you did not name — check your path first. Cannot be "
             "combined with --out or --beside.",
    )
    parser.add_argument(
        "--spectra-type", default=None, choices=sorted(SPECTROSCOPY_TYPES),
        help="Measurement type for spectral exports, which record none of their "
             "own. Required to convert one; images and TRPL need it not.",
    )
    parser.add_argument(
        "--prefix", default=None,
        help="Filename prefix to select. Narrows which files form a TRPL sweep, "
             "and names a TIFF stack.",
    )
    parser.add_argument(
        "--compression", default="gzip",
        help="HDF5 compression filter, or 'none' (default: gzip).",
    )
    parser.add_argument(
        "--stack", action="store_true",
        help="Write one multi-page TIFF per folder, in _iter_N order, instead of "
             "one file per frame.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Descend into subdirectories when PATH is a directory.",
    )
    parser.add_argument(
        "--dtype", default="auto", choices=_DTYPES,
        help="Pixel type. Default 'auto': integer counts kept exactly, "
             "anything else narrowed to float32.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing output files. Default: refuse.",
    )
    args = parser.parse_args(argv)

    try:
        report = convert_path(
            args.path,
            out          = args.out,
            recursive    = args.recursive,
            stack        = args.stack,
            dtype        = args.dtype,
            overwrite    = args.overwrite,
            spectra_type = args.spectra_type,
            prefix       = args.prefix,
            from_raw     = args.from_raw,
            beside       = args.beside,
            # argparse cannot express "a filter name or nothing", so the word is
            # spelled on the command line and turned into None here.
            compression  = None if args.compression.lower() == "none"
                           else args.compression,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1

    for target in report.outputs:
        print(f"wrote {target}")
    # A deferred TRPL folder is the one skip that asks the caller to do something,
    # so it is named rather than left inside a count.
    for folder, kind in report.skipped.items():
        if kind == _TRPL_DIR_KIND:
            print(f"deferred {folder}: {_SKIP_REASON[_TRPL_DIR_KIND]}")
    for src, message in report.errors:
        print(f"ERROR {src}: {message}", file=sys.stderr)

    summary = f"\n{len(report.outputs)} file(s) written"
    if report.skipped:
        summary += f", {len(report.skipped)} not converted"
    summary += f", {len(report.errors)} error(s)."
    print(summary)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
