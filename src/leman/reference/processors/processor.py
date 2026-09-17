# src/leman/reference/processor.py

import datetime
import io
import requests
import zipfile
import ftplib, ssl
import numpy as np
import h5py
from pathlib import Path
from abc import ABC, abstractmethod

HEADERS = {"User-Agent": "LANES-Tools/1.0"}

REF_FORMAT_NAME = "leman.reference"
REF_FORMAT_VERSION = "1.0"

# Keys that must be present in every registry entry
_REQUIRED_META_KEYS = ("material", "source", "title", "doi", "dataset_doi")


class Processor(ABC):
    """
    Base class for all reference dataset processors.

    Subclasses must implement :meth:`run`, which downloads the raw data,
    processes it, and writes the HDF5 file.  All shared machinery
    (_fetch_zip, _write_metadata, out_path construction) lives here so
    individual processors stay focused on their own parsing logic.

    Parameters
    ----------
    meta : dict
        Registry entry for this dataset. Must contain:
        ``material``, ``source``, ``title``, ``doi``, ``dataset_doi``.
        Optional: ``about`` (default ``""``), ``spectroscopy`` (default ``"PL"``).
    out_dir : Path
        Directory where the ``.h5`` file will be written.
    """

    def __init__(self, meta: dict, out_dir: Path):
        missing = [k for k in _REQUIRED_META_KEYS if k not in meta]
        if missing:
            raise ValueError(f"Registry entry is missing required keys: {missing}")

        self.meta     = meta
        self.out_path = out_dir / f"{meta['material']}__{meta['source']}.h5"

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _fetch_zip(self, url: str) -> zipfile.ZipFile:
        """Download a ZIP archive from *url* and return it as a ZipFile object."""
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return zipfile.ZipFile(io.BytesIO(r.content))
    
    def _fetch_file(self, url: str, verify_ssl: bool = True) -> bytes:
        """
        Download a single file from a URL and return the raw bytes.

        Parameters
        ----------
        url        : direct download URL
        verify_ssl : set False for servers with self-signed certificates
                    (e.g. dataserv.ub.tum.de)
        """
        r = requests.get(url, headers=HEADERS, verify=verify_ssl)
        r.raise_for_status()
        return r.content
    
    def _fetch_ftp_file(
        self,
        remote_path : str,
        host        : str,
        user        : str,
        password    : str = None,
    ) -> bytes:
        """
        Download a single file from an FTP server into memory and return
        the raw bytes.

        Uses FTPS (FTP over TLS, RFC 4217) with an unverified SSL context
        to handle self-signed certificates as used by dataserv.ub.tum.de.

        Parameters
        ----------
        remote_path : str
            Path to the file on the FTP server.
        host : str
            FTP hostname, e.g. ``"dataserv.ub.tum.de"``.
        user : str
            FTP username.
        password : str, optional
            FTP password. Defaults to *user* when ``None`` — the mediaTUM
            convention where the dataset ID serves as both.
        """
        if password is None:
            password = user

        # mediaTUM uses a self-signed certificate, so we disable verification.
        # This is acceptable here because we are only downloading public
        # read-only research data, not transmitting sensitive information.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode   = ssl.CERT_NONE

        buf = io.BytesIO()
        with ftplib.FTP_TLS(host=host, context=ctx) as ftp:
            ftp.login(user=user, passwd=password)
            ftp.prot_p()    # encrypt the data channel as well as the control channel
            ftp.retrbinary(f"RETR {remote_path}", buf.write)
        return buf.getvalue()

    def _write_metadata(self, hf: h5py.File) -> None:
        """
        Write the standard file-level metadata attributes to an open HDF5 file.

        Call this once at the top of every processor's :meth:`run` method,
        immediately after opening the file, before creating any groups.
        """
        from ... import __version__

        hf.attrs["format"]          = REF_FORMAT_NAME
        hf.attrs["format_version"]  = REF_FORMAT_VERSION
        hf.attrs["created"]         = datetime.datetime.now().astimezone().isoformat()
        hf.attrs["toolkit_version"] = __version__
        hf.attrs["material"]        = self.meta["material"]
        hf.attrs["source"]          = self.meta["source"]
        hf.attrs["title"]           = self.meta["title"]
        hf.attrs["doi"]             = self.meta["doi"]
        hf.attrs["dataset_doi"]     = self.meta["dataset_doi"]
        hf.attrs["about"]           = self.meta.get("about", "")
        hf.attrs["spectroscopy"]    = self.meta.get("spectroscopy", "PL")

    # ------------------------------------------------------------------
    # Schema helpers — write the ``leman.reference`` v1.0 layouts
    # ------------------------------------------------------------------

    @staticmethod
    def _write_provenance(
        hf: h5py.File, summary: str, **extra
    ) -> None:
        """
        Write an optional provenance group with free-form attributes.

        Parameters
        ----------
        hf      : open HDF5 file
        summary : one-line description of the processing pipeline
        **extra : processor-specific attributes (e.g. ``smooth_kernel=3``)
        """
        grp = hf.create_group("provenance")
        grp.attrs["pipeline_summary"] = summary
        for key, value in extra.items():
            grp.attrs[key] = value

    @staticmethod
    def _write_single_spectrum(
        hf: h5py.File,
        axis_values: np.ndarray,
        axis_name: str,
        axis_units: str,
        axis_label: str,
        intensity: np.ndarray,
        intensity_units: str,
    ) -> None:
        """
        Write layout A (single spectrum, no sweep parameter).

        Parameters
        ----------
        hf              : open HDF5 file (metadata already written)
        axis_values     : 1-D spectral axis (energy or wavelength)
        axis_name       : ``"energy"`` or ``"wavelength"``
        axis_units      : ``"eV"`` or ``"nm"``
        axis_label      : display label, e.g. ``"Energy"``
        intensity       : 1-D signal array, same length as *axis_values*
        intensity_units : ``"counts"``, ``"arb. u."``, ``"dimensionless"``, etc.
        """
        axis_values = np.asarray(axis_values, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)
        if axis_values.shape != intensity.shape:
            raise ValueError(
                f"Axis and intensity shapes differ: "
                f"{axis_values.shape} vs {intensity.shape}"
            )

        axes = hf.create_group("axes")
        ds = axes.create_dataset(axis_name, data=axis_values)
        ds.attrs["units"] = axis_units
        ds.attrs["label"] = axis_label

        spectra = hf.create_group("spectra")
        ds = spectra.create_dataset("intensity", data=intensity)
        ds.attrs["units"] = intensity_units

    @staticmethod
    def _write_1d_sweep(
        hf: h5py.File,
        axis_values: np.ndarray,
        axis_name: str,
        axis_units: str,
        axis_label: str,
        sweep_values: np.ndarray,
        sweep_name: str,
        sweep_units: str,
        sweep_label: str,
        default_index: int,
        intensity_2d: np.ndarray,
        intensity_units: str,
    ) -> None:
        """
        Write layout B (1-D sweep).

        Parameters
        ----------
        hf              : open HDF5 file (metadata already written)
        axis_values     : 1-D spectral axis, length *n_pixels*
        axis_name       : ``"energy"`` or ``"wavelength"``
        axis_units      : ``"eV"`` or ``"nm"``
        axis_label      : display label
        sweep_values    : 1-D sweep parameter, length *n_sweeps*
        sweep_name      : e.g. ``"gate_voltage"``
        sweep_units     : e.g. ``"V"``
        sweep_label     : display label, e.g. ``"Gate voltage"``
        default_index   : index into *sweep_values* for the default spectrum
        intensity_2d    : ``(n_pixels, n_sweeps)`` signal array
        intensity_units : signal units
        """
        axis_values = np.asarray(axis_values, dtype=np.float64)
        sweep_values = np.asarray(sweep_values, dtype=np.float64)
        intensity_2d = np.asarray(intensity_2d, dtype=np.float64)
        expected = (axis_values.shape[0], sweep_values.shape[0])
        if intensity_2d.shape != expected:
            raise ValueError(
                f"Intensity shape {intensity_2d.shape} does not match "
                f"(n_pixels, n_sweeps) = {expected}"
            )

        axes = hf.create_group("axes")
        ds = axes.create_dataset(axis_name, data=axis_values)
        ds.attrs["units"] = axis_units
        ds.attrs["label"] = axis_label

        sweep = hf.create_group("sweep")
        ds = sweep.create_dataset("values", data=sweep_values)
        ds.attrs["name"] = sweep_name
        ds.attrs["units"] = sweep_units
        ds.attrs["label"] = sweep_label
        sweep.create_dataset("default_index", data=np.int64(default_index))

        spectra = hf.create_group("spectra")
        ds = spectra.create_dataset("intensity", data=intensity_2d)
        ds.attrs["units"] = intensity_units
        ds.attrs["axes"] = f"{axis_name}, sweep"

    @staticmethod
    def _write_nested_sweep(
        hf: h5py.File,
        conditions: list[dict],
        sweep_meta: dict,
    ) -> None:
        """
        Write layout C (nested sweep with per-condition axes).

        Parameters
        ----------
        hf         : open HDF5 file (metadata already written)
        conditions : one dict per outer condition, each with keys:
            ``axis_values``, ``axis_name``, ``axis_units``, ``axis_label``,
            ``inner_values``, ``inner_name``, ``inner_units``, ``inner_label``,
            ``default_index``, ``intensity_2d``, ``intensity_units``,
            ``outer_value``, ``is_default``
        sweep_meta : dict with ``outer_name``, ``outer_units``,
            ``outer_label``, ``default_value``
        """
        meta_grp = hf.create_group("sweep_meta")
        meta_grp.attrs["outer_name"] = sweep_meta["outer_name"]
        meta_grp.attrs["outer_units"] = sweep_meta["outer_units"]
        meta_grp.attrs["outer_label"] = sweep_meta["outer_label"]
        meta_grp.attrs["default_value"] = float(sweep_meta["default_value"])

        cond_root = hf.create_group("conditions")
        for i, cond in enumerate(conditions):
            axis_values = np.asarray(cond["axis_values"], dtype=np.float64)
            inner_values = np.asarray(cond["inner_values"], dtype=np.float64)
            intensity_2d = np.asarray(cond["intensity_2d"], dtype=np.float64)

            expected = (axis_values.shape[0], inner_values.shape[0])
            if intensity_2d.shape != expected:
                raise ValueError(
                    f"Condition {i}: intensity shape {intensity_2d.shape} "
                    f"does not match (n_pixels, n_inner) = {expected}"
                )

            grp = cond_root.create_group(str(i))
            grp.attrs["outer_value"] = float(cond["outer_value"])
            grp.attrs["is_default"] = bool(cond["is_default"])

            axes = grp.create_group("axes")
            ds = axes.create_dataset(cond["axis_name"], data=axis_values)
            ds.attrs["units"] = cond["axis_units"]
            ds.attrs["label"] = cond["axis_label"]

            sweep = grp.create_group("sweep")
            ds = sweep.create_dataset("values", data=inner_values)
            ds.attrs["name"] = cond["inner_name"]
            ds.attrs["units"] = cond["inner_units"]
            ds.attrs["label"] = cond["inner_label"]
            sweep.create_dataset(
                "default_index", data=np.int64(cond["default_index"])
            )

            spectra = grp.create_group("spectra")
            ds = spectra.create_dataset("intensity", data=intensity_2d)
            ds.attrs["units"] = cond["intensity_units"]

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> None:
        """
        Download, process, and write the dataset to :attr:`out_path`.

        Subclasses must override this.  The typical structure is::

            def run(self):
                z = self._fetch_zip("https://...")
                with h5py.File(self.out_path, "w") as hf:
                    self._write_metadata(hf)
                    # ... dataset-specific parsing ...
        """