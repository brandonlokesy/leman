# src/leman/reference/processor.py

import io
import requests
import zipfile
import ftplib, ssl
import h5py
from pathlib import Path
from abc import ABC, abstractmethod

HEADERS = {"User-Agent": "LANES-Tools/1.0"}

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
        hf.attrs["material"]    = self.meta["material"]
        hf.attrs["source"]      = self.meta["source"]
        hf.attrs["title"]       = self.meta["title"]
        hf.attrs["doi"]         = self.meta["doi"]
        hf.attrs["dataset_doi"] = self.meta["dataset_doi"]
        hf.attrs["about"]       = self.meta.get("about", "")
        hf.attrs["spectroscopy"] = self.meta.get("spectroscopy", "PL")

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