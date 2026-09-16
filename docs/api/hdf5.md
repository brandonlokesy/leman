# HDF5

Self-describing HDF5 storage for AttoCube sweeps.

A raw CSV export records the numbers but not what they mean. These functions
write the measurement metadata — spectroscopy type, what was swept, which
acquisition channel drove which gate, the device stack — alongside the data, so
a scan re-read later comes back as the same object.

One format serves both measured axes: `axes/wavelength` for a spectral sweep,
`axes/time` for TRPL. A TRPL sweep also arrives as a *directory* of per-point
files, so writing it collapses them into a single archive — the committed
11.57 MB example becomes 0.069 MB.

Usually reached through `to_hdf5` on a sweep, and by passing an `.h5` path to the
loader, rather than called directly.

!!! note "Corrections are recorded, not replayed"
    The Jacobian flag, background windows and the `bg_spectrum` / `reference`
    spectra are all stored as **provenance** and are not re-applied on read —
    loading is not deciding. The auxiliary spectra are stored as arrays rather
    than paths, so a contrast can still be rebuilt from the archive alone:

    ```python
    back = ACSpectralSweep("scan.h5", spectra_type="R")
    again = ACSpectralSweep(
        "scan.h5", spectra_type="R",
        reference=back.source_metadata["reference"],
    )
    ```

::: leman.hdf5
