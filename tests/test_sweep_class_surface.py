"""What each sweep class must still expose, and where its warnings point.

Both tests here guard failures that leave the suite and the docs build green.

**The surface.** Moving a member between the sweep classes and their private
bases is invisible to every other test: the behaviour is identical when it lands
in the right place, and when it lands in the wrong one the member simply is not
there any more. `mkdocs build --strict` does not catch it either -- nothing is
*wrong* on the rendered page, there is just less of it. That is a real
regression this repository has had before (C7 in `dev/defects.md`, where a page
went from 32 members to 6). The lists below are therefore checked as a whole,
against `dir()`, so a member that quietly stops being reachable fails here.

They are **minimum** sets, not exact ones: adding a member is normal and must
not fail, so the assertion is one-directional.

**The warning depth.** `stacklevel` is measured, not read off the `def` lines,
because the frame count is what decides where a warning is blamed -- and a wrong
value also keys the wrong module's `__warningregistry__`, so the default filter
dedups against an entry nobody is looking at. See
`tests/test_loaders_nesting.py` for the same reasoning about `_bind_nesting`.
"""

import warnings
from pathlib import Path

import pytest

from leman import hdf5
from leman.loaders import ACSpectralSweep, ACTRPLSweep

from _paths import DATA

SPECTRAL = str(DATA / "stark-shift"
                    / "PL-dual-gate-sweep_26_05_15_14_03_18_iter_0.csv")

# Everything a caller, a plotting function or a fitting function reaches on a
# spectral sweep. Split by where it comes from, so a failure says which class
# lost it rather than only that something went missing.
_FROM_THE_SWEEP_BASE = {
    "as_grid", "as_image_grid", "carrier_density", "curated_parameters", "ef",
    "gate_mode", "gates", "get_parameter", "i_bot", "i_channel", "i_top",
    "is_dual_gated", "is_nested", "n_declared_sweeps", "n_points", "n_sweeps",
    "nearest_index", "nesting", "parameter_labels", "power", "scanner_x",
    "scanner_y", "signal_label", "signal_name", "signal_unit", "spectroscopy",
    "sweep_axis", "sweep_axis_label", "sweep_grid", "sweep_label",
    "sweep_unit", "to_hdf5", "v_bot", "v_channel", "v_top",
    "varying_parameters",
}

# Independent of which instrument wrote the file: a sweep of spectra has an
# energy axis, a correction ladder and a way to pick a pixel window or a column.
_FROM_A_SPECTRAL_SWEEP = {
    "best_energy_spectra", "best_spectra", "contrast_label",
    "get_spectrum_at", "get_spectrum_by_index", "n_pixels", "pixel_slice",
}

# AttoCube-only: two CCD regions of interest, and the two aliases kept until
# plot_pl_map_Vab_scan is updated.
_FROM_THE_ATTOCUBE_SPECTRAL_CLASS = {"roi", "gate_axis", "gate_axis_label"}

# The temporal class shares the base and nothing else: no `spectra`, no energy
# machinery, and `decays` in place of both.
_FROM_THE_TRPL_CLASS = {"axis_label", "best_decays", "n_bins"}


# --- The public surface ----------------------------------------------------

@pytest.mark.parametrize(
    "cls, expected",
    [
        (ACSpectralSweep,
         _FROM_THE_SWEEP_BASE | _FROM_A_SPECTRAL_SWEEP
         | _FROM_THE_ATTOCUBE_SPECTRAL_CLASS),
        (ACTRPLSweep,
         _FROM_THE_SWEEP_BASE | _FROM_THE_TRPL_CLASS),
    ],
    ids=["spectral", "trpl"],
)
def test_every_public_member_is_still_reachable(cls, expected):
    missing = expected - set(dir(cls))
    assert not missing, (
        f"{cls.__name__} no longer exposes {sorted(missing)}. A member that "
        f"moved to the wrong class is invisible to every other test and to "
        f"mkdocs --strict."
    )


def test_the_spectral_and_temporal_surfaces_stay_distinct():
    # Handing a TRPL sweep to a spectral plot must keep raising rather than
    # drawing time as if it were wavelength, which is what 0008 turns on.
    spectral_only = _FROM_A_SPECTRAL_SWEEP | _FROM_THE_ATTOCUBE_SPECTRAL_CLASS
    leaked = spectral_only & set(dir(ACTRPLSweep))
    assert not leaked, (
        f"ACTRPLSweep has grown {sorted(leaked)}, which only means "
        f"anything for a wavelength axis."
    )
    assert "spectra" not in dir(ACTRPLSweep)
    assert "decays" not in dir(ACSpectralSweep)


# --- Which arrays an archive carries ---------------------------------------

def test_the_signal_datasets_come_from_the_class_not_the_layout():
    # Two instruments can share a wavelength axis and still write a different
    # number of signal arrays, so `_LAYOUT_KIND == "spectral"` is not a valid
    # test for "carries two regions of interest".
    assert ACSpectralSweep._HDF5_SIGNALS == (
        ("roi1", "spectra_roi1"), ("roi2", "spectra_roi2"))
    assert ACTRPLSweep._HDF5_SIGNALS == (("counts", "decays"),)


def test_an_undeclared_signal_table_is_refused_by_name():
    # Reached, not merely written: a refusal nothing can trigger is the defect
    # A29 records. A class that inherits the empty default must say so rather
    # than write an archive with no signal in it.
    class _Undeclared:
        _HDF5_SIGNALS = ()

    with pytest.raises(NotImplementedError, match="_Undeclared declares no"):
        hdf5._signal_arrays(_Undeclared())


# --- Where a warning is blamed ---------------------------------------------

def test_the_jacobian_warning_is_not_blamed_on_the_package():
    """
    The Jacobian-without-background warning must not name a line in the loader.

    It is raised inside ``_build_correction_ladder``, one frame further from the
    caller than ``__init__`` is, so its depth is threaded in as a parameter
    rather than written as a literal at the ``warnings.warn`` call.  Get that
    count wrong and the warning blames a line inside the package, which tells
    the researcher nothing and dedups against the wrong registry.

    This pins the *frame count*, not the exact target.  Measured here, the
    warning currently lands one level above the line that constructed the scan
    rather than on it, so it reads as ``sys:1`` from a module-level call.  That
    off-by-one predates this test and belongs to the standing ``stacklevel``
    audit in `.claude/CLAUDE.md`; asserting the exact target would freeze it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ACSpectralSweep(SPECTRAL, spectra_type="PL", apply_jacobian=True)

    jacobian = [w for w in caught
                if "apply_jacobian=True with no background" in str(w.message)]
    assert jacobian, "the warning was not raised at all"
    assert Path(jacobian[0].filename).name != "loaders.py", (
        f"blamed {jacobian[0].filename}:{jacobian[0].lineno}, a line inside "
        f"the package"
    )
