"""
The spectra ladder: three cumulative rungs per axis, and the two axes mirrored.

``spectra`` / ``spectra_cr`` / ``spectra_bg`` in wavelength space and
``energy_spectra`` / ``energy_spectra_cr`` / ``energy_spectra_bg`` on the energy
axis, each rung the one above it plus one further correction. What is pinned here
is the *table*, which is why every test is parametrised over the four declarations
a caller can make — neither correction, each alone, and both — rather than over
one interesting case:

* a rung exists exactly when its correction was asked for;
* each energy rung is its wavelength rung reordered, times the Jacobian when one
  was asked for and never otherwise;
* both ``best_*`` return the *same* rung, which is the property whose absence
  produced A16.

The spike recipe comes from the loader-level cosmic-ray tests, so both files
describe one measurement. Several assertions would hold vacuously on a scan whose
spike was never flagged, so the first test rules that out.
"""

import numpy as np
import pytest

from leman.constants import HC_EV_NM
from leman.loaders import (
    ACSpectralSweep,
    SingleSpectrum,
    _resolve_spectra,
)

from test_loaders import make_spectral_csv
from test_loaders_cosmic_rays import (
    BG_REGION,
    CR_PIXEL,
    CR_SWEEP,
    WL,
    _sweep,
    _with_spikes,
)

# The four declarations, as loader kwargs. Named so a parametrised failure says
# which corrections were in play.
DECLARATIONS = {
    "none":  {},
    "cr":    {"cosmic_rays": {}},
    "bg":    {"bg_region_nm": BG_REGION},
    "cr+bg": {"cosmic_rays": {}, "bg_region_nm": BG_REGION},
}

# rung name -> (wavelength attribute, energy attribute), lowest correction last.
RUNGS = {
    "raw": ("spectra",    "energy_spectra"),
    "cr":  ("spectra_cr", "energy_spectra_cr"),
    "bg":  ("spectra_bg", "energy_spectra_bg"),
}


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path, roi1=_with_spikes(_sweep()), wavelength=WL)
    return path


@pytest.fixture
def load(csv_path):
    def _load(**kwargs):
        return ACSpectralSweep(str(csv_path), spectra_type="PL", **kwargs)
    return _load


def _ascending_energy(spectra: np.ndarray) -> np.ndarray:
    """Reorder a wavelength-space array onto the ascending energy axis."""
    return spectra[np.argsort(HC_EV_NM / WL)]


# ---------------------------------------------------------------------------
# The shape of the table
# ---------------------------------------------------------------------------


def test_the_repair_actually_moves_a_pixel(load):
    """The guard: several assertions below are vacuous if no spike was flagged."""
    scan = load(**DECLARATIONS["cr"])
    assert scan.cosmic_ray_mask[CR_PIXEL, CR_SWEEP]
    assert not np.allclose(scan.spectra, scan.spectra_cr)


@pytest.mark.parametrize("declared", list(DECLARATIONS))
def test_a_rung_exists_exactly_when_its_correction_was_asked_for(load, declared):
    scan     = load(**DECLARATIONS[declared])
    expected = {"raw": True,
                "cr":  "cr" in declared,
                "bg":  "bg" in declared}

    for rung, (wl_attr, en_attr) in RUNGS.items():
        for attr in (wl_attr, en_attr):
            present = getattr(scan, attr) is not None
            assert present is expected[rung], f"{attr} with {declared!r}"


@pytest.mark.parametrize("declared", list(DECLARATIONS))
def test_each_energy_rung_mirrors_its_wavelength_rung(load, declared):
    """The symmetry itself — the one test to keep if only one survives."""
    scan = load(**DECLARATIONS[declared])

    for wl_attr, en_attr in RUNGS.values():
        wl_rung = getattr(scan, wl_attr)
        if wl_rung is None:
            continue
        np.testing.assert_allclose(getattr(scan, en_attr),
                                   _ascending_energy(wl_rung))


@pytest.mark.parametrize("declared", list(DECLARATIONS))
def test_best_picks_the_same_rung_on_both_axes(load, declared):
    """Asserted together, because the two drifting apart is what A16 was."""
    scan     = load(**DECLARATIONS[declared])
    expected = ("bg" if "bg" in declared
                else "cr" if "cr" in declared
                else "raw")
    wl_attr, en_attr = RUNGS[expected]

    assert scan.best_spectra        is getattr(scan, wl_attr)
    assert scan.best_energy_spectra is getattr(scan, en_attr)


# ---------------------------------------------------------------------------
# What each rung holds
# ---------------------------------------------------------------------------


def test_the_first_rung_is_the_file_counts_even_with_a_repair(load):
    # The regression that would silently reappear: folding the repair into the
    # first rung leaves no way to reach the counts the file actually holds.
    plain, repaired = load(), load(**DECLARATIONS["cr"])

    np.testing.assert_allclose(repaired.spectra, plain.spectra)
    np.testing.assert_allclose(repaired.energy_spectra, plain.energy_spectra)
    assert not np.allclose(repaired.energy_spectra, repaired.energy_spectra_cr)


def test_the_background_rung_carries_the_repair(load):
    # Pins the cumulative contract and the cr-before-bg ordering in one: the
    # pedestal comes off the repaired counts, so a spike inside the window does
    # not inflate the estimate that is subtracted.
    both = load(**DECLARATIONS["cr+bg"])
    bg_only = load(**DECLARATIONS["bg"])

    window = (WL >= BG_REGION[0]) & (WL <= BG_REGION[1])
    # (1, n_sweeps) pedestal per sweep, from the repaired counts.
    pedestal = both.spectra_cr[window, :].mean(axis=0)
    np.testing.assert_allclose(both.spectra_bg,
                               both.spectra_cr - pedestal[None, :])
    # And it is not the same as subtracting a spike-inflated pedestal.
    assert not np.allclose(both.spectra_bg, bg_only.spectra_bg)


def test_the_contrast_sits_outside_the_ladder(tmp_path, csv_path):
    ref = tmp_path / "bare.csv"
    ref.write_text(
        ",".join(f"{w}" for w in WL) + "\n"
        + ",".join(f"{v}" for v in np.full(WL.size, 500.0)) + "\n"
    )
    scan = ACSpectralSweep(str(csv_path), spectra_type="R",
                                 reference=str(ref), cosmic_rays={})

    assert scan.contrast is not None and scan.energy_contrast is not None
    for served in (scan.best_spectra, scan.best_energy_spectra):
        assert not np.allclose(served, scan.contrast)
        assert not np.allclose(served, scan.energy_contrast)


# ---------------------------------------------------------------------------
# The Jacobian is a property of the energy representation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reaching the rungs by name: a source is a correction, x_axis is the axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source, rung", [("raw", "raw"), ("cr", "cr"), ("bg", "bg")])
@pytest.mark.parametrize("x_axis", ["wavelength", "energy"])
def test_every_state_is_reachable_on_both_axes(load, source, rung, x_axis):
    scan = load(**DECLARATIONS["cr+bg"])
    attr = RUNGS[rung][0 if x_axis == "wavelength" else 1]
    assert np.array_equal(_resolve_spectra(scan, source, x_axis),
                          getattr(scan, attr))


@pytest.mark.parametrize("retired", ["energy", "energy_bg", "contrast_wavelength",
                                     "energy_pre_jacobian"])
def test_the_retired_keys_raise_and_name_the_vocabulary(load, retired):
    # These keys baked the axis into the name. Refusing them is what stops a call
    # that used to mean one array quietly meaning another.
    scan = load(**DECLARATIONS["cr+bg"])
    with pytest.raises(ValueError, match="is not recognised") as excinfo:
        _resolve_spectra(scan, retired, "energy")
    assert "'raw'" in str(excinfo.value) and "'bg'" in str(excinfo.value)


def test_an_unrequested_correction_names_the_argument_that_enables_it(load):
    scan = load()
    with pytest.raises(ValueError, match=r"cosmic_rays="):
        _resolve_spectra(scan, "cr", "wavelength")
    with pytest.raises(ValueError, match=r"bg_region_nm="):
        _resolve_spectra(scan, "bg", "energy")


def test_a_correction_the_class_does_not_offer_names_the_class(tmp_path):
    # A SingleSpectrum takes no cosmic_rays=, so advising it would send the caller
    # after an argument that does not exist.
    path = tmp_path / "one.csv"
    path.write_text(
        ",".join(f"{w}" for w in WL) + "\n"
        + ",".join(f"{v}" for v in _sweep()[:, 0]) + "\n"
    )
    with pytest.raises(ValueError, match="SingleSpectrum has no 'cr'"):
        _resolve_spectra(SingleSpectrum(str(path)), "cr", "wavelength")


def test_the_jacobian_reaches_every_energy_rung_and_no_wavelength_one(load):
    from leman.processing import jacobian_correction_wvl2E

    on  = load(**DECLARATIONS["cr+bg"], apply_jacobian=True)
    off = load(**DECLARATIONS["cr+bg"])

    for wl_attr, en_attr in RUNGS.values():
        # Untouched in wavelength space, whatever apply_jacobian says.
        np.testing.assert_allclose(getattr(on, wl_attr), getattr(off, wl_attr))
        np.testing.assert_allclose(
            getattr(on, en_attr),
            _ascending_energy(jacobian_correction_wvl2E(
                getattr(on, wl_attr), WL, axis=0)),
        )


def test_pre_jacobian_is_refused_on_the_wavelength_axis(load):
    # The wavelength arrays already hold the values the Jacobian is applied to, so
    # there is a correct alternative to name rather than a warning to emit.
    scan = load()
    with pytest.raises(ValueError, match="energy axis only"):
        _resolve_spectra(scan, "pre_jacobian", "wavelength")
    assert _resolve_spectra(scan, "pre_jacobian", "energy") is not None


def test_pre_jacobian_is_the_first_rung_without_the_correction(load):
    off = load()
    assert off.energy_spectra_pre_jacobian is off.energy_spectra

    on = load(apply_jacobian=True, bg_region_nm=BG_REGION)
    np.testing.assert_allclose(on.energy_spectra_pre_jacobian,
                               _ascending_energy(on.spectra))
    assert not np.allclose(on.energy_spectra_pre_jacobian, on.energy_spectra)
