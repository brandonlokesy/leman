"""
Key-parity invariants between the two vocabulary tables in ``constants``.

``SPECTROSCOPY_TYPES`` and ``SIGNAL_LABELS`` are keyed identically but reach
different code. The first gates ``spectra_type=`` at load time, supplies the
option list in the resulting error messages, and is re-exported to the
``reference`` subpackage; the second is read only by ``_Sweep``'s label
properties and by ``contrast_label``. Nothing structural holds them in step, so
adding a measurement type means editing two dicts.

The two ways they can fall out of step are **not** equally harmless, which is
why this is two tests and not one ``==``. A key present only in
``SPECTROSCOPY_TYPES`` validates, loads, and then raises ``KeyError`` at the
first label access — typically inside a plotting call, far from the load that
caused it. A key present only in ``SIGNAL_LABELS`` is merely unreachable. One
is a latent crash; the other is dead weight.

The tables are deliberately *not* merged. Their audiences differ (only the
first crosses into ``reference``), and two of the six rows do not map
one-to-one: ``SIGNAL_LABELS["RC"]`` is read by an ``"R"`` scan's
``contrast_label`` rather than by an "RC" scan, and TRPL's row duplicates PL's
because the signal is identical and only the x-axis differs. The argument is in
``dev/plan-E12.md``.
"""

import pytest

from leman.constants import SIGNAL_LABELS, SPECTROSCOPY_TYPES


def test_every_spectroscopy_type_has_a_signal_label():
    """The crash direction: validates as a type, then cannot be labelled."""
    missing = set(SPECTROSCOPY_TYPES) - set(SIGNAL_LABELS)
    assert not missing, (
        f"{sorted(missing)} pass spectra_type= validation but have no "
        f"SIGNAL_LABELS row, so such a scan loads cleanly and then raises "
        f"KeyError at the first signal_name / signal_label access. Add a "
        f"(name, unit) row for each."
    )


def test_signal_labels_declare_no_unknown_type():
    """
    The dead-weight direction: a label no scan can reach.

    A row read by direct lookup rather than through ``spectra_type`` would be a
    legitimate exception — ``contrast_label`` does exactly that with ``"RC"``.
    If this test fails for such a row, the fix may be to document the direct
    consumer rather than to add a technique.
    """
    orphans = set(SIGNAL_LABELS) - set(SPECTROSCOPY_TYPES)
    assert not orphans, (
        f"{sorted(orphans)} have a signal label but are not valid "
        f"spectra_type= values, so no scan can carry them."
    )


@pytest.mark.parametrize("spectra_type", sorted(SIGNAL_LABELS))
def test_signal_label_rows_unpack_to_name_and_unit(spectra_type):
    """
    ``signal_label`` does ``name, unit = SIGNAL_LABELS[...]``, so the arity is
    a contract: a bare string would unpack into its first two characters.
    """
    row = SIGNAL_LABELS[spectra_type]
    assert isinstance(row, tuple) and len(row) == 2, row

    name, unit = row
    assert isinstance(name, str) and name, f"{spectra_type}: empty signal name"
    # An empty unit is correct for a dimensionless ratio, so only the type is
    # checked here — never the truthiness.
    assert isinstance(unit, str), f"{spectra_type}: unit {unit!r} is not a str"


@pytest.mark.parametrize("spectra_type", sorted(SPECTROSCOPY_TYPES))
def test_technique_names_are_non_empty(spectra_type):
    """``scan.spectroscopy`` returns these verbatim, including into __repr__."""
    technique = SPECTROSCOPY_TYPES[spectra_type]
    assert isinstance(technique, str) and technique.strip(), (
        f"{spectra_type}: technique name {technique!r}"
    )
