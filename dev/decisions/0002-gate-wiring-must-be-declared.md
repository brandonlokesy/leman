# 0002 — Gate wiring is declared with `gates=`, and the loader refuses without it

| | |
|---|---|
| **Status** | Accepted · extended by [0003](0003-gates-declares-device-topology.md) |
| **Date** | 2026-08-05 |
| **Audit** | E7b |

## Context

The gate electrodes can be wired either way round, so which acquisition channel drove
which gate is a per-session fact that no file records. The curated registry nonetheless
hardwired `v_top → "V_A"` and `v_bot → "V_B"`, and applied it silently.

It bit on a real measurement: a voltage applied to a device's **bottom** gate was read
as the top gate. Three consequences, in increasing severity:

- `sweep="bottom_voltage"` passed validation and returned the *top* channel, with a
  plot that looked entirely normal.
- `ef` was mirrored, so any dipole extracted from it had the wrong sign.
- `gate_mode` returned `"top-gate only"`. The repr did not merely assume wrong, it
  **asserted** wrong.

Inherited material offers no help: the old MATLAB used `dm(4,…)`/`dm(6,…)` with no note
of which was which, and inherited group material is internally inconsistent about the
field's sign.

An earlier partial fix recorded the mapping through `curated_labels` and surfaced it in
`__repr__` and in exported HDF5, which made a transposition visible on sight. It still
defaulted when nothing was said, which is the part that failed.

## Decision

A keyword-only `gates={"top": <row>, "bottom": <row>}` on both sweep classes,
resolving onto the existing curated labels so that `curated_parameters`, `__repr__` and
HDF5 needed no new plumbing.

**What refuses without it:** `v_top`, `v_bot`, `ef` (only when a geometry was supplied —
reporting that no field was computed needs no wiring), and the gate sweep types, derived
from the intersection of the sweep requirements and the gate-role registry rather than
relisted. `v_top` / `v_bot` were removed as `curated_labels` *label* keys so the mapping
has exactly one spelling; `curated_scales` still reaches them.

**What deliberately does not raise**, because a diagnostic that dies is no diagnostic:

- `gate_mode` needs no mapping. How many channels moved together is a property of the
  data, and the Pearson correlation is symmetric in the two rows, so only the wording
  differs — a single driven gate is `"bottom-gate only"` with a mapping and
  `"single gate driven ('V_B')"` without.
- `__repr__` renders either way, guards its `E_F` line, and states the undeclared case
  using the `gates=` call shape. It is where someone looks after loading, and the
  failure being guarded against is a plot that looks fine.

The mapping is recorded on the scan and written into HDF5 as **its own attribute**,
present only when declared.

## Rejected

**Continuing to default.** The package cannot supply the wiring, but it can refuse to
proceed without it, and it was not refusing. This was the last place the package
guessed at something no file records — `spectra_type=` is already required with no
default, and an undeclared `sweep=` already means the index rather than an
auto-detected parameter.

**A warning instead of a refusal.** The failure mode is a plot that looks entirely
normal, so a warning would be read past.

**Renaming `v_top` / `v_bot` to `V_A` / `V_B`** — i.e. abolishing the role layer rather
than declaring into it. The role layer is where the field's sign convention lives, so
channel names there would make `electric_field` *look* unambiguous while still being
wrong half the time. That moves the ambiguity out of a recorded mapping and into the
researcher's head. Channels are already reachable as raw rows.

**Declaring the mapping anywhere other than `gates=`**, in particular as
`curated_labels` keys. One fact, one spelling.

**A separate list of gate-backed sweep types.** A constant
`("electric_field", "top_voltage", "bottom_voltage")` was not created: that information
is already in the sweep-requirements table, and a second copy could drift — add a
gate-backed sweep type next year, forget the list, and the requirement silently does not
apply. The requirement is instead derived by mapping a sweep's required curated attribute
back to its role, so a new sweep type inherits it from the declaration it already has to
make.

**Letting the blanket `curated_labels` dump serve as the HDF5 record.** It always
carries a resolved label for both gate rows and therefore cannot tell a declared
mapping from a defaulted one. A separate attribute is what stops a CSV → HDF5 → reload
round trip laundering an unstated wiring into provenance.

## Consequences

- **Channel-level work needs no declaration.** `scan["V_A"]` and `sweep="V_A"` are
  unaffected.
- The deprecated `ACPLVabScan` shim passes the historical `V_A`→top /
  `V_B`→bottom mapping **explicitly**, so existing notebooks keep producing the numbers
  they always did. Its `FutureWarning` is what asks callers to confirm the wiring rather
  than inherit it.
- Two plotting call sites that fell back to `scan.v_top` and labelled the axis
  `$V_\mathrm{top}$` now use `sweep_axis` / `sweep_axis_label`, which asserts nothing
  the scan was not told. Wiring is checked before `scan.ef` is read, since reading it
  undeclared raises.
- **Nothing in the code is still open.** The fact itself has to come from the lab
  notebook per session — that was always true, and is now enforced rather than assumed.

Verified against `examples/data/stark-shift/`: the shim reproduces
`ef[:3] = [-171.2652, -165.5573, -159.8484]` mV/nm, and the transposed mapping gives
the exact negation.
