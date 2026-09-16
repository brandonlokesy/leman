# 0029 — A curated row's unit is declared alongside its scale, and lives in one place

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-20 |
| **Audit** | B6 |

## Context

`_CURATED` promotes a handful of instrument rows to named properties, storing each as
`(row label, scale, unit)`. Two of those three were declarable — `curated_labels=` and
`curated_scales=` — and the unit was not.

The package actively recommends the scale override. A piezo row carries drive voltage,
and converting it to a distance needs a per-stage µm/V calibration that is not in the
file, so both the `_SWEEP_TYPES` comment and the `scanner_x` / `scanner_y` attribute
entry told the caller to supply one through `curated_scales`. Following that advice
produced micrometre numbers under a "V" label.

The audit filed this as one missing override. It was not. Reading every reader of the
registry showed the unit slot was consumed by exactly one thing — the
`curated_parameters` property, a documentation view — while every unit a researcher
actually sees came from a **second copy in `_SWEEP_TYPES`**, and the `__repr__` power
line from a **third, a hardcoded literal**. One quantity, three homes, one of them
overridable. Any override desynchronised them, and adding a fourth declarable copy
would have fixed the view and left the labels wrong.

## Decision

**1. A curated row's unit is declared through `curated_units=`, beside its scale.**

Same keys as `curated_scales`, same store in the same override loop, same
argument-beats-file-beats-default resolution. It reaches the gate and current entries,
which `curated_labels` refuses: a row is a wiring claim and `gates=` is its one
spelling, but a unit claims nothing about wiring — the reasoning already applied to
`curated_scales` in 0002. `""` is the dimensionless spelling; `None` is refused, because
`str(None)` would label an axis `"None"`.

**2. The unit lives in the registry, and `_SWEEP_TYPES` stops holding a second copy.**

A curated-backed sweep type now spells `None` for its unit, meaning *read it from the
registry*. Only the three axes with no curated row behind them keep a literal: the sweep
index, and the two quantities derived from gate voltages rather than promoted from a row.
`__repr__`'s power line reads the registry too. So one declaration reaches the property,
the repr, the axis label, the held-axis warning and every legend that names the
coordinate — which is the property that makes the argument worth having.

**3. A scale the caller changed with no unit beside it warns.**

Naming the entry, both scale values, and the unit left standing. Not raised: the numbers
are correct either way, and a researcher may legitimately be mid-calibration. Not
silent: a label that misstates its numbers is a misread, and nothing downstream can
detect it. Restating the unit silences it, which is exactly what a polarity flip does —
`-1e9` is still nA.

The warning is for a *caller*. A file records a value for every entry, so reading an
archive back never triggers it; otherwise every stored non-default scale would warn on
every load.

## Rejected

**`curated_units=` on its own, with no deduplication.** The shape the audit proposed.
It fixes `curated_parameters` and nothing a researcher looks at — the axis label, the
repr line and the legend all read `_SWEEP_TYPES`. It would have closed the entry while
leaving the reported symptom intact, and added a fourth place the same string lives.

**`curated_units=` with no warning.** Cheapest, and detects nothing. The argument exists
because a rescaled row with a stale unit is undetectable downstream; shipping the door
without the detection leaves the same trap for anyone who does not know the door is
there — which is everyone reading the advice that caused the problem.

**Refusing a scale override that arrives without a unit.** Impossible to misuse, and it
breaks the polarity flip 0005 documents (`{"i_bot": -1e9}`, where nanoamps is genuinely
still the unit) by demanding a restatement before the call is legal. It also cannot apply
to units read from a file without inventing a second silent-for-archives /
raise-for-callers asymmetry beside the one `curated_labels` already needs. A warning buys
the detection without either cost.

**One argument taking a `(scale, unit)` pair.** The most principled: two facts that must
agree cannot then disagree, which is how `gates=` is shaped. Rejected on cost, not on
merit. Every existing call site and the stored HDF5 shape for that key would change, and
accepting *either* a float or a pair would make the parameter's type depend on how much
the caller happened to know. Worth revisiting if the two ever drift again in practice.

**Keeping the `_SWEEP_TYPES` unit as a fallback the registry overrides.** Two live
copies with a precedence rule between them, which is harder to reason about than one
copy, and it leaves the dead literal in place for someone to "fix" back into use.

## Consequences

- One declaration states what a rescaled row now is, and it reaches every reader.
- `FORMAT_VERSION` moves 2.2 → 2.3. The unit was previously not persisted at all — it
  was re-derived from the class default on read, so an archive could restore a scale and
  silently revert its unit. The version gate is on the major, so this needs no migration.
- `_SWEEP_TYPES` is no longer the home of a curated-backed axis's unit. Putting a literal
  back there for one of those five axes reintroduces B6.
- The five collapsed pairs agreed before the change, so nothing about an un-overridden
  scan moved. That is pinned by a test, not asserted.
- `ACPLVabScan` keeps no unit argument. It forwards `power_scale` through
  `super().__init__`, so using it warns through one extra frame and points at its own
  call rather than the caller's line. It is deprecated; the frame is documented where it
  happens rather than paid for with a new parameter on a retiring class.
- The `plotting` panel defaults (`sweep_unit="V"`, `sweep_units="V"`) are a further copy
  and are **not** touched here — they belong to **E12**, which owes that function a
  rename in the same breaking change.

## Load-bearing choices

**That the warning keys on what the caller passed, not on what the values are.** A
value-based test — "the scale differs from the default and the unit does not" — would
fire on the polarity flip and, worse, could not be silenced by restating the correct
unit, because the correct unit *is* the default. Key presence is what makes restating
the unit both meaningful and sufficient.

**That a changed scale is compared against the value in force, not the class default.**
Restating the scale a file already recorded is not a rescale and must not warn.
