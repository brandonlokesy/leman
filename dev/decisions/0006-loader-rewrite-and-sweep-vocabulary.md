# 0006 — The spectral loader declares what it is, and never auto-detects the sweep axis

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-07-30 |
| **Audit** | G1 · closed E1, and part of E7b |

## Context

`ACPLVabScan` was named and built for one measurement — a gate sweep on PL — and
was being used for reflectance, power series and rasters. It hardcoded "PL" in labels,
took eight separate `*_label` / `power_scale` arguments, and raised `KeyError` at load if
*any* curated row was missing, so a file from a different instrument configuration could
not be loaded at all.

## Decision

Renamed and rewritten as `ACSpectralSweep`.

- **`spectra_type=` is required, keyword-only, and has no default.** It is written into
  exported metadata and trusted thereafter, so a default would let a guess outlive the
  session. Downstream code asks the scan for its signal label rather than hardcoding
  "PL".
- **One `sweep=` argument** takes a registry key *or* any raw CSV row label.
  **Undeclared means the sweep axis is the sweep index**, never an auto-detected
  parameter.
- **No curated row is mandatory.** The file loads, and each curated property raises —
  listing the available labels — only if accessed. The one remaining fail-fast is for the
  row the *declared* `sweep=` needs, so the requirement follows what the caller said they
  measured rather than a fixed list.
- The eight `*_label` / `power_scale` arguments collapsed into `curated_labels=` /
  `curated_scales=`.
- Both ROIs are always loaded; `roi=` only chooses what `spectra` points at.
- The spectroscopy-type vocabulary **moved** to `constants.py` (gaining `"RC"`) and is
  re-exported from its old location. One vocabulary for the package.

The old name survives as a deprecated subclass emitting **`FutureWarning`**.

## Rejected

**A default `spectra_type`.** Any default is a guess that gets written into an archive
and trusted by everything downstream.

**Auto-detecting the sweep axis from which rows varied.** Mislabelling an axis is worse
than not labelling one, and `V_A` + `V_B` both varying is genuinely ambiguous between a
field sweep and independent gating. `varying_parameters()` and `gate_mode` return the
evidence instead, and the caller declares.

**`DeprecationWarning` for the shim.** Python filters it out by default outside
`__main__`, so a library raising one warns nobody.

**Deleting `gate_axis` / `gate_axis_label`.** Kept as aliases of `sweep_axis` /
`sweep_axis_label` so existing plotting keeps working; they cannot go until the map
plotting function is updated.

**Keeping a second copy of the spectroscopy-type vocabulary** in the reference loader.
The second copy is the bug.

## Consequences

- A file missing `Scanner X` loads; only touching `scanner_x` raises.
- `varying_parameters()` is **evidence, not a detector**: it ranks rows by span relative
  to their own RMS magnitude, so a small channel swinging across its whole range
  outranks a large one stepped through part of its. The top entry is often a leakage
  current.
- Existing notebooks keep working through the shim, with a warning that asks them to
  restate what the measurement was.
