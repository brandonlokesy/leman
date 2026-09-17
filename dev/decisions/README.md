# Decision records

One file per decision, **append-only**. A record describes a choice that was made,
the alternatives that were rejected, and what became true as a result.

The field that earns this folder is **Rejected**. Everything else here is recoverable
from somewhere else — the decision is in the source, the date and diff are in
`git log`, the defect is in the audit. Rejected alternatives are recoverable from
nothing, because rejected code was never committed. If you write only one section
properly, write that one.

## The rule that makes it work

**A record is never rewritten after it is accepted.** If the decision changes, write a
*new* record and point the two at each other. Editing an accepted record destroys the
thing the folder exists to keep, and produces a file whose sentences belong to
different years with no way to tell which is which.

## Naming and numbering

```
dev/decisions/NNNN-short-slug.md
```

`NNNN` is **allocation order, not chronology** — take the next free number when you
write the record, whatever date the decision was made. Numbers are permanent
identifiers: never renumber, never re-use, never close a gap. The `Date` field carries
chronology; that is its job.

## The template

```markdown
# NNNN — One line naming the decision, not the problem

| | |
|---|---|
| **Status** | Accepted |
| **Date** | YYYY-MM-DD |
| **Audit** | E13 · (omit if the decision had no audit finding behind it) |

## Context
What forced a choice. The measurement, the failure, the constraint. Enough that the
decision reads as necessary rather than arbitrary.

## Decision
What was chosen, stated as behaviour.

## Rejected
Each alternative that was genuinely considered, and what specifically was wrong with
it. Not a list of everything imaginable — a list of what someone would otherwise
propose again.

## Consequences
What is now true. What a caller must do. What this closes off.

## Load-bearing choices        ← optional
The parts most worth revisiting if something downstream later feels wrong.
```

**Status** is one of:

| Status | Meaning |
|---|---|
| `Accepted` | in force |
| `Amended by NNNN` | still in force, with a later record adjusting part of it |
| `Superseded by NNNN` | no longer in force; the record stays for the rejected alternatives it holds |

Adding a status line to an old record is the **only** permitted edit to it.

## One record covers one change, not one choice

A single change often settles several coupled choices — where a correction runs *and*
what shape its argument takes. Keep those in one record with a numbered `Decision`
section. Splitting them produces two files that are each unreadable without the other,
which is worse than one file with two parts.

## What does not go here

| Content | Home |
|---|---|
| How the exporter lays out a file; what a column physically is | `dev/instruments/<system>.md` |
| A quantity's defining equation, units, domain of validity | `dev/physics-conventions.md` |
| How the package works *now* — shapes, vocabulary, call flow | `dev/architecture.md` |
| A defect: something was wrong, here is the fix and the test | the audit |
| Work not yet done, or done and reverted | `dev/TODO.md`, `dev/plan-*.md` |
| The one-line imperative a future contributor must not violate | `.claude/CLAUDE.md` |

The last row is the one to get right: CLAUDE.md carries the **rule**, this folder
carries the **reasoning**. A rule with its reasoning attached is too long to stay
loaded; reasoning with no rule extracted never gets followed.

Implementation traces — call flows with line numbers — do not belong in a record
either. They rot immediately and they describe the code, which `dev/architecture.md`
already does for the current state.

## Index

Ordered by number, which is allocation order — see the `Date` column for chronology.

| # | Decision | Date | Status |
|---|---|---|---|
| [0001](0001-cosmic-ray-repair-at-load-time.md) | Cosmic-ray repair runs at load time, declared by one dict | 2026-08-05 | Accepted · amended (A17) |
| [0002](0002-gate-wiring-must-be-declared.md) | Gate wiring is declared with `gates=`, and the loader refuses without it | 2026-08-05 | Accepted · extended by 0003 |
| [0003](0003-gates-declares-device-topology.md) | `gates` declares device topology, and carries the carrier-density path | 2026-08-05 | Accepted |
| [0004](0004-nested-sweeps-fast-and-slow.md) | Nested sweeps use `fast_sweep=`/`slow_sweep=`; flat stays canonical | 2026-08-06 | Accepted · amended (A13) · §3 superseded by 0016 |
| [0005](0005-electrode-currents-are-role-named.md) | Electrode currents are role-named and resolved from `gates=` | 2026-08-07 | Accepted |
| [0006](0006-loader-rewrite-and-sweep-vocabulary.md) | The spectral loader declares what it is, and never auto-detects the sweep axis | 2026-07-30 | Accepted |
| [0007](0007-hdf5-stores-no-derived-arrays.md) | One loader class reads both formats, and HDF5 stores no derived arrays | 2026-07-30 | Accepted |
| [0008](0008-trpl-is-a-separate-class.md) | TRPL is a separate class; its metadata companion is evidence, not the source | 2026-07-30 | Accepted |
| [0009](0009-background-before-jacobian.md) | Background is subtracted in wavelength space, before the Jacobian | 2026-08-04 | Accepted |
| [0010](0010-file-order-from-iter-index.md) | Directory order comes from the `_iter_N` integer; anomalies warn, never repair | 2026-08-06 | Accepted |
| [0011](0011-label-contract-derive-or-verbatim.md) | Labels: `None` derives, a string is verbatim, nothing is appended | 2026-08-06 | Accepted |
| [0012](0012-plot-spectrum-selects-like-the-accessors.md) | `plot_spectrum` selects by coordinate, keyword-only, through the accessors | 2026-08-10 | Accepted |
| [0013](0013-power-series-thinning-and-stacking.md) | `plot_power_series` thins by slice step and stacks by an absolute offset | 2026-08-05 | Accepted |
| [0014](0014-animate-by-frame-index.md) | An animation is driven by a sequence of frame indices | 2026-08-12 | Accepted |
| [0015](0015-reversed-frame-window-endpoints-are-refused.md) | A reversed coordinate window is refused, not reversed for you | 2026-08-13 | Accepted |
| [0016](0016-nest-levels-must-hold-apart.md) | A nest is verified by levels holding apart, not by a tolerance | 2026-08-17 | Accepted |
| [0017](0017-asserted-shape-and-grouping-row.md) | A nest's shape and its grouping row are declared with named keywords | 2026-08-17 | Accepted |
| [0018](0018-a-held-axis-is-recognised-by-two-signs.md) | An axis that never moved is recognised by two signs, either sufficient | 2026-08-18 | Accepted · §1 amended by 0019 |
| [0019](0019-a-driven-axis-is-also-recognised-by-direction.md) | A driven axis is also recognised by which way it steps | 2026-08-18 | Accepted |
| [0020](0020-plot-image-annotates-and-returns-the-circle.md) | `plot_image` carries the laser annotation, and returns the circle it drew | 2026-08-18 | Accepted |
| [0021](0021-the-conjugate-axis-has-one-implementation.md) | The conjugate spectral axis has one implementation, and a plot returns the one it drew | 2026-08-18 | Accepted |
| [0022](0022-the-animation-wrapper-returns-its-panels.md) | The animation wrapper forwards one dict to its spectrum panel, and returns its panels | 2026-08-18 | Accepted |
| [0023](0023-the-cosmic-ray-fill-ignores-flagged-pixels.md) | The cosmic-ray replacement median ignores the pixels it is repairing | 2026-08-18 | Accepted |
| [0024](0024-long-plotting-returns-are-named.md) | A plotting return longer than `(fig, ax, artist)` is a named tuple | 2026-08-19 | Accepted · extended by 0025 |
| [0025](0025-plot-image-carries-its-colorbar.md) | `plot_image` carries its colorbar, and a new member is appended | 2026-08-19 | Accepted |
| [0026](0026-plot-image-carries-the-coordinate-mapping.md) | `plot_image` carries the coordinate mapping, as two named parameters | 2026-08-19 | Accepted |
| [0027](0027-the-suite-runs-in-ci-on-three-systems.md) | The test suite runs in CI on three operating systems, and a red run blocks a merge | 2026-08-20 | Accepted |
| [0028](0028-the-spectral-map-pins-a-nest.md) | The spectral map pins a nest, through the resolver the series already used | 2026-08-20 | Accepted |
| [0029](0029-a-curated-rows-unit-is-declared-with-its-scale.md) | A curated row's unit is declared alongside its scale, and lives in one place | 2026-08-20 | Accepted |
| [0030](0030-clim-and-rescale-img-are-refused-together.md) | `clim` and `rescale_img` are refused together, in both functions | 2026-08-21 | Accepted |
| [0031](0031-the-spectral-maps-mesh-runs-one-row-per-sweep-point.md) | The spectral map's mesh runs one row per sweep point | 2026-08-21 | Accepted |
| [0032](0032-converted-files-land-in-a-processed-folder.md) | Converted files land in a `processed/` folder | 2026-08-21 | Superseded by 0034 (name only) |
| [0033](0033-a-trpl-directory-converts-only-when-named.md) | A TRPL directory converts only when it is named | 2026-08-21 | Accepted |
| [0034](0034-converted-files-land-in-a-converted-folder.md) | Converted files land in a `converted/` folder, not `processed/` | 2026-08-21 | Accepted · §2 extended by 0036 |
| [0035](0035-out-is-an-output-root-and-the-tree-is-mirrored.md) | `out=` is an output root, and the tree is mirrored beneath it | 2026-08-21 | Accepted |
| [0036](0036-the-default-root-comes-from-the-folder-you-named.md) | The default output root comes from the folder you named; `from_raw=` opts into searching | 2026-08-21 | Accepted |
| [0037](0037-beside-is-shorthand-not-a-placement-rule.md) | `beside=` is shorthand for one `out=`, not a third placement rule | 2026-08-22 | Accepted |
| [0038](0038-the-spectral-machinery-is-not-per-instrument.md) | A sweep of spectra gets its own base, and it has no `__init__` | 2026-08-26 | Accepted |
| [0039](0039-an-unnamed-row-beats-a-guessed-one.md) | An unnamed parameter row beats a guessed one, and one signal beats a fictional pair | 2026-08-26 | Accepted |
| [0040](0040-reference-hdf5-has-a-schema.md) | Reference HDF5 files use a versioned schema with three layouts | 2026-09-17 | Accepted |

This table is the one mutable part of the folder.
