# 0008 — TRPL is a separate class, and its metadata companion is evidence rather than the source

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-07-30 |
| **Audit** | G3, G4 |

## Context

A TRPL sweep is a directory of single-decay files, each carrying its own full parameter
snapshot, plus a metadata companion whose parameter columns hold one snapshot per point.
The measured axis is time, not wavelength — and the export names that column
`Wavelength` anyway.

## Decision

### 1. A sibling class, not a mode

`ACTRPLSweep` sits beside `ACSpectralSweep` over a shared private base, and
accepts one file *or* a directory. It has **no `spectra` attribute**.

### 2. Parameters come from the data files, not the companion

Each data file's own 57-row snapshot is contemporaneous with its decay, so a sweep loads
without the companion at all. The companion supplies the declared sweep count — which is
what makes an aborted sweep visible — and its table is exposed for inspection.

Its values are **not** cross-checked row by row.

## Rejected

**One class with a mode flag.** A single decay is just `n_sweeps == 1`, so that part
would have worked. But `energy = hc/t` is meaningless and divides by zero at `t = 0`, so
a mode flag would leave roughly a third of the public API conditionally meaningful.

**Giving the TRPL class a `spectra` attribute** for interface compatibility. Without one,
a TRPL sweep handed to a spectral plot raises instead of drawing time as wavelength. The
cost is that decay plotting is unwritten rather than broken.

**Row-by-row validation against the companion.** It is written seconds after the last
decay, so genuinely drifting channels disagree — the leakage currents and
`Fianium_Select_A6` do, while the swept gates agree to seven figures. Nothing in the file
says which channels are stable, so a value check would fire on every real sweep, which is
how warnings get ignored. Contrast the same file's over-allocation, which *is* keyed on,
because there the sentinel is exact.

**Trusting the companion as the parameter source.** It collides with the first data file
on `iter_0` and is written last, so it must be classified by **content, not filename** —
and the per-file snapshots are the contemporaneous record anyway.

## Consequences

- A directory missing its companion loads, without a declared-count cross-check.
- Time-axis plotting and lifetime fitting are both absent rather than approximated:
  the plotting x-axis resolver knows only energy and wavelength, and there is no
  exponential model in `fitting`. The baseline machinery is already generic over the
  model function, so a decay fit is a small generalisation rather than a parallel
  implementation.
