# BigTable export — format and hardware facts

One file per acquisition system. This is the **record** for the BigTable export:
established from the two files committed under `examples/data/big-table-spectral/`
on 2026-08-26, and not to be re-derived by inference from the parser.

What belongs here: how the exporter lays out a file, and the hardware facts a column
encodes. What does not: why the loader responds to any of it the way it does — that is
a decision record in `dev/decisions/`. This file says *what the instrument and its
exporter did*; a decision record says *what we chose to do about it*.

**Where a number appears below, it is evidence from two exports, not a value a caller
should expect.** The mechanisms generalise; the arithmetic does not.

---

## Provenance and unknowns

Read this first, because it qualifies everything after it.

Established from exactly two files, both monolayer WSe₂ in a dual-gated structure,
both written on 2025-05-08:

| File | Blocks | Rows | What the numbers show |
|---|---|---|---|
| `PL_Doping_1LWSe2_p20uW_exp1sec_250508_190421.csv` | 61 | 1340 | the two gate rows move **together** |
| `PL_Ez_1LWSe2_p20uW_exp1sec_250508_190009.csv` | 101 | 1340 | the two gate rows move **oppositely** |

**The most important thing on this page: the parameter row names in
`loaders.py` are ours, not the file's.** The export carries no labels anywhere.
The names come from a MATLAB snippet written by a colleague who uses this setup,
checked against what the numbers do. That snippet is not a document that can be
cited, and it contains at least one internal contradiction (below), so treat the
naming as **inherited group practice** rather than as a specification.

Still unknown, and no committed file can answer it:

- **Which acquisition program and version emits this format**, and therefore whether
  the layout, the parameter row order, or the row count is version-stable. The layout
  is the AttoCube's four-column block minus the header, which suggests a shared
  LabVIEW lineage, but the parameter row order is different and that is an
  expectation, not a finding.
- **What columns 3 and 4 of a block are meant to be.** They are byte-identical in all
  217 080 pairs across the two files. The AttoCube's third and fourth fields are two
  CCD regions of interest, one of which is blank in every ordinary measurement; here
  the second is a copy rather than blank.
- **Rows 10, 12, 13, 17 and 30.** Row 10 is zero in both files. The other four carry
  values that are the same in both files and are described below, unidentified.
- **The unit of the gate-current rows (5 and 7).** The MATLAB snippet calls them
  amperes. They read between 0.01 and 0.4, which is 10–400 mA through a gate
  dielectric, so the label is not credible as written. The loader carries scale 1.0
  and **no unit string** rather than asserting one.
- **The contradiction.** The snippet maps row 9 to *both* `vg3` (gate voltage 3) and
  `lambda`, with the same trailing comment on two different lines, and names row 13
  `ig3` while leaving row 12 unmentioned. Row 9 is taken as gate voltage 3 on the
  user's instruction that the wavelength reading is not meaningful here; rows 12 and
  13 are left unnamed.
- **Whether the parameter block is 30 rows.** Nothing past row 30 is non-zero in
  either file, and no header states a count. The loader exposes 30 and warns if a row
  beyond it carries a value.
- **Whether row 8 is a measurement or a setpoint.** It is *exactly* constant to all
  five exported decimals across 61 and 101 points respectively, where the power row
  fluctuates by ~10% point to point. A live photodiode reading would fluctuate too,
  so row 8 looks like one reading copied into every block rather than one taken per
  point.

---

## Export format

### No header, and the layout is not recoverable from the file

The first line is already data. There is no label column and no field names, so
**nothing in the file states the block width, the block count, or what any parameter
row is.** The reading class supplies all of it, which is why it refuses anything whose
shape does not fit rather than reading it a different way.

### Block structure

One column **block** per sweep point, four columns wide:

```
│ par wl signal signal │ par wl signal signal │ …
│ ←── block 0 ───────→ │ ←── block 1 ───────→ │
   (one sweep point)
```

Contrast the AttoCube, whose otherwise identical block is preceded by a
`"Parameters Labels"` column and named by a header line — see
`dev/instruments/attocube.md`.

### Rectangular, with no padding of any kind

Both files are exactly `n_blocks × 4` columns wide on every row, and exactly
`n_pixels` rows tall. There is **no trailing field pad** and **no over-allocated
zero-filled block**, both of which the AttoCube export has. Nothing needs stripping.

### The parameter column is overlaid on the pixel rows

There is no separate parameter section. Row *i* of a block's first column is
parameter *i*; the same row of the block's second column is pixel *i*'s wavelength.
So the file has one row per spectrometer pixel — 1340 in both files — and the
parameter block occupies the first 30 of them. A file with fewer rows than the
parameter block is not something this format produces.

### The axis is repeated in every block

The second field of every block carries the same wavelength axis, identical across
all 61 and all 101 blocks respectively (0 mismatches). The redundancy is what
separates this format from a real-space image, whose width also divides by four often
enough — a 512-column frame is committed under `examples/data/exciton-diffusion/`.

### The signal column is written twice

The third and fourth fields of a block are byte-identical everywhere. Counts are
integers on a pedestal of roughly 600: the Doping file spans 594 to 33 847 and the Ez
file 589 to 17 722.

### The parameter rows

1-based, as the MATLAB snippet counts them. Only these carry a value; every other row
of the block is zero in both files.

| Row | Reading in the two files | What it is |
|---|---|---|
| 1, 2, 3 | 59 / 108 or 109 / 23, constant integers | stage x, y, z in piezo controller units |
| 4 | −3.00→+2.99 V; −5.00→+4.97 V | gate voltage 1 |
| 5 | small, noisy, both signs | the current at gate 1 — **unit unconfirmed** |
| 6 | −3.37→+3.35 V; +5.60→−5.58 V | gate voltage 2 |
| 7 | small, noisy, both signs | the current at gate 2 — **unit unconfirmed** |
| 8 | −4.92900; −4.78949 — *exactly* constant | reflection intensity in V, or a setpoint |
| 9 | exactly zero in both | gate voltage 3 |
| 10 | zero in both | unidentified |
| 11 | ≈21, fluctuating ~10% point to point | excitation power, **already in µW** |
| 12 | ≈−5×10⁻⁷, noisy | unidentified — the magnitude of a real gate current |
| 13 | ≈−0.019, noisy | unidentified — the snippet calls this the gate-3 current |
| 17 | exactly 1.0 | unidentified — both filenames say `exp1sec` |
| 30 | exactly 100000.0 | unidentified |

### The spectral axis

702.026 90 to 977.973 10 nm over 1340 pixels in both files, stepping 0.206 08 to
0.206 09 nm — uniform to the five decimals the exporter writes. Exported to five
decimal places throughout, so two files taken on the same grating position are
bit-identical on this axis, unlike the AttoCube's TRPL time axis.

### The filename carries what the file does not

`PL_Doping_1LWSe2_p20uW_exp1sec_250508_190421.csv` reads as
measurement / scan kind / sample / power / exposure / date / time. The power and
exposure it names agree with rows 11 and 17. **Nothing in the file states the
measurement type**, which is why `spectra_type=` is required.

---

## Hardware facts the format encodes

These are properties of the instrument, not of the file layout, but a column cannot be
read correctly without them.

**The stage rows are in piezo controller units.** Integers, three digits at most.
Not volts (unlike the AttoCube's `Scanner X`/`Scanner Y`, which carry drive voltage)
and not micrometres. Converting to a distance needs a per-stage calibration that no
file contains.

**Excitation power is already in microwatts.** Row 11 reads ≈21 in files whose names
say `p20uW`. The AttoCube writes a raw photodiode voltage needing a ×0.303×10⁶ scale;
this system does not.

**The two gates are driven at a fixed ratio of about 1.12.** Measured at both ends of
both sweeps: |row 6 / row 4| is 1.1226 and 1.1176 in the Doping file, 1.1211 and
1.1235 in the Ez file. Same magnitude in both, same sign as row 4 in the Doping file
and opposite in the Ez file. That is what a capacitance-matched dual-gate pair looks
like — one sweep holds the field at zero and moves the density, the other the reverse
— but **the capacitances themselves are not recorded anywhere**, so the ratio is a
measurement of what the acquisition program drove, not a derivation from the stack.

**Which gate is top and which is bottom varies between samples.** Stated by the user,
not derivable from any file. Nothing in this package assumes it; `gates=` declares it
per load.

---

## Measurements not yet represented here

Only the spectral sweep. Whether this setup also writes real-space images, TRPL, or
2-D spatial rasters, and in what shape, is unknown — no such file has been seen. A
raster arriving flattened would need `fast_sweep=`/`slow_sweep=` exactly as the
AttoCube's does, but that is an expectation rather than a finding.
