# 0039 — An unnamed parameter row beats a guessed one, and one signal beats a fictional pair

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-26 |

## Context

The BigTable spectral export carries **no labels anywhere**. Its parameter column
is positional, so a loader has to supply the names. The only description of what
each row is comes from a MATLAB snippet written by a colleague who uses the
setup. That snippet cannot be cited, and it contradicts itself: it maps row 9 to
both `vg3` and `lambda` with the same trailing comment, it calls row 13 a gate
current in amperes where the row reads −0.019 (19 mA through a gate dielectric),
and it never mentions row 12, which reads ≈10⁻⁶ and is the magnitude a real gate
current would have.

Two more things about the export force a choice. Columns 3 and 4 of every block
are byte-identical in all 217 080 pairs across the two committed files, and
nothing says what the second is for. And the export is a bare numeric grid, so
its shape is the only evidence of what it is.

Format facts and what each row was measured to be: `dev/instruments/big-table.md`.

## Decision

1. **Ten rows are named; five keep their position as their name.** Rows 1–9 and
   11 get names. Rows 10, 12, 13, 17 and 30 become `Row 10`, `Row 12`, and so
   on. An unnamed row is still in `parameters`, still readable, and still usable
   as a `sweep=` axis, so not naming it costs nothing — while a wrong name would
   reach every axis label and every archive and be trusted there. This is
   0006's rule applied to a row rather than an axis: mislabelling is worse than
   not labelling.

2. **The gate-current rows carry scale 1.0 and an empty unit string.** That is a
   statement of ignorance, not a claim of dimensionlessness. A caller who knows
   sets both through `curated_scales=` and `curated_units=`.

3. **One signal, and no `roi=`.** `spectra` is the third field of each block. The
   fourth is compared against it and a disagreement **warns**, naming the first
   pixel and sweep point that differ.

4. **HDF5 refuses in both directions**, with one shared message. The archive
   records which axis a file holds but not which instrument wrote it, so a saved
   BigTable sweep would read back as an `AttoCubeSpectralSweep`.

5. **A grid whose blocks do not share one axis is refused.** In this export the
   second field of every block holds the same wavelength axis. That is the check
   that separates a sweep from a real-space image, and it is also the only
   verification the axis gets.

## Rejected

**Taking the MATLAB snippet's names as given.** The fast option. It would write
`Gate Current 3` onto a row reading 19 mA, and `Gate Voltage 3` onto a row the
same list also calls a wavelength — into every figure legend and every saved
file, where nothing downstream could question it.

**Guessing that rows 12 and 13 are swapped**, on the grounds that row 12's
magnitude fits a gate current and row 13's does not. It is the most likely single
explanation and there is no evidence for it. A plausible guess recorded as a name
is indistinguishable from a fact six months later.

**Leaving the unidentified rows out of `parameters` entirely.** Then a
researcher who *does* know what row 13 is cannot reach it without editing the
package. Naming it positionally costs nothing and keeps it in reach.

**Keeping `roi=` and treating column 4 as a second region of interest**, by
analogy with the AttoCube's `ExpROI2`. The analogy is the only argument for it:
the AttoCube's second region is *blank* in every ordinary measurement, whereas
this one is a copy. An argument that admits both readings is not a reason to
build the API around one.

**Dropping column 4 silently.** Then the one observation that would settle what
it is — a file where the two disagree — would never be seen. It warns instead.

**Refusing on a disagreement rather than warning.** The third column is what a
researcher wants either way, and the file may be perfectly good.

**Storing a BigTable scan in the existing archive format**, writing the single
signal into both region slots. It round-trips today and reads back as the wrong
class, which is the problem 0038 exists to avoid.

**Separating a sweep from an image by column count alone.** A real-space image's
width divides by four often enough — a 512-column frame is committed under
`examples/data/exciton-diffusion/` — so the count would have let one through to
be read as a 128-block sweep.

**Naming the class after the export format** (`HeaderlessBlockSpectralSweep`)
rather than the setup. It survives the setup being renamed, and reads badly in
every notebook that will use it.

## Consequences

- Naming a row later is a one-line edit to `_BIGTABLE_ROWS`, and nothing that
  reads `Row 13` today was told it was anything else.
- `_BIGTABLE_N_PARAM_ROWS` is 30 because nothing past row 30 carries a value in
  either committed file. It is not declared by the format, so a value below it
  **warns** rather than being dropped.
- The parameter column is overlaid on the pixel rows, so a file with fewer rows
  than the parameter block leaves the axis padded with zeros — finite, so the
  pixel mask keeps them, and then `hc/λ` divides by zero. `_validate_payload`
  refuses a non-increasing axis, which covers that and any other misread stride.
- Reflectance on this instrument will work when a reference spectrum turns up:
  `reference=` and `contrast=` are inherited and untouched. Nothing has been
  tested against a BigTable reflectance file, because none exists here.

## Load-bearing choices

- **The repeated-axis check is a refusal, not a warning.** If it is ever
  downgraded, a real-space image becomes loadable as a sweep and every number
  after that point is meaningless.
- **`_BIGTABLE_ROWS` is keyed 1-based**, matching the MATLAB, the instrument
  record and every note anyone will write about this format. `_row_labels` is the
  one place that converts, and it says so.
