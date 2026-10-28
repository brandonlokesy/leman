# Olympus microscope (main lab) — export format facts

The bright-field microscope in the main lab, an Olympus, used for imaging samples
and devices. The group calls it "the microscope"; the loader is `Micrograph`. This
is the **record** for its JPEG export, established from the 29 files committed
under `examples/data/optical-microscope/` on 2026-10-27.

What belongs here: what the software writes and where. Why the loader responds to it
the way it does is `dev/decisions/0041-a-micrograph-reads-its-own-label.md`.

---

## Provenance and unknowns

| Files | What they are |
|---|---|
| `<M>x_example_<n>.jpg`, M ∈ {5, 10, 20, 50, 100} | default export, with the info bar: 2 / 4 / 3 / 8 / 7 files |
| `BW_example_{50,100}x.jpg`, `aperture_example_{50,100}x.jpg` | same microscope, exported without the info bar |
| `example_reject.png` | the **transfer stage**, a different setup; must be refused |

Still unknown: the acquisition software's name and version, and therefore whether
any of the geometry below survives an update. The magnification-to-scale table was
entered as **nominal** values; no stage-micrometer calibration is on record
(`dev/physics-conventions.md` §9).

## The file

- JPEG, 2448 × 2048 px (width × height), RGB, 8 bit. Every export seen, at every
  magnification, with or without the bar.
- No EXIF. The JPEG comment holds only `Intel(R) JPEG Library, version [2.0.14.46]`.
  **The magnification is recorded nowhere but in the label pixels.**
- The `BW_` files are RGB too, not single-channel.

## The info bar

All coordinates are 0-based `(row, column)` in the full frame.

| Element | Where | Notes |
|---|---|---|
| bar | rows 2002–2047 (46 rows), full width | **translucent** grey overlay: the image shows through |
| `X: … mm  Y: … mm   Sample: <name>` | left, text rows ≈2010–2046, starting at column 7 | white text; `X:` is at columns 7–52 in every file |
| scale bar | columns 1800–2162 (363 px), rows 2025–2034 | white; core rows 2026–2033 are solid, the outer two are softened by JPEG |
| label | starts at column 2201 or 2204, text rows ≈2010–2046 | `500 um`, `250 um`, `125 um`, `50 um`, `25 um` |

| Label | Magnification | Pixel size (µm / px) |
|---|---|---|
| 500 um | 5× | 500 / 363 = 1.377 |
| 250 um | 10× | 0.6887 |
| 125 um | 20× | 0.3444 |
| 50 um | 50× | 0.1377 |
| 25 um | 100× | 0.06887 |

The scale bar is 363 px at every magnification; only the label changes. The table
is exactly 6.887 µm / M, which is what nominal objective magnifications would give.
One camera arrangement that produces that constant is a 3.45 µm sensor pixel behind
a 0.5× adapter; that is an inference from the numbers, not a record of the
hardware.

### The overlay is translucent

The bar's grey varies from file to file (median 115–168 across the committed
exports) because it is blended with whatever lies under it. Consequences:

- There is **no reliable edge at row 2002.** It is sharp where the image above is
  dark and invisible where the image is close to the bar's own grey (`5x_example_1`,
  `50x_example_1`).
- The 46 rows hold sample, tinted and under text. Removing the bar loses them.
- The brightest non-text pixel measured inside the bar, by darkest channel, is 197
  (a white flake beneath it). Text is white in every channel.

### The sample name can collide with the scale bar

The sample name has no length limit. In `5x_example_1`
("… top gate after evaporation strip") it runs into the left end of the scale bar,
which then measures 366 px as a white run. A long enough name would reach the label
at column 2201; none of the committed files does.

## The transfer stage, for contrast

`example_reject.png`: PNG, 2848 × 2848 px; the bar sits mid-frame, not at the bottom;
text reads `X: 1.83   Y: 0.36` (no unit) and the label is `250um`, without a space.
Different software, different layout.
