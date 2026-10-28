# 0041 — A micrograph reads its magnification from its own label, and an override restates the wrong label

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-10-28 |

## Context

The main-lab Olympus microscope saves a JPEG whose only record of the magnification
is the text of the scale-bar label drawn into the info bar: no EXIF, no sidecar
(`dev/instruments/olympus-microscope.md`). A scale bar on a figure needs the pixel
size, so either the image is read or the user states it every time. Three further
facts shape the choice:

- Some exports have no bar at all (`BW_`, `aperture_`), and must still load.
- People sometimes leave the software set to the wrong objective, so the label lies
  and the true magnification is known only to them.
- The magnification table is nominal, so a careful user may measure a pixel size
  against a structure of known size and want that instead.

## Decision

1. **The label is matched against stored masks.** Text pixels (darkest channel
   ≥ 230) in a fixed box are compared with one stored mask per label by Jaccard
   index; the best match is accepted at ≥ 0.9, otherwise the label is unreadable.
   On the committed examples the right label scores ≥ 0.950 and the best wrong one
   ≤ 0.760, measured on files the masks were not cut from.
2. **The bar is recognised by its opening `X:`**, matched the same way at ≥ 0.8,
   not by the scale bar. The scale bar is then checked only for "white across its
   footprint".
3. **An unreadable label is refused, not guessed.** The user passes
   `magnification=` or `pixel_size=`.
4. **`magnification=` must agree with a readable label.** To correct a wrong label
   the user passes `label_reads=` — the value the label shows — and the loader
   refuses unless that is what the label does show.
5. **`pixel_size=` is a measured calibration.** It replaces the nominal value, is
   not compared with the label, and cannot be combined with `magnification=`.
   With it, `magnification` is `None` and the label's value stays in
   `label_magnification`; `repr` shows the equivalent magnification at the
   nominal calibration (2500 µm / 363 px ÷ pixel size) as a reading aid only.
6. **The masks ship inside the package** (`src/leman/data/micrograph_masks.npz`,
   3.4 kB, via `package-data`), built by `dev/build_micrograph_masks.py` from the
   `*_example_1` files only.

## Rejected

- **OCR (Tesseract or similar).** An external binary to install on every lab machine
  and on three CI systems, for five possible strings drawn in one fixed font.
- **Reading the label's width.** Digits are equal-width, so `500`, `250` and `125`
  are indistinguishable.
- **Detecting the bar by an edge at row 2002.** The overlay is translucent; the edge
  vanishes wherever the image is close to the bar's grey (two committed files).
- **Detecting the bar by the scale bar's exact 363 px extent** — white at 1800–2162
  and dark at 2163. A long sample name joins the bar's white (`5x_example_1`
  measures 366 px), and one that reaches column 2163 would make a bar image read as
  bar-less.
- **`override_magnification=True`.** A switch costs nothing to add, so it gets added
  to make an error go away and then copied with the notebook cell, switching the
  check off for every image afterwards — including ones whose label was right.
  `label_reads=` only ever passes on an image whose label shows that exact value.
- **Overriding a wrong label with `pixel_size=`.** It made the user look up a number
  that follows from the magnification they already know; the friction bought
  nothing. `pixel_size=` stays, for measured values only.
- **Letting `pixel_size=` win over `magnification=` when both are given.** The two
  can contradict each other; which one was meant cannot be known.
- **Keeping the label's value as `magnification` when `pixel_size=` is given.** A
  measured pixel size is often the remedy for a wrong label, and since
  `magnification=` cannot accompany `pixel_size=`, nothing could correct the
  stored value: the object would carry a right pixel size and a wrong
  magnification.
- **Allowing `magnification=` + `label_reads=` + `pixel_size=` together** to record
  the objective alongside a measured pixel size. It reopens the two-sources
  ambiguity refused above, for a value nothing downstream uses.
- **An `equivalent_magnification` attribute.** It rests on the nominal table, which
  has no calibration record; as an attribute it would be computed with.
- **Installing `examples/data` with the package to build the masks at run time.**
  625 MB downloaded for an 8 kB need, and an edited example would change how the
  loader behaves.
- **A threshold at the midpoint of the label gap (0.855).** Rejecting a right label
  costs one argument; accepting a wrong one scales every length by 2× or more,
  silently. The threshold leans towards refusing.

## Consequences

- A new label, a software update that moves the bar, or a font change makes every
  affected image refuse with a message naming `magnification=`. Fix by adding
  examples and re-running the build script; the test comparing the shipped masks to
  a rebuild fails until that is done.
- An image of the right size with no bar, from any camera, loads as a bar-less
  micrograph: nothing in the file says where it came from.
- The pixel size is nominal unless `pixel_size=` is given; `repr` says which.

## Load-bearing choices

- The two thresholds (0.9 label, 0.8 `X:`) and the text cut (230) rest on 24
  labelled files. Only 2 are at 5× and 3 at 20×. More exports at those
  magnifications are the cheapest way to tighten them.
- The label box opens at column 2190; glyphs start at 2201. A sample name ending
  between 2190 and 2201 makes the label unreadable even though it is intact.
