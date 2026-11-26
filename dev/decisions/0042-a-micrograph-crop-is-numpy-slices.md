# 0042 — A micrograph is cropped with one `(rows, columns)` pair of slices

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-11-26 |

## Context

Figures for publication show part of a micrograph, with the scale bar. Cropping
with `ax.set_xlim` / `ax.set_ylim` after `plot_micrograph` works, but
`ax.set_ylim(low, high)` turns the image upside down: `imshow` with
`origin="upper"` draws on an inverted y-axis, and limits given low-to-high
un-invert it. `plot_micrograph` needed a crop argument that cannot flip.

## Decision

1. **`crop=` takes a tuple of two slices, `(rows, columns)`**, written
   `crop=np.s_[r0:r1, c0:c1]`. This is the form `roi=` and `bg_region=` already
   take in `diffusion` — the same image-region concept spelled the same way.
2. **The array is sliced, and `imshow` is given an `extent`** placing the slice at
   its full-image pixel coordinates. Row 0 stays on top, ticks read the same
   numbers as on the full image, and a saved figure carries only the cropped
   pixels.
3. **Bounds are pixel indices of `img`**, resolved by `slice.indices`, so negative
   and open bounds mean what they mean in NumPy.
4. **A step other than 1 is refused**: a subsampled image has a different pixel
   size from the one the scale bar is built from. An empty selection is refused.

## Rejected

- **`xcrop=(c0, c1), ycrop=(r0, r1)`.** It names the axes, so it cannot be given in
  the wrong order — the case for it. But it is a second spelling of a region the
  package already spells as slices, and "y" is ambiguous on an image: up the plot,
  or down the rows. A swapped `crop=` is visible the moment the figure is drawn.
- **Setting axis limits instead of slicing.** Keeps the whole array in the
  artist, so a vector export can embed the full frame, and leaves
  `im.get_array()` disagreeing with what is shown.
- **Crop in µm.** The bounds would be rounded to pixels anyway, and the pixel size
  is nominal unless measured; pixel indices are exact.
- **Re-indexing the cropped image from 0.** Loses the link between the cropped
  figure's ticks and the full image's, which is how a crop is chosen.

## Consequences

- The row bound comes first. Someone thinking in `(x, y)` will get a wrong crop,
  visible immediately.
- `ax.get_xlim()` / `ax.get_ylim()` after a crop are in full-image pixels, not
  starting at 0.
