# 0020 — `plot_image` carries the laser annotation, and returns the circle it drew

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-18 |

## Context

A grid of real-space PL frames — several panels in one `plt.subplots` figure — could not
carry the 1/e² laser-spot overlay. Two functions could draw an image, and neither served
the case:

- `plot_image` accepts an `ax=`, so it composes into a grid, but had no annotation.
- `_ACImg.show_image` annotates, but creates its own figure unconditionally, so
  it cannot draw into a caller's axes.

Callers wanting both therefore reached into the private `plotting._draw_laser_circle`.
A private helper used from notebooks is a signature nobody has agreed to, and it made
the overlay's styling a copy-paste decision at each call site rather than a property of
the package.

## Decision

1. `plot_image` gains `laser_annotation` and `laser_ref`. `laser_annotation` is the only
   switch; `laser_ref` selects *which* reference and does not enable the overlay. An
   explicit `laser_ref` wins, then the image object's own `laser_ref` attribute, then
   nothing — the resolution rule `plot_diffusion_cloud` already used, so the package
   spells this concept one way rather than two.
2. Drawing goes through the existing shared circle helper, so a static panel and an
   animation of the same scan carry identical annotations, dashes included.
3. `plot_image` returns the circle it drew as a fourth element, `None` when it drew
   none.

## Rejected

**An `ax=` parameter on `_ACImg.show_image`.** The obvious fix, and the reason
this record exists. `show_image` is a convenience viewer: it owns its whole figure, so
nothing composes on top of it and its internals cost nothing later. `ax=` converts it
into a composable plotting function that happens to live in `loaders`, and
composability attracts the parameters composition needs — a colormap, shared colour
limits, colorbar suppression. The end state is a second image-plotting function inside
the loader module, free to drift from the first. That drift is not hypothetical: the
package already has two entry points for one diffusion-cloud concept whose default
thresholds disagree, so the same image analysed statically and animated gives different
contours. One axes-accepting image function, in `plotting`, is the whole point.

**A single `laser_ref` argument, truthiness deciding whether to draw.** One parameter
instead of two, and it cannot silently do nothing when handed a reference. But it
cannot express "annotate using whatever reference this object already carries" without
a sentinel value, and it would be a third spelling of a switch the package already
spells `laser_annotation` in two public functions and one loader method. Consistency of
vocabulary across the surface beat saving one keyword. The cost is accepted and
documented: `laser_ref` alone draws nothing, and a test pins that so it is not later
"fixed" into an implicit enable.

**Keeping the three-element return.** No caller would have broken. Rejected because a
caller who wants the circle in a different colour or weight then has to recover it from
the axes' patch list — re-deriving which patch is the annotation — and the pressure to
add `laser_color`, `laser_lw` and `laser_linestyle` comes straight back. Those style
parameters are the standing counter-example to *parameters earn their place*; returning
the artist is what makes refusing them honest rather than merely restrictive.

**Style parameters for the circle.** Not added, for the reason above.

## Consequences

- `plot_image` returns four values. Every caller unpacking three must add a slot; at the
  time of the change the only such callers were in the test suite.
- A grid of annotated panels is written with the public API alone. The private circle
  helper stays private.
- A bare 2-D array remains a valid `image` argument. It carries no reference, so
  `laser_annotation=True` alone draws nothing for one — documented rather than raising,
  because the array input is deliberate and an overlay is optional.
- `show_image` keeps its contract: no `ax=`, owns its figure, stays a viewer. Adding one
  is now a decision to reverse rather than an omission to fill.

## Load-bearing choices

The gating rule is the part most worth revisiting. `laser_annotation` gating a
`laser_ref` that is otherwise inert is defensible only as consistency with the existing
call sites; if those are ever reworked, all of them should change together rather than
this one drifting to a fourth spelling.
