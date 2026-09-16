# 0014 — An animation is driven by a sequence of frame indices

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-12 |
| **Audit** | C8 · (and it is what E18 and E20 were cleared to make room for) |

## Context

`animate_panels` could animate the first *N* frames and nothing else: `n_frames=N` took a
count, panels received that count, and `FuncAnimation` was handed `frames=N`, so every
index was a position from zero. A researcher with a 2091-point scan who wanted the twenty
frames around a feature had no way to ask for them, and rendering all 2091 to look at
twenty is what made long scans impractical to inspect or embed.

The request came from PR #16 (`dev/spectra-colour`, original commit `de45fb14`, Felicia
Iacob), which proposed `pos_start`/`pos_end` — a pair of `(x, y)` scanner positions
resolved to the nearest frame. That framing surfaced the real questions: what a frame
index *means* across panels of different kinds, and what should happen when panels
disagree about how many frames they have.

Two facts constrained the answer. `ACImgSweep` carries no coordinates at all
— it is a folder of images with `n_frames` and `load_frame`, so a coordinate cannot
address it. And the flat order of a declared nest is fast-inside-slow, so a contiguous
index range between two grid points is a snake across rows, not a region.

## Decision

`animate_panels(panels, *, frames=None)` takes **a sequence of frame indices**, in order.
`None` means every frame.

Indices are the panels' own. `init_artists(ax, frames)` receives the whole selection so a
panel can fix its axes limits over exactly the frames that will be shown, and `update` is
handed each index unchanged. A panel keeps its arrays at full length and reads them at the
index it is given; it never records where a selection began and never offsets an index.

One parameter therefore expresses a window (`range(500, 520)`), a stride
(`range(0, 2091, 10)`), a single frame (`[7]`), and an arbitrary order.

**Panels must agree on `n_frames` when no selection is given**, and a mismatch raises. With
an explicit selection there is nothing to guess, so panels of differing length are allowed
provided every index is valid for all of them.

The frame counter reports the frame's own index: `"Frame 203/2091"`, not `"Frame 3/40"`.
`{position}` and `{n_shown}` are available for the other reading.

## Rejected

**`pos_start`/`pos_end` as `(x, y)` scanner positions.** The original proposal. It serves
one case — a line scan, where one coordinate is constant, so it is a 1-D lookup in a 2-D
costume — and misleads on two others: a nested raster, where the frames between two grid
points are not a region, and an image sequence, which has no coordinates to match against.
It also has to *find* the coordinates, which it did by scanning the panel list for the
first object exposing `scanner_x`, silently picking one panel's frame of reference for the
whole figure. Selecting by coordinate remains worth having, but as a helper that takes the
scan as an argument, so the caller writes down which scan the coordinates belong to.

**A bespoke nearest-index search.** The proposal computed `np.hypot` + `argmin` in
`plotting`. `loaders` already resolves a coordinate to an index — `nearest_index`,
`_index_for_value`, `_sweep_selector`, `_lookup_axis` — with settled policies for a
coordinate that falls between points and one that matches several. A second implementation
inherits none of them and cannot warn about anything.

**Window-relative indices in `update`.** The proposal passed `frame_start` to
`init_artists` and window-relative frames to `update`, so each panel stored the offset and
added it back. Every panel then carries the same arithmetic, and getting it wrong is
silent: the animation plays real frames in a plausible order while the shared title names
different ones. Native indices delete the bookkeeping from all three panels.

**Keeping `n_frames=` alongside `frames=`.** `n_frames=N` is `frames=range(N)` spelled as a
count, so keeping both means either a precedence rule or a parameter whose only job is to
be refused in combination. It is removed rather than deprecated: pre-adoption, and there
are no in-repo callers.

**Taking the minimum across panels, as before.** This is what made the AttoCube's extra
white-light frame invisible — a figure built from an image sequence and a sweep that do not
correspond rendered happily and looked right. Silence was the defect, not the truncation.

**Handling the white-light off-by-one inside `animate_panels`.** The engine sees a row of
`ImageSequencePanel`s and cannot tell which is the white light, so it would have to guess
that "whichever panel is one longer" is the extra one. `animate_wl_pl_spectra` knows,
because its argument is named `wl`.

**Negative indices.** They would work by accident on array-backed panels and silently
reorder an animation when mixed with positives. `range(n - 10, n)` says the same thing and
cannot be misread.

## Consequences

`animate_panels`' window group is **keyword-only**. `n_frames=` occupied that positional
slot and took a count, so a stale `animate_panels(panels, 40)` would otherwise be read as a
different selection instead of failing.

`AnimationPanel.init_artists` takes `frames` rather than a count — a breaking change to the
protocol, and the feature itself. All three panels lose their truncation:
`SpectrumLinePanel` keeps its full arrays and computes limits over the shown columns;
`ImageSequencePanel` and `DiffusionCloudPanel` draw `frames[0]` rather than frame 0.

A figure whose panels disagree now raises where it used to render. That is the point, but
it will surface in any notebook that was relying on the silent truncation.

`animate_wl_pl_spectra` drops the AttoCube's trailing white-light frame when `wl` is
**exactly one** longer than the shortest other panel, warning with both counts. Any other
disagreement falls through to the engine's refusal. It has to act rather than advise
because it constructs its scans internally from paths, so a caller who passes paths has no
place to insert a fix.

Selecting frames by physical coordinate is still owed, as
`plotting.frame_window(scan, start, end, axis=)` over the loaders' existing resolvers.

## Load-bearing choices

**The counter reports the native index.** A window-relative counter would make two clips of
one scan indistinguishable and a still lifted from a GIF impossible to trace back. If a
caller wants the other reading, `frame_count_fmt="Frame {position}/{n_shown}"` gives it.

**An explicit selection relaxes the agreement rule.** Refusing is about the default, where
the engine would otherwise pick for the caller. If that turns out to hide something, the
place to tighten is `_resolve_frames`, which already validates every index against the
shortest panel.
