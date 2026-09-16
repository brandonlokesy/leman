# 0033 — A TRPL directory converts only when it is named

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-21 |
| **Audit** | — |

## Context

`converters.convert_path` walks a folder, or a tree with `recursive=True`, and
routes each file by content. Images and spectral exports are per-file: one CSV in,
one output out. **A TRPL sweep is not.** One TCSPC decay is one file and the sweep
is the whole directory, so converting it means deciding which files in that
directory belong to one measurement.

`ACTRPLSweep._decode_dir` already answers most of that, and answers it well:
an IRF reference is excluded by *name* (its `[Par, Wavelength, Exp]` header is
identical to a real decay, so content cannot settle it), a spectral-header file in
a TRPL folder is read as the parameter-table companion rather than a sweep, and
anything else is skipped.

What it cannot answer is how many measurements are present.
`examples/data/TRPL/right_spots` holds `right1_*` and `right2_*`. With no `prefix`
every temporal file is taken as a point of one sweep, and the two merge. This is
detected — both claim the same `_iter_N` indices, so `_order_by_iter` warns (A12) —
but a warning is not a refusal, and the sweep loads.

That is survivable when a person typed the load and is reading the output. It is
not survivable in an unattended `--recursive` run over a tree, which is the case
this module exists for: the warning scrolls past among dozens of `wrote …` lines
and a wrong archive is left on disk looking exactly like a right one.

The measured example: the folder's data is 0.43 MB of decays plus an 11.14 MB
parameter-table companion, and the archive is 0.070 MB. The prize is real, which
is why the answer is not "skip TRPL".

## Decision

1. A directory holding temporal CSVs converts to one `.h5` **only when it is the
   directory the caller named** — `convert_path(path)` where `folder == path`.
2. A directory reached by `recursive=True` is **not** converted. It is recorded in
   `ConversionReport.skipped` under the kind `"trpl_directory"`.
3. `prefix=` narrows which files form the sweep, and is what makes a
   multi-measurement folder convertible: name it, pass the prefix, convert twice.
4. The command line **names** a deferred directory on stdout rather than leaving it
   inside the skipped count, because it is the one skip that asks the caller to do
   something.
5. In a folder that is a TRPL sweep, a spectral CSV is that sweep's companion and
   is not converted as a sweep of its own. Images in such a folder still convert —
   a laser-spot or white-light reference beside a sweep is an ordinary frame.

## Rejected

**Convert every TRPL directory, recursion included.** Fewest commands, and the
loader does warn. Rejected on the `right_spots` case: the output is a single
archive holding two interleaved measurements, with no marker in the file saying
so. Nothing downstream can detect it, and the warning that would have said so is
long gone by the time anyone opens the archive. A conversion tool that can write a
silently wrong archive from an unattended run is worse than one that asks.

**Require `prefix=` for every TRPL directory.** Removes the ambiguity outright.
Rejected because it makes the common case — one measurement, one folder, which is
what `examples/data/TRPL` is — into an error, and the prefix it would demand is
derivable from nothing the caller knows better than the loader does.

**Refuse (raise) when a TRPL directory is found without a prefix.** Honest, and it
cannot write a wrong file. Rejected because a tree containing one TRPL folder would
then fail the whole run, or fill `errors` with something that is not an error: the
folder is fine, it simply needs its own command. `skipped` already means "not
converted, here is why", which is exactly the situation.

**Group temporal files by an inferred prefix**, stripping `_<timestamp>_iter_N`.
Would convert `right_spots` into two correct archives with no declaration.
Rejected as the same class of guess the package refuses elsewhere — *don't
auto-detect the sweep axis*, and *don't repair a gap or pick a winner among
duplicate `_iter_N` indices*. Filename shape is a convention the exporter is not
contractually held to, and this package has already watched its padding width vary
between two committed folders (A7).

## Consequences

- `tmdc-convert examples/data/TRPL` writes one archive. `tmdc-convert <tree>
  --recursive` writes every image and spectral output in the tree and prints
  `deferred …` for each TRPL folder, exiting 0 — nothing failed.
- A multi-measurement folder takes one command per measurement, each with its own
  `--prefix`. The archive is then named after the prefix, so the two do not collide.
- `ConversionReport.skipped` now holds directories as well as files. Its values are
  looked up in `converters._SKIP_REASON`, which extends `loaders._CSV_KIND_REASON`
  with the one kind that is not a CSV kind.
- The rule is positional, so `convert_path(folder)` and
  `convert_path(folder.parent, recursive=True)` deliberately do different things to
  the same folder. That is the point, and both tests exist.

## Load-bearing choices

That a warning is not enough. The argument rests on the failure being **invisible
in the output**: a merged sweep is a well-formed archive with a plausible point
count. If `_order_by_iter` were ever changed to *refuse* colliding indices rather
than warn, the merge could no longer happen silently and the guard here would be
worth re-examining — the deferral exists to cover a hole in that helper's
strictness, not because recursion is inherently unsafe.
