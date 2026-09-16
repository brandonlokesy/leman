# Converters

Turn AttoCube CSV exports into formats that are smaller and that other tools can
open. Images become TIFF; spectral and TRPL sweeps become HDF5.

An export leaves the acquisition software as text. A frame is a bare grid of
comma-separated numbers with no header; a spectral sweep is a wide grid of
`[Par, Wavelength, ExpROI1, ExpROI2]` blocks; a TRPL sweep is a whole *directory*,
one decay per file. Text costs roughly six bytes per count, so every one of them is
far larger than the same data in a binary format, and no viewer will open a frame.

Nothing here interprets a measurement. No correction is applied, no axis is
derived, and every question a loader already answers is left to it. The output
holds the same numbers the export held.

Measured on the committed example data:

| Source | As exported | Converted |
|---|---|---|
| `position-scan/PL`, 46 frames | 96.49 MB | 24.12 MB (one stack) |
| `stark-shift` spectral sweep | 4.59 MB | 0.142 MB |
| `TRPL`, 3 decays + companion | 11.57 MB | 0.070 MB |

## Quick start

### One image frame

```python
from leman import converters

converters.convert_image_csv_to_tiff("scan/raw/frame_iter_0.csv")
# -> scan/converted/frame_iter_0.tif
```

### A folder of frames, as one stack

A gate or position sweep of frames is a sequence, so it can go into a single
multi-page TIFF — the object ImageJ opens most naturally.

```python
converters.convert_image_dir_to_tiff_stack("scan/raw", prefix="PL-dual-gate-sweep_")
# -> scan/converted/PL-dual-gate-sweep_stack.tif
```

Pages are ordered by the `_iter_N` suffix, so `iter_10` follows `iter_2`. Without
`--stack` (or this function), each frame becomes its own `.tif` instead.

### A spectral sweep

```python
converters.convert_spectral_csv_to_hdf5("scan/raw/sweep.csv", "PL")
# -> scan/converted/sweep.h5
```

The measurement type is required, and is the only measurement fact asked for. A raw
export records none, and it cannot be inferred from the numbers.

### A TRPL sweep

A TRPL sweep is a whole **directory** — one decay per file — so converting it
collapses many files into one archive.

```python
converters.convert_trpl_dir_to_hdf5("trpl-sweep", prefix="TRPL_")
# -> converted/trpl.h5
```

No type argument: `ACTRPLSweep` defaults it to `"TRPL"`, the class name
having already declared the modality.

### A file, a folder, or a whole tree

```python
report = converters.convert_path("data", recursive=True, spectra_type="PL")
outputs, skipped, errors = report
```

Failures are collected rather than raised, so one unreadable frame does not abandon
a sweep. Call a single-purpose converter directly to get the exception instead.

## Reading an archive back

The HDF5 side owns no format. It loads with `ACSpectralSweep` or
`ACTRPLSweep` and writes with [`write_sweep`](hdf5.md), so an archive reopens
by handing the `.h5` straight back to its loader.

Only the measurement *type* is stored at conversion time. What was swept, which
channel reached which gate, and the device stack are declared when you **open** the
archive — the loader takes each from its argument and only falls back to what the
file recorded:

```python
scan = ACSpectralSweep(
    "scan/converted/sweep.h5", spectra_type="PL",
    sweep="electric_field",
    gates={"top": "V_A", "bottom": "V_B"},
    geometry=geom,
)
```

So an archive is declared against exactly as the CSV was, and nothing is lost by
converting before you have decided how to analyse.

## The command

Installing the package provides `tmdc-convert`. An editable install needs
re-installing once for the command to appear:

```
conda activate viz-sci-plot
pip install -e ".[docs,test,colormaps]"
```

```
tmdc-convert PATH [--out ROOT | --from-raw | --beside]
             [--spectra-type {A,PL,R,RC,T,TRPL}] [--prefix P] [--stack]
             [--recursive] [--dtype auto|uint16|uint32|float32]
             [--compression gzip|none] [--overwrite]
```

`PATH` is a CSV file or a directory of them. The command returns a non-zero exit
status if anything failed, and a batch continues past a failure rather than
stopping at it.

### Every flag

| Flag | Takes | Default | What it does |
|---|---|---|---|
| `--out` | a path | off | Output root. The tree under `PATH` is copied beneath it. A path with a suffix is taken as one filename instead. |
| `--beside` | — | off | Write each output into the folder its source came from, with no `converted/` level. |
| `--from-raw` | — | off | Search *upward* for the nearest `raw/` folder and place output relative to that. |
| `--spectra-type` | `A` `PL` `R` `RC` `T` `TRPL` | none | Measurement type for spectral exports. Required to convert one; images and TRPL do not need it. |
| `--prefix` | a string | none | Filename prefix to select. Narrows which files form a TRPL sweep, and names a stack. |
| `--stack` | — | off | One multi-page TIFF per folder, in `_iter_N` order, instead of one file per frame. |
| `--recursive` | — | off | Descend into subdirectories. |
| `--dtype` | `auto` `uint16` `uint32` `float32` | `auto` | Pixel type for images. `auto` keeps integer counts exactly. |
| `--compression` | a filter name, or `none` | `gzip` | HDF5 compression. |
| `--overwrite` | — | off | Replace existing output. Off means an existing file raises. |

`--out`, `--beside` and `--from-raw` are **mutually exclusive** — all three answer
where output goes, so giving two names both in the refusal.

### How a run works

Three steps, in order:

1. **Collect.** Every `*.csv` under `PATH`, or under `PATH` and its subfolders with
   `--recursive`. `--prefix` narrows this. Non-CSV files are ignored.
2. **Classify by content, per folder.** Not by filename. A numeric grid of three
   rows or more is an image; a block header decides spectral or temporal; a two-row
   grid is a single spectrum and is skipped. Images become TIFF, spectral exports
   become HDF5, and a folder holding temporal files is one TRPL sweep.
3. **Place.** A destination is worked out once for the whole run, then each output
   keeps its source's position within it.

## Where output lands

Output goes into a **`converted/`** folder, created if it is not already there.
Which `converted/` depends on **the folder you name in the command** — nothing is
searched for unless you ask, so you can work the answer out from what you typed.

!!! note "`converted/`, not `processed/`"
    A conversion is not an analysis. `processed/` is for what an analysis pulls
    *out* of the raw data — fitted positions, integrated intensities, diffusion
    lengths — which took a decision to produce and may not be reproducible.
    Everything in `converted/` is the raw data in a better container, so it can be
    deleted and regenerated at any time. Which folder a file is in tells you which
    kind it is.

The rule, in order of precedence:

| Situation | Where output goes |
|---|---|
| `--beside` given | the source's own folder, no `converted/` at all |
| `--out ROOT` given | `ROOT`, with the tree under `PATH` copied beneath it |
| `--from-raw` given | the nearest `raw/` folder's sibling `converted/`, tree copied beneath |
| the folder you named is `raw` | its sibling `converted/`, tree copied beneath |
| anything else | a `converted/` inside each source folder |

Whichever applies, a source's **position** is preserved. That is what lets two spot
folders hold the same `laser_ref.csv` without one overwriting the other.

## Worked examples

Every tree below is real output. The starting point:

```
EXP/
`-- raw/
    |-- spot01/
    |   |-- 01-PL-Vbot/
    |   |   |-- PL_10uW_iter_0.csv
    |   |   `-- PL_1uW_iter_0.csv
    |   |-- 02-diffusion/
    |   |   |-- pl_iter_0.csv
    |   |   |-- pl_iter_1.csv
    |   |   `-- pl_iter_2.csv
    |   `-- ref/
    |       |-- laser_ref.csv
    |       `-- wl.csv
    `-- spot02/
        |-- 01-PL-Vbot/
        |   `-- PL_1uW_iter_0.csv
        `-- ref/
            `-- laser_ref.csv
```

Sweeps in `01-PL-Vbot`, a real-space frame sequence in `02-diffusion`, and
reference frames in `ref`.

### A. Point at `raw/` — the usual case

```
tmdc-convert EXP/raw --recursive --spectra-type PL
```

```
EXP/
|-- converted/
|   |-- spot01/
|   |   |-- 01-PL-Vbot/
|   |   |   |-- PL_10uW_iter_0.h5     <-- new
|   |   |   `-- PL_1uW_iter_0.h5      <-- new
|   |   |-- 02-diffusion/
|   |   |   |-- pl_iter_0.tif         <-- new
|   |   |   |-- pl_iter_1.tif         <-- new
|   |   |   `-- pl_iter_2.tif         <-- new
|   |   `-- ref/
|   |       |-- laser_ref.tif         <-- new
|   |       `-- wl.tif                <-- new
|   `-- spot02/
|       |-- 01-PL-Vbot/
|       |   `-- PL_1uW_iter_0.h5      <-- new
|       `-- ref/
|           `-- laser_ref.tif         <-- new
`-- raw/
    ... unchanged ...
```

`9 file(s) written, 0 error(s).` The folder named is `raw`, so its sibling
`converted/` is used and the spot and measurement folders are copied underneath.
`raw/` is untouched. Sweeps became `.h5` and frames became `.tif` in the same run.

### B. Point at one measurement folder

```
tmdc-convert EXP/raw/spot01/01-PL-Vbot --spectra-type PL
```

```
spot01/
|-- 01-PL-Vbot/
|   |-- converted/
|   |   |-- PL_10uW_iter_0.h5         <-- new
|   |   `-- PL_1uW_iter_0.h5          <-- new
|   |-- PL_10uW_iter_0.csv
|   `-- PL_1uW_iter_0.csv
|-- 02-diffusion/
`-- ref/
```

The folder named is `01-PL-Vbot`, not `raw`, so output lands beside the files —
**inside `raw/`**. Nothing searched upward, which is deliberate: the destination is
readable off the command. If that is not what you wanted, see C.

### C. The same folder, with `--from-raw`

```
tmdc-convert EXP/raw/spot01/01-PL-Vbot --spectra-type PL --from-raw
```

```
EXP/
|-- converted/
|   `-- spot01/
|       `-- 01-PL-Vbot/
|           |-- PL_10uW_iter_0.h5     <-- new
|           `-- PL_1uW_iter_0.h5      <-- new
`-- raw/
    ... unchanged ...
```

Now the command walks up from `01-PL-Vbot` until it finds `raw`, and places output
relative to that — so this one measurement lands in exactly the slot it would have
had in run A. Convert one folder today and the rest next week, and they stack up in
the same tree.

!!! warning "Why `--from-raw` is not the default"
    It reads folders you did not name. If a folder called `raw` sits high up an
    unrelated path — `X:/Brandon/raw/01_Projects/…` — then everything beneath it
    anchors there, and you could not tell from the command you typed. Check your
    path, then use it. If no `raw` is found it warns and falls back to the default.

### D. One file, with `--beside`

```
tmdc-convert EXP/raw/spot01/01-PL-Vbot/PL_1uW_iter_0.csv --spectra-type PL --beside
```

```
01-PL-Vbot/
|-- PL_10uW_iter_0.csv
|-- PL_1uW_iter_0.csv
`-- PL_1uW_iter_0.h5                  <-- new
```

No `converted/` folder at all — the archive sits next to its CSV. This is for
targeted conversions, and it is how the committed `examples/data` archives are laid
out.

`--beside` is shorthand: `--out EXP/raw/spot01/01-PL-Vbot` does the identical
thing. Prefer `--beside` for a one-off, because a mistyped path is still a valid
destination and the output would land somewhere else without complaint, whereas a
flag cannot be mistyped that way.

### E. A whole tree, with `--beside`

```
tmdc-convert EXP/raw --recursive --spectra-type PL --beside
```

```
raw/
|-- spot01/
|   |-- 01-PL-Vbot/
|   |   |-- PL_10uW_iter_0.csv
|   |   |-- PL_10uW_iter_0.h5         <-- new
|   |   |-- PL_1uW_iter_0.csv
|   |   `-- PL_1uW_iter_0.h5          <-- new
|   |-- 02-diffusion/
|   |   |-- pl_iter_0.csv
|   |   |-- pl_iter_0.tif             <-- new
|   |   |-- pl_iter_1.csv
|   |   |-- pl_iter_1.tif             <-- new
|   |   |-- pl_iter_2.csv
|   |   `-- pl_iter_2.tif             <-- new
|   `-- ref/
|       |-- laser_ref.csv
|       |-- laser_ref.tif             <-- new
|       |-- wl.csv
|       `-- wl.tif                    <-- new
`-- spot02/
    ...
```

Every output lands with its own source and no `converted/` appears anywhere. This
converts a tree in place. Note it writes **inside `raw/`**, which the default goes
out of its way to avoid — the flag does not warn, because you asked for it.

### F. A frame sequence as one stack

```
tmdc-convert EXP/raw/spot01/02-diffusion --stack
```

```
02-diffusion/
|-- converted/
|   `-- 02-diffusion_stack.tif        <-- new
|-- pl_iter_0.csv
|-- pl_iter_1.csv
`-- pl_iter_2.csv
```

One multi-page TIFF instead of three files, named after the folder, with pages in
`_iter_N` order. `--prefix pl_iter_` would name it `pl_stack.tif` and select only
those frames. No `--spectra-type` was needed: nothing here is a spectral export.

## Walking a measurement tree

`--recursive` picks the converter per file, so a benchmarking run converts in one
command. Empty folders and non-CSV files (a stray `.stackdump`) are ignored.

**Two things to watch.**

`--spectra-type` is **one value for the whole run**. A tree holding both PL and
reflectance sweeps needs one pass per type, or the reflectance sweeps are archived
as PL:

```powershell
Get-ChildItem raw -Recurse -Directory -Filter "*PL*" | ForEach-Object { tmdc-convert $_.FullName --spectra-type PL --from-raw }
Get-ChildItem raw -Recurse -Directory -Filter "*R-*" | ForEach-Object { tmdc-convert $_.FullName --spectra-type R  --from-raw }
Get-ChildItem raw -Recurse -Directory -Filter "ref"  | ForEach-Object { tmdc-convert $_.FullName --from-raw }
```

`--from-raw` is what makes that work: each command names a folder inside `raw/`,
and they all land in the same `converted/` tree.

Asking for no `--spectra-type` at all is not fatal: the frames still convert, and
each spectral export is reported as an error carrying the loader's own message,
which names the flag and lists the valid types.

**A TRPL directory converts only when you name it.** Reached by `--recursive` it is
reported as deferred and printed, because two measurements in one folder would
merge into a single archive with nothing in the file saying so:

```
tmdc-convert data/trpl-sweep                    # converts
tmdc-convert data --recursive                   # defers, and names the folder
tmdc-convert data/right_spots --prefix right1_  # converts just right1
```

## Which files are converted

Selection is by content, not by filename. A two-row single spectrum is excluded: it
is numeric on its first line exactly like an image, so the row count is what
separates them. In a TRPL folder, a file with a spectral header is that sweep's
parameter-table companion and is read as part of the sweep rather than converted on
its own. Anything excluded is reported rather than passed through.

Pixel type for images follows the data. `"auto"` keeps integer counts exactly —
`uint16` up to 65535, `uint32` above — and falls back to `float32` for anything
else, which is a narrowing from the `float64` the CSV is read as.

::: leman.converters
