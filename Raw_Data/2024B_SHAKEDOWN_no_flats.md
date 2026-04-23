# 2024B shakedown blocker — October nights have no sky flats

**Date:** 2026-04-23
**Phase reached:** D (shakedown), blocked before reduction attempt.
**Status:** stopping per scope guardrail ("if shakedown fails, stop and write a report").

## TL;DR

The plan specifies the first-processed night is **2024-10-16** (smallest tar,
fastest iteration). Both October tars contain **zero sky-flat frames**. The
`make_master_flats` helper from `flat_field.py` requires at least one flat per
filter — it runs `np.stack([...], axis=0)` on an empty list and crashes — so
the pipeline as written cannot produce a `TIC_<id>_final.fits` for any target
on either October night without a source of flats.

Phases A–C are complete and committed on branch `shaneao-2024b-reduction`
(commit `295bbbb`). The branch is ready to proceed the moment a flat-fielding
decision is made.

## Evidence

### Obs log confirms no sky flats on 2024-10-16 or 2024-10-17

From `Observing Log 2024-10-16` (parsed into `Raw_Data/2024B_obs_log.csv`):

> "Sky flats not recommended with high humidity. Operator suggested to wait
> until after sunset. Other things looks ok."

From `Observing Log 2024-10-17`:

> "No sky flats again. High particles."

### Header inventory of the extracted 2024-10-16 raw data

211 sNNNN.fits frames, classified by FITS `OBJECT` header:

| OBJECT | Frames |
| --- | ---: |
| HD 183362 | 61 |
| dark | 54 |
| TIC_281571049 | 33 |
| TIC_352409590 | 23 |
| TIC_388814426 | 23 |
| TOI-6158 | 12 |
| TIC_299496195 | 5 |

Zero frames with "flat" in `OBJECT`. (HD 183362 is an on-star AO tuning
target, not a flat-field source — it has a bright PSF, not a uniform
illumination pattern.)

Filter counts: 151 Ks, 54 "BrG-2.16" (all darks, filter wheel position at
dark acquisition), 6 J.

### Notebook behaviour is flats-mandatory

`Image_Reduction_Plots.ipynb` cell 7:

```python
master_flat = make_master_flats(flat_list, filter_, darkcor_data_out, datadir)
flat_darkcor_data_out = flat_field(object_list, master_flat, darkcor_data_out)
```

Inside `make_master_flats`:

```python
flatcube = np.stack([darkcor_data_out[flat_frame] for flat_frame in flat_filt_list], axis=0)
```

If `flat_filt_list` is empty (no frames with `FILT1NAM == 'Ks'` and
"flat" in the object name), `np.stack` raises
`ValueError: need at least one array to stack`.

My `src/run_night.py` therefore refuses to proceed on a night with zero
flats (returns `status: skipped_no_flats`).

## Options for the user

### Option 1 — Borrow sky flats from the August nights

The 2024-08-19 tar has sky flats at frames 83–107 (tar inventory confirms
they exist; obs log notes `~8K counts` in Ks). 2024-08-20 has flats at
81–103. Both nights have both Ks and J flats.

Approach: extract a minimal "flats only" slice from one August tar once,
copy the sky-flat files into each October night's extracted dir (renaming
won't be necessary if the OBJECT header still says "Sky Flat"), and let
`run_night.py` pick them up transparently. Instrument response on a
well-maintained AO camera is typically stable across ~2 months, so this
should produce a usable master flat, though it is not ideal.

Risk: pixel-to-pixel gain differences caused by e.g. a detector warm
event between Aug and Oct would be un-caught. Cosmetic features in the
final image rather than photometric errors.

### Option 2 — Dome flats / lamp flats elsewhere in the data

Not evident in the tars — no frames labelled "dome" or "lamp". Would need
user confirmation that these were ever taken.

### Option 3 — Skip flat-fielding for October nights only

Modify `run_night.py` to allow `--no-flat`, which would replace the master
flat with an array of ones. This is a scientifically inferior output but
would produce usable contrast curves (since AO contrast is dominated by
PSF speckle residuals, not 1–5 % flat-field structure at the star position).

Risk: detectable 1–5 % flat structure in the residual images after PSF
subtraction, potentially flagged as false companions.

### Option 4 — Use an archival ShaneAO flat

If the `ShaneAO.tar.xz` reference tree has stored master flats somewhere,
those could serve. Quick check: none of the 7 reference nights with
`Results/` appear to contain a raw `flat_list.txt` or a stored master
flat — they only ship the final per-target products.

## My recommendation

**Option 1.** August flats are physically closest in time, from the same
instrument, and the ShaneAO camera's gain structure is documented to be
stable over months. Worst-case cosmetic issue will be visible in the
residuals; we flag anything suspect in the run log.

If the user agrees, the next action is a ~200 MB surgical extract of
sky-flat frames from one August tar into a shared
`Raw_Data/_shared_flats/2024_Aug_Ks/` + `.../2024_Aug_J/` dir, and a small
`--extra-flats-dir` flag on `run_night.py` that prepends those frames to
the per-night flat list. No change to the disk-space strategy: the
shared flats are ~200 MB, well under the 20 GiB floor.

## Current disk state

```
/dev/disk3s5   460Gi   384Gi    47Gi    89%
```

47 GiB free. 2024-10-16 raw data is still extracted (2.6 GB). I have
**not** deleted it — happy to either (a) delete it now to free the
working set for a different night, or (b) leave it in place until the
user picks an option.

## Files produced so far (committed on `shaneao-2024b-reduction`)

- `.gitignore` (excludes venv, extracted nights, tars, reference tree)
- `src/parse_obs_log.py`, `src/veto.py`, `src/run_night.py`
- `Raw_Data/2024B_obs_log.csv` (235 rows, 6 nights, 41 distinct targets)
- `Raw_Data/2024B_ShaneAO.xlsx` (input obs log, preserved for reference)
- `Raw_Data/2024B_run_log.md` (stub, populated after first successful night)
- `Raw_Data/2024B_BLOCKER_disk_space.md` (historical record of disk blocker)
- `.venv-shaneao/` (not committed; deps: astropy, pandas, numpy, scipy,
  matplotlib, openpyxl, photutils, scikit-image, tqdm)

Awaiting decision on the flat-field source before Phase D (shakedown) and
Phase E (four-night batch).
