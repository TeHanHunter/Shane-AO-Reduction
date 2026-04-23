# 2024B ShaneAO reduction — blocked on disk space

**Date:** 2026-04-23
**Phase reached:** A (staging) — aborted before any extraction.

## What I checked

Free space on `/Users/tehan/PycharmProjects/` (volume `/dev/disk3s5`):

```
Filesystem      Size    Used   Avail Capacity
/dev/disk3s5   460Gi   395Gi    37Gi    92%
```

**37 GiB free.** The briefing instructed: "If disk is tight (<50 GB free), stop and report — do not proceed." I stopped.

## Why 37 GiB is not enough

Raw tars (already on disk, uncompressed `tar` format):

| Tar | Size |
|---|---|
| `data-2024-08-19-AO-Paul.Robertson.tar` | 14.9 GB |
| `data-2024-08-20-AO-Paul.Robertson.tar` | 9.3 GB |
| `data-2024-10-16-AO-Paul.Robertson.tar` | 2.6 GB |
| `data-2024-10-17-AO-Paul.Robertson.tar` | 4.8 GB |
| **Subtotal (tars)** | **31.6 GB** |
| `ShaneAO.tar.xz` (xz-compressed reference) | 1.15 GB → expands to ~5-15 GB |

Extraction of the four `.tar` files is ~1:1 (tar stores files uncompressed, so the extracted tree is roughly the same 31.6 GB). Budget after extraction:

- 37 GiB − 31.6 GB (four nights) − ~10 GB (reference) ≈ **~-5 GB, i.e. overcommitted**.
- Reduction products are on top of that: per-target master flats, sky frames, dark-corrected cubes, `final.fits`, contrast-curve dirs. Rough rule of thumb is +20-40 % of raw data volume for intermediates.

## What I verified safely (no extraction)

- Repo `/Users/tehan/PycharmProjects/Shane-AO-Reduction/` exists, on `main`, with a clean `src/` layout matching the plan (`dark_correct.py`, `flat_field.py`, `sky_subtraction.py`, `reduce_astralux.py`, `utils.py`, plus `Image_Reduction_Plots.ipynb` and `Veto.ipynb`).
- Git remote: `https://github.com/TeHanHunter/Shane-AO-Reduction.git`, branch `main`. No `shaneao-2024b-reduction` branch created yet — deferred until we can actually proceed.
- `tar -tf data-2024-10-16-AO-Paul.Robertson.tar | head` confirms internal layout is `data-2024-10-16-AO-Paul.Robertson/sNNNN.fits`, matching the naming convention the notebooks expect.

## Pre-existing dirty state (unrelated to this task)

`git status` shows modifications I did **not** make — `.DS_Store` churn and `src/contrast_curve.ipynb` modified. Leaving these alone.

## Options for the user

1. **Free ~30-40 GB** on `/System/Volumes/Data`, then re-invoke. Target > 70 GB free for comfort (raw + extracted + reduction intermediates + safety margin).
2. **Stage extraction on external volume.** Symlink `Raw_Data/data-2024-MM-DD-AO-Paul.Robertson/` and `Raw_Data/_reference/ShaneAO/` onto an external drive with > 50 GB free. The notebooks hardcode `datadir` paths, but the new `src/run_night.py` will take `--data-dir` as a CLI flag, so external paths are fine for the extracted trees. Reference archive likewise.
3. **Process one night at a time,** deleting the extracted tree after each night's `Results/` is written. Smallest night (2024-10-16, 2.6 GB) fits with room, but this requires running the shakedown + batch serially and cleaning up between nights. Riskier; loses the ability to re-run without re-extracting.

My recommendation: option 1 or 2. Option 3 is viable only after the pipeline is shaken down on at least one night where we kept the raw data around.

Please advise. I'll resume at Phase A as soon as disk is available or an alternate staging path is approved.
