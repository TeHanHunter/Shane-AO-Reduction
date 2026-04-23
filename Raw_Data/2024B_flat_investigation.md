# 2024B flat-field investigation

**Date:** 2026-04-23
**Scope:** 2024-10-16 and 2024-10-17 (Phase D blocker follow-up)
**Motivation:** user correction — "sky flats are not the only flat option;
Shane AO also takes dome flats". Check for dome flats before falling back to
borrowing August sky flats.

## Method

For each extracted night, read the FITS primary header of every
`s[0-9][0-9][0-9][0-9].fits` frame and record:

- `OBJECT` (notebook's flat test is `'flat' in OBJECT.lower()` — this already
  matches "flat", "sky flat", "dome flat", "lamp flat")
- `ITIME` (ms) -> seconds
- `FILT1NAM`, `FILT2NAM`
- `LAMPNAM1..5,A..K` + `LAMPSTA1..5,A..K` (any lamp ON is a dome-flat signature)
- `HA`, `RA`, `DEC`, `LOOPSTAT`, `LASONSKY` (for context)

Per-frame inventories written to
`Raw_Data/data-2024-10-16-AO-Paul.Robertson/_header_inventory.tsv` and
`Raw_Data/data-2024-10-17-AO-Paul.Robertson/_header_inventory.tsv`.

Shane AO has 16 lamp name / state keyword pairs. The only lamp that is
ever ON in 2024B is `LAMPNAMC="Red"` with `LAMPSTAC="on"`. All others
("Blue", "Laser", "Neon", "He", "Hg-Cd", "Hg-A", "Sup_Blue",
"Dim_Neon", "Spare_Ar", "Spare1..4,9") are off on every frame sampled.

## 2024-10-16 — NO flats of any kind

211 frames. OBJECT histogram:

| OBJECT | count |
| --- | ---: |
| `HD 183362` | 61 |
| `dark` | 54 |
| `TIC_281571049` | 33 |
| `TIC_352409590` | 23 |
| `TIC_388814426` | 23 |
| `TOI-6158` | 12 |
| `TIC_299496195` | 5 |

- Zero frames with `OBJECT` containing `flat` / `dome` / `dflat` / `lamp`.
- Zero frames with any lamp ON.
- Obs log (xlsx): row 4 of "Observing Log 2024-10-16" = "Sky flats not
  recommended with high humidity. Operator suggested to wait until after
  sunset. Other things looks ok." No mention of dome flats anywhere in the
  sheet.

**Verdict: no-flat-available** for 2024-10-16.

## 2024-10-17 — dome flats present (not in obs log)

343 frames. OBJECT histogram:

| OBJECT | count |
| --- | ---: |
| `TIC_283410775` | 131 |
| `dark` | 111 |
| `HD 183362` | 62 |
| `dome flat` | **33** |
| `TOI-6158` | 3 |
| `TIC_388814426` | 3 |

Dome flats are frames `s0079.fits` through `s0111.fits`, all with
`LAMPNAMC="Red"` ON (all other lamps off):

| Frames | ITIME | FILT1NAM | FILT2NAM | Count |
| --- | --- | --- | --- | ---: |
| s0079-s0089 | 12.0 s | Ks | Open | 11 |
| s0090-s0100 | 18.0 s | H | Open | 11 |
| s0101-s0111 | 60.0 s | J | Open | 11 |

All 33 are lamps-on frames at a **single** lamp state (no lamps-off pair).
This is the standard Shane AO 11-frame-per-filter dome-flat sequence: one
master flat per filter is built by averaging the 11 frames after
dark-subtraction and normalizing. `dark_correct.py` already normalizes
flat frames by their exposure time, and the dark frames for this night
include matching exposures (dark 12.0s: 11, 18.0s: 11, 60.0s: 21), so the
per-filter master-flat build works without pair subtraction.

- Obs log (xlsx): row 4 of "Observing Log 2024-10-17" = "No sky flats
  again. High particles." No mention of dome flats anywhere in the sheet —
  the operator's log entries begin at frame 208 (science), so the
  frames 79-111 dome-flat block is in the FITS headers only.
- Science frames on 2024-10-17 use FILT1NAM ∈ {J, Ks}. The J and Ks dome
  flats from this same night cover both science filters at exposures that
  differ from science exposures, but Shane AO flat-field correction
  divides by the normalized master flat regardless of exposure, so this
  is the intended workflow.

**Verdict: dome-flat-available** for 2024-10-17.

## Decision

- **2024-10-17** reduces normally with its own dome flats. The pipeline's
  existing `'flat' in OBJECT.lower()` detection already matches
  `"dome flat"`, so `make_master_flats` picks them up without change.
  For robustness I added a small `src/flat_utils.py` helper,
  `is_flat_object(obj)`, that also recognizes `"dflat"`, `"dome"`,
  `"lamp"` OBJECT strings. `veto.py` and `run_night.py` now call it.
- **2024-10-16** remains blocked. Per user directive, I did **not**
  fall back to borrowing sky flats from August or dome flats from
  2024-10-17 without explicit approval. The nearest available flats
  would be the 2024-10-17 dome flats (adjacent night, same run, same
  instrument state), which is substantially more defensible than the
  original proposal of borrowing August sky flats two months earlier,
  but still requires the user to sign off.

## Pipeline changes

Added `src/flat_utils.py` defining `is_flat_object(obj)`. Updated
`src/veto.py::frame_counts` and `src/run_night.py::pick_flat_dark` to
use it. No changes to `flat_field.py` (per-filter averaging works as-is).
No pair-subtraction logic needed (no lamps-off frames in 2024B).
