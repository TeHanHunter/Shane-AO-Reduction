# 2024B ShaneAO reduction — run log

Strategy: serial one-night-at-a-time. Extract tar, reduce, commit `Results/`,
delete raw `sNNNN.fits` frames before extracting the next night. Refuse to
extract if less than 20 GiB free.

Order (smallest tar first): 2024-10-16 -> 2024-10-17 -> 2024-08-20 -> 2024-08-19.

## Night 20241016 (2.6 GB tar)

**Status: done** — reduced using **borrowed 2024-10-17 dome flats**
(per user directive). 33 dome flats (Ks 12s, H 18s, J 60s) + their
matching 12/18/60s darks were re-extracted from the 2024-10-17 tar,
renamed to `s9079`-`s9389` to avoid collision with 2024-10-16's own
`s0107`-`s0317` frames, and copied into the 10-16 data dir. The pipeline
recognises them via `is_flat_object(OBJECT)` regardless of filename.

Caveat: cross-night flats. Same instrument state, adjacent night — more
defensible than the original proposal (August sky flats two months
earlier), but it is still a borrow.

Reduced (3 targets, 4 final.fits):
- TIC_352409590 J + Ks
- TIC_388814426 Ks
- TIC_281571049 Ks

Failed:
- TOI-6158 Ks — only 1 dither position populated at 240s.
- TIC_299496195 Ks — all 5 frames dropped by sparse-itime filter
  (30/60/90s singletons).
- TO-6158 (typo in xlsx) — skipped (too few frames after veto).

## Night 20241017 (dome flats; tar extracted and processed in prior session)

**Status: done (committed 6331a69).**

- Reduced: TIC_283410775 J.
- Skipped: TIC_283410775 Ks (single dither position after manual vetoes);
  TOI_6158, TIC_388814426 (<5 frames after manual vetoes).
- Dome flats used in place of absent sky flats.
- Raws deleted after commit.

## Night 20240820 (8 GB tar; processed in two passes)

**Status: done.**

Pass 1 (prior session, committed 08d5c6f):
- TIC_17540944 J
- TIC_192415411 Ks
- TIC_233574265 J
- TIC_299496195 Ks
- TIC_427314633 J
- TIC_441056236 J
- TIC_441056236 Ks
- TIC_89502706 Ks

Pass 2 (this session):
- TIC_299496195 J — reduced successfully.
- TIC_283410775 J — **skipped_too_few**: only 4 frames remained after
  applying obs-log manual vetoes (`expose_per_posn=20, dither=5` sequence
  was marked veto due to widespread overexposures); below the 5-frame
  per-ITIME floor enforced in `run_night._run_single_filter`.

Raws deleted post-commit.

## Night 20240819 (15 GB tar)

**Status: done.**

Reduced (7 targets, 8 final.fits):
- TIC_189571727 Ks
- TIC_285181196 J + Ks
- TIC_298559217 J
- TIC_313874586 J
- TIC_352409577 Ks
- TIC_60922830 Ks
- TIC_71038757 Ks

Failed:
- TIC_298559217 Ks — NaN crash in sky-subtraction (338-center at
  30s exposure; likely a bad centroid).
- TIC_313874586 Ks — (not produced; centroid/crop issue, J succeeded).
- TIC_346416638 Ks — only 1 dither position populated at 8s.

Raws deleted post-commit.

## Night 20241217 (raws delivered post-Apr-2026; processed 2026-05-01)

**Status: done.**

Reduced (5 targets, 6 final.fits):
- TIC_352409590 Ks
- TIC_35760711 Ks
- TIC_435903839 Ks
- TIC_12632044 Ks
- TIC_172572159 J + Ks

Failed (NaN-centroid in sky subtraction):
- TIC_257397333 Ks
- TIC_60764070 Ks

Skipped (too few frames after veto): TIC_283866910, TIC_117880865,
TIC_77552918, TIC_364898.

## Night 20241219 (raws delivered post-Apr-2026; processed 2026-05-01)

**Status: done.**

Reduced (9 targets, 9 final.fits):
- TIC_148251101 Ks
- TIC_151058955 Ks
- TIC_10056120 Ks
- TIC_46739994 Ks
- TIC_265168621 Ks
- TIC_60764070 Ks
- TIC_149766251 Ks
- TIC_77552918 Ks

Failed (single-dither-position at one ITIME):
- TIC_289706625 Ks

Skipped (too few frames after veto): TIC_345778835, TIC_117880865,
TIC_333620087, TIC_471012349.

## Phase F — contrast curves

All 22 2024B-original `*_final*.fits` passed through
`src/reduce_astralux.py --input-file` on 2026-04-23. Three 08-19
targets (`TIC_313874586_final_filter_J`, `TIC_352409577_final`,
`TIC_60922830_final`) needed a second pass with a 600s per-file timeout.

The 14 December `*_final*.fits` (6 from 12-17 + 8 from 12-19) passed
through `reduce_astralux.py` on 2026-05-01 with no per-file timeouts.

## SURFSUP integration

After each reduction batch, `pipeline/shaneao_binary_detect.py` and
`pipeline/merge_shaneao_into_v2.py` (in the SURFSUP repo) refresh the
`ao_*` columns in `paper_tables/surfsup_master_v2.csv`. The 2026-05-01
December merge added 13 targets to v2 (TIC_172572159 and TIC_352409590
already had Oct/Jan epochs and were updated to multi-night entries);
all 13 returned `ao_binary_flag=single`.

## Final tally (2024B)

- 39 targets had raw data on disk (6 nights extracted; Aug/Oct raws
  deleted post-commit).
- **32 targets** reduced to at least one `final.fits`; 36 per-filter
  outputs in total.
- 10 targets failed (mostly single-dither-position or sparse-exposure
  cases; three NaN crashes).
- 11 targets remain in the obs log only — too few frames after veto.

See `Raw_Data/2024B_summary.md` for the per-target table and
`Raw_Data/2024B_summary.csv` for the machine-readable version.
