# 2024B ShaneAO reduction — run log

Strategy: serial one-night-at-a-time. Extract tar, reduce, commit `Results/`,
delete raw `sNNNN.fits` frames before extracting the next night. Refuse to
extract if less than 20 GiB free.

Order (smallest tar first): 2024-10-16 -> 2024-10-17 -> 2024-08-20 -> 2024-08-19.

## Night 20241016 (2.6 GB tar)

**Status: BLOCKED during shakedown** — no sky-flat frames on either October
night. See `Raw_Data/2024B_SHAKEDOWN_no_flats.md` for details and options.

- 2026-04-23 14:00 UTC: Disk check: 50 GiB free. Extracted tar (2.6 GB).
  Remaining free: 47 GiB.
- Header inventory: 211 sNNNN.fits frames, **0 flats**, 54 darks,
  5 TIC targets + TOI-6158 + HD 183362 calibration star.
- Obs log parses cleanly (5 science TICs + TOI-6158).
- Pipeline dry-run returns expected target partitioning
  (23 / 12 / 23 / 33 / 5 frames per target).
- Halted before any reduction attempt per scope guardrail.

Raw frames kept on disk (2.6 GB) pending user decision on flats.

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
