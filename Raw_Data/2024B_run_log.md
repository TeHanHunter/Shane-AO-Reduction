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
