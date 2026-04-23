"""
Parse Raw_Data/2024B_ShaneAO.xlsx into a normalized per-(night, target, filter) CSV.

The xlsx has per-night "Observing Log YYYY-MM-DD" sheets, each with columns:
    File Number, Object, Start Time (UT), Exposure Time (s),
    Filter, #Expose/Pos'n, #Dither Pos'n, Notes

File Number can be a single int (e.g. "168") or a hyphen range (e.g. "170-184").
Object can be "Sky Flat", "TIC_352409590", "TIC 352409590", "TOI-6158", etc.
Notes contains free-text human annotations; we extract a coarse manual-veto
flag from keywords.

Emits Raw_Data/2024B_obs_log.csv with columns:
    night, target, tic_id, filter, frame_first, frame_last, exp_time_s,
    expose_per_posn, dither_posns, is_flat, notes, manual_veto

One row per contiguous File Number range in the log (so a single target may
have several rows for different exposure times / repeat visits / dither
sequences). This is friendlier for reduction than trying to collapse them.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


NIGHT_SHEET_RE = re.compile(r"^Observing Log (\d{4}-\d{2}-\d{2})$")
VETO_KEYWORDS = (
    "overexpose", "saturat", "lost", "too high", "too low",
    "failed", "closing", "closed", "bad", "junk", "aborted",
)


def parse_frame_range(cell: str) -> tuple[int, int] | None:
    """'168' -> (168, 168), '170-184' -> (170, 184), 'H and J sky flats...' -> None."""
    if pd.isna(cell):
        return None
    s = str(cell).strip()
    # Pure int
    if s.isdigit():
        v = int(s)
        return (v, v)
    # Range "a-b"
    m = re.match(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a, b) if a <= b else (b, a)
    return None


def extract_tic_id(obj: str) -> str | None:
    """'TIC_352409590' / 'TIC 352409590' -> '352409590'. Non-TIC -> None."""
    if not isinstance(obj, str):
        return None
    m = re.search(r"TIC[ _]?(\d+)", obj, flags=re.I)
    return m.group(1) if m else None


def classify_object(obj) -> tuple[str, bool]:
    """Return (canonical_target_name, is_flat)."""
    if pd.isna(obj):
        return ("", False)
    s = str(obj).strip()
    low = s.lower()
    if "flat" in low:
        return ("SkyFlat", True)
    if "dark" in low:
        return ("Dark", False)
    # Normalize TIC formatting: "TIC 12345" -> "TIC_12345"
    m = re.match(r"^(TIC)[ _]?(\d+)$", s, flags=re.I)
    if m:
        return (f"TIC_{m.group(2)}", False)
    # Keep other names (TOI-6158, TO-6158, etc.) as-is
    return (s, False)


def manual_veto_flag(notes) -> bool:
    if pd.isna(notes):
        return False
    low = str(notes).lower()
    return any(k in low for k in VETO_KEYWORDS)


def parse_sheet(xlsx_path: Path, sheet: str, night: str) -> pd.DataFrame:
    # Read with header=None so we can find the real header row ourselves.
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
    # Find the "File Number" header row
    header_row = None
    for i in range(min(10, len(raw))):
        row = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if "file number" in row and "object" in row:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"[{sheet}] Could not locate header row")

    df = pd.read_excel(xlsx_path, sheet_name=sheet, header=header_row)
    # Keep only the 8 standard cols (some sheets have junk trailing cols)
    wanted = [
        "File Number", "Object", "Start Time (UT)", "Exposure Time (s)",
        "Filter", "#Expose/Pos'n", "#Dither Pos'n", "Notes",
    ]
    present = [c for c in wanted if c in df.columns]
    if "File Number" not in present or "Object" not in present:
        raise ValueError(f"[{sheet}] Missing File Number / Object columns")
    df = df[present].copy()

    rows = []
    for _, r in df.iterrows():
        fr = parse_frame_range(r.get("File Number"))
        if fr is None:
            continue  # free-text note row
        target, is_flat = classify_object(r.get("Object"))
        if not target:
            continue
        tic = extract_tic_id(r.get("Object"))
        filt = r.get("Filter") if "Filter" in present else None
        if isinstance(filt, str):
            filt = filt.strip()
        exp = r.get("Exposure Time (s)") if "Exposure Time (s)" in present else None
        epp = r.get("#Expose/Pos'n") if "#Expose/Pos'n" in present else None
        dpp = r.get("#Dither Pos'n") if "#Dither Pos'n" in present else None
        notes = r.get("Notes") if "Notes" in present else None

        rows.append({
            "night": night.replace("-", ""),  # 2024-10-16 -> 20241016
            "target": target,
            "tic_id": tic,
            "filter": filt,
            "frame_first": fr[0],
            "frame_last": fr[1],
            "exp_time_s": exp if not pd.isna(exp) else None,
            "expose_per_posn": epp if not pd.isna(epp) else None,
            "dither_posns": dpp if not pd.isna(dpp) else None,
            "is_flat": is_flat,
            "notes": notes if not pd.isna(notes) else "",
            "manual_veto": manual_veto_flag(notes),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="Raw_Data/2024B_ShaneAO.xlsx")
    ap.add_argument("--out", default="Raw_Data/2024B_obs_log.csv")
    args = ap.parse_args()

    xlsx = Path(args.xlsx)
    xls = pd.ExcelFile(xlsx)

    frames = []
    for sheet in xls.sheet_names:
        m = NIGHT_SHEET_RE.match(sheet)
        if not m:
            continue
        night = m.group(1)
        df = parse_sheet(xlsx, sheet, night)
        frames.append(df)
        print(f"[{sheet}] -> {len(df)} rows "
              f"({df['is_flat'].sum()} flats, "
              f"{df['target'].nunique()} distinct targets)")

    if not frames:
        raise SystemExit("No 'Observing Log YYYY-MM-DD' sheets found")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}: {len(out)} rows, "
          f"{out['night'].nunique()} nights, "
          f"{out[~out['is_flat']]['target'].nunique()} distinct non-flat targets")
    print("\nPer-night summary:")
    print(out.groupby("night").agg(
        rows=("target", "size"),
        targets=("target", lambda s: s[~s.eq("SkyFlat")].nunique()),
        flats=("is_flat", "sum"),
        vetoed=("manual_veto", "sum"),
    ))


if __name__ == "__main__":
    main()
