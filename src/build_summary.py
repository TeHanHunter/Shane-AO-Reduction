"""Build a per-target summary of the 2024B reduction run.

For each (night, target) row block in the obs log:
  * n_frames_observed   = sum of (frame_last - frame_first + 1) for non-flat rows
  * n_frames_vetoed     = how many of those frame IDs appear in
                          data-<night>/veto_list.txt (0 if no veto file)
  * result              = "ok" / "failed" / "skipped" based on whether a
                          final.fits (or per-filter finals) exist in
                          Raw_Data/2024B_Results/<night>/
  * final_fits_path     = relative path(s) to the final.fits file(s); empty
                          string if none
  * notes               = concatenation of obs-log notes plus any pipeline
                          failure reason captured in 2024B_run_log.md

Outputs:
  Raw_Data/2024B_summary.csv
  Raw_Data/2024B_summary.md   (human-readable table)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parent.parent
OBS_LOG = REPO / "Raw_Data" / "2024B_obs_log.csv"
RESULTS_ROOT = REPO / "Raw_Data" / "2024B_Results"
OUT_CSV = REPO / "Raw_Data" / "2024B_summary.csv"
OUT_MD = REPO / "Raw_Data" / "2024B_summary.md"

NIGHT_TO_DATADIR = {
    20240819: "data-2024-08-19-AO-Paul.Robertson",
    20240820: "data-2024-08-20-AO-Paul.Robertson",
    20241016: "data-2024-10-16-AO-Paul.Robertson",
    20241017: "data-2024-10-17-AO-Paul.Robertson",
}
# Nights we have raw tars for (others are log-only, no data on disk).
NIGHTS_WITH_DATA = set(NIGHT_TO_DATADIR.keys())


def sanitize_target(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(t).strip()).strip("_")


def load_veto(night: int) -> set[int]:
    datadir = REPO / "Raw_Data" / NIGHT_TO_DATADIR.get(night, "")
    vf = datadir / "veto_list.txt"
    if not vf.exists():
        return set()
    frames = set()
    for line in vf.read_text().splitlines():
        m = re.search(r"s(\d+)", line.strip())
        if m:
            frames.add(int(m.group(1)))
    return frames


def find_finals(night: int, target_sanitized: str) -> list[Path]:
    night_dir = RESULTS_ROOT / str(night)
    if not night_dir.exists():
        return []
    out = []
    for p in sorted(night_dir.glob(f"{target_sanitized}_final*.fits")):
        out.append(p)
    return out


def main() -> None:
    df = pd.read_csv(OBS_LOG)
    df["night"] = df["night"].astype(int)
    df["frame_first"] = pd.to_numeric(df["frame_first"], errors="coerce")
    df["frame_last"] = pd.to_numeric(df["frame_last"], errors="coerce")
    df["is_flat"] = df["is_flat"].fillna(False).astype(bool)
    df["target"] = df["target"].fillna("").astype(str)

    rows = []
    for (night, target), grp in df[~df["is_flat"]].groupby(["night", "target"],
                                                             dropna=False):
        if not target.strip():
            continue
        ts = sanitize_target(target)
        tic_ids = sorted({str(int(x)) for x in grp["tic_id"].dropna()
                          if not pd.isna(x)})
        filts = sorted({f for f in grp["filter"].fillna("").astype(str)
                        if f})

        # Frame coverage
        frames = set()
        for _, r in grp.iterrows():
            if pd.isna(r["frame_first"]) or pd.isna(r["frame_last"]):
                continue
            a, b = int(r["frame_first"]), int(r["frame_last"])
            frames.update(range(a, b + 1))
        n_obs = len(frames)

        vetoed = frames & load_veto(int(night))
        n_vet = len(vetoed)

        finals = find_finals(int(night), ts)
        filters_produced = []
        for p in finals:
            m = re.search(r"_final(?:_filter_(\w+))?\.fits$", p.name)
            if not m:
                continue
            filters_produced.append(m.group(1) or "Ks")
        filters_produced = sorted(set(filters_produced))

        if int(night) not in NIGHTS_WITH_DATA:
            result = "no_data"
        elif finals:
            result = "ok"
        else:
            result = "failed"

        notes_list = [n for n in grp["notes"].fillna("").astype(str)
                      if n.strip()]
        notes = " | ".join(dict.fromkeys(notes_list))[:400]

        rows.append({
            "night": int(night),
            "target": target,
            "tic_id": ";".join(tic_ids),
            "filters_planned": ",".join(filts),
            "filters_produced": ",".join(filters_produced),
            "n_frames_observed": n_obs,
            "n_frames_vetoed": n_vet,
            "result": result,
            "final_fits_path": ";".join(
                str(p.relative_to(REPO)) for p in finals),
            "notes": notes,
        })

    out = pd.DataFrame(rows).sort_values(["night", "target"]).reset_index(
        drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(out)} rows)")

    # Markdown
    lines = ["# 2024B ShaneAO reduction — per-target summary",
             "",
             f"Source: `Raw_Data/2024B_obs_log.csv` + `Raw_Data/2024B_Results/`",
             ""]
    for night, sub in out.groupby("night"):
        lines.append(f"## {night}  ({len(sub)} targets)")
        lines.append("")
        lines.append("| target | TIC | filt (plan / produced) | N obs | N vet | result | final.fits |")
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
        for _, r in sub.iterrows():
            finals = r["final_fits_path"] or "—"
            finals_short = ";<br>".join(
                Path(p).name for p in finals.split(";") if p and p != "—")
            if not finals_short:
                finals_short = "—"
            lines.append(
                f"| {r['target']} | {r['tic_id'] or '—'} | "
                f"{r['filters_planned']} / {r['filters_produced'] or '—'} | "
                f"{r['n_frames_observed']} | {r['n_frames_vetoed']} | "
                f"**{r['result']}** | {finals_short} |"
            )
        lines.append("")

    # Totals
    lines.append("## Totals")
    lines.append("")
    total = len(out)
    ok = (out["result"] == "ok").sum()
    fail = (out["result"] == "failed").sum()
    nodata = (out["result"] == "no_data").sum()
    with_data = out[out["result"] != "no_data"]
    lines.append(f"- Targets in obs log: **{total}**")
    lines.append(f"- Nights with raw data on disk (tars extracted): "
                 f"**{sorted(NIGHTS_WITH_DATA)}**")
    lines.append(f"- Targets with raw data: **{len(with_data)}**  "
                 f"(ok={ok}, failed={fail})")
    lines.append(f"- Targets without raw data (log-only): **{nodata}**")
    lines.append(f"- Total frames observed (with data): "
                 f"**{int(with_data['n_frames_observed'].sum())}**")
    lines.append(f"- Total frames vetoed (with data): "
                 f"**{int(with_data['n_frames_vetoed'].sum())}**")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
