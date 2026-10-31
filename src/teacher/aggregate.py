"""Aggregate cached teacher time series into one row per video. Cheap; rerun anytime.

    python src/teacher/aggregate.py   # -> gt/teacher_signals.csv (+ gt/whisper_check.csv)

All rates are per minute of *analysed* time (the windows in gt/windows.csv).
Teacher-only: these columns are never classifier features.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import cache_path  # noqa: E402

EDGE_BAND = 0.15  # text_area_central drops boxes centred in the top/bottom 15% (captions, tickers, logos)

COLUMNS = ["video_id", "n_windows", "analysed_sec", "speech_source", "speech_coverage",
           "words_per_min", "cuts_per_min", "motion_mean", "motion_p90", "global_motion_mean",
           "residual_motion_mean", "global_motion_ratio", "text_area_mean", "text_area_central",
           "text_present_frac", "text_persistence"]


def central_area(boxes_json):
    area = 0.0
    for _, _, (x, y, w, h) in json.loads(boxes_json):
        if EDGE_BAND <= y + h / 2 <= 1 - EDGE_BAND:  # symmetric band, so Vision's bottom-left origin is fine
            area += w * h
    return min(1.0, area)


def text_persistence(tx):
    """Among samples with text that have a predecessor in the same window: share whose text_hash equals it."""
    same, total = 0, 0
    for _, g in tx.sort_values(["window_idx", "t_sec"]).groupby("window_idx"):
        h = g["text_hash"].tolist()
        for prev, cur in zip(h, h[1:]):
            if cur:
                total += 1
                same += cur == prev
    return same / total if total else 0.0


def speech_stats(sp, minutes):
    return sp["speech_active"].mean(), sp["word_count"].sum() / minutes


def aggregate(video_id, wins):
    analysed = float((wins.end_sec - wins.start_sec).sum())
    minutes = analysed / 60
    sp, cu, mo, tx = (pd.read_parquet(cache_path(s, video_id)) for s in ("speech", "cuts", "motion", "text"))
    cov, wpm = speech_stats(sp, minutes)
    motion_mean = mo["flow_mag_mean"].mean()
    glob = mo["global_motion"].mean()
    return {
        "video_id": video_id, "n_windows": len(wins), "analysed_sec": round(analysed, 3),
        "speech_source": sp["source"].iloc[0], "speech_coverage": cov, "words_per_min": wpm,
        "cuts_per_min": len(cu) / minutes, "motion_mean": motion_mean,
        "motion_p90": mo["flow_mag_p90"].mean(), "global_motion_mean": glob,
        "residual_motion_mean": mo["residual_motion"].mean(),
        "global_motion_ratio": glob / motion_mean if motion_mean > 0 else 0.0,
        "text_area_mean": tx["text_area_frac"].mean(),
        "text_area_central": tx["boxes_json"].map(central_area).mean(),
        "text_present_frac": (tx["n_boxes"] > 0).mean(),
        "text_persistence": text_persistence(tx),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", default="gt/windows.csv")
    ap.add_argument("--out", default="gt/teacher_signals.csv")
    ap.add_argument("--whisper-check-out", default="gt/whisper_check.csv")
    args = ap.parse_args()

    windows = pd.read_csv(args.windows)
    rows, missing = [], []
    for vid, wins in windows.groupby("video_id", sort=True):
        try:
            rows.append(aggregate(vid, wins))
        except FileNotFoundError as e:
            missing.append(f"{vid}: {e.filename}")
    if missing:
        sys.exit("Missing cache files (run src/teacher/extract.py):\n  " + "\n  ".join(missing))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.round(6).to_csv(args.out, index=False)
    print(f"{len(df)} videos -> {args.out}")

    # srt vs whisper consistency check for videos that have both.
    check = []
    for vid, wins in windows.groupby("video_id", sort=True):
        wp = cache_path("speech_whisper", vid)
        if not wp.exists():
            continue
        minutes = float((wins.end_sec - wins.start_sec).sum()) / 60
        srt_cov, srt_wpm = speech_stats(pd.read_parquet(cache_path("speech", vid)), minutes)
        wsp = pd.read_parquet(wp)
        w_cov, w_wpm = speech_stats(wsp, minutes)
        agree = (pd.read_parquet(cache_path("speech", vid))["speech_active"].values == wsp["speech_active"].values).mean()
        check.append({"video_id": vid, "srt_coverage": srt_cov, "whisper_coverage": w_cov,
                      "srt_wpm": srt_wpm, "whisper_wpm": w_wpm, "per_second_agreement": agree})
    if check:
        pd.DataFrame(check).round(4).to_csv(args.whisper_check_out, index=False)
        print(f"{len(check)} whisper-check videos -> {args.whisper_check_out}")


if __name__ == "__main__":
    main()
