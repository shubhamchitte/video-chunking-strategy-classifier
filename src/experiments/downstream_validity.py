"""Experiment 1: does each class's own chunker produce the most usable chunks? (teacher cache only)

    .venv/bin/python src/experiments/downstream_validity.py   # -> artifacts/results/downstream_validity*.csv

Simulates the 4 chunkers on the analysed windows from cached teacher signals:
  shot        boundaries = cached cuts
  transcript  boundaries = middle of speech pauses > 1.5 s (per-second speech cache; no cue/speaker ids)
  action_peak boundaries = local maxima of motion (flow_mag_mean) above the video's p80, >= 5 s apart
  keyframe    boundaries = OCR text_hash change or |text-area change| > TEXT_AREA_JUMP between 0.5 fps samples
Window edges also end chunks. Chunk lengths are only true lengths for videos analysed whole
(one window, duration <= 600 s): the PRIMARY table uses those. For all videos, the
empty-window rate (share of windows with no boundary) is reported as a secondary view.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teacher"))
from extract import cache_path  # noqa: E402

PAUSE_SEC = 1.5
PEAK_PCTL = 80
PEAK_MIN_GAP = 5.0
TEXT_AREA_JUMP = 0.02
SANE = (5.0, 120.0)
GIANT, FRAGMENT = 120.0, 2.0
CHUNKERS = ["transcript", "keyframe", "action_peak", "shot"]  # same order as classes 1..4
OWN = dict(zip(["1_conversational", "2_info_dense", "3_high_motion", "4_narrative"], CHUNKERS))


def transcript_bounds(sp, s, e):
    t, a = sp["t_sec"].to_numpy(), sp["speech_active"].to_numpy()
    out, i = [], 0
    while i < len(a):
        if a[i] == 0:
            j = i
            while j < len(a) and a[j] == 0:
                j += 1
            start, end = t[i], (t[j] if j < len(t) else e)
            if end - start > PAUSE_SEC and i > 0 and j < len(a):  # interior pauses only
                out.append((start + end) / 2)
            i = j
        else:
            i += 1
    return out


def peak_bounds(mo, thr):
    t, m = mo["t_sec"].to_numpy(), mo["flow_mag_mean"].to_numpy()
    cand = [i for i in range(1, len(m) - 1) if m[i] > thr and m[i] >= m[i - 1] and m[i] >= m[i + 1]]
    chosen = []
    for i in sorted(cand, key=lambda i: -m[i]):  # greedy by height, enforce min gap
        if all(abs(t[i] - t[k]) >= PEAK_MIN_GAP for k in chosen):
            chosen.append(i)
    return sorted(t[i] for i in chosen)


def keyframe_bounds(tx):
    t, h, a = tx["t_sec"].to_numpy(), tx["text_hash"].fillna("").to_numpy(), tx["text_area_frac"].to_numpy()
    return [(t[i - 1] + t[i]) / 2 for i in range(1, len(t))
            if h[i] != h[i - 1] or abs(a[i] - a[i - 1]) > TEXT_AREA_JUMP]


def chunk_lengths(bounds, s, e):
    pts = [s] + sorted(b for b in bounds if s < b < e) + [e]
    return np.diff(pts), len(pts) == 2


def metrics(lengths, analysed):
    L = np.asarray(lengths, float)
    return {
        "chunks_per_min": len(L) / (analysed / 60),
        "median_len": float(np.median(L)),
        "giant_time": L[L > GIANT].sum() / analysed,
        "fragment_chunks": float((L < FRAGMENT).mean()),
        "sane_coverage": L[(L >= SANE[0]) & (L <= SANE[1])].sum() / analysed,
    }


def main():
    labels = pd.read_csv("gt/gt_labels_v1.csv").set_index("video_id")
    windows = pd.read_csv("gt/windows.csv")
    rows = []
    for vid, wins in windows.groupby("video_id"):
        sp, cu, mo, tx = (pd.read_parquet(cache_path(k, vid)) for k in ("speech", "cuts", "motion", "text"))
        thr = np.percentile(mo["flow_mag_mean"], PEAK_PCTL) if len(mo) else np.inf
        for ch in CHUNKERS:
            lengths, empty_w = [], 0
            for wi, w in wins.reset_index(drop=True).iterrows():
                s, e = w.start_sec, w.end_sec
                if ch == "shot":
                    b = cu.loc[cu.window_idx == wi, "t_sec"].tolist()
                elif ch == "transcript":
                    b = transcript_bounds(sp[sp.window_idx == wi], s, e)
                elif ch == "action_peak":
                    b = peak_bounds(mo[mo.window_idx == wi], thr)
                else:
                    b = keyframe_bounds(tx[tx.window_idx == wi])
                L, empty = chunk_lengths(b, s, e)
                lengths += list(L)
                empty_w += empty
            analysed = float((wins.end_sec - wins.start_sec).sum())
            rows.append({"video_id": vid, "gt_class": labels.at[vid, "gt_class"], "chunker": ch,
                         "single_window": len(wins) == 1, "empty": empty_w == len(wins),
                         "empty_window_rate": empty_w / len(wins), **metrics(lengths, analysed)})
    df = pd.DataFrame(rows)
    out = Path("artifacts/results")
    df.round(4).to_csv(out / "downstream_validity_per_video.csv", index=False)

    prim = df[df.single_window]
    agg = prim.groupby(["gt_class", "chunker"]).agg(
        n=("video_id", "count"), sane_coverage=("sane_coverage", "mean"), giant_time=("giant_time", "mean"),
        fragment_chunks=("fragment_chunks", "mean"), chunks_per_min=("chunks_per_min", "mean"),
        median_len=("median_len", "median"), empty_videos=("empty", "mean")).reset_index()
    sec = df.groupby(["gt_class", "chunker"]).agg(empty_window_rate_all90=("empty_window_rate", "mean")).reset_index()
    agg = agg.merge(sec, on=["gt_class", "chunker"])
    agg["own_chunker"] = agg.apply(lambda r: OWN[r.gt_class] == r.chunker, axis=1)
    agg.round(4).to_csv(out / "downstream_validity.csv", index=False)

    piv = agg.pivot(index="gt_class", columns="chunker", values="sane_coverage")[CHUNKERS]
    print(f"PRIMARY: {prim.video_id.nunique()} single-window videos; mean sane coverage (5-120 s chunks)")
    print(piv.round(3).to_string())
    for m in ("giant_time", "fragment_chunks", "chunks_per_min", "empty_videos"):
        print(f"\n{m}:\n" + agg.pivot(index="gt_class", columns="chunker", values=m)[CHUNKERS].round(3).to_string())
    print("\nSECONDARY (all 90): empty-window rate\n"
          + agg.pivot(index="gt_class", columns="chunker", values="empty_window_rate_all90")[CHUNKERS].round(3).to_string())
    print("\nn per class (primary):", prim.drop_duplicates("video_id").gt_class.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
