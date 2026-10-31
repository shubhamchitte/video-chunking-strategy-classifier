"""Deterministic analysis-window schedule shared by teacher extractors and cheap features.

    python src/teacher/windows.py [--labels gt/gt_labels_v0.csv] [--out gt/windows.csv]

- duration <= 600 s: one window covering the whole video.
- duration >  600 s: 10 windows of 60 s, evenly spaced between 5% and 95% of duration
  (window starts span [0.05*d, 0.95*d - 60]).
"""
import argparse
import csv

SHORT_MAX_SEC = 600.0
N_WINDOWS = 10
WINDOW_SEC = 60.0
LO_FRAC, HI_FRAC = 0.05, 0.95


def windows_for(duration):
    """Return [(start_sec, end_sec), ...] for a video of the given duration."""
    if duration <= SHORT_MAX_SEC:
        return [(0.0, round(duration, 3))]
    first = LO_FRAC * duration
    last = HI_FRAC * duration - WINDOW_SEC
    step = (last - first) / (N_WINDOWS - 1)
    return [(round(first + i * step, 3), round(first + i * step + WINDOW_SEC, 3))
            for i in range(N_WINDOWS)]


def load_windows(path="gt/windows.csv"):
    """{video_id: [(start_sec, end_sec), ...]} in window_idx order."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["video_id"], []).append((float(r["start_sec"]), float(r["end_sec"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default="gt/gt_labels_v0.csv")
    ap.add_argument("--out", default="gt/windows.csv")
    args = ap.parse_args()

    with open(args.labels, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["video_id"])
    n = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "window_idx", "start_sec", "end_sec"])
        for r in rows:
            if not r["duration_sec"]:
                raise SystemExit(f"{r['video_id']}: no duration_sec; run src/data/probe.py first")
            for i, (s, e) in enumerate(windows_for(float(r["duration_sec"]))):
                w.writerow([r["video_id"], i, f"{s:.3f}", f"{e:.3f}"])
                n += 1
    print(f"{len(rows)} videos, {n} windows -> {args.out}")


if __name__ == "__main__":
    main()
