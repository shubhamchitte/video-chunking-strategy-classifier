"""Fill ffprobe metadata into gt/gt_labels_v0.csv. Stdlib only; ffprobe must be on PATH.

    python src/data/probe.py [--labels gt/gt_labels_v0.csv]

Fills duration_sec, fps, resolution, has_audio, has_subtitles. Files that are
missing or fail to open, and videos with no usable frame rate, are flagged in
`notes` and listed in gt/probe_issues.csv.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

FFPROBE_ARGS = ["-v", "error", "-show_entries",
                "format=duration:stream=codec_type,avg_frame_rate,width,height", "-of", "json"]


def parse_fps(rate):
    """'30000/1001' -> 29.97; '0/0', empty or missing -> None."""
    try:
        f = Fraction(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return round(float(f), 3) if f > 0 else None


def probe(path, ffprobe):
    res = subprocess.run([ffprobe, *FFPROBE_ARGS, str(path)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or f"ffprobe exit {res.returncode}")
    info = json.loads(res.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("no video stream")
    duration = info.get("format", {}).get("duration")
    return {
        "duration_sec": round(float(duration), 3) if duration else "",
        "fps": parse_fps(video.get("avg_frame_rate")),
        "resolution": f"{video['width']}x{video['height']}" if video.get("width") else "",
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def add_note(row, note):
    row["notes"] = "; ".join(n for n in (row["notes"], note) if n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default="gt/gt_labels_v0.csv")
    ap.add_argument("--subtitles-dir", default="data/subtitles")
    ap.add_argument("--issues", default="gt/probe_issues.csv")
    args = ap.parse_args()

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        sys.exit("ERROR: ffprobe not found on PATH")

    with open(args.labels, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)

    issues = []
    for row in rows:
        # Drop notes from a previous probe run so reruns stay deterministic.
        row["notes"] = "; ".join(n for n in row["notes"].split("; ")
                                 if n and not n.startswith("probe:"))
        row["has_subtitles"] = (Path(args.subtitles_dir) / f"{row['video_id']}.srt").is_file()
        path = Path(row["video_path"])
        if not path.is_file():
            meta, err = None, "missing file"
        else:
            try:
                meta, err = probe(path, ffprobe), None
            except (RuntimeError, ValueError, KeyError) as e:
                meta, err = None, f"unreadable: {e}"
        if meta is None:
            for k in ("duration_sec", "fps", "resolution", "has_audio"):
                row[k] = ""
            add_note(row, f"probe: {err}")
            issues.append({"video_id": row["video_id"], "issue": err})
            continue
        row.update({k: ("" if v is None else v) for k, v in meta.items()})
        if meta["fps"] is None:
            add_note(row, "probe: no usable avg_frame_rate")
            issues.append({"video_id": row["video_id"], "issue": "no usable avg_frame_rate"})

    with open(args.labels, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(args.issues, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "issue"])
        w.writeheader()
        w.writerows(issues)

    ok = len(rows) - len({i["video_id"] for i in issues})
    print(f"probed {len(rows)} videos: {ok} ok, {len(issues)} issue(s) -> {args.issues}")
    print(f"no audio: {sum(r['has_audio'] is False for r in rows)}; "
          f"with subtitles: {sum(r['has_subtitles'] for r in rows)}")


if __name__ == "__main__":
    main()
