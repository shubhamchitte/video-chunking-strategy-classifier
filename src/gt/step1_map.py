"""GT v0: map each video's sub_category to a chunking class and validate counts.

Deterministic and rerunnable, stdlib only. Run from the repo root:

    python src/gt/step1_map.py [--qa data/meta/test.jsonl]

Reads the per-question metadata, deduplicates to one row per video, validates
the expected counts (exits non-zero on any mismatch) and writes:

    gt/gt_labels_v0.csv       one row per video
    gt/count_validation.csv   per-sub_category expected vs found

ffprobe columns (duration_sec, fps, resolution, has_audio) and has_subtitles
are left empty here; src/data/probe.py fills them in.
"""
import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

EXPECTED_VIDEOS = OrderedDict([
    # (domain, sub_category): videos
    (("Artistic Performance", "Acrobatics"), 2),
    (("Artistic Performance", "Magic Show"), 3),
    (("Artistic Performance", "Stage Play"), 5),
    (("Artistic Performance", "Variety Show"), 1),
    (("Film & Television", "Animation"), 2),
    (("Film & Television", "Documentary"), 3),
    (("Film & Television", "Movie & TV Show"), 2),
    (("Film & Television", "News Report"), 4),
    (("Knowledge", "Astronomy"), 1),
    (("Knowledge", "Biology & Medicine"), 3),
    (("Knowledge", "Finance & Commerce"), 2),
    (("Knowledge", "Geography"), 7),
    (("Knowledge", "Humanity & History"), 2),
    (("Knowledge", "Life Tip"), 2),
    (("Knowledge", "Literature & Art"), 5),
    (("Knowledge", "Technology"), 1),
    (("Life Record", "Daily Life"), 4),
    (("Life Record", "Exercise"), 1),
    (("Life Record", "Fashion"), 7),
    (("Life Record", "Food"), 3),
    (("Life Record", "Handicraft"), 4),
    (("Life Record", "Travel"), 4),
    (("Multilingual", "Multilingual"), 2),
    (("Sports Competition", "Athletics"), 8),
    (("Sports Competition", "Basketball"), 2),
    (("Sports Competition", "Esports"), 3),
    (("Sports Competition", "Football"), 3),
    (("Sports Competition", "Other Sports"), 4),
])

CLASS_MAP = {
    "1_conversational": ["News Report", "Documentary", "Finance & Commerce",
                         "Humanity & History", "Literature & Art", "Multilingual"],
    "2_info_dense": ["Astronomy", "Biology & Medicine", "Geography", "Technology",
                     "Life Tip", "Handicraft", "Fashion", "Food"],
    "3_high_motion": ["Athletics", "Basketball", "Esports", "Football",
                      "Other Sports", "Exercise", "Acrobatics"],
    "4_narrative": ["Stage Play", "Movie & TV Show", "Animation", "Magic Show",
                    "Variety Show", "Daily Life", "Travel"],
}
SUBCAT_TO_CLASS = {s: c for c, subs in CLASS_MAP.items() for s in subs}
EXPECTED_CLASS_TOTALS = {"1_conversational": 18, "2_info_dense": 28,
                         "3_high_motion": 23, "4_narrative": 21}

NEEDS_SIGNAL_CHECK = {"Fashion", "Daily Life", "Travel", "Documentary", "Esports", "Exercise"}
EXPECTED_FLAGGED = 22

EXPECTED_N_VIDEOS = 90
EXPECTED_N_QUESTIONS = 270
EXPECTED_Q_PER_VIDEO = 3
EXPECTED_DURATION = {"short": 30, "medium": 30, "long": 30}

LABEL_COLUMNS = ["video_id", "domain", "sub_category", "duration_bucket", "duration_sec",
                 "fps", "resolution", "has_audio", "has_subtitles", "n_questions",
                 "video_path", "gt_class", "gt_source", "needs_signal_check", "notes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qa", default="data/meta/test.jsonl")
    ap.add_argument("--out-dir", default="gt")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.qa).read_text().splitlines() if line.strip()]
    errors = []

    # Deduplicate to one row per video, checking per-video consistency.
    videos = OrderedDict()
    q_count = Counter()
    for r in rows:
        vid = r["video_id"]
        q_count[vid] += 1
        key = (r["domain"], r["sub_category"], r["duration"])
        if vid in videos and videos[vid] != key:
            errors.append(f"{vid}: inconsistent metadata across questions {videos[vid]} vs {key}")
        videos.setdefault(vid, key)

    if len(rows) != EXPECTED_N_QUESTIONS:
        errors.append(f"questions: expected {EXPECTED_N_QUESTIONS}, found {len(rows)}")
    if len(videos) != EXPECTED_N_VIDEOS:
        errors.append(f"videos: expected {EXPECTED_N_VIDEOS}, found {len(videos)}")
    for vid, n in q_count.items():
        if n != EXPECTED_Q_PER_VIDEO:
            errors.append(f"{vid}: expected {EXPECTED_Q_PER_VIDEO} questions, found {n}")
    dur = Counter(k[2] for k in videos.values())
    if dict(dur) != EXPECTED_DURATION:
        errors.append(f"duration buckets: expected {EXPECTED_DURATION}, found {dict(dur)}")

    # Per-sub_category validation.
    found_v = Counter((d, s) for d, s, _ in videos.values())
    found_q = Counter((r["domain"], r["sub_category"]) for r in rows)
    validation = []
    for key in list(EXPECTED_VIDEOS) + sorted(set(found_v) - set(EXPECTED_VIDEOS)):
        ev = EXPECTED_VIDEOS.get(key, 0)
        fv, fq = found_v.get(key, 0), found_q.get(key, 0)
        match = ev == fv and ev * EXPECTED_Q_PER_VIDEO == fq
        if not match:
            errors.append(f"{key}: expected {ev} videos/{ev * 3} questions, found {fv}/{fq}")
        validation.append({"domain": key[0], "sub_category": key[1], "expected_videos": ev,
                           "found_videos": fv, "expected_questions": ev * EXPECTED_Q_PER_VIDEO,
                           "found_questions": fq, "match": match})

    # Map to classes.
    labels = []
    for vid, (domain, subcat, bucket) in videos.items():
        flagged = subcat in NEEDS_SIGNAL_CHECK
        labels.append({
            "video_id": vid, "domain": domain, "sub_category": subcat,
            "duration_bucket": bucket, "duration_sec": "", "fps": "", "resolution": "",
            "has_audio": "", "has_subtitles": "", "n_questions": q_count[vid],
            "video_path": f"data/videos/{vid}.mp4",
            "gt_class": SUBCAT_TO_CLASS.get(subcat, "UNMAPPED"),
            "gt_source": "subcat_map", "needs_signal_check": flagged,
            "notes": "provisional: ambiguous sub_category, pending signal check" if flagged else "",
        })
    labels.sort(key=lambda r: r["video_id"])

    cls = Counter(r["gt_class"] for r in labels)
    if dict(cls) != EXPECTED_CLASS_TOTALS:
        errors.append(f"class totals: expected {EXPECTED_CLASS_TOTALS}, found {dict(cls)}")
    n_flag = sum(r["needs_signal_check"] for r in labels)
    if n_flag != EXPECTED_FLAGGED:
        errors.append(f"needs_signal_check: expected {EXPECTED_FLAGGED}, found {n_flag}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "count_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(validation[0]))
        w.writeheader()
        w.writerows(validation)
    with open(out / "gt_labels_v0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerows(labels)

    print(f"videos={len(videos)} questions={len(rows)} durations={dict(sorted(dur.items()))}")
    print("classes:", dict(sorted(cls.items())), f"flagged={n_flag}")
    if errors:
        print("VALIDATION FAILED:", *errors, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print("All validations passed.")


if __name__ == "__main__":
    main()
