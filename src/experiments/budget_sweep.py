"""Accuracy vs. latency: re-extract the v1 cheap features under different analysis budgets.

    CHEAP_FFMPEG_THREADS=1 python src/experiments/budget_sweep.py extract --jobs 6
    python src/experiments/budget_sweep.py evaluate      # -> artifacts/results/budget_sweep*.csv, docs/tradeoff.svg

Same feature code (src/features/cheap.py) and same 17 features; only the window schedule and
the video fps change. Windows are evenly spaced within 5-95% of the duration (a single window is
centred at 50%); a video no longer than the total budget is analysed whole. CPU time per video
includes ffmpeg children; run with CHEAP_FFMPEG_THREADS=1 so it reflects one-core cost.
Experiment only: the graded path (run.yaml, src/main.py) still uses 10 x 60 s at 2 fps.
"""
import argparse
import csv
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "features"), str(ROOT / "src" / "classify"), str(ROOT / "src" / "teacher")]
import cheap  # noqa: E402

# name: (n_windows, window_sec, video_fps)
BUDGETS = {
    "1x30s_2fps": (1, 30.0, 2.0),
    "3x30s_2fps": (3, 30.0, 2.0),
    "5x60s_2fps": (5, 60.0, 2.0),
    "10x60s_1fps": (10, 60.0, 1.0),
    "10x60s_2fps": (10, 60.0, 2.0),  # shipped
}
SHIPPED = "10x60s_2fps"
OUT_DIR = Path("artifacts/features/budget_sweep")
RES = Path("artifacts/results")
LONG_VIDEO = "eZ5lFBPpRkM"  # 3221 s, 720p: wall time on one core
CAPACITY_HOURS_PER_DAY, UTILIZATION = 10_000, 0.6


def schedule(n, length):
    def windows_for(duration):
        if duration <= n * length:
            return [(0.0, round(duration, 3))]
        if n == 1:
            s = 0.5 * duration - length / 2
            return [(round(s, 3), round(s + length, 3))]
        first, last = 0.05 * duration, 0.95 * duration - length
        step = (last - first) / (n - 1)
        return [(round(first + i * step, 3), round(first + i * step + length, 3)) for i in range(n)]
    return windows_for


def cpu():
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + ch.ru_utime + ch.ru_stime


def _job(args):
    budget, vid, path = args
    n, length, fps = BUDGETS[budget]
    cheap.windows_for = schedule(n, length)  # module globals read at call time
    cheap.VIDEO_FPS = fps
    c0, t0 = cpu(), time.perf_counter()
    feats, analysed = cheap.extract(path)
    return budget, vid, feats, analysed, cpu() - c0, time.perf_counter() - t0


def extract(args):
    with open("gt/gt_labels_v1.csv", newline="") as f:
        videos = sorted((r["video_id"], r["video_path"]) for r in csv.DictReader(f))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for budget in args.budgets or BUDGETS:
        out = OUT_DIR / f"{budget}.csv"
        if out.exists() and not args.force:
            print(f"{budget}: cached")
            continue
        rows = []
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for b, vid, feats, analysed, c, w in ex.map(_job, [(budget, v, p) for v, p in videos]):
                rows.append([vid, round(analysed, 3), round(c, 3), round(w, 3), *(round(feats[k], 6) for k in cheap.FEATURES)])
        with open(out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["video_id", "analysed_sec", "cpu_sec", "wall_sec", *cheap.FEATURES])
            wr.writerows(rows)
        print(f"{budget}: {len(rows)} videos, mean CPU {np.mean([r[2] for r in rows]):.2f} s -> {out}", flush=True)
    # Wall time for one long video on one core, sequential, each budget.
    walls = {}
    path = dict(videos)[LONG_VIDEO]
    for budget in BUDGETS:
        _, _, _, analysed, c, w = _job((budget, LONG_VIDEO, path))
        walls[budget] = {"analysed_sec": analysed, "cpu_sec": round(c, 2), "wall_sec": round(w, 2)}
        print(f"long video {LONG_VIDEO} {budget}: wall {w:.1f}s cpu {c:.1f}s", flush=True)
    json.dump(walls, open(OUT_DIR / "long_video_wall.json", "w"), indent=1)


def evaluate(_args):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    from models import SEED, load_dataset, make_model
    _, y, ids, _, labels = load_dataset()
    unflagged = np.array([labels[v]["needs_signal_check"] != "True" for v in ids])
    durations = {v: float(labels[v]["duration_sec"]) for v in ids}
    total_hours = sum(durations.values()) / 3600
    walls = json.load(open(OUT_DIR / "long_video_wall.json"))

    per_budget = {}
    for budget in BUDGETS:
        X, yb, idb, names, _ = load_dataset(OUT_DIR / f"{budget}.csv")
        assert idb == ids and (yb == y).all(), "row order differs: folds would not be paired"
        X = X[:, [names.index(f) for f in cheap.FEATURES]]  # exactly the 17 v1 features (drop cpu_sec, wall_sec)
        cpu_rows = {r["video_id"]: float(r["cpu_sec"]) for r in csv.DictReader(open(OUT_DIR / f"{budget}.csv"))}
        reps, folds = [], []
        for r in range(5):
            p = np.empty_like(y)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
                p[te] = make_model("logreg").fit(X[tr], y[tr]).predict(X[te])
                folds.append(f1_score(y[te], p[te], average="macro", labels=range(4), zero_division=0))
            reps.append((accuracy_score(y, p), f1_score(y, p, average="macro"),
                         accuracy_score(y[unflagged], p[unflagged])))
        c = np.array([cpu_rows[v] for v in ids])
        cpu_per_video_hour = c.sum() / total_hours
        busy = CAPACITY_HOURS_PER_DAY * cpu_per_video_hour / 86_400
        per_budget[budget] = {
            "budget": budget, "accuracy": np.mean([x[0] for x in reps]), "accuracy_std": np.std([x[0] for x in reps]),
            "macro_f1": np.mean([x[1] for x in reps]), "macro_f1_std": np.std([x[1] for x in reps]),
            "unflagged_acc": np.mean([x[2] for x in reps]),
            "cpu_s_per_video_mean": c.mean(), "cpu_s_per_video_p95": np.percentile(c, 95),
            "cpu_s_per_video_hour": cpu_per_video_hour,
            "long_video_wall_s_1core": walls[budget]["wall_sec"],
            "cores_10k_h_per_day": busy / UTILIZATION,
            "_rep_f1": np.array([x[1] for x in reps]), "_fold_f1": np.array(folds),
        }
    base = per_budget[SHIPPED]
    rows = []
    for b, r in per_budget.items():
        d_rep, d_fold = r["_rep_f1"] - base["_rep_f1"], r["_fold_f1"] - base["_fold_f1"]
        r.update(d_macro_f1=d_rep.mean(), d_macro_f1_std=d_rep.std(), folds_better=int((d_fold > 0).sum()),
                 folds_worse=int((d_fold < 0).sum()),
                 verdict="shipped" if b == SHIPPED else ("within noise" if abs(d_rep.mean()) < 0.03 else
                                                          ("better" if d_rep.mean() > 0 else "worse")))
        rows.append({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                     for k, v in r.items() if not k.startswith("_")})
    with open(RES / "budget_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    model_size_speed(RES / "model_size_speed.json")
    plot(rows, Path("docs/tradeoff.svg"))
    print(f"{'budget':14s}{'acc':>7s}{'mF1':>7s}{'unfl':>7s}{'CPU-s/vid':>10s}{'p95':>7s}{'CPU-s/h':>9s}"
          f"{'wall long':>10s}{'cores':>7s}{'dF1':>8s}{'folds+/-':>10s}  verdict")
    for r in rows:
        print(f"{r['budget']:14s}{r['accuracy']:>7.3f}{r['macro_f1']:>7.3f}{r['unflagged_acc']:>7.3f}"
              f"{r['cpu_s_per_video_mean']:>10.2f}{r['cpu_s_per_video_p95']:>7.2f}{r['cpu_s_per_video_hour']:>9.1f}"
              f"{r['long_video_wall_s_1core']:>10.1f}{r['cores_10k_h_per_day']:>7.1f}{r['d_macro_f1']:>+8.3f}"
              f"{r['folds_better']:>5d}/{r['folds_worse']:<4d}  {r['verdict']}")


def model_size_speed(out):
    import pickle
    from models import JsonLogReg, load_dataset, make_model
    X, y, _, _, _ = load_dataset()
    res = {}
    lr = JsonLogReg("artifacts/model/logreg_v1.json")
    gbm = make_model("lightgbm").fit(X, y)
    tmp = RES / "_lgbm_model.txt"
    gbm.booster_.save_model(str(tmp))
    for name, model, size in (("logreg_json", lr, Path("artifacts/model/logreg_v1.json").stat().st_size),
                              ("lightgbm_txt", gbm, tmp.stat().st_size)):
        model.predict_proba(X[:1])
        t0 = time.perf_counter()
        for _ in range(20):
            for i in range(len(X)):
                model.predict_proba(X[i:i + 1])
        res[name] = {"bytes_on_disk": size, "inference_ms_per_video": round((time.perf_counter() - t0) / (20 * len(X)) * 1000, 4)}
    res["lightgbm_pickle_bytes"] = len(pickle.dumps(gbm))
    tmp.unlink()
    json.dump(res, open(out, "w"), indent=1)
    print("model size/speed:", res)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for r in rows:
        shipped = r["budget"] == SHIPPED
        ax.errorbar(r["cpu_s_per_video_mean"], r["macro_f1"], yerr=r["macro_f1_std"], fmt="o",
                    color="#d62728" if shipped else "#1f77b4", ms=7 if shipped else 5, capsize=2)
        ax.annotate(r["budget"].replace("_", " ") + (" (shipped)" if shipped else ""),
                    (r["cpu_s_per_video_mean"], r["macro_f1"]), textcoords="offset points", xytext=(6, -10), fontsize=7)
    ax.set_xlabel("CPU-s per video (mean, one core)")
    ax.set_ylabel("macro-F1 (5×5 CV)")
    ax.set_title("Accuracy vs. analysis budget", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    print("plot ->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--jobs", type=int, default=6)
    ex.add_argument("--budgets", nargs="*")
    ex.add_argument("--force", action="store_true")
    sub.add_parser("evaluate")
    args = ap.parse_args()
    {"extract": extract, "evaluate": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
