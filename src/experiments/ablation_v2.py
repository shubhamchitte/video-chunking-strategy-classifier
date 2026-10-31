"""Experiment 2 (+3): feature-group ablation on the SAME 5x5 folds as v1, plus a teacher-feature reference.

    python src/experiments/ablation_v2.py   # -> artifacts/results/ablation_v2.csv, ablation_v2_paired.csv

Logistic regression with the shipped settings. Paired comparison vs v1 on identical folds:
per-repeat (pooled out-of-fold) and per-fold macro-F1 differences. |diff| < NOISE is treated as noise.
Experiment 3: the 5 teacher features (gt/teacher_signals.csv) with (a) nearest class anchor
(robust-z and per-class medians fitted on training folds only) and (b) logistic regression.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classify"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from models import CLASSES, SEED, load_dataset, make_model  # noqa: E402
from cheap_v2 import GROUPS  # noqa: E402

V2_CSV = Path("artifacts/features/cheap_features_v2.csv")
TEACHER_CSV = Path("gt/teacher_signals.csv")
TEACHER_5 = ["speech_coverage", "cuts_per_min", "global_motion_mean", "residual_motion_mean", "text_area_central"]
N_SPLITS, N_REPEATS, NOISE = 5, 5, 0.03
OUT = Path("artifacts/results")


class NearestAnchor:
    """Robust-z (train median/IQR) + per-class median anchors; predict nearest anchor (Euclidean)."""

    def fit(self, X, y):
        self.med = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        self.iqr = np.where(iqr == 0, 1.0, iqr)
        z = (X - self.med) / self.iqr
        self.anchors = np.stack([np.median(z[y == k], axis=0) for k in range(len(CLASSES))])
        return self

    def predict(self, X):
        z = (X - self.med) / self.iqr
        return np.linalg.norm(z[:, None, :] - self.anchors[None], axis=2).argmin(axis=1)


def run(X, y, factory, unflagged):
    reps, folds = [], []
    for r in range(N_REPEATS):
        p = np.empty_like(y)
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + r).split(X, y):
            p[te] = factory().fit(X[tr], y[tr]).predict(X[te])
            folds.append(f1_score(y[te], p[te], average="macro", labels=range(4), zero_division=0))
        f1c = f1_score(y, p, average=None, labels=range(4), zero_division=0)
        reps.append({"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
                     **{f"f1_{c}": v for c, v in zip(CLASSES, f1c)},
                     "unflagged_acc": accuracy_score(y[unflagged], p[unflagged]),
                     "unflagged_macro_f1": f1_score(y[unflagged], p[unflagged], average="macro")})
    return reps, np.array(folds)


def main():
    X, y, ids, names, labels = load_dataset(V2_CSV)
    X1, y1, ids1, names1, _ = load_dataset()
    assert ids == ids1 and (y == y1).all(), "v1/v2 row order differs: folds would not be paired"
    unflagged = np.array([labels[v]["needs_signal_check"] != "True" for v in ids])
    col = {n: i for i, n in enumerate(names)}
    v1 = names1
    rows_v2 = list(csv.DictReader(open(V2_CSV)))
    analysed_min = sum(float(r["analysed_sec"]) for r in rows_v2) / 60
    extra_cpu = {g: sum(float(r[f"cpu_{g}"]) for r in rows_v2) / analysed_min for g in GROUPS}

    def cols(groups):
        return [col[c] for c in v1 + [c for g in groups for c in GROUPS[g]]]

    sets = {"v1": []} | {f"v1+{g}": [g] for g in GROUPS} | {"v1+all": list(GROUPS)}
    results = {name: run(X[:, cols(gs)], y, lambda: make_model("logreg"), unflagged) for name, gs in sets.items()}
    base_rep = np.array([r["macro_f1"] for r in results["v1"][0]])
    base_fold = results["v1"][1]
    best = [g for g in GROUPS if np.mean([r["macro_f1"] for r in results[f"v1+{g}"][0]]) - base_rep.mean() > 0]
    if best and len(best) < len(GROUPS):
        sets[f"v1+best({''.join(best)})"] = best
        results[f"v1+best({''.join(best)})"] = run(X[:, cols(best)], y, lambda: make_model("logreg"), unflagged)

    # Experiment 3: teacher features (reference; macOS-only tools, ~5.6 CPU-s/analysed min).
    teach = {r["video_id"]: r for r in csv.DictReader(open(TEACHER_CSV))}
    T = np.array([[float(teach[v][f]) for f in TEACHER_5] for v in ids])
    results["teacher5_nearest_anchor"] = run(T, y, NearestAnchor, unflagged)
    results["teacher5_logreg"] = run(T, y, lambda: make_model("logreg"), unflagged)

    out_rows, paired = [], []
    for name, (reps, folds) in results.items():
        row = {"feature_set": name, "n_features": X[:, cols(sets[name])].shape[1] if name in sets else len(TEACHER_5)}
        for k in reps[0]:
            row[k] = round(float(np.mean([r[k] for r in reps])), 4)
        row["macro_f1_std"] = round(float(np.std([r["macro_f1"] for r in reps])), 4)
        gs = sets.get(name, [])
        row["extra_cpu_s_per_min"] = round(sum(extra_cpu[g] for g in gs), 2) if name in sets else None
        out_rows.append(row)
        d_rep = np.array([r["macro_f1"] for r in reps]) - base_rep
        d_fold = folds - base_fold
        paired.append({"feature_set": name, "d_macro_f1_mean": round(d_rep.mean(), 4),
                       "d_macro_f1_std_repeats": round(d_rep.std(), 4),
                       "repeats_better": int((d_rep > 0).sum()), "folds_better": int((d_fold > 0).sum()),
                       "folds_worse": int((d_fold < 0).sum()),
                       "verdict": "noise" if abs(d_rep.mean()) < NOISE else ("better" if d_rep.mean() > 0 else "worse")})
    for fname, data in (("ablation_v2.csv", out_rows), ("ablation_v2_paired.csv", paired)):
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print(f"{'feature set':26s}{'n':>4s}{'acc':>7s}{'mF1':>7s}{'conv':>7s}{'info':>7s}{'motion':>7s}{'narr':>7s}"
          f"{'unfl acc':>9s}{'+CPU-s/min':>11s}{'dF1':>8s}{'folds +/-':>11s}  verdict")
    for r, p in zip(out_rows, paired):
        cpu = "" if r["extra_cpu_s_per_min"] is None else f"{r['extra_cpu_s_per_min']:.2f}"
        print(f"{r['feature_set']:26s}{r['n_features']:>4d}{r['accuracy']:>7.3f}{r['macro_f1']:>7.3f}"
              f"{r['f1_1_conversational']:>7.3f}{r['f1_2_info_dense']:>7.3f}{r['f1_3_high_motion']:>7.3f}"
              f"{r['f1_4_narrative']:>7.3f}{r['unflagged_acc']:>9.3f}{cpu:>11s}{p['d_macro_f1_mean']:>+8.3f}"
              f"{p['folds_better']:>6d}/{p['folds_worse']:<4d}  {p['verdict']}")
    print("extra CPU-s per analysed minute by group:", {g: round(v, 2) for g, v in extra_cpu.items()})


if __name__ == "__main__":
    main()
