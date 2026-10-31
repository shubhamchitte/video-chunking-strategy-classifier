"""Repeated stratified 5-fold CV of the cheap-feature classifiers vs baselines.

    python src/classify/evaluate.py   # -> artifacts/results/

Splits are stratified on gt_class (GT v1), 5 folds x 5 repeats, fixed seeds.
Metrics are computed on each repeat's pooled out-of-fold predictions, reported as
mean +- std over repeats. The unflagged subset (68 videos) is scored from the same
out-of-fold predictions. The sub_category mapping is a metadata *oracle* baseline:
it uses information the classifier never sees.
"""
import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import CLASSES, SEED, JsonLogReg, export_logreg, feature_schedule, load_dataset, make_model  # noqa: E402

# Older lightgbm/sklearn pairs warn about feature names on plain numpy input; harmless.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

LATENCY_REPS = 20
N_SPLITS = 5
N_REPEATS = 5
MODELS = ["logreg", "lightgbm"]
BASELINES = ["majority", "always_shot_detection", "subcat_oracle"]


def oof_predictions(name, X, y, seed, v0_idx):
    """Out-of-fold class predictions, plus probabilities for trained models (else None)."""
    pred = np.empty_like(y)
    proba = np.zeros((len(y), len(CLASSES))) if name in MODELS else None
    for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y):
        if name == "majority":
            pred[te] = np.bincount(y[tr], minlength=len(CLASSES)).argmax()
        elif name == "always_shot_detection":
            pred[te] = CLASSES.index("4_narrative")
        elif name == "subcat_oracle":
            pred[te] = v0_idx[te]
        else:
            m = make_model(name).fit(X[tr], y[tr])
            proba[te] = m.predict_proba(X[te])
            pred[te] = proba[te].argmax(axis=1)
    return pred, proba


MERGE = {0: 0, 1: 0, 2: 1, 3: 2}  # conversational + info_dense -> one "talk/explain" class
CONF_BINS = [0.0, 0.5, 0.7, 1.01]


def merged_scores(y, p):
    ym, pm = np.vectorize(MERGE.get)(y), np.vectorize(MERGE.get)(p)
    return {"accuracy": accuracy_score(ym, pm), "macro_f1": f1_score(ym, pm, average="macro")}


def confidence_table(y, probas):
    """Pooled over repeats: coverage and accuracy per max-probability bin."""
    conf = np.concatenate([pr.max(axis=1) for pr in probas])
    hit = np.concatenate([pr.argmax(axis=1) == y for pr in probas])
    rows = []
    for lo, hi in zip(CONF_BINS, CONF_BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        rows.append({"bin": f"[{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi > 1 else ')'}", "share": round(m.mean(), 3),
                     "accuracy": round(hit[m].mean(), 3) if m.any() else None})
    return rows


def scores(y, p):
    return {"accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", labels=range(len(CLASSES)), zero_division=0),
            **{f"f1_{c}": f for c, f in zip(CLASSES, f1_score(y, p, average=None, labels=range(len(CLASSES)),
                                                            zero_division=0))}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="artifacts/results")
    ap.add_argument("--model-out", default="artifacts/model/logreg_v1.json")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, y, ids, names, labels = load_dataset()
    v0_idx = np.array([CLASSES.index(labels[v]["gt_class_v0"]) for v in ids])
    unflagged = np.array([labels[v]["needs_signal_check"] != "True" for v in ids])
    print(f"{len(ids)} videos, {len(names)} features, classes {np.bincount(y).tolist()}, "
          f"unflagged {unflagged.sum()}")

    summary, oof_rows, confusions, analyses = [], {}, {}, {"merged_conv_info": {}, "confidence": {}}
    for name in MODELS + BASELINES:
        per_repeat = {"all": [], "unflagged": []}
        merged, probas = [], []
        t0 = time.time()
        for r in range(N_REPEATS):
            p, pr = oof_predictions(name, X, y, SEED + r, v0_idx)
            per_repeat["all"].append(scores(y, p))
            per_repeat["unflagged"].append(scores(y[unflagged], p[unflagged]))
            merged.append(merged_scores(y, p))
            if pr is not None:
                probas.append(pr)
            if r == 0:
                oof_rows[name] = p
                confusions[name] = confusion_matrix(y, p, labels=range(len(CLASSES)))
        fit_sec = time.time() - t0
        analyses["merged_conv_info"][name] = {k: round(float(np.mean([m[k] for m in merged])), 4) for k in merged[0]}
        if probas:
            analyses["confidence"][name] = confidence_table(y, probas)
        for subset, reps in per_repeat.items():
            row = {"model": name, "subset": subset, "n": int(len(y) if subset == "all" else unflagged.sum())}
            for k in reps[0]:
                vals = [rr[k] for rr in reps]
                row[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{k}_std"] = round(float(np.std(vals)), 4)
            row["cv_wall_sec"] = round(fit_sec, 2)
            summary.append(row)
        a = [x for x in summary if x["model"] == name]
        print(f"{name:22s} acc {a[0]['accuracy_mean']:.3f}±{a[0]['accuracy_std']:.3f}  "
              f"macroF1 {a[0]['macro_f1_mean']:.3f}±{a[0]['macro_f1_std']:.3f}  "
              f"| unflagged acc {a[1]['accuracy_mean']:.3f} macroF1 {a[1]['macro_f1_mean']:.3f}")

    with open(out / "cv_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(out / "oof_predictions_repeat0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "gt_class", "needs_signal_check", *oof_rows])
        for i, v in enumerate(ids):
            w.writerow([v, CLASSES[y[i]], not unflagged[i], *(CLASSES[oof_rows[m][i]] for m in oof_rows)])
    json.dump({m: {"labels": CLASSES, "rows_true_cols_pred": c.tolist()} for m, c in confusions.items()},
              open(out / "confusion_repeat0.json", "w"), indent=1)

    # Interpretability: full-data fits (not used for scoring).
    # (coefficients are on standardized features, so magnitudes are comparable)
    lr = make_model("logreg").fit(X, y)
    coef = lr[-1].coef_
    export_logreg(lr, names, args.model_out, feature_schedule())  # final model used by `main.py predict`
    gbm = make_model("lightgbm").fit(X, y)
    gain = gbm.booster_.feature_importance("gain")
    with open(out / "feature_importance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "lgbm_gain_share", *[f"logreg_coef_{c}" for c in CLASSES]])
        for j, n in enumerate(names):
            w.writerow([n, round(gain[j] / gain.sum(), 4), *(round(coef[k, j], 3) for k in range(len(CLASSES)))])

    # Inference latency (features already extracted): the shipped JSON logreg exactly as `predict` runs it,
    # and LightGBM for comparison. One video per call, LATENCY_REPS passes over all videos, one process.
    lat = {}
    for m, model in (("logreg_json", JsonLogReg(args.model_out)), ("lightgbm", gbm)):
        model.predict_proba(X[:1])
        t0 = time.perf_counter()
        for _ in range(LATENCY_REPS):
            for i in range(len(X)):
                model.predict_proba(X[i:i + 1])
        lat[m] = (time.perf_counter() - t0) / (LATENCY_REPS * len(X)) * 1000
    json.dump({"inference_ms_per_video": {k: round(v, 4) for k, v in lat.items()},
               "method": f"single-row predict_proba, {LATENCY_REPS} x {len(X)} calls, one process, features precomputed",
               "model_bytes": {"logreg_json": Path(args.model_out).stat().st_size}},
              open(out / "inference_latency.json", "w"), indent=1)
    top = {c: sorted(zip(names, coef[k]), key=lambda t: -abs(t[1]))[:3] for k, c in enumerate(CLASSES)}
    analyses["logreg_top_coefficients"] = {c: [[n, round(float(v), 3)] for n, v in t] for c, t in top.items()}
    json.dump(analyses, open(out / "analyses.json", "w"), indent=1)
    print_report(summary, analyses, lat)
    print(f"results -> {out}/")


def print_report(summary, analyses, lat):
    line = "-" * 91
    print(f"\n{line}\nCV results: {N_SPLITS}-fold stratified x {N_REPEATS} repeats on GT v1 (mean ± std over repeats)\n{line}")
    print(f"{'model':24s}{'accuracy':>16s}{'macro-F1':>16s}{'unflagged acc':>16s}{'merged 3-cls acc':>19s}")
    for name in MODELS + BASELINES:
        a, u = [r for r in summary if r["model"] == name]
        print(f"{name:24s}{a['accuracy_mean']:>9.3f} ± {a['accuracy_std']:.3f}{a['macro_f1_mean']:>9.3f} ± "
              f"{a['macro_f1_std']:.3f}{u['accuracy_mean']:>16.3f}{analyses['merged_conv_info'][name]['accuracy']:>19.3f}")
    print(f"{line}\nlogreg confidence vs accuracy (pooled over repeats):")
    for row in analyses["confidence"]["logreg"]:
        print(f"  max prob {row['bin']:12s} share {row['share']:.2f}  accuracy {row['accuracy']}")
    print("logreg top coefficients (standardized features):")
    for c, t in analyses["logreg_top_coefficients"].items():
        print(f"  {c:18s} " + ", ".join(f"{n} {v:+.2f}" for n, v in t))
    print("inference ms/video: " + ", ".join(f"{k} {v:.4f}" for k, v in lat.items()) + f"\n{line}")


if __name__ == "__main__":
    main()
