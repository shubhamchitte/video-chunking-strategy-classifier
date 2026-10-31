"""GT v1: reassign flagged videos to the nearest class anchor of teacher signals.

    python src/teacher/relabel.py   # -> gt/gt_labels_v1.csv, gt/class_anchors.csv, gt/relabel_sensitivity.csv

1. Robust-standardize each feature over all 90 videos: (x - median) / IQR.
2. Class anchor = per-class median of standardized features over the UNFLAGGED videos.
3. Flagged video: Euclidean distance to every anchor. Switch to the nearest class
   only if d_nearest < margin * d_v0_class, otherwise keep the v0 label.
4. Unflagged video: never changed; marked outlier_own_class if its distance to its
   own anchor exceeds the OUTLIER_PCTL percentile of all unflagged own-anchor distances.
5. Manual review overrides from gt/manual_overrides.csv (video_id, gt_class, reason)
   are applied last, with gt_source = "manual_review".

PRIMARY_RULE produces the labels. Every rule in RULES is also evaluated and written
to gt/relabel_sensitivity.csv so the effect of the feature set and margin is visible.
gt_labels_v0.csv is read, never modified. Teacher signals never become classifier features.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["1_conversational", "2_info_dense", "3_high_motion", "4_narrative"]
OUTLIER_PCTL = 90

# One feature per axis: speech, editing, camera motion, subject motion, on-screen text
# (central only, so captions, tickers and logos in the top/bottom 15% don't count).
FEATURES_5 = ["speech_coverage", "cuts_per_min", "global_motion_mean", "residual_motion_mean",
              "text_area_central"]
FEATURES_11 = ["speech_coverage", "words_per_min", "cuts_per_min", "motion_mean", "motion_p90",
               "global_motion_mean", "residual_motion_mean", "global_motion_ratio",
               "text_area_mean", "text_present_frac", "text_persistence"]
RULES = {
    "5feat_m0.7": (FEATURES_5, 0.7),
    "11feat_m0.8": (FEATURES_11, 0.8),  # first version, superseded; kept for sensitivity
}
PRIMARY_RULE = "5feat_m0.7"


def robust_z(df):
    med = df.median()
    iqr = df.quantile(0.75) - df.quantile(0.25)
    return (df - med) / iqr.replace(0, 1.0)


def apply_rule(sig, cls_v0, flagged, features, margin):
    """Return (labels DataFrame indexed by video_id, standardized anchors)."""
    z = robust_z(sig[features])
    unflag = ~flagged
    anchors = z[unflag].groupby(cls_v0[unflag]).median().reindex(CLASSES)
    dist = pd.DataFrame({c: np.linalg.norm(z.values - anchors.loc[c].values, axis=1) for c in CLASSES},
                        index=z.index)
    own = pd.Series({v: dist.at[v, cls_v0[v]] for v in z.index})
    thr = np.percentile(own[unflag], OUTLIER_PCTL)

    rows = {}
    for vid in z.index:
        c0, d = cls_v0[vid], dist.loc[vid]
        nearest = d.idxmin()
        ratio = d[nearest] / d[c0] if d[c0] > 0 else 1.0
        cls, outlier = c0, False
        if flagged[vid]:
            if nearest == c0:
                reason = "flagged; nearest is v0 class: kept"
            elif d[nearest] < margin * d[c0]:
                cls = nearest
                reason = f"flagged; nearest={nearest}, margin {ratio:.2f} < {margin}: switched"
            else:
                reason = f"flagged; nearest={nearest}, margin {ratio:.2f} >= {margin}: kept"
        else:
            outlier = own[vid] > thr
            reason = (f"unflagged; own-anchor distance {own[vid]:.2f} > p{OUTLIER_PCTL} {thr:.2f} "
                      f"(nearest={nearest}): reported, not changed") if outlier else "unflagged"
        rows[vid] = {"cls": cls, "ratio": ratio, "nearest": nearest, "outlier": outlier,
                     "reason": reason, **{f"d_{c}": d[c] for c in CLASSES}}
    return pd.DataFrame.from_dict(rows, orient="index"), anchors


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default="gt/gt_labels_v0.csv")
    ap.add_argument("--signals", default="gt/teacher_signals.csv")
    ap.add_argument("--overrides", default="gt/manual_overrides.csv")
    ap.add_argument("--out", default="gt/gt_labels_v1.csv")
    ap.add_argument("--anchors-out", default="gt/class_anchors.csv")
    ap.add_argument("--sensitivity-out", default="gt/relabel_sensitivity.csv")
    args = ap.parse_args()

    v0 = pd.read_csv(args.labels, keep_default_na=False)
    sig = pd.read_csv(args.signals).set_index("video_id").sort_index()
    if set(sig.index) != set(v0.video_id):
        raise SystemExit("teacher_signals.csv and gt_labels_v0.csv cover different videos")
    idx = v0.set_index("video_id").reindex(sig.index)
    flagged = idx["needs_signal_check"].astype(str).eq("True")
    cls_v0 = idx["gt_class"]

    results = {name: apply_rule(sig, cls_v0, flagged, f, m) for name, (f, m) in RULES.items()}
    res, _ = results[PRIMARY_RULE]

    # Sensitivity table: every flagged video under every rule.
    sens = pd.DataFrame({"sub_category": idx["sub_category"], "gt_class_v0": cls_v0})[flagged]
    for name, (r, _) in results.items():
        sens[f"{name}_class"] = r["cls"]
        sens[f"{name}_ratio"] = r["ratio"].round(3)
        sens[f"{name}_switched"] = r["cls"] != cls_v0
    sens.rename_axis("video_id").to_csv(args.sensitivity_out)

    out = v0.copy()
    r = res.reindex(out.video_id)
    out["gt_class_v0"] = out["gt_class"]
    out["gt_class_v1"] = r["cls"].values
    out["gt_source"] = np.where(r["cls"].values != out["gt_class_v0"].values, "teacher_anchor", "subcat_map")
    for c in CLASSES:
        out[f"d_{c}"] = r[f"d_{c}"].round(4).values
    out["outlier_own_class"] = r["outlier"].values
    out["reason"] = r["reason"].values

    n_manual = 0
    if Path(args.overrides).exists():
        ov = pd.read_csv(args.overrides, keep_default_na=False)
        for _, o in ov.iterrows():
            if o["gt_class"] not in CLASSES:
                raise SystemExit(f"override {o['video_id']}: unknown class {o['gt_class']}")
            m = out.video_id == o["video_id"]
            if not m.any():
                raise SystemExit(f"override {o['video_id']}: unknown video_id")
            out.loc[m, "reason"] = (f"manual_review: {o['reason']} (rule said {out.loc[m, 'gt_class_v1'].iloc[0]})")
            out.loc[m, "gt_class_v1"] = o["gt_class"]
            out.loc[m, "gt_source"] = "manual_review"
            n_manual += 1
    out["gt_class"] = out["gt_class_v1"]
    out["changed"] = out["gt_class_v1"] != out["gt_class_v0"]
    cols = [c for c in out.columns if c not in ("changed", "outlier_own_class", "reason")]
    out[cols[:cols.index("gt_class_v1") + 1] + ["changed"] + cols[cols.index("gt_class_v1") + 1:]
        + ["outlier_own_class", "reason"]].to_csv(args.out, index=False)

    # Anchor profile in raw units (per-class medians over unflagged videos), all features.
    raw = sig[FEATURES_11 + ["text_area_central"]][~flagged].groupby(cls_v0[~flagged]).median().reindex(CLASSES)
    raw.insert(0, "n_unflagged", (~flagged).groupby(cls_v0).sum().reindex(CLASSES))
    raw.round(4).to_csv(args.anchors_out)

    for name, (rr, _) in results.items():
        print(f"{name}: switched {int((rr['cls'] != cls_v0).sum())}/{int(flagged.sum())}, "
              f"outliers {int(rr['outlier'].sum())}")
    print(f"primary={PRIMARY_RULE}, manual overrides={n_manual}, changed={int(out.changed.sum())} -> {args.out}")
    print("v1 distribution:", out.gt_class_v1.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
