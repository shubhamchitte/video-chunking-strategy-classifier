"""Dataset loading and model definitions shared by evaluation and prediction."""
import csv
from pathlib import Path

import numpy as np

FEATURES_CSV = Path("artifacts/features/cheap_features.csv")
LABELS_CSV = Path("gt/gt_labels_v1.csv")
CLASSES = ["1_conversational", "2_info_dense", "3_high_motion", "4_narrative"]
CHUNKERS = {
    "1_conversational": "semantic_transcript",
    "2_info_dense": "visual_keyframe_change",
    "3_high_motion": "action_peak_audio_motion",
    "4_narrative": "shot_scene_boundary",
}
SEED = 42
DEFAULT_MODEL = "logreg"  # chosen by CV macro-F1, see docs/results.md


def load_dataset(features_csv=FEATURES_CSV, labels_csv=LABELS_CSV):
    """Return (X, y, video_ids, feature_names, labels_by_id). Only cheap features enter X."""
    with open(features_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    names = [c for c in rows[0] if c not in ("video_id", "analysed_sec", "extract_sec")]
    with open(labels_csv, newline="") as f:
        labels = {r["video_id"]: r for r in csv.DictReader(f)}
    rows = [r for r in rows if r["video_id"] in labels]
    X = np.array([[float(r[n]) for n in names] for r in rows])
    y = np.array([CLASSES.index(labels[r["video_id"]]["gt_class"]) for r in rows])
    return X, y, [r["video_id"] for r in rows], names, labels


def make_model(name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED))
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=5,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              class_weight="balanced", random_state=SEED, n_jobs=1, verbose=-1)
    raise ValueError(name)


def feature_schedule(features_csv=FEATURES_CSV):
    """Analysis schedule the features were extracted with (sidecar written by src/features/cheap.py)."""
    import json
    return json.loads(Path(features_csv).with_suffix(".meta.json").read_text())["schedule"]


def export_logreg(pipeline, feature_names, path, schedule):
    """Save a fitted StandardScaler+LogisticRegression as plain JSON (portable, no pickle).

    `schedule` (windows, fps) is stored so predict can refuse features computed differently.
    """
    import json
    import sklearn
    scaler, lr = pipeline[0], pipeline[-1]
    model = {
        "format": "logreg-json-v1", "classes": [CLASSES[i] for i in lr.classes_],
        "features": list(feature_names), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
        "trained_on": {"features": str(FEATURES_CSV), "labels": str(LABELS_CSV), "n_videos": int(scaler.n_samples_seen_)},
        "sklearn_version": sklearn.__version__,
        "schedule": schedule,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(model, indent=1))


class JsonLogReg:
    """Multinomial logistic regression from export_logreg JSON: softmax(((x - mean) / scale) @ coef.T + b)."""

    def __init__(self, path):
        import json
        m = json.loads(Path(path).read_text())
        self.classes, self.features = m["classes"], m["features"]
        self.mean, self.scale = np.array(m["mean"]), np.array(m["scale"])
        self.coef, self.intercept = np.array(m["coef"]), np.array(m["intercept"])
        self.schedule = m.get("schedule")

    def predict_proba(self, X):
        z = ((np.atleast_2d(X) - self.mean) / self.scale) @ self.coef.T + self.intercept
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)
