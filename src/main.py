"""Video -> chunking-strategy classifier.

    python src/main.py evaluate                  # CV results + baselines from committed features (no videos)
    python src/main.py predict --video FILE.mp4  # JSON prediction for one raw video (needs ffmpeg/ffprobe)

Classes and the chunker each one routes to:
    1_conversational -> semantic_transcript        3_high_motion -> action_peak_audio_motion
    2_info_dense     -> visual_keyframe_change     4_narrative   -> shot_scene_boundary
Low-confidence predictions (max probability < 0.5) fall back to shot detection + transcript.
"""
import argparse
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC / "classify"), str(SRC / "features")]

MODEL_PATH = Path("artifacts/model/logreg_v1.json")
CONFIDENCE_THRESHOLD = 0.5
FALLBACK_CHUNKER = "shot_scene_boundary+semantic_transcript"


def cmd_evaluate(args):
    import evaluate
    evaluate.main(["--out-dir", args.out_dir, "--model-out", str(args.model)])


def decide(proba, classes, threshold=CONFIDENCE_THRESHOLD):
    """Map class probabilities to the routing decision (pure; no I/O)."""
    from models import CHUNKERS
    best = max(range(len(classes)), key=lambda i: proba[i])
    cls = classes[best]
    fallback = bool(proba[best] < threshold)
    return {
        "predicted_class": cls,
        "chunker": FALLBACK_CHUNKER if fallback else CHUNKERS[cls],
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "confidence": "low" if fallback else "high",
        "fallback_used": fallback,
        "fallback_reason": (f"max probability {proba[best]:.2f} < {threshold}; "
                            f"using default chunker instead of {CHUNKERS[cls]}") if fallback else None,
    }


def predict_video(video, model_path=MODEL_PATH):
    """Full prediction for one video file: features -> model -> routing decision."""
    from cheap import SCHEDULE, extract
    from models import JsonLogReg

    model = JsonLogReg(model_path)
    if model.schedule != SCHEDULE:
        raise RuntimeError(f"model was trained on features with schedule {model.schedule}, "
                           f"but the extractor uses {SCHEDULE}; re-extract features and rerun evaluate")
    t0 = time.perf_counter()
    feats, analysed = extract(video)
    feature_sec = time.perf_counter() - t0
    proba = model.predict_proba([[feats[n] for n in model.features]])[0]
    return {"video": str(video), **decide(proba, model.classes),
            "feature_seconds": round(feature_sec, 2), "analysed_seconds": round(analysed, 1)}


def cmd_predict(args):
    if not Path(args.video).is_file():
        sys.exit(f"ERROR: no such file: {args.video}")
    print(json.dumps(predict_video(args.video, args.model), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="cross-validated results from committed features")
    ev.add_argument("--out-dir", default="artifacts/results")
    ev.add_argument("--model", type=Path, default=MODEL_PATH, help="where to write the final model")
    pr = sub.add_parser("predict", help="classify one raw video")
    pr.add_argument("--video", required=True)
    pr.add_argument("--model", type=Path, default=MODEL_PATH)
    args = ap.parse_args()
    {"evaluate": cmd_evaluate, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
