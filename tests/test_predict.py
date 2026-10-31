import json
import subprocess
import sys

from conftest import ROOT, needs_ffmpeg

from main import CONFIDENCE_THRESHOLD, FALLBACK_CHUNKER, decide
from models import CHUNKERS, CLASSES

REQUIRED = {"video", "predicted_class", "chunker", "probabilities", "confidence", "fallback_used",
            "feature_seconds"}


def test_confident_prediction_routes_to_class_chunker():
    d = decide([0.05, 0.05, 0.85, 0.05], CLASSES)
    assert d["predicted_class"] == "3_high_motion"
    assert d["chunker"] == CHUNKERS["3_high_motion"]
    assert d["fallback_used"] is False and d["confidence"] == "high"


def test_low_confidence_uses_fallback_chunker():
    d = decide([0.30, 0.25, 0.25, 0.20], CLASSES)  # max 0.30 < 0.5
    assert CONFIDENCE_THRESHOLD == 0.5
    assert d["predicted_class"] == "1_conversational"
    assert d["fallback_used"] is True and d["confidence"] == "low"
    assert d["chunker"] == FALLBACK_CHUNKER
    assert "0.30 < 0.5" in d["fallback_reason"]


def test_probabilities_are_reported_for_all_classes():
    d = decide([0.1, 0.2, 0.3, 0.4], CLASSES)
    assert list(d["probabilities"]) == CLASSES


@needs_ffmpeg
def test_cli_predict_outputs_json_with_required_keys(synthetic_video):
    out = subprocess.run([sys.executable, "src/main.py", "predict", "--video", str(synthetic_video)],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    res = json.loads(out)
    assert REQUIRED <= set(res)
    assert res["predicted_class"] in CLASSES
    assert abs(sum(res["probabilities"].values()) - 1) < 1e-3
    assert res["fallback_used"] == (max(res["probabilities"].values()) < CONFIDENCE_THRESHOLD)
