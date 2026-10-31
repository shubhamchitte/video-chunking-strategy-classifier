"""The model must be used on features computed with exactly the schedule it was trained on."""
import json

import pytest
from conftest import needs_ffmpeg

import cheap
from cheap import SCHEDULE, windows_for
from models import JsonLogReg, feature_schedule

MODEL = "artifacts/model/logreg_v1.json"


def test_shipped_schedule_is_3x30s_at_2fps():
    assert SCHEDULE["n_windows"] == 3 and SCHEDULE["window_sec"] == 30.0 and SCHEDULE["video_fps"] == 2.0


@pytest.mark.parametrize("duration", [20.0, 89.9, 90.0])
def test_short_video_is_analysed_whole(duration):
    assert windows_for(duration) == [(0.0, round(duration, 3))]


@pytest.mark.parametrize("duration", [90.1, 600.0, 3552.1])
def test_long_video_has_three_30s_windows_within_5_to_95_percent(duration):
    wins = windows_for(duration)
    assert len(wins) == 3
    assert all(e - s == pytest.approx(30.0, abs=1e-3) for s, e in wins)
    assert wins[0][0] == pytest.approx(0.05 * duration, abs=1e-3)
    assert wins[-1][1] == pytest.approx(0.95 * duration, abs=1e-3)


def test_model_features_and_extractor_share_one_schedule():
    """Training features (sidecar), model JSON and the predict-time extractor all agree."""
    assert feature_schedule() == SCHEDULE
    assert JsonLogReg(MODEL).schedule == SCHEDULE


def test_committed_features_were_extracted_with_this_schedule():
    """analysed_sec in the committed CSV equals what the schedule gives for each video's duration."""
    import csv
    labels = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(open("gt/gt_labels_v1.csv"))}
    for r in csv.DictReader(open("artifacts/features/cheap_features.csv")):
        expected = sum(e - s for s, e in windows_for(labels[r["video_id"]]))
        assert float(r["analysed_sec"]) == pytest.approx(expected, abs=0.01), r["video_id"]


@needs_ffmpeg
def test_predict_refuses_model_with_a_different_schedule(tmp_path, synthetic_video):
    from main import predict_video
    m = json.loads(open(MODEL).read())
    m["schedule"] = dict(m["schedule"], n_windows=10, window_sec=60.0)
    bad = tmp_path / "model.json"
    bad.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="schedule"):
        predict_video(synthetic_video, bad)


def test_predict_checks_schedule_before_extracting(tmp_path, monkeypatch):
    """Mismatch is caught without decoding anything (no ffmpeg needed)."""
    from main import predict_video
    monkeypatch.setattr(cheap, "extract", lambda *_: pytest.fail("extract must not run"))
    m = json.loads(open(MODEL).read())
    m["schedule"] = dict(m["schedule"], video_fps=1.0)
    bad = tmp_path / "model.json"
    bad.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="schedule"):
        predict_video("unused.mp4", bad)
