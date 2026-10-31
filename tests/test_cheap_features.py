import math

from conftest import needs_ffmpeg

from cheap import FEATURES, extract


@needs_ffmpeg
def test_extract_returns_all_17_finite_features(synthetic_video):
    feats, analysed = extract(synthetic_video)
    assert len(FEATURES) == 17
    assert list(feats) == FEATURES
    assert all(math.isfinite(v) for v in feats.values()), feats
    assert analysed == pytest_approx(20.0)


@needs_ffmpeg
def test_synthetic_content_gives_expected_signals(synthetic_video):
    feats, _ = extract(synthetic_video)
    # A steady sine tone is loud, not silent, and not syllabic speech.
    assert feats["silence_frac"] < 0.05
    assert feats["vad_speech_ratio"] < 0.2
    # testsrc is a moving pattern with sharp edges and no cuts.
    assert feats["edge_mean"] > 0.01
    assert feats["static_frac"] < 0.5


def pytest_approx(x, tol=0.5):
    import pytest
    return pytest.approx(x, abs=tol)
