import pytest

from windows import N_WINDOWS, WINDOW_SEC, windows_for


@pytest.mark.parametrize("duration", [5.0, 47.02, 599.9, 600.0])
def test_short_video_is_one_whole_window(duration):
    assert windows_for(duration) == [(0.0, round(duration, 3))]


@pytest.mark.parametrize("duration", [600.1, 746.0, 2656.5, 3552.1])
def test_long_video_has_ten_60s_windows_within_5_to_95_percent(duration):
    wins = windows_for(duration)
    assert len(wins) == N_WINDOWS
    for s, e in wins:
        assert e - s == pytest.approx(WINDOW_SEC, abs=1e-3)
        assert s >= 0.05 * duration - 1e-3
        assert e <= 0.95 * duration + 1e-3
    starts = [s for s, _ in wins]
    assert starts == sorted(starts)
    assert starts[0] == pytest.approx(0.05 * duration, abs=1e-3)
    assert wins[-1][1] == pytest.approx(0.95 * duration, abs=1e-3)
