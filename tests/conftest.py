"""Make src/ modules importable and run every test from the repo root (paths in src/ are repo-relative)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "classify", "features", "teacher"):
    sys.path.insert(0, str(ROOT / "src" / sub))

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """~20 s 320x240 test pattern with a 440 Hz tone, built-in codecs only (mpeg4 + aac)."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    out = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=20",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=20",
                    "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out
