"""Teacher signal extractors (offline, macOS / Apple Silicon only). NEVER classifier features.

    python src/teacher/extract.py                         # all signals, all videos
    python src/teacher/extract.py --only cuts --force     # recompute one signal
    python src/teacher/extract.py --videos ID1,ID2        # subset (e.g. smoke test)

Each signal is a raw per-window time series cached per video at
artifacts/teacher_cache/{signal}/v{version}/{video_id}.parquet. A video is skipped
if its cache file exists for the signal's current version. Only the windows in
gt/windows.csv are decoded (ffmpeg -ss before -i for fast seek).

Signals:
  speech  per second: speech_active, word_count, source ("srt" | "whisper")
  cuts    one row per cut: t_sec (PySceneDetect AdaptiveDetector, 320 px wide)
  motion  2 fps, 320 px, Farneback flow: flow_mag_mean, flow_mag_p90,
          global_motion (|median flow vector|), residual_motion (mean |flow - median|)
  text    0.5 fps, Apple Vision OCR via ocrmac: text_area_frac, n_boxes, text_hash
  speech_whisper  (only via --only) Whisper even when an .srt exists, for the
                  srt-vs-whisper consistency check; never feeds aggregation
"""
import argparse
import csv
import hashlib
import html
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windows import load_windows  # noqa: E402

CACHE_DIR = Path("artifacts/teacher_cache")
LOG_PATH = Path("data/logs/teacher_run.log")
TIMINGS_PATH = CACHE_DIR / "timings.csv"

SPEECH_VERSION = 1
CUTS_VERSION = 1
MOTION_VERSION = 1
TEXT_VERSION = 1

SMALL_WIDTH = 320
MOTION_FPS = 2.0
TEXT_FPS = 0.5
OCR_MIN_CONF = 0.5
WHISPER_MODEL = "mlx-community/whisper-small-mlx"
CUT_EDGE_SEC = 0.5  # cuts this close to a window edge are window artifacts, not cuts

log = logging.getLogger("teacher")


# ---------------------------------------------------------------- decoding

def decode_frames(path, start, dur, width, height, fps=None, gray=False):
    """Yield (frame_index, ndarray) for [start, start+dur) scaled to width x height."""
    vf = (f"fps={fps}," if fps else "") + f"scale={width}:{height}"
    pix, ch = ("gray", 1) if gray else ("bgr24", 3)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(path), "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = width * height * ch
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            shape = (height, width) if gray else (height, width, 3)
            yield i, np.frombuffer(buf, np.uint8).reshape(shape)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def decode_audio(path, start, dur, sr=16000):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32)


def scaled_size(meta, width):
    w, h = (int(x) for x in meta["resolution"].split("x"))
    if width >= w:
        return w, h
    return width, max(2, int(round(width * h / w / 2)) * 2)


# ---------------------------------------------------------------- speech

_TAG = re.compile(r"<[^>]+>")
_NONSPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪+")
_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt(path):
    """[(start, end, n_words)] for cues that contain speech words.

    YouTube auto-captions keep each cue on screen long after it is spoken, so the
    effective end is clipped to the next cue's start and to 1 s + 0.5 s/word.
    """
    cues = []
    for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8", errors="replace")):
        m = _TS.search(block)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(block[m.end():].splitlines())
        text = _NONSPEECH.sub(" ", html.unescape(_TAG.sub(" ", text)))
        n_words = len(re.findall(r"\w+", text))
        if n_words:
            cues.append([start, end, n_words])
    cues.sort()
    for i, c in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else c[1]
        c[1] = max(c[0] + 0.2, min(c[1], max(nxt, c[0] + 0.2), c[0] + 1.0 + 0.5 * c[2]))
    return [tuple(c) for c in cues]


def _bins(start, end):
    return np.arange(np.floor(start), np.ceil(end))


def extract_speech(video_id, meta, windows, force_whisper=False):
    srt = Path(f"data/subtitles/{video_id}.srt")
    rows = []
    if srt.is_file() and not force_whisper:
        cues = parse_srt(srt)
        for wi, (s, e) in enumerate(windows):
            for t in _bins(s, e):
                active, words = 0, 0.0
                for cs, ce, nw in cues:
                    ov = min(ce, t + 1) - max(cs, t)
                    if ov > 0:
                        active = 1
                        words += nw * ov / (ce - cs)
                rows.append((wi, float(t), active, round(words, 3), "srt"))
        return pd.DataFrame(rows, columns=["window_idx", "t_sec", "speech_active", "word_count", "source"])

    import mlx_whisper
    for wi, (s, e) in enumerate(windows):
        audio = decode_audio(meta["video_path"], s, e - s)
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL, word_timestamps=True,
                                     condition_on_previous_text=False, verbose=None)
        words = [w for seg in res.get("segments", [])
                 if not (seg.get("no_speech_prob", 0) > 0.6 and seg.get("avg_logprob", 0) < -1.0)
                 for w in seg.get("words", []) if re.search(r"\w", w.get("word", ""))]
        for t in _bins(s, e):
            rel = t - s
            hit = [w for w in words if w["start"] < rel + 1 and w["end"] > rel]
            n = sum(1 for w in words if rel <= w["start"] < rel + 1)
            rows.append((wi, float(t), int(bool(hit)), float(n), "whisper"))
    return pd.DataFrame(rows, columns=["window_idx", "t_sec", "speech_active", "word_count", "source"])


# ---------------------------------------------------------------- cuts

def extract_cuts(video_id, meta, windows):
    from scenedetect import AdaptiveDetector
    from scenedetect.common import FrameTimecode
    fps = float(meta["fps"])
    w, h = scaled_size(meta, SMALL_WIDTH)
    rows = []
    for wi, (s, e) in enumerate(windows):
        det = AdaptiveDetector()
        cut_frames, last = [], None
        for i, frame in decode_frames(meta["video_path"], s, e - s, w, h):
            last = FrameTimecode(i, fps)
            cut_frames += det.process_frame(last, frame)
        if last is not None:
            cut_frames += det.post_process(last)
        for tc in cut_frames:
            t = s + tc.frame_num / fps
            if s + CUT_EDGE_SEC < t < e - CUT_EDGE_SEC:
                rows.append((wi, round(t, 3)))
    return pd.DataFrame(rows, columns=["window_idx", "t_sec"]).astype({"window_idx": "int64", "t_sec": "float64"})


# ---------------------------------------------------------------- motion

def extract_motion(video_id, meta, windows):
    import cv2
    w, h = scaled_size(meta, SMALL_WIDTH)
    rows = []
    for wi, (s, e) in enumerate(windows):
        prev = None
        for i, frame in decode_frames(meta["video_path"], s, e - s, w, h, fps=MOTION_FPS, gray=True):
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.linalg.norm(flow, axis=2)
                med = np.median(flow.reshape(-1, 2), axis=0)
                resid = np.linalg.norm(flow - med, axis=2)
                rows.append((wi, round(s + i / MOTION_FPS, 3), float(mag.mean()),
                             float(np.percentile(mag, 90)), float(np.linalg.norm(med)),
                             float(resid.mean())))
            prev = frame
    return pd.DataFrame(rows, columns=["window_idx", "t_sec", "flow_mag_mean", "flow_mag_p90",
                                       "global_motion", "residual_motion"])


# ---------------------------------------------------------------- text

def normalize_text(texts):
    return " ".join(t for t in (re.sub(r"[^\w]+", " ", x.lower()).strip() for x in texts) if t)


def ocr_frame(frame_bgr):
    """[(text, conf, (x, y, w, h) normalized, origin bottom-left)] via Apple Vision."""
    from PIL import Image
    from ocrmac import ocrmac
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    res = ocrmac.OCR(img, recognition_level="accurate").recognize()
    return [(t, c, tuple(b)) for t, c, b in res if c >= OCR_MIN_CONF]


def extract_text(video_id, meta, windows):
    w, h = scaled_size(meta, 10_000)  # native resolution; OCR needs the detail
    rows = []
    for wi, (s, e) in enumerate(windows):
        for i, frame in decode_frames(meta["video_path"], s, e - s, w, h, fps=TEXT_FPS):
            boxes = ocr_frame(frame)
            area = min(1.0, sum(b[2] * b[3] for _, _, b in boxes))
            norm = normalize_text(t for t, _, _ in boxes)
            rows.append((wi, round(s + i / TEXT_FPS, 3), area, len(boxes),
                         hashlib.sha1(norm.encode()).hexdigest()[:12] if norm else "",
                         json.dumps([[t, round(c, 3), [round(x, 4) for x in b]] for t, c, b in boxes])))
    return pd.DataFrame(rows, columns=["window_idx", "t_sec", "text_area_frac", "n_boxes",
                                       "text_hash", "boxes_json"])


def extract_speech_whisper(video_id, meta, windows):
    """Whisper even when an .srt exists; only for the srt-vs-whisper consistency check."""
    return extract_speech(video_id, meta, windows, force_whisper=True)


EXTRACTORS = {
    "speech": (extract_speech, SPEECH_VERSION),
    "cuts": (extract_cuts, CUTS_VERSION),
    "motion": (extract_motion, MOTION_VERSION),
    "text": (extract_text, TEXT_VERSION),
    "speech_whisper": (extract_speech_whisper, SPEECH_VERSION),
}
DEFAULT_SIGNALS = ["speech", "cuts", "motion", "text"]  # speech_whisper only via --only


def cache_path(signal, video_id):
    return CACHE_DIR / signal / f"v{EXTRACTORS[signal][1]}" / f"{video_id}.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=sorted(EXTRACTORS), action="append")
    ap.add_argument("--videos", help="comma-separated video_ids (default: all)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--labels", default="gt/gt_labels_v0.csv")
    ap.add_argument("--windows", default="gt/windows.csv")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: ffmpeg not found on PATH")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()])

    with open(args.labels, newline="") as f:
        meta = {r["video_id"]: r for r in csv.DictReader(f)}
    windows = load_windows(args.windows)
    ids = args.videos.split(",") if args.videos else sorted(meta)
    signals = args.only or DEFAULT_SIGNALS

    TIMINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_timings = not TIMINGS_PATH.exists()
    with open(TIMINGS_PATH, "a", newline="") as tf:
        tw = csv.writer(tf)
        if new_timings:
            tw.writerow(["signal", "version", "video_id", "analysed_sec", "runtime_sec"])
        for n, vid in enumerate(ids, 1):
            analysed = sum(e - s for s, e in windows[vid])
            for sig in signals:
                fn, ver = EXTRACTORS[sig]
                out = cache_path(sig, vid)
                if out.exists() and not args.force:
                    continue
                t0 = time.time()
                try:
                    df = fn(vid, meta[vid], windows[vid])
                except Exception as e:  # keep going; failures are logged and retried next run
                    log.error("[%d/%d] %s %s FAILED: %r", n, len(ids), sig, vid, e)
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                df.to_parquet(tmp, index=False)
                tmp.replace(out)
                dt = time.time() - t0
                tw.writerow([sig, ver, vid, round(analysed, 1), round(dt, 2)])
                tf.flush()
                log.info("[%d/%d] %s %s: %d rows, %.1fs for %.0fs of video",
                         n, len(ids), sig, vid, len(df), dt, analysed)


if __name__ == "__main__":
    main()
