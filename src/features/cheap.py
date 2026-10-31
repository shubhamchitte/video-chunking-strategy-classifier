"""Cheap, cross-platform, CPU-only features computed from the raw video only.

    python src/features/cheap.py                 # all videos in gt/gt_labels_v1.csv -> artifacts/features/cheap_features.csv
    python src/features/cheap.py --jobs 4        # parallel over videos
    python src/features/cheap.py --video data/videos/X.mp4   # one file, prints JSON

Never reads subtitles, the teacher cache, domain or sub_category. Needs
ffmpeg/ffprobe on PATH plus numpy and opencv-python-headless.

Analysis budget (SCHEDULE): 3 windows x 30 s at 2 fps, evenly spaced within 5-95% of the
duration; videos <= 90 s are analysed whole. Chosen in src/experiments/budget_sweep.py as the
cheapest budget within noise of the best (the teacher's GT windows, src/teacher/windows.py,
are a separate, larger schedule). The schedule is written next to the features
(*.meta.json), copied into the model JSON, and checked at predict time.

Audio (16 kHz mono, numpy): speech-vs-music and loudness.
Video (2 fps, 160 px wide, grayscale): editing, motion, edge density.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

N_WINDOWS = 3
WINDOW_SEC = 30.0
LO_FRAC, HI_FRAC = 0.05, 0.95
SR = 16000
N_FFT = 512                 # 32 ms
HOP = 160                   # 10 ms -> 100 frames/s
VIDEO_FPS = 2.0
VIDEO_WIDTH = 160
HIST_BINS = 32
HIST_JUMP = 0.3             # histogram L1/2 distance counted as an abrupt change
STATIC_DIFF = 0.01          # mean |frame diff| / 255 below this = static
SILENCE_DB = -45.0          # dBFS
# ffmpeg threads per decode (0 = ffmpeg default/auto). Set CHEAP_FFMPEG_THREADS=1 for one-core workers.
FFMPEG_THREADS = os.environ.get("CHEAP_FFMPEG_THREADS", "0")
ONSET_RISE_DB = 9.0         # loudness rise within ONSET_LAG frames counted as an onset
ONSET_LAG = 10              # frames (100 ms)
VAD_MOD4 = 0.6              # 1 s block is speech-like if 2-8 Hz modulation share exceeds this
SCHEDULE = {"n_windows": N_WINDOWS, "window_sec": WINDOW_SEC, "video_fps": VIDEO_FPS,
            "span": [LO_FRAC, HI_FRAC], "video_width": VIDEO_WIDTH}

FEATURES = [
    # audio
    "vad_speech_ratio", "mod4_energy", "flatness_mean", "flatness_std",
    "rms_db_mean", "rms_db_std", "rms_db_p90", "silence_frac", "onset_rate",
    # video
    "hist_delta_mean", "hist_delta_p90", "hist_jump_rate",
    "frame_diff_mean", "frame_diff_p90", "static_frac", "edge_mean", "edge_std",
]


# ---------------------------------------------------------------- schedule

def windows_for(duration):
    """[(start_sec, end_sec)]: whole video if it fits the budget, else N_WINDOWS evenly in 5-95%."""
    if duration <= N_WINDOWS * WINDOW_SEC:
        return [(0.0, round(duration, 3))]
    if N_WINDOWS == 1:
        s = 0.5 * duration - WINDOW_SEC / 2
        return [(round(s, 3), round(s + WINDOW_SEC, 3))]
    first, last = LO_FRAC * duration, HI_FRAC * duration - WINDOW_SEC
    step = (last - first) / (N_WINDOWS - 1)
    return [(round(first + i * step, 3), round(first + i * step + WINDOW_SEC, 3)) for i in range(N_WINDOWS)]


# ---------------------------------------------------------------- decoding

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    return float(info["format"]["duration"]), int(v["width"]), int(v["height"]), has_audio


def decode_audio(path, start, dur):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", FFMPEG_THREADS, "-ss", f"{start:.3f}",
           "-t", f"{dur:.3f}", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)


def decode_gray(path, start, dur, w, h):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", FFMPEG_THREADS, "-filter_threads", "1"
           if FFMPEG_THREADS == "1" else "0", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(path), "-an", "-vf", f"fps={VIDEO_FPS},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[: n * w * h], np.uint8).reshape(n, h, w)


# ---------------------------------------------------------------- audio

def audio_frames(x):
    """Per-10 ms frame stats: rms_db, spectral flatness, 300-3400 Hz band energy."""
    if len(x) < N_FFT:
        return None
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(N_FFT)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    band = (freqs >= 300) & (freqs <= 3400)
    rms_db = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    band_energy = power[:, band].sum(axis=1)
    return rms_db, flatness, band_energy


BLOCK = 100  # 1 s at 100 frames/s


def mod4_blocks(band_energy):
    """Per 1 s block: share of envelope modulation energy at 2-8 Hz (syllable rate) vs 0.5-25 Hz.

    Speech modulates loudness at ~4 Hz; sustained music and noise mostly do not.
    """
    env = np.log(band_energy + 1e-9)
    f = np.fft.rfftfreq(BLOCK, 1 / 100)
    vals = []
    for i in range(0, len(env) - BLOCK + 1, BLOCK):
        seg = env[i:i + BLOCK] - env[i:i + BLOCK].mean()
        spec = np.abs(np.fft.rfft(seg * np.hanning(BLOCK))) ** 2
        tot = spec[(f >= 0.5) & (f <= 25)].sum()
        vals.append(spec[(f >= 2) & (f <= 8)].sum() / tot if tot > 0 else 0.0)
    return np.array(vals)


def onsets(rms_db):
    """Count abrupt loudness rises (>= ONSET_RISE_DB within ONSET_LAG frames): hits, whistles, crowd bursts."""
    rise = (rms_db[ONSET_LAG:] - rms_db[:-ONSET_LAG]) > ONSET_RISE_DB
    return int(np.sum(np.diff(rise.astype(np.int8)) == 1))


# ---------------------------------------------------------------- video

def video_stats(frames):
    import cv2
    hist = np.stack([np.histogram(f, bins=HIST_BINS, range=(0, 256))[0] / f.size for f in frames])
    hd = 0.5 * np.abs(np.diff(hist, axis=0)).sum(axis=1)
    fd = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2)) / 255.0
    edges = np.array([(cv2.Canny(f, 100, 200) > 0).mean() for f in frames])
    return hd, fd, edges


# ---------------------------------------------------------------- per video

def extract(path, duration=None, width=None, height=None, has_audio=None):
    """Return (feature dict, analysed_sec) for one video file."""
    if duration is None:
        duration, width, height, has_audio = probe(path)
    w = VIDEO_WIDTH if width >= VIDEO_WIDTH else width
    h = max(2, int(round(w * height / width / 2)) * 2)

    rms, flat, mod, speech_blocks, hd, fd, edges = [], [], [], [], [], [], []
    n_onsets = 0
    analysed = 0.0
    for s, e in windows_for(duration):
        analysed += e - s
        if has_audio:
            a = audio_frames(decode_audio(path, s, e - s))
            if a is not None:
                r, fl, be = a
                rms.append(r)
                flat.append(fl)
                m = mod4_blocks(be)
                mod.append(m)
                # Block energy gate: louder than the window's noise floor and not silent.
                blk = r[: len(m) * BLOCK].reshape(len(m), BLOCK).mean(axis=1)
                loud = blk > max(np.percentile(r, 10) + 6.0, SILENCE_DB)
                speech_blocks.append(loud & (m > VAD_MOD4))
                n_onsets += onsets(r)
        frames = decode_gray(path, s, e - s, w, h)
        if len(frames) >= 2:
            a, b, c = video_stats(frames)
            hd.append(a)
            fd.append(b)
            edges.append(c)

    minutes = analysed / 60
    f = dict.fromkeys(FEATURES, 0.0)
    if rms:
        rms, flat, mod = np.concatenate(rms), np.concatenate(flat), np.concatenate(mod)
        sb = np.concatenate(speech_blocks)
        active = rms > max(np.percentile(rms, 10) + 6.0, SILENCE_DB)
        f.update(vad_speech_ratio=sb.mean() if len(sb) else 0.0, mod4_energy=mod.mean() if len(mod) else 0.0,
                 flatness_mean=flat[active].mean() if active.any() else flat.mean(),
                 flatness_std=flat[active].std() if active.any() else 0.0,
                 rms_db_mean=rms.mean(), rms_db_std=rms.std(), rms_db_p90=np.percentile(rms, 90),
                 silence_frac=(rms < SILENCE_DB).mean(), onset_rate=n_onsets / minutes)
    else:
        f.update(rms_db_mean=-100.0, rms_db_p90=-100.0, silence_frac=1.0)
    if hd:
        hd, fd, edges = np.concatenate(hd), np.concatenate(fd), np.concatenate(edges)
        f.update(hist_delta_mean=hd.mean(), hist_delta_p90=np.percentile(hd, 90),
                 hist_jump_rate=(hd > HIST_JUMP).sum() / minutes,
                 frame_diff_mean=fd.mean(), frame_diff_p90=np.percentile(fd, 90),
                 static_frac=(fd < STATIC_DIFF).mean(), edge_mean=edges.mean(), edge_std=edges.std())
    return {k: float(v) for k, v in f.items()}, analysed


def _job(row):
    t0 = time.time()
    feats, analysed = extract(row["video_path"])
    return row["video_id"], feats, analysed, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default="gt/gt_labels_v1.csv")
    ap.add_argument("--out", default="artifacts/features/cheap_features.csv")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--video", help="extract one file and print JSON")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"ERROR: {tool} not found on PATH")
    if args.video:
        feats, analysed = extract(args.video)
        print(json.dumps({"analysed_sec": analysed, **feats}, indent=2))
        return

    with open(args.labels, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["video_id"])
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for n, (vid, feats, analysed, dt) in enumerate(ex.map(_job, rows), 1):
            results.append((vid, feats, analysed, dt))
            print(f"[{n}/{len(rows)}] {vid}: {dt:.1f}s for {analysed:.0f}s analysed", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "analysed_sec", "extract_sec", *FEATURES])
        for vid, feats, analysed, dt in results:
            w.writerow([vid, round(analysed, 3), round(dt, 3), *(round(feats[k], 6) for k in FEATURES)])
    meta = Path(args.out).with_suffix(".meta.json")
    meta.write_text(json.dumps({"schedule": SCHEDULE, "features": FEATURES}, indent=1))
    tot_min = sum(r[2] for r in results) / 60
    tot_sec = sum(r[3] for r in results)
    print(f"{len(results)} videos -> {args.out}; {tot_sec / tot_min:.2f} s per analysed video-minute "
          f"(per process, jobs={args.jobs})")


if __name__ == "__main__":
    main()
