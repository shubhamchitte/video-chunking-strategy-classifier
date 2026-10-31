"""Feature groups v2: EXPERIMENT ONLY, not used by run.yaml or the shipped model (see docs/writeup_appendix.md).

Install requirements-experiments.txt (OpenCV 4.12 for Haar faces) in a separate venv.

    python src/features/cheap_v2.py --jobs 4   # -> artifacts/features/cheap_features_v2.csv (+ keyframe_vs_cuts.csv)

v1 columns are copied unchanged from artifacts/features/cheap_features.csv. New groups:
  A  bitstream (ffprobe -show_packets, no decode): packet sizes, bits/pixel, keyframe rate and interval CV
  B  faces (OpenCV Haar frontal face, 1 fps, 320 px): face-present fraction, largest-face area
  C  color/light (2 fps, 160 px BGR): saturation, color variance, brightness, lighting key, colorfulness
  D  editing rhythm from the v1 histogram-jump detector: shot length mean/std/CV
  E  audio+: ZCR, spectral centroid, low-energy ratio, 13 MFCC means (numpy/scipy)
Per group, extra CPU seconds (incl. ffmpeg/ffprobe children) are recorded. D and E reuse
v1's decoded frames/audio in a real pipeline, so only their compute is timed; B and C need
their own decode, which is included (C is an upper bound: one color decode could replace v1's gray one).
"""
import argparse
import csv
import resource
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teacher"))
from cheap import (BLOCK, HIST_BINS, HIST_JUMP, HOP, N_FFT, SR, decode_audio, probe)  # noqa: E402,F401
from windows import windows_for  # noqa: E402

V1_CSV = Path("artifacts/features/cheap_features.csv")
OUT_CSV = Path("artifacts/features/cheap_features_v2.csv")
KF_CSV = Path("artifacts/results/keyframe_vs_cuts.csv")
GROUPS = {
    "A": ["a_pkt_size_mean", "a_pkt_size_std", "a_pkt_size_cv", "a_pkt_p90_p50", "a_bitrate_kbps",
          "a_bits_per_pixel", "a_keyframe_rate", "a_keyframe_interval_cv"],
    "B": ["b_face_present_frac", "b_face_area_mean"],
    "C": ["c_saturation_mean", "c_color_var", "c_brightness_mean", "c_brightness_std", "c_colorfulness"],
    "D": ["d_shot_len_mean", "d_shot_len_std", "d_shot_len_cv"],
    "E": ["e_zcr_mean", "e_zcr_var", "e_centroid_mean", "e_centroid_std", "e_low_energy_ratio",
          *[f"e_mfcc{i}" for i in range(13)]],
}
KF_TOL = 0.5


def cpu():
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + ch.ru_utime + ch.ru_stime


def decode(path, s, d, w, h, fps, pix):
    ch = 3 if pix == "bgr24" else 1
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{s:.3f}", "-t", f"{d:.3f}", "-i", str(path),
                          "-an", "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", pix, "-"],
                         capture_output=True, check=True).stdout
    n = len(raw) // (w * h * ch)
    return np.frombuffer(raw[: n * w * h * ch], np.uint8).reshape((n, h, w, ch) if ch == 3 else (n, h, w))


def size_for(width, height, target):
    w = min(target, width)
    return w, max(2, int(round(w * height / width / 2)) * 2)


# ---------------------------------------------------------------- A bitstream

def group_a(path, wins, width, height, fps_nominal):
    intervals = ",".join(f"{s:.3f}%+{e - s:.3f}" for s, e in wins)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", intervals,
                          "-show_entries", "packet=pts_time,size,flags", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    t, size, key = [], [], []
    for line in out.splitlines():
        p = line.split(",")
        if len(p) < 3 or p[0] in ("", "N/A"):
            continue
        ts = float(p[0])
        if any(s <= ts < e for s, e in wins):  # read_intervals snaps to keyframes; trim to the windows
            t.append(ts)
            size.append(int(p[1]))
            key.append("K" in p[2])
    t, size, key = np.array(t), np.array(size, float), np.array(key)
    analysed = sum(e - s for s, e in wins)
    kt = np.sort(t[key])
    gaps = np.diff(kt)
    gaps = gaps[gaps < 61]  # don't count the jump between windows
    feats = {
        "a_pkt_size_mean": size.mean(), "a_pkt_size_std": size.std(),
        "a_pkt_size_cv": size.std() / size.mean(), "a_pkt_p90_p50": np.percentile(size, 90) / np.percentile(size, 50),
        "a_bitrate_kbps": size.sum() * 8 / analysed / 1000,
        "a_bits_per_pixel": size.sum() * 8 / (len(size) * width * height),
        "a_keyframe_rate": key.sum() / (analysed / 60),
        "a_keyframe_interval_cv": gaps.std() / gaps.mean() if len(gaps) > 1 else 0.0,
    }
    return feats, kt


def keyframe_vs_cuts(kt, vid, wins):
    """Precision/recall of keyframe times vs teacher cuts (+-KF_TOL s), ignoring first keyframe of each window."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teacher"))
    from extract import cache_path
    cuts = pd.read_parquet(cache_path("cuts", vid))["t_sec"].to_numpy()
    kt = np.array([k for k in kt if not any(abs(k - s) < 1.0 for s, _ in wins)])
    tp_k = sum(np.any(np.abs(cuts - k) <= KF_TOL) for k in kt)
    tp_c = sum(np.any(np.abs(kt - c) <= KF_TOL) for c in cuts)
    return {"video_id": vid, "n_keyframes": len(kt), "n_cuts": len(cuts),
            "precision": tp_k / len(kt) if len(kt) else np.nan, "recall": tp_c / len(cuts) if len(cuts) else np.nan}


# ---------------------------------------------------------------- B faces

def group_b(path, wins, width, height):
    import cv2
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    w, h = size_for(width, height, 320)
    present, areas = [], []
    for s, e in wins:
        for f in decode(path, s, e - s, w, h, 1, "gray"):
            faces = det.detectMultiScale(f, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            present.append(len(faces) > 0)
            areas.append(max((fw * fh for _, _, fw, fh in faces), default=0) / (w * h))
    return {"b_face_present_frac": float(np.mean(present)), "b_face_area_mean": float(np.mean(areas))}


# ---------------------------------------------------------------- C color / D rhythm

def decode_color(path, wins, width, height):
    w, h = size_for(width, height, 160)
    return [decode(path, s, e - s, w, h, 2, "bgr24") for s, e in wins]


def group_c(frames_by_win):
    import cv2
    sat, cvar, bright, bstd, colorful = [], [], [], [], []
    for frames in frames_by_win:
        for f in frames:
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            sat.append(hsv[..., 1].mean() / 255)
            luma = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float) / 255
            bright.append(luma.mean())
            bstd.append(luma.std())
            ff = f.astype(float)
            cvar.append(ff.reshape(-1, 3).var(axis=0).mean() / 255 ** 2)
            b, g, r = ff[..., 0], ff[..., 1], ff[..., 2]
            rg, yb = r - g, 0.5 * (r + g) - b
            colorful.append((np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 255)
    return {"c_saturation_mean": np.mean(sat), "c_color_var": np.mean(cvar), "c_brightness_mean": np.mean(bright),
            "c_brightness_std": np.mean(bstd), "c_colorfulness": np.mean(colorful)}


def group_d(frames_by_win, wins):
    import cv2
    lengths = []
    for frames, (s, e) in zip(frames_by_win, wins):
        if len(frames) < 2:
            continue
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        hist = np.stack([np.histogram(g, bins=HIST_BINS, range=(0, 256))[0] / g.size for g in gray])
        hd = 0.5 * np.abs(np.diff(hist, axis=0)).sum(axis=1)
        cuts = [s + (i + 1) / 2.0 for i in np.flatnonzero(hd > HIST_JUMP)]
        lengths += list(np.diff([s, *cuts, e]))
    L = np.array(lengths)
    return {"d_shot_len_mean": L.mean(), "d_shot_len_std": L.std(), "d_shot_len_cv": L.std() / L.mean()}


# ---------------------------------------------------------------- E audio+

def mel_filterbank(n_mels=26, fmin=0.0, fmax=SR / 2):
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    bins = np.floor((N_FFT + 1) * mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)) / SR).astype(int)
    fb = np.zeros((n_mels, N_FFT // 2 + 1))
    for i in range(1, n_mels + 1):
        lo, c, hi = bins[i - 1], bins[i], bins[i + 1]
        fb[i - 1, lo:c] = (np.arange(lo, c) - lo) / max(c - lo, 1)
        fb[i - 1, c:hi] = (hi - np.arange(c, hi)) / max(hi - c, 1)
    return fb


def group_e(audio_by_win):
    from scipy.fft import dct
    fb = mel_filterbank()
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    zcr, cent, low, mfcc = [], [], [], []
    for x in audio_by_win:
        if len(x) < N_FFT:
            continue
        n = 1 + (len(x) - N_FFT) // HOP
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
        fr = x[idx]
        zcr.append(np.mean(np.abs(np.diff(np.signbit(fr), axis=1)), axis=1))
        power = np.abs(np.fft.rfft(fr * np.hanning(N_FFT), axis=1)) ** 2 + 1e-12
        cent.append((power * freqs).sum(axis=1) / power.sum(axis=1))
        rms = np.sqrt((fr ** 2).mean(axis=1))
        for i in range(0, len(rms) - BLOCK + 1, BLOCK):  # low-energy: frames below half the 1 s mean RMS
            low.append((rms[i:i + BLOCK] < 0.5 * rms[i:i + BLOCK].mean()).mean())
        mfcc.append(dct(np.log(power @ fb.T), type=2, norm="ortho", axis=1)[:, :13])
    if not zcr:
        return dict.fromkeys(GROUPS["E"], 0.0)
    zcr, cent, mfcc = np.concatenate(zcr), np.concatenate(cent), np.concatenate(mfcc)
    return {"e_zcr_mean": zcr.mean(), "e_zcr_var": zcr.var(), "e_centroid_mean": cent.mean(),
            "e_centroid_std": cent.std(), "e_low_energy_ratio": float(np.mean(low)) if low else 0.0,
            **{f"e_mfcc{i}": v for i, v in enumerate(mfcc.mean(axis=0))}}


# ---------------------------------------------------------------- per video

def _job(row):
    path, vid = row["video_path"], row["video_id"]
    duration, width, height, has_audio = probe(path)
    wins = windows_for(duration)
    feats, extra = {}, {}

    c0 = cpu()
    a, kt = group_a(path, wins, width, height, row.get("fps"))
    extra["A"] = cpu() - c0
    feats.update(a)
    kf = keyframe_vs_cuts(kt, vid, wins)

    c0 = cpu()
    feats.update(group_b(path, wins, width, height))
    extra["B"] = cpu() - c0

    c0 = cpu()
    frames = decode_color(path, wins, width, height)
    feats.update(group_c(frames))
    extra["C"] = cpu() - c0

    c0 = cpu()
    feats.update(group_d(frames, wins))
    extra["D"] = cpu() - c0

    audio = [decode_audio(path, s, e - s) for s, e in wins] if has_audio else []
    c0 = cpu()
    feats.update(group_e(audio))
    extra["E"] = cpu() - c0
    return vid, {k: float(v) for k, v in feats.items()}, extra, kf


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default="gt/gt_labels_v1.csv")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    with open(V1_CSV, newline="") as f:
        v1 = {r["video_id"]: r for r in csv.DictReader(f)}
    with open(args.labels, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["video_id"])
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for n, res in enumerate(ex.map(_job, rows), 1):
            results.append(res)
            print(f"[{n}/{len(rows)}] {res[0]} extra CPU s: " + " ".join(f"{k}={v:.1f}" for k, v in res[2].items()),
                  flush=True)

    new_cols = [c for g in GROUPS.values() for c in g]
    v1_cols = list(next(iter(v1.values())))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*v1_cols, *new_cols, *[f"cpu_{g}" for g in GROUPS]])
        for vid, feats, extra, _ in results:
            w.writerow([*(v1[vid][c] for c in v1_cols), *(round(feats[c], 6) for c in new_cols),
                        *(round(extra[g], 3) for g in GROUPS)])
    with open(KF_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0][3]))
        w.writeheader()
        w.writerows(r[3] for r in results)

    analysed_min = sum(float(v1[r[0]]["analysed_sec"]) for r in results) / 60
    print("extra CPU-s per analysed minute: " + ", ".join(
        f"{g} {sum(r[2][g] for r in results) / analysed_min:.2f}" for g in GROUPS) + f" -> {OUT_CSV}")


if __name__ == "__main__":
    main()
