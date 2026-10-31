"""Build one self-contained HTML page per video to eyeball cached teacher signals.

    python src/teacher/verify_html.py ID1 ID2 ...   # -> data/verify/{video_id}.html (not committed)

Shows: before/after frames for every detected cut, a timeline (speech, motion,
text area, cut markers), 5 OCR sample frames with boxes drawn, and a plain list
of cut timestamps for jumping to in a player.
"""
import argparse
import base64
import csv
import html
import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import cache_path  # noqa: E402
from windows import load_windows  # noqa: E402

OUT_DIR = Path("data/verify")
CUT_OFFSET_SEC = 0.12


def grab(path, t, width=320):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(t, 0):.3f}", "-i", str(path),
           "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "-"]
    return Image.open(io.BytesIO(subprocess.run(cmd, capture_output=True, check=True).stdout)).convert("RGB")


def b64(img_or_fig, fmt="JPEG"):
    buf = io.BytesIO()
    if isinstance(img_or_fig, Image.Image):
        img_or_fig.save(buf, fmt, quality=80)
        mime = "jpeg"
    else:
        img_or_fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(img_or_fig)
        mime = "png"
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def ts(t):
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:06.3f}" if h else f"{m}:{s:06.3f}"


def timeline(windows, sp, mo, tx, cu):
    """Concatenate windows on one 'analysed time' axis; dashed lines separate windows."""
    offsets, acc = [], 0.0
    for s, e in windows:
        offsets.append(acc - s)
        acc += e - s

    def x(df):
        return df["t_sec"] + df["window_idx"].map(dict(enumerate(offsets)))

    fig, axes = plt.subplots(3, 1, figsize=(13, 6), sharex=True)
    axes[0].step(x(sp), sp["speech_active"], where="post", color="tab:blue")
    axes[0].set_ylabel("speech")
    axes[0].set_ylim(-0.1, 1.1)
    axes[1].plot(x(mo), mo["flow_mag_mean"], label="flow mean", color="tab:orange")
    axes[1].plot(x(mo), mo["global_motion"], label="global (camera)", color="tab:green")
    axes[1].plot(x(mo), mo["residual_motion"], label="residual (subject)", color="tab:red", alpha=0.7)
    axes[1].set_ylabel("motion px/0.5s")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(x(tx), tx["text_area_frac"], marker=".", color="tab:purple")
    axes[2].set_ylabel("text area")
    for ax in axes:
        for cx in x(cu):
            ax.axvline(cx, color="k", lw=0.6, alpha=0.5)
        for i in range(1, len(windows)):
            ax.axvline(offsets[i] + windows[i][0], color="gray", ls="--", lw=1)
    starts = [offsets[i] + s for i, (s, _) in enumerate(windows)]
    axes[2].set_xticks(starts, [ts(s) for s, _ in windows], rotation=45, fontsize=8)
    axes[2].set_xlabel("window start (video time); black lines = cuts, dashed = window boundary")
    return fig


def ocr_samples(path, tx, n=5):
    with_text = tx[tx["n_boxes"] > 0]
    pick = with_text.iloc[np.linspace(0, len(with_text) - 1, min(n, len(with_text))).astype(int)] \
        if len(with_text) else tx.head(0)
    out = []
    for _, r in pick.iterrows():
        img = grab(path, r["t_sec"], width=640)
        d = ImageDraw.Draw(img)
        W, H = img.size
        texts = []
        for text, conf, (bx, by, bw, bh) in json.loads(r["boxes_json"]):
            # Vision boxes are normalized with origin at bottom-left.
            d.rectangle([bx * W, (1 - by - bh) * H, (bx + bw) * W, (1 - by) * H], outline=(255, 0, 0), width=2)
            texts.append(f"{text} ({conf:.2f})")
        out.append((r["t_sec"], r["text_area_frac"], b64(img), texts))
    return out


def page(video_id, meta, windows):
    path = meta["video_path"]
    sp, cu, mo, tx = (pd.read_parquet(cache_path(s, video_id)) for s in ("speech", "cuts", "motion", "text"))
    cut_rows = "".join(
        f"<div class=cut><div class=ts>{ts(t)}</div>"
        f"<img src='{b64(grab(path, t - CUT_OFFSET_SEC))}'><img src='{b64(grab(path, t + CUT_OFFSET_SEC))}'></div>"
        for t in cu["t_sec"])
    ocr = "".join(
        f"<figure><img src='{img}'><figcaption>t={ts(t)} &middot; area={a:.3f}<br>"
        f"{html.escape(' | '.join(texts))}</figcaption></figure>"
        for t, a, img, texts in ocr_samples(path, tx))
    analysed = sum(e - s for s, e in windows)
    summary = (f"{meta['sub_category']} &middot; v0 class {meta['gt_class']} &middot; {meta['duration_bucket']} "
               f"{float(meta['duration_sec']):.0f}s &middot; {len(windows)} window(s), {analysed:.0f}s analysed "
               f"&middot; speech source {sp['source'].iloc[0] if len(sp) else '-'} &middot; {len(cu)} cuts")
    return f"""<!doctype html><html><head><meta charset=utf-8><title>{video_id} teacher check</title>
<style>body{{font:14px system-ui;margin:16px;background:#fff;color:#111}} img{{max-width:100%}}
.cuts{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px}}
.cut img{{width:49%;margin-right:1%}} .ts{{font-family:monospace;font-weight:600}}
figure{{display:inline-block;width:640px;margin:0 12px 12px 0;vertical-align:top}} pre{{background:#f4f4f4;padding:8px}}</style>
</head><body><h1>{video_id}</h1><p>{summary}<br><code>{path}</code></p>
<h2>Timeline</h2><img src='{b64(timeline(windows, sp, mo, tx, cu))}'>
<h2>Cut timestamps ({len(cu)})</h2><pre>{chr(10).join(ts(t) for t in cu['t_sec']) or '(none)'}</pre>
<h2>Cuts: frame {CUT_OFFSET_SEC}s before | {CUT_OFFSET_SEC}s after</h2><div class=cuts>{cut_rows or '(none)'}</div>
<h2>OCR samples</h2>{ocr or '(no frames with text)'}
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--labels", default="gt/gt_labels_v0.csv")
    args = ap.parse_args()
    with open(args.labels, newline="") as f:
        meta = {r["video_id"]: r for r in csv.DictReader(f)}
    windows = load_windows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for vid in args.video_ids:
        out = OUT_DIR / f"{vid}.html"
        out.write_text(page(vid, meta[vid], windows[vid]))
        print(out)


if __name__ == "__main__":
    main()
