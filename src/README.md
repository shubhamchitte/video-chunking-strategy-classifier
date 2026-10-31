# Code

Entry point: **`main.py`**: `evaluate` (no videos needed) and `predict --video FILE` (needs ffmpeg). See `run.yaml`.

| Path | Role | Deps |
|---|---|---|
| `main.py` | CLI: evaluate / predict (JSON output, low-confidence fallback) | requirements.txt |
| `features/cheap.py` | 17 cheap audio/video features; shipped schedule 3 × 30 s at 2 fps (stored with features and model) | numpy, opencv, ffmpeg |
| `classify/models.py` | dataset loading, model definitions, JSON model export/load | numpy, scikit-learn, lightgbm |
| `classify/evaluate.py` | 5×5 stratified CV, baselines, analyses, final model fit | same |
| `teacher/windows.py` | window schedule shared by teacher and cheap features | stdlib |
| `teacher/extract.py`, `aggregate.py`, `relabel.py`, `verify_html.py` | offline GT pipeline (Whisper, OCR, flow, shot detection); never classifier features | requirements-teacher.txt (macOS) |
| `gt/step1_map.py` | GT v0: sub_category → class mapping and count validation | stdlib |
| `data/download.sh`, `data/probe.py` | dataset download with checksums; ffprobe metadata | curl, ffprobe |
| `features/cheap_v2.py`, `experiments/*.py` | experiments only (feature-group ablation, downstream chunk check, teacher-vs-cheap, analysis-budget sweep); not imported by `main.py` | requirements-experiments.txt |
| `tools/make_architecture.py` | regenerates `docs/architecture.svg` + `.drawio` from one layout spec | stdlib |
