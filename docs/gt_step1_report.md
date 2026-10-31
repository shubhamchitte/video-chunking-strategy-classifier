# GT v0 report: sub_category mapping, download and probe (Steps 1–3)

Ground truth v0 for the chunking-strategy classifier on the 90-video Video-MME
sample. Labels come from a fixed sub_category → class map. 22 videos in
ambiguous sub_categories are flagged as provisional.

Reproduce from the repo root:

```bash
bash src/data/download.sh      # metadata + 90 videos + 78 subtitles into ./data/ (gitignored), checksum-verified
python src/gt/step1_map.py     # gt/gt_labels_v0.csv, gt/count_validation.csv (fails loudly on any count mismatch)
python src/data/probe.py       # fills duration_sec, fps, resolution, has_audio, has_subtitles; gt/probe_issues.csv
```

## 1. Count validation

**All checks pass.** Details in [`gt/count_validation.csv`](../gt/count_validation.csv).

| Check | Expected | Found |
|---|---|---|
| Questions (rows in `qa/test.jsonl`) | 270 | 270 |
| Unique videos | 90 | 90 |
| Questions per video | 3 each | 3 each |
| Duration buckets short / medium / long | 30 / 30 / 30 | 30 / 30 / 30 |
| Sub_category video and question counts | 28 rows | 28 / 28 match |
| UNMAPPED sub_categories | 0 | 0 |

Each video has a single (domain, sub_category, duration) across its 3 questions.

## 2. Class distribution (GT v0)

| Class | Sub_categories | Expected | Found |
|---|---|---|---|
| 1_conversational | News Report, Documentary, Finance & Commerce, Humanity & History, Literature & Art, Multilingual | 18 | 18 |
| 2_info_dense | Astronomy, Biology & Medicine, Geography, Technology, Life Tip, Handicraft, Fashion, Food | 28 | 28 |
| 3_high_motion | Athletics, Basketball, Esports, Football, Other Sports, Exercise, Acrobatics | 23 | 23 |
| 4_narrative | Stage Play, Movie & TV Show, Animation, Magic Show, Variety Show, Daily Life, Travel | 21 | 21 |
| **Total** | | **90** | **90** |

## 3. Flagged videos (`needs_signal_check = True`): 22

Labels for these videos are **provisional**. Their sub_categories (Fashion,
Daily Life, Travel, Documentary, Esports, Exercise) could plausibly fit more
than one chunking strategy.

| video_id | sub_category | gt_class (v0) | duration_bucket | duration_sec |
|---|---|---|---|---|
| 5i5Be73ESUc | Daily Life | 4_narrative | long | 1843.8 |
| NeLj-gLWgug | Daily Life | 4_narrative | medium | 352.3 |
| RU8IfXp7LbE | Daily Life | 4_narrative | short | 73.0 |
| vzfTpidE5wg | Daily Life | 4_narrative | long | 1987.4 |
| 3EQLFHRHpag | Documentary | 1_conversational | medium | 746.0 |
| MPfuSfiZ96A | Documentary | 1_conversational | short | 77.9 |
| QrmxMNBhH2Q | Documentary | 1_conversational | medium | 606.0 |
| LlajSKnbcGk | Esports | 3_high_motion | short | 40.8 |
| S8w44Tl5atk | Esports | 3_high_motion | long | 2123.0 |
| xvQOEdwAgj0 | Esports | 3_high_motion | long | 2468.3 |
| lNvBXLbInXE | Exercise | 3_high_motion | medium | 728.7 |
| 2iHmlBmLKUE | Fashion | 2_info_dense | short | 67.5 |
| 9v9NHoTXc-Q | Fashion | 2_info_dense | short | 56.7 |
| K7YNoRZMv2k | Fashion | 2_info_dense | long | 1838.2 |
| OGQ7KwUXuBk | Fashion | 2_info_dense | short | 116.0 |
| frX8GujpmkI | Fashion | 2_info_dense | short | 60.1 |
| uVR-DZQuRbM | Fashion | 2_info_dense | short | 89.9 |
| x-ccfWcbht4 | Fashion | 2_info_dense | medium | 871.7 |
| 27YZLpw0M98 | Travel | 4_narrative | short | 102.1 |
| 2gFxZmb0dMo | Travel | 4_narrative | short | 75.3 |
| A30IuIjQYYg | Travel | 4_narrative | medium | 250.9 |
| eZ5lFBPpRkM | Travel | 4_narrative | long | 3221.5 |

## 4. Download, checksums and file health

| Item | Result |
|---|---|
| Videos downloaded / checksum OK | 90 / 90 |
| Subtitles downloaded / checksum OK | 78 / 78 |
| Persistent download failures | 0 (2 transient `HTTP/2 stream reset` errors recovered on the automatic retry) |
| Rerun of `download.sh` | all 168 files skipped as cached-and-valid |
| Missing or unreadable videos (ffprobe) | 0 ([`gt/probe_issues.csv`](../gt/probe_issues.csv) is empty) |
| Videos without an audio stream | 0 |
| Videos without a usable frame rate | 0 |
| Videos without subtitles | 12 |
| `git status` under `data/` | nothing tracked or untracked (ignored via `/data/`) |

Checksums are verified against the `videos/*.mp4` and `subtitles/*.srt`
entries in the dataset's `checksums.sha256` (168 entries; other entries are ignored).

Frame rates: 25 (32 videos), 29.97 (26), 30 (14), 23.976 (13), 24 (3), 29.839 (1), 15 (1).
Resolutions: 1280x720 (75), 640x360 (13), 1280x714 (1), 1280x676 (1).

## 5. Total duration

| Bucket | All: videos | All: hours | All: range (s) | Flagged: videos | Flagged: hours |
|---|---|---|---|---|---|
| short | 30 | 0.70 | 41 – 116 | 10 | 0.21 |
| medium | 30 | 4.41 | 251 – 1081 | 6 | 0.99 |
| long | 30 | 20.77 | 1838 – 3552 | 6 | 3.75 |
| **Total** | **90** | **25.88** | | **22** | **4.94** |

The teacher pass on the flagged set therefore covers about 4.9 hours of video.

## 6. Subtitles are teacher-only

78 of the 90 videos have a `.srt` file (in `data/subtitles/`, gitignored). Of the 12
without one, 5 are in `3_high_motion`; the rest are spread over the other classes.

- **Allowed:** as a teacher signal when correcting GT. For videos with subtitles,
  the `.srt` can stand in for Whisper when measuring speech density. Whisper is
  the fallback for the 12 without subtitles.
- **Never allowed:** as a classifier feature. At inference the classifier sees
  only the raw video. Subtitle presence also correlates with class, so using
  `has_subtitles` would leak information. The `has_subtitles` column in
  `gt_labels_v0.csv` is metadata for the teacher pipeline only.

## 7. Evaluation design notes

- Cross-validation will be stratified on `gt_class` (4 classes), **not** on sub_category.
- Astronomy, Technology, Exercise and Variety Show each have only one video, so
  per-sub_category metrics will be **descriptive only**, not used for model selection.
- `domain` and `sub_category` are never used as classifier features. They define
  GT v0 only.

## 8. Data source & citation

Data comes from a Hugging Face-hosted sampled subset (10%, duration-stratified, seed 42) of the public Video-MME benchmark.
Its dataset card says it is not the original dataset or an official release, and
it asks that results be reported against the original:

> Fu, C. et al. *Video-MME: The First-Ever Comprehensive Evaluation Benchmark of
> Multi-modal LLMs in Video Analysis.* CVPR 2025. arXiv:2405.21075.
> Original dataset: https://huggingface.co/datasets/lmms-eval/Video-MME

The sample's dataset card **states no license**, and the videos are YouTube
content. Videos and subtitles are kept locally in `./data/` for research use only.
They are never committed and never uploaded anywhere. Only small derived artifacts
(CSVs, this report) are committed.

## 9. NOT DONE YET

- **Signal-based correction of the 22 flagged videos is pending.** Their labels are provisional (GT v0 = subcat_map only).
- Teacher signals (Whisper/subtitles, shot detection, optical flow, OCR) have not started, so there is no GT v1 yet.
- Cheap features, the classifier, baselines, 5-fold CV benchmark and speed metrics have not started.
- `run.yaml` is still the template. It references `requirements.txt` and `src/main.py`, and neither exists yet.
