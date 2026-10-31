# GT step 2: teacher signals and GT v1

GT v0 mapped sub_category → chunking class, and 22 videos in ambiguous
sub_categories were flagged. This step measures objective signal profiles for all
90 videos. The 68 unflagged videos then define class anchors, and each flagged
video is reassigned to the nearest anchor when it is clearly closer. Uncertain
switches were checked by hand.

**Result:** GT v1 = **18 / 27 / 21 / 24** (conversational / info_dense /
high_motion / narrative). 5 labels changed versus v0: 4 by the teacher rule and
1 by manual override. A second manual review kept its v0 label.

> **Teacher-only.** Whisper, OCR, optical flow, shot detection and subtitles are
> offline labeling tools (macOS / Apple Silicon). They are **never** classifier
> features. The classifier sees only the raw video, using cheap cross-platform features.

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-teacher.txt   # macOS only
.venv/bin/python src/teacher/windows.py      # gt/windows.csv
.venv/bin/python src/teacher/extract.py      # artifacts/teacher_cache/ (cached, resumable; ~55 min)
.venv/bin/python src/teacher/aggregate.py    # gt/teacher_signals.csv, gt/whisper_check.csv
.venv/bin/python src/teacher/relabel.py      # gt/gt_labels_v1.csv (+ anchors, sensitivity; applies gt/manual_overrides.csv)
.venv/bin/python src/teacher/verify_html.py <video_id> ...   # data/verify/*.html for eyeballing (not committed)
```

## 1. Window schedule

`src/teacher/windows.py` → `gt/windows.csv`. The schedule is computed once and
shared by the teacher extractors and the later cheap-feature step.

- Duration ≤ 600 s: one window covering the whole video.
- Duration > 600 s: 10 windows of 60 s, evenly spaced between 5% and 95% of the duration.

That gives 468 windows and **9.63 h of video analysed** out of 25.88 h. Long
videos are capped at 600 s analysed.

## 2. Extractors and design choices

Each signal is a raw time series per window, cached at
`artifacts/teacher_cache/{signal}/v{EXTRACTOR_VERSION}/{video_id}.parquet`.
Only the window segments are decoded, using `ffmpeg -ss` before `-i`.

| Signal | Rate / resolution | Output | Method and choices |
|---|---|---|---|
| speech | per second | speech_active, word_count, source | `.srt` when present (78 videos), else mlx-whisper `small` on the window audio (12 videos). |
| cuts | native fps, 320 px wide | cut timestamps | PySceneDetect `AdaptiveDetector` (defaults). Cuts within 0.5 s of a window edge are dropped. |
| motion | 2 fps, 320 px wide, grayscale | flow_mag_mean, flow_mag_p90, global_motion, residual_motion | Farneback optical flow. global = magnitude of the median flow vector (camera); residual = mean magnitude after subtracting it (subject). Units: px per 0.5 s step at 320 px. |
| text | 0.5 fps, **native resolution** | text_area_frac, n_boxes, text_hash, boxes_json | Apple Vision via ocrmac. 320 px is too small to read text, so OCR runs on full frames. Boxes with confidence < 0.5 are dropped. text_hash = SHA-1 of lowercased alphanumeric text. |

Design choices, accepted at review:

- **Auto-caption timing.** YouTube auto-captions keep each line on screen long
  after it is spoken. Each cue's effective end is clipped to the next cue's start
  and to 1 s + 0.5 s per word. `<font>` tags and non-speech markers (`[Music]`,
  `(...)`, `♪`) are stripped. Word counts are spread over the cue's clipped duration.
- **Whisper hallucination filter.** Segments with `no_speech_prob > 0.6` *and*
  `avg_logprob < -1` are dropped. `condition_on_previous_text=False`.
- **OCR at full resolution** with a 0.5 confidence floor (see table).
- **Motion units** are px per 0.5 s at 320 px width, so they are comparable across source resolutions.
- `boxes_json` (text and box per OCR sample) is stored for the verification
  pages and for `text_area_central`.

## 3. Runtime and cache size

M1 Pro, sequential, one process.

**Smoke test** (3 videos):

| Video | Analysed | speech | cuts | motion | text |
|---|---|---|---|---|---|
| M69Sn3OERZo short, 720p | 47 s | 4.5 s (Whisper, warm) | 1.1 s | 0.7 s | 3.7 s |
| 7fnVcXfPe5c medium, 360p | 276 s | <0.1 s (srt) | 2.5 s | 4.0 s | 7.8 s |
| qd2ivr-5oEM long, 720p | 600 s | 0.2 s (srt) | 9.6 s | 10.9 s | 43.9 s |

The smoke test extrapolated to about 65 min for all 90 videos.

**Full run** (90 videos, 9.63 h analysed): **54 min wall clock**, 0 failures.

| Signal | Runtime | Seconds per analysed video-minute | Cache size |
|---|---|---|---|
| text (OCR) | 30.8 min | 3.20 | 1.5 MB |
| motion | 9.1 min | 0.94 | 2.9 MB |
| cuts | 8.0 min | 0.83 | 0.4 MB |
| speech (78 srt + 12 Whisper; Whisper ≈ 5.9 min) | 6.0 min | 0.62 | 0.8 MB |
| speech_whisper (consistency check, 5 videos) | 1.0 min | 2.37 | 0.03 MB |
| **Total** | **≈ 55 min** | | **5.6 MB** |

The one-time 136 s Whisper model download is excluded. These tools are far too
slow for the real-time classifier, which is why they are teacher-only.

## 4. Aggregates

`src/teacher/aggregate.py` → `gt/teacher_signals.csv`, one row per video. Rates
are per minute of *analysed* time.

speech_coverage, words_per_min, cuts_per_min, motion_mean, motion_p90 (mean of
per-frame p90), global_motion_mean, residual_motion_mean, global_motion_ratio
(global / total), text_area_mean, **text_area_central** (excludes boxes centered
in the top or bottom 15% of the frame: captions, tickers, logos), text_present_frac,
text_persistence (among samples with text, the share whose text_hash equals the
previous sample in the same window; 0 when there is no text), speech_source.

## 5. Whisper vs subtitles

Whisper was also run on 5 videos that have subtitles. It used a separate cache
(`speech_whisper`), so the srt-based cache was not overwritten.

| video | type | srt coverage | Whisper coverage | srt wpm | Whisper wpm | per-second agreement |
|---|---|---|---|---|---|---|
| YlAefotZNAc | News, short | 1.00 | 1.00 | 205 | 190 | 1.00 |
| xKiRmesHWIA | History, long | 0.99 | 1.00 | 156 | 156 | 0.99 |
| HK13nVxBYxI | News, medium | 0.99 | 0.98 | 177 | 164 | 0.97 |
| a7jZszvFpTY | Acrobatics, medium | 0.51 | 0.52 | 69 | 76 | 0.81 |
| **2QTiAmvygC4** | **Stage Play, short** | **0.21** | **0.80** | 15 | 101 | **0.38** |

On speech-heavy video the two sources agree closely. On music- or performance-heavy
video they diverge: Whisper transcribes singing and lyrics as speech, and
auto-captions skip them.

**Music caveat.** 7 of the 12 no-subtitle videos (Whisper-sourced) are
music-heavy genres, so their speech_coverage may be inflated:

| video | sub_category | Whisper coverage | wpm |
|---|---|---|---|
| _w4XRiUVfY4 | Stage Play | 0.79 | 118 |
| a8IGDMrohnY | Magic Show | 0.77 | 120 |
| SBssmtYTJpM | Athletics | 0.49 | 49 |
| 026dzf-vc5g | Acrobatics | 0.42 | 46 |
| awHc8IySU-I | Other Sports | 0.27 | 43 |
| vf3sPa2W0sA | Athletics | 0.23 | 82 |
| ACvLYz9nHvg | Basketball | 0.22 | 24 |

Manual review confirmed this for **frX8GujpmkI** (Fashion, no subtitles): it is a
Gucci ad, and its 0.74 Whisper speech coverage is **song lyrics**, not speech.
Sports commentary is real speech, so low or mid coverage there may be genuine.

## 6. Anchor-based relabel

`src/teacher/relabel.py`:

1. Robust-standardize each feature over all 90 videos: (x − median) / IQR.
2. Class anchor = per-class median over the **68 unflagged** videos.
3. Each flagged video gets a Euclidean distance to every anchor. It switches only
   if `d_nearest < MARGIN × d_v0`.
4. Unflagged videos are never changed. They are reported when their own-anchor
   distance exceeds the 90th percentile of unflagged own-anchor distances.
5. Manual overrides from `gt/manual_overrides.csv` are applied last.

### Two rule versions (sensitivity)

| Rule | Features | Margin | Flagged switched | Status |
|---|---|---|---|---|
| A | 11: speech_coverage, words_per_min, cuts_per_min, motion_mean, motion_p90, global_motion_mean, residual_motion_mean, global_motion_ratio, text_area_mean, text_present_frac, text_persistence | 0.8 | 11 / 22 | superseded |
| **B** | **5, one per axis:** speech_coverage, cuts_per_min, global_motion_mean, residual_motion_mean, text_area_central | **0.7** | **6 / 22** | **primary** |

Rule A over-weighted motion (5 of its 11 features are correlated motion
measures), and its anchors were weakly separated. Half the flagged videos
switched, several of them implausibly (e.g. 4 of 7 Fashion videos). Rule B uses
one feature per axis and a stricter margin. Only 5 switches are shared by both
rules, and `lNvBXLbInXE` switches under both but to different classes. The
labels are therefore **sensitive to the feature set and margin**, which is why
the remaining switches went to manual review. Per-video results for both rules
are in `gt/relabel_sensitivity.csv`.

### Class anchor profile (median raw values, unflagged videos)

| Class | n | speech_cov | words/min | cuts/min | motion_mean | motion_p90 | global | residual | global_ratio | text_area | **text_central** | text_present | text_persist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1_conversational | 15 | 0.985 | 156 | 7.2 | 2.22 | 5.15 | 0.63 | 1.99 | 0.30 | 0.031 | 0.018 | 0.53 | 0.32 |
| 2_info_dense | 21 | 0.842 | 125 | 6.7 | 2.64 | 6.29 | 0.54 | 2.36 | 0.23 | 0.030 | 0.014 | 0.59 | 0.25 |
| 3_high_motion | 19 | 0.734 | 104 | 5.7 | 5.09 | 10.88 | 1.93 | 4.74 | 0.37 | 0.017 | 0.012 | 0.80 | 0.04 |
| 4_narrative | 13 | 0.553 | 69 | 17.4 | 3.23 | 7.76 | 0.67 | 3.17 | 0.23 | 0.004 | 0.001 | 0.09 | 0.00 |

Narrative stands out on cuts (17.4 per minute) and near-zero text. High_motion
stands out on camera motion (about 3× the others). Conversational has the most
speech, but only moderately more than info_dense.

**Finding: on-screen text does not separate info_dense in this sample.** The
info_dense anchor has *less* central text than conversational (0.014 vs 0.018),
and its total text area is equal (0.030 vs 0.031). The info_dense sub_categories
here are mostly Geography, Fashion, Food and Handicraft. Their videos are largely
scenic or demonstrative, not slide or workspace content, so "info-dense" by
sub_category is not "text-dense" in these videos.

### All 22 flagged videos

Distances are under rule B (standardized units). The ratio in parentheses is
`d_nearest / d_v0`; a switch happens when it is below the margin.

| video_id | sub_category | v0 | rule A 11f/0.8 (ratio) | rule B 5f/0.7 (ratio) | **v1** | source | d_conv | d_info | d_motion | d_narr |
|---|---|---|---|---|---|---|---|---|---|---|
| 5i5Be73ESUc | Daily Life | narr | narr (0.98) | narr (0.77) | **narr** | subcat_map | 1.53 | 1.32 | 1.15 | 1.50 |
| NeLj-gLWgug | Daily Life | narr | narr (1.00) | narr (1.00) | **narr** | subcat_map | 2.11 | 1.72 | 2.21 | 1.21 |
| RU8IfXp7LbE | Daily Life | narr | narr (1.00) | narr (1.00) | **narr** | subcat_map | 2.62 | 2.42 | 2.07 | 1.70 |
| vzfTpidE5wg | Daily Life | narr | narr (0.94) | narr (0.91) | **narr** | subcat_map | 1.02 | 0.75 | 1.47 | 0.82 |
| 3EQLFHRHpag | Documentary | conv | motion (0.69) | motion (0.57) | **conv** | manual_review | 2.17 | 2.08 | 1.23 | 1.97 |
| MPfuSfiZ96A | Documentary | conv | narr (0.60) | narr (0.41) | **narr** | teacher_anchor | 1.90 | 1.68 | 1.74 | 0.77 |
| QrmxMNBhH2Q | Documentary | conv | narr (0.69) | conv (0.70) | **conv** | subcat_map | 2.11 | 2.01 | 1.70 | 1.49 |
| LlajSKnbcGk | Esports | motion | motion (1.00) | motion (1.00) | **motion** | subcat_map | 2.48 | 2.22 | 1.18 | 1.88 |
| S8w44Tl5atk | Esports | motion | info (0.62) | info (0.49) | **conv** | manual_review | 1.39 | 1.19 | 2.42 | 1.78 |
| xvQOEdwAgj0 | Esports | motion | motion (1.00) | motion (1.00) | **motion** | subcat_map | 1.39 | 1.26 | 1.07 | 1.59 |
| lNvBXLbInXE | Exercise | motion | conv (0.46) | info (0.50) | **info** | teacher_anchor | 1.06 | 0.91 | 1.80 | 1.03 |
| 2iHmlBmLKUE | Fashion | info | motion (0.65) | info (0.73) | **info** | subcat_map | 3.43 | 3.31 | 2.40 | 2.70 |
| 9v9NHoTXc-Q | Fashion | info | narr (0.65) | narr (0.63) | **narr** | teacher_anchor | 2.95 | 2.70 | 2.91 | 1.71 |
| K7YNoRZMv2k | Fashion | info | info (1.00) | info (0.93) | **info** | subcat_map | 1.30 | 1.01 | 1.48 | 0.94 |
| OGQ7KwUXuBk | Fashion | info | info (1.00) | info (1.00) | **info** | subcat_map | 1.48 | 1.23 | 1.69 | 1.27 |
| frX8GujpmkI | Fashion | info | narr (0.64) | narr (0.69) | **narr** | teacher_anchor | 3.12 | 2.96 | 2.60 | 2.05 |
| uVR-DZQuRbM | Fashion | info | narr (0.75) | info (0.71) | **info** | subcat_map | 2.55 | 2.43 | 2.13 | 1.73 |
| x-ccfWcbht4 | Fashion | info | info (0.83) | info (1.00) | **info** | subcat_map | 0.84 | 0.83 | 1.74 | 1.20 |
| 27YZLpw0M98 | Travel | narr | narr (1.00) | narr (1.00) | **narr** | subcat_map | 1.77 | 1.62 | 1.85 | 0.97 |
| 2gFxZmb0dMo | Travel | narr | motion (0.71) | narr (0.77) | **narr** | subcat_map | 2.67 | 2.70 | 1.93 | 2.51 |
| A30IuIjQYYg | Travel | narr | narr (1.00) | narr (1.00) | **narr** | subcat_map | 2.60 | 2.28 | 2.08 | 1.17 |
| eZ5lFBPpRkM | Travel | narr | motion (0.80) | narr (0.75) | **narr** | subcat_map | 2.30 | 2.13 | 1.26 | 1.68 |

## 7. Manual review

**Why.** Rule B still left 6 flagged videos with switches whose margins were
moderate and sensitive to the rule (section 6). The user watched each one and
labeled it by which chunker fits best, meaning *where the information changes*:
in speech, in stepwise visual updates, in action peaks, or at edited cuts.
Overrides are recorded in `gt/manual_overrides.csv` and applied by `relabel.py`
with `gt_source = "manual_review"`. Confirmed switches stay `teacher_anchor`.

| video_id | sub_category | v0 | teacher (rule B) | **human (v1)** | agree | rationale |
|---|---|---|---|---|---|---|
| MPfuSfiZ96A | Documentary | conv | narr | **narr** | ✓ | 1-min documentary, edited cuts |
| 9v9NHoTXc-Q | Fashion | info | narr | **narr** | ✓ | fashion commercial, cuts, no speech |
| frX8GujpmkI | Fashion | info | narr | **narr** | ✓ | Gucci ad, song + cuts (its Whisper "speech" is lyrics) |
| lNvBXLbInXE | Exercise | motion | info | **info** | ✓ | yoga follow-along, step-by-step |
| 3EQLFHRHpag | Documentary | conv | motion | **conv** | ✗ | news-style; meaning in speech, cuts secondary (keeps v0) |
| S8w44Tl5atk | Esports | motion | info | **conv** | ✗ | talking-head YouTuber with face cam, no cuts, no gameplay (high confidence) |

**Result: teacher and human agreed on 4 of 6.**

**Key example: S8w44Tl5atk.** The dataset labels it *Esports*, so a genre-based
mapping would send it to high-motion (action-peak) chunking. It is actually a
talking-head YouTuber explaining to a face cam, with no gameplay and no cuts. Its
information lives in the speech, so semantic transcript chunking is right and
action-peak chunking would be wrong. The signals had already moved it away from
high_motion (d_motion = 2.42, its farthest class). This is the core argument for
the project: **the chunking strategy must be predicted from signals in the
video, not from its category.**

## 8. Unflagged outliers (reported, not relabeled)

Unflagged videos whose distance to their own class anchor exceeds the 90th
percentile. The set depends on the rule, so both lists are shown.

**Rule B (primary, threshold 2.36):**

| video | sub_category | v0 | own-anchor distance | nearest anchor |
|---|---|---|---|---|
| Qyg_91gNHCc | **Geography** | info | **7.50** | conversational |
| 7fnVcXfPe5c | Life Tip | info | 4.75 | conversational |
| 5LU_XY0z2ZY | Biology & Medicine | info | 4.52 | conversational |
| y9Fne3oUwX4 | Food | info | 3.06 | high_motion |
| HK13nVxBYxI | News Report | conv | 2.77 | conversational (own) |
| ooqdg9Wr-mo | Food | info | 2.57 | narrative |
| _qepWb_NVj4 | **Geography** | info | 2.39 | conversational |

**Rule A (superseded, threshold 3.26):** Qyg_91gNHCc (Geography) 6.21,
5LU_XY0z2ZY (Bio & Med) 4.96, 7fnVcXfPe5c (Life Tip) 4.64, y9Fne3oUwX4 (Food)
4.03, jkNxmTrrZSk (Bio & Med) 3.56, 2QTiAmvygC4 (Stage Play) 3.72,
_w4XRiUVfY4 (Stage Play) 3.42.

Four videos are outliers under both rules: Qyg_91gNHCc, 7fnVcXfPe5c,
5LU_XY0z2ZY and y9Fne3oUwX4. 6 of the 7 rule-B outliers are info_dense, which
matches the finding that the info_dense anchor is not coherent.

**Taxonomy finding: Geography may not fit info_dense.** The most extreme outlier
in either rule is a Geography video, and both rule-B Geography outliers sit
nearer the conversational anchor. Geography content in this sample ranges from
scenic footage with narration (conversational or narrative) to instructional,
map-based explainers (info_dense). A single sub_category → class mapping cannot
cover both. These videos are left unchanged in v1 because they were not flagged,
but they are the first place to look if labels are revisited.

## 9. Changes and final distribution

| | conv | info | motion | narr | total |
|---|---|---|---|---|---|
| GT v0 | 18 | 28 | 23 | 21 | 90 |
| changed out | −1 (MPfu) | −2 (9v9N, frX8) | −2 (lNvB, S8w4) | 0 | −5 |
| changed in | +1 (S8w4) | +1 (lNvB) | 0 | +3 (MPfu, 9v9N, frX8) | +5 |
| **GT v1** | **18** | **27** | **21** | **24** | **90** |

`gt_source`: 84 `subcat_map`, 4 `teacher_anchor`, 2 `manual_review`. In
`gt/gt_labels_v1.csv`, `gt_class` = `gt_class_v1`, and `gt_class_v0` keeps the
original label.

## 10. Limitations

- **Thresholds are design choices.** The margin (0.7 vs 0.8), the feature set,
  the 90th-percentile outlier cutoff, the auto-caption timing heuristic, the
  Whisper filter and the OCR confidence floor were all chosen by hand. Section 6
  shows the labels are sensitive to the first two.
- **Anchors come from a noisy prior.** They are medians of sub_category-mapped
  labels, and the unflagged set itself contains outliers (section 8). Small
  classes (13–21 unflagged each) make the anchors unstable.
- **Subtitles may be auto-generated.** Auto-captions skip singing and music, and
  Whisper transcribes them, so speech_coverage is source-dependent on
  music-heavy video (section 5).
- **Windowed analysis.** Long videos are sampled at 10 × 60 s (the teacher's and the previous classifier budget; the shipped classifier uses 3 × 30 s), so rare events can be missed.
- **Default detector settings.** PySceneDetect AdaptiveDetector was not tuned per
  video. Fades and dissolves may be under-counted, and flashes over-counted.
- **One reviewer.** Manual labels come from a single reviewer; there is no inter-rater agreement.
- **Small sample.** 90 videos, with 1–8 per sub_category.

## NOT DONE YET

- Cheap cross-platform features, the classifier, baselines, the 5-fold CV benchmark and speed metrics.
- `run.yaml` is still the template.
