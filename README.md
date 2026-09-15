# HoopVision

[![tests](https://github.com/AndiLata/HoopVision/actions/workflows/tests.yml/badge.svg)](https://github.com/AndiLata/HoopVision/actions/workflows/tests.yml)

Single-camera basketball shot analysis. One phone on a tripod, one click to
mark the rim, and a dashboard that reports field goals, the shot arc, dribbles
before the shot, and shooting-arm angle.

## Three ways to run it

| Mode | Needs | For |
|---|---|---|
| `--replay` | nothing | The synthetic clip with exact ground truth. Verifies the pipeline. |
| `--bootstrap` | a clip + one rim click | **Your own footage, today.** Zero-shot COCO ball detection. Indicative numbers. |
| `--weights` | a fine-tuned model | Real numbers. |

### Verify the pipeline (30 seconds, no model, no torch)

```bash
pip install -r requirements.txt
python scripts/make_sample_clip.py --out sample
python scripts/verify.py                     # 45/45 checks against truth.json
python -m hoopvision.cli --replay sample --video sample/clip.mp4 \
    --out out/dashboard.mp4 --csv out/shots.csv
```

### Run it on your own footage before labelling anything

```bash
pip install ultralytics                       # needed from here on

# 1. is this clip even usable? run it on a 60-second test shoot
python scripts/check_footage.py --video clips/test.mov

# 2. mark the rim - two clicks, once per camera setup
python scripts/mark_rim.py --video clips/test.mov --out rim.json

# 3. go
python -m hoopvision.cli --bootstrap --rim rim.json \
    --video clips/test.mov --out out/test.mp4 --csv out/test.csv
```

Bootstrap mode leans on two facts. **The rim never moves** — the camera is on
a tripod, so the rim is a constant that needs one click, not a detector and
not 600 labelled frames. And **COCO already has a `sports ball` class** — weak
on a small, blurred, backlit basketball, but free and available now.

It is how you answer "does my camera setup work?" before committing an
afternoon to labelling. It is not a substitute for fine-tuning, and it turns
off the camera-drift check (the rim is asserted rather than detected), so
don't quote its numbers.

## What it measures

| Metric | Notes |
|---|---|
| field goals | made / attempted, FG% |
| make / miss | two-zone rim-plane crossing, tested against the ball's real clearance |
| shot trajectory | the tracked arc, drawn raw, with the previous six shots ghosted |
| dribbles before the shot | floor-contact bounces in the run-up, plus last dribble → release |
| shooting-arm angle | elbow at release, set point (the pocket), extension |
| entry angle | with the 43–47° optimal band |
| release speed / angle / height | depth-corrected via the ball's own diameter |
| follow-through, knee dip | pose-derived |
| shot chart | needs a four-click court homography |
| session σ | release angle, entry angle, elbow — the numbers that move first |

## The three ideas worth stealing

**The ball is a ruler that travels with the action.** A size-7 ball is 0.2395 m
across, always. So `m_per_px(t) = 0.2395 / ball_diameter_px(t)` gives a
depth-corrected scale at every frame, for free, with no calibration step. The
rim (0.4572 m, regulation) gives a second, exact ruler at the rim plane. A
third falls out of gravity itself: fit the pixel trajectory of a free ball and
the quadratic coefficient is `(g/2) / m_per_px`. Three independent rulers that
should agree — when they don't, the track is contaminated.

**The ball never has to be seen inside the hoop.** Take the last sample above
the rim plane and the first below it, interpolate x at exactly rim height, and
test it against `(rim − ball) / 2 = ±10.9 cm` — the only room a ball has to
pass a ring. At 30 fps the gap you interpolate across is 61% of that whole
clearance, which is why the filming protocol says 120 fps.

**Release is where the ball stops being pushed.** A state machine can't know
the ball has gone until it has visibly separated from the hand — several
frames late, in the same direction every time, which reads as a release speed
a few percent low on every shot and never looks like noise. Fit the free-flight
parabola, walk it backwards, and the release is the last frame the ball still
agreed with it.

## Layout

```
hoopvision/
  constants.py     regulation dimensions - the reason metres are possible
  detect.py        Detection types; ReplayDetector | YoloDetector
  pose.py          COCO-17 keypoints; ReplayPose | YoloPose
  geometry.py      RimScale, BallScale, CourtHomography, camera-drift check
  trajectory.py    ballistic fitting, rim-plane crossing, release refinement
  track.py         gravity-gated ball tracker
  events.py        shot state machine + the make/miss call
  form.py          arm angle, follow-through, knee dip, dribble counting
  metrics.py       per-shot records, session aggregates, CSV
  render/          theme, panels, dashboard composition
  pipeline.py      orchestration
  cli.py           entry point
  bootstrap.py     zero-shot mode: COCO ball + a clicked rim, footage triage
scripts/
  make_sample_clip.py   synthetic clip + ground truth
  verify.py             score the pipeline against that truth
  check_footage.py      is this clip worth labelling?
  mark_rim.py           two clicks -> rim.json
  train.py              fine-tune on Apple Silicon
  calibrate_court.py    four clicks -> homography
tests/                  29 unit tests, no torch required
```

## Going live on real footage

1. **Film at 120 fps**, fixed tripod, ~45° off the shot line, sun behind the
   camera. Details and the reasoning are in the build plan.
2. **Label `ball` and `rim` only** — 300–600 frames, oversampling the ball
   against open sky, overlapping the rim, and motion-blurred at release.
   `person` comes free from COCO pretraining; don't label it.
3. **Train:** `python scripts/train.py --data path/to/data.yaml`
4. **Run:** swap `--replay` for `--weights runs/detect/hoopvision/weights/best.pt`.
   Nothing else changes — every stage downstream of detection is already
   verified against the synthetic clip.

## Known limits

A single camera measures motion in the image plane only, so movement toward or
away from the lens is invisible. A shot along the optical axis reads slow, and
a ball passing *behind* the rim can project to an x that reads as inside. The
apparent-diameter check catches most of the second case; neither is fully
solvable without a second camera. Measure your own error on a hand-scored clip
and put that number in place of this paragraph — a measured error rate with a
stated cause is worth more than an unqualified accuracy claim.

## Licence note

`yolo26n` is AGPL-3.0. That's fine for a portfolio project and for anything you
publish the source of. If this ever needs to be closed-source, RF-DETR's
Nano–Large weights are Apache 2.0 and the detector interface in `detect.py` is
the only thing that would have to change.
