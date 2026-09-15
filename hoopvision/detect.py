"""Detection types and backends.

The pipeline talks to a Detector protocol, never to ultralytics directly.
That is what lets the whole thing run end to end - state machine, metrics,
dashboard - before a single frame has been labelled, using ReplayDetector.
Swap in YoloDetector once best.pt exists and nothing downstream changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol


@dataclass(frozen=True)
class Detection:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def radius_px(self) -> float:
        """Mean half-extent. Averaging both axes is more stable than either
        alone when the box is slightly clipped by motion blur."""
        return (self.w + self.h) / 4

    def to_dict(self) -> dict:
        return {
            "cls": self.cls, "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2, "conf": self.conf,
        }

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        return Detection(d["cls"], d["x1"], d["y1"], d["x2"], d["y2"], d.get("conf", 1.0))


@dataclass
class FrameDetections:
    frame_idx: int
    ball: Optional[Detection] = None
    rim: Optional[Detection] = None
    backboard: Optional[Detection] = None


class Detector(Protocol):
    def detect(self, frame_idx: int, image) -> FrameDetections: ...


def _best(dets: Iterable[Detection], cls: str) -> Optional[Detection]:
    cands = [d for d in dets if d.cls == cls]
    return max(cands, key=lambda d: d.conf) if cands else None


class ReplayDetector:
    """Replays boxes from a JSON sidecar.

    Two jobs: (1) run the pipeline on the synthetic clip before any model
    exists, (2) freeze a real clip's detections so downstream changes can be
    tested without re-running inference.
    """

    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text())
        self._frames: Dict[int, List[Detection]] = {
            int(k): [Detection.from_dict(d) for d in v] for k, v in raw["frames"].items()
        }
        self.meta = raw.get("meta", {})

    def detect(self, frame_idx: int, image=None) -> FrameDetections:
        dets = self._frames.get(frame_idx, [])
        return FrameDetections(
            frame_idx,
            ball=_best(dets, "ball"),
            rim=_best(dets, "rim"),
            backboard=_best(dets, "backboard"),
        )


class YoloDetector:
    """Wraps a fine-tuned ultralytics model. Imported lazily so the repo
    installs and tests without torch."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 960, device: str = "mps"):
        from ultralytics import YOLO  # noqa: PLC0415 - lazy on purpose

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.names = self.model.names

    def detect(self, frame_idx: int, image) -> FrameDetections:
        res = self.model.predict(
            image, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        dets: List[Detection] = []
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            dets.append(Detection(self.names[int(box.cls)], x1, y1, x2, y2, float(box.conf)))
        return FrameDetections(
            frame_idx,
            ball=_best(dets, "ball"),
            rim=_best(dets, "rim"),
            backboard=_best(dets, "backboard"),
        )
