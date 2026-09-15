#!/usr/bin/env python3
"""Generate a synthetic shooting clip with exact ground truth.

Why this exists: it lets every stage after detection - tracking, the state
machine, the make/miss test, the form metrics, the dashboard - be built and
verified before a single frame has been labelled. The ball follows a real
parabola under real gravity, the skeleton's elbow angle is real two-link
IK, and the dribbles are real bounces. So when the pipeline reports a 47
degree entry angle, truth.json says whether it is right.

    python scripts/make_sample_clip.py --out sample/

Writes clip.mp4, detections.json, poses.json, truth.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoopvision.constants import BALL_DIAMETER_M, G, RIM_HEIGHT_M, RIM_INNER_DIAMETER_M

# ---------------------------------------------------------------- scene
W, H = 960, 640
FPS = 120.0
S = 110.0                      # px per metre
SHOOTER_X_PX = 300.0
FLOOR_Y_PX = 600.0
X_RIM_M = 4.18
BALL_R_PX = BALL_DIAMETER_M / 2 * S
RIM_W_PX = RIM_INNER_DIAMETER_M * S

UPPER_ARM_M = 0.31
FOREARM_M = 0.31
SHOULDER_Y_M = 1.45
HIP_Y_M = 0.95
KNEE_Y_M = 0.50
ANKLE_Y_M = 0.06
HALF_SHOULDER_M = 0.19
HALF_HIP_M = 0.13


def to_px(x_m: float, y_m: float) -> Tuple[float, float]:
    return SHOOTER_X_PX + x_m * S, FLOOR_Y_PX - y_m * S


def elbow_from_ik(sh: Tuple[float, float], wr: Tuple[float, float],
                  l1: float = UPPER_ARM_M, l2: float = FOREARM_M,
                  flip: float = 1.0) -> Tuple[Tuple[float, float], float]:
    """Two-link IK in metres. Returns (elbow, interior angle at elbow, deg)."""
    sx, sy = sh
    wx, wy = wr
    dx, dy = wx - sx, wy - sy
    d = math.hypot(dx, dy)
    d = min(max(d, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    ux, uy = dx / max(math.hypot(dx, dy), 1e-9), dy / max(math.hypot(dx, dy), 1e-9)
    px, py = sx + a * ux, sy + a * uy
    ex, ey = px + flip * h * uy, py - flip * h * ux
    cos = (l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2)
    ang = math.degrees(math.acos(min(max(cos, -1.0), 1.0)))
    return (ex, ey), ang


def bounce_y(t: float, h: float) -> float:
    """One dribble cycle: apex h, floor, apex h."""
    half = math.sqrt(2 * h / G)
    t = t % (2 * half)
    dt = t if t <= half else (2 * half - t)
    return max(h - 0.5 * G * dt * dt, 0.0)


def solve_v0(dx: float, dy: float, theta_deg: float) -> float:
    th = math.radians(theta_deg)
    denom = 2 * math.cos(th) ** 2 * (math.tan(th) * dx - dy)
    if denom <= 0:
        raise ValueError("unreachable target at that release angle")
    return math.sqrt(G * dx * dx / denom)


# ---------------------------------------------------------------- frames
def build(shots: List[dict]) -> Tuple[List[dict], Dict[int, list], Dict[int, list], List[dict]]:
    frames: List[dict] = []          # per-frame scene state
    dets: Dict[int, list] = {}
    poses: Dict[int, list] = {}
    truth: List[dict] = []

    idx = 0
    release_y_m = 2.20
    wrist_offset_m = 0.15            # ball sits this far above the hand

    for shot in shots:
        n_drib = shot["dribbles"]
        theta = shot["release_angle_deg"]
        err = shot["lateral_error_m"]

        # --- dribbles -------------------------------------------------
        h_drib = 0.35
        cycle = int(round(2 * math.sqrt(2 * h_drib / G) * FPS))
        for k in range(n_drib):
            for f in range(cycle):
                y = bounce_y(f / FPS, h_drib) + BALL_DIAMETER_M / 2
                bx, by = 0.35, y
                wx, wy = 0.33, max(y + 0.24, 0.86)
                frames.append(dict(ball=(bx, by), wrist=(wx, wy), dip=0.0, phase="dribble"))
                idx += 1

        # --- gather to the pocket ------------------------------------
        pocket = (0.26, 1.40)
        start = (0.35, h_drib + BALL_DIAMETER_M / 2)
        for f in range(24):
            u = f / 23
            bx = start[0] + u * (pocket[0] - start[0])
            by = start[1] + u * (pocket[1] - start[1])
            frames.append(dict(ball=(bx, by), wrist=(bx - 0.02, by - wrist_offset_m),
                               dip=0.10 * u, phase="gather"))
            idx += 1
        for f in range(14):          # hold the pocket -> LOADED
            frames.append(dict(ball=pocket, wrist=(pocket[0] - 0.02, pocket[1] - wrist_offset_m),
                               dip=0.10, phase="load"))
            idx += 1

        # --- lift to release -----------------------------------------
        # The ball rides up the outside of the shoulder. Without the bulge
        # the wrist would pass straight through the shoulder joint, which is
        # anatomically impossible and drives the IK elbow angle to zero.
        lift = 22
        for f in range(lift):
            u = (f + 1) / lift
            bx = pocket[0] * (1 - u) + 0.22 * math.sin(math.pi * u)
            by = pocket[1] + u * (release_y_m - pocket[1])
            frames.append(dict(ball=(bx, by), wrist=(bx, by - wrist_offset_m),
                               dip=0.10 * (1 - u), phase="lift"))
            idx += 1
        release_idx = idx - 1

        # --- flight ---------------------------------------------------
        target_x = X_RIM_M + err
        dy = RIM_HEIGHT_M - release_y_m
        v0 = solve_v0(target_x, dy, theta)
        th = math.radians(theta)
        vx, vy0 = v0 * math.cos(th), v0 * math.sin(th)
        t_rim = target_x / vx
        n_flight = int(round((t_rim + 0.22) * FPS))
        for f in range(n_flight):
            t = (f + 1) / FPS
            bx = vx * t
            by = release_y_m + vy0 * t - 0.5 * G * t * t
            wy = release_y_m - wrist_offset_m + min(t * 2.0, 0.12)
            frames.append(dict(ball=(bx, by), wrist=(0.02, wy), dip=0.0, phase="flight"))
            idx += 1

        # --- reset: retrieve the ball, no teleporting -----------------
        last_b = frames[-1]["ball"]
        home = (0.35, h_drib + BALL_DIAMETER_M / 2)
        for f in range(120):          # ~1 s: the player walks it back
            u = (f + 1) / 120
            bx = last_b[0] + u * (home[0] - last_b[0])
            by = last_b[1] + u * (home[1] - last_b[1])
            frames.append(dict(ball=(bx, by), wrist=(bx - 0.02, max(by + 0.22, 0.88)),
                               dip=0.0, phase="reset"))
            idx += 1

        # --- ground truth ---------------------------------------------
        vy_rim = vy0 - G * t_rim
        entry = math.degrees(math.atan2(abs(vy_rim), abs(vx)))
        x_at_rim = X_RIM_M + err
        offset = x_at_rim - X_RIM_M
        made = abs(offset) <= (RIM_INNER_DIAMETER_M - BALL_DIAMETER_M) / 2
        # The set point is measured at the pocket hold, where the knee dip is
        # at its deepest - so the shoulder has dropped with it. Computing it
        # from the upright shoulder would be truth that never occurs in the clip.
        _, elbow_rel = elbow_from_ik((HALF_SHOULDER_M, SHOULDER_Y_M),
                                     (0.0, release_y_m - wrist_offset_m))
        _, elbow_set = elbow_from_ik((HALF_SHOULDER_M, SHOULDER_Y_M - 0.10),
                                     (pocket[0] - 0.02, pocket[1] - wrist_offset_m))
        truth.append(dict(
            dribbles=n_drib,
            release_frame=release_idx,
            release_speed_ms=v0,
            release_angle_deg=theta,
            entry_angle_deg=entry,
            miss_offset_m=offset,
            outcome="make" if made else "miss",
            elbow_at_release_deg=elbow_rel,
            elbow_set_point_deg=elbow_set,
        ))

    # --- detections + poses -------------------------------------------
    rim_px = to_px(X_RIM_M, RIM_HEIGHT_M)
    for i, st in enumerate(frames):
        bx, by = to_px(*st["ball"])
        dets[i] = [
            dict(cls="ball", x1=bx - BALL_R_PX, y1=by - BALL_R_PX,
                 x2=bx + BALL_R_PX, y2=by + BALL_R_PX, conf=0.97),
            dict(cls="rim", x1=rim_px[0] - RIM_W_PX / 2, y1=rim_px[1] - 4,
                 x2=rim_px[0] + RIM_W_PX / 2, y2=rim_px[1] + 4, conf=0.99),
        ]
        poses[i] = make_pose(st)
    return frames, dets, poses, truth


def make_pose(state: dict) -> list:
    """17 COCO keypoints in pixels, with real IK for the shooting arm."""
    dip = state["dip"]
    wr = state["wrist"]
    sh_y = SHOULDER_Y_M - dip
    hip_y = HIP_Y_M - dip
    knee_y = KNEE_Y_M - dip * 0.6

    r_sh = (HALF_SHOULDER_M, sh_y)
    l_sh = (-HALF_SHOULDER_M, sh_y)
    r_el, _ = elbow_from_ik(r_sh, wr, flip=1.0)
    l_wr = (wr[0] - 0.26, wr[1] - 0.02)
    l_el, _ = elbow_from_ik(l_sh, l_wr, flip=-1.0)

    pts_m = {
        "nose": (0.0, sh_y + 0.24),
        "left_eye": (-0.03, sh_y + 0.27), "right_eye": (0.03, sh_y + 0.27),
        "left_ear": (-0.08, sh_y + 0.26), "right_ear": (0.08, sh_y + 0.26),
        "left_shoulder": l_sh, "right_shoulder": r_sh,
        "left_elbow": l_el, "right_elbow": r_el,
        "left_wrist": l_wr, "right_wrist": wr,
        "left_hip": (-HALF_HIP_M, hip_y), "right_hip": (HALF_HIP_M, hip_y),
        "left_knee": (-HALF_HIP_M, knee_y), "right_knee": (HALF_HIP_M, knee_y),
        "left_ankle": (-HALF_HIP_M, ANKLE_Y_M), "right_ankle": (HALF_HIP_M, ANKLE_Y_M),
    }
    order = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
    out = []
    for name in order:
        x, y = to_px(*pts_m[name])
        out.append([x, y, 0.95])
    return out


def render(frames, dets, poses, path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
    rim_px = to_px(X_RIM_M, RIM_HEIGHT_M)
    for i, st in enumerate(frames):
        img = np.full((H, W, 3), (168, 172, 176), dtype=np.uint8)
        cv2.rectangle(img, (0, int(FLOOR_Y_PX)), (W, H), (120, 128, 136), -1)
        cv2.line(img, (0, int(FLOOR_Y_PX)), (W, int(FLOOR_Y_PX)), (200, 205, 210), 2)
        # backboard + rim
        cv2.rectangle(img, (int(rim_px[0] + RIM_W_PX / 2), int(rim_px[1] - 110)),
                      (int(rim_px[0] + RIM_W_PX / 2 + 12), int(rim_px[1] + 10)),
                      (235, 235, 235), -1)
        cv2.line(img, (int(rim_px[0] - RIM_W_PX / 2), int(rim_px[1])),
                 (int(rim_px[0] + RIM_W_PX / 2), int(rim_px[1])), (30, 70, 200), 5)
        # skeleton
        kp = np.asarray(poses[i])
        for a, b in [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
                     (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
            cv2.line(img, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])),
                     (40, 40, 45), 4, cv2.LINE_AA)
        cv2.circle(img, (int(kp[0][0]), int(kp[0][1])), 14, (40, 40, 45), -1, cv2.LINE_AA)
        # ball
        bx, by = to_px(*st["ball"])
        cv2.circle(img, (int(bx), int(by)), int(BALL_R_PX), (32, 92, 208), -1, cv2.LINE_AA)
        cv2.circle(img, (int(bx), int(by)), int(BALL_R_PX), (20, 50, 120), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shots = [
        dict(dribbles=2, release_angle_deg=56.0, lateral_error_m=0.00),
        dict(dribbles=0, release_angle_deg=54.0, lateral_error_m=0.05),
        dict(dribbles=3, release_angle_deg=51.0, lateral_error_m=0.22),   # miss
        dict(dribbles=1, release_angle_deg=57.0, lateral_error_m=-0.04),
        dict(dribbles=2, release_angle_deg=53.0, lateral_error_m=-0.19),  # miss
    ]
    frames, dets, poses, truth = build(shots)

    (out / "detections.json").write_text(json.dumps(
        {"meta": {"fps": FPS, "w": W, "h": H},
         "frames": {str(k): v for k, v in dets.items()}}))
    (out / "poses.json").write_text(json.dumps(
        {"poses": {str(k): v for k, v in poses.items()}}))
    (out / "truth.json").write_text(json.dumps(
        {"fps": FPS, "scale_m_per_px": 1.0 / S, "floor_y_px": FLOOR_Y_PX,
         "shots": truth}, indent=2))

    if not args.no_video:
        render(frames, dets, poses, out / "clip.mp4")

    made = sum(1 for t in truth if t["outcome"] == "make")
    print(f"{len(frames)} frames, {len(truth)} shots, {made} makes -> {out}")


if __name__ == "__main__":
    main()
