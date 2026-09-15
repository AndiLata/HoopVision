"""Bootstrap mode: the fixed rim, and the footage triage report.

Nothing here touches torch - the model wrapper is a thin call, but the
geometry and the go/no-go logic are where a wrong answer costs you an
afternoon of shooting.
"""
import pytest

from hoopvision.bootstrap import FixedRim, analyse_detections, verdicts
from hoopvision.constants import BALL_DIAMETER_M, RIM_INNER_DIAMETER_M
from hoopvision.geometry import RimScale


def test_two_clicks_give_a_usable_scale():
    """Clicking the ring's inner edges must reproduce the regulation scale."""
    rim = FixedRim.from_two_clicks((400.0, 300.0), (450.0, 302.0))
    rs = RimScale.from_rim_box(rim.x1, rim.y1, rim.x2, rim.y2)
    assert rs.rim_width_px == pytest.approx(50.0)
    assert rs.px_to_m(50.0) == pytest.approx(RIM_INNER_DIAMETER_M)
    # and the ball should then measure ~26 px at that depth
    assert rs.m_to_px(BALL_DIAMETER_M) == pytest.approx(26.2, abs=0.5)


def test_clicks_in_either_order_give_the_same_rim():
    a = FixedRim.from_two_clicks((400.0, 300.0), (450.0, 300.0))
    b = FixedRim.from_two_clicks((450.0, 300.0), (400.0, 300.0))
    assert (a.x1, a.x2) == (b.x1, b.x2)


def test_rim_round_trips_through_json(tmp_path):
    rim = FixedRim(400.0, 297.0, 450.0, 303.0)
    p = tmp_path / "rim.json"
    rim.save(p)
    assert FixedRim.load(p) == rim


# ---------------------------------------------------------------- report

def hits(n, present, diam=26.0, cy=200.0):
    """n frames, `present` a set of frame indices where the ball was found."""
    return [(i, diam, cy) if i in present else (i, None, None) for i in range(n)]


def test_longest_gap_is_measured_not_guessed():
    r = analyse_detections(hits(20, set(range(20)) - {5, 6, 7}), 20, 120.0, 1920, 1080)
    assert r.longest_gap_frames == 3
    assert r.longest_gap_ms == pytest.approx(25.0)


def test_gap_at_the_very_end_still_counts():
    r = analyse_detections(hits(20, set(range(15))), 20, 120.0, 1920, 1080)
    assert r.longest_gap_frames == 5


def test_detection_rate_and_median_size():
    r = analyse_detections(hits(100, set(range(80))), 100, 120.0, 1920, 1080)
    assert r.detection_rate == pytest.approx(0.8)
    assert r.median_ball_px == pytest.approx(26.0)


def test_thirty_fps_fails_the_check():
    r = analyse_detections(hits(100, set(range(100))), 100, 30.0, 1920, 1080)
    levels = {h: lvl for lvl, h, _ in verdicts(r)}
    assert any(lvl == "fail" and "fps" in h for lvl, h, _ in verdicts(r))


def test_120fps_good_footage_passes_everything():
    r = analyse_detections(hits(300, set(range(300))), 300, 120.0, 1920, 1080)
    assert all(lvl == "ok" for lvl, _, _ in verdicts(r))


def test_tiny_ball_fails():
    r = analyse_detections(hits(300, set(range(300)), diam=8.0), 300, 120.0, 1920, 1080)
    assert any(lvl == "fail" and "px across" in h for lvl, h, _ in verdicts(r))


def test_no_detections_at_all_is_a_clear_failure():
    r = analyse_detections(hits(100, set()), 100, 120.0, 1920, 1080)
    assert r.median_ball_px is None
    assert any(lvl == "fail" for lvl, _, _ in verdicts(r))


def test_ball_lost_in_flight_is_flagged():
    """Detections only ever in the bottom half - the backlit-sky signature."""
    h = [(i, 26.0, 900.0) for i in range(200)]
    r = analyse_detections(h, 200, 120.0, 1920, 1080)
    assert r.upper_half_rate == 0.0
    assert any("upper half" in headline for _, headline, _ in verdicts(r))


def test_zero_shot_detection_rate_warns_rather_than_fails():
    """60% is normal for COCO and is an argument for fine-tuning, not a
    reason to throw the clip away - the report must say so."""
    r = analyse_detections(hits(300, set(range(180))), 300, 120.0, 1920, 1080)
    got = [(lvl, h) for lvl, h, _ in verdicts(r) if "of frames" in h]
    assert got and got[0][0] == "warn"
