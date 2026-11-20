import math
from datetime import datetime

from parking_rules.tracker import Tracker
from parking_rules.zones import Zone, check_pose, zones_from_json, zones_from_landmarks, zones_to_json


def test_tracker_confirms_after_min_hits_and_averages():
    t = Tracker(gate=0.5, min_hits=3)
    t.update('fire hydrant', 2.0, 1.0, 0)
    t.update('fire hydrant', 2.2, 1.0, 1)
    assert t.confirmed() == []
    t.update('fire hydrant', 2.1, 1.0, 2)
    [lm] = t.confirmed()
    assert math.isclose(lm.x, 2.1) and math.isclose(lm.y, 1.0)


def test_tracker_keeps_far_apart_hydrants_separate():
    t = Tracker(gate=0.5, min_hits=1)
    t.update('fire hydrant', 0, 0, 0)
    t.update('fire hydrant', 3, 0, 0)
    assert len(t.confirmed()) == 2


def test_zones_roundtrip_json():
    t = Tracker(min_hits=1)
    t.update('fire hydrant', 1, 2, 0)
    zones = zones_from_landmarks(t.confirmed())
    assert zones_from_json(zones_to_json(zones)) == zones


HYDRANT = [Zone('fire hydrant', 5.0, 0.0, 'too close to a fire hydrant', radius=1.0)]


def test_parking_on_top_of_hydrant_is_illegal():
    assert check_pose(4.5, 0.0, 0.0, HYDRANT)


def test_parking_just_inside_radius_is_illegal():
    # Front bumper at x + 1.05; hydrant zone starts at 4.0 -> base_link at 3.0 puts the bumper at 4.05.
    assert check_pose(3.0, 0.0, 0.0, HYDRANT)


def test_parking_well_away_is_legal():
    assert check_pose(1.0, 0.0, 0.0, HYDRANT) == []
    assert check_pose(5.0, 2.0, 0.0, HYDRANT) == []


def test_yaw_matters():
    # base_link at 3.5 facing +x: the front bumper reaches 4.55 (inside the zone that starts at 4.0, illegal).
    # Facing -x: the rear bumper is the closest part and reaches only 3.72 (legal).
    assert check_pose(3.5, 0.0, 0.0, HYDRANT)
    assert check_pose(3.5, 0.0, math.pi, HYDRANT) == []


# --- signs -------------------------------------------------------------------------------------------------------

def _sign_zone(side=-1):
    """Car drove along +x; a no-parking sign at (5, -1.5) on its right (side=-1)."""
    t = Tracker(min_hits=1)
    t.update('no_parking', 5.0, -1.5 * -side, 0, heading=0.0, side=side)
    return zones_from_landmarks(t.confirmed())


def test_sign_strip_starts_at_sign_and_runs_with_traffic():
    [z] = _sign_zone()
    xs = [p[0] for p in z.polygon]
    ys = [p[1] for p in z.polygon]
    assert min(xs) == 5.0 and math.isclose(max(xs), 8.0)        # 3 m from the sign onward
    assert math.isclose(min(ys), -1.5) and math.isclose(max(ys), -0.3)  # 1.2 m from the sign into the road


def test_parking_after_sign_on_its_side_is_illegal():
    zones = _sign_zone()
    [bad] = check_pose(6.0, -0.9, 0.0, zones)
    assert 'Zeichen 286' in bad.reason


def test_parking_before_sign_is_legal():
    assert check_pose(3.0, -0.9, 0.0, _sign_zone()) == []   # car ends at x = 4.05, sign at 5


def test_parking_on_the_other_side_of_the_road_is_legal():
    assert check_pose(6.0, 1.0, 0.0, _sign_zone()) == []


def test_sign_on_left_makes_strip_to_its_right():
    [z] = _sign_zone(side=+1)   # sign at (5, +1.5)
    ys = [p[1] for p in z.polygon]
    assert math.isclose(max(ys), 1.5) and math.isclose(min(ys), 0.3)


def test_sign_without_heading_gives_no_zone():
    t = Tracker(min_hits=1)
    t.update('no_parking', 5.0, -1.5, 0)
    assert zones_from_landmarks(t.confirmed()) == []


def test_misread_sign_stays_one_landmark_with_majority_label():
    t = Tracker(min_hits=1)
    for label in ('no_parking', 'no_stopping', 'no_parking'):
        t.update(label, 5.0, -1.5, 0, heading=0.0, side=-1)
    [lm] = t.confirmed()
    assert lm.label == 'no_parking' and lm.hits == 3


def test_hydrant_and_sign_at_same_place_stay_separate():
    t = Tracker(min_hits=1)
    t.update('fire hydrant', 5.0, -1.5, 0)
    t.update('no_parking', 5.0, -1.5, 0, heading=0.0, side=-1)
    assert len(t.confirmed()) == 2


def test_strip_zone_json_roundtrip():
    zones = _sign_zone()
    assert zones_from_json(zones_to_json(zones)) == zones


# --- signs with extra plates: legality depends on time and duration -----------------------------------------------

def _timed_zone(label, plate_text, status='read'):
    from parking_rules.plates import parse
    t = Tracker(min_hits=1)
    lm = t.update(label, 5.0, -1.5, 0, heading=0.0, side=-1)
    return zones_from_landmarks(t.confirmed(), plates={lm.id: (status, parse(plate_text) if plate_text else None)})


FRIDAY_9 = datetime(2026, 11, 20, 9, 0)


def test_parking_sign_max_duration_in_zone():
    zones = _timed_zone('parking', 'Mo-Fr 8-18 h max. 2 Std.')
    assert check_pose(6.0, -0.9, 0.0, zones, when=FRIDAY_9, minutes=60) == []
    [bad] = check_pose(6.0, -0.9, 0.0, zones, when=FRIDAY_9, minutes=180)
    assert bad.reason.startswith('max. 2 h parking')
    assert check_pose(6.0, -0.9, 0.0, zones, when=datetime(2026, 11, 21, 9), minutes=600) == []   # Saturday


def test_parking_sign_without_plate_never_blocks():
    zones = _timed_zone('parking', None, status='none')
    assert check_pose(6.0, -0.9, 0.0, zones, when=FRIDAY_9, minutes=24 * 60) == []


def test_no_parking_sign_only_during_plate_hours():
    zones = _timed_zone('no_parking', 'Mo-Fr 7-9 h')
    assert check_pose(6.0, -0.9, 0.0, zones, when=datetime(2026, 11, 20, 8), minutes=30)
    assert check_pose(6.0, -0.9, 0.0, zones, when=FRIDAY_9, minutes=120) == []


def test_unread_plate_blocks_until_read():
    [bad] = check_pose(6.0, -0.9, 0.0, _timed_zone('parking', None, status='unread'), when=FRIDAY_9)
    assert 'not read yet' in bad.reason


def test_timed_zone_json_roundtrip():
    zones = _timed_zone('parking', 'Mo-Fr 8-18 h max. 2 Std.')
    back = zones_from_json(zones_to_json(zones))
    assert check_pose(6.0, -0.9, 0.0, back, when=FRIDAY_9, minutes=180)[0].reason == \
        check_pose(6.0, -0.9, 0.0, zones, when=FRIDAY_9, minutes=180)[0].reason
