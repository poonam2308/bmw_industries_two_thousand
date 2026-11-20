"""Parking rules: landmarks -> no-parking zones -> is this parking pose legal?

Pure Python, no ROS. Use `check_pose` from your parking planner to drop illegal candidate spots before sending a
Nav2 goal.

Two kinds of zone:
  - circle around an object (fire hydrant: keep it clear)
  - strip that starts at a sign and runs along the road in the direction of travel, on the sign's side of the road
    (StVO: no-parking / no-stopping signs apply from the sign onward until the next junction or an end sign)
A sign's zone can carry the schedule read from its extra plate ("Mo-Fr 8-18 h, max. 2 Std."); then legality also
depends on *when* and *how long* you park (see plates.py). A parking sign (Zeichen 314) without a plate allows
parking, so it never makes a pose illegal.

Distances are scaled for BITT (about 1.3 m long, roughly a quarter of a real car). For reference, real German rules
(StVO §12): no parking within 5 m of a junction, none in front of driveways or lowered kerbs; hydrants and fire-brigade
access must stay clear.
"""
from dataclasses import dataclass, asdict, replace
from datetime import datetime
import json
import math

from .plates import Schedule, violation

# Car outline in base_link (centre of the rear axle, x forward, y left), with padding. From the sim's
# robot_config.yaml (footprint_with_padding_base_link).
FOOTPRINT = [(-0.22, -0.3), (-0.22, 0.3), (1.05, 0.3), (1.05, -0.3)]


@dataclass
class CircleRule:
    radius: float
    reason: str


@dataclass
class StripRule:
    length: float  # m along the road from the sign (toy scale; on a real street: until the next junction)
    width: float   # m from the sign towards the road (covers the parking lane)
    reason: str


# label -> rule. Add a line per new rule.
RULES = {
    'fire hydrant': CircleRule(1.0, 'too close to a fire hydrant'),
    'no_parking': StripRule(3.0, 1.2, 'no-parking sign (Zeichen 286)'),
    'no_stopping': StripRule(3.0, 1.2, 'no-stopping sign (Zeichen 283)'),
    'parking': StripRule(3.0, 1.2, 'parking area (Zeichen 314)'),
}


@dataclass
class Zone:
    kind: str
    x: float              # the object / sign position
    y: float
    reason: str
    radius: float = 0.0   # circle zones
    polygon: list = None  # strip zones: [[x, y], ...] corners in the fixed frame, convex
    plate: str = 'none'   # extra plate: 'none' (not seen), 'unread' (seen, not understood yet), 'read'
    schedule: dict = None  # Schedule.to_dict() of the plate, when read


def strip_polygon(x, y, heading, side, length, width):
    """Rectangle starting at the sign, `length` along `heading`, `width` from the sign towards the road.

    side: +1 if the sign stood left of the car (road is to its right), -1 if right (road to its left)."""
    dx, dy = math.cos(heading), math.sin(heading)
    nx, ny = (dy, -dx) if side > 0 else (-dy, dx)
    return [[x, y], [x + length * dx, y + length * dy],
            [x + length * dx + width * nx, y + length * dy + width * ny], [x + width * nx, y + width * ny]]


def zones_from_landmarks(landmarks, rules=RULES, plates=None):
    """plates: {landmark id: (status, Schedule or None)} from the plate reader; missing = no plate seen."""
    plates = plates or {}
    zones = []
    for lm in landmarks:
        rule = rules.get(lm.label)
        if isinstance(rule, CircleRule):
            zones.append(Zone(lm.label, lm.x, lm.y, rule.reason, radius=rule.radius))
        elif isinstance(rule, StripRule) and lm.heading is not None and lm.side:
            status, schedule = plates.get(lm.id, ('none', None))
            zones.append(Zone(lm.label, lm.x, lm.y, rule.reason,
                              polygon=strip_polygon(lm.x, lm.y, lm.heading, lm.side, rule.length, rule.width),
                              plate=status, schedule=schedule.to_dict() if schedule else None))
    return zones


def zones_to_json(zones):
    return json.dumps([asdict(z) for z in zones])


def zones_from_json(text):
    return [Zone(**z) for z in json.loads(text)]


def footprint_at(x, y, yaw, footprint=FOOTPRINT):
    """Car outline in the fixed frame when base_link is at (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in footprint]


def _edges(poly):
    return list(zip(poly, poly[1:] + poly[:1]))


def _point_in_polygon(px, py, poly):
    inside = False
    for (x1, y1), (x2, y2) in _edges(poly):
        if (y1 > py) != (y2 > py) and px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _dist_point_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def polygon_hits_circle(poly, cx, cy, r):
    if _point_in_polygon(cx, cy, poly):
        return True
    return any(_dist_point_segment(cx, cy, *a, *b) < r for a, b in _edges(poly))


def convex_polygons_overlap(a, b):
    """Separating axis test for two convex polygons (touching edges do not count as overlap)."""
    for poly in (a, b):
        for (x1, y1), (x2, y2) in _edges(poly):
            ax, ay = y1 - y2, x2 - x1  # edge normal
            pa = [ax * x + ay * y for x, y in a]
            pb = [ax * x + ay * y for x, y in b]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                return False
    return True


def touches(poly, zone):
    if zone.polygon:
        return convex_polygons_overlap(poly, [tuple(p) for p in zone.polygon])
    return polygon_hits_circle(poly, zone.x, zone.y, zone.radius)


def check_pose(x, y, yaw, zones, when=None, minutes=60, footprint=FOOTPRINT):
    """Zones that make parking with base_link at (x, y, yaw) illegal, starting `when` (local time at the signs,
    default now) for `minutes`. Empty list = legal. Each returned zone's `reason` says why."""
    when = when or datetime.now()
    poly = footprint_at(x, y, yaw, footprint)
    bad = []
    for z in zones:
        if not touches(poly, z):
            continue
        schedule = Schedule.from_dict(z.schedule) if z.schedule else None
        why = violation(z.kind, z.reason, schedule, z.plate, when, minutes)
        if why:
            bad.append(replace(z, reason=why))
    return bad
