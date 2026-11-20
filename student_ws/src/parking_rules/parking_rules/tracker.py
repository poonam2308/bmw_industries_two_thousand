"""Fuse noisy single-frame detections into stable landmarks in a fixed frame.

One detection can be wrong (a red bin looks like a hydrant for a frame). A landmark only counts once it was seen
`min_hits` times at about the same place; its position is the running mean of all sightings.

Signs also need to know *which way the road goes* and *which side of the road they are on*, because a no-parking
sign applies from the sign onward in the direction of travel, on its side only. So each sighting can carry the car's
heading and whether the sign was left or right of the car; both are averaged as well.
The label is a majority vote, so a sign misread once as "no stopping" instead of "no parking" stays one landmark.
"""
from collections import Counter
from dataclasses import dataclass, field
import math

# Labels in the same group are the same kind of object for association (one physical sign, two possible readings).
GROUPS = {'no_parking': 'sign', 'no_stopping': 'sign'}


def group_of(label):
    return GROUPS.get(label, label)


@dataclass
class Landmark:
    id: int
    label: str
    x: float
    y: float
    hits: int = 0
    last_seen: float = 0.0
    xs: list = field(default_factory=list, repr=False)
    ys: list = field(default_factory=list, repr=False)
    votes: Counter = field(default_factory=Counter, repr=False)
    _hx: float = field(default=0.0, repr=False)   # sum of heading unit vectors
    _hy: float = field(default=0.0, repr=False)
    _side: float = field(default=0.0, repr=False)  # sum of +1 (left of the car) / -1 (right)

    @property
    def heading(self):
        """Mean heading of the car when it saw this landmark (rad), or None if never given."""
        return math.atan2(self._hy, self._hx) if (self._hx or self._hy) else None

    @property
    def side(self):
        """+1: the landmark was on the car's left, -1: on its right, 0: unknown."""
        return (self._side > 0) - (self._side < 0)

    def add(self, label, x, y, stamp, window, heading=None, side=None):
        self.xs = (self.xs + [x])[-window:]
        self.ys = (self.ys + [y])[-window:]
        self.x = sum(self.xs) / len(self.xs)
        self.y = sum(self.ys) / len(self.ys)
        self.hits += 1
        self.last_seen = stamp
        self.votes[label] += 1
        self.label = self.votes.most_common(1)[0][0]
        if heading is not None:
            self._hx += math.cos(heading)
            self._hy += math.sin(heading)
        if side:
            self._side += 1 if side > 0 else -1


class Tracker:
    def __init__(self, gate=0.75, min_hits=3, window=30):
        self.gate = gate          # m: a sighting closer than this to a landmark of the same group belongs to it
        self.min_hits = min_hits  # sightings needed before the landmark is trusted
        self.window = window      # average over the last N sightings
        self.landmarks = []
        self._next_id = 0

    def update(self, label, x, y, stamp, heading=None, side=None):
        best, best_d = None, self.gate
        for lm in self.landmarks:
            d = math.hypot(lm.x - x, lm.y - y)
            if group_of(lm.label) == group_of(label) and d < best_d:
                best, best_d = lm, d
        if best is None:
            best = Landmark(self._next_id, label, x, y)
            self._next_id += 1
            self.landmarks.append(best)
        best.add(label, x, y, stamp, self.window, heading, side)
        return best

    def confirmed(self):
        return [lm for lm in self.landmarks if lm.hits >= self.min_hits]
