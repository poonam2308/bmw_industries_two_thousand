"""From detections to what the car shows. Pure Python, no ROS, no OpenCV: easy to test.

classify()  one frame's detections -> a scene name and the detection it is about
Director    turns the stream of per-frame scenes into stable reactions: a scene must be seen for a few frames in a row
            before the display changes, a message stays up for a minimum time, and while the scene stays the same
            the text rotates now and then (we are stuck in traffic, after all).
"""
import time
from dataclasses import dataclass

from .detector import BIKES, VEHICLES
from .messages import ANIMATIONS, FLIRT_PROFILE, MESSAGES, PROFILE


TIER_SCENES = {'premium': 'fancy_car', 'budget': 'budget_car'}  # brand check tier -> scene

# Minimum box sizes as a fraction of the image (bigger = closer). People need more, so that someone far away in
# the background does not count as "behind the car": in a recording from the car, people far down a corridor were
# 1.6-1.8 % of the image, people close to the car 8-30 %. Set from the command line with --min-person.
MIN_AREA = 0.01
MIN_PERSON_AREA = 0.05

# With --pitch-people, people on foot get the job pitch ('person_pitch') instead of the pedestrian hello.
PITCH_PEOPLE = False


def overlap(a, b):
    """Fraction of box a that lies inside box b."""
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = (ax2 - ax1) * (ay2 - ay1)
    return w * h / area if area > 0 else 0.0


def classify(detections, min_area=MIN_AREA, min_person_area=None):
    """The scene for one frame: fancy_car, budget_car, car, cyclist, pedestrian or nobody, plus the detection it is
    about.

    The closest (biggest) subject wins. Tiny boxes (far away, below min_area of the image) are ignored, and people
    on foot only count from min_person_area (default: MIN_PERSON_AREA). Riders and drivers do not need it: a
    bicycle or a car is judged by its own box.
    """
    if min_person_area is None:
        min_person_area = MIN_PERSON_AREA
    dets = [d for d in detections if d.area >= min_area]
    bikes = [d for d in dets if d.label in BIKES]
    vehicles = [d for d in dets if d.label in VEHICLES]
    candidates = [('cyclist', b) for b in bikes]
    for v in vehicles:
        candidates.append((TIER_SCENES.get(v.tier, 'car'), v))
    for p in (d for d in dets if d.label == 'person'):
        if any(overlap(p, v) > 0.7 for v in vehicles):
            continue  # the driver seen through the windscreen belongs to the car
        if any(overlap(p, b) > 0.2 or overlap(b, p) > 0.2 for b in bikes):
            continue  # the rider belongs to the bicycle
        if p.area < min_person_area:
            continue  # in the background, not behind the car
        candidates.append(('pedestrian', p))
    if not candidates:
        return 'nobody', None
    return max(candidates, key=lambda c: c[1].area)


@dataclass
class Reaction:
    scene: str      # classify() scene, 'stuck' (a car behind us while we stand still) or 'cyclist_flirt'
    text: str       # for the display
    animation: str  # for the rear LEDs


class Director:
    def __init__(self, confirm_frames=3, hold_s=8.0, rotate_s=20.0, stuck_s=30.0, flirt=False, pitch_people=None,
                 clock=time.monotonic):
        self.flirt = flirt                    # demo only: cyclists get the 'cyclist_flirt' texts
        self.pitch_people = PITCH_PEOPLE if pitch_people is None else pitch_people  # people get the job pitch
        self.confirm_frames = confirm_frames  # frames a new scene must be seen in a row
        self.hold_s = hold_s                  # a message stays up at least this long
        self.rotate_s = rotate_s              # same scene this long -> next text
        self.stuck_s = stuck_s                # standing still this long with a car behind -> 'stuck' texts
        self.clock = clock
        self.current = None
        self.shown_at = 0.0
        self.candidate, self.count = None, 0
        self.next_index = {scene: 0 for scene in MESSAGES}

    def update(self, scene, stopped_s=None, battery_v=None):
        """Feed one frame's scene. Returns a new Reaction when the display should change, else None.

        stopped_s: seconds the car has been standing still (None = unknown), battery_v: volts (None = unknown).
        """
        if scene == 'car' and stopped_s is not None and stopped_s >= self.stuck_s:
            scene = 'stuck'
        if scene == 'pedestrian' and self.pitch_people:
            scene = 'person_pitch'
        if scene == 'cyclist' and self.flirt:
            scene = 'cyclist_flirt'
        now = self.clock()
        self.count = self.count + 1 if scene == self.candidate else 1
        self.candidate = scene
        if self.current is None:
            return self._show(scene, now, battery_v)
        shown_for = now - self.shown_at
        if scene != self.current.scene:
            # Something appearing behind us is shown at once (after confirmation); leaving needs the hold time.
            if self.count >= self.confirm_frames and (shown_for >= self.hold_s or self.current.scene == 'nobody'):
                return self._show(scene, now, battery_v)
        elif shown_for >= self.rotate_s and len(MESSAGES[scene]) > 1:
            return self._show(scene, now, battery_v)
        return None

    def _show(self, scene, now, battery_v):
        self.current = Reaction(scene, self._next_text(scene, battery_v), ANIMATIONS[scene])
        self.shown_at = now
        return self.current

    def _next_text(self, scene, battery_v):
        texts = MESSAGES[scene]
        for _ in range(len(texts)):
            i = self.next_index[scene]
            self.next_index[scene] = (i + 1) % len(texts)
            if '{battery' in texts[i] and battery_v is None:
                continue
            return texts[i].format(battery=battery_v, **PROFILE, **FLIRT_PROFILE)
        return ''
