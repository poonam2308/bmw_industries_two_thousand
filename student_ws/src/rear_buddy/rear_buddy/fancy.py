"""Is this car fancy? A pluggable check: any callable (frame, detection) -> bool works.

For now only fixed answers exist, which is enough to develop and demo the rest of the pipeline. Better checks to
plug in later: a car make/model classifier (e.g. trained on Stanford Cars), or a vision LLM asked
"is this a luxury or sports car?" on the cropped box. They are slow, so the pipeline only asks about the closest
vehicle.
"""


class FixedFancy:
    """Always gives the same answer: 'always' for a demo with a sports-car photo, 'never' to test normal cars."""

    def __init__(self, answer: bool):
        self.answer = answer

    def __call__(self, frame, detection):
        return self.answer


def make_fancy_check(name):
    checks = {'never': FixedFancy(False), 'always': FixedFancy(True)}
    if name not in checks:
        raise ValueError(f'unknown fancy check {name!r}, choose from {sorted(checks)}')
    return checks[name]


def crop(frame, detection):
    """The detection's box cut out of the frame, for checks that look at the car itself."""
    x1, y1, x2, y2 = (int(v) for v in detection.box)
    return frame[max(y1, 0):y2, max(x1, 0):x2]
