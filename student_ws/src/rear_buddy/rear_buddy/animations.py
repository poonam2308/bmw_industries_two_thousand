"""Rear LED animations. Pure Python: render(name, t) gives the 32 rear pixels at time t (seconds since start).

The car has 4 rear bars x 8 WS2812 pixels in one chain. Here they are one strip of 32 pixels, index 0 = bar 0
pixel 0 (left) ... index 31 = bar 3 pixel 7 (right), which matches LightsCmd.rear_rgb (bar i pixel p at 8*i+p).
Check the direction on the real car: if a sweep runs the wrong way, flip it with reverse=True in to_rear_rgb().

Add an animation: write a function (i, t) -> (r, g, b) floats 0..1 and put it in ANIMATIONS.
"""
import colorsys
import math

N_PIXELS = 32
CENTRE = (N_PIXELS - 1) / 2


def _triangle(t, period):
    """0 -> 1 -> 0 over one period."""
    x = (t / period) % 1.0
    return 2 * x if x < 0.5 else 2 * (1 - x)


def scanner(i, t):
    """Red dot running left and right with a fading tail."""
    pos = _triangle(t, 2.5) * (N_PIXELS - 1)
    glow = max(0.0, 1 - abs(i - pos) / 3) ** 2
    return (glow, 0.0, 0.0)


def breathing(i, t):
    """Calm cyan, slowly brighter and dimmer: 'I see you, all fine'."""
    level = 0.15 + 0.85 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 4.0))
    return (0.2 * level, 0.6 * level, level)


def sleepy(i, t):
    """Dim blue, very slow, a bit brighter in the middle: we have been standing here forever."""
    level = 0.05 + 0.35 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 6.0))
    middle = 1 - 0.5 * abs(i - CENTRE) / CENTRE
    return (0.0, 0.1 * level * middle, level * middle)


def gold_shine(i, t):
    """Gold with a white glint sweeping across: for the fancy car."""
    base = (1.0 * 0.4, 0.65 * 0.4, 0.0)
    glint_pos = (t / 1.5 % 1.0) * (N_PIXELS + 8) - 4
    glint = max(0.0, 1 - abs(i - glint_pos) / 2.5)
    return tuple(min(1.0, b + glint) for b in base)


def green_wave(i, t):
    """Green waves rolling across: cheering for the cyclist."""
    level = 0.5 + 0.5 * math.sin(2 * math.pi * (i / 8 - t * 1.5))
    return (0.0, level, 0.15 * level)


def rainbow(i, t):
    """Moving rainbow: hello pedestrian."""
    return colorsys.hsv_to_rgb((i / N_PIXELS + t * 0.3) % 1.0, 1.0, 1.0)


def hearts(i, t):
    """Pink heartbeat (ba-dum ... ba-dum), strongest in the middle: flirt mode."""
    # Pulses last a few tenths of a second so that the ~10 Hz updates over Wi-Fi do not skip them.
    x = t % 1.4
    beat = math.exp(-x / 0.2) + (0.8 * math.exp(-(x - 0.4) / 0.2) if x >= 0.4 else 0.0)
    level = 0.25 + 0.75 * min(1.0, beat)
    middle = 1 - 0.6 * abs(i - CENTRE) / CENTRE
    return (level * middle, 0.08 * level * middle, 0.4 * level * middle)


def off(i, t):
    return (0.0, 0.0, 0.0)


ANIMATIONS = {
    'scanner': scanner,
    'breathing': breathing,
    'sleepy': sleepy,
    'gold_shine': gold_shine,
    'green_wave': green_wave,
    'rainbow': rainbow,
    'hearts': hearts,
    'off': off,
}


def render(name, t, brightness=0.5):
    """32 (r, g, b) tuples, ints 0..255. Unknown names render 'off'. brightness 0..1 caps every channel."""
    fn = ANIMATIONS.get(name, off)
    scale = 255 * max(0.0, min(1.0, brightness))
    return [tuple(int(round(max(0.0, min(1.0, c)) * scale)) for c in fn(i, t)) for i in range(N_PIXELS)]


def to_rear_rgb(pixels, reverse=False):
    """32 (r, g, b) tuples -> the flat 96-value list for LightsCmd.rear_rgb."""
    ordered = pixels[::-1] if reverse else pixels
    return [c for px in ordered for c in px]
