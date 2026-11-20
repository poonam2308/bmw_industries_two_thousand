"""Tests for the LED animations: python3 -m pytest test/"""
import pytest

from rear_buddy.animations import ANIMATIONS, N_PIXELS, render, to_rear_rgb
from rear_buddy.messages import ANIMATIONS as SCENE_ANIMATIONS


def test_every_scene_has_an_animation():
    assert set(SCENE_ANIMATIONS.values()) <= set(ANIMATIONS)


@pytest.mark.parametrize('name', sorted(ANIMATIONS))
def test_frames_are_32_valid_pixels_and_respect_brightness(name):
    for k in range(50):
        pixels = render(name, k * 0.137, brightness=0.5)
        assert len(pixels) == N_PIXELS
        assert all(len(px) == 3 and all(0 <= c <= 128 for c in px) for px in pixels)


@pytest.mark.parametrize('name', sorted(set(ANIMATIONS) - {'off'}))
def test_animations_light_up_and_move(name):
    frames = [render(name, k * 0.1) for k in range(40)]
    assert any(any(c > 0 for px in f for c in px) for f in frames)
    assert len({tuple(f) for f in frames}) > 1


def test_unknown_animation_is_off():
    assert render('nope', 1.0) == [(0, 0, 0)] * N_PIXELS


def test_rear_rgb_layout():
    pixels = [(i, 0, 0) for i in range(N_PIXELS)]
    flat = to_rear_rgb(pixels)
    assert len(flat) == 96
    assert flat[24 * 1 + 3 * 2] == 8 + 2  # bar 1, pixel 2 is strip index 10
    assert to_rear_rgb(pixels, reverse=True)[0] == 31
