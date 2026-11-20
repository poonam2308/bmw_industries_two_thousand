"""Tests for the car brand check, without loading the CLIP model: python3 -m pytest test/"""
import builtins

import numpy as np

from rear_buddy import fancy
from rear_buddy.detector import Detection
from rear_buddy.fancy import FixedTier, decide, labels_from, make_fancy_check
from rear_buddy.pipeline import analyse

BRANDS = {'premium': ['Porsche', 'BMW'], 'budget': ['Toyota'], 'other': ['Volkswagen']}


def test_labels_cover_brands_and_descriptions():
    labels = labels_from(BRANDS, ['a small cheap city car'])
    assert [(name, tier) for name, tier, _ in labels] == [
        ('Porsche', 'premium'), ('BMW', 'premium'), ('Toyota', 'budget'), ('Volkswagen', None),
        ('a small cheap city car', 'budget')]
    assert 'a photo of a Porsche car' in labels[0][2]


def test_decide_picks_the_best_label():
    labels = labels_from(BRANDS, ['a small cheap city car'])
    assert decide([0.9, 0.05, 0.02, 0.02, 0.01], labels) == ('premium', 'Porsche')
    assert decide([0.0, 0.0, 0.1, 0.1, 0.8], labels) == ('budget', 'a small cheap city car')
    assert decide([0.0, 0.0, 0.0, 0.95, 0.05], labels) == (None, 'Volkswagen')  # other brand: normal hello


def test_decide_unsure_is_a_normal_car():
    labels = labels_from(BRANDS, [])
    assert decide([0.45, 0.4, 0.1, 0.05], labels, min_conf=0.5) == (None, None)


def test_fixed_checks():
    assert make_fancy_check('always')(None, None) == ('premium', None)
    assert make_fancy_check('budget')(None, None) == ('budget', None)
    assert make_fancy_check('never')(None, None) == (None, None)


def test_clip_falls_back_to_normal_cars_when_unavailable(monkeypatch):
    real_import = builtins.__import__

    def no_open_clip(name, *args, **kwargs):
        if name == 'open_clip':
            raise ImportError('No module named open_clip')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_open_clip)
    logs = []
    check = make_fancy_check('clip', log=logs.append)
    assert isinstance(check, FixedTier) and check(None, None) == (None, None)
    assert any('unavailable' in m for m in logs)


def test_only_the_closest_car_is_checked_and_sets_the_scene():
    near = Detection('car', 0.9, (0, 0, 300, 200), 0.4)
    far = Detection('car', 0.9, (400, 0, 440, 30), 0.02)
    asked = []

    def check(frame, d):
        asked.append(d)
        return 'budget', 'Toyota'

    scene, subject, _ = analyse(np.zeros((480, 640, 3), np.uint8), lambda f: [far, near], check)
    assert asked == [near] and scene == 'budget_car' and subject.brand == 'Toyota'
    assert far.tier is None


def test_tiny_crops_are_not_classified():
    check = fancy.ClipBrandCheck.__new__(fancy.ClipBrandCheck)  # no model needed for this path
    tiny = Detection('car', 0.9, (0, 0, 10, 10), 0.001)
    assert check(np.zeros((480, 640, 3), np.uint8), tiny) == (None, None)
