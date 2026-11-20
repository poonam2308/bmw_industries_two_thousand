"""Tests for the scene rules and the Director. No camera, no model, no ROS needed:  python3 -m pytest test/"""
from rear_buddy.detector import Detection
from rear_buddy.messages import MESSAGES
from rear_buddy.rules import Director, classify


def det(label, box, tier=None, size=(640, 480)):
    x1, y1, x2, y2 = box
    return Detection(label, 0.9, box, (x2 - x1) * (y2 - y1) / (size[0] * size[1]), tier)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_nothing_behind():
    assert classify([]) == ('nobody', None)


def test_car_tiers():
    assert classify([det('car', (100, 100, 400, 300), tier='premium')])[0] == 'fancy_car'
    assert classify([det('car', (100, 100, 400, 300), tier='budget')])[0] == 'budget_car'
    assert classify([det('car', (100, 100, 400, 300), tier=None)])[0] == 'car'


def test_budget_car_says_old_is_gold():
    r = Director(confirm_frames=1, clock=FakeClock()).update('budget_car')
    assert r.text == 'Hi :) Old is gold.' and r.animation == 'breathing'


def test_standing_still_behind_a_budget_car_stays_old_is_gold():
    assert Director(confirm_frames=1, clock=FakeClock()).update('budget_car', stopped_s=99).scene == 'budget_car'


def test_rider_on_bicycle_is_a_cyclist():
    bike = det('bicycle', (200, 250, 400, 450))
    rider = det('person', (230, 100, 380, 400))
    scene, subject = classify([bike, rider])
    assert scene == 'cyclist' and subject is bike


def test_driver_behind_windscreen_belongs_to_the_car():
    car = det('car', (50, 50, 600, 450))
    driver = det('person', (250, 120, 350, 250))
    assert classify([car, driver])[0] == 'car'


def test_pedestrian_alone():
    assert classify([det('person', (200, 50, 300, 450))])[0] == 'pedestrian'


def test_closest_subject_wins():
    far_car = det('car', (10, 10, 60, 40))
    near_walker = det('person', (200, 50, 400, 470))
    assert classify([far_car, near_walker])[0] == 'pedestrian'


def test_tiny_far_away_boxes_are_ignored():
    assert classify([det('car', (0, 0, 10, 10))])[0] == 'nobody'


def test_new_scene_needs_confirmation_but_not_hold_when_nothing_was_shown():
    clock = FakeClock()
    d = Director(confirm_frames=3, clock=clock)
    assert d.update('nobody').scene == 'nobody'
    assert d.update('car') is None
    assert d.update('car') is None
    assert d.update('car').scene == 'car'


def test_flicker_does_not_change_the_display():
    clock = FakeClock()
    d = Director(confirm_frames=3, clock=clock)
    d.update('nobody')
    for scene in ['car', 'nobody', 'car', 'nobody', 'car']:
        assert d.update(scene) is None


def test_message_is_held_before_leaving():
    clock = FakeClock()
    d = Director(confirm_frames=1, hold_s=8.0, clock=clock)
    d.update('car')
    clock.t = 5.0
    assert d.update('nobody') is None
    clock.t = 9.0
    assert d.update('nobody').scene == 'nobody'


def test_text_rotates_while_the_scene_stays():
    clock = FakeClock()
    d = Director(rotate_s=20.0, clock=clock)
    first = d.update('cyclist', battery_v=12.3)
    clock.t = 10.0
    assert d.update('cyclist', battery_v=12.3) is None
    clock.t = 21.0
    second = d.update('cyclist', battery_v=12.3)
    assert second.scene == 'cyclist' and second.text != first.text


def test_standing_still_behind_a_car_becomes_stuck():
    d = Director(confirm_frames=1, stuck_s=30.0, clock=FakeClock())
    assert d.update('car', stopped_s=40.0).scene == 'stuck'


def test_battery_texts_are_skipped_without_battery_data():
    d = Director(confirm_frames=1, rotate_s=0.0, clock=FakeClock())
    for _ in range(len(MESSAGES['cyclist']) * 2):
        r = d.update('cyclist')
        if r:
            assert 'V' not in r.text.split()  # no "{battery:.1f} V" text was chosen


def test_job_pitch_fills_in_the_profile():
    r = Director(clock=FakeClock()).update('fancy_car')
    assert '{' not in r.text and r.animation == 'gold_shine'


def test_flirt_mode_is_off_by_default():
    assert Director(confirm_frames=1, clock=FakeClock()).update('cyclist').scene == 'cyclist'


def test_flirt_mode_flirts_with_cyclists_only():
    d = Director(confirm_frames=1, hold_s=0, flirt=True, clock=FakeClock())
    r = d.update('cyclist')
    assert r.scene == 'cyclist_flirt' and r.animation == 'hearts' and 'coffee' in r.text.lower()
    assert d.update('car').scene == 'car'


def test_flirt_texts_name_the_flirt_profile():
    from rear_buddy.messages import FLIRT_PROFILE
    d = Director(confirm_frames=1, rotate_s=0.0, flirt=True, clock=FakeClock())
    texts = [d.update('cyclist').text for _ in MESSAGES['cyclist_flirt']]
    assert all('{' not in t for t in texts)
    assert all(FLIRT_PROFILE['flirt_name'] in t for t in texts)


def test_people_far_in_the_background_do_not_count():
    far = det('person', (300, 200, 396, 296))   # 3 % of the image: above MIN_AREA, below MIN_PERSON_AREA
    near = det('person', (200, 50, 400, 470))   # 27 %
    assert classify([far])[0] == 'nobody'
    assert classify([far, near])[1] is near
    assert classify([far], min_person_area=0.001)[0] == 'pedestrian'  # threshold can be lowered


def test_small_cars_and_bikes_still_count():
    assert classify([det('car', (0, 0, 90, 60))])[0] == 'car'          # 1.8 %: cars keep the low minimum
    bike = det('bicycle', (300, 300, 380, 380))                        # 2 %, rider even smaller
    rider = det('person', (310, 250, 370, 330))
    assert classify([bike, rider])[0] == 'cyclist'


def test_pitch_people_gives_people_the_job_pitch_with_qr():
    from rear_buddy.messages import QR_SCENES
    plain = Director(confirm_frames=1, clock=FakeClock())
    assert plain.update('pedestrian').scene == 'pedestrian'  # off by default
    d = Director(confirm_frames=1, hold_s=0, pitch_people=True, clock=FakeClock())
    r = d.update('pedestrian')
    assert r.scene == 'person_pitch' and r.animation == 'gold_shine' and 'person_pitch' in QR_SCENES
    assert '{' not in r.text
    assert d.update('cyclist').scene == 'cyclist' and d.update('budget_car').scene == 'budget_car'
