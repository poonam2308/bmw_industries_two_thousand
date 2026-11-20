"""Tests for reading bags and replaying them: python3 -m pytest test/"""
import sqlite3

import cv2
import numpy as np
import pytest
from mcap.writer import Writer

from rear_buddy.bag import bag_topics, read_bag
from rear_buddy.bridge import Runner
from rear_buddy.cdr import Codec
from rear_buddy.detector import Detection
from rear_buddy.images import to_bgr
from rear_buddy.replay import CAMERA, BagClock, pick_camera, play
from rear_buddy.rules import Director

codec = Codec()
TYPES = {CAMERA: 'sensor_msgs/msg/CompressedImage', '/wheel_odom': 'car_interfaces/msg/WheelOdom',
         '/batt_status': 'car_interfaces/msg/BattStatus', '/tf': 'tf2_msgs/msg/TFMessage'}


def drive(seconds=4.0):
    """A small recording like the car's: camera 10 Hz, wheels 20 Hz (standing still), battery 1 Hz."""
    ok, jpg = cv2.imencode('.jpg', np.zeros((48, 64, 3), np.uint8))
    image = codec.encode('sensor_msgs/CompressedImage', {'format': 'jpeg', 'data': jpg.tobytes()})
    wheels = codec.encode('car_interfaces/WheelOdom', {})
    battery = codec.encode('car_interfaces/BattStatus', {'voltage_total': 12.2, 'estop_clear': True})
    msgs = []
    for i in range(int(seconds * 20)):
        t = 1_000_000_000_000 + i * 50_000_000  # bags use absolute times
        msgs.append((t, '/wheel_odom', wheels))
        if i % 2 == 0:
            msgs.append((t, CAMERA, image))
        if i % 20 == 0:
            msgs.append((t, '/batt_status', battery))
    msgs.append((msgs[-1][0], '/tf', b'\x00\x01\x00\x00'))  # a topic the replay does not use
    return msgs


def write_mcap(path, msgs):
    with open(path, 'wb') as f:
        w = Writer(f)
        w.start(profile='ros2', library='test')
        channels = {}
        for topic, msg_type in TYPES.items():
            sid = w.register_schema(name=msg_type, encoding='ros2msg', data=b'')
            channels[topic] = w.register_channel(topic=topic, message_encoding='cdr', schema_id=sid)
        for t, topic, data in msgs:
            w.add_message(channel_id=channels[topic], log_time=t, data=data, publish_time=t)
        w.finish()


def write_db3(path, msgs):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, '
               'offered_qos_profiles TEXT)')
    db.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)')
    ids = {topic: i for i, topic in enumerate(TYPES, 1)}
    db.executemany('INSERT INTO topics VALUES (?, ?, ?, "cdr", "")', [(i, t, TYPES[t]) for t, i in ids.items()])
    db.executemany('INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)',
                   [(ids[topic], t, data) for t, topic, data in reversed(msgs)])  # stored out of order on purpose
    db.commit()
    db.close()


@pytest.fixture(params=['mcap', 'db3'])
def bag(tmp_path, request):
    folder = tmp_path / 'drive_1'
    folder.mkdir()
    (folder / 'metadata.yaml').write_text('rosbag2_bagfile_information: {}\n')
    msgs = drive()
    (write_mcap if request.param == 'mcap' else write_db3)(folder / f'drive_1_0.{request.param}', msgs)
    return folder, msgs


def test_topics_and_messages_in_recording_order(bag):
    folder, msgs = bag
    assert bag_topics(folder) == TYPES
    got = list(read_bag(folder, {CAMERA, '/batt_status'}))
    assert [t for t, *_ in got] == sorted(t for t, *_ in got)
    assert {topic for _, topic, _, _ in got} == {CAMERA, '/batt_status'}
    assert len(got) == sum(1 for _, topic, _ in msgs if topic in (CAMERA, '/batt_status'))
    assert got[0][2] in (TYPES[CAMERA], TYPES['/batt_status'])


def test_pick_camera():
    assert pick_camera(TYPES) == CAMERA
    assert pick_camera({'/back_camera/image': 'sensor_msgs/msg/Image', '/x': 'std_msgs/msg/Bool'}) == '/back_camera/image'
    with pytest.raises(SystemExit):
        pick_camera({'/x': 'std_msgs/msg/Bool'})
    with pytest.raises(SystemExit):
        pick_camera(TYPES, wanted='/nope')


class FakeDisplay:
    def __init__(self):
        self.shown = []

    def show(self, reaction):
        self.shown.append(reaction['scene'])


def test_replay_runs_the_car_code_on_the_bag_clock(bag):
    folder, _ = bag
    clock, display, reactions = BagClock(), FakeDisplay(), []
    cyclist = [Detection('bicycle', 0.9, (0, 0, 40, 40), 0.5)]
    runner = Runner(lambda f: cyclist, lambda f, d: (None, None), Director(confirm_frames=1, clock=clock), CAMERA,
                    display=display, log=lambda m: None, clock=clock,
                    on_reaction=lambda r: reactions.append((round(clock.t, 1), r.scene)))
    duration = play(folder, runner, clock, CAMERA, rate=0)
    assert duration == pytest.approx(3.95, abs=0.1)
    assert reactions[0] == (0.1, 'cyclist') and display.shown[0] == 'cyclist'  # first image, looked at on the next tick
    assert runner.images == 40
    assert runner.car.stopped_s() == pytest.approx(4.0, abs=0.15)  # wheels recorded standing still the whole time
    assert runner.car.battery_v() == pytest.approx(12.2)


def test_replay_start_skips_the_beginning(bag):
    folder, _ = bag
    clock = BagClock()
    runner = Runner(lambda f: [], lambda f, d: (None, None), Director(clock=clock), CAMERA, log=lambda m: None,
                    clock=clock, on_reaction=lambda r: None)
    assert play(folder, runner, clock, CAMERA, rate=0, start_s=3.0) == pytest.approx(0.95, abs=0.1)
    assert runner.images == 10


def test_raw_images_from_the_simulation():
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[..., 0] = 255  # pure red in RGB
    payload = codec.encode('sensor_msgs/Image', {'height': 2, 'width': 3, 'encoding': 'rgb8', 'step': 9,
                                                 'data': rgb.tobytes()})
    frame, problem = to_bgr(codec, 'sensor_msgs/msg/Image', payload)
    assert problem is None and frame.shape == (2, 3, 3) and frame[0, 0].tolist() == [0, 0, 255]  # BGR
    mono = codec.encode('sensor_msgs/Image', {'height': 1, 'width': 2, 'encoding': 'mono8', 'step': 4,
                                              'data': bytes([7, 9, 0, 0])})  # rows padded to step
    assert to_bgr(codec, 'sensor_msgs/Image', mono)[0][0, 1].tolist() == [9, 9, 9]
    odd = codec.encode('sensor_msgs/Image', {'height': 1, 'width': 1, 'encoding': '16UC1', 'step': 2,
                                             'data': b'\x00\x00'})
    assert to_bgr(codec, 'sensor_msgs/Image', odd) == (None, "unsupported image encoding '16uc1'")
