"""Tests for the bridge mode: a fake foxglove_bridge for the client, and the Runner without any network.
python3 -m pytest test/"""
import json
import queue
import struct
import threading
import time

import cv2
import numpy as np
import pytest
from websockets.sync.server import serve

from rear_buddy.bridge import LIGHTS, Runner
from rear_buddy.cdr import Codec
from rear_buddy.detector import Detection
from rear_buddy.foxglove_client import SUBPROTOCOL, FoxgloveClient
from rear_buddy.rules import Director

CAMERA = '/back_camera/image/compressed'
codec = Codec()


def jpeg_payload():
    ok, jpg = cv2.imencode('.jpg', np.zeros((48, 64, 3), np.uint8))
    return codec.encode('sensor_msgs/CompressedImage', {'format': 'jpeg', 'data': jpg.tobytes()})


# ---- the client against a fake bridge ---------------------------------------------------------------------------

class FakeBridge:
    """Speaks the server side of the Foxglove protocol, just enough for the client."""

    def __init__(self):
        self.received = queue.Queue()  # everything the client sends, decoded
        self.server = serve(self.handler, '127.0.0.1', 0, subprotocols=[SUBPROTOCOL])
        self.url = f'ws://127.0.0.1:{self.server.socket.getsockname()[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def handler(self, ws):
        ws.send(json.dumps({'op': 'serverInfo', 'name': 'fake bridge', 'capabilities': ['clientPublish']}))
        ws.send(json.dumps({'op': 'advertise', 'channels': [
            {'id': 10, 'topic': CAMERA, 'encoding': 'cdr', 'schemaName': 'sensor_msgs/msg/CompressedImage',
             'schema': 'std_msgs/Header header\nstring format\nuint8[] data', 'schemaEncoding': 'ros2msg'},
            {'id': 11, 'topic': '/not_wanted', 'encoding': 'cdr', 'schemaName': 'std_msgs/msg/Bool'},
        ]}))
        for raw in ws:
            if isinstance(raw, str):
                msg = json.loads(raw)
                self.received.put(('json', msg))
                if msg['op'] == 'subscribe':
                    for sub in msg['subscriptions']:
                        ws.send(struct.pack('<BIQ', 1, sub['id'], 123) + jpeg_payload())
            else:
                self.received.put(('binary', raw[0], struct.unpack_from('<I', raw, 1)[0], raw[5:]))

    def next(self, kind):
        while True:
            item = self.received.get(timeout=5)
            if item[0] == kind:
                return item

    def close(self):
        self.server.shutdown()


@pytest.fixture
def bridge():
    b = FakeBridge()
    yield b
    b.close()


def test_client_subscribes_receives_advertises_and_publishes(bridge):
    got = queue.Queue()
    client = FoxgloveClient(bridge.url, {CAMERA}, on_message=lambda t, p, c: got.put((t, p, c)), log=lambda m: None)
    client.advertise(LIGHTS, 'car_interfaces/msg/LightsCmd')
    client.start()
    try:
        assert client.connected.wait(5)
        # Only the wanted topic is subscribed, and its message arrives with the channel's schema.
        subs = [m for m in (bridge.next('json')[1], bridge.next('json')[1]) if m['op'] == 'subscribe']
        assert [s['channelId'] for s in subs[0]['subscriptions']] == [10]
        topic, payload, channel = got.get(timeout=5)
        assert topic == CAMERA and channel['schemaName'] == 'sensor_msgs/msg/CompressedImage'
        assert codec.decode('sensor_msgs/CompressedImage', payload)['format'] == 'jpeg'
        # Publishing: the advertisement went out on connect, the message carries our channel id.
        assert client.publish(LIGHTS, codec.encode('car_interfaces/LightsCmd', {'mode': 3}))
        _, opcode, channel_id, body = bridge.next('binary')
        assert opcode == 1 and channel_id == client.client_channels[LIGHTS][0]
        assert codec.decode('car_interfaces/LightsCmd', body)['mode'] == 3
    finally:
        client.close()


def test_client_advertisement_is_sent(bridge):
    client = FoxgloveClient(bridge.url, set(), on_message=lambda *a: None, log=lambda m: None)
    client.advertise(LIGHTS, 'car_interfaces/msg/LightsCmd')
    client.start()
    try:
        msg = bridge.next('json')[1]
        assert msg == {'op': 'advertise', 'channels': [{'id': 1, 'topic': LIGHTS, 'encoding': 'cdr',
                                                         'schemaName': 'car_interfaces/msg/LightsCmd'}]}
    finally:
        client.close()


def test_publish_while_disconnected_is_dropped():
    client = FoxgloveClient('ws://127.0.0.1:9', set(), on_message=lambda *a: None, log=lambda m: None)
    client.advertise(LIGHTS, 'car_interfaces/msg/LightsCmd')
    assert client.publish(LIGHTS, b'x') is False


# ---- the runner, no network --------------------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeDisplay:
    def __init__(self):
        self.shown = []

    def show(self, reaction):
        self.shown.append(reaction)


def make_runner(clock, published, display, flirt=True):
    cyclist = [Detection('bicycle', 0.9, (0, 0, 40, 40), 0.5)]
    return Runner(detector=lambda frame: cyclist, fancy=lambda f, d: (None, None),
                  director=Director(confirm_frames=1, flirt=flirt, clock=clock), camera_topic=CAMERA,
                  publish=lambda topic, payload: published.append((topic, payload)), display=display,
                  log=lambda m: None, clock=clock)


def test_runner_reacts_to_the_camera_and_drives_leds_and_display():
    clock, published, display = FakeClock(), [], FakeDisplay()
    runner = make_runner(clock, published, display)
    runner.tick()  # no image yet: LEDs play the idle scanner
    assert codec.decode('car_interfaces/LightsCmd', published[-1][1])['rear_rgb'] != bytes(96)
    runner.on_message(CAMERA, jpeg_payload(), {})
    clock.t += 1
    runner.tick()
    assert display.shown[-1]['scene'] == 'cyclist_flirt'
    assert runner.animation == 'hearts'
    lights = codec.decode('car_interfaces/LightsCmd', published[-1][1])
    assert published[-1][0] == LIGHTS and lights['front_rgb'] == bytes([255] * 12)
    rear = lights['rear_rgb']
    assert max(rear[0::3]) > max(rear[1::3])  # hearts are pink: more red than green


def test_runner_uses_wheels_and_battery():
    clock, published = FakeClock(), []
    runner = make_runner(clock, published, None)
    runner.on_message('/wheel_odom', codec.encode('car_interfaces/WheelOdom', {}), {})  # standing still
    runner.on_message('/batt_status', codec.encode('car_interfaces/BattStatus', {'voltage_total': 12.4}), {})
    clock.t += 2
    assert runner.car.stopped_s() == pytest.approx(2) and runner.car.battery_v() == pytest.approx(12.4)


def test_runner_lights_off_and_no_leds():
    clock, published = FakeClock(), []
    runner = make_runner(clock, published, None)
    runner.lights_off()
    off = codec.decode('car_interfaces/LightsCmd', published[-1][1])
    assert off['rear_rgb'] == bytes(96) and off['front_rgb'] == bytes(12)
    quiet = Runner(lambda f: [], lambda f, d: (None, None), Director(), CAMERA, publish=None, log=lambda m: None)
    quiet.tick()
    quiet.lights_off()  # nothing to publish to, must not fail


def test_runner_warns_about_a_different_message_definition():
    logs = []
    runner = Runner(lambda f: [], lambda f, d: (None, None), Director(), CAMERA, log=logs.append)
    runner.on_message('/batt_status', codec.encode('car_interfaces/BattStatus', {}), {
        'schemaName': 'car_interfaces/msg/BattStatus', 'schemaEncoding': 'ros2msg',
        'schema': 'std_msgs/Header header\nfloat32 volts'})
    assert any('different fields' in m for m in logs)
