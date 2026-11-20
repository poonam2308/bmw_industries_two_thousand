#!/usr/bin/env python3
"""Rear Buddy with the real car, from your laptop, without ROS or Docker: through the car's Foxglove bridge.

    python3 -m rear_buddy.bridge                          # rear camera in, rear LEDs out (ws://10.42.0.1:8765)
    python3 -m rear_buddy.bridge --display --flirt        # + the phone display (open the printed address)
    python3 -m rear_buddy.bridge --show                   # + a window: rear camera, boxes, message, LED bar
    python3 -m rear_buddy.bridge --no-leds                # leave the car's lights alone
    python3 -m rear_buddy.bridge --topic /preview/left/compressed --show    # front camera, if there is no rear one

Connect the laptop to the BITT Wi-Fi first. Subscribes to the camera, /wheel_odom (standing still -> traffic-jam
jokes) and /batt_status (battery jokes); publishes /lights_cmd 10 times per second. While it runs, the light buttons in
Foxglove are overridden ("last message wins"); Ctrl-C switches the lights off and quits.
"""
import argparse
import time

import cv2
import numpy as np

from .animations import render, to_rear_rgb
from .car_state import CarState
from .cdr import Codec, parse, schema_sections
from .detector import Detector
from .display_server import DisplayServer
from .fancy import make_fancy_check
from .foxglove_client import FoxgloveClient
from .messages import QR_SCENES, QR_URL
from .offline import print_reaction, with_leds
from .pipeline import analyse, draw
from .rules import Director

WHEELS, BATTERY, LIGHTS = '/wheel_odom', '/batt_status', '/lights_cmd'
TYPES = {WHEELS: 'car_interfaces/WheelOdom', BATTERY: 'car_interfaces/BattStatus'}
FRONT = {'white': (255, 255, 255), 'off': (0, 0, 0)}
NO_IMAGE_WARN_S = 10.0


class Runner:
    """The brain, independent of the connection: feed it bridge messages, call tick() regularly."""

    def __init__(self, detector, fancy, director, camera_topic, publish=None, display=None, show=False,
                 brightness=0.5, front='white', reverse=False, detect_hz=2.0, log=print, clock=time.monotonic):
        self.detector, self.fancy, self.director = detector, fancy, director
        self.camera_topic, self.publish, self.display, self.show = camera_topic, publish, display, show
        self.brightness, self.front, self.reverse = brightness, FRONT[front], reverse
        self.detect_every = 1.0 / detect_hz
        self.log, self.clock = log, clock
        self.codec = Codec()
        self.car = CarState(clock)
        self.latest_image = None
        self.images = 0
        self.checked = set()
        self.started = clock()
        self.last_detect = -1e9
        self.animation, self.anim_since = 'scanner', self.started
        self.text, self.view = '', None

    def on_message(self, topic, payload, channel):
        """Runs on the client's thread: keep it cheap. Images are only stored, decoding happens in tick()."""
        if topic not in self.checked:
            self.checked.add(topic)
            self._check_schema(topic, channel)
        if topic == self.camera_topic:
            self.latest_image = payload
            self.images += 1
            if self.images == 1:
                self.log(f'first image from {topic}')
        elif topic == WHEELS:
            w = self.codec.decode(TYPES[WHEELS], payload)
            self.car.on_wheels(w['left_speed_mps'], w['right_speed_mps'])
        elif topic == BATTERY:
            self.car.on_battery(self.codec.decode(TYPES[BATTERY], payload)['voltage_total'])

    def _check_schema(self, topic, channel):
        """Warn if the car's message definition differs from ours (then decoding would be wrong)."""
        ours = TYPES.get(topic, 'sensor_msgs/CompressedImage' if topic == self.camera_topic else None)
        if not ours or channel.get('schemaEncoding') != 'ros2msg' or not channel.get('schema'):
            return
        theirs = schema_sections(channel.get('schemaName', ours), channel['schema']).get(ours.replace('/msg/', '/'))
        if theirs is None:
            theirs = schema_sections(channel.get('schemaName', ours), channel['schema'])[
                next(iter(schema_sections(channel.get('schemaName', ours), channel['schema'])))]
        if [f.name for f in parse(theirs)] != [f.name for f in self.codec.fields(ours)]:
            self.log(f'warning: {topic} on the car has different fields than expected; values may be wrong')

    def tick(self):
        now = self.clock()
        if now - self.last_detect >= self.detect_every and self.latest_image is not None:
            self.last_detect = now
            self._detect(self.latest_image)
        if self.publish:
            pixels = render(self.animation, now - self.anim_since, self.brightness)
            self.publish(LIGHTS, self.codec.encode('car_interfaces/LightsCmd', {
                'front_rgb': list(self.front) * 4, 'rear_rgb': to_rear_rgb(pixels, self.reverse),
                'mode': 0, 'blink_period_ms': 500, 'flags': 0}))
        if self.images == 0 and now - self.started > NO_IMAGE_WARN_S and not getattr(self, '_warned', False):
            self._warned = True
            self.log(f'no images on {self.camera_topic} after {NO_IMAGE_WARN_S:.0f} s: is the rear camera connected? '
                     'Try --topic /preview/left/compressed to test with the front camera.')

    def _detect(self, payload):
        image = self.codec.decode('sensor_msgs/CompressedImage', payload)
        frame = cv2.imdecode(np.frombuffer(image['data'], np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.log(f'could not decode an image ({image["format"]})')
            return
        scene, subject, dets = analyse(frame, self.detector, self.fancy)
        reaction = self.director.update(scene, self.car.stopped_s(), self.car.battery_v())
        if reaction:
            print_reaction(reaction)
            self.text = reaction.text
            if self.display:
                self.display.show(reaction.__dict__)
            if reaction.animation != self.animation:
                self.animation, self.anim_since = reaction.animation, self.clock()
        if self.show:
            self.view = draw(frame, dets, subject, self.text)

    def lights_off(self):
        if self.publish:
            self.publish(LIGHTS, self.codec.encode('car_interfaces/LightsCmd', {'blink_period_ms': 500}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', default='ws://10.42.0.1:8765', help="the car's Foxglove bridge")
    p.add_argument('--topic', default='/back_camera/image/compressed', help='camera topic (CompressedImage)')
    p.add_argument('--fancy', default='never', choices=['never', 'always'], help='fancy-car check to use')
    p.add_argument('--flirt', action='store_true', help='demo only: cyclists get the flirt texts')
    p.add_argument('--display', action='store_true', help='serve the rear display page for a phone')
    p.add_argument('--port', type=int, default=8080, help='port of the rear display page')
    p.add_argument('--show', action='store_true', help='window with the camera, boxes, message and LED bar')
    p.add_argument('--no-leds', action='store_true', help='do not touch the car lights')
    p.add_argument('--brightness', type=float, default=0.5, help='rear LED brightness 0..1')
    p.add_argument('--front', default='white', choices=sorted(FRONT), help='headlights while running')
    p.add_argument('--reverse', action='store_true', help='flip the LED direction if sweeps run backwards')
    p.add_argument('--led-hz', type=float, default=10.0, help='LED updates per second')
    p.add_argument('--detect-hz', type=float, default=2.0, help='detections per second')
    p.add_argument('--model', default='yolo11n.pt', help='YOLO weights')
    args = p.parse_args()

    display = None
    if args.display:
        display = DisplayServer(port=args.port, qr_url=QR_URL, qr_scenes=QR_SCENES)
        print('rear display: open ' + ' or '.join(display.urls()) + ' on the phone (same Wi-Fi as this laptop)')

    client = FoxgloveClient(args.url, {args.topic, WHEELS, BATTERY}, on_message=lambda *a: runner.on_message(*a))
    runner = Runner(Detector(args.model), make_fancy_check(args.fancy), Director(flirt=args.flirt), args.topic,
                    publish=None if args.no_leds else client.publish, display=display, show=args.show,
                    brightness=args.brightness, front=args.front, reverse=args.reverse, detect_hz=args.detect_hz)
    if not args.no_leds:
        client.advertise(LIGHTS, 'car_interfaces/msg/LightsCmd')
    print(f'connecting to {args.url} (laptop on the BITT Wi-Fi?) ... Ctrl-C to stop')
    client.start()

    period = 1.0 / args.led_hz
    try:
        while True:
            start = time.monotonic()
            runner.tick()
            if args.show and runner.view is not None:
                cv2.imshow('rear buddy (car)', with_leds(runner.view, runner.animation,
                                                         time.monotonic() - runner.anim_since))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(max(0.0, period - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):  # make sure "off" arrives
            runner.lights_off()
            time.sleep(0.05)
        client.close()
        if display:
            display.close()
        cv2.destroyAllWindows()
        print('lights off, disconnected')


if __name__ == '__main__':
    main()
