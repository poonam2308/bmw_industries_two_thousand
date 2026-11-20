#!/usr/bin/env python3
"""Rear Buddy's rear LEDs: plays the animation of the latest reaction on the car's rear bars.

    ros2 run rear_buddy rear_buddy_leds                                       # follows /rear_buddy/reaction
    ros2 run rear_buddy rear_buddy_leds --ros-args -p animation:=rainbow      # one animation, to test the lights
    ros2 run rear_buddy rear_buddy_leds --ros-args -p brightness:=0.3 -p rate_hz:=10.0 -p front:=off

Subscribes  /rear_buddy/reaction   std_msgs/String, JSON with an "animation" field (from rear_buddy_node)
Publishes   /lights_cmd            car_interfaces/LightsCmd, rate_hz times per second

/lights_cmd is "last message wins": while this node runs it overrides the light buttons in Foxglove / controls.py
(indicators, hazard). The firmware still adds its own brake light. On Ctrl-C all lights are switched off.
"""
import json
import time

import rclpy
from car_interfaces.msg import LightsCmd
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .animations import ANIMATIONS, render, to_rear_rgb

FRONT = {'white': (255, 255, 255), 'off': (0, 0, 0)}


class RearLeds(Node):
    def __init__(self):
        super().__init__('rear_buddy_leds')
        self.animation = self.declare_parameter('animation', 'scanner').value  # until the first reaction
        self.brightness = self.declare_parameter('brightness', 0.5).value
        rate_hz = self.declare_parameter('rate_hz', 10.0).value
        front = self.declare_parameter('front', 'white').value                 # headlights: white or off
        self.reverse = self.declare_parameter('reverse', False).value          # flip if sweeps run backwards
        if front not in FRONT:
            raise ValueError(f'front must be one of {sorted(FRONT)}')
        self.front = FRONT[front]
        self.started = time.monotonic()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)  # as in the ESP contract
        self.pub = self.create_publisher(LightsCmd, '/lights_cmd', qos)
        self.create_subscription(String, '/rear_buddy/reaction', self.on_reaction, 10)
        self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(f'playing {self.animation!r}, following /rear_buddy/reaction')

    def on_reaction(self, msg):
        try:
            name = json.loads(msg.data)['animation']
        except (ValueError, KeyError, TypeError):
            self.get_logger().warning(f'ignoring reaction without animation: {msg.data!r}')
            return
        if name not in ANIMATIONS:
            self.get_logger().warning(f'unknown animation {name!r}, playing off')
        if name != self.animation:
            self.get_logger().info(f'animation -> {name}')
            self.animation, self.started = name, time.monotonic()

    def lights(self, rear_rgb, front):
        msg = LightsCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.front_rgb = [c for _ in range(4) for c in front]
        msg.rear_rgb = rear_rgb
        msg.mode = LightsCmd.MODE_SOLID
        msg.blink_period_ms = 500
        return msg

    def tick(self):
        pixels = render(self.animation, time.monotonic() - self.started, self.brightness)
        self.pub.publish(self.lights(to_rear_rgb(pixels, self.reverse), self.front))

    def switch_off(self):
        self.pub.publish(self.lights([0] * 96, (0, 0, 0)))


def main():
    rclpy.init()
    node = RearLeds()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.switch_off()
    rclpy.spin_once(node, timeout_sec=0.3)  # let the last message leave
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
