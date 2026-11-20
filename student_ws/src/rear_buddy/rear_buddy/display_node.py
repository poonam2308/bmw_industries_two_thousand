#!/usr/bin/env python3
"""Rear Buddy's rear display on the car: serves the display page and pushes every reaction to it.

    ros2 run rear_buddy rear_buddy_display                       # port 8080
    ros2 run rear_buddy rear_buddy_display --ros-args -p port:=8090
    ros2 run rear_buddy rear_buddy_display --ros-args -p qr_url:=''          # no QR code

Subscribes  /rear_buddy/reaction   std_msgs/String, JSON {scene, text, animation} (from rear_buddy_node)

Open the printed address on the phone (or the small screen) in the rear window. It must be on the same network as
the machine running this node, e.g. the car's BITT Wi-Fi.
"""
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .display_server import DisplayServer
from .messages import QR_SCENES, QR_URL


class RearDisplay(Node):
    def __init__(self):
        super().__init__('rear_buddy_display')
        port = self.declare_parameter('port', 8080).value
        qr_url = self.declare_parameter('qr_url', QR_URL).value  # shown as a QR code on the job pitch
        self.server = DisplayServer(port=port, qr_url=qr_url or None, qr_scenes=QR_SCENES)
        if qr_url and not self.server.qr:
            self.get_logger().warning('no QR code on the display: pip install segno')
        self.create_subscription(String, '/rear_buddy/reaction', self.on_reaction, 10)
        for url in self.server.urls():
            self.get_logger().info(f'rear display: open {url} on the phone')

    def on_reaction(self, msg):
        try:
            self.server.show(json.loads(msg.data))
        except (ValueError, AttributeError):
            self.get_logger().warning(f'ignoring bad reaction: {msg.data!r}')


def main():
    rclpy.init()
    node = RearDisplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.server.close()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
