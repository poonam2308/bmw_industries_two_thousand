"""Is it legal to park with the car at (x, y, yaw)? Reads the zones published by parking_rules_mapper.

    ros2 run parking_rules check_spot 3.0 1.5 0.0          # x y [m] yaw [rad] of base_link in odom
    ros2 run parking_rules check_spot --here               # the car's current pose

Exit code 0 = legal, 1 = illegal, 2 = no zones received. In your own parking planner do not shell out to this:
subscribe to /parking_rules/zones and call parking_rules.zones.check_pose() directly, as below.
"""
import argparse
import math
import sys

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .zones import check_pose, zones_from_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pose', nargs='*', type=float, help='x y yaw of base_link in odom')
    p.add_argument('--here', action='store_true', help='use the current pose of the car')
    p.add_argument('--timeout', type=float, default=3.0)
    args = p.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    if not args.here and len(args.pose) != 3:
        p.error('give x y yaw, or --here')

    rclpy.init()
    node = Node('check_spot')
    got = {}
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, '/parking_rules/zones', lambda m: got.setdefault('zones', m.data), latched)
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    end = node.get_clock().now() + Duration(seconds=args.timeout)
    while rclpy.ok() and 'zones' not in got and node.get_clock().now() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if 'zones' not in got:
        print('no zones received: is parking_rules_mapper running?')
        sys.exit(2)
    zones = zones_from_json(got['zones'])

    if args.here:
        try:
            t = tf_buffer.lookup_transform('odom', 'base_link', Time(), Duration(seconds=args.timeout)).transform
        except TransformException as e:
            print(f'no pose: {e}')
            sys.exit(2)
        q = t.rotation
        x, y = t.translation.x, t.translation.y
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    else:
        x, y, yaw = args.pose

    bad = check_pose(x, y, yaw, zones)
    print(f'pose ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f}°), {len(zones)} zone(s) known')
    if bad:
        for z in bad:
            print(f'  ILLEGAL: {z.reason} at ({z.x:.2f}, {z.y:.2f})')
    else:
        print('  legal')
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
