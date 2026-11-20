"""Rear Buddy on the car: camera reactions, rear LED animations and the rear display page for the phone.

    ros2 launch rear_buddy rear_buddy.launch.py
    ros2 launch rear_buddy rear_buddy.launch.py fancy:=always flirt:=true brightness:=0.3
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        'fancy': ('clip', 'car brand check: clip, always, budget or never'),
        'flirt': ('false', 'demo only: the car flirts with cyclists'),
        'brightness': ('0.5', 'rear LED brightness 0..1'),
        'front': ('white', 'headlights while running: white or off'),
        'port': ('8080', 'port of the rear display page for the phone'),
    }
    cfg = {name: LaunchConfiguration(name) for name in args}
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=d, description=desc) for name, (d, desc) in args.items()],
        Node(package='rear_buddy', executable='rear_buddy_node', output='screen', parameters=[{
            'fancy': cfg['fancy'],
            'flirt': ParameterValue(cfg['flirt'], value_type=bool),
        }]),
        Node(package='rear_buddy', executable='rear_buddy_leds', output='screen', parameters=[{
            'brightness': ParameterValue(cfg['brightness'], value_type=float),
            'front': cfg['front'],
        }]),
        Node(package='rear_buddy', executable='rear_buddy_display', output='screen', parameters=[{
            'port': ParameterValue(cfg['port'], value_type=int),
        }]),
    ])
