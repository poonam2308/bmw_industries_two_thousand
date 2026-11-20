from setuptools import find_packages, setup

setup(
    name='rear_buddy',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={'rear_buddy': ['display.html']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/rear_buddy']),
        ('share/rear_buddy', ['package.xml', 'README.md']),
        ('share/rear_buddy/launch', ['launch/rear_buddy.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description="Looks through BITT's rear camera and entertains whoever is behind the car.",
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rear_buddy_node = rear_buddy.node:main',
            'rear_buddy_leds = rear_buddy.led_node:main',
            'rear_buddy_display = rear_buddy.display_node:main',
            'rear_buddy_offline = rear_buddy.offline:main',
            'rear_buddy_bridge = rear_buddy.bridge:main',
            'rear_buddy_replay = rear_buddy.replay:main',
        ],
    },
)
