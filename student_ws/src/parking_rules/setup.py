from setuptools import find_packages, setup

setup(
    name='parking_rules',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/parking_rules']),
        ('share/parking_rules', ['package.xml', 'README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Finds things that make a parking spot illegal and checks parking goals against them.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'parking_rules_mapper = parking_rules.mapper_node:main',
            'check_spot = parking_rules.check_spot:main',
            'parking_rules_offline = parking_rules.offline:main',
        ],
    },
)
