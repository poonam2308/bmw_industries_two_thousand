"""Tests for the CDR codec: python3 -m pytest test/"""
import struct

import pytest

from rear_buddy.cdr import Codec, parse, schema_sections


@pytest.fixture
def codec():
    return Codec()


def test_lights_cmd_byte_layout(codec):
    payload = codec.encode('car_interfaces/LightsCmd', {
        'header': {'stamp': {'sec': 7, 'nanosec': 9}, 'frame_id': ''},
        'front_rgb': list(range(12)), 'rear_rgb': [200] * 96, 'mode': 3, 'blink_period_ms': 500, 'flags': 1})
    assert payload[:4] == b'\x00\x01\x00\x00'
    body = payload[4:]
    assert struct.unpack_from('<iiI', body, 0) == (7, 9, 1)   # stamp, frame_id length incl. NUL
    assert body[12] == 0                                       # the NUL
    assert list(body[13:25]) == list(range(12))                # front_rgb, fixed array: no length
    assert body[25:121] == bytes([200] * 96)                   # rear_rgb
    assert body[121] == 3                                      # mode
    assert struct.unpack_from('<H', body, 122)[0] == 500       # uint16 aligned to 2
    assert body[124] == 1 and len(body) == 125                 # flags


def test_round_trips(codec):
    image = {'header': {'stamp': {'sec': 1, 'nanosec': 2}, 'frame_id': 'back_camera'}, 'format': 'jpeg',
             'data': b'\xff\xd8 jpeg bytes \xff\xd9'}
    assert codec.decode('sensor_msgs/CompressedImage', codec.encode('sensor_msgs/CompressedImage', image)) == image
    batt = codec.decode('car_interfaces/BattStatus', codec.encode('car_interfaces/BattStatus', {
        'voltage_total': 12.5, 'estop_clear': True}))
    assert batt['voltage_total'] == pytest.approx(12.5) and batt['estop_clear'] is True
    wheels = codec.decode('car_interfaces/msg/WheelOdom', codec.encode('car_interfaces/WheelOdom', {
        'left_count': -5, 'right_count': 2 ** 31 - 1, 'left_speed_mps': 0.25, 'right_speed_mps': -0.5}))
    assert wheels['left_count'] == -5 and wheels['right_speed_mps'] == pytest.approx(-0.5)


def test_alignment_after_odd_string(codec):
    # frame_id 'ab' -> 3 bytes, so the next uint32 length must be padded to a multiple of 4.
    image = {'header': {'frame_id': 'ab'}, 'format': 'png', 'data': b'x'}
    payload = codec.encode('sensor_msgs/CompressedImage', image)
    assert codec.decode('sensor_msgs/CompressedImage', payload)['format'] == 'png'
    assert payload[4 + 15] == 0 and struct.unpack_from('<I', payload, 4 + 16)[0] == 4  # pad, then len('png\0')


def test_wrong_fixed_array_size_is_an_error(codec):
    with pytest.raises(ValueError):
        codec.encode('car_interfaces/LightsCmd', {'front_rgb': [0] * 3})


def test_parse_skips_comments_and_constants():
    fields = parse('uint8 MODE_SOLID=0\n# comment\nuint8[12] front_rgb  # lamps\nfloat64[] xs\nint32[<=4] ys\n'
                   'std_msgs/Header header\nTime t', package='pkg')
    assert [(f.name, f.type, f.array) for f in fields] == [
        ('front_rgb', 'uint8', 12), ('xs', 'float64', 'seq'), ('ys', 'int32', 'seq'),
        ('header', 'std_msgs/Header', None), ('t', 'pkg/Time', None)]


def test_schema_sections_like_the_bridge_sends_them():
    schema = ('std_msgs/Header header\nfloat32 voltage_total\n'
              '================================================================================\n'
              'MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n')
    sections = schema_sections('car_interfaces/msg/BattStatus', schema)
    assert set(sections) == {'car_interfaces/BattStatus', 'std_msgs/Header'}
    assert [f.name for f in parse(sections['std_msgs/Header'])] == ['stamp', 'frame_id']
