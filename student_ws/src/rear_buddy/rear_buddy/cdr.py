"""Minimal ROS 2 message (de)serialisation without ROS: CDR, little endian, as DDS and the foxglove_bridge use it.

    codec = Codec()
    msg = codec.decode('sensor_msgs/CompressedImage', payload)       # -> dict, uint8 arrays as bytes
    payload = codec.encode('car_interfaces/LightsCmd', {'mode': 0})   # missing fields are zero / empty

Driven by .msg definitions: the ones Rear Buddy needs are built in (copied from car_interfaces and ROS), more can
be added with add_definition(). Supports primitives, strings, fixed arrays T[N], sequences T[] / T[<=N] and nested
messages. Not supported (not needed here): wstring.
"""
import re
import struct
from dataclasses import dataclass

ENCAPSULATION = b'\x00\x01\x00\x00'  # CDR, little endian

PRIMITIVES = {  # ROS type: (struct format, size in bytes)
    'bool': ('?', 1), 'byte': ('B', 1), 'char': ('B', 1), 'uint8': ('B', 1), 'int8': ('b', 1),
    'uint16': ('H', 2), 'int16': ('h', 2), 'uint32': ('I', 4), 'int32': ('i', 4),
    'uint64': ('Q', 8), 'int64': ('q', 8), 'float32': ('f', 4), 'float64': ('d', 8),
}
BYTE_TYPES = {'uint8', 'byte', 'char'}  # arrays of these are returned as bytes

DEFINITIONS = {
    'builtin_interfaces/Time': 'int32 sec\nuint32 nanosec',
    'std_msgs/Header': 'builtin_interfaces/Time stamp\nstring frame_id',
    'sensor_msgs/CompressedImage': 'std_msgs/Header header\nstring format\nuint8[] data',
    'sensor_msgs/Image': 'std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\n'
                         'uint32 step\nuint8[] data',
    # From ros_ws/src/car_interfaces/msg (constants left out, they are not serialised).
    'car_interfaces/WheelOdom': 'std_msgs/Header header\nint32 left_count\nint32 right_count\n'
                                'float32 left_speed_mps\nfloat32 right_speed_mps',
    'car_interfaces/BattStatus': 'std_msgs/Header header\nfloat32 voltage_1\nfloat32 voltage_2\n'
                                 'float32 voltage_total\nfloat32 current_a\nbool estop_clear',
    'car_interfaces/LightsCmd': 'std_msgs/Header header\nuint8[12] front_rgb\nuint8[96] rear_rgb\nuint8 mode\n'
                                'uint16 blink_period_ms\nuint8 flags',
}

_TYPE = re.compile(r'^(?P<base>[A-Za-z_][\w/]*?)(?:<=\d+)?(?:\[(?P<bound><=)?(?P<size>\d*)\])?$')
_SECTION = re.compile(r'^={10,}\s*$', re.M)
_CONSTANT = re.compile(r'^\w+\s*=')


@dataclass
class Field:
    name: str
    type: str          # primitive, 'string', or 'pkg/Type'
    array: object      # None (single value), int (fixed size) or 'seq' (length-prefixed)


def normalize(name):
    """'pkg/msg/Type' -> 'pkg/Type'."""
    parts = name.split('/')
    return f'{parts[0]}/{parts[-1]}' if len(parts) == 3 else name


def parse(definition, package=''):
    """Fields of one .msg definition. Comments, blank lines and constants are skipped."""
    fields = []
    for raw in definition.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        typ, rest = line.split(None, 1)
        if _CONSTANT.match(rest):  # NAME=value after the type (bounded types like int32[<=4] also contain '=')
            continue
        name = rest.split()[0]
        m = _TYPE.match(typ)
        if not m:
            raise ValueError(f'cannot parse field type {typ!r}')
        base = m['base']
        if base not in PRIMITIVES and base not in ('string', 'wstring'):
            base = normalize(base) if '/' in base else f'{package}/{base}'
        array = None
        if '[' in typ:
            array = int(m['size']) if m['size'] and not m['bound'] else 'seq'
        fields.append(Field(name, base, array))
    return fields


def schema_sections(name, schema):
    """A ros2msg schema as the bridge advertises it (main definition, then '=====' + 'MSG: pkg/Type' sections)
    -> {type name: definition}."""
    parts = _SECTION.split(schema)
    out = {normalize(name): parts[0]}
    for part in parts[1:]:
        lines = part.strip('\n').split('\n', 1)
        if lines and lines[0].startswith('MSG:'):
            out[normalize(lines[0][4:].strip())] = lines[1] if len(lines) > 1 else ''
    return out


class Codec:
    def __init__(self, definitions=None):
        self.types = {}
        for name, text in (definitions or DEFINITIONS).items():
            self.add_definition(name, text)

    def add_definition(self, name, text):
        name = normalize(name)
        self.types[name] = parse(text, name.split('/')[0])

    def fields(self, type_name):
        try:
            return self.types[normalize(type_name)]
        except KeyError:
            raise KeyError(f'no definition for {type_name}') from None

    # ---- decoding -------------------------------------------------------------------------------------------

    def decode(self, type_name, payload):
        buf = memoryview(payload)
        if bytes(buf[:2]) not in (b'\x00\x01', b'\x00\x03'):  # CDR_LE / PL_CDR_LE
            raise ValueError('not little-endian CDR')
        reader = _Reader(buf)
        return reader.message(self, normalize(type_name))

    # ---- encoding -------------------------------------------------------------------------------------------

    def encode(self, type_name, value=None):
        writer = _Writer()
        writer.message(self, normalize(type_name), value or {})
        return ENCAPSULATION + bytes(writer.out)


class _Reader:
    def __init__(self, buf):
        self.buf, self.pos = buf, 4  # alignment counts from after the 4-byte encapsulation header

    def align(self, n):
        self.pos += (-(self.pos - 4)) % n

    def primitive(self, t):
        fmt, size = PRIMITIVES[t]
        self.align(size)
        value = struct.unpack_from('<' + fmt, self.buf, self.pos)[0]
        self.pos += size
        return value

    def string(self):
        n = self.primitive('uint32')  # includes the terminating NUL
        raw = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return raw.rstrip(b'\x00').decode('utf-8', 'replace')

    def value(self, codec, t):
        if t in PRIMITIVES:
            return self.primitive(t)
        if t == 'string':
            return self.string()
        return self.message(codec, t)

    def message(self, codec, t):
        out = {}
        for f in codec.fields(t):
            if f.array is None:
                out[f.name] = self.value(codec, f.type)
                continue
            count = f.array if isinstance(f.array, int) else self.primitive('uint32')
            if f.type in BYTE_TYPES:
                out[f.name] = bytes(self.buf[self.pos:self.pos + count])
                self.pos += count
            elif f.type in PRIMITIVES and count:
                fmt, size = PRIMITIVES[f.type]
                self.align(size)
                out[f.name] = list(struct.unpack_from(f'<{count}{fmt}', self.buf, self.pos))
                self.pos += size * count
            else:
                out[f.name] = [self.value(codec, f.type) for _ in range(count)]
        return out


class _Writer:
    def __init__(self):
        self.out = bytearray()

    def align(self, n):
        self.out += b'\x00' * ((-len(self.out)) % n)

    def primitive(self, t, v):
        fmt, size = PRIMITIVES[t]
        self.align(size)
        self.out += struct.pack('<' + fmt, v)

    def string(self, s):
        raw = s.encode('utf-8') + b'\x00'
        self.primitive('uint32', len(raw))
        self.out += raw

    def value(self, codec, t, v):
        if t in PRIMITIVES:
            self.primitive(t, v if v is not None else 0)
        elif t == 'string':
            self.string(v or '')
        else:
            self.message(codec, t, v or {})

    def message(self, codec, t, value):
        for f in codec.fields(t):
            v = value.get(f.name)
            if f.array is None:
                self.value(codec, f.type, v)
                continue
            items = v if v is not None else ([0] * f.array if isinstance(f.array, int) else [])
            if isinstance(f.array, int) and len(items) != f.array:
                raise ValueError(f'{t}.{f.name} needs exactly {f.array} values, got {len(items)}')
            if f.array == 'seq':
                self.primitive('uint32', len(items))
            if f.type in BYTE_TYPES:
                self.out += bytes(items)
            else:
                for item in items:
                    self.value(codec, f.type, item)
