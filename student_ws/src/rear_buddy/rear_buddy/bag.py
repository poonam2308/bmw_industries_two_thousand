"""Read ROS 2 bags (rosbag2) without ROS: MCAP (.mcap) and SQLite (.db3) storage.

    topics = bag_topics('bags/drive_1')                    # {'/back_camera/image/compressed': 'sensor_msgs/msg/...'}
    for t_ns, topic, msg_type, payload in read_bag('bags/drive_1', topics={'/wheel_odom'}):
        ...                                                 # payload: CDR bytes, decode with cdr.Codec

A bag is a folder (metadata.yaml + one or more .mcap / .db3 files, read in name order) or a single file.
Messages come in recording order.
"""
import pathlib
import sqlite3


def bag_files(path):
    path = pathlib.Path(path).expanduser()
    if path.is_file():
        return [path]
    files = sorted(path.glob('*.mcap')) or sorted(path.glob('*.db3'))
    if not files:
        raise FileNotFoundError(f'no .mcap or .db3 files in {path}')
    return files


def bag_topics(path):
    """{topic: message type} of everything recorded."""
    topics = {}
    for f in bag_files(path):
        if f.suffix == '.mcap':
            from mcap.reader import make_reader
            with open(f, 'rb') as fh:
                summary = make_reader(fh).get_summary()
                if summary is None:  # no summary section: scan the whole file
                    fh.seek(0)
                    for schema, channel, _ in make_reader(fh).iter_messages():
                        topics[channel.topic] = schema.name if schema else ''
                    continue
                for ch in summary.channels.values():
                    schema = summary.schemas.get(ch.schema_id)
                    topics[ch.topic] = schema.name if schema else ''
        else:
            with sqlite3.connect(f'file:{f}?mode=ro', uri=True) as db:
                topics.update(dict(db.execute('SELECT name, type FROM topics')))
    return topics


def read_bag(path, topics=None):
    """Yields (time in ns, topic, message type, CDR payload) in recording order."""
    for f in bag_files(path):
        if f.suffix == '.mcap':
            from mcap.reader import make_reader
            with open(f, 'rb') as fh:
                for schema, channel, message in make_reader(fh).iter_messages(
                        topics=list(topics) if topics else None, log_time_order=True):
                    yield message.log_time, channel.topic, schema.name if schema else '', message.data
        else:
            with sqlite3.connect(f'file:{f}?mode=ro', uri=True) as db:
                query = ('SELECT m.timestamp, t.name, t.type, m.data FROM messages m JOIN topics t ON m.topic_id = t.id')
                args = []
                if topics:
                    query += f' WHERE t.name IN ({",".join("?" * len(topics))})'
                    args = list(topics)
                yield from db.execute(query + ' ORDER BY m.timestamp', args)
