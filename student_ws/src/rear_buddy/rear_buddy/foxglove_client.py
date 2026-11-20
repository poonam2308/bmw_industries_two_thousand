"""A small client for the car's foxglove_bridge (Foxglove WebSocket protocol v1): subscribe and publish ROS topics
over Wi-Fi from a laptop without ROS.

    client = FoxgloveClient('ws://10.42.0.1:8765', topics={'/wheel_odom'}, on_message=handle).start()
    client.advertise('/lights_cmd', 'car_interfaces/msg/LightsCmd')
    client.publish('/lights_cmd', cdr_payload)

on_message(topic, payload, channel) runs on the client's thread with the raw CDR payload (decode it with cdr.Codec);
channel is the bridge's advertisement (schemaName, schema, ...). The client reconnects by itself after Wi-Fi drops
and then subscribes and advertises again.
Protocol: https://github.com/foxglove/ws-protocol/blob/main/docs/spec.md
"""
import json
import struct
import threading

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

SUBPROTOCOL = 'foxglove.websocket.v1'
OP_MESSAGE_DATA = 0x01  # server -> client: subscription id, receive time, payload; client -> server: channel id, payload


class FoxgloveClient:
    def __init__(self, url, topics, on_message, log=print, retry_s=2.0):
        self.url, self.topics, self.on_message, self.log = url, set(topics), on_message, log
        self.retry_s = retry_s
        self.capabilities = set()
        self.server_channels = {}   # channel id -> advertisement, everything the bridge offers
        self.subscriptions = {}     # our subscription id -> topic
        self.client_channels = {}   # topic -> (our channel id, schema name)
        self.connected = threading.Event()  # set once the bridge sent serverInfo
        self._ws = None
        self._lock = threading.Lock()       # serialises sends from the main thread and the receive thread
        self._stop = threading.Event()
        self._next_sub = 1
        self._thread = threading.Thread(target=self._run, name='foxglove-client', daemon=True)

    def start(self):
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        ws = self._ws
        if ws is not None:
            ws.close()
        self._thread.join(timeout=3)

    def advertised_topics(self):
        return sorted({c['topic'] for c in self.server_channels.values()})

    # ---- publishing -----------------------------------------------------------------------------------------

    def advertise(self, topic, schema_name):
        with self._lock:
            if topic not in self.client_channels:
                self.client_channels[topic] = (len(self.client_channels) + 1, schema_name)
        if self.connected.is_set():
            self._advertise([topic])

    def publish(self, topic, payload):
        """Send one message. False when not connected (the message is dropped, the next one may get through)."""
        channel_id = self.client_channels[topic][0]
        return self._send(struct.pack('<BI', OP_MESSAGE_DATA, channel_id) + payload)

    def _advertise(self, topics):
        channels = [{'id': self.client_channels[t][0], 'topic': t, 'encoding': 'cdr',
                     'schemaName': self.client_channels[t][1]} for t in topics]
        self._send(json.dumps({'op': 'advertise', 'channels': channels}))

    def _send(self, data):
        with self._lock:
            ws = self._ws
            if ws is None or not self.connected.is_set():
                return False
            try:
                ws.send(data)
                return True
            except (ConnectionClosed, OSError):
                return False

    # ---- receiving ------------------------------------------------------------------------------------------

    def _run(self):
        last_error = None
        while not self._stop.is_set():
            try:
                with connect(self.url, subprotocols=[SUBPROTOCOL], max_size=None, open_timeout=5) as ws:
                    with self._lock:
                        self._ws = ws
                        self.server_channels.clear()
                        self.subscriptions.clear()
                    last_error = None
                    for raw in ws:
                        if isinstance(raw, str):
                            self._on_text(json.loads(raw))
                        else:
                            self._on_binary(raw)
                    raise ConnectionError('connection closed')
            except Exception as e:  # noqa: BLE001 - any connection problem: report once and keep retrying
                error = str(e) or type(e).__name__
                if not self._stop.is_set() and error != last_error:
                    self.log(f'bridge {self.url}: {error}; retrying every {self.retry_s:.0f} s')
                last_error = error
            finally:
                self.connected.clear()
                with self._lock:
                    self._ws = None
            self._stop.wait(self.retry_s)

    def _on_text(self, msg):
        op = msg.get('op')
        if op == 'serverInfo':
            self.capabilities = set(msg.get('capabilities', []))
            self.connected.set()
            self.log(f'connected to {msg.get("name", "bridge")} at {self.url}')
            if self.client_channels:
                if 'clientPublish' not in self.capabilities:
                    self.log('warning: this bridge does not accept messages from clients (no clientPublish)')
                self._advertise(list(self.client_channels))
        elif op == 'advertise':
            new = []
            subscribed = set(self.subscriptions.values())
            for ch in msg.get('channels', []):
                self.server_channels[ch['id']] = ch
                if ch['topic'] in self.topics and ch['topic'] not in subscribed:
                    new.append({'id': self._next_sub, 'channelId': ch['id']})
                    self.subscriptions[self._next_sub] = ch['topic']
                    subscribed.add(ch['topic'])
                    self._next_sub += 1
            if new:
                self._send(json.dumps({'op': 'subscribe', 'subscriptions': new}))
        elif op == 'unadvertise':
            gone = {self.server_channels.pop(cid, {}).get('topic') for cid in msg.get('channelIds', [])}
            for sid, topic in list(self.subscriptions.items()):
                if topic in gone:
                    del self.subscriptions[sid]
        elif op == 'status' and msg.get('level', 0) >= 1:
            self.log(f'bridge says: {msg.get("message")}')

    def _on_binary(self, data):
        if len(data) < 13 or data[0] != OP_MESSAGE_DATA:
            return
        sub_id = struct.unpack_from('<I', data, 1)[0]
        topic = self.subscriptions.get(sub_id)
        if topic is None:
            return
        channel = next((c for c in self.server_channels.values() if c['topic'] == topic), {})
        try:
            self.on_message(topic, data[13:], channel)
        except Exception as e:  # noqa: BLE001 - a bad message must not kill the connection
            self.log(f'error handling {topic}: {e}')
