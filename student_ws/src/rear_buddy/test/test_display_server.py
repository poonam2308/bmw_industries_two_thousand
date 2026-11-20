"""Tests for the rear display web server: python3 -m pytest test/"""
import json
import urllib.request

import pytest

from rear_buddy.display_server import DisplayServer


@pytest.fixture
def server():
    s = DisplayServer(host='127.0.0.1', port=0)
    yield s
    s.close()


def get(server, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{server.port}{path}', timeout=5) as r:
        return r.status, r.headers['Content-Type'], r.read()


def test_page_is_served(server):
    status, ctype, body = get(server, '/')
    assert status == 200 and ctype.startswith('text/html') and b'EventSource' in body


def test_state_follows_show(server):
    assert json.loads(get(server, '/state')[2])['scene'] == 'nobody'
    server.show({'scene': 'cyclist_flirt', 'text': 'Coffee?', 'animation': 'hearts', 'extra': 'ignored'})
    assert json.loads(get(server, '/state')[2]) == {'scene': 'cyclist_flirt', 'text': 'Coffee?', 'animation': 'hearts', 'qr': False}


def test_events_stream_current_then_new_messages(server):
    with urllib.request.urlopen(f'http://127.0.0.1:{server.port}/events', timeout=5) as stream:
        assert stream.headers['Content-Type'] == 'text/event-stream'

        def next_event():
            line = stream.readline()
            assert stream.readline() == b'\n'
            return json.loads(line.removeprefix(b'data: '))

        assert next_event()['scene'] == 'nobody'
        server.show({'scene': 'car', 'text': 'Hello!', 'animation': 'breathing'})
        assert next_event() == {'scene': 'car', 'text': 'Hello!', 'animation': 'breathing', 'qr': False}


def test_unknown_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        get(server, '/nope')
    assert err.value.code == 404


def test_qr_code_only_on_its_scenes():
    s = DisplayServer(host='127.0.0.1', port=0, qr_url='https://www.linkedin.com/in/someone', qr_scenes={'fancy_car'})
    try:
        status, ctype, body = get(s, '/qr.svg')
        assert status == 200 and ctype == 'image/svg+xml' and b'viewBox' in body
        s.show({'scene': 'fancy_car', 'text': 'Hire me', 'animation': 'gold_shine'})
        assert json.loads(get(s, '/state')[2])['qr'] is True
        s.show({'scene': 'car', 'text': 'Hi', 'animation': 'breathing'})
        assert json.loads(get(s, '/state')[2])['qr'] is False
    finally:
        s.close()


def test_no_qr_code_without_url(server):
    server.show({'scene': 'fancy_car', 'text': 'Hire me', 'animation': 'gold_shine'})
    assert json.loads(get(server, '/state')[2])['qr'] is False
    with pytest.raises(urllib.error.HTTPError):
        get(server, '/qr.svg')
