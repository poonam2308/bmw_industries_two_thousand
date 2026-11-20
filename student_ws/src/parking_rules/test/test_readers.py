import shutil
import sys
import types

import numpy as np
import pytest

from parking_rules.plates import parse
from parking_rules.readers import ClaudeReader, FixedReader, make_reader
from parking_rules.signs import render_plate

IMG = np.full((40, 80, 3), 255, np.uint8)


def test_fixed_reader_and_none():
    assert make_reader('none') is None
    assert make_reader('fixed', fixed_text='8-11 h')(IMG, IMG) == '8-11 h'
    assert FixedReader('')(IMG, IMG) is None


def test_unknown_reader():
    with pytest.raises(ValueError):
        make_reader('magic')


# --- Claude reader against a fake `anthropic` module (no network, no cost) --------------------------------------

class _Err(Exception):
    def __init__(self, message='x', status_code=500):
        super().__init__(message)
        self.message, self.status_code = message, status_code


def _fake_anthropic(behaviour):
    m = types.ModuleType('anthropic')
    for name in ('AuthenticationError', 'PermissionDeniedError', 'RateLimitError', 'APIStatusError',
                 'APIConnectionError'):
        setattr(m, name, type(name, (_Err,), {}))
    calls = []

    class Messages:
        def create(self, **kw):
            calls.append(kw)
            return behaviour(m, kw)

    class Client:
        def __init__(self):
            self.beta = types.SimpleNamespace(messages=Messages())

    m.Anthropic = Client
    return m, calls


def _response(text, stop='end_turn'):
    return types.SimpleNamespace(stop_reason=stop, content=[types.SimpleNamespace(type='text', text=text)])


@pytest.fixture
def fake(monkeypatch):
    def install(behaviour):
        m, calls = _fake_anthropic(behaviour)
        monkeypatch.setitem(sys.modules, 'anthropic', m)
        return calls
    return install


def test_claude_reader_request_and_text(fake):
    calls = fake(lambda m, kw: _response('8-11 h\nmax. 3 Std.'))
    text = ClaudeReader(log=lambda *_: None)(IMG, IMG)
    assert parse(text).describe() == '8-11 h, max. 3 h'
    [kw] = calls
    assert kw['model'] == 'claude-opus-5-5'
    assert kw['fallbacks'] == 'default' and 'server-side-fallback-2026-07-01' in kw['betas']
    content = kw['messages'][0]['content']
    assert content[0]['type'] == 'image' and content[0]['source']['media_type'] == 'image/jpeg'


def test_claude_reader_none_and_refusal(fake):
    fake(lambda m, kw: _response('NONE'))
    assert ClaudeReader(log=lambda *_: None)(IMG, IMG) is None
    fake(lambda m, kw: _response('', stop='refusal'))
    assert ClaudeReader(log=lambda *_: None)(IMG, IMG) is None


def test_claude_reader_offline_returns_none_and_keeps_trying(fake):
    def offline(m, kw):
        raise m.APIConnectionError('no route')
    calls = fake(offline)
    logs = []
    r = ClaudeReader(log=logs.append)
    assert r(IMG, IMG) is None and r(IMG, IMG) is None
    assert len(calls) == 2 and 'no internet' in logs[0]


def test_claude_reader_stops_after_auth_error(fake):
    def denied(m, kw):
        raise m.AuthenticationError('invalid x-api-key', 401)
    calls = fake(denied)
    r = ClaudeReader(log=lambda *_: None)
    assert r(IMG, IMG) is None and r(IMG, IMG) is None
    assert len(calls) == 1   # does not hammer the API with a bad key


# --- tesseract: only if installed ------------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which('tesseract') is None, reason='tesseract program not installed')
@pytest.mark.parametrize('lines', [['8-11 h', 'max. 3 Std.'], ['Mo-Fr', '7-9 h'], ['2 Std.']])
def test_tesseract_reads_rendered_plates(lines):
    pytest.importorskip('pytesseract')
    plate = render_plate(lines, 300)
    text = make_reader('tesseract')(plate, plate)
    assert parse(text).key() == parse('\n'.join(lines)).key()
