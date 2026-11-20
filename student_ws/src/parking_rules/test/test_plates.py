from datetime import datetime

import pytest

from parking_rules.plates import ALL_DAYS, WORKDAYS, Schedule, minutes_inside, parse, violation

MOFR = (0, 1, 2, 3, 4)
FRI = datetime(2026, 11, 20)   # a Friday
SAT = datetime(2026, 11, 21)


def at(day, hh, mm=0):
    return day.replace(hour=hh, minute=mm)


# --- parsing --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('text, windows, max_minutes', [
    ('8-11 h\nmax. 3 Std.', [(ALL_DAYS, 480, 660)], 180),          # the example from the task
    ('8 - 11 Uhr', [(ALL_DAYS, 480, 660)], None),
    ('von 8 bis 11 Uhr', [(ALL_DAYS, 480, 660)], None),
    ('8.30-18 h', [(ALL_DAYS, 510, 1080)], None),
    ('8:30-18:00', [(ALL_DAYS, 510, 1080)], None),
    ('Mo-Fr 8-18 h', [(MOFR, 480, 1080)], None),
    ('Mo-Fr 8-18 h  Sa 8-13 h', [(MOFR, 480, 1080), ((5,), 480, 780)], None),
    ('werktags 9-18 h', [(WORKDAYS, 540, 1080)], None),
    ('Mo, Mi, Fr 7-9 h', [((0, 2, 4), 420, 540)], None),
    ('18-6 h', [(ALL_DAYS, 1080, 360)], None),                       # overnight
    ('0-24 h', [(ALL_DAYS, 0, 1440)], None),
    ('2 Std.', [], 120),
    ('1/2 Std.', [], 30),
    ('1,5 Std.', [], 90),
    ('90 Min.', [], 90),
    ('Mon-Fri 8-11 h, max. 2 h', [(MOFR, 480, 660)], 120),
    ('max 3 hours', [], 180),
])
def test_parse(text, windows, max_minutes):
    s = parse(text)
    assert s.windows == windows
    assert s.max_minutes == max_minutes


@pytest.mark.parametrize('ocr', [
    '8–11 h max. 3 Std.',          # en dash
    '8—11h max.3Std.',             # em dash, no spaces
    '8-1l h max. 3 Std.',          # l instead of 1
    'O8-11 h max. 3 Std.',         # O instead of 0
    '8 -11 Uhr\nmax. 3 Stunden',
])
def test_parse_tolerates_ocr_slips(ocr):
    assert parse(ocr).key() == parse('8-11 h max. 3 Std.').key()


def test_parse_flags():
    s = parse('mit Parkscheibe\n2 Std.')
    assert s.disc and s.max_minutes == 120 and not s.ticket
    assert parse('Mo-Sa 9-19 h mit Parkschein').ticket


@pytest.mark.parametrize('junk', ['', 'Anlieger frei', 'xx', 'Lieferverkehr frei', 'Mofa'])
def test_parse_nothing_understood(junk):
    assert parse(junk) is None


def test_days_do_not_match_inside_words():
    # 'mit', 'frei', 'sonn' must not be read as Mi / Fr / So.
    assert parse('mit Parkscheibe 8-11 h').windows == [(ALL_DAYS, 480, 660)]


def test_schedule_roundtrip_and_describe():
    s = parse('Mo-Fr 8-11 h max. 3 Std.')
    assert Schedule.from_dict(s.to_dict()).key() == s.key()
    assert s.describe() == 'Mo-Fr 8-11 h, max. 3 h'


# --- time arithmetic ------------------------------------------------------------------------------------------------

def test_minutes_inside():
    w = [(ALL_DAYS, 480, 660)]
    assert minutes_inside(w, at(FRI, 9), 60) == 60
    assert minutes_inside(w, at(FRI, 10, 30), 60) == 30
    assert minutes_inside(w, at(FRI, 12), 60) == 0
    assert minutes_inside(w, at(FRI, 7), 24 * 60) == 180       # one full window in a day


def test_minutes_inside_overnight_and_weekdays():
    assert minutes_inside([(ALL_DAYS, 1080, 360)], at(FRI, 23), 120) == 120
    assert minutes_inside([(ALL_DAYS, 1080, 360)], at(FRI, 5), 120) == 60   # window of Thursday night ends at 6
    assert minutes_inside([(MOFR, 480, 1080)], at(SAT, 9), 60) == 0


# --- legality -------------------------------------------------------------------------------------------------------

EXAMPLE = parse('8-11 h\nmax. 3 Std.')   # under a parking sign (Zeichen 314)


@pytest.mark.parametrize('start, minutes, legal', [
    ((9, 0), 120, True),      # 2 h inside the window
    ((8, 0), 180, True),      # exactly 3 h
    ((7, 0), 300, True),      # 7-12: only 3 h fall inside 8-11
    ((12, 0), 600, True),     # after 11: no limit
])
def test_example_parking_sign(start, minutes, legal):
    assert (violation('parking', 'parking area', EXAMPLE, 'read', at(FRI, *start), minutes) is None) == legal


def test_example_limit_can_never_bite_under_stvo_meaning():
    # The window 8-11 is itself 3 h long, so 'max. 3 Std.' inside it is never exceeded (see README).
    assert all(violation('parking', 'p', EXAMPLE, 'read', at(FRI, h), 24 * 60) is None for h in range(24))


def test_max_duration_violation_message():
    s = parse('Mo-Fr 8-18 h max. 2 Std.')
    why = violation('parking', 'parking area', s, 'read', at(FRI, 9), 180)
    assert why == 'max. 2 h parking (Mo-Fr 8-18 h, max. 2 h); this stay would be 3 h'
    assert violation('parking', 'parking area', s, 'read', at(SAT, 9), 600) is None   # Saturday: free


def test_parking_sign_without_plate_is_legal():
    assert violation('parking', 'parking area', None, 'none', at(FRI, 9), 600) is None


def test_parking_only_during_window():
    s = parse('Mo-Fr 8-18 h')
    assert violation('parking', 'p', s, 'read', at(FRI, 9), 60) is None
    assert violation('parking', 'p', s, 'read', at(FRI, 17), 120) == 'parking only Mo-Fr 8-18 h'


def test_ticket_needed():
    s = parse('Mo-Sa 9-19 h mit Parkschein')
    assert 'ticket' in violation('parking', 'p', s, 'read', at(FRI, 10), 30)
    assert violation('parking', 'p', s, 'read', at(FRI, 20), 30) is None


def test_no_parking_sign_with_time_plate():
    s = parse('Mo-Fr 7-9 h')
    assert violation('no_parking', 'no-parking sign', s, 'read', at(FRI, 8), 30) == 'no-parking sign, Mo-Fr 7-9 h'
    assert violation('no_parking', 'no-parking sign', s, 'read', at(FRI, 10), 120) is None
    assert violation('no_parking', 'no-parking sign', s, 'read', at(SAT, 8), 30) is None
    assert violation('no_parking', 'no-parking sign', s, 'read', at(FRI, 6), 90) is not None   # runs into 7-9


def test_no_parking_sign_without_plate_is_always_illegal():
    assert violation('no_parking', 'no-parking sign', None, 'none', at(SAT, 3), 5) == 'no-parking sign'


def test_unread_plate_is_illegal_to_be_safe():
    assert 'not read yet' in violation('parking', 'parking area', None, 'unread', at(FRI, 9), 30)


# --- command-line helpers -------------------------------------------------------------------------------------------

from parking_rules.plates import parse_minutes, parse_when   # noqa: E402


@pytest.mark.parametrize('text, minutes', [('2h', 120), ('1.5h', 90), ('1,5h', 90), ('90m', 90), ('90', 90),
                                           ('1h30', 90), ('45min', 45)])
def test_parse_minutes(text, minutes):
    assert parse_minutes(text) == minutes


def test_parse_minutes_rejects_junk():
    with pytest.raises(ValueError):
        parse_minutes('soon')


def test_parse_when():
    assert parse_when('2026-11-21 09:00') == datetime(2026, 11, 21, 9, 0)
    t = parse_when('10:30')
    assert (t.hour, t.minute) == (10, 30)
    assert abs((parse_when('now') - parse_when('')).total_seconds()) < 5
