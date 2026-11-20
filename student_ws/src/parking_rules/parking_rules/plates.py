"""Extra plates (Zusatzzeichen) under parking signs: text -> Schedule -> legal at this time for this long?

Pure Python (no OpenCV, no ROS). The text comes from OCR or a vision model (readers.py), so the parser is forgiving:
it fixes common OCR slips (O/0, l/1, dashes) and ignores what it does not understand.

Understood (German and English, case-insensitive):
    time windows   8-11 h · 8 - 11 Uhr · 8.30-18 h · 8:30-18:00 · von 8 bis 11 Uhr · 18-6 h (overnight)
    days           Mo-Fr · Mo, Mi, Fr · Sa · werktags (Mo-Sa) · täglich · Mon-Fri ...; they apply to the time
                   windows that follow them ("Mo-Fr 8-18 h Sa 8-13 h")
    max. duration  max. 3 Std. · 2 Std. · 1/2 Std. · 1,5 Std. · 90 Min. · 2 hours · max. 2 h
    flags          mit Parkscheibe (parking disc) · mit Parkschein / Ticket (pay at the machine)

How a schedule is applied depends on the sign it hangs under (StVO meaning):
    no_parking / no_stopping (286 / 283)  the ban applies only during the windows
    parking (314) + max. duration         during the windows you may stay at most that long; outside them: free
    parking (314) + ticket                a ticket is needed during the windows (BITT has none -> illegal)
    parking (314) + windows only          parking is only allowed during the windows
Simplification: the time limit is compared with the minutes of your stay that fall inside the windows.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re

DAY_NAMES = {
    'mo': 0, 'di': 1, 'mi': 2, 'do': 3, 'fr': 4, 'sa': 5, 'so': 6,
    'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6,
}
DAY_LABEL = ('Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So')
ALL_DAYS = tuple(range(7))
WORKDAYS = tuple(range(6))   # werktags = Mo-Sa in Germany


@dataclass
class Schedule:
    windows: list = field(default_factory=list)  # [(days tuple, start minute, end minute)]; end <= start = overnight
    max_minutes: int | None = None
    ticket: bool = False
    disc: bool = False

    def empty(self):
        return not (self.windows or self.max_minutes or self.ticket)

    def key(self):
        """Canonical form, used to vote between several readings of the same plate."""
        return repr((sorted(self.windows), self.max_minutes, self.ticket, self.disc))

    def to_dict(self):
        return {'windows': [[list(d), s, e] for d, s, e in self.windows], 'max_minutes': self.max_minutes,
                'ticket': self.ticket, 'disc': self.disc}

    @classmethod
    def from_dict(cls, d):
        return cls([(tuple(w[0]), w[1], w[2]) for w in d['windows']], d['max_minutes'], d['ticket'], d['disc'])

    def describe(self):
        parts = [f'{_days_text(d)} {_hm(s)}-{_hm(e)} h'.strip() for d, s, e in self.windows]
        if self.max_minutes:
            parts.append(f'max. {_dur(self.max_minutes)}')
        if self.ticket:
            parts.append('ticket')
        if self.disc:
            parts.append('parking disc')
        return ', '.join(parts)


def _hm(minutes):
    h, m = divmod(minutes, 60)
    return f'{h}' if m == 0 else f'{h}.{m:02d}'


def _dur(minutes):
    return f'{minutes // 60} h' if minutes % 60 == 0 else (f'{minutes} min' if minutes < 60 else f'{minutes / 60:g} h')


def _days_text(days):
    days = tuple(sorted(days))
    if days == ALL_DAYS:
        return ''
    if len(days) > 2 and days == tuple(range(days[0], days[-1] + 1)):
        return f'{DAY_LABEL[days[0]]}-{DAY_LABEL[days[-1]]}'
    return ', '.join(DAY_LABEL[d] for d in days)


# --- parsing --------------------------------------------------------------------------------------------------------

_DAY = r'(?:mon|tue|wed|thu|fri|sat|sun|mo|di|mi|do|fr|sa|so)'
_TIME = r'(\d{1,2})(?:[.:](\d{2}))?'
_TOKEN = re.compile(
    rf'(?P<days>{_DAY}\.?\s*-\s*{_DAY}\b|{_DAY}\b(?:\.?\s*,\s*{_DAY}\b)*)'
    rf'|(?P<werktags>werktags?|workdays?|weekdays?)'
    rf'|(?P<daily>t(?:ä|ae|a)glich|daily)'
    rf'|(?P<window>{_TIME}\s*-\s*{_TIME}\s*(?:h\b|uhr\b)?)'
    rf'|(?P<half>1\s*/\s*2)\s*(?:std|stunden?|h\b)'
    rf'|max\.?\s*(?P<maxh>\d+(?:[.,]\d+)?)\s*h\b'
    rf'|(?P<dur>\d+(?:[.,]\d+)?)\s*(?P<unit>std|stunden?|hours?|hrs?|min(?:uten|utes)?)'
    rf'|(?P<disc>parkscheibe|parking\s*disc)'
    rf'|(?P<ticket>parkschein|ticket)',
)


_KINDS = ('days', 'werktags', 'daily', 'window', 'half', 'maxh', 'dur', 'disc', 'ticket')


def _normalise(text):
    t = text.lower()
    t = re.sub(r'[‐-―−~]', '-', t)                        # all kinds of dashes
    t = re.sub(r'(?<=\d)\s*(?:bis|to|until)\s*(?=\d)', '-', t)           # 8 bis 11 -> 8-11
    t = re.sub(r'\b(?:von|from)\s+(?=\d)', '', t)
    t = re.sub(r'(?<=\d)[o](?=\d|\b)|(?<=\b)[o](?=\d)', '0', t)          # OCR: 1O -> 10, O8 -> 08
    t = re.sub(r'(?<=\d)[il|](?=\d|\b)|(?<=[\s-])[il|](?=\d)', '1', t)   # OCR: l8 -> 18, 1l -> 11
    return t


def _day_set(text):
    names = re.findall(_DAY, text)
    if '-' in text and len(names) == 2:
        a, b = DAY_NAMES[names[0]], DAY_NAMES[names[1]]
        return tuple(range(a, b + 1)) if a <= b else tuple(list(range(a, 7)) + list(range(0, b + 1)))
    return tuple(sorted({DAY_NAMES[n] for n in names}))


def parse(text):
    """Plate text -> Schedule, or None if nothing in it was understood."""
    if not text:
        return None
    s = Schedule()
    days = ALL_DAYS
    for m in _TOKEN.finditer(_normalise(text)):
        # The first named alternative that matched (lastgroup would report inner groups such as 'unit').
        g = next(k for k in _KINDS if m.group(k) is not None)
        if g == 'days':
            days = _day_set(m.group('days'))
        elif g == 'werktags':
            days = WORKDAYS
        elif g == 'daily':
            days = ALL_DAYS
        elif g == 'window':
            h1, m1, h2, m2 = (int(x) if x else 0 for x in m.group(5, 6, 7, 8))
            if h1 <= 24 and h2 <= 24 and m1 < 60 and m2 < 60:
                start, end = h1 * 60 + m1, h2 * 60 + m2
                if start != end:
                    s.windows.append((days, start % 1440, end if end == 1440 else end % 1440))
        elif g == 'half':
            s.max_minutes = 30
        elif g == 'maxh':
            s.max_minutes = int(round(float(m.group('maxh').replace(',', '.')) * 60))
        elif g == 'dur':
            value = float(m.group('dur').replace(',', '.'))
            s.max_minutes = int(round(value if m.group('unit').startswith('min') else value * 60))
        elif g == 'disc':
            s.disc = True
        elif g == 'ticket':
            s.ticket = True
    return None if s.empty() and not s.disc else s


# --- applying -------------------------------------------------------------------------------------------------------

def minutes_inside(windows, start, minutes):
    """How many minutes of the stay [start, start + minutes] fall inside the schedule's windows."""
    end = start + timedelta(minutes=minutes)
    total = 0.0
    day = datetime(start.year, start.month, start.day) - timedelta(days=1)  # overnight windows from the day before
    while day <= end:
        for days, s, e in windows:
            if day.weekday() in days:
                w0 = day + timedelta(minutes=s)
                w1 = day + timedelta(minutes=e if e > s else e + 1440)
                total += max(0.0, (min(end, w1) - max(start, w0)).total_seconds() / 60)
        day += timedelta(days=1)
    return round(total)


def violation(kind, base_reason, schedule, plate, start, minutes):
    """Why parking under this sign from `start` for `minutes` is illegal, or None if it is legal.

    kind: the sign's label; plate: 'none' (no extra plate seen), 'unread' (seen, text not understood yet) or 'read'.
    """
    if plate == 'unread':
        return f'{base_reason}: extra plate not read yet'
    if kind == 'parking':
        if schedule is None:
            return None
        windows = schedule.windows or [(ALL_DAYS, 0, 1440)]
        inside = minutes_inside(windows, start, minutes)
        when = schedule.describe()
        if schedule.ticket and inside > 0:
            return f'parking ticket required ({when})'
        if schedule.max_minutes:
            if inside > schedule.max_minutes:
                return f'max. {_dur(schedule.max_minutes)} parking ({when}); this stay would be {_dur(inside)}'
            return None
        if schedule.windows and not (schedule.ticket or schedule.disc) and inside < minutes:
            return f'parking only {when}'
        return None
    # A ban (no parking / no stopping): always, or only during the plate's windows.
    if schedule is None or not schedule.windows:
        return base_reason
    if minutes_inside(schedule.windows, start, minutes) > 0:
        return f'{base_reason}, {schedule.describe()}'
    return None


# --- command-line helpers -------------------------------------------------------------------------------------------

def local_now(tz='Europe/Berlin'):
    """Local time at the signs as a naive datetime. Uses the laptop's clock (the car's clock is often wrong) and an
    explicit time zone, because the dev container usually runs in UTC."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz)).replace(tzinfo=None)
    except Exception:   # no tz database in the container: fall back to the machine's local time
        return datetime.now()


def parse_when(text, tz='Europe/Berlin'):
    """'09:30' (today) or '2026-11-20 09:30' -> datetime; '' / 'now' -> local_now()."""
    if not text or text == 'now':
        return local_now(tz)
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    hh, mm = (int(v) for v in text.split(':'))
    return local_now(tz).replace(hour=hh, minute=mm, second=0, microsecond=0)


def parse_minutes(text):
    """'2h' / '1.5h' / '90m' / '90' (minutes) / '1h30' -> minutes."""
    t = text.strip().lower().replace(',', '.')
    m = re.fullmatch(r'(\d+(?:\.\d+)?)h(?:(\d+)m?)?', t)
    if m:
        return int(round(float(m.group(1)) * 60)) + int(m.group(2) or 0)
    m = re.fullmatch(r'(\d+)(?:m|min)?', t)
    if m:
        return int(m.group(1))
    raise ValueError(f'cannot read duration {text!r} (use e.g. 2h, 1.5h, 90m, 1h30)')
