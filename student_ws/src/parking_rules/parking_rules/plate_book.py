"""Per-sign bookkeeping of extra-plate readings (thread-safe, no ROS).

A sign's plate is read a few times from different sightings and the readings vote, because a single OCR read is
often wrong. Reading stops once one schedule was read `settle` times or after `max_reads` attempts.

    status 'none'    no plate seen under this sign (in fewer than `min_seen` sightings)
    status 'unread'  a plate is there, but no reading was understood yet  -> zones treat this as illegal (safe side)
    status 'read'    the schedule with the most votes
"""
from collections import Counter
import threading

from .plates import parse


class PlateBook:
    def __init__(self, max_reads=5, settle=2, min_seen=2):
        self.max_reads, self.settle, self.min_seen = max_reads, settle, min_seen
        self._lock = threading.Lock()
        self._seen = Counter()       # sign id -> sightings with a plate below it
        self._reads = Counter()      # sign id -> reads attempted
        self._votes = {}             # sign id -> Counter(schedule key)
        self._schedules = {}         # schedule key -> Schedule
        self._pending = set()
        self.texts = {}              # sign id -> last raw text (for logging / debugging)

    def saw_plate(self, sign_id):
        with self._lock:
            self._seen[sign_id] += 1

    def start_read(self, sign_id):
        """True if a read should be started now for this sign (and marks it pending)."""
        with self._lock:
            votes = self._votes.get(sign_id)
            settled = votes and votes.most_common(1)[0][1] >= self.settle
            if sign_id in self._pending or settled or self._reads[sign_id] >= self.max_reads:
                return False
            self._pending.add(sign_id)
            self._reads[sign_id] += 1
            return True

    def finish_read(self, sign_id, text):
        """Record a reading (text may be None). Returns the parsed Schedule or None."""
        schedule = parse(text) if text else None
        with self._lock:
            self._pending.discard(sign_id)
            self.texts[sign_id] = text
            if schedule:
                self._votes.setdefault(sign_id, Counter())[schedule.key()] += 1
                self._schedules[schedule.key()] = schedule
        return schedule

    def status(self, sign_id):
        with self._lock:
            votes = self._votes.get(sign_id)
            if votes:
                return 'read', self._schedules[votes.most_common(1)[0][0]]
            if self._seen[sign_id] >= self.min_seen:
                return 'unread', None
            return 'none', None

    def all_status(self, sign_ids):
        return {i: self.status(i) for i in sign_ids}
