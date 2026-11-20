"""What Rear Buddy wants to know about the car itself: how long it has been standing still, and the battery.
Fed from /wheel_odom and /batt_status by the ROS node or the bridge mode."""
import time

STOPPED_MPS = 0.05  # below this wheel speed the car counts as standing still
STALE_S = 3.0       # wheel data older than this is ignored (battery: twice that, it comes at 1 Hz)


class CarState:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.stopped_since = None
        self.wheels_at = self.battery_at = -1e9
        self.volts = None

    def on_wheels(self, left_mps, right_mps):
        now = self.clock()
        self.wheels_at = now
        if max(abs(left_mps), abs(right_mps)) < STOPPED_MPS:
            self.stopped_since = self.stopped_since if self.stopped_since is not None else now
        else:
            self.stopped_since = None

    def on_battery(self, volts):
        self.volts, self.battery_at = volts, self.clock()

    def stopped_s(self):
        """Seconds standing still, 0 while moving, None without recent wheel data."""
        now = self.clock()
        if now - self.wheels_at >= STALE_S:
            return None
        return now - self.stopped_since if self.stopped_since is not None else 0.0

    def battery_v(self):
        return self.volts if self.clock() - self.battery_at < 2 * STALE_S else None
