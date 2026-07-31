"""Small standalone helpers used by both the CLI and GUI front ends."""

import time


class FPSCounter:
    """Rolling FPS estimate based on an exponential moving average."""

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self._last_time = None
        self.fps = 0.0

    def tick(self):
        now = time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps = (self.smoothing * self.fps) + ((1 - self.smoothing) * instant_fps)
        self._last_time = now
        return self.fps
