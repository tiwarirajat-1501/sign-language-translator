"""
Phase 1: Smart Text Construction.

Turns a per-frame "stable_prediction" signal into typed text, with no
SPACE key required:

- Automatic Letter Acceptance: a sign held steadily for HOLD_SECONDS gets
  accepted on its own.
- Duplicate Prevention: as long as the same sign stays in front of the
  camera, it won't be accepted a second time.
- Hand Removal Detection: pulling the hand out of frame clears the
  "already accepted" memory, so the same letter can be typed twice in a
  row (e.g. the double L in "HELLO") by showing it, removing the hand,
  and showing it again.
- Cooldown Timer: a short pause after every acceptance so a slightly
  early re-recognition of the same sign doesn't sneak in a duplicate.
"""

import time

from config import HOLD_SECONDS, COOLDOWN_SECONDS, CLEAR_HOLD_SECONDS, SPACE_LABEL, DELETE_LABEL


class TextBuilder:
    def __init__(self, hold_seconds=HOLD_SECONDS, cooldown_seconds=COOLDOWN_SECONDS,
                 clear_hold_seconds=CLEAR_HOLD_SECONDS):
        self.text = ""

        self._hold_seconds = hold_seconds
        self._cooldown_seconds = cooldown_seconds
        self._clear_hold_seconds = clear_hold_seconds

        self._candidate = None          # sign currently being "held"
        self._candidate_since = None    # when it started being held
        self._last_accepted = None      # last sign we actually typed (duplicate guard)
        self._cooldown_until = 0.0

    def on_hand_lost(self):
        """Call this whenever the current frame has no stable prediction
        (no hand, or hand present but not yet confidently classified)."""
        self._candidate = None
        self._candidate_since = None
        self._last_accepted = None

    def update(self, stable_prediction, now=None):
        """
        Call once per frame with the current stable_prediction (or None).

        Returns a dict:
          accepted   -> the label just accepted this frame, or None
          progress   -> 0.0-1.0, how far through the hold time we are (for a progress bar)
          cleared    -> True if this frame triggered a full clear (long-held DELETE)
        """
        now = now if now is not None else time.time()

        if stable_prediction is None:
            self.on_hand_lost()
            return {"accepted": None, "progress": 0.0, "cleared": False}

        if stable_prediction != self._candidate:
            self._candidate = stable_prediction
            self._candidate_since = now
            return {"accepted": None, "progress": 0.0, "cleared": False}

        held_for = now - self._candidate_since
        progress = min(held_for / self._hold_seconds, 1.0)

        # Long-held DELETE clears the whole line instead of deleting one char.
        if stable_prediction == DELETE_LABEL and held_for >= self._clear_hold_seconds:
            self.clear()
            self._candidate = None
            self._candidate_since = None
            return {"accepted": None, "progress": 1.0, "cleared": True}

        if held_for < self._hold_seconds:
            return {"accepted": None, "progress": progress, "cleared": False}

        if now < self._cooldown_until:
            return {"accepted": None, "progress": progress, "cleared": False}

        if stable_prediction == self._last_accepted:
            # Same sign, still being held -> already typed it, don't repeat.
            return {"accepted": None, "progress": progress, "cleared": False}

        self._apply(stable_prediction)
        self._last_accepted = stable_prediction
        self._cooldown_until = now + self._cooldown_seconds

        return {"accepted": stable_prediction, "progress": 1.0, "cleared": False}

    def _apply(self, label):
        if label == SPACE_LABEL:
            self.text += " "
        elif label == DELETE_LABEL:
            self.text = self.text[:-1]
        else:
            self.text += label

    def manual_delete(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""
