"""
Phase 1: Smart Text Construction.

Turns a per-frame "stable_prediction" signal into typed text, with no
SPACE key required:

- Automatic Letter Acceptance: a sign held steadily gets accepted on its
  own. The hold time is adaptive when a calibrated confidence score is
  available (see below) -- longer for borderline signs, shorter for
  clear ones.
- Confidence-Based Filtering: if a calibrated confidence is passed in, a
  sign below MIN_CONFIDENCE_TO_ACCEPT is never accepted no matter how
  long it's held. Pass confidence=None (the default) to disable this --
  useful before you've retrained with probability=True, since the
  fallback confidence estimate doesn't separate correct from wrong
  predictions well enough to gate on.
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

from config import (HOLD_SECONDS, COOLDOWN_SECONDS, CLEAR_HOLD_SECONDS,
                     SPACE_LABEL, DELETE_LABEL, NOTHING_LABEL,
                     MIN_CONFIDENCE_TO_ACCEPT, hold_seconds_for_confidence)


class TextBuilder:
    def __init__(self, hold_seconds=HOLD_SECONDS, cooldown_seconds=COOLDOWN_SECONDS,
                 clear_hold_seconds=CLEAR_HOLD_SECONDS,
                 min_confidence_to_accept=MIN_CONFIDENCE_TO_ACCEPT):
        self.text = ""

        self._default_hold_seconds = hold_seconds  # used when confidence is None
        self._cooldown_seconds = cooldown_seconds
        self._clear_hold_seconds = clear_hold_seconds
        self._min_confidence_to_accept = min_confidence_to_accept

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

    def update(self, stable_prediction, confidence=None, now=None):
        """
        Call once per frame with the current stable_prediction (or None)
        and, optionally, a calibrated confidence in [0, 1] for that
        prediction (pass None if the model wasn't trained with
        probability=True -- see predictor.has_calibrated_confidence).

        Returns a dict:
          accepted   -> the label just accepted this frame, or None
          progress   -> 0.0-1.0, how far through the hold time we are (for a progress bar)
          cleared    -> True if this frame triggered a full clear (long-held DELETE)
        """
        now = now if now is not None else time.time()

        # Your model has a "nothing" class (hand present, no clear sign) --
        # treat it exactly like no hand at all: never typed, and it clears
        # the duplicate-prevention memory so the next real sign is fresh.
        if stable_prediction is None or stable_prediction == NOTHING_LABEL:
            self.on_hand_lost()
            return {"accepted": None, "progress": 0.0, "cleared": False}

        if stable_prediction != self._candidate:
            self._candidate = stable_prediction
            self._candidate_since = now
            return {"accepted": None, "progress": 0.0, "cleared": False}

        # Adaptive hold time: only kicks in with a real calibrated confidence.
        required_hold = (hold_seconds_for_confidence(confidence)
                          if confidence is not None else self._default_hold_seconds)

        held_for = now - self._candidate_since
        progress = min(held_for / required_hold, 1.0)

        # Long-held DELETE clears the whole line instead of deleting one char.
        if stable_prediction == DELETE_LABEL and held_for >= self._clear_hold_seconds:
            self.clear()
            self._candidate = None
            self._candidate_since = None
            return {"accepted": None, "progress": 1.0, "cleared": True}

        if held_for < required_hold:
            return {"accepted": None, "progress": progress, "cleared": False}

        # Confidence gate: only enforced when a calibrated confidence was given.
        if confidence is not None and confidence < self._min_confidence_to_accept:
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

