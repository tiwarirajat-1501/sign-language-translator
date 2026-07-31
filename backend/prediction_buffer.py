"""
Smarter Prediction Buffer.

The original approach was a flat majority vote over the last BUFFER_SIZE
frames: whichever label appeared most often won, full stop. That treats
a confident, just-now prediction the same as a shaky one from half a
second ago.

This buffer instead weights every vote by confidence * recency, so:
  - a high-confidence prediction counts for more than a shaky one
  - a recent prediction counts for more than a stale one (RECENCY_DECAY < 1)

With no confidence available (confidence=None), every vote is weighted
1.0 and this degenerates back to recency-only weighting -- still an
improvement over a flat vote, and harmless if you haven't retrained with
probability=True yet.
"""

from collections import deque

from config import BUFFER_SIZE, RECENCY_DECAY


class SmartBuffer:
    def __init__(self, maxlen=BUFFER_SIZE, recency_decay=RECENCY_DECAY):
        self._items = deque(maxlen=maxlen)  # (label, confidence)
        self._recency_decay = recency_decay

    def append(self, label, confidence=None):
        self._items.append((label, confidence if confidence is not None else 1.0))

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)

    def vote(self):
        """
        Returns (best_label, agreement, avg_confidence_for_best):
          best_label            -> the winning label, or None if buffer is empty
          agreement             -> winning weight / total weight, in [0, 1]
          avg_confidence_for_best -> mean raw confidence of frames that voted
                                     for the winner (None if none had confidence)
        """
        if not self._items:
            return None, 0.0, None

        n = len(self._items)
        weight_by_label = {}
        conf_sum_by_label = {}
        conf_count_by_label = {}
        total_weight = 0.0

        for i, (label, confidence) in enumerate(self._items):
            age = n - 1 - i  # 0 = most recent frame
            recency_weight = self._recency_decay ** age
            w = confidence * recency_weight

            weight_by_label[label] = weight_by_label.get(label, 0.0) + w
            conf_sum_by_label[label] = conf_sum_by_label.get(label, 0.0) + confidence
            conf_count_by_label[label] = conf_count_by_label.get(label, 0) + 1
            total_weight += w

        best_label = max(weight_by_label, key=weight_by_label.get)
        agreement = weight_by_label[best_label] / total_weight if total_weight > 0 else 0.0
        avg_confidence = conf_sum_by_label[best_label] / conf_count_by_label[best_label]

        return best_label, agreement, avg_confidence
