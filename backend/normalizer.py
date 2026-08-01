"""Turns raw MediaPipe hand landmarks into the 63-feature vector the SVM expects."""

import numpy as np


def normalize_landmarks(hand_landmarks, hand_label):
    """
    Convert MediaPipe's 21 hand landmarks into a translation- and scale-
    invariant 63-feature vector (21 points x 3 coords), matching training.

    Returns None if the landmarks are malformed or degenerate (e.g. a hand
    collapsed to a single point), so callers can skip that frame safely.
    """
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float64
    )
    if hand_label == "Right":
        landmarks[:, 0] = 1.0 - landmarks[:, 0]

    if landmarks.shape != (21, 3):
        return None

    # center on the wrist (landmark 0)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist

    # scale so the farthest point from the wrist is distance 1
    distances = np.linalg.norm(landmarks, axis=1)
    scale = distances.max()
    if scale <= 1e-8:
        return None
    landmarks = landmarks / scale

    features = landmarks.flatten()
    if len(features) != 63:
        return None

    return features
