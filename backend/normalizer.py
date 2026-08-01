"""Turns raw MediaPipe hand landmarks into the 63-feature vector the SVM expects."""

import numpy as np

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    return float(angle)


def normalize_landmarks(hand_landmarks, hand_label):
    """
    Convert MediaPipe's 21 hand landmarks into a translation- and scale-
    invariant 73-feature vector (21 points x 3 coords), matching training.

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

    features = landmarks.flatten().tolist()
    # Thumb
    features.append(calculate_angle(landmarks[1], landmarks[2], landmarks[3]))
    features.append(calculate_angle(landmarks[2], landmarks[3], landmarks[4]))

# Index
    features.append(calculate_angle(landmarks[5], landmarks[6], landmarks[7]))
    features.append(calculate_angle(landmarks[6], landmarks[7], landmarks[8]))

# Middle
    features.append(calculate_angle(landmarks[9], landmarks[10], landmarks[11]))
    features.append(calculate_angle(landmarks[10], landmarks[11], landmarks[12]))

# Ring
    features.append(calculate_angle(landmarks[13], landmarks[14], landmarks[15]))
    features.append(calculate_angle(landmarks[14], landmarks[15], landmarks[16]))

# Pinky
    features.append(calculate_angle(landmarks[17], landmarks[18], landmarks[19]))
    features.append(calculate_angle(landmarks[18], landmarks[19], landmarks[20]))

    if len(features) != 73:
        return None

    return np.array(features, dtype=np.float64)
