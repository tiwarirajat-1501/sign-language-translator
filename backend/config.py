"""
All tunable constants live here so nothing is hardcoded across modules.
Tweak these instead of hunting through the codebase.
"""

from pathlib import Path

# ---- paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "svm_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# ---- mediapipe ----
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.30
MIN_TRACKING_CONFIDENCE = 0.30

# ---- prediction smoothing (raw predictions -> stable prediction) ----
BUFFER_SIZE = 15
AGREEMENT_THRESHOLD = 0.70

# Instead of a flat majority vote, weight each buffered frame by
# (confidence * recency) so a strong, recent prediction outweighs a
# handful of stale or low-confidence ones. RECENCY_DECAY < 1 means older
# frames count for less; 1.0 is equivalent to the old plain vote.
RECENCY_DECAY = 0.90

# ---- confidence-based filtering + adaptive hold time ----
# These ONLY activate when the predictor reports has_calibrated_confidence
# (i.e. the model was retrained with probability=True -- see
# scripts/retrain_with_probability.py). With the uncalibrated fallback
# confidence, these thresholds would reject almost everything -- verified
# on this project's actual model before adding this.
#
# Each tier is (confidence_threshold, hold_seconds), highest first. A sign
# held with very high confidence gets accepted fast; one near the floor
# still gets accepted, just after a longer hold to be safe.
CONFIDENCE_HOLD_TIERS = [
    (0.99, 0.4),
    (0.95, 0.7),
    (0.80, 1.2),
]
MIN_CONFIDENCE_TO_ACCEPT = 0.80  # below this, never accept, regardless of hold time


def hold_seconds_for_confidence(confidence):
    """Look up the adaptive hold time for a given calibrated confidence."""
    for threshold, hold in CONFIDENCE_HOLD_TIERS:
        if confidence >= threshold:
            return hold
    return HOLD_SECONDS  # below the lowest tier -- MIN_CONFIDENCE_TO_ACCEPT gates it out anyway


# ---- Phase 1: smart text construction ----
HOLD_SECONDS = 1.0          # how long a stable sign must be held before it's auto-accepted
COOLDOWN_SECONDS = 0.5      # short pause after accepting, before another letter can be accepted
CLEAR_HOLD_SECONDS = 2.5    # holding the "del" sign this long clears the whole line instead

# special classifier labels that map to actions instead of literal characters
SPACE_LABEL = "space"
DELETE_LABEL = "del"
NOTHING_LABEL = "nothing"  # "hand visible but not forming any sign" -- never typed

# ---- Phase 5: performance ----
PROCESS_EVERY_N_FRAMES = 1  # set to 2 or 3 on slower machines to skip mediapipe on some frames

# ---- Phase 4: prediction history ----
HISTORY_LENGTH = 8

# ---- window / UI ----
WINDOW_NAME = "American Sign Language Translator"
DARK_BG = (24, 24, 24)
