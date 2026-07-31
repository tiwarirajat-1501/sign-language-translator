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

# ---- Phase 1: smart text construction ----
HOLD_SECONDS = 1.0          # how long a stable sign must be held before it's auto-accepted
COOLDOWN_SECONDS = 0.5      # short pause after accepting, before another letter can be accepted
CLEAR_HOLD_SECONDS = 2.5    # holding the "del" sign this long clears the whole line instead

# special classifier labels that map to actions instead of literal characters
SPACE_LABEL = "space"
DELETE_LABEL = "del"

# ---- Phase 5: performance ----
PROCESS_EVERY_N_FRAMES = 1  # set to 2 or 3 on slower machines to skip mediapipe on some frames

# ---- Phase 4: prediction history ----
HISTORY_LENGTH = 8

# ---- window / UI ----
WINDOW_NAME = "American Sign Language Translator"
DARK_BG = (24, 24, 24)
