import cv2
import mediapipe as mp
import numpy as np
import joblib

from pathlib import Path
from collections import deque, Counter


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "svm_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

print("Project root:", PROJECT_ROOT)
print("V2 model path:", MODEL_PATH)


# ============================================================
# 2. CHECK MODEL FILES
# ============================================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# 3. LOAD V2 MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

print("Model, scaler and label encoder loaded successfully!")


# ============================================================
# 4. MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.30,
    min_tracking_confidence=0.30
)


# ============================================================
# 5. PREDICTION SMOOTHING
# ============================================================

BUFFER_SIZE = 15
AGREEMENT_THRESHOLD = 0.70

prediction_buffer = deque(
    maxlen=BUFFER_SIZE
)

stable_prediction = None


# ============================================================
# 6. TEXT CONSTRUCTION
# ============================================================

translated_text = ""


# ============================================================
# 7. LANDMARK NORMALIZATION
# ============================================================

def normalize_landmarks(hand_landmarks):
    """
    Converts MediaPipe's 21 hand landmarks into the same
    normalized 63-feature representation used for SVM V2.
    """

    # --------------------------------------------------------
    # Extract 21 x 3 coordinates
    # --------------------------------------------------------

    landmarks = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float64
    )

    # Safety check
    if landmarks.shape != (21, 3):
        return None


    # --------------------------------------------------------
    # Translation normalization
    # Landmark 0 = wrist
    # --------------------------------------------------------

    wrist = landmarks[0].copy()

    landmarks = landmarks - wrist


    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    scale = distances.max()

    if scale <= 1e-8:
        return None

    landmarks = landmarks / scale


    # --------------------------------------------------------
    # 21 x 3 -> 63 features
    # --------------------------------------------------------

    features = landmarks.flatten()

    if len(features) != 63:
        return None

    return features


# ============================================================
# 8. START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    hands.close()

    raise RuntimeError(
        "Could not open webcam."
    )


print("Webcam opened successfully!")
print("Running ASL Sign Language Translator")
print("=" * 60)
print("American Sign Language Translator")
print("=" * 60)
print("Controls")
print(" SPACEBAR  : Accept current stable letter")
print(" BACKSPACE : Delete last character")
print(" C         : Clear translated text")
print(" Q         : Quit")
print("=" * 60)


# ============================================================
# 9. REAL-TIME LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam frame.")
        break


    # --------------------------------------------------------
    # Mirror webcam
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe hand detection
    # --------------------------------------------------------

    results = hands.process(
        frame_rgb
    )


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )


        # ----------------------------------------------------
        # Draw hand landmarks
        # ----------------------------------------------------

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


        # ----------------------------------------------------
        # Normalize landmarks
        # ----------------------------------------------------

        features = normalize_landmarks(
            hand_landmarks
        )


        if features is not None:


            # ------------------------------------------------
            # (63,) -> (1, 63)
            # ------------------------------------------------

            features_array = features.reshape(
                1,
                -1
            )


            # ------------------------------------------------
            # StandardScaler V2
            # ------------------------------------------------

            features_scaled = scaler.transform(
                features_array
            )


            # ------------------------------------------------
            # SVM prediction
            # ------------------------------------------------

            prediction_encoded = model.predict(
                features_scaled
            )


            # ------------------------------------------------
            # Decode prediction
            # ------------------------------------------------

            predicted_label = (
                label_encoder.inverse_transform(
                    prediction_encoded
                )[0]
            )


            # =================================================
            # PREDICTION SMOOTHING
            # =================================================

            prediction_buffer.append(
                predicted_label
            )


            prediction_counts = Counter(
                prediction_buffer
            )


            most_common_label, count = (
                prediction_counts
                .most_common(1)[0]
            )


            agreement = (
                count /
                len(prediction_buffer)
            )


            # ------------------------------------------------
            # Update stable prediction
            # ------------------------------------------------

            if agreement >= AGREEMENT_THRESHOLD:

                stable_prediction = (
                    most_common_label
                )


            # =================================================
            # DISPLAY RAW PREDICTION
            # =================================================

            cv2.putText(
                frame,
                f"Raw: {predicted_label}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # DISPLAY STABLE PREDICTION
            # =================================================

            if stable_prediction is not None:

                cv2.putText(
                    frame,
                    f"Stable: {stable_prediction}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )


                cv2.putText(
                    frame,
                    f"Agreement: {agreement * 100:.0f}%",
                    (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )


            else:

                cv2.putText(
                    frame,
                    "Stable: detecting...",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )


        # ====================================================
        # INVALID LANDMARKS
        # ====================================================

        else:

            prediction_buffer.clear()

            stable_prediction = None

            cv2.putText(
                frame,
                "Invalid landmarks",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )


    # ========================================================
    # NO HAND DETECTED
    # ========================================================

    else:

        prediction_buffer.clear()

        stable_prediction = None

        cv2.putText(
            frame,
            "No hand detected",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # DISPLAY TRANSLATED TEXT
    # ========================================================

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]


    # --------------------------------------------------------
    # Bottom text panel
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (20, frame_height - 85),
        (frame_width - 20, frame_height - 20),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"Text: {translated_text}",
        (30, frame_height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # DISPLAY CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "SPACE: Accept | C: Clear | BACKSPACE: Delete | Q: Quit",
        (20, frame_height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "American Sign Language Translator",
        frame
    )


    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # Q -> Quit
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # SPACEBAR -> Accept stable prediction
    # --------------------------------------------------------

    elif key == ord(" "):

        if stable_prediction is not None:


            # ------------------------------------------------
            # SPACE gesture
            # ------------------------------------------------

            if stable_prediction == "space":

                translated_text += " "


            # ------------------------------------------------
            # DELETE gesture
            # ------------------------------------------------

            elif stable_prediction == "del":

                translated_text = (
                    translated_text[:-1]
                )


            # ------------------------------------------------
            # Normal alphabet letter
            # ------------------------------------------------

            else:

                translated_text += (
                    stable_prediction
                )


            print(
                "Accepted:",
                stable_prediction,
                "| Text:",
                translated_text
            )


    # --------------------------------------------------------
    # BACKSPACE -> Delete last character manually
    # --------------------------------------------------------

    elif key == 8:

        if len(translated_text) > 0:

            translated_text = (
                translated_text[:-1]
            )

        print(
            "Deleted last character | Text:",
            translated_text
        )


    # --------------------------------------------------------
    # C -> Clear translated text
    # --------------------------------------------------------

    elif key == ord("c"):

        translated_text = ""

        print("Text cleared.")


# ============================================================
# 10. CLEANUP
# ============================================================

hands.close()

cap.release()

cv2.destroyAllWindows()

print("\nTranslator stopped successfully.")

if translated_text:
    print(f"Final translated text: {translated_text}")
else:
    print("No text was translated.")