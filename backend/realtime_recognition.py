"""
Real-time ASL translator - main entry point.

This file only orchestrates: capture a frame, get landmarks, predict,
smooth, build text, draw overlay, handle keys. All the actual logic
lives in the other modules (Phase 6: modular backend):

    predictor.py           model + scaler + label encoder
    normalizer.py           landmark -> feature vector
    text_builder.py         Phase 1: auto-accept / duplicate prevention / cooldown
    gesture_controller.py   Phase 2: which labels are commands vs letters
    ui.py                   all overlay drawing
    utils.py                FPS counter
    tts.py                  Phase 8: speak the current text
    config.py               every tunable constant
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TensorFlow Lite INFO/WARNING spam
os.environ["GLOG_minloglevel"] = "3"       # silence glog/absl (used internally by mediapipe)

import warnings
# sklearn: scaler was fit on a DataFrame (column names) but we feed it a plain
# array every frame -- harmless, just noisy at 30fps.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# protobuf: internal deprecation notice from inside mediapipe, nothing we control.
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import mediapipe as mp
from collections import deque

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

import config
import ui
from normalizer import normalize_landmarks
from predictor import Predictor
from text_builder import TextBuilder
from prediction_buffer import SmartBuffer
from utils import FPSCounter
from tts import Speaker


def main():
    print("Project root:", config.PROJECT_ROOT)
    print("Model path:", config.MODEL_PATH)

    predictor = Predictor()
    print("Model, scaler and label encoder loaded successfully!")
    if predictor.has_calibrated_confidence:
        print("Calibrated confidence available -- confidence gating and adaptive hold are ACTIVE.")
    else:
        print("No calibrated confidence (model wasn't trained with probability=True).")
        print("Confidence gating and adaptive hold are OFF; using fixed hold time instead.")
        print("Run scripts/retrain_with_probability.py to enable them.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
    )

    text_builder = TextBuilder()
    speaker = Speaker()
    fps_counter = FPSCounter()
    history = deque(maxlen=config.HISTORY_LENGTH)

    prediction_buffer = SmartBuffer()
    stable_prediction = None
    last_confidence = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        hands.close()
        raise RuntimeError("Could not open webcam.")

    print("Webcam opened successfully!")
    print("Running ASL Sign Language Translator")
    print("=" * 60)
    print("Signs are now auto-accepted after a short hold - no SPACE needed.")
    print("Controls (manual fallback / debug)")
    print(" SPACEBAR  : Accept current stable letter manually")
    print(" BACKSPACE : Delete last character")
    print(" C         : Clear translated text")
    print(" T         : Speak the current text out loud")
    print(" Q         : Quit")
    print("=" * 60)

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            fps = fps_counter.tick()
            frame_index += 1

            # Phase 5: optionally skip mediapipe on some frames to save CPU,
            # while still rendering every frame for a smooth-looking video.
            run_detection = (frame_index % config.PROCESS_EVERY_N_FRAMES == 0)

            hand_present = False

            if run_detection:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                

                if results.multi_hand_landmarks:
                    hand_present = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_label = results.multi_handedness[0].classification[0].label
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    features = normalize_landmarks(hand_landmarks, hand_label)

                    if features is not None:
                        predicted_label, confidence = predictor.predict(features)
                        last_confidence = confidence

                        prediction_buffer.append(predicted_label, confidence)
                        most_common_label, agreement, stable_confidence = prediction_buffer.vote()

                        if agreement >= config.AGREEMENT_THRESHOLD:
                            stable_prediction = most_common_label
                        else:
                            stable_prediction = None

                        ui.draw_raw_prediction(frame, predicted_label)

                        # Only pass confidence through when it's calibrated (real
                        # predict_proba) -- otherwise text_builder falls back to the
                        # fixed hold time / no gating, which is the safe default.
                        confidence_for_gating = (stable_confidence
                                                  if predictor.has_calibrated_confidence else None)
                        result = text_builder.update(stable_prediction, confidence=confidence_for_gating)
                        if result["accepted"]:
                            history.append(result["accepted"])
                            print(
    f"Accepted: {result['accepted']} | "
    f"Hand: {hand_label} | "
    f"Confidence: {confidence:.2%} | "
    f"Text: {text_builder.text}"
)

                        if stable_prediction is not None:
                            ui.draw_stable_prediction(frame, stable_prediction, agreement,
                                                       last_confidence, result["progress"])
                        else:
                            ui.draw_detecting(frame)
                    else:
                        prediction_buffer.clear()
                        stable_prediction = None
                        text_builder.on_hand_lost()
                        ui.draw_invalid_landmarks(frame)

            if not hand_present:
                prediction_buffer.clear()
                stable_prediction = None
                text_builder.on_hand_lost()
                ui.draw_no_hand(frame)

            ui.draw_history(frame, history)
            ui.draw_fps(frame, fps)
            ui.draw_text_panel(frame, text_builder.text)
            ui.draw_controls(frame, "SPACE: Accept | C: Clear | BACKSPACE: Delete | T: Speak | Q: Quit")

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                if stable_prediction is not None:
                    text_builder._apply(stable_prediction)
                    print("Manually accepted:", stable_prediction, "| Text:", text_builder.text)
            elif key == 8:
                text_builder.manual_delete()
                print("Deleted last character | Text:", text_builder.text)
            elif key == ord("c"):
                text_builder.clear()
                print("Text cleared.")
            elif key == ord("t"):
                speaker.speak(text_builder.text)

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    print("\nTranslator stopped successfully.")
    if text_builder.text:
        print(f"Final translated text: {text_builder.text}")
    else:
        print("No text was translated.")


if __name__ == "__main__":
    main()
