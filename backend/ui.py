"""All the cv2.putText / cv2.rectangle overlay drawing lives here, so
realtime_recognition.py only has to call a few functions instead of being
full of drawing code."""

import cv2


def draw_raw_prediction(frame, label):
    cv2.putText(frame, f"Raw: {label}", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_stable_prediction(frame, label, agreement, confidence, hold_progress):
    cv2.putText(frame, f"Stable: {label}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.putText(frame, f"Agreement: {agreement * 100:.0f}%", (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if confidence is not None:
        cv2.putText(frame, f"Confidence: {confidence * 100:.0f}%", (30, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # hold-to-accept progress bar
    bar_x, bar_y, bar_w, bar_h = 30, 170, 200, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    fill_w = int(bar_w * hold_progress)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)


def draw_detecting(frame):
    cv2.putText(frame, "Stable: detecting...", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)


def draw_invalid_landmarks(frame):
    cv2.putText(frame, "Invalid landmarks", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)


def draw_no_hand(frame):
    cv2.putText(frame, "No hand detected", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)


def draw_history(frame, history):
    """history: iterable of recently accepted labels, most recent last."""
    if not history:
        return
    text = "History: " + " ".join(history)
    cv2.putText(frame, text, (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 1, cv2.LINE_AA)


def draw_fps(frame, fps):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def draw_text_panel(frame, text):
    frame_height, frame_width = frame.shape[:2]
    cv2.rectangle(frame, (20, frame_height - 85), (frame_width - 20, frame_height - 20), (0, 0, 0), -1)
    cv2.putText(frame, f"Text: {text}", (30, frame_height - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_controls(frame, text):
    frame_height = frame.shape[0]
    cv2.putText(frame, text, (20, frame_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
