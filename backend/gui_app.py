"""
Phase 7: GUI.

A Tkinter desktop window around the exact same pipeline used in
realtime_recognition.py (predictor, normalizer, text_builder, gesture
rules). Tkinter ships with Python so there's nothing extra to install
beyond Pillow (for converting OpenCV frames into something Tkinter can
display).

Run with:  python gui_app.py
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

import tkinter as tk
from tkinter import filedialog, messagebox
from collections import deque, Counter

import cv2
from PIL import Image, ImageTk
import mediapipe as mp

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

import config
from normalizer import normalize_landmarks
from predictor import Predictor
from text_builder import TextBuilder
from utils import FPSCounter
from tts import Speaker

DARK_BG = "#181818"
DARK_PANEL = "#242424"
ACCENT = "#4CAF50"
TEXT_COLOR = "#EAEAEA"
MUTED = "#9A9A9A"


class ASLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Translator")
        self.root.configure(bg=DARK_BG)
        self.root.geometry("900x700")

        # ---- pipeline (same modules as the CLI version) ----
        self.predictor = Predictor()
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.text_builder = TextBuilder()
        self.speaker = Speaker()
        self.fps_counter = FPSCounter()
        self.prediction_buffer = deque(maxlen=config.BUFFER_SIZE)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open webcam.")
            self.root.destroy()
            return

        self._build_ui()
        self._tick()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        video_frame = tk.Frame(self.root, bg=DARK_BG)
        video_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        info_frame = tk.Frame(self.root, bg=DARK_PANEL)
        info_frame.pack(side="top", fill="x", padx=10)

        self.stable_var = tk.StringVar(value="Stable: detecting...")
        self.confidence_var = tk.StringVar(value="")
        self.fps_var = tk.StringVar(value="FPS: 0")

        tk.Label(info_frame, textvariable=self.stable_var, font=("Segoe UI", 16, "bold"),
                 fg=ACCENT, bg=DARK_PANEL).pack(side="left", padx=10, pady=6)
        tk.Label(info_frame, textvariable=self.confidence_var, font=("Segoe UI", 11),
                 fg=MUTED, bg=DARK_PANEL).pack(side="left", padx=10)
        tk.Label(info_frame, textvariable=self.fps_var, font=("Segoe UI", 11),
                 fg=MUTED, bg=DARK_PANEL).pack(side="right", padx=10)

        text_frame = tk.Frame(self.root, bg=DARK_BG)
        text_frame.pack(side="top", fill="x", padx=10, pady=(10, 0))

        tk.Label(text_frame, text="Sentence:", font=("Segoe UI", 11),
                 fg=TEXT_COLOR, bg=DARK_BG).pack(anchor="w")

        self.text_box = tk.Text(text_frame, height=3, font=("Segoe UI", 14),
                                 bg=DARK_PANEL, fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                                 wrap="word", relief="flat")
        self.text_box.pack(fill="x", pady=(2, 8))
        self.text_box.configure(state="disabled")

        button_frame = tk.Frame(self.root, bg=DARK_BG)
        button_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        def make_button(label, command):
            return tk.Button(button_frame, text=label, command=command,
                              bg=DARK_PANEL, fg=TEXT_COLOR, activebackground=ACCENT,
                              relief="flat", padx=12, pady=6)

        make_button("Clear", self._on_clear).pack(side="left", padx=4)
        make_button("Backspace", self._on_backspace).pack(side="left", padx=4)
        make_button("Speak", self._on_speak).pack(side="left", padx=4)
        make_button("Save to file", self._on_save).pack(side="left", padx=4)
        make_button("Quit", self._on_quit).pack(side="right", padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

    # ------------------------------------------------------------------
    # main loop (Tkinter-friendly: scheduled with root.after, not a while loop)
    # ------------------------------------------------------------------

    def _tick(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            fps = self.fps_counter.tick()
            self.fps_var.set(f"FPS: {fps:.0f}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            stable_prediction = None
            confidence = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                features = normalize_landmarks(hand_landmarks)
                if features is not None:
                    label, confidence = self.predictor.predict(features)
                    self.prediction_buffer.append(label)
                    most_common_label, count = Counter(self.prediction_buffer).most_common(1)[0]
                    agreement = count / len(self.prediction_buffer)
                    if agreement >= config.AGREEMENT_THRESHOLD:
                        stable_prediction = most_common_label
                else:
                    self.prediction_buffer.clear()
                    self.text_builder.on_hand_lost()
            else:
                self.prediction_buffer.clear()
                self.text_builder.on_hand_lost()

            result = self.text_builder.update(stable_prediction)
            self._update_text_box()

            if stable_prediction is not None:
                self.stable_var.set(f"Stable: {stable_prediction}  ({result['progress'] * 100:.0f}%)")
            else:
                self.stable_var.set("Stable: detecting...")

            self.confidence_var.set(f"Confidence: {confidence * 100:.0f}%" if confidence is not None else "")

            self._render_frame(frame)

        # ~30 fps target; frame processing time already eats into this
        self.root.after(15, self._tick)

    def _render_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # keep a reference or Tkinter garbage-collects it

    def _update_text_box(self):
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", self.text_builder.text)
        self.text_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # button handlers
    # ------------------------------------------------------------------

    def _on_clear(self):
        self.text_builder.clear()
        self._update_text_box()

    def _on_backspace(self):
        self.text_builder.manual_delete()
        self._update_text_box()

    def _on_speak(self):
        self.speaker.speak(self.text_builder.text)

    def _on_save(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text file", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text_builder.text)

    def _on_quit(self):
        self.hands.close()
        self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    ASLTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
