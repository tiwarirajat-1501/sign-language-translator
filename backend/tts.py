"""
Phase 8: Text-to-Speech.

pyttsx3 works fully offline (uses SAPI5 on Windows, NSSpeech on macOS,
espeak on Linux) so there's no API key or internet dependency.

Speaking is done on a background thread so it never blocks the webcam
loop or freezes the GUI while a sentence is being read out.
"""

import threading

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


class Speaker:
    def __init__(self, rate=170):
        self._rate = rate
        self._lock = threading.Lock()
        self._engine = None
        if _TTS_AVAILABLE:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)

    @property
    def available(self):
        return self._engine is not None

    def speak(self, text):
        """Fire-and-forget: speaks `text` on a background thread."""
        if not text or not text.strip():
            return
        if not self.available:
            print("[tts] pyttsx3 not installed, skipping speech. Run: pip install pyttsx3")
            return

        thread = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        thread.start()

    def _speak_blocking(self, text):
        # pyttsx3 engines aren't thread-safe if called concurrently, so
        # serialize speech requests instead of letting them overlap.
        with self._lock:
            self._engine.say(text)
            self._engine.runAndWait()
