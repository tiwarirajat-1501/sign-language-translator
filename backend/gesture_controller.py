"""
Phase 2: Gesture Commands.

Your classifier already includes "space" and "del" as trained sign
classes (see label_encoder.pkl). Once Phase 1's auto-accept is in place,
holding the "space" or "del" sign IS the gesture command -- no keyboard
needed for those two anymore.

The one command the model can't give you directly is CLEAR, since there's
no "clear" sign in the training data. Rather than retrain the model just
for that, TextBuilder treats a *long-held* "del" (CLEAR_HOLD_SECONDS,
default 2.5s) as CLEAR, and a normal hold as a single delete. That gives
you a fully gesture-controlled translator without adding a new class:

    hold a letter          -> types the letter
    hold "space"            -> types a space
    hold "del" briefly       -> deletes one character
    hold "del" a bit longer  -> clears the whole line

If you later add real "clear"/"enter" signs to your dataset and retrain,
just add them as new SPECIAL_LABELS here and branch on them in
TextBuilder._apply() -- the rest of the pipeline doesn't need to change.

The keyboard controls (SPACE/BACKSPACE/C/Q) are kept in realtime_recognition.py
as a manual fallback/debug path -- they don't get removed, gestures just make
them optional.
"""

from config import SPACE_LABEL, DELETE_LABEL

COMMAND_LABELS = {SPACE_LABEL, DELETE_LABEL}


def is_command_label(label):
    return label in COMMAND_LABELS
