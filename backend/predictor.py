"""Wraps the trained SVM + scaler + label encoder behind one simple call."""

import numpy as np
import joblib

from config import MODEL_PATH, SCALER_PATH, ENCODER_PATH


class Predictor:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH, encoder_path=ENCODER_PATH):
        for path in (model_path, scaler_path, encoder_path):
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)

        # Phase 3/6 needs a confidence number. Only some SVMs expose predict_proba
        # (requires probability=True at training time) -- when it's missing we
        # fall back to a softmax over decision_function, which is NOT calibrated
        # (it doesn't reliably separate correct from wrong predictions). Callers
        # that want to gate acceptance on confidence should check
        # has_calibrated_confidence first and only trust the number if True.
        self._has_proba = hasattr(self.model, "predict_proba")
        self._has_decision = hasattr(self.model, "decision_function")

    @property
    def has_calibrated_confidence(self):
        return self._has_proba

    def predict(self, features):
        """
        features: 1D array of length 73.
        Returns (label: str, confidence: float or None).
        confidence is in [0, 1] when available, otherwise None.
        """
        features_array = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)

        prediction_encoded = self.model.predict(features_scaled)
        label = self.label_encoder.inverse_transform(prediction_encoded)[0]
        confidence = self._confidence(features_scaled)

        return label, confidence

    def _confidence(self, features_scaled):
        if self._has_proba:
            probs = self.model.predict_proba(features_scaled)[0]
            return float(np.max(probs))

        if self._has_decision:
            # decision_function gives distances to hyperplanes, not probabilities.
            # Softmax them so we still get a usable 0-1 "confidence" for the UI.
            scores = np.atleast_1d(self.model.decision_function(features_scaled)[0])
            exp_scores = np.exp(scores - np.max(scores))
            softmax = exp_scores / exp_scores.sum()
            return float(np.max(softmax))

        return None
