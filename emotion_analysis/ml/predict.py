# ml/predict.py
import numpy as np
import pickle
import time
import os

from .config import (
    MODEL_PATH_SVM, SCALER_PATH,
    EMOTION_LABELS, EMOTION_LABELS_RU, EMOTION_AFFECT_MAP
)
from .preprocess import load_and_clean
from .features import extract_features

_model = None
_scaler = None


def _load_model_and_scaler():
    global _model, _scaler

    print("[DEBUG] _load_model_and_scaler called")

    if _model is None:
        # Используем SVM модель
        svm_path = MODEL_PATH_SVM
        print(f"[predict] Загружаем SVM модель из {svm_path}")

        if not os.path.exists(svm_path):
            print(f"[ERROR] Model file not found: {svm_path}")
            print(f"[ERROR] Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Model file not found: {svm_path}")

        try:
            with open(svm_path, "rb") as f:
                _model = pickle.load(f)
            print("[predict] SVM модель загружена успешно")
            print(f"[DEBUG] Model type: {type(_model)}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise

    if _scaler is None:
        print(f"[predict] Загружаем scaler из {SCALER_PATH}")

        if not os.path.exists(SCALER_PATH):
            print(f"[ERROR] Scaler file not found: {SCALER_PATH}")
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

        try:
            with open(SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)
            print("[predict] Scaler загружен успешно")
            print(f"[DEBUG] Scaler type: {type(_scaler)}")
        except Exception as e:
            print(f"[ERROR] Failed to load scaler: {e}")
            raise


def predict_emotion(audio_path: str) -> dict:
    _load_model_and_scaler()

    start_time = time.time()

    signal, sr = load_and_clean(audio_path)

    feat = extract_features(signal, sr)

    # Проверяем что feat не пустой и имеет правильную размерность
    if feat is None or len(feat) == 0:
        raise ValueError("Failed to extract features from audio")

    feat_scaled = _scaler.transform(feat.reshape(1, -1))

    probs_raw = _model.predict_proba(feat_scaled)[0]

    probabilities = {
        EMOTION_LABELS[cls_idx]: round(float(prob), 4)
        for cls_idx, prob in enumerate(probs_raw)
    }

    dominant_emotion = max(probabilities, key=probabilities.get)
    confidence = probabilities[dominant_emotion]

    valence = sum(
        prob * EMOTION_AFFECT_MAP[emotion]["valence"]
        for emotion, prob in probabilities.items()
    )
    arousal = sum(
        prob * EMOTION_AFFECT_MAP[emotion]["arousal"]
        for emotion, prob in probabilities.items()
    )

    processing_time = round(time.time() - start_time, 3)

    return {
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_ru": EMOTION_LABELS_RU[dominant_emotion],
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "valence": round(valence, 4),
        "arousal": round(arousal, 4),
        "intensity": round(confidence, 4),
        "processing_time": processing_time,
    }