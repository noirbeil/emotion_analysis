# =============================================================
#  ФАЙЛ: ml/analyzer.py  (обновлённая версия)
#  ИЗМЕНЕНИЯ:
#    + подключён speech_analysis.py (Whisper + ключевые слова)
#    + вероятности сегментов объединяются: аудио × 0.7 + текст × 0.3
#    + подключён report_builder.py (итоговый словарь для frontend)
#    + run_full_analysis() теперь возвращает полный словарь
# =============================================================

import numpy as np
import librosa
import time
import psycopg2
from datetime import datetime

# ИСПРАВЛЕНИЕ: используем относительные импорты
from .config import (
    SAMPLE_RATE, SEGMENT_DURATION, SEGMENT_OVERLAP,
    EMOTION_LABELS, EMOTION_LABELS_RU, EMOTION_AFFECT_MAP,
    MODEL_PATH_SVM  # Добавьте эту строку
)
from .preprocess import load_and_clean
from .features import extract_features, extract_features_for_db
from .predict import predict_emotion, _load_model_and_scaler
import ml.predict as _predict_module

# Новые модули
from .speech_analysis import analyze_speech, fuse_audio_and_text
from .report_builder import build_frontend_result, infer_emotional_pattern

def get_or_create_ml_model_id(conn) -> int:
    """
    Возвращает id SVM-модели из таблицы ml_models.
    Если записи нет — создаёт её автоматически.
    """
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id FROM ml_models
            WHERE file_path LIKE '%emotion_model_svm%'
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            return row[0]

        cur.execute("""
            INSERT INTO ml_models (model_name, version, accuracy, trained_date, file_path)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            'SVM_Emotion_Classifier',
            '1.0',
            None,
            datetime.now().date(),
            'ml/models/emotion_model_svm.pkl'
        ))
        model_id = cur.fetchone()[0]
        conn.commit()
        print(f"[DB] Создана запись ml_models с id={model_id}")
        return model_id
    finally:
        cur.close()


# ============================================================
# ПОДКЛЮЧЕНИЕ К БД
# ============================================================
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="aeog_db",
        user="aeog_user",
        password="user12345"   # ← замени на свой пароль
    )


# ============================================================
# МАРКЕРЫ РИСКА
# ============================================================
def detect_risk_markers(segments: list[dict]) -> list[str]:
    """
    Определяет наблюдаемые маркеры эмоционального неблагополучия.
    НЕ диагноз — описательные паттерны для психолога.
    """
    markers = []
    if not segments:
        return ["Недостаточно данных для определения маркеров"]

    emotions_seq = [s["dominant_emotion"] for s in segments]
    all_probs    = [s["probabilities"] for s in segments]

    # 1. Преобладание негативных эмоций (>60% времени)
    negative = {"sadness", "anger", "fear", "disgust"}
    neg_count = sum(1 for e in emotions_seq if e in negative)
    neg_ratio = neg_count / len(emotions_seq)
    if neg_ratio > 0.6:
        markers.append(
            f"Преобладание негативных эмоций — {neg_ratio * 100:.0f}% времени сессии"
        )

    # 2. Высокая вариативность — частая смена эмоций
    changes = sum(1 for i in range(1, len(emotions_seq))
                  if emotions_seq[i] != emotions_seq[i - 1])
    change_rate = changes / max(len(emotions_seq) - 1, 1)
    if change_rate > 0.55:
        markers.append(
            f"Высокая эмоциональная нестабильность — смена эмоции в {change_rate * 100:.0f}% переходов"
        )

    # 3. Устойчивый страх (≥3 сегмента подряд = 30 сек при SEGMENT_DURATION=10)
    max_fear_streak = cur_streak = 0
    for e in emotions_seq:
        cur_streak = cur_streak + 1 if e == "fear" else 0
        max_fear_streak = max(max_fear_streak, cur_streak)
    if max_fear_streak >= 4:  # ≥4 сегмента = 20 сек при SEGMENT_DURATION=5
        seconds = max_fear_streak * SEGMENT_DURATION
        markers.append(f"Устойчивое состояние страха — {seconds} сек непрерывно")

    # 4. Высокая средняя вероятность грусти
    avg_sad = float(np.mean([p.get("sadness", 0) for p in all_probs]))
    if avg_sad > 0.40:
        markers.append(
            f"Повышенный уровень грусти — средняя вероятность {avg_sad * 100:.0f}%"
        )

    # 5. Низкое возбуждение (подавленность)
    avg_arousal = float(np.mean([s["arousal"] for s in segments]))
    if avg_arousal < 0.28:
        markers.append(
            f"Сниженная энергетика речи — возможный маркер подавленности "
            f"(среднее возбуждение {avg_arousal:.2f})"
        )

    # 6. Высокая тревожность (страх + высокое возбуждение)
    avg_fear = float(np.mean([p.get("fear", 0) for p in all_probs]))
    if avg_fear > 0.35 and avg_arousal > 0.65:
        markers.append(
            f"Признаки тревожного состояния — страх {avg_fear * 100:.0f}%, "
            f"возбуждение {avg_arousal:.2f}"
        )

    if not markers:
        markers.append("Значимых маркеров эмоционального неблагополучия не выявлено")

    return markers


# ============================================================
# СЕГМЕНТНЫЙ АНАЛИЗ С ОБЪЕДИНЕНИЕМ АУДИО + ТЕКСТ
# ============================================================
def analyze_by_segments(audio_path: str, speech_result: dict) -> list[dict]:
    """
    Разбивает аудио на перекрывающиеся сегменты и анализирует каждый.
    """
    try:
        import ml.predict as _predict_module

        _load_model_and_scaler()

        if _predict_module._model is None or _predict_module._scaler is None:
            print("[ERROR] Model or scaler is None after loading")
            return []

        print(f"[DEBUG] Model type: {type(_predict_module._model)}")
        print(f"[DEBUG] Scaler type: {type(_predict_module._scaler)}")

        signal, sr = load_and_clean(audio_path)

        if len(signal) == 0:
            raise ValueError("Empty audio signal")

        segment_samples = int(SEGMENT_DURATION * sr)
        overlap_samples = int(SEGMENT_OVERLAP * sr)
        step_samples = segment_samples - overlap_samples

        if len(signal) < segment_samples:
            print(f"[WARNING] Audio too short: {len(signal)} < {segment_samples} samples")
            return []

        text_scores = speech_result.get("text_scores", {})
        text_available = speech_result.get("text_available", False)

        segments = []
        start_sample = 0

        while start_sample + segment_samples <= len(signal):
            end_sample = start_sample + segment_samples
            segment_sig = signal[start_sample:end_sample]
            start_sec = round(start_sample / sr, 2)
            end_sec = round(end_sample / sr, 2)

            try:
                feat = extract_features(segment_sig, sr)

                if feat is None or len(feat) == 0:
                    print(f"[!] Сегмент {start_sec}-{end_sec} сек: пустые признаки")
                    start_sample += step_samples
                    continue

                feat_scaled = _predict_module._scaler.transform(feat.reshape(1, -1))
                probs_raw = _predict_module._model.predict_proba(feat_scaled)[0]

                audio_probs = {
                    EMOTION_LABELS[i]: round(float(probs_raw[i]), 4)
                    for i in range(len(EMOTION_LABELS))
                }

                fused_probs = fuse_audio_and_text(audio_probs, text_scores, text_available)

                dominant = max(fused_probs, key=fused_probs.get)
                confidence = fused_probs[dominant]

                valence = sum(p * EMOTION_AFFECT_MAP[e]["valence"] for e, p in fused_probs.items())
                arousal = sum(p * EMOTION_AFFECT_MAP[e]["arousal"] for e, p in fused_probs.items())

                segments.append({
                    "timestamp_start": start_sec,
                    "timestamp_end": end_sec,
                    "dominant_emotion": dominant,
                    "confidence": round(confidence, 4),
                    "probabilities": fused_probs,
                    "audio_probs": audio_probs,
                    "valence": round(float(valence), 4),
                    "arousal": round(float(arousal), 4),
                    "intensity": round(confidence, 4),
                })

            except Exception as e:
                print(f"[!] Сегмент {start_sec}-{end_sec} сек: {e}")
                import traceback
                traceback.print_exc()

            start_sample += step_samples

        print(f"[analysis] Проанализировано сегментов: {len(segments)}")
        return segments

    except Exception as e:
        print(f"[analysis] Ошибка в analyze_by_segments: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================
# СОХРАНЕНИЕ В БД
# ============================================================
def save_analysis_to_db(
    audio_record_id: int,
    segments: list[dict],
    features_dict: dict,
    processing_time: float,
) -> dict:
    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        ml_model_id = get_or_create_ml_model_id(conn)
        avg_confidence = round(
            float(np.mean([s["confidence"] for s in segments])), 4
        ) if segments else 0.0

        cur.execute("""
            INSERT INTO emotion_analysis
                (audio_record_id, ml_model_id, analysis_date, processing_time, confidence)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """, (audio_record_id, ml_model_id, datetime.now(), processing_time, avg_confidence))
        analysis_id = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO audio_features
                (emotion_analysis_id,
                 pitch, tempo, energy, spectral_centroid,
                 mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7,
                 mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, mfcc_13)
            VALUES (%s, %s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s)
        """, (
            analysis_id,
            features_dict["pitch"],          features_dict["tempo"],
            features_dict["energy"],         features_dict["spectral_centroid"],
            features_dict["mfcc_1"],  features_dict["mfcc_2"],  features_dict["mfcc_3"],
            features_dict["mfcc_4"],  features_dict["mfcc_5"],  features_dict["mfcc_6"],
            features_dict["mfcc_7"],  features_dict["mfcc_8"],  features_dict["mfcc_9"],
            features_dict["mfcc_10"], features_dict["mfcc_11"], features_dict["mfcc_12"],
            features_dict["mfcc_13"],
        ))

        emotion_result_ids = []
        for seg in segments:
            cur.execute("""
                INSERT INTO emotion_results
                    (emotion_analysis_id, emotion, intensity,
                     valence, arousal, timestamp_start, timestamp_end)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id
            """, (
                analysis_id,
                seg["dominant_emotion"], seg["intensity"],
                seg["valence"],          seg["arousal"],
                seg["timestamp_start"],  seg["timestamp_end"],
            ))
            emotion_result_ids.append(cur.fetchone()[0])

        conn.commit()
        print(f"[DB] analysis_id={analysis_id}, сегментов={len(segments)}")
        return {"analysis_id": analysis_id, "emotion_result_ids": emotion_result_ids}

    except Exception as e:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ============================================================
# СОХРАНЕНИЕ ОТЧЁТА
# ============================================================
def save_report_to_db(
    session_id: int,
    report_text: str,
    emotion_result_ids: list[int],
) -> int:
    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO reports (session_id, report_date, report_type, content, format)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """, (session_id, datetime.now(), "detailed", report_text, "pdf"))
        report_id = cur.fetchone()[0]

        for er_id in emotion_result_ids:
            cur.execute("""
                INSERT INTO report_emotion_results (report_id, emotion_result_id)
                VALUES (%s, %s)
            """, (report_id, er_id))

        conn.commit()
        print(f"[DB] report_id={report_id}")
        return report_id

    except Exception as e:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ — вызывается из FastAPI
# ============================================================
def run_full_analysis(
    audio_record_id: int,
    audio_path:      str,
    session_id:      int,
) -> dict:
    """
    Полный пайплайн:
      1. Очистка аудио
      2. Речевой анализ (Whisper STT + ключевые слова)
      3. Извлечение признаков для БД
      4. Сегментный анализ (аудио + текст)
      5. Маркеры риска
      6. Сохранение в БД
      7. Генерация отчёта
      8. Итоговый словарь для frontend

    Возвращает словарь из report_builder.build_frontend_result()
    """
    print(f"\n{'='*60}")
    print(f"[АНАЛИЗ] audio_record_id={audio_record_id}  файл={audio_path}")
    print(f"{'='*60}")
    start = time.time()

    # 1. Загрузка аудио (нужна для извлечения признаков)
    signal, sr = load_and_clean(audio_path)

    # 2. Речевой анализ (Whisper + ключевые слова)
    print("[1/6] Речевой анализ...")
    speech_result = analyze_speech(audio_path)

    # 3. Признаки для таблицы audio_features
    print("[2/6] Извлечение признаков...")
    features_dict = extract_features_for_db(signal, sr)

    # 4. Сегментный анализ
    print("[3/6] Сегментный анализ...")
    segments = analyze_by_segments(audio_path, speech_result)
    if not segments:
        raise ValueError("Файл слишком короткий или повреждён")

    # 5. Маркеры риска
    print("[4/6] Определение маркеров...")
    markers = detect_risk_markers(segments)

    processing_time = round(time.time() - start, 2)

    # 6. Сохраняем в БД
    print("[5/6] Сохранение в БД...")
    db_result = save_analysis_to_db(
        audio_record_id=audio_record_id,
        segments=segments,
        features_dict=features_dict,
        processing_time=processing_time,
    )

    # 7. Формируем итоговый словарь (содержит report_text)
    print("[6/6] Формирование отчёта...")
    frontend_result = build_frontend_result(
        segments=segments,
        speech_result=speech_result,
        markers=markers,
        audio_record_id=audio_record_id,
        session_id=session_id,
        analysis_id=db_result["analysis_id"],
        report_id=0,               # временно 0, обновим ниже
        processing_time=processing_time,
    )

    # Сохраняем отчёт в БД
    report_id = save_report_to_db(
        session_id=session_id,
        report_text=frontend_result["report_text"],
        emotion_result_ids=db_result["emotion_result_ids"],
    )
    frontend_result["meta"]["report_id"] = report_id
    frontend_result["meta"]["report_id"] = report_id

    print(f"\n[ГОТОВО] Время обработки: {processing_time} сек")
    print(f"  Паттерн: {frontend_result['pattern']['pattern_ru']}")
    print(f"  Маркеры: {len(markers)}")
    print(f"  Транскрипт: {'есть' if speech_result['text_available'] else 'нет (whisper не установлен)'}")

    return frontend_result