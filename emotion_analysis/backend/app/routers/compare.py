from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import date
import json

from .. import database, models, auth

router = APIRouter(prefix="/compare", tags=["Compare"])


# ── Даты сеансов пациента (для подсказок в datepicker) ──────────
@router.get("/patient-session-dates")
def get_patient_session_dates(
        patient_id: int,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist),
):
    """
    Возвращает список дат и id сеансов пациента у текущего психолога.
    Используется для подсказок при выборе дат сравнения.
    """
    sessions = (
        db.query(models.Session.id, models.Session.session_date, models.Session.notes)
        .filter(
            models.Session.patient_id == patient_id,
            models.Session.psychologist_id == current_psychologist.id,
        )
        .order_by(desc(models.Session.session_date))
        .all()
    )

    return [
        {
            "session_id": s.id,
            "date": s.session_date.isoformat(),
            "label": s.notes or f"Сеанс {s.session_date}",
        }
        for s in sessions
    ]


# ── Сравнение двух сеансов ──────────────────────────────────────
@router.get("/sessions")
def compare_sessions(
        session_id_1: int,
        session_id_2: int,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist),
):
    """
    Сравнивает два сеанса одного пациента:
    - агрегированные эмоции по каждому сеансу
    - дельта (изменение) по каждой эмоции
    - временные шкалы обоих сеансов для наложения на график
    """

    def _get_session_data(sid: int):
        session = db.query(models.Session).filter(
            models.Session.id == sid,
            models.Session.psychologist_id == current_psychologist.id,
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {sid} not found")

        config = session.analysis_config
        if not config or config.analysis_status != "completed":
            raise HTTPException(
                status_code=422,
                detail=f"Анализ сеанса {sid} ещё не завершён",
            )

        # Собираем все эмоциональные результаты через аудиозаписи
        audio_records = (
            db.query(models.AudioRecord)
            .filter(models.AudioRecord.session_id == sid)
            .all()
        )

        timeline = []
        emotion_totals: dict[str, float] = {}
        emotion_counts: dict[str, int] = {}

        for audio in audio_records:
            analysis = (
                db.query(models.EmotionAnalysis)
                .filter(models.EmotionAnalysis.audio_record_id == audio.id)
                .first()
            )
            if not analysis:
                continue

            results = (
                db.query(models.EmotionResult)
                .filter(models.EmotionResult.emotion_analysis_id == analysis.id)
                .order_by(models.EmotionResult.timestamp_start)
                .all()
            )

            for r in results:
                timeline.append({
                    "timestamp_start": r.timestamp_start,
                    "timestamp_end": r.timestamp_end,
                    "emotion": r.emotion,
                    "intensity": r.intensity,
                    "valence": r.valence,
                    "arousal": r.arousal,
                })
                emotion_totals[r.emotion] = emotion_totals.get(r.emotion, 0.0) + r.intensity
                emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1

        # Средние значения по эмоциям
        aggregated = {
            em: round(emotion_totals[em] / emotion_counts[em], 4)
            for em in emotion_totals
        }

        dominant = max(aggregated, key=aggregated.get) if aggregated else None

        return {
            "session_id": sid,
            "session_date": session.session_date.isoformat(),
            "patient_id": session.patient_id,
            "project_name": session.notes or f"Сеанс {session.session_date}",
            "aggregated_emotions": aggregated,
            "dominant_emotion": dominant,
            "timeline": timeline,
        }

    data1 = _get_session_data(session_id_1)
    data2 = _get_session_data(session_id_2)

    # Проверяем что это один пациент
    if data1["patient_id"] != data2["patient_id"]:
        raise HTTPException(
            status_code=400,
            detail="Сеансы принадлежат разным пациентам",
        )

    # Считаем дельту
    all_emotions = set(data1["aggregated_emotions"]) | set(data2["aggregated_emotions"])
    delta = {}
    for em in all_emotions:
        v1 = data1["aggregated_emotions"].get(em, 0.0)
        v2 = data2["aggregated_emotions"].get(em, 0.0)
        delta[em] = round(v2 - v1, 4)

    return {
        "session_1": data1,
        "session_2": data2,
        "delta": delta,                  # положительное = выросло, отрицательное = упало
        "patient_id": data1["patient_id"],
    }