# =============================================================
#  ФАЙЛ: ml/report_builder.py
#  ЧТО ДЕЛАЕТ:
#    Собирает итоговый словарь результата анализа для frontend.
#    Frontend получает этот словарь и рисует:
#      - временной график эмоций по сегментам
#      - круговую диаграмму распределения
#      - маркеры риска
#      - текст расшифровки и найденные слова
#      - сводную таблицу отчёта
# =============================================================

import numpy as np
from datetime import datetime
from .config import EMOTION_LABELS, EMOTION_LABELS_RU, EMOTION_AFFECT_MAP, SEGMENT_DURATION


# ==============================================================
# ПРИБЛИЗИТЕЛЬНОЕ ПСИХОЛОГИЧЕСКОЕ СОСТОЯНИЕ
# Это НЕ диагноз — описательный паттерн для психолога
# ==============================================================
def infer_emotional_pattern(
    avg_valence: float,
    avg_arousal: float,
    dominant_counts: dict[str, int],
    total_segments: int,
) -> dict:
    """
    На основе валентности и возбуждения определяет общий
    наблюдаемый эмоциональный паттерн сессии.

    Использует модель Рассела (четыре квадранта):
      высокое возбуждение + позитивная валентность → активированное позитивное
      высокое возбуждение + негативная валентность → стресс/тревога/гнев
      низкое возбуждение  + негативная валентность → подавленное состояние
      низкое возбуждение  + позитивная валентность → расслабленное/спокойное

    Возвращает словарь с:
      pattern_key   — короткий идентификатор для frontend
      pattern_ru    — русское название паттерна
      description   — описание для психолога
      recommendation — общая рекомендация (не лечение!)
    """
    # Квадранты по валентности и возбуждению
    positive = avg_valence > 0.15
    high_arousal = avg_arousal > 0.55

    # Дополнительно смотрим что преобладает
    negative_emotions = {"sadness", "anger", "fear", "disgust"}
    neg_ratio = sum(dominant_counts.get(e, 0) for e in negative_emotions) / max(total_segments, 1)

    if positive and high_arousal:
        return {
            "pattern_key": "activated_positive",
            "pattern_ru":  "Активированное позитивное состояние",
            "description": (
                "Высокая эмоциональная активность с преобладанием позитивных эмоций. "
                "Клиент вовлечён, энергичен, открыт к взаимодействию."
            ),
            "recommendation": (
                "Благоприятное состояние для продуктивной работы. "
                "Рекомендуется закрепить позитивные паттерны."
            ),
        }
    elif not positive and high_arousal:
        if dominant_counts.get("anger", 0) > dominant_counts.get("fear", 0):
            return {
                "pattern_key": "stress_anger",
                "pattern_ru":  "Напряжение и агрессивность",
                "description": (
                    "Высокий уровень активации с преобладанием негативных эмоций: "
                    "гнев, раздражение, напряжение. Возможны конфликтные переживания."
                ),
                "recommendation": (
                    "Рекомендуется работа с агрессией и напряжением. "
                    "Техники телесного расслабления и когнитивного переструктурирования."
                ),
            }
        else:
            return {
                "pattern_key": "anxiety",
                "pattern_ru":  "Тревожно-возбуждённое состояние",
                "description": (
                    "Высокая активация с выраженной тревогой или страхом. "
                    "Возможны признаки ситуативной или хронической тревожности."
                ),
                "recommendation": (
                    "Рекомендуется работа со страхами и тревогой. "
                    "Техники заземления, дыхательные практики."
                ),
            }
    elif not positive and not high_arousal:
        return {
            "pattern_key": "depressive",
            "pattern_ru":  "Подавленное состояние",
            "description": (
                "Низкая активация с преобладанием негативных эмоций: "
                "грусть, апатия, подавленность. Возможны признаки депрессивного эпизода."
            ),
            "recommendation": (
                "Рекомендуется оценка депрессивной симптоматики. "
                "Техники активации поведения, поиск ресурсов."
            ),
        }
    else:
        return {
            "pattern_key": "calm_positive",
            "pattern_ru":  "Спокойное / ресурсное состояние",
            "description": (
                "Умеренная или низкая активация с позитивным или нейтральным фоном. "
                "Состояние устойчивости и относительного благополучия."
            ),
            "recommendation": (
                "Стабильное состояние. Рекомендуется поддерживающая работа."
            ),
        }


# ==============================================================
# ДАННЫЕ ДЛЯ ВРЕМЕННОГО ГРАФИКА (frontend)
# ==============================================================
def build_timeline_chart_data(segments: list[dict]) -> list[dict]:
    """
    Преобразует сегменты в формат для отрисовки временного графика во frontend.

    Каждый элемент — одна точка на графике:
    {
        x:            14.0,             ← середина сегмента в секундах
        label:        "0:14",           ← метка времени для оси X
        dominant:     "sadness",
        dominant_ru:  "Грусть",
        confidence:   0.72,
        valence:      -0.75,
        arousal:      0.25,
        joy:    0.05, sadness: 0.72, anger: 0.08, ...   ← все вероятности
    }
    """
    chart_data = []
    for seg in segments:
        mid = round((seg["timestamp_start"] + seg["timestamp_end"]) / 2, 2)
        total_sec = int(mid)
        label = f"{total_sec // 60}:{total_sec % 60:02d}"

        point = {
            "x":           mid,
            "label":       label,
            "dominant":    seg["dominant_emotion"],
            "dominant_ru": EMOTION_LABELS_RU.get(seg["dominant_emotion"],
                                                  seg["dominant_emotion"]),
            "confidence":  seg["confidence"],
            "intensity":   seg["intensity"],
            "valence":     seg["valence"],
            "arousal":     seg["arousal"],
        }
        # Добавляем все вероятности как отдельные поля — recharts умеет их читать
        for emotion, prob in seg.get("probabilities", {}).items():
            point[emotion] = prob

        chart_data.append(point)

    return chart_data


# ==============================================================
# ИТОГОВЫЙ СЛОВАРЬ ДЛЯ FRONTEND
# ==============================================================
def build_frontend_result(
    segments:      list[dict],
    speech_result: dict,
    markers:       list[str],
    audio_record_id: int,
    session_id:    int,
    analysis_id:   int,
    report_id:     int,
    processing_time: float,
) -> dict:
    """
    Собирает ОДИН итоговый словарь, который backend отдаёт во frontend.
    Frontend использует его чтобы:
      - нарисовать временной график (emotion_timeline)
      - нарисовать круговую диаграмму (emotion_summary)
      - отобразить маркеры (risk_markers)
      - показать транскрипт и ключевые слова
      - показать паттерн и рекомендацию
      - сформировать PDF-отчёт

    СТРУКТУРА которую ждёт React:
    {
      "meta": {...},
      "overall": {...},
      "emotion_timeline": [...],    ← временной график
      "emotion_summary":  {...},    ← круговая диаграмма
      "speech": {...},              ← текстовый анализ
      "risk_markers": [...],
      "pattern": {...},
      "report_text": "...",
    }
    """
    if not segments:
        return {"status": "error", "message": "Нет сегментов для анализа"}

    # --- Агрегация по сегментам ---
    dominant_counts: dict[str, int] = {}
    for seg in segments:
        e = seg["dominant_emotion"]
        dominant_counts[e] = dominant_counts.get(e, 0) + 1

    total_segments  = len(segments)
    overall_dominant = max(dominant_counts, key=dominant_counts.get)
    avg_valence = float(np.mean([s["valence"] for s in segments]))
    avg_arousal = float(np.mean([s["arousal"] for s in segments]))
    avg_confidence = float(np.mean([s["confidence"] for s in segments]))

    # Средние вероятности по каждой эмоции (для круговой диаграммы)
    emotion_summary = {}
    for emotion in EMOTION_LABELS:
        avg_prob = float(np.mean([
            s.get("probabilities", {}).get(emotion, 0.0)
            for s in segments
        ]))
        emotion_summary[emotion] = round(avg_prob, 4)

    # --- Нормализуем summary до 1.0 ---
    total_prob = sum(emotion_summary.values())
    if total_prob > 0:
        emotion_summary = {e: round(v / total_prob, 4) for e, v in emotion_summary.items()}

    # --- Данные временного графика ---
    timeline = build_timeline_chart_data(segments)

    # --- Паттерн ---
    pattern = infer_emotional_pattern(
        avg_valence, avg_arousal, dominant_counts, total_segments
    )

    # --- Текстовый анализ ---
    speech = {
        "transcript":     speech_result.get("transcript", ""),
        "whisper_enabled": speech_result.get("whisper_enabled", False),
        "text_available": speech_result.get("text_available", False),
        "found_keywords": speech_result.get("found_keywords", []),
        "keyword_hits":   speech_result.get("keyword_hits", {}),
    }

    # --- Текст отчёта ---
    report_text = _build_report_text(
        segments, dominant_counts, avg_valence, avg_arousal,
        avg_confidence, markers, pattern, speech_result
    )

    # --- Итоговый словарь ---
    result = {
        "status": "ok",

        # Мета-информация
        "meta": {
            "analysis_id":    analysis_id,
            "report_id":      report_id,
            "session_id":     session_id,
            "audio_record_id": audio_record_id,
            "analysis_date":  datetime.now().isoformat(),
            "processing_time": processing_time,
            "total_segments": total_segments,
            "duration_sec":   round(segments[-1]["timestamp_end"], 1),
        },

        # Общий итог сессии
        "overall": {
            "dominant_emotion":    overall_dominant,
            "dominant_emotion_ru": EMOTION_LABELS_RU.get(overall_dominant, overall_dominant),
            "avg_confidence":      round(avg_confidence, 4),
            "avg_valence":         round(avg_valence, 4),
            "avg_arousal":         round(avg_arousal, 4),
            # Интерпретация для интерфейса
            "valence_label": (
                "Позитивный фон" if avg_valence > 0.15
                else "Негативный фон" if avg_valence < -0.15
                else "Нейтральный фон"
            ),
            "arousal_label": (
                "Высокое возбуждение" if avg_arousal > 0.6
                else "Умеренная активность" if avg_arousal > 0.3
                else "Низкая активность"
            ),
        },

        # Для временного графика во frontend (ось X = время, ось Y = вероятность)
        # Передаём все сегменты + рассчитанные точки
        "emotion_timeline": timeline,

        # Для круговой диаграммы: {emotion → доля}
        "emotion_summary": emotion_summary,

        # Текстовый анализ (Whisper + ключевые слова)
        "speech": speech,

        # Маркеры риска
        "risk_markers": markers,

        # Паттерн и рекомендация
        "pattern": pattern,

        # Готовый текст отчёта для сохранения в reports.content
        "report_text": report_text,
    }

    return result


# ==============================================================
# ТЕКСТ ОТЧЁТА
# ==============================================================
def _build_report_text(
    segments, dominant_counts, avg_valence, avg_arousal,
    avg_confidence, markers, pattern, speech_result
) -> str:
    """
    Генерирует полный текст отчёта на русском языке.
    Сохраняется в таблицу reports.content.
    """
    total_seg  = len(segments)
    duration   = segments[-1]["timestamp_end"] if segments else 0
    dom_sorted = sorted(dominant_counts.items(), key=lambda x: x[1], reverse=True)

    lines = [
        "ОТЧЁТ ПО АНАЛИЗУ ЭМОЦИОНАЛЬНОЙ ОКРАСКИ ГОЛОСА",
        "=" * 60,
        f"Дата:                {datetime.now().strftime('%d.%m.%Y  %H:%M')}",
        f"Длительность:        {int(duration // 60)} мин {int(duration % 60)} сек",
        f"Сегментов:           {total_seg}",
        f"Уверенность модели:  {avg_confidence * 100:.1f}%",
        "",
        "РАСПРЕДЕЛЕНИЕ ЭМОЦИЙ",
        "-" * 40,
    ]
    for emotion, count in dom_sorted:
        pct = count / total_seg * 100
        ru = EMOTION_LABELS_RU.get(emotion, emotion)
        lines.append(f"  {ru:<20}: {pct:.1f}%")

    lines += [
        "",
        "АФФЕКТИВНЫЕ ПОКАЗАТЕЛИ",
        "-" * 40,
        f"  Валентность:  {avg_valence:+.2f}  ({pattern['pattern_ru']})",
        f"  Возбуждение:  {avg_arousal:.2f}",
        "",
        "НАБЛЮДАЕМЫЙ ПАТТЕРН",
        "-" * 40,
        f"  {pattern['pattern_ru']}",
        f"  {pattern['description']}",
        "",
        "РЕКОМЕНДАЦИИ",
        "-" * 40,
        f"  {pattern['recommendation']}",
        "",
        "МАРКЕРЫ ЭМОЦИОНАЛЬНОГО СОСТОЯНИЯ",
        "-" * 40,
    ]
    for m in markers:
        lines.append(f"  • {m}")

    # Блок текстового анализа (если Whisper работал)
    if speech_result.get("text_available"):
        lines += [
            "",
            "АНАЛИЗ РЕЧЕВОГО СОДЕРЖАНИЯ (Whisper STT)",
            "-" * 40,
            f"  Транскрипт: «{speech_result['transcript'][:300]}»",
        ]
        kw_found = speech_result.get("found_keywords", [])
        if kw_found:
            lines.append(f"  Найдено эмоциональных маркеров: {len(kw_found)}")
            by_emotion: dict[str, list] = {}
            for item in kw_found:
                by_emotion.setdefault(item["emotion_ru"], []).append(item["word"])
            for emo_ru, words in by_emotion.items():
                lines.append(f"    {emo_ru}: {', '.join(words[:5])}")
        else:
            lines.append("  Эмоциональных слов-маркеров не обнаружено.")
    else:
        lines += [
            "",
            "АНАЛИЗ РЕЧЕВОГО СОДЕРЖАНИЯ",
            "-" * 40,
            "  Расшифровка речи недоступна (whisper не установлен).",
            "  Эмоции определены только по акустическим признакам.",
        ]

    lines += [
        "",
        "=" * 60,
        "ПРИМЕЧАНИЕ",
        "Данный отчёт является вспомогательным инструментом для специалиста.",
        "Результаты анализа не заменяют профессиональную психологическую оценку",
        "и не являются медицинским диагнозом.",
    ]

    return "\n".join(lines)