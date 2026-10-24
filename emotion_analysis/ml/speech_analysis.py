# =============================================================
#  ФАЙЛ: ml/speech_analysis.py
#  ЧТО ДЕЛАЕТ:
#    1. Расшифровывает аудио в текст через Whisper
#    2. Ищет слова-триггеры по каждой эмоции
#    3. Возвращает текстовую оценку эмоций
#
#  УСТАНОВКА: pip install openai-whisper
#  (Whisper работает локально, без интернета и без API-ключа)
# =============================================================

import re
import numpy as np
import os

# Пробуем загрузить Whisper — если не установлен, STT будет отключён
try:
    import whisper as _whisper_lib
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[speech_analysis] whisper не установлен. "
          "pip install openai-whisper  →  STT будет отключён, "
          "работаем только на аудио-модели.")

from .config import EMOTION_LABELS, EMOTION_LABELS_RU


# ------------------------------------------------------------------
# СЛОВАРИ СЛОВ-ТРИГГЕРОВ
# Ключи совпадают с EMOTION_LABELS из config.py
# Слова — нижний регистр, без знаков препинания
# Добавляй свои слова — чем больше, тем точнее
# ------------------------------------------------------------------
EMOTION_KEYWORDS: dict[str, list[str]] = {

    "joy": [
        # позитивные состояния
        "счастл", "рад", "рада", "весел", "радост", "отлично", "замечательно",
        "прекрасно", "хорошо", "люблю", "восторг", "восхитительно", "улыбаюсь",
        "смешно", "здорово", "класс", "ура", "отдыхаю", "улучшилось",
        "доволен", "довольна", "приятно", "нравится", "хочу", "жду",
        "с удовольствием", "интересно", "воодушевлён", "влюблён",
    ],

    "sadness": [
        # депрессивные, упадочные состояния
        "грустно", "грусть", "печально", "печаль", "плачу", "слёзы", "тоска",
        "одиноко", "одиночество", "скучаю", "скучно", "устал", "устала",
        "бессмысленно", "безнадёжно", "депрессия", "подавлен", "подавлена",
        "никому не нужен", "никому не нужна", "нет сил", "не хочу",
        "не могу", "тяжело", "больно", "горе", "потерял", "потеряла",
        "ушёл", "ушла", "умер", "умерла", "конец", "всё плохо",
        "ничего хорошего", "нет смысла", "зачем", "не вижу смысла",
    ],

    "anger": [
        # агрессия, раздражение, конфликт
        "злюсь", "злость", "бешенство", "ненавижу", "раздражает",
        "раздражён", "раздражена", "достали", "идиот", "идиоты",
        "несправедливо", "виноват", "виновата", "не могу терпеть",
        "взбесило", "взбесила", "хватит", "надоело", "бесит",
        "всё равно не слушают", "игнорируют", "кричит", "орёт",
        "угрожает", "агрессия", "конфликт", "скандал", "обидел",
        "обидела", "унизил", "унизила",
    ],

    "fear": [
        # тревога, страх, паника
        "боюсь", "страшно", "страх", "тревога", "тревожно", "паника",
        "паникую", "не могу успокоиться", "сердце колотится", "дрожу",
        "опасность", "угроза", "что будет", "а вдруг", "катастрофа",
        "не справлюсь", "потеряю", "умру", "ужасно", "кошмар",
        "невозможно", "не знаю что делать", "беспокоюсь", "волнуюсь",
        "не сплю", "бессонница", "тревожные мысли", "навязчивые мысли",
    ],

    "disgust": [
        # отвращение, брезгливость, неприятие
        "отвратительно", "мерзко", "противно", "гадость", "гадко",
        "тошнит", "брезгую", "не могу смотреть", "ужасный человек",
        "не хочу", "не буду", "низко", "недостойно", "презираю",
        "омерзительно", "стыдно за них",
    ],

    "neutral": [
        # спокойные, нейтральные маркеры — без эмоциональной окраски
        # (используем как «балласт», если ничего не нашли)
        "обычно", "нормально", "как всегда", "в целом", "в принципе",
        "ничего особенного", "так же как", "стандартно",
    ],
}

# Весовой коэффициент для текстовой оценки (0.0 – 1.0)
# 0.3 означает: текст даёт 30% итогового результата, аудио — 70%
TEXT_WEIGHT = 0.30
AUDIO_WEIGHT = 1.0 - TEXT_WEIGHT

# Минимальная длина слова для поиска (чтобы не ловить предлоги)
MIN_KEYWORD_LEN = 4


# ==============================================================
# ЗАГРУЗКА МОДЕЛИ WHISPER
# ==============================================================
_whisper_model = None


def _load_whisper(model_size: str = "small"):
    """
    Загружает модель Whisper один раз и кэширует.

    Размеры моделей (точность vs скорость):
      tiny   — очень быстро, точность ~60%
      base   — быстро, точность ~70%
      small  — умеренно, точность ~80%  ← рекомендуем для диплома
      medium — медленно, точность ~85%
      large  — очень медленно, точность ~90%
    """
    global _whisper_model
    if _whisper_model is None and WHISPER_AVAILABLE:
        print(f"[Whisper] Загружаем модель '{model_size}'...")
        _whisper_model = _whisper_lib.load_model(model_size)
        print("[Whisper] Модель загружена.")
    return _whisper_model


# ==============================================================
# РАСШИФРОВКА АУДИО В ТЕКСТ
# ==============================================================
def transcribe_audio(audio_path: str, model_size: str = "small") -> str:
    """
    Расшифровывает аудиофайл в текст через Whisper.
    """
    if not WHISPER_AVAILABLE:
        return ""

    # Проверяем существование файла
    if not os.path.exists(audio_path):
        print(f"[Whisper] Файл не найден: {audio_path}")
        return ""

    try:
        model = _load_whisper(model_size)
        # Используем абсолютный путь к файлу
        abs_path = os.path.abspath(audio_path)
        print(f"[Whisper] Расшифровка файла: {abs_path}")

        result = model.transcribe(abs_path, language="ru", fp16=False)
        text = result.get("text", "").strip()
        print(f"[Whisper] Расшифровано: «{text[:120]}{'...' if len(text) > 120 else ''}»")
        return text
    except Exception as e:
        print(f"[Whisper] Ошибка расшифровки: {e}")
        import traceback
        traceback.print_exc()
        return ""


# ==============================================================
# ПОИСК СЛОВ-ТРИГГЕРОВ В ТЕКСТЕ
# ==============================================================
def find_emotion_keywords(text: str) -> dict:
    """
    Ищет слова-триггеры каждой эмоции в расшифрованном тексте.

    Использует частичное совпадение (prefix-match): слово «боюсь» совпадёт
    с триггером «боюсь» и с производными «боятся», «боится» и т.д.

    Возвращает словарь:
    {
      "anger":   {"count": 2, "found_words": ["злюсь", "ненавижу"]},
      "sadness": {"count": 0, "found_words": []},
      ...
    }
    """
    if not text:
        return {emotion: {"count": 0, "found_words": []} for emotion in EMOTION_LABELS}

    # Нормализуем текст: нижний регистр, убираем знаки препинания
    clean_text = text.lower()
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    words_in_text = clean_text.split()

    result = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        found = []
        for kw in keywords:
            if len(kw) < MIN_KEYWORD_LEN:
                continue
            # Ищем триггер как подстроку в каждом слове текста
            for word in words_in_text:
                if kw in word and word not in found:
                    found.append(word)
        result[emotion] = {
            "count":       len(found),
            "found_words": found[:10],  # максимум 10 для отчёта
        }

    return result


# ==============================================================
# ТЕКСТОВАЯ ОЦЕНКА ЭМОЦИЙ (вероятности по тексту)
# ==============================================================
def compute_text_emotion_scores(keyword_hits: dict) -> dict[str, float]:
    """
    Переводит счётчики ключевых слов в вероятности (сумма = 1.0).

    Если ни одного слова не найдено — возвращает равномерное распределение
    (текстовый канал не даёт информации, решение принимает только аудио).

    Возвращает: {"anger": 0.12, "joy": 0.05, ...}
    """
    counts = {emotion: data["count"] for emotion, data in keyword_hits.items()}
    total = sum(counts.values())

    if total == 0:
        # Нет ключевых слов — равномерное распределение (нейтральный вклад)
        n = len(EMOTION_LABELS)
        return {emotion: round(1.0 / n, 4) for emotion in EMOTION_LABELS}

    return {
        emotion: round(counts[emotion] / total, 4)
        for emotion in EMOTION_LABELS
    }


# ==============================================================
# ОБЪЕДИНЕНИЕ АУДИО + ТЕКСТ
# ==============================================================
def fuse_audio_and_text(
    audio_probs: dict[str, float],
    text_scores: dict[str, float],
    text_available: bool,
) -> dict[str, float]:
    """
    Взвешенно объединяет вероятности от аудио-модели и текстового анализа.

    Если text_available=False (Whisper не установлен или текст пустой) —
    возвращает только аудио-вероятности без изменений.

    Формула: итог = AUDIO_WEIGHT × audio + TEXT_WEIGHT × text

    Возвращает: итоговые вероятности (сумма = 1.0)
    """
    if not text_available:
        return audio_probs

    fused = {}
    for emotion in EMOTION_LABELS:
        a = audio_probs.get(emotion, 0.0)
        t = text_scores.get(emotion, 0.0)
        fused[emotion] = round(AUDIO_WEIGHT * a + TEXT_WEIGHT * t, 4)

    # Нормализуем чтобы сумма была ровно 1.0
    total = sum(fused.values())
    if total > 0:
        fused = {e: round(v / total, 4) for e, v in fused.items()}

    return fused


# ==============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==============================================================
def analyze_speech(audio_path: str) -> dict:
    """
    Полный текстовый анализ одного аудиофайла:
      1. Whisper → текст
      2. Поиск ключевых слов
      3. Текстовые вероятности эмоций

    Возвращает словарь, который передаётся в run_full_analysis():
    {
        "transcript":       "текст расшифровки",
        "keyword_hits":     {emotion: {count, found_words}, ...},
        "text_scores":      {emotion: float, ...},
        "text_available":   True / False,
        "whisper_enabled":  True / False,
    }
    """
    transcript = transcribe_audio(audio_path)
    text_available = bool(transcript)

    keyword_hits = find_emotion_keywords(transcript)
    text_scores  = compute_text_emotion_scores(keyword_hits)

    # Собираем найденные слова в один список для отчёта
    all_found = []
    for emotion, data in keyword_hits.items():
        for word in data["found_words"]:
            all_found.append({
                "word":    word,
                "emotion": emotion,
                "emotion_ru": EMOTION_LABELS_RU.get(emotion, emotion),
            })

    return {
        "transcript":      transcript,
        "keyword_hits":    keyword_hits,
        "text_scores":     text_scores,
        "found_keywords":  all_found,    # плоский список для отчёта
        "text_available":  text_available,
        "whisper_enabled": WHISPER_AVAILABLE,
    }


# ==============================================================
# БЫСТРАЯ ПРОВЕРКА
# ==============================================================
if __name__ == "__main__":
    import sys, os
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Простая проверка без файла — только keyword-поиск
        print("=== Тест поиска ключевых слов ===")
        test_text = "Мне очень грустно и одиноко, я боюсь что всё плохо и нет смысла"
        hits   = find_emotion_keywords(test_text)
        scores = compute_text_emotion_scores(hits)
        for em, data in hits.items():
            if data["count"] > 0:
                print(f"  {EMOTION_LABELS_RU[em]:15s}: {data['count']} слов → {data['found_words']}")
        print("\nТекстовые вероятности:")
        for em, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {EMOTION_LABELS_RU[em]:15s}: {score:.2%}")
        sys.exit(0)

    result = analyze_speech(path)
    print(f"\nТранскрипт: {result['transcript']}")
    print(f"\nНайдено ключевых слов: {len(result['found_keywords'])}")
    for item in result["found_keywords"]:
        print(f"  «{item['word']}» → {item['emotion_ru']}")