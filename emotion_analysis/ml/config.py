

import os

# ----------------------------------------------------------
# ПУТИ
# ----------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESD_DIR        = os.path.join(BASE_DIR, "data", "RESD")
RESD_TRAIN_CSV  = os.path.join(RESD_DIR, "train.csv")
RESD_TEST_CSV   = os.path.join(RESD_DIR, "test.csv")
RESD_TRAIN_DIR  = os.path.join(RESD_DIR, "train")
RESD_TEST_DIR   = os.path.join(RESD_DIR, "test")
FEATURES_CSV = os.path.join(BASE_DIR, "data", "features.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "models", "emotion_model.h5")
MODEL_PATH_SVM = os.path.join(BASE_DIR, "models", "emotion_model_svm.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")
AUDIO_UPLOAD = os.path.join(BASE_DIR, "..", "audio_uploads")

os.makedirs(os.path.join(BASE_DIR, "data"),   exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(AUDIO_UPLOAD,                     exist_ok=True)

# ----------------------------------------------------------
# ЭМОЦИИ
# CREMA-D коды → названия для БД
# БД принимает: 'joy','sadness','anger','fear','disgust','neutral'
# ----------------------------------------------------------
CREMA_LABEL_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "joy",
    "NEU": "neutral",
    "SAD": "sadness",
}

# Порядок важен — индексы совпадают с выходом нейросети
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness"]

# Русские названия для отчёта
EMOTION_LABELS_RU = {
    "anger":   "Гнев",
    "disgust": "Отвращение",
    "fear":    "Страх",
    "joy":     "Радость",
    "neutral": "Нейтральное",
    "sadness": "Грусть",
}

# ----------------------------------------------------------
# ТАБЛИЦА ВАЛЕНТНОСТИ И ВОЗБУЖДЕНИЯ
# Валентность: -1 (негативная) ... +1 (позитивная)
# Возбуждение:  0 (низкое)    ...  1 (высокое)
# ----------------------------------------------------------
EMOTION_AFFECT_MAP = {
    "joy":     {"valence":  0.80, "arousal": 0.70},
    "neutral": {"valence":  0.00, "arousal": 0.30},
    "sadness": {"valence": -0.75, "arousal": 0.25},
    "anger":   {"valence": -0.70, "arousal": 0.85},
    "fear":    {"valence": -0.65, "arousal": 0.90},
    "disgust": {"valence": -0.60, "arousal": 0.60},
}

RESD_LABEL_MAP = {
    "anger":       "anger",
    "disgust":     "disgust",
    "fear":        "fear",
    "happiness":   "joy",
    "sadness":     "sadness",
    "neutral":     "neutral",
    "enthusiasm":  "joy",   # enthusiasm близко к joy
}

# ----------------------------------------------------------
# ПАРАМЕТРЫ LIBROSA
# ----------------------------------------------------------
SAMPLE_RATE = 22050
N_MFCC      = 40
HOP_LENGTH  = 512
N_FFT       = 2048

# ----------------------------------------------------------
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ----------------------------------------------------------
TEST_SIZE     = 0.20
RANDOM_SEED   = 42
EPOCHS      = 150
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

# ----------------------------------------------------------
# ПАРАМЕТРЫ СЕГМЕНТНОГО АНАЛИЗА (для длинных записей сессий)
# ----------------------------------------------------------
SEGMENT_DURATION = 5   # секунд на один отрезок
SEGMENT_OVERLAP  = 1    # секунд перекрытие