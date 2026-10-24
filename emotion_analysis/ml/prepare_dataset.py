

import os
import re
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

from config import (
    FEATURES_CSV,
    RESD_TRAIN_CSV, RESD_TEST_CSV,
    RESD_TRAIN_DIR, RESD_TEST_DIR,
    RESD_LABEL_MAP
)
from preprocess import load_and_clean
from features import extract_features


def augment(signal: np.ndarray, sr: int):
    """Аугментация — только для train."""
    variants = [signal]
    try:
        variants.append(librosa.effects.pitch_shift(signal, sr=sr, n_steps=2))
        variants.append(librosa.effects.pitch_shift(signal, sr=sr, n_steps=-2))
    except Exception:
        pass
    try:
        variants.append(librosa.effects.time_stretch(signal, rate=0.9))
        variants.append(librosa.effects.time_stretch(signal, rate=1.1))
    except Exception:
        pass
    noise = np.random.normal(0, 0.005, signal.shape)
    variants.append(signal + noise)
    return variants


def process_split(csv_path: str, audio_dir: str, use_augmentation: bool) -> list:

    df = pd.read_csv(csv_path)
    rows = []
    errors = 0

    print(f"\nОбрабатываем {csv_path} ({len(df)} записей)...")
    print(f"Уникальные эмоции в CSV: {df['emotion'].unique()}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(csv_path)):
        # Маппинг эмоции
        raw_emotion = str(row["emotion"]).strip().lower()
        emotion = RESD_LABEL_MAP.get(raw_emotion)
        if emotion is None:
            continue  # пропускаем неизвестные эмоции

        # Путь к файлу: audio_dir / path из CSV
        rel_path = str(row["path"]).strip()
        file_path = os.path.join(audio_dir, rel_path)

        if not os.path.exists(file_path):
            errors += 1
            print(f"\n[!] Файл не найден: {file_path}")
            continue

        try:
            signal, sr = load_and_clean(file_path)
            if len(signal) < sr * 0.5:
                continue

            filename_base = str(row["name"]).strip()

            if use_augmentation:
                variants = augment(signal, sr)
            else:
                variants = [signal]  # тест — только оригинал

            for i, aug_signal in enumerate(variants):
                feat = extract_features(aug_signal, sr)
                feat_row = {f"feat_{j}": feat[j] for j in range(len(feat))}
                feat_row["emotion"] = emotion
                # aug0 = оригинал, aug1+ = аугментации
                feat_row["filename"] = f"{filename_base}_aug{i}" if i > 0 else filename_base
                feat_row["split"] = "train" if use_augmentation else "test"
                rows.append(feat_row)

        except Exception as e:
            errors += 1
            print(f"\n[!] Ошибка {rel_path}: {e}")

    print(f"Готово. Строк: {len(rows)}, ошибок: {errors}")
    return rows


def prepare_dataset():
    all_rows = []

    # Train — с аугментацией
    train_rows = process_split(
        csv_path=RESD_TRAIN_CSV,
        audio_dir=RESD_TRAIN_DIR,
        use_augmentation=True
    )
    all_rows.extend(train_rows)

    # Test — только оригиналы, без аугментации
    test_rows = process_split(
        csv_path=RESD_TEST_CSV,
        audio_dir=RESD_TEST_DIR,
        use_augmentation=False
    )
    all_rows.extend(test_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(FEATURES_CSV, index=False)

    print(f"\n{'='*50}")
    print(f"Итого примеров: {len(df)}")
    print(f"  train: {len(df[df['split']=='train'])}")
    print(f"  test:  {len(df[df['split']=='test'])}")
    print(f"\nРаспределение эмоций:")
    print(df["emotion"].value_counts())
    print(f"\nСохранено в: {FEATURES_CSV}")


if __name__ == "__main__":
    prepare_dataset()