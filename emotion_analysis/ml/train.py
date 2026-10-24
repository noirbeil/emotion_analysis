

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from config import (
    FEATURES_CSV, MODEL_PATH, SCALER_PATH,
    EMOTION_LABELS, TEST_SIZE, RANDOM_SEED, BASE_DIR
)
import os


def load_data():
    print("Загружаем данные...")
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]

    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["emotion"].values
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["emotion"].values

    print(f"Train: {len(X_train)}, Test: {len(X_test)}, признаков: {X_train.shape[1]}")
    print(f"Классы: {np.unique(y_train)}")
    return X_train, X_test, y_train, y_test


def split_no_leakage(X, y, originals):

    unique_files = np.unique(originals)
    np.random.seed(RANDOM_SEED)

    # Определяем эмоцию каждого оригинального файла
    file_emotion = {}
    for orig in unique_files:
        mask = originals == orig
        file_emotion[orig] = y[mask][0]

    files = np.array(list(file_emotion.keys()))
    emotions = np.array([file_emotion[f] for f in files])

    from sklearn.model_selection import train_test_split as tts
    train_files, test_files = tts(
        files, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=emotions
    )

    train_set = set(train_files)
    test_set  = set(test_files)

    train_mask = np.array([o in train_set for o in originals])
    test_mask  = np.array([o in test_set  for o in originals])

    # В тест берём ТОЛЬКО оригиналы (aug0), не аугментированные
    orig_only = np.array([not f.endswith(tuple([f"_aug{i}" for i in range(1, 10)]))
                          for f in originals])
    # Упрощённая версия: файлы с _aug в тест не идут
    import re
    no_aug_mask = np.array([not re.search(r'_aug[1-9]', str(f)) for f in
                             pd.read_csv(FEATURES_CSV)["filename"].values])

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test  = X[test_mask & no_aug_mask]
    y_test  = y[test_mask & no_aug_mask]

    print(f"Train: {len(X_train)}, Test (только оригиналы): {len(X_test)}")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title("Матрица ошибок классификации")
    plt.ylabel("Истинная")
    plt.xlabel("Предсказанная")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Матрица ошибок: {path}")



def train():
    X_train, X_test, y_train, y_test = load_data()

    le = LabelEncoder()
    le.classes_ = np.array(EMOTION_LABELS)
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler сохранён: {SCALER_PATH}")

    print("\nОбучаем SVM...")
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_SEED,
        verbose=True
    )
    svm.fit(X_train_sc, y_train_enc)

    svm_path = MODEL_PATH.replace(".h5", "_svm.pkl")
    with open(svm_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"Модель сохранена: {svm_path}")

    y_pred = svm.predict(X_test_sc)
    acc = np.mean(y_pred == y_test_enc)
    print(f"\nТочность: {acc*100:.1f}%")
    print(classification_report(y_test_enc, y_pred, target_names=EMOTION_LABELS))

    plots_dir = os.path.join(BASE_DIR, "models")
    plot_confusion_matrix(y_test_enc, y_pred, plots_dir)
    return acc



if __name__ == "__main__":
    train()