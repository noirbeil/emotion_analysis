import numpy as np
import librosa
import noisereduce as nr
from .config import SAMPLE_RATE


def load_and_clean(file_path: str) -> tuple[np.ndarray, int]:


    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    signal_trimmed, _ = librosa.effects.trim(signal, top_db=20)

    noise_sample = signal_trimmed[:sr // 2]  # первые 0.5 сек
    signal_denoised = nr.reduce_noise(
        y=signal_trimmed,
        y_noise=noise_sample,
        sr=sr,
        stationary=False,   # нестационарный шум (фон может меняться)
        prop_decrease=0.8   # убираем 80% шума (не 100%, чтобы не исказить голос)
    )

    max_val = np.max(np.abs(signal_denoised))
    if max_val > 0:
        signal_normalized = signal_denoised / max_val
    else:
        signal_normalized = signal_denoised

    return signal_normalized, sr


def get_audio_duration(file_path: str) -> float:

    duration = librosa.get_duration(path=file_path)
    return round(duration, 2)