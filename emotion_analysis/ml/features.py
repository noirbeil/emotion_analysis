import numpy as np
import librosa
from .config import SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT


def extract_features(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    features = []

    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    for m in [mfcc, mfcc_delta, mfcc_delta2]:
        features.extend(np.mean(m, axis=1).tolist())
        features.extend(np.std(m,  axis=1).tolist())

    chroma = librosa.feature.chroma_stft(
        y=signal, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.extend(np.mean(chroma, axis=1).tolist())
    features.extend(np.std(chroma,  axis=1).tolist())

    sc = librosa.feature.spectral_centroid(
        y=signal, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.append(float(np.mean(sc)))
    features.append(float(np.std(sc)))

    bw = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.append(float(np.mean(bw)))
    features.append(float(np.std(bw)))

    contrast = librosa.feature.spectral_contrast(
        y=signal, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.extend(np.mean(contrast, axis=1).tolist())
    features.extend(np.std(contrast,  axis=1).tolist())

    rolloff = librosa.feature.spectral_rolloff(
        y=signal, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.append(float(np.mean(rolloff)))
    features.append(float(np.std(rolloff)))

    rms = librosa.feature.rms(y=signal, hop_length=HOP_LENGTH)
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=HOP_LENGTH)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr, hop_length=HOP_LENGTH)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)
    features.append(tempo)

    result = np.array(features, dtype=np.float32)
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result



def extract_features_for_db(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            signal, fmin=50, fmax=500, sr=sr, hop_length=HOP_LENGTH
        )
        f0_voiced = f0[voiced_flag]
        pitch = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    except Exception:
        pitch = 0.0

    try:
        tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
        tempo = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
    except Exception:
        tempo = 0.0

    rms = librosa.feature.rms(y=signal, hop_length=HOP_LENGTH)
    energy = float(np.mean(rms))
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr, hop_length=HOP_LENGTH)))
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc_means = np.mean(mfcc, axis=1)

    return {
        "pitch": round(pitch, 4),
        "tempo": round(tempo, 4),
        "energy": round(energy, 6),
        "spectral_centroid": round(spec_centroid, 4),
        **{f"mfcc_{i+1}": round(float(mfcc_means[i]), 4) for i in range(13)},
    }