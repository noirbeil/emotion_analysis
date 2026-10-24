import os
import subprocess
import librosa
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from datetime import datetime
from .. import database, models, schemas, auth
from datetime import datetime, timezone

router = APIRouter(prefix="/audio", tags=["Audio"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def convert_to_wav(input_path: str) -> str:
    """
    Перекодирует любой аудиофайл в WAV (22050 Hz, mono) через ffmpeg.
    Возвращает путь к новому файлу. Оригинал удаляется.
    """
    output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-ar', '22050', '-ac', '1', output_path],
            capture_output=True,
            timeout=120
        )
        if result.returncode == 0 and os.path.exists(output_path):
            os.remove(input_path)
            print(f"[ffmpeg] Конвертация успешна: {output_path}")
            return output_path
        else:
            print(f"[ffmpeg] Ошибка конвертации: {result.stderr.decode('utf-8', errors='ignore')}")
            return input_path
    except FileNotFoundError:
        print("[ffmpeg] ffmpeg не найден в PATH. Конвертация пропущена.")
        return input_path
    except subprocess.TimeoutExpired:
        print("[ffmpeg] Таймаут конвертации.")
        return input_path
    except Exception as e:
        print(f"[ffmpeg] Неожиданная ошибка: {e}")
        return input_path


@router.post("/upload", response_model=schemas.AudioUploadResponse)
async def upload_audio(
        session_id: int = Form(...),
        file: UploadFile = File(...),
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    # Проверка, что сессия принадлежит психологу
    session = db.query(models.Session).filter(
        models.Session.id == session_id,
        models.Session.psychologist_id == current_psychologist.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Определение формата файла
    file_extension = os.path.splitext(file.filename)[1].lower().replace('.', '')
    allowed_formats = ['wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm']

    if file_extension not in allowed_formats:
        raise HTTPException(status_code=400, detail=f"Invalid format. Allowed: {allowed_formats}")

    # Сохранение оригинального файла
    safe_filename = f"session_{session_id}_{int(datetime.now(timezone.utc).timestamp())}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Конвертируем в WAV если это не WAV
    # (браузерный MediaRecorder часто пишет WebM/Opus даже с расширением .wav)
    if file_extension != 'wav':
        file_path = convert_to_wav(file_path)
    else:
        # Даже .wav из браузера может быть WebM внутри — перекодируем на всякий случай
        file_path = convert_to_wav(file_path)

    final_extension = os.path.splitext(file_path)[1].lower().replace('.', '')

    # Извлечение аудио-параметров с помощью librosa
    duration = None
    sample_rate = 22050

    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        sample_rate = sr
        print(f"[librosa] Длительность: {duration:.2f}s, Sample rate: {sr}")
    except Exception as e:
        print(f"[librosa] Не удалось извлечь параметры: {e}")

    # Запись в базу данных
    audio_record = models.AudioRecord(
        session_id=session_id,
        file_path=file_path,
        duration=duration,
        format=final_extension,
        sample_rate=sample_rate
    )
    db.add(audio_record)
    db.commit()
    db.refresh(audio_record)

    return audio_record


@router.get("/session/{session_id}", response_model=List[schemas.AudioRecordResponse])
def get_session_audio(
        session_id: int,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    session = db.query(models.Session).filter(
        models.Session.id == session_id,
        models.Session.psychologist_id == current_psychologist.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    audio_records = db.query(models.AudioRecord).filter(
        models.AudioRecord.session_id == session_id
    ).all()

    return audio_records


@router.get("/{audio_id}", response_model=schemas.AudioRecordResponse)
def get_audio_record(
        audio_id: int,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    audio = db.query(models.AudioRecord).join(models.Session).filter(
        models.AudioRecord.id == audio_id,
        models.Session.psychologist_id == current_psychologist.id
    ).first()

    if not audio:
        raise HTTPException(status_code=404, detail="Audio record not found")

    return audio


@router.delete("/{audio_id}")
def delete_audio_record(
        audio_id: int,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    audio = db.query(models.AudioRecord).join(models.Session).filter(
        models.AudioRecord.id == audio_id,
        models.Session.psychologist_id == current_psychologist.id
    ).first()

    if not audio:
        raise HTTPException(status_code=404, detail="Audio record not found")

    if os.path.exists(audio.file_path):
        os.remove(audio.file_path)

    db.delete(audio)
    db.commit()

    return {"message": "Audio record deleted"}