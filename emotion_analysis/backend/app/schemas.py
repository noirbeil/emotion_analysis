from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, date, time
from typing import Optional, List
from typing import List
from enum import Enum


# ========== AUTH SCHEMAS ==========
class PsychologistCreate(BaseModel):
    name: str
    surname: str
    login: str
    password: str
    specialization: Optional[str] = None
    license_number: Optional[str] = None
    role: str = "psychologist"


class PsychologistLogin(BaseModel):
    login: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class PsychologistResponse(BaseModel):
    id: int
    name: str
    surname: str
    login: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


# ========== PATIENT SCHEMAS ==========
class PatientCreate(BaseModel):
    name: str
    surname: str
    birth_date: Optional[date] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None


class PatientResponse(BaseModel):
    id: int
    name: str
    surname: str
    birth_date: Optional[date]
    gender: Optional[str]
    status: str
    medical_history: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class PatientUpdate(BaseModel):
    name: Optional[str] = None
    surname: Optional[str] = None
    birth_date: Optional[date] = None
    gender: Optional[str] = None
    status: Optional[str] = None
    medical_history: Optional[str] = None


# ========== SESSION SCHEMAS ==========
class SessionCreate(BaseModel):
    patient_id: int
    session_date: date
    session_time: Optional[time] = None
    duration: Optional[int] = None
    notes: Optional[str] = None
    status: str = "planned"


class SessionResponse(BaseModel):
    id: int
    psychologist_id: int
    patient_id: int
    session_date: date
    session_time: Optional[time]
    duration: Optional[int]
    notes: Optional[str]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class SessionUpdate(BaseModel):
    session_date: Optional[date] = None
    session_time: Optional[time] = None
    duration: Optional[int] = None
    notes: Optional[str] = None
    status: Optional[str] = None


# ========== AUDIO SCHEMAS ==========
class AudioUploadResponse(BaseModel):
    id: int
    session_id: int
    file_path: str
    duration: Optional[float]
    format: Optional[str]
    sample_rate: Optional[int]
    record_date: datetime

    class Config:
        from_attributes = True


class AudioRecordResponse(BaseModel):
    id: int
    session_id: int
    file_path: str
    record_date: datetime
    duration: Optional[float]
    format: Optional[str]
    sample_rate: Optional[int]

    class Config:
        from_attributes = True


class EmotionType(str, Enum):
    anger = "anger"  # злость
    disgust = "disgust"  # брезгливость
    fear = "fear"  # страх
    joy = "joy"  # радость
    sadness = "sadness"  # грусть


class ResultFormat(str, Enum):
    table = "table"
    chart = "chart"


class MLModelType(str, Enum):
    emotionnet = "EmotionNet"
    lstm = "LSTM"
    cnn = "CNN"


class SessionAnalysisConfigCreate(BaseModel):
    emotions_to_analyze: List[EmotionType]  # массив эмоций
    threshold_value: float = Field(0.5, ge=0.0, le=1.0)
    result_format: ResultFormat = ResultFormat.table
    ml_model_type: MLModelType
    sensitivity: float = Field(0.7, ge=0.1, le=1.0)


class SessionCreateWithConfig(BaseModel):
    # Основные данные сессии
    patient_id: int
    session_date: date
    session_time: Optional[time] = None
    duration: Optional[int] = None
    notes: Optional[str] = None
    status: str = "planned"

    # Настройки анализа
    analysis_config: SessionAnalysisConfigCreate


class SessionAnalysisConfigResponse(BaseModel):
    id: int
    session_id: int
    emotions_to_analyze: List[str]
    threshold_value: float
    result_format: str
    ml_model_type: str
    sensitivity: float
    analysis_status: str
    aggregated_results: Optional[dict]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Добавьте в schemas.py
class PsychologistUpdate(BaseModel):
    name: Optional[str] = None
    surname: Optional[str] = None
    specialization: Optional[str] = None
    license_number: Optional[str] = None
    login: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None


class PsychologistCreateByAdmin(BaseModel):
    name: str
    surname: str
    login: str
    password: str
    specialization: Optional[str] = None
    license_number: Optional[str] = None
    role: str = "psychologist"


class SessionWithConfigResponse(SessionResponse):
    analysis_config: Optional[SessionAnalysisConfigResponse] = None