from sqlalchemy import Column, Integer, String, DateTime, Date, Time, Float, Text, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Psychologist(Base):
    __tablename__ = "psychologists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    surname = Column(String(100), nullable=False)
    specialization = Column(String(200), nullable=True)
    license_number = Column(String(100), nullable=True)
    login = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default='psychologist')
    created_at = Column(DateTime, default=datetime.utcnow)

    # Связи
    sessions = relationship("Session", back_populates="psychologist")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    surname = Column(String(100), nullable=False)
    birth_date = Column(Date, nullable=True)
    gender = Column(String(10), nullable=True)
    status = Column(String(50), default='active')
    medical_history = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Связи
    sessions = relationship("Session", back_populates="patient")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    psychologist_id = Column(Integer, ForeignKey("psychologists.id", ondelete="RESTRICT"), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)  # Изменили с RESTRICT на CASCADE
    session_date = Column(Date, nullable=False)
    session_time = Column(Time, nullable=True)
    duration = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    status = Column(String(20), default='planned')
    created_at = Column(DateTime, default=datetime.utcnow)

    # Связи
    psychologist = relationship("Psychologist", back_populates="sessions")
    patient = relationship("Patient", back_populates="sessions")
    audio_records = relationship("AudioRecord", back_populates="session", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="session", cascade="all, delete-orphan")
    analysis_config = relationship("SessionAnalysisConfig", back_populates="session", uselist=False, cascade="all, delete-orphan")


class AudioRecord(Base):
    __tablename__ = "audio_records"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    file_path = Column(String(500), nullable=False)
    record_date = Column(DateTime, default=datetime.utcnow)
    duration = Column(Float, nullable=True)  # длительность в секундах
    format = Column(String(10), nullable=True)
    sample_rate = Column(Integer, default=22050)

    # Связи
    session = relationship("Session", back_populates="audio_records")
    emotion_analyses = relationship("EmotionAnalysis", back_populates="audio_record")


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=True)
    accuracy = Column(Float, nullable=True)
    trained_date = Column(Date, nullable=True)
    file_path = Column(String(500), nullable=True)

    # Связи
    emotion_analyses = relationship("EmotionAnalysis", back_populates="ml_model")


class EmotionAnalysis(Base):
    __tablename__ = "emotion_analysis"

    id = Column(Integer, primary_key=True, index=True)
    audio_record_id = Column(Integer, ForeignKey("audio_records.id", ondelete="CASCADE"), nullable=False)
    ml_model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="RESTRICT"), nullable=False)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)

    # Связи
    audio_record = relationship("AudioRecord", back_populates="emotion_analyses")
    ml_model = relationship("MLModel", back_populates="emotion_analyses")
    audio_features = relationship("AudioFeatures", back_populates="emotion_analysis", uselist=False)
    emotion_results = relationship("EmotionResult", back_populates="emotion_analysis")


class AudioFeatures(Base):
    __tablename__ = "audio_features"

    id = Column(Integer, primary_key=True, index=True)
    emotion_analysis_id = Column(Integer, ForeignKey("emotion_analysis.id", ondelete="CASCADE"), unique=True,
                                 nullable=False)
    pitch = Column(Float, nullable=True)
    tempo = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    spectral_centroid = Column(Float, nullable=True)
    mfcc_1 = Column(Float, nullable=True)
    mfcc_2 = Column(Float, nullable=True)
    mfcc_3 = Column(Float, nullable=True)
    mfcc_4 = Column(Float, nullable=True)
    mfcc_5 = Column(Float, nullable=True)
    mfcc_6 = Column(Float, nullable=True)
    mfcc_7 = Column(Float, nullable=True)
    mfcc_8 = Column(Float, nullable=True)
    mfcc_9 = Column(Float, nullable=True)
    mfcc_10 = Column(Float, nullable=True)
    mfcc_11 = Column(Float, nullable=True)
    mfcc_12 = Column(Float, nullable=True)
    mfcc_13 = Column(Float, nullable=True)

    # Связи
    emotion_analysis = relationship("EmotionAnalysis", back_populates="audio_features")


class EmotionResult(Base):
    __tablename__ = "emotion_results"

    id = Column(Integer, primary_key=True, index=True)
    emotion_analysis_id = Column(Integer, ForeignKey("emotion_analysis.id", ondelete="CASCADE"), nullable=False)
    emotion = Column(String(50), nullable=False)
    intensity = Column(Float, nullable=True)
    valence = Column(Float, nullable=True)
    arousal = Column(Float, nullable=True)
    timestamp_start = Column(Float, nullable=True)
    timestamp_end = Column(Float, nullable=True)

    # Связи
    emotion_analysis = relationship("EmotionAnalysis", back_populates="emotion_results")
    reports = relationship("Report", secondary="report_emotion_results", back_populates="emotion_results")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    report_date = Column(DateTime, default=datetime.utcnow)
    report_type = Column(String(50), default='standard')
    content = Column(Text, nullable=True)
    format = Column(String(10), default='pdf')

    # Связи
    session = relationship("Session", back_populates="reports")
    emotion_results = relationship("EmotionResult", secondary="report_emotion_results", back_populates="reports")


class SessionAnalysisConfig(Base):
    __tablename__ = "session_analysis_config"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True)

    # Какие эмоции анализировать - храним как JSON строку
    emotions_to_analyze = Column(Text, nullable=False)  # JSON array

    threshold_value = Column(Float, nullable=False, default=0.5)
    result_format = Column(String(20), nullable=False, default='table')
    ml_model_type = Column(String(100), nullable=False)
    sensitivity = Column(Float, nullable=False, default=0.7)

    # Статус и результаты
    analysis_status = Column(String(20), nullable=False, default='pending')
    aggregated_results = Column(Text, nullable=True)  # JSON с результатами
    analysis_started_at = Column(DateTime, nullable=True)
    analysis_completed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Связи
    session = relationship("Session", back_populates="analysis_config")

# Связующая таблица для many-to-many
class ReportEmotionResult(Base):
    __tablename__ = "report_emotion_results"

    report_id = Column(Integer, ForeignKey("reports.id", ondelete="CASCADE"), primary_key=True)
    emotion_result_id = Column(Integer, ForeignKey("emotion_results.id", ondelete="CASCADE"), primary_key=True)