# routers/patients.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from .. import database, models, schemas, auth

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("/", response_model=List[schemas.PatientResponse])
def get_patients(
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist),
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        status_filter: Optional[str] = None
):
    """Получить список пациентов (доступно всем авторизованным)"""
    query = db.query(models.Patient)
    if status_filter:
        query = query.filter(models.Patient.status == status_filter)
    patients = query.offset(skip).limit(limit).all()
    return patients


@router.post("/", response_model=schemas.PatientResponse)
def create_patient(
        patient: schemas.PatientCreate,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    """Создать нового пациента (доступно всем авторизованным)"""
    db_patient = models.Patient(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@router.get("/{patient_id}", response_model=schemas.PatientResponse)
def get_patient(
        patient_id: int,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    """Получить пациента по ID"""
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}", response_model=schemas.PatientResponse)
def update_patient(
        patient_id: int,
        patient_update: schemas.PatientUpdate,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    """Обновить данные пациента"""
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    for key, value in patient_update.dict(exclude_unset=True).items():
        setattr(patient, key, value)

    db.commit()
    db.refresh(patient)
    return patient


@router.delete("/{patient_id}")
def delete_patient(
        patient_id: int,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    """Удалить пациента (только для админа)"""
    # Проверяем права доступа
    if current_psychologist.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only administrators can delete patients"
        )

    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        # Находим все сессии пациента
        sessions = db.query(models.Session).filter(models.Session.patient_id == patient_id).all()

        # Удаляем все связанные данные вручную
        for session in sessions:
            # Удаляем аудиозаписи и связанные с ними данные
            audio_records = db.query(models.AudioRecord).filter(models.AudioRecord.session_id == session.id).all()
            for audio in audio_records:
                # Удаляем анализ эмоций
                emotion_analyses = db.query(models.EmotionAnalysis).filter(
                    models.EmotionAnalysis.audio_record_id == audio.id
                ).all()
                for analysis in emotion_analyses:
                    # Удаляем признаки аудио
                    db.query(models.AudioFeatures).filter(
                        models.AudioFeatures.emotion_analysis_id == analysis.id
                    ).delete()
                    # Удаляем результаты эмоций
                    db.query(models.EmotionResult).filter(
                        models.EmotionResult.emotion_analysis_id == analysis.id
                    ).delete()
                    # Удаляем сам анализ
                    db.delete(analysis)
                # Удаляем аудиозапись
                db.delete(audio)

            # Удаляем отчеты
            db.query(models.Report).filter(models.Report.session_id == session.id).delete()

            # Удаляем конфигурацию анализа
            db.query(models.SessionAnalysisConfig).filter(
                models.SessionAnalysisConfig.session_id == session.id
            ).delete()

            # Удаляем сессию
            db.delete(session)

        # Теперь удаляем пациента
        db.delete(patient)
        db.commit()

        return {"message": "Patient and all related data deleted successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting patient: {str(e)}")