from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
import io
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc, or_, func
from typing import List, Optional
from datetime import date, datetime
import json
import os
import sys
from .. import database, models, schemas, auth
from pydantic import BaseModel
from ml.analyzer import run_full_analysis
from datetime import datetime, timezone


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ── Схема одного элемента списка проектов ──────────────────────
class ProjectListItem(BaseModel):
    id: int
    project_name: str
    session_date: date
    status: str
    patient_name: str
    patient_surname: str
    last_audio_date: Optional[datetime] = None
    analysis_status: str
    dominant_emotion: Optional[str] = None

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    items: List[ProjectListItem]
    total: int
    page: int
    pages: int

class RecentVoiceRecord(BaseModel):
    id: int
    session_id: int
    project_name: str
    patient_name: str
    patient_surname: str
    record_date: datetime
    duration: Optional[float] = None
    analysis_status: str

    class Config:
        from_attributes = True

@router.get("/recent-voice-records", response_model=List[RecentVoiceRecord])
def get_recent_voice_records(
        limit: int = Query(5, ge=1, le=20),
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    """
    Возвращает последние записи голоса для психолога.
    """
    # Получаем последние аудиозаписи для сессий текущего психолога
    recent_records = (
        db.query(
            models.AudioRecord,
            models.Session,
            models.Patient
        )
        .join(models.Session, models.AudioRecord.session_id == models.Session.id)
        .join(models.Patient, models.Session.patient_id == models.Patient.id)
        .filter(models.Session.psychologist_id == current_psychologist.id)
        .order_by(desc(models.AudioRecord.record_date))
        .limit(limit)
        .all()
    )

    result = []
    for audio, session, patient in recent_records:
        project_name = session.notes if session.notes else "Без названия"

        # Проверяем, есть ли результаты анализа для этой аудиозаписи
        has_analysis = (
                           db.query(models.EmotionAnalysis)
                           .filter(models.EmotionAnalysis.audio_record_id == audio.id)
                           .first()
                       ) is not None

        # Статус анализа
        if has_analysis:
            # Проверяем, есть ли emotion_results
            analysis = db.query(models.EmotionAnalysis).filter(
                models.EmotionAnalysis.audio_record_id == audio.id
            ).first()

            has_results = (
                              db.query(models.EmotionResult)
                              .filter(models.EmotionResult.emotion_analysis_id == analysis.id)
                              .first()
                          ) is not None

            analysis_status = "completed" if has_results else "processing"
        else:
            analysis_status = "pending"

        result.append(RecentVoiceRecord(
            id=audio.id,
            session_id=session.id,
            project_name=project_name,
            patient_name=patient.name,
            patient_surname=patient.surname,
            record_date=audio.record_date,
            duration=audio.duration,
            analysis_status=analysis_status
        ))

    return result


# ── Вспомогательная функция: регистрация кириллического шрифта ─
def _register_cyrillic_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # sessions.py находится в backend/app/routers/
    # поднимаемся на 4 уровня вверх до emotion_analysis/
    base_dir = os.path.dirname(  # emotion_analysis/
        os.path.dirname(         # backend/
            os.path.dirname(     # backend/app/
                os.path.dirname( # backend/app/routers/
                    os.path.abspath(__file__)
                )
            )
        )
    )

    reg_path  = os.path.join(base_dir, "fonts", "DejaVuSans.ttf")
    bold_path = os.path.join(base_dir, "fonts", "DejaVuSans-Bold.ttf")

    if os.path.exists(reg_path) and os.path.exists(bold_path):
        pdfmetrics.registerFont(TTFont("DejaVu",      reg_path))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold_path))
        return "DejaVu", "DejaVu-Bold"

    # Fallback на системные шрифты
    for sys_reg, sys_bold in [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]:
        if os.path.exists(sys_reg) and os.path.exists(sys_bold):
            pdfmetrics.registerFont(TTFont("DejaVu",      sys_reg))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", sys_bold))
            return "DejaVu", "DejaVu-Bold"

    return "Helvetica", "Helvetica-Bold"


# ── Генерация PDF из готового текста отчёта ────────────────────
def _build_pdf_from_text(report_text: str, session, session_id: int) -> bytes:
    """
    Принимает готовый текст отчёта (из reports.content),
    оборачивает его в красивый PDF с кириллическим шрифтом.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    FONT, FONT_BOLD = _register_cyrillic_font()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm
    )

    PURPLE      = colors.HexColor("#7b5ba8")
    PURPLE_LIGHT = colors.HexColor("#f3eeff")

    title_style = ParagraphStyle(
        "Title", fontName=FONT_BOLD, fontSize=16,
        textColor=PURPLE, spaceAfter=4, leading=20
    )
    meta_style = ParagraphStyle(
        "Meta", fontName=FONT, fontSize=10,
        textColor=colors.HexColor("#444444"), spaceAfter=2
    )
    section_style = ParagraphStyle(
        "Section", fontName=FONT_BOLD, fontSize=11,
        textColor=PURPLE, spaceBefore=10, spaceAfter=3
    )
    body_style = ParagraphStyle(
        "Body", fontName=FONT, fontSize=9,
        leading=14, spaceAfter=1,
        textColor=colors.HexColor("#222222")
    )
    footer_style = ParagraphStyle(
        "Footer", fontName=FONT, fontSize=8,
        textColor=colors.grey, spaceBefore=10
    )

    story = []

    # Заголовок
    story.append(Paragraph("Отчёт по эмоциональному анализу речи", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PURPLE, spaceAfter=6))

    # Мета-информация
    story.append(Paragraph(f"<b>ID сессии:</b> {session_id}", meta_style))
    story.append(Paragraph(f"<b>Дата сессии:</b> {session.session_date}", meta_style))
    story.append(Paragraph(
        f"<b>Пациент:</b> {session.patient.name} {session.patient.surname}", meta_style
    ))
    story.append(Paragraph(
        f"<b>Дата экспорта:</b> {datetime.now().strftime('%d.%m.%Y %H:%M')}", meta_style
    ))
    story.append(Spacer(1, 8))

    # Парсим текст отчёта по строкам и красиво разбиваем на секции
    if report_text:
        lines = report_text.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 3))
                continue

            # Заголовки секций (все заглавные или начинаются со знаков)
            if stripped == stripped.upper() and len(stripped) > 3 and not stripped.startswith("•"):
                story.append(Paragraph(stripped, section_style))
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#ccbbee"), spaceAfter=3))
            elif stripped.startswith("="):
                # Разделители пропускаем — у нас своя HR
                pass
            elif stripped.startswith("-") and len(stripped) > 3 and stripped == stripped[0] * len(stripped):
                # Строка из дефисов — пропускаем
                pass
            else:
                story.append(Paragraph(stripped, body_style))
    else:
        story.append(Paragraph("Текст отчёта недоступен.", body_style))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccbbee")))
    story.append(Paragraph(
        "Данный отчёт является вспомогательным инструментом для специалиста. "
        "Результаты не заменяют профессиональную психологическую оценку и не являются медицинским диагнозом.",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ── СПИСОК ПРОЕКТОВ ─────────────────────────────────────────────
@router.get("/list", response_model=ProjectListResponse)
def get_project_list(
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist),
        search: Optional[str] = Query(None, description="Поиск по ФИО пациента"),
        date_from: Optional[date] = Query(None),
        date_to:   Optional[date] = Query(None),
        page:      int = Query(1, ge=1),
        page_size: int = Query(8, ge=1, le=100),
):
    query = (
        db.query(models.Session)
        .options(
            joinedload(models.Session.patient),
            joinedload(models.Session.analysis_config),
        )
        .filter(models.Session.psychologist_id == current_psychologist.id)
    )

    if search and search.strip():
        search_lower = search.strip().lower()
        query = query.join(models.Patient).filter(
            or_(
                func.lower(models.Patient.name).contains(search_lower),
                func.lower(models.Patient.surname).contains(search_lower),
                func.lower(
                    func.concat(models.Patient.name, " ", models.Patient.surname)
                ).contains(search_lower),
            )
        )

    if date_from:
        query = query.filter(models.Session.session_date >= date_from)
    if date_to:
        query = query.filter(models.Session.session_date <= date_to)

    total = query.count()

    sessions = (
        query
        .order_by(desc(models.Session.session_date), desc(models.Session.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = []
    for s in sessions:
        last_audio = (
            db.query(models.AudioRecord.record_date)
            .filter(models.AudioRecord.session_id == s.id)
            .order_by(desc(models.AudioRecord.record_date))
            .first()
        )
        last_audio_date = last_audio[0] if last_audio else None

        config = s.analysis_config
        analysis_status = config.analysis_status if config else "pending"
        project_name = s.notes if s.notes else "Без названия"

        dominant_emotion = None
        if config and config.aggregated_results:
            try:
                agg = json.loads(config.aggregated_results)
                dominant_emotion = agg.get("dominant_emotion")
            except Exception:
                pass

        items.append(ProjectListItem(
            id=s.id,
            project_name=project_name,
            session_date=s.session_date,
            status=s.status,
            patient_name=s.patient.name,
            patient_surname=s.patient.surname,
            last_audio_date=last_audio_date,
            analysis_status=analysis_status,
            dominant_emotion=dominant_emotion,
        ))

    return ProjectListResponse(
        items=items,
        total=total,
        page=page,
        pages=max(1, (total + page_size - 1) // page_size),
    )


# ── СПИСОК СЕССИЙ ───────────────────────────────────────────────
@router.get("/", response_model=List[schemas.SessionResponse])
def get_sessions(
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist),
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        patient_id: Optional[int] = None,
        status_filter: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
):
    query = db.query(models.Session).filter(
        models.Session.psychologist_id == current_psychologist.id
    )
    if patient_id:    query = query.filter(models.Session.patient_id == patient_id)
    if status_filter: query = query.filter(models.Session.status == status_filter)
    if date_from:     query = query.filter(models.Session.session_date >= date_from)
    if date_to:       query = query.filter(models.Session.session_date <= date_to)
    return query.order_by(desc(models.Session.session_date)).offset(skip).limit(limit).all()


# ── ОДНА СЕССИЯ ─────────────────────────────────────────────────
@router.get("/{session_id}", response_model=schemas.SessionWithConfigResponse)
def get_session(
        session_id: int,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist),
):
    session = db.query(models.Session).filter(
        models.Session.id == session_id,
        models.Session.psychologist_id == current_psychologist.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.analysis_config:
        session.analysis_config.emotions_to_analyze = json.loads(
            session.analysis_config.emotions_to_analyze
        )
    return session


# ── СОЗДАНИЕ СЕССИИ ─────────────────────────────────────────────
@router.post("/", response_model=schemas.SessionWithConfigResponse)
def create_session(
        session_data: schemas.SessionCreateWithConfig,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist),
):
    patient = db.query(models.Patient).filter(models.Patient.id == session_data.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    db_session = models.Session(
        psychologist_id=current_psychologist.id,
        patient_id=session_data.patient_id,
        session_date=session_data.session_date,
        session_time=session_data.session_time,
        duration=session_data.duration,
        notes=session_data.notes,
        status=session_data.status,
    )
    db.add(db_session)
    db.flush()

    emotions_list = [e.value for e in session_data.analysis_config.emotions_to_analyze]
    analysis_config = models.SessionAnalysisConfig(
        session_id=db_session.id,
        emotions_to_analyze=json.dumps(emotions_list),
        threshold_value=session_data.analysis_config.threshold_value,
        result_format=session_data.analysis_config.result_format.value,
        ml_model_type=session_data.analysis_config.ml_model_type.value,
        sensitivity=session_data.analysis_config.sensitivity,
        analysis_status="pending",
    )
    db.add(analysis_config)
    db.commit()
    db.refresh(db_session)

    if db_session.analysis_config:
        db_session.analysis_config.emotions_to_analyze = json.loads(
            db_session.analysis_config.emotions_to_analyze
        )
    return db_session


# ── УДАЛЕНИЕ СЕССИИ ─────────────────────────────────────────────
@router.delete("/{session_id}")
def delete_session(
        session_id: int,
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist),
):
    session = db.query(models.Session).filter(
        models.Session.id == session_id,
        models.Session.psychologist_id == current_psychologist.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"message": "Session deleted"}


# ── ЗАПУСК АНАЛИЗА ──────────────────────────────────────────────
@router.post("/{session_id}/analyze")
def start_session_analysis(
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

    config = session.analysis_config
    if config:
        config.analysis_status = "processing"
        config.analysis_started_at = datetime.now(timezone.utc)
        db.commit()

    import threading
    thread = threading.Thread(
        target=_run_analysis_background,
        args=(session_id, db.bind.url)
    )
    thread.start()

    return {"status": "processing", "message": "Analysis started"}


def _run_analysis_background(session_id: int, db_url: str):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        audio_records = db.query(models.AudioRecord).filter(
            models.AudioRecord.session_id == session_id
        ).all()

        for audio in audio_records:
            if not os.path.exists(audio.file_path):
                continue

            result = run_full_analysis(
                audio_record_id=audio.id,
                audio_path=audio.file_path,
                session_id=session_id
            )
            print(f"Analysis complete for audio {audio.id}: "
                  f"{result.get('overall', {}).get('dominant_emotion')}")

        session = db.query(models.Session).filter(models.Session.id == session_id).first()
        if session and session.analysis_config:
            session.analysis_config.analysis_status = "completed"
            session.analysis_config.analysis_completed_at = datetime.now(timezone.utc)
            db.commit()

    except Exception as e:
        print(f"Analysis error for session {session_id}: {e}")
        session = db.query(models.Session).filter(models.Session.id == session_id).first()
        if session and session.analysis_config:
            session.analysis_config.analysis_status = "failed"
            db.commit()
    finally:
        db.close()


# ── СТАТУС АНАЛИЗА ──────────────────────────────────────────────
@router.get("/{session_id}/analysis-status")
def get_analysis_status(
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

    config = session.analysis_config
    return {
        "session_id": session_id,
        "analysis_status": config.analysis_status if config else "pending",
        "analysis_started_at": config.analysis_started_at if config else None,
        "analysis_completed_at": config.analysis_completed_at if config else None
    }


# ── РЕЗУЛЬТАТЫ АНАЛИЗА ──────────────────────────────────────────
@router.get("/{session_id}/analysis-results")
def get_analysis_results(
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

    results = []
    for audio in audio_records:
        analysis = db.query(models.EmotionAnalysis).filter(
            models.EmotionAnalysis.audio_record_id == audio.id
        ).first()

        if analysis:
            emotion_results = db.query(models.EmotionResult).filter(
                models.EmotionResult.emotion_analysis_id == analysis.id
            ).all()

            results.append({
                "audio_id": audio.id,
                "duration": audio.duration,
                "analysis_date": analysis.analysis_date,
                "confidence": analysis.confidence,
                "emotions": [
                    {
                        "emotion": er.emotion,
                        "intensity": er.intensity,
                        "valence": er.valence,
                        "arousal": er.arousal,
                        "timestamp_start": er.timestamp_start,
                        "timestamp_end": er.timestamp_end
                    }
                    for er in emotion_results
                ]
            })

    return {
        "session_id": session_id,
        "analysis_status": session.analysis_config.analysis_status if session.analysis_config else "pending",
        "results": results
    }


# ── ЭКСПОРТ ОТЧЁТА ──────────────────────────────────────────────
@router.get("/{session_id}/export-report")
def export_report(
        session_id: int,
        format: str = "pdf",
        db: Session = Depends(database.get_db),
        current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)
):
    """Экспорт отчёта в PDF или JSON.
    PDF строится из готового текста в таблице reports.content,
    который был сгенерирован ml/report_builder.py во время анализа.
    """

    # Проверяем сессию
    session = db.query(models.Session).filter(
        models.Session.id == session_id,
        models.Session.psychologist_id == current_psychologist.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── JSON ────────────────────────────────────────────────────
    if format == "json":
        audio_records = db.query(models.AudioRecord).filter(
            models.AudioRecord.session_id == session_id
        ).all()

        results = []
        for audio in audio_records:
            analysis = db.query(models.EmotionAnalysis).filter(
                models.EmotionAnalysis.audio_record_id == audio.id
            ).first()

            if analysis:
                emotion_results = db.query(models.EmotionResult).filter(
                    models.EmotionResult.emotion_analysis_id == analysis.id
                ).all()
                results.append({
                    "audio_id": audio.id,
                    "duration": audio.duration,
                    "analysis_date": analysis.analysis_date.isoformat() if analysis.analysis_date else None,
                    "emotions": [
                        {
                            "emotion": er.emotion,
                            "intensity": er.intensity,
                            "valence": er.valence,
                            "arousal": er.arousal,
                            "timestamp_start": er.timestamp_start,
                            "timestamp_end": er.timestamp_end
                        }
                        for er in emotion_results
                    ]
                })

        report_data = {
            "session_id": session_id,
            "session_date": session.session_date.isoformat() if session.session_date else None,
            "patient": f"{session.patient.name} {session.patient.surname}",
            "analysis_results": results,
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        return Response(
            content=json_str.encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=report_{session_id}.json"}
        )

    # ── PDF ─────────────────────────────────────────────────────
    try:
        # 1. Берём готовый текст из таблицы reports (последний по дате)
        report_record = (
            db.query(models.Report)
            .filter(models.Report.session_id == session_id)
            .order_by(desc(models.Report.report_date))
            .first()
        )

        if report_record and report_record.content:
            report_text = report_record.content
        else:
            # Анализ ещё не проводился — генерируем заглушку
            report_text = (
                f"ОТЧЁТ ПО АНАЛИЗУ ЭМОЦИОНАЛЬНОЙ ОКРАСКИ ГОЛОСА\n"
                f"{'=' * 60}\n"
                f"Дата экспорта: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
                f"Анализ для сессии #{session_id} ещё не был выполнен.\n"
                f"Запустите анализ и повторите экспорт."
            )

        pdf_bytes = _build_pdf_from_text(report_text, session, session_id)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=report_{session_id}.pdf"
            }
        )

    except ImportError as e:
        # reportlab не установлен
        raise HTTPException(
            status_code=500,
            detail=f"reportlab не установлен: {e}. Выполните: pip install reportlab"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка генерации PDF: {e}")



