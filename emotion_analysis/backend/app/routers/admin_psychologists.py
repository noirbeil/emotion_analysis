# routers/admin_psychologists.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from .. import database, models, schemas, auth

router = APIRouter(prefix="/admin/psychologists", tags=["Admin Psychologists"])


@router.get("/", response_model=List[schemas.PsychologistResponse])
def get_all_psychologists(
        db: Session = Depends(database.get_db),
        current_admin: models.Psychologist = Depends(auth.get_current_admin)
):
    """Получить всех психологов (только для админа)"""
    psychologists = db.query(models.Psychologist).all()
    return psychologists


@router.get("/{psychologist_id}", response_model=schemas.PsychologistResponse)
def get_psychologist(
        psychologist_id: int,
        db: Session = Depends(database.get_db),
        current_admin: models.Psychologist = Depends(auth.get_current_admin)
):
    """Получить психолога по ID (только для админа)"""
    psychologist = db.query(models.Psychologist).filter(
        models.Psychologist.id == psychologist_id
    ).first()

    if not psychologist:
        raise HTTPException(status_code=404, detail="Psychologist not found")

    return psychologist


@router.put("/{psychologist_id}", response_model=schemas.PsychologistResponse)
def update_psychologist(
        psychologist_id: int,
        psychologist_update: schemas.PsychologistUpdate,
        db: Session = Depends(database.get_db),
        current_admin: models.Psychologist = Depends(auth.get_current_admin)
):
    """Обновить данные психолога (только для админа)"""
    psychologist = db.query(models.Psychologist).filter(
        models.Psychologist.id == psychologist_id
    ).first()

    if not psychologist:
        raise HTTPException(status_code=404, detail="Psychologist not found")

    # Нельзя изменить роль админа у последнего админа
    if psychologist.role == "admin" and psychologist_update.role == "psychologist":
        admin_count = db.query(models.Psychologist).filter(
            models.Psychologist.role == "admin"
        ).count()
        if admin_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot demote the last admin"
            )

    # Обновляем только переданные поля
    update_data = psychologist_update.dict(exclude_unset=True)

    # Если обновляется пароль - хэшируем его
    if "password" in update_data and update_data["password"]:
        update_data["password_hash"] = auth.get_password_hash(update_data.pop("password"))

    for key, value in update_data.items():
        setattr(psychologist, key, value)

    db.commit()
    db.refresh(psychologist)
    return psychologist


@router.delete("/{psychologist_id}")
def delete_psychologist(
        psychologist_id: int,
        db: Session = Depends(database.get_db),
        current_admin: models.Psychologist = Depends(auth.get_current_admin)
):
    """Удалить психолога (только для админа)"""
    psychologist = db.query(models.Psychologist).filter(
        models.Psychologist.id == psychologist_id
    ).first()

    if not psychologist:
        raise HTTPException(status_code=404, detail="Psychologist not found")

    # Нельзя удалить самого себя
    if psychologist.id == current_admin.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete your own account"
        )

    # Нельзя удалить последнего админа
    if psychologist.role == "admin":
        admin_count = db.query(models.Psychologist).filter(
            models.Psychologist.role == "admin"
        ).count()
        if admin_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete the last admin"
            )

    db.delete(psychologist)
    db.commit()
    return {"message": "Psychologist deleted successfully"}