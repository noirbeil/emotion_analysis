from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from .. import database, models, auth, schemas

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=schemas.Token)
def register(psychologist: schemas.PsychologistCreate, db: Session = Depends(database.get_db)):
    # Проверка существования пользователя по логину
    existing = db.query(models.Psychologist).filter(
        models.Psychologist.login == psychologist.login
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Login already registered")

    # Проверка роли
    if psychologist.role not in ["psychologist", "admin"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    hashed = auth.get_password_hash(psychologist.password)
    db_psychologist = models.Psychologist(
        name=psychologist.name,
        surname=psychologist.surname,
        login=psychologist.login,
        password_hash=hashed,
        specialization=psychologist.specialization,
        license_number=psychologist.license_number,
        role=psychologist.role
    )
    db.add(db_psychologist)
    db.commit()
    db.refresh(db_psychologist)

    access_token = auth.create_access_token(data={"sub": db_psychologist.login})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login", response_model=schemas.Token)
def login(login_data: schemas.PsychologistLogin, db: Session = Depends(database.get_db)):
    psychologist = auth.authenticate_psychologist(db, login_data.login, login_data.password)
    if not psychologist:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login or password"
        )

    access_token = auth.create_access_token(data={"sub": psychologist.login})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=schemas.PsychologistResponse)
def get_current_user(current_psychologist: models.Psychologist = Depends(auth.get_current_psychologist)):
    return current_psychologist