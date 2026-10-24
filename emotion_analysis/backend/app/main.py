from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from .routers import auth, patients, sessions, audio, compare, admin_psychologists
from .database import engine, Base

# Создание таблиц в БД (если их нет)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AEOG Backend", version="1.0", description="Анализ эмоций в голосе")

# CORS для React фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем папку с загруженными файлами для статической раздачи (опционально)
if os.path.exists("uploads"):
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Подключение роутеров
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(sessions.router)
app.include_router(audio.router)
app.include_router(compare.router)
app.include_router(admin_psychologists.router)


@app.get("/")
def root():
    return {
        "message": "AEOG Backend is running",
        "docs": "/docs",
        "version": "1.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}