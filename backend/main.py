"""FastAPI app: serves the frontend and the two modules' APIs. Run: uvicorn main:app --reload (from /backend)."""
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import ai_service
import interview_service as iv
import resume_analyzer
from models import AnswerRequest, ReportRequest, StartRequest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="AI Interview Preparation Coach")

# In-memory only: extracted CV text kept briefly so the interview module can use it as context.
CV_STORE: dict[str, tuple[float, str]] = {}
CV_TTL = 2 * 60 * 60


def _purge_cvs():
    now = time.time()
    for k in [k for k, (t, _) in CV_STORE.items() if now - t > CV_TTL]:
        CV_STORE.pop(k, None)
    while len(CV_STORE) > 200:
        CV_STORE.pop(next(iter(CV_STORE)))


@app.exception_handler(iv.InterviewError)
async def _interview_error(_, exc: iv.InterviewError):
    return JSONResponse({"detail": str(exc)}, status_code=exc.status)


@app.get("/api/status")
def status():
    return {"mode": "ai" if ai_service.AI_ENABLED else "demo", "model": ai_service.MODEL if ai_service.AI_ENABLED else None}


@app.post("/api/cv/analyze")
async def analyze_cv(file: UploadFile = File(...)):
    data = await file.read(resume_analyzer.MAX_FILE_BYTES + 1)
    try:
        text = resume_analyzer.extract_text(file.filename or "", data)
    except resume_analyzer.CVError as e:
        raise HTTPException(400, str(e))
    del data  # the uploaded file is never written to disk
    analysis = resume_analyzer.analyze(text)
    _purge_cvs()
    cv_id = uuid.uuid4().hex
    CV_STORE[cv_id] = (time.time(), text)
    return {
        "cv_id": cv_id,
        "filename": file.filename,
        "analysis": analysis,
        "extracted_preview": text[:1500],
        "extracted_chars": len(text),
    }


@app.post("/api/interview/start")
def start_interview(req: StartRequest):
    cv_text = None
    if req.cv_id:
        entry = CV_STORE.get(req.cv_id)
        cv_text = entry[1] if entry else None  # CV is optional - silently continue without it if expired
    return iv.start(req.domain, req.difficulty, cv_text)


@app.post("/api/interview/answer")
def answer(req: AnswerRequest):
    return iv.answer(req.session_id, req.answer)


@app.post("/api/interview/report")
def report(req: ReportRequest):
    return iv.report(req.session_id)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="static")
