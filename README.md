# AI Interview Preparation Coach

TCS Tech Day AI Hackathon prototype. Two modules in one web app:

1. **CV Analyzer** - upload a PDF/DOCX CV, get a transparent rule-based ATS score (/100), detected sections and skills,
   strengths, weaknesses and recommendations (AI-written when an API key is set).
2. **Interview Preparation** - adaptive 5-question mock interview.
   - Domains: **Technical** (programming, DSA, OOP, DBMS, OS, networks, cybersecurity, CS fundamentals - practical *and* conceptual)
     or **HR** (introduction, teamwork, leadership, conflict, motivation, behavioral, situational).
   - Difficulty: Easy / Medium / Hard. Optional CV context personalises questions (never assumes skills not in the CV).
   - After every answer: score /10, strengths, improvements, feedback, sample strong answer, communication / technical depth
     (relevance for HR) / confidence scores, then the next question adapts to your previous answers and scores.
   - Final readiness report computed from your real scores + AI summary and top-3 tips.

## Run (Windows PowerShell)

```powershell
cd ai-interview-coach
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
# put your key in .env  ->  ANTHROPIC_API_KEY=sk-ant-...
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

No API key? The app runs in a clearly labelled **DEMO mode** (orange badge): local question bank and simple heuristic
feedback. The ATS score is rule-based in both modes.

## Structure

```
backend/  main.py (API) · resume_analyzer.py (extract + ATS) · interview_service.py (interview flow + demo mode)
          ai_service.py (only file that talks to the AI provider) · prompts.py · models.py (validated JSON schemas)
frontend/ index.html · style.css · app.js
```

## Privacy

No database, no file storage. The uploaded CV is parsed in memory, and its text plus interview state live in server
memory only (expiring; interview data is deleted when the report is generated). Resume and answer contents are not logged.
The API key lives only in `.env` (git-ignored) and never reaches the browser.

## Configuration

`ANTHROPIC_API_KEY` (required for AI mode) and optional `ANTHROPIC_MODEL` (default `claude-opus-5`) in `.env`.
To use another provider, replace `generate_json()` in `backend/ai_service.py`.
