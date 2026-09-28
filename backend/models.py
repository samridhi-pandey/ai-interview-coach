"""Pydantic models = the JSON contracts for AI output. Every AI response is validated against these."""
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, field_validator

# Exactly two interview domains.
DOMAINS = ("Technical", "HR")
DIFFICULTIES = ("Easy", "Medium", "Hard")
TOTAL_QUESTIONS = 5

QuestionType = Literal["conceptual", "practical", "behavioral", "situational"]

# AI models sometimes return 7.0 / "7" / 6.5 - coerce to int, then enforce 0-10.
Score10 = Annotated[int, BeforeValidator(lambda v: round(float(v))), Field(ge=0, le=10)]


def _clip_list(v):
    return [str(x).strip() for x in v if str(x).strip()][:5] if isinstance(v, list) else v


ShortList = Annotated[list[str], BeforeValidator(_clip_list)]


# ---------- Module 2: interview ----------
class Question(BaseModel):
    question: str = Field(min_length=10)
    topic: str = Field(min_length=2)  # e.g. "DBMS - indexing", "Teamwork"
    type: QuestionType

    @field_validator("type", mode="before")
    @classmethod
    def _lower(cls, v):
        return str(v).strip().lower()


class Evaluation(BaseModel):
    score: Score10
    strengths: ShortList
    improvements: ShortList
    feedback: str = Field(min_length=5)
    ideal_answer: str = Field(min_length=5)
    communication_score: Score10
    technical_depth_score: Score10  # for HR this is "relevance / content quality"
    confidence_score: Score10


class TurnResult(Evaluation):
    """One AI call = evaluation of the answer + the adaptive next question (null after question 5)."""
    next_question: Optional[Question] = None


class FinalReport(BaseModel):
    """AI-written parts of the final report. The numbers are computed from real scores, not by the AI."""
    strong_areas: ShortList
    areas_to_improve: ShortList
    top_3_tips: Annotated[list[str], BeforeValidator(lambda v: _clip_list(v)[:3] if isinstance(v, list) else v)]


# ---------- Module 1: CV analyzer ----------
class CVNarrative(BaseModel):
    """AI-written qualitative feedback. ATS numbers are rule-based and never come from the AI."""
    strengths: ShortList
    areas_to_improve: ShortList
    recommendations: ShortList


# ---------- API request bodies ----------
class StartRequest(BaseModel):
    domain: str
    difficulty: str
    cv_id: Optional[str] = None


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


class ReportRequest(BaseModel):
    session_id: str
