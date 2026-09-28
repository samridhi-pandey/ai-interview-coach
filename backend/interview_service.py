"""Module 2: adaptive 5-question mock interview (Technical or HR) with in-memory sessions only."""
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import ai_service
from models import (
    DIFFICULTIES, DOMAINS, TOTAL_QUESTIONS, Evaluation, FinalReport, Question, TurnResult,
)
from prompts import REPORT_SYSTEM, first_question_system, turn_system

MAX_ANSWER_CHARS = 4000
MAX_SESSIONS = 200
SESSION_TTL = 2 * 60 * 60


class InterviewError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Session:
    domain: str
    difficulty: str
    cv_text: Optional[str]
    questions: list = field(default_factory=list)  # list[Question]
    answers: list = field(default_factory=list)  # list[str]
    evals: list = field(default_factory=list)  # list[Evaluation]
    created: float = field(default_factory=time.time)


SESSIONS: dict[str, Session] = {}


def _cleanup():
    now = time.time()
    for sid in [s for s, v in SESSIONS.items() if now - v.created > SESSION_TTL]:
        SESSIONS.pop(sid, None)
    while len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))


def validate_setup(domain: str, difficulty: str) -> tuple[str, str]:
    d = {x.lower(): x for x in DOMAINS}.get((domain or "").strip().lower())
    if not d:
        raise InterviewError(f"Invalid domain '{domain}'. Please choose 'Technical' or 'HR'.")
    lvl = {x.lower(): x for x in DIFFICULTIES}.get((difficulty or "").strip().lower())
    if not lvl:
        raise InterviewError(f"Invalid difficulty '{difficulty}'. Please choose Easy, Medium or Hard.")
    return d, lvl


def _get(session_id: str) -> Session:
    s = SESSIONS.get(session_id)
    if not s:
        raise InterviewError("Interview session not found or expired. Please start a new interview.", 404)
    return s


def _cv_block(s: Session) -> str:
    if s.cv_text:
        return f"<candidate_cv>\n{s.cv_text}\n</candidate_cv>"
    return "No CV was provided. Do not reference a CV, projects or experience."


def _history_block(s: Session) -> str:
    if not s.evals:
        return "(no previous questions)"
    parts = []
    for i, (q, a, e) in enumerate(zip(s.questions, s.answers, s.evals), 1):
        parts.append(
            f"Q{i} [{q.type}, topic: {q.topic}]: {q.question}\n"
            f"Answer summary/score: {e.score}/10; improvements needed: {'; '.join(e.improvements)}"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------- start
def start(domain: str, difficulty: str, cv_text: Optional[str]) -> dict:
    domain, difficulty = validate_setup(domain, difficulty)
    _cleanup()
    s = Session(domain=domain, difficulty=difficulty, cv_text=cv_text)
    if ai_service.AI_ENABLED:
        user = (
            f"Difficulty: {difficulty}\n{_cv_block(s)}\n\n"
            f"Write question 1 of {TOTAL_QUESTIONS}. "
            + ("Start with a warm, common opening question." if domain == "HR" else "Start at a suitable warm-up level.")
        )
        try:
            q = ai_service.generate_json(first_question_system(domain), user, Question, max_tokens=1500)
        except ai_service.AIError as e:
            raise InterviewError(str(e), 502)
    else:
        q = _mock_question(s, prev_score=None)
    s.questions.append(q)
    sid = uuid.uuid4().hex
    SESSIONS[sid] = s
    return {
        "session_id": sid,
        "domain": domain,
        "difficulty": difficulty,
        "total": TOTAL_QUESTIONS,
        "cv_used": bool(cv_text),
        "mode": "ai" if ai_service.AI_ENABLED else "demo",
        "question": _q_out(1, q),
    }


def _q_out(n: int, q: Question) -> dict:
    return {"number": n, "text": q.question, "topic": q.topic, "type": q.type}


# ---------------------------------------------------------------- answer
def answer(session_id: str, text: str) -> dict:
    s = _get(session_id)
    text = (text or "").strip()
    if not text:
        raise InterviewError("Please type an answer before submitting.")
    if len(text) > MAX_ANSWER_CHARS:
        raise InterviewError(f"Answer is too long (max {MAX_ANSWER_CHARS} characters).")
    if len(s.answers) >= len(s.questions):
        raise InterviewError("This interview is already complete. Please view your report.")

    n = len(s.answers) + 1  # question being answered
    q = s.questions[n - 1]
    last = n == TOTAL_QUESTIONS

    if ai_service.AI_ENABLED:
        user = (
            f"Difficulty: {s.difficulty}\n{_cv_block(s)}\n\n"
            f"Interview so far:\n{_history_block(s)}\n\n"
            f"Current question (question {n} of {TOTAL_QUESTIONS}) [{q.type}, topic: {q.topic}]: {q.question}\n"
            f"<answer>\n{text}\n</answer>\n\n"
            + ("This is the last question: set next_question to null." if last
               else f"Then write question {n + 1} of {TOTAL_QUESTIONS}, adapting to this answer.")
        )
        try:
            result = ai_service.generate_json(turn_system(s.domain), user, TurnResult, max_tokens=6000)
        except ai_service.AIError as e:
            raise InterviewError(str(e), 502)
        if not last and result.next_question is None:
            # model forgot the next question - ask for it separately rather than failing the demo
            try:
                result.next_question = ai_service.generate_json(
                    first_question_system(s.domain),
                    f"Difficulty: {s.difficulty}\n{_cv_block(s)}\n\nInterview so far:\n{_history_block(s)}\n"
                    f"Previous question: {q.question} (scored {result.score}/10)\n"
                    f"Write question {n + 1} of {TOTAL_QUESTIONS}; do not repeat earlier topics.",
                    Question, max_tokens=1500,
                )
            except ai_service.AIError as e:
                raise InterviewError(str(e), 502)
        ev = Evaluation.model_validate(result.model_dump(exclude={"next_question"}))
        nxt = None if last else result.next_question
    else:
        ev = _mock_evaluate(s, q, text)
        nxt = None if last else _mock_question(s, prev_score=ev.score, exclude=q)

    s.answers.append(text)
    s.evals.append(ev)
    if nxt:
        s.questions.append(nxt)
    return {
        "evaluation": ev.model_dump(),
        "next_question": _q_out(n + 1, nxt) if nxt else None,
        "finished": last,
        "mode": "ai" if ai_service.AI_ENABLED else "demo",
    }


# ---------------------------------------------------------------- report
def _avg(xs) -> float:
    return sum(xs) / len(xs)


def report(session_id: str) -> dict:
    s = _get(session_id)
    if len(s.evals) < TOTAL_QUESTIONS:
        raise InterviewError(f"Please answer all {TOTAL_QUESTIONS} questions first ({len(s.evals)} answered).")

    scores = [e.score for e in s.evals]
    comm = round(_avg([e.communication_score for e in s.evals]), 1)
    tech = round(_avg([e.technical_depth_score for e in s.evals]), 1)
    conf = round(_avg([e.confidence_score for e in s.evals]), 1)
    overall = round(_avg(scores) * 10)
    # Readiness: half answer quality, half the three skill dimensions. Fully derived from real scores.
    readiness = round((0.5 * _avg(scores) + 0.5 * _avg([comm, tech, conf])) * 10)

    narrative, mode, note = None, "demo", None
    if ai_service.AI_ENABLED:
        lines = []
        for i, (q, a, e) in enumerate(zip(s.questions, s.answers, s.evals), 1):
            lines.append(
                f"Q{i} [{q.type}, {q.topic}] {q.question}\n<answer>{a}</answer>\n"
                f"score {e.score}/10 (communication {e.communication_score}, "
                f"{'relevance' if s.domain == 'HR' else 'technical depth'} {e.technical_depth_score}, "
                f"confidence {e.confidence_score}); improvements: {'; '.join(e.improvements)}"
            )
        user = f"Domain: {s.domain}, difficulty: {s.difficulty}\n\n" + "\n\n".join(lines)
        try:
            narrative = ai_service.generate_json(REPORT_SYSTEM, user, FinalReport, max_tokens=2500)
            mode = "ai"
        except ai_service.AIError as e:
            note = f"AI summary unavailable ({e}). Showing a score-based summary instead."
    if narrative is None:
        narrative = _rule_report(s)

    out = {
        "domain": s.domain,
        "difficulty": s.difficulty,
        "overall_score": overall,
        "communication_score": comm,
        "technical_depth_score": tech,
        "technical_depth_label": "Content Quality" if s.domain == "HR" else "Technical Depth",
        "confidence_score": conf,
        "readiness_percentage": readiness,
        "strong_areas": narrative.strong_areas,
        "areas_to_improve": narrative.areas_to_improve,
        "top_3_tips": narrative.top_3_tips,
        "questions": [
            {"number": i, "topic": q.topic, "question": q.question, "score": e.score}
            for i, (q, e) in enumerate(zip(s.questions, s.evals), 1)
        ],
        "mode": mode,
        "note": note,
    }
    SESSIONS.pop(session_id, None)  # nothing about the interview is kept after the report
    return out


def _rule_report(s: Session) -> FinalReport:
    ranked = sorted(zip(s.questions, s.evals), key=lambda p: p[1].score, reverse=True)
    strong = [f"{q.topic} (scored {e.score}/10)" for q, e in ranked if e.score >= 6][:3]
    weak = [f"{q.topic} (scored {e.score}/10)" for q, e in reversed(ranked) if e.score < 6][:3]
    tips = [
        "Structure answers: definition or context first, then an example, then the outcome or trade-off.",
        "Practise speaking your answers aloud in 60-90 seconds to build confidence.",
        "Revise the topics listed under 'Areas to improve' and re-attempt them.",
    ]
    return FinalReport(
        strong_areas=strong or ["Keep practising - no topic scored 6/10 or higher yet."],
        areas_to_improve=weak or ["No topic scored below 6/10 - try a harder difficulty next."],
        top_3_tips=tips,
    )


# ---------------------------------------------------------------- DEMO mode (no API key)
# Clearly labelled in the UI. Local question bank + simple, transparent answer heuristics.
def _q(question, topic, type_, ideal):
    return (Question(question=question, topic=topic, type=type_), ideal)


BANK = {
    "Technical": {
        "Easy": [
            _q("What is the difference between a process and a thread?", "Operating Systems", "conceptual",
               "A process is an independent program in execution with its own memory space; a thread is a lightweight unit inside a process that shares its memory. Threads are cheaper to create and switch, but sharing memory needs synchronization to avoid race conditions."),
            _q("Explain the four pillars of OOP with a short example.", "OOP", "conceptual",
               "Encapsulation bundles data and methods and hides internals; abstraction exposes only essentials; inheritance lets a class reuse another's behaviour; polymorphism lets one interface have many implementations, e.g. a Shape with area() overridden by Circle and Rectangle."),
            _q("What is the difference between an array and a linked list?", "DSA", "conceptual",
               "An array stores elements contiguously giving O(1) index access but costly insertion; a linked list stores nodes with pointers giving O(1) insertion at a known node but O(n) access."),
            _q("What is a primary key and how is it different from a foreign key?", "DBMS", "conceptual",
               "A primary key uniquely identifies each row in a table and cannot be null; a foreign key is a column that references the primary key of another table to enforce relationships and referential integrity."),
            _q("What happens at a high level when you type a URL into a browser?", "Computer Networks", "conceptual",
               "The browser resolves the domain via DNS, opens a TCP connection (and TLS for HTTPS) to the server, sends an HTTP request, receives the response and renders the page."),
            _q("Write, in words or pseudo-code, how you would reverse a string.", "Programming", "practical",
               "Use two pointers at the start and end and swap characters while moving inward until they meet; this is O(n) time and O(1) extra space for a mutable array (or build a new string by iterating backwards)."),
        ],
        "Medium": [
            _q("Explain database normalization and why we use it. When might you denormalize?", "DBMS", "conceptual",
               "Normalization organizes tables (1NF, 2NF, 3NF) to remove redundancy and update anomalies. You may denormalize in read-heavy systems to avoid expensive joins, accepting some redundancy."),
            _q("How would you detect a cycle in a linked list? Explain the time and space complexity.", "DSA", "practical",
               "Use Floyd's tortoise and hare: a slow pointer moves one step and a fast pointer two; if they ever meet there is a cycle. It is O(n) time and O(1) space, better than a hash set which needs O(n) space."),
            _q("What is a deadlock? What are the necessary conditions and how can it be prevented?", "Operating Systems", "conceptual",
               "A deadlock is when processes wait on each other forever. The four Coffman conditions are mutual exclusion, hold and wait, no preemption and circular wait; breaking any one, for example by ordering resource acquisition, prevents it."),
            _q("Your SQL query on a large table is slow. How would you investigate and fix it?", "DBMS", "practical",
               "Look at the query plan (EXPLAIN), check for full table scans, add appropriate indexes on filter and join columns, avoid SELECT *, rewrite subqueries or joins, and consider caching or pagination."),
            _q("Explain the difference between TCP and UDP and give a use case for each.", "Computer Networks", "conceptual",
               "TCP is connection-oriented, reliable and ordered with flow control, used for web and file transfer; UDP is connectionless and faster without guarantees, used for streaming, gaming and DNS."),
            _q("Design a simple URL shortener. What components and data would you need?", "System Design", "practical",
               "A service that generates a unique short key (base62 of an id or hash), stores key-to-URL mappings in a database with an index on the key, redirects on lookup, and adds a cache for hot links plus collision handling and expiry."),
        ],
        "Hard": [
            _q("Explain how a hash map works internally and how collisions are handled. What is the worst-case complexity?", "DSA", "conceptual",
               "A hash function maps keys to bucket indexes; collisions are handled by chaining or open addressing, with resizing when the load factor grows. Average lookup is O(1), worst case O(n) when many keys collide (O(log n) with tree buckets)."),
            _q("Design a rate limiter for an API used by millions of users. Discuss the algorithm and trade-offs.", "System Design", "practical",
               "Use a token bucket or sliding window counter per user/IP stored in a fast store such as Redis with atomic increments and expiry; discuss distributed consistency, burst handling, and returning 429 with retry headers."),
            _q("Explain ACID properties and isolation levels. What problems does each level prevent?", "DBMS", "conceptual",
               "ACID: atomicity, consistency, isolation, durability. Read committed prevents dirty reads, repeatable read prevents non-repeatable reads, and serializable also prevents phantom reads, at increasing cost to concurrency."),
            _q("How does virtual memory and paging work, and what causes thrashing?", "Operating Systems", "conceptual",
               "Virtual memory maps virtual pages to physical frames through page tables with a TLB cache; pages not in RAM cause page faults and are loaded from disk. Thrashing occurs when the working sets exceed RAM so the system spends most time swapping pages."),
            _q("A production service is intermittently returning 500 errors under load. How would you debug it?", "Software Engineering", "practical",
               "Check logs and metrics for the error pattern, correlate with load and deployments, look at resource limits (CPU, memory, connections, DB pool), reproduce with load testing, then fix the root cause and add monitoring and alerts."),
            _q("What is the difference between symmetric and asymmetric encryption, and how does HTTPS use both?", "Cybersecurity", "conceptual",
               "Symmetric uses one shared key and is fast; asymmetric uses a public/private key pair and is slower. In HTTPS, asymmetric cryptography and certificates authenticate the server and exchange a session key, which then encrypts the traffic symmetrically."),
        ],
    },
    "HR": {
        "Easy": [
            _q("Please tell me about yourself.", "Self introduction", "conceptual",
               "Give a short structured intro: your name and course, key skills, one or two highlights such as a project or internship, and why you are interested in this role - about a minute."),
            _q("What are your key strengths?", "Strengths", "conceptual",
               "Name two or three strengths relevant to the role and back each with a brief real example that shows the result."),
            _q("What is one weakness of yours, and how are you working on it?", "Weaknesses", "conceptual",
               "Choose a genuine, non-critical weakness, show self-awareness and describe concrete steps you are taking to improve, with early results."),
            _q("Why do you want to join our company?", "Motivation", "conceptual",
               "Show you researched the company, connect its work or values with your skills and interests, and explain what you hope to learn and contribute."),
            _q("Where do you see yourself in five years?", "Career goals", "conceptual",
               "Describe realistic growth: building strong skills, taking on responsibility and contributing meaningfully, aligned with the opportunities the company offers."),
        ],
        "Medium": [
            _q("Tell me about a time you worked in a team. What was your role and what was the outcome?", "Teamwork", "behavioral",
               "Use STAR: describe the situation and goal, your specific role, the actions you took to collaborate, and the measurable result, plus what you learned."),
            _q("Describe a situation where you had a conflict with a teammate. How did you handle it?", "Conflict handling", "behavioral",
               "Explain the disagreement neutrally, how you listened and communicated, the compromise or solution reached, and the improved outcome."),
            _q("Tell me about a time you led a project or took initiative.", "Leadership", "behavioral",
               "Use STAR to show how you organised people or tasks, made decisions, handled obstacles and delivered a result."),
            _q("Tell me about a time you failed or made a mistake. What did you learn?", "Resilience", "behavioral",
               "Own the mistake honestly, explain what you did to fix it and what you changed afterwards so it does not repeat."),
            _q("How do you handle pressure and tight deadlines?", "Time management", "behavioral",
               "Describe prioritising tasks, breaking work down, communicating early about risks, and give a short real example of delivering under pressure."),
        ],
        "Hard": [
            _q("Your team is behind schedule and a teammate is not delivering. What would you do?", "Situational leadership", "situational",
               "Talk to the teammate privately to understand blockers, offer help, re-plan and redistribute work if needed, keep the manager informed early, and focus on the shared goal rather than blame."),
            _q("You disagree with your manager's technical decision. How do you respond?", "Communication", "situational",
               "Seek to understand their reasoning, share your view respectfully with data, listen, and commit to the final decision while documenting concerns if the risk is high."),
            _q("You are given two urgent tasks by two different seniors with conflicting priorities. What do you do?", "Prioritization", "situational",
               "Clarify deadlines and impact, communicate transparently with both seniors, escalate for a priority decision if needed, and avoid silently dropping either task."),
            _q("Tell me about the most difficult decision you have made and how you made it.", "Decision making", "behavioral",
               "Set the context and stakes, describe the options and criteria, how you consulted others and decided, and the result and reflection."),
            _q("You notice a serious error in a colleague's work just before the client presentation. What do you do?", "Ethics and teamwork", "situational",
               "Speak to the colleague immediately and privately, help fix it or flag it to the lead in time, prioritise honesty toward the client, and handle it constructively without blame."),
        ],
    },
}
for _q_, _ in BANK["HR"]["Easy"]:
    _q_.type = "behavioral"  # HR warm-ups are shown as behavioral questions

LEVELS = list(DIFFICULTIES)


def _mock_question(s: Session, prev_score: Optional[int], exclude: Optional[Question] = None) -> Question:
    """Adaptive demo logic: move up a level after a strong answer, down after a weak one; never repeat a question."""
    idx = LEVELS.index(s.difficulty)
    if prev_score is not None:
        if prev_score >= 8:
            idx = min(idx + 1, 2)
        elif prev_score <= 3:
            idx = max(idx - 1, 0)
    asked = {q.question for q in s.questions}
    asked_topics = {q.topic for q in s.questions}
    # Prefer alternating conceptual/practical for Technical to mix styles.
    want_practical = s.domain == "Technical" and bool(s.questions) and s.questions[-1].type == "conceptual"
    for level in [idx] + [i for i in (1, 0, 2) if i != idx]:
        pool = [q for q, _ in BANK[s.domain][LEVELS[level]] if q.question not in asked]
        fresh = [q for q in pool if q.topic not in asked_topics] or pool
        if s.domain == "Technical" and fresh:
            fresh.sort(key=lambda q: (q.type == "practical") != want_practical)
        if fresh:
            return fresh[0]
    raise InterviewError("No more demo questions available.", 500)


IDEAL = {q.question: ideal for domain in BANK.values() for level in domain.values() for q, ideal in level}


def _ideal_for(q: Question) -> str:
    return IDEAL.get(q.question, "Give a clear, structured answer with a short example.")


def _mock_evaluate(s: Session, q: Question, text: str) -> Evaluation:
    """Transparent heuristic scoring for DEMO mode (NOT real AI): length, structure words, examples, numbers."""
    words = re.findall(r"\b\w+\b", text.lower())
    n = len(words)
    idk = re.search(r"\b(don'?t know|no idea|not sure|can'?t remember)\b", text.lower()) and n < 25
    structure = len(re.findall(r"\b(first|second|then|because|therefore|for example|for instance|however|finally|result|so that)\b", text.lower()))
    example = bool(re.search(r"\b(for example|for instance|e\.g\.|when i|in my project|i built|i worked)\b", text.lower()))
    qwords = {w for w in re.findall(r"[a-z]{5,}", q.question.lower())}
    overlap = len(qwords & set(words))
    score = 2 + min(n, 80) / 80 * 4 + min(structure, 3) * 0.5 + (1 if example else 0) + min(overlap, 2) * 0.25
    if idk or n < 5:
        score = 1 if n < 5 else 2
    score = max(0, min(10, round(score)))
    comm = max(0, min(10, round(2 + min(n, 70) / 70 * 4 + min(structure, 3) * 0.7 + (1 if n >= 20 else 0))))
    tech = max(0, min(10, round(score + (1 if example else 0) - (1 if n < 20 else 0))))
    hedges = len(re.findall(r"\b(maybe|i think|i guess|probably|not sure|kind of|sort of)\b", text.lower()))
    conf = max(0, min(10, round(4 + min(n, 60) / 60 * 3 - hedges * 1.2 + (1 if n >= 30 else 0))))

    strengths, improvements = [], []
    if n >= 40:
        strengths.append("You gave a reasonably detailed answer.")
    if structure >= 2:
        strengths.append("Your answer had a logical flow.")
    if example:
        strengths.append("You backed your answer with an example.")
    if not strengths:
        strengths.append("You attempted the question - that is the first step.")
    if n < 30:
        improvements.append("Expand your answer - aim for 4-6 sentences covering the concept and an example.")
    if not example:
        improvements.append("Add a concrete example or project experience.")
    if hedges:
        improvements.append("Reduce hedging words such as 'maybe' or 'I think' to sound more confident.")
    if structure < 2:
        improvements.append("Structure your answer (definition -> explanation -> example).")
    return Evaluation(
        score=score,
        strengths=strengths[:3],
        improvements=improvements[:3] or ["Add one more depth point, such as a trade-off or edge case."],
        feedback=(
            "[DEMO MODE - heuristic feedback, not generated by AI] "
            f"Your answer was {n} words long. Compare it with the sample answer below and add any missing points."
        ),
        ideal_answer=_ideal_for(q),
        communication_score=comm,
        technical_depth_score=tech,
        confidence_score=conf,
    )
