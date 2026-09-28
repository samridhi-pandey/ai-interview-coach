// AI Interview Preparation Coach - vanilla JS frontend. No secrets here; all AI calls go through the backend.
const $ = (id) => document.getElementById(id);
const show = (el, on = true) => el.classList.toggle("hidden", !on);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fill = (id, items, tag = "li") => ($(id).innerHTML = (items || []).map((t) => `<${tag}>${esc(t)}</${tag}>`).join(""));

const state = { file: null, cvId: null, cvName: null, domain: "Technical", difficulty: "Medium", sessionId: null, finished: false };

async function api(path, options) {
  let res;
  try { res = await fetch(path, options); } catch { throw new Error("Cannot reach the server. Is the backend running?"); }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    let msg = data?.detail;
    if (Array.isArray(msg)) msg = "Please check your input and try again.";
    throw new Error(msg || `Request failed (${res.status}).`);
  }
  return data;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const setError = (id, msg) => { $(id).textContent = msg || ""; show($(id), !!msg); };

// ---------- status badge + tabs ----------
api("/api/status").then((s) => {
  const b = $("modeBadge");
  b.textContent = s.mode === "ai" ? `AI mode · ${s.model}` : "DEMO mode · no API key (heuristic feedback)";
  b.classList.add(s.mode);
}).catch(() => { $("modeBadge").textContent = "server offline"; });

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  show($("tab-cv"), name === "cv");
  show($("tab-interview"), name === "interview");
  if (name === "interview") renderCvContext();
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));

// ================= MODULE 1: CV ANALYZER =================
const dz = $("dropzone");
["dragenter", "dragover"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (ev) => pickFile(ev.dataTransfer.files[0]));
$("cvFile").addEventListener("change", (ev) => pickFile(ev.target.files[0]));

function pickFile(f) {
  setError("cvError", "");
  if (!f) return;
  if (!/\.(pdf|docx)$/i.test(f.name)) return setError("cvError", "Unsupported file type. Please upload a PDF or DOCX file.");
  if (f.size > 5 * 1024 * 1024) return setError("cvError", "File is too large (max 5 MB).");
  state.file = f;
  const kb = f.size / 1024;
  $("fileInfo").textContent = `📎 ${f.name} · ${kb > 1024 ? (kb / 1024).toFixed(1) + " MB" : Math.round(kb) + " KB"}`;
  show($("fileInfo"));
  $("analyzeBtn").disabled = false;
}

$("analyzeBtn").addEventListener("click", async () => {
  if (!state.file) return;
  setError("cvError", "");
  $("analyzeBtn").disabled = true;
  show($("cvLoading"));
  try {
    const fd = new FormData();
    fd.append("file", state.file);
    const data = await api("/api/cv/analyze", { method: "POST", body: fd });
    state.cvId = data.cv_id;
    state.cvName = data.filename;
    renderCv(data);
  } catch (e) {
    setError("cvError", e.message);
  } finally {
    show($("cvLoading"), false);
    $("analyzeBtn").disabled = false;
  }
});

const CAT_LABELS = { structure: "Structure", formatting: "Formatting", content: "Content", keywords: "Keywords", completeness: "Completeness" };
const scoreColor = (pct) => (pct >= 75 ? "#16a34a" : pct >= 50 ? "#f59e0b" : "#dc2626");

function renderCv(data) {
  const a = data.analysis;
  show($("cvResult"));
  $("atsScore").textContent = a.ats_score;
  $("atsRing").style.setProperty("--p", a.ats_score);
  $("atsRing").style.setProperty("--c", scoreColor(a.ats_score));
  $("catBars").innerHTML = Object.keys(CAT_LABELS).map((k) => {
    const pct = a.category_percent[k];
    return `<div class="bar"><span>${CAT_LABELS[k]}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${scoreColor(pct)}"></div></div><span class="bar-val">${pct}%</span></div>`;
  }).join("");
  const chips = (list, empty) => (list.length ? list.map((s) => `<span class="chip">${esc(s)}</span>`).join("") : `<span class="muted">${empty}</span>`);
  $("sections").innerHTML = chips(a.detected_sections, "No standard section headings detected.");
  $("skills").innerHTML = chips(a.detected_skills, "No recognised skills detected.");
  fill("strengths", a.strengths);
  fill("areas", a.areas_to_improve);
  fill("recs", a.recommendations);
  $("cvPreview").textContent = data.extracted_preview + (data.extracted_chars > 1500 ? "\n…" : "");
  $("cvNote").textContent = a.note || "";
  show($("cvNote"), !!a.note);
  if (a.mode === "demo") {
    $("cvNote").textContent = (a.note ? a.note + " " : "") + "DEMO mode: the ATS score is rule-based; the written feedback is generated from rules, not AI.";
    show($("cvNote"));
  }
  $("cvResult").scrollIntoView({ behavior: "smooth" });
}

$("goInterview").addEventListener("click", () => switchTab("interview"));

// ================= MODULE 2: INTERVIEW =================
function bindSeg(id, key) {
  $(id).querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
    $(id).querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state[key] = b.dataset.value;
  }));
}
bindSeg("domainSeg", "domain");
bindSeg("diffSeg", "difficulty");

function renderCvContext() {
  const el = $("cvContext");
  if (state.cvId) {
    el.innerHTML = `✅ Using CV: <strong>${esc(state.cvName)}</strong> to personalise questions <button class="btn ghost" id="dropCv">Remove</button>`;
    $("dropCv").addEventListener("click", () => { state.cvId = null; state.cvName = null; renderCvContext(); });
  } else {
    el.innerHTML = `No CV attached — you'll get general questions. <button class="btn ghost" id="toCv">Analyze a CV first</button>`;
    $("toCv").addEventListener("click", () => switchTab("cv"));
  }
}
renderCvContext();

function showQuestion(q, total) {
  ["setup", "fbCard", "reportCard"].forEach((id) => show($(id), false));
  show($("qCard"));
  $("qCount").textContent = `Question ${q.number} of ${total}`;
  $("qTopic").textContent = q.topic;
  $("qType").textContent = q.type;
  $("qText").textContent = q.text;
  $("progressBar").style.width = `${((q.number - 1) / total) * 100}%`;
  $("answerBox").value = "";
  $("charCount").textContent = "0 / 4000";
  setError("answerError", "");
  $("answerBox").focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("startBtn").addEventListener("click", async () => {
  setError("startError", "");
  $("startBtn").disabled = true;
  show($("startLoading"));
  try {
    const d = await post("/api/interview/start", { domain: state.domain, difficulty: state.difficulty, cv_id: state.cvId });
    state.sessionId = d.session_id;
    state.total = d.total;
    state.finished = false;
    state.currentDomain = d.domain;
    showQuestion(d.question, d.total);
  } catch (e) {
    setError("startError", e.message);
  } finally {
    show($("startLoading"), false);
    $("startBtn").disabled = false;
  }
});

$("answerBox").addEventListener("input", (e) => ($("charCount").textContent = `${e.target.value.length} / 4000`));

$("submitBtn").addEventListener("click", async () => {
  const answer = $("answerBox").value.trim();
  if (!answer) return setError("answerError", "Please type an answer before submitting.");
  setError("answerError", "");
  $("submitBtn").disabled = true;
  $("answerBox").disabled = true;
  show($("evalLoading"));
  try {
    const d = await post("/api/interview/answer", { session_id: state.sessionId, answer });
    state.next = d.next_question;
    state.finished = d.finished;
    renderFeedback(d.evaluation, d.finished, d.mode);
  } catch (e) {
    setError("answerError", e.message);
  } finally {
    show($("evalLoading"), false);
    $("submitBtn").disabled = false;
    $("answerBox").disabled = false;
  }
});

function renderFeedback(ev, finished, mode) {
  show($("qCard"), false);
  show($("fbCard"));
  $("fbScore").textContent = ev.score;
  $("fbComm").textContent = ev.communication_score;
  $("fbTech").textContent = ev.technical_depth_score;
  $("fbConf").textContent = ev.confidence_score;
  $("fbTechLabel").textContent = state.currentDomain === "HR" ? "Relevance / Content" : "Technical Depth";
  fill("fbStrengths", ev.strengths);
  fill("fbImprove", ev.improvements);
  $("fbText").textContent = ev.feedback;
  $("fbIdeal").textContent = ev.ideal_answer;
  $("nextBtn").textContent = finished ? "View final report →" : "Next question →";
  setError("fbError", "");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("nextBtn").addEventListener("click", async () => {
  if (!state.finished) return showQuestion(state.next, state.total);
  setError("fbError", "");
  $("nextBtn").disabled = true;
  show($("reportLoading"));
  try {
    renderReport(await post("/api/interview/report", { session_id: state.sessionId }));
  } catch (e) {
    setError("fbError", e.message);
  } finally {
    show($("reportLoading"), false);
    $("nextBtn").disabled = false;
  }
});

function renderReport(r) {
  show($("fbCard"), false);
  show($("reportCard"));
  $("reportMeta").textContent = `${r.domain} interview · ${r.difficulty} difficulty`;
  $("readyPct").textContent = r.readiness_percentage;
  $("readyRing").style.setProperty("--p", r.readiness_percentage);
  $("readyRing").style.setProperty("--c", scoreColor(r.readiness_percentage));
  $("rOverall").textContent = r.overall_score;
  $("rComm").textContent = r.communication_score;
  $("rTech").textContent = r.technical_depth_score;
  $("rTechLabel").textContent = r.technical_depth_label;
  $("rConf").textContent = r.confidence_score;
  $("qScores").innerHTML = r.questions.map((q) => {
    const pct = q.score * 10;
    return `<div class="bar"><span>Q${q.number} · ${esc(q.topic)}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${scoreColor(pct)}"></div></div><span class="bar-val">${q.score}/10</span></div>`;
  }).join("");
  fill("rStrong", r.strong_areas);
  fill("rImprove", r.areas_to_improve);
  fill("rTips", r.top_3_tips);
  const notes = [r.note, r.mode === "demo" ? "DEMO mode: scores come from simple heuristics and a local question bank, not from AI." : ""].filter(Boolean);
  $("reportNote").textContent = notes.join(" ");
  show($("reportNote"), notes.length > 0);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("restartBtn").addEventListener("click", () => {
  state.sessionId = null;
  show($("reportCard"), false);
  show($("setup"));
  renderCvContext();
  window.scrollTo({ top: 0, behavior: "smooth" });
});
